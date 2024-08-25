# %%
# Imports
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import particles.distributions as dists
from jax import grad, vmap
from jax.tree_util import Partial
from jaxtyping import Array, PRNGKeyArray, Real

from diffusionlib.conditioning_method import ConditioningMethodName, get_conditioning_method
from diffusionlib.sampler import SamplerName, get_sampler
from diffusionlib.sampler.base import Sampler
from diffusionlib.sampler.ddim import DDIMVP
from diffusionlib.sampler.predictor_corrector import PCSampler
from diffusionlib.sde import SDE, VP
from diffusionlib.solver import EulerMaruyama

COLOR_POSTERIOR = "#a2c4c9"
COLOR_ALGORITHM = "#ff7878"
NUM_STEPS = 1000
NUM_SAMPLES = 1000
SIGMA_Y = 1.0
DIM_X = 8
DIM_Y = 2
SEED = 100

SIZE = 8.0  # mean multiplier
CENTER_RANGE = np.array((-2, 2))
BETA_MIN = 0.01
BETA_MAX = 20.0

CHART_LIMS = SIZE * (CENTER_RANGE + np.array((-0.5, 0.5)))
CHART_TICKS = np.arange(CENTER_RANGE[0] * SIZE, CENTER_RANGE[1] * (SIZE + 1), SIZE)

methods = (
    SamplerName.SMC_DIFF_OPT,
    ConditioningMethodName.DIFFUSION_POSTERIOR_SAMPLING,
    ConditioningMethodName.PSEUDO_INVERSE_GUIDANCE,
    ConditioningMethodName.VJP_GUIDANCE,
)


# Define epsilon functions
# NOTE: model/epsilon function is just the negative score by the variance
def get_model_fn(
    ou_dist: Callable[[Array], dist.MixtureSameFamily], sde: SDE
) -> Callable[[Array, Array], Array]:
    return vmap(
        grad(
            lambda x, t: -jnp.sqrt(sde.marginal_variance(t))
            * ou_dist(sde.marginal_mean_coeff(t)).log_prob(x)
        )
    )


def get_score_fn(
    ou_dist: Callable[[Array], dist.MixtureSameFamily], sde: SDE
) -> Callable[[Array, Array], Array]:
    return vmap(grad(lambda x, t: ou_dist(sde.marginal_mean_coeff(t)).log_prob(x)))


# Define mixture model function
def ou_mixt(mean_coeff: float, means: Array, dim_x: int, weights: Array) -> dist.MixtureSameFamily:
    means = jnp.vstack(means) * mean_coeff
    covs = jnp.repeat(jnp.eye(dim_x)[None], axis=0, repeats=means.shape[0])
    return dist.MixtureSameFamily(
        component_distribution=dist.MultivariateNormal(loc=means, covariance_matrix=covs),
        mixing_distribution=dist.CategoricalProbs(weights),
    )


# Define posterior
def gaussian_posterior(
    y: Real[Array, " dim_y"],
    likelihood_a: Real[Array, "dim_y dim_x"],
    likelihood_bias: Real[Array, " dim_y"],
    likelihood_precision: Real[Array, "dim_y dim_y"],
    prior_loc: Real[Array, " dim_x"],
    prior_covar: Real[Array, "dim_x dim_x"],
) -> dist.MultivariateNormal:
    # Compute the precision matrix of the prior distribution
    prior_precision_matrix = jnp.linalg.inv(prior_covar)

    # Calculate the precision matrix of the posterior distribution
    posterior_precision_matrix = (
        prior_precision_matrix + likelihood_a.T @ likelihood_precision @ likelihood_a
    )

    # Calculate the covariance matrix of the posterior distribution
    posterior_covariance_matrix = jnp.linalg.inv(posterior_precision_matrix)

    # Calculate the mean of the posterior distribution
    posterior_mean = posterior_covariance_matrix @ (
        likelihood_a.T @ likelihood_precision @ (y - likelihood_bias)
        + prior_precision_matrix @ prior_loc
    )

    # Ensure symmetry and numerical stability of the covariance matrix
    # Handle potential numerical issues by regularization
    try:
        posterior_covariance_matrix = (
            posterior_covariance_matrix + posterior_covariance_matrix.T
        ) / 2
    except ValueError:
        u, s, v = jnp.linalg.svd(posterior_covariance_matrix, full_matrices=False)
        s = jnp.clip(s, 1e-12, 1e6).real
        posterior_covariance_matrix = u.real @ jnp.diag(s) @ v.real
        posterior_covariance_matrix = (
            posterior_covariance_matrix + posterior_covariance_matrix.T
        ) / 2

    return dist.MultivariateNormal(
        loc=posterior_mean, covariance_matrix=posterior_covariance_matrix
    )


# Define posterior for the mixture model
def get_posterior(
    obs: Array, prior: dist.MixtureSameFamily, a: Array, sigma_y: Array
) -> dist.MixtureSameFamily:
    mixing_dist: dist.CategoricalProbs = prior.mixing_distribution
    component_dist: dist.MultivariateNormal = prior.component_distribution  # type: ignore
    comp_mean = component_dist.mean
    comp_cov: Array = component_dist.covariance_matrix  # type: ignore

    # Precompute the inverse of the observation noise covariance matrix
    precision = jnp.linalg.inv(sigma_y)
    modified_means = []
    modified_covars = []
    weights = []

    # Iterate through the components of the prior distribution
    for loc, cov, weight in zip(comp_mean, comp_cov, mixing_dist.probs):
        # Compute the posterior distribution for the current component
        new_dist = gaussian_posterior(obs, a, jnp.zeros_like(obs), precision, loc, cov)
        modified_means.append(new_dist.mean)
        modified_covars.append(new_dist.covariance_matrix)

        # Calculate the prior likelihood and residual
        prior_x = dist.MultivariateNormal(loc, covariance_matrix=cov)
        residue = obs - a @ new_dist.loc

        # Compute log-probability contributions
        log_constant = (
            -0.5 * residue @ precision @ residue.T
            + prior_x.log_prob(new_dist.mean)
            - new_dist.log_prob(new_dist.mean)
        )

        # Compute the log weight for the component
        weights.append(jnp.log(weight) + log_constant)

    # Normalize weights
    weights = jnp.array(weights)
    normalized_weights = weights - jax.scipy.special.logsumexp(weights)

    # Construct categorical distribution from the normalized weights
    categorical_distribution = dist.CategoricalLogits(logits=normalized_weights)

    # Construct a mixture distribution of multivariate normals
    multivariate_mixture = dist.MultivariateNormal(
        loc=jnp.stack(modified_means, axis=0),
        covariance_matrix=jnp.stack(modified_covars, axis=0),
    )

    return dist.MixtureSameFamily(categorical_distribution, multivariate_mixture)


# Inverse problem functions
def extended_svd(a: Array) -> tuple[Array, Array, Array, Array]:
    # Compute the singular value decomposition
    u, s, v = jnp.linalg.svd(a, full_matrices=False)

    # Create a coordinate mask based on the length of the singular values
    coordinate_mask = jnp.concatenate([jnp.ones(len(s)), jnp.zeros(v.shape[0] - len(s))]).astype(
        bool
    )

    return u, s, v, coordinate_mask


def generate_measurement_equations(
    dim_x: int,
    dim_y: int,
    mixt: dist.MixtureSameFamily,
    noise_std: float,
    key: PRNGKeyArray,
):
    # Generate random keys for different sources of randomness
    key_a, key_diag, key_init_sample, key_init_obs = random.split(key, 4)

    # Create random matrix
    a = random.normal(key_a, (dim_y, dim_x))

    # Build extended SVD
    u, s, v, coordinate_mask = extended_svd(a)

    # Re-create `s` using uniform sampling, sorting the generated values to align with
    # properties of singular values being ordered in the SVD Sigma (`s`) matrix
    s_new = jnp.sort(random.uniform(key_diag, s.shape), descending=True)
    s_new_mat = jnp.diag(s_new)

    # Re-construct `a` using the sorted diag and coordinate mask
    a_recon: Real[Array, "{dim_y} {dim_x}"] = u @ s_new_mat @ v[coordinate_mask]

    # Sample initial data and simulate initial observations
    init_sample: Real[Array, "{dim_x}"] = mixt.sample(key_init_sample)

    init_obs: Real[Array, "{dim_y}"] = a_recon @ init_sample
    init_obs += random.normal(key_init_obs, init_obs.shape) * noise_std

    # Construct observation noise covariance matrix
    sigma_y = jnp.diag(jnp.full(dim_y, noise_std**2))

    return a_recon, sigma_y, u, s_new, v, coordinate_mask, init_obs, init_sample


def make_sampler(
    method,
    base_sampler,
    sde,
    model,
    score,
    dim_x,
    init_obs,
    a,
    measurement_noise_std,
    num_samples,
    num_steps,
) -> Sampler:
    if isinstance(method, SamplerName):
        guided_sampler = get_sampler(
            method,
            base_sampler=base_sampler,
            obs_matrix=a,
            obs_noise=measurement_noise_std,
            num_particles=num_samples,
            stack_samples=True,
        )
    else:
        cond_method = get_conditioning_method(
            name=method,
            sde=sde.reverse(score),
            y=jnp.tile(init_obs, (num_samples, 1)),
            H=a,
            HHT=a @ a.T,
            observation_map=lambda x: a @ x.flatten(),
            shape=(num_samples, dim_x),
            scale=num_steps + 0.05,  # DPS scale
            noise_std=measurement_noise_std,
        )
        solver = EulerMaruyama(
            num_steps=1000, sde=sde.reverse(model).guide(cond_method.guidance_score_func)
        )
        guided_sampler = PCSampler(
            shape=(num_samples, dim_x), outer_solver=solver, stack_samples=True
        )

    return guided_sampler


# %%
key = random.PRNGKey(SEED)

# Build prior (equal weighted, grid GMM)
means = [
    jnp.array([-SIZE * i, -SIZE * j] * (DIM_X // 2))
    for i in range(CENTER_RANGE[0], CENTER_RANGE[1] + 1)
    for j in range(CENTER_RANGE[0], CENTER_RANGE[1] + 1)
]
weights = jnp.ones(len(means))
weights = weights / weights.sum()

ou_mixt_fun = Partial(ou_mixt, means=means, dim_x=DIM_X, weights=weights)
mixt = ou_mixt_fun(1)

sde = VP(jnp.array(BETA_MIN), jnp.array(BETA_MAX))
model = get_model_fn(ou_mixt_fun, sde)  # epsilon estimator
score = get_score_fn(ou_mixt_fun, sde)

sampler = DDIMVP(
    num_steps=NUM_STEPS,
    shape=(NUM_SAMPLES, DIM_X),
    model=model,
    beta_min=BETA_MIN,
    beta_max=BETA_MAX,
    eta=1.0,  # NOTE: equates to using DDPM
    stack_samples=True,
)

# %%
key, sub_key = random.split(key)
prior_samples = sampler.sample(sub_key)

# Plot model prior samples
fig, ax = plt.subplots(figsize=(6, 6))

# Axes
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)

# Samples
ax.scatter(
    x=prior_samples[0, :, 0],
    y=prior_samples[0, :, 1],
    color=COLOR_ALGORITHM,
    alpha=0.5,
    edgecolors="black",
    lw=0.5,
    s=10,
)

# Limits
ax.set_xlim(*CHART_LIMS)
ax.set_ylim(*CHART_LIMS)

ax.set_xticks(CHART_TICKS)
ax.set_yticks(CHART_TICKS)

# Labels
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

ax.grid(True)

plt.tight_layout()
plt.savefig("paper/assets/gmm_prior_samples.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

# %%
# Setup inverse problem
key, sub_key = random.split(key)

a, sigma_y, u, diag, v, coordinate_mask, init_obs, init_sample = generate_measurement_equations(
    DIM_X, DIM_Y, mixt, SIGMA_Y, sub_key
)

# Get posterior samples
posterior = get_posterior(init_obs, mixt, a, sigma_y)
key, sub_key = random.split(key)
posterior_samples = posterior.sample(sub_key, (NUM_SAMPLES,))

guided_samplers = {
    method: make_sampler(
        method,
        sampler,
        sde,
        model,
        score,
        DIM_X,
        init_obs,
        a,
        SIGMA_Y,
        NUM_SAMPLES,
        NUM_STEPS,
    )
    for method in methods
}

particle_samples = {
    method: jnp.array(sampler.sample(key, y=init_obs, ESSrmin=0.8))[
        :: 1 if isinstance(method, SamplerName) else -1
    ]
    for method, sampler in guided_samplers.items()
    if not print(method)
}

# %%
title_map = {
    SamplerName.SMC_DIFF_OPT: "SMCDiffOpt",
    SamplerName.MCG_DIFF: "MCGDiff",
    ConditioningMethodName.DIFFUSION_POSTERIOR_SAMPLING: "DPS",
    ConditioningMethodName.PSEUDO_INVERSE_GUIDANCE: r"$\Pi$IGD",
    ConditioningMethodName.VJP_GUIDANCE: "TMPD",
}

smc_samples = particle_samples[SamplerName.SMC_DIFF_OPT]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

ax.axhline(0, color="black", lw=0.5)
ax.axvline(0, color="black", lw=0.5)

# True posterior samples
ax.scatter(
    x=posterior_samples[:, 0],
    y=posterior_samples[:, 1],
    color=COLOR_POSTERIOR,
    alpha=0.5,
    edgecolors="black",
    lw=0.5,
    s=10,
)

# Particle samples
ax.scatter(
    x=smc_samples[-1, :, 0],
    y=smc_samples[-1, :, 1],
    color=COLOR_ALGORITHM,
    alpha=0.5,
    edgecolors="black",
    lw=0.5,
    s=10,
)

# Limits
ax.set_xlim(*CHART_LIMS)
ax.set_ylim(*CHART_LIMS)

ax.set_xticks(CHART_TICKS)
ax.set_yticks(CHART_TICKS)

# Labels
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

ax.grid(True)

plt.tight_layout()
plt.savefig("paper/assets/gmm_smc_samples.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

# %%
fig, axs = plt.subplots(nrows=1, ncols=len(methods), figsize=(6 * len(methods), 6), sharey=True)

for i, ((method, samples), ax) in enumerate(zip(particle_samples.items(), axs.flatten())):
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)

    # True posterior samples
    ax.scatter(
        x=posterior_samples[:, 0],
        y=posterior_samples[:, 1],
        color=COLOR_POSTERIOR,
        alpha=0.5,
        edgecolors="black",
        lw=0.5,
        s=10,
    )

    # Particle samples
    ax.scatter(
        x=samples[-1, :, 0],
        y=samples[-1, :, 1],
        color=COLOR_ALGORITHM,
        alpha=0.5,
        edgecolors="black",
        lw=0.5,
        s=10,
    )

    # Limits
    ax.set_xlim(*CHART_LIMS)
    ax.set_ylim(*CHART_LIMS)

    ax.set_xticks(CHART_TICKS)
    ax.set_yticks(CHART_TICKS)

    # Labels
    ax.set_title(f"{title_map[method]}")
    ax.set_xlabel("$x_1$")
    if i == 0:
        ax.set_ylabel("$x_2$")

    ax.grid(True)

plt.tight_layout()
plt.savefig("paper/assets/gmm_samples.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

# %%
c_t = lambda t: sampler.sqrt_alphas_cumprod[t]
d_t = lambda t: sampler.sqrt_1m_alphas_cumprod[t]

y_s = init_obs * sampler.sqrt_alphas_cumprod[:, None][::-1]


def p_y_t(t, x):
    cov = (c_t(t) ** 2 * SIGMA_Y**2 * jnp.eye(DIM_Y)) + (d_t(t) ** 2 * a @ a.T)
    return dists.MvNormal(loc=x @ a.T, cov=cov).logpdf(y_s[t])


fig, ax = plt.subplots(figsize=(10, 6))

for method, samples in particle_samples.items():
    liks = jnp.array([p_y_t(NUM_STEPS - t, samples[t]) for t in range(NUM_STEPS)])
    mean_liks = jnp.mean(liks, axis=1)
    ax.plot(mean_liks[:-1], label=title_map[method])

ax.set_xlabel("$t$")
ax.set_ylabel("Log Likelihood")

ticks = np.arange(0, 1001, 200)  # Generate ticks at intervals (0, 200, 400, ..., 1000)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks[::-1])

ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("paper/assets/gmm_log_likelihoods.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

# %%
posterior_log_prob = jnp.repeat(jnp.mean(posterior.log_prob(posterior_samples)), NUM_STEPS)

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    posterior_log_prob,
    label="Analytic Samples",
    color=COLOR_POSTERIOR,
    lw=2,
)

for method, samples in particle_samples.items():
    particle_log_probs = posterior.log_prob(samples)
    mean_log_probs = jnp.mean(particle_log_probs, axis=1)
    ax.plot(mean_log_probs, label=title_map[method])


ax.set_xlabel("$t$")
ax.set_ylabel("Mean Posterior Density")

ticks = np.arange(0, 1001, 200)  # Generate ticks at intervals (0, 200, 400, ..., 1000)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks[::-1])

ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("paper/assets/gmm_posterior_densities.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
