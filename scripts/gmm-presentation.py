# Imports
from pathlib import Path
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
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal

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

OUT_BASE_PATH = Path() / "presentation" / "assets"

methods = (
    SamplerName.SMC_DIFF_OPT,
    ConditioningMethodName.VJP_GUIDANCE,
)
title_map = {
    SamplerName.SMC_DIFF_OPT: "SMCDiffOpt",
    SamplerName.MCG_DIFF: "MCGDiff",
    ConditioningMethodName.DIFFUSION_POSTERIOR_SAMPLING: "DPS",
    ConditioningMethodName.PSEUDO_INVERSE_GUIDANCE: r"$\Pi$IGD",
    ConditioningMethodName.VJP_GUIDANCE: "TMPD",
}


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
    init_sample: Real[Array, " {dim_x}"] = mixt.sample(key_init_sample)

    init_obs: Real[Array, " {dim_y}"] = a_recon @ init_sample
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
            num_steps=num_steps, sde=sde.reverse(model).guide(cond_method.guidance_score_func)
        )
        guided_sampler = PCSampler(
            shape=(num_samples, dim_x), outer_solver=solver, stack_samples=True
        )

    return guided_sampler


def animate_samples(
    title: str,
    file_name: str,
    samples: Real[Array, "{num_steps} {num_samples} {dim_x}"],
    posterior_samples: Real[Array, "{num_samples} {dim_x}"] | None = None,
    skip: int = 5,
    reverse: bool = False,
    figsize: tuple[int, int] = (6, 6),
    base_sampler: DDIMVP | None = None,
    rescale: bool = False,
    init_obs: Array | None = None,
) -> FuncAnimation:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor((0.98, 0.98, 0.98))

    if base_sampler is not None:
        c_t = base_sampler.sqrt_alphas_cumprod[::-1]
        d_t = base_sampler.sqrt_1m_alphas_cumprod[::-1]

    def animate(t: int):
        # Clear axis for next frame
        ax.clear()

        # Axes
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.grid(True)

        if posterior_samples is not None:
            ax.scatter(
                x=posterior_samples[:, 0],
                y=posterior_samples[:, 1],
                color=COLOR_POSTERIOR,
                alpha=0.5,
                edgecolors="black",
                lw=0.5,
                s=10,
            )

        if rescale and (base_sampler is not None):
            total_size = 10 * NUM_SAMPLES
            unscaled_size = dists.MvNormal(
                loc=c_t[t] * init_obs, cov=c_t[t] ** 2 * sigma_y**2 + d_t[t] ** 2 * a @ a.T
            ).pdf(samples[t, :, :] @ a.T)
            scaled_size = unscaled_size / unscaled_size.sum()
            size = total_size * scaled_size
        else:
            size = 10

        # Particle samples
        ax.scatter(
            x=samples[t, :, 0],
            y=samples[t, :, 1],
            s=size,
            color=COLOR_ALGORITHM,
            alpha=0.5,
            edgecolors="black",
            lw=0.5,
        )

        # Limits
        ax.set_xlim(*CHART_LIMS)
        ax.set_ylim(*CHART_LIMS)

        ax.set_xticks(CHART_TICKS)
        ax.set_yticks(CHART_TICKS)

        # Labels
        ax.set_xlabel("Coordinate 1")
        ax.set_ylabel("Coordinate 2")
        ax.set_title(f"{title}\nt={t if reverse else (NUM_STEPS - t)}")

    frames = (
        jnp.arange(NUM_STEPS, -skip, -skip) if reverse else jnp.arange(0, NUM_STEPS + skip, skip)
    )
    ani = FuncAnimation(fig, animate, frames=frames, interval=25)

    out_path = OUT_BASE_PATH / file_name / f"{file_name}.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer="pillow", dpi=300)

    plt.close()

    return ani


def plot_measurement_system(
    prior_samples: Real[Array, "{num_samples} {dim_x}"],
    posterior_samples: Real[Array, "{num_samples} {dim_x}"],
    init_sample: Real[Array, " {dim_x}"],
    init_obs: Real[Array, " {dim_y}"],
    a: Real[Array, "{dim_y} {dim_x}"],
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col", sharey="col")
    fig.patch.set_facecolor((0.98, 0.98, 0.98))

    transformed_samples = prior_samples[0] @ a.T
    transformed_init_sample = a @ init_sample

    noisy_samples = transformed_samples + SIGMA_Y * np.random.randn(*transformed_samples.shape)

    mu = transformed_init_sample
    grid_range = 4
    x_1_points = np.linspace(mu[0] - grid_range, mu[0] + grid_range, 100)
    x_2_points = np.linspace(mu[1] - grid_range, mu[1] + grid_range, 100)
    x_1_meshpoints, x_2_meshpoints = np.meshgrid(x_1_points, x_2_points)
    z_vals = multivariate_normal(mu, sigma_y * np.eye(DIM_Y)).pdf(
        np.dstack((x_1_meshpoints, x_2_meshpoints))
    )

    axes[0, 0].set_xlim(*CHART_LIMS)
    axes[0, 0].set_ylim(*CHART_LIMS)
    axes[0, 0].set_xticks(CHART_TICKS)
    axes[0, 0].set_yticks(CHART_TICKS)
    axes[0, 0].set_xlabel("Coordinate 1")
    axes[0, 0].set_ylabel("Coordinate 2")
    axes[0, 0].scatter(
        prior_samples[0, :, 0],
        prior_samples[0, :, 1],
        alpha=0.5,
        color=COLOR_ALGORITHM,
        s=10,
        edgecolors="black",
        lw=0.5,
    )
    axes[0, 0].scatter(
        init_sample[0], init_sample[1], edgecolors="black", lw=0.5, color="red", s=100
    )
    axes[0, 0].set_title("GMM Prior in $\\mathbf{x}_*$ space")
    axes[0, 0].grid(True)

    axes[0, 1].set_xlabel("Coordinate 1")
    axes[0, 1].set_ylabel("Coordinate 2")
    axes[0, 1].scatter(
        transformed_samples[:, 0],
        transformed_samples[:, 1],
        alpha=0.5,
        color="orange",
        s=10,
        edgecolors="black",
        lw=0.5,
    )
    axes[0, 1].scatter(
        transformed_init_sample[0],
        transformed_init_sample[1],
        edgecolors="black",
        lw=0.5,
        color="red",
        s=100,
    )
    axes[0, 1].set_title("Transformed GMM Prior in $\\mathbf{y}$ space")
    axes[0, 1].grid(True)

    axes[1, 0].set_xlim(*CHART_LIMS)
    axes[1, 0].set_ylim(*CHART_LIMS)
    axes[1, 0].set_xticks(CHART_TICKS)
    axes[1, 0].set_yticks(CHART_TICKS)
    axes[1, 0].set_xlabel("Coordinate 1")
    axes[1, 0].set_ylabel("Coordinate 2")
    axes[1, 0].scatter(
        posterior_samples[:, 0],
        posterior_samples[:, 1],
        alpha=0.5,
        color=COLOR_POSTERIOR,
        s=10,
        edgecolors="black",
        lw=0.5,
    )
    axes[1, 0].scatter(
        init_sample[0], init_sample[1], edgecolors="black", lw=0.5, color="red", s=100
    )
    axes[1, 0].set_title("Posterior in $\\mathbf{x}_*$ space")
    axes[1, 0].grid(True)

    axes[1, 1].set_xlabel("Coordinate 1")
    axes[1, 1].set_ylabel("Coordinate 2")
    axes[1, 1].scatter(
        noisy_samples[:, 0],
        noisy_samples[:, 1],
        alpha=0.5,
        color="orange",
        s=10,
        edgecolors="black",
        lw=0.5,
    )
    axes[1, 1].contour(x_1_meshpoints, x_2_meshpoints, z_vals, levels=10, zorder=1)
    axes[1, 1].scatter(
        transformed_init_sample[0],
        transformed_init_sample[1],
        edgecolors="black",
        lw=0.5,
        color="red",
        s=100,
    )
    axes[1, 1].scatter(
        init_obs[0], init_obs[1], color="dodgerblue", edgecolors="black", lw=0.5, s=100
    )
    axes[1, 1].set_title("Noisy observations in $\\mathbf{y}$ space")
    axes[1, 1].grid(True)

    return fig, axes


if __name__ == "__main__":
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

    # Setup inverse problem
    key, sub_key = random.split(key)

    a, sigma_y, u, diag, v, coordinate_mask, init_obs, init_sample = generate_measurement_equations(
        DIM_X, DIM_Y, mixt, SIGMA_Y, sub_key
    )

    # Setup samplers
    prior_sampler = DDIMVP(
        num_steps=NUM_STEPS,
        shape=(NUM_SAMPLES, DIM_X),
        model=model,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        eta=1.0,  # NOTE: equates to using DDPM
        stack_samples=True,
    )

    guided_samplers = {
        method: make_sampler(
            method,
            prior_sampler,
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

    # Make samples
    key, sub_key = random.split(key)

    prior_samples: Real[Array, "{num_steps} {num_samples} {dim_x}"] = prior_sampler.sample(sub_key)

    posterior_samples = get_posterior(init_obs, mixt, a, sigma_y).sample(sub_key, (NUM_SAMPLES,))

    particle_samples = {
        method: jnp.array(sampler.sample(key, y=init_obs, ESSrmin=0.8))[
            :: 1 if isinstance(method, SamplerName) else -1
        ]
        for method, sampler in guided_samplers.items()
        if not print(method)
    }

    # Animate samples and save
    prior_animation = animate_samples("Prior Sampling", "gmm-prior", prior_samples, reverse=True)
    posterior_animations = {
        method: animate_samples(
            f"Posterior Sampling ({title_map[method]})",
            f"gmm-posterior-{method}",
            samples,
            posterior_samples,
            base_sampler=prior_sampler,
            rescale=method == SamplerName.SMC_DIFF_OPT,
            init_obs=init_obs,
        )
        for method, samples in particle_samples.items()
    }

    # Plot measurement system demo and save
    fig, ax = plot_measurement_system(prior_samples, posterior_samples, init_sample, init_obs, a)
    out_path = OUT_BASE_PATH / "gmm-measurement-system.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
