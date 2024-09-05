# # Gaussian Mixture Model Example


# Imports
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from IPython.display import HTML
from jax import grad, vmap
from jax.tree_util import Partial
from jaxtyping import Array, PRNGKeyArray, Real

from diffusionlib.sampler.ddim import DDIMVP
from diffusionlib.sampler.smc import SMCDiffOptSampler
from diffusionlib.sde import SDE, VP

matplotlib.rcParams["animation.embed_limit"] = 2**128
COLOR_POSTERIOR = "#a2c4c9"
COLOR_ALGORITHM = "#ff7878"


# Config
key = random.PRNGKey(100)
num_steps = 1000
num_samples = 1000
dim_x = 80
dim_y = 1
measurement_noise_std = 1
size = 8.0  # mean multiplier
center_range = np.array((-2, 2))

beta_min = 0.01
beta_max = 20.0

# plotting
chart_lims = size * (center_range + np.array((-0.5, 0.5)))


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


# Build prior (equal weighted, grid GMM)
means = [
    jnp.array([-size * i, -size * j] * (dim_x // 2))
    for i in range(center_range[0], center_range[1] + 1)
    for j in range(center_range[0], center_range[1] + 1)
]
weights = jnp.ones(len(means))
weights = weights / weights.sum()

ou_mixt_fun = Partial(ou_mixt, means=means, dim_x=dim_x, weights=weights)
mixt = ou_mixt_fun(1)


# Get model prior samples (i.e. DDPM)
key, sub_key = random.split(key)

sde = VP(jnp.array(beta_min), jnp.array(beta_max))
model = get_model_fn(ou_mixt_fun, sde)  # epsilon estimator

sampler = DDIMVP(
    num_steps=num_steps,
    shape=(num_samples, dim_x),
    model=model,
    beta_min=beta_min,
    beta_max=beta_max,
    eta=1.0,  # NOTE: equates to using DDPM
    stack_samples=True,
)

# NOTE: lead axis of `prior_samples` is such that index 0 corresponds to X_0 (not X_T).
prior_samples: Real[Array, "{num_steps} {num_samples} {dim_x}"] = sampler.sample(sub_key)


# Create animation of particle posterior sampling
subhist = [*prior_samples[::10], prior_samples[-1]][::-1]

fig, ax = plt.subplots(figsize=(6, 6))


def animate(i: int):
    # Clear axis for next frame
    ax.clear()

    # Axes
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)

    # Particle samples
    ax.scatter(
        x=subhist[i][:, 0],
        y=subhist[i][:, 1],
        color=COLOR_ALGORITHM,
        alpha=0.5,
        edgecolors="black",
        lw=0.5,
        s=10,
    )

    # Limits
    ax.set_xlim(*chart_lims)
    ax.set_ylim(*chart_lims)

    # Labels
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    ax.set_title(f"Particle posterior sampling\nt={num_steps - (i * 10)}")

    ax.grid(True)


ani = animation.FuncAnimation(fig, animate, frames=len(subhist), interval=50)
plt.close()

ani.save("gmm-prior.gif", writer="pillow")


# # Setup inverse problem
# key, sub_key = random.split(key)

# a, sigma_y, u, diag, v, coordinate_mask, init_obs, init_sample = generate_measurement_equations(
#     dim_x, dim_y, mixt, measurement_noise_std, sub_key
# )


# # Get posterior samples
# posterior = get_posterior(init_obs, mixt, a, sigma_y)
# key, sub_key = random.split(key)

# posterior_samples = posterior.sample(sub_key, (num_samples,))


# smc_guided_sampler = SMCDiffOptSampler(
#     base_sampler=sampler,
#     obs_matrix=a,
#     obs_noise=measurement_noise_std,
#     num_particles=num_samples,
#     stack_samples=True,
# )

# particle_samples = smc_guided_sampler.sample(key, y=init_obs)


# # Create animation of particle posterior sampling
# subhist = [*particle_samples[::10], particle_samples[-1]]

# fig, ax = plt.subplots(figsize=(6, 6))


# def animate(i: int):
#     # Clear axis for next frame
#     ax.clear()

#     # Axes
#     ax.axhline(0, color="black", lw=0.5)
#     ax.axvline(0, color="black", lw=0.5)

#     # True posterior samples
#     ax.scatter(
#         x=posterior_samples[:, 0],
#         y=posterior_samples[:, 1],
#         color=COLOR_POSTERIOR,
#         alpha=0.5,
#         edgecolors="black",
#         lw=0.5,
#         s=10,
#     )

#     # Particle samples
#     ax.scatter(
#         x=subhist[i][:, 0],
#         y=subhist[i][:, 1],
#         color=COLOR_ALGORITHM,
#         alpha=0.5,
#         edgecolors="black",
#         lw=0.5,
#         s=10,
#     )

#     # Limits
#     ax.set_xlim(*chart_lims)
#     ax.set_ylim(*chart_lims)

#     # Labels
#     ax.set_xlabel("Coordinate 1")
#     ax.set_ylabel("Coordinate 2")
#     ax.set_title(f"Particle posterior sampling\nt={num_steps - (i * 10)}")

#     ax.grid(True)


# ani = animation.FuncAnimation(fig, animate, frames=len(subhist), interval=100)
# plt.close()

# ani.save("gmm-posterior.gif", writer="pillow")
