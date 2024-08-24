import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Imports
import itertools as it
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import torch
from jax import grad, vmap
from jax.tree_util import Partial
from jaxtyping import Array, PRNGKeyArray, Real
from scipy.stats import wasserstein_distance

from diffusionlib.conditioning_method import ConditioningMethodName, get_conditioning_method
from diffusionlib.sampler import SamplerName, get_sampler
from diffusionlib.sampler.ddim import DDIMVP
from diffusionlib.sampler.predictor_corrector import PCSampler
from diffusionlib.sde import SDE, VP
from diffusionlib.solver import EulerMaruyama

from diffusionlib.mcg_diff.particle_filter import mcg_diff
from diffusionlib.mcg_diff.sgm import ScoreModel
from diffusionlib.mcg_diff.utils import (
    NetReparametrized,
    get_optimal_timesteps_from_singular_values,
)


# Config
num_steps = 1000
num_samples = 1000
size = 8.0  # mean multiplier
center_range = np.array((-2, 2))
beta_min = 0.01
beta_max = 20.0

# Experimental Grid
methods = (
    ConditioningMethodName.DIFFUSION_POSTERIOR_SAMPLING,
    ConditioningMethodName.PSEUDO_INVERSE_GUIDANCE,
    ConditioningMethodName.VJP_GUIDANCE,
    SamplerName.MCG_DIFF,
    SamplerName.SMC_DIFF_OPT,
)
sigma_ys = (0.01, 0.1, 1.0)
dim_xs = (8, 80)
dim_ys = (1, 2, 4)
seeds = (0, 1, 10, 100, 1000)
grid = it.product(methods, sigma_ys, dim_xs, dim_ys, seeds)


def _jax_to_torch(array: Array) -> torch.Tensor:
    return torch.from_numpy(jax.device_get(array).copy())


def sliced_wasserstein(dist_1, dist_2, n_slices=100):
    projections = torch.randn(size=(n_slices, dist_1.shape[1]))
    projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
    dist_1_projected = projections @ dist_1.T
    dist_2_projected = projections @ dist_2.T
    return np.mean(
        [
            wasserstein_distance(u_values=d1.cpu().numpy(), v_values=d2.cpu().numpy())
            for d1, d2 in zip(dist_1_projected, dist_2_projected)
        ]
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


def ou_mixt_torch(alpha_t, means, dim, weights):
    cat = torch.distributions.Categorical(weights, validate_args=False)

    ou_norm = torch.distributions.MultivariateNormal(
        torch.vstack(tuple((alpha_t**0.5) * m for m in means)),
        torch.eye(dim).repeat(len(means), 1, 1),
        validate_args=False,
    )
    return torch.distributions.MixtureSameFamily(cat, ou_norm, validate_args=False)


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


def build_extended_svd_torch(A: torch.Tensor):
    U, d, V = torch.linalg.svd(A, full_matrices=True)
    coordinate_mask = torch.ones_like(V[0])
    coordinate_mask[len(d) :] = 0
    return U, d, coordinate_mask, V


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


def experiment(
    method: SamplerName | ConditioningMethodName,
    dim_x: int,
    dim_y: int,
    measurement_noise_std: float,
    seed: int,
) -> float:
    print(f"{method=}, {dim_x=}, {dim_y=}, {measurement_noise_std=}, {seed=}")
    key = random.PRNGKey(seed)
    torch.manual_seed(seed)

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

    sde = VP(jnp.array(beta_min), jnp.array(beta_max))
    model = get_model_fn(ou_mixt_fun, sde)  # epsilon estimator
    score = get_score_fn(ou_mixt_fun, sde)

    sampler = DDIMVP(
        num_steps=num_steps,
        shape=(num_samples, dim_x),
        model=model,
        beta_min=beta_min,
        beta_max=beta_max,
        eta=1.0,  # NOTE: equates to using DDPM
        stack_samples=True,
    )

    # Setup inverse problem
    key, sub_key = random.split(key)

    a, sigma_y, u, diag, v, coordinate_mask, init_obs, init_sample = generate_measurement_equations(
        dim_x, dim_y, mixt, measurement_noise_std, sub_key
    )

    # Get posterior samples
    posterior = get_posterior(init_obs, mixt, a, sigma_y)
    key, sub_key = random.split(key)
    posterior_samples = posterior.sample(sub_key, (num_samples,))

    if method == SamplerName.SMC_DIFF_OPT:
        guided_sampler = get_sampler(
            method,
            base_sampler=sampler,
            obs_matrix=a,
            obs_noise=measurement_noise_std,
            num_particles=num_samples,
        )
    # NOTE: janking in to ensure not user issue for bad results...
    elif method == SamplerName.MCG_DIFF:
        a = _jax_to_torch(a)
        init_obs = _jax_to_torch(init_obs)
        alphas_cumprod = _jax_to_torch(sampler.alphas_cumprod)
        means = [_jax_to_torch(mean) for mean in means]
        weights = _jax_to_torch(weights)

        u, diag, coordinate_mask, v = build_extended_svd_torch(a)

        from functools import partial

        ou_mixt_fun = partial(ou_mixt_torch, means=means, dim=dim_x, weights=weights)
        score_network = lambda x, alpha_t: torch.func.grad(
            lambda y: ou_mixt_fun(alpha_t).log_prob(y).sum()
        )(x)

        score_model = ScoreModel(
            NetReparametrized(
                base_score_module=lambda x, t: -score_network(x, alphas_cumprod[t])
                * ((1 - alphas_cumprod[t]) ** 0.5),
                orthogonal_transformation=v,
            ),
            alphas_cumprod=alphas_cumprod,
            device="cpu",
        )

        adapted_timesteps = get_optimal_timesteps_from_singular_values(
            alphas_cumprod=alphas_cumprod,
            singular_value=diag,
            n_timesteps=100,
            var=measurement_noise_std,
            mode="else",
        )

        def mcg_diff_fun(initial_samples):
            samples, log_weights = mcg_diff(
                initial_particles=initial_samples,
                observation=(u.T @ init_obs),
                score_model=score_model,
                likelihood_diagonal=diag,
                coordinates_mask=coordinate_mask.bool(),
                var_observation=measurement_noise_std,
                timesteps=adapted_timesteps,
                eta=1,
                gaussian_var=1e-8,
            )
            return (
                samples[
                    torch.distributions.Categorical(logits=log_weights, validate_args=False).sample(
                        sample_shape=(num_samples,)
                    )
                ] @ v
            )

        # particle_samples = torch.func.vmap(mcg_diff_fun, in_dims=(0,), randomness="different")(
        #     torch.randn(size=(num_samples, 1000, dim_x))
        # )
        particle_samples = mcg_diff_fun(torch.randn(size=(num_samples, dim_x)))
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
        guided_sampler = PCSampler(shape=(num_samples, dim_x), outer_solver=solver)

    if method != SamplerName.MCG_DIFF:
        particle_samples = guided_sampler.sample(key, y=init_obs, ESSrmin=0.8)

    sliced_wasserstein_distance = sliced_wasserstein(
        dist_1=np.array(posterior_samples),
        dist_2=np.array(particle_samples),
        n_slices=10_000,
    )

    return sliced_wasserstein_distance


def load_existing_results(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename).to_dict("records")
    return []


def save_results(results, filename):
    df = pd.DataFrame(results)
    df.sort_values(["method", "sigma_y", "dim_x", "dim_y", "seed"]).to_csv(filename, index=False)


if __name__ == "__main__":
    results_file = "gmm-2.csv"
    exp_results = load_existing_results(results_file)

    # Convert existing results to a set of tuples for quick lookup
    existing_combinations = set(
        (res["method"], res["sigma_y"], res["dim_x"], res["dim_y"], res["seed"])
        for res in exp_results
    )

    for method, sigma_y, dim_x, dim_y, seed in grid:
        if (method, sigma_y, dim_x, dim_y, seed) in existing_combinations:
            print(f"Skipping: {method=}, {sigma_y=}, {dim_x=}, {dim_y=}, {seed=} (already done)")
            continue

        sliced_wasserstein_distance = experiment(method, dim_x, dim_y, sigma_y, seed)
        exp_result = {
            "method": str(method),
            "sigma_y": sigma_y,
            "dim_x": dim_x,
            "dim_y": dim_y,
            "seed": seed,
            "sliced_wasserstein_distance": sliced_wasserstein_distance,
        }
        exp_results.append(exp_result)

        # Save the current results to the CSV
        save_results(exp_results, results_file)
        print(
            f"Saved results for: method={method}, sigma_y={sigma_y}, dim_x={dim_x}, dim_y={dim_y}"
        )
