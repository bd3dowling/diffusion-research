# Imports
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import optax
from jax import random
from jaxtyping import Array, Integer, PRNGKeyArray, PyTree, Real
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from optax import GradientTransformation

from diffusionlib.optimizer import SMCDiffOptOptimizer
from diffusionlib.sampler.ddim import DDIMVP

# Define Branin function
# Branin
A = 1
B = 5.1 / (4 * jnp.pi**2)
C = 5 / jnp.pi
R = 6
S = 10
T_BRANIN = 1 / (8 * jnp.pi)

# Plotting
MIN_X_1 = -8
MAX_X_1 = 12
MIN_X_2 = -6
MAX_X_2 = 18

# Elliptical region
ELLIPSE_CENTRE = jnp.array([-0.2, 7.5])
ELLIPSE_SEMI_AXES = jnp.array([3.6, 8.0])
ELLIPSE_ANGLE_DEG = 25
ELLIPSE_ANGLE_RAD = jnp.deg2rad(ELLIPSE_ANGLE_DEG)

# Training
NUM_SAMPLES = 6_000
NUM_EPOCHS = 20_000

# Diffusion
BETA_MIN = 0.0001
BETA_MAX = 0.02
T = jnp.array(1000)

# Config
key = random.PRNGKey(100)
matplotlib.rcParams["animation.embed_limit"] = 2**128


class FullyConnectedWithTime(eqx.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time
    variable.
    """

    layers: list[eqx.nn.Linear]

    def __init__(self, in_size: int, key: PRNGKeyArray):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        out_size = in_size

        self.layers = [
            eqx.nn.Linear(in_size + 4, 256, key=key1),
            eqx.nn.Linear(256, 256, key=key2),
            eqx.nn.Linear(256, 256, key=key3),
            eqx.nn.Linear(256, out_size, key=key4),
        ]

    def __call__(self, x: Array, t: Array) -> Real[Array, "1"]:
        t_fourier = jnp.array(
            [t - 0.5, jnp.cos(2 * jnp.pi * t), jnp.sin(2 * jnp.pi * t), -jnp.cos(4 * jnp.pi * t)],
        )

        x = jnp.concatenate([x, t_fourier])

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))

        x = self.layers[-1](x)

        return x


# Define Branin function
def get_branin(
    a: float = A, b: float = B, c: float = C, r: float = R, s: float = S, t: float = T_BRANIN
) -> Callable[[Real[Array, "..."], Real[Array, "..."]], Real[Array, "..."]]:
    def branin(x_1: Real[Array, "..."], x_2: Real[Array, "..."]) -> Real[Array, "..."]:
        return a * (x_2 - b * x_1**2 + c * x_1 - r) ** 2 + s * (1 - t) * jnp.cos(x_1) + s

    return branin


def get_branin_vec(
    a: float = A, b: float = B, c: float = C, r: float = R, s: float = S, t: float = T_BRANIN
) -> Callable[[Real[Array, "..."]], Real[Array, "..."]]:
    def branin(x: Real[Array, "..."]) -> Real[Array, "..."]:
        return (
            a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2
            + s * (1 - t) * jnp.cos(x[:, 0])
            + s
        )

    return branin


# Adam optimizer for parallel optimization
def optimize_with_adam_and_trajectories(
    func: Callable[[Array, Array], Array],
    initial_points: Array,
    learning_rate: float = 0.01,
    num_iterations: int = 1000,
) -> Array:
    # Create the Adam optimizer
    optimizer = optax.adam(learning_rate)

    # Initialize the optimizer's state for all particles
    opt_state = jax.vmap(optimizer.init)(initial_points)

    # Initialize trajectory recording
    trajectories = [initial_points]  # List to store all particle positions at each step

    # Define the optimization loop
    def loop_body(i, loop_carry):
        params, opt_state = loop_carry
        grads = jax.vmap(jax.grad(lambda p: func(p[0], p[1])))(params)
        updates, opt_state = jax.vmap(optimizer.update)(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    params = initial_points
    for i in range(num_iterations):
        params, opt_state = loop_body(i, (params, opt_state))
        trajectories.append(params)  # Record the current positions

    return jnp.stack(trajectories)  # Stack trajectories into an array


# Generate data points
def sample_unit_circle_uniform(n: int, key: PRNGKeyArray) -> Real[Array, "..."]:
    sub_key_1, sub_key_2 = random.split(key)

    angles = random.uniform(sub_key_1, (n,), minval=0, maxval=2 * jnp.pi)
    radii = jnp.sqrt(random.uniform(sub_key_2, (n,)))
    x = radii * jnp.cos(angles)
    y = radii * jnp.sin(angles)
    return jnp.vstack((x, y)).T


def unit_circle_to_ellipse(
    unit_circle_points: Real[Array, "batch 2"],
    semi_axes: Real[Array, "2"] = ELLIPSE_SEMI_AXES,
    angle_rad: Real[Array, ""] = ELLIPSE_ANGLE_RAD,
    center: Real[Array, "2"] = ELLIPSE_CENTRE,
) -> Real[Array, "batch 2"]:
    # Scale points to the ellipse dimensions
    ellipse_points = unit_circle_points * semi_axes

    # Rotation matrix for the given angle
    rotation_matrix = jnp.array(
        [[jnp.cos(angle_rad), -jnp.sin(angle_rad)], [jnp.sin(angle_rad), jnp.cos(angle_rad)]]
    )

    # Rotate the points
    rotated_points = ellipse_points.dot(rotation_matrix.T)

    # Translate the points to the specified center
    return rotated_points + center


# Define the diffusion
def beta_t(
    t: Real[Array, " batch"],
    beta_min: Real[Array, ""] = BETA_MIN,
    beta_max: Real[Array, ""] = BETA_MAX,
) -> Real[Array, " batch"]:
    return beta_min + t * (beta_max - beta_min) / T


def alpha_t(t: Real[Array, " batch"]) -> Real[Array, " batch"]:
    return 1 - beta_t(t)


alpha = alpha_t(jnp.arange(T + 1))
cumulative_alpha_values = jnp.cumprod(alpha)


def a_t(t: Integer[Array, " batch"]) -> Real[Array, " batch"]:
    return jnp.sqrt(alpha_t(t))


def b_t(t: Integer[Array, " batch"]) -> Real[Array, " batch"]:
    return jnp.sqrt(beta_t(t))


def c_t(t: Integer[Array, " batch"]) -> Real[Array, " batch"]:
    return jnp.sqrt(cumulative_alpha_values[t])


def d_t(t: Integer[Array, " batch"]) -> Real[Array, " batch"]:
    return jnp.sqrt(1 - cumulative_alpha_values[t])


def u_t(t: Integer[Array, " batch"]) -> Real[Array, " batch"]:
    return jnp.sqrt(cumulative_alpha_values[t - 1])


def v_t(t: Integer[Array, " batch"]) -> Real[Array, " batch"]:
    return -jnp.sqrt(alpha_t(t)) * (1 - cumulative_alpha_values[t - 1])


def w_t(t: Integer[Array, " batch"]) -> Real[Array, " batch"]:
    return jnp.sqrt(
        beta_t(t) * (1 - cumulative_alpha_values[t - 1]) / (1 - cumulative_alpha_values[t])
    )


def forward_marginal(
    key: PRNGKeyArray, x_0: Real[Array, "batch dim"], t: Integer[Array, " batch"]
) -> Real[Array, "batch dim"]:
    return c_t(t) * x_0 + d_t(t) ** 2 * random.normal(key, x_0.shape)


@jax.jit
@jax.value_and_grad
def loss(model: FullyConnectedWithTime, data: Array, key: PRNGKeyArray) -> Real[Array, ""]:
    key1, key2 = random.split(key, 2)

    random_times = random.randint(key1, (data.shape[0],), minval=0, maxval=T)

    # NOTE: noise will match as both use key2
    noise = random.normal(key2, data.shape)
    noised_data = forward_marginal(key2, data, random_times[:, jnp.newaxis])

    # NOTE: rescale time to in [0, 1]
    output = jax.vmap(model)(noised_data, random_times / (T - 1))

    loss = jnp.mean((noise - output) ** 2)

    return loss


def fit(
    model: FullyConnectedWithTime,
    steps: int,
    optimizer: GradientTransformation,
    data: Array,
    rng: PRNGKeyArray,
    print_every: int = 5_000,
) -> FullyConnectedWithTime:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses: list[float] = []

    @eqx.filter_jit
    def make_step(
        model: FullyConnectedWithTime,
        opt_state: PyTree,
        data: Array,
        step_rng: PRNGKeyArray,
    ):
        loss_value, grads = loss(model, data, step_rng)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss_value

    for step in range(steps):
        step_rng, rng = random.split(rng, 2)
        model, opt_state, train_loss = make_step(
            model,
            opt_state,
            data,
            step_rng,
        )
        losses.append(train_loss)

        if (step % print_every) == 0 or (step == steps - 1):
            mean_loss = jnp.mean(jnp.array(losses))
            print(f"{step=},\t avg_train_loss={mean_loss}")

    return model


def plot_branin_contour(
    x_1_meshpoints: Real[Array, "x_1 x_2"],
    x_2_meshpoints: Real[Array, "x_1 x_2"],
    branin_values: Real[Array, "x_1 x_2"],
    minima_points: Real[Array, "num_minima 2"],
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(8, 6))

    cp = ax.contourf(x_1_meshpoints, x_2_meshpoints, branin_values, levels=50, cmap="coolwarm")
    fig.colorbar(cp, ax=ax)

    ax.contour(
        x_1_meshpoints, x_2_meshpoints, branin_values, levels=50, colors="black", linewidths=0.5
    )

    ax.scatter(
        minima_points[:, 0],
        minima_points[:, 1],
        color="red",
        zorder=5,
        marker="x",
        s=100,
        label="Globabl minima",
    )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(MIN_X_1, MAX_X_1)
    ax.set_ylim(MIN_X_2, MAX_X_2)
    ax.legend()

    return fig, ax


# Define ellipse
def get_ellipse() -> Ellipse:
    return Ellipse(
        xy=ELLIPSE_CENTRE,
        width=2 * ELLIPSE_SEMI_AXES[0],
        height=2 * ELLIPSE_SEMI_AXES[1],
        angle=ELLIPSE_ANGLE_DEG,
        edgecolor="lightgreen",
        facecolor="none",
        linewidth=2,
        zorder=6,
        label="Data Manifold",
    )


def add_trajectories(ax: Axes, trajectories: Real[Array, "num_particles num_steps 2"]) -> Axes:
    # Plot the trajectories of the particles
    for i in range(trajectories.shape[1]):  # Iterate over each particle
        ax.plot(
            trajectories[:, i, 0],
            trajectories[:, i, 1],
            linestyle="-",
            color="orange",
            label="Trajectory" if i == 0 else "",
            zorder=7,
        )
        ax.scatter(
            trajectories[0, i, 0],
            trajectories[0, i, 1],
            color="black",
            marker="o",
            s=5,
            label="Start" if i == 0 else "",
            zorder=8,
        )
        ax.scatter(
            trajectories[-1, i, 0],
            trajectories[-1, i, 1],
            color="orange",
            marker="o",
            s=5,
        )

    ax.legend()

    return ax

if __name__ == "__main__":
    # Create a grid of points
    x_1_points = jnp.linspace(MIN_X_1, MAX_X_1, 2000)
    x_2_points = jnp.linspace(MIN_X_2, MAX_X_2, 2000)
    x_1_meshpoints, x_2_meshpoints = jnp.meshgrid(x_1_points, x_2_points)

    # Evaluate the Branin function at each grid point
    branin_values = get_branin()(x_1_meshpoints, x_2_meshpoints)

    minima_points = jnp.array([(-jnp.pi, 12.275), (jnp.pi, 2.275), (9.42478, 2.475)])

    # Get Adam trajectories
    start_points = random.normal(key, (10, 2))

    adam_trajectories = optimize_with_adam_and_trajectories(
        get_branin(),
        start_points,
        learning_rate=0.05,
        num_iterations=1000,
    )

    adam_trajectories_2 = optimize_with_adam_and_trajectories(
        get_branin(),
        4 * start_points,
        learning_rate=0.05,
        num_iterations=1000,
    )

    # Plot the contour with trajectories
    fig, ax = plot_branin_contour(x_1_meshpoints, x_2_meshpoints, branin_values, minima_points)
    ax.add_patch(get_ellipse())
    add_trajectories(ax, adam_trajectories)
    plt.savefig("paper/assets/adam_branin.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plot_branin_contour(x_1_meshpoints, x_2_meshpoints, branin_values, minima_points)
    ax.add_patch(get_ellipse())
    add_trajectories(ax, adam_trajectories_2)
    plt.savefig("paper/assets/adam_larger_branin.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Train model
    key, sub_key_1, sub_key_2, sub_key_3 = random.split(key, 4)

    unit_circle_points = sample_unit_circle_uniform(NUM_SAMPLES, sub_key_1)
    ellipse_points = unit_circle_to_ellipse(unit_circle_points)

    model_optimizer = optax.adamw(learning_rate=1e-3)
    denoiser_model = FullyConnectedWithTime(2, key=sub_key_2)
    fitted_model = fit(denoiser_model, NUM_EPOCHS, model_optimizer, ellipse_points, sub_key_3)

    # Get sampler
    sampler = DDIMVP(
        num_steps=T,
        shape=(1000, 2),
        model=jax.vmap(fitted_model),  # assumes epsilon model (not score), so okay here!
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        eta=1.0,  # NOTE: equates to using DDPM
    )

    # Prior samples
    key, sub_key = random.split(key)
    prior_samples = sampler.sample(sub_key)

    # Get optimizer
    optimizer = SMCDiffOptOptimizer(base_sampler=sampler, gamma_t=lambda t: 1 - d_t(t))

    # Get optimizer trajectories
    particle_samples = jnp.array(optimizer.optimize(key, get_branin_vec(), stack_samples=True))

    # Prior samples plot
    fig, ax = plot_branin_contour(x_1_meshpoints, x_2_meshpoints, branin_values, minima_points)
    ax.add_patch(get_ellipse())
    ax.scatter(
        x=prior_samples[:, 0],
        y=prior_samples[:, 1],
        alpha=0.8,
        color="orange",
        edgecolors="black",
        lw=0.5,
        s=10,
        label="Prior Samples",
    )
    ax.legend()
    plt.savefig("paper/assets/smc_branin_prior.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Particle samples plot
    fig, ax = plot_branin_contour(x_1_meshpoints, x_2_meshpoints, branin_values, minima_points)
    ax.add_patch(get_ellipse())
    ax.scatter(
        x=particle_samples[-1, :, 0],
        y=particle_samples[-1, :, 1],
        alpha=0.8,
        color="orange",
        edgecolors="black",
        lw=0.5,
        s=10,
        label="Particles",
    )
    ax.legend()
    plt.savefig("paper/assets/smc_branin.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot the mean values over time
    branin = get_branin_vec()
    particle_branins = branin(particle_samples.reshape(-1, 2)).reshape(1000, 1000)
    mean_particle_branins = jnp.mean(particle_branins, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(mean_particle_branins, label="Particle Mean")
    ax.axhline(y=0.397887, color="red", linestyle="--", label="Global Minimum")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Branin Value")

    ticks = jnp.arange(0, 1001, 200)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks[::-1])

    ax.legend()
    ax.grid()

    # Display the plot
    plt.tight_layout()
    plt.savefig("paper/assets/smc_branin_mean_val.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
