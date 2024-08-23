import pandas as pd

df = pd.read_csv("gmm.csv")

col_names = {
    "sigma_y": r"$\sigma_y$",
    "dim_x": r"$d_x$",
    "dim_y": r"$d_y$",
    "smc_diff_opt": "SMCDiffOpt",
    # "mcg_diff": "MCGDiff",
    "diffusion_posterior_sampling": "DPS",
    "pseudo_inverse_guidance": r"$\Pi$IGD",
    "vjp_guidance": "TMPD",
}

print(
    df
    .groupby(["method", "dim_x", "dim_y"])
    [["sliced_wasserstein_distance"]]
    .agg(["min", "max"])
    ["sliced_wasserstein_distance"]
    .assign(
        midpoint = lambda frame: (frame["max"] + frame["min"]) / 2,
        range_pm = lambda frame: (frame["max"] - frame["min"]) / 2,
        val = lambda frame: (
            frame["midpoint"].round(2).astype(str) + ' ± ' + frame["range_pm"].round(2).astype(str)
        )
    )
    .loc[:, ["val"]]
    .pivot_table(
        index=["dim_x", "dim_y"],
        columns="method",
        values="val",
        aggfunc=lambda x: ''.join(x)
    )
    .reset_index()
    # .reindex(columns=["dim_x", "dim_y", "smc_diff_opt", "mcg_diff", "diffusion_posterior_sampling", "pseudo_inverse_guidance", "vjp_guidance"])
    .reindex(columns=["dim_x", "dim_y", "smc_diff_opt", "diffusion_posterior_sampling", "pseudo_inverse_guidance", "vjp_guidance"])
    .rename(columns=col_names)
    .to_latex(index=False)
)

print(
    df
    .groupby(["method", "sigma_y", "dim_x", "dim_y"])
    [["sliced_wasserstein_distance"]]
    .agg(["min", "max"])
    ["sliced_wasserstein_distance"]
    .assign(
        midpoint = lambda frame: (frame["max"] + frame["min"]) / 2,
        range_pm = lambda frame: (frame["max"] - frame["min"]) / 2,
        val = lambda frame: (
            frame["midpoint"].round(2).astype(str) + ' ± ' + frame["range_pm"].round(2).astype(str)
        )
    )
    .loc[:, ["val"]]
    .pivot_table(
        index=["sigma_y", "dim_x", "dim_y"],
        columns="method",
        values="val",
        aggfunc=lambda x: ''.join(x)
    )
    .reset_index()
    .assign(sigma_y = lambda frame: frame["sigma_y"].round(1).astype(str))
    # .reindex(columns=["sigma_y", "dim_x", "dim_y", "smc_diff_opt", "mcg_diff", "diffusion_posterior_sampling", "pseudo_inverse_guidance", "vjp_guidance"])
    .reindex(columns=["sigma_y", "dim_x", "dim_y", "smc_diff_opt", "diffusion_posterior_sampling", "pseudo_inverse_guidance", "vjp_guidance"])
    .rename(columns=col_names)
    .to_latex(index=False)
)
