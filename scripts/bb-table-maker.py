import pandas as pd

name_map = {
    "cbas": "CbAS",
    "cma-es": "CMA-ES",
    "gradient-ascent": "Gradient Ascent",
    "reinforce": "REINFORCE",
    "mins": "MINs",
    "ddom": "DDOM",
    "diff-opt": "DiffOpt",
    "smc-diff-opt": "SMCDiffOpt",
}

others = pd.read_csv("superconductor-others.csv", index_col=0)
others_unnorm = pd.read_csv("superconductor-others-unnormalized.csv", index_col=0)

perf = (
    pd.read_csv("superconductor.csv")
    .loc[:, ["score", "score_norm"]]
    .agg(["mean", "std"])
    .transpose()
    .assign(
        val=lambda frame: frame["mean"].round(3).astype(str)
        + " ± "
        + frame["std"].round(3).astype(str)
    )
    .loc[:, ["val"]]
)

perf_row = pd.DataFrame({"superconductor": [perf.loc["score_norm", "val"]]}, index=["smc-diff-opt"])
perf_row_unnorm = pd.DataFrame(
    {"superconductor": [perf.loc["score", "val"]]}, index=["smc-diff-opt"]
)

# Normalized
print(pd.concat([others, perf_row]).transpose().rename(columns=name_map).to_latex(index=False))

# Unnormalized
print(
    pd.concat([others_unnorm, perf_row_unnorm])
    .transpose()
    .rename(columns=name_map)
    .to_latex(index=False)
)
