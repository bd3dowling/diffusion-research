#!/bin/zsh

# Base directory path
base_dir="presentation/assets"

# Define an array of asset file names (without the extensions)
assets=(
    "branin-adam"
    "branin-adam-larger"
    "branin-backwards"
    "branin-forward"
    "branin-particles"
    "branin-particles-dynamic"
    "gmm-posterior-smc_diff_opt"
    "gmm-posterior-vjp_guidance"
    "gmm-prior"
)

# Loop through each asset and convert the gif to png
for asset in $assets; do
    magick convert -coalesce "${base_dir}/${asset}/${asset}.gif" "${base_dir}/${asset}/${asset}.png"
done
