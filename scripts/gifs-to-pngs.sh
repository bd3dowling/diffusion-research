#!/bin/zsh

magick convert -coalesce \
    presentation/assets/gmm-posterior-smc_diff_opt/gmm-posterior-smc_diff_opt.gif \
    presentation/assets/gmm-posterior-smc_diff_opt/gmm-posterior-smc_diff_opt.png

magick convert -coalesce \
    presentation/assets/gmm-posterior-vjp_guidance/gmm-posterior-vjp_guidance.gif \
    presentation/assets/gmm-posterior-vjp_guidance/gmm-posterior-vjp_guidance.png

magick convert -coalesce \
    presentation/assets/gmm-prior/gmm-prior.gif \
    presentation/assets/gmm-prior/gmm-prior.png
