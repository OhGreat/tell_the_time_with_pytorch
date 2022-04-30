#/bin/bash

python train.py \
-approach "cse_loss" \
-bs 64 \
-lr 1e-5 \
-epochs 150 \
-patience 10 \
-weights_name "CSE loss" \
-save_plots \
-v 1