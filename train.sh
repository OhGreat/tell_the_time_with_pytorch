#/bin/bash

python train.py \
-mode "cse_loss" \
-data_splits 16500 1000 500 \
-bs 64 \
-lr 1e-4 \
-epochs 100 \
-patience 10 \
-weights_name "temp_weights" \
-save_plots \
-v 1 