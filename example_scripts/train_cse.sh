#/bin/bash

python train.py \
-approach "minute_distance" \
-bs 64 \
-lr 1e-4 \
-epochs 150 \
-patience 10 \
-weights_name "CSE loss approach" \
-save_plots \
-v 1