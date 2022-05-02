#/bin/bash

python train.py \
-data_aug \
-approach "minute_distance" \
-bs 64 \
-lr 1e-4 \
-epochs 200 \
-patience 10 \
-weights_name "Minutes-distance loss approach" \
-save_plots \
-v 1