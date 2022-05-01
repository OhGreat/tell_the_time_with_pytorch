#/bin/bash

python train.py \
-approach "minute_distance" \
-bs 64 \
-lr 1e-5 \
-epochs 200 \
-patience 10 \
-weights_name "Minutes-distance loss approach_2" \
-save_plots \
-v 1