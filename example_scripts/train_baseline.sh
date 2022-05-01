#/bin/bash

python train.py \
-approach "baseline" \
-bs 64 \
-lr 1e-4 \
-epochs 200 \
-patience 10 \
-weights_name "Baseline approach" \
-save_plots \
-v 1