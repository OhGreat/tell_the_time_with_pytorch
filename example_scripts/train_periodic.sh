#/bin/bash

python train.py \
-data_aug \
-approach "periodic_labels" \
-bs 64 \
-lr 1e-4 \
-epochs 200 \
-patience 10 \
-weights_name "Periodic labels approach" \
-save_plots \
-v 1
