#!/bin/bash
source /home/noushad/miniconda3/etc/profile.d/conda.sh
conda activate oneclick
cd /home/noushad/deepfake-detection
DATA="/home/noushad/deepfake-detection/data/faces_dataset"

for MODE in spatial_only freq_only semantic_only spatial_freq spatial_semantic; do
  echo "=== Ablation: $MODE ==="
  python scripts/train.py \
    --config configs/config.yaml \
    --data-dir "$DATA" \
    --ablation-mode "$MODE" \
    2>&1 | tee "logs/train_ablation_${MODE}_clean.log"
  echo "Done: $MODE"
done
echo 'ALL ABLATIONS COMPLETE'
