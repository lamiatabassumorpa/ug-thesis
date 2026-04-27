#!/bin/bash
source /home/noushad/miniconda3/etc/profile.d/conda.sh
conda activate oneclick
cd /home/noushad/deepfake-detection

echo "=== Cross-Generator Ablation (SD v1.x → SDXL) ==="
echo ""

for MODE in spatial_only freq_only semantic_only spatial_freq spatial_semantic; do
  CKPT="checkpoints/ablation_${MODE}/best_model.pth"
  echo -n "${MODE}: "
  python scripts/cross_generator_eval.py \
    --config configs/config.yaml \
    --checkpoint "$CKPT" \
    --real-dir data/real \
    --fake-dir data/fake/sdxl_faces/imgs \
    --generator-name "SDXL" \
    --max-images 1500 2>&1 | grep "SDXL"
done

echo -n "full_model: "
python scripts/cross_generator_eval.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --real-dir data/real \
  --fake-dir data/fake/sdxl_faces/imgs \
  --generator-name "SDXL" \
  --max-images 1500 2>&1 | grep "SDXL"

echo ""
echo "Done."
