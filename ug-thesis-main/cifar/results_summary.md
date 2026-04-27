# CiFake Phase 1 — Results Summary

> **Status:** Phase 1 Complete — Dataset Replaced  
> This is the archived result from when the project used CiFake as its primary dataset.

---

## Dataset Info

| Property | Value |
|:---------|:------|
| **Name** | CiFake (Real and AI-Generated Synthetic Images) |
| **Source** | Kaggle: `birdy654/cifake-real-and-ai-generated-synthetic-images` |
| **Real class** | CIFAR-10 (50,000 natural photos: 10 categories) |
| **Fake class** | Stable Diffusion generated (50,000 CIFAR-10 style images) |
| **Original resolution** | 32×32 px → upscaled to 256×256 for training |
| **Class balance** | Perfectly balanced (50k / 50k) |

---

## Standard Metrics (Clean Test Set)

*Recorded from Phase 1 training run (epoch 15, seed=42)*

| Metric | Value |
|:-------|------:|
| AUC-ROC | ~100% |
| Accuracy | ~99.8% |
| F1 Score | ~99.8% |
| Recall | ~99.9% |

> ⚠️ Near-100% AUC in first few epochs is a **red flag** — the model was learning texture/style differences, not genuine forgery semantics.

---

## ❌ Robustness Under JPEG Compression

This is the key finding that caused the dataset to be abandoned:

| JPEG Quality | AUC | Drop |
|:------------|----:|-----:|
| Clean | ~100% | — |
| Q=100 | ~99.9% | −0.1pp |
| Q=90 | ~95.2% | −4.8pp |
| **Q=75** | **~67.8%** | **−32.2pp** ❌ |
| Q=50 | ~58.4% | −41.6pp ❌ |

**Social media and messaging apps typically apply JPEG compression at Q=75-85.**  
A detector that drops to 67.8% at Q=75 is essentially useless for real-world deployment.

---

## Root Cause Analysis

| Issue | Explanation |
|:------|:------------|
| **Small images** | 32×32 CIFAR-10 images contain very little spatial detail. The model latches onto upscaling artifacts (introduced during 32→256 resize), not genuine forgery artifacts. |
| **Wrong domain** | CIFAR-10 = cars, ships, frogs, dogs — NOT faces. The thesis goal is face deepfake detection. |
| **Generator frequency artifacts are fragile** | Stable Diffusion leaves characteristic high-frequency patterns in the DCT spectrum. JPEG compression removes these patterns, causing the model to fail. |
| **No generalization challenge** | CiFake uses only one generator (SD). There's no way to test cross-generator generalization. |

---

## Decision: Switch to FFHQ + DiffusionDB

| Criterion | CiFake ❌ | FFHQ + DiffusionDB ✅ |
|:----------|:--------:|:--------------------:|
| Face images | ❌ No | ✅ Yes |
| Resolution | ❌ 32px | ✅ 256px |
| JPEG robustness (Q=75) | ❌ 67.8% | ✅ ~96%+ |
| Cross-generator eval | ❌ None | ✅ SDXL (unseen) |
| Academic credibility | ⚠️ Toy dataset | ✅ Publication-grade |

---

## Scripts in This Folder

```bash
# 1. Download CiFake from Kaggle
python cifar/download_cifake.py

# 2. Resize 32x32 → 256x256 and organize
python cifar/prepare_cifake.py

# 3. Train MS-DFD on CiFake
python cifar/train_cifake.py --data-dir cifar/data --epochs 30

# 4. Evaluate + robustness test
python cifar/evaluate_cifake.py \
    --checkpoint cifar/checkpoints/best_model_cifake.pth \
    --data-dir cifar/data
```
