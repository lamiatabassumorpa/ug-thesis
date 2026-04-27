# 📦 CiFake — Phase 1 Dataset (Archived)

This folder contains all code and documentation related to the **CiFake** dataset, which was used in the **very first phase** of this thesis project before switching to FFHQ + DiffusionDB.

---

## What is CiFake?

**CiFake** is a publicly available image dataset on Kaggle that combines:
- **Real images** → CIFAR-10 (60,000 natural photos: animals, vehicles, etc.) — 32×32 px
- **Fake images** → AI-generated versions of CIFAR-10 using Stable Diffusion

| Class  | Source        | Count  | Resolution |
|:-------|:-------------|-------:|:----------:|
| Real   | CIFAR-10      | 50,000 | 32×32 px   |
| Fake   | Stable Diffusion (CIFAR-10 style) | 50,000 | 32×32 px |

🔗 **Kaggle:** https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

---

## Why Was It Used?

Phase 1 goal: quickly prototype and verify the 3-stream architecture worked before scaling to larger, more complex face datasets.

CiFake was attractive because:
- ✅ 100,000 images — large enough to train deep models
- ✅ Clean 50/50 class balance — no pos_weight tuning needed
- ✅ Publicly available on Kaggle with a single download command
- ✅ Fast to download (~1.2 GB)

---

## ❌ Why Was It Abandoned?

| Problem | Detail |
|:--------|:-------|
| **Image size too small** | 32×32 px — too small for meaningful texture/frequency analysis. We had to upscale to 256×256 which introduces artifacts. |
| **Not face images** | CIFAR-10 contains cars, ships, dogs, frogs — NOT faces. This thesis is specifically about face deepfake detection. |
| **Robustness failure** | JPEG compression at Q=75 caused accuracy to drop from ~100% → **67.8%** (−32 percentage points) |
| **Wrong domain** | A face deepfake detector should be trained on faces vs AI-generated faces, not generic objects |
| **Instant 100% AUC** | The model hit 100% AUC in epoch 1 — a red flag indicating the task was trivially easy (texture/style difference, not genuine forgery detection) |

---

## ✅ What Replaced It?

| Dataset         | Type        | Count   | Notes                      |
|:---------------|:-----------:|--------:|:---------------------------|
| FFHQ 256px      | Real faces  | ~15,000 | NVIDIA's Flickr Faces HQ   |
| CelebA-HQ       | Real faces  | ~3,000  | Google celebrity faces     |
| DiffusionDB (face-filtered) | Fake faces (SD v1.x) | 5,524 | OpenCV face-detected |
| SDXL Faces      | Fake (eval) | 1,920   | Cross-generator evaluation |

**Final result with new dataset:** AUC 99.92%, only −1.91% cross-generator drop ✅

---

## Files in This Folder

| File | Description |
|:-----|:------------|
| `download_cifake.py` | Download CiFake from Kaggle using kagglehub |
| `prepare_cifake.py` | Resize images from 32×32 → 256×256 and organize into real/fake folders |
| `dataset_cifake.py` | PyTorch Dataset class for CiFake (adapted from main `data/dataset.py`) |
| `train_cifake.py` | Training script adapted for CiFake |
| `evaluate_cifake.py` | Evaluation + robustness test (JPEG compression drop) |
| `results_summary.md` | Recorded results from Phase 1 CiFake experiments |
