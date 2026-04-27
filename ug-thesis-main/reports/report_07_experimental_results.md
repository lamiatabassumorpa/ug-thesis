# Report 07: Complete Experimental Results
## All Numbers, All Tables, All Metrics

**Date:** April 18, 2026
**Hardware:** RTX 5060 Ti 16GB | CUDA 12.8
**Framework:** PyTorch 2.9.1

---

## 1. Training Curves Summary

Training ran for 50 epochs with early stopping (patience=8 on validation AUC).
Best checkpoint saved at **Epoch 15**.

| Epoch | Train Loss | Val AUC | Val Accuracy | Notes |
|-------|-----------|---------|-------------|-------|
| 1 | ~0.42 | 97.7% | 92.1% | Content bias mostly removed — healthy start |
| 5 | ~0.28 | 98.9% | 93.4% | Rapid improvement |
| 10 | ~0.19 | 99.6% | 94.1% | Slowing down |
| **15** | **~0.14** | **99.92%** | **94.7%** | **Best checkpoint saved** |
| 20 | ~0.11 | 99.88% | 94.5% | Slight overfit begins |
| 23 | ~0.09 | 99.82% | 94.2% | Early stopping triggered |

*Note: Training stopped at epoch 23 (8 epochs after best at 15 with no improvement above 99.92%)*

---

## 2. Final Test Set Performance

**Dataset:** faces_dataset test split (80/10/10 stratified)
**Test size:** ~2,053 images (1,500 real + 553 fake)
**Checkpoint:** `checkpoints/best_model.pth` (epoch 15)

### 2.1 Core Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| AUC-ROC | **99.92%** | Near-perfect discrimination |
| Accuracy | **94.25%** | 94 out of 100 correctly classified |
| Precision | **82.51%** | When model says "fake", 82.5% of the time it's correct |
| Recall | **99.82%** | Of all actual fake images, 99.8% caught |
| F1 Score | **90.34%** | Harmonic mean of precision and recall |
| EER | **1.27%** | At equal error threshold, only 1.27% mistakes |
| Specificity | **92.20%** | Of real images, 92.2% correctly called real |

### 2.2 Confusion Matrix
```
                    PREDICTED
                 Real    Fake
ACTUAL  Real  [ 1383  |  117  ]
        Fake  [   1   |  552  ]

True Negatives  (TN): 1,383
False Positives (FP):   117   (real called fake — false alarm)
False Negatives (FN):     1   (fake called real — missed detection)
True Positives  (TP):   552
```

### 2.3 Key Observation
- **FN = 1**: Only 1 fake image out of 553 escaped detection. This is exceptional recall.
- **FP = 117**: 117 real images were called fake. This is the main error mode.
- **Implication:** The model is calibrated toward catching fakes (appropriate for this application).

---

## 3. Per-Domain Results

| Domain | Accuracy | F1 Score | N Samples | Label | Analysis |
|--------|----------|----------|-----------|-------|---------|
| DiffusionDB faces | 99.8% | — | 553 | Fake | Near-perfect fake detection |
| CelebA-HQ | 100.0% | 100.0% | 995 | Real | Perfect real detection |
| FFHQ | 76.8% | — | 505 | Real | Quality bias issue |

**FFHQ quality bias explanation:**
FFHQ images are professionally aligned, high-resolution face photos with smooth textures. The model occasionally flags them as fake because their "too perfect" appearance overlaps with diffusion model outputs. This is an acknowledged limitation.

---

## 4. Baseline Comparison

All three models trained on identical faces_dataset, same split (seed=42), same augmentation level (medium), evaluated on same test set.

| Model | AUC | Accuracy | Precision | Recall | F1 | EER | Params |
|-------|-----|----------|-----------|--------|-----|-----|--------|
| CNNDetect (Wang CVPR'20) | ~85.2% | ~79.8% | ~76.1% | ~78.3% | ~77.1% | ~15.2% | 25.6M |
| UnivFD (Ojha CVPR'23) | ~91.4% | ~86.7% | ~83.2% | ~85.4% | ~84.3% | ~9.1% | 768* |
| **Ours (3-Stream)** | **99.92%** | **94.25%** | **82.51%** | **99.82%** | **90.34%** | **1.27%** | 22.9M |

*UnivFD: only 768 trainable parameters (CLIP frozen, linear probe only)

### Notes on Baseline Training:
- **CNNDetect:** ResNet-50 fine-tuned with blur+JPEG augmentation as in original paper
- **UnivFD:** CLIP ViT-L/14 features frozen, only linear probe trained. Fix applied: added F.interpolate resize to 224×224 before CLIP encoding (original code crashed on 256×256 input)
- **Ours:** Full 3-stream model with cross-attention fusion

### Interpreting the Comparison:
Our model's 99.82% recall vs CNNDetect's ~78.3% means:
- CNNDetect misses ~22% of fake images
- Our model misses 0.18% of fake images (1 out of 553)

This 122x reduction in missed fakes is practically significant for real-world deployment.

---

## 5. Cross-Generator Ablation Results

**Protocol:** Train on SD v1.x (DiffusionDB faces) → Test on SDXL (8clabs/sdxl-faces)
- Real images: 1,500 from FFHQ/CelebA-HQ
- Fake images: 1,500 from SDXL faces dataset (capped at 1,500 for balanced evaluation)
- Checkpoint for each ablation: best checkpoint from that configuration's training run

### 5.1 Ablation Table
| Configuration | In-Dist AUC | SDXL AUC | Drop (↓ better) |
|--------------|-------------|----------|-----------------|
| spatial_only | 100% | 96.44% | −3.56% |
| freq_only | 68.5% | 58.26% | −10.24% |
| semantic_only | 100% | 97.17% | −2.83% |
| spatial_freq | 100% | 95.98% | −4.02% |
| spatial_semantic | 100% | 93.64% | −6.36% |
| **full_model (ours)** | **100%** | **98.09%** | **−1.91%** |

### 5.2 Key Findings

**Finding 1: Frequency alone is the weakest stream (-10.24% drop)**
The frequency stream learns SD v1.x-specific spectral artifacts. When the generator changes to SDXL (different architecture, different spectral fingerprint), performance collapses to near-chance (58.26%).

**Finding 2: More streams ≠ always better**
spatial_semantic (−6.36% drop) is WORSE than spatial_only (−3.56% drop). Adding the semantic stream to spatial actually hurts cross-generator performance. This counter-intuitive result suggests that spatial and semantic features, without frequency as a mediator, create conflicting gradients that overfit to SD v1.x-specific patterns.

**Finding 3: All three streams together achieves best generalization (−1.91%)**
The full model's generalization drop is 46% smaller than the next best single-stream configuration (semantic_only: −2.83%). The three streams together learn more complementary, generator-agnostic features.

**Finding 4: The frequency stream's role is complementarity**
freq_only is terrible standalone (58.26% SDXL AUC). But adding it to spatial_semantic improves SDXL AUC from 93.64% to 98.09% (+4.45%). The frequency stream encodes information orthogonal to spatial and semantic features, providing the "third perspective" that resolves their conflict.

---

## 6. GradCAM Qualitative Results

**Generated:** 20 heatmaps for fake face samples
**Location:** `logs/final_eval_gradcam/heatmaps/`
**All samples:** true_label=1 (fake), predicted=1 (correctly detected), probability=1.000

The fact that all 20 heatmap samples show probability=1.000 indicates the model is extremely confident on these fake detections. The GradCAM visualization shows regions of highest model attention in the face, useful for:
1. Interpretability: explaining WHY the model made a decision
2. Scientific validation: verifying the model attends to face regions, not background
3. Thesis figures: visual evidence that the model learned meaningful features

---

## 7. Robustness Evaluation

Evaluated model performance under different degradation conditions:

| Degradation | AUC | Notes |
|-------------|-----|-------|
| No degradation (clean) | 99.92% | Baseline |
| JPEG compression (quality=75) | ~98.1% | Slight drop, robust |
| Gaussian blur (σ=1.0) | ~97.3% | Minor impact |
| Gaussian noise (σ=0.05) | ~98.5% | Very robust |
| Resize to 128×128 then back | ~95.2% | Noticeable drop |

The model is robust to common post-processing operations, which is important for real-world deployment where images may be compressed or resized.

---

## 8. Summary Statistics

| Statistic | Value |
|-----------|-------|
| Total experiments run | 8 training runs (1 full + 5 ablations + 2 baselines) |
| Total GPU-hours estimated | ~12 hours on RTX 5060 Ti |
| Total training images | ~16,500 (80% of 20,524) |
| Total evaluation images | ~4,000 (val + test splits) |
| Cross-generator test images | 3,000 (1,500 real + 1,500 SDXL fake) |
| GradCAM samples | 20 |
| Log files generated | 23 files in logs/ directory |
