# Report 10: Publication & Future Work Guide
## How to Turn This Into a Published Paper

**Date:** April 18, 2026

---

## 1. Target Venues (Ranked by Fit)

### Tier 1: BSc Thesis (Immediate)
**Requirements:** None beyond university guidelines
**Timeline:** Write now, defend in 2-4 weeks
**Format:** Whatever your department requires (usually 40-80 pages)
**Verdict:** READY. All experiments complete.

---

### Tier 2: ICCIT 2025 (Best Fit for Conference)
**Full name:** 18th International Conference on Computer and Information Technology
**Organizer:** IEEE Bangladesh Section
**Website:** Check ieee.org.bd/iccit
**Format:** IEEE 6-page double-column
**Typical deadline:** August-September 2025
**Notification:** October-November 2025
**Venue:** Bangladesh (Dhaka/BUET usually)

**Why this is the best fit:**
- IEEE-indexed (appears in IEEE Xplore)
- Bangladesh-based (familiar academic culture)
- Accepts papers at this level of contribution regularly
- Image processing/deep learning track always present
- Co-authored work from Bangladeshi institutions welcomed

---

### Tier 3: ICAICT (Backup)
**Full name:** International Conference on Advancement in Computer and Intelligent Technology
**Format:** Similar to ICCIT
**Verdict:** Good backup if ICCIT deadline is missed

---

### Tier 4: IEEE Access (Journal — Stretch Goal)
**Type:** Open-access IEEE journal
**Timeline:** 3-6 months review
**Requirements:** More thorough evaluation (add GenImage generators)
**Verdict:** Achievable with 2-3 additional experiments (ForenSynths + GenImage DALL-E + multi-seed)

---

## 2. Recommended Paper Structure (ICCIT IEEE Format)

### Page Budget: 6 pages maximum

```
Title (with authors and affiliations)
Abstract (150 words)
─────────────────────────────────────────
I.    Introduction                    (0.5 pages)
II.   Related Work                    (0.5 pages)  
III.  Methodology                     (1.5 pages)
IV.   Experiments                     (2.0 pages)
V.    Results & Discussion            (1.0 pages)
VI.   Conclusion & Future Work        (0.5 pages)
References                            (last column)
─────────────────────────────────────────
Total: 6 pages
```

---

## 3. Section-by-Section Writing Guide

### Title
> "Multi-Stream Deepfake Face Detection with Cross-Generator Generalization via Learnable Frequency Fusion"

Key words to include: "multi-stream," "cross-generator," "deepfake detection"

---

### Abstract (write this LAST)
Template:
> "Deepfake face detection systems trained on one generative model often fail to detect faces from unseen generators. We propose a three-stream detection architecture that jointly analyzes spatial artifacts (EfficientNet-B0), frequency signatures (ResNet-18 with learnable FFT), and semantic inconsistencies (ViT-Tiny) through cross-stream multi-head attention fusion. Training exclusively on Stable Diffusion v1.x faces, our model achieves 98.09% AUC on unseen SDXL-generated faces — a degradation of only 1.91% versus 10.24% for a frequency-only baseline. Extensive ablation analysis reveals that the frequency stream, despite weak standalone performance (68.5% AUC), provides orthogonal information that critically improves cross-generator generalization. Our model outperforms CNNDetect and UnivFD baselines with 99.92% AUC on in-distribution testing."

---

### Introduction
Must contain (in this order):
1. **Hook:** Deepfakes are a threat (1-2 sentences, cite a news incident if possible)
2. **Problem statement:** Existing detectors don't generalize across generators
3. **Research gap:** "No prior work systematically analyzes how stream complementarity affects cross-generator generalization"
4. **Our contribution (bullet list):**
   - Multi-stream architecture combining spatial, frequency, semantic analysis
   - Cross-stream attention fusion for inter-stream communication
   - Ablation showing frequency stream provides orthogonal generalization benefit
   - 98.09% AUC on unseen SDXL generator, smallest drop among all configurations

---

### Related Work (cite these specifically)
**Deepfake detection:**
- Wang et al. (2020) CNNDetect — CVPR 2020
- Ojha et al. (2023) UnivFD — CVPR 2023
- Zhao et al. (2021) Multi-attentional deepfake detection — CVPR 2021

**Frequency-domain analysis:**
- Qian et al. (2020) F3-Net — ECCV 2020
- Durall et al. (2020) Watch Your Up-Convolution — CVPR 2020

**Foundation models for detection:**
- Dosovitskiy et al. (2021) ViT — ICLR 2021
- Radford et al. (2021) CLIP — ICML 2021

**Datasets:**
- Rombach et al. (2022) Stable Diffusion — CVPR 2022 (cite for SD v1.x)
- Podell et al. (2023) SDXL — cite arXiv

---

### Methodology Figures (needed in paper)
**Figure 1: Overall Architecture**
```
[Input Face Image]
    ↓          ↓          ↓
[Spatial]  [Frequency] [Semantic]
    ↓          ↓          ↓
[128-dim]  [64-dim]  [384-dim]
    └──────────┴──────────┘
         [Cross-Attention]
                ↓
         [Classifier]
                ↓
         [Real / Fake]
```

**Figure 2: Cross-Generator Ablation Table** (Table 2 in paper)

**Figure 3: 2-3 GradCAM Heatmaps** (select clearest examples)

---

### Experiments Section (Tables)

**Table 1: Dataset Statistics**
| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | 12,000 | 4,419 | 16,419 |
| Val | 1,500 | 552 | 2,052 |
| Test | 1,500 | 553 | 2,053 |
| SDXL eval | 1,500 | 1,920 | 3,420 |

**Table 2: Comparison with Baselines**

**Table 3: Cross-Generator Ablation**

---

## 4. Minimum Additional Work for IEEE Access (Journal)

The current work is conference-ready. For a journal submission:

| Additional Experiment | Effort | Impact |
|----------------------|--------|--------|
| ForenSynths evaluation (ProGAN/StyleGAN faces) | 2-3 hours | +1 cross-generator data point |
| GenImage DALL-E/Midjourney (download ~2GB) | 1-2 days | +2 data points, much stronger claim |
| Multi-seed training (3 seeds, report mean±std) | ~3× compute | Statistical significance |
| Concatenation vs. attention fusion ablation | 1 training run | Addresses examiner Q7 |
| GradCAM per-stream visualization | 2-4 hours | Strong qualitative contribution |

Total estimated additional effort: **1-2 weeks** to reach journal level.

---

## 5. Ethics Statement (Copy-Paste Ready)

> This research was conducted solely for academic purposes using publicly available datasets. The FFHQ dataset was collected by NVIDIA under Creative Commons license; DiffusionDB was published by Wang et al. under CC0. The 8clabs/sdxl-faces dataset is available on HuggingFace Hub under open license. No private or non-consensual face data was used. The deepfake detection system described herein is intended to serve a defensive role — helping protect individuals from non-consensual synthetic media and supporting media authenticity verification. We acknowledge the dual-use potential of detection research and encourage responsible deployment. The performance of this system across demographic groups was not exhaustively evaluated; users should exercise caution before applying it in sensitive, high-stakes contexts.

---

## 6. Open Science Checklist (Before Submission)

- [ ] Upload code to GitHub with README
- [ ] Upload best checkpoint to HuggingFace Hub: `Noushad999/deepfake-detector-3stream`
- [ ] Replace `E:/deepfake-detection` hardcoded path with `${DATA_ROOT:-data}` ✅ Done
- [ ] Add `open-clip-torch` to requirements.txt ✅ Done
- [ ] Create `requirements-lock.txt` with pinned versions ✅ Done
- [ ] Add CUDA version to README (CUDA 12.8, PyTorch 2.9.1) ✅ Done
- [ ] Write reproducibility instructions in README
- [ ] Link to datasets (FFHQ: HuggingFace, DiffusionDB: HuggingFace, SDXL-faces: HuggingFace)

---

## 7. Complete Reference List for Paper

```
[1] Wang, S., et al. "CNN-generated images are surprisingly easy to spot... for now." CVPR 2020.
[2] Ojha, U., et al. "Towards Universal Fake Image Detection by Exploiting CLIP's Potential." CVPR 2023.
[3] Zhao, H., et al. "Multi-attentional deepfake detection." CVPR 2021.
[4] Qian, Y., et al. "Thinking in frequency: Face forgery detection by mining frequency-aware clues." ECCV 2020.
[5] Durall, R., et al. "Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions." CVPR 2020.
[6] Tan, M., Le, Q.V. "EfficientNet: Rethinking model scaling for convolutional neural networks." ICML 2019.
[7] Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." ICLR 2021.
[8] Radford, A., et al. "Learning transferable visual models from natural language supervision." ICML 2021.
[9] Rombach, R., et al. "High-resolution image synthesis with latent diffusion models." CVPR 2022.
[10] Podell, D., et al. "SDXL: Improving latent diffusion models for high-resolution image synthesis." arXiv 2023.
[11] Wang, A., et al. "DiffusionDB: A large-scale prompt gallery dataset for text-to-image generative models." ACL 2023.
[12] Karras, T., et al. "A style-based generator architecture for generative adversarial networks." CVPR 2019.
[13] He, K., et al. "Deep residual learning for image recognition." CVPR 2016.
[14] Shiohara, K., Yamasaki, T. "Detecting deepfakes with self-blended images." CVPR 2022.
```

---

## 8. Timeline: Thesis to Conference

```
Week 1-2  (Now):
  □ Write thesis draft using this project's reports as sections
  □ Add methodology figures (architecture diagram, ablation table)
  □ Select 3 best GradCAM heatmaps for visual analysis section

Week 3:
  □ Thesis review with supervisor
  □ Revisions

Week 4:
  □ Defense preparation using report_05_defense_qa.md
  □ Practice presentation

Week 5-6:
  □ Convert thesis to 6-page IEEE format for ICCIT
  □ Check ICCIT 2025 submission deadline

Week 7-8:
  □ Submit to ICCIT 2025

Optional (for stronger paper):
  □ Download GenImage DALL-E/Midjourney images
  □ Run cross-generator test on additional generators
  □ Submit to IEEE Access
```
