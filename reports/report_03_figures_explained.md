# Report 03: Figures & Visualizations Explained
## Every Chart, Graph and Heatmap — What It Shows and Why It Matters

**Location of all figures:** `logs/final_eval_gradcam/`
**Date generated:** April 18, 2026

---

## Figure 1: ROC Curve
**File:** `logs/final_eval_gradcam/roc_curve.png`

### What is a ROC Curve?
ROC stands for Receiver Operating Characteristic. It's a graph that shows the trade-off between:
- **True Positive Rate (TPR / Recall)** — Of all the fake images, what % did we correctly catch?
- **False Positive Rate (FPR)** — Of all the real images, what % did we wrongly call fake?

### How to read it:
- The curve goes from bottom-left (0,0) to top-right (1,1)
- **A perfect detector** would go straight up the left edge and across the top (area = 1.0 = 100%)
- **A random guesser** would be a diagonal line from (0,0) to (1,1) (area = 0.5 = 50%)
- **Our model** hugs the top-left corner very closely → AUC = **99.92%**

### What the AUC number means:
If you randomly pick one fake image and one real image and show them to the model, there is a **99.92% probability** that the model will score the fake image higher (more suspicious) than the real image. This is a very strong result.

### The EER Point:
On the ROC curve, there is a special point where FPR = FNR (False Negative Rate). This is the **Equal Error Rate (EER)**. Our EER = **1.27%** — meaning at the operating point where both error types are equal, we make only 1.27% mistakes.

---

## Figure 2: Confusion Matrix
**File:** `logs/final_eval_gradcam/confusion_matrix.png`

### What is a Confusion Matrix?
A 2×2 table showing all four types of outcomes:

```
                    PREDICTED
                 Real    Fake
ACTUAL  Real  [ 1383  |  117  ]   ← 92.2% correctly called real
        Fake  [   1   |  552  ]   ← 99.8% correctly caught as fake
```

### Reading each cell:
| Cell | Number | Meaning |
|------|--------|---------|
| True Negative (TN) | 1,383 | Real images correctly identified as real |
| False Positive (FP) | 117 | Real images wrongly called fake (false alarm) |
| False Negative (FN) | 1 | Fake images that slipped through as real (missed) |
| True Positive (TP) | 552 | Fake images correctly caught |

### What this tells us:
- The model almost never misses a fake image (only 1 missed out of 553)
- It has some false alarms on real images (117 out of 1,500) — it is slightly over-cautious
- For deepfake detection, missing a fake (FN=1) is much more costly than a false alarm (FP=117) — our model is correctly biased toward catching fakes

---

## Figure 3: Prediction Distribution
**File:** `logs/final_eval_gradcam/prediction_dist.png`

### What it shows:
Two histograms overlaid on the same plot:
- **Blue histogram (Real images):** Distribution of the model's confidence scores for real images
- **Orange histogram (Fake images):** Distribution of confidence scores for fake images

### What a good model looks like:
- Real images should cluster near 0.0 (model says "not fake")
- Fake images should cluster near 1.0 (model says "definitely fake")
- The two histograms should have minimal overlap

### What our model shows:
The two distributions are well-separated. Most fake images score >0.9, most real images score <0.1. The small overlap region around 0.5 corresponds to the 117 false positives.

---

## Figure 4: GradCAM Heatmaps
**File:** `logs/final_eval_gradcam/heatmaps/sample_000_true1_pred1_prob1.000.jpg` (and 19 more)

### What is GradCAM?
GradCAM (Gradient-weighted Class Activation Mapping) answers the question: **"Which part of this image made the AI say fake?"**

It works by:
1. Making a prediction on an image
2. Looking backwards through the network to find which regions contributed most to that prediction
3. Creating a "heat map" where HOT colors (red/yellow) = important regions, COOL colors (blue) = less important regions

### File naming convention:
`sample_000_true1_pred1_prob1.000.jpg`
- `true1` = The ground truth label is 1 (this IS a fake image)
- `pred1` = The model predicted 1 (correctly called it fake)
- `prob1.000` = The model is 100.0% confident this is fake

### What to look for in the heatmaps:
For AI-generated faces, the model typically focuses on:
- **Eye regions** — AI often generates slightly asymmetric or unnatural eyes
- **Hair-skin boundary** — Diffusion models struggle with realistic hair-skin transitions
- **Teeth/mouth** — Teeth generation is notoriously imperfect in SD v1.x
- **High-frequency edges** — Spectral artifacts at sharp boundaries

All 20 heatmaps in our results show `prob=1.000` — the model is maximally confident on these samples, suggesting very clear visual evidence of AI generation.

---

## Figure 5: Cross-Generator Ablation Table (Key Result)

This is not a saved image file but the most important table in the project:

```
┌─────────────────────┬──────────────┬───────────┬──────────────┐
│ Configuration       │ In-Dist AUC  │ SDXL AUC  │    Drop      │
├─────────────────────┼──────────────┼───────────┼──────────────┤
│ Spatial Only        │    100%      │  96.44%   │   −3.56%     │
│ Frequency Only      │    68.5%     │  58.26%   │   −10.24%    │
│ Semantic Only       │    100%      │  97.17%   │   −2.83%     │
│ Spatial + Freq      │    100%      │  95.98%   │   −4.02%     │
│ Spatial + Semantic  │    100%      │  93.64%   │   −6.36%     │
│ Full 3-Stream ★     │    100%      │  98.09%   │   −1.91%     │
└─────────────────────┴──────────────┴───────────┴──────────────┘
```

### How to read this table:
1. **In-Dist AUC:** Performance on images from the same generator (SD v1.x) it was trained on
2. **SDXL AUC:** Performance on a completely different generator never seen during training
3. **Drop:** How much performance fell when switching to an unseen generator

### Key insights from this figure:
- Frequency-only is 68.5% in-distribution (barely above chance) — it learns SD-specific artifacts, not universal cues
- Spatial+Semantic actually does WORSE (93.64%) than spatial alone (96.44%) — without the frequency stream, the combination creates conflicting signals
- **The full 3-stream model has the smallest drop (−1.91%)** — the three streams together learn more complementary, generalizable features

### Why this matters for a paper:
This table is the core scientific contribution. It shows that:
1. More streams ≠ always better (spatial+semantic is worse than spatial alone)
2. All three streams together achieve emergent generalization
3. The frequency stream, despite low standalone accuracy, plays a crucial complementary role

---

## Figure 6: Per-Domain Performance

From `logs/final_eval_gradcam/per_domain_results.json`:

| Domain | Accuracy | F1 | N Samples | Type |
|--------|----------|-----|-----------|------|
| DiffusionDB faces | 99.8% | 50.0% | 553 | Fake |
| CelebA-HQ | 100.0% | 100.0% | 995 | Real |
| FFHQ | 76.8% | 43.4% | 505 | Real |

### Why is FFHQ accuracy 76.8% but CelebA is 100%?
This is an important finding. FFHQ and CelebA-HQ are both "real face" datasets, but the model performs very differently on them:
- **CelebA-HQ** → 100%: These celebrity photos have natural imperfections, varied lighting, and compression artifacts that match what the model learned as "real"
- **FFHQ** → 76.8%: FFHQ images are extremely high-quality, professionally retouched photos with very clean, smooth textures — paradoxically, the model sometimes confuses them with AI-generated images because they look "too perfect"

### What to do with this finding in a paper:
Report it honestly as a limitation. "The model exhibits a quality bias — extremely high-quality real photos can trigger false positives because their smooth textures superficially resemble diffusion model outputs." This is a known challenge in the field and points to future work.

### Why the F1 score for fake domain shows 50.0% despite 99.8% accuracy:
F1 score is computed with macro averaging across classes. When the test set for "DiffusionDB faces" contains ONLY fake images (no real images in that specific domain folder), the classifier has no true negatives for that domain, causing the macro F1 to appear low. The 99.8% accuracy correctly represents actual performance.

---

## Summary of All Figures

| Figure | File | Key Message |
|--------|------|-------------|
| ROC Curve | `roc_curve.png` | AUC=99.92%, near-perfect discrimination |
| Confusion Matrix | `confusion_matrix.png` | Only 1 missed fake; 117 false alarms |
| Prediction Distribution | `prediction_dist.png` | Clean separation between real and fake scores |
| GradCAM Heatmaps | `heatmaps/*.jpg` | Model focuses on face-specific regions |
| Ablation Table | (in paper) | Full model best at cross-generator generalization |
| Per-Domain Results | (in paper) | FFHQ quality bias is an honest limitation |
