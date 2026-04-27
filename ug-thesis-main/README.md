<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:1a1a3e,70:cc785c,100:0d1117&height=240&section=header&text=Multi-Stream%20Deepfake%20Detection&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=42&desc=Spatial%20%C2%B7%20Frequency%20%C2%B7%20Semantic%20Fusion%20with%20Cross-Generator%20Generalization&descSize=17&descFontColor=cccccc&descAlignY=63" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com/api?font=JetBrains+Mono&size=17&duration=2600&pause=1400&color=CC785C&center=true&vCenter=true&width=800&lines=AUC+99.92%25+In-Distribution+%7C+AUC+98.09%25+on+Unseen+SDXL+Generator;Only+-1.91%25+Cross-Generator+Drop+vs+-10.24%25+Frequency-Only+Baseline;Spatial+%2B+Frequency+%2B+Semantic+Fusion+via+Cross-Stream+Attention;22.9M+Parameters+%7C+Epoch+15+Best+%7C+RTX+5060+Ti+16GB+CUDA+12.8" alt="Animated metrics"/>

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-Academic_Research-6e7681?style=for-the-badge&logoColor=white)](LICENSE)

[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-99.92%25-2ea043?style=for-the-badge&logo=checkmarx&logoColor=white)]()
[![EER](https://img.shields.io/badge/EER-1.27%25-2ea043?style=for-the-badge&logo=checkmarx&logoColor=white)]()
[![Cross‑Gen AUC](https://img.shields.io/badge/Cross--Gen_AUC_(SDXL)-98.09%25-0969da?style=for-the-badge)]()
[![Params](https://img.shields.io/badge/Parameters-22.9M-cc785c?style=for-the-badge)]()
[![Epoch](https://img.shields.io/badge/Best_Epoch-15-8957e5?style=for-the-badge)]()

</div>

---

## How It Works — 3-Stream Architecture

<p align="center">
  <img src="assets/streams.svg" width="100%" alt="Animated 3-Stream Pipeline"/>
</p>

Each input face is processed simultaneously by three independent streams, each capturing a different class of forgery evidence:

<table>
<tr>
<td width="33%" align="center">
<h3>🔷 Spatial Stream</h3>
<b>EfficientNet-B0 → 128-dim</b><br/><br/>
Detects <b>pixel-level texture artifacts</b> and <b>boundary inconsistencies</b> — the visible seams and blending errors left by generative upsampling.
</td>
<td width="33%" align="center">
<h3>🟠 Frequency Stream</h3>
<b>ResNet-18 + Learnable FFT Mask → 64-dim</b><br/><br/>
Detects <b>spectral anomalies</b> and <b>periodic noise patterns</b> in the frequency domain — invisible to the eye but present in the Fourier spectrum of every generated image.
</td>
<td width="33%" align="center">
<h3>🟣 Semantic Stream</h3>
<b>ViT-Tiny (FAT-Lite) → 384-dim</b><br/><br/>
Detects <b>high-level structural inconsistencies</b> — unnatural face geometry, irregular landmark relationships, and global compositional errors.
</td>
</tr>
</table>

All three feature vectors are fused by a **Multi-Level Attention Fusion (MLAF)** module via cross-stream attention (3-token sequence, 4 heads, 256-dim projection), producing the final P(fake) score.

---

## Abstract

AI-generated face imagery (deepfakes) poses escalating threats to digital trust, media integrity, and identity security. Existing single-stream detectors trained on one generative model family consistently fail to generalize to novel, unseen generators — a critical limitation for real-world deployment where the threat landscape is constantly evolving.

We present **MS-DFD** (*Multi-Stream Deepfake Face Detector*), a unified framework that fuses three orthogonal views of forgery evidence through a **Multi-Level Attention Fusion (MLAF)** module. The cross-stream attention enables the model to weight each stream's contribution dynamically per sample, learning generator-agnostic forgery representations that transfer to unseen generators.

**Key result:** Our model achieves only **−1.91% generalization drop** on SDXL (never seen during training), versus **−10.24%** for a frequency-only baseline — a **5.3× improvement** in cross-generator robustness.

---

## Performance at a Glance

<div align="center">

| Metric | **MS-DFD (Ours)** | UnivFD (CVPR '23) | CNNDetect (CVPR '20) |
|:-------|:-----------------:|:-----------------:|:--------------------:|
| AUC-ROC | **99.92%** | 91.4% | 85.2% |
| Accuracy | **94.25%** | 86.7% | 79.8% |
| Recall | **99.82%** | 85.4% | 78.3% |
| F1 Score | **90.34%** | 84.3% | 77.1% |
| EER | **1.27%** | 9.1% | 15.2% |
| Cross-Gen AUC (SDXL) | **98.09%** | — | — |
| Generalization Drop | **−1.91%** | — | — |

*All models trained and evaluated on the same face dataset (stratified split, seed=42). No data leakage.*

</div>

---

## Architecture

```mermaid
flowchart LR
    INPUT[/"🖼️ Input Face\n256 × 256"/]

    subgraph SP ["🔷 Spatial Stream  ·  NPR Branch"]
        direction TB
        sp1["EfficientNet-B0\nImageNet Pretrained"]
        sp2["FC → 128-dim\nFeature Vector"]
        sp1 --> sp2
    end

    subgraph FR ["🟠 Frequency Stream  ·  FreqBlender"]
        direction TB
        fr1["Learnable\nFFT Mask"]
        fr2["ResNet-18\nBackbone"]
        fr3["FC → 64-dim\nFeature Vector"]
        fr1 --> fr2 --> fr3
    end

    subgraph SE ["🟣 Semantic Stream  ·  FAT-Lite"]
        direction TB
        se1["ViT-Tiny\nPatch16 / 224px"]
        se2["CLS Token → 384-dim\nFeature Vector"]
        se1 --> se2
    end

    subgraph FU ["⚡ MLAF Fusion"]
        direction TB
        fu1["Stream Projections\n128 + 64 + 384 → 256 each"]
        fu2["Cross-Stream Attention\n3-token sequence, 4 heads"]
        fu3["Classifier Head\n256 → 1  ·  Sigmoid"]
        fu1 --> fu2 --> fu3
    end

    OUT[/"🎯 P(fake)\n∈ [0, 1]"/]

    INPUT --> SP & FR & SE
    sp2 & fr3 & se2 --> fu1
    fu3 --> OUT

    style SP fill:#0d2137,stroke:#4a90d9,color:#a8d4f5
    style FR fill:#2d1200,stroke:#cc6b2c,color:#f5c4a0
    style SE fill:#160d2d,stroke:#7b52d9,color:#c4aef5
    style FU fill:#0d2d17,stroke:#3ab56e,color:#a0f0c0
```

<div align="center">

| Stream | Backbone | Output Dim | Forgery Cues Captured |
|:-------|:--------:|:----------:|:----------------------|
| **Spatial (NPR)** | EfficientNet-B0 | 128 | Pixel texture artifacts, blending boundaries |
| **Frequency (FreqBlender)** | ResNet-18 + Learnable FFT | 64 | Spectral anomalies, up-convolution artifacts |
| **Semantic (FAT-Lite)** | ViT-Tiny patch16 | 384 | Global structural inconsistencies, face geometry |
| **Fusion (MLAF)** | Cross-Stream Attention | 256 | Inter-stream relationships → final classification |

**Total Parameters:** 22.9M &nbsp;·&nbsp; **Optimizer:** AdamW (lr=3×10⁻⁴, cosine LR) &nbsp;·&nbsp; **Loss:** Weighted BCE (pos_weight=2.72)

</div>

---

## Results

### ROC Curve & Confusion Matrix

<p align="center">
  <img src="assets/figures/roc_curve.png" width="45%" alt="ROC Curve — AUC 99.92%"/>
  &nbsp;&nbsp;
  <img src="assets/figures/confusion_matrix.png" width="45%" alt="Confusion Matrix"/>
</p>

<div align="center">

| | Metric | Value |
|--|:-------|------:|
| ✅ | AUC-ROC | **99.92%** |
| ✅ | False Negatives (missed fakes) | **1** / 553 |
| ⚠️ | False Positives (real flagged as fake) | 117 / ~1,500 |
| ✅ | Equal Error Rate | **1.27%** |

</div>

The model is intentionally conservative — in high-stakes detection tasks, missing a fake (FN) is costlier than a false alarm (FP). The single missed fake (FN=1) confirms near-perfect recall at 99.82%.

### Prediction Score Distribution

<p align="center">
  <img src="assets/figures/prediction_dist.png" width="72%" alt="Score Distribution"/>
</p>

Score distributions are sharply bimodal with minimal overlap. Fakes cluster near **1.0**, real images near **0.0** — high decision confidence with minimal ambiguous predictions near the 0.5 threshold.

### Baseline Comparison

<p align="center">
  <img src="assets/figures/baseline_comparison.png" width="80%" alt="Baseline Comparison"/>
</p>

---

## Cross-Generator Generalization

> **Experiment:** Train on **Stable Diffusion v1.x** face images exclusively → evaluate on **SDXL** (architecturally different, never seen during training). This stress-tests generalization to a novel, more capable generator family.

<p align="center">
  <img src="assets/figures/ablation_crossgen.png" width="80%" alt="Cross-Generator Ablation Study"/>
</p>

<div align="center">

| Ablation Configuration | In-Dist AUC | SDXL AUC | Gen. Drop |
|:-----------------------|:-----------:|:--------:|:---------:|
| Frequency Only | 68.5% | 58.26% | −10.24% |
| Spatial + Frequency | 100% | 95.98% | −4.02% |
| Spatial Only | 100% | 96.44% | −3.56% |
| Semantic Only | 100% | 97.17% | −2.83% |
| Spatial + Semantic | 100% | 93.64% | −6.36% |
| **Full 3-Stream (MS-DFD)** | **100%** | **98.09%** | **−1.91% ✓** |

</div>

**Key finding:** Spatial + Semantic (−6.36%) performs *worse* than Spatial alone (−3.56%). When spatial and semantic streams provide conflicting signals on out-of-distribution SDXL images, accuracy degrades. The **frequency stream acts as a tie-breaker** — its spectral view resolves inter-stream conflicts and enables the full model to achieve the lowest generalization drop of any configuration.

---

## GradCAM++ Interpretability

Gradient-weighted class activation maps reveal which spatial regions the model uses as evidence. All samples below: `true_label=fake, predicted=fake, confidence=100%`.

<p align="center">
  <img src="assets/heatmaps/heatmap_01.jpg" width="15%" alt="GradCAM 1"/>
  <img src="assets/heatmaps/heatmap_02.jpg" width="15%" alt="GradCAM 2"/>
  <img src="assets/heatmaps/heatmap_03.jpg" width="15%" alt="GradCAM 3"/>
  <img src="assets/heatmaps/heatmap_04.jpg" width="15%" alt="GradCAM 4"/>
  <img src="assets/heatmaps/heatmap_05.jpg" width="15%" alt="GradCAM 5"/>
  <img src="assets/heatmaps/heatmap_06.jpg" width="15%" alt="GradCAM 6"/>
</p>

Hot regions (red/yellow) consistently localize to **facial boundaries**, **periocular regions**, and **skin texture transitions** — precisely where diffusion model up-convolution artifacts are theoretically expected to appear. This provides interpretable, physically motivated evidence that the model has learned genuine forgery semantics rather than spurious correlations.

---

## Dataset

### Composition

<div align="center">

| Source | Category | Count | Notes |
|:-------|:--------:|------:|:------|
| FFHQ 256px | Real | ~15,000 | High-quality aligned faces |
| CelebA-HQ | Real | ~3,000 | Celebrity faces, diverse conditions |
| DiffusionDB (face-filtered) | Fake | 5,524 | SD v1.x; OpenCV face detector applied |
| 8clabs/sdxl-faces | Fake (eval only) | 1,920 | SDXL; zero overlap with training |

</div>

### Train / Val / Test Splits

<div align="center">

| Split | Real | Fake | Total | Purpose |
|:------|-----:|-----:|------:|:--------|
| Train | ~12,000 | ~4,419 | ~16,419 | Supervised training |
| Validation | ~1,500 | ~552 | ~2,052 | Early stopping on AUC |
| Test | ~1,500 | ~553 | ~2,053 | Final reported metrics |
| Cross-Gen (SDXL) | 1,500 | 1,920 | 3,420 | Generalization evaluation |

</div>

> Stratified per-class split, single RNG seed=42. Zero data leakage across splits — verified by single-pass indexing before any split assignment.

---

## Quick Start

### Requirements

```
Python    3.10+
CUDA      12.8
GPU       8 GB+ VRAM  (tested: RTX 5060 Ti 16GB)
Storage   ~5 GB for datasets
```

### Installation

```bash
git clone https://github.com/noushad999/thesis_grp_3.git
cd thesis_grp_3

# Pinned versions — exact reproducibility guaranteed
pip install -r requirements-lock.txt
```

### Data Setup

```bash
export DATA_ROOT=/path/to/data   # or edit configs/config.yaml directly

# Step 1: Download datasets
python scripts/download_datasets.py

# Step 2: Filter DiffusionDB → face images only (OpenCV detector)
python scripts/filter_faces.py \
  --input  $DATA_ROOT/fake/diffusiondb \
  --output $DATA_ROOT/fake/diffusiondb_faces

# Step 3: Organize
#   faces_dataset/real/{ffhq, celebahq}
#   faces_dataset/fake/diffusiondb_faces
```

### Train

```bash
python scripts/train.py \
  --config   configs/config.yaml \
  --data-dir $DATA_ROOT/faces_dataset

# Saves best checkpoint → checkpoints/best_model.pth  (epoch 15)
# Early stopping patience = 8 epochs on validation AUC
```

### Evaluate

```bash
# Full evaluation + GradCAM++ heatmaps
python scripts/evaluate.py \
  --config       configs/config.yaml \
  --checkpoint   checkpoints/best_model.pth \
  --data-dir     $DATA_ROOT/faces_dataset \
  --output-dir   logs/eval \
  --num-heatmaps 20

# Baseline comparison (CNNDetect, UnivFD)
python scripts/compare_baselines.py \
  --config   configs/config.yaml \
  --data-dir $DATA_ROOT/faces_dataset

# Cross-generator evaluation (SDXL)
python scripts/cross_generator_eval.py \
  --config         configs/config.yaml \
  --checkpoint     checkpoints/best_model.pth \
  --real-dir       $DATA_ROOT/real \
  --fake-dir       $DATA_ROOT/fake/sdxl_faces/imgs \
  --generator-name SDXL \
  --max-images     1500

# Ablation study — trains all 5 configurations
bash scripts/run_ablations_clean.sh
bash scripts/cross_gen_ablation.sh
```

### Single-Image Inference

```bash
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image      path/to/face.jpg

# Output: P(fake) score ∈ [0, 1]  |  threshold = 0.5
```

---

## Project Structure

<details>
<summary><strong>Expand full directory tree</strong></summary>

```
deepfake-detection/
│
├── assets/
│   ├── streams.svg                   # ← Animated 3-stream pipeline diagram
│   ├── figures/                      # ROC curve, confusion matrix, ablation charts
│   └── heatmaps/                     # GradCAM++ visualizations (heatmap_01–06.jpg)
│
├── configs/
│   └── config.yaml                   # All hyperparameters — DATA_ROOT portable
│
├── data/
│   └── dataset.py                    # DeepfakeDataset — stratified split, augmentation
│
├── models/
│   ├── spatial_stream.py             # NPRBranch — EfficientNet-B0 → 128-dim
│   ├── freq_stream.py                # FreqBlender — ResNet-18 + LearnableFFTMask → 64-dim
│   ├── semantic_stream.py            # FATLiteTransformer — ViT-Tiny → 384-dim
│   ├── fusion.py                     # MLAFFusion — 3-token cross-stream attention
│   ├── full_model.py                 # MultiStreamDeepfakeDetector + ablation modes
│   ├── baselines.py                  # CNNDetect + UnivFD baselines
│   └── localization.py               # GradCAM++ heatmap generation
│
├── scripts/
│   ├── train.py                      # AdamW, cosine LR, early stopping on AUC
│   ├── evaluate.py                   # Evaluation + GradCAM++ heatmaps
│   ├── compare_baselines.py          # Side-by-side vs CNNDetect / UnivFD
│   ├── cross_generator_eval.py       # Cross-generator AUC (SDXL, unseen generators)
│   ├── filter_faces.py               # OpenCV face detector for DiffusionDB filtering
│   ├── inference.py                  # Single-image prediction
│   ├── robustness_eval.py            # Robustness under JPEG compression, blur, noise
│   ├── run_ablations_clean.sh        # All 5 ablation training configurations
│   └── cross_gen_ablation.sh         # Cross-generator ablation table
│
├── reports/                          # 10 detailed technical reports
├── requirements.txt
└── requirements-lock.txt             # Pinned — CUDA 12.8, PyTorch 2.9.1
```

</details>

---

## Engineering Notes

<details>
<summary><strong>Critical bug fixes applied</strong></summary>

<br/>

| File | Fix Applied | Impact |
|:-----|:------------|:-------|
| `data/dataset.py:97` | `os.walk(followlinks=True)` | Symlinked dataset directories returned 0 images without this |
| `data/dataset.py:110` | Single-RNG stratified split | Original implementation had data leakage across train/val/test |
| `models/baselines.py:77` | `F.interpolate(x, (224,224))` in UnivFD | CLIP ViT-L/14 requires 224×224 — crashed at runtime on 256×256 input |
| `configs/config.yaml:51` | `pos_weight: 2.72` (was 1.25) | Corrected for actual 15k:5.5k real:fake class imbalance |
| `configs/config.yaml:7` | `${DATA_ROOT:-data}` (was absolute path) | Portable across all machines without manual config edits |
| `models/localization.py:79` | GradCAM++ alpha sum over `axis=(1,2)` | Per-pixel alpha computation was mathematically incorrect |
| `models/fusion.py:22` | 3-token sequence cross-stream attention | `seq_len=1` self-attention is a mathematical no-op (no cross-stream information flow) |

</details>

<details>
<summary><strong>Technical reports index (10 reports)</strong></summary>

<br/>

| # | Report | Primary Audience |
|:-:|:-------|:----------------:|
| 01 | Project Overview — complete summary of all work | General |
| 02 | Code Explained — line-by-line in plain language | Non-technical |
| 03 | Figures Explained — every chart interpreted | General |
| 04 | Architecture — full technical model specification | Technical |
| 05 | Defense Q&A — 15 thesis viva questions with answers | Student |
| 06 | Dataset & Pipeline — data collection and preprocessing | Technical |
| 07 | Experimental Results — all metrics and tables | Researcher |
| 08 | Baseline Comparison — detailed analysis vs CNNDetect/UnivFD | Researcher |
| 09 | Ablation Study — per-stream contribution analysis | Researcher |
| 10 | Publication Guide — ICCIT 2025 paper structure | Student |

</details>

---

## Reproducibility

All experiments are fully reproducible from a clean environment:

```bash
# Environment
Python 3.10   |   PyTorch 2.9.1   |   CUDA 12.8   |   RTX 5060 Ti 16GB

# Determinism
seed: 42
deterministic: true   # torch.backends.cudnn.deterministic = True

# Exact package versions
pip install -r requirements-lock.txt
```

Training configuration is fully declarative in `configs/config.yaml`. No hardcoded paths, no absolute references.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{ramim2025multistreamdeepfake,
  title   = {Multi-Stream Deepfake Detection via Spatial, Frequency,
             and Semantic Fusion with Cross-Generator Generalization},
  author  = {Ramim, Md Noushad Jahan and {Thesis Group 3}},
  year    = {2025},
  note    = {BSc Thesis — Multi-Stream Deepfake Face Detection},
  url     = {https://github.com/noushad999/thesis_grp_3}
}
```

---

## References

<details>
<summary><strong>Show all references</strong></summary>

<br/>

1. Wang, S. et al. *"CNN-generated images are surprisingly easy to spot — for now."* CVPR 2020.
2. Ojha, U. et al. *"Towards Universal Fake Image Detection by Exploiting CLIP's Potential."* CVPR 2023.
3. Qian, Y. et al. *"Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues."* ECCV 2020.
4. Durall, R. et al. *"Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions."* CVPR 2020.
5. Dosovitskiy, A. et al. *"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale."* ICLR 2021.
6. Rombach, R. et al. *"High-Resolution Image Synthesis with Latent Diffusion Models."* CVPR 2022.
7. Podell, D. et al. *"SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis."* arXiv 2023.

</details>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:1a1a3e,70:cc785c,100:0d1117&height=130&section=footer" width="100%"/>

**Md Noushad Jahan Ramim** &nbsp;·&nbsp; BSc Thesis Group 3

*Built with PyTorch &nbsp;·&nbsp; CUDA 12.8 &nbsp;·&nbsp; RTX 5060 Ti 16GB*

<br/>

[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Datasets-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![GitHub](https://img.shields.io/badge/GitHub-noushad999-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/noushad999/thesis_grp_3)

</div>
