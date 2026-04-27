# Report 04: Model Architecture
## A Complete Technical Description of the Multi-Stream Deepfake Detection System

**Date:** April 18, 2026
**Framework:** PyTorch 2.9.1
**Input:** RGB face images, 256×256 pixels
**Output:** Binary classification (real=0, fake=1) + probability score [0,1]

---

## 1. System Overview

```
Input Image (3 × 256 × 256)
         │
    ┌────┴────┐
    │  Resize │ (to stream-specific sizes)
    └────┬────┘
         │
    ┌────┴────────────────────────────────┐
    │                                     │
  Stream 1          Stream 2           Stream 3
 (Spatial)        (Frequency)         (Semantic)
EfficientNet-B0   ResNet-18+FFT      ViT-Tiny patch16
  ↓ 128-dim         ↓ 64-dim           ↓ 384-dim
    │                 │                   │
    └────────┬────────┘                   │
             │      Project all → 256-dim │
             └──────────────┬─────────────┘
                            │
                   Cross-Stream Attention
                   (4 heads, 256-dim hidden)
                            │
                    Global Average Pool
                            │
                    Linear (256 → 1)
                            │
                      Sigmoid → [0,1]
                            │
                    Classification Decision
                    (threshold = 0.5)
```

---

## 2. Stream 1: Spatial Stream (NPR Branch)

**File:** `models/spatial_stream.py`
**Input size:** 256×256
**Output:** 128-dimensional feature vector

### Architecture:
```
Input (3×256×256)
    → EfficientNet-B0 (pretrained on ImageNet-1K)
       [Removes original 1000-class head]
    → Global Average Pooling
       [1280 → 1280 vector]
    → Linear(1280, 256)
    → BatchNorm1d(256)
    → ReLU
    → Dropout(0.3)
    → Linear(256, 128)
    → L2 Normalize
Output: 128-dim unit vector
```

### Why EfficientNet-B0?
EfficientNet-B0 was chosen because:
1. Compound scaling — balances depth, width, and resolution optimally
2. Relatively lightweight (5.3M parameters) for fast training
3. Pretrained ImageNet weights provide excellent general texture understanding
4. Proven effective for forensic image analysis tasks

### What does this stream detect?
- **Noise patterns:** AI generators introduce characteristic noise signatures
- **Edge artifacts:** Diffusion models produce slight blurring at object boundaries
- **Texture inconsistency:** Generated skin textures often lack natural micro-variations
- **Compression artifacts:** Different from JPEG compression artifacts seen in real photos

---

## 3. Stream 2: Frequency Stream (FreqBlender Branch)

**File:** `models/frequency_stream.py`
**Input size:** 256×256
**Output:** 64-dimensional feature vector

### Architecture:
```
Input (3×256×256)
    → Learnable FFT Module
       [Apply 2D FFT → frequency domain]
       [Learnable magnitude scaling per frequency]
       [Convert back to spatial domain]
    → ResNet-18 (pretrained, first conv modified to accept FFT output)
       [Removes original 1000-class head]
    → Global Average Pooling
       [512 → 512 vector]
    → Linear(512, 128)
    → BatchNorm1d(128)
    → ReLU
    → Linear(128, 64)
    → L2 Normalize
Output: 64-dim unit vector
```

### The Learnable FFT Module (Novel Component):
```python
class LearnableFFT(nn.Module):
    def __init__(self, channels=3):
        self.magnitude_scale = nn.Parameter(torch.ones(channels, 1, 1))
        self.phase_shift = nn.Parameter(torch.zeros(channels, 1, 1))
    
    def forward(self, x):
        fft = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(fft) * self.magnitude_scale
        phase = torch.angle(fft) + self.phase_shift
        fft_modified = magnitude * torch.exp(1j * phase)
        return torch.fft.irfft2(fft_modified, s=x.shape[-2:], norm='ortho')
```

**Plain language:** Instead of a fixed FFT transform, we learn the best way to emphasize certain frequencies. The network decides during training which frequency bands are most informative for detecting fakes. Diffusion models often introduce characteristic periodic patterns in specific frequency ranges.

### Why only 64-dim output?
Frequency features are intrinsically lower-dimensional than spatial features. The spectral artifacts relevant to deepfake detection occupy a narrow band in frequency space. 64 dimensions captures this compactly without over-parameterizing.

### Ablation finding:
Frequency-only: 68.5% AUC (barely above chance). This is expected and intentional — frequency artifacts are generator-specific, not universal. The stream's value is **complementary**: combined with spatial+semantic, it reduces cross-generator drop from −2.83% to −1.91%.

---

## 4. Stream 3: Semantic Stream (FAT-Lite Branch)

**File:** `models/semantic_stream.py`
**Input size:** 256×256 (resized to 224×224 internally for ViT)
**Output:** 384-dimensional feature vector

### Architecture:
```
Input (3×256×256)
    → Resize to 224×224 (ViT requirement)
    → ViT-Tiny patch16_224 (pretrained on ImageNet-21K via DINO)
       [Split image into 14×14 = 196 patches of size 16×16]
       [Add [CLS] token → 197 tokens total]
       [12 transformer encoder layers, 3 attention heads each]
       [Hidden dim: 192]
       [Removes original classification head]
    → Extract [CLS] token (global representation)
       [192-dim vector]
    → Linear(192, 384)
    → LayerNorm(384)
    → GELU activation
    → Dropout(0.1)
Output: 384-dim feature vector
```

### Why Vision Transformer?
Vision Transformers capture **long-range dependencies** that CNNs miss. For example:
- If the left eye is located 150 pixels from the right eye, a CNN needs many layers to "connect" them
- A ViT's attention mechanism can directly compare any two patches regardless of distance

This is important for deepfake detection because AI generators often produce locally plausible but **globally inconsistent** faces. The [CLS] token aggregates global face context.

### Why ViT-Tiny specifically?
ViT-Tiny (5.7M parameters) was chosen over larger variants (ViT-Base, ViT-Large) because:
1. Training resources (RTX 5060 Ti 16GB) are sufficient for Tiny but might struggle with Large
2. Tiny still achieves 74.5% on ImageNet, providing strong representations
3. Overfitting risk is lower with smaller models on our ~20K dataset

---

## 5. Fusion Module: Cross-Stream Multi-Head Attention

**File:** `models/fusion.py`
**Input:** Three feature vectors [128-dim, 64-dim, 384-dim]
**Output:** Single classification logit

### Architecture:
```
Spatial (128-dim) ──┐
                    ├→ Linear projections → all 256-dim
Frequency (64-dim) ─┤
                    │
Semantic (384-dim) ─┘
         ↓
Stack as sequence: [spatial', freq', semantic'] shape=(3, 256)
         ↓
Multi-Head Self-Attention (4 heads, 256-dim)
  • Head 1: Attends to spatial-frequency correlations
  • Head 2: Attends to spatial-semantic correlations  
  • Head 3: Attends to frequency-semantic correlations
  • Head 4: Global stream integration
         ↓
Residual connection (attended + original)
         ↓
LayerNorm
         ↓
Mean pool across 3 streams → 256-dim
         ↓
Linear(256, 128) → ReLU → Dropout(0.4)
         ↓
Linear(128, 1) → logit
         ↓
Sigmoid → probability [0,1]
```

### Why Cross-Attention and not Simple Concatenation?
Simple concatenation would give a 128+64+384 = 576-dim vector, which is then classified. The problem: the classifier treats all dimensions as independent — it doesn't know that spatial dim 45 is related to semantic dim 203.

Cross-stream attention explicitly models these inter-stream relationships. Each stream's output is informed by what the other streams found:
- If the spatial stream sees texture artifacts AND the semantic stream sees structural inconsistency at the same image region, their attention scores reinforce each other
- If streams disagree, the attention mechanism learns to weight the more reliable stream higher

---

## 6. Loss Function

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.72]))
```

**BCEWithLogitsLoss** = Binary Cross-Entropy Loss with numerically stable sigmoid built in.

For a prediction probability p and true label y:
```
Loss = −[y × log(p) + (1−y) × log(1−p)]
```

With `pos_weight=2.72`, when y=1 (fake image):
```
Loss = −[2.72 × log(p) + (1−1) × log(1−p)]
     = −2.72 × log(p)
```

This means missing a fake image is penalized 2.72× more than a false alarm on a real image. This compensates for having 2.72× more real training images.

---

## 7. Training Configuration

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| Optimizer | AdamW | Weight decay regularization built in |
| Learning Rate | 3×10⁻⁴ | Linear scaling: 1×10⁻⁴ × (128/32) |
| Weight Decay | 1×10⁻⁴ | Standard L2 regularization |
| Warmup Epochs | 3 | Prevents large gradient updates at start |
| Gradient Clip | 1.0 | Prevents gradient explosion |
| Early Stopping | Patience=8 on Val AUC | Prevents overfitting |
| Best Epoch | 15 | Validation AUC peaked here |

---

## 8. Model Size Summary

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| Spatial (EfficientNet-B0 + head) | ~5.4M | Yes |
| Frequency (ResNet-18 + FFT + head) | ~11.3M | Yes |
| Semantic (ViT-Tiny + projection) | ~5.8M | Yes |
| Fusion (attention + classifier) | ~0.4M | Yes |
| **Total** | **~22.9M** | **Yes** |

For comparison:
- CNNDetect (ResNet-50): 25.6M parameters
- UnivFD (CLIP linear probe): only 768 trainable parameters (CLIP frozen)

---

## 9. Inference Pipeline

```python
# Complete inference on a single image
from models.full_model import MultiStreamDeepfakeDetector
from data.dataset import get_transforms
from PIL import Image
import torch
import numpy as np

model = MultiStreamDeepfakeDetector.load_from_checkpoint('checkpoints/best_model.pth')
model.eval()

transform = get_transforms('test', img_size=256)
image = np.array(Image.open('face.jpg').convert('RGB'))
tensor = transform(image=image)['image'].unsqueeze(0)

with torch.no_grad():
    logit, _ = model(tensor)
    prob = torch.sigmoid(logit).item()

print(f"Probability of being fake: {prob:.3f}")
print("FAKE" if prob > 0.5 else "REAL")
```

Typical inference time: ~8ms per image on RTX 5060 Ti (batch_size=1)
