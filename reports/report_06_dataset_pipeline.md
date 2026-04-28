# Report 06: Dataset & Data Pipeline
## How Data Was Collected, Cleaned, and Prepared

**Date:** April 18, 2026

---

## 1. The Problem With the Original Dataset

Before starting actual experiments, we discovered a critical scientific flaw in the original dataset setup. This section documents what it was, why it mattered, and how we fixed it.

### Original Setup (Flawed):
- **Fake images:** DiffusionDB — 50,000 AI-generated images of all types: landscapes, objects, anime art, abstract images, faces
- **Real images:** FFHQ — 70,000 high-resolution human face photographs

### Why This Was Wrong (Content Bias):
The model was not learning to detect deepfakes. It was learning to distinguish:
- "Images containing only a human face" (real) vs.
- "Images that might be landscapes, anime, objects, or faces" (fake)

This is like training a fraud detector on counterfeit art and real ID documents — and then claiming you built a document fraud detector. The model learned the wrong thing.

**Evidence:** AUC hit 100% in epoch 1. Legitimate face detectors take many epochs to converge and rarely reach 100%. Instant 100% is a red flag indicating the task is too easy (content bias, not genuine forgery detection).

---

## 2. The Fix: Face Filtering Pipeline

### Tool Used:
OpenCV's deep neural network face detector: `res10_300x300_ssd_iter_140000.caffemodel`
- Architecture: ResNet-10 SSD (Single Shot Detector)
- Input size: 300×300
- Output: Bounding box coordinates + confidence score for detected faces

### Face Filtering Script: `scripts/filter_faces.py`

```python
# Pseudocode explanation of what filter_faces.py does:

# 1. Load the face detector model
detector = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'caffemodel')

# 2. For each image in DiffusionDB (50,000 images):
for image_path in all_diffusiondb_images:
    image = load_image(image_path)
    
    # 3. Run face detection
    detections = detector.detect(image, confidence_threshold=0.5)
    
    # 4. Keep only images with at least one detected face
    if len(detections) > 0:
        save_to_faces_output_folder(image_path)

# Result: 5,524 images with faces out of 50,000 total (11% face rate)
```

### Troubleshooting During Filtering:

**Problem:** The caffemodel file downloaded as a corrupt file (wrong size, failed to parse)
**Diagnosis:** The URL returned an HTML error page instead of the binary model file
**Fix:** Used Python's `urllib.request` to download directly and verified file size (must be ~10.7MB)

```python
import urllib.request
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
    'res10_300x300_ssd_iter_140000.caffemodel'
)
# Verify: file must be exactly 10,666,620 bytes
```

---

## 3. Final Dataset Composition

### Fake Images (AI-Generated Faces):
| Source | Generator | Count | Notes |
|--------|-----------|-------|-------|
| DiffusionDB (face-filtered) | Stable Diffusion v1.x | 5,524 | Face-detected subset |
| SDXL Faces (evaluation only) | SDXL | 1,920 | HuggingFace: 8clabs/sdxl-faces |

### Real Images (Genuine Photographs):
| Source | Count | Notes |
|--------|-------|-------|
| FFHQ 256px | ~15,000 | NVIDIA's Flickr Faces HQ |
| CelebA-HQ | ~3,000 | Google's celebrity faces |

**Total training/evaluation dataset:** ~20,524 images

---

## 4. Dataset Directory Structure

```
data/
├── faces_dataset/           ← Main training dataset (symlinked)
│   ├── real/
│   │   ├── ffhq/            ← symlink → data/real/ffhq/
│   │   └── celebahq/        ← symlink → data/real/celebahq/
│   └── fake/
│       └── diffusiondb_faces/ ← symlink → data/fake/diffusiondb_faces/
│
├── real/
│   ├── ffhq/                ← 15,000 FFHQ face images
│   └── celebahq/            ← CelebA-HQ images
│
└── fake/
    ├── diffusiondb/         ← Original 50k DiffusionDB (unfiltered)
    ├── diffusiondb_faces/   ← Face-filtered subset (5,524 images)
    ├── sdxl_faces/
    │   └── imgs/            ← 1,920 SDXL faces (evaluation only)
    └── genimage/            ← Directory structure exists (empty)
        ├── DALL-E/
        ├── Midjourney/
        ├── ADM/
        ├── glide/
        ├── stable_diffusion_v_1_4/
        └── stable_diffusion_v_1_5/
```

### Why Symlinks?
Instead of copying 15,000 images twice, we created symbolic links (shortcuts) from `faces_dataset/real/` to `data/real/`. This saves disk space and ensures a single source of truth — if we add more images to `data/real/ffhq/`, they automatically appear in `faces_dataset/real/ffhq/`.

**Critical fix required:** `os.walk()` does not follow symlinks by default. We added `followlinks=True` to `data/dataset.py` line 97.

---

## 5. Data Split Strategy

### Method: Stratified 80/10/10 Split

```python
def _split_data(self):
    rng = np.random.RandomState(seed=42)  # Fixed seed for reproducibility
    
    real_indices = [i for i, label in enumerate(labels) if label == 0]
    fake_indices = [i for i, label in enumerate(labels) if label == 1]
    
    rng.shuffle(real_indices)   # Shuffle real independently
    rng.shuffle(fake_indices)   # Shuffle fake independently
    
    # Partition each class into 80/10/10
    # Real: 12,000 train / 1,500 val / 1,500 test
    # Fake: 4,419 train / 552 val / 553 test
    
    # Combine and sort to maintain determinism
    split_indices = sorted(real_split + fake_split)
```

### Why Stratified?
If we shuffled all images together and split, we might accidentally get:
- Training set: 85% real, 15% fake
- Test set: 50% real, 50% fake

This would make training think the world has mostly real images while testing assumes equal distribution. Stratified splitting ensures each split has the same class ratio as the full dataset.

### Why Two Separate Shuffles?
Using one shuffle for all images, then splitting, can produce overlap if the split boundaries happen to fall unevenly. Shuffling each class separately with a single RNG and then partitioning is mathematically guaranteed to produce non-overlapping splits.

---

## 6. Augmentation Pipeline

During training, each image is randomly modified to improve robustness:

| Augmentation | Probability | Effect |
|-------------|-------------|--------|
| Random Horizontal Flip | 50% | Mirror image left-right |
| Random Crop (from 288→256) | 100% | Slight zoom/position variation |
| Color Jitter (±15% brightness/contrast) | 50% | Lighting variation |
| Affine Transform (±10° rotation, ±5% translate) | 50% | Slight angle/position change |
| Gaussian Noise (scale=0.1) | 30% | Add random pixel noise |
| JPEG Compression (quality 75-100%) | 30% | Simulate re-compression |

### Why Augment?
Real-world deepfakes are often:
- Shared on social media (JPEG compressed multiple times)
- Screenshotted (lossy, slight noise)
- Posted at different orientations
- Viewed under different lighting conditions

Training on augmented images means the model sees a variety of degraded versions and learns features that survive these transformations.

**For validation and testing: NO augmentation.** Only resize + normalize. We evaluate on "clean" images to measure true performance.

---

## 7. Normalization

All images are normalized with ImageNet statistics:
```python
mean = [0.485, 0.456, 0.406]  # R, G, B channel means
std  = [0.229, 0.224, 0.225]  # R, G, B channel standard deviations
```

**Why these specific values?** These are computed from the ImageNet training set (~1.28M images). All three backbones (EfficientNet-B0, ResNet-18, ViT-Tiny) were pretrained on ImageNet and their weights were learned with this normalization. Using the same normalization in our training ensures the pretrained feature detectors work correctly.

---

## 8. Class Imbalance Handling

**Problem:** 15,000 real images vs 5,524 fake images ≈ 2.72:1 ratio

**Solution used:** Weighted loss function
```python
pos_weight = 15000 / 5524 ≈ 2.72
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.72]))
```

**Alternative available but not used:** WeightedRandomSampler
```python
# This oversamples fake images during training
# So the model sees roughly equal numbers of real and fake per epoch
sample_weights = [2.72 if label==1 else 1.0 for label in dataset.labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

We chose weighted loss over oversampling because:
1. The model still sees ALL real images (no undersampling)
2. Mathematical equivalence to having 2.72× more fake images
3. Cleaner implementation with no sampling bias artifacts

---

## 9. Download Scripts

All datasets can be downloaded using `scripts/download_datasets.py`:

```bash
# Download FFHQ faces
python scripts/download_datasets.py --dataset ffhq --output data/real/ffhq

# Download DiffusionDB subset  
python scripts/download_datasets.py --dataset diffusiondb --output data/fake/diffusiondb

# Download SDXL faces (for cross-generator eval)
python scripts/download_datasets.py --dataset sdxl-faces --output data/fake/sdxl_faces

# Filter faces from DiffusionDB
python scripts/filter_faces.py --input data/fake/diffusiondb --output data/fake/diffusiondb_faces
```

Total download size:
- FFHQ 256px: ~2GB
- DiffusionDB subset: ~2GB  
- SDXL faces: ~154MB
- **Total: ~4.2GB**
