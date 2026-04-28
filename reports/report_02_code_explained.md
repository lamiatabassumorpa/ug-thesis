# Report 02: Complete Code Documentation
## প্রতিটি Line কী করে — সহজ বাংলায় ও ইংরেজিতে
### Full Project Code Walkthrough — Zero Prior Knowledge Required

**Author:** Md Noushad Jahan Ramim
**Date:** April 18, 2026

> **এই report টা কীভাবে পড়বে:**
> প্রতিটি section এ actual code আছে, তারপর সেটার "plain language" explanation আছে।
> কোনো programming জ্ঞান না থাকলেও বুঝতে পারবে।

---

## THE BIG PICTURE: কোড কীভাবে সাজানো

```
একটি ছবি দাও
     ↓
[৩টি আলাদা "বিশেষজ্ঞ" একসাথে দেখে]
     ↓          ↓          ↓
 Texture     Frequency   Structure
 (চামড়া?)   (তরঙ্গ?)    (চোখ ঠিক আছে?)
     ↓          ↓          ↓
[তিনজন মিলে আলোচনা করে সিদ্ধান্ত নেয়]
     ↓
REAL বা FAKE
```

---

# PART 1: CONFIGS — `configs/config.yaml`

এটা পুরো project এর "control panel"। এখানে সব settings আছে।

```yaml
# Line 7-8
data:
  root_dir: "${DATA_ROOT:-data}"
```
> **কী করে:** Data কোথায় আছে সেটা বলে।
> `${DATA_ROOT:-data}` মানে: "DATA_ROOT নামের environment variable থেকে নাও।
> সেটা না থাকলে 'data' folder ব্যবহার করো।"
> এটা আগে `"E:/deepfake-detection/data"` ছিল — যেটা শুধু আমার কম্পিউটারে কাজ করতো।
> এখন যেকোনো কম্পিউটারে কাজ করবে।

```yaml
# Line 12-14
model:
  spatial:
    backbone: "efficientnet_b0"
    feature_dim: 128
```
> **কী করে:** প্রথম "বিশেষজ্ঞ" (Texture analyst) কে কীভাবে বানাবে।
> `efficientnet_b0` = এই AI architecture ব্যবহার করো।
> `feature_dim: 128` = এই বিশেষজ্ঞ তার মতামত ১২৮টি সংখ্যায় প্রকাশ করবে।

```yaml
# Line 17-20
  frequency:
    backbone: "resnet18"
    feature_dim: 64
    fft_learnable: true
```
> **কী করে:** দ্বিতীয় "বিশেষজ্ঞ" (Frequency analyst)।
> `feature_dim: 64` = এই বিশেষজ্ঞ মতামত ৬৪টি সংখ্যায় দেয় (ছোট, কারণ frequency info compact)।
> `fft_learnable: true` = FFT filter টা AI নিজেই শিখবে কোন frequency গুরুত্বপূর্ণ।

```yaml
# Line 23-25
  semantic:
    backbone: "vit_tiny_patch16_224"
    feature_dim: 384
```
> **কী করে:** তৃতীয় "বিশেষজ্ঞ" (Structure analyst — Vision Transformer)।
> `feature_dim: 384` = সবচেয়ে বড় output — কারণ semantic/structural info সবচেয়ে সমৃদ্ধ।

```yaml
# Line 38-40
training:
  batch_size: 128
  epochs: 50
  lr: 3e-4
```
> **কী করে:**
> `batch_size: 128` = একসাথে ১২৮টি ছবি দেখে শেখে (একটার বদলে ১২৮টা = দ্রুততর)।
> `epochs: 50` = পুরো dataset ৫০ বার দেখবে।
> `lr: 3e-4` = Learning rate = 0.0003. কতটুকু ভুল থেকে শিখবে।
>   খুব বড় → এলোমেলো শেখা। খুব ছোট → অনেক ধীরে শেখা। 0.0003 = সঠিক ভারসাম্য।

```yaml
# Line 49-51
  loss:
    type: "weighted_bce"
    pos_weight: 2.72
```
> **কী করে:** Loss function — AI কে ভুলের জন্য কতটুকু "শাস্তি" দেওয়া হবে।
> `pos_weight: 2.72` = FAKE image কে REAL বললে শাস্তি ২.৭২ গুণ বেশি।
> **কেন?** আমাদের কাছে ১৫,০০০ real এবং ৫,৫২৪ fake আছে (১৫০০০/৫৫২৪ ≈ ২.৭২)।
> এই বেশি শাস্তি class imbalance ঠিক করে।

---

# PART 2: DATA LOADING — `data/dataset.py`

## ক্লাস: `DeepfakeDataset`

```python
# Line 25-26
class DeepfakeDataset(Dataset):
    """
    Stratified deepfake detection dataset.
    ...
    """
```
> **কী করে:** PyTorch Dataset এর একটা "blueprint"।
> এটা বলে: "আমি এভাবে data দেব।" PyTorch এই blueprint পড়ে training এর সময় data নেয়।

```python
# Line 42-50
def __init__(
    self,
    data_root: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    max_samples_per_class: Optional[int] = None,
    domain_labels: bool = False,
    seed: int = 42
):
```
> **কী করে:** Dataset তৈরির সময় কী কী parameter নেয়।
> `data_root` = ছবিগুলো কোন folder এ আছে।
> `split = 'train'` = কোন ভাগ চাই (train/val/test)।
> `seed = 42` = Random shuffling এর "বীজ"। একই seed মানে একই shuffle — reproducibility।

```python
# Line 64-65
self._collect_data()
self._split_data()
```
> **কী করে:** দুটো কাজ করে — প্রথমে সব ছবি খুঁজে বের করো (`collect`), তারপর ভাগ করো (`split`)।

---

## মেথড: `_collect_from_directory()`

```python
# Line 95-103
def _collect_from_directory(self, directory: Path, label: int):
    collected = 0
    for root, _, files in os.walk(directory, followlinks=True):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in IMG_EXTENSIONS:
                self.image_paths.append(Path(root) / fname)
                self.labels.append(label)
                self.domain_names.append(Path(root).name)
                collected += 1
```

> **Line by line:**
> - `for root, _, files in os.walk(directory, followlinks=True):` →
>   directory এর ভেতরে সব subdirectory তে ঘুরে বেড়াও।
>   **`followlinks=True`** = shortcut folder (symlink) ও follow করো।
>   ⚠️ এটা না থাকলে `faces_dataset` থেকে 0টা ছবি পেতাম — এই একটা word সব ঠিক করেছে।
>
> - `if Path(fname).suffix.lower() in IMG_EXTENSIONS:` →
>   শুধু image file রাখো (.jpg, .png, .bmp etc)। .txt বা .log বাদ দাও।
>
> - `self.image_paths.append(Path(root) / fname)` →
>   ছবির পুরো path মনে রাখো। যেমন: `data/real/ffhq/img_0001.png`
>
> - `self.labels.append(label)` →
>   label মনে রাখো। real=0, fake=1।
>
> - `self.domain_names.append(Path(root).name)` →
>   কোন folder থেকে এসেছে মনে রাখো। `ffhq` বা `diffusiondb_faces` ইত্যাদি।

---

## মেথড: `_split_data()` — সবচেয়ে গুরুত্বপূর্ণ Fix

```python
# Line 110-150
def _split_data(self):
    rng = np.random.RandomState(self.seed)    # Line 117
```
> **কী করে:** একটা fixed random number generator বানায়।
> `seed=42` মানে প্রতিবার একই ক্রমে shuffle হবে। তাই যেকেউ এই code চালালে একই split পাবে।

```python
    labels_arr = np.array(self.labels)        # Line 119
    real_idx   = np.where(labels_arr == 0)[0].copy()  # Line 120
    fake_idx   = np.where(labels_arr == 1)[0].copy()  # Line 121
```
> **কী করে:**
> `labels_arr` = সব label এর array: [0, 0, 1, 0, 1, 1, ...]
> `real_idx` = শুধু real (0) গুলোর index বের করো: [0, 1, 3, ...]
> `fake_idx` = শুধু fake (1) গুলোর index বের করো: [2, 4, 5, ...]

```python
    rng.shuffle(real_idx)   # Line 123
    rng.shuffle(fake_idx)   # Line 124
```
> **কী করে:** Real আর Fake আলাদাভাবে shuffle করো।
> ⚠️ একসাথে shuffle করলে class ratio নষ্ট হতে পারে।
> আলাদা shuffle → guarantee: প্রতিটা split এ ~৩:১ ratio থাকবে।

```python
    def partition(idx: np.ndarray) -> Dict[str, np.ndarray]:  # Line 126
        n       = len(idx)
        n_train = int(0.80 * n)    # ৮০% train
        n_val   = int(0.10 * n)    # ১০% val
        return {
            'train': idx[:n_train],
            'val':   idx[n_train: n_train + n_val],
            'test':  idx[n_train + n_val:]   # বাকি ১০% test
        }
```
> **কী করে:** একটা array কে ৮০/১০/১০ তে ভাগ করে।
> Slice করা হচ্ছে: প্রথম ৮০% = train, পরের ১০% = val, শেষ ১০% = test।
> এগুলো কখনো overlap করে না — mathematical guarantee।

```python
    selected = np.sort(np.concatenate([selected_real, selected_fake]))  # Line 146
```
> **কী করে:** real আর fake এর selected index গুলো একসাথে করে sort করো।
> Sort করলে deterministic order পাওয়া যায়।

---

## মেথড: `__getitem__()` — একটা ছবি লোড করা

```python
# Line 159-190
def __getitem__(self, idx: int) -> Dict:
    img_path = self.image_paths[idx]    # ছবির path নাও
    label    = self.labels[idx]         # label নাও (0 বা 1)

    try:
        image    = Image.open(img_path).convert('RGB')   # ছবি খোলো, RGB তে convert করো
        image_np = np.array(image)                        # PIL Image → numpy array
    except Exception as e:
        print(f"Warning: Could not load {img_path}: {e}")
        image_np = np.zeros((256, 256, 3), dtype=np.uint8)  # ভুল হলে black image দাও
```
> **কী করে:** একটি index দিলে সেই index এর ছবি আর label return করে।
> `.convert('RGB')` = grayscale বা RGBA হলেও সব ছবি ৩-channel RGB তে convert।
> `try/except` = ছবি corrupt থাকলে crash না করে black image দাও।

```python
    if self.transform:
        augmented    = self.transform(image=image_np)
        image_tensor = augmented['image']
```
> **কী করে:** যদি augmentation transform থাকে, সেটা apply করো।
> Training এ থাকে (flip, noise, etc)। Validation/Test এ নেই।

```python
    result = {
        'image': image_tensor,
        'label': torch.tensor(label, dtype=torch.float32),
        'path':  str(img_path)
    }
```
> **কী করে:** একটা dictionary return করে।
> `image` = ছবির tensor (সংখ্যার matrix).
> `label` = 0.0 বা 1.0 (float কারণ BCE loss এটাই চায়).
> `path` = debugging এর জন্য।

---

## Function: `get_transforms()` — Augmentation Pipeline

```python
# Line 231-242 (medium augmentation)
elif augmentation_level == 'medium':
    return A.Compose([
        A.Resize(img_size + 32, img_size + 32),          # 256+32=288 তে বড় করো
        A.RandomCrop(img_size, img_size),                  # তারপর 256 তে crop করো
        A.HorizontalFlip(p=0.5),                           # ৫০% chance মিরর করো
        A.ColorJitter(brightness=0.15, contrast=0.15,
                      saturation=0.15, hue=0.05, p=0.5),  # রঙ সামান্য পরিবর্তন
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                 rotate=(-10, 10), p=0.5),                 # সামান্য ঘোরাও/সরাও
        A.GaussNoise(noise_scale_factor=0.1, p=0.3),      # সামান্য noise যোগ করো
        A.ImageCompression(quality_range=(75, 100), p=0.3),# JPEG compress করো
        normalize, to_tensor                               # Normalize আর Tensor বানাও
    ])
```
> **Line by line:**
> - `Resize(288,288)` → একটু বড় করো যাতে crop করার জায়গা থাকে।
> - `RandomCrop(256,256)` → random জায়গা থেকে 256×256 কেটে নাও। এই দুটো একসাথে = "zoom in" effect।
> - `HorizontalFlip(p=0.5)` → প্রতিটা ছবির ৫০% chance বাম-ডান উল্টে যাবে। Model কে শেখায় যে left face আর right face একই।
> - `ColorJitter` → brightness ±15%, contrast ±15% randomly পরিবর্তন। বিভিন্ন lighting condition simulate করে।
> - `Affine` → সামান্য ঘোরানো (±10°) ও সরানো (±5%). Tilted selfie simulate করে।
> - `GaussNoise` → random pixel noise (30% chance). Screenshot বা compressed image simulate করে।
> - `ImageCompression` → JPEG quality 75-100% (30% chance). Social media re-upload simulate করে।

---

# PART 3: STREAM 1 — `models/spatial_stream.py`

## ক্লাস: `NPRBranch` (Spatial Stream)

```python
# Line 14-15
class NPRBranch(nn.Module):
    """
    Spatial stream using EfficientNet-B0 backbone for pixel-level artifact detection.
    ...
    """
```
> **কী করে:** ছবির pixel-level texture দেখে। চামড়া কি স্বাভাবিক? এজ কি ঠিক?

```python
# Line 35-40
self.backbone = timm.create_model(
    'efficientnet_b0',
    pretrained=pretrained,
    num_classes=0,
    global_pool='avg'
)
```
> **কী করে:**
> `timm.create_model(...)` = pre-built AI model লোড করো।
> `'efficientnet_b0'` = কোন model? EfficientNet-B0।
> `pretrained=True` = ImageNet এ already trained weights নাও। এটা না করলে scratch থেকে শুরু হয়।
> `num_classes=0` = classification head সরিয়ে দাও। আমরা নিজেরা head বানাবো।
> `global_pool='avg'` = পুরো ছবির features কে average করে একটা vector বানাও।
> Output: একটা 1280-dimension vector।

```python
# Line 46-53
self.projection = nn.Sequential(
    nn.Linear(backbone_feat_dim, 256),   # 1280 → 256
    nn.BatchNorm1d(256),                  # Normalize করো
    nn.ReLU(inplace=True),               # Negative values কে 0 করো
    nn.Dropout(dropout_rate),            # Training এ 30% neurons বন্ধ রাখো
    nn.Linear(256, output_dim),          # 256 → 128
    nn.BatchNorm1d(output_dim)            # আবার Normalize
)
```
> **কী করে:** ১২৮০টা সংখ্যা → ১২৮টায় কমানো।
>
> - `nn.Linear(1280, 256)` = ১২৮০টা number কে weigh করে ২৫৬টা বানাও।
>   Think of it as: ২৫৬ জন judge আছে, প্রত্যেকে ১২৮০টা feature দেখে মত দেয়।
>
> - `nn.BatchNorm1d(256)` = সব ২৫৬টা সংখ্যা normalize করো (mean=0, std=1 তে রাখো)।
>   কেন? Training stable রাখার জন্য। এটা ছাড়া numbers অনেক বড় বা ছোট হয়ে যায়।
>
> - `nn.ReLU(inplace=True)` = যেকোনো negative number কে 0 বানিয়ে দাও।
>   কেন? Negative signals ignore করা — শুধু positive evidence গুলো রাখো।
>   `inplace=True` = memory save করো (result আলাদা জায়গায় না রেখে overwrite করো)।
>
> - `nn.Dropout(0.3)` = Training এর সময় random ভাবে ৩০% neurons বন্ধ রাখো।
>   কেন? এটা "regularization" — model কে নির্ভরযোগ্য feature শিখতে বাধ্য করে।
>   Testing/inference এ dropout বন্ধ থাকে।
>
> - `nn.Linear(256, 128)` = ২৫৬ → ১২৮।
> - শেষ `BatchNorm1d(128)` = final normalize।

```python
# Line 65-79
def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.backbone(x)    # [B, 3, 256, 256] → [B, 1280]
    output   = self.projection(features)  # [B, 1280] → [B, 128]
    return output
```
> **কী করে:** একটা ছবির batch দাও, ১২৮-dim feature vector পাও।
> `[B, 3, 256, 256]` মানে: B=batch size, 3=RGB channels, 256×256=resolution।
> Return: `[B, 128]` = প্রতিটা ছবির জন্য ১২৮টা number।

---

# PART 4: STREAM 2 — `models/freq_stream.py`

## ক্লাস: `LearnableFFTMask` — The Novel Component

```python
# Line 13-32
class LearnableFFTMask(nn.Module):
    def __init__(self, img_size: int = 256, n_channels: int = 3):
        super(LearnableFFTMask, self).__init__()
        self.mask = nn.Parameter(
            torch.ones(n_channels, img_size, img_size),
            requires_grad=True
        )
        self.band_weights = nn.Parameter(
            torch.ones(4),  # Low, Mid-Low, Mid-High, High
            requires_grad=True
        )
```
> **কী করে:** এই class টা ছবির "frequency domain" নিয়ে কাজ করে।
>
> **Frequency domain মানে কী?**
> একটা ছবিকে তরঙ্গের সমষ্টি হিসেবে দেখা যায়।
> - Low frequency = ধীরে পরিবর্তন হওয়া (একটা মানুষের মুখের সামগ্রিক আকার)
> - High frequency = দ্রুত পরিবর্তন (চুলের প্রতিটি গোছা, চামড়ার texture)
>
> `self.mask = nn.Parameter(torch.ones(...))` →
>   একটা learnable filter তৈরি করো যেটা AI শিখবে।
>   `nn.Parameter` মানে: training এ এটার value update হবে।
>   শুরুতে সব 1.0 (সব frequency সমান গুরুত্ব)।
>   Training এর পর: গুরুত্বপূর্ণ frequencies এর কাছে বড় value, বাকিগুলো ছোট।
>
> `self.band_weights = nn.Parameter(torch.ones(4))` →
>   ৪টা frequency band: Low, Mid-Low, Mid-High, High।
>   প্রতিটার আলাদা weight। AI শিখবে কোন band সবচেয়ে informative।

```python
# Line 50-51
x_fft = torch.fft.fft2(x, dim=(-2, -1))
x_fft_shifted = torch.fft.fftshift(x_fft)
```
> **কী করে:** ছবিকে frequency domain এ নিয়ে যাও।
> `fft2` = 2D Fast Fourier Transform। Pixel values → Frequency coefficients।
> `fftshift` = Low frequency কে center এ নিয়ে আসো (visualization এর জন্য)।

```python
# Line 53-55
magnitude = torch.abs(x_fft_shifted)   # কতটা শক্তিশালী প্রতিটা frequency
phase = torch.angle(x_fft_shifted)     # কোন phase এ আছে প্রতিটা frequency
```
> **কী করে:** Complex number কে দুইভাগে ভাগ করো।
> Magnitude = তরঙ্গের উচ্চতা (কতটা জোরালো)।
> Phase = তরঙ্গ কোথায় আছে।

```python
# Line 58-71
# Create radial frequency bands
dist_norm = dist / (max_dist + 1e-8)    # 0-1 এর মধ্যে normalize করো

low_mask      = (dist_norm < 0.25).float()        # ০-২৫% = Low freq
mid_low_mask  = ((dist_norm >= 0.25) & (dist_norm < 0.5)).float()   # ২৫-৫০%
mid_high_mask = ((dist_norm >= 0.5)  & (dist_norm < 0.75)).float()  # ৫০-৭৫%
high_mask     = (dist_norm >= 0.75).float()       # ৭৫-১০০% = High freq
```
> **কী করে:** FFT spectrum কে ৪টা "ring" এ ভাগ করো।
> Center = Low frequency (মুখের overall shape)।
> Edge = High frequency (সূক্ষ্ম texture, চামড়ার grain)।
> AI-generated images প্রায়ই specific frequency rings এ অস্বাভাবিক pattern দেখায়।

```python
# Line 82-93
magnitude_filtered = magnitude * combined_mask    # mask apply করো
x_fft_filtered = magnitude_filtered * torch.exp(1j * phase)   # phase আবার যোগ করো
x_ifft = torch.fft.ifft2(torch.fft.ifftshift(x_fft_filtered))  # আবার ছবিতে ফিরে যাও
return torch.abs(x_ifft).real    # real part নাও
```
> **কী করে:** Filter করা frequency দিয়ে ছবি পুনরায় বানাও।
> এই নতুন ছবিতে specific frequency artifacts highlighted থাকে।

---

## ক্লাস: `FreqBlender`

```python
# Line 97-168
class FreqBlender(nn.Module):
    def __init__(self, output_dim: int = 64, ...):
        self.fft_mask = LearnableFFTMask(...)   # Learnable FFT filter
        self.backbone = timm.create_model('resnet18', ...)  # ResNet-18
        self.projection = nn.Sequential(
            nn.Linear(512, 128),   # 512 → 128
            ...
            nn.Linear(128, 64)    # 128 → 64
        )
    
    def forward(self, x):
        x_freq   = self.fft_mask(x)        # ছবি → frequency-filtered ছবি
        features = self.backbone(x_freq)   # filtered ছবি → 512 features
        output   = self.projection(features)  # 512 → 64
        return output
```
> **কী করে (সহজে):**
> ১. ছবিটা frequency domain এ নিয়ে যাও।
> ২. Learnable mask দিয়ে গুরুত্বপূর্ণ frequencies highlight করো।
> ৩. Filtered ছবিটা ResNet-18 দিয়ে analyze করো।
> ৪. ৬৪টা সংখ্যায় সারসংক্ষেপ করো।

---

# PART 5: STREAM 3 — `models/semantic_stream.py`

## ক্লাস: `FATLiteTransformer` (Semantic Stream)

```python
# Line 41-47
self.backbone = timm.create_model(
    'vit_tiny_patch16_224',
    pretrained=pretrained,
    num_classes=0,
    global_pool='token',
    img_size=256          # pos-embed interpolation
)
```
> **কী করে:** Vision Transformer (ViT) লোড করো।
>
> **ViT কীভাবে কাজ করে?**
> ছবিকে ১৬×১৬ pixel এর ছোট ছোট patch এ কেটে নাও।
> ২৫৬×২৫৬ ছবি থেকে = ১৬×১৬ = ২৫৬টা patch।
> প্রতিটা patch কে একটা "word" হিসেবে treat করো।
> এই "words" দিয়ে একটা "sentence" তৈরি করো।
> Transformer এই sentence এর মধ্যে সম্পর্ক বোঝে।
>
> **কেন এটা গুরুত্বপূর্ণ?**
> AI-generated faces প্রায়ই "locally OK but globally inconsistent" —
> বাম চোখ ঠিক আছে, ডান চোখ একটু অদ্ভুত, কিন্তু প্রতিটা আলাদাভাবে দেখলে বোঝা যায় না।
> ViT এই দূরত্বের সম্পর্ক বুঝতে পারে।
>
> `global_pool='token'` = [CLS] token ব্যবহার করো।
> [CLS] token = পুরো ছবির summary। প্রতিটা patch এর সাথে interact করে global representation নেয়।
>
> `img_size=256` = ২২৪ এর বদলে ২৫৬ px। timm automatically position embeddings interpolate করে।

```python
# Line 53-58
self.projection = nn.Sequential(
    nn.Linear(backbone_feat_dim, output_dim),  # 192 → 384
    nn.LayerNorm(output_dim),
    nn.GELU(),
    nn.Dropout(dropout_rate)
)
```
> **কী করে:** ViT এর 192-dim output কে 384-dim এ বাড়াও।
> `nn.GELU()` = ReLU এর smooth version। কোনো sharp cutoff নেই।
> `nn.LayerNorm` = BatchNorm এর মতোই, কিন্তু sequence data এর জন্য ভালো।

```python
# Line 69-79
def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.backbone(x)          # [B, 3, 256, 256] → [B, 192]
    output   = self.projection(features)  # [B, 192] → [B, 384]
    return output
```
> **সহজে:** ছবি দাও → ViT পুরো ছবির structure বোঝে → ৩৮৪টা সংখ্যায় summary।

---

# PART 6: FUSION — `models/fusion.py`

## ক্লাস: `StreamAttention` — তিন বিশেষজ্ঞের আলোচনা

```python
# Line 39-50
self.spatial_proj = nn.Sequential(
    nn.Linear(spatial_dim, hidden_dim),   # 128 → 256
    nn.LayerNorm(hidden_dim)
)
self.freq_proj = nn.Sequential(
    nn.Linear(freq_dim, hidden_dim),      # 64 → 256
    nn.LayerNorm(hidden_dim)
)
self.semantic_proj = nn.Sequential(
    nn.Linear(semantic_dim, hidden_dim),  # 384 → 256
    nn.LayerNorm(hidden_dim)
)
```
> **কী করে:** তিনটা stream এর output কে একই "language" এ translate করো।
> Spatial বলে ১২৮ ভাষায়, Freq বলে ৬৪ ভাষায়, Semantic বলে ৩৮৪ ভাষায়।
> Attention এর আগে সবাইকে ২৫৬ ভাষায় translate করতে হবে।

```python
# Line 53-58
self.attn = nn.MultiheadAttention(
    embed_dim=hidden_dim,    # 256
    num_heads=n_heads,       # 4
    dropout=dropout,
    batch_first=True
)
```
> **কী করে:** Multi-Head Attention। এটা Transformer এর core।
>
> **Attention কীভাবে কাজ করে?**
> ৩টা stream আছে: Spatial (S), Freq (F), Semantic (T)।
> Attention জিজ্ঞেস করে: "S বলছে কিছু সন্দেহজনক আছে। F আর T কি একমত?"
> যদি F আর T ও সন্দেহজনক কিছু দেখে, তাহলে S এর confidence বাড়ে।
>
> `num_heads=4` = ৪টা parallel "conversation" হয়।
> Head 1: "Texture artifact আছে কি? Freq আর Semantic কী মনে করে?"
> Head 2: "Eye region কি suspicious? বাকিরা কী বলছে?"
> Head 3: "Overall face structure consistent কি না?"
> Head 4: "সব মিলিয়ে কতটা sure?"
> এই ৪টা parallel analysis combine হয়।

```python
# Line 79-95
def forward(self, spatial_feat, freq_feat, semantic_feat):
    # প্রতিটা stream কে project করো → [B, 1, 256]
    s = self.spatial_proj(spatial_feat).unsqueeze(1)
    f = self.freq_proj(freq_feat).unsqueeze(1)
    t = self.semantic_proj(semantic_feat).unsqueeze(1)
    
    # তিনটাকে একসাথে রাখো → [B, 3, 256]
    tokens = torch.cat([s, f, t], dim=1)
    
    # Attention apply করো
    attended, attn_weights = self.attn(tokens, tokens, tokens)
    attended = self.dropout(attended)
    
    # Residual connection: পুরনো + নতুন information
    tokens = self.norm(tokens + attended)
    
    # ৩টা stream এর average নাও → [B, 256]
    fused = tokens.mean(dim=1)
    
    return fused, attn_weights
```
> **Line by line:**
> - `.unsqueeze(1)` = [B, 256] → [B, 1, 256]। "1" মানে ১টা token।
> - `torch.cat([s, f, t], dim=1)` = তিনটা token জোড়া লাগাও → [B, 3, 256]।
>   এখন ৩টা stream একটা "sequence" হিসেবে attention দেখতে পাবে।
> - `self.attn(tokens, tokens, tokens)` = self-attention।
>   ৩টা token নিজেরা নিজেদের দেখে। প্রতিটা token বাকি দুটোর দিকে attend করে।
> - `tokens + attended` = Residual connection।
>   নতুন (attended) আর পুরনো (tokens) দুটোই রাখো। Training stable থাকে।
> - `.mean(dim=1)` = ৩টা informed token এর গড় নাও → single vector।

---

## ক্লাস: `MLAFFusion` — Final Decision Maker

```python
# Line 136-142
self.classifier = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),  # 256 → 128
    nn.BatchNorm1d(hidden_dim // 2),
    nn.GELU(),
    nn.Dropout(dropout_rate),               # 40% dropout
    nn.Linear(hidden_dim // 2, 1)           # 128 → 1
)
```
> **কী করে:** Attention এর পর একটাই সংখ্যায় আসো।
> ২৫৬ → ১২৮ → ১। সেই ১টা সংখ্যা = logit।
> Logit positive = FAKE। Negative = REAL।

```python
# Line 156-180
def forward(self, spatial_feat, freq_feat, semantic_feat):
    fused_features, attn_weights = self.cross_stream_attn(
        spatial_feat, freq_feat, semantic_feat
    )  # [B, 256]
    
    logits = self.classifier(fused_features)  # [B, 1]
    
    return logits, fused_features
```
> **সহজে:**
> ৩টা stream এর output নাও → attention দিয়ে combine করো → ১টা number (logit) বানাও।
> এই logit কে Sigmoid এ দিলে 0-1 probability পাওয়া যায়।

---

# PART 7: FULL MODEL — `models/full_model.py`

## ক্লাস: `MultiStreamDeepfakeDetector`

```python
# Line 31-38
ABLATION_MODES = {
    None:               (True,  True,  True),   # সব চালু
    "spatial_only":     (True,  False, False),   # শুধু spatial
    "freq_only":        (False, True,  False),   # শুধু frequency
    "semantic_only":    (False, False, True),    # শুধু semantic
    "spatial_freq":     (True,  True,  False),   # spatial + freq
    "spatial_semantic": (True,  False, True),    # spatial + semantic
}
```
> **কী করে:** Ablation study এর জন্য কোন stream চালু বা বন্ধ রাখতে হবে সেটা define করে।
> (True, False, False) মানে: spatial=on, freq=off, semantic=off।

```python
# Line 112-115
spatial_feat  = self.spatial_stream(x)  if self._use_spatial  else torch.zeros(...)
freq_feat     = self.freq_stream(x)     if self._use_freq     else torch.zeros(...)
semantic_feat = self.semantic_stream(x) if self._use_semantic else torch.zeros(...)
```
> **কী করে:** Ablation mode অনুযায়ী stream চালাও বা শূন্য দাও।
> `torch.zeros(...)` = সেই dimension এর সব 0 এর tensor। মানে ওই stream কোনো তথ্য দিচ্ছে না।

```python
# Line 134-149
def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    self.eval()
    with torch.no_grad():
        logits, _ = self(x)
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).long().squeeze(-1)
    return predictions
```
> **কী করে:** User-friendly prediction method।
> `torch.no_grad()` = gradient calculation বন্ধ রাখো (inference সময় দরকার নেই, memory বাঁচে)।
> `torch.sigmoid(logits)` = logit → probability [0, 1]।
> `probs > 0.5` = 50% এর বেশি হলে FAKE (1), নয়তো REAL (0)।

---

# PART 8: GRADCAM — `models/localization.py`

## ক্লাস: `GradCAMLocalization`

```python
# Line 35-47
def _register_hooks(self):
    if self.target_layer is None:
        self.target_layer = self.model.spatial_stream.backbone.conv_head

    def forward_hook(module, input, output):
        self.activations = output   # forward pass এ activation save করো

    def backward_hook(module, grad_input, grad_output):
        self.gradients = grad_output[0]   # backward pass এ gradient save করো
```
> **কী করে:** "Hook" লাগানো — AI এর ভেতরে spy বসানো।
> Forward pass এ: কোন neuron কতটা active হলো সেটা record করো।
> Backward pass এ: ওই neuron টা final decision এ কতটা contribute করেছে সেটা record করো।

```python
# Line 70-99 — GradCAM++ formula
grad_sq    = gradients ** 2
grad_cube  = gradients ** 3
spatial_sum = (activations * grad_cube).sum(axis=(1, 2), keepdims=True)
alpha = grad_sq / (2.0 * grad_sq + spatial_sum + 1e-8)
weights = (alpha * np.maximum(gradients, 0.0)).sum(axis=(1, 2))
cam = np.einsum('c,chw->hw', weights, activations)
cam = np.maximum(cam, 0.0)
```
> **কী করে (সহজে):**
> ১. কোন neurons সবচেয়ে বেশি "excited" হয়েছে? (activations)
> ২. ওই excitement কতটা decision এ গেছে? (gradients)
> ৩. দুটো combine করে "importance map" তৈরি করো।
> ৪. গুরুত্বপূর্ণ regions = bright (hot) color, কম গুরুত্বপূর্ণ = cool color।
>
> `+1e-8` = division by zero থেকে বাঁচতে একটা tiny number যোগ করো।

```python
# Line 101-132
def generate_heatmap(self, input_tensor, target_class=None):
    input_tensor = input_tensor.requires_grad_(True)  # gradient চাই
    output, _ = self.model(input_tensor)              # forward pass
    target_output = output[0, 0]
    target_output.backward()                          # backward pass
    heatmap = self._compute_gradcampp()
    return heatmap
```
> **কী করে:** একটা ছবি দাও → কোথায় AI এর attention সেটা দেখাও।

---

# PART 9: TRAINING — `scripts/train.py`

## ক্লাস: `MetricsTracker`

```python
# Line 44-109
class MetricsTracker:
    def update(self, loss, logits, labels):
        scores = torch.sigmoid(logits).detach().cpu()   # logit → probability
        preds  = (scores > 0.5).long()                  # probability → 0/1
        self.correct += (preds == labels.long().cpu()).sum().item()  # সঠিক গুনো
        self.all_scores.extend(scores.numpy().tolist())  # AUC এর জন্য রাখো
```
> **কী করে:** প্রতিটা batch এর পর metrics update করো।
> `.detach()` = gradient graph থেকে বিচ্ছিন্ন করো (মেমরি বাঁচাও)।
> `.cpu()` = GPU থেকে CPU তে নিয়ে আসো (numpy calculation এর জন্য)।
> `.item()` = Python number এ convert করো।

```python
# Line 88-99
auc = roc_auc_score(self.all_labels, self.all_scores)
fpr, tpr, _ = roc_curve(self.all_labels, self.all_scores)
eer = brentq(lambda x: 1.0 - x - float(interp1d(fpr, tpr)(x)), 0.0, 1.0)
```
> **কী করে:** AUC আর EER হিসেব করো।
> `roc_auc_score` = ROC curve এর area।
> `brentq` = mathematical equation solver। EER এ FPR = FNR — এটা solve করে।
> `interp1d(fpr, tpr)` = ROC curve কে continuous function বানায় তারপর solve করা যায়।

---

## Early Stopping

```python
# Line 116-144
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience   = patience    # কতবার অপেক্ষা করবো
        self.min_delta  = min_delta   # কতটুকু improve হলে "improvement" গণ্য হবে
        self.counter    = 0           # কতবার improve হয়নি

    def __call__(self, current_value):
        if improved:
            self.best_value = current_value
            self.counter    = 0       # counter reset
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True   # বলো: training বন্ধ করো
                return True
```
> **কী করে:** Training automatically বন্ধ করে যখন model improve করা বন্ধ করে।
> `patience=8` = ৮ epoch পরপর improve না হলে training শেষ।
> কেন? Overfitting থেকে বাঁচতে — model নতুন কিছু শিখছে না কিন্তু training data memorize করছে।

---

## Optimizer Setup

```python
# Line 151-185
def create_optimizer_and_scheduler(model, lr, weight_decay, warmup_epochs, total_epochs):
    no_decay = {'bias', 'norm', 'layernorm', 'layer_norm'}
    grouped  = [
        {'params': [...not bias/norm...], 'weight_decay': weight_decay},
        {'params': [...bias/norm...],     'weight_decay': 0.0},         # bias নিয়ে regularize না
    ]
    optimizer = AdamW(grouped, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                             total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs,
                                      eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])
```
> **কী করে:**
>
> **AdamW:**
> - `betas=(0.9, 0.999)` = গত ১০টা update এর গড় নাও (momentum)। Jump করা কমায়।
> - `weight_decay` = L2 regularization। বড় weights কে penalty দাও।
> - Bias আর LayerNorm এর weights এ weight decay প্রযোজ্য নয় (standard practice)।
>
> **Learning Rate Schedule:**
> - `warmup_sched` = প্রথম ৩ epoch এ LR ধীরে ধীরে বাড়াও (0.1x → 1x)।
>   কেন? শুরুতে weights random — বড় LR দিলে training unstable।
> - `cosine_sched` = Epoch বাড়ার সাথে LR cosine curve এ কমতে থাকে।
>   কেন? শেষের দিকে fine-tuning — ছোট steps দরকার।
> - `SequentialLR` = প্রথমে warmup, তারপর cosine।

---

## Training Loop

```python
# Line 192-243
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, ...):
    model.train()    # dropout, batchnorm training mode এ থাকে
    
    for step, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)  # GPU তে পাঠাও
        labels = batch['label'].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):    # Mixed precision
            logits, _ = model(images)                         # Forward pass
            logits    = logits.squeeze(-1)                    # [B, 1] → [B]
            loss = criterion(logits, labels) / gradient_accumulation_steps
        
        loss.backward()    # Backward pass: কোন parameter কতটা দায়ী?
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clip
            optimizer.step()    # Parameters update করো
            optimizer.zero_grad()   # Gradients reset করো
```
> **Line by line:**
>
> - `model.train()` = Training mode। Dropout আর BatchNorm এ বলো "তুমি training mode এ আছ"।
>   (Inference এ `model.eval()` — dropout বন্ধ, batchnorm stable statistics ব্যবহার করে।)
>
> - `.to(device, non_blocking=True)` = GPU তে পাঠাও।
>   `non_blocking=True` = CPU GPU transfer এর সময় CPU অন্য কাজ করতে পারে (parallel)।
>
> - `torch.amp.autocast('cuda')` = Mixed Precision Training।
>   32-bit float এর বদলে 16-bit float ব্যবহার করো → 2x দ্রুত, অর্ধেক memory।
>
> - `logits.squeeze(-1)` = [B, 1] → [B]। BCELoss এর জন্য দরকার।
>
> - `loss.backward()` = Backpropagation। Chain rule দিয়ে প্রতিটা parameter এর gradient হিসেব করো।
>
> - `clip_grad_norm_(..., 1.0)` = Gradient explosion থেকে বাঁচাও।
>   যদি gradient অনেক বড় হয়ে যায় → scale down করো max norm = 1.0।
>
> - `optimizer.step()` = gradients দিয়ে parameters update করো।
>
> - `optimizer.zero_grad()` = পুরনো gradient মুছে দাও।
>   এটা না করলে gradients cumulate হয় → wrong updates।

---

# PART 10: BASELINES — `models/baselines.py`

## CNNDetect Fix

```python
# কোনো fix ছিল না। Directly কাজ করেছে।
backbone.fc = nn.Linear(2048, 1)  # ResNet-50 এর শেষ layer replace
```
> **কী করে:** ResNet-50 এর original 1000-class head কে সরিয়ে 1-class head বসাও।
> ২০৪৮ = ResNet-50 এর penultimate layer এর size।

## UnivFD Fix (Critical)

```python
# BEFORE (crashed with RuntimeError):
feats = self.clip.encode_image(x)   # x is [B, 3, 256, 256] — WRONG size for CLIP

# AFTER (fixed):
def forward(self, x: torch.Tensor):
    with torch.no_grad():
        x = torch.nn.functional.interpolate(
            x, size=(224, 224),       # ← এই line টা add করা হয়েছে
            mode='bilinear',
            align_corners=False
        )
        feats = self.clip.encode_image(x)    # এখন 224×224 — CLIP খুশি
        feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalize
    logits = self.linear(feats.float())
    return logits, None
```
> **কেন crash হয়েছিল?**
> CLIP ViT-L/14 এ position embeddings আছে।
> ২২৪×২২৪ ছবি = 14×14 = ১৯৬ patches + 1 [CLS] = ১৯৭ tokens।
> ২৫৬×২৫৬ ছবি = 16×16 = ২৫৬ patches + 1 [CLS] = ২৫৭ tokens।
> CLIP এর weight matrix এ ১৯৭টা position embedding আছে, ২৫৭টার জন্য নেই।
> → Dimension mismatch → RuntimeError।
>
> **Fix:** `F.interpolate(x, size=(224, 224))` দিয়ে resize করো।
> `mode='bilinear'` = smooth interpolation (nearest neighbor এর মতো pixelated নয়)।

---

# সারসংক্ষেপ: সব Bug Fix এক জায়গায়

| File | Line | Bug | Fix |
|------|------|-----|-----|
| `data/dataset.py` | 97 | `os.walk()` symlink follow করেনা → 0 images | `followlinks=True` যোগ করো |
| `data/dataset.py` | 110-150 | Data leakage: different RNG seeds overlap করতে পারে | Single RNG, per-class partition |
| `models/baselines.py` | 75-78 | UnivFD crash: 256px ≠ CLIP's 224px position embeddings | `F.interpolate(x, (224,224))` যোগ করো |
| `configs/config.yaml` | 7 | Hardcoded Windows path: অন্য machine এ কাজ করবে না | `${DATA_ROOT:-data}` |
| `configs/config.yaml` | 51 | `pos_weight=1.25` wrong: actual ratio is 2.72 | `pos_weight: 2.72` |
| `models/localization.py` | 78-85 | GradCAM++ alpha per-pixel (wrong) | Sum over spatial dims `axis=(1,2)` |
| `models/fusion.py` | 22 | seq_len=1 attention = mathematical no-op | 3 tokens (one per stream) |
| `models/semantic_stream.py` | 53 | Redundant 192→384→384 projection | Single 192→384 |

---

> **Final Note:**
> প্রতিটা bug real experiment এ পাওয়া গেছে।
> কোনো change hypothetical বা arbitrary নয়।
> প্রতিটা fix এর পেছনে একটা concrete error, crash, বা scientific concern আছে।
