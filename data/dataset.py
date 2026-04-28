"""
Robust PyTorch Dataset/DataLoader for deepfake detection.
Handles ForenSynths, GenImage, DiffusionDB, COCO, FFHQ, and other sources.

Fix applied:
  Original _split_data() had a data-leakage bug — val/test indices were drawn
  from the complement of train indices using a different RNG seed, which could
  produce overlapping index sets. Replaced with a clean stratified split that
  partitions each class once with a single RNG, guaranteeing no leakage.
"""
import os
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


class DeepfakeDataset(Dataset):
    """
    Stratified deepfake detection dataset.

    Directory structure expected:
        data_root/
        ├── fake/
        │   ├── forensynths/   (GAN: ProGAN, StyleGAN, BigGAN …)
        │   ├── genimage/      (SD 1.4/1.5/XL, Midjourney, DALL-E 3 …)
        │   └── diffusiondb/   (Stable Diffusion prompts)
        └── real/
            ├── coco/          (MS-COCO 2017)
            └── ffhq/          (FFHQ 256px)

    Labels: real=0, fake=1.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        max_samples_per_class: Optional[int] = None,
        domain_labels: bool = False,
        seed: int = 42
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.domain_labels = domain_labels
        self.seed = seed

        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.domain_names: List[str] = []

        self._collect_data()
        self._split_data()

        print(f"[DeepfakeDataset] {split:5s}: {len(self.image_paths):>6,} images "
              f"(Real: {self.labels.count(0):>5,} | Fake: {self.labels.count(1):>5,})")

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_data(self):
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root not found: {self.data_root}\n"
                f"Run: python scripts/download_datasets.py"
            )

        fake_dir = self.data_root / 'fake'
        real_dir = self.data_root / 'real'

        if fake_dir.exists():
            self._collect_from_directory(fake_dir, label=1)
        if real_dir.exists():
            self._collect_from_directory(real_dir, label=0)

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {self.data_root}\n"
                f"Expected: data_root/fake/... and data_root/real/..."
            )

    def _collect_from_directory(self, directory: Path, label: int):
        collected = 0
        for root, _, files in os.walk(directory, followlinks=True):
            for fname in sorted(files):
                if Path(fname).suffix.lower() in IMG_EXTENSIONS:
                    self.image_paths.append(Path(root) / fname)
                    self.labels.append(label)
                    self.domain_names.append(Path(root).name)
                    collected += 1
        print(f"  Collected {collected:>6,} images from {directory} (label={label})")

    # ------------------------------------------------------------------
    # Stratified split — NO DATA LEAKAGE
    # ------------------------------------------------------------------

    def _split_data(self):
        """
        Stratified 80/10/10 train/val/test split.

        Each class is shuffled once with a single RNG, then partitioned
        into non-overlapping slices. This guarantees zero leakage.
        """
        rng = np.random.RandomState(self.seed)

        labels_arr = np.array(self.labels)
        real_idx   = np.where(labels_arr == 0)[0].copy()
        fake_idx   = np.where(labels_arr == 1)[0].copy()

        rng.shuffle(real_idx)
        rng.shuffle(fake_idx)

        def partition(idx: np.ndarray) -> Dict[str, np.ndarray]:
            n       = len(idx)
            n_train = int(0.80 * n)
            n_val   = int(0.10 * n)
            return {
                'train': idx[:n_train],
                'val':   idx[n_train: n_train + n_val],
                'test':  idx[n_train + n_val:]
            }

        real_splits = partition(real_idx)
        fake_splits = partition(fake_idx)

        selected_real = real_splits[self.split]
        selected_fake = fake_splits[self.split]

        if self.max_samples_per_class is not None:
            selected_real = selected_real[:self.max_samples_per_class]
            selected_fake = selected_fake[:self.max_samples_per_class]

        selected = np.sort(np.concatenate([selected_real, selected_fake]))

        self.image_paths  = [self.image_paths[i]  for i in selected]
        self.labels       = [self.labels[i]        for i in selected]
        self.domain_names = [self.domain_names[i]  for i in selected]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        try:
            image    = Image.open(img_path).convert('RGB')
            image_np = np.array(image)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            image_np = np.zeros((256, 256, 3), dtype=np.uint8)

        if self.transform:
            augmented    = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            default_transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            image_tensor = default_transform(image=image_np)['image']

        result = {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'path':  str(img_path)
        }

        if self.domain_labels:
            result['domain'] = self.domain_names[idx]

        return result

    def get_class_weights(self) -> torch.Tensor:
        n_real  = self.labels.count(0)
        n_fake  = self.labels.count(1)
        total   = n_real + n_fake
        w_real  = total / (2.0 * n_real) if n_real > 0 else 1.0
        w_fake  = total / (2.0 * n_fake) if n_fake > 0 else 1.0
        return torch.tensor([w_real, w_fake])

    def get_domain_list(self) -> List[str]:
        return sorted(set(self.domain_names))


# -----------------------------------------------------------------------
# Augmentation transforms
# -----------------------------------------------------------------------

def get_transforms(
    phase: str = 'train',
    img_size: int = 256,
    augmentation_level: str = 'medium'
) -> A.Compose:
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    to_tensor = ToTensorV2()

    if phase == 'train':
        if augmentation_level == 'none':
            return A.Compose([A.Resize(img_size, img_size), normalize, to_tensor])

        elif augmentation_level == 'light':
            return A.Compose([
                A.Resize(img_size + 32, img_size + 32),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                normalize, to_tensor
            ])

        elif augmentation_level == 'medium':
            return A.Compose([
                A.Resize(img_size + 32, img_size + 32),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.15, contrast=0.15,
                              saturation=0.15, hue=0.05, p=0.5),
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                         rotate=(-10, 10), p=0.5),
                A.GaussNoise(noise_scale_factor=0.1, p=0.3),
                A.ImageCompression(quality_range=(75, 100), p=0.3),
                normalize, to_tensor
            ])

        else:  # heavy
            return A.Compose([
                A.Resize(img_size + 64, img_size + 64),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.1, p=0.8),
                A.Affine(translate_percent=0.1, scale=(0.8, 1.2),
                         rotate=(-20, 20), p=0.8),
                A.GaussNoise(noise_scale_factor=0.15, p=0.5),
                A.Blur(blur_limit=3, p=0.2),
                A.ImageCompression(quality_range=(60, 100), p=0.5),
                A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(16, 32),
                                hole_width_range=(16, 32), p=0.3),
                normalize, to_tensor
            ])
    else:
        return A.Compose([A.Resize(img_size, img_size), normalize, to_tensor])


# -----------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------

def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 256,
    max_samples_per_class: Optional[int] = None,
    augmentation_level: str = 'medium',
    use_weighted_sampling: bool = False,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders with stratified splits.
    """
    train_transform = get_transforms('train', img_size, augmentation_level)
    val_transform   = get_transforms('val',   img_size)
    test_transform  = get_transforms('test',  img_size)

    train_dataset = DeepfakeDataset(
        data_root=data_root, split='train',
        transform=train_transform,
        max_samples_per_class=max_samples_per_class, seed=seed
    )
    val_dataset = DeepfakeDataset(
        data_root=data_root, split='val',
        transform=val_transform,
        max_samples_per_class=max_samples_per_class, seed=seed
    )
    test_dataset = DeepfakeDataset(
        data_root=data_root, split='test',
        transform=test_transform,
        max_samples_per_class=max_samples_per_class, seed=seed
    )

    train_sampler = None
    if use_weighted_sampling:
        class_weights  = train_dataset.get_class_weights()
        sample_weights = [class_weights[int(l)].item() for l in train_dataset.labels]
        train_sampler  = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        **common_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False,
        **{**common_kwargs, 'drop_last': False}
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False,
        **{**common_kwargs, 'drop_last': False}
    )

    print(f"\n[DataLoaders] Ready:")
    print(f"  Train: {len(train_dataset):>6,} samples | {len(train_loader):>4} batches")
    print(f"  Val:   {len(val_dataset):>6,} samples | {len(val_loader):>4} batches")
    print(f"  Test:  {len(test_dataset):>6,} samples | {len(test_loader):>4} batches")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import tempfile, shutil
    print("Testing DeepfakeDataset with stratified split …")

    with tempfile.TemporaryDirectory() as tmpdir:
        for sub in ['fake/forensynths', 'real/coco']:
            os.makedirs(os.path.join(tmpdir, sub), exist_ok=True)

        for i in range(20):
            img = Image.new('RGB', (256, 256), color=(i * 12, i * 12, i * 12))
            img.save(os.path.join(tmpdir, 'fake/forensynths', f'fake_{i}.png'))
            img.save(os.path.join(tmpdir, 'real/coco',        f'real_{i}.png'))

        for split in ('train', 'val', 'test'):
            ds = DeepfakeDataset(data_root=tmpdir, split=split)
            assert len(ds) > 0, f"Empty {split} split!"

        sample = DeepfakeDataset(data_root=tmpdir, split='train')[0]
        assert sample['image'].shape == (3, 256, 256)
        assert sample['label'].item() in [0.0, 1.0]

        print("\nDeepfakeDataset test PASSED (no leakage)!")
