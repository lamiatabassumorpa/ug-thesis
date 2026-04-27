"""
PyTorch Dataset class for CiFake.

Adapted from data/dataset.py for the CiFake-specific folder structure.

CiFake directory structure:
    cifar/data/
    ├── real/     <- CIFAR-10 real images (upscaled 256x256)
    └── fake/     <- SD-generated fake images (upscaled 256x256)

Usage:
    from cifar.dataset_cifake import CiFakeDataset, get_cifake_loaders

    train_loader, val_loader, test_loader = get_cifake_loaders(
        data_dir="cifar/data",
        batch_size=128
    )
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}


class CiFakeDataset(Dataset):
    """
    Dataset for CiFake (CIFAR-10 real + Stable Diffusion fake).

    Labels: real=0, fake=1
    Split: stratified 80/10/10 train/val/test (no data leakage)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        max_per_class: Optional[int] = None,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.seed = seed

        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        self._collect()
        self._split(max_per_class)

        n_real = self.labels.count(0)
        n_fake = self.labels.count(1)
        print(f"[CiFakeDataset] {split:5s}: {len(self.image_paths):>6,} images "
              f"(Real: {n_real:>5,} | Fake: {n_fake:>5,})")

    def _collect(self):
        """Collect all image paths and labels from real/ and fake/ directories."""
        for label, subdir in [(0, 'real'), (1, 'fake')]:
            d = self.data_dir / subdir
            if not d.exists():
                print(f"  WARNING: {d} does not exist.")
                continue
            count = 0
            for root, _, files in os.walk(d, followlinks=True):
                for fname in sorted(files):
                    if Path(fname).suffix.lower() in IMG_EXTENSIONS:
                        self.image_paths.append(Path(root) / fname)
                        self.labels.append(label)
                        count += 1
            print(f"  Collected {count:>6,} from {d} (label={label})")

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {self.data_dir}\n"
                f"Run: python cifar/prepare_cifake.py"
            )

    def _split(self, max_per_class: Optional[int] = None):
        """Stratified 80/10/10 split — zero data leakage."""
        rng = np.random.RandomState(self.seed)
        labels_arr = np.array(self.labels)

        real_idx = np.where(labels_arr == 0)[0].copy()
        fake_idx = np.where(labels_arr == 1)[0].copy()
        rng.shuffle(real_idx)
        rng.shuffle(fake_idx)

        def partition(idx: np.ndarray) -> Dict:
            n = len(idx)
            n_train = int(0.80 * n)
            n_val   = int(0.10 * n)
            return {
                'train': idx[:n_train],
                'val':   idx[n_train: n_train + n_val],
                'test':  idx[n_train + n_val:]
            }

        real_parts = partition(real_idx)
        fake_parts = partition(fake_idx)

        sel_real = real_parts[self.split]
        sel_fake = fake_parts[self.split]

        if max_per_class is not None:
            sel_real = sel_real[:max_per_class]
            sel_fake = sel_fake[:max_per_class]

        selected = np.sort(np.concatenate([sel_real, sel_fake]))
        self.image_paths = [self.image_paths[i] for i in selected]
        self.labels      = [self.labels[i]       for i in selected]

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
            image_tensor = self.transform(image=image_np)['image']
        else:
            default = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            image_tensor = default(image=image_np)['image']

        return {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'path':  str(img_path)
        }

    def pos_weight(self) -> float:
        """Compute pos_weight for BCEWithLogitsLoss (real_count / fake_count)."""
        n_real = self.labels.count(0)
        n_fake = self.labels.count(1)
        return n_real / max(n_fake, 1)


def get_cifake_transforms(phase: str = 'train') -> A.Compose:
    """Return albumentations transforms for CiFake training."""
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    to_tensor = ToTensorV2()

    if phase == 'train':
        return A.Compose([
            A.Resize(288, 288),
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15,
                          saturation=0.15, hue=0.05, p=0.5),
            A.GaussNoise(noise_scale_factor=0.1, p=0.3),
            # NOTE: JPEG compression is key for testing robustness
            A.ImageCompression(quality_range=(75, 100), p=0.3),
            normalize, to_tensor
        ])
    else:
        return A.Compose([A.Resize(256, 256), normalize, to_tensor])


def get_cifake_loaders(
    data_dir: str = "cifar/data",
    batch_size: int = 128,
    num_workers: int = 4,
    max_per_class: Optional[int] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for CiFake."""

    train_ds = CiFakeDataset(data_dir, 'train',
                             get_cifake_transforms('train'),
                             max_per_class, seed)
    val_ds   = CiFakeDataset(data_dir, 'val',
                             get_cifake_transforms('val'),
                             max_per_class, seed)
    test_ds  = CiFakeDataset(data_dir, 'test',
                             get_cifake_transforms('test'),
                             max_per_class, seed)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **kwargs)

    print(f"\n[CiFake DataLoaders]")
    print(f"  Train: {len(train_ds):>6,} samples | {len(train_loader):>4} batches")
    print(f"  Val:   {len(val_ds):>6,} samples | {len(val_loader):>4} batches")
    print(f"  Test:  {len(test_ds):>6,} samples | {len(test_loader):>4} batches")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing CiFakeDataset...")
    import tempfile, shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "real").mkdir()
        (tmp / "fake").mkdir()

        for i in range(30):
            img = Image.new('RGB', (256, 256), color=(i * 8, i * 8, i * 8))
            img.save(tmp / "real" / f"real_{i:05d}.png")
            img.save(tmp / "fake" / f"fake_{i:05d}.png")

        for split in ('train', 'val', 'test'):
            ds = CiFakeDataset(str(tmp), split=split)
            sample = ds[0]
            assert sample['image'].shape == (3, 256, 256)
            assert sample['label'].item() in [0.0, 1.0]
            print(f"  {split}: {len(ds)} samples — OK")

    print("CiFakeDataset test PASSED ✅")
