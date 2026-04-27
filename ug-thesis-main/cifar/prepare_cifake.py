"""
Prepare CiFake Dataset: Resize 32x32 → 256x256 and organize.

After running download_cifake.py, run this script to:
  1. Resize all images from 32x32 to 256x256
  2. Convert to PNG format
  3. Organize into final folder structure for training

Input:  cifar/raw_data/real/ and cifar/raw_data/fake/
Output: cifar/data/real/ and cifar/data/fake/

Usage:
    python cifar/prepare_cifake.py
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Paths
CIFAR_DIR      = Path(__file__).parent
RAW_DATA_DIR   = CIFAR_DIR / "raw_data"
PREPARED_DIR   = CIFAR_DIR / "data"

RAW_REAL_DIR   = RAW_DATA_DIR / "real"
RAW_FAKE_DIR   = RAW_DATA_DIR / "fake"
PREP_REAL_DIR  = PREPARED_DIR / "real"
PREP_FAKE_DIR  = PREPARED_DIR / "fake"

TARGET_SIZE    = 256   # pixels (same as MS-DFD training resolution)
UPSAMPLE_FILTER = Image.LANCZOS  # High-quality upscaling


def prepare_split(src_dir: Path, dst_dir: Path, label: str, max_count: int = None):
    """Resize images from src_dir and save to dst_dir."""
    if not src_dir.exists():
        print(f"  WARNING: {src_dir} does not exist. Run download_cifake.py first.")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([
        f for f in src_dir.rglob("*")
        if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ])

    if max_count:
        img_files = img_files[:max_count]

    count = 0
    for src in tqdm(img_files, desc=f"Preparing {label}"):
        try:
            img = Image.open(src).convert("RGB")

            # CiFake is 32x32 — upscale to 256x256
            if img.size != (TARGET_SIZE, TARGET_SIZE):
                img = img.resize((TARGET_SIZE, TARGET_SIZE), UPSAMPLE_FILTER)

            out_path = dst_dir / f"{label}_{count:05d}.png"
            img.save(out_path, "PNG")
            count += 1

        except Exception as e:
            print(f"  Skipping {src.name}: {e}")
            continue

    return count


def print_stats():
    """Print dataset statistics."""
    real_count = sum(1 for _ in PREP_REAL_DIR.rglob("*.png")) if PREP_REAL_DIR.exists() else 0
    fake_count = sum(1 for _ in PREP_FAKE_DIR.rglob("*.png")) if PREP_FAKE_DIR.exists() else 0

    print("\n" + "=" * 60)
    print("CiFake Prepared Dataset Statistics")
    print("=" * 60)
    print(f"  Real images: {real_count:,}")
    print(f"  Fake images: {fake_count:,}")
    print(f"  Total:       {real_count + fake_count:,}")
    print(f"  Resolution:  {TARGET_SIZE}×{TARGET_SIZE} px")
    print(f"  Location:    {PREPARED_DIR}/")
    print("=" * 60)

    if real_count > 0 and fake_count > 0:
        ratio = real_count / fake_count
        print(f"\n  Class ratio (real/fake): {ratio:.2f}:1")
        if 0.9 <= ratio <= 1.1:
            print("  Class balance: ✅ Balanced (no pos_weight needed)")
        else:
            print(f"  Class balance: ⚠️ Imbalanced — set pos_weight={ratio:.2f}")

    print(f"\nNext step: Run train_cifake.py to train the model")


def main():
    print("=" * 60)
    print("CiFake Dataset Preparation")
    print(f"Input:  {RAW_DATA_DIR}/")
    print(f"Output: {PREPARED_DIR}/")
    print(f"Target resolution: {TARGET_SIZE}×{TARGET_SIZE} px (upscaled from 32×32)")
    print("=" * 60)

    # Prepare real images
    print("\n[1/2] Processing REAL images (CIFAR-10)...")
    real_count = prepare_split(RAW_REAL_DIR, PREP_REAL_DIR, label="real")

    # Prepare fake images
    print("\n[2/2] Processing FAKE images (Stable Diffusion)...")
    fake_count = prepare_split(RAW_FAKE_DIR, PREP_FAKE_DIR, label="fake")

    print_stats()

    if real_count == 0 or fake_count == 0:
        print("\n⚠️  No images found. Make sure to run download_cifake.py first.")
        print(f"     Expected data in: {RAW_DATA_DIR}/real/ and {RAW_DATA_DIR}/fake/")


if __name__ == "__main__":
    main()
