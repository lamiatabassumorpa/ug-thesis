"""
Download CiFake Dataset from Kaggle.

CiFake: Real and AI-Generated Synthetic Images
  - 50,000 REAL images (CIFAR-10)
  - 50,000 FAKE images (Stable Diffusion generated, CIFAR-10 style)
  - Size: ~1.2 GB
  - Resolution: 32x32 px (upscaled to 256x256 by prepare_cifake.py)

Kaggle URL: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Usage:
    python cifar/download_cifake.py

Requirements:
    pip install kagglehub pillow tqdm
    (Kaggle account + API key in ~/.kaggle/kaggle.json)
"""

import os
import shutil
from pathlib import Path

# Output directory: inside the cifar folder
OUTPUT_DIR = Path(__file__).parent / "raw_data"
FAKE_DIR = OUTPUT_DIR / "fake"
REAL_DIR = OUTPUT_DIR / "real"


def download_cifake_kagglehub():
    """Download CiFake via kagglehub (recommended method)."""
    print("=" * 60)
    print("CiFake Dataset Downloader")
    print("Source: birdy654/cifake-real-and-ai-generated-synthetic-images")
    print("=" * 60)

    try:
        import kagglehub
        print("\nDownloading from Kaggle via kagglehub...")
        path = kagglehub.dataset_download(
            "birdy654/cifake-real-and-ai-generated-synthetic-images"
        )
        print(f"Downloaded to: {path}")
        return Path(path)

    except ImportError:
        print("kagglehub not installed.")
        print("Install with: pip install kagglehub")
        return None
    except Exception as e:
        print(f"Error: {e}")
        print("\nAlternative: Download manually from Kaggle and extract to:")
        print(f"  {OUTPUT_DIR}/")
        return None


def organize_cifake(raw_path: Path):
    """
    Organize downloaded CiFake into:
        cifar/raw_data/
            real/   <- CIFAR-10 real images
            fake/   <- Stable Diffusion fake images
    """
    print(f"\nOrganizing dataset from: {raw_path}")
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    real_count = 0
    fake_count = 0

    # CiFake folder structure: REAL/ and FAKE/ at the top level
    for root, dirs, files in os.walk(raw_path):
        root_path = Path(root)
        folder_name = root_path.name.upper()

        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            src = root_path / fname

            if folder_name == "REAL" or "real" in str(root_path).lower():
                dst = REAL_DIR / f"real_{real_count:05d}.png"
                shutil.copy2(src, dst)
                real_count += 1
            elif folder_name == "FAKE" or "fake" in str(root_path).lower():
                dst = FAKE_DIR / f"fake_{fake_count:05d}.png"
                shutil.copy2(src, dst)
                fake_count += 1

    print(f"\nOrganization complete:")
    print(f"  Real images: {real_count}")
    print(f"  Fake images: {fake_count}")
    print(f"  Total:       {real_count + fake_count}")
    print(f"\nSaved to: {OUTPUT_DIR}/")
    print("\nNext step: Run prepare_cifake.py to resize 32x32 → 256x256")

    return real_count, fake_count


def download_cifake_hf_fallback():
    """
    Fallback: Download CiFake via HuggingFace datasets library.
    Dataset: 'jlbaker361/cifake-real-vs-ai-generated'
    """
    print("\nTrying HuggingFace fallback...")
    try:
        from datasets import load_dataset
        from PIL import Image
        from tqdm import tqdm

        print("Loading CiFake from HuggingFace...")
        ds = load_dataset("jlbaker361/cifake-real-vs-ai-generated", split="train")

        print(f"Dataset loaded: {len(ds)} samples")
        print(f"Columns: {ds.column_names}")

        REAL_DIR.mkdir(parents=True, exist_ok=True)
        FAKE_DIR.mkdir(parents=True, exist_ok=True)

        real_count = 0
        fake_count = 0

        for item in tqdm(ds, desc="Saving images"):
            img = item.get("image")
            label = item.get("label", item.get("labels", 0))

            if img is None:
                continue

            if not isinstance(img, Image.Image):
                continue

            # label: 0 = REAL, 1 = FAKE
            if label == 0:
                img.save(REAL_DIR / f"real_{real_count:05d}.png")
                real_count += 1
            else:
                img.save(FAKE_DIR / f"fake_{fake_count:05d}.png")
                fake_count += 1

        print(f"\nHF Download complete: {real_count} real, {fake_count} fake")
        return real_count, fake_count

    except Exception as e:
        print(f"HuggingFace fallback failed: {e}")
        return 0, 0


def main():
    print("Starting CiFake download...\n")

    # Method 1: kagglehub
    raw_path = download_cifake_kagglehub()

    if raw_path is not None:
        organize_cifake(raw_path)
    else:
        # Method 2: HuggingFace fallback
        print("\nkagglehub failed. Trying HuggingFace...")
        real_count, fake_count = download_cifake_hf_fallback()

        if real_count == 0 and fake_count == 0:
            print("\n" + "=" * 60)
            print("MANUAL DOWNLOAD REQUIRED")
            print("=" * 60)
            print("1. Go to: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
            print("2. Download the zip file")
            print(f"3. Extract to: {OUTPUT_DIR}/")
            print("   Expected structure:")
            print(f"     {OUTPUT_DIR}/REAL/  <- real images")
            print(f"     {OUTPUT_DIR}/FAKE/  <- fake images")

    # Verify
    actual_real = sum(1 for _ in REAL_DIR.rglob("*.png")) if REAL_DIR.exists() else 0
    actual_fake = sum(1 for _ in FAKE_DIR.rglob("*.png")) if FAKE_DIR.exists() else 0
    print(f"\nVerification: {actual_real} real, {actual_fake} fake in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
