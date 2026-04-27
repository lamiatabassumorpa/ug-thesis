"""
Production Inference Script — Multi-Stream Deepfake Detector
=============================================================
Supports:
  - Single image inference
  - Batch inference from directory
  - Optional GradCAM++ heatmap generation
  - JSON result export

Usage:
  # Single image
  python scripts/inference.py --input path/to/image.jpg --checkpoint checkpoints/best_model.pth

  # Batch (directory)
  python scripts/inference.py --input path/to/dir/ --checkpoint checkpoints/best_model.pth --batch

  # With heatmaps
  python scripts/inference.py --input image.jpg --checkpoint best_model.pth --heatmap

  # Save results to JSON
  python scripts/inference.py --input dir/ --batch --output results.json
"""
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Union

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from models.localization import GradCAMLocalization
from utils.utils import get_device

# Supported image extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


# -----------------------------------------------------------------------
# Model loader
# -----------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: torch.device,
    spatial_dim: int = 128,
    freq_dim: int = 64,
    semantic_dim: int = 384,
    fusion_hidden: int = 256,
    fusion_heads: int = 4
) -> MultiStreamDeepfakeDetector:
    """Load model from checkpoint."""
    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=spatial_dim,
        freq_output_dim=freq_dim,
        semantic_output_dim=semantic_dim,
        fusion_hidden_dim=fusion_hidden,
        fusion_attention_heads=fusion_heads,
        pretrained_backbones=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    loss  = checkpoint.get('loss',  '?')
    print(f"Model loaded — epoch {epoch}, val_loss {loss:.4f}" if isinstance(loss, float)
          else f"Model loaded — epoch {epoch}")

    return model


# -----------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------

def get_inference_transform(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def preprocess_image(
    image_path: Union[str, Path],
    transform: A.Compose
) -> Optional[torch.Tensor]:
    """Load and preprocess a single image → [1, 3, H, W]."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        tensor = transform(image=img_np)['image']
        return tensor.unsqueeze(0)   # [1, 3, H, W]
    except Exception as e:
        print(f"  WARN: could not load {image_path}: {e}")
        return None


# -----------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------

@torch.no_grad()
def predict_single(
    model: MultiStreamDeepfakeDetector,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5
) -> Dict:
    """Run inference on a single preprocessed image tensor."""
    image_tensor = image_tensor.to(device)

    t0 = time.perf_counter()
    logits, features = model(image_tensor, return_features=True)
    latency_ms = (time.perf_counter() - t0) * 1000

    prob      = torch.sigmoid(logits).item()
    label     = "FAKE" if prob >= threshold else "REAL"
    is_fake   = prob >= threshold

    return {
        "probability_fake": round(prob, 6),
        "probability_real": round(1.0 - prob, 6),
        "prediction":       label,
        "is_fake":          is_fake,
        "confidence":       round(abs(prob - 0.5) * 2, 4),   # 0 = uncertain, 1 = certain
        "latency_ms":       round(latency_ms, 2),
        "stream_features": {
            "spatial_norm":   round(features['spatial'].norm().item(), 4),
            "freq_norm":      round(features['frequency'].norm().item(), 4),
            "semantic_norm":  round(features['semantic'].norm().item(), 4),
        }
    }


def predict_batch(
    model: MultiStreamDeepfakeDetector,
    image_dir: Union[str, Path],
    device: torch.device,
    transform: A.Compose,
    threshold: float = 0.5,
    recursive: bool = True
) -> List[Dict]:
    """Run inference on all images in a directory."""
    image_dir = Path(image_dir)
    glob_fn   = image_dir.rglob if recursive else image_dir.glob

    image_paths = sorted(
        p for p in glob_fn("*")
        if p.suffix.lower() in IMG_EXTS
    )

    if not image_paths:
        print(f"No images found in {image_dir}")
        return []

    print(f"Found {len(image_paths)} images in {image_dir}")
    results = []

    for i, path in enumerate(image_paths, 1):
        tensor = preprocess_image(path, transform)
        if tensor is None:
            continue

        result = predict_single(model, tensor, device, threshold)
        result['image_path'] = str(path)
        results.append(result)

        pct  = i / len(image_paths) * 100
        bar  = '#' * (int(pct) // 4)
        print(f"\r  [{bar:<25}] {pct:5.1f}%  {path.name[:40]:<40}  "
              f"{result['prediction']}  ({result['probability_fake']:.3f})",
              end='', flush=True)

    print()
    return results


# -----------------------------------------------------------------------
# Heatmap generation
# -----------------------------------------------------------------------

def generate_heatmap_for_image(
    model: MultiStreamDeepfakeDetector,
    image_tensor: torch.Tensor,
    device: torch.device,
    output_path: str
) -> str:
    """Generate and save a GradCAM++ heatmap."""
    cam = GradCAMLocalization(model)
    saved = cam.save_heatmap(
        input_tensor=image_tensor.to(device),
        output_path=output_path,
        save_visualization=True
    )
    cam.remove_hooks()
    return saved


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

def print_summary(results: List[Dict]):
    if not results:
        return

    total    = len(results)
    n_fake   = sum(1 for r in results if r['is_fake'])
    n_real   = total - n_fake
    avg_prob = sum(r['probability_fake'] for r in results) / total
    avg_lat  = sum(r['latency_ms'] for r in results) / total

    print("\n" + "=" * 55)
    print("INFERENCE SUMMARY")
    print("=" * 55)
    print(f"  Total images:   {total:>6}")
    print(f"  Predicted REAL: {n_real:>6}  ({n_real/total*100:.1f}%)")
    print(f"  Predicted FAKE: {n_fake:>6}  ({n_fake/total*100:.1f}%)")
    print(f"  Avg fake prob:  {avg_prob:>6.3f}")
    print(f"  Avg latency:    {avg_lat:>5.1f} ms / image")

    # High-confidence fakes
    confident_fakes = [r for r in results if r['is_fake'] and r['confidence'] > 0.8]
    if confident_fakes:
        print(f"\n  High-confidence fakes (conf > 0.8): {len(confident_fakes)}")
        for r in confident_fakes[:5]:
            name = Path(r['image_path']).name if 'image_path' in r else 'N/A'
            print(f"    {name:40s}  prob={r['probability_fake']:.3f}")

    print("=" * 55)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Stream Deepfake Detector — Inference"
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Path to image file or directory'
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--batch', '-b', action='store_true',
        help='Run batch inference on a directory'
    )
    parser.add_argument(
        '--threshold', '-t', type=float, default=0.5,
        help='Decision threshold for fake/real (default: 0.5)'
    )
    parser.add_argument(
        '--heatmap', action='store_true',
        help='Generate GradCAM++ heatmaps'
    )
    parser.add_argument(
        '--heatmap-dir', type=str, default='./logs/heatmaps',
        help='Directory to save heatmaps'
    )
    parser.add_argument(
        '--img-size', type=int, default=256,
        help='Input image size (default: 256)'
    )
    parser.add_argument(
        '--no-recursive', action='store_true',
        help='Do not recurse into subdirectories (batch mode)'
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device()

    # Load model
    checkpoint = args.checkpoint
    if not Path(checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("Train a model first:  python scripts/train.py --config configs/config.yaml")
        sys.exit(1)

    print(f"\nLoading model from: {checkpoint}")
    model = load_model(checkpoint, device)

    transform = get_inference_transform(args.img_size)
    input_path = Path(args.input)

    # ── Batch mode ──────────────────────────────────────────────────────
    if args.batch or input_path.is_dir():
        results = predict_batch(
            model, input_path, device, transform,
            threshold=args.threshold,
            recursive=not args.no_recursive
        )

        print_summary(results)

        if args.heatmap:
            print(f"\nGenerating heatmaps → {args.heatmap_dir}")
            heatmap_dir = Path(args.heatmap_dir)
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            for r in results[:20]:   # limit to 20
                if not r.get('image_path'):
                    continue
                tensor = preprocess_image(r['image_path'], transform)
                if tensor is not None:
                    name = Path(r['image_path']).stem
                    generate_heatmap_for_image(
                        model, tensor, device,
                        str(heatmap_dir / f"{name}_heatmap.jpg")
                    )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved → {args.output}")

    # ── Single image mode ────────────────────────────────────────────────
    else:
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}")
            sys.exit(1)

        print(f"\nAnalyzing: {input_path.name}")
        tensor = preprocess_image(input_path, transform)
        if tensor is None:
            sys.exit(1)

        result = predict_single(model, tensor, device, args.threshold)
        result['image_path'] = str(input_path)

        # Pretty print
        print("\n" + "=" * 45)
        print(f"  Image:       {input_path.name}")
        print(f"  Prediction:  {result['prediction']}")
        print(f"  Fake prob:   {result['probability_fake']:.4f}")
        print(f"  Real prob:   {result['probability_real']:.4f}")
        print(f"  Confidence:  {result['confidence']:.4f}")
        print(f"  Latency:     {result['latency_ms']:.1f} ms")
        print("=" * 45)

        if args.heatmap:
            heatmap_dir = Path(args.heatmap_dir)
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            heatmap_path = str(heatmap_dir / f"{input_path.stem}_heatmap.jpg")
            generate_heatmap_for_image(model, tensor, device, heatmap_path)
            print(f"  Heatmap:     {heatmap_path}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved → {args.output}")


if __name__ == '__main__':
    main()
