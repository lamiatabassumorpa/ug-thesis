"""
Robustness Evaluation Script
Tests model performance under JPEG compression, Gaussian noise, and resize attacks.
Produces a table suitable for the paper's robustness analysis section.

Usage:
  python scripts/robustness_eval.py \
    --checkpoint checkpoints/best_model.pth \
    --data-dir /path/to/data \
    --output-dir logs/robustness
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import io
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from data.dataset import create_dataloaders
from utils.utils import set_seed, get_device, load_checkpoint


# -----------------------------------------------------------------------
# Perturbation transforms
# -----------------------------------------------------------------------

class JPEGCompression:
    """Re-encode image as JPEG at given quality."""
    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: [3, H, W] float in [0, 1]
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        compressed = np.array(Image.open(buf)).astype(np.float32) / 255.0
        return torch.from_numpy(compressed).permute(2, 0, 1)

    def __repr__(self):
        return f"JPEG(q={self.quality})"


class GaussianNoise:
    """Add Gaussian noise with given sigma."""
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.sigma
        return torch.clamp(img + noise, 0.0, 1.0)

    def __repr__(self):
        return f"GaussNoise(σ={self.sigma})"


class ResizeAttack:
    """Downscale then upscale to destroy high-freq forgery traces."""
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        small_h, small_w = max(1, int(h * self.scale)), max(1, int(w * self.scale))
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        small = pil.resize((small_w, small_h), Image.BILINEAR)
        restored = small.resize((w, h), Image.BILINEAR)
        return torch.from_numpy(np.array(restored).astype(np.float32) / 255.0).permute(2, 0, 1)

    def __repr__(self):
        return f"Resize(scale={self.scale})"


# -----------------------------------------------------------------------
# Perturbed dataset wrapper
# -----------------------------------------------------------------------

class PerturbedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img = item['image']  # [3, H, W] tensor
        img = self.transform(img)
        return {**item, 'image': img}


# -----------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].cpu().numpy()
        logits, _ = model(images)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs) * 100 if len(set(all_labels)) > 1 else 0.0
    acc = accuracy_score(all_labels, all_preds) * 100
    f1  = f1_score(all_labels, all_preds, zero_division=0) * 100

    return {'auc': auc, 'accuracy': acc, 'f1': f1}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation")
    parser.add_argument('--config',      type=str, default='../configs/config.yaml')
    parser.add_argument('--checkpoint',  type=str, default='../checkpoints/best_model.pth')
    parser.add_argument('--data-dir',    type=str, default=None)
    parser.add_argument('--output-dir',  type=str, default='../logs/robustness')
    parser.add_argument('--batch-size',  type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir

    set_seed(config['seed'])
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=config['model']['spatial']['feature_dim'],
        freq_output_dim=config['model']['frequency']['feature_dim'],
        semantic_output_dim=config['model']['semantic']['feature_dim'],
        fusion_hidden_dim=config['model']['fusion']['hidden_dim'],
        fusion_attention_heads=config['model']['fusion']['attention_heads'],
        pretrained_backbones=False
    ).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        try:
            load_checkpoint(model, torch.optim.AdamW(model.parameters()), args.checkpoint, device)
        except ValueError:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded from epoch {checkpoint['epoch']}")
    else:
        print(f"WARNING: checkpoint not found: {args.checkpoint}")

    # Load base test set
    _, _, test_loader = create_dataloaders(
        data_root=config['data']['root_dir'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=256
    )
    base_dataset = test_loader.dataset

    # Define perturbations
    perturbations = [
        ("Clean (baseline)",          None),
        ("JPEG q=90",                 JPEGCompression(90)),
        ("JPEG q=70",                 JPEGCompression(70)),
        ("JPEG q=50",                 JPEGCompression(50)),
        ("Gaussian noise σ=0.01",     GaussianNoise(0.01)),
        ("Gaussian noise σ=0.05",     GaussianNoise(0.05)),
        ("Gaussian noise σ=0.10",     GaussianNoise(0.10)),
        ("Resize 0.75× (↓↑)",         ResizeAttack(0.75)),
        ("Resize 0.50× (↓↑)",         ResizeAttack(0.50)),
    ]

    print("\n" + "=" * 70)
    print("ROBUSTNESS EVALUATION")
    print("=" * 70)
    print(f"{'Perturbation':<30} {'AUC':>8} {'Accuracy':>10} {'F1':>8}")
    print("─" * 60)

    results = {}

    for name, transform in perturbations:
        if transform is None:
            loader = test_loader
        else:
            perturbed = PerturbedDataset(base_dataset, transform)
            loader = DataLoader(
                perturbed,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda')
            )

        metrics = evaluate_loader(model, loader, device)
        results[name] = metrics
        print(f"{name:<30} {metrics['auc']:>7.2f}% {metrics['accuracy']:>9.2f}% {metrics['f1']:>7.2f}%")

    print("─" * 60)

    # Save results
    out_path = output_dir / "robustness_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Print AUC degradation table
    baseline_auc = results["Clean (baseline)"]['auc']
    print("\n=== AUC Degradation vs Baseline ===")
    print(f"{'Perturbation':<30} {'AUC Drop':>10}")
    print("─" * 42)
    for name, m in results.items():
        if name == "Clean (baseline)":
            continue
        drop = baseline_auc - m['auc']
        print(f"{name:<30} {drop:>+9.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
