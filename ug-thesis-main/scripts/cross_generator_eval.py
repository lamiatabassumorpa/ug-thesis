"""
Cross-Generator Evaluation Script
Tests model on unseen generators (images NOT seen during training).

Training generators (seen):    DiffusionDB (SD v1.x), sd_mix, research_grade
Unseen generators to test on:  Midjourney, DALL-E 3, Flux.1, SDXL, ProGAN (ForenSynths)

Usage:
  # Evaluate on a folder of unseen fake images:
  python scripts/cross_generator_eval.py \\
    --checkpoint checkpoints/best_model.pth \\
    --fake-dir /path/to/unseen_generator_images \\
    --real-dir /path/to/real_faces \\
    --generator-name "Midjourney_v6" \\
    --output-dir logs/cross_generator

  # Evaluate on ForenSynths test split:
  python scripts/cross_generator_eval.py \\
    --checkpoint checkpoints/best_model.pth \\
    --forensynths-dir /path/to/forensynths_test \\
    --output-dir logs/cross_generator

  # Batch evaluate all subdirs in a folder:
  python scripts/cross_generator_eval.py \\
    --checkpoint checkpoints/best_model.pth \\
    --batch-dir /path/to/generators/ \\
    --real-dir /path/to/real_faces \\
    --output-dir logs/cross_generator
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.full_model import MultiStreamDeepfakeDetector
from utils.utils import set_seed, get_device, load_checkpoint


IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

TRANSFORM = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int]):
        self.paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = np.array(Image.open(self.paths[idx]).convert('RGB'))
        except Exception:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        tensor = TRANSFORM(image=img)['image']
        return {
            'image': tensor,
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'path':  str(self.paths[idx])
        }


def collect_images(directory: Path, label: int, max_images: int = None) -> Tuple[List[Path], List[int]]:
    paths = []
    for f in sorted(directory.rglob('*')):
        if f.suffix.lower() in IMG_EXTENSIONS:
            paths.append(f)
    if max_images:
        paths = paths[:max_images]
    return paths, [label] * len(paths)


@torch.no_grad()
def evaluate_dataset(model, paths, labels, device, batch_size=32) -> Dict:
    ds = SimpleImageDataset(paths, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch['image'].to(device)
        logits, _ = model(images)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch['label'].numpy().tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr  = (probs_arr > 0.5).astype(int)

    auc = roc_auc_score(labels_arr, probs_arr) * 100 if len(set(labels_arr)) > 1 else 0.0
    acc = accuracy_score(labels_arr, preds_arr) * 100
    f1  = f1_score(labels_arr, preds_arr, zero_division=0) * 100
    eer = 0.0
    if len(set(labels_arr)) > 1:
        try:
            fpr, tpr, _ = roc_curve(labels_arr, probs_arr)
            eer = brentq(lambda x: 1.0 - x - float(interp1d(fpr, tpr)(x)), 0., 1.) * 100
        except Exception:
            pass

    return {'auc': auc, 'accuracy': acc, 'f1': f1, 'eer': eer,
            'n_real': int((labels_arr == 0).sum()),
            'n_fake': int((labels_arr == 1).sum())}


def load_model(checkpoint_path, config, device):
    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=config['model']['spatial']['feature_dim'],
        freq_output_dim=config['model']['frequency']['feature_dim'],
        semantic_output_dim=config['model']['semantic']['feature_dim'],
        fusion_hidden_dim=config['model']['fusion']['hidden_dim'],
        fusion_attention_heads=config['model']['fusion']['attention_heads'],
        pretrained_backbones=False
    ).to(device)

    if os.path.exists(checkpoint_path):
        try:
            load_checkpoint(model, torch.optim.AdamW(model.parameters()), checkpoint_path, device)
        except ValueError:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded from epoch {ckpt['epoch']}")
    else:
        print(f"WARNING: checkpoint not found: {checkpoint_path}")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Cross-generator evaluation")
    parser.add_argument('--config',         type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint',     type=str, required=True)
    parser.add_argument('--real-dir',       type=str, default=None,
                        help='Real face images (if not provided, uses data/real/ffhq)')
    parser.add_argument('--fake-dir',       type=str, default=None,
                        help='Single unseen generator folder')
    parser.add_argument('--generator-name', type=str, default='unseen_generator')
    parser.add_argument('--batch-dir',      type=str, default=None,
                        help='Each subdir = one generator (batch mode)')
    parser.add_argument('--forensynths-dir',type=str, default=None,
                        help='ForenSynths test split root (has real/ and fake/ subdirs)')
    parser.add_argument('--output-dir',     type=str, default='logs/cross_generator')
    parser.add_argument('--max-images',     type=int, default=1000)
    parser.add_argument('--batch-size',     type=int, default=32)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, config, device)

    # Default real images
    data_root = Path(config['data']['root_dir'])
    real_dir = Path(args.real_dir) if args.real_dir else data_root / 'real'

    real_paths, real_labels = [], []
    if real_dir.exists():
        rp, rl = collect_images(real_dir, label=0, max_images=args.max_images)
        real_paths.extend(rp)
        real_labels.extend(rl)
        print(f"Real images: {len(real_paths)} from {real_dir}")

    results = {}

    # --- Mode 1: Single fake-dir ---
    if args.fake_dir:
        fake_paths, fake_labels = collect_images(Path(args.fake_dir), label=1, max_images=args.max_images)
        all_paths  = real_paths + fake_paths
        all_labels = real_labels + fake_labels
        print(f"\nEvaluating [{args.generator_name}]: {len(fake_paths)} fake + {len(real_paths)} real")
        metrics = evaluate_dataset(model, all_paths, all_labels, device, args.batch_size)
        results[args.generator_name] = metrics

    # --- Mode 2: Batch dir (each subdir = one generator) ---
    elif args.batch_dir:
        batch_dir = Path(args.batch_dir)
        for gen_dir in sorted(batch_dir.iterdir()):
            if not gen_dir.is_dir():
                continue
            fake_paths, fake_labels = collect_images(gen_dir, label=1, max_images=args.max_images)
            if len(fake_paths) == 0:
                continue
            all_paths  = real_paths + fake_paths
            all_labels = real_labels + fake_labels
            print(f"\nEvaluating [{gen_dir.name}]: {len(fake_paths)} fake + {len(real_paths)} real")
            metrics = evaluate_dataset(model, all_paths, all_labels, device, args.batch_size)
            results[gen_dir.name] = metrics

    # --- Mode 3: ForenSynths ---
    elif args.forensynths_dir:
        forensynths = Path(args.forensynths_dir)
        for gen_dir in sorted((forensynths / 'fake').iterdir()) if (forensynths / 'fake').exists() else []:
            if not gen_dir.is_dir():
                continue
            real_src = forensynths / 'real'
            rp, rl = collect_images(real_src, label=0, max_images=args.max_images) if real_src.exists() else (real_paths, real_labels)
            fp, fl = collect_images(gen_dir, label=1, max_images=args.max_images)
            print(f"\nForenSynths [{gen_dir.name}]: {len(fp)} fake + {len(rp)} real")
            metrics = evaluate_dataset(model, rp + fp, rl + fl, device, args.batch_size)
            results[gen_dir.name] = metrics

    # --- Print results table ---
    print("\n" + "=" * 70)
    print("CROSS-GENERATOR EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Generator':<25} {'AUC':>8} {'Accuracy':>10} {'F1':>8} {'EER':>8} {'N_fake':>8}")
    print("─" * 70)
    aucs = []
    for gen, m in results.items():
        print(f"{gen:<25} {m['auc']:>7.2f}% {m['accuracy']:>9.2f}% {m['f1']:>7.2f}% {m['eer']:>7.2f}% {m['n_fake']:>8}")
        aucs.append(m['auc'])

    if len(aucs) > 1:
        print("─" * 70)
        print(f"{'Mean ± Std':<25} {np.mean(aucs):>7.2f}% ± {np.std(aucs):.2f}%")

    # Save
    out_path = output_dir / 'cross_generator_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("\nNOTE: Add unseen generator images to run this properly.")
    print("  Midjourney/DALL-E/Flux images → --fake-dir or --batch-dir")
    print("  ForenSynths test split        → --forensynths-dir")


if __name__ == "__main__":
    main()
