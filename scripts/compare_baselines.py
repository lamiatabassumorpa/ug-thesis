"""
Full Comparison: Our Model vs CNNDetect vs UnivFD
==================================================
Evaluates all models on the same test set.
Produces the results table for Section 4 of the paper.

Usage:
  python scripts/compare_baselines.py --config configs/config.yaml --data-dir data/
"""
import os
import sys
import json
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.full_model import MultiStreamDeepfakeDetector
from models.baselines import CNNDetect, UnivFD
from data.dataset import create_dataloaders
from utils.utils import set_seed, get_device, load_checkpoint


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].cpu().numpy()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(images)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])
        all_labels.extend(labels.tolist())

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr  = (probs_arr > 0.5).astype(int)

    auc = roc_auc_score(labels_arr, probs_arr) * 100
    acc = accuracy_score(labels_arr, preds_arr) * 100
    f1  = f1_score(labels_arr, preds_arr, zero_division=0) * 100
    eer = 0.0
    try:
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr)
        eer = brentq(lambda x: 1.0 - x - float(interp1d(fpr, tpr)(x)), 0., 1.) * 100
    except Exception:
        pass

    return {'auc': auc, 'accuracy': acc, 'f1': f1, 'eer': eer}


def load_our_model(ckpt_path, config, device):
    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=config['model']['spatial']['feature_dim'],
        freq_output_dim=config['model']['frequency']['feature_dim'],
        semantic_output_dim=config['model']['semantic']['feature_dim'],
        fusion_hidden_dim=config['model']['fusion']['hidden_dim'],
        fusion_attention_heads=config['model']['fusion']['attention_heads'],
        pretrained_backbones=False
    ).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    return model


def load_baseline(model_name, ckpt_path, device):
    model = CNNDetect(pretrained=False) if model_name == 'cnndetect' else UnivFD()
    model = model.to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='configs/config.yaml')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default='logs/comparison')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir

    set_seed(config['seed'])
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(config['logging']['checkpoint_dir'])

    _, _, test_loader = create_dataloaders(
        data_root=config['data']['root_dir'],
        batch_size=32,
        num_workers=0,
        img_size=256
    )

    models_to_eval = {
        'Ours (3-Stream MLAF)':
            load_our_model(str(ckpt_dir / 'best_model.pth'), config, device),
        'CNNDetect (Wang et al. 2020)':
            load_baseline('cnndetect', str(ckpt_dir / 'cnndetect/seed42/best_model.pth'), device),
        'UnivFD (Ojha et al. 2023)':
            load_baseline('univfd', str(ckpt_dir / 'univfd/seed42/best_model.pth'), device),
    }

    print("\n" + "=" * 75)
    print("COMPARISON: Our Method vs Baselines (Diffusion Deepfake Detection)")
    print("=" * 75)
    print(f"{'Method':<35} {'AUC':>8} {'Acc':>8} {'F1':>8} {'EER':>8}")
    print("─" * 75)

    all_results = {}
    for name, model in models_to_eval.items():
        try:
            metrics = evaluate(model, test_loader, device)
            all_results[name] = metrics
            print(f"{name:<35} {metrics['auc']:>7.2f}% {metrics['accuracy']:>7.2f}% "
                  f"{metrics['f1']:>7.2f}% {metrics['eer']:>7.2f}%")
        except Exception as e:
            print(f"{name:<35}  ERROR: {e}")

    print("─" * 75)

    out_path = output_dir / 'comparison_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
