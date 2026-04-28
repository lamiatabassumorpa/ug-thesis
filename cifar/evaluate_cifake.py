"""
Evaluate trained model on CiFake — including the critical robustness test.

This script reproduces the Phase 1 finding:
  ❌ CIFAKE: 100% clean, 67.8% at JPEG Q=75 (-32pp)

Which is the key reason we switched to FFHQ + DiffusionDB.

Usage:
    python cifar/evaluate_cifake.py \\
        --checkpoint cifar/checkpoints/best_model_cifake.pth \\
        --data-dir   cifar/data

Output:
    - Full classification metrics (AUC, accuracy, F1, recall)
    - Robustness breakdown under JPEG compression
    - results_summary.md update
"""

import sys
import argparse
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    recall_score, precision_score, confusion_matrix
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent.parent))
from cifar.dataset_cifake import CiFakeDataset


# -----------------------------------------------------------------------
# Robustness: JPEG Compression
# -----------------------------------------------------------------------

def apply_jpeg_compression(image_np: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression to a numpy image array."""
    img = Image.fromarray(image_np)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert('RGB'))


def get_eval_transform(jpeg_quality: int = None) -> A.Compose:
    """Eval transform, optionally with JPEG compression."""
    ops = [A.Resize(256, 256)]
    if jpeg_quality is not None:
        ops.append(A.ImageCompression(
            quality_range=(jpeg_quality, jpeg_quality), p=1.0))
    ops += [
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    return A.Compose(ops)


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, device, jpeg_quality=None, desc="Eval"):
    """Run evaluation on a CiFakeDataset, optionally with JPEG compression."""
    transform = get_eval_transform(jpeg_quality=jpeg_quality)
    dataset.transform = transform

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, num_workers=4,
        shuffle=False, pin_memory=True
    )

    model.eval()
    all_labels = []
    all_probs  = []
    all_preds  = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].numpy().tolist()

        with torch.amp.autocast('cuda'):
            logits = model(images)
        probs = torch.sigmoid(logits).cpu().squeeze().numpy()
        preds = (probs >= 0.5).astype(int)

        if isinstance(probs, np.floating):
            probs = [float(probs)]
            preds = [int(preds)]

        all_labels.extend(labels)
        all_probs.extend(probs.tolist() if hasattr(probs, 'tolist') else probs)
        all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else preds)

    auc      = roc_auc_score(all_labels, all_probs)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    recall   = recall_score(all_labels, all_preds, zero_division=0)
    precision= precision_score(all_labels, all_preds, zero_division=0)
    cm       = confusion_matrix(all_labels, all_preds)

    return {
        'auc': auc, 'acc': acc, 'f1': f1,
        'recall': recall, 'precision': precision, 'cm': cm
    }


def print_metrics(metrics: dict, label: str):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)")
    print(f"  Accuracy:  {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    cm = metrics['cm']
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")


def run_robustness_test(model, data_dir, device):
    """
    Run JPEG robustness test — the key Phase 1 finding.
    Tests accuracy at JPEG quality: 100, 90, 75, 50
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS TEST: JPEG Compression")
    print("This is why CiFake was replaced with FFHQ + DiffusionDB")
    print("=" * 60)

    jpeg_qualities = [100, 90, 75, 50]
    results = {}

    test_ds = CiFakeDataset(data_dir, split='test', seed=42)
    baseline = evaluate(model, test_ds, device, jpeg_quality=None,
                        desc="Clean (no compression)")
    results['clean'] = baseline['auc']
    print_metrics(baseline, "Clean — No Compression")

    for quality in jpeg_qualities:
        test_ds = CiFakeDataset(data_dir, split='test', seed=42)
        metrics = evaluate(model, test_ds, device,
                           jpeg_quality=quality,
                           desc=f"JPEG Q={quality}")
        results[f'jpeg_{quality}'] = metrics['auc']
        drop = (baseline['auc'] - metrics['auc']) * 100
        print_metrics(metrics, f"JPEG Quality={quality}  (drop: −{drop:.2f}pp)")

    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY (AUC vs JPEG Quality)")
    print("=" * 60)
    print(f"{'Quality':<12} {'AUC':>8} {'Drop':>10}")
    print("-" * 32)
    base_auc = results['clean']
    print(f"{'Clean':<12} {base_auc:>8.4f} {'—':>10}")
    for q in jpeg_qualities:
        auc  = results[f'jpeg_{q}']
        drop = (base_auc - auc) * 100
        flag = "❌" if drop > 10 else ("⚠️ " if drop > 5 else "✅")
        print(f"{'Q='+str(q):<12} {auc:>8.4f} {'-'+f'{drop:.2f}pp':>10}  {flag}")

    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")

    # Load model
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from models.full_model import MultiStreamDeepfakeDetector
        model = MultiStreamDeepfakeDetector(mode='full').to(device)
    except ImportError:
        import torchvision.models as tv_models
        model = tv_models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model = model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Standard test set evaluation
    print("\n" + "=" * 60)
    print("STANDARD TEST SET EVALUATION")
    print("=" * 60)
    test_ds = CiFakeDataset(args.data_dir, split='test', seed=42)
    test_metrics = evaluate(model, test_ds, device, desc="Test Set")
    print_metrics(test_metrics, "Test Set — Clean Images")

    # Robustness test
    robustness = run_robustness_test(model, args.data_dir, device)

    # Save summary
    summary_path = Path(__file__).parent / "results_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# CiFake Phase 1 Results Summary\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n\n")
        f.write("## Standard Metrics (Clean Test Set)\n\n")
        f.write(f"| Metric | Value |\n|:-------|------:|\n")
        f.write(f"| AUC-ROC | {test_metrics['auc']*100:.2f}% |\n")
        f.write(f"| Accuracy | {test_metrics['acc']*100:.2f}% |\n")
        f.write(f"| F1 | {test_metrics['f1']*100:.2f}% |\n")
        f.write(f"| Recall | {test_metrics['recall']*100:.2f}% |\n\n")
        f.write("## Robustness Under JPEG Compression\n\n")
        f.write("| JPEG Quality | AUC | Drop |\n|:------------|----:|-----:|\n")
        base = robustness['clean']
        f.write(f"| Clean | {base*100:.2f}% | — |\n")
        for q in [100, 90, 75, 50]:
            auc  = robustness.get(f'jpeg_{q}', 0)
            drop = (base - auc) * 100
            f.write(f"| Q={q} | {auc*100:.2f}% | −{drop:.2f}pp |\n")
        f.write("\n## Key Finding\n\n")
        auc_q75  = robustness.get('jpeg_75', 0)
        drop_q75 = (base - auc_q75) * 100
        f.write(f"JPEG Q=75 drop: **−{drop_q75:.2f}pp** ")
        if drop_q75 > 20:
            f.write("❌ — Dataset too fragile. Replaced with FFHQ + DiffusionDB.\n")
        else:
            f.write("✅\n")

    print(f"\n✅ Results saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CiFake model")
    parser.add_argument("--checkpoint", default="cifar/checkpoints/best_model_cifake.pth")
    parser.add_argument("--data-dir",   default="cifar/data")
    args = parser.parse_args()
    main(args)
