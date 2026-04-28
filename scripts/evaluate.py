"""
Evaluation script for Multi-Stream Deepfake Detection.
Computes accuracy, precision, recall, F1, AUC across generator domains.
Generates GradCAM++ heatmaps for visual analysis.
"""
import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from models.localization import GradCAMLocalization, generate_batch_heatmaps
from data.dataset import create_dataloaders, DeepfakeDataset
from utils.utils import set_seed, get_device, load_checkpoint


class DomainEvaluator:
    """Evaluate model performance per generator domain."""
    
    def __init__(self):
        self.domain_predictions = {}
        self.domain_labels = {}
    
    def update(self, domain: str, preds: np.ndarray, labels: np.ndarray):
        if domain not in self.domain_predictions:
            self.domain_predictions[domain] = []
            self.domain_labels[domain] = []
        
        self.domain_predictions[domain].extend(preds.tolist())
        self.domain_labels[domain].extend(labels.tolist())
    
    def compute_per_domain(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for domain in self.domain_predictions:
            preds = np.array(self.domain_predictions[domain])
            labels = np.array(self.domain_labels[domain])

            # Compute accuracy
            accuracy = accuracy_score(labels, preds) * 100

            # Compute metrics with macro average to handle single-class domains
            precision = precision_score(labels, preds, average='macro', zero_division=0) * 100
            recall = recall_score(labels, preds, average='macro', zero_division=0) * 100
            f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100

            results[domain] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_samples': len(labels)
            }
        return results


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str = None
) -> Dict[str, float]:
    """
    Full model evaluation with all metrics.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        output_dir: Directory to save results
    
    Returns:
        Dictionary of all metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].cpu().numpy()
            paths = batch['path']
            
            # Forward pass
            logits, _ = model(images)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())
            all_paths.extend(paths)
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    auc_val = roc_auc_score(all_labels, all_probs) * 100
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    try:
        eer_val = brentq(lambda x: 1.0 - x - float(interp1d(fpr, tpr)(x)), 0., 1.) * 100
    except Exception:
        eer_val = 0.0

    metrics = {
        'accuracy':  accuracy_score(all_labels, all_preds) * 100,
        'precision': precision_score(all_labels, all_preds, zero_division=0) * 100,
        'recall':    recall_score(all_labels, all_preds, zero_division=0) * 100,
        'f1_score':  f1_score(all_labels, all_preds, zero_division=0) * 100,
        'auc_roc':   auc_val,
        'eer':       eer_val,
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    metrics.update({
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) * 100 if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) * 100 if (tn + fn) > 0 else 0  # Negative Predictive Value
    })
    
    # Print results
    print(f"\n{'Metric':<20} {'Value':>10}")
    print(f"{'─'*30}")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:<20} {value:>9.2f}%")
        else:
            print(f"{name:<20} {value:>10}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['Real', 'Fake'], digits=4))
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')
        
        # Plot ROC curve
        plot_roc_curve(all_labels, all_probs, output_dir / 'roc_curve.png')
        
        # Plot prediction distribution
        plot_prediction_distribution(all_probs, all_labels, 
                                     output_dir / 'prediction_dist.png')
        
        print(f"\nResults saved to: {output_dir}")
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: str):
    """Plot and save ROC curve."""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved: {save_path}")


def plot_prediction_distribution(probs: np.ndarray, labels: np.ndarray, save_path: str):
    """Plot prediction probability distribution."""
    plt.figure(figsize=(10, 6))
    
    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]
    
    plt.hist(real_probs, bins=50, alpha=0.6, label='Real', color='green', edgecolor='black')
    plt.hist(fake_probs, bins=50, alpha=0.6, label='Fake', color='red', edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Fake Probability')
    plt.ylabel('Count')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction distribution saved: {save_path}")


def generate_heatmaps(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 20
):
    """Generate GradCAM++ heatmaps for sample images."""
    print(f"\nGenerating GradCAM++ heatmaps for {num_samples} samples...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    cam = GradCAMLocalization(model)
    
    samples_generated = 0
    
    # Don't use no_grad() - GradCAM needs gradients
    for batch in tqdm(test_loader, desc="Generating heatmaps"):
        if samples_generated >= num_samples:
            break
        
        images = batch['image'].to(device)
        labels = batch['label'].cpu().numpy()
        paths = batch['path']
        
        batch_size = images.shape[0]
        
        for i in range(min(batch_size, num_samples - samples_generated)):
            img = images[i:i+1]
            label = int(labels[i])
            
            # Get prediction (with grad for GradCAM)
            model.zero_grad()
            logits, _ = model(img)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob > 0.5 else 0
            
            # Generate heatmap (this needs gradients)
            try:
                heatmap = cam.generate_heatmap(img, target_class=label)
                vis = cam.visualize_heatmap(img, heatmap)
                
                # Save
                filename = f"sample_{samples_generated:03d}_true{label}_pred{pred}_prob{prob:.3f}.jpg"
                filepath = output_dir / filename
                
                import cv2
                cv2.imwrite(str(filepath), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                samples_generated += 1
                
            except Exception as e:
                print(f"  Warning: Could not generate heatmap for sample {samples_generated}: {e}")
                continue
        
        if samples_generated >= num_samples:
            break
    
    cam.remove_hooks()
    print(f"\nGenerated {samples_generated} heatmaps in: {output_dir}")


def evaluate_per_domain(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str
):
    """Evaluate model performance broken down by generator domain."""
    print("\n" + "="*70)
    print("PER-DOMAIN EVALUATION")
    print("="*70)
    
    model.eval()
    domain_evaluator = DomainEvaluator()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Per-domain evaluation"):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].cpu().numpy()
            paths = batch['path']
            
            logits, _ = model(images)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            # Extract domain from path
            for i in range(len(paths)):
                path = paths[i]
                # Domain is the deepest directory name
                domain = Path(path).parent.name
                domain_evaluator.update(domain, preds[i:i+1], labels[i:i+1])
    
    # Compute and print per-domain results
    domain_results = domain_evaluator.compute_per_domain()
    
    print(f"\n{'Domain':<20} {'Accuracy':>10} {'F1':>10} {'Samples':>10}")
    print(f"{'─'*50}")
    
    for domain, metrics in sorted(domain_results.items()):
        print(f"{domain:<20} {metrics['accuracy']:>9.1f}% {metrics['f1_score']:>9.1f}% {metrics['n_samples']:>10}")
    
    # Save per-domain results
    if output_dir:
        output_path = Path(output_dir) / 'per_domain_results.json'
        with open(output_path, 'w') as f:
            json.dump(domain_results, f, indent=2)
        print(f"\nPer-domain results saved: {output_path}")
    
    return domain_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Stream Deepfake Detector")
    parser.add_argument('--config', type=str, default='../configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='../logs/evaluation')
    parser.add_argument('--num-heatmaps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir
    
    # Setup
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
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        try:
            load_checkpoint(model, torch.optim.AdamW(model.parameters()), args.checkpoint, device)
        except ValueError:
            # Fallback: load only model weights
            print("Loading model weights only...")
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint['epoch']}")
    else:
        print(f"WARNING: Checkpoint not found: {args.checkpoint}")
        print("Evaluating with untrained weights.")
    
    # Create test loader
    _, _, test_loader = create_dataloaders(
        data_root=config['data']['root_dir'],
        batch_size=args.batch_size,
        num_workers=0,
        img_size=256
    )
    
    # Full evaluation
    metrics = evaluate_model(model, test_loader, device, str(output_dir))
    
    # Per-domain evaluation
    evaluate_per_domain(model, test_loader, device, str(output_dir))
    
    # Generate heatmaps
    generate_heatmaps(model, test_loader, device, 
                     str(output_dir / 'heatmaps'), args.num_heatmaps)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"All results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
