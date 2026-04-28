"""
Training script for Multi-Stream Deepfake Detection.
Includes: early stopping, LR scheduling, TensorBoard logging, checkpointing.

Added metrics: AUC-ROC and Equal Error Rate (EER) — standard for deepfake detection papers.
"""
import os
import sys
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Optional wandb — only imported if enabled in config
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from models.full_model import MultiStreamDeepfakeDetector
from data.dataset import create_dataloaders
from utils.utils import set_seed, get_device, save_checkpoint, load_checkpoint


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

class MetricsTracker:
    """
    Tracks loss, accuracy, precision, recall, F1, AUC-ROC, and EER.
    AUC and EER are the standard metrics reported in deepfake detection papers.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.losses     = []
        self.correct    = 0
        self.total      = 0
        self.all_preds  = []
        self.all_labels = []
        self.all_scores = []   # sigmoid probabilities for AUC/EER

    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor):
        self.losses.append(loss)
        scores = torch.sigmoid(logits).detach().cpu()

        # Handle scalar case (batch_size=1 after squeeze)
        preds = (scores > 0.5).long()
        if preds.dim() == 0:
            preds  = preds.unsqueeze(0)
            scores = scores.unsqueeze(0)

        self.correct += (preds == labels.long().cpu()).sum().item()
        self.total   += labels.size(0)
        self.all_preds.extend(preds.numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        self.all_scores.extend(scores.numpy().tolist())

    def compute(self) -> Dict[str, float]:
        avg_loss = float(np.mean(self.losses))
        accuracy = 100.0 * self.correct / max(self.total, 1)

        tp = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 0 and l == 1)

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)

        # AUC-ROC and EER require both classes to be present
        auc = 0.0
        eer = 0.0
        if len(set(self.all_labels)) > 1:
            try:
                auc = roc_auc_score(self.all_labels, self.all_scores)
                fpr, tpr, _ = roc_curve(self.all_labels, self.all_scores)
                # EER: point where FPR == 1 - TPR (i.e. FAR == FRR)
                eer = brentq(lambda x: 1.0 - x - float(interp1d(fpr, tpr)(x)), 0.0, 1.0)
            except Exception:
                pass

        return {
            'loss':      avg_loss,
            'accuracy':  accuracy,
            'precision': precision * 100,
            'recall':    recall    * 100,
            'f1_score':  f1        * 100,
            'auc':       auc       * 100,
            'eer':       eer       * 100,
        }


# -----------------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience    = patience
        self.min_delta   = min_delta
        self.mode        = mode
        self.counter     = 0
        self.best_value  = None
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            return False

        improved = (
            current_value > self.best_value + self.min_delta
            if self.mode == 'max'
            else current_value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = current_value
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


# -----------------------------------------------------------------------
# Optimizer & scheduler
# -----------------------------------------------------------------------

def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    total_epochs: int
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW with weight-decay exclusion for bias/norm + warmup + cosine decay."""

    no_decay = {'bias', 'norm', 'layernorm', 'layer_norm'}
    grouped  = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n.lower() for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n.lower() for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]

    optimizer = AdamW(grouped, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                              total_iters=warmup_epochs)
    cosine_sched  = CosineAnnealingLR(optimizer,
                                       T_max=max(total_epochs - warmup_epochs, 1),
                                       eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup_sched, cosine_sched],
                              milestones=[warmup_epochs])

    return optimizer, scheduler


# -----------------------------------------------------------------------
# Train / validate
# -----------------------------------------------------------------------

def _orthogonality_loss(*stream_feats: torch.Tensor) -> torch.Tensor:
    """
    Encourage streams to capture orthogonal (non-redundant) features.
    Projects each stream to unit norm, then penalizes off-diagonal cosine
    similarity in the batch Gram matrix across streams.
    """
    norms = [nn.functional.normalize(f.mean(dim=0, keepdim=True), dim=-1)
             for f in stream_feats]
    loss = torch.tensor(0.0, device=stream_feats[0].device)
    n = len(norms)
    for i in range(n):
        for j in range(i + 1, n):
            # Project to same space via dot product of norms (scalar cosine sim)
            sim = (norms[i] * norms[j]).sum().abs()
            loss = loss + sim
    return loss / max(n * (n - 1) / 2, 1)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    scaler: torch.amp.GradScaler = None
) -> MetricsTracker:
    model.train()
    metrics = MetricsTracker()
    pbar    = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    use_amp = scaler is not None

    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, features = model(images, return_features=True)
            logits    = logits.squeeze(-1)
            loss = criterion(logits, labels) / gradient_accumulation_steps
            # Orthogonality loss: minimize cosine similarity between stream norms.
            # Uses batch-level Gram matrix so dimension mismatch is not an issue.
            if features is not None:
                orth_loss = _orthogonality_loss(
                    features['spatial'], features['frequency'], features['semantic'])
                loss = loss + 0.01 * orth_loss / gradient_accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        metrics.update(loss.item() * gradient_accumulation_steps, logits.float(), labels)
        computed = metrics.compute()
        pbar.set_postfix({
            'loss': f"{computed['loss']:.4f}",
            'acc':  f"{computed['accuracy']:.1f}%",
            'auc':  f"{computed['auc']:.1f}%"
        })

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> MetricsTracker:
    model.eval()
    metrics = MetricsTracker()
    pbar    = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    use_amp = device.type == 'cuda'

    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(images)
            logits    = logits.squeeze(-1)
            loss = criterion(logits, labels)

        metrics.update(loss.item(), logits.float(), labels)

        computed = metrics.compute()
        pbar.set_postfix({
            'loss': f"{computed['loss']:.4f}",
            'acc':  f"{computed['accuracy']:.1f}%",
            'auc':  f"{computed['auc']:.1f}%"
        })

    return metrics


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def train(
    config_path: str   = '../configs/config.yaml',
    resume_from: str   = None,
    data_dir: str      = None,
    epochs: int        = None,
    batch_size: int    = None,
    ablation_mode: str = None
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if data_dir:    config['data']['root_dir']        = data_dir
    if epochs:      config['training']['epochs']      = epochs
    if batch_size:  config['training']['batch_size']  = batch_size

    set_seed(config['seed'])
    device = get_device()

    ablation_suffix = f"_{ablation_mode}" if ablation_mode else ""
    checkpoint_dir = Path(config['logging']['checkpoint_dir']) / f"ablation{ablation_suffix}" if ablation_mode else Path(config['logging']['checkpoint_dir'])
    log_dir        = Path(config['logging']['log_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))

    # Wandb (optional)
    use_wandb = (
        _WANDB_AVAILABLE
        and config.get('logging', {}).get('wandb', {}).get('enabled', False)
    )
    if use_wandb:
        wandb.init(
            project=config['logging']['wandb'].get('project', 'deepfake-detection'),
            entity=config['logging']['wandb'].get('entity') or None,
            config=config,
            name=f"run_{int(time.time())}"
        )
        print("Wandb logging enabled.")
    else:
        print("Wandb disabled (set logging.wandb.enabled: true in config to enable).")

    print("=" * 70)
    print("MULTI-STREAM DEEPFAKE DETECTION — TRAINING")
    print("=" * 70)
    print(f"Device:     {device}")
    print(f"Data:       {config['data']['root_dir']}")
    print(f"Epochs:     {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"LR:         {config['training']['lr']}")
    print("=" * 70)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=config['data']['root_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            img_size=256,
            augmentation_level=config['training'].get('augmentation_level', 'medium')
        )
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        print("Run: python scripts/download_datasets.py")
        return

    if ablation_mode:
        print(f"ABLATION MODE: {ablation_mode}")

    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=config['model']['spatial']['feature_dim'],
        freq_output_dim=config['model']['frequency']['feature_dim'],
        semantic_output_dim=config['model']['semantic']['feature_dim'],
        fusion_hidden_dim=config['model']['fusion']['hidden_dim'],
        fusion_attention_heads=config['model']['fusion']['attention_heads'],
        pretrained_backbones=config['model']['spatial']['pretrained'],
        ablation_mode=ablation_mode
    ).to(device)

    n_params = model.count_parameters()
    print(f"\nModel: {n_params:,} parameters ({n_params / 1e6:.2f}M)")

    # Loss
    loss_type = config['training']['loss']['type']
    if loss_type == 'weighted_bce':
        pw = float(config['training']['loss'].get('pos_weight', 1.5))
        pos_weight = torch.tensor([pw]).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Weighted BCE: pos_weight={pw:.2f}")
    elif loss_type == 'focal':
        # Focal loss approximation via label smoothing + BCE
        criterion = nn.BCEWithLogitsLoss(label_smoothing=0.05) \
            if hasattr(nn.BCEWithLogitsLoss, 'label_smoothing') \
            else nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay']),
        warmup_epochs=int(config['training']['warmup_epochs']),
        total_epochs=int(config['training']['epochs'])
    )

    # AMP (Automatic Mixed Precision) — critical for Blackwell GPUs
    scaler = None
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        scaler = torch.amp.GradScaler('cuda')
        print("AMP (fp16) enabled for GPU training.")

    early_stopping = EarlyStopping(patience=config['training']['patience'], mode='max')

    start_epoch  = 1
    best_val_auc = 0.0

    if resume_from and os.path.exists(resume_from):
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}")

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            gradient_accumulation_steps=config['training'].get('gradient_accumulation', 1),
            max_grad_norm=config['training'].get('max_grad_norm', 1.0),
            scaler=scaler
        )
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t0

        tr = train_metrics.compute()
        vl = val_metrics.compute()

        print(
            f"\nEpoch {epoch:>3}/{config['training']['epochs']} | "
            f"{epoch_time:.0f}s | LR={current_lr:.2e}"
        )
        print(
            f"  Train | Loss={tr['loss']:.4f} | Acc={tr['accuracy']:.1f}% | "
            f"F1={tr['f1_score']:.1f}% | AUC={tr['auc']:.1f}% | EER={tr['eer']:.1f}%"
        )
        print(
            f"  Val   | Loss={vl['loss']:.4f} | Acc={vl['accuracy']:.1f}% | "
            f"F1={vl['f1_score']:.1f}% | AUC={vl['auc']:.1f}% | EER={vl['eer']:.1f}%"
        )

        # TensorBoard
        tb_metrics = {
            'Loss/train': tr['loss'],    'Loss/val':   vl['loss'],
            'Acc/train':  tr['accuracy'],'Acc/val':    vl['accuracy'],
            'F1/train':   tr['f1_score'],'F1/val':     vl['f1_score'],
            'AUC/train':  tr['auc'],     'AUC/val':    vl['auc'],
            'EER/train':  tr['eer'],     'EER/val':    vl['eer'],
            'LR':         current_lr
        }
        for tag, val in tb_metrics.items():
            writer.add_scalar(tag, val, epoch)

        # Wandb
        if use_wandb:
            wandb.log({
                'epoch':      epoch,
                'train/loss': tr['loss'],   'val/loss':  vl['loss'],
                'train/acc':  tr['accuracy'],'val/acc':  vl['accuracy'],
                'train/auc':  tr['auc'],    'val/auc':   vl['auc'],
                'train/eer':  tr['eer'],    'val/eer':   vl['eer'],
                'train/f1':   tr['f1_score'],'val/f1':   vl['f1_score'],
                'lr':         current_lr
            })

        # Save best model (by AUC — more reliable than accuracy)
        if vl['auc'] > best_val_auc:
            best_val_auc = vl['auc']
            save_checkpoint(model, optimizer, epoch, vl['loss'],
                            str(checkpoint_dir / 'best_model.pth'))
            print(f"  >>> Best model saved  (AUC={best_val_auc:.1f}%)")

        save_checkpoint(model, optimizer, epoch, vl['loss'],
                        str(checkpoint_dir / 'latest_checkpoint.pth'))

        if early_stopping(vl['auc']):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best Validation AUC: {best_val_auc:.1f}%")
    print(f"Checkpoint:          {checkpoint_dir / 'best_model.pth'}")
    print(f"TensorBoard:         {log_dir / 'tensorboard'}")
    print("=" * 70)

    writer.close()
    if use_wandb:
        wandb.finish()

    return model, best_val_auc


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Stream Deepfake Detector")
    parser.add_argument('--config',     type=str, default='../configs/config.yaml')
    parser.add_argument('--resume',     type=str, default=None)
    parser.add_argument('--data-dir',   type=str, default=None)
    parser.add_argument('--epochs',     type=int, default=None)
    parser.add_argument('--batch-size',    type=int,  default=None)
    parser.add_argument('--ablation-mode', type=str,  default=None,
                        choices=[None, 'spatial_only', 'freq_only', 'semantic_only',
                                 'spatial_freq', 'spatial_semantic'],
                        help='Ablation: which streams to enable (others zeroed)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        config_path=args.config,
        resume_from=args.resume,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ablation_mode=args.ablation_mode
    )
