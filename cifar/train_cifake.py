"""
Train MS-DFD model on CiFake Dataset.

This is the Phase 1 training script — using CiFake (CIFAR-10 real +
Stable Diffusion fake) as the dataset.

NOTE: CiFake results showed near-100% AUC during training, but poor
robustness under JPEG compression (67.8% at Q=75 — a 32pp drop).
This is why the dataset was eventually replaced with FFHQ + DiffusionDB.

Usage:
    python cifar/train_cifake.py --data-dir cifar/data --epochs 30

Requirements:
    pip install torch torchvision timm albumentations scikit-learn tqdm
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cifar.dataset_cifake import get_cifake_loaders


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

DEFAULT_CONFIG = {
    "data_dir":      "cifar/data",
    "epochs":        30,
    "batch_size":    128,
    "lr":            3e-4,
    "weight_decay":  1e-4,
    "patience":      8,         # early stopping on validation AUC
    "num_workers":   4,
    "seed":          42,
    "img_size":      256,
    "save_dir":      "cifar/checkpoints",
}


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------

def setup_logger(save_dir: Path) -> logging.Logger:
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().squeeze().numpy()
        all_probs.extend(probs if hasattr(probs, '__iter__') else [probs])
        all_labels.extend(labels.cpu().squeeze().numpy().tolist()
                          if labels.numel() > 1 else [labels.cpu().item()])

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), auc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for batch in tqdm(loader, desc="Val  ", leave=False):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().squeeze().numpy()
        all_probs.extend(probs if hasattr(probs, '__iter__') else [probs])
        all_labels.extend(labels.cpu().squeeze().numpy().tolist()
                          if labels.numel() > 1 else [labels.cpu().item()])

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), auc


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    logger = setup_logger(save_dir)

    logger.info("=" * 60)
    logger.info("CiFake Phase 1 Training")
    logger.info(f"Device: {device}")
    logger.info(f"Data:   {args.data_dir}")
    logger.info("=" * 60)

    # Data
    train_loader, val_loader, test_loader = get_cifake_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # Model — import full MS-DFD model
    try:
        from models.full_model import MultiStreamDeepfakeDetector
        model = MultiStreamDeepfakeDetector(mode='full').to(device)
        logger.info("Model: MultiStreamDeepfakeDetector (full 3-stream)")
    except ImportError:
        logger.warning("MS-DFD model not found. Using simple CNN baseline.")
        import torchvision.models as tv_models
        backbone = tv_models.efficientnet_b0(weights='DEFAULT')
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, 1)
        model = backbone.to(device)
        logger.info("Model: EfficientNet-B0 (spatial only, fallback)")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params / 1e6:.1f}M")

    # Loss — CiFake is balanced (50k/50k) so pos_weight ~ 1.0
    # But compute it from actual split just in case
    n_real = train_loader.dataset.labels.count(0)
    n_fake = train_loader.dataset.labels.count(1)
    pos_weight_val = n_real / max(n_fake, 1)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"pos_weight: {pos_weight_val:.2f}")

    # Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda')

    # Training
    best_val_auc  = 0.0
    patience_left = args.patience
    best_ckpt     = save_dir / "best_model_cifake.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss,   val_auc   = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_left = args.patience
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_auc':   val_auc,
                'args':      vars(args)
            }, best_ckpt)
            logger.info(f"  ✅ New best: {best_val_auc:.4f} — saved to {best_ckpt}")
        else:
            patience_left -= 1
            logger.info(f"  No improvement. Patience: {patience_left}/{args.patience}")
            if patience_left == 0:
                logger.info("Early stopping triggered.")
                break

    # Final test evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Final Test Evaluation")
    logger.info("=" * 60)
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    test_loss, test_auc = eval_epoch(model, test_loader, criterion, device)
    logger.info(f"Test AUC:  {test_auc:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Best Epoch: {ckpt['epoch']}")
    logger.info("\nNOTE: Run evaluate_cifake.py for full metrics + robustness test")


if __name__ == "__main__":
    cfg = DEFAULT_CONFIG

    parser = argparse.ArgumentParser(description="Train on CiFake Dataset")
    parser.add_argument("--data-dir",    default=cfg["data_dir"])
    parser.add_argument("--epochs",      type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size",  type=int, default=cfg["batch_size"])
    parser.add_argument("--lr",          type=float, default=cfg["lr"])
    parser.add_argument("--weight-decay",type=float, default=cfg["weight_decay"])
    parser.add_argument("--patience",    type=int, default=cfg["patience"])
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--seed",        type=int, default=cfg["seed"])
    parser.add_argument("--save-dir",    default=cfg["save_dir"])
    args = parser.parse_args()

    main(args)
