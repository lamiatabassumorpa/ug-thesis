"""
Baseline Models for Comparison
================================
CNNDetect  — Wang et al., "CNN-generated images are surprisingly easy to spot", CVPR 2020.
             ResNet-50 fine-tuned for binary real/fake classification.

UnivFD     — Ojha et al., "Towards Universal Fake Image Detection by Exploiting
             CLIP's Potential", CVPR 2023.
             CLIP ViT-L/14 features + linear layer. No fine-tuning of CLIP.

XceptionDetect — Rossler et al., "FaceForensics++: Learning to Detect Manipulated
             Facial Images", ICCV 2019. THE standard baseline for FF++ protocol.
             Xception pretrained on ImageNet, fine-tuned for binary classification.

F3Net      — Qian et al., "Thinking in Frequency: Face Forgery Detection by
             Mining Frequency-Aware Clues", ECCV 2020.
             Dual-branch: spatial (Xception-lite) + frequency (DCT + learnable weights).
             Attention-based fusion with frequency-aware discriminative features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import Tuple, Optional

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# -----------------------------------------------------------------------
# CNNDetect baseline (Wang et al. CVPR 2020)
# -----------------------------------------------------------------------

class CNNDetect(nn.Module):
    """
    ResNet-50 pretrained on ImageNet, last FC replaced with binary classifier.
    Exactly matches Wang et al. 2020 architecture.
    Training: blur augmentation + JPEG augmentation during training.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Replace final FC: 2048 → 1 (binary logit)
        backbone.fc = nn.Linear(2048, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        logits = self.backbone(x)   # [B, 1]
        return logits, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# UnivFD baseline (Ojha et al. CVPR 2023)
# -----------------------------------------------------------------------

class UnivFD(nn.Module):
    """
    CLIP ViT-L/14 features (frozen) + single linear layer.
    CLIP is NOT fine-tuned — only the linear probe is trained.
    This is the key insight of UnivFD: CLIP features generalise across generators.
    """

    def __init__(self, clip_model: str = 'ViT-L-14', pretrained: str = 'openai'):
        super().__init__()
        try:
            import open_clip
            self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model, pretrained=pretrained
            )
        except Exception as e:
            raise ImportError(f"open_clip required: pip install open-clip-torch\n{e}")

        # Freeze all CLIP parameters
        for p in self.clip.parameters():
            p.requires_grad = False

        # Linear probe: 768 (ViT-L/14 dim) → 1
        clip_dim = self.clip.visual.output_dim
        self.linear = nn.Linear(clip_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            feats = self.clip.encode_image(x)           # [B, 768]
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalise

        logits = self.linear(feats.float())             # [B, 1]
        return logits, None

    def count_parameters(self) -> int:
        # Only the linear probe is trainable
        return sum(p.numel() for p in self.linear.parameters())


# -----------------------------------------------------------------------
# XceptionDetect (Rossler et al. ICCV 2019)
# THE standard baseline for FaceForensics++ protocol.
# -----------------------------------------------------------------------

class XceptionDetect(nn.Module):
    """
    Xception fine-tuned for deepfake detection.
    Standard baseline used in FaceForensics++ and most subsequent papers.
    Trained end-to-end (unlike UnivFD which freezes the backbone).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm required: pip install timm")

        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features  # 2048

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 1),
        )
        nn.init.xavier_uniform_(self.classifier[-1].weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Xception expects 299×299, resize if needed
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear",
                              align_corners=False)
        feats = self.backbone(x)        # [B, 2048]
        logits = self.classifier(feats) # [B, 1]
        return logits, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# F3Net (Qian et al. ECCV 2020)
# Frequency-Aware Discriminative Feature Learning
# -----------------------------------------------------------------------

class DCTLayer(nn.Module):
    """
    Differentiable DCT-based frequency decomposition on 8×8 blocks.
    Applies learned per-frequency-component weights to emphasize
    manipulation-indicative frequencies.
    """

    def __init__(self, n_freqs: int = 64):
        super().__init__()
        self.n_freqs = n_freqs
        # Learnable per-frequency weight vector (initialized to ones)
        self.freq_weights = nn.Parameter(torch.ones(n_freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] image tensor (C=1, grayscale)

        Returns:
            freq_features: [B, n_freqs, H//8, W//8]
        """
        B, C, H, W = x.shape
        # Pad to multiple of 8
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        x = F.pad(x, (0, pw, 0, ph))
        _, _, H2, W2 = x.shape

        # Extract 8×8 non-overlapping patches: [B, C, H2/8, 8, W2/8, 8]
        x = x.view(B, C, H2 // 8, 8, W2 // 8, 8)
        x = x.permute(0, 1, 2, 4, 3, 5)  # [B, C, H/8, W/8, 8, 8]

        # 2D DCT via orthogonal FFT approximation
        x_dct = torch.fft.rfft2(x.float(), norm="ortho")  # [B, C, H/8, W/8, 8, 5]
        x_dct = torch.abs(x_dct)

        # Flatten 8×5 → 40 components, pad/truncate to n_freqs
        B2, C2, Hb, Wb, _, _ = x_dct.shape
        x_dct = x_dct.reshape(B2, C2, Hb, Wb, -1)  # [B, C, Hb, Wb, 40]

        # Take first n_freqs components
        n = min(self.n_freqs, x_dct.shape[-1])
        x_dct = x_dct[..., :n]  # [B, C, Hb, Wb, n]
        if n < self.n_freqs:
            pad = torch.zeros(*x_dct.shape[:-1], self.n_freqs - n,
                              device=x.device, dtype=x_dct.dtype)
            x_dct = torch.cat([x_dct, pad], dim=-1)

        # Apply learnable frequency weights
        w = torch.sigmoid(self.freq_weights).view(1, 1, 1, 1, self.n_freqs)
        x_dct = x_dct * w

        # Rearrange: [B, n_freqs, Hb, Wb] by averaging over channels
        x_dct = x_dct.mean(dim=1)                     # [B, Hb, Wb, n_freqs]
        x_dct = x_dct.permute(0, 3, 1, 2).contiguous()  # [B, n_freqs, Hb, Wb]

        return x_dct


class F3Net(nn.Module):
    """
    Simplified F3Net: dual-stream frequency-aware forgery detection.

    Stream 1 (FAD — Frequency-Aware Decomposition):
        Grayscale → DCT → learnable frequency weights → CNN features

    Stream 2 (LFS — Local Frequency Statistics):
        RGB → high-frequency residuals (Laplacian) → CNN features

    Fusion: channel attention + concatenation → binary classifier.

    This is a faithful simplification of Qian et al. ECCV 2020 that
    captures the core frequency-discriminative learning idea without
    requiring the original SRM filter bank (which needs external files).
    """

    def __init__(self, pretrained: bool = True, img_size: int = 224):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm required: pip install timm")

        # --- FAD branch: DCT features ---
        self.n_freqs = 12  # keep 12 low+mid DCT components
        self.dct = DCTLayer(n_freqs=self.n_freqs)

        # Lightweight CNN to process frequency maps
        self.fad_encoder = nn.Sequential(
            nn.Conv2d(self.n_freqs, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(14),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(7),
        )
        fad_dim = 64 * 7 * 7  # 3136

        # --- LFS branch: high-frequency residual features ---
        # Laplacian kernel for sharpness / manipulation residuals
        self.register_buffer(
            "laplacian",
            torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                         dtype=torch.float32),
        )

        # EfficientNet-B0 backbone for LFS features
        self.lfs_backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=4,  # 3 RGB + 1 Laplacian residual
        )
        lfs_dim = self.lfs_backbone.num_features  # 1280

        # --- Channel attention for FAD branch ---
        fad_proj_dim = 256
        self.fad_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fad_dim, fad_proj_dim),
            nn.LayerNorm(fad_proj_dim),
            nn.ReLU(inplace=True),
        )

        lfs_proj_dim = 256
        self.lfs_proj = nn.Sequential(
            nn.Linear(lfs_dim, lfs_proj_dim),
            nn.LayerNorm(lfs_proj_dim),
            nn.ReLU(inplace=True),
        )

        # Attention gate (learn which branch to trust per sample)
        combined = fad_proj_dim + lfs_proj_dim
        self.attention = nn.Sequential(
            nn.Linear(combined, combined // 4),
            nn.ReLU(inplace=True),
            nn.Linear(combined // 4, 2),
            nn.Softmax(dim=-1),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        B = x.size(0)

        # FAD: grayscale → DCT features
        gray = x.mean(dim=1, keepdim=True)       # [B, 1, H, W]
        freq_maps = self.dct(gray)               # [B, n_freqs, Hb, Wb]
        fad_feat = self.fad_proj(self.fad_encoder(freq_maps))  # [B, 256]

        # LFS: RGB + Laplacian residual
        lap = F.conv2d(gray, self.laplacian, padding=1)  # [B, 1, H, W]
        x_aug = torch.cat([x, lap], dim=1)      # [B, 4, H, W]
        lfs_feat = self.lfs_proj(self.lfs_backbone(x_aug))  # [B, 256]

        # Attention-weighted combination
        combined = torch.cat([fad_feat, lfs_feat], dim=1)  # [B, 512]
        attn = self.attention(combined)                      # [B, 2]
        fused = (attn[:, 0:1] * fad_feat +
                 attn[:, 1:2] * lfs_feat)                   # [B, 256]

        logits = self.classifier(combined)
        return logits, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# Convenience registry
# -----------------------------------------------------------------------

BASELINE_REGISTRY = {
    "cnndetect": CNNDetect,
    "univfd": UnivFD,
    "xception": XceptionDetect,
    "f3net": F3Net,
}


def build_baseline(name: str, **kwargs) -> nn.Module:
    """Factory function to build a baseline by name."""
    name_lower = name.lower()
    if name_lower not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. "
            f"Available: {list(BASELINE_REGISTRY.keys())}"
        )
    return BASELINE_REGISTRY[name_lower](**kwargs)
