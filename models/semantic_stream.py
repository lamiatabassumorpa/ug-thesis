"""
Semantic Domain Stream - FAT-Lite (Frequency-Aware Transformer Lite)
Lightweight Vision Transformer for global structural inconsistency detection.
Output: EXACTLY 384-dimensional feature vector.

Fixes applied:
  1. Redundant 192→384→384 projection simplified to 192→384 (single linear).
  2. Input handled natively at 256×256 via timm's img_size pos-embed interpolation
     — no more silent resize in forward().
"""
import torch
import torch.nn as nn
import timm


class FATLiteTransformer(nn.Module):
    """
    Semantic stream using ViT-Tiny for global structural anomaly detection.
    Particularly effective against Diffusion model-generated images.

    Architecture:
        Input:    [B, 3, 256, 256]
        Backbone: ViT-Tiny-Patch16 (pos-embeds interpolated to 256)
        Project:  Linear(192 → 384) + LayerNorm + GELU + Dropout
        Output:   [B, 384]
    """

    def __init__(
        self,
        output_dim: int = 384,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super(FATLiteTransformer, self).__init__()

        assert output_dim == 384, f"FAT-Lite output must be 384, got {output_dim}"

        # ViT-Tiny pretrained on ImageNet-21k at 224px.
        # img_size=256 tells timm to interpolate position embeddings to 256×256,
        # so we feed 256px images without any resize in forward().
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='token',
            img_size=256          # pos-embed interpolation: 196 tokens → 256 tokens
        )

        backbone_feat_dim = self.backbone.num_features  # 192 for vit_tiny

        # Single-step projection: 192 → 384
        # Previous version did 192→384→384 (second Linear was redundant).
        self.projection = nn.Sequential(
            nn.Linear(backbone_feat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 256, 256]  — no internal resize needed

        Returns:
            [B, 384]
        """
        features = self.backbone(x)        # [B, 192]
        output   = self.projection(features)  # [B, 384]
        return output

    def get_feature_dim(self) -> int:
        return 384


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING: FAT-Lite Transformer (Semantic Stream)")
    print("=" * 60)

    model = FATLiteTransformer(output_dim=384, pretrained=False)
    model.eval()

    x = torch.randn(4, 3, 256, 256)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (4, 384), \
        f"Dimension mismatch! Expected (4, 384), got {output.shape}"
    print("\nFAT-Lite test PASSED!")
