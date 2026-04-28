"""
Spatial Domain Stream - NPR (Neural Pattern Reconstruction) Branch
Extracts pixel-level GAN artifacts and boundary inconsistencies.
Output: EXACTLY 128-dimensional feature vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple


class NPRBranch(nn.Module):
    """
    Spatial stream using EfficientNet-B0 backbone for pixel-level artifact detection.
    
    Architecture:
        Input: [B, 3, 256, 256]
        Backbone: EfficientNet-B0 (pretrained)
        Projection: MLP -> 128-dim
        Output: [B, 128]
    """
    
    def __init__(
        self,
        output_dim: int = 128,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super(NPRBranch, self).__init__()
        
        assert output_dim == 128, f"NPR Branch output must be 128, got {output_dim}"
        
        # EfficientNet-B0 as feature extractor
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        # Get backbone output dimension
        backbone_feat_dim = self.backbone.num_features
        
        # Project to 128-dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection layer weights."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, 256, 256]
        
        Returns:
            Feature tensor [B, 128]
        """
        # Backbone feature extraction
        features = self.backbone(x)  # [B, 1280]
        
        # Project to 128-dim
        output = self.projection(features)  # [B, 128]
        
        return output
    
    def get_feature_dim(self) -> int:
        return 128


if __name__ == "__main__":
    print("="*60)
    print("TESTING: NPR Branch (Spatial Stream)")
    print("="*60)
    
    # Test with dummy tensor
    model = NPRBranch(output_dim=128, pretrained=False)
    model.eval()
    
    # Input: [Batch, Channels, Height, Width]
    x = torch.randn(4, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dim:   {output.shape[1]}")
    
    assert output.shape == (4, 128), f"Dimension mismatch! Expected (4, 128), got {output.shape}"
    print("\nNPR Branch test PASSED!")
