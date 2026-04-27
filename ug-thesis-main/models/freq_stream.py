"""
Frequency Domain Stream - FreqBlender
Uses learnable FFT masks to detect spectral inconsistencies.
Output: EXACTLY 64-dimensional feature vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple


class LearnableFFTMask(nn.Module):
    """
    Learnable frequency mask applied in Fourier domain.
    Allows the model to focus on frequency bands most indicative of manipulation.
    """
    
    def __init__(self, img_size: int = 256, n_channels: int = 3):
        super(LearnableFFTMask, self).__init__()
        # Learnable mask (initialized to ones - passthrough)
        # Mask shape: [C, H, W] for per-channel filtering
        self.mask = nn.Parameter(
            torch.ones(n_channels, img_size, img_size),
            requires_grad=True
        )
        
        # Frequency band attention
        self.band_weights = nn.Parameter(
            torch.ones(4),  # Low, Mid-Low, Mid-High, High
            requires_grad=True
        )
        
        self.img_size = img_size
        self.n_channels = n_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable frequency mask.
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            Filtered image [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # FFT: spatial -> frequency
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft)
        
        # Get magnitude and phase
        magnitude = torch.abs(x_fft_shifted)
        phase = torch.angle(x_fft_shifted)
        
        # Create radial frequency bands for structured filtering
        center_h, center_w = H // 2, W // 2
        y_coords = torch.arange(H, device=x.device).float() - center_h
        x_coords = torch.arange(W, device=x.device).float() - center_w
        dist = torch.sqrt(y_coords.unsqueeze(1)**2 + x_coords.unsqueeze(0)**2)
        
        # Normalize distance to [0, 1]
        max_dist = torch.sqrt(torch.tensor(center_h**2 + center_w**2, device=x.device))
        dist_norm = dist / (max_dist + 1e-8)
        
        # Create band masks
        low_mask = (dist_norm < 0.25).float()
        mid_low_mask = ((dist_norm >= 0.25) & (dist_norm < 0.5)).float()
        mid_high_mask = ((dist_norm >= 0.5) & (dist_norm < 0.75)).float()
        high_mask = (dist_norm >= 0.75).float()
        
        # Weighted combination
        band_mask = (
            self.band_weights[0] * low_mask +
            self.band_weights[1] * mid_low_mask +
            self.band_weights[2] * mid_high_mask +
            self.band_weights[3] * high_mask
        )  # [H, W]
        
        # Expand mask for batch and channels: [B, C, H, W]
        combined_mask = self.mask.unsqueeze(0).expand(B, -1, -1, -1) * \
                        band_mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        
        # Apply mask to magnitude
        magnitude_filtered = magnitude * combined_mask
        
        # Reconstruct complex spectrum
        x_fft_filtered = magnitude_filtered * torch.exp(1j * phase)
        
        # Inverse FFT: frequency -> spatial
        x_ifft = torch.fft.ifft2(torch.fft.ifftshift(x_fft_filtered, dim=(-2, -1)), dim=(-2, -1))
        
        return torch.abs(x_ifft).real


class FreqBlender(nn.Module):
    """
    Frequency stream using learnable FFT filtering + ResNet-18 backbone.
    
    Architecture:
        Input: [B, 3, 256, 256]
        FFT Mask: Learnable spectral filtering
        Backbone: ResNet-18 (pretrained)
        Projection: MLP -> 64-dim
        Output: [B, 64]
    """
    
    def __init__(
        self,
        output_dim: int = 64,
        pretrained: bool = True,
        img_size: int = 256,
        dropout_rate: float = 0.3
    ):
        super(FreqBlender, self).__init__()
        
        assert output_dim == 64, f"FreqBlender output must be 64, got {output_dim}"
        
        # Learnable FFT mask
        self.fft_mask = LearnableFFTMask(img_size=img_size, n_channels=3)
        
        # ResNet-18 backbone (lightweight for frequency features)
        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        backbone_feat_dim = self.backbone.num_features
        
        # Project to 64-dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
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
            Feature tensor [B, 64]
        """
        # Frequency filtering
        x_freq = self.fft_mask(x)  # [B, 3, 256, 256]
        
        # Backbone feature extraction
        features = self.backbone(x_freq)  # [B, 512]
        
        # Project to 64-dim
        output = self.projection(features)  # [B, 64]
        
        return output
    
    def get_feature_dim(self) -> int:
        return 64


if __name__ == "__main__":
    print("="*60)
    print("TESTING: FreqBlender (Frequency Stream)")
    print("="*60)
    
    model = FreqBlender(output_dim=64, pretrained=False, img_size=256)
    model.eval()
    
    x = torch.randn(4, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dim:   {output.shape[1]}")
    
    assert output.shape == (4, 64), f"Dimension mismatch! Expected (4, 64), got {output.shape}"
    print("\nFreqBlender test PASSED!")
