"""
Complete Multi-Stream Deepfake Detection Model
Integrates all 3 streams + MLAF fusion + GradCAM++ localization.
Target: ~25M parameters, ~95MB model size.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .spatial_stream import NPRBranch
from .freq_stream import FreqBlender
from .semantic_stream import FATLiteTransformer
from .fusion import MLAFFusion


class MultiStreamDeepfakeDetector(nn.Module):
    """
    Multi-Stream Fusion Deepfake Detector.
    
    Architecture:
        Input: [B, 3, 256, 256]
        ├── Stream 1 (Spatial/NPR): [B, 128]
        ├── Stream 2 (FreqBlender): [B, 64]
        ├── Stream 3 (FAT-Lite):    [B, 384]
        └── MLAF Fusion:            [B, 1] (logit)
    
    Total Parameters: ~25M
    """
    
    # Valid ablation modes — which streams are ACTIVE (others zeroed)
    ABLATION_MODES = {
        None: (True, True, True),               # full model
        "spatial_only":   (True,  False, False),
        "freq_only":      (False, True,  False),
        "semantic_only":  (False, False, True),
        "spatial_freq":   (True,  True,  False),
        "spatial_semantic": (True, False, True),
    }

    def __init__(
        self,
        spatial_output_dim: int = 128,
        freq_output_dim: int = 64,
        semantic_output_dim: int = 384,
        fusion_hidden_dim: int = 256,
        fusion_attention_heads: int = 4,
        pretrained_backbones: bool = True,
        ablation_mode: str = None,
        stream_dropout_p: float = 0.1,
    ):
        assert ablation_mode in self.ABLATION_MODES, \
            f"ablation_mode must be one of {list(self.ABLATION_MODES.keys())}"
        super(MultiStreamDeepfakeDetector, self).__init__()
        self.ablation_mode = ablation_mode
        self._use_spatial, self._use_freq, self._use_semantic = \
            self.ABLATION_MODES[ablation_mode]
        self.stream_dropout_p = stream_dropout_p
        
        # Three parallel streams
        self.spatial_stream = NPRBranch(
            output_dim=spatial_output_dim,
            pretrained=pretrained_backbones
        )
        
        self.freq_stream = FreqBlender(
            output_dim=freq_output_dim,
            pretrained=pretrained_backbones
        )
        
        self.semantic_stream = FATLiteTransformer(
            output_dim=semantic_output_dim,
            pretrained=pretrained_backbones
        )
        
        # Fusion module
        self.fusion = MLAFFusion(
            spatial_dim=spatial_output_dim,
            freq_dim=freq_output_dim,
            semantic_dim=semantic_output_dim,
            hidden_dim=fusion_hidden_dim,
            attention_heads=fusion_attention_heads
        )
        
        # Store dimensions
        self.combined_dim = spatial_output_dim + freq_output_dim + semantic_output_dim
        
        # Initialize fusion-specific weights
        self._init_fusion_weights()
    
    def _init_fusion_weights(self):
        """Additional weight initialization for fusion."""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through all streams and fusion.
        
        Args:
            x: Input images [B, 3, 256, 256]
            return_features: If True, return per-stream features
        
        Returns:
            logits: [B, 1] classification logits
            features: Dict of per-stream and fused features (if return_features=True)
        """
        # Parallel forward — disabled streams are replaced with zeros
        spatial_feat  = self.spatial_stream(x)  if self._use_spatial  else torch.zeros(x.size(0), self.fusion.cross_stream_attn.spatial_proj[0].in_features,  device=x.device)
        freq_feat     = self.freq_stream(x)     if self._use_freq     else torch.zeros(x.size(0), self.fusion.cross_stream_attn.freq_proj[0].in_features,     device=x.device)
        semantic_feat = self.semantic_stream(x) if self._use_semantic else torch.zeros(x.size(0), self.fusion.cross_stream_attn.semantic_proj[0].in_features, device=x.device)

        # Stream Dropout: randomly zero entire streams during training to prevent co-adaptation.
        # Each stream is dropped independently; at least one always survives via sequential masking.
        if self.training and self.stream_dropout_p > 0.0 and self.ablation_mode is None:
            streams = [spatial_feat, freq_feat, semantic_feat]
            drop_mask = torch.rand(3, device=x.device) < self.stream_dropout_p
            # Ensure at least one stream is active
            if drop_mask.all():
                drop_mask[torch.randint(3, (1,))] = False
            if drop_mask[0]: spatial_feat  = torch.zeros_like(spatial_feat)
            if drop_mask[1]: freq_feat     = torch.zeros_like(freq_feat)
            if drop_mask[2]: semantic_feat = torch.zeros_like(semantic_feat)
        
        # Fusion and classification
        logits, fused_features = self.fusion(
            spatial_feat, freq_feat, semantic_feat
        )
        
        features = None
        if return_features:
            features = {
                'spatial': spatial_feat,
                'frequency': freq_feat,
                'semantic': semantic_feat,
                'fused': fused_features,
                'combined': torch.cat([spatial_feat, freq_feat, semantic_feat], dim=1)
            }
        
        return logits, features
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Binary prediction.
        
        Args:
            x: Input images [B, 3, 256, 256]
            threshold: Sigmoid threshold
        
        Returns:
            predictions: [B] binary predictions (0=real, 1=fake)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self(x)
            probs = torch.sigmoid(logits)
            predictions = (probs > threshold).long().squeeze(-1)
        return predictions
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get fake probability."""
        self.eval()
        with torch.no_grad():
            logits, _ = self(x)
            probs = torch.sigmoid(logits)
        return probs.squeeze(-1)
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def count_parameters_per_stream(self) -> Dict[str, int]:
        """Count parameters per stream for analysis."""
        counts = {
            'spatial': sum(p.numel() for p in self.spatial_stream.parameters()),
            'frequency': sum(p.numel() for p in self.freq_stream.parameters()),
            'semantic': sum(p.numel() for p in self.semantic_stream.parameters()),
            'fusion': sum(p.numel() for p in self.fusion.parameters())
        }
        counts['total'] = sum(counts.values())
        return counts


if __name__ == "__main__":
    print("="*70)
    print("FULL MODEL TEST: Multi-Stream Deepfake Detector")
    print("="*70)
    
    # Create model
    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=128,
        freq_output_dim=64,
        semantic_output_dim=384,
        fusion_hidden_dim=256,
        fusion_attention_heads=4,
        pretrained_backbones=False
    )
    model.eval()
    
    # Test forward pass
    x = torch.randn(4, 3, 256, 256)
    
    with torch.no_grad():
        logits, features = model(x, return_features=True)
    
    print(f"\nInput shape:          {x.shape}")
    print(f"Logits output:        {logits.shape}")
    print(f"Spatial features:     {features['spatial'].shape}")
    print(f"Frequency features:   {features['frequency'].shape}")
    print(f"Semantic features:    {features['semantic'].shape}")
    print(f"Fused features:       {features['fused'].shape}")
    print(f"Combined features:    {features['combined'].shape}")
    
    # Dimension verification
    assert logits.shape == (4, 1), f"Logits mismatch: {logits.shape}"
    assert features['spatial'].shape == (4, 128), f"Spatial mismatch"
    assert features['frequency'].shape == (4, 64), f"Freq mismatch"
    assert features['semantic'].shape == (4, 384), f"Semantic mismatch"
    assert features['fused'].shape == (4, 256), f"Fused mismatch"
    assert features['combined'].shape == (4, 576), f"Combined mismatch"
    
    # Parameter count
    total_params = model.count_parameters()
    per_stream = model.count_parameters_per_stream()
    
    print(f"\n{'='*70}")
    print("PARAMETER COUNT:")
    print(f"  Spatial (NPR):      {per_stream['spatial']:>10,}")
    print(f"  Frequency (Blend):  {per_stream['frequency']:>10,}")
    print(f"  Semantic (FAT):     {per_stream['semantic']:>10,}")
    print(f"  Fusion (MLAF):      {per_stream['fusion']:>10,}")
    print(f"  {'─'*30}")
    print(f"  TOTAL:              {per_stream['total']:>10,}")
    print(f"{'='*70}")
    
    # Test prediction
    preds = model.predict(x)
    probs = model.get_probabilities(x)
    
    print(f"\nPredictions: {preds}")
    print(f"Probabilities: {probs}")
    
    assert preds.shape == (4,), f"Pred shape mismatch: {preds.shape}"
    assert probs.shape == (4,), f"Prob shape mismatch: {probs.shape}"
    assert all(p in [0, 1] for p in preds.tolist()), "Invalid predictions"
    assert all(0 <= p <= 1 for p in probs.tolist()), "Invalid probabilities"
    
    print("\nFULL MODEL TEST PASSED!")
    print("All dimensions verified. Model is ready for training.")
