"""
Multi-Level Adaptive Fusion (MLAF) Module
Cross-stream attention fusion of spatial (128), frequency (64), and semantic (384) streams.
Each stream is treated as a separate token — proper multi-head cross-attention.
Output: Binary classification logit (Real/Fake)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StreamAttention(nn.Module):
    """
    Cross-stream attention: each stream feature is projected to a common
    hidden_dim and treated as one token in a 3-token sequence.
    Streams attend to each other, learning which stream matters most
    for a given input.

    Bug fixed: original code used seq_len=1 (self-attention on a single token),
    which is mathematically a no-op. This version uses 3 tokens — one per stream.
    """

    def __init__(
        self,
        spatial_dim: int = 128,
        freq_dim: int = 64,
        semantic_dim: int = 384,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super(StreamAttention, self).__init__()

        assert hidden_dim % n_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"

        # Project each stream to common hidden_dim before attention
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Multi-head attention over the 3 stream tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable stream-type embeddings (like positional encoding but for streams)
        # Helps the model distinguish spatial vs frequency vs semantic tokens
        self.stream_embeddings = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.trunc_normal_(self.stream_embeddings, std=0.02)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        semantic_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_feat:  [B, spatial_dim]
            freq_feat:     [B, freq_dim]
            semantic_feat: [B, semantic_dim]

        Returns:
            fused:        [B, hidden_dim]  — mean-pooled attended tokens
            attn_weights: [B, 3, 3]        — per-sample stream attention weights
        """
        # Project each stream → [B, 1, hidden_dim]
        s = self.spatial_proj(spatial_feat).unsqueeze(1)
        f = self.freq_proj(freq_feat).unsqueeze(1)
        t = self.semantic_proj(semantic_feat).unsqueeze(1)

        # Stack → 3-token sequence: [B, 3, hidden_dim]
        tokens = torch.cat([s, f, t], dim=1)

        # Add stream-type embeddings (helps attention distinguish stream types)
        tokens = tokens + self.stream_embeddings.unsqueeze(0)

        # Cross-stream attention
        attended, attn_weights = self.attn(tokens, tokens, tokens)
        attended = self.dropout(attended)

        # Residual + LayerNorm
        tokens = self.norm(tokens + attended)   # [B, 3, hidden_dim]

        # Weighted pool: use attention weights to weight stream contributions
        # attn_weights: [B, 3, 3] — mean over query dim → per-stream importance [B, 3]
        stream_importance = attn_weights.mean(dim=1)              # [B, 3]
        stream_importance = torch.softmax(stream_importance, dim=-1).unsqueeze(-1)  # [B, 3, 1]
        fused = (tokens * stream_importance).sum(dim=1)           # [B, hidden_dim]

        return fused, attn_weights


class MLAFFusion(nn.Module):
    """
    Multi-Level Adaptive Fusion module.

    Architecture:
        1. Project each stream to hidden_dim
        2. Cross-stream multi-head attention (3 stream tokens)
        3. Mean-pool → [B, hidden_dim]
        4. Classification head → [B, 1]

    Input:  [B, 128], [B, 64], [B, 384]
    Output: logit [B, 1], fused features [B, hidden_dim]
    """

    def __init__(
        self,
        spatial_dim: int = 128,
        freq_dim: int = 64,
        semantic_dim: int = 384,
        hidden_dim: int = 256,
        attention_heads: int = 4,
        dropout_rate: float = 0.4
    ):
        super(MLAFFusion, self).__init__()

        # Cross-stream attention
        self.cross_stream_attn = StreamAttention(
            spatial_dim=spatial_dim,
            freq_dim=freq_dim,
            semantic_dim=semantic_dim,
            hidden_dim=hidden_dim,
            n_heads=attention_heads,
            dropout=0.1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        semantic_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_feat:  [B, 128]
            freq_feat:     [B, 64]
            semantic_feat: [B, 384]

        Returns:
            logits:         [B, 1]
            fused_features: [B, hidden_dim]
        """
        # Cross-stream attention fusion
        fused_features, attn_weights = self.cross_stream_attn(
            spatial_feat, freq_feat, semantic_feat
        )  # [B, hidden_dim]

        # Classification
        logits = self.classifier(fused_features)  # [B, 1]

        return logits, fused_features


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING: MLAF Fusion Module (Cross-Stream Attention)")
    print("=" * 60)

    model = MLAFFusion(
        spatial_dim=128,
        freq_dim=64,
        semantic_dim=384,
        hidden_dim=256,
        attention_heads=4
    )
    model.eval()

    spatial  = torch.randn(4, 128)
    freq     = torch.randn(4, 64)
    semantic = torch.randn(4, 384)

    with torch.no_grad():
        logits, fused = model(spatial, freq, semantic)

    print(f"Spatial input:  {spatial.shape}")
    print(f"Freq input:     {freq.shape}")
    print(f"Semantic input: {semantic.shape}")
    print(f"Logits output:  {logits.shape}")
    print(f"Fused features: {fused.shape}")

    assert logits.shape == (4, 1),   f"Logits shape mismatch! {logits.shape}"
    assert fused.shape  == (4, 256), f"Fused shape mismatch!  {fused.shape}"

    print("\nMLAF Fusion test PASSED!")
