"""
Temporal Transformer for Multi-SAR Fusion.

This module takes N SAR images and produces a single fused feature map using
temporal attention mechanisms. It allows the model to:
1. Learn which SAR images contain the most useful information
2. Handle variable number of input images
3. Aggregate temporal information effectively

Architecture:
    Input: [B, N, C, H, W] - N SAR images per batch
    Spatial Encoder: Extract features from each image → [B, N, D, H', W']
    Temporal Transformer: Cross-attention across temporal dimension
    Output: [B, D, H', W'] - Single fused feature map

Reference: Similar to approaches in video understanding (ViViT, TimeSformer)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal dimension."""

    def __init__(self, d_model, max_len=16):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x, num_frames):
        """
        Args:
            x: [B, N, D] or [B, N, D, H, W]
            num_frames: actual number of frames (for masking)
        """
        return x + self.pos_embed[:, :x.size(1)]


class SpatialEncoder(nn.Module):
    """CNN-based spatial encoder for extracting features from each SAR image."""

    def __init__(self, in_channels=3, embed_dim=256, num_downsample=2):
        """
        Args:
            in_channels: Input channels (3 for RGB SAR: VV, VH, VV/VH)
            embed_dim: Output embedding dimension
            num_downsample: Number of downsampling operations (each halves spatial dim)
        """
        super().__init__()

        layers = []
        ch = in_channels
        out_ch = 64

        for i in range(num_downsample + 1):
            if i < num_downsample:
                # Downsample block
                layers.extend([
                    nn.Conv2d(ch, out_ch, 3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                ])
                ch = out_ch
                out_ch = min(out_ch * 2, embed_dim)
            else:
                # Final projection
                layers.extend([
                    nn.Conv2d(ch, embed_dim, 3, padding=1),
                    nn.GroupNorm(8, embed_dim),
                    nn.SiLU(),
                ])

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B*N, C, H, W]
        Returns:
            features: [B*N, D, H', W']
        """
        return self.encoder(x)


class TemporalAttentionBlock(nn.Module):
    """Multi-head self-attention for temporal dimension."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, D] where N is temporal dimension
            mask: [B, N] boolean mask (True = valid, False = padding)
        Returns:
            out: [B, N, D]
            attn_weights: [B, num_heads, N, N]
        """
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]

        # Apply mask if provided
        if mask is not None:
            # mask: [B, N] → [B, 1, 1, N]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, attn_weights


class TemporalTransformerBlock(nn.Module):
    """Transformer block for temporal attention."""

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = TemporalAttentionBlock(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, D]
            mask: [B, N] boolean mask
        Returns:
            out: [B, N, D]
            attn_weights: [B, num_heads, N, N]
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x), mask)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class TemporalFusion(nn.Module):
    """Learnable fusion module to aggregate temporal features into single output."""

    def __init__(self, embed_dim, max_frames=8, fusion_type='attention'):
        """
        Args:
            embed_dim: Feature dimension
            max_frames: Maximum number of temporal frames
            fusion_type: 'attention' (learnable weights) or 'mean' (simple average)
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim

        if fusion_type == 'attention':
            # Learnable query for aggregation
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
        elif fusion_type == 'weighted':
            # Learnable weights per position
            self.weights = nn.Parameter(torch.ones(1, max_frames, 1) / max_frames)
        # 'mean' fusion doesn't need extra parameters

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, D] temporal features
            mask: [B, N] boolean mask (True = valid)
        Returns:
            out: [B, D] fused feature
        """
        B, N, D = x.shape

        if self.fusion_type == 'attention':
            # Use learnable query to attend to temporal features
            query = self.query.expand(B, -1, -1)  # [B, 1, D]

            # Create attention mask for padding
            if mask is not None:
                key_padding_mask = ~mask  # MultiheadAttention expects True for padding
            else:
                key_padding_mask = None

            out, _ = self.cross_attn(query, x, x, key_padding_mask=key_padding_mask)
            out = self.norm(out + query)
            return out.squeeze(1)  # [B, D]

        elif self.fusion_type == 'weighted':
            weights = self.weights[:, :N, :]  # [1, N, 1]
            if mask is not None:
                weights = weights * mask.unsqueeze(-1).float()
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            out = (x * weights).sum(dim=1)  # [B, D]
            return out

        else:  # mean
            if mask is not None:
                mask = mask.unsqueeze(-1).float()  # [B, N, 1]
                out = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                out = x.mean(dim=1)
            return out


class TemporalTransformer(nn.Module):
    """
    Complete Temporal Transformer for Multi-SAR fusion.

    Takes N SAR images and produces a single fused feature map that can be used
    as condition for the diffusion model.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dim=256,
        out_channels=None,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4.0,
        dropout=0.1,
        max_frames=8,
        spatial_downsample=2,
        fusion_type='attention',
        output_spatial_size=64,
        input_size=256,
    ):
        """
        Args:
            in_channels: Input channels per SAR image (3 for VV, VH, VV/VH)
            embed_dim: Transformer embedding dimension
            out_channels: Output channels (default: embed_dim)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            max_frames: Maximum number of SAR images
            spatial_downsample: Number of spatial downsampling in encoder (2^n)
            fusion_type: 'attention', 'weighted', or 'mean'
            output_spatial_size: Target spatial size for output feature map
            input_size: Input image size
        """
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_channels = out_channels or embed_dim
        self.max_frames = max_frames
        self.spatial_downsample = spatial_downsample
        self.output_spatial_size = output_spatial_size
        self.input_size = input_size

        # Calculate intermediate spatial size after encoding
        self.encoded_spatial_size = input_size // (2 ** spatial_downsample)

        # Spatial encoder for each SAR image
        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_downsample=spatial_downsample
        )

        # Positional encoding for temporal dimension
        self.temporal_pos_enc = PositionalEncoding(embed_dim, max_len=max_frames)

        # Temporal transformer blocks
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Fusion module
        self.fusion = TemporalFusion(embed_dim, max_frames, fusion_type)

        # Output projection (if needed)
        if self.out_channels != embed_dim:
            self.out_proj = nn.Conv2d(embed_dim, self.out_channels, 1)
        else:
            self.out_proj = nn.Identity()

        # Spatial adjustment to match output_spatial_size
        if self.encoded_spatial_size != output_spatial_size:
            self.spatial_adjust = nn.Upsample(
                size=(output_spatial_size, output_spatial_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            self.spatial_adjust = nn.Identity()

        # Store attention weights for visualization
        self.register_buffer('last_attn_weights', None)

    def forward(self, x, num_sar=None):
        """
        Args:
            x: [B, N, C, H, W] - N SAR images per batch
            num_sar: [B] or int - actual number of valid SAR images per sample
        Returns:
            out: [B, out_channels, H', W'] - fused feature map
        """
        B, N, C, H, W = x.shape

        # Create mask for valid frames
        if num_sar is not None:
            if isinstance(num_sar, int):
                num_sar = torch.tensor([num_sar] * B, device=x.device)
            # mask: [B, N], True for valid frames
            indices = torch.arange(N, device=x.device).unsqueeze(0)  # [1, N]
            mask = indices < num_sar.unsqueeze(1)  # [B, N]
        else:
            mask = None

        # === Spatial Encoding ===
        # Reshape for batch processing: [B, N, C, H, W] → [B*N, C, H, W]
        x = rearrange(x, 'b n c h w -> (b n) c h w')

        # Extract spatial features: [B*N, C, H, W] → [B*N, D, H', W']
        spatial_features = self.spatial_encoder(x)
        _, D, Hs, Ws = spatial_features.shape

        # Reshape back: [B*N, D, H', W'] → [B, N, D, H', W']
        spatial_features = rearrange(spatial_features, '(b n) d h w -> b n d h w', b=B, n=N)

        # === Temporal Attention (per spatial location) ===
        # Rearrange for temporal attention: [B, N, D, H', W'] → [B*H'*W', N, D]
        temporal_input = rearrange(spatial_features, 'b n d h w -> (b h w) n d')

        # Expand mask for spatial locations
        if mask is not None:
            mask_expanded = repeat(mask, 'b n -> (b hw) n', hw=Hs * Ws)
        else:
            mask_expanded = None

        # Add temporal positional encoding
        temporal_input = self.temporal_pos_enc(temporal_input, N)

        # Apply temporal transformer blocks
        attn_weights_list = []
        for block in self.temporal_blocks:
            temporal_input, attn_w = block(temporal_input, mask_expanded)
            attn_weights_list.append(attn_w)

        # Store attention weights (sample first spatial location for visualization)
        if len(attn_weights_list) > 0:
            # Take attention from last layer, first head, first spatial location per batch
            self.last_attn_weights = attn_weights_list[-1][:B, 0]  # [B, N, N]

        # === Temporal Fusion ===
        # Fuse temporal dimension: [B*H'*W', N, D] → [B*H'*W', D]
        fused = self.fusion(temporal_input, mask_expanded)

        # Reshape back to spatial: [B*H'*W', D] → [B, D, H', W']
        fused = rearrange(fused, '(b h w) d -> b d h w', b=B, h=Hs, w=Ws)

        # === Output Projection ===
        out = self.out_proj(fused)

        # Adjust spatial size to match target
        out = self.spatial_adjust(out)

        return out

    def get_attention_weights(self):
        """Return last attention weights for visualization."""
        return self.last_attn_weights


class TemporalTransformerLight(nn.Module):
    """
    Lightweight version of Temporal Transformer.

    Uses simpler pooling-based fusion instead of full transformer,
    suitable for faster training and smaller models.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dim=128,
        out_channels=None,
        max_frames=8,
        spatial_downsample=2,
        output_spatial_size=64,
        input_size=256,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels or embed_dim
        self.max_frames = max_frames

        # Per-frame encoder
        self.encoder = SpatialEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_downsample=spatial_downsample
        )

        # Temporal attention weights (learnable per-frame importance)
        self.temporal_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),  # Score per frame
        )

        # Output projection
        if self.out_channels != embed_dim:
            self.out_proj = nn.Conv2d(embed_dim, self.out_channels, 1)
        else:
            self.out_proj = nn.Identity()

        # Spatial adjustment
        encoded_size = input_size // (2 ** spatial_downsample)
        if encoded_size != output_spatial_size:
            self.spatial_adjust = nn.Upsample(
                size=(output_spatial_size, output_spatial_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            self.spatial_adjust = nn.Identity()

    def forward(self, x, num_sar=None):
        """
        Args:
            x: [B, N, C, H, W]
            num_sar: [B] or int
        Returns:
            out: [B, out_channels, H', W']
        """
        B, N, C, H, W = x.shape

        # Create mask
        if num_sar is not None:
            if isinstance(num_sar, int):
                num_sar = torch.tensor([num_sar] * B, device=x.device)
            indices = torch.arange(N, device=x.device).unsqueeze(0)
            mask = indices < num_sar.unsqueeze(1)  # [B, N]
        else:
            mask = torch.ones(B, N, device=x.device, dtype=torch.bool)

        # Encode each frame
        x = rearrange(x, 'b n c h w -> (b n) c h w')
        features = self.encoder(x)  # [B*N, D, H', W']
        _, D, Hs, Ws = features.shape

        # Compute attention scores
        scores = self.temporal_attn(features)  # [B*N, 1]
        scores = rearrange(scores, '(b n) 1 -> b n', b=B, n=N)

        # Apply mask (set padding to -inf before softmax)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1)  # [B, N]

        # Reshape features
        features = rearrange(features, '(b n) d h w -> b n d h w', b=B, n=N)

        # Weighted sum
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1, 1]
        fused = (features * weights).sum(dim=1)  # [B, D, H', W']

        # Output
        out = self.out_proj(fused)
        out = self.spatial_adjust(out)

        return out
