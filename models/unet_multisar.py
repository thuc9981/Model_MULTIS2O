"""
UNet Model with Multi-SAR support via Temporal Transformer.

This module extends UNetModelSwin to accept multiple SAR images as input
and uses a Temporal Transformer to fuse them into a single feature map
that conditions the diffusion process.

Architecture:
    Input: x (noisy latent) + lq (N SAR images)
    TemporalTransformer(lq) → fused_features [B, D, H', W']
    UNet(x concat fused_features) → output
"""

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .swin_transformer import BasicLayer
from .temporal_transformer import TemporalTransformer, TemporalTransformerLight
from .unet import TimestepBlock, TimestepEmbedSequential, ResBlock, Downsample, Upsample


class UNetModelSwinMultiSAR(nn.Module):
    """
    UNet model with Temporal Transformer for Multi-SAR to Optical translation.

    This model takes multiple SAR images, fuses them using a Temporal Transformer,
    and uses the fused features as conditioning for the diffusion process.

    :param image_size: Resolution of the latent space (e.g., 64 for 256px input with 4x downsampling)
    :param in_channels: channels in the input latent Tensor
    :param model_channels: base channel count for the model
    :param out_channels: channels in the output Tensor
    :param num_res_blocks: number of residual blocks per downsample
    :param attention_resolutions: resolutions at which attention is applied
    :param dropout: dropout probability
    :param channel_mult: channel multiplier for each level
    :param conv_resample: use learned convolutions for up/downsampling
    :param dims: 1D, 2D, or 3D convolutions
    :param use_fp16: use half precision
    :param num_heads: attention heads
    :param num_head_channels: channels per attention head
    :param use_scale_shift_norm: FiLM-like conditioning
    :param resblock_updown: use residual blocks for up/downsampling
    :param swin_depth: depth of swin transformer blocks
    :param swin_embed_dim: embedding dimension for swin transformer
    :param window_size: window size for swin attention
    :param mlp_ratio: MLP ratio for swin transformer
    :param patch_norm: patch normalization in swin transformer
    :param cond_lq: whether to condition on LQ (SAR) images
    :param lq_size: spatial size of LQ (SAR) images
    :param lq_channels: channels per SAR image (3 for VV, VH, VV/VH)
    :param use_confidence: predict confidence map (C-DiffSET)

    Multi-SAR specific parameters:
    :param max_sar_frames: maximum number of SAR images to process
    :param temporal_embed_dim: embedding dimension for temporal transformer
    :param temporal_num_heads: attention heads in temporal transformer
    :param temporal_num_layers: number of transformer layers
    :param temporal_fusion_type: 'attention', 'weighted', or 'mean'
    :param use_light_temporal: use lightweight temporal transformer
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        swin_depth=2,
        swin_embed_dim=96,
        window_size=8,
        mlp_ratio=2.0,
        patch_norm=False,
        cond_lq=True,
        lq_size=256,
        lq_channels=3,
        use_confidence=False,
        # Multi-SAR specific parameters
        max_sar_frames=8,
        temporal_embed_dim=256,
        temporal_num_heads=8,
        temporal_num_layers=4,
        temporal_fusion_type='attention',
        use_light_temporal=False,
    ):
        super().__init__()
        self.use_confidence = use_confidence
        self.max_sar_frames = max_sar_frames

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.lq_channels = lq_channels

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # === Temporal Transformer for Multi-SAR fusion ===
        # Calculate number of spatial downsamples needed
        num_spatial_downsample = int(math.log(lq_size / image_size) / math.log(2))

        # Output channels from temporal transformer
        temporal_out_channels = temporal_embed_dim

        if use_light_temporal:
            self.temporal_transformer = TemporalTransformerLight(
                in_channels=lq_channels,
                embed_dim=temporal_embed_dim,
                out_channels=temporal_out_channels,
                max_frames=max_sar_frames,
                spatial_downsample=num_spatial_downsample,
                output_spatial_size=image_size,
                input_size=lq_size,
            )
        else:
            self.temporal_transformer = TemporalTransformer(
                in_channels=lq_channels,
                embed_dim=temporal_embed_dim,
                out_channels=temporal_out_channels,
                num_heads=temporal_num_heads,
                num_layers=temporal_num_layers,
                mlp_ratio=4.0,
                dropout=0.1,
                max_frames=max_sar_frames,
                spatial_downsample=num_spatial_downsample,
                fusion_type=temporal_fusion_type,
                output_spatial_size=image_size,
                input_size=lq_size,
            )

        # Feature map from temporal transformer
        base_chn = temporal_out_channels

        # === UNet architecture ===
        ch = input_ch = int(channel_mult[0] * model_channels)
        in_channels_total = in_channels + base_chn  # latent + fused SAR features

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels_total, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size

        for level, mult in enumerate(channel_mult):
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj == 0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                in_chans=ch,
                embed_dim=swin_embed_dim,
                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                window_size=window_size,
                depth=swin_depth,
                img_size=ds,
                patch_size=1,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=dropout,
                attn_drop=0.,
                drop_path=0.,
                use_checkpoint=False,
                norm_layer=normalization,
                patch_norm=patch_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions and i == 0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output convolution
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

        # Confidence head (C-DiffSET): predicts pixel-wise confidence map
        if self.use_confidence:
            confidence_conv = conv_nd(dims, input_ch, 1, 3, padding=1)
            nn.init.zeros_(confidence_conv.weight)
            nn.init.constant_(confidence_conv.bias, math.log(math.e - 1))
            self.confidence_head = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                confidence_conv,
                nn.Softplus(),
            )

    def forward(self, x, timesteps, lq=None, num_sar=None, mask=None):
        """
        Apply the model to an input batch.

        :param x: [B, C, H, W] Tensor of noisy latent inputs
        :param timesteps: [B] 1-D batch of timesteps
        :param lq: [B, N, C, H, W] Tensor of N SAR images per sample
        :param num_sar: [B] or int, actual number of valid SAR images per sample
        :param mask: (optional) mask tensor for inpainting
        :return: [B, C, H, W] Tensor of outputs (and confidence if enabled)
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        if lq is not None:
            assert self.cond_lq
            # Fuse multiple SAR images using Temporal Transformer
            # lq: [B, N, C, H, W] → fused: [B, D, H', W']
            fused_features = self.temporal_transformer(lq.type(self.dtype), num_sar)

            # Concatenate fused features with noisy latent
            x = th.cat([x, fused_features], dim=1)

        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)

        if self.use_confidence:
            confidence = self.confidence_head(h)
            return out, confidence
        return out

    def convert_to_fp16(self):
        """Convert the torso of the model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.temporal_transformer.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the torso of the model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.temporal_transformer.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def get_temporal_attention_weights(self):
        """Get attention weights from the temporal transformer for visualization."""
        return self.temporal_transformer.get_attention_weights()
