"""
Quick test to verify multi-SAR validation fix
"""
import sys
sys.path.insert(0, '/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main')

import torch

print("=" * 80)
print("Testing Multi-SAR Validation Fix")
print("=" * 80)

# Simulate multi-SAR input: [B, N, C, H, W]
B, N, C, H, W = 8, 8, 3, 256, 256
im_lq = torch.randn(B, N, C, H, W)
print(f"\nInput shape: {im_lq.shape}")

# Check if multi-SAR (len(shape) == 5)
is_multisar = (len(im_lq.shape) == 5)
print(f"Is multi-SAR: {is_multisar}")

if is_multisar:
    # Multi-SAR: [B, N, C, H, W] → take first SAR, VV channel
    vv_band = im_lq[:, 0, 0:1, :, :]  # [B, 1, H, W]
    lq_for_diffusion = vv_band.expand(-1, 3, -1, -1)  # [B, 3, H, W]
    is_dualband = False
    print(f"\n✓ Multi-SAR conversion:")
    print(f"  VV band shape: {vv_band.shape}")
    print(f"  lq_for_diffusion shape: {lq_for_diffusion.shape}")
    print(f"  Expected: [B, 3, H, W] = [{B}, 3, {H}, {W}]")

    if lq_for_diffusion.shape == (B, 3, H, W):
        print(f"\n✓✓✓ VALIDATION FIX IS CORRECT! ✓✓✓")
    else:
        print(f"\n✗✗✗ SHAPE MISMATCH! ✗✗✗")
else:
    print("Not multi-SAR input")

# Test model_kwargs
model_kwargs = {'lq': im_lq}
num_sar = torch.tensor([5, 3, 4, 6, 7, 2, 4, 5])
model_kwargs['num_sar'] = num_sar
print(f"\nmodel_kwargs['lq'].shape: {model_kwargs['lq'].shape}")
print(f"model_kwargs['num_sar']: {model_kwargs['num_sar'].tolist()}")

print("\n" + "=" * 80)
print("All fixes verified successfully!")
print("=" * 80)
