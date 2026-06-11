"""
Test script to verify new Multi-SAR visualization
"""
import sys
sys.path.insert(0, '/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main')

import torch
from torchvision.utils import make_grid
import torch.nn.functional as F

print("=" * 80)
print("Testing Multi-SAR Visualization Logic")
print("=" * 80)

# Simulate multi-SAR batch
B, N, C, H, W = 4, 5, 3, 256, 256
lq = torch.randn(B, N, C, H, W)
num_sar = torch.tensor([5, 3, 4, 5])  # First sample has 5 valid SAR, second has 3, etc.

print(f"\nInput shape: {lq.shape}")
print(f"Num SAR per sample: {num_sar.tolist()}")

# Test SAR grid creation for first sample
sar_grid_sample = lq[0]  # [N, C, H, W]
n_valid = num_sar[0].item()
print(f"\nFirst sample: {n_valid} valid SAR images")

# Only show valid (non-padded) SAR images
sar_grid_sample = sar_grid_sample[:n_valid]  # [n_valid, C, H, W]
print(f"SAR grid sample shape: {sar_grid_sample.shape}")

# Create grid
sar_grid = make_grid(sar_grid_sample, nrow=n_valid, normalize=True, scale_each=False, padding=2)
print(f"SAR grid shape: {sar_grid.shape}")
print(f"  Expected: [3, H, W*n_valid + padding]")

# Resize for composite
sar_grid_resized = F.interpolate(
    sar_grid.unsqueeze(0),
    size=(H, W),
    mode='bilinear',
    align_corners=False
)
print(f"\nSAR grid resized: {sar_grid_resized.shape}")

# Simulate GT, prediction, diffused
gt_vis = torch.randn(1, C, H, W)
x0_vis = torch.randn(1, C, H, W)
xt_vis = torch.randn(1, C, H, W)

print(f"GT shape: {gt_vis.shape}")
print(f"x0 pred shape: {x0_vis.shape}")
print(f"Diffused shape: {xt_vis.shape}")

# Create composite
composite = torch.cat([sar_grid_resized, gt_vis, x0_vis, xt_vis], dim=3)
print(f"\nComposite shape: {composite.shape}")
print(f"  Expected: [1, 3, H, 4*W] = [1, 3, {H}, {4*W}]")

if composite.shape == (1, 3, H, 4*W):
    print("\n✓ Composite visualization logic is CORRECT!")
    print("\nComposite layout: [SAR Grid | GT | Prediction | Diffused]")
    print("  - SAR Grid: All valid SAR images combined")
    print("  - GT: Ground truth optical RGB")
    print("  - Prediction: Model's x0 prediction")
    print("  - Diffused: Output from diffusion process")
else:
    print("\n✗ Composite shape mismatch!")

print("\n" + "=" * 80)
print("Test completed successfully!")
print("=" * 80)
