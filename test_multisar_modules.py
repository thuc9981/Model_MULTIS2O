#!/usr/bin/env python
"""
Test script for Multi-SAR to Optical translation modules.

This script tests:
1. MultiSARDataset - Loading multiple SAR images per sample
2. TemporalTransformer - Fusing N SAR images into single feature map
3. UNetModelSwinMultiSAR - Full model with temporal transformer
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_temporal_transformer():
    """Test TemporalTransformer module."""
    print("\n" + "=" * 60)
    print("Testing TemporalTransformer")
    print("=" * 60)

    from models.temporal_transformer import TemporalTransformer, TemporalTransformerLight

    # Test parameters
    batch_size = 2
    num_sar = 5
    channels = 3
    height = 256
    width = 256

    # Create dummy input: [B, N, C, H, W]
    x = torch.randn(batch_size, num_sar, channels, height, width)
    num_sar_tensor = torch.tensor([5, 3])  # Different number of SAR images per sample

    # Test full temporal transformer
    print("\n1. Testing full TemporalTransformer...")
    model = TemporalTransformer(
        in_channels=channels,
        embed_dim=256,
        out_channels=256,
        num_heads=8,
        num_layers=4,
        max_frames=8,
        spatial_downsample=2,
        output_spatial_size=64,
        input_size=256,
    )
    print(f"   Input shape: {x.shape}")

    output = model(x, num_sar_tensor)
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: [2, 256, 64, 64]")
    assert output.shape == (batch_size, 256, 64, 64), f"Shape mismatch: {output.shape}"
    print("   ✓ Full TemporalTransformer test passed!")

    # Test attention weights
    attn_weights = model.get_attention_weights()
    if attn_weights is not None:
        print(f"   Attention weights shape: {attn_weights.shape}")

    # Test lightweight temporal transformer
    print("\n2. Testing TemporalTransformerLight...")
    model_light = TemporalTransformerLight(
        in_channels=channels,
        embed_dim=128,
        out_channels=128,
        max_frames=8,
        spatial_downsample=2,
        output_spatial_size=64,
        input_size=256,
    )

    output_light = model_light(x, num_sar_tensor)
    print(f"   Output shape: {output_light.shape}")
    print(f"   Expected: [2, 128, 64, 64]")
    assert output_light.shape == (batch_size, 128, 64, 64), f"Shape mismatch: {output_light.shape}"
    print("   ✓ TemporalTransformerLight test passed!")

    # Calculate model sizes
    full_params = sum(p.numel() for p in model.parameters())
    light_params = sum(p.numel() for p in model_light.parameters())
    print(f"\n   Full model params: {full_params/1e6:.2f}M")
    print(f"   Light model params: {light_params/1e6:.2f}M")

    return True


def test_unet_multisar():
    """Test UNetModelSwinMultiSAR module."""
    print("\n" + "=" * 60)
    print("Testing UNetModelSwinMultiSAR")
    print("=" * 60)

    from models.unet_multisar import UNetModelSwinMultiSAR

    # Test parameters
    batch_size = 2
    num_sar = 5
    lq_channels = 3
    lq_size = 256
    image_size = 64  # Latent space size

    # Create dummy inputs
    x = torch.randn(batch_size, 3, image_size, image_size)  # Noisy latent
    timesteps = torch.randint(0, 4, (batch_size,))
    lq = torch.randn(batch_size, num_sar, lq_channels, lq_size, lq_size)  # Multi-SAR
    num_sar_tensor = torch.tensor([5, 3])

    # Create model
    print("\nCreating UNetModelSwinMultiSAR...")
    model = UNetModelSwinMultiSAR(
        image_size=image_size,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=[2, 2, 2, 2],
        attention_resolutions=[64, 32, 16, 8],
        channel_mult=[1, 2, 2, 4],
        use_fp16=False,
        num_head_channels=32,
        use_scale_shift_norm=True,
        swin_depth=2,
        swin_embed_dim=128,
        window_size=8,
        cond_lq=True,
        lq_size=lq_size,
        lq_channels=lq_channels,
        use_confidence=True,
        max_sar_frames=8,
        temporal_embed_dim=256,
        temporal_num_heads=8,
        temporal_num_layers=4,
        temporal_fusion_type='attention',
        use_light_temporal=False,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")

    # Test forward pass
    print("\nTesting forward pass...")
    print(f"   x shape: {x.shape}")
    print(f"   lq shape: {lq.shape}")
    print(f"   timesteps: {timesteps}")

    output, confidence = model(x, timesteps, lq=lq, num_sar=num_sar_tensor)

    print(f"   Output shape: {output.shape}")
    print(f"   Expected: [{batch_size}, 3, {image_size}, {image_size}]")
    assert output.shape == (batch_size, 3, image_size, image_size), f"Shape mismatch: {output.shape}"

    print(f"   Confidence shape: {confidence.shape}")
    print(f"   Expected: [{batch_size}, 1, {image_size}, {image_size}]")
    assert confidence.shape == (batch_size, 1, image_size, image_size), f"Shape mismatch: {confidence.shape}"

    print("   ✓ UNetModelSwinMultiSAR test passed!")

    # Test temporal attention weights
    attn_weights = model.get_temporal_attention_weights()
    if attn_weights is not None:
        print(f"   Temporal attention weights shape: {attn_weights.shape}")

    return True


def test_multisar_dataset():
    """Test MultiSARDataset module."""
    print("\n" + "=" * 60)
    print("Testing MultiSARDataset")
    print("=" * 60)

    from basicsr.data.multisar_dataset import MultiSARDataset

    # Check if data directories exist
    data_root_lq = Path("/mnt/hdd12tb/code/thucnd/Data_after_process_patches/train/A")
    data_root_gt = Path("/mnt/hdd12tb/code/thucnd/Data_after_process_patches/train/B")

    if not data_root_lq.exists():
        print(f"   ⚠ Data directory not found: {data_root_lq}")
        print("   Skipping dataset test...")
        return True

    if not data_root_gt.exists():
        print(f"   ⚠ Data directory not found: {data_root_gt}")
        print("   Skipping dataset test...")
        return True

    # Create dataset config
    opt = {
        'dataroot_gt': str(data_root_gt),
        'dataroot_lq': str(data_root_lq),
        'io_backend': {'type': 'disk'},
        'phase': 'train',
        'gt_size': 256,
        'use_hflip': True,
        'use_rot': True,
        'max_sar_images': 8,
        'min_sar_images': 2,
        'scale': 1,
    }

    # Create dataset
    print("\nCreating MultiSARDataset...")
    dataset = MultiSARDataset(opt)
    print(f"   Dataset size: {len(dataset)}")

    # Get a sample
    print("\nLoading sample...")
    sample = dataset[0]

    print(f"   LQ shape: {sample['lq'].shape}")
    print(f"   GT shape: {sample['gt'].shape}")
    print(f"   Num SAR: {sample['num_sar']}")
    print(f"   GT path: {sample['gt_path']}")
    print(f"   Num LQ paths: {len(sample['lq_paths'])}")

    # Verify shapes
    assert len(sample['lq'].shape) == 4, f"LQ should be 4D: [N, C, H, W], got {sample['lq'].shape}"
    assert len(sample['gt'].shape) == 3, f"GT should be 3D: [C, H, W], got {sample['gt'].shape}"

    print("   ✓ MultiSARDataset test passed!")

    return True


def test_dataloader():
    """Test DataLoader with MultiSARDataset."""
    print("\n" + "=" * 60)
    print("Testing DataLoader with MultiSARDataset")
    print("=" * 60)

    from basicsr.data.multisar_dataset import MultiSARDataset
    from torch.utils.data import DataLoader

    # Check if data directories exist
    data_root_lq = Path("/mnt/hdd12tb/code/thucnd/Data_after_process_patches/train/A")
    data_root_gt = Path("/mnt/hdd12tb/code/thucnd/Data_after_process_patches/train/B")

    if not data_root_lq.exists() or not data_root_gt.exists():
        print("   ⚠ Data directories not found, skipping...")
        return True

    # Create dataset
    opt = {
        'dataroot_gt': str(data_root_gt),
        'dataroot_lq': str(data_root_lq),
        'io_backend': {'type': 'disk'},
        'phase': 'train',
        'gt_size': 256,
        'use_hflip': False,
        'use_rot': False,
        'max_sar_images': 8,
        'min_sar_images': 2,
        'scale': 1,
    }

    dataset = MultiSARDataset(opt)

    # Test with DataLoader
    print("\nTesting DataLoader...")

    def collate_fn(batch):
        """Custom collate function for multi-SAR data."""
        lq = torch.stack([b['lq'] for b in batch])
        gt = torch.stack([b['gt'] for b in batch])
        num_sar = torch.tensor([b['num_sar'] for b in batch])
        return {'lq': lq, 'gt': gt, 'num_sar': num_sar}

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

    batch = next(iter(loader))
    print(f"   Batch LQ shape: {batch['lq'].shape}")
    print(f"   Batch GT shape: {batch['gt'].shape}")
    print(f"   Batch num_sar: {batch['num_sar']}")

    assert batch['lq'].shape[0] == 4, "Batch size mismatch"
    assert batch['lq'].shape[1] == 8, "Max SAR images mismatch"

    print("   ✓ DataLoader test passed!")

    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Multi-SAR to Optical Translation Module Tests")
    print("#" * 60)

    tests = [
        ("TemporalTransformer", test_temporal_transformer),
        ("UNetModelSwinMultiSAR", test_unet_multisar),
        ("MultiSARDataset", test_multisar_dataset),
        ("DataLoader", test_dataloader),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n   ✗ {name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"   {name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
