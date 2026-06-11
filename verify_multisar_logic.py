"""
Script to verify Multi-SAR dataset logic matches user requirements:
- Multiple SAR images from different dates (same patch) → 1 optical image
"""
import sys
sys.path.insert(0, '/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main')

from basicsr.data.multisar_dataset import MultiSARDataset
import torch

# Test with real data
config = {
    'dataroot_gt': '/mnt/hdd12tb/code/thucnd/Data_after_process_patches/train/B',
    'dataroot_lq': '/mnt/hdd12tb/code/thucnd/Data_after_process_patches/train/A',
    'io_backend': {'type': 'disk'},
    'phase': 'train',
    'gt_size': 256,
    'use_hflip': False,
    'use_rot': False,
    'scale': 1,
    'max_sar_images': 8,
    'min_sar_images': 2,
}

print("=" * 80)
print("VERIFYING MULTI-SAR DATASET LOGIC")
print("=" * 80)

dataset = MultiSARDataset(config)

# Find the specific sample from user's example
target_subdir = 'C-48-8-D-c-2_2022-01-15'
target_gt = 'C-48-8-D-c-2_2022-01-15_HLS_2021-12-31_RGB_aligned_p0.png'

sample_found = None
for idx, sample in enumerate(dataset.samples):
    if target_subdir in sample['gt_path'] and sample['patch_idx'] == 0:
        sample_found = (idx, sample)
        break

if sample_found:
    idx, sample = sample_found
    print(f"\n✓ Found sample at index {idx}")
    print(f"\nSubdirectory: {sample['subdir']}")
    print(f"Patch Index: p{sample['patch_idx']}")
    print(f"\n{'='*80}")
    print(f"GT (Optical RGB):")
    print(f"{'='*80}")
    print(f"  {sample['gt_path']}")

    print(f"\n{'='*80}")
    print(f"Input SAR images (N={len(sample['sar_paths'])}):")
    print(f"{'='*80}")

    # Parse dates from SAR filenames
    import re
    for i, sar_path in enumerate(sample['sar_paths']):
        fname = sar_path.split('/')[-1]
        # Extract OP date: pattern OP_YYYY-MM-DD
        match = re.search(r'OP_(\d{4}-\d{2}-\d{2})', fname)
        op_date = match.group(1) if match else 'unknown'
        print(f"  [{i+1}] OP_date: {op_date} - {fname}")

    # Verify expected dates
    expected_dates = ['2021-12-16', '2021-12-17', '2021-12-22', '2021-12-28', '2021-12-29']
    actual_dates = []
    for sar_path in sample['sar_paths']:
        match = re.search(r'OP_(\d{4}-\d{2}-\d{2})', sar_path)
        if match:
            actual_dates.append(match.group(1))

    print(f"\n{'='*80}")
    print(f"VERIFICATION RESULT:")
    print(f"{'='*80}")

    # Check if dates match
    dates_match = set(actual_dates) == set(expected_dates)

    print(f"  Expected SAR dates: {expected_dates}")
    print(f"  Actual SAR dates:   {actual_dates}")
    print(f"  Dates match: {'✓ YES' if dates_match else '✗ NO'}")
    print(f"  Number of SAR images: {len(sample['sar_paths'])}")
    print(f"  SAR images sorted by date: {'✓ YES' if actual_dates == sorted(actual_dates) else '✗ NO'}")

    # Test loading the sample
    print(f"\n{'='*80}")
    print(f"TESTING DATA LOADING:")
    print(f"{'='*80}")

    data_sample = dataset[idx]
    print(f"  LQ shape: {data_sample['lq'].shape} - Expected: [8, 3, H, W]")
    print(f"  GT shape: {data_sample['gt'].shape} - Expected: [3, H, W]")
    print(f"  Num SAR: {data_sample['num_sar']} - Actual valid SAR images")
    print(f"  LQ paths length: {len(data_sample['lq_paths'])}")

    # Check for other patches
    print(f"\n{'='*80}")
    print(f"CHECKING OTHER PATCHES (p1-p8):")
    print(f"{'='*80}")

    patches_found = []
    for sample in dataset.samples:
        if target_subdir in sample['gt_path']:
            patches_found.append(sample['patch_idx'])

    patches_found = sorted(set(patches_found))
    print(f"  Found patches: {patches_found}")
    print(f"  Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8] (9 patches)")
    print(f"  All patches present: {'✓ YES' if len(patches_found) >= 9 else '✗ NO (only ' + str(len(patches_found)) + ' patches)'}")

    # Sample 3 more patches to verify
    print(f"\n  Verifying samples for patches p1, p4, p8:")
    for patch_id in [1, 4, 8]:
        for sample in dataset.samples:
            if target_subdir in sample['gt_path'] and sample['patch_idx'] == patch_id:
                print(f"    p{patch_id}: {len(sample['sar_paths'])} SAR images → 1 optical RGB")
                break

    print(f"\n{'='*80}")
    print(f"FINAL CONCLUSION:")
    print(f"{'='*80}")
    if dates_match and len(sample['sar_paths']) == 5:
        print("  ✓✓✓ LOGIC IS CORRECT! ✓✓✓")
        print("  Multiple SAR images (different dates, same patch) → 1 optical image")
    else:
        print("  ✗✗✗ LOGIC HAS ISSUES! ✗✗✗")
else:
    print(f"\n✗ Sample not found for {target_subdir} patch p0")
    print("\nAvailable samples (first 10):")
    for i in range(min(10, len(dataset.samples))):
        sample = dataset.samples[i]
        print(f"  [{i}] {sample['subdir']} p{sample['patch_idx']} - {len(sample['sar_paths'])} SAR images")

print("\n" + "=" * 80)
