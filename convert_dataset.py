"""
Convert Dataset_Setup to format compatible with MultiSARDataset.

Changes:
- Convert .tif to .png
- Rename _patch_0.tif -> _p0.png
- Keep folder structure same

Usage:
    python convert_dataset.py --input /path/to/Dataset_Setup --output /path/to/Dataset_Converted
"""

import os
import re
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed, using PIL only")

def convert_filename(old_name, output_format='npy'):
    """Convert old filename pattern to new pattern.

    Old: C-48-10-A-a_2022-02-04_HLS_2022-02-04_OP_2022-01-21_VVVH_aligned_patch_0.tif
    New: C-48-10-A-a_2022-02-04_HLS_2022-02-04_OP_2022-01-21_VVVH_aligned_p0.npy (or .png)

    Old: C-48-10-A-a_2022-02-04_HLS_2022-02-04_RGB_aligned_patch_0.tif
    New: C-48-10-A-a_2022-02-04_HLS_2022-02-04_RGB_aligned_p0.npy (or .png)
    """
    # Replace _patch_X.tif with _pX.npy or _pX.png
    new_name = re.sub(r'_patch_(\d+)\.tif$', rf'_p\1.{output_format}', old_name)
    return new_name

def read_tif_image(input_path):
    """Read TIF/GeoTIFF file and return numpy array."""
    try:
        # Try PIL first
        img = Image.open(input_path)
        img_array = np.array(img)
        return img_array
    except Exception as e:
        # Fallback to rasterio for GeoTIFF
        if HAS_RASTERIO:
            try:
                with rasterio.open(input_path) as src:
                    img_array = src.read()
                    # rasterio returns (C, H, W), convert to (H, W, C)
                    if len(img_array.shape) == 3:
                        img_array = np.transpose(img_array, (1, 2, 0))
                    return img_array
            except Exception as e2:
                print(f"Error reading {input_path}: {e2}")
                return None
        else:
            print(f"Error reading {input_path}: {e}")
            return None

def normalize_to_uint8(img_array):
    """Normalize array to uint8 [0, 255]."""
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        img_min = img_array.min()
        img_max = img_array.max()
        if img_max > img_min:
            img_array = (img_array - img_min) / (img_max - img_min) * 255
        img_array = img_array.astype(np.uint8)
    elif img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    return img_array

def ensure_3_channels(img_array):
    """Ensure array has 3 channels (H, W, 3)."""
    if len(img_array.shape) == 2:
        # Grayscale -> RGB
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[2] == 1:
        # Single channel -> RGB
        img_array = np.stack([img_array[:,:,0]]*3, axis=-1)
    elif img_array.shape[2] == 4:
        # RGBA -> RGB
        img_array = img_array[:, :, :3]
    elif img_array.shape[2] > 3:
        # Use first 3 channels
        img_array = img_array[:, :, :3]
    return img_array

def convert_tif_to_npy(input_path, output_path):
    """Convert TIF to numpy .npy format (preserves original data)."""
    img_array = read_tif_image(input_path)
    if img_array is None:
        return False

    try:
        # Ensure 3 channels
        img_array = ensure_3_channels(img_array)
        # Save as numpy
        np.save(output_path, img_array)
        return True
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        return False

def convert_tif_to_png(input_path, output_path):
    """Convert TIF to PNG, handling multi-channel images and GeoTIFF."""
    img_array = read_tif_image(input_path)
    if img_array is None:
        return False

    try:
        # Normalize to 0-255
        img_array = normalize_to_uint8(img_array)
        # Ensure 3 channels
        img_array = ensure_3_channels(img_array)

        # Save as PNG
        img_out = Image.fromarray(img_array)
        if img_out.mode != 'RGB':
            img_out = img_out.convert('RGB')
        img_out.save(output_path, 'PNG')
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False

def convert_dataset(input_dir, output_dir, output_format='npy'):
    """Convert entire dataset.

    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
        output_format: 'npy' or 'png'
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    convert_func = convert_tif_to_npy if output_format == 'npy' else convert_tif_to_png

    # Process train, val, test
    for split in ['train', 'val', 'test']:
        split_input = input_dir / split
        if not split_input.exists():
            print(f"Skipping {split} - not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split}... (output format: {output_format})")
        print(f"{'='*60}")

        for folder_type in ['A', 'B']:  # A=SAR, B=HLS
            type_input = split_input / folder_type
            type_output = output_dir / split / folder_type

            if not type_input.exists():
                print(f"  Skipping {folder_type} - not found")
                continue

            # Get all subdirs
            subdirs = [d for d in type_input.iterdir() if d.is_dir()]
            print(f"  {folder_type}: {len(subdirs)} subdirectories")

            for subdir in tqdm(subdirs, desc=f"  {split}/{folder_type}"):
                # Create output subdir
                out_subdir = type_output / subdir.name
                out_subdir.mkdir(parents=True, exist_ok=True)

                # Convert all TIF files
                for tif_file in subdir.glob("*.tif"):
                    new_name = convert_filename(tif_file.name, output_format)
                    out_path = out_subdir / new_name

                    if not out_path.exists():
                        convert_func(tif_file, out_path)

def main():
    parser = argparse.ArgumentParser(description="Convert dataset format")
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--format", default="npy", choices=["npy", "png"],
                        help="Output format: 'npy' (numpy, preserves original) or 'png' (8-bit)")
    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")

    convert_dataset(args.input, args.output, args.format)

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)

if __name__ == "__main__":
    main()
