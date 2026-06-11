"""
Script to extract and view validation progress images.
The progress images are grids created by torchvision.utils.make_grid.
Layout: rows = samples in batch, cols = diffusion timesteps (left=noisy, right=final)
The LAST COLUMN is the final denoised output from the EMA model.
"""
import cv2
import numpy as np
import os
import sys

base = 'experiments/sar_dualband_vv_vh/2026-02-07-16-16-38/images/val'
out_dir = '/mnt/fwgpu/hdd12tb/code/thucnd/check_results_val'
os.makedirs(out_dir, exist_ok=True)

pad = 2
cell_size = 256

for idx in [586, 601, 725]:
    # Read progress grid
    img = cv2.imread(f'{base}/progress-{idx}.png')
    if img is None:
        print(f'Cannot read progress-{idx}.png')
        continue
    
    h, w = img.shape[:2]
    ncols = (w - pad) // (cell_size + pad)
    nrows_grid = (h - pad) // (cell_size + pad)
    print(f'progress-{idx}: {img.shape}, grid={nrows_grid} rows x {ncols} cols')
    print(f'  Cols represent diffusion timesteps: col0=noisy ... col{ncols-1}=final output')
    print(f'  Rows represent different val samples in the batch')
    
    # Extract each cell
    for row in range(nrows_grid):
        for col in range(ncols):
            y = pad + row * (cell_size + pad)
            x = pad + col * (cell_size + pad)
            cell = img[y:y+cell_size, x:x+cell_size]
            
            # Only save last column (final output) and first column (initial)
            if col == ncols - 1:
                out_path = f'{out_dir}/val_final_{idx}_sample{row}.png'
                cv2.imwrite(out_path, cell)
                print(f'  Final output sample {row}: {out_path}')
            elif col == 0:
                out_path = f'{out_dir}/val_initial_{idx}_sample{row}.png'
                cv2.imwrite(out_path, cell)
    
    # Extract GT
    gt = cv2.imread(f'{base}/gt-{idx}.png')
    if gt is not None:
        gt_ncols = (gt.shape[1] - pad) // (cell_size + pad)
        gt_nrows = (gt.shape[0] - pad) // (cell_size + pad)
        print(f'  gt-{idx}: {gt.shape}, grid={gt_nrows}x{gt_ncols}')
        for row in range(gt_nrows):
            for col in range(gt_ncols):
                y = pad + row * (cell_size + pad)
                x = pad + col * (cell_size + pad)
                cell = gt[y:y+cell_size, x:x+cell_size]
                out_path = f'{out_dir}/val_gt_{idx}_sample{row}_col{col}.png'
                cv2.imwrite(out_path, cell)
    
    # Extract LQ VV
    lq_vv = cv2.imread(f'{base}/lq_VV-{idx}.png')
    if lq_vv is not None:
        lq_ncols = (lq_vv.shape[1] - pad) // (cell_size + pad)
        lq_nrows = (lq_vv.shape[0] - pad) // (cell_size + pad)
        print(f'  lq_VV-{idx}: {lq_vv.shape}, grid={lq_nrows}x{lq_ncols}')
        for row in range(lq_nrows):
            for col in range(lq_ncols):
                y = pad + row * (cell_size + pad)
                x = pad + col * (cell_size + pad)
                cell = lq_vv[y:y+cell_size, x:x+cell_size]
                out_path = f'{out_dir}/val_lq_VV_{idx}_sample{row}_col{col}.png'
                cv2.imwrite(out_path, cell)
    
    # Extract LQ VH
    lq_vh = cv2.imread(f'{base}/lq_VH-{idx}.png')
    if lq_vh is not None:
        lq_ncols = (lq_vh.shape[1] - pad) // (cell_size + pad)
        lq_nrows = (lq_vh.shape[0] - pad) // (cell_size + pad)
        for row in range(lq_nrows):
            for col in range(lq_ncols):
                y = pad + row * (cell_size + pad)
                x = pad + col * (cell_size + pad)
                cell = lq_vh[y:y+cell_size, x:x+cell_size]
                out_path = f'{out_dir}/val_lq_VH_{idx}_sample{row}_col{col}.png'
                cv2.imwrite(out_path, cell)

    # Create a side-by-side comparison: LQ_VV | Final_Output | GT
    print(f'  Creating comparison strips...')
    for row in range(nrows_grid):
        parts = []
        
        # LQ VV
        lq_path = f'{out_dir}/val_lq_VV_{idx}_sample{row}_col{row}.png'
        if os.path.exists(lq_path):
            parts.append(cv2.imread(lq_path))
        
        # Final output
        final_path = f'{out_dir}/val_final_{idx}_sample{row}.png'
        if os.path.exists(final_path):
            parts.append(cv2.imread(final_path))
        
        # GT
        gt_path = f'{out_dir}/val_gt_{idx}_sample{row}_col{row}.png'
        if os.path.exists(gt_path):
            parts.append(cv2.imread(gt_path))
        
        if len(parts) >= 2:
            strip = np.hstack(parts)
            strip_path = f'{out_dir}/comparison_{idx}_sample{row}.png'
            cv2.imwrite(strip_path, strip)
            print(f'  Comparison strip: {strip_path} ({len(parts)} panels)')

print('\nDone! All extracted images in:', out_dir)
print('Key files:')
print('  val_final_*.png = Model output (last diffusion step, EMA model)')
print('  val_gt_*.png = Ground truth')
print('  val_lq_VV_*.png = Input VV band')
print('  comparison_*.png = Side-by-side: LQ | Output | GT')
