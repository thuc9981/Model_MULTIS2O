#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Confidence-Guided Local Refinement Inference for SAR Despeckling.

Pipeline:
  1. Global Pass: Run full diffusion → get prediction + confidence map
  2. Mask Regions: Identify low-confidence patches (threshold on confidence map)
  3. Local SDEdit: Add noise to weak patches at intermediate t, re-denoise
  4. Fusion: Blend refined patches back with smooth weighting

Usage:
  python inference_local_refine.py \
    --cfg_path configs/sar_dualband_vv_vh.yaml \
    --ckpt_path experiments/sar_paris_finetune/ckpts/model_10000.pth \
    -i /path/to/input/folder \
    -o /path/to/output/folder \
    --refine_threshold 0.5 \
    --refine_topk 0.15 \
    --refine_patch_size 96 \
    --refine_context_pad 16 \
    --refine_t_start 2 \
    --save_confidence
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
import math
import glob
import cv2
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext

from omegaconf import OmegaConf
from sampler import ResShiftSampler
from utils import util_image
from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

import torch as th


# ──────────────────────────────────────────────────────────────
#  Argument Parser
# ──────────────────────────────────────────────────────────────
def get_parser():
    parser = argparse.ArgumentParser(description="Confidence-Guided Local Refinement")

    # I/O
    parser.add_argument("-i", "--in_path", type=str, required=True, help="Input path (folder or single image).")
    parser.add_argument("-o", "--out_path", type=str, default="./results_refine", help="Output folder.")
    parser.add_argument("--input_path", type=str, default=None, help="Alias for in_path")
    parser.add_argument("--output_path", type=str, default=None, help="Alias for out_path")

    # Model
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to config .yaml")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Chopping batch size.")
    parser.add_argument("--chop_size", type=int, default=256, help="Chopping size.")
    parser.add_argument("--chop_stride", type=int, default=-1, help="Chopping stride.")

    # Refinement params
    parser.add_argument("--refine_threshold", type=float, default=0.0,
                        help="Absolute confidence threshold. Regions below this are candidates for refinement. "
                             "If 0 (default), uses adaptive threshold: mean - refine_sigma * std.")
    parser.add_argument("--refine_sigma", type=float, default=1.5,
                        help="For adaptive threshold: threshold = mean - sigma * std. "
                             "Lower sigma = more aggressive (more pixels refined).")
    parser.add_argument("--refine_topk", type=float, default=0.15,
                        help="Top-K fraction: refine at most this fraction of lowest-confidence pixels "
                             "(0.15 = bottom 15%%). Both threshold AND topk must be satisfied.")
    parser.add_argument("--refine_patch_size", type=int, default=96,
                        help="Patch size for local refinement crops.")
    parser.add_argument("--refine_context_pad", type=int, default=16,
                        help="Context padding around each refinement patch (for seamless blending).")
    parser.add_argument("--refine_t_start", type=int, default=2,
                        help="SDEdit: start re-denoising from this timestep (0-indexed, lower=less noise). "
                             "Model has 4 steps total (0,1,2,3). t_start=2 adds moderate noise.")
    parser.add_argument("--min_region_size", type=int, default=64,
                        help="Minimum connected component size (pixels) to trigger refinement.")
    parser.add_argument("--max_refine_patches", type=int, default=20,
                        help="Maximum number of patches to refine per image.")

    # Cycle spinning
    parser.add_argument("--cycle_spin", type=str2bool, default=False,
                        help="Enable cycle spinning for global pass.")

    # Output options
    parser.add_argument("--save_confidence", type=str2bool, default=True,
                        help="Save confidence maps as visualization.")
    parser.add_argument("--save_mask", type=str2bool, default=True,
                        help="Save refinement masks.")
    parser.add_argument("--save_before_refine", type=str2bool, default=False,
                        help="Also save the result BEFORE refinement for comparison.")

    args = parser.parse_args()
    if args.input_path is not None:
        args.in_path = args.input_path
    if args.output_path is not None:
        args.out_path = args.output_path
    return args


#  Config Loading (reused from inference_resshift.py)
_LINK_VQGAN = 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth'

def load_configs(args):
    """Load config and set up paths."""
    configs = OmegaConf.load(args.cfg_path)

    # VQGAN weights
    ckpt_dir = Path('./weights')
    ckpt_dir.mkdir(exist_ok=True)
    vqgan_path = ckpt_dir / 'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
        print("[Info] Downloading VQGAN...")
        load_file_from_url(url=_LINK_VQGAN, model_dir=str(ckpt_dir), progress=True, file_name=vqgan_path.name)
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # Model checkpoint
    configs.model.ckpt_path = args.ckpt_path

    # Scale
    if hasattr(configs, 'diffusion') and hasattr(configs.diffusion, 'params'):
        if hasattr(configs.diffusion.params, 'sf'):
            args.scale = configs.diffusion.params.sf

    # Output dir
    Path(args.out_path).mkdir(parents=True, exist_ok=True)

    # Chop stride
    if args.chop_stride < 0:
        sf_ratio = 4 // args.scale
        if args.chop_size == 512:
            chop_stride = (512 - 64) * sf_ratio
        elif args.chop_size == 256:
            chop_stride = (256 - 32) * sf_ratio
        elif args.chop_size == 64:
            chop_stride = (64 - 16) * sf_ratio
        else:
            chop_stride = (args.chop_size - 32) * sf_ratio
    else:
        chop_stride = args.chop_stride * (4 // args.scale)

    args.chop_size *= (4 // args.scale)

    return configs, chop_stride


# ──────────────────────────────────────────────────────────────
#  Core: Single Image Processing with Confidence
# ──────────────────────────────────────────────────────────────
def process_single_image_with_confidence(sampler, im_lq_tensor):
    """
    Run global diffusion pass and return both result and confidence map.

    Input:
        im_lq_tensor: [1, C, H, W], torch tensor, [-1, 1]
    Output:
        im_sr_tensor: [1, 3, H, W], torch tensor, [0, 1]
        confidence:   [1, 1, H, W], torch tensor, or None
    """
    context = torch.cuda.amp.autocast if sampler.use_amp else nullcontext
    with context():
        with torch.no_grad():
            im_sr_tensor, confidence = sampler.sample_func_with_confidence(
                im_lq_tensor, noise_repeat=False, mask=None
            )

    # [-1,1] → [0,1]
    im_sr_tensor = im_sr_tensor * 0.5 + 0.5
    return im_sr_tensor, confidence


# ──────────────────────────────────────────────────────────────
#  Core: Build Refinement Mask from Confidence Map
# ──────────────────────────────────────────────────────────────
def build_refinement_mask(confidence, threshold, topk_frac, min_region_size, sigma=1.5):
    """
    Build a binary mask of low-confidence regions.

    Args:
        confidence: [1, 1, H, W] tensor
        threshold: absolute threshold (0 = use adaptive: mean - sigma*std)
        topk_frac: top-K fraction of lowest confidence
        min_region_size: minimum connected component area
        sigma: for adaptive threshold

    Returns:
        mask_np: [H, W] uint8 numpy array, 255 = refine, 0 = keep
        stats: dict with diagnostic info
    """
    conf_np = confidence[0, 0].cpu().numpy()  # [H, W]
    H, W = conf_np.shape

    # 1. Threshold (adaptive or absolute)
    conf_mean = float(conf_np.mean())
    conf_std = float(conf_np.std())
    if threshold <= 0:
        # Adaptive: low-confidence = below mean - sigma * std
        actual_threshold = conf_mean - sigma * conf_std
    else:
        actual_threshold = threshold

    mask_abs = conf_np < actual_threshold

    # 2. Top-K: only refine the bottom topk_frac of pixels
    total_pixels = H * W
    k = int(total_pixels * topk_frac)
    if k > 0:
        flat = conf_np.flatten()
        kth_value = np.partition(flat, k)[k]  # k-th smallest value
        mask_topk = conf_np <= kth_value
    else:
        mask_topk = np.zeros_like(mask_abs)

    # Combined: both conditions must hold
    mask = (mask_abs & mask_topk).astype(np.uint8) * 255

    # 3. Morphological cleanup: remove tiny regions, connect nearby ones
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove tiny blobs

    # 4. Connected components filter: remove regions smaller than min_region_size
    num_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label_id in range(1, num_labels):
        area = comp_stats[label_id, cv2.CC_STAT_AREA]
        if area < min_region_size:
            mask[labels == label_id] = 0

    stats = {
        'conf_mean': float(conf_np.mean()),
        'conf_std': float(conf_np.std()),
        'conf_min': float(conf_np.min()),
        'conf_max': float(conf_np.max()),
        'actual_threshold': float(actual_threshold),
        'refine_pixel_count': int(mask.sum() // 255),
        'refine_pixel_frac': float((mask > 0).sum()) / total_pixels,
        'num_components': int((np.unique(labels[mask > 0])).shape[0]) if mask.sum() > 0 else 0,
    }

    return mask, stats


# ──────────────────────────────────────────────────────────────
#  Core: Extract Refinement Patches from Mask
# ──────────────────────────────────────────────────────────────
def extract_refine_patches(mask, patch_size, context_pad, max_patches):
    """
    Extract patch coordinates covering the refinement mask regions.

    Returns list of dicts:
        {'y': crop_y, 'x': crop_x, 'h': crop_h, 'w': crop_w,
         'inner_y': ..., 'inner_x': ..., 'inner_h': ..., 'inner_w': ...}
    """
    H, W = mask.shape
    if mask.sum() == 0:
        return []

    # Find bounding boxes of connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    patches = []
    for label_id in range(1, num_labels):
        cx = int(centroids[label_id][0])
        cy = int(centroids[label_id][1])
        bx = stats[label_id, cv2.CC_STAT_LEFT]
        by = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]

        # If the component fits in one patch, use its centroid
        if bw <= patch_size and bh <= patch_size:
            # Center patch on component centroid
            inner_y = max(0, cy - patch_size // 2)
            inner_x = max(0, cx - patch_size // 2)
            inner_y = min(inner_y, H - patch_size)
            inner_x = min(inner_x, W - patch_size)
            inner_h = patch_size
            inner_w = patch_size

            # Add context padding
            crop_y = max(0, inner_y - context_pad)
            crop_x = max(0, inner_x - context_pad)
            crop_y2 = min(H, inner_y + inner_h + context_pad)
            crop_x2 = min(W, inner_x + inner_w + context_pad)

            patches.append({
                'crop_y': crop_y, 'crop_x': crop_x,
                'crop_h': crop_y2 - crop_y, 'crop_w': crop_x2 - crop_x,
                'inner_y': inner_y - crop_y, 'inner_x': inner_x - crop_x,
                'inner_h': inner_h, 'inner_w': inner_w,
                'global_y': inner_y, 'global_x': inner_x,
            })
        else:
            # Large component: tile with overlapping patches
            for tile_y in range(by, by + bh, patch_size // 2):
                for tile_x in range(bx, bx + bw, patch_size // 2):
                    inner_y = max(0, min(tile_y, H - patch_size))
                    inner_x = max(0, min(tile_x, W - patch_size))
                    inner_h = min(patch_size, H - inner_y)
                    inner_w = min(patch_size, W - inner_x)

                    crop_y = max(0, inner_y - context_pad)
                    crop_x = max(0, inner_x - context_pad)
                    crop_y2 = min(H, inner_y + inner_h + context_pad)
                    crop_x2 = min(W, inner_x + inner_w + context_pad)

                    patches.append({
                        'crop_y': crop_y, 'crop_x': crop_x,
                        'crop_h': crop_y2 - crop_y, 'crop_w': crop_x2 - crop_x,
                        'inner_y': inner_y - crop_y, 'inner_x': inner_x - crop_x,
                        'inner_h': inner_h, 'inner_w': inner_w,
                        'global_y': inner_y, 'global_x': inner_x,
                    })

    # Deduplicate overlapping patches (keep unique by global position)
    seen = set()
    unique_patches = []
    for p in patches:
        key = (p['global_y'] // (patch_size // 2), p['global_x'] // (patch_size // 2))
        if key not in seen:
            seen.add(key)
            unique_patches.append(p)

    # Limit number of patches
    if len(unique_patches) > max_patches:
        # Prioritize patches with more mask coverage
        for p in unique_patches:
            gy, gx = p['global_y'], p['global_x']
            gh, gw = p['inner_h'], p['inner_w']
            p['mask_coverage'] = mask[gy:gy+gh, gx:gx+gw].sum() / 255.0
        unique_patches.sort(key=lambda x: x['mask_coverage'], reverse=True)
        unique_patches = unique_patches[:max_patches]

    return unique_patches


# ──────────────────────────────────────────────────────────────
#  Core: Local SDEdit Refinement
# ──────────────────────────────────────────────────────────────
def local_sdedit_refine(sampler, im_sr, im_lq, patches, mask_np, t_start):
    """
    SDEdit-style local refinement: add noise at timestep t_start, then re-denoise.

    Args:
        sampler: ResShiftSampler
        im_sr: [1, 3, H, W] tensor [0, 1] — current global prediction
        im_lq: [1, C, H, W] tensor [-1, 1] — original LQ input
        patches: list of patch dicts
        mask_np: [H, W] uint8 refinement mask
        t_start: SDEdit starting timestep

    Returns:
        im_refined: [1, 3, H, W] tensor [0, 1] — refined result
    """
    if len(patches) == 0:
        return im_sr

    device = im_sr.device
    im_refined = im_sr.clone()
    diffusion = sampler.base_diffusion
    model = sampler.model
    autoencoder = sampler.autoencoder
    context = torch.cuda.amp.autocast if sampler.use_amp else nullcontext

    # Create soft blending weight (cosine taper at edges)
    def make_blend_weight(h, w, taper=16):
        """Create a soft blending weight map with cosine taper at edges."""
        weight = torch.ones(1, 1, h, w, device=device)
        if taper <= 0:
            return weight
        for i in range(min(taper, h // 2)):
            alpha = 0.5 * (1 - math.cos(math.pi * i / taper))
            weight[:, :, i, :] *= alpha
            weight[:, :, h - 1 - i, :] *= alpha
        for j in range(min(taper, w // 2)):
            alpha = 0.5 * (1 - math.cos(math.pi * j / taper))
            weight[:, :, :, j] *= alpha
            weight[:, :, :, w - 1 - j] *= alpha
        return weight

    # Accumulator for blending overlapping patches
    H, W = im_sr.shape[2], im_sr.shape[3]
    blend_accum = torch.zeros(1, 3, H, W, device=device)
    weight_accum = torch.zeros(1, 1, H, W, device=device)

    num_timesteps = diffusion.num_timesteps
    t_start = min(t_start, num_timesteps - 1)  # clamp

    print(f"  [Refine] Processing {len(patches)} patches at t_start={t_start} "
          f"(total diffusion steps={num_timesteps})")

    for idx, patch in enumerate(patches):
        cy, cx = patch['crop_y'], patch['crop_x']
        ch, cw = patch['crop_h'], patch['crop_w']

        # Crop from current prediction [0,1] → [-1,1] for diffusion
        sr_crop = im_refined[:, :, cy:cy+ch, cx:cx+cw]  # [1, 3, ch, cw]
        sr_crop_norm = (sr_crop - 0.5) / 0.5             # [0,1] → [-1,1]

        # Crop LQ condition
        lq_crop = im_lq[:, :, cy:cy+ch, cx:cx+cw]       # [1, 2, ch, cw]

        # For VQGAN: expand VV to 3ch for diffusion y
        is_dualband = (lq_crop.shape[1] == 2)
        if is_dualband:
            lq_for_diffusion = lq_crop[:, 0:1, :, :].expand(-1, 3, -1, -1)
        else:
            lq_for_diffusion = lq_crop

        # Padding to make divisible by model offset
        offset = sampler.padding_offset
        pad_h = (math.ceil(ch / offset)) * offset - ch
        pad_w = (math.ceil(cw / offset)) * offset - cw
        if pad_h > 0 or pad_w > 0:
            # Use 'reflect' only when padding < dimension, otherwise 'replicate'
            pad_mode = 'reflect' if (pad_h < ch and pad_w < cw) else 'replicate'
            sr_crop_norm = F.pad(sr_crop_norm, (0, pad_w, 0, pad_h), mode=pad_mode)
            lq_crop = F.pad(lq_crop, (0, pad_w, 0, pad_h), mode=pad_mode)
            lq_for_diffusion = F.pad(lq_for_diffusion, (0, pad_w, 0, pad_h), mode=pad_mode)

        with context():
            with torch.no_grad():
                # Encode to latent space
                z_sr = diffusion.encode_first_stage(sr_crop_norm, autoencoder, up_sample=False)
                z_lq = diffusion.encode_first_stage(lq_for_diffusion, autoencoder, up_sample=True)

                # SDEdit: add noise at t_start
                noise = torch.randn_like(z_sr)
                t_tensor = torch.tensor([t_start], device=device)

                # Forward diffuse z_sr to timestep t_start
                z_t = diffusion.q_sample(z_sr, z_lq, t_tensor, noise=noise)

                # Set up model kwargs (pass original 2ch LQ as condition)
                if sampler.configs.model.params.cond_lq:
                    model_kwargs = {'lq': lq_crop}
                else:
                    model_kwargs = None

                # Denoise from t_start down to 0
                z_sample = z_t
                for step_i in range(t_start, -1, -1):
                    t_step = torch.tensor([step_i] * z_sample.shape[0], device=device)
                    out = diffusion.p_sample(
                        model, z_sample, z_lq, t_step,
                        clip_denoised=(autoencoder is None),
                        denoised_fn=None,
                        model_kwargs=model_kwargs,
                        noise_repeat=False,
                    )
                    z_sample = out["sample"]

                # Decode back to pixel space
                refined_crop = diffusion.decode_first_stage(z_sample, first_stage_model=autoencoder)
                refined_crop = refined_crop.clamp(-1.0, 1.0)
                refined_crop = refined_crop * 0.5 + 0.5  # [-1,1] → [0,1]

                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    refined_crop = refined_crop[:, :, :ch, :cw]

        # Build blend weight
        blend_w = make_blend_weight(ch, cw, taper=patch.get('inner_x', 16))

        # Only refine where the mask says so — soft mask
        mask_crop = torch.from_numpy(
            mask_np[cy:cy+ch, cx:cx+cw].astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,ch,cw]

        # Combined weight = blend * mask
        combined_weight = blend_w * mask_crop

        blend_accum[:, :, cy:cy+ch, cx:cx+cw] += refined_crop * combined_weight
        weight_accum[:, :, cy:cy+ch, cx:cx+cw] += combined_weight

    # Merge: where weight > 0, use blended; else keep original
    has_weight = (weight_accum > 1e-6).float()
    safe_weight = weight_accum.clamp(min=1e-6)
    blended = blend_accum / safe_weight

    im_refined = im_sr * (1 - has_weight) + blended * has_weight

    return im_refined.clamp(0, 1)


# ──────────────────────────────────────────────────────────────
#  Visualization Helpers
# ──────────────────────────────────────────────────────────────
def save_confidence_vis(confidence, save_path):
    """Save confidence map as a heatmap image."""
    if confidence is None:
        return
    conf_np = confidence[0, 0].cpu().numpy()
    # Normalize to [0, 255]
    conf_min, conf_max = conf_np.min(), conf_np.max()
    if conf_max - conf_min > 1e-6:
        conf_norm = ((conf_np - conf_min) / (conf_max - conf_min) * 255).astype(np.uint8)
    else:
        conf_norm = np.zeros_like(conf_np, dtype=np.uint8)
    # Apply colormap
    conf_color = cv2.applyColorMap(conf_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(save_path), conf_color)


def save_mask_vis(mask_np, save_path):
    """Save refinement mask as an image."""
    cv2.imwrite(str(save_path), mask_np)


# ──────────────────────────────────────────────────────────────
#  Load SAR Input (dual-band VV+VH)
# ──────────────────────────────────────────────────────────────
def load_sar_dualband(img_path, vv_channel=2, vh_channel=1):
    """
    Load SAR image and extract VV + VH channels.

    Returns:
        img_lq_tensor: [1, 2, H, W] tensor, [-1, 1]
    """
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img_bgr = img_bgr.astype(np.float32) / 255.0

    if len(img_bgr.shape) == 2:
        # Grayscale → stack as 2 channels (VV = VH)
        vv = img_bgr
        vh = img_bgr
    elif img_bgr.shape[2] >= 3:
        vv = img_bgr[:, :, vv_channel]
        vh = img_bgr[:, :, vh_channel]
    else:
        vv = img_bgr[:, :, 0]
        vh = img_bgr[:, :, min(1, img_bgr.shape[2] - 1)]

    # Stack [VV, VH] → [2, H, W]
    img_lq = np.stack([vv, vh], axis=0)  # [2, H, W]
    img_lq_tensor = torch.from_numpy(img_lq).float().unsqueeze(0)  # [1, 2, H, W]

    # [0,1] → [-1,1]
    img_lq_tensor = img_lq_tensor * 2.0 - 1.0

    return img_lq_tensor


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    args = get_parser()
    configs, chop_stride = load_configs(args)

    print("=" * 70)
    print("  Confidence-Guided Local Refinement Inference")
    print("=" * 70)
    print(f"  Config:    {args.cfg_path}")
    print(f"  Checkpoint:{args.ckpt_path}")
    print(f"  Input:     {args.in_path}")
    print(f"  Output:    {args.out_path}")
    print(f"  Threshold: {args.refine_threshold} (sigma={args.refine_sigma})")
    print(f"  Top-K:     {args.refine_topk}")
    print(f"  Patch:     {args.refine_patch_size}")
    print(f"  t_start:   {args.refine_t_start}")
    print("=" * 70)

    # Build sampler
    sampler = ResShiftSampler(
        configs,
        sf=args.scale,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=args.bs,
        use_amp=True,
        seed=args.seed,
        padding_offset=configs.model.params.get('lq_size', 64),
    )

    # Determine VV/VH channels from config
    vv_ch = configs.data.val.get('vv_channel_bgr', 2)
    vh_ch = configs.data.val.get('vh_channel_bgr', 1)

    # Scan input files
    if os.path.isfile(args.in_path):
        files = [args.in_path]
    else:
        files = sorted(glob.glob(os.path.join(args.in_path, '*')))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]

    if len(files) == 0:
        print(f"[Error] No images found at {args.in_path}")
        sys.exit(1)

    print(f"[Info] Found {len(files)} images.")
    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_confidence:
        (out_dir / 'confidence').mkdir(exist_ok=True)
    if args.save_mask:
        (out_dir / 'masks').mkdir(exist_ok=True)
    if args.save_before_refine:
        (out_dir / 'before_refine').mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_refined = 0
    total_skipped = 0

    for path in tqdm(files, desc="Processing"):
        try:
            img_name = os.path.splitext(os.path.basename(path))[0]

            # 1. Load SAR input
            img_lq_tensor = load_sar_dualband(path, vv_ch, vh_ch).to(device)

            # 2. Global pass → prediction + confidence
            im_sr, confidence = process_single_image_with_confidence(sampler, img_lq_tensor)

            # 3. Save before-refine if requested
            if args.save_before_refine:
                sr_before = util_image.tensor2img(im_sr, rgb2bgr=True, min_max=(0, 1))
                util_image.imwrite(sr_before, str(out_dir / 'before_refine' / f"{img_name}.png"))

            # 4. Save confidence map
            if args.save_confidence and confidence is not None:
                save_confidence_vis(confidence, out_dir / 'confidence' / f"{img_name}_conf.png")

            # 5. Build refinement mask
            if confidence is not None:
                mask_np, stats = build_refinement_mask(
                    confidence,
                    threshold=args.refine_threshold,
                    topk_frac=args.refine_topk,
                    min_region_size=args.min_region_size,
                    sigma=args.refine_sigma,
                )
                print(f"  [{img_name}] conf mean={stats['conf_mean']:.4f}±{stats['conf_std']:.4f}, "
                      f"min={stats['conf_min']:.4f}, max={stats['conf_max']:.4f}, "
                      f"thresh={stats['actual_threshold']:.4f}, "
                      f"refine_frac={stats['refine_pixel_frac']:.2%}, "
                      f"components={stats['num_components']}")

                if args.save_mask:
                    save_mask_vis(mask_np, out_dir / 'masks' / f"{img_name}_mask.png")

                # 6. Extract patches
                patches = extract_refine_patches(
                    mask_np,
                    patch_size=args.refine_patch_size,
                    context_pad=args.refine_context_pad,
                    max_patches=args.max_refine_patches,
                )

                # 7. Local SDEdit refinement
                if len(patches) > 0:
                    im_sr = local_sdedit_refine(
                        sampler, im_sr, img_lq_tensor,
                        patches, mask_np,
                        t_start=args.refine_t_start,
                    )
                    total_refined += 1
                else:
                    print(f"  [{img_name}] No patches need refinement → skip")
                    total_skipped += 1
            else:
                print(f"  [{img_name}] No confidence map available → skip refinement")
                total_skipped += 1

            # 8. Save final result
            output_img = util_image.tensor2img(im_sr, rgb2bgr=True, min_max=(0, 1))
            util_image.imwrite(output_img, str(out_dir / f"{img_name}.png"))

            # Cleanup
            del img_lq_tensor, im_sr, confidence
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"[Error] Failed to process {path}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

    print("=" * 70)
    print(f"[Done] Refined: {total_refined}, Skipped: {total_skipped}, Total: {len(files)}")
    print(f"[Done] Results saved to {args.out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
