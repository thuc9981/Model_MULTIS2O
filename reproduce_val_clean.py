#!/usr/bin/env python
"""
Reproduce validation inference EXACTLY as trainer.py does.
No cycle spinning, no chopping, no extra logic.
Directly replicates the val pipeline step by step.
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from pathlib import Path
from contextlib import nullcontext
from omegaconf import OmegaConf

from utils import util_common, util_net, util_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of S1 (LQ) images")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    os.makedirs(args.output_dir, exist_ok=True)

    # ──────────────────────────────────────
    # 1. Load config (EXACTLY like trainer)
    # ──────────────────────────────────────
    configs = OmegaConf.load(args.cfg_path)
    print(f"Config: {args.cfg_path}")

    # ──────────────────────────────────────
    # 2. Build diffusion model (EXACTLY like BaseSampler.build_model)
    # ──────────────────────────────────────
    base_diffusion = util_common.instantiate_from_config(configs.diffusion)

    model = util_common.instantiate_from_config(configs.model).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        util_net.reload_model(model, ckpt['state_dict'])
    else:
        util_net.reload_model(model, ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Model loaded from {args.ckpt_path}")

    # ──────────────────────────────────────
    # 3. Load EMA weights (EXACTLY like trainer uses ema_model for val)
    # ──────────────────────────────────────
    if args.use_ema:
        ckpt_dir = Path(args.ckpt_path).parent
        ema_dir = ckpt_dir.parent / 'ema_ckpts'
        ema_name = f"ema_{Path(args.ckpt_path).name}"
        ema_path = ema_dir / ema_name
        if ema_path.exists():
            print(f"Loading EMA from {ema_path}")
            ema_state = torch.load(str(ema_path), map_location=device)
            util_net.reload_model(model, ema_state)
        else:
            print(f"WARNING: EMA not found at {ema_path}")

    # ──────────────────────────────────────
    # 4. Load autoencoder (EXACTLY like BaseSampler.build_model)
    # ──────────────────────────────────────
    vqgan_path = configs.autoencoder.ckpt_path
    if not os.path.exists(vqgan_path):
        vqgan_path = 'weights/autoencoder_vq_f4.pth'
    params = configs.autoencoder.get('params', dict)
    autoencoder = util_common.get_obj_from_str(configs.autoencoder.target)(**params)
    autoencoder.to(device)
    ae_state = torch.load(vqgan_path, map_location=device)
    if 'state_dict' in ae_state:
        ae_state = ae_state['state_dict']
    util_net.reload_model(autoencoder, ae_state)
    autoencoder.eval()
    print(f"Autoencoder loaded from {vqgan_path}")

    # ──────────────────────────────────────
    # 5. Read dataset config
    # ──────────────────────────────────────
    data_cfg = configs.data.val if hasattr(configs.data, 'val') else configs.data.train
    vv_ch = int(data_cfg.get('vv_channel_bgr', 2))
    vh_ch = int(data_cfg.get('vh_channel_bgr', 1))
    print(f"Band extraction: VV from BGR ch{vv_ch}, VH from BGR ch{vh_ch}")

    # ──────────────────────────────────────
    # 6. Process images EXACTLY like validation
    # ──────────────────────────────────────
    files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ])
    if args.max_images:
        files = files[:args.max_images]
    print(f"Processing {len(files)} images...")

    use_amp = configs.train.get('use_amp', True)
    context = torch.cuda.amp.autocast if use_amp else nullcontext

    for idx, fname in enumerate(files):
        path = os.path.join(args.input_dir, fname)
        
        # ── Step A: Read image (EXACTLY like DualBandSARDataset.__getitem__) ──
        img_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            print(f"  Skip: cannot read {path}")
            continue
        img_raw = img_raw.astype(np.float32) / 255.0
        
        vv = img_raw[:, :, vv_ch]
        vh = img_raw[:, :, vh_ch]
        img_lq = np.stack([vv, vh], axis=-1)  # [H,W,2] → ch0=VV, ch1=VH

        # ── Step B: Convert to tensor (EXACTLY like dataset) ──
        img_lq_t = torch.from_numpy(img_lq.transpose(2, 0, 1)).float()  # [2,H,W]
        img_lq_t = img_lq_t * 2.0 - 1.0  # normalize to [-1,1]
        im_lq = img_lq_t.unsqueeze(0).to(device)  # [1,2,H,W]

        # ── Step C: Validation-style resolution handling (EXACTLY like prepare_data) ──
        val_resolution = configs.train.get('val_resolution', 256)
        h, w = im_lq.shape[2:]
        if h > val_resolution and w > val_resolution:
            h_end = int((h // val_resolution) * val_resolution)
            w_end = int((w // val_resolution) * val_resolution)
            im_lq = im_lq[:, :, :h_end, :w_end]
        else:
            h_pad = math.ceil(h / val_resolution) * val_resolution - h
            w_pad = math.ceil(w / val_resolution) * val_resolution - w
            padding_mode = configs.train.get('val_padding_mode', 'reflect')
            im_lq = F.pad(im_lq, pad=(0, w_pad, 0, h_pad), mode=padding_mode)

        # ── Step D: Setup model_kwargs (EXACTLY like trainer.py validation_step) ──
        model_kwargs = {'lq': im_lq}  # [1,2,H,W], [-1,1]

        # ── Step E: Expand VV to 3ch for diffusion (EXACTLY like trainer.py) ──
        vv_band = im_lq[:, 0:1, :, :]
        lq_for_diffusion = vv_band.expand(-1, 3, -1, -1)  # [1,3,H,W]

        # ── Step F: Run diffusion (EXACTLY like trainer.py validation) ──
        # trainer.py uses p_sample_loop_progressive and decodes at the end
        # p_sample_loop does the same thing internally
        with context():
            with torch.no_grad():
                final = None
                for sample in base_diffusion.p_sample_loop_progressive(
                    y=lq_for_diffusion,
                    model=model,
                    first_stage_model=autoencoder,
                    noise=None,
                    clip_denoised=False,  # clip_denoised=True if autoencoder is None else False
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=False,
                ):
                    final = sample

                # ── Step G: Decode (EXACTLY like trainer.py) ──
                im_sr = base_diffusion.decode_first_stage(
                    final['sample'],
                    autoencoder,
                ).clamp(-1.0, 1.0)

        # ── Step H: Convert to image ──
        # Use linear mapping [-1,1] → [0,1] → [0,255] (not make_grid normalize)
        im_sr_np = im_sr.squeeze(0).float().cpu()  # [3,H,W]
        im_sr_np = im_sr_np * 0.5 + 0.5  # [-1,1] → [0,1]
        im_sr_np = im_sr_np.clamp(0, 1)
        im_sr_np = im_sr_np.permute(1, 2, 0).numpy()  # [H,W,3] RGB
        im_sr_np = (im_sr_np * 255.0).round().astype(np.uint8)
        im_sr_bgr = cv2.cvtColor(im_sr_np, cv2.COLOR_RGB2BGR)

        # ── Step I: Save ──
        stem = os.path.splitext(fname)[0]
        save_path = os.path.join(args.output_dir, f"{stem}.png")
        cv2.imwrite(save_path, im_sr_bgr)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(files)}] {fname} → saved")

    print(f"Done! Results in {args.output_dir}")


if __name__ == '__main__':
    main()
