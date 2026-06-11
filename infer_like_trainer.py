#!/usr/bin/env python
"""
Inference script that runs the EXACT same code path as trainer.py validation().

This is NOT a reimplementation — it literally uses the same:
  - PairedImageDataset (same data loading)
  - prepare_data(data, phase='val') (same preprocessing)
  - p_sample_loop_progressive (same diffusion sampling)
  - decode_first_stage(...).clamp(-1.0, 1.0)  (same decoding)
  - logging_image → make_grid(normalize=True, scale_each=True) (same saving)
  - batch_PSNR(pred*0.5+0.5, gt*0.5+0.5, ycbcr=True) (same PSNR)

Usage:
  python infer_like_trainer.py \
    --cfg_path configs/multisar_opera_hls_l2.yaml \
    --ckpt_path /mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/experiments/multisar_opera_hls_l2/2026-04-10-15-45-36/ckpts/model_760000.pth \
    --test_lq /mnt/hdd12tb/code/thucnd/Data_HLS_opera_4/Dataset_256x256_test_NPY/val/OPERA \
    --test_gt /mnt/hdd12tb/code/thucnd/Data_HLS_opera_4/Dataset_256x256_test_NPY/val/HLS \
    --output_dir val_test_22 \
    --use_ema \
    --gpu 0
"""

import os
import sys
import math
import argparse
import numpy as np
from pathlib import Path
from contextlib import nullcontext

import cv2
import torch
import torch.nn.functional as F
import torch.utils.data as udata
import torchvision.utils as vutils
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm

# ── project imports (same as trainer.py) ──
sys.path.insert(0, os.path.dirname(__file__))
from utils import util_common, util_net, util_image
from datapipe.datasets import create_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Run trainer-identical validation on test set")
    p.add_argument("--cfg_path", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--test_lq", type=str, required=True, help="Test LQ dir (test/A)")
    p.add_argument("--test_gt", type=str, required=True, help="Test GT dir (test/B)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--use_ema", action="store_true", default=False)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override val batch size (default: use config train.batch[1])")
    p.add_argument("--max_images", type=int, default=None, help="Limit number of test images")
    p.add_argument("--save_individual", action="store_true", default=False,
                   help="Also save each image individually (pred, gt, lq) as separate PNGs")
    return p.parse_args()


def build_model(configs, ckpt_path, use_ema, device):
    """Build diffusion + UNet + autoencoder exactly as trainer.__init__"""

    # ── diffusion ──
    base_diffusion = util_common.instantiate_from_config(configs.diffusion)

    # ── UNet model ──
    model = util_common.instantiate_from_config(configs.model).to(device)

    # ── load checkpoint (same as trainer.reload_model) ──
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if 'state_dict' in ckpt:
        util_net.reload_model(model, ckpt['state_dict'])
    else:
        util_net.reload_model(model, ckpt)

    # ── EMA weights ──
    if use_ema:
        ckpt_dir = Path(ckpt_path).parent
        ema_dir = ckpt_dir.parent / 'ema_ckpts'
        ema_name = f"ema_{Path(ckpt_path).name}"
        ema_path = ema_dir / ema_name
        if ema_path.exists():
            print(f"[EMA] Loading from {ema_path}")
            ema_state = torch.load(str(ema_path), map_location="cpu")
            # EMA was saved from DDP, keys have 'module.' prefix → use reload_model to strip
            util_net.reload_model(model, ema_state)
        else:
            print(f"[WARN] EMA not found at {ema_path}, using base model weights")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── Autoencoder (VQGAN) ──
    vqgan_path = configs.autoencoder.ckpt_path
    if not os.path.exists(vqgan_path):
        vqgan_path = 'weights/autoencoder_vq_f4.pth'
    params = configs.autoencoder.get('params', dict)
    autoencoder = util_common.get_obj_from_str(configs.autoencoder.target)(**params)
    ae_state = torch.load(vqgan_path, map_location="cpu")
    if 'state_dict' in ae_state:
        ae_state = ae_state['state_dict']
    util_net.reload_model(autoencoder, ae_state)
    autoencoder.to(device).eval()

    return base_diffusion, model, autoencoder


def prepare_data_val(data, configs, device, dtype=torch.float32):
    """
    Exact copy of trainer.prepare_data(data, phase='val').
    """
    offset = configs.train.get('val_resolution', 256)
    for key, value in data.items():
        if not hasattr(value, 'shape'):
            continue
        h, w = value.shape[2:]
        if h > offset and w > offset:
            h_end = int((h // offset) * offset)
            w_end = int((w // offset) * offset)
            data[key] = value[:, :, :h_end, :w_end]
        else:
            h_pad = math.ceil(h / offset) * offset - h
            w_pad = math.ceil(w / offset) * offset - w
            padding_mode = configs.train.get('val_padding_mode', 'reflect')
            data[key] = F.pad(value, pad=(0, w_pad, 0, h_pad), mode=padding_mode)

    output = {}
    for key, value in data.items():
        if hasattr(value, 'cuda'):
            output[key] = value.to(device).to(dtype=dtype)
        else:
            output[key] = value
    return output


def logging_image_to_file(im_tensor, save_path, nrow=8):
    """
    Exact copy of trainer.logging_image — uses make_grid(normalize=True, scale_each=True).
    Then saves to disk with util_image.imwrite (same as trainer).
    """
    im_grid = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True)
    im_np = im_grid.cpu().permute(1, 2, 0).numpy()
    util_image.imwrite(im_np, str(save_path))


def save_individual_image(tensor, save_path):
    """
    Save a single [1,3,H,W] tensor clamped to [0,1] as a PNG.
    Uses the same path as trainer for individual saves.
    """
    img = tensor.squeeze(0).cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    img_u8 = (img * 255.0).round().astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img_bgr)


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    # ── Load config ──
    configs = OmegaConf.load(args.cfg_path)

    # ── Build dataset: override val paths to point to test set ──
    val_cfg = dict(configs.data.val)
    val_cfg['dataroot_lq'] = args.test_lq
    val_cfg['dataroot_gt'] = args.test_gt
    val_cfg['phase'] = 'val'  # keep as val so PairedImageDataset skips augmentation
    dataset = create_dataset(val_cfg)
    print(f"Test dataset: {len(dataset)} images")

    # ── Dataloader: same as trainer ──
    batch_size = args.batch_size if args.batch_size else configs.train.batch[1]
    dataloader = udata.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    # ── Build model ──
    base_diffusion, model, autoencoder = build_model(
        configs, args.ckpt_path, args.use_ema, device
    )

    # ── Prepare output dirs (same structure as trainer's image_dir/val/) ──
    out_dir = Path(args.output_dir)
    grid_dir = out_dir / "grid"    # progress/gt/lq grids (same as trainer)
    indiv_dir = out_dir / "individual"  # per-image saves
    grid_dir.mkdir(parents=True, exist_ok=True)
    if args.save_individual:
        (indiv_dir / "pred").mkdir(parents=True, exist_ok=True)
        (indiv_dir / "gt").mkdir(parents=True, exist_ok=True)
        (indiv_dir / "lq").mkdir(parents=True, exist_ok=True)

    # ── indices: exact same as trainer.validation() ──
    num_timesteps = base_diffusion.num_timesteps
    indices = np.linspace(
        0,
        num_timesteps,
        num_timesteps if num_timesteps < 5 else 4,
        endpoint=False,
        dtype=np.int64,
    ).tolist()
    if not (num_timesteps - 1) in indices:
        indices.append(num_timesteps - 1)
    print(f"Diffusion timesteps: {num_timesteps}, decode at indices: {indices}")

    # ── context manager: same as trainer ──
    context = torch.cuda.amp.autocast if configs.train.use_amp else nullcontext

    # ── Metrics ──
    mean_psnr = 0.0
    mean_lpips = 0.0
    mean_mse = 0.0
    val_count = 0
    has_gt = False
    log_step = 0

    max_images = args.max_images

    print(f"\nRunning validation-style inference on test set...")
    print(f"  Batch size: {batch_size}")
    print(f"  AMP: {configs.train.use_amp}")
    print(f"  EMA: {args.use_ema}")
    print(f"  val_y_channel: {configs.train.val_y_channel}")
    print(f"  val_resolution: {configs.train.get('val_resolution', 256)}")
    print()

    for ii, data in enumerate(tqdm(dataloader, desc="Inference")):
        if max_images is not None and val_count >= max_images:
            break

        # ── prepare_data(data, phase='val') — SAME as trainer ──
        data = prepare_data_val(data, configs, device)

        if 'gt' in data:
            im_lq, im_gt = data['lq'], data['gt']
        else:
            im_lq = data['lq']

        is_dualband = (im_lq.shape[1] == 2)

        # ── model_kwargs — SAME as trainer ──
        num_iters = 0
        if configs.model.params.cond_lq:
            model_kwargs = {'lq': data['lq']}
        else:
            model_kwargs = None

        if is_dualband:
            vv_band = im_lq[:, 0:1, :, :]
            lq_for_diffusion = vv_band.expand(-1, 3, -1, -1)
        else:
            lq_for_diffusion = im_lq

        # ── Diffusion sampling — SAME as trainer ──
        with context():
            for sample in base_diffusion.p_sample_loop_progressive(
                y=lq_for_diffusion,
                model=model,
                first_stage_model=autoencoder,
                noise=None,
                clip_denoised=True if autoencoder is None else False,
                model_kwargs=model_kwargs,
                device=device,
                progress=False,
            ):
                sample_decode = {}
                if num_iters in indices:
                    for key, value in sample.items():
                        if key in ['sample']:
                            # ── decode + clamp: SAME as trainer line 1122 ──
                            sample_decode[key] = base_diffusion.decode_first_stage(
                                value,
                                autoencoder,
                            ).clamp(-1.0, 1.0)
                    im_sr_progress = sample_decode['sample']
                    if num_iters + 1 == 1:
                        im_sr_all = im_sr_progress
                    else:
                        im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                num_iters += 1

        # ── Metrics — SAME as trainer ──
        if 'gt' in data:
            has_gt = True
            bs = im_gt.shape[0]
            batch_mse = F.mse_loss(sample_decode['sample'], im_gt, reduction='mean')

            mean_psnr += util_image.batch_PSNR(
                sample_decode['sample'] * 0.5 + 0.5,
                im_gt * 0.5 + 0.5,
                ycbcr=configs.train.val_y_channel,
            )
            mean_mse += batch_mse.item() * bs
            val_count += bs

        # ── Save images — SAME as trainer.logging_image ──
        # Progress grid (all intermediate + final timesteps concatenated)
        vis_channels = 3
        im_sr_vis = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=vis_channels)
        logging_image_to_file(
            im_sr_vis,
            grid_dir / f"progress-{log_step}.png",
            nrow=len(indices),
        )
        if 'gt' in data:
            logging_image_to_file(im_gt, grid_dir / f"gt-{log_step}.png")
        if is_dualband:
            vv_vis = im_lq[:, 0:1, :, :].expand(-1, 3, -1, -1)
            logging_image_to_file(vv_vis, grid_dir / f"lq_VV-{log_step}.png")
            vh_vis = im_lq[:, 1:2, :, :].expand(-1, 3, -1, -1)
            logging_image_to_file(vh_vis, grid_dir / f"lq_VH-{log_step}.png")
        else:
            logging_image_to_file(im_lq, grid_dir / f"lq-{log_step}.png")

        # ── Also save individual images if requested ──
        if args.save_individual and 'gt' in data:
            pred = sample_decode['sample']  # [-1, 1]
            for b_idx in range(pred.shape[0]):
                global_idx = val_count - bs + b_idx
                fname = f"{global_idx:05d}.png"
                # pred: clamp to [0,1] for individual save (raw pixel values)
                save_individual_image(pred[b_idx:b_idx+1].clamp(0, 1),
                                      indiv_dir / "pred" / fname)
                save_individual_image(im_gt[b_idx:b_idx+1],
                                      indiv_dir / "gt" / fname)
                save_individual_image(im_lq[b_idx:b_idx+1],
                                      indiv_dir / "lq" / fname)

        log_step += 1

    # ── Summary — SAME as trainer ──
    print()
    print("=" * 70)
    if has_gt and val_count > 0:
        mean_psnr /= val_count
        mean_mse /= val_count
        print(f"Validation Metric (trainer-identical):")
        print(f"  PSNR  = {mean_psnr:.4f} dB  (same formula as trainer: *0.5+0.5, Y-channel, uint8)")
        print(f"  MSE   = {mean_mse:.6f}")
        print(f"  Count = {val_count} images")
    print(f"\nGrid images saved to: {grid_dir}/")
    if args.save_individual:
        print(f"Individual images saved to: {indiv_dir}/")
    print(f"  progress-N.png = diffusion sampling progress (all timesteps)")
    print(f"  gt-N.png       = ground truth")
    print(f"  lq-N.png       = input (noisy SAR)")
    print(f"\nThese grid images use make_grid(normalize=True, scale_each=True)")
    print(f"which is IDENTICAL to how trainer.py saves validation images.")
    print("=" * 70)


if __name__ == '__main__':
    main()
