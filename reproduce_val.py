"""
Reproduce the exact validation pipeline from trainer.py as a standalone script.
This will process a few val images and save the output EXACTLY as val does,
so we can compare with inference_resshift.py output.
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import math
from pathlib import Path
from contextlib import nullcontext
from omegaconf import OmegaConf
from utils import util_common, util_net, util_image
from torchvision import utils as vutils

def main():
    # 1. Load config
    cfg_path = 'configs/sar_dualband_vv_vh.yaml'
    configs = OmegaConf.load(cfg_path)
    
    # 2. Build diffusion
    print("Building diffusion model...")
    base_diffusion = util_common.instantiate_from_config(configs.diffusion)
    
    # 3. Build model
    print("Building UNet model...")
    model = util_common.instantiate_from_config(configs.model).cuda()
    
    # 4. Load EMA weights (same as validation)
    ckpt_path = 'experiments/sar_dualband_vv_vh/2026-02-07-16-16-38/ckpts/model_145000.pth'
    ema_path = 'experiments/sar_dualband_vv_vh/2026-02-07-16-16-38/ema_ckpts/ema_model_145000.pth'
    
    print(f"Loading EMA weights from {ema_path}...")
    ema_state = torch.load(ema_path, map_location="cuda:0")
    util_net.reload_model(model, ema_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # 5. Build autoencoder (VQGAN)
    print("Building autoencoder...")
    configs.autoencoder.ckpt_path = 'weights/autoencoder_vq_f4.pth'
    params = configs.autoencoder.get('params', dict)
    autoencoder = util_common.get_obj_from_str(configs.autoencoder.target)(**params)
    autoencoder.cuda()
    ae_ckpt = torch.load(configs.autoencoder.ckpt_path, map_location="cuda:0")
    if 'state_dict' in ae_ckpt:
        util_net.reload_model(autoencoder, ae_ckpt['state_dict'])
    else:
        util_net.reload_model(autoencoder, ae_ckpt)
    autoencoder.eval()
    
    # 6. Load a few val images using DualBandSARDataset logic
    val_s1_dir = '/mnt/fwgpu/hdd12tb/code/thucnd/Data_3_26k/val/s1'
    val_s2_dir = '/mnt/fwgpu/hdd12tb/code/thucnd/Data_3_26k/val/s2'
    out_dir = '/mnt/fwgpu/hdd12tb/code/thucnd/check_test/val_reproduce'
    os.makedirs(out_dir, exist_ok=True)
    
    vv_idx = 2  # BGR index for VV (R channel)
    vh_idx = 1  # BGR index for VH (G channel)
    
    files = sorted(os.listdir(val_s1_dir))[:4]  # First 4 images
    
    for i, fname in enumerate(files):
        print(f"\n--- Processing {fname} ({i+1}/{len(files)}) ---")
        
        # Read SAR image (same as DualBandSARDataset)
        lq_path = os.path.join(val_s1_dir, fname)
        img_raw = cv2.imread(lq_path, cv2.IMREAD_UNCHANGED)
        img_raw = img_raw.astype(np.float32) / 255.0
        
        vv = img_raw[:, :, vv_idx]
        vh = img_raw[:, :, vh_idx]
        img_lq = np.stack([vv, vh], axis=-1)  # [H, W, 2]
        
        # Read GT
        gt_path = os.path.join(val_s2_dir, fname)
        if os.path.exists(gt_path):
            img_gt_gray = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            img_gt_gray = img_gt_gray.astype(np.float32) / 255.0
            img_gt = np.stack([img_gt_gray]*3, axis=-1)  # [H, W, 3]
        
        # Convert to tensor (same as dataset)
        im_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float()  # [2, H, W]
        im_lq = im_lq * 2.0 - 1.0  # normalize to [-1, 1]
        im_lq = im_lq.unsqueeze(0).cuda()  # [1, 2, H, W]
        
        if os.path.exists(gt_path):
            im_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float()
            im_gt = im_gt * 2.0 - 1.0
            im_gt = im_gt.unsqueeze(0).cuda()
        
        print(f"  im_lq: shape={im_lq.shape}, min={im_lq.min():.3f}, max={im_lq.max():.3f}")
        
        # 7. Run diffusion (EXACTLY like trainer.py validation)
        model_kwargs = {'lq': im_lq}  # Full 2-channel as condition
        
        # Expand VV to 3ch for diffusion y
        vv_band = im_lq[:, 0:1, :, :]
        lq_for_diffusion = vv_band.expand(-1, 3, -1, -1)
        
        print(f"  lq_for_diffusion: shape={lq_for_diffusion.shape}")
        print(f"  model_kwargs['lq']: shape={model_kwargs['lq'].shape}")
        
        context = torch.cuda.amp.autocast
        
        with context():
            with torch.no_grad():
                # p_sample_loop already includes decode_first_stage internally
                output = base_diffusion.p_sample_loop(
                    y=lq_for_diffusion,
                    model=model,
                    first_stage_model=autoencoder,
                    noise=None,
                    noise_repeat=False,
                    clip_denoised=False,  # autoencoder is not None
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=False,
                )
        
        output = output.clamp(-1.0, 1.0)
        
        print(f"  output: shape={output.shape}, min={output.min():.3f}, max={output.max():.3f}")
        
        # === SAVE METHOD 1: Same as validation (make_grid with normalize=True) ===
        grid_val = vutils.make_grid(output, nrow=1, normalize=True, scale_each=True)
        grid_np = grid_val.cpu().permute(1, 2, 0).numpy()
        val_style_path = os.path.join(out_dir, f'{fname}_val_style.png')
        util_image.imwrite(grid_np, val_style_path)
        print(f"  Saved val-style: {val_style_path}")
        
        # === SAVE METHOD 2: Same as inference (tensor2img with [0,1]) ===
        output_01 = output * 0.5 + 0.5  # [-1,1] -> [0,1]
        inf_img = util_image.tensor2img(output_01, rgb2bgr=True, min_max=(0, 1))
        inf_style_path = os.path.join(out_dir, f'{fname}_inf_style.png')
        util_image.imwrite(inf_img, inf_style_path)
        print(f"  Saved inf-style: {inf_style_path}")
        
        # === SAVE METHOD 3: Direct raw output values ===
        output_np = output[0].cpu().numpy()  # [3, H, W] in [-1, 1]
        print(f"  Raw output stats: ch0 min={output_np[0].min():.4f} max={output_np[0].max():.4f} mean={output_np[0].mean():.4f}")
        print(f"                    ch1 min={output_np[1].min():.4f} max={output_np[1].max():.4f} mean={output_np[1].mean():.4f}")
        print(f"                    ch2 min={output_np[2].min():.4f} max={output_np[2].max():.4f} mean={output_np[2].mean():.4f}")
        
        # Save LQ VV for comparison
        lq_vv_01 = (im_lq[0, 0:1] * 0.5 + 0.5).expand(3, -1, -1)
        lq_vv_img = util_image.tensor2img(lq_vv_01.unsqueeze(0), rgb2bgr=True, min_max=(0, 1))
        lq_path_out = os.path.join(out_dir, f'{fname}_lq_vv.png')
        util_image.imwrite(lq_vv_img, lq_path_out)
        
        if os.path.exists(gt_path):
            gt_01 = im_gt * 0.5 + 0.5
            gt_img = util_image.tensor2img(gt_01, rgb2bgr=True, min_max=(0, 1))
            gt_path_out = os.path.join(out_dir, f'{fname}_gt.png')
            util_image.imwrite(gt_img, gt_path_out)
        
        del im_lq, lq_for_diffusion, output
        torch.cuda.empty_cache()
    
    print(f"\nDone! Results saved to {out_dir}")
    print("Compare *_val_style.png (how val shows it) vs *_inf_style.png (how inference saves it)")

if __name__ == '__main__':
    main()
