import os
import sys
from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
import lpips

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import util_common, util_image, util_net

# ============ CONFIG ============
CHECKPOINT_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/experiments_10m/multisar_opera_hls_l2/2026-04-10-15-45-36/ema_ckpts/ema_model_2160000.pth"
CONFIG_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/configs/multisar_opera_hls_l2.yaml"
OUTPUT_DIR = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/test_2160000_OP2HLS_EndToEnd_DualGPU_Batched"
AUTOENCODER_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/weights/autoencoder_vq_f4.pth"

# ----- CẤU HÌNH XỬ LÝ -----
ROOT_PAIRS = Path("/mnt/hdd12tb/code/thucnd/Data_S1_S2_2_5_10m/pairs_aligned/test")
PATCH_SIZE = 256
GRID_SIZE = 6

# BATCH SIZE (12-18 LÀ ĐẸP CHO VRAM 24GB)
BATCH_SIZE = 36

os.makedirs(OUTPUT_DIR, exist_ok=True)
PATCH_VIS_DIR = os.path.join(OUTPUT_DIR, "Patches_Visualizations")
os.makedirs(PATCH_VIS_DIR, exist_ok=True)

# ==========================================
# CÁC HÀM TIỆN ÍCH CƠ BẢN
# ==========================================
def load_config(config_path):
    return OmegaConf.load(config_path)

def _maybe_load_ema_model(model, ckpt_path):
    ckpt_path = Path(ckpt_path)
    ema_path = ckpt_path.parent.parent / "ema_ckpts" / f"ema_{ckpt_path.name}"
    if not ema_path.exists(): return False
    ema_state = util_common.torch_load(str(ema_path), map_location="cpu")
    util_net.reload_model(model, ema_state)
    return True

def get_dynamic_window_by_coords(x, y, P_size, W, H, device):
    win_y = torch.hann_window(P_size, periodic=False)
    win_x = torch.hann_window(P_size, periodic=False)
    
    tolerance = 5
    if y <= tolerance: win_y[:P_size//2] = 1.0
    if y + P_size >= H - tolerance: win_y[P_size//2:] = 1.0
    if x <= tolerance: win_x[:P_size//2] = 1.0
    if x + P_size >= W - tolerance: win_x[P_size//2:] = 1.0
        
    win_2d = (win_y.unsqueeze(1) * win_x.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return win_2d.to(device)

def tensor_to_image(tensor):
    x = tensor.detach().float().cpu()
    if x.ndim == 4: x = x[0]
    arr = (x.numpy().transpose(1, 2, 0) + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))

def save_full_res_image(tensor, save_path):
    img = tensor_to_image(tensor)
    img.save(save_path)

def create_visualization(sar_images, output_img, gt_img, base_name, save_path):
    img_size = 256
    padding = 5
    text_height = 45
    header_height = 30

    n_sar = len(sar_images)
    max_display = min(n_sar, 6)
    cols = 3
    sar_rows = (max_display + cols - 1) // cols
    total_rows = sar_rows + 1

    canvas_width = cols * img_size + (cols + 1) * padding
    canvas_height = header_height + total_rows * (img_size + text_height) + (total_rows + 1) * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except: font = ImageFont.load_default()

    def resize_img(img): return img.resize((img_size, img_size), Image.LANCZOS)

    draw.text((padding, 5), f"Result: {base_name}", fill=(0, 0, 128), font=font)

    y_offset = header_height
    for i in range(max_display):
        row, col = i // cols, i % cols
        x = padding + col * (img_size + padding)
        y = y_offset + row * (img_size + text_height + padding)
        canvas.paste(resize_img(sar_images[i]), (x, y))
        draw.text((x, y + img_size + 2), f"SAR Input {i + 1}", fill=(0, 0, 0), font=font)

    result_row_y = y_offset + sar_rows * (img_size + text_height + padding)
    
    output_x = padding
    canvas.paste(resize_img(output_img), (output_x, result_row_y))
    draw.text((output_x, result_row_y + img_size + 2), "OUTPUT (Predicted)", fill=(0, 128, 0), font=font)

    gt_x = padding + 2 * (img_size + padding)
    canvas.paste(resize_img(gt_img), (gt_x, result_row_y))
    draw.text((gt_x, result_row_y + img_size + 2), "GROUND TRUTH", fill=(0, 0, 255), font=font)

    canvas.save(save_path)

# ==========================================
# HÀM XỬ LÝ LÕI TRÊN TỪNG GPU (Tích hợp BATCHING)
# ==========================================
def worker_process(gpu_id, folder_list, config_path, checkpoint_path, autoencoder_path):
    device = f"cuda:{gpu_id}"
    
    from ldm.models.autoencoder import VQModelTorch
    from models.unet_multisar import UNetModelSwinMultiSAR

    config = load_config(config_path)
    val_y_channel = bool(config.train.get("val_y_channel", False))

    model = UNetModelSwinMultiSAR(**config.model.params)
    checkpoint = util_common.torch_load(checkpoint_path, map_location="cpu")
    util_net.reload_model(model, checkpoint)
    _maybe_load_ema_model(model, checkpoint_path)
    model = model.to(device).float().eval()

    autoencoder = VQModelTorch(**config.autoencoder.params)
    ae_ckpt = util_common.torch_load(autoencoder_path, map_location="cpu")
    util_net.reload_model(autoencoder, ae_ckpt)
    autoencoder = autoencoder.to(device).float().eval()

    diffusion = util_common.instantiate_from_config(config.diffusion)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    total_psnr, total_ssim, total_lpips, total_images = 0.0, 0.0, 0.0, 0

    for folder_path in tqdm(folder_list, desc=f"GPU {gpu_id}", position=gpu_id):
        row_name = folder_path.name
        
        s2_files = list(folder_path.glob("*_S2L2A_*RGB*.[tT][iI][fF]"))
        s1_files = list(folder_path.glob("*_S1_*.[tT][iI][fF]"))
        
        if not s2_files or not s1_files:
            continue
            
        s2_path = s2_files[0]
        total_s1_files = len(s1_files)
        
        try:
            with rasterio.open(s2_path) as src_s2_meta:
                H, W = src_s2_meta.height, src_s2_meta.width
        except Exception:
            continue
            
        cv_data = {
            'pred': torch.zeros(1, 3, H, W, device=device),
            'gt': torch.zeros(1, 3, H, W, device=device),
            'weight': torch.zeros(1, 1, H, W, device=device),
            'sars': [torch.zeros(1, 3, H, W, device=device) for _ in range(total_s1_files)],
            'count': 0
        }

        stride_x = (W - PATCH_SIZE) // (GRID_SIZE - 1) if W > PATCH_SIZE else 0
        stride_y = (H - PATCH_SIZE) // (GRID_SIZE - 1) if H > PATCH_SIZE else 0

        src_s2 = rasterio.open(s2_path)
        src_s1_list = [rasterio.open(p) for p in s1_files]

        patch_idx = 0
        
        # Danh sách Gom Batch
        batch_lq = []
        batch_gt = []
        batch_num_sar = []
        batch_coords = []
        
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                patch_idx += 1
                y_off, x_off = int(r * stride_y), int(c * stride_x)
                window = Window(x_off, y_off, PATCH_SIZE, PATCH_SIZE)
                
                # --- ĐỌC QUANG HỌC (S2) ---
                s2_patch = src_s2.read(window=window)[:3]
                s2_patch = np.nan_to_num(s2_patch, nan=0.0) 
                s2_patch = np.clip(s2_patch, 0, 3000).astype(np.float32)
                
                s2_tensor = torch.from_numpy(s2_patch)
                s2_tensor = (s2_tensor / 3000.0) * 2.0 - 1.0
                
                # --- ĐỌC VÀ LỌC SAR ---
                valid_s1_tensors = []
                for idx_s1, src_s1 in enumerate(src_s1_list):
                    s1_patch = src_s1.read(window=window)
                    
                    nan_count = np.isnan(s1_patch[0]).sum()
                    nan_ratio = nan_count / s1_patch[0].size
                    
                    if nan_ratio > 0.30: continue
                        
                    s1_patch = np.nan_to_num(s1_patch, nan=0.0)
                    
                    vv_linear = s1_patch[0]
                    vh_linear = s1_patch[1]
                    ratio_linear = vv_linear - vh_linear
                    
                    vv_clipped = np.clip(vv_linear, -25, 5)
                    vh_clipped = np.clip(vh_linear, -35, -5)
                    ratio_clipped = np.clip(ratio_linear, 0, 16)
                    
                    vv_norm = (vv_clipped - (-25)) / 30.0 * 2.0 - 1.0
                    vh_norm = (vh_clipped - (-35)) / 40.0 * 2.0 - 1.0
                    ratio_norm = (ratio_clipped - 0) / 16.0 * 2.0 - 1.0
                    
                    s1_processed = np.stack([vv_norm, vh_norm, ratio_norm], axis=0).astype(np.float32)
                    valid_s1_tensors.append((idx_s1, torch.from_numpy(s1_processed)))
                
                if not valid_s1_tensors:
                    blank_s1 = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                    valid_s1_tensors.append((0, torch.from_numpy(blank_s1)))
                
                n_valid = len(valid_s1_tensors)
                s1_stack = torch.stack([t[1] for t in valid_s1_tensors], dim=0) 
                
                padded_s1 = torch.zeros((total_s1_files, 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)
                padded_s1[:n_valid] = s1_stack
                
                orig_indices = [t[0] for t in valid_s1_tensors]

                batch_lq.append(padded_s1)
                batch_gt.append(s2_tensor)
                batch_num_sar.append(n_valid)
                batch_coords.append((x_off, y_off, orig_indices, patch_idx))

                # ===============================================
                # KIỂM TRA ĐIỀU KIỆN KÍCH HOẠT GPU
                # ===============================================
                is_last_patch = (patch_idx == GRID_SIZE * GRID_SIZE)
                
                if len(batch_lq) == BATCH_SIZE or (is_last_patch and len(batch_lq) > 0):
                    
                    lq_batch = torch.stack(batch_lq).to(device)  
                    gt_batch = torch.stack(batch_gt).to(device)  
                    num_sar_batch = torch.tensor(batch_num_sar, dtype=torch.long, device=device) 

                    with torch.no_grad():
                        model_kwargs = {"lq": lq_batch, "num_sar": num_sar_batch}
                        y_diff = lq_batch[:, 0, :, :, :] 

                        final_sample = None
                        for sample in diffusion.p_sample_loop_progressive(
                            y=y_diff, model=model, first_stage_model=autoencoder, noise=None,
                            clip_denoised=False, model_kwargs=model_kwargs, device=device, progress=False
                        ):
                            final_sample = sample
                            
                        output_batch = diffusion.decode_first_stage(final_sample["sample"], autoencoder).clamp(-1.0, 1.0)

                        current_b_size = len(batch_lq)
                        for b in range(current_b_size):
                            out_p = output_batch[b]
                            gt_p = gt_batch[b]
                            b_n_valid = batch_num_sar[b]
                            b_x, b_y, b_orig_idxs, b_p_idx = batch_coords[b]
                            
                            # CẬP NHẬT VÁ LỖI TẠI ĐÂY: Thêm .unsqueeze(0) để bảo vệ số chiều
                            patch_vis_path = os.path.join(PATCH_VIS_DIR, f"{row_name}_patch_{b_p_idx}_x{b_x}_y{b_y}.png")
                            sar_patch_imgs = [tensor_to_image(lq_batch[b, i, 0:1].unsqueeze(0).expand(-1, 3, -1, -1)) for i in range(b_n_valid)]
                            create_visualization(sar_images=sar_patch_imgs, output_img=tensor_to_image(out_p), gt_img=tensor_to_image(gt_p), base_name=f"{row_name} (P:{b_p_idx})", save_path=patch_vis_path)

                            # TÍNH TOÁN DÁN CHÍNH XÁC VÀO CANVAS
                            x1, y1 = b_x, b_y
                            x2, y2 = x1 + PATCH_SIZE, y1 + PATCH_SIZE
                            
                            win_2d = get_dynamic_window_by_coords(x1, y1, PATCH_SIZE, W, H, device)
                            
                            cv_data['pred'][:, :, y1:y2, x1:x2] += out_p * win_2d
                            cv_data['gt'][:, :, y1:y2, x1:x2] += gt_p * win_2d
                            cv_data['weight'][:, :, y1:y2, x1:x2] += win_2d
                            
                            # CẬP NHẬT VÁ LỖI TẠI ĐÂY: Thêm .unsqueeze(0)
                            for list_idx, orig_idx in enumerate(b_orig_idxs):
                                sar_patch_t = lq_batch[b, list_idx, 0:1].unsqueeze(0).expand(-1, 3, -1, -1)
                                cv_data['sars'][orig_idx][:, :, y1:y2, x1:x2] += sar_patch_t * win_2d
                                
                            cv_data['count'] += 1

                    # Reset Batch List
                    batch_lq.clear()
                    batch_gt.clear()
                    batch_num_sar.clear()
                    batch_coords.clear()

        src_s2.close()
        for src in src_s1_list: src.close()

        # --- XẢ CANVAS KHI ĐÃ CẮT & KHÂU XONG ---
        if cv_data['count'] > 0:
            with torch.no_grad():
                valid_mask = (cv_data['weight'] > 1e-5).float()
                weight_safe = torch.where(valid_mask > 0, cv_data['weight'], torch.ones_like(cv_data['weight']))
                
                pred_full = (cv_data['pred'] / weight_safe).clamp(-1.0, 1.0)
                gt_full = (cv_data['gt'] / weight_safe).clamp(-1.0, 1.0)
                sar_fulls = [s / weight_safe for s in cv_data['sars']]

                pred_full = pred_full * valid_mask + (-1.0) * (1.0 - valid_mask)
                gt_full = gt_full * valid_mask + (-1.0) * (1.0 - valid_mask)
                
                max_sars = [s for s in sar_fulls if s.abs().sum() > 0]
                for i in range(len(max_sars)):
                    max_sars[i] = max_sars[i] * valid_mask + (-1.0) * (1.0 - valid_mask)

                curr_lpips = lpips_fn(pred_full, gt_full).sum().item()
                curr_psnr = util_image.batch_PSNR(pred_full * 0.5 + 0.5, gt_full * 0.5 + 0.5, ycbcr=val_y_channel)
                curr_ssim = util_image.batch_SSIM(pred_full * 0.5 + 0.5, gt_full * 0.5 + 0.5, ycbcr=val_y_channel)

                total_psnr += curr_psnr
                total_ssim += curr_ssim
                total_lpips += curr_lpips
                total_images += 1

                pred_save_path = os.path.join(OUTPUT_DIR, f"{row_name}_Stitched_Full.png")
                save_full_res_image(pred_full, pred_save_path)
                
                vis_save_path = os.path.join(OUTPUT_DIR, f"{row_name}_Stitched_Compare.png")
                sar_imgs = [tensor_to_image(s) for s in max_sars]
                create_visualization(
                    sar_images=sar_imgs, output_img=tensor_to_image(pred_full),
                    gt_img=tensor_to_image(gt_full), base_name=row_name, save_path=vis_save_path
                )

# ==========================================
# KHỞI CHẠY ĐA TIẾN TRÌNH (DUAL GPU)
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    print("=" * 80)
    print(" END-TO-END INFERENCE: DUAL GPU & BATCH PROCESSING (MAX VRAM)")
    print("=" * 80)

    all_subfolders = [f for f in ROOT_PAIRS.iterdir() if f.is_dir()]
    total_folders = len(all_subfolders)
    
    if total_folders == 0:
        print(" KHÔNG TÌM THẤY ẢNH NÀO TRONG THƯ MỤC ROOT_PAIRS!")
        sys.exit()

    mid_point = total_folders // 2
    folders_gpu_0 = all_subfolders[:mid_point]
    folders_gpu_1 = all_subfolders[mid_point:]

    print(f" Tìm thấy {total_folders} ảnh. Phân bổ: GPU 0 ({len(folders_gpu_0)} ảnh) | GPU 1 ({len(folders_gpu_1)} ảnh).")

    p0 = mp.Process(target=worker_process, args=(0, folders_gpu_0, CONFIG_PATH, CHECKPOINT_PATH, AUTOENCODER_PATH))
    p1 = mp.Process(target=worker_process, args=(1, folders_gpu_1, CONFIG_PATH, CHECKPOINT_PATH, AUTOENCODER_PATH))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    print("\n" + "=" * 80)
    print(" QUÁ TRÌNH INFERENCE TRÊN 2 GPU ĐÃ HOÀN TẤT!")
    print("=" * 80)