import os
import sys
import math
from pathlib import Path
import rasterio
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import util_common, util_net

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHECKPOINT_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/experiments/multisar_opera_hls_l2/2026-04-10-15-45-36/ema_ckpts/ema_model_1145000.pth"
CONFIG_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/configs/inference_HLS_OP.yaml"
OUTPUT_DIR = "/mnt/hdd12tb/code/thucnd/Data_HLS_opera_4/Data_HLS_infer_Batch_3_TEST"
AUTOENCODER_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/weights/autoencoder_vq_f4.pth"

DATA_DIRS = [
    Path("/mnt/hdd12tb/code/thucnd/Data_HLS_opera_4/pairs_aligned/test"),
    Path("/mnt/hdd12tb/code/thucnd/Data_HLS_opera_4/pairs_aligned/val"),
    Path("/mnt/hdd12tb/code/thucnd/Data_HLS_opera_4/pairs_aligned/train")
]

# ----- CẤU HÌNH LƯỚI & ĐA LUỒNG -----
PATCH_SIZE = 256
GRID_N = 2  
MAX_SAR_IMAGES = 8

# Cấu hình tận dụng VRAM (24GB)
NUM_GPUS = 2
WORKERS_PER_GPU = 4  # 4 workers * 5GB = ~20GB VRAM mỗi GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. HÀM TIỆN ÍCH DỮ LIỆU
# ==========================================
def load_config(config_path):
    return OmegaConf.load(config_path)

def load_and_norm_sar(tif_path):
    with rasterio.open(tif_path) as src:
        img = src.read()
        nodata_val = src.profile.get('nodata', np.nan)
        
    vv = img[0].astype('float32')
    vh = img[1].astype('float32')
    
    if np.isnan(nodata_val):
        valid_mask = ~np.isnan(vv) & ~np.isnan(vh)
    else:
        valid_mask = (vv != nodata_val) & (vh != nodata_val) & ~np.isnan(vv) & ~np.isnan(vh)

    EPSILON = 1e-10
    
    if img.shape[0] >= 3:
        ratio = img[2].astype('float32')
    else:
        ratio = np.full_like(vv, np.nan)
    
    if np.nanmin(vv) >= 0:
        if img.shape[0] < 3:
            with np.errstate(divide='ignore', invalid='ignore'):
                linear_ratio = np.divide(vv, vh)
                linear_ratio[~valid_mask] = np.nan
                ratio = linear_ratio
                
        vv_db = np.full_like(vv, np.nan)
        vh_db = np.full_like(vh, np.nan)
        ratio_db = np.full_like(ratio, np.nan)
        
        valid_vv = np.clip(vv[valid_mask], 0, None) + EPSILON
        valid_vh = np.clip(vh[valid_mask], 0, None) + EPSILON
        valid_ratio = np.clip(ratio[valid_mask], 0, None) + EPSILON
        
        vv_db[valid_mask] = 10 * np.log10(valid_vv)
        vh_db[valid_mask] = 10 * np.log10(valid_vh)
        ratio_db[valid_mask] = 10 * np.log10(valid_ratio)
            
        vv = vv_db
        vh = vh_db
        ratio = ratio_db
    else:
        if img.shape[0] < 3:
            ratio = vv - vh
            
    vv = np.where(valid_mask, vv, np.nan)
    vh = np.where(valid_mask, vh, np.nan)
    ratio = np.where(valid_mask, ratio, np.nan)

    img_vv = np.clip((vv + 25.0) / 30.0, 0.0, 1.0)
    img_vh = np.clip((vh + 35.0) / 40.0, 0.0, 1.0)
    img_ratio = np.clip(ratio / 16.0, 0.0, 1.0)
    
    sar_stack = np.stack([img_vv, img_vh, img_ratio], axis=0)
    sar_stack = sar_stack * 2.0 - 1.0
    # sar_stack = np.nan_to_num(sar_stack, nan=-1.0) 
    
    return torch.from_numpy(sar_stack).float()

def get_crop_coords(H, W):
    stride_x = (W - PATCH_SIZE) // (GRID_N - 1) if W > PATCH_SIZE and GRID_N > 1 else 0
    stride_y = (H - PATCH_SIZE) // (GRID_N - 1) if H > PATCH_SIZE and GRID_N > 1 else 0
    coords = []
    for r in range(GRID_N):
        for c in range(GRID_N):
            y1 = r * stride_y
            x1 = c * stride_x
            if r == GRID_N - 1: y1 = max(0, H - PATCH_SIZE)
            if c == GRID_N - 1: x1 = max(0, W - PATCH_SIZE)
            coords.append({"y1": y1, "y2": y1 + PATCH_SIZE, "x1": x1, "x2": x1 + PATCH_SIZE})
    return coords

def create_soft_blending_mask(patch_size, device):
    linspace = torch.linspace(0.5, 1.0, steps=patch_size // 2, device=device)
    mask_1d = torch.cat([linspace, torch.flip(linspace, dims=[0])])
    mask_2d = mask_1d.unsqueeze(1) * mask_1d.unsqueeze(0)
    return mask_2d.unsqueeze(0).unsqueeze(0)

def _maybe_load_ema_model(model, ckpt_path):
    ckpt_path = Path(ckpt_path)
    ema_path = ckpt_path.parent.parent / "ema_ckpts" / f"ema_{ckpt_path.name}"
    if not ema_path.exists(): 
        return False
    ema_state = util_common.torch_load(str(ema_path), map_location="cpu")
    util_net.reload_model(model, ema_state)
    return True

# ==========================================
# 3. QUY TRÌNH GPU WORKER (BATCH INFERENCE)
# ==========================================
def worker_process(gpu_id, worker_id, folder_list, config_path, checkpoint_path, autoencoder_path):
    device = f"cuda:{gpu_id}"
    
    from ldm.models.autoencoder import VQModelTorch
    from models.unet_multisar import UNetModelSwinMultiSAR

    config = load_config(config_path)

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
    win_2d = create_soft_blending_mask(PATCH_SIZE, device)

    # Thêm worker_id để thanh tiến trình hiển thị không bị đè lên nhau
    for folder in tqdm(folder_list, desc=f"GPU {gpu_id} | Worker {worker_id}", position=worker_id, leave=False):
        base_name = folder.name
        sar_files = sorted(list(folder.glob("*_OP_*.tif")))
        if not sar_files: continue
        
        sar_files = sar_files[:MAX_SAR_IMAGES]
        num_sar = len(sar_files)
        
        try:
            sar_tensors = [load_and_norm_sar(f) for f in sar_files]
            lq_full = torch.stack(sar_tensors, dim=0).to(device) # [N, 3, H, W]
            _, _, H, W = lq_full.shape
            
            coords = get_crop_coords(H, W)

            pred_canvas = torch.zeros((1, 3, H, W), device=device)
            weight_canvas = torch.zeros((1, 1, H, W), device=device)

            lq_patches_list = []
            valid_coords = [] 
            
            for c in coords:
                y1, y2, x1, x2 = c["y1"], c["y2"], c["x1"], c["x2"]
                lq_patch = lq_full[:, :, y1:y2, x1:x2]
                
                # Bỏ qua patch nếu chứa NaN (áp dụng fix từ lần trước)
                if torch.isnan(lq_patch).any():
                    continue
                    
                lq_patches_list.append(lq_patch)
                valid_coords.append(c)
                
            if not lq_patches_list:
                continue

            lq_batch = torch.stack(lq_patches_list, dim=0)
            num_valid_patches = len(valid_coords)
            num_sar_batch = torch.tensor([num_sar] * num_valid_patches, dtype=torch.long, device=device)

            y_cond = lq_batch[:, :num_sar].mean(dim=1)
            model_kwargs = {"lq": lq_batch, "num_sar": num_sar_batch}
            
            final_sample = None
            with torch.no_grad():
                for sample in diffusion.p_sample_loop_progressive(
                    y=y_cond, model=model, first_stage_model=autoencoder, noise=None,
                    clip_denoised=False, model_kwargs=model_kwargs, device=device, progress=False
                ):
                    final_sample = sample
                
                output_batch = diffusion.decode_first_stage(final_sample["sample"], autoencoder).clamp(-1.0, 1.0)
            
            for i, c in enumerate(valid_coords):
                y1, y2, x1, x2 = c["y1"], c["y2"], c["x1"], c["x2"]
                out_patch = output_batch[i].unsqueeze(0)
                pred_canvas[:, :, y1:y2, x1:x2] += out_patch * win_2d
                weight_canvas[:, :, y1:y2, x1:x2] += win_2d

            valid_mask = weight_canvas > 0.01
            weight_safe = torch.where(valid_mask, weight_canvas, torch.ones_like(weight_canvas))
            pred_stitched = (pred_canvas / weight_safe).clamp(-1.0, 1.0)
            
            out_np = pred_stitched[0].cpu().numpy()                
            out_np = (out_np + 1.0) / 2.0                          
            out_np = np.clip(out_np, 0.0, 1.0).astype(np.float32)  
            
            valid_mask_np = valid_mask[0].cpu().numpy()
            out_np = np.where(valid_mask_np, out_np, np.nan)
            
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_Optical_Predicted.npy")
            np.save(save_path, out_np)
            
        except Exception as e:
            print(f"\n[GPU {gpu_id} | W{worker_id}] Lỗi khi xử lý {base_name}: {e}")
            continue

# ==========================================
# 4. TRÌNH QUẢN LÝ TIẾN TRÌNH CHÍNH
# ==========================================
def main():
    mp.set_start_method('spawn', force=True)

    print("=" * 80)
    print(f" VRAM OPTIMIZED BATCH INFERENCE | {NUM_GPUS} GPUs | {WORKERS_PER_GPU} WORKERS/GPU")
    print("=" * 80)

    print("[1/2] Quét tất cả thư mục dữ liệu...")
    all_folders = []
    for d in DATA_DIRS:
        if d.exists():
            all_folders.extend([f for f in d.iterdir() if f.is_dir()])
            
    total_folders = len(all_folders)
    if total_folders == 0:
        print(" Không tìm thấy thư mục nào!")
        sys.exit(1)

    # Chia việc cho 8 luồng (2 GPU * 4 workers)
    total_workers = NUM_GPUS * WORKERS_PER_GPU
    folder_chunks = [all_folders[i::total_workers] for i in range(total_workers)]
    
    print(f" Phân bổ tác vụ: Tổng {total_folders} ảnh chia cho {total_workers} luồng xử lý đồng thời.")

    print("[2/2] Bắt đầu kích hoạt các luồng Inference song song...\n")
    
    processes = []
    for worker_id in range(total_workers):
        gpu_id = worker_id % NUM_GPUS
        chunk = folder_chunks[worker_id]
        
        p = mp.Process(
            target=worker_process, 
            args=(gpu_id, worker_id, chunk, CONFIG_PATH, CHECKPOINT_PATH, AUTOENCODER_PATH)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\n" + "=" * 80)
    print(" HOÀN TẤT TOÀN BỘ QUÁ TRÌNH DỰ ĐOÁN TỐC ĐỘ CAO!")
    print(f" Dữ liệu lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()