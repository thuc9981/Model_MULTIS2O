import os
import sys
from pathlib import Path
import rasterio
from rasterio.windows import Window
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

# ============ CONFIG ============
CHECKPOINT_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/experiments_10m/multisar_opera_hls_l2/2026-04-10-15-45-36/ckpts/model_2160000.pth"
CONFIG_PATH     = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/configs/multisar_opera_hls_l2.yaml"
OUTPUT_DIR      = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/Data-inferS2_ALL"
AUTOENCODER_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/weights/autoencoder_vq_f4.pth"

# ----- DATA ROOT: xử lý tất cả 3 splits -----
DATA_ROOT  = Path("/mnt/hdd12tb/code/thucnd/Data_S1_S2_2_5_10m/pairs_aligned")
SPLITS     = ["test"]

PATCH_SIZE = 256
GRID_SIZE  = 6
BATCH_SIZE = 36   

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# UTILITY
# ==========================================
def load_config(config_path):
    return OmegaConf.load(config_path)


def _maybe_load_ema_model(model, ckpt_path):
    ckpt_path = Path(ckpt_path)
    ema_path  = ckpt_path.parent.parent / "ema_ckpts" / f"ema_{ckpt_path.name}"
    if not ema_path.exists():
        return False
    ema_state = util_common.torch_load(str(ema_path), map_location="cpu")
    util_net.reload_model(model, ema_state)
    return True


def get_dynamic_window_by_coords(x, y, P_size, W, H, device):
    """
    Hann window 2D, flatten biên (=1.0) để không bị mờ mép ảnh.
    Shape trả về: (1, 1, P_size, P_size)
    """
    win_y = torch.hann_window(P_size, periodic=False)
    win_x = torch.hann_window(P_size, periodic=False)

    tolerance = 5
    if y <= tolerance:              win_y[:P_size // 2] = 1.0
    if y + P_size >= H - tolerance: win_y[P_size // 2:] = 1.0
    if x <= tolerance:              win_x[:P_size // 2] = 1.0
    if x + P_size >= W - tolerance: win_x[P_size // 2:] = 1.0

    win_2d = (win_y.unsqueeze(1) * win_x.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return win_2d.to(device)


# ==========================================
# WORKER CHẠY TRÊN 1 GPU
# ==========================================
def worker_process(gpu_id, folder_list, config_path, checkpoint_path, autoencoder_path):
    device = f"cuda:{gpu_id}"

    # Import nặng chỉ làm 1 lần trong mỗi process con
    from ldm.models.autoencoder import VQModelTorch
    from models.unet_multisar import UNetModelSwinMultiSAR

    config = load_config(config_path)

    # --- Load model ---
    model = UNetModelSwinMultiSAR(**config.model.params)
    checkpoint = util_common.torch_load(checkpoint_path, map_location="cpu")
    util_net.reload_model(model, checkpoint)
    _maybe_load_ema_model(model, checkpoint_path)
    model = model.to(device).float().eval()

    # --- Load autoencoder ---
    autoencoder = VQModelTorch(**config.autoencoder.params)
    ae_ckpt = util_common.torch_load(autoencoder_path, map_location="cpu")
    util_net.reload_model(autoencoder, ae_ckpt)
    autoencoder = autoencoder.to(device).float().eval()

    # --- Diffusion scheduler ---
    diffusion = util_common.instantiate_from_config(config.diffusion)

    # -------------------------------------------------------
    for folder_path in tqdm(folder_list, desc=f"GPU {gpu_id}", position=gpu_id):
        row_name = folder_path.name

        # Tìm file S1 và S2 (S2 chỉ dùng để lấy H, W)
        s2_files = list(folder_path.glob("*_S2L2A_*RGB*.[tT][iI][fF]"))
        s1_files = list(folder_path.glob("*_S1_*.[tT][iI][fF]"))

        if not s2_files or not s1_files:
            continue

        s2_path        = s2_files[0]
        total_s1_files = len(s1_files)

        # Đọc kích thước ảnh gốc từ S2
        try:
            with rasterio.open(s2_path) as src_meta:
                H, W = src_meta.height, src_meta.width
        except Exception as e:
            print(f"[GPU {gpu_id}] Bỏ qua {row_name}: {e}")
            continue

        # ----- Canvas tích lũy (weighted sum, Hann blending) -----
        # Chỉ cần canvas cho pred — không cần gt nữa
        canvas_pred   = torch.zeros(1, 3, H, W, device=device)
        canvas_weight = torch.zeros(1, 1, H, W, device=device)

        # Tính stride để tạo grid 6×6 đều khắp ảnh
        stride_x = (W - PATCH_SIZE) // (GRID_SIZE - 1) if W > PATCH_SIZE else 0
        stride_y = (H - PATCH_SIZE) // (GRID_SIZE - 1) if H > PATCH_SIZE else 0

        # Mở file một lần, đọc nhiều patch
        src_s2   = rasterio.open(s2_path)
        src_s1_list = [rasterio.open(p) for p in s1_files]

        # --- Batch accumulator ---
        batch_lq      = []   # [(total_s1, 3, 256, 256), ...]
        batch_num_sar = []   # [n_valid, ...]
        batch_coords  = []   # [(x_off, y_off), ...]
        patch_idx     = 0

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                patch_idx += 1
                y_off = int(r * stride_y)
                x_off = int(c * stride_x)
                window = Window(x_off, y_off, PATCH_SIZE, PATCH_SIZE)

                # ── Đọc & xử lý SAR ──────────────────────────────
                valid_s1_tensors = []
                for src_s1 in src_s1_list:
                    s1_patch = src_s1.read(window=window)   # (2, H, W) — VV, VH

                    nan_ratio = np.isnan(s1_patch[0]).sum() / s1_patch[0].size
                    if nan_ratio > 0.30:
                        continue

                    s1_patch = np.nan_to_num(s1_patch, nan=0.0)

                    vv_linear = s1_patch[0]
                    vh_linear = s1_patch[1]
                    ratio_linear = vv_linear - vh_linear

                    # Clip về dải hợp lệ (dB)
                    vv_clipped    = np.clip(vv_linear,    -25,  5)
                    vh_clipped    = np.clip(vh_linear,    -35, -5)
                    ratio_clipped = np.clip(ratio_linear,   0, 16)

                    # Normalize về [-1, 1]
                    vv_norm    = (vv_clipped    - (-25)) / 30.0 * 2.0 - 1.0
                    vh_norm    = (vh_clipped    - (-35)) / 40.0 * 2.0 - 1.0
                    ratio_norm = (ratio_clipped -    0)  / 16.0 * 2.0 - 1.0

                    s1_proc = np.stack([vv_norm, vh_norm, ratio_norm], axis=0).astype(np.float32)
                    valid_s1_tensors.append(torch.from_numpy(s1_proc))

                # Nếu tất cả SAR đều NaN → dùng blank
                if not valid_s1_tensors:
                    blank = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                    valid_s1_tensors.append(torch.from_numpy(blank))

                n_valid  = len(valid_s1_tensors)
                s1_stack = torch.stack(valid_s1_tensors, dim=0)   # (n_valid, 3, 256, 256)

                # Pad về (total_s1_files, 3, 256, 256)
                padded_s1 = torch.zeros((total_s1_files, 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)
                padded_s1[:n_valid] = s1_stack

                batch_lq.append(padded_s1)
                batch_num_sar.append(n_valid)
                batch_coords.append((x_off, y_off))

                # ── Kích hoạt GPU khi đủ batch hoặc patch cuối ──
                is_last_patch = (patch_idx == GRID_SIZE * GRID_SIZE)
                if len(batch_lq) == BATCH_SIZE or (is_last_patch and len(batch_lq) > 0):

                    lq_batch      = torch.stack(batch_lq).to(device)          # (B, N_s1, 3, 256, 256)
                    num_sar_batch = torch.tensor(batch_num_sar, dtype=torch.long, device=device)

                    with torch.no_grad():
                        model_kwargs = {"lq": lq_batch, "num_sar": num_sar_batch}
                        y_diff       = lq_batch[:, 0, :, :, :]               # (B, 3, 256, 256)

                        final_sample = None
                        for sample in diffusion.p_sample_loop_progressive(
                            y=y_diff, model=model, first_stage_model=autoencoder,
                            noise=None, clip_denoised=False,
                            model_kwargs=model_kwargs, device=device, progress=False
                        ):
                            final_sample = sample

                        # Decode latent → pixel space, clamp về [-1, 1]
                        output_batch = diffusion.decode_first_stage(
                            final_sample["sample"], autoencoder
                        ).clamp(-1.0, 1.0)  # (B, 3, 256, 256), float32, [-1,1]

                    # --- Stitch từng patch vào canvas ---
                    for b in range(len(batch_lq)):
                        out_p = output_batch[b]    # (3, 256, 256), trên device
                        bx, by = batch_coords[b]
                        x2, y2 = bx + PATCH_SIZE, by + PATCH_SIZE

                        win_2d = get_dynamic_window_by_coords(bx, by, PATCH_SIZE, W, H, device)

                        canvas_pred  [:, :, by:y2, bx:x2] += out_p.unsqueeze(0) * win_2d
                        canvas_weight[:, :, by:y2, bx:x2] += win_2d

                    # Reset batch
                    batch_lq.clear()
                    batch_num_sar.clear()
                    batch_coords.clear()

        src_s2.close()
        for src in src_s1_list:
            src.close()

        # ── Normalize canvas & lưu numpy ─────────────────────────
        with torch.no_grad():
            valid_mask   = (canvas_weight > 1e-5).float()
            weight_safe  = torch.where(valid_mask > 0, canvas_weight, torch.ones_like(canvas_weight))

            # Chia có trọng số → ảnh đã stitch, fill vùng trống bằng -1.0
            pred_full = (canvas_pred / weight_safe).clamp(-1.0, 1.0)
            pred_full = pred_full * valid_mask + (-1.0) * (1.0 - valid_mask)

        # Chuyển về CPU numpy float32, shape (3, H, W), range [-1, 1]
        pred_np = pred_full[0].float().cpu().numpy()   # loại bỏ batch dim → (3, H, W)

        # Lưu: OUTPUT_DIR / split / <row_name>.npy
        # folder_path có dạng: DATA_ROOT / split / row_name
        split_name = folder_path.parent.name           # "train" / "val" / "test"
        split_out  = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_out, exist_ok=True)

        npy_save_path = os.path.join(split_out, f"{row_name}.npy")
        np.save(npy_save_path, pred_np)
        # → file .npy chứa array float32, shape (3, H, W), giá trị [-1.0, 1.0]


# ==========================================
# MAIN — DUAL GPU, 3 SPLITS
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    print("=" * 80)
    print(" INFERENCE — DUAL GPU | BATCH | SAVE NPY | TRAIN+VAL+TEST")
    print("=" * 80)

    # Thu thập tất cả thư mục từ 3 splits
    all_folders = []
    for split in SPLITS:
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            print(f"  [WARN] Không tìm thấy split: {split_dir}")
            continue
        folders = [f for f in split_dir.iterdir() if f.is_dir()]
        print(f"  Split [{split:5s}]: {len(folders)} ảnh")
        all_folders.extend(folders)

    total_folders = len(all_folders)
    if total_folders == 0:
        print(" Không tìm thấy thư mục nào!")
        sys.exit(1)

    print(f"\n  Tổng cộng: {total_folders} ảnh → phân bổ cho 2 GPU")
    print("=" * 80)

    mid = total_folders // 2
    folders_gpu_0 = all_folders[:mid]
    folders_gpu_1 = all_folders[mid:]
    print(f"  GPU 0: {len(folders_gpu_0)} ảnh | GPU 1: {len(folders_gpu_1)} ảnh")

    p0 = mp.Process(target=worker_process,
                    args=(0, folders_gpu_0, CONFIG_PATH, CHECKPOINT_PATH, AUTOENCODER_PATH))
    p1 = mp.Process(target=worker_process,
                    args=(1, folders_gpu_1, CONFIG_PATH, CHECKPOINT_PATH, AUTOENCODER_PATH))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print("\n" + "=" * 80)
    print(" HOÀN TẤT! Kết quả lưu tại:", OUTPUT_DIR)
    print("   Cấu trúc: OUTPUT_DIR/train/*.npy | /val/*.npy | /test/*.npy")
    print("   Format  : numpy float32, shape (3, H, W), range [-1.0, 1.0]")
    print("=" * 80)