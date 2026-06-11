import os
import re
import sys
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from tqdm import tqdm
import rasterio  # Thêm thư viện để đọc metadata ảnh gốc nhanh chóng

import lpips

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import util_common, util_image, util_net

# ============ CONFIG ============
CHECKPOINT_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/experiments/multisar_opera_hls_l2/2026-04-10-15-45-36/ckpts/model_1145000.pth"
CONFIG_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/configs/multisar_opera_hls_l2.yaml"
OUTPUT_DIR = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/test_OP_HLS_1145000_visualizations"
AUTOENCODER_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/weights/autoencoder_vq_f4.pth"

# Đường dẫn đến thư mục chứa ảnh gốc (Dùng để lấy kích thước thật tự động)
PAIRS_BASE_DIR = Path("/mnt/hdd12tb/code/thucnd/Data_HLS_opera_4/pairs")

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 400

PATCH_SIZE = 256
GRID_N = 2  
TOTAL_PATCHES = GRID_N * GRID_N

os.makedirs(OUTPUT_DIR, exist_ok=True)
PATCH_VIS_DIR = os.path.join(OUTPUT_DIR, "Patches_Visualizations")
os.makedirs(PATCH_VIS_DIR, exist_ok=True)

# ==========================================
# CÁC HÀM TIỆN ÍCH CƠ BẢN
# ==========================================
def load_config(config_path): return OmegaConf.load(config_path)

def _prepare_spatial_val_4d(x, offset, padding_mode):
    h, w = x.shape[-2:]
    if h > offset and w > offset:
        h_end = int((h // offset) * offset)
        w_end = int((w // offset) * offset)
        return x[:, :, :h_end, :w_end]
    h_pad = math.ceil(h / offset) * offset - h
    w_pad = math.ceil(w / offset) * offset - w
    return F.pad(x, pad=(0, w_pad, 0, h_pad), mode=padding_mode)

def _prepare_val_batch_like_trainer(data, config, device):
    offset = int(config.train.get("val_resolution", 256))
    padding_mode = config.train.get("val_padding_mode", "reflect")
    lq = data["lq"].float()
    if lq.ndim == 4: lq = lq.unsqueeze(1)
    bsz, n_frames, chn, h, w = lq.shape
    lq_4d = lq.view(bsz * n_frames, chn, h, w)
    lq_4d = _prepare_spatial_val_4d(lq_4d, offset=offset, padding_mode=padding_mode)
    lq = lq_4d.view(bsz, n_frames, chn, lq_4d.shape[-2], lq_4d.shape[-1])
    gt = data.get("gt", None)
    if gt is not None:
        gt = gt.float()
        gt = _prepare_spatial_val_4d(gt, offset=offset, padding_mode=padding_mode)
    raw_num_sar = data.get("num_sar", None)
    if raw_num_sar is None: num_sar = torch.full((bsz,), n_frames, dtype=torch.long)
    elif isinstance(raw_num_sar, torch.Tensor): num_sar = raw_num_sar.view(-1).to(dtype=torch.long)
    else: num_sar = torch.full((bsz,), int(raw_num_sar), dtype=torch.long)
    if num_sar.numel() == 1 and bsz > 1: num_sar = num_sar.repeat(bsz)
    num_sar = num_sar.clamp(min=1, max=n_frames)
    lq = lq.to(device).float()
    if gt is not None: gt = gt.to(device).float()
    num_sar = num_sar.to(device=device, dtype=torch.long)
    return lq, gt, num_sar

def _maybe_load_ema_model(model, ckpt_path):
    ckpt_path = Path(ckpt_path)
    ema_path = ckpt_path.parent.parent / "ema_ckpts" / f"ema_{ckpt_path.name}"
    if not ema_path.exists(): return False
    ema_state = util_common.torch_load(str(ema_path), map_location="cpu")
    util_net.reload_model(model, ema_state)
    return True

# ========================================================
# 🚀 THUẬT TOÁN TỰ ĐỘNG LẤY SIZE VÀ TÍNH TỌA ĐỘ 4 GÓC CHUẨN ĐẾN TỪNG PIXEL
# ========================================================
def get_original_shape(base_name):
    """
    Tự động tìm kiếm ảnh gốc trong thư mục pairs để lấy chính xác Height và Width.
    """
    try:
        # Trích xuất tên thư mục gốc từ base_name (ví dụ: F-48-80-B-b_2021-01-13)
        match = re.match(r'^([A-Za-z0-9\-]+_\d{4}-\d{2}-\d{2})', base_name)
        if match:
            folder_name = match.group(1)
            # Tìm thư mục này bên trong PAIRS_BASE_DIR
            candidates = list(PAIRS_BASE_DIR.rglob(folder_name))
            if candidates:
                # Tìm bất kỳ file .tif nào bên trong để đọc thông số kích thước hình học
                tif_files = list(candidates[0].glob("*.tif"))
                if tif_files:
                    with rasterio.open(tif_files[0]) as src:
                        return src.height, src.width
    except Exception as e:
        print(f"⚠️ Cảnh báo lỗi đọc kích thước tự động cho {base_name}: {e}")
    
    # Fallback mặc định an toàn nếu hệ thống file bị trục trặc
    return 512, 512 

def get_4corner_crop_coords(H, W, P_size):
    """
    Trả về chính xác tọa độ 4 góc khớp 100% với file cắt patch của bạn.
    """
    return [
        {"y1": 0, "y2": P_size, "x1": 0, "x2": P_size}, # Patch 0: Trên - Trái
        {"y1": 0, "y2": P_size, "x1": max(0, W - P_size), "x2": max(0, W - P_size) + P_size}, # Patch 1: Trên - Phải
        {"y1": max(0, H - P_size), "y2": max(0, H - P_size) + P_size, "x1": 0, "x2": P_size}, # Patch 2: Dưới - Trái
        {"y1": max(0, H - P_size), "y2": max(0, H - P_size) + P_size, "x1": max(0, W - P_size), "x2": max(0, W - P_size) + P_size} # Patch 3: Dưới - Phải
    ]

def tensor_to_image(tensor):
    """KHÓA MÀU SẮC ĐỒNG BỘ: Sử dụng công thức toán học tĩnh (x + 1)/2"""
    x = tensor.detach().float().cpu()
    if x.ndim == 4: x = x[0]
    arr = (x.numpy().transpose(1, 2, 0) + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))

def save_full_res_image(tensor, save_path):
    img = tensor_to_image(tensor)
    img.save(save_path)

def create_visualization(sar_images, output_img, gt_img, base_name, save_path, metrics=None):
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
    except Exception: font = ImageFont.load_default()

    def resize_img(img): return img.resize((img_size, img_size), Image.LANCZOS)

    header_text = f"Result: {base_name}"
    if metrics: header_text += f" | PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f}"
    draw.text((padding, 5), header_text, fill=(0, 0, 128), font=font)
    
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
# MAIN LOOP
# ==========================================
def main():
    print("=" * 80)
    print(" Multi-SAR to Optical Inference (4 Corners Auto-Resizing Stitching)")
    print("=" * 80)

    from basicsr.data.multisar_dataset import MultiSARDataset
    from ldm.models.autoencoder import VQModelTorch
    from models.unet_multisar import UNetModelSwinMultiSAR

    config = load_config(CONFIG_PATH)
    val_y_channel = bool(config.train.get("val_y_channel", False))
    use_amp = bool(config.train.get("use_amp", False))

    print("[1/6] Loading UNet Model...")
    model = UNetModelSwinMultiSAR(**config.model.params)
    checkpoint = util_common.torch_load(CHECKPOINT_PATH, map_location="cpu")
    util_net.reload_model(model, checkpoint)
    _maybe_load_ema_model(model, CHECKPOINT_PATH)
    model = model.to(DEVICE).float().eval()

    print("[2/6] Loading VQGAN Autoencoder...")
    autoencoder = VQModelTorch(**config.autoencoder.params)
    ae_ckpt = util_common.torch_load(AUTOENCODER_PATH, map_location="cpu")
    util_net.reload_model(autoencoder, ae_ckpt)
    autoencoder = autoencoder.to(DEVICE).float().eval()

    print("[3/6] Creating Diffusion Engine...")
    diffusion = util_common.instantiate_from_config(config.diffusion)
    
    print("[4/6] Initializing LPIPS Model...")
    lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE).eval()

    print("[5/6] Loading Validation Dataset...")
    val_config = dict(config.data.val)
    val_config["phase"] = "val"
    dataset = MultiSARDataset(val_config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    print("[6/6] Running Inference & Stitching...")

    win_1d = torch.hann_window(PATCH_SIZE, periodic=False).to(DEVICE)
    win_2d = (win_1d.unsqueeze(1) * win_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0) 

    current_base = None
    canvas_data = None
    coords = None # Sẽ tính toán động theo từng ảnh
    
    total_psnr, total_ssim, total_lpips, total_images = 0.0, 0.0, 0.0, 0
    context = torch.cuda.amp.autocast if use_amp else nullcontext

    def flush_canvas(base_name, cv_data):
        nonlocal total_psnr, total_ssim, total_lpips, total_images
        if cv_data['count'] == 0: return
        
        weight_safe = torch.where(cv_data['weight'] > 0, cv_data['weight'], torch.ones_like(cv_data['weight']))
        pred_full = cv_data['pred'] / weight_safe
        gt_full = cv_data['gt'] / weight_safe
        sar_fulls = [s / weight_safe for s in cv_data['sars']]
        
        pred_full = pred_full.clamp(-1.0, 1.0)
        gt_full = gt_full.clamp(-1.0, 1.0)

        # Tính Metrics toàn cục trên kích thước thực
        current_lpips = lpips_fn(pred_full, gt_full).sum().item()
        current_psnr = util_image.batch_PSNR(pred_full * 0.5 + 0.5, gt_full * 0.5 + 0.5, ycbcr=val_y_channel)
        current_ssim = util_image.batch_SSIM(pred_full * 0.5 + 0.5, gt_full * 0.5 + 0.5, ycbcr=val_y_channel)

        total_psnr += current_psnr
        total_ssim += current_ssim
        total_lpips += current_lpips
        total_images += 1
        metrics = {"psnr": current_psnr, "ssim": current_ssim, "lpips": current_lpips}

        # Lưu ảnh Full độ nét cao sạch viền
        pred_save_path = os.path.join(OUTPUT_DIR, f"{base_name}_Stitched_Full_Thuong.png")
        save_full_res_image(pred_full, pred_save_path)
        
        # Lưu ảnh so sánh tổng quan
        vis_save_path = os.path.join(OUTPUT_DIR, f"{base_name}_Stitched_Full_SoSanh.png")
        sar_imgs = [tensor_to_image(s) for s in sar_fulls]
        create_visualization(
            sar_images=sar_imgs[:cv_data['n_valid']],
            output_img=tensor_to_image(pred_full), gt_img=tensor_to_image(gt_full),
            base_name=base_name, save_path=vis_save_path, metrics=metrics
        )

    # VÒNG LẶP DỮ LIỆU
    for data in tqdm(dataloader, desc="Inference"):
        with torch.no_grad():
            lq, gt, num_sar_dev = _prepare_val_batch_like_trainer(data, config, DEVICE)
            gt_path = data.get("gt_path", [""])[0]

            match = re.search(r'(.*)_patch_(\d+)\.\w+$', os.path.basename(gt_path))
            if not match: continue
            
            base_name = match.group(1)
            patch_idx = int(match.group(2)) 
            n_valid = int(num_sar_dev[0].item())

            # PHÁT HIỆN CHUYỂN SANG KHU VỰC ẢNH MỚI
            if base_name != current_base:
                if current_base is not None:
                    flush_canvas(current_base, canvas_data)
                    if total_images >= MAX_SAMPLES: break
                
                current_base = base_name
                
                # 🟢 TỰ ĐỘNG LẤY SIZE GỐC TỪ THƯ MỤC PAIRS
                ORIGINAL_H, ORIGINAL_W = get_original_shape(base_name)
                coords = get_4corner_crop_coords(ORIGINAL_H, ORIGINAL_W, PATCH_SIZE)
                
                # Khởi tạo ma trận Canvas động theo size thực vừa tìm được
                canvas_data = {
                    'pred': torch.zeros(1, 3, ORIGINAL_H, ORIGINAL_W, device=DEVICE),
                    'gt': torch.zeros(1, 3, ORIGINAL_H, ORIGINAL_W, device=DEVICE),
                    'weight': torch.zeros(1, 1, ORIGINAL_H, ORIGINAL_W, device=DEVICE),
                    'sars': [torch.zeros(1, 3, ORIGINAL_H, ORIGINAL_W, device=DEVICE) for _ in range(lq.shape[1])],
                    'count': 0,
                    'n_valid': n_valid
                }

            # INFERENCE PATCH
            model_kwargs = {"lq": lq, "num_sar": num_sar_dev}
            vv_band = lq[:, 0, 0:1, :, :]
            y = vv_band.expand(-1, 3, -1, -1)

            final_sample = None
            with context():
                for sample in diffusion.p_sample_loop_progressive(
                    y=y, model=model, first_stage_model=autoencoder, noise=None,
                    clip_denoised=False, model_kwargs=model_kwargs, device=DEVICE, progress=False
                ):
                    final_sample = sample
            output_patch = diffusion.decode_first_stage(final_sample["sample"], autoencoder).clamp(-1.0, 1.0)
            
            # Lưu Visual Patch lẻ (để phân tích)
            patch_vis_path = os.path.join(PATCH_VIS_DIR, f"{base_name}_patch_{patch_idx}_Vis.png")
            sar_patch_imgs = [tensor_to_image(lq[:, i, 0:1, :, :].expand(-1, 3, -1, -1)) for i in range(n_valid)]
            create_visualization(
                sar_images=sar_patch_imgs, output_img=tensor_to_image(output_patch), gt_img=tensor_to_image(gt),
                base_name=f"{base_name} (Patch {patch_idx})", save_path=patch_vis_path, metrics=None
            )

            # CỘNG DỒN VÀO CANVAS THEO LƯỚI TỌA ĐỘ ĐỘNG
            c = coords[patch_idx]
            y1, y2, x1, x2 = c['y1'], c['y2'], c['x1'], c['x2']
            
            canvas_data['pred'][:, :, y1:y2, x1:x2] += output_patch * win_2d
            canvas_data['gt'][:, :, y1:y2, x1:x2] += gt * win_2d
            canvas_data['weight'][:, :, y1:y2, x1:x2] += win_2d
            
            for i in range(lq.shape[1]):
                canvas_data['sars'][i][:, :, y1:y2, x1:x2] += (lq[:, i, 0:1, :, :].expand(-1, 3, -1, -1)) * win_2d

            canvas_data['count'] += 1

    # XẢ ẢNH CUỐI CÙNG
    if current_base is not None and total_images < MAX_SAMPLES:
        flush_canvas(current_base, canvas_data)

    print("\n" + "=" * 80)
    print(" 🎉 ĐÃ HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH KHÂU ẢNH TỰ ĐỘNG CO GIÃN!")
    print("=" * 80)

if __name__ == "__main__":
    main()