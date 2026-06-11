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

# --- THÊM THƯ VIỆN ĐỂ TÍNH ĐỘ ĐO ---
import lpips

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import util_common, util_image, util_net

# ============ CONFIG ============
CHECKPOINT_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/experiments_10m/multisar_opera_hls_l2/2026-04-10-15-45-36/ckpts/model_1860000.pth"
CONFIG_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/configs/multisar_opera_hls_l2.yaml"
OUTPUT_DIR = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/experiments_10m/multisar_opera_hls_l2/2026-04-10-15-45-36/inference_results_1860000"
AUTOENCODER_PATH = "/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/weights/autoencoder_vq_f4.pth"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_config(config_path):
    return OmegaConf.load(config_path)

def _prepare_spatial_val_4d(x, offset, padding_mode):
    """Match trainer.prepare_data(..., phase='val') for 4D tensors [B, C, H, W]."""
    h, w = x.shape[-2:]
    if h > offset and w > offset:
        h_end = int((h // offset) * offset)
        w_end = int((w // offset) * offset)
        return x[:, :, :h_end, :w_end]

    h_pad = math.ceil(h / offset) * offset - h
    w_pad = math.ceil(w / offset) * offset - w
    return F.pad(x, pad=(0, w_pad, 0, h_pad), mode=padding_mode)

def _prepare_val_batch_like_trainer(data, config, device):
    """Prepare lq/gt/num_sar using the same val logic as trainer."""
    offset = int(config.train.get("val_resolution", 256))
    padding_mode = config.train.get("val_padding_mode", "reflect")

    lq = data["lq"].float()
    if lq.ndim == 4:
        lq = lq.unsqueeze(1)
    if lq.ndim != 5:
        raise ValueError(f"Expected lq with 4D or 5D shape, got {tuple(lq.shape)}")

    bsz, n_frames, chn, h, w = lq.shape
    lq_4d = lq.view(bsz * n_frames, chn, h, w)
    lq_4d = _prepare_spatial_val_4d(lq_4d, offset=offset, padding_mode=padding_mode)
    lq = lq_4d.view(bsz, n_frames, chn, lq_4d.shape[-2], lq_4d.shape[-1])

    gt = data.get("gt", None)
    if gt is not None:
        gt = gt.float()
        if gt.ndim != 4:
            raise ValueError(f"Expected gt with 4D shape, got {tuple(gt.shape)}")
        gt = _prepare_spatial_val_4d(gt, offset=offset, padding_mode=padding_mode)

    raw_num_sar = data.get("num_sar", None)
    if raw_num_sar is None:
        num_sar = torch.full((bsz,), n_frames, dtype=torch.long)
    elif isinstance(raw_num_sar, torch.Tensor):
        num_sar = raw_num_sar.view(-1).to(dtype=torch.long)
    elif isinstance(raw_num_sar, (list, tuple)):
        num_sar = torch.tensor(list(raw_num_sar), dtype=torch.long)
    else:
        num_sar = torch.full((bsz,), int(raw_num_sar), dtype=torch.long)

    if num_sar.numel() == 1 and bsz > 1:
        num_sar = num_sar.repeat(bsz)
    if num_sar.numel() != bsz:
        num_sar = torch.full((bsz,), n_frames, dtype=torch.long)
    num_sar = num_sar.clamp(min=1, max=n_frames)

    lq = lq.to(device).float()
    if gt is not None:
        gt = gt.to(device).float()
    num_sar = num_sar.to(device=device, dtype=torch.long)
    return lq, gt, num_sar

def _maybe_load_ema_model(model, ckpt_path):
    """Load EMA weights if file exists at experiments/.../ema_ckpts/ema_<ckpt>.pth."""
    ckpt_path = Path(ckpt_path)
    ema_path = ckpt_path.parent.parent / "ema_ckpts" / f"ema_{ckpt_path.name}"
    if not ema_path.exists():
        print(f"[WARN] EMA checkpoint not found: {ema_path}")
        return False

    print(f"[EMA] Loading EMA weights from: {ema_path}")
    ema_state = util_common.torch_load(str(ema_path), map_location="cpu")
    util_net.reload_model(model, ema_state)
    return True

def _uncollate_lq_paths(lq_paths, batch_idx=0):
    out = []
    if not isinstance(lq_paths, (list, tuple)): return out
    for item in lq_paths:
        if isinstance(item, (list, tuple)):
            if len(item) > batch_idx: out.append(str(item[batch_idx]))
        elif isinstance(item, str): out.append(item)
    return out

# ==========================================
# VISUALIZATION
# ==========================================
def tensor_to_image(tensor):
    """Hàm dùng chung cho cả SAR và Quang học vì dữ liệu đã là PNG chuẩn"""
    x = tensor.detach().float().cpu()
    if x.ndim == 4: x = x[0]
    
    # 1. Ép từ dải [-1, 1] của model về dải [0, 1]
    arr = (x.numpy().transpose(1, 2, 0) + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    
    # 2. Chuyển thẳng ra ảnh 8-bit [0, 255]
    return Image.fromarray((arr * 255.0).astype(np.uint8))

# --- CẬP NHẬT: Thêm tham số metrics để in lên ảnh ---
def create_visualization(sar_images, output_img, gt_img, sar_paths, gt_path, save_path, total_sar_count=None, metrics=None):
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

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_header = font

    def resize_img(img): return img.resize((img_size, img_size), Image.LANCZOS)
    def get_short_name(path):
        if not path: return ""
        basename = os.path.basename(str(path))
        match = re.search(r"OP_(\d{4}-\d{2}-\d{2})", basename)
        if match: return f"OP_{match.group(1)}"
        return basename[:14] + "..." + basename[-12:] if len(basename) > 30 else basename

    actual_count = total_sar_count if total_sar_count is not None else n_sar
    
    # --- CẬP NHẬT: Ghi Metrics lên Header của ảnh ---
    header_text = f"Multi-SAR -> Optical | {actual_count} SAR images"
    if metrics:
        header_text += f" | PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f}"
        
    draw.text((padding, 5), header_text, fill=(0, 0, 128), font=font_header)

    y_offset = header_height
    for i in range(max_display):
        row, col = i // cols, i % cols
        x = padding + col * (img_size + padding)
        y = y_offset + row * (img_size + text_height + padding)
        canvas.paste(resize_img(sar_images[i]), (x, y))
        sar_label = f"SAR{i + 1}: {get_short_name(sar_paths[i]) if i < len(sar_paths) else ''}"
        draw.text((x, y + img_size + 2), sar_label, fill=(0, 0, 0), font=font)

    result_row_y = y_offset + sar_rows * (img_size + text_height + padding)

    output_x = padding
    canvas.paste(resize_img(output_img), (output_x, result_row_y))
    draw.text((output_x, result_row_y + img_size + 2), "OUTPUT (Predicted)", fill=(0, 128, 0), font=font)

    gt_x = padding + 2 * (img_size + padding)
    canvas.paste(resize_img(gt_img), (gt_x, result_row_y))
    draw.text((gt_x, result_row_y + img_size + 2), f"GT: {get_short_name(gt_path)}", fill=(0, 0, 255), font=font)

    arrow_x = padding + img_size + padding + img_size // 2
    arrow_y = result_row_y + img_size // 2
    draw.text((arrow_x - 10, arrow_y - 10), "->", fill=(100, 100, 100), font=font_header)

    canvas.save(save_path)

def main():
    print("=" * 80)
    print(" Multi-SAR to Optical Inference & Metrics Calculation")
    print("=" * 80)

    from basicsr.data.multisar_dataset import MultiSARDataset
    from ldm.models.autoencoder import VQModelTorch
    from models.unet_multisar import UNetModelSwinMultiSAR

    config = load_config(CONFIG_PATH)
    config.model.params.use_fp16 = False
    val_y_channel = bool(config.train.get("val_y_channel", False))
    use_amp = bool(config.train.get("use_amp", False))
    use_ema_val = bool(config.train.get("use_ema_val", False))

    print("[1/6] Loading UNet Model...")
    model = UNetModelSwinMultiSAR(**config.model.params)
    checkpoint = util_common.torch_load(CHECKPOINT_PATH, map_location="cpu")
    util_net.reload_model(model, checkpoint)

    if use_ema_val:
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
    # LPIPS yêu cầu dữ liệu ảnh dạng Tensor dải [-1, 1]
    lpips_net = config.lpips.net if hasattr(config, "lpips") and hasattr(config.lpips, "net") else "vgg"
    lpips_fn = lpips.LPIPS(net=lpips_net).to(DEVICE).eval()

    print("[5/6] Loading Validation Dataset...")
    val_config = dict(config.data.val)
    val_config["phase"] = "val"
    
    dataset = MultiSARDataset(val_config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    print("[6/6] Running Inference...")

    count = 0  # số ảnh visualization đã lưu (1 ảnh/batch)
    total_images = 0
    
    # --- CẬP NHẬT: Biến lưu trữ tổng các metrics để tính trung bình ---
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    context = torch.cuda.amp.autocast if use_amp else nullcontext

    for data in tqdm(dataloader, desc="Inference"):
        if count >= MAX_SAMPLES: break

        with torch.no_grad():
            lq, gt, num_sar_dev = _prepare_val_batch_like_trainer(data, config, DEVICE)
            lq_paths = data.get("lq_paths", [])
            gt_path = data.get("gt_path", "")

            n_valid = int(num_sar_dev[0].item())

            model_kwargs = {
                "lq": lq,
                "num_sar": num_sar_dev,
            }

            # Match trainer: y uses VV channel of first SAR, expanded to 3 channels.
            y = lq[:, 0, :, :, :]

            final_sample = None
            with context():
                for sample in diffusion.p_sample_loop_progressive(
                    y=y,
                    model=model,
                    first_stage_model=autoencoder,
                    noise=None,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=DEVICE,
                    progress=False,
                ):
                    final_sample = sample

            if final_sample is not None and "sample" in final_sample:
                output = diffusion.decode_first_stage(final_sample["sample"], autoencoder).clamp(-1.0, 1.0)
            else:
                output = gt

            # --- Metrics theo cùng chuẩn với validation trainer ---
            batch_size = output.shape[0]
            current_lpips = lpips_fn(output, gt).sum().item()
            current_psnr = util_image.batch_PSNR(
                output * 0.5 + 0.5,
                gt * 0.5 + 0.5,
                ycbcr=val_y_channel,
            )
            current_ssim = util_image.batch_SSIM(
                output * 0.5 + 0.5,
                gt * 0.5 + 0.5,
                ycbcr=val_y_channel,
            )

            # Chuyển Tensors về ảnh dạng PIL Image [0, 255]
            output_img = tensor_to_image(output[0:1])
            gt_img = tensor_to_image(gt[0:1])
            
            # Cộng dồn
            total_psnr += current_psnr
            total_ssim += current_ssim
            total_lpips += current_lpips
            total_images += batch_size
            
            # Lưu dict để hiển thị
            current_metrics = {
                "psnr": current_psnr / max(1, batch_size),
                "ssim": current_ssim / max(1, batch_size),
                "lpips": current_lpips / max(1, batch_size),
            }

            lq_paths_one = _uncollate_lq_paths(lq_paths, batch_idx=0)
            gt_path_one = gt_path[0] if isinstance(gt_path, (list, tuple)) else gt_path

            sar_images = []
            valid_paths = []
            max_display = min(n_valid, 6, lq.shape[1])
            for j in range(max_display):
                sar_images.append(tensor_to_image(lq[0:1, j]))
                valid_paths.append(lq_paths_one[j] if j < len(lq_paths_one) else "")

            if gt_path_one:
                gt_basename = os.path.basename(gt_path_one)
                gt_name_without_ext = os.path.splitext(gt_basename)[0]
                save_filename = f"{gt_name_without_ext}.png"
            else:
                # Fallback nếu không có đường dẫn GT
                save_filename = f"result_{count:04d}.png"
                
            save_path = os.path.join(OUTPUT_DIR, save_filename)

            create_visualization(
                sar_images=sar_images,
                output_img=output_img,
                gt_img=gt_img,
                sar_paths=valid_paths,
                gt_path=gt_path_one,
                save_path=save_path,
                total_sar_count=n_valid,
                metrics=current_metrics # Truyền metrics vào vẽ
            )
            count += 1

    # --- CẬP NHẬT: Tổng hợp và In báo cáo kết quả Metrics ---
    if total_images > 0:
        avg_psnr = total_psnr / total_images
        avg_ssim = total_ssim / total_images
        avg_lpips = total_lpips / total_images
        
        print("\n" + "=" * 80)
        print(" TỔNG HỢP CHỈ SỐ ĐÁNH GIÁ (AVERAGE METRICS):")
        print(f"   - Tổng số ảnh đã chạy: {total_images}")
        print(f"   - Average PSNR : {avg_psnr:.4f} (Trainer-style)")
        print(f"   - Average SSIM : {avg_ssim:.4f} (Càng gần 1 càng tốt)")
        print(f"   - Average LPIPS: {avg_lpips:.4f} (Càng thấp càng tốt)")
        print(f"   - use_ema_val   : {use_ema_val}")
        print(f"   - val_y_channel : {val_y_channel}")
        print("=" * 80)

    print(f"  {OUTPUT_DIR}")

if __name__ == "__main__":
    main()