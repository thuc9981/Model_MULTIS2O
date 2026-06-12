# SAR-to-Optical ResShift (C-DiffSET)

## Khung khuếch tán có điều khiển độ tin cậy cho bài toán khử nhiễu SAR và chuyển đổi SAR sang ảnh quang học

---

## Tóm tắt

Kho lưu trữ này giới thiệu **C-DiffSET**, một phiên bản mở rộng của kiến trúc ResShift (Resolution Shift), được thiết kế chuyên biệt cho:
- Khử nhiễu ảnh Radar khẩu độ tổng hợp (SAR Despeckling)
- Chuyển đổi miền SAR sang ảnh quang học (SAR-to-Optical Translation)
Phương pháp đề xuất tích hợp mô hình khuếch tán (Diffusion Model) với cơ chế tối ưu hóa có trọng số theo bản đồ độ tin cậy (Confidence-weighted Optimization), nhằm:
- Giảm hiện tượng “ảo giác” (hallucination) trong mô hình sinh ảnh
- Bảo toàn cấu trúc hình học
- Duy trì tính nhất quán phổ màu và độ tương phản
Hệ thống được thiết kế cho dữ liệu SAR hai băng (VV, VH) từ Sentinel-1 và huấn luyện theo cấu trúc dữ liệu ghép cặp (paired supervision).
---
##  Tổng quan hệ thống

C-DiffSET mở rộng ResShift gốc với các thành phần thích nghi miền Viễn thám:
- Hỗ trợ đầu vào SAR hai băng (VV/VH)
- Hàm mất mát khuếch tán có điều khiển độ tin cậy (C-Diff)
- Ràng buộc gradient có cổng (Gated Gradient)
- Điều chuẩn màu sắc và độ tương phản
Mục tiêu chính là tăng độ chính xác tái tạo cấu trúc trong khi hạn chế sinh vân giả tại các vùng nhiễu hoặc độ tin cậy thấp (ví dụ: mặt nước, thảm thực vật).

## Cài đặt môi trường

### Yêu cầu

- Python ≥ 3.8
- PyTorch ≥ 2.0
- Khuyến nghị sử dụng GPU CUDA

---

### Thiết lập bằng Conda
```bash
conda env create -f environment.yml
conda activate resshift_sar
```
### Data 
Cấu trúc dữ liệu huấn luyện dạng ghép cặp:
```
Data_SAR/
├── train/
│   ├── S1_SAR/   # Ảnh SAR
│   └── S2_Optical/   # Ảnh quang học (Ground Truth)
├── val/
│   ├── S1_SAR/
│   └── S2_Optical/
├── test/
│   ├── S1_SAR/
│   └── S2_Optical

```
Cập nhật đường dẫn trong file cấu hình config
Ví dụ :
```
dataroot_lq: /duong/dan/train/A
dataroot_gt: /duong/dan/train/B
```
###Tham số cấu hình (Điều chỉnh trong config) : 

| Tham số      | Mô tả                                                   |
| ------------ | ------------------------------------------------------- |
| `steps`      | Số bước khuếch tán (phải khớp checkpoint nếu fine-tune) |
| `loss_coef`  | Trọng số `[C-Diff, LPIPS, Gradient]`             |
| `cdiff_beta` | Độ nhạy của cơ chế độ tin cậy                           |

ví dụ :
```
loss_coef: [2.0, 0.5, 0.2, 0.5]
cdiff_beta: 1.0
```

### Train model :
Huấn luyện trên 1 GPU : 
``` python main.py --cfg_path config_path ```

Huấn luyện đa GPU : 
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --nnodes=1 \
  main.py \
  --cfg_path config_path \
  --save_dir save_dir/sar_journal_2gpu
```
Huấn luyện tiếp tục checkpoint old : 
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --nnodes=1 \
  main.py --cfg_path config_path \
  --save_dir save_dir/sar_journal_2gpu --resume path_checkpoint
```
Link checkpoint : (https://drive.google.com/drive/folders/1l4OpMHi8gSguY2cTW_AhXXWY4hcC8An6?usp=sharing) 
với :
| Model                   | Mô tả                                                   |
| ----------------------- | ------------------------------------------------------- |
| `model_165000.pth`      | Model phiên bản gốc ( No confidence and Gradient loss)  |
| `model_190000.pth`      | Model phiên bản ứng dụng Confidence and Gradient loss   |
| `model_255000.pth`      | Model cải tiến cân bằng tham số                         |

### Test - inference
With model Opera to HLS : /mnt/hdd1tb/SAR2Optical/Model_MultiS2O/inference_multisar_HLS.py (Cập nhật các đường dẫn trong file  )

```
python ./inference_multisar_HLS.py
```
With model S1 to S2 : /mnt/hdd1tb/SAR2Optical/Model_MultiS2O/inference_multisar_S2.py (Cập nhật các đường dẫn trong file  )
```
python ./inference_multisar_S2.py
```

### Chỉ số đánh giá 
# Model-MultiS2O
# Model-MultiS2O
# Model_MULTIS2O
# Model_MULTIS2O
