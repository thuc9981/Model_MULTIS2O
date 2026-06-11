import cv2
import numpy as np
import os
import glob
import re

# ================= CẤU HÌNH LẠI =================
# Đường dẫn chứa ảnh (Dựa trên thông tin bạn cung cấp)
INPUT_DIR = "/mnt/fwgpu/hdd12tb/code/thucnd/ResShift-journal/experiments/sar_dualband_vv_vh/2026-02-07-16-16-38/images/val"

# Thư mục xuất kết quả
OUTPUT_DIR = "/mnt/fwgpu/hdd12tb/code/thucnd/check_results/check_results_val"

# Cấu hình Batch size
BATCH_SIZE = 4
# Số cột mặc định khi lưu grid d(torchvision.make_grid default nrow=8)
NROW_DEFAULT = 8
# Padding mặc định của torchvision.make_grid
GRID_PADDING = 2
# Chiều cao dòng tiêu đề
LABEL_HEIGHT = 28
# ================================================

def process_visuals():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f" Đã tạo thư mục kết quả: {OUTPUT_DIR}")

    # 1. Tìm các file lq (mẫu: lq-5881.png) hoặc dual-band (lq_VV-5881.png, lq_VH-5881.png)
    lq_files = sorted(glob.glob(os.path.join(INPUT_DIR, "lq-*.png")))
    lq_vv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "lq_VV-*.png")))
    lq_vh_files = sorted(glob.glob(os.path.join(INPUT_DIR, "lq_VH-*.png")))
    
    print(f" Đang quét thư mục: {INPUT_DIR}")
    if len(lq_vv_files) > 0 and len(lq_vh_files) > 0:
        print(f" Tìm thấy {len(lq_vv_files)} file LQ_VV và {len(lq_vh_files)} file LQ_VH.")
        lq_mode = "dual"
        lq_iter_files = lq_vv_files
    else:
        print(f" Tìm thấy {len(lq_files)} file LQ.")
        lq_mode = "single"
        lq_iter_files = lq_files

    if len(lq_iter_files) == 0:
        print(" Vẫn không tìm thấy file. Bạn hãy kiểm tra lại đường dẫn INPUT_DIR xem có đúng 100% không.")
        return

    for lq_path in lq_iter_files:
        # Lấy tên file: lq-5881.png
        filename = os.path.basename(lq_path)
        
        # Tách lấy số step (5881) bằng cách xóa 'lq-' và '.png'
        # Cách này an toàn hơn split
        if lq_mode == "dual":
            step_name = filename.replace("lq_VV-", "").replace(".png", "")
        else:
            step_name = filename.replace("lq-", "").replace(".png", "")
        
        # Tạo đường dẫn tương ứng cho GT và Progress
        gt_path = os.path.join(INPUT_DIR, f"gt-{step_name}.png")
        prog_path = os.path.join(INPUT_DIR, f"progress-{step_name}.png")
        lq_vv_path = os.path.join(INPUT_DIR, f"lq_VV-{step_name}.png")
        lq_vh_path = os.path.join(INPUT_DIR, f"lq_VH-{step_name}.png")

        # Kiểm tra sự tồn tại
        if not os.path.exists(gt_path):
            print(f"  Không thấy file GT: {gt_path}")
            continue
        if not os.path.exists(prog_path):
            print(f"  Không thấy file Progress: {prog_path}")
            continue
        if lq_mode == "dual":
            if not os.path.exists(lq_vv_path):
                print(f"  Không thấy file LQ_VV: {lq_vv_path}")
                continue
            if not os.path.exists(lq_vh_path):
                print(f"  Không thấy file LQ_VH: {lq_vh_path}")
                continue

        print(f" Đang ghép ảnh bước: {step_name} ...")

        # 2. Đọc ảnh
        img_lq_grid = cv2.imread(lq_path)
        img_gt_grid = cv2.imread(gt_path)
        img_prog_grid = cv2.imread(prog_path)
        img_lq_vv_grid = cv2.imread(lq_vv_path) if lq_mode == "dual" else None
        img_lq_vh_grid = cv2.imread(lq_vh_path) if lq_mode == "dual" else None

        if img_lq_grid is None:
            print(f" Lỗi đọc ảnh LQ: {lq_path}")
            continue

        # 3. Cắt nhỏ Grid LQ và GT (tự động theo nrow + padding)
        h, w, _ = img_lq_grid.shape
        ncol_lq = min(NROW_DEFAULT, BATCH_SIZE)
        nrow_lq = int(np.ceil(BATCH_SIZE / ncol_lq))
        tile_w = (w - GRID_PADDING * (ncol_lq - 1)) // ncol_lq
        tile_h = (h - GRID_PADDING * (nrow_lq - 1)) // nrow_lq
        
        lq_list = []
        lq_vv_list = []
        lq_vh_list = []
        gt_list = []

        # Cắt theo thứ tự trái -> phải, trên -> xuống
        for r in range(nrow_lq):
            for c in range(ncol_lq):
                if (r * ncol_lq + c) >= BATCH_SIZE:
                    continue
                y1 = r * (tile_h + GRID_PADDING)
                y2 = y1 + tile_h
                x1 = c * (tile_w + GRID_PADDING)
                x2 = x1 + tile_w
                lq_list.append(img_lq_grid[y1:y2, x1:x2])
                gt_list.append(img_gt_grid[y1:y2, x1:x2])
                if lq_mode == "dual":
                    lq_vv_list.append(img_lq_vv_grid[y1:y2, x1:x2])
                    lq_vh_list.append(img_lq_vh_grid[y1:y2, x1:x2])

        # 4. Xử lý ảnh Progress (grid nrow = số bước indices)
        ph, pw, _ = img_prog_grid.shape

        # Suy ra số cột progress từ kích thước tile
        ncol_prog = max(1, int(round((pw + GRID_PADDING) / (tile_w + GRID_PADDING))))
        nrow_prog = max(1, int(round((ph + GRID_PADDING) / (tile_h + GRID_PADDING))))

        # Lấy 2 cột cuối cùng (Kết quả dự đoán 2 ảnh gần cuối và cuối)
        p_tile_w = tile_w
        if ncol_prog >= 2:
            pred_cols = [ncol_prog - 2, ncol_prog - 1]
        else:
            pred_cols = [ncol_prog - 1]

        prog_list = []
        for i in range(BATCH_SIZE):
            row = i  # mỗi row tương ứng một sample
            preds_for_row = []
            for col in pred_cols:
                y1 = row * (tile_h + GRID_PADDING)
                y2 = y1 + tile_h
                x1 = col * (tile_w + GRID_PADDING)
                x2 = x1 + p_tile_w

                crop = img_prog_grid[y1:y2, x1:x2]

                # Resize về kích thước LQ nếu bị lệch pixel
                if crop.shape[:2] != (tile_h, tile_w):
                    crop = cv2.resize(crop, (tile_w, tile_h))

                preds_for_row.append(crop)
            prog_list.append(preds_for_row)

        # 5. Ghép lại:
        #   Single: [LQ] | [Progress Cuối] | [GT]
        #   Dual:   [LQ_VV] | [LQ_VH] | [Progress Cuối] | [GT]
        final_rows = []
        # Giới hạn số lượng ảnh để file không quá dài (ví dụ chỉ lấy 8 ảnh đầu)
        limit_show = min(BATCH_SIZE, 8) 
        
        for i in range(limit_show):
            # Thêm vạch ngăn cách màu trắng
            sep = 255 * np.ones((tile_h, 10, 3), dtype=np.uint8)
            
            # Ghép: LQ - Progress - GT
            try:
                if lq_mode == "dual":
                    if len(prog_list[i]) == 2:
                        combined = np.hstack([
                            lq_vv_list[i], sep, lq_vh_list[i], sep,
                            prog_list[i][0], sep, prog_list[i][1], sep,
                            gt_list[i]
                        ])
                    else:
                        combined = np.hstack([
                            lq_vv_list[i], sep, lq_vh_list[i], sep,
                            prog_list[i][0], sep, gt_list[i]
                        ])
                else:
                    if len(prog_list[i]) == 2:
                        combined = np.hstack([
                            lq_list[i], sep,
                            prog_list[i][0], sep, prog_list[i][1], sep,
                            gt_list[i]
                        ])
                    else:
                        combined = np.hstack([
                            lq_list[i], sep, prog_list[i][0], sep, gt_list[i]
                        ])
                final_rows.append(combined)
            except Exception as e:
                print(f"Lỗi ghép dòng {i}: {e}")
                continue

        if len(final_rows) > 0:
            # Tạo dòng tiêu đề để dễ quan sát
            if lq_mode == "dual":
                if len(pred_cols) == 2:
                    col_labels = ["VV", "VH", "Pred-1", "Pred-2", "GT"]
                    col_widths = [tile_w, tile_w, tile_w, tile_w, tile_w]
                else:
                    col_labels = ["VV", "VH", "Pred", "GT"]
                    col_widths = [tile_w, tile_w, tile_w, tile_w]
            else:
                if len(pred_cols) == 2:
                    col_labels = ["LQ", "Pred-1", "Pred-2", "GT"]
                    col_widths = [tile_w, tile_w, tile_w, tile_w]
                else:
                    col_labels = ["LQ", "Pred", "GT"]
                    col_widths = [tile_w, tile_w, tile_w]

            sep_w = 10
            total_w = sum(col_widths) + sep_w * (len(col_widths) - 1)
            label_row = 255 * np.ones((LABEL_HEIGHT, total_w, 3), dtype=np.uint8)
            x = 0
            for idx, label in enumerate(col_labels):
                cv2.putText(
                    label_row,
                    label,
                    (x + 6, LABEL_HEIGHT - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                x += col_widths[idx] + sep_w

            # Chèn dòng tiêu đề lên đầu
            full_image = np.vstack([label_row] + final_rows)
            save_path = os.path.join(OUTPUT_DIR, f"compare_{step_name}.jpg")
            cv2.imwrite(save_path, full_image)
            print(f" Đã xong! File tại: {save_path}")

if __name__ == "__main__":
    process_visuals()