import os
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DIR_S1_S2 = Path("/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/test_S1_s2_1648000_visualizations")
DIR_OP_HLS = Path("/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/test_OP_HLS_1145000_visualizations")
PAIRS_BASE_DIR = Path("/mnt/hdd12tb/code/thucnd/Data_S1_S2_2_5_10m/pairs")

# Thư mục xuất kết quả so sánh đối chứng
OUTPUT_COMPARE_DIR = Path("/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/Cross_Model_Comparisons")
OUTPUT_COMPARE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. HÀM TRÍCH XUẤT ẢNH GT GỐC TỪ THƯ MỤC PAIRS
# ==========================================
def find_and_read_gt(base_name, target_h, target_w):
    """
    Tìm file ảnh quang học gốc trong thư mục pairs để làm Ground Truth chuẩn.
    """
    try:
        # Tách tên thư mục chứa ảnh (Ví dụ: F-48-80-B-b_2021-01-13)
        match = re.match(r'^([A-Za-z0-9\-]+_\d{4}-\d{2}-\d{2})', base_name)
        if match:
            folder_name = match.group(1)
            candidates = list(PAIRS_BASE_DIR.rglob(folder_name))
            if candidates:
                # Tìm file quang học gốc (HLS hoặc S2 tùy thuộc vào folder cấu trúc)
                # Ưu tiên các file không chứa ký tự hệ SAR như _S1_ hoặc _OP_
                for f in candidates[0].glob("*.tif"):
                    if "_S1_" not in f.name and "_OP_" not in f.name:
                        import rasterio
                        with rasterio.open(f) as src:
                            # Đọc 3 băng tầng RGB
                            img = src.read([1, 2, 3]) # Đọc dạng kênh chuẩn
                            # Chuyển đổi định dạng phù hợp OpenCV
                            img_hwc = np.transpose(img, (1, 2, 0)).astype(np.float32)
                            
                            # Chuẩn hóa dải màu về [0, 255] đồng bộ tĩnh
                            if img_hwc.max() > 255:
                                # Nếu ảnh ở hệ số phản xạ nguyên [0, 10000], clip về [0, 3000] rồi scale
                                img_hwc = np.clip(img_hwc, 0, 3000)
                                img_hwc = (img_hwc / 3000.0) * 255.0
                            elif img_hwc.max() <= 1.0:
                                img_hwc = img_hwc * 255.0
                                
                            img_uint8 = np.clip(img_hwc, 0, 255).astype(np.uint8)
                            # Chuyển đổi RGB sang BGR cho OpenCV
                            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                            # Resize về cùng kích thước với ảnh dự đoán để xếp hàng hàng ngang
                            return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        print(f"⚠️ Không thể đọc GT gốc cho {base_name}: {e}")
    return None

# ==========================================
# 3. TIẾN HÀNH GHÉP CẶP VÀ KHÂU ĐỐI CHỨNG
# ==========================================
def main():
    print("🚀 BẮT ĐẦU QUY TRÌNH SO SÁNH ĐỐI CHỨNG LIÊN MÔ HÌNH...")
    
    # Quét toàn bộ ảnh kết quả dạng "Thuong" của cả 2 mô hình
    s1_s2_files = list(DIR_S1_S2.glob("*_Stitched_Full_Thuong.png"))
    op_hls_files = list(DIR_OP_HLS.glob("*_Stitched_Full_Thuong.png"))
    
    print(f"-> Tìm thấy {len(s1_s2_files)} kết quả từ Mô hình 1 (S1->S2)")
    print(f"-> Tìm thấy {len(op_hls_files)} kết quả từ Mô hình 2 (OPERA->HLS)")

    # Xây dựng từ điển ánh xạ theo mã Khu vực (Prefix) để tìm cặp trùng nhau
    # Ví dụ mã: F-48-80-B-b
    op_hls_map = {}
    for f in op_hls_files:
        prefix_match = re.match(r'^([A-Za-z0-9\-]+)', f.name)
        if prefix_match:
            op_hls_map[prefix_match.group(1)] = f

    match_count = 0

    # Duyệt qua các file của Mô hình 1 để đối chiếu sang Mô hình 2
    for f_s1_s2 in tqdm(s1_s2_files, desc="Đang so sánh đối chứng"):
        prefix_match = re.match(r'^([A-Za-z0-9\-]+)', f_s1_s2.name)
        if not prefix_match:
            continue
            
        region_prefix = prefix_match.group(1)
        
        # Kiểm tra xem mô hình 2 có kết quả tại khu vực này không
        if region_prefix in op_hls_map:
            f_op_hls = op_hls_map[region_prefix]
            
            # Đọc hai ảnh dự đoán của 2 model
            img_s1_s2 = cv2.imread(str(f_s1_s2))
            img_op_hls = cv2.imread(str(f_op_hls))
            
            if img_s1_s2 is None or img_op_hls is None:
                continue
                
            H, W, C = img_s1_s2.shape
            
            # Đồng bộ kích thước ảnh mô hình 2 theo mô hình 1 để đưa vào ma trận phẳng
            if img_op_hls.shape != img_s1_s2.shape:
                img_op_hls = cv2.resize(img_op_hls, (W, H), interpolation=cv2.INTER_LANCZOS4)
                
            # Trích xuất ảnh Ground Truth thực tế chuẩn từ dữ liệu gốc
            img_gt = find_and_read_gt(f_op_hls.name, H, W)
            
            if img_gt is None:
                # Nếu không tìm thấy file .tif gốc, tạo một canvas trống chữ để không lỗi mạch code
                img_gt = np.zeros_like(img_s1_s2)
                cv2.putText(img_gt, "GT Not Found", (W//4, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- DÁN NHÃN TEXT LÊN ĐẦU MỖI ẢNH ĐỂ KHÔNG BỊ NHẦM LẪN ---
            # Tạo bản sao tránh ghi đè lên file gốc
            view_gt = img_gt.copy()
            view_m1 = img_s1_s2.copy()
            view_m2 = img_op_hls.copy()
            
            # Hàm vẽ nền đen chữ trắng cho dễ nhìn
            def draw_label(img, text):
                cv2.rectangle(img, (5, 5), (int(W * 0.65), 45), (0, 0, 0), -1)
                cv2.putText(img, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
            draw_label(view_gt, "1. GROUND TRUTH (Actual)")
            draw_label(view_m1, f"2. MODEL S1->S2 (Ckpt 1648k)")
            draw_label(view_m2, f"3. MODEL OPERA->HLS (Ckpt 1145k)")

            # --- KHÂU 3 ẢNH LẠI THÀNH MỘT DẢI NGANG (HORIZONTAL STITCH) ---
            comparison_matrix = np.hstack([view_gt, view_m1, view_m2])
            
            # Lưu file đối chứng toàn cảnh
            out_name = f"{region_prefix}_Cross_Model_Comparison.png"
            cv2.imwrite(str(OUTPUT_COMPARE_DIR / out_name), comparison_matrix)
            
            match_count += 1

    print("\n" + "="*60)
    print(" 🎉 QUA TRÌNH TRÍCH XUẤT VÀ SO SÁNH HOÀN TẤT!")
    print(f" -> Đã tạo thành công {match_count} dải ảnh đối chứng liên mô hình.")
    print(f" -> Vị trí lưu kết quả: {OUTPUT_COMPARE_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()