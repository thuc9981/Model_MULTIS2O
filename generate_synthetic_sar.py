import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import argparse

def add_speckle_noise(img, L=1):
    """
    Thêm nhiễu đốm (Speckle noise) tuân theo phân phối Gamma.
    Mô phỏng theo cách của SAR-DDPM.
    
    Args:
        img: Ảnh đầu vào (0-1 float).
        L: Number of looks (L càng nhỏ nhiễu càng mạnh). L=1 là chuẩn cho Single-look SAR.
    """
    h, w = img.shape
    
    # Sinh nhiễu từ phân phối Gamma
    # Mean = 1, Variance = 1/L
    noise = np.random.gamma(shape=L, scale=1.0/L, size=(h, w))
    
    # Áp dụng mô hình nhiễu nhân: Y = X * N
    noisy_img = img * noise
    
    return noisy_img

def process_dataset(input_dir, output_dir, L=1):
    # Tạo cấu trúc thư mục output
    save_gt = os.path.join(output_dir, 'GT') # Ảnh sạch (Gray)
    save_lq = os.path.join(output_dir, 'LQ') # Ảnh nhiễu (Speckled)
    
    os.makedirs(save_gt, exist_ok=True)
    os.makedirs(save_lq, exist_ok=True)
    
    # Lấy danh sách ảnh
    # Hỗ trợ các đuôi ảnh phổ biến
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
    img_list = []
    for ext in extensions:
        img_list.extend(glob.glob(os.path.join(input_dir, ext)))
        # Thử tìm trong subfolder nếu có (như train/val cũ)
        img_list.extend(glob.glob(os.path.join(input_dir, '*', ext)))
        img_list.extend(glob.glob(os.path.join(input_dir, '*', '*', ext)))

    print(f"Tìm thấy {len(img_list)} ảnh trong {input_dir}")
    print(f"Đang xử lý với Number of Looks L={L} (L=1 là nhiễu mạnh nhất)...")

    for path in tqdm(img_list):
        # Lấy tên file
        img_name = os.path.basename(path)
        basename, _ = os.path.splitext(img_name)
        
        # 1. Đọc ảnh và chuẩn hóa về [0, 1]
        img = cv2.imread(path)
        if img is None:
            continue
            
        # 2. Chuyển sang Grayscale (Mô phỏng ảnh SAR chỉ có 1 kênh cường độ)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float32) / 255.0
        
        # 3. Tạo ảnh nhiễu (LQ)
        img_speckled = add_speckle_noise(img_gray, L=L)
        
        # Clip giá trị về [0, 1] để tránh lỗi khi lưu
        img_speckled = np.clip(img_speckled, 0, 1)
        
        # 4. Chuẩn bị lưu ảnh
        # ResShift yêu cầu input 3 kênh (RGB). 
        # Ta sẽ nhân bản kênh Gray thành 3 kênh để đánh lừa mạng (R=G=B).
        
        gt_save = (img_gray * 255).astype(np.uint8)
        gt_save_3c = cv2.merge([gt_save, gt_save, gt_save])
        
        lq_save = (img_speckled * 255).astype(np.uint8)
        lq_save_3c = cv2.merge([lq_save, lq_save, lq_save])
        
        # 5. Lưu ra file (giữ nguyên tên để Dataloader bắt cặp)
        # Lưu định dạng PNG để không bị nén mất mát thêm
        cv2.imwrite(os.path.join(save_gt, f"{basename}.png"), gt_save_3c)
        cv2.imwrite(os.path.join(save_lq, f"{basename}.png"), lq_save_3c)

    print(f"Xử lý hoàn tất! Dữ liệu đã lưu tại: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Đường dẫn thư mục chứa ảnh Optical gốc')
    parser.add_argument('--output', type=str, required=True, help='Đường dẫn thư mục lưu dataset đã tạo')
    parser.add_argument('--L', type=float, default=1.0, help='Number of looks. 1.0 = Nhiễu mạnh nhất (Chuẩn SAR), tăng lên thì nhiễu giảm.')
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.L)