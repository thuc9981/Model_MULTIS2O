import os

file_path = 'trainer.py'

# 1. Tìm đoạn code khởi tạo ngữ cảnh AMP trong validation (Hiện tại chưa có)
# Chúng ta sẽ tìm dòng 'mean_psnr = mean_lpips = 0' để chèn code khởi tạo context ngay sau đó
target_anchor = "mean_psnr = mean_lpips = 0"

insert_code = """            
            # FIX: Thêm context cho AMP (Mixed Precision)
            context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext"""

# 2. Tìm vòng lặp inference để bọc nó trong context
loop_start = """for ii, data in enumerate(self.dataloaders[phase]):"""

# 3. Tìm đoạn gọi mô hình để thêm 'with context():'
# Đoạn này nằm trong vòng lặp p_sample_loop_progressive
# Chúng ta sẽ thay thế cả cụm xử lý vòng lặp diffusion
old_diffusion_loop = """                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).cuda()
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
                        model_kwargs=model_kwargs,
                        device=f"cuda:{self.rank}",
                        progress=False,
                        ):"""

new_diffusion_loop = """                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).cuda()
                
                # FIX: Bọc vòng lặp inference trong autocast context
                with context():
                    for sample in self.base_diffusion.p_sample_loop_progressive(
                            y=im_lq,
                            model=self.ema_model if self.configs.train.use_ema_val else self.model,
                            first_stage_model=self.autoencoder,
                            noise=None,
                            clip_denoised=True if self.autoencoder is None else False,
                            model_kwargs=model_kwargs,
                            device=f"cuda:{self.rank}",
                            progress=False,
                            ):"""

#   THỰC HIỆN SỬA  
if not os.path.exists(file_path):
    print(f"Lỗi: Không tìm thấy file {file_path}")
    exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

changed = False

# Thêm định nghĩa context
if insert_code.strip() not in content:
    if target_anchor in content:
        print("-> Đang thêm định nghĩa context AMP...")
        content = content.replace(target_anchor, target_anchor + "\n" + insert_code)
        changed = True
    else:
        print("Cảnh báo: Không tìm thấy vị trí neo 'mean_psnr = ...'")

# Bọc vòng lặp trong with context():
# Lưu ý: Do việc thay thế cả khối code lớn có thể gặp vấn đề về khoảng trắng (indentation),
# nên ta sẽ dùng logic replace chính xác chuỗi text cũ.
if old_diffusion_loop in content:
    print("-> Đang bọc vòng lặp validation trong 'with context():'...")
    content = content.replace(old_diffusion_loop, new_diffusion_loop)
    
    # Do Python yêu cầu thụt đầu dòng (indent), ta cần thụt dòng cho nội dung bên trong vòng lặp for
    # Đoạn code bên trong vòng lặp bắt đầu bằng 'sample_decode = {}'
    # Ta sẽ tìm và thụt dòng thủ công cho đoạn này
    
    # Tìm đoạn code xử lý bên trong loop để thụt vào thêm 4 spaces
    inner_loop_old = """                    sample_decode = {}
                    if num_iters in indices:
                        for key, value in sample.items():
                            if key in ['sample', ]:
                                sample_decode[key] = self.base_diffusion.decode_first_stage(
                                        value,
                                        self.autoencoder,
                                        ).clamp(-1.0, 1.0)
                        im_sr_progress = sample_decode['sample']
                        if num_iters + 1 == 1:
                            im_sr_all = im_sr_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                    num_iters += 1
                    tt -= 1"""
    
    inner_loop_new = """                        sample_decode = {}
                        if num_iters in indices:
                            for key, value in sample.items():
                                if key in ['sample', ]:
                                    sample_decode[key] = self.base_diffusion.decode_first_stage(
                                            value,
                                            self.autoencoder,
                                            ).clamp(-1.0, 1.0)
                            im_sr_progress = sample_decode['sample']
                            if num_iters + 1 == 1:
                                im_sr_all = im_sr_progress
                            else:
                                im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                        num_iters += 1
                        tt -= 1"""
                        
    if inner_loop_old in content:
        content = content.replace(inner_loop_old, inner_loop_new)
        changed = True
    else:
        print("Lỗi: Không tìm thấy nội dung bên trong vòng lặp để thụt dòng. Hãy kiểm tra lại file trainer.py")
        changed = False # Hủy thay đổi nếu không thụt dòng được để tránh lỗi syntax

else:
    # Thử tìm phiên bản đã sửa (để tránh sửa 2 lần)
    if "with context():" in content and "p_sample_loop_progressive" in content:
        print("-> File có vẻ đã được sửa rồi.")
    else:
        print("Lỗi: Không tìm thấy đoạn code vòng lặp diffusion cũ.")

if changed:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("\n[OK] Đã sửa file trainer.py thành công!")
else:
    print("\n[INFO] Không có thay đổi nào được thực hiện.")