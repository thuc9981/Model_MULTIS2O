#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 13:04:06

import os, sys, math, time, random, datetime, functools
import lpips
from models.basic_ops import mean_flat
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext
import wandb 
from datapipe.datasets import create_dataset

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

    def setup_dist(self):
        if "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])

            torch.cuda.set_device(self.local_rank)

            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=datetime.timedelta(seconds=3600),
                )

            self.num_gpus = self.world_size
            print(f"[DDP Trainer] world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}")
        else:
            # Single GPU / no torchrun
            self.rank = 0
            self.local_rank = 0
            self.num_gpus = 1



    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            resume_path = Path(self.configs.resume)
            if not resume_path.is_absolute():
                resume_path = (Path.cwd() / resume_path).resolve()
            if resume_path.exists():
                # If checkpoint is under a ckpts/ folder, use its parent as save_dir
                if resume_path.parent.name == "ckpts":
                    save_dir = resume_path.parent.parent
                else:
                    save_dir = resume_path.parent
            else:
                # Fallback: use save_dir with timestamp if resume path is invalid
                save_dir = Path(self.configs.save_dir) / datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                if not save_dir.exists() and self.rank == 0:
                    save_dir.mkdir(parents=True)
            project_id = save_dir.name
            # Update resume path to resolved path for later loading
            self.configs.resume = str(resume_path)
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume, "Log file exists but resume is False!"
            
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        if self.rank == 0 and hasattr(self.configs, 'wandb'):
            self.logger.info(f"Initializing WandB project: {self.configs.wandb.project}")
            wandb.init(
                project=self.configs.wandb.project,
                name=self.configs.wandb.name,
                config=OmegaConf.to_container(self.configs, resolve=True),
                mode=self.configs.wandb.get("mode", "online")
            )

        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))
            
    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()
        if self.rank == 0 and wandb.run is not None:
            wandb.finish()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                
                clean_key = key.replace('_orig_mod.', '')
                
                possible_keys = [
                    key,                          
                    clean_key,                   
                    'module.' + clean_key,       
                    'module.' + key,           
                    key.replace('module.', '')    
                ]

                found = False
                for k_try in possible_keys:
                    if k_try in ckpt:
                        ema_state[key] = deepcopy(ckpt[k_try].detach().data)
                        found = True
                        break
                
                if not found:
                    pass 

        if self.configs.resume:
            resume_path = Path(self.configs.resume)
            if not resume_path.is_absolute():
                resume_path = (Path.cwd() / resume_path).resolve()

            # If not found, try save_dir/ckpts/model_xxx.pth
            if not resume_path.is_file():
                ckpt_candidate = Path(self.configs.save_dir) / "ckpts" / Path(self.configs.resume).name
                if not ckpt_candidate.is_absolute():
                    ckpt_candidate = (Path.cwd() / ckpt_candidate).resolve()
                if ckpt_candidate.is_file():
                    resume_path = ckpt_candidate

            if not resume_path.is_file():
                search_root = Path(self.configs.save_dir)
                if not search_root.is_absolute():
                    search_root = (Path.cwd() / search_root).resolve()
                matches = list(search_root.rglob(Path(self.configs.resume).name))
                matches = [m for m in matches if m.is_file() and m.name.endswith(".pth")]
                if len(matches) == 1:
                    resume_path = matches[0]
                elif len(matches) > 1:
                    # Prefer checkpoints inside ckpts/ folders
                    ckpt_matches = [m for m in matches if m.parent.name == "ckpts"]
                    resume_path = ckpt_matches[0] if ckpt_matches else matches[0]

            self.configs.resume = str(resume_path)
            if not (self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)):
                if self.rank == 0:
                    self.logger.warning(f"Resume checkpoint not found: {self.configs.resume}. Starting from scratch.")
                return

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            
            if hasattr(self, 'autoencoder') and 'autoencoder' in ckpt:
                util_net.reload_model(self.autoencoder, ckpt['autoencoder'])
                if self.rank == 0:
                    self.logger.info("=> Phục hồi trọng số Autoencoder đã được tune từ Checkpoint thành công!")
            
            torch.cuda.empty_cache()

            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)

            if self.rank == 0:
                self.log_step = ckpt.get('log_step', {'train': 0, 'val': 0}) 
                self.log_step_img = ckpt.get('log_step_img', {'train': 0, 'val': 0})

            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                if ema_ckpt_path.is_file():
                    self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                    ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                    _load_ema_state(self.ema_state, ema_ckpt)
                else:
                    self.logger.warning(
                        f"EMA checkpoint not found at {str(ema_ckpt_path)}. "
                        "Continuing without EMA state."
                    )
            torch.cuda.empty_cache()

            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")

            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

        # amp settings
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        #   local_rank đúng chuẩn torchrun  
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)

        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model = model.to(self.local_rank)

        #   Load checkpoint  
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.local_rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)

        #   torch.compile (optional)  
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")

        #   Wrap DDP đúng chuẩn  
        if self.num_gpus > 1:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                static_graph=False
            )
        else:
            self.model = model

        #   EMA  
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).to(self.local_rank)
            self.ema_state = OrderedDict(
                {key: deepcopy(value.data) for key, value in self.model.state_dict().items()}
            )
            self.ema_ignore_keys = [
                x for x in self.ema_state.keys()
                if ('running_' in x or 'num_batches_tracked' in x)
            ]

        #   model info  
        self.print_model_info()


    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data):
        dtype = torch.float16 if self.configs.train.use_amp else torch.float32
        device = torch.device(f"cuda:{self.local_rank}")
        
        output = {}
        for key, value in data.items():
            if hasattr(value, 'to'):
                output[key] = value.to(device, dtype=dtype)
            else:
                output[key] = value
        return output

    def validation(self):
        pass

    def train(self):
        self.init_logger()       # setup logger: self.logger

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']))

            # training phase
            self.training_step(data)

            # validation phase
            if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0:
                self.save_ckpt()

            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard
        self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            # 1. LƯU MODEL CHÍNH
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            ckpt = {
                    'iters_start': self.current_iters,
                    'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                    'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                    'state_dict': self.model.state_dict(),
                    }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
                
            # ĐOẠN THÊM VÀO ĐỂ CỨU AUTOENCODER
            if hasattr(self, 'autoencoder') and self.autoencoder is not None:
                ckpt['autoencoder'] = self.autoencoder.state_dict()
                
            torch.save(ckpt, ckpt_path)
            
            # ==========================================================
            # 2. ĐOẠN THÊM MỚI: LƯU TRỌNG SỐ EMA (EMA MODEL)
            # ==========================================================
            if hasattr(self, 'ema_state') and hasattr(self, 'ema_ckpt_dir'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                
                # EMA state bản chất đã là một OrderedDict chứa params, nên lưu trực tiếp
                torch.save(self.ema_state, ema_ckpt_path)
                self.logger.info(f"Saved EMA model to {ema_ckpt_path.name}")
    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    @torch.no_grad()
    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        # Đã nới lỏng assert để pass qua nếu chỉ dùng wandb
        assert self.tf_logging or self.local_logging or (wandb.run is not None)
        
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path)
            
        if self.tf_logging:
            self.writer.add_image(
                f"{phase}/{tag}",
                im_tensor,
                self.log_step_img[phase],
                )
                
        if self.rank == 0 and wandb.run is not None:
            # Lấy Global Step để ảnh đồng bộ với biểu đồ Loss
            wandb_step = getattr(self, 'current_iters', self.log_step_img[phase])
            
            # Chuyển Tensor (C, H, W) dải [0,1] thành Numpy (H, W, C) dải [0,255] uint8
            im_np_wandb = im_tensor.cpu().permute(1, 2, 0).numpy()
            im_np_wandb = (np.clip(im_np_wandb, 0.0, 1.0) * 255.0).astype(np.uint8)
            
            # Đẩy lên bảng điều khiển WandB (Phân nhóm theo train/val)
            wandb.log({f"{phase}/images/{tag}": wandb.Image(im_np_wandb)}, step=wandb_step)
        # ==========================================================

        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag='Loss', phase='train', add_global_step=True):
        def _to_float(value):
            if isinstance(value, torch.Tensor):
                return value.mean().item() if value.numel() > 1 else value.item()
            if isinstance(value, (list, tuple)):
                return float(np.mean(value))
            return float(value)

        # Use current_iters as wandb step (must be monotonically increasing globally)
        # TensorBoard can use per-phase step counters since it supports separate steps per tag
        wandb_step = getattr(self, 'current_iters', self.log_step[phase])

        if isinstance(metrics, dict):
            safe_metrics = {k: _to_float(v) for k, v in metrics.items()}
            self.writer.add_scalars(tag, safe_metrics, self.log_step[phase])
            
            if self.rank == 0 and wandb.run is not None:
                wandb_data = {f"{phase}/{tag}/{k}": v for k, v in safe_metrics.items()}
                wandb.log(wandb_data, step=wandb_step)
        else:
            val = _to_float(metrics)
            self.writer.add_scalar(tag, val, self.log_step[phase])
            
            if self.rank == 0 and wandb.run is not None:
                wandb.log({f"{phase}/{tag}": val}, step=wandb_step)

        if add_global_step:
            self.log_step[phase] += 1

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def load_model(self, model, ckpt_path=None, tag='model', strict=True):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if strict:
            util_net.reload_model(model, ckpt)
        else:
            model.load_state_dict(ckpt, strict=False)
        if self.rank == 0:
            self.logger.info('Loaded Done')

class TrainerDifIR(TrainerBase):
    def setup_optimizaton(self):
        super().setup_optimizaton()
        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )

    def build_model(self):
        super().build_model()
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])

        # autoencoder
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            if self.configs.autoencoder.get("tune_decoder", False):
                self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
                if self.rank == 0:
                    num_params = 0
                    for key, value in autoencoder.named_parameters():
                        if 'decoder' in key or 'post_quant_conv' in key:
                            num_params += value.numel()
                        else:
                            value.requires_grad = False
                    self.logger.info(f'Finetuning Decoder module: {num_params/10**6:.2f}M...')
            else:
                self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
                self.freeze_model(autoencoder)
                autoencoder.eval()
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        if self.configs.autoencoder.params.lora_tune_decoder or self.configs.autoencoder.tune_decoder:
            self.freeze_model(self.model)

        # LPIPS metric
        if hasattr(self.configs, 'lpips'):
            lpips_net = self.configs.lpips.net
        else:
            lpips_net = 'vgg'
        if self.rank == 0:
            self.logger.info(f"Loading LIIPS Metric: {lpips_net}...")
        lpips_loss = lpips.LPIPS(net=lpips_net).to(f"cuda:{self.rank}")
        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        self.lpips_loss = lpips_loss

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b*10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        if realesrgan is None:
            realesrgan = self.configs.data.get(phase, dict).type == 'realesrgan'
        if self.configs.train.get('disable_degradation', False):
            realesrgan = False
        if realesrgan and phase == 'train':
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            if not hasattr(self, 'use_sharpener'):
                self.use_sharpener = USMSharp().cuda()

            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]
            if isinstance(self.configs.degradation.sf, int):
                sf = self.configs.degradation.sf
            else:
                assert len(self.configs.degradation.sf) == 2
                sf = random.uniform(*self.configs.degradation.sf)

            if self.configs.degradation.use_sharp:
                im_gt = self.use_sharpener(im_gt)

            #        The first degradation process        #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.configs.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.configs.degradation['gray_noise_prob']
            if random.random() < self.configs.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            #        The second degradation process        #
            if random.random() < self.configs.degradation['second_order_prob']:
                # blur
                if random.random() < self.configs.degradation['second_blur_prob']:
                    out = filter2D(out, kernel2)
                # random resize
                updown_type = random.choices(
                        ['up', 'down', 'keep'],
                        self.configs.degradation['resize_prob2'],
                        )[0]
                if updown_type == 'up':
                    scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
                elif updown_type == 'down':
                    scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                        mode=mode,
                        )
                # add noise
                gray_noise_prob = self.configs.degradation['gray_noise_prob2']
                if random.random() < self.configs.degradation['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.configs.degradation['noise_range2'],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                        )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.configs.degradation['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                        )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)

            # resize back
            if self.configs.degradation.resize_back:
                out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')
                temp_sf = self.configs.degradation['sf']
            else:
                temp_sf = self.configs.degradation['sf']

            # clamp and round
            im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.configs.degradation['gt_size']
            im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, temp_sf)
            im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
            self.lq, self.gt, flag_nan = replace_nan_in_batch(im_lq, im_gt)
            if flag_nan:
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return {'lq':self.lq, 'gt':self.gt}
        elif phase == 'val':
            offset = self.configs.train.get('val_resolution', 256)
            for key, value in data.items():
                if not hasattr(value, 'shape'): continue
                h, w = value.shape[2:]
                if h > offset and w > offset:
                    h_end = int((h // offset) * offset)
                    w_end = int((w // offset) * offset)
                    data[key] = value[:, :, :h_end, :w_end]
                else:
                    h_pad = math.ceil(h / offset) * offset - h
                    w_pad = math.ceil(w / offset) * offset - w
                    padding_mode = self.configs.train.get('val_padding_mode', 'reflect')
                    data[key] = F.pad(value, pad=(0, w_pad, 0, h_pad), mode=padding_mode)
            
            #   FIX: Kiểm tra type trước khi đẩy lên GPU  
            output = {}
            for key, value in data.items():
                if hasattr(value, 'cuda'):
                    output[key] = value.cuda().to(dtype=dtype)
                else:
                    output[key] = value
            return output
        else:
            #   FIX: Kiểm tra type trước khi đẩy lên GPU  
            output = {}
            for key, value in data.items():
                if hasattr(value, 'cuda'):
                    output[key] = value.cuda().to(dtype=dtype)
                else:
                    output[key] = value
            return output

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        # Check if we have dual-band SAR input (2-channel LQ)
        is_dualband = (data['lq'].shape[1] == 2)

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items() if hasattr(value, 'shape')} # Filter tensors
            if 'mask' in data and not hasattr(data['mask'], 'shape'):
                 # Handle special case if needed, generally micro_data assumes tensors
                 pass

            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"cuda:{self.rank}",
                    )
            latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
            if 'autoencoder' in self.configs:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]
            noise = torch.randn(
                    size= (micro_data['gt'].shape[0], noise_chn,) + (latent_resolution, ) * 2,
                    device=micro_data['gt'].device,
                    )
            if self.configs.model.params.cond_lq:
                # model_kwargs['lq'] → fed directly to UNet (can be 2-ch for dual-band)
                model_kwargs = {'lq':micro_data['lq'],}
                if 'mask' in micro_data:
                    model_kwargs['mask'] = micro_data['mask']
            else:
                model_kwargs = None

            # For diffusion y: autoencoder needs 3-channel input.
            # If dual-band (2ch), expand VV (channel 0) to 3 channels.
            if is_dualband:
                vv_band = micro_data['lq'][:, 0:1, :, :]  # [B, 1, H, W]
                lq_for_diffusion = vv_band.expand(-1, 3, -1, -1)  # [B, 3, H, W]
            else:
                lq_for_diffusion = micro_data['lq']

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                lq_for_diffusion,
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
                cdiff_beta=self.configs.train.get('cdiff_beta', None),
            )
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)

            # make logging
            if last_batch:
                self.log_step_train(losses, tt, micro_data, z_t, z0_pred.detach())

        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # grad zero
        self.model.zero_grad()

        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]

            #   Reset loss accumulator  
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key: torch.zeros(len(record_steps), dtype=torch.float64)
                                for key in loss.keys()}
                self.loss_count = torch.zeros(len(record_steps), dtype=torch.float64)

            #   Accumulate loss  
            for jj in range(len(record_steps)):
                index = record_steps[jj] - 1
                mask = (tt == index).float()
                for key, value in loss.items():
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            #   Log scalar loss  
            if self.current_iters % self.configs.train.log_freq[0] == 0:
                self.loss_count[self.loss_count == 0] = 1e-6
                for key in self.loss_mean:
                    self.loss_mean[key] /= self.loss_count

                log_str = f"Train: {self.current_iters:06d}/{self.configs.train.iterations:06d}, "
                for jj, tval in enumerate(record_steps):
                    log_str += f"t({tval}):{self.loss_mean['loss'][jj]:.2e}/{self.loss_mean['mse'][jj]:.2e}, "
                # Log confidence mean if available
                if 'confidence_mean' in self.loss_mean:
                    avg_conf = self.loss_mean['confidence_mean'].mean().item()
                    log_str += f"conf:{avg_conf:.3f}, "
                log_str += f"lr:{self.optimizer.param_groups[0]['lr']:.2e}"
                self.logger.info(log_str)

                #  Log scalar (0-dim)
                scalar_loss = {f"t{record_steps[i]}_{k}": float(self.loss_mean[k][i])
                            for k in self.loss_mean for i in range(len(record_steps))}
                self.logging_metric(scalar_loss, tag='Loss', phase=phase, add_global_step=True)

            #   Log images  
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                # Handle 2-channel LQ: expand VV to 3ch for visualization
                lq_vis = batch['lq']
                if lq_vis.shape[1] == 2:
                    # Log VV (channel 0) as grayscale image (expanded to 3ch)
                    vv_vis = lq_vis[:, 0:1, :, :].expand(-1, 3, -1, -1)
                    self.logging_image(vv_vis, tag='lq_VV', phase=phase, add_global_step=False)
                    # Log VH (channel 1) as grayscale image
                    vh_vis = lq_vis[:, 1:2, :, :].expand(-1, 3, -1, -1)
                    self.logging_image(vh_vis, tag='lq_VH', phase=phase, add_global_step=False)
                else:
                    self.logging_image(lq_vis, tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x0_pred = self.base_diffusion.decode_first_stage(z0_pred, self.autoencoder)
                self.logging_image(x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

            #   Time logging  
            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()

            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elapsed = self.toc - self.tic
                self.logger.info(f"Elapsed time: {elapsed:.2f}s")
                self.logger.info("=" * 100)


    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            indices = list(range(self.base_diffusion.num_timesteps))
            if not (self.base_diffusion.num_timesteps-1) in indices:
                indices.append(self.base_diffusion.num_timesteps-1)
            batch_size = self.configs.train.batch[1]
            max_val_images = self.configs.train.get('val_max_images', None)
            if max_val_images is not None:
                max_val_images = int(max_val_images)
                num_iters_epoch = math.ceil(min(len(self.datasets[phase]), max_val_images) / batch_size)
            else:
                num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            mean_mse = 0
            mean_val_loss = 0
            loss_coef = self.configs.train.get('loss_coef', [1.0, 1.0])
            mean_vh_consistency = 0  # VH cross-consistency metric
            val_count = 0
            has_gt = False
            
            context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
            for ii, data in enumerate(self.dataloaders[phase]):
                if max_val_images is not None and val_count >= max_val_images:
                    break
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                else:
                    im_lq = data['lq']

                # Detect input type first
                is_multisar = (len(im_lq.shape) == 5)
                is_dualband = False  # Will be set properly below

                num_iters = 0
                if self.configs.model.params.cond_lq:
                    model_kwargs = {'lq':data['lq'],}
                    # Add num_sar for multi-SAR models
                    if 'num_sar' in data:
                        model_kwargs['num_sar'] = data['num_sar']
                    if 'mask' in data:
                        model_kwargs['mask'] = data['mask']
                else:
                    model_kwargs = None
                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).cuda()

                # Handle multi-SAR input [B, N, C, H, W]
                is_multisar = (len(im_lq.shape) == 5)
                if is_multisar:
                    # Multi-SAR: [B, N, C, H, W] → take first SAR, VV channel
                    vv_band = im_lq[:, 0, 0:1, :, :]  # [B, 1, H, W]
                    lq_for_diffusion = vv_band.expand(-1, 3, -1, -1)  # [B, 3, H, W]
                    is_dualband = False
                else:
                    is_dualband = (im_lq.shape[1] == 2)
                    if is_dualband:
                        vv_band = im_lq[:, 0:1, :, :]
                        lq_for_diffusion = vv_band.expand(-1, 3, -1, -1)
                    else:
                        lq_for_diffusion = im_lq
                
                with context():
                    for sample in self.base_diffusion.p_sample_loop_progressive(
                            y=lq_for_diffusion,
                            model=self.ema_model if self.configs.train.use_ema_val else self.model,
                            first_stage_model=self.autoencoder,
                            noise=None,
                            clip_denoised=True if self.autoencoder is None else False,
                            model_kwargs=model_kwargs,
                            device=f"cuda:{self.rank}",
                            progress=False,
                            ):
                        sample_decode = {}
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
                        tt -= 1

                if 'gt' in data:
                    has_gt = True
                    batch_size = im_gt.shape[0]
                    
                    # =======================================================
                    # ĐÃ SỬA: CHUẨN HÓA DẢI GIÁ TRỊ VỀ [0, 1] ĐỂ TÍNH METRIC
                    # =======================================================
                    pred_01 = sample_decode['sample'] * 0.5 + 0.5
                    gt_01 = im_gt * 0.5 + 0.5
                    
                    # 1. Tính MSE chuẩn (trên dải 0-1)
                    batch_mse_01 = F.mse_loss(pred_01, gt_01, reduction='mean')
                    mean_mse += batch_mse_01.item() * batch_size

                    # 2. Tính PSNR (ĐÃ VÁ LỖI NHÂN BATCH_SIZE)
                    current_batch_psnr = util_image.batch_PSNR(
                            pred_01,
                            gt_01,
                            ycbcr=self.configs.train.val_y_channel,
                            )
                    mean_psnr += current_batch_psnr * batch_size  # <--- SỬA LỖI CHÍ MẠNG Ở ĐÂY

                    # 3. Tính LPIPS (Giữ nguyên vì hàm này trả về sum)
                    batch_lpips_sum = self.lpips_loss(
                            sample_decode['sample'],
                            im_gt,
                        ).sum().item()
                    mean_lpips += batch_lpips_sum
                    batch_lpips_mean = batch_lpips_sum / max(1, batch_size)

                    # 4. G-Grad loss (pixel-space gradient)
                    grad_x_pred = pred_01[:, :, :, :-1] - pred_01[:, :, :, 1:]
                    grad_y_pred = pred_01[:, :, :-1, :] - pred_01[:, :, 1:, :]
                    grad_x_gt   = gt_01[:, :, :, :-1]   - gt_01[:, :, :, 1:]
                    grad_y_gt   = gt_01[:, :, :-1, :]    - gt_01[:, :, 1:, :]
                    batch_grad = ((grad_x_pred - grad_x_gt) ** 2).mean() + ((grad_y_pred - grad_y_gt) ** 2).mean()

                    # 5. ValLoss
                    lambda_grad = loss_coef[2] if len(loss_coef) > 2 else 0.0
                    batch_val_loss = (loss_coef[0] * batch_mse_01.item()
                                      + loss_coef[1] * batch_lpips_mean
                                      + lambda_grad * batch_grad.item())
                    mean_val_loss += batch_val_loss * batch_size

                    # 6. VH cross-consistency validation
                    if is_dualband:
                        pred_gray = pred_01[:, 0:1, :, :]  # Đã ở dải 0-1
                        vh_gray = im_lq[:, 1:2, :, :] * 0.5 + 0.5 
                        vh_gray = vh_gray.to(dtype=pred_gray.dtype)
                        
                        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=pred_gray.dtype, device=pred_gray.device).view(1,1,3,3)
                        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=pred_gray.dtype, device=pred_gray.device).view(1,1,3,3)
                        
                        pred_edge = torch.sqrt(F.conv2d(pred_gray, sobel_x, padding=1)**2 + F.conv2d(pred_gray, sobel_y, padding=1)**2 + 1e-6)
                        vh_edge = torch.sqrt(F.conv2d(vh_gray, sobel_x, padding=1)**2 + F.conv2d(vh_gray, sobel_y, padding=1)**2 + 1e-6)
                        
                        pred_edge_flat = pred_edge.view(pred_edge.shape[0], -1)
                        vh_edge_flat = vh_edge.view(vh_edge.shape[0], -1)
                        pred_norm = pred_edge_flat - pred_edge_flat.mean(dim=1, keepdim=True)
                        vh_norm = vh_edge_flat - vh_edge_flat.mean(dim=1, keepdim=True)
                        ncc = (pred_norm * vh_norm).sum(dim=1) / (pred_norm.norm(dim=1) * vh_norm.norm(dim=1) + 1e-8)
                        mean_vh_consistency += ncc.sum().item()

                val_count += im_lq.shape[0]

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')
                    # For visualization, use 3-channel version
                    vis_channels = 3  # output is always 3ch
                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=vis_channels)
                    self.logging_image(
                            im_sr_all,
                            tag='progress',
                            phase=phase,
                            add_global_step=False,
                            nrow=len(indices),
                            )
                    if 'gt' in data:
                        self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                    # Handle visualization based on input type
                    if is_multisar:
                        # Multi-SAR: [B, N, C, H, W] → show first SAR image
                        lq_vis = im_lq[:, 0, :, :, :]  # [B, C, H, W] - first SAR
                        self.logging_image(lq_vis, tag='lq_SAR', phase=phase, add_global_step=True)
                    elif is_dualband:
                        vv_vis = im_lq[:, 0:1, :, :].expand(-1, 3, -1, -1)
                        self.logging_image(vv_vis, tag='lq_VV', phase=phase, add_global_step=False)
                        vh_vis = im_lq[:, 1:2, :, :].expand(-1, 3, -1, -1)
                        self.logging_image(vh_vis, tag='lq_VH', phase=phase, add_global_step=True)
                    else:
                        self.logging_image(im_lq, tag='lq', phase=phase, add_global_step=True)

            if has_gt and val_count > 0:
                mean_psnr /= val_count
                mean_lpips /= val_count
                mean_mse /= val_count
                mean_val_loss /= val_count
                log_msg = f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}'
                self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)
                self.logging_metric(mean_lpips, tag='LPIPS', phase=phase, add_global_step=False)
                self.logging_metric(mean_mse, tag='MSE', phase=phase, add_global_step=False)
                self.logging_metric(mean_val_loss, tag='ValLoss', phase=phase, add_global_step=False)
                # Log VH cross-consistency if dual-band
                if is_dualband and val_count > 0:
                    mean_vh_consistency /= val_count
                    log_msg += f', VH_Consistency={mean_vh_consistency:.4f}'
                    self.logging_metric(mean_vh_consistency, tag='VH_Consistency', phase=phase, add_global_step=False)
                log_msg += '...'
                self.logger.info(log_msg)
                # Final global step increment
                self.logging_metric(0.0, tag='_val_step', phase=phase, add_global_step=True)

            self.logger.info("="*100)

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

class TrainerDifIRLPIPS(TrainerDifIR):
    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        loss_coef = self.configs.train.get('loss_coef')
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        # diffusion loss
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            x0_pred = self.base_diffusion.decode_first_stage(
                    z0_pred,
                    self.autoencoder,
                    ) # f16
            self.current_x0_pred = x0_pred.detach()

            # ---- λ2 · LPIPS loss ----
            losses["lpips"] = self.lpips_loss(
                    x0_pred,
                    micro_data['gt'],
                    ).to(z0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)
            losses["lpips"] *= loss_coef[1]

            # ---- λ1 · C-Diff / MSE loss ----
            if loss_coef[0] > 0:    # calculate mse in latent space
                losses["mse"] *= loss_coef[0]
            else:                   # calculate mse in pixel space
                assert len(loss_coef) > 2 and loss_coef[2] > 0
                losses["mse"] = mean_flat((x0_pred - micro_data['gt']) ** 2)
                losses["mse"] *= loss_coef[2]

            # ---- λ3 · G-Grad (gradient) loss ----
            lambda_grad = loss_coef[2] if (loss_coef[0] > 0 and len(loss_coef) > 2) else 0.0
            if "grad" in losses and lambda_grad > 0:
                losses["grad"] *= lambda_grad
            else:
                # If diffusion did not produce grad term, set it to zero
                losses["grad"] = torch.zeros_like(losses["mse"])

            assert losses["mse"].shape == losses["lpips"].shape
            # L_total = λ1·L_CDiff + λ2·L_LPIPS + λ3·L_GGrad
            if flag_nan:
                losses["loss"] = losses["mse"] + losses["grad"]
            else:
                losses["loss"] = losses["mse"] + losses["lpips"] + losses["grad"]
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    assert value.shape == mask.shape
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, MSE/LPIPS/Grad: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    grad_val = self.loss_mean['grad'][jj].item() if 'grad' in self.loss_mean else 0.0
                    log_str += 't({:d}):{:.1e}/{:.1e}/{:.1e}, '.format(
                            current_record,
                            self.loss_mean['mse'][jj].item(),
                            self.loss_mean['lpips'][jj].item(),
                            grad_val,
                            )
                # Log confidence mean if available
                if 'confidence_mean' in self.loss_mean:
                    avg_conf = self.loss_mean['confidence_mean'].mean().item()
                    log_str += 'conf:{:.3f}, '.format(avg_conf)
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                        )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                self.logging_image(self.current_x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                if not hasattr(self, 'tic'):
                    self.tic = time.time()
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)


class TrainerDifIRLPIPSMultiSAR(TrainerDifIRLPIPS):
    """
    Trainer for Multi-SAR to Optical translation using Temporal Transformer.

    This trainer handles multi-SAR input where each sample consists of N SAR images
    that need to be fused into a single feature map to condition the diffusion process.

    Key differences from TrainerDifIRLPIPS:
    1. Input LQ has shape [B, N, C, H, W] instead of [B, C, H, W]
    2. Passes num_sar to model for masking padded frames
    3. Custom visualization for multi-SAR inputs
    """

    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        """Prepare multi-SAR data for training/validation."""
        # Multi-SAR doesn't use RealESRGAN degradation
        if phase == 'train':
            # Move tensors to GPU
            output = {}
            for key, value in data.items():
                if hasattr(value, 'cuda'):
                    output[key] = value.cuda().to(dtype=dtype)
                elif isinstance(value, (int, float)):
                    output[key] = value
                elif hasattr(value, '__len__'):  # handle lists like num_sar
                    if isinstance(value, torch.Tensor):
                        output[key] = value.cuda()
                    else:
                        output[key] = value
                else:
                    output[key] = value
            return output

        elif phase == 'val':
            offset = self.configs.train.get('val_resolution', 256)
            output = {}

            for key, value in data.items():
                if not hasattr(value, 'shape'):
                    if isinstance(value, torch.Tensor):
                        output[key] = value.cuda()
                    else:
                        output[key] = value
                    continue

                # Handle multi-SAR input: [B, N, C, H, W]
                if key == 'lq' and len(value.shape) == 5:
                    B, N, C, h, w = value.shape
                    if h > offset and w > offset:
                        h_end = int((h // offset) * offset)
                        w_end = int((w // offset) * offset)
                        value = value[:, :, :, :h_end, :w_end]
                    else:
                        h_pad = math.ceil(h / offset) * offset - h
                        w_pad = math.ceil(w / offset) * offset - w
                        padding_mode = self.configs.train.get('val_padding_mode', 'reflect')
                        # Pad each SAR image
                        value = F.pad(value.view(B*N, C, h, w), pad=(0, w_pad, 0, h_pad), mode=padding_mode)
                        value = value.view(B, N, C, h + h_pad, w + w_pad)
                    output[key] = value.cuda().to(dtype=dtype)
                # Handle GT: [B, C, H, W]
                elif key == 'gt':
                    h, w = value.shape[2:]
                    if h > offset and w > offset:
                        h_end = int((h // offset) * offset)
                        w_end = int((w // offset) * offset)
                        value = value[:, :, :h_end, :w_end]
                    else:
                        h_pad = math.ceil(h / offset) * offset - h
                        w_pad = math.ceil(w / offset) * offset - w
                        padding_mode = self.configs.train.get('val_padding_mode', 'reflect')
                        value = F.pad(value, pad=(0, w_pad, 0, h_pad), mode=padding_mode)
                    output[key] = value.cuda().to(dtype=dtype)
                else:
                    if hasattr(value, 'cuda'):
                        output[key] = value.cuda().to(dtype=dtype)
                    else:
                        output[key] = value

            return output
        else:
            # Test phase
            output = {}
            for key, value in data.items():
                if hasattr(value, 'cuda'):
                    output[key] = value.cuda().to(dtype=dtype)
                else:
                    output[key] = value
            return output

    def training_step(self, data):
        """Training step for multi-SAR input."""
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        for jj in range(0, current_batchsize, micro_batchsize):
            # Slice micro-batch
            micro_data = {}
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    micro_data[key] = value[jj:jj+micro_batchsize]
                elif isinstance(value, torch.Tensor):
                    micro_data[key] = value[jj:jj+micro_batchsize]
                elif isinstance(value, (list, tuple)):
                    micro_data[key] = value[jj:jj+micro_batchsize] if len(value) == current_batchsize else value
                else:
                    micro_data[key] = value

            last_batch = (jj + micro_batchsize >= current_batchsize)
            tt = torch.randint(
                0, self.base_diffusion.num_timesteps,
                size=(micro_data['gt'].shape[0],),
                device=f"cuda:{self.rank}",
            )

            latent_downsamping_sf = 2 ** (len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf

            if 'autoencoder' in self.configs:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]

            noise = torch.randn(
                size=(micro_data['gt'].shape[0], noise_chn) + (latent_resolution,) * 2,
                device=micro_data['gt'].device,
            )

            if self.configs.model.params.cond_lq:
                # Multi-SAR: lq has shape [B, N, C, H, W]
                model_kwargs = {
                    'lq': micro_data['lq'],
                    'num_sar': micro_data.get('num_sar', None),
                }
                if 'mask' in micro_data:
                    model_kwargs['mask'] = micro_data['mask']
            else:
                model_kwargs = None

            # For diffusion y: use first SAR image, channel 0 (VV) expanded to 3 channels
            # lq: [B, N, C, H, W] → take first SAR, first channel
            if len(micro_data['lq'].shape) == 5:
                vv_band = micro_data['lq'][:, 0, 0:1, :, :]  # [B, 1, H, W]
                lq_for_diffusion = vv_band.expand(-1, 3, -1, -1)  # [B, 3, H, W]
            else:
                lq_for_diffusion = micro_data['lq']

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                lq_for_diffusion,
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
                cdiff_beta=self.configs.train.get('cdiff_beta', None),
            )

            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)

            # Make logging
            if last_batch:
                self.log_step_train(losses, tt, micro_data, z_t, z0_pred.detach())

        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # Grad zero
        self.model.zero_grad()

        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        if self.rank == 0:
            # =======================================================
            # 1. TÍNH TOÁN & GOM NHÓM LOSS CHUNG (BỎ CHIA THEO TIMESTEP)
            # =======================================================
            if self.current_iters % self.configs.train.log_freq[0] == 1 or not hasattr(self, 'loss_mean'):
                self.loss_mean = {key: 0.0 for key in loss.keys()}
                self.loss_count = 0.0

            # Cộng dồn loss của toàn bộ batch
            batch_size = tt.shape[0]
            for key, value in loss.items():
                self.loss_mean[key] += torch.sum(value.detach()).item()
            self.loss_count += batch_size

            # In Log ra màn hình
            if self.current_iters % self.configs.train.log_freq[0] == 0:
                safe_count = max(self.loss_count, 1e-6)
                avg_loss = {key: val / safe_count for key, val in self.loss_mean.items()}

                log_str = f"Train (MultiSAR): {self.current_iters:06d}/{self.configs.train.iterations:06d} | "
                
                # Lấy các chỉ số với default 0.0 để tránh lỗi nếu chưa có
                total_loss = avg_loss.get('loss', 0.0)
                mse_val = avg_loss.get('mse', 0.0)
                lpips_val = avg_loss.get('lpips', 0.0)
                grad_val = avg_loss.get('grad', 0.0)
                
                log_str += f"Loss: {total_loss:.3e} | MSE: {mse_val:.3e} | LPIPS: {lpips_val:.3e} | Grad: {grad_val:.3e} | "
                
                if 'confidence_mean' in avg_loss:
                    log_str += f"Conf: {avg_loss['confidence_mean']:.3f} | "
                    
                log_str += f"lr: {self.optimizer.param_groups[0]['lr']:.2e}"
                
                self.logger.info(log_str)
                self.logging_metric(avg_loss, tag='Loss', phase=phase, add_global_step=True)

            # =======================================================
            # 2. LOG HÌNH ẢNH (GIỮ NGUYÊN CODE BẢN GỐC CỦA BẠN)
            # =======================================================
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                # Visualize multi-SAR input
                lq = batch['lq']
                if len(lq.shape) == 5:
                    # [B, N, C, H, W] - create grid of all SAR images
                    B, N, C, H, W = lq.shape

                    # Get num_sar for masking padded frames
                    num_sar = batch.get('num_sar', None)
                    if num_sar is None:
                        num_sar = torch.full((B,), N, dtype=torch.long)

                    # Create SAR grid: show all valid SAR images for first sample in batch
                    sar_grid_sample = lq[0]  # [N, C, H, W]

                    # Ensure n_valid is an integer
                    if isinstance(num_sar, torch.Tensor):
                        n_valid = int(num_sar[0].item())
                    elif isinstance(num_sar, (list, tuple)):
                        n_valid = int(num_sar[0])
                    else:
                        n_valid = int(num_sar)
                    n_valid = max(1, min(n_valid, N))  # Clamp to valid range

                    # Only show valid (non-padded) SAR images
                    sar_grid_sample = sar_grid_sample[:n_valid]  # [n_valid, C, H, W]

                    # Create grid: arrange SAR images in a row
                    from torchvision.utils import make_grid
                    sar_grid = make_grid(sar_grid_sample, nrow=int(n_valid), normalize=True, scale_each=False, padding=2)

                    # Log SAR grid
                    self.logging_image(sar_grid.unsqueeze(0), tag='lq_SAR_all', phase=phase, add_global_step=False)

                    # Log number of SAR images used
                    if isinstance(num_sar, torch.Tensor):
                        self.logger.info(f"Sample 1 uses {n_valid} SAR images | Batch stats: min={num_sar.min()}, max={num_sar.max()}, mean={num_sar.float().mean():.1f}")

                    # Create composite visualization: SAR Grid | GT | Prediction | Diffused
                    gt_vis = batch['gt'][0:1]  # [1, 3, H, W]
                    x0_vis = self.current_x0_pred[0:1]  # [1, 3, H, W]

                    # Compute diffused
                    x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                    )
                    xt_vis = x_t[0:1]  # [1, 3, H, W]

                    # Resize SAR grid to match GT height for concatenation
                    import torch.nn.functional as F
                    sar_grid_resized = F.interpolate(
                        sar_grid.unsqueeze(0),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )  # [1, 3, H, W]

                    # Concatenate horizontally: SAR | GT | Pred | Diffused
                    composite = torch.cat([sar_grid_resized, gt_vis, x0_vis, xt_vis], dim=3)  # [1, 3, H, 4*W]

                    self.logging_image(composite, tag='composite', phase=phase, add_global_step=True)

                else:
                    self.logging_image(lq, tag='lq', phase=phase, add_global_step=False)

                    self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                    x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                    )
                    self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                    self.logging_image(self.current_x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

            # =======================================================
            # 3. TÍNH THỜI GIAN
            # =======================================================
            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                if not hasattr(self, 'tic'):
                    self.tic = time.time()
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("=" * 100)


def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    from utils import util_image
    from  einops import rearrange
    im1 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00012685_crop000.png',
                            chn = 'rgb', dtype='float32')
    im2 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00014886_crop000.png',
                            chn = 'rgb', dtype='float32')
    im = rearrange(np.stack((im1, im2), 3), 'h w c b -> b c h w')
    im_grid = im.copy()
    for alpha in [0.8, 0.4, 0.1, 0]:
        im_new = im * alpha + np.random.randn(*im.shape) * (1 - alpha)
        im_grid = np.concatenate((im_new, im_grid), 1)

    im_grid = np.clip(im_grid, 0.0, 1.0)
    im_grid = rearrange(im_grid, 'b (k c) h w -> (b k) c h w', k=5)
    xx = vutils.make_grid(torch.from_numpy(im_grid), nrow=5, normalize=True, scale_each=True).numpy()
    util_image.imshow(np.concatenate((im1, im2), 0))
    util_image.imshow(xx.transpose((1,2,0)))
