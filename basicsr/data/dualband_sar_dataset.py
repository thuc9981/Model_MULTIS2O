"""
DualBand SAR Dataset for VV (primary) + VH (auxiliary) → Optical-Gray translation.

S1 images: 3-channel PNG stored as (VV, VH, VV/VH) in RGB order.
    - OpenCV reads as BGR, so: ch0=VV/VH(B), ch1=VH(G), ch2=VV(R)
    - We extract VV (ch2 in BGR) and VH (ch1 in BGR), discard VV/VH (ch0).
S2 images: grayscale PNG (optical ground truth).
    - Read as grayscale, then expand to 3 channels for autoencoder compatibility.

The dataset returns:
    lq: [2, H, W] tensor  → channel 0 = VV (primary), channel 1 = VH (auxiliary)
    gt: [3, H, W] tensor  → grayscale expanded to 3 channels (for autoencoder)
    lq_path, gt_path: file paths
"""

import os
from os import path as osp
import cv2
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DualBandSARDataset(data.Dataset):
    """Dual-band SAR dataset: VV (primary) + VH (auxiliary) → optical gray.

    Args:
        opt (dict): Config for dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt (optical grayscale).
            dataroot_lq (str): Data root path for lq (SAR 3-band: VV, VH, VV/VH).
            io_backend (dict): IO backend type and other kwargs.
            gt_size (int): Cropped patch size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation.
            scale (int): Scale factor (1 for SAR-to-optical).
            phase (str): 'train' or 'val'.
            vv_channel_bgr (int): VV channel index in BGR order. Default: 2 (R channel).
            vh_channel_bgr (int): VH channel index in BGR order. Default: 1 (G channel).
    """

    def __init__(self, opt):
        super(DualBandSARDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        # Channel indices for VV and VH in BGR order
        # Default: VV is R channel (BGR index 2), VH is G channel (BGR index 1)
        self.vv_channel = opt.get('vv_channel_bgr', 2)
        self.vh_channel = opt.get('vh_channel_bgr', 1)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # Optional filename mapping between LQ and GT
        # Example: LQ has suffix "_S1" before patch index; GT does not.
        # Set in config:
        #   lq_suffix_remove: "_S1"
        # or:
        #   lq_suffix_replace: ["_S1", ""]
        self.lq_suffix_remove = opt.get('lq_suffix_remove', None)
        self.lq_suffix_replace = opt.get('lq_suffix_replace', None)

        self.paths = self._build_paired_paths()

    def _build_paired_paths(self):
        # Default behavior: same filenames
        if self.lq_suffix_remove is None and self.lq_suffix_replace is None:
            return paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl
            )

        gt_files = {}
        for fname in os.listdir(self.gt_folder):
            if not osp.isfile(osp.join(self.gt_folder, fname)):
                continue
            base, _ = osp.splitext(fname)
            gt_files[base] = osp.join(self.gt_folder, fname)

        paths = []
        missing = []
        for fname in os.listdir(self.lq_folder):
            if not osp.isfile(osp.join(self.lq_folder, fname)):
                continue
            lq_base, ext = osp.splitext(fname)
            gt_base = lq_base
            if self.lq_suffix_remove:
                gt_base = gt_base.replace(self.lq_suffix_remove, '')
            if self.lq_suffix_replace:
                if isinstance(self.lq_suffix_replace, (list, tuple)) and len(self.lq_suffix_replace) == 2:
                    gt_base = gt_base.replace(self.lq_suffix_replace[0], self.lq_suffix_replace[1])
                else:
                    raise ValueError('lq_suffix_replace must be a list/tuple of length 2, e.g., ["_S1", ""]')

            gt_path = gt_files.get(gt_base, None)
            if gt_path is None:
                missing.append(gt_base + ext)
                continue

            lq_path = osp.join(self.lq_folder, fname)
            paths.append({'lq_path': lq_path, 'gt_path': gt_path})

        if len(paths) == 0:
            raise FileNotFoundError(
                f'No paired images found between {self.lq_folder} and {self.gt_folder}. '
                'Check file naming and lq/gt roots.'
            )
        if missing:
            print(f'[WARN] {len(missing)} lq files are missing for gt images. These pairs will be skipped.')

        return paths

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt
            )

        scale = self.opt['scale']

        #   Load GT (optical grayscale)  
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        # Read as grayscale
        img_gt_gray = np.frombuffer(img_bytes, np.uint8)
        img_gt_gray = cv2.imdecode(img_gt_gray, cv2.IMREAD_GRAYSCALE)
        if img_gt_gray is None:
            raise ValueError(f"Failed to read GT image: {gt_path}")
        img_gt_gray = img_gt_gray.astype(np.float32) / 255.0
        # Expand to 3 channels for autoencoder compatibility: [H, W] → [H, W, 3]
        img_gt = np.stack([img_gt_gray, img_gt_gray, img_gt_gray], axis=-1)

        #   Load LQ (SAR 3-band)  
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq_raw = np.frombuffer(img_bytes, np.uint8)
        img_lq_raw = cv2.imdecode(img_lq_raw, cv2.IMREAD_UNCHANGED)
        if img_lq_raw is None:
            raise ValueError(f"Failed to read LQ image: {lq_path}")
        img_lq_raw = img_lq_raw.astype(np.float32) / 255.0

        # Extract VV and VH only (discard VV/VH band)
        # img_lq_raw is in BGR order from OpenCV
        vv = img_lq_raw[:, :, self.vv_channel]  # VV (primary)
        vh = img_lq_raw[:, :, self.vh_channel]  # VH (auxiliary)
        img_lq = np.stack([vv, vh], axis=-1)  # [H, W, 2]

        #   Augmentation for training  
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Random crop — need to handle 2-channel LQ and 3-channel GT
            h_lq, w_lq = img_lq.shape[:2]
            h_gt, w_gt = img_gt.shape[:2]
            top = np.random.randint(0, max(1, h_gt - gt_size * scale + 1))
            left = np.random.randint(0, max(1, w_gt - gt_size * scale + 1))
            img_gt = img_gt[top:top + gt_size * scale, left:left + gt_size * scale, :]
            top_lq = top // scale
            left_lq = left // scale
            img_lq = img_lq[top_lq:top_lq + gt_size, left_lq:left_lq + gt_size, :]

            # Flip and rotation (apply same transform to both)
            # Horizontal flip
            if self.opt.get('use_hflip', False) and np.random.random() < 0.5:
                img_gt = np.flip(img_gt, axis=1).copy()
                img_lq = np.flip(img_lq, axis=1).copy()
            # Vertical flip + transpose (rotation augmentation)
            if self.opt.get('use_rot', False) and np.random.random() < 0.5:
                img_gt = np.flip(img_gt, axis=0).copy()
                img_lq = np.flip(img_lq, axis=0).copy()
            if self.opt.get('use_rot', False) and np.random.random() < 0.5:
                img_gt = np.transpose(img_gt, (1, 0, 2)).copy()
                img_lq = np.transpose(img_lq, (1, 0, 2)).copy()

        # Crop unmatched GT images during validation
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        #   Convert to tensor  
        # GT: [H, W, 3] → [3, H, W]
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float()
        # LQ: [H, W, 2] → [2, H, W]
        img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float()

        # Normalize to [-1, 1] 
        img_gt = img_gt * 2.0 - 1.0
        img_lq = img_lq * 2.0 - 1.0

        return {
            'lq': img_lq,        # [2, H, W] — VV + VH
            'gt': img_gt,        # [3, H, W] — grayscale expanded to 3ch
            'lq_path': lq_path,
            'gt_path': gt_path,
        }

    def __len__(self):
        return len(self.paths)
