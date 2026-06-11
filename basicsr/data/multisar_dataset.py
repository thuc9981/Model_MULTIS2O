"""
Multi-SAR Dataset for SAR (S1 + OPERA) -> optical (S2/HLS RGB) translation.

This dataset loads multiple SAR images (from different acquisition dates/sources)
for each patch and maps them to a single optical image.

Supported naming styles:
1) Legacy OP/HLS style
   - SAR: {region}_HLS_{target_date}_OP_{acq_date}_VVVH_aligned_p0.npy
   - GT:  {region}_HLS_{target_date}_RGB_aligned_p0.npy

2) New S1/OP + S2 style
   - SAR: {region}_S2_{target_date}_S1_{acq_date}_VVVH_aligned_DB_patch_r_c.npy
   - SAR: {region}_HLS_{target_date}_OP_{acq_date}_VVVH_aligned_DB_patch_r_c.npy
   - GT:  {region}_S2_{target_date}_RGB_aligned_patch_r_c.npy

The dataset returns:
    lq: [N, 3, H, W] tensor  -> N SAR images (VV, VH, VV/VH)
    gt: [3, H, W] tensor     -> RGB optical image
    num_sar: int             -> actual number of SAR images (for masking in transformer)
    lq_paths: list           -> paths to SAR images
    gt_path: str             -> path to optical image
"""

import os
import re
from os import path as osp
from collections import defaultdict
import cv2
import numpy as np
import torch
from torch.utils import data as data

from basicsr.utils import FileClient
from basicsr.utils.registry import DATASET_REGISTRY

# Supported file extensions
SUPPORTED_EXTENSIONS = ('.png', '.npy')


@DATASET_REGISTRY.register()
class MultiSARDataset(data.Dataset):
    """Multi-SAR dataset: multiple SAR images → single optical RGB image.

    Args:
        opt (dict): Config for dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt (optical RGB images).
            dataroot_lq (str): Data root path for lq (SAR images).
            io_backend (dict): IO backend type and other kwargs.
            gt_size (int): Cropped patch size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation.
            scale (int): Scale factor (1 for SAR-to-optical).
            phase (str): 'train' or 'val'.
            max_sar_images (int): Maximum number of SAR images to use. Default: 8.
            min_sar_images (int): Minimum number of SAR images required. Default: 2.
    """

    def __init__(self, opt):
        super(MultiSARDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        self.max_sar_images = opt.get('max_sar_images', 8)
        self.min_sar_images = opt.get('min_sar_images', 2)

        # Optional extension filter: auto | npy | png.
        file_format = str(opt.get('file_format', 'auto')).strip().lower()
        if file_format in ('auto', 'all', '*'):
            self.allowed_extensions = SUPPORTED_EXTENSIONS
        elif file_format in ('npy', '.npy'):
            self.allowed_extensions = ('.npy',)
        elif file_format in ('png', '.png', 'rgb', 'rgb_png', 'uint8'):
            self.allowed_extensions = ('.png',)
        else:
            raise ValueError(
                f"Unsupported file_format={file_format}. Use one of: auto, npy, png"
            )

        # Optional source filter for SAR inputs, e.g. ['S1', 'OP'].
        input_sources = opt.get('input_sar_sources', None)
        if input_sources is None:
            self.input_sar_sources = None
        else:
            if isinstance(input_sources, str):
                input_sources = [input_sources]
            self.input_sar_sources = {str(x).strip().upper() for x in input_sources if str(x).strip()}
            if len(self.input_sar_sources) == 0:
                self.input_sar_sources = None

        # If True, keep only samples that contain all requested source types.
        self.require_all_input_sources = bool(opt.get('require_all_input_sources', False))

        # Build paired paths: list of (gt_path, [list of sar_paths])
        self.samples = self._build_paired_paths()

        print(f"[MultiSARDataset] Loaded {len(self.samples)} samples")
        print(f"[MultiSARDataset] Max SAR images per sample: {self.max_sar_images}")
        print(f"[MultiSARDataset] Allowed extensions: {self.allowed_extensions}")
        if self.input_sar_sources is None:
            print("[MultiSARDataset] SAR source filter: all (S1/OP as available)")
        else:
            print(f"[MultiSARDataset] SAR source filter: {sorted(self.input_sar_sources)}")
            print(f"[MultiSARDataset] Require all selected sources: {self.require_all_input_sources}")

    @staticmethod
    def _build_patch_key(patch_1, patch_2=None):
        """Create a stable patch key that supports p0 and patch_row_col styles."""
        if patch_2 is None:
            return str(int(patch_1))
        return f"{int(patch_1)}_{int(patch_2)}"

    def _parse_sar_filename(self, filename):
        """Parse SAR filename."""
        fname = osp.basename(filename)
        m = re.match(
            r'^(?P<region>.+?)_'
            r'(?P<target_tag>HLS|S2)_(?P<target_date>\d{4}-\d{2}-\d{2})_'
            r'(?P<source>OP|S1)_(?P<acq_date>\d{4}-\d{2}-\d{2})_'
            r'VVVH_aligned_'
            r'(?:[A-Za-z0-9_]*?)?'   # <<< FIX: bỏ qua mọi thứ giữa aligned và patch
            r'patch_(?P<patch_1>\d+)(?:_(?P<patch_2>\d+))?'
            r'\.(?P<ext>png|npy)$',
            fname,
            flags=re.IGNORECASE,
        )
        if m is None:
            return None
        return {
            'region': m.group('region'),
            'target_date': m.group('target_date'),
            'source': m.group('source').upper(),
            'acq_date': m.group('acq_date'),
            'patch_key': self._build_patch_key(m.group('patch_1'), m.group('patch_2')),
            'ext': m.group('ext').lower(),
        }

    def _parse_hls_filename(self, filename):
        """Parse HLS filename."""
        fname = osp.basename(filename)
        m = re.match(
            r'^(?P<region>.+?)_'
            r'(?P<target_tag>HLS|S2)_(?P<target_date>\d{4}-\d{2}-\d{2})_'
            r'RGB_aligned(?:_[A-Za-z0-9]+)?_'
            r'(?:[A-Za-z0-9_]*?)?'   # <<< FIX: bỏ qua mọi thứ giữa aligned và patch
            r'patch_(?P<patch_1>\d+)(?:_(?P<patch_2>\d+))?\.(?P<ext>png|npy)$',
            fname,
            flags=re.IGNORECASE,
        )
        if m is None:
            return None
        return {
            'region': m.group('region'),
            'target_date': m.group('target_date'),
            'patch_key': self._build_patch_key(m.group('patch_1'), m.group('patch_2')),
            'ext': m.group('ext').lower(),
        }

    def _build_paired_paths(self):
        """Recursive + key-based pairing."""
        if not osp.isdir(self.lq_folder):
            raise FileNotFoundError(f"LQ folder not found: {self.lq_folder}")
        if not osp.isdir(self.gt_folder):
            raise FileNotFoundError(f"GT folder not found: {self.gt_folder}")

        sar_by_key = defaultdict(list)  # (region,target_date,patch) -> [(acq_date, source, path)]
        gt_by_key = {}                  # (region,target_date,patch) -> gt_path
        lq_source_counts = defaultdict(int)

        lq_total = lq_parsed = 0
        gt_total = gt_parsed = 0

        for root, _, files in os.walk(self.lq_folder):
            for fname in files:
                if not fname.lower().endswith(self.allowed_extensions):
                    continue
                lq_total += 1
                info = self._parse_sar_filename(fname)
                if info is None:
                    continue
                if self.input_sar_sources is not None and info['source'] not in self.input_sar_sources:
                    continue
                lq_parsed += 1
                lq_source_counts[info['source']] += 1
                key = (info['region'], info['target_date'], info['patch_key'])
                sar_by_key[key].append((info['acq_date'], info['source'], osp.join(root, fname)))

        for root, _, files in os.walk(self.gt_folder):
            for fname in files:
                if not fname.lower().endswith(self.allowed_extensions):
                    continue
                gt_total += 1
                info = self._parse_hls_filename(fname)
                if info is None:
                    continue
                gt_parsed += 1
                key = (info['region'], info['target_date'], info['patch_key'])
                gt_by_key[key] = osp.join(root, fname)

        samples = []
        matched_keys = 0
        for key, sar_list in sar_by_key.items():
            gt_path = gt_by_key.get(key)
            if gt_path is None:
                continue
            matched_keys += 1
            sar_list = sorted(sar_list, key=lambda x: (x[0], x[1]))

            if self.input_sar_sources is not None and self.require_all_input_sources:
                present_sources = {src for _, src, _ in sar_list}
                if not self.input_sar_sources.issubset(present_sources):
                    continue

            sar_paths = [p for _, _, p in sar_list]

            if len(sar_paths) < self.min_sar_images:
                continue

            samples.append({
                'gt_path': gt_path,
                'sar_paths': sar_paths,
            })

        print(
            f"[MultiSARDataset] Pairing stats: "
            f"LQ total={lq_total}, LQ parsed={lq_parsed}, "
            f"GT total={gt_total}, GT parsed={gt_parsed}, "
            f"matched_keys={matched_keys}, samples={len(samples)}"
        )
        if len(lq_source_counts) > 0:
            source_stats = ', '.join([f"{k}={v}" for k, v in sorted(lq_source_counts.items())])
            print(f"[MultiSARDataset] LQ by source: {source_stats}")

        if len(samples) == 0:
            raise FileNotFoundError(
                f"No paired samples found between {self.lq_folder} and {self.gt_folder}. "
                f"Parsed LQ={lq_parsed}, Parsed GT={gt_parsed}, matched_keys={matched_keys}."
            )

        return samples
    def _load_image(self, path):
        path_lower = path.lower()
        path_upper = path.upper()

        try:
            if path_lower.endswith('.npy'):
                img = np.load(path)

                # ===== HARD CHECK SHAPE =====
                if img.size != 3 * 256 * 256:
                    raise ValueError(f"Invalid size: {img.size}")

                # reshape về đúng format nếu cần
                if img.ndim == 1:
                    img = img.reshape(3, 256, 256)

                if img.ndim == 3 and img.shape[0] in [1, 2, 3, 4]:
                    img = img.transpose(1, 2, 0)

                if img.ndim != 3 or img.shape[2] < 3:
                    raise ValueError(f"Invalid shape after processing: {img.shape}")

                img = img[:, :, :3].astype(np.float32)

                # ===== NORMALIZATION =====
                if '_RGB_' in path_upper:
                    img = np.clip(img, 0.0, 3000.0) / 3000.0

                elif '_OP_' in path_upper or '_S1_' in path_upper:
                    img_vv = np.clip((img[:, :, 0] + 25.0) / 30.0, 0.0, 1.0)
                    img_vh = np.clip((img[:, :, 1] + 35.0) / 40.0, 0.0, 1.0)
                    img_ratio = np.clip(img[:, :, 2] / 16.0, 0.0, 1.0)
                    img = np.stack([img_vv, img_vh, img_ratio], axis=-1)
                else:
                    raise ValueError(f"Unknown type: {path}")

            else:
                img_bytes = self.file_client.get(path, 'img')
                img = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("cv2 decode failed")

                img = img.astype(np.float32) / 255.0
                img = img[:, :, ::-1]

            return img.astype(np.float32)

        except Exception as e:
            print(f"[LOAD ERROR] {path} | {e}")
            return None   # <<< QUAN TRỌNG
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt
            )

        sample = self.samples[index]
        scale = self.opt.get('scale', 1)

        # === Load GT ===
        gt_path = sample['gt_path']
        img_gt = self._load_image(gt_path)

        sar_paths = sample['sar_paths']

        sar_images = []
        valid_paths = []

        for p in sar_paths:
            try:
                img = self._load_image(p)

                # check shape chuẩn
                if img.shape != (256, 256, 3):
                    print(f"[BAD SHAPE] {p} | {img.shape}")
                    continue

                sar_images.append(img)
                valid_paths.append(p)

            except Exception as e:
                print(f"[LOAD ERROR] {p} | {e}")
                continue

        # === HANDLE CASE KHÔNG CÓ SAR ===
        if len(sar_images) == 0:
            raise ValueError(f"No valid SAR images in sample: {gt_path}")

        actual_num_sar = len(sar_images)

        # === LIMIT max_sar_images ===
        if actual_num_sar > self.max_sar_images:
            if self.opt['phase'] == 'train':
                indices = np.random.choice(actual_num_sar, self.max_sar_images, replace=False)
                indices = np.sort(indices)
            else:
                indices = np.linspace(0, actual_num_sar - 1, self.max_sar_images, dtype=int)

            sar_images = [sar_images[i] for i in indices]
            valid_paths = [valid_paths[i] for i in indices]
            actual_num_sar = self.max_sar_images

        # === STACK (SAFE) ===
        try:
            img_lq = np.stack(sar_images, axis=0)
        except Exception as e:
            raise ValueError(f"Stack failed at {gt_path} | shapes={[x.shape for x in sar_images]}")

        # === AUGMENT ===
        if self.opt['phase'] == 'train':
            gt_size = self.opt.get('gt_size', 256)
            h_gt, w_gt = img_gt.shape[:2]

            top = np.random.randint(0, max(1, h_gt - gt_size * scale + 1))
            left = np.random.randint(0, max(1, w_gt - gt_size * scale + 1))

            img_gt = img_gt[top:top + gt_size * scale, left:left + gt_size * scale, :]

            top_lq = top // scale
            left_lq = left // scale
            img_lq = img_lq[:, top_lq:top_lq + gt_size, left_lq:left_lq + gt_size, :]

            if self.opt.get('use_hflip', False) and np.random.random() < 0.5:
                img_gt = np.flip(img_gt, axis=1).copy()
                img_lq = np.flip(img_lq, axis=2).copy()

            if self.opt.get('use_rot', False) and np.random.random() < 0.5:
                img_gt = np.flip(img_gt, axis=0).copy()
                img_lq = np.flip(img_lq, axis=1).copy()

            if self.opt.get('use_rot', False) and np.random.random() < 0.5:
                img_gt = np.transpose(img_gt, (1, 0, 2)).copy()
                img_lq = np.transpose(img_lq, (0, 2, 1, 3)).copy()

        # === VALIDATION CROP ===
        if self.opt['phase'] != 'train':
            h_lq, w_lq = img_lq.shape[1:3]
            img_gt = img_gt[0:h_lq * scale, 0:w_lq * scale, :]

        # === TO TENSOR ===
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float()
        img_lq = torch.from_numpy(img_lq.transpose(0, 3, 1, 2)).float()

        img_gt = img_gt * 2.0 - 1.0
        img_lq = img_lq * 2.0 - 1.0

        # === PAD ===
        if actual_num_sar < self.max_sar_images:
            pad_size = self.max_sar_images - actual_num_sar
            padding = torch.zeros(pad_size, *img_lq.shape[1:])
            img_lq = torch.cat([img_lq, padding], dim=0)

        padded_lq_paths = list(valid_paths)
        while len(padded_lq_paths) < self.max_sar_images:
            padded_lq_paths.append('')

        return {
            'lq': img_lq,
            'gt': img_gt,
            'num_sar': actual_num_sar,
            'lq_paths': padded_lq_paths,
            'gt_path': gt_path,
        }

    def __len__(self):
        return len(self.samples)
