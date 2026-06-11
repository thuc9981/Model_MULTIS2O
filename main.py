#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from utils.util_common import get_obj_from_str


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./save_dir",
        help="Folder to save the checkpoints and training log",
    )
    parser.add_argument(
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from the save_dir or checkpoint",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="./configs/training/ffhq256_bicubic8.yaml",
        help="Configs of yaml file",
    )
    return parser.parse_args()


def init_distributed():
    """Init torch.distributed if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

        print(f"[DDP INIT] world_size={world_size}, rank={rank}, local_rank={local_rank}")
        return rank, local_rank, world_size
    else:
        print("[Single GPU / No DDP]")
        return 0, 0, 1


if __name__ == "__main__":
    args = get_parser()

    # Load yaml config
    configs = OmegaConf.load(args.cfg_path)

    # Merge CLI args into config
    for key in vars(args):
        if key in ["cfg_path", "save_dir", "resume"]:
            configs[key] = getattr(args, key)

    # Init distributed (safe for 1 GPU + torchrun)
    rank, local_rank, world_size = init_distributed()

    # Create trainer
    trainer = get_obj_from_str(configs.trainer.target)(configs)

    # Start training
    trainer.train()
