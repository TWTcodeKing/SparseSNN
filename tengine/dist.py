"""
Distributed training utilities.

Supports flexible multi-GPU via user-specified GPU IDs, e.g. --gpu-ids 0,1,2,3.
Uses torchrun / torch.distributed.launch under the hood.
"""

import os
import torch
import torch.distributed as dist


def setup_distributed(backend='nccl'):
    """
    Initialize distributed process group from environment variables
    set by torchrun / torch.distributed.launch.

    Returns (rank, local_rank, world_size).
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_tensor(tensor, world_size):
    """Average a tensor across all processes."""
    if world_size <= 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt
