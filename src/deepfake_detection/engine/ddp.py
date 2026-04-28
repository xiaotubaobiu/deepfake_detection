from __future__ import annotations

import os
import torch
import torch.distributed as dist


def init_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier():
    if dist.is_initialized():
        dist.barrier()
