"""
Distributed training utilities.
Supports single GPU (Colab), multi GPU via mp.spawn, and torchrun.
"""

import os
import socket
import torch
import torch.distributed as dist


def setup_distributed(rank: int = None, world_size: int = None,
                      backend: str = "nccl", port: str = None):
    """Initialize the process group.

    Handles three launch modes: torchrun (env vars set automatically),
    mp.spawn (rank and world_size passed explicitly), and single GPU
    (no distributed setup, returns immediately).

    For mp.spawn, the port must be determined ONCE in the parent process
    and passed to all children.
    """
    if is_torchrun():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    elif rank is not None and world_size is not None and world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port or "12355"
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ and "RANK" in os.environ


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_local_rank() -> int:
    if is_torchrun():
        return int(os.environ["LOCAL_RANK"])
    if dist.is_initialized():
        return dist.get_rank() % max(torch.cuda.device_count(), 1)
    return 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def find_free_port() -> str:
    """Find a free port on localhost. Call ONCE in the parent process,
    then pass the result to all children via the port argument."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def gather_predictions(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Gather variable length tensors from all ranks to rank 0.

    Necessary for distributed evaluation: metrics like AUROC must be
    computed on the full prediction set, not per GPU subsets.
    """
    if world_size <= 1:
        return tensor

    local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
    size_list = [torch.zeros(1, device=tensor.device, dtype=torch.long)
                 for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = int(max(s.item() for s in size_list))

    if tensor.shape[0] < max_size:
        padding = torch.zeros(
            max_size - tensor.shape[0], *tensor.shape[1:],
            device=tensor.device, dtype=tensor.dtype
        )
        tensor = torch.cat([tensor, padding], dim=0)

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    if is_main_process():
        result = []
        for i, g in enumerate(gathered):
            actual_size = int(size_list[i].item())
            result.append(g[:actual_size])
        return torch.cat(result, dim=0)

    return tensor
