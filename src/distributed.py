"""Small torch.distributed helpers for single-node data parallel training."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup_distributed(config: Any) -> None:
    """Initialize NCCL when launched with torchrun and select the local GPU."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA GPUs")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    config.device = f"cuda:{local_rank}"
    config.teacher_device = config.device

    run_id = [config.run_id if local_rank == 0 else None]
    dist.broadcast_object_list(run_id, src=0)
    config.run_id = run_id[0]


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def wrap_ddp(model: torch.nn.Module, device: str) -> torch.nn.Module:
    if not is_distributed():
        return model
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return DistributedDataParallel(
        model,
        device_ids=[device_index],
        output_device=device_index,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_metrics(metrics: dict[str, float], device: str) -> dict[str, float]:
    if not is_distributed():
        return metrics
    keys = list(metrics)
    values = torch.tensor(
        [metrics[key] for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: value.item() for key, value in zip(keys, values)}


def set_epoch(dataloader: Any, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)