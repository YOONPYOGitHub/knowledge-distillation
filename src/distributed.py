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
    if config.global_batch_size and config.batch_size * world_size != config.global_batch_size:
        raise ValueError("batch_size * WORLD_SIZE must equal global_batch_size")
    teacher_devices = config.teacher_devices
    if teacher_devices:
        if len(teacher_devices) != world_size:
            raise ValueError("teacher_devices must contain one device per DDP rank")
        if int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))) != world_size:
            raise ValueError("Dedicated teacher placement currently supports one node only")
        for name in teacher_devices:
            device = torch.device(name)
            if (device.type != "cuda" or device.index is None
                    or device.index < world_size or device.index >= torch.cuda.device_count()):
                raise ValueError(f"Teacher GPU must be available and disjoint from student ranks: {name}")
    if world_size == 1:
        if teacher_devices:
            config.teacher_device = teacher_devices[0]
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA GPUs")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    config.device = f"cuda:{local_rank}"
    config.teacher_device = teacher_devices[local_rank] if teacher_devices else config.device

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


def _fp32_allreduce_hook(process_group, bucket):
    """Reduce BF16 gradients in FP32, then restore the parameter gradient dtype."""
    buffer = bucket.buffer()
    reduced = buffer.float()
    future = dist.all_reduce(reduced, group=process_group, async_op=True).get_future()
    return future.then(
        lambda result: result.value()[0].div_(dist.get_world_size(process_group)).to(buffer.dtype)
    )


def wrap_ddp(model: torch.nn.Module, device: str, fp32_reduce: bool = False) -> torch.nn.Module:
    if not is_distributed():
        return model
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    wrapped = DistributedDataParallel(
        model,
        device_ids=[device_index],
        output_device=device_index,
        gradient_as_bucket_view=True,
    )
    if fp32_reduce:
        wrapped.register_comm_hook(None, _fp32_allreduce_hook)
    return wrapped


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
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
        return
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def normalize_batch_loss(loss, metrics, labels, config):
    """Match a single global token-mean loss, including a zero-token final rank.

    DDP averages gradients across ranks. Scale each local mean by
    world_size * local_valid_tokens / global_valid_tokens before backward.
    Metrics are global batch means, preserving the legacy mean-over-steps history.
    """
    if not config.global_batch_size:
        return loss, metrics
    local_tokens = int((labels[..., 1:] != -100).sum().item())
    keys = list(metrics)
    values = torch.tensor(
        [local_tokens] + [metrics[key] * local_tokens for key in keys],
        dtype=torch.float64, device=labels.device,
    )
    if is_distributed():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    global_tokens = values[0].item()
    if not global_tokens:
        raise ValueError("A global batch contains no valid prediction targets")
    means = {key: values[i + 1].item() / global_tokens for i, key in enumerate(keys)}
    return loss * (world_size() * local_tokens / global_tokens), means