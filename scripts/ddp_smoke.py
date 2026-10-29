#!/usr/bin/env python3
"""Verify two-GPU DDP initialization, data sharding, and gradient sync."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from src.config import KDConfig
from src.distributed import cleanup_distributed, setup_distributed


def main() -> None:
    config = KDConfig(device="cuda", seed=42)
    setup_distributed(config)
    try:
        current_rank = dist.get_rank()
        current_world_size = dist.get_world_size()

        dataset = TensorDataset(torch.arange(12))
        sampler = DistributedSampler(
            dataset,
            num_replicas=current_world_size,
            rank=current_rank,
            shuffle=False,
        )
        local_ids = [int(batch[0].item()) for batch in DataLoader(dataset, sampler=sampler)]

        model = torch.nn.Linear(4, 1, bias=False, dtype=torch.bfloat16).to(config.device)
        model = DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.full(
            (2, 4),
            current_rank + 1,
            dtype=torch.bfloat16,
            device=config.device,
        )
        loss = model(inputs).float().sum()
        loss.backward()
        optimizer.step()

        gathered_ids: list[list[int] | None] = [None] * current_world_size
        dist.all_gather_object(gathered_ids, local_ids)
        checksum = next(model.parameters()).detach().float().sum()
        gathered_checksums = [torch.zeros_like(checksum) for _ in range(current_world_size)]
        dist.all_gather(gathered_checksums, checksum)

        if current_rank == 0:
            flattened = [sample_id for shard in gathered_ids if shard for sample_id in shard]
            if sorted(flattened) != list(range(len(dataset))):
                raise RuntimeError(f"Invalid sampler shards: {gathered_ids}")
            if not all(
                torch.equal(gathered_checksums[0], value)
                for value in gathered_checksums[1:]
            ):
                raise RuntimeError("DDP parameters diverged across ranks")
            print(
                f"DDP smoke passed: shards={gathered_ids}, "
                f"checksums={gathered_checksums}"
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()