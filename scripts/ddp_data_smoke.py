#!/usr/bin/env python3
"""Verify Korean dataset loading and DDP sharding without model weights."""

from __future__ import annotations

import argparse

from src.config import from_yaml
from src.dataset import create_dataloaders, load_tokenizer
from src.distributed import cleanup_distributed, rank, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--samples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = from_yaml(args.config)
    config.dataset_max_samples = args.samples
    config.validation_ratio = 0.1
    config.test_ratio = 0.1
    config.num_workers = 0
    setup_distributed(config)
    try:
        tokenizer = load_tokenizer(config.student_model)
        loaders = create_dataloaders(config, tokenizer)
        batch = next(iter(loaders["train"]))
        print(
            f"rank={rank()} device={config.device} "
            f"samples={len(loaders['train'].dataset)} "
            f"batch={tuple(batch['input_ids'].shape)}"
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()