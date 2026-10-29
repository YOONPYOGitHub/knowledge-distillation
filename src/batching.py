"""Partition global batches without dropping or repeating real samples."""

from collections.abc import Iterator

import torch
from torch.utils.data import Dataset, Sampler


class GlobalBatchSampler(Sampler[list[int]]):
    """Yield one rank's consecutive slice of each shared global batch.

    ``batch_size`` is global, and ``num_replicas`` counts student ranks, not
    frozen-teacher devices. All ranks must use the same size, seed and epoch.
    An empty final slice is represented by ``[-1]`` for ``MaskedDataset``;
    nonempty slices are never padded. Use as a DataLoader's ``batch_sampler``.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        for name, value in (
            ("dataset_size", dataset_size),
            ("batch_size", batch_size),
            ("num_replicas", num_replicas),
            ("rank", rank),
            ("seed", seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if batch_size % num_replicas != 0:
            raise ValueError("global batch_size must be divisible by num_replicas")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.local_batch_size = batch_size // num_replicas

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(
                self.dataset_size, generator=generator, device="cpu"
            ).tolist()
        else:
            indices = list(range(self.dataset_size))

        for global_start in range(0, self.dataset_size, self.batch_size):
            local_start = global_start + self.rank * self.local_batch_size
            local_indices = indices[local_start : local_start + self.local_batch_size]
            yield local_indices if local_indices else [-1]

    def __len__(self) -> int:
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic permutation seeded with ``seed + epoch``."""
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        self.epoch = epoch


class MaskedDataset(Dataset):
    """Pass through real samples and turn ``-1`` into an all-ignored sample.

    Sentinel requests shallow-copy sample zero and allocate new labels, leaving
    its inputs and attention mask unchanged for a valid forward pass. Losses
    must honor ``-100`` and use a sum or an empty-safe mean: ordinary mean
    cross-entropy can be NaN when every label is ignored. KD losses must also
    mask ignored tokens. This wrapper does not change loss normalization.
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if index < -1:
            raise IndexError("only -1 is a valid negative sentinel index")
        if index == -1:
            sample = dict(self.dataset[0])
            sample["labels"] = torch.full_like(sample["labels"], -100)
            return sample
        return self.dataset[index]