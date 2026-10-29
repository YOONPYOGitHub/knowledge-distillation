"""CPU-only batching tests; no models, datasets or downloads are required."""

import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from src.batching import GlobalBatchSampler, MaskedDataset


class _ToyDataset(Dataset):
    """Return cached dictionaries so accidental mutations are observable."""

    def __init__(self, size: int) -> None:
        self.samples = [
            {
                "input_ids": torch.tensor([index + 1, 2, 3], dtype=torch.long),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "labels": torch.tensor([index + 1, 2, -100], dtype=torch.long),
            }
            for index in range(size)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class GlobalBatchSamplerTests(unittest.TestCase):
    def test_exhaustive_global_step_parity(self) -> None:
        for size in range(1, 10):
            for global_batch, replicas_options in ((4, (1, 2, 4)), (2, (1, 2))):
                for replicas in replicas_options:
                    for shuffle in (False, True):
                        for epoch in (0, 3):
                            with self.subTest(
                                size=size,
                                global_batch=global_batch,
                                replicas=replicas,
                                shuffle=shuffle,
                                epoch=epoch,
                            ):
                                self._assert_step_parity(
                                    size, global_batch, replicas, shuffle, epoch
                                )

    def _assert_step_parity(
        self,
        size: int,
        global_batch: int,
        replicas: int,
        shuffle: bool,
        epoch: int,
    ) -> None:
        single = GlobalBatchSampler(size, global_batch, shuffle=shuffle)
        single.set_epoch(epoch)
        single_batches = list(single)
        steps = (size + global_batch - 1) // global_batch
        self.assertEqual(len(single), steps)
        self.assertEqual(len(single_batches), steps)

        order = (
            torch.randperm(
                size, generator=torch.Generator().manual_seed(42 + epoch)
            ).tolist()
            if shuffle
            else list(range(size))
        )
        self.assertEqual(
            single_batches,
            [order[start : start + global_batch] for start in range(0, size, global_batch)],
        )

        rank_batches = []
        for rank in range(replicas):
            sampler = GlobalBatchSampler(
                size, global_batch, replicas, rank, shuffle=shuffle
            )
            sampler.set_epoch(epoch)
            batches = list(sampler)
            self.assertEqual(len(sampler), steps)
            self.assertEqual(len(batches), steps)
            self.assertEqual(list(sampler), batches)
            rank_batches.append(batches)

        all_real_indices = []
        local_size = global_batch // replicas
        for step, global_indices in enumerate(single_batches):
            real_indices = []
            for rank, batches in enumerate(rank_batches):
                local_indices = batches[step]
                expected = global_indices[rank * local_size : (rank + 1) * local_size]
                self.assertEqual(local_indices, expected if expected else [-1])
                if not expected:
                    self.assertEqual(step, steps - 1)
                else:
                    self.assertNotIn(-1, local_indices)
                real_indices.extend(index for index in local_indices if index != -1)
            self.assertEqual(real_indices, global_indices)
            all_real_indices.extend(real_indices)

        self.assertEqual(all_real_indices, order)
        self.assertEqual(sorted(all_real_indices), list(range(size)))
        self.assertEqual(len(set(all_real_indices)), size)

    def test_global_batch_two_with_two_student_ranks(self) -> None:
        self.assertEqual(list(GlobalBatchSampler(3, 2)), [[0, 1], [2]])
        self.assertEqual(list(GlobalBatchSampler(3, 2, 2, 0)), [[0], [2]])
        self.assertEqual(list(GlobalBatchSampler(3, 2, 2, 1)), [[1], [-1]])

    def test_partial_nonempty_batches_are_not_padded(self) -> None:
        self.assertEqual(list(GlobalBatchSampler(3, 4, 2, 0)), [[0, 1]])
        self.assertEqual(list(GlobalBatchSampler(3, 4, 2, 1)), [[2]])
        self.assertEqual(list(GlobalBatchSampler(1, 4, 2, 0)), [[0]])
        self.assertEqual(list(GlobalBatchSampler(1, 4, 2, 1)), [[-1]])
        self.assertEqual(list(GlobalBatchSampler(1, 4)), [[0]])

    def test_shuffle_epoch_stability_and_change(self) -> None:
        sampler = GlobalBatchSampler(97, 4, shuffle=True, seed=123)
        self.assertEqual(sampler.epoch, 0)
        epoch_zero = list(sampler)
        self.assertEqual(epoch_zero, list(sampler))
        self.assertEqual(
            epoch_zero, list(GlobalBatchSampler(97, 4, shuffle=True, seed=123))
        )

        sampler.set_epoch(1)
        epoch_one = list(sampler)
        self.assertNotEqual(epoch_zero, epoch_one)
        self.assertEqual(epoch_one, list(sampler))
        self.assertEqual(
            [index for batch in epoch_one for index in batch],
            torch.randperm(97, generator=torch.Generator().manual_seed(124)).tolist(),
        )
        sampler.set_epoch(0)
        self.assertEqual(epoch_zero, list(sampler))

    def test_shuffle_is_independent_of_default_rng(self) -> None:
        sampler = GlobalBatchSampler(19, 4, shuffle=True)
        original_state = torch.random.get_rng_state().clone()
        try:
            first = list(sampler)
            self.assertTrue(torch.equal(original_state, torch.random.get_rng_state()))
            torch.rand(32)
            advanced_state = torch.random.get_rng_state().clone()
            self.assertEqual(first, list(sampler))
            self.assertTrue(torch.equal(advanced_state, torch.random.get_rng_state()))
        finally:
            torch.random.set_rng_state(original_state)

    def test_unshuffled_order_does_not_depend_on_epoch_or_seed(self) -> None:
        sampler = GlobalBatchSampler(9, 4, seed=987)
        self.assertIsInstance(sampler, Sampler)
        expected = [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
        self.assertEqual(list(sampler), expected)
        sampler.set_epoch(12)
        self.assertEqual(list(sampler), expected)

    def test_invalid_configuration(self) -> None:
        invalid = (
            {"dataset_size": 0},
            {"dataset_size": -1},
            {"batch_size": 0},
            {"batch_size": -4},
            {"num_replicas": 0},
            {"num_replicas": -2},
            {"rank": -1},
            {"rank": 1},
            {"num_replicas": 2, "rank": 2},
            {"num_replicas": 2, "rank": 3},
            {"batch_size": 3, "num_replicas": 2},
            {"batch_size": 2, "num_replicas": 4},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                config = {"dataset_size": 9, "batch_size": 4}
                config.update(overrides)
                with self.assertRaises(ValueError):
                    GlobalBatchSampler(**config)

    def test_integer_configuration_is_required(self) -> None:
        for name in ("dataset_size", "batch_size", "num_replicas", "rank", "seed"):
            for value in (1.5, "2", None, True):
                with self.subTest(name=name, value=value):
                    config = {"dataset_size": 9, "batch_size": 4, name: value}
                    with self.assertRaises(TypeError):
                        GlobalBatchSampler(**config)

        sampler = GlobalBatchSampler(9, 4)
        for epoch in (1.5, "2", None, True):
            with self.subTest(epoch=epoch):
                with self.assertRaises(TypeError):
                    sampler.set_epoch(epoch)
                self.assertEqual(sampler.epoch, 0)


class MaskedDatasetTests(unittest.TestCase):
    def test_normal_samples_pass_through(self) -> None:
        base = _ToyDataset(3)
        dataset = MaskedDataset(base)
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(len(dataset), len(base))
        for index in range(len(base)):
            self.assertIs(dataset[index], base[index])

    def test_sentinel_is_fresh_and_does_not_mutate_base(self) -> None:
        base = _ToyDataset(3)
        dataset = MaskedDataset(base)
        snapshots = [{key: value.clone() for key, value in sample.items()} for sample in base]
        first = dataset[-1]
        second = dataset[-1]
        self.assertIsNot(first, base[0])
        self.assertIsNot(first, second)
        self.assertEqual(set(first), set(base[0]))
        self.assertIsNot(first["labels"], base[0]["labels"])
        self.assertIsNot(first["labels"], second["labels"])
        self.assertTrue(torch.equal(first["labels"], torch.full_like(base[0]["labels"], -100)))
        self.assertEqual(first["labels"].shape, base[0]["labels"].shape)
        self.assertEqual(first["labels"].dtype, base[0]["labels"].dtype)
        self.assertEqual(first["labels"].device, base[0]["labels"].device)
        for key in ("input_ids", "attention_mask"):
            self.assertIs(first[key], base[0][key])
            self.assertTrue(torch.equal(first[key], snapshots[0][key]))

        first["labels"].fill_(17)
        first["extra"] = torch.tensor(1)
        self.assertNotIn("extra", base[0])
        self.assertNotIn("extra", second)
        self.assertTrue(torch.all(second["labels"] == -100).item())
        self.assertTrue(torch.all(dataset[-1]["labels"] == -100).item())
        for sample, snapshot in zip(base, snapshots):
            for key in snapshot:
                self.assertTrue(torch.equal(sample[key], snapshot[key]))

    def test_invalid_indices_do_not_wrap(self) -> None:
        dataset = MaskedDataset(_ToyDataset(3))
        for index in (-2, -3, -4, -100, 3, 4):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    dataset[index]

    def test_odd_final_batch_sentinel_has_zero_loss_and_gradient(self) -> None:
        base = _ToyDataset(3)
        dataset = MaskedDataset(base)
        loaders = [
            DataLoader(
                dataset,
                batch_sampler=GlobalBatchSampler(len(dataset), 2, 2, rank),
                num_workers=0,
            )
            for rank in range(2)
        ]
        rank_batches = [list(loader) for loader in loaders]
        self.assertEqual([len(loader) for loader in loaders], [2, 2])
        self.assertEqual([len(batches) for batches in rank_batches], [2, 2])

        real_documents = []
        for step in range(2):
            for batches in rank_batches:
                batch = batches[step]
                for row in range(batch["labels"].shape[0]):
                    if torch.any(batch["labels"][row] != -100).item():
                        real_documents.append(batch["input_ids"][row, 0].item())
        self.assertEqual(real_documents, [1, 2, 3])

        sentinel = rank_batches[1][-1]
        self.assertTrue(torch.all(sentinel["labels"] == -100).item())
        self.assertTrue(torch.equal(sentinel["input_ids"][0], base[0]["input_ids"]))
        self.assertTrue(torch.equal(sentinel["attention_mask"][0], base[0]["attention_mask"]))

        model = torch.nn.Sequential(torch.nn.Embedding(16, 4), torch.nn.Linear(4, 16))
        logits = model(sentinel["input_ids"])
        self.assertTrue(torch.isfinite(logits).all().item())
        labels = sentinel["labels"][:, 1:].reshape(-1)
        # Sum plus a guarded denominator also covers an all-ignored token mean.
        loss_sum = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, 16),
            labels,
            ignore_index=-100,
            reduction="sum",
        )
        loss = loss_sum / (labels != -100).sum().clamp_min(1)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter)))


if __name__ == "__main__":
    unittest.main()