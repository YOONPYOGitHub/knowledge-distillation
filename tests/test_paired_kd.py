"""Offline loss and configuration regressions for the paired-GPU path."""

import os
import unittest
from unittest.mock import patch

import torch

from src.config import KDConfig
from src.distill import kd_loss
from src.distributed import normalize_batch_loss, setup_distributed


class PairedKDTests(unittest.TestCase):
    def test_masked_rank_loss_and_gradients_are_zero(self):
        for divergence in ("forward_kl", "reverse_kl"):
            config = KDConfig(device="cpu", kd_reduction="tokenmean", kd_divergence=divergence)
            logits = torch.randn(1, 4, 8, requires_grad=True)
            loss, ce, kd = kd_loss(logits, torch.randn_like(logits), torch.full((1, 4), -100), config)
            self.assertEqual((loss.item(), ce, kd), (0.0, 0.0, 0.0))
            loss.backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_global_batch_mismatch_fails_before_cuda(self):
        config = KDConfig(device="cpu", batch_size=1, global_batch_size=2, kd_reduction="tokenmean")
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(ValueError, "global_batch_size"):
                setup_distributed(config)

    def test_teacher_mapping_does_not_overlap_students(self):
        config = KDConfig(device="cuda", teacher_devices=["cuda:0", "cuda:3"])
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"}):
            with patch("torch.cuda.device_count", return_value=4):
                with self.assertRaisesRegex(ValueError, "disjoint"):
                    setup_distributed(config)

    def test_teacher_mapping_selects_rank_specific_gpu(self):
        config = KDConfig(device="cuda", teacher_devices=["cuda:2", "cuda:3"], run_id="test")
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "LOCAL_RANK": "1"}),
            patch("torch.cuda.device_count", return_value=4),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_device"),
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.broadcast_object_list"),
        ):
            setup_distributed(config)
        self.assertEqual(config.device, "cuda:1")
        self.assertEqual(config.teacher_device, "cuda:3")

    def test_single_rank_normalization_and_empty_batch_guard(self):
        config = KDConfig(device="cpu", batch_size=2, global_batch_size=2, kd_reduction="tokenmean")
        labels = torch.tensor([[1, 2, -100]])
        loss = torch.tensor(3.0, requires_grad=True)
        scaled, metrics = normalize_batch_loss(loss, {"loss": 3.0}, labels, config)
        self.assertEqual(scaled.item(), 3.0)
        self.assertEqual(metrics, {"loss": 3.0})
        with self.assertRaisesRegex(ValueError, "no valid"):
            normalize_batch_loss(loss, {"loss": 3.0}, torch.full_like(labels, -100), config)


if __name__ == "__main__":
    unittest.main()