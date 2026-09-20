"""Offline regressions for split Teacher/Student GPU evaluation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.config import KDConfig
from src.evaluate import evaluate_all, evaluate_speed


class EvaluationDeviceTests(unittest.TestCase):
    def test_teacher_evaluated_on_its_own_device(self):
        teacher = MagicMock()
        teacher.parameters.side_effect = lambda: iter(
            [SimpleNamespace(device=torch.device("cuda:1"))]
        )
        loader = SimpleNamespace(dataset=[0])
        with tempfile.TemporaryDirectory() as directory:
            config = KDConfig(
                output_dir=directory, run_id="test", device="cuda:0",
                teacher_device="cuda:1",
            )
            with (
                patch("src.evaluate.load_tokenizer"),
                patch("src.evaluate.create_dataloaders", return_value={"test": loader}) as create_loaders,
                patch("src.evaluate.load_teacher", return_value=teacher),
                patch("src.evaluate.AutoModelForCausalLM.from_pretrained"),
                patch("src.evaluate.evaluate_model", return_value={}) as evaluate,
                patch("torch.cuda.empty_cache"),
            ):
                evaluate_all(config)
            self.assertIs(create_loaders.call_args.kwargs["distributed"], False)
            self.assertEqual(evaluate.call_args_list[0].args[3], "cuda:1")
            self.assertEqual(evaluate.call_args_list[1].args[3], "cuda:0")
            self.assertTrue((Path(directory) / "logs/test/evaluation_results.json").is_file())

    def test_speed_synchronizes_the_measured_device(self):
        tensor = MagicMock()
        tensor.to.return_value = tensor
        tensor.sum.return_value.item.return_value = 4
        loader = [{"input_ids": tensor, "attention_mask": tensor}]
        with patch("torch.cuda.synchronize") as synchronize:
            result = evaluate_speed(MagicMock(), loader, "cuda:1", num_batches=1)
        self.assertEqual(synchronize.call_count, 2)
        for call in synchronize.call_args_list:
            self.assertEqual(call.args, ("cuda:1",))
        self.assertGreater(result["tokens_per_sec"], 0)


if __name__ == "__main__":
    unittest.main()