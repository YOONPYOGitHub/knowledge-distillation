"""Offline regressions for split Teacher/Student GPU evaluation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.config import KDConfig
from src.evaluate import evaluate_all, evaluate_speed, get_model_size


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
            self.assertEqual(call.args, (torch.device("cuda:1"),))
        self.assertGreater(result["tokens_per_sec"], 0)

    def test_speed_synchronizes_every_shard_of_a_split_model(self):
        """분할 로드된 Teacher는 모든 shard를 동기화해야 속도가 부풀지 않는다."""
        tensor = MagicMock()
        tensor.to.return_value = tensor
        tensor.sum.return_value.item.return_value = 4
        loader = [{"input_ids": tensor, "attention_mask": tensor}]
        sharded = MagicMock()
        sharded.parameters.side_effect = lambda: iter(
            [
                SimpleNamespace(device=torch.device("cuda:0")),
                SimpleNamespace(device=torch.device("cuda:1")),
                SimpleNamespace(device=torch.device("cuda:1")),
            ]
        )
        with patch("torch.cuda.synchronize") as synchronize:
            evaluate_speed(sharded, loader, "cuda:0", num_batches=1)
        synchronized = {call.args[0] for call in synchronize.call_args_list}
        self.assertEqual(
            synchronized, {torch.device("cuda:0"), torch.device("cuda:1")}
        )

    def test_vision_tower_is_reported_separately(self):
        """Gemma 3 멀티모달 Teacher의 vision tower는 텍스트 파라미터와 분리해 기록한다."""
        model = torch.nn.Module()
        model.language_model = torch.nn.Linear(4, 4)  # 20 params
        model.vision_tower = torch.nn.Linear(3, 2)  # 8 params
        info = get_model_size(model)
        self.assertEqual(info["total_params"], 28)
        self.assertEqual(info["vision_params"], 8)
        self.assertEqual(info["text_params"], 20)

    def test_plain_model_has_no_vision_breakdown(self):
        info = get_model_size(torch.nn.Linear(4, 4))
        self.assertNotIn("vision_params", info)


if __name__ == "__main__":
    unittest.main()