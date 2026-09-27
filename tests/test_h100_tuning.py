"""Download-free CPU tests for the parent-facing H100 trial contract."""

import io
import json
import math
import random
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.config import KDConfig
from src import h100_tuning as tuning


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(3, 4)
        self.projection = torch.nn.Linear(4, 3)
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)
        self.config = SimpleNamespace(use_cache=True)
        self.calls = []

    def forward(self, input_ids, attention_mask=None):
        self.calls.append((self.training, torch.is_grad_enabled(), self.config.use_cache))
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def examples(count=16):
    return [{"input_ids": torch.tensor([i % 3, 1, 1, 1]),
             "attention_mask": torch.ones(4, dtype=torch.long),
             "labels": torch.tensor([-100, 1, 1, 1])} for i in range(count)]


class EvaluationTests(unittest.TestCase):
    def test_masked_token_weighting_float32_shift_and_mode(self):
        class TableLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.tensor(
                    [[0, math.log(2), math.log(4)], [math.log(3), 0, 0]],
                    dtype=torch.float16))

            def forward(self, input_ids):
                self_test.assertFalse(torch.is_grad_enabled())
                self_test.assertFalse(self.training)
                return SimpleNamespace(logits=self.table[input_ids])

        self_test = self
        model = TableLM()
        data = [
            {"input_ids": torch.tensor([0, 0, 0, 0]),
             "labels": torch.tensor([99, 0, 1, -100])},
            {"input_ids": torch.tensor([1, 1, 1, 1]),
             "labels": torch.tensor([99, 2, -100, -100])},
        ]
        rows = model.table.float().tolist()
        expected = sum(math.log(sum(math.exp(v) for v in rows[row])) - rows[row][target]
                       for row, target in [(0, 0), (0, 1), (1, 2)]) / 3
        for was_training in (True, False):
            model.train(was_training)
            result = tuning.evaluate_ce(model, tuning.make_loader(data, 1), "cpu")
            self.assertEqual(set(result), {"ce", "ppl", "tokens"})
            self.assertEqual(result["tokens"], 3)
            self.assertAlmostEqual(result["ce"], expected, places=6)
            self.assertAlmostEqual(result["ppl"], math.exp(expected), places=6)
            self.assertEqual(model.training, was_training)

    def test_empty_masked_and_nonfinite_validation_restore_mode(self):
        model = TinyLM()
        masked = examples(1)
        masked[0]["labels"].fill_(-100)
        for data in ([], masked):
            with self.subTest(data=data), self.assertRaises(ValueError):
                tuning.evaluate_ce(model, tuning.make_loader(data, 1), "cpu")
            self.assertTrue(model.training)
        with torch.no_grad():
            model.projection.bias.fill_(float("nan"))
        model.eval()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            tuning.evaluate_ce(model, tuning.make_loader(examples(1), 1), "cpu")
        self.assertFalse(model.training)

    def test_loader_is_deterministic_without_consuming_global_rng(self):
        data = torch.arange(40)
        torch.manual_seed(123)
        state = torch.random.get_rng_state().clone()
        first = torch.cat(list(tuning.make_loader(data, 3, seed=11)))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.rand(100)
        second = torch.cat(list(tuning.make_loader(data, 3, seed=11)))
        other = torch.cat(list(tuning.make_loader(data, 3, seed=12)))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        state = torch.random.get_rng_state().clone()
        self.assertTrue(torch.equal(torch.cat(list(tuning.make_loader(data, 3))), data))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        loader = tuning.make_loader(data, 2, seed=11, num_workers=2)
        self.assertEqual(loader.num_workers, 2)
        self.assertEqual(loader.pin_memory, torch.cuda.is_available())


class TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.initial = self.root / "original_ft.pt"
        torch.save(TinyLM().state_dict(), self.initial)
        self.teacher_path = self.root / "teacher"
        self.teacher_path.mkdir()
        self.config = KDConfig(
            device="cpu", teacher_device="cpu", batch_size=2, epochs=2,
            optimizer="adafactor", learning_rate=0.02, alpha=1.0,
            distill_student_checkpoint=str(self.initial),
            teacher_checkpoint=str(self.teacher_path), kd_reduction="tokenmean",
            early_stopping_patience=1, early_stopping_min_delta=100, max_eval_steps=1,
        )

        class TrainValidationOnly(dict):
            def __getitem__(self, key):
                if key == "test":
                    raise AssertionError("Test split must never be accessed")
                return super().__getitem__(key)

        self.data = TrainValidationOnly(train=examples(), validation=examples(3))
        self.students, self.teachers = [], []

        def load_student(config):
            model = TinyLM()
            self.students.append(model)
            return model

        def load_teacher(config):
            model = TinyLM()
            self.teachers.append(model)
            return model

        self.load_student = self.stack.enter_context(patch.object(
            tuning, "load_student", side_effect=load_student))
        self.load_teacher = self.stack.enter_context(patch.object(
            tuning, "load_teacher", side_effect=load_teacher))
        self.output = self.stack.enter_context(redirect_stdout(io.StringIO()))

    def run_trial(self, config=None, name="trial", **kwargs):
        return tuning.run_trial(config or self.config, self.data, self.root / name,
                                training_seed=42, **kwargs)

    def test_two_epochs_initial_quarters_best_and_atomic_artifacts(self):
        original = self.initial.read_bytes()
        with patch.object(tuning, "make_loader", wraps=tuning.make_loader) as loaders:
            result = self.run_trial()
        self.assertEqual(result["final_step"], 16)
        self.assertEqual([v["step"] for v in result["history"]], list(range(0, 17, 2)))
        self.assertEqual([v["epoch"] for v in result["history"]], [i / 4 for i in range(9)])
        self.assertEqual([c.kwargs["seed"] for c in loaders.call_args_list if "seed" in c.kwargs],
                         [43, 44])
        self.assertEqual(result["initial_validation"]["tokens"], 9)
        self.assertAlmostEqual(result["initial_validation"]["ce"], math.log(3), places=6)
        self.assertLess(result["best_validation"]["ce"], result["initial_validation"]["ce"])
        self.assertEqual(result["best_checkpoint"], str(self.root / "trial" / "best.pt"))
        self.assertGreater(result["best_step"], 0)
        self.assertEqual(result["final_validation"]["ce"], result["history"][-1]["ce"])
        self.assertEqual(result["peak_memory_gb"], 0.0)
        self.assertIsNone(result["matched_validation"])
        self.assertIsNone(result["matched_checkpoint"])
        self.load_teacher.assert_not_called()
        self.assertFalse(self.students[0].config.use_cache)
        self.assertTrue(any(training and grad for training, grad, _ in self.students[0].calls))
        self.assertEqual(original, self.initial.read_bytes())
        checkpoint = torch.load(result["best_checkpoint"], weights_only=True)
        self.assertEqual(set(checkpoint), set(self.students[0].state_dict()))
        directory = self.root / "trial"
        self.assertEqual(json.loads((directory / "result.json").read_text()), result)
        self.assertEqual(json.loads((directory / "validation_history.json").read_text()), result["history"])
        saved_config = json.loads((directory / "config.json").read_text())
        self.assertEqual(saved_config["training_seed"], 42)
        self.assertEqual(saved_config["validation_checks_per_epoch"], 4)
        self.assertIsNone(saved_config["runner"]["scheduler"])
        self.assertEqual(saved_config["runner"]["warmup_steps"], 0)
        self.assertFalse(saved_config["runner"]["early_stopping"])
        self.assertFalse(list(directory.glob("*.tmp")))
        self.assertFalse((directory / "error.json").exists())
        self.assertEqual(self.output.getvalue().count("validation step="), 9)

    def test_ce_only_never_loads_teacher_or_calls_kd_and_keeps_best_zero(self):
        with patch.object(tuning, "kd_loss", side_effect=AssertionError("CE-only called KD")):
            result = self.run_trial(replace(self.config, learning_rate=0.0))
        self.assertEqual(result["best_step"], 0)
        self.assertEqual(result["best_epoch"], 0.0)
        self.assertEqual(result["best_checkpoint"], str(self.initial))
        self.assertEqual(result["best_validation"], result["initial_validation"])
        self.assertEqual(result["final_step"], 16)  # Configured early stopping is ignored.
        self.assertFalse((self.root / "trial" / "best.pt").exists())
        self.load_teacher.assert_not_called()

    def test_matched_step_between_quarters_and_step_limit(self):
        result = self.run_trial(replace(self.config, max_train_steps=5), matched_step=3)
        self.assertEqual([v["step"] for v in result["history"]], [0, 2, 3, 4, 5])
        matched = next(v for v in result["history"] if v["step"] == 3)
        self.assertEqual(result["matched_validation"], {k: matched[k] for k in ("ce", "ppl", "tokens")})
        model = TinyLM()
        model.load_state_dict(torch.load(result["matched_checkpoint"], weights_only=True))
        measured = tuning.evaluate_ce(model, tuning.make_loader(self.data["validation"], 2), "cpu")
        self.assertEqual(result["matched_validation"], measured)
        self.assertEqual(result["final_step"], 5)
        self.assertEqual(result["history"][-1]["epoch"], 5 / 8)

    def test_matched_initial_step_saves_original_weights(self):
        result = self.run_trial(replace(self.config, max_train_steps=1), matched_step=0)
        self.assertEqual(result["matched_validation"], result["initial_validation"])
        original = torch.load(self.initial, weights_only=True)
        matched = torch.load(result["matched_checkpoint"], weights_only=True)
        self.assertTrue(all(torch.equal(original[k], matched[k]) for k in original))

    def test_kd_uses_existing_loss_and_frozen_teacher(self):
        with patch.object(tuning, "kd_loss", wraps=tuning.kd_loss) as kd:
            result = self.run_trial(replace(self.config, alpha=0.4, max_train_steps=2))
        self.assertEqual(kd.call_count, 2)
        self.assertEqual(result["final_step"], 2)
        self.load_teacher.assert_called_once()
        teacher = self.teachers[0]
        self.assertFalse(teacher.training)
        self.assertFalse(teacher.config.use_cache)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(not training and not grad for training, grad, _ in teacher.calls))
        self.assertFalse(torch.equal(self.students[0].projection.bias, teacher.projection.bias))

    def test_rejects_any_preexisting_directory_without_modifying_it(self):
        for name, artifact in (("empty", None), ("done", "result.json"), ("partial", "best.pt")):
            directory = self.root / name
            directory.mkdir()
            if artifact:
                (directory / artifact).write_text("keep")
            before = {p.name: p.read_bytes() for p in directory.iterdir()}
            with self.assertRaises(FileExistsError):
                self.run_trial(name=name)
            self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
        self.load_student.assert_not_called()

    def test_rejects_missing_or_directory_student_checkpoint(self):
        for index, path in enumerate(("", str(self.root / "missing.pt"), str(self.teacher_path))):
            with self.assertRaises(FileNotFoundError):
                self.run_trial(replace(self.config, distill_student_checkpoint=path), name=f"bad{index}")
            self.assertTrue((self.root / f"bad{index}" / "error.json").is_file())
            self.assertFalse((self.root / f"bad{index}" / "result.json").exists())
        self.load_student.assert_not_called()

    def test_rejects_missing_teacher_and_unreachable_matched_step(self):
        with self.assertRaises(FileNotFoundError):
            self.run_trial(replace(self.config, teacher_checkpoint=""), name="teacher_missing")
        with self.assertRaises(ValueError):
            self.run_trial(matched_step=17, name="unreachable")
        with self.assertRaises(ValueError):
            self.run_trial(matched_step=-1, name="negative")
        self.load_student.assert_not_called()

    def test_error_is_sanitized_and_complete_result_is_not_written(self):
        real_evaluate = tuning.evaluate_ce

        def evaluate_then_fail(*args, **kwargs):
            if len(self.students[0].calls) > 2:
                raise RuntimeError("secret-token-do-not-persist")
            return real_evaluate(*args, **kwargs)

        with patch.object(tuning, "evaluate_ce", side_effect=evaluate_then_fail):
            with self.assertRaisesRegex(RuntimeError, "secret-token"):
                self.run_trial()
        directory = self.root / "trial"
        self.assertFalse((directory / "result.json").exists())
        error = (directory / "error.json").read_text()
        self.assertNotIn("secret-token", error)
        self.assertEqual(json.loads(error)["error_type"], "RuntimeError")
        self.assertEqual(json.loads(error)["step"], 2)
        self.assertEqual([v["step"] for v in json.loads((directory / "validation_history.json").read_text())], [0])

    def test_training_seed_covers_python_numpy_and_torch(self):
        draws = []

        def seeded_student(config):
            draws.append((random.random(), float(np.random.random()), torch.rand(1).item()))
            return TinyLM()

        self.load_student.side_effect = seeded_student
        config = replace(self.config, max_train_steps=1)
        first = self.run_trial(config, name="first")
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        second = self.run_trial(config, name="second")
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(first["final_validation"], second["final_validation"])


if __name__ == "__main__":
    unittest.main()