"""Download-free CPU interruption tests, including real dropout and Adafactor."""
import json
import random
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.research_runner import train_stage
from src.resume_checkpoint import load_checkpoint

class DropoutLM(torch.nn.Module):
    def __init__(self, trace):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 7)
        self.dropout = torch.nn.Dropout(0.4)
        self.projection = torch.nn.Linear(7, 11)
        self.config = SimpleNamespace(use_cache=True)
        self.trace = trace

    def forward(self, input_ids, attention_mask=None):
        if self.training:
            self.trace.append((input_ids[:, 0].tolist(), random.random(),
                               float(np.random.random())))
        return SimpleNamespace(logits=self.projection(self.dropout(self.embedding(input_ids))))


def examples(count):
    return [{"input_ids": torch.tensor([i % 11, 2, 3, 4]),
             "labels": torch.tensor([-100, (i + 1) % 11, 3, 4])}
            for i in range(count)]


class ResumeTests(unittest.TestCase):
    """The interrupted draft's original acceptance case on the fixed engine API."""
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(7)
        self.initial = {k: v.clone() for k, v in DropoutLM([]).state_dict().items()}
        self.settings = dict(seed=42, batch_size=2, max_steps=10, epochs=2,
                             validation_interval=2, patience=0, min_delta=0.,
                             checkpoint_interval_steps=2, alpha=1., temperature=1.,
                             divergence='forward_kl', vocab_size=0, device='cpu')
        self.trace = []

    def run_stage(self, name, resume=False):
        from transformers.optimization import Adafactor
        model = DropoutLM(self.trace)
        model.load_state_dict(self.initial)
        optimizer = Adafactor(model.parameters(), lr=0.03, relative_step=False,
                              scale_parameter=False, warmup_init=False)
        return train_stage(model, examples(10), examples(3), optimizer, self.root / name,
                           settings=self.settings, identity={'base_revision': 'tiny-fixed'},
                           resume=resume)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_dropout_exact_continuation_after_optimizer_crash(self):
        from transformers.optimization import Adafactor
        reference = self.run_stage('reference')
        expected_trace = list(self.trace)
        expected = load_checkpoint(self.root / 'reference')
        self.trace.clear()
        actual_step = Adafactor.step
        calls = 0

        def crash_after_step(optimizer, *args, **kwargs):
            nonlocal calls
            result = actual_step(optimizer, *args, **kwargs)
            calls += 1
            if calls == 3:
                raise RuntimeError('injected interruption after optimizer mutation')
            return result

        with patch.object(Adafactor, 'step', new=crash_after_step):
            with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                self.run_stage('trial')
        self.assertEqual(load_checkpoint(self.root / 'trial')['step'], 2)
        durable_trace = self.trace[:2]
        self.trace.clear()
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        resumed = self.run_stage('trial', resume=True)
        self.assertEqual(durable_trace + self.trace, expected_trace)
        actual = load_checkpoint(self.root / 'trial')
        for key in ('model_state', 'optimizer', 'rng', 'cursor'):
            self.assert_nested_equal(actual[key], expected[key])
        for key in ('history', 'initial_validation', 'best_validation', 'best_step', 'final_step'):
            self.assertEqual(resumed[key], reference[key])
        self.assertFalse(list((self.root / 'trial').glob('*.tmp')))


# Verbatim interrupted draft retained for provenance; deliberately never executed.
LEGACY_H100_DRAFT = r'''"""Download-free CPU interruption tests, including real dropout and Adafactor."""
import io
import json
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


class DropoutLM(torch.nn.Module):
    def __init__(self, trace):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 7)
        self.dropout = torch.nn.Dropout(0.4)
        self.projection = torch.nn.Linear(7, 11)
        self.config = SimpleNamespace(use_cache=True)
        self.trace = trace

    def forward(self, input_ids, attention_mask=None):
        if self.training:
            self.trace.append((input_ids[:, 0].tolist(), random.random(),
                               float(np.random.random())))
        return SimpleNamespace(logits=self.projection(self.dropout(self.embedding(input_ids))))


def examples(count):
    return [{"input_ids": torch.tensor([i % 11, 2, 3, 4]),
             "labels": torch.tensor([-100, (i + 1) % 11, 3, 4])}
            for i in range(count)]


class ResumeTests(unittest.TestCase):
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
        self.initial = self.root / "initial.pt"
        torch.manual_seed(7)
        torch.save(DropoutLM([]).state_dict(), self.initial)
        self.teacher = self.root / "teacher"
        self.teacher.mkdir()
        (self.teacher / "adapter_config.json").write_text('{"base_model":"fixed"}')
        self.config = KDConfig(device="cpu", teacher_device="cpu", batch_size=2,
                               epochs=2, optimizer="adafactor", learning_rate=0.03,
                               alpha=1.0, distill_student_checkpoint=str(self.initial),
                               teacher_checkpoint=str(self.teacher))
        self.data = {"train": examples(10), "validation": examples(3)}
        self.trace, self.losses, self.optimizers = [], [], []
        self.stack.enter_context(patch.object(tuning, "load_student",
                                             side_effect=lambda _: DropoutLM(self.trace)))
        real_optimizer = tuning.create_optimizer

        def optimizer(*args):
            value = real_optimizer(*args)
            self.optimizers.append(value)
            return value

        self.stack.enter_context(patch.object(tuning, "create_optimizer", side_effect=optimizer))
        real_ce = tuning._ce_sum

        def ce(logits, labels):
            value = real_ce(logits, labels)
            if logits.requires_grad:
                self.losses.append(float(value.detach() / (labels[:, 1:] != -100).sum()))
            return value

        self.stack.enter_context(patch.object(tuning, "_ce_sum", side_effect=ce))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def run_trial(self, name="trial", **kwargs):
        config = kwargs.pop("config", self.config)
        data = kwargs.pop("data", self.data)
        return tuning.run_trial(config, data, self.root / name, training_seed=42,
                                validation_checks_per_epoch=2, matched_step=3,
                                checkpoint_interval_steps=2, **kwargs)

    def checkpoint(self, name):
        from src.resume_checkpoint import load_checkpoint
        return load_checkpoint(self.root / name / "last.pt")

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    @staticmethod
    def metrics(history):
        return [{k: v for k, v in row.items() if k != "elapsed_seconds"} for row in history]

    def test_dropout_exact_continuation_after_optimizer_crash(self):
        reference = self.run_trial("reference")
        expected_trace, expected_losses = list(self.trace), list(self.losses)
        expected = self.checkpoint("reference")
        self.trace.clear()
        self.losses.clear()
        actual_step = torch.optim.Optimizer.step  # replaced below on the real Adafactor
        from transformers.optimization import Adafactor
        actual_step = Adafactor.step
        calls = 0

        def crash_after_step(optimizer, *args, **kwargs):
            nonlocal calls
            result = actual_step(optimizer, *args, **kwargs)
            calls += 1
            if calls == 3:
                raise RuntimeError("injected interruption after optimizer mutation")
            return result

        with patch.object(Adafactor, "step", new=crash_after_step):
            with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                self.run_trial()
        self.assertEqual(self.checkpoint("trial")["step"], 2)
        durable_trace, durable_losses = self.trace[:2], self.losses[:2]
        self.trace.clear()
        self.losses.clear()
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        resumed = self.run_trial(resume=True)
        self.assertEqual(durable_trace + self.trace, expected_trace)
        self.assertEqual(durable_losses + self.losses, expected_losses)
        actual = self.checkpoint("trial")
        self.assert_nested_equal(actual["model"], expected["model"])
        self.assert_nested_equal(actual["optimizer"], expected["optimizer"])
        self.assertEqual(self.metrics(resumed["history"]), self.metrics(reference["history"]))
        for key in ("initial_validation", "final_validation", "best_validation", "best_step",
                    "best_epoch", "matched_validation", "final_step"):
            self.assertEqual(resumed[key], reference[key])
        self.assertFalse((self.root / "trial" / "error.json").exists())
        self.assertFalse(list((self.root / "trial").glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
'''

if __name__ == '__main__':
    unittest.main()
