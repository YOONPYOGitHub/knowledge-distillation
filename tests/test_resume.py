"""Epoch 단위 재개: 끊었다가 이어간 학습이 끊지 않은 학습과 같아야 한다."""

import tempfile
import unittest

import torch

from src.config import KDConfig
from src.models import create_optimizer
from src.resume import clear_state, load_state, save_state, state_path


def make_config(output_dir, **overrides):
    values = dict(
        device="cpu", output_dir=output_dir, run_id="resume-test", resume_training=True,
        optimizer="adafactor", learning_rate=1e-3,
    )
    values.update(overrides)
    config = KDConfig(**values)
    config.ensure_dirs()
    return config


def make_model(frozen_head=False):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4)).to(torch.bfloat16)
    if frozen_head:
        model[1].requires_grad_(False)
    return model


def train_epoch(model, optimizer, epoch):
    generator = torch.Generator().manual_seed(epoch)
    for _ in range(3):
        x = torch.randn(4, 8, generator=generator).to(torch.bfloat16)
        loss = model(x).float().pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_resumed_training_matches_uninterrupted(self):
        config = make_config(self.tmp.name)

        straight = make_model()
        optimizer = create_optimizer(straight, config, config.learning_rate)
        for epoch in (1, 2):
            train_epoch(straight, optimizer, epoch)

        first = make_model()
        optimizer = create_optimizer(first, config, config.learning_rate)
        train_epoch(first, optimizer, 1)
        save_state(config, "distill", first, optimizer, 1, 2.5, 0, [{"epoch": 1}])

        resumed = make_model()
        optimizer = create_optimizer(resumed, config, config.learning_rate)
        state = load_state(config, "distill", resumed, optimizer)
        self.assertEqual((state["epoch"], state["best_val_loss"]), (1, 2.5))
        self.assertEqual(state["history"], [{"epoch": 1}])
        train_epoch(resumed, optimizer, 2)

        for a, b in zip(straight.parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_optimizer_state_keeps_fp32_moments(self):
        config = make_config(self.tmp.name)
        model = make_model()
        optimizer = create_optimizer(model, config, config.learning_rate)
        train_epoch(model, optimizer, 1)
        save_state(config, "distill", model, optimizer, 1, 1.0, 0, [])

        fresh = make_model()
        fresh_optimizer = create_optimizer(fresh, config, config.learning_rate)
        load_state(config, "distill", fresh, fresh_optimizer)
        for p_old, p_new in zip(model.parameters(), fresh.parameters()):
            old, new = optimizer.state[p_old], fresh_optimizer.state[p_new]
            for key, value in old.items():
                if torch.is_tensor(value):
                    self.assertEqual(new[key].dtype, value.dtype, key)
                    self.assertTrue(torch.equal(new[key], value), key)

    def test_only_trainable_parameters_are_saved(self):
        config = make_config(self.tmp.name)
        model = make_model(frozen_head=True)
        optimizer = create_optimizer(model, config, config.learning_rate)
        save_state(config, "teacher", model, optimizer, 1, 1.0, 0, [])
        saved = torch.load(state_path(config, "teacher"), weights_only=True)
        self.assertEqual(set(saved["trainable"]), {"0.weight", "0.bias"})

    def test_changed_config_starts_fresh(self):
        config = make_config(self.tmp.name)
        model = make_model()
        optimizer = create_optimizer(model, config, config.learning_rate)
        save_state(config, "distill", model, optimizer, 1, 1.0, 0, [])

        changed = make_config(self.tmp.name, learning_rate=5e-4)
        self.assertIsNone(load_state(changed, "distill", model, optimizer))
        # 학습 길이와 장치는 바꿔도 이어간다.
        longer = make_config(self.tmp.name, epochs=8, early_stopping_patience=2)
        self.assertIsNotNone(load_state(longer, "distill", model, optimizer))

    def test_disabled_writes_nothing_and_clear_removes(self):
        disabled = make_config(self.tmp.name, resume_training=False)
        model = make_model()
        optimizer = create_optimizer(model, disabled, disabled.learning_rate)
        save_state(disabled, "baseline", model, optimizer, 1, 1.0, 0, [])
        self.assertFalse(state_path(disabled, "baseline").exists())

        enabled = make_config(self.tmp.name)
        save_state(enabled, "baseline", model, optimizer, 1, 1.0, 0, [])
        self.assertTrue(state_path(enabled, "baseline").exists())
        clear_state(enabled, "baseline")
        self.assertFalse(state_path(enabled, "baseline").exists())


if __name__ == "__main__":
    unittest.main()
