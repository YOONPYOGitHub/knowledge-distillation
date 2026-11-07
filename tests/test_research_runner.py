"""Download-free CPU contract tests for the reusable research engine."""
import copy
import json
import math
import random
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 7)
        self.dropout = torch.nn.Dropout(0.4)
        self.head = torch.nn.Linear(7, 11)
        self.trace = []
        self.fail_at = None
        self.signal_at = None

    def forward(self, input_ids, attention_mask=None):
        if self.training:
            self.trace.append((input_ids[:, 0].tolist(), random.random(), float(np.random.random())))
            if len(self.trace) == self.fail_at:
                raise RuntimeError('injected crash')
            if len(self.trace) == self.signal_at:
                signal.raise_signal(signal.SIGTERM)
        return SimpleNamespace(logits=self.head(self.dropout(self.embedding(input_ids))))


def examples(n):
    return [{'input_ids': torch.tensor([i % 11, 2, 3, 4]),
             'attention_mask': torch.tensor([1, 1, 1, 1]),
             'labels': torch.tensor([-100, (i + 1) % 11, 3, -100 if i % 2 else 4])}
            for i in range(n)]


class EngineTests(unittest.TestCase):
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
        self.initial = copy.deepcopy(TinyLM().state_dict())
        self.train, self.valid = examples(9), examples(3)
        self.settings = dict(seed=42, batch_size=2, max_steps=8, epochs=2,
                             validation_interval=2, patience=0, min_delta=0.0,
                             checkpoint_interval_steps=2, alpha=1.0,
                             temperature=2.0, divergence='forward_kl', vocab_size=0,
                             device='cpu')
        self.identity = {'base': 'tiny', 'revision': 'fixed', 'adapter_hash': 'none'}

    def run_stage(self, name, *, model=None, optimizer=None, **kwargs):
        from src.research_runner import train_stage
        if model is None:
            model = TinyLM()
            model.load_state_dict(self.initial)
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        result = train_stage(model, self.train, self.valid, optimizer, self.root / name,
                             settings=kwargs.pop('settings', self.settings),
                             identity=kwargs.pop('identity', self.identity), **kwargs)
        return result, model, optimizer

    def test_ce_budget_token_weighted_validation_and_generation(self):
        result, model, _ = self.run_stage('ce')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['final_step'], 8)
        self.assertEqual([r['step'] for r in result['history']], [0, 2, 4, 6, 8])
        model.eval()
        numerator, tokens = 0.0, 0
        with torch.no_grad():
            for row in self.valid:
                logits = model(row['input_ids'][None], row['attention_mask'][None]).logits[:, :-1].float()
                labels = row['labels'][None, 1:]
                numerator += F.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1),
                                             ignore_index=-100, reduction='sum').item()
                tokens += int((labels != -100).sum())
        last = result['history'][-1]
        self.assertAlmostEqual(last['ce'], numerator / tokens, places=6)
        self.assertEqual(last['tokens'], tokens)
        self.assertAlmostEqual(last['ppl'], math.exp(last['ce']))
        json.dumps(result, allow_nan=False)
        checkpoint = Path(result['last_checkpoint'])
        self.assertTrue((checkpoint / 'COMPLETE').is_file())
        self.assertTrue((checkpoint / 'state.pt').is_file())
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        self.assertEqual(manifest['step'], 8)
        self.assertIn('state.pt', manifest['files'])
        pointer = json.loads((self.root / 'ce' / 'latest.json').read_text())
        self.assertEqual((self.root / 'ce' / pointer['checkpoint']).resolve(), checkpoint.resolve())
        self.assertTrue((Path(result['best_checkpoint']) / 'COMPLETE').is_file())

    def test_resume_refuses_changed_data_identity_settings_and_corruption_before_writes(self):
        from src.resume_checkpoint import load_checkpoint
        result, _, _ = self.run_stage('protected')
        root = self.root / 'protected'
        snapshot = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
        for key in ('input_ids', 'labels', 'attention_mask'):
            for dataset in (self.train, self.valid):
                original = dataset[0][key].clone()
                dataset[0][key][1] = 0
                with self.subTest(key=key, dataset=id(dataset)):
                    with self.assertRaisesRegex(ValueError, 'contract'):
                        self.run_stage('protected', resume=True)
                dataset[0][key] = original
        with self.assertRaisesRegex(ValueError, 'contract'):
            self.run_stage('protected', resume=True, identity={**self.identity, 'revision': 'new'})
        with self.assertRaisesRegex(ValueError, 'contract'):
            self.run_stage('protected', resume=True, settings={**self.settings, 'temperature': 4.0})
        with self.assertRaisesRegex(ValueError, 'resume'):
            self.run_stage('protected')
        self.assertEqual(snapshot, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()})
        payload = Path(result['last_checkpoint']) / 'state.pt'
        payload.write_bytes(payload.read_bytes() + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.run_stage('protected', resume=True)
        with self.assertRaisesRegex(ValueError, 'COMPLETE'):
            incomplete = root / 'checkpoints' / 'step-99999999-uncommitted'
            incomplete.mkdir()
            load_checkpoint(incomplete)
    def test_kd_forward_reverse_kl_temperature_mask_and_frozen_teacher(self):
        for divergence in ('forward_kl', 'reverse_kl'):
            for vocab in (0, 7):
                with self.subTest(divergence=divergence, vocab=vocab):
                    student = TinyLM()
                    student.load_state_dict(self.initial)
                    student.dropout.p = 0
                    expected = copy.deepcopy(student)
                    teacher = TinyLM()
                    teacher.dropout.p = 0
                    teacher.train()
                    teacher_before = copy.deepcopy(teacher.state_dict())
                    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
                    manual_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
                    order = torch.randperm(len(self.train), generator=torch.Generator().manual_seed(42)).tolist()[:2]
                    ids = torch.stack([self.train[i]['input_ids'] for i in order])
                    labels = torch.stack([self.train[i]['labels'] for i in order])
                    s = expected(ids).logits[:, :-1].float()
                    with torch.no_grad():
                        t = teacher(ids).logits[:, :-1].float()
                    ce = F.cross_entropy(s.reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
                    limit = vocab or 11
                    log_s, log_t = (s[..., :limit] / 2).log_softmax(-1), (t[..., :limit] / 2).log_softmax(-1)
                    if divergence == 'forward_kl':
                        kd = (log_t.exp() * (log_t - log_s)).sum(-1)
                    else:
                        kd = (log_s.exp() * (log_s - log_t)).sum(-1)
                    kd = kd[labels[:, 1:] != -100].mean() * 4
                    (0.3 * ce + 0.7 * kd).backward()
                    manual_optimizer.step()
                    settings = {**self.settings, 'max_steps': 1, 'alpha': 0.3,
                                'divergence': divergence, 'vocab_size': vocab}
                    result, _, _ = self.run_stage(f'{divergence}-{vocab}', model=student,
                                                  optimizer=optimizer, teacher=teacher, settings=settings)
                    self.assert_nested_equal(student.state_dict(), expected.state_dict())
                    self.assert_nested_equal(teacher.state_dict(), teacher_before)
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
                    self.assertFalse(teacher.training)
                    self.assertEqual(result['final_step'], 1)

    def test_validation_only_patience_selects_step_zero(self):
        model = TinyLM()
        model.load_state_dict(self.initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        result, _, _ = self.run_stage('patience', model=model, optimizer=optimizer,
                                      settings={**self.settings, 'patience': 2})
        self.assertEqual(result['final_step'], 4)
        self.assertEqual(result['best_step'], 0)
        self.assertEqual(result['best_validation'], result['initial_validation'])
        self.assertEqual(result['stop_reason'], 'early_stopping')
        self.assertEqual([r['step'] for r in result['history']], [0, 2, 4])
        loaded_model = TinyLM()
        loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.0)
        resumed, _, _ = self.run_stage('patience', resume=True, model=loaded_model,
                                       optimizer=loaded_optimizer,
                                       settings={**self.settings, 'patience': 2})
        self.assertEqual(resumed, result)

    def test_sigterm_safe_boundary_resume_and_handler_restored(self):
        reference, _, _ = self.run_stage('signal-reference')
        interrupted = TinyLM()
        interrupted.load_state_dict(self.initial)
        interrupted.signal_at = 3
        previous = signal.getsignal(signal.SIGTERM)
        safe_handler = lambda *_: None
        signal.signal(signal.SIGTERM, safe_handler)
        try:
            result, _, _ = self.run_stage('signal', model=interrupted)
            self.assertIs(signal.getsignal(signal.SIGTERM), safe_handler)
            self.assertEqual(result['status'], 'interrupted')
            self.assertEqual(result['final_step'], 3)
            self.assertEqual(result['stop_reason'], 'sigterm')
            resumed, _, _ = self.run_stage('signal', resume=True)
            self.assertEqual(resumed['history'], reference['history'])
            from src.resume_checkpoint import load_checkpoint
            actual = load_checkpoint(Path(resumed['last_checkpoint']))
            expected = load_checkpoint(Path(reference['last_checkpoint']))
            self.assert_nested_equal(actual['model_state'], expected['model_state'])
            crash = TinyLM()
            crash.load_state_dict(self.initial)
            crash.fail_at = 1
            with self.assertRaisesRegex(RuntimeError, 'injected crash'):
                self.run_stage('signal-crash', model=crash)
            self.assertIs(signal.getsignal(signal.SIGTERM), safe_handler)
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_manifest_payload_sha_size_and_pointer_manifest_hash(self):
        import hashlib
        result, _, _ = self.run_stage('manifest')
        generation = Path(result['last_checkpoint'])
        manifest = json.loads((generation / 'manifest.json').read_text())
        payload = (generation / 'state.pt').read_bytes()
        self.assertEqual(manifest['files']['state.pt'],
                         {'sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload)})
        pointer = json.loads((self.root / 'manifest' / 'latest.json').read_text())
        self.assertEqual(pointer['manifest_sha256'], hashlib.sha256((generation / 'manifest.json').read_bytes()).hexdigest())

    def test_rejects_unsupported_or_invalid_settings_without_output(self):
        cases = [dict(gradient_accumulation_steps=2), dict(scheduler='cosine'),
                 dict(test_dataset=[]), dict(max_steps=0), dict(epochs=0),
                 dict(batch_size=0), dict(validation_interval=0),
                 dict(checkpoint_interval_steps=0), dict(patience=-1),
                 dict(min_delta=-0.1), dict(alpha=1.1), dict(alpha=-0.1),
                 dict(alpha=0.5), dict(temperature=0), dict(divergence='js'),
                 dict(vocab_size=-1), dict(max_steps=True)]
        for i, extra in enumerate(cases):
            with self.subTest(extra=extra):
                with self.assertRaises((ValueError, TypeError)):
                    self.run_stage(f'invalid-{i}', settings={**self.settings, **extra})
                self.assertFalse((self.root / f'invalid-{i}').exists())
        from src.research_runner import train_stage
        with self.assertRaises(TypeError):
            train_stage(None, [], [], None, self.root / 'test', test_dataset=[])

    def test_relocated_resume_and_corrupt_best_refusal(self):
        import shutil
        result, _, _ = self.run_stage('original', settings={**self.settings, 'min_delta': 100.0})
        shutil.copytree(self.root / 'original', self.root / 'relocated')
        shutil.rmtree(self.root / 'original')
        resumed, _, _ = self.run_stage('relocated', resume=True,
                                       settings={**self.settings, 'min_delta': 100.0})
        best = Path(resumed['best_checkpoint'])
        self.assertTrue(best.is_relative_to((self.root / 'relocated').resolve()))
        self.assertTrue((best / 'COMPLETE').is_file())
        self.assertEqual(resumed['best_step'], 0)
        self.assertNotEqual(best, Path(resumed['last_checkpoint']))
        (best / 'state.pt').write_bytes(b'corrupt best')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.run_stage('relocated', resume=True, settings={**self.settings, 'min_delta': 100.0})

    def test_trainable_only_buffers_schema_and_optimizer_identity(self):
        from src.resume_checkpoint import load_checkpoint
        model = TinyLM()
        model.load_state_dict(self.initial)
        model.embedding.requires_grad_(False)
        model.register_buffer('required_buffer', torch.arange(3))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.03)
        result, _, _ = self.run_stage('adapter', model=model, optimizer=optimizer)
        payload = load_checkpoint(Path(result['last_checkpoint']))
        self.assertEqual(set(payload['model_state']), {'head.weight', 'head.bias', 'required_buffer'})
        bad = TinyLM()
        bad.embedding.requires_grad_(False)
        bad.register_buffer('required_buffer', torch.arange(3))
        wrong_optimizer = torch.optim.SGD([p for p in bad.parameters() if p.requires_grad], lr=0.03)
        with self.assertRaisesRegex(ValueError, 'contract'):
            self.run_stage('adapter', model=bad, optimizer=wrong_optimizer, resume=True)
        wrong_shape = TinyLM()
        wrong_shape.embedding.requires_grad_(False)
        wrong_shape.register_buffer('required_buffer', torch.arange(4))
        with self.assertRaisesRegex(ValueError, 'shape|contract'):
            self.run_stage('adapter', model=wrong_shape, resume=True)
        missing = TinyLM()
        missing.embedding.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'keys|contract'):
            self.run_stage('adapter', model=missing, resume=True)

    def test_rejects_alias_teacher_and_nonfinite_training(self):
        model = TinyLM()
        model.load_state_dict(self.initial)
        with self.assertRaisesRegex(ValueError, 'teacher'):
            self.run_stage('alias', model=model, teacher=model,
                           settings={**self.settings, 'alpha': 0.5})
        self.assertFalse((self.root / 'alias').exists())
        bad = TinyLM()
        bad.load_state_dict(self.initial)
        optimizer = torch.optim.SGD(bad.parameters(), lr=float('nan'))
        with self.assertRaises((ValueError, FloatingPointError)):
            self.run_stage('bad-lr', model=bad, optimizer=optimizer)
        self.assertFalse((self.root / 'bad-lr').exists())

    def test_nonfinite_training_loss_cannot_publish_a_mutated_checkpoint(self):
        class Nonfinite(TinyLM):
            def forward(self, *args, **kwargs):
                result = super().forward(*args, **kwargs)
                if self.training:
                    result.logits = result.logits * float('nan')
                return result
        bad = Nonfinite()
        bad.load_state_dict(self.initial)
        with self.assertRaisesRegex(FloatingPointError, 'finite'):
            self.run_stage('nan', model=bad, settings={**self.settings, 'checkpoint_interval_steps': 1})
        from src.resume_checkpoint import load_checkpoint
        self.assertEqual(load_checkpoint(self.root / 'nan')['step'], 0)

    def test_empty_masked_or_nonfinite_validation_refuses_initial_checkpoint(self):
        original = self.valid
        cases = [[], [{'input_ids': [1, 2], 'labels': [-100, -100]}]]
        for i, data in enumerate(cases):
            self.valid = data
            with self.assertRaisesRegex(ValueError, 'validation|empty'):
                self.run_stage(f'bad-validation-{i}')
            self.assertFalse((self.root / f'bad-validation-{i}').exists())
        self.valid = original
        model = TinyLM()
        model.load_state_dict(self.initial)
        with torch.no_grad():
            model.head.weight.fill_(float('nan'))
        with self.assertRaisesRegex(FloatingPointError, 'finite'):
            self.run_stage('nan-validation', model=model)
        self.assertFalse((self.root / 'nan-validation').exists())

    def test_empty_existing_output_requires_explicit_resume(self):
        (self.root / 'empty').mkdir()
        with self.assertRaisesRegex(ValueError, 'resume'):
            self.run_stage('empty')
        with self.assertRaisesRegex(ValueError, 'COMPLETE'):
            self.run_stage('empty', resume=True)
        self.assertEqual(list((self.root / 'empty').iterdir()), [])

    def test_empty_training_refused_without_checkpoint(self):
        self.train = []
        with self.assertRaisesRegex(ValueError, 'empty|training'):
            self.run_stage('empty-training')
        self.assertFalse((self.root / 'empty-training').exists())

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_dropout_optimizer_rng_exact_resume_and_idempotence(self):
        from src.resume_checkpoint import load_checkpoint
        reference, reference_model, _ = self.run_stage('reference')
        crash = TinyLM()
        crash.load_state_dict(self.initial)
        crash.fail_at = 3
        with self.assertRaisesRegex(RuntimeError, 'injected crash'):
            self.run_stage('crash', model=crash)
        durable = load_checkpoint(self.root / 'crash')
        self.assertEqual(durable['step'], 2)
        resumed, resumed_model, _ = self.run_stage('crash', resume=True)
        self.assertEqual(crash.trace[:2] + resumed_model.trace, reference_model.trace)
        actual = load_checkpoint(Path(resumed['last_checkpoint']))
        expected = load_checkpoint(Path(reference['last_checkpoint']))
        for key in ('model_state', 'optimizer', 'rng', 'history', 'best_step', 'best_validation', 'cursor'):
            self.assert_nested_equal(actual[key], expected[key])
        self.assertEqual(resumed['initial_validation'], {k: v for k, v in reference['history'][0].items() if k != 'step'})
        paths = sorted(str(p) for p in (self.root / 'crash').rglob('*'))
        again, loaded, _ = self.run_stage('crash', resume=True)
        self.assertEqual(again, resumed)
        self.assertEqual(loaded.trace, [])
        self.assertEqual(paths, sorted(str(p) for p in (self.root / 'crash').rglob('*')))
        self.assert_nested_equal(loaded.state_dict(), resumed_model.state_dict())


if __name__ == '__main__':
    unittest.main()
