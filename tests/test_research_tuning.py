"""Validation-only sweep tests; every metric comes from the real CPU engine."""
import json
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace


class TinyTokenizer:
    eos_token_id = 7
    special_tokens_map = {'eos_token': '<eos>'}

    def __init__(self):
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: '{}')

    def get_vocab(self):
        return {str(i): i for i in range(8)}

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 5)
        self.dropout = torch.nn.Dropout(.2)
        self.head = torch.nn.Linear(5, 8)
        self.signal_at = None
        self.steps = 0

    def forward(self, input_ids, attention_mask=None):
        if self.training:
            self.steps += 1
            if self.steps == self.signal_at:
                import signal
                signal.raise_signal(signal.SIGTERM)
        return SimpleNamespace(logits=self.head(self.dropout(self.embedding(input_ids))))


@pytest.fixture
def base(tmp_path, request):
    import yaml
    from scripts.run_research import main
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'configs/research_qwen35.yaml').read_text())
    config['data'].update(max_documents=10, seq_length=4)
    if getattr(request, 'param', False):
        config['data']['exposure_exclusion'] = {'source_ids': ['pilot-id'], 'text_sha256': ['a' * 64]}
    config['training'].update(device='cpu', max_steps=2, validation_interval=1,
                              checkpoint_interval_steps=1)
    cfg = tmp_path / 'base.yaml'
    cfg.write_text(yaml.safe_dump(config))
    trace = []

    def model(spec, **kw):
        torch.manual_seed(17)
        result = TinyLM()
        trace.append((spec['id'], result))
        return result

    deps = dict(source=lambda n: [{'id': str(i), 'text': '한국어 문서 자료입니다. ' * 30 + str(i)}
                                 for i in range(n)],
                tokenizer=lambda spec: TinyTokenizer(), model=model,
                adapter=lambda model, adapter, seed: model)
    directory = tmp_path / 'base'
    main(['teacher', '--config', str(cfg), '--run-dir', str(directory)], dependencies=deps)
    trace.clear()
    yield directory, deps, trace
    torch.set_num_threads(previous)


def snapshot(directory):
    return {str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in directory.rglob('*') if p.is_file()}


@pytest.mark.parametrize('base', [True], indirect=True)
def test_load_base_preserves_frozen_exposure_identity(base, monkeypatch):
    from src.research_tuning import load_base
    import src.research_data as data
    directory, deps, _ = base
    before = snapshot(directory)
    original = data.pack_split

    def pack(path, *args):
        assert Path(path).name != 'test.jsonl'
        return original(path, *args)

    monkeypatch.setattr(data, 'pack_split', pack)
    config, prepared, datasets, teacher = load_base(directory, deps)
    assert prepared['raw']['identity']['exposure_exclusion'] == config['data']['exposure_exclusion']
    assert set(datasets) == {'train', 'validation'}
    assert before == snapshot(directory)


def small_trials():
    from src.research_tuning import default_trials
    return [default_trials()[0], default_trials()[4]]


def test_real_sweep_freezes_plan_and_selection_never_touches_test_or_base(base, tmp_path, monkeypatch):
    from src import research_tuning as tuning
    import src.research_data as data
    import scripts.run_research as cli
    directory, deps, trace = base
    before = snapshot(directory)
    real_pack = data.pack_split
    packed = []

    def pack(path, *args):
        assert Path(path).name != 'test.jsonl', 'test tokenization forbidden'
        packed.append(Path(path).name)
        return real_pack(path, *args)

    def forbidden(*a, **kw):
        raise AssertionError('test evaluation forbidden')

    monkeypatch.setattr(data, 'pack_split', pack)
    monkeypatch.setattr(cli, 'pack_split', pack)
    monkeypatch.setattr(cli, 'evaluate', forbidden)
    output = tmp_path / 'sweep'
    result = tuning.run_sweep(directory, output, max_steps=2, dependencies=deps,
                              trials=small_trials())
    assert result['status'] == 'completed'
    assert len(result['trials']) == 2
    plan = json.loads((output / 'plan.json').read_text())
    assert plan['initial_adapter_sha256']
    assert plan['numerical_policy']['deterministic_algorithms'] is True
    assert all(r['contract']['identity']['numerical_policy'] == plan['numerical_policy'] for r in result['trials'])
    assert plan['provenance']['base_config_hash']
    assert plan['provenance']['teacher']['manifest_sha256']
    assert plan['provenance']['prepared_fingerprint']
    assert plan['common_settings']['max_steps'] == 2
    assert plan['common_settings']['seed'] == 42
    assert len({r['contract']['identity']['initial_adapter_sha256'] for r in result['trials']}) == 1
    for row in result['trials']:
        assert row['result']['final_step'] == 2
        assert row['result']['best_validation']['ppl'] > 0
        assert row['result']['best_validation']['tokens'] > 0
        assert (Path(row['result']['best_checkpoint']) / 'COMPLETE').is_file()
    assert packed == ['train.jsonl', 'validation.jsonl']
    assert before == snapshot(directory)
    selection = json.loads((output / 'selection.json').read_text())
    for method in ('ce', 'kd'):
        candidate = min((r for r in result['trials'] if r['trial']['method'] == method),
                        key=lambda r: (r['result']['best_validation']['ce'], r['trial']['id']))
        assert selection['methods'][method]['trial_id'] == candidate['trial']['id']
        assert selection['methods'][method]['best_validation'] == candidate['result']['best_validation']
    assert 'no effectiveness conclusion' in result['interpretation']
    assert result['test_evaluated'] is False
    assert not (output / 'test-results.json').exists()
    assert (output / 'trials.csv').read_text().count('\n') == 3
    assert 'not a blind test' in (output / 'summary.md').read_text()
    assert [role for role, _ in trace].count('Qwen/Qwen3.5-9B') == 1
    assert not trace[-1][1].training
    assert all(not p.requires_grad for p in trace[-1][1].parameters())


def assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


def test_sigterm_sweep_resumes_exact_optimizer_rng_cursor_and_is_idempotent(base, tmp_path):
    from src.research_tuning import run_sweep
    from src.research_runner import train_stage
    from src.resume_checkpoint import load_checkpoint
    directory, deps, trace = base
    reference = run_sweep(directory, tmp_path / 'reference', max_steps=2,
                          dependencies=deps, trials=small_trials())
    output = tmp_path / 'interrupted'
    def interrupt(model, *args, **kw):
        model.signal_at = 1
        return train_stage(model, *args, **kw)
    partial = run_sweep(directory, output, max_steps=2,
                        dependencies={**deps, 'runner': interrupt}, trials=small_trials())
    assert partial['status'] == 'interrupted'
    assert partial['trials'][0]['result']['final_step'] == 1
    assert not (output / 'selection.json').exists()
    plan_before = (output / 'plan.json').read_bytes()
    resumed = run_sweep(directory, output, resume=True, max_steps=2,
                        dependencies=deps, trials=small_trials())
    assert resumed['status'] == 'completed'
    assert (output / 'plan.json').read_bytes() == plan_before
    for actual, expected in zip(resumed['trials'], reference['trials']):
        a = load_checkpoint(actual['result']['last_checkpoint'])
        e = load_checkpoint(expected['result']['last_checkpoint'])
        for key in ('model_state', 'optimizer', 'rng', 'history', 'best_step', 'best_validation', 'cursor'):
            assert_nested_equal(a[key], e[key])
    before = snapshot(output)
    trace.clear()
    def forbidden(*args, **kw):
        raise AssertionError('completed trials must not be trained again')
    assert run_sweep(directory, output, resume=True, max_steps=2,
                     dependencies={**deps, 'runner': forbidden}, trials=small_trials()) == resumed
    assert before == snapshot(output)
    assert not any(role == 'Qwen/Qwen3.5-9B' for role, _ in trace)


def test_resume_contract_plan_budget_and_corrupt_completed_checkpoint_refused(base, tmp_path):
    from src.research_tuning import run_sweep
    directory, deps, _ = base
    output = tmp_path / 'sweep'
    result = run_sweep(directory, output, max_steps=2, dependencies=deps, trials=small_trials())
    before = snapshot(output)
    for kwargs in ({'max_steps': 3}, {'max_steps': 2, 'trials': list(reversed(small_trials()))}):
        kwargs.setdefault('trials', small_trials())
        with pytest.raises(ValueError, match='plan'):
            run_sweep(directory, output, resume=True, dependencies=deps, **kwargs)
        assert before == snapshot(output)
    with pytest.raises(ValueError, match='resume'):
        run_sweep(directory, output, dependencies=deps)
    path = Path(result['trials'][0]['result']['best_checkpoint']) / 'state.pt'
    path.write_bytes(b'corrupt')
    corrupted = snapshot(output)
    with pytest.raises(ValueError, match='hash'):
        run_sweep(directory, output, resume=True, max_steps=2, dependencies=deps, trials=small_trials())
    assert corrupted == snapshot(output)


def test_cli_runs_all_eight_trials_and_help_is_lightweight(base, tmp_path):
    import os
    import subprocess
    import sys
    from scripts.tune_research import main, parse_args
    from src.research_tuning import default_trials
    directory, deps, _ = base
    output = tmp_path / 'cli'
    args = ['--base-run', str(directory), '--output-dir', str(output), '--max-steps', '2']
    report = main(args, dependencies=deps)
    assert report['status'] == 'completed'
    assert [r['trial'] for r in report['trials']] == default_trials()
    assert all(r['result']['final_step'] == 2 for r in report['trials'])
    assert main(args + ['--resume'], dependencies=deps) == report
    assert parse_args(args[:-2]).max_steps == 200
    root = Path(__file__).resolve().parents[1]
    probe = """import sys,runpy
sys.addaudithook(lambda event,args: (_ for _ in ()).throw(AssertionError('heavy import')) if event=='import' and args[0].split('.')[0] in ('torch','transformers','peft','datasets') else None)
sys.argv=['tune_research.py','--help']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-c', probe], cwd=root / 'scripts',
        capture_output=True, text=True, timeout=20, env={**os.environ, 'HF_HUB_OFFLINE': '1'})
    assert result.returncode == 0, result.stderr
    assert all(flag in result.stdout for flag in ('--base-run', '--output-dir', '--resume', '--max-steps'))
    assert all(flag in result.stdout for flag in ('--trials-file', '--seed',
        '--validation-interval', '--patience', '--checkpoint-interval'))


def test_unbalanced_duplicate_or_path_escaping_trial_injections_are_rejected_before_training(base, tmp_path):
    from src.research_tuning import run_sweep
    directory, deps, trace = base
    ce, kd = small_trials()
    for index, trials in enumerate(([ce], [ce, ce, kd, kd], [{**ce, 'id': '../escape'}, kd])):
        with pytest.raises(ValueError, match='trial'):
            run_sweep(directory, tmp_path / f'invalid-{index}', max_steps=2,
                      dependencies=deps, trials=trials)
    assert trace == []


def test_frozen_selection_never_allows_retraining_missing_trial(base, tmp_path):
    import shutil
    from src.research_tuning import run_sweep
    directory, deps, _ = base
    output = tmp_path / 'sweep'
    run_sweep(directory, output, max_steps=2, dependencies=deps, trials=small_trials())
    shutil.rmtree(output / 'trials/kd-1')
    before = snapshot(output)
    with pytest.raises(ValueError, match='selection'):
        run_sweep(directory, output, resume=True, max_steps=2, dependencies=deps, trials=small_trials())
    assert before == snapshot(output)


def test_initial_adapter_hash_refuses_stochastic_student_reinitialization(base, tmp_path):
    from src.research_tuning import run_sweep
    directory, deps, _ = base
    calls = []
    original = deps['model']
    def changed(spec, **kwargs):
        model = original(spec, **kwargs)
        if calls:
            with torch.no_grad():
                model.head.bias.add_(1)
        calls.append(spec['id'])
        return model
    output = tmp_path / 'unpaired'
    with pytest.raises(ValueError, match='initial adapter hash'):
        run_sweep(directory, output, max_steps=2, dependencies={**deps, 'model': changed}, trials=small_trials())
    assert (output / 'trials/ce-1/result.json').exists()
    assert not (output / 'selection.json').exists()


@pytest.mark.parametrize('target', ['prepared', 'raw', 'teacher', 'config'])
def test_changed_base_provenance_is_rejected_on_resume(base, tmp_path, target):
    from src.research_tuning import run_sweep
    directory, deps, _ = base
    output = tmp_path / 'sweep'
    run_sweep(directory, output, max_steps=2, dependencies=deps, trials=small_trials())
    if target == 'prepared':
        path = directory / 'prepared.json'
        value = json.loads(path.read_text())
        value['fingerprint'] = 'tampered'
        path.write_text(json.dumps(value))
    elif target == 'raw':
        path = directory / 'raw/train.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
    elif target == 'teacher':
        record = json.loads((directory / 'teacher.json').read_text())
        path = Path(record['result']['best_checkpoint']) / 'state.pt'
        path.write_bytes(b'corrupted teacher')
    else:
        path = directory / 'config.json'
        value = json.loads(path.read_text())
        value['mode'] = 'changed'
        path.write_text(json.dumps(value))
    before = snapshot(output)
    with pytest.raises(ValueError):
        run_sweep(directory, output, resume=True, max_steps=2, dependencies=deps, trials=small_trials())
    assert before == snapshot(output)


def test_completed_metrics_and_selection_tampering_cannot_change_selection(base, tmp_path):
    from src.research_tuning import run_sweep
    directory, deps, _ = base
    output = tmp_path / 'sweep'
    run_sweep(directory, output, max_steps=2, dependencies=deps, trials=small_trials())
    path = output / 'trials/ce-1/result.json'
    original = path.read_bytes()
    record = json.loads(original)
    record['result']['best_validation']['ce'] = 0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='result/checkpoint'):
        run_sweep(directory, output, resume=True, max_steps=2, dependencies=deps, trials=small_trials())
    path.write_bytes(original)
    selection_path = output / 'selection.json'
    selection = json.loads(selection_path.read_text())
    selection['methods']['ce']['trial_id'] = 'kd-1'
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='selection'):
        run_sweep(directory, output, resume=True, max_steps=2, dependencies=deps, trials=small_trials())


def test_output_lock_and_unsafe_output_scope_prevent_writes(base, tmp_path):
    from scripts.run_research import run_lock
    from src.research_tuning import run_sweep
    directory, deps, trace = base
    for unsafe in (directory, directory / 'sweep', tmp_path):
        with pytest.raises(ValueError, match='separate'):
            run_sweep(directory, unsafe, max_steps=2, dependencies=deps, trials=small_trials())
    output = tmp_path / 'sweep'
    run_sweep(directory, output, max_steps=2, dependencies=deps, trials=small_trials())
    before = snapshot(output)
    trace.clear()
    with run_lock(output):
        with pytest.raises(RuntimeError, match='locked'):
            run_sweep(directory, output, resume=True, max_steps=2, dependencies=deps, trials=small_trials())
    assert before == snapshot(output)
    assert trace == []


def test_kd_interruption_restores_selected_teacher_and_recovers_missing_result_record(base, tmp_path):
    from src.research_tuning import run_sweep
    from src.research_runner import train_stage
    from src.resume_checkpoint import load_checkpoint
    directory, deps, _ = base
    selected = json.loads((directory / 'teacher.json').read_text())
    weights = load_checkpoint(selected['result']['best_checkpoint'])['model_state']
    seen = []
    def interrupt_kd(model, *args, **kwargs):
        teacher = kwargs['teacher']
        if teacher is not None:
            assert not teacher.training
            assert all(not p.requires_grad for p in teacher.parameters())
            for name, tensor in weights.items():
                assert torch.equal(teacher.state_dict()[name], tensor)
            seen.append(kwargs['resume'])
            if not kwargs['resume']:
                model.signal_at = 1
        return train_stage(model, *args, **kwargs)
    output = tmp_path / 'kd-interrupt'
    partial = run_sweep(directory, output, max_steps=2,
                       dependencies={**deps, 'runner': interrupt_kd}, trials=small_trials())
    assert [r['result']['status'] for r in partial['trials']] == ['completed', 'interrupted']
    (output / 'trials/kd-1/result.json').unlink()  # crash after durable commit, before result metadata
    resumed = run_sweep(directory, output, resume=True, max_steps=2,
                       dependencies={**deps, 'runner': interrupt_kd}, trials=small_trials())
    assert resumed['status'] == 'completed'
    assert seen == [False, True]
    assert [r['result']['final_step'] for r in resumed['trials']] == [2, 2]


def test_plan_is_committed_before_first_optimizer_update_and_all_trials_continue(base, tmp_path):
    from src.research_tuning import run_sweep, default_trials
    from src.research_runner import train_stage
    directory, deps, _ = base
    output = tmp_path / 'frozen-before-step'
    seen = []
    def checked(model, train, validation, optimizer, trial_dir, **kwargs):
        plan = json.loads((output / 'plan.json').read_text())
        assert plan['trials'] == default_trials()
        assert not (output / 'selection.json').exists()
        seen.append(kwargs['identity']['trial_id'])
        return train_stage(model, train, validation, optimizer, trial_dir, **kwargs)
    result = run_sweep(directory, output, max_steps=2, dependencies={**deps, 'runner': checked})
    assert seen == [trial['id'] for trial in default_trials()]
    assert len(result['trials']) == 8
    assert all(r['result']['final_step'] == 2 for r in result['trials'])


def test_round_one_plan_is_balanced_and_budget_is_shared():
    from src import research_tuning as tuning
    trials = tuning.default_trials()
    assert len(trials) == 8
    ce = [t for t in trials if t['method'] == 'ce']
    kd = [t for t in trials if t['method'] == 'kd']
    assert [t['learning_rate'] for t in ce] == [2e-5, 5e-5, 1e-4, 2e-4]
    assert [(t['learning_rate'], t['alpha'], t['temperature'], t['divergence']) for t in kd] == [
        (1e-4, .7, 1., 'forward_kl'), (5e-5, .7, 2., 'forward_kl'),
        (5e-5, .7, 1., 'reverse_kl'), (2e-5, .9, 1., 'reverse_kl')]
    assert len({t['id'] for t in trials}) == 8
    settings = tuning.sweep_settings({'device': 'cpu', 'batch_size': 1}, 200, 8)
    assert settings == dict(seed=42, device='cpu', batch_size=1, max_steps=200,
        epochs=100, validation_interval=20, patience=3, min_delta=1e-4,
        checkpoint_interval_steps=20, vocab_size=8)
    assert tuning.sweep_settings({'device': 'cpu', 'batch_size': 1}, 2, 8)['validation_interval'] == 2
    with pytest.raises(ValueError):
        tuning.sweep_settings({'device': 'cpu', 'batch_size': 1}, 0, 8)
