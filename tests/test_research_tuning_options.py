"""Bounded validation-only controls and conditional fixed-Teacher replication."""
import json
from pathlib import Path

import pytest

from test_research_tuning import base, small_trials, snapshot, TinyLM


def test_shared_settings_are_bounded_and_student_seed_is_explicit():
    from src.research_tuning import sweep_settings
    training = dict(device='cpu', batch_size=1, seed=42)
    settings = sweep_settings(training, 1000, 8, seed=43, settings_overrides={
        'validation_interval': 200, 'patience': 5, 'checkpoint_interval_steps': 100})
    assert settings['seed'] == 43
    assert settings['validation_interval'] == 200
    assert settings['patience'] == 5
    assert settings['checkpoint_interval_steps'] == 100
    assert training['seed'] == 42
    for overrides in ({'learning_rate': 1e-4}, {'alpha': .7}, {'device': 'cuda'},
                      {'seed': 44}, {'validation_interval': 0}, {'patience': True},
                      {'checkpoint_interval_steps': 1.5}, {'validation_interval': 1001},
                      {'checkpoint_interval_steps': 1001}):
        with pytest.raises(ValueError):
            sweep_settings(training, 1000, 8, settings_overrides=overrides)
    for seed in (-1, 2**32, True, 1.5):
        with pytest.raises(ValueError, match='seed'):
            sweep_settings(training, 1000, 8, seed=seed)
    assert sweep_settings(training, 1, 8, seed=0)['seed'] == 0
    assert sweep_settings(training, 1, 8, seed=2**32 - 1)['seed'] == 2**32 - 1


def test_cli_custom_pair_is_validated_and_forwards_shared_options(tmp_path, monkeypatch):
    from scripts import tune_research as cli
    from src import research_tuning as tuning
    trials_file = tmp_path / 'pair.json'
    trials_file.write_text(json.dumps(small_trials()))
    calls = []
    monkeypatch.setattr(tuning, 'run_sweep', lambda *a, **kw: calls.append((a, kw)) or kw)
    args = ['--base-run', 'base', '--output-dir', 'out', '--max-steps', '1000',
            '--trials-file', str(trials_file), '--seed', '44', '--validation-interval', '200',
            '--patience', '5', '--checkpoint-interval', '100']
    result = cli.main(args)
    assert result['trials'] == small_trials()
    assert result['seed'] == 44
    assert result['settings_overrides'] == dict(validation_interval=200, patience=5,
                                               checkpoint_interval_steps=100)
    assert len(calls) == 1
    calls.clear()
    invalid_json = ['{', '{}', '[]', json.dumps(small_trials()[:1]),
                    json.dumps([{**t, 'id': f"{t['id']}-{i}"} for i in range(5)
                                for t in small_trials()]),
                    json.dumps(small_trials()).replace('2e-05', 'NaN'),
                    json.dumps([{**small_trials()[0], 'method': []}, small_trials()[1]]),
                    json.dumps([{**small_trials()[0], 'learning_rate': 10**400}, small_trials()[1]]),
                    json.dumps(small_trials()).replace('"id": "ce-1"', '"id": "other", "id": "ce-1"')]
    for content in invalid_json:
        trials_file.write_text(content)
        with pytest.raises(SystemExit):
            cli.main(args)
    trials_file.write_text(json.dumps(small_trials()))
    for flag, value in (('--seed', '-1'), ('--seed', str(2**32)), ('--seed', '1.5'),
                        ('--validation-interval', '0'), ('--patience', '-1'),
                        ('--checkpoint-interval', '1001'), ('--validation-interval', '1001')):
        with pytest.raises(SystemExit):
            cli.main(args + [flag, value])
    assert calls == []


def test_api_rejects_oversized_trials_and_invalid_options_before_loading(monkeypatch, tmp_path):
    from src import research_tuning as tuning
    def forbidden(*a, **kw):
        raise AssertionError('base/model loading is forbidden')
    monkeypatch.setattr(tuning, 'load_base', forbidden)
    too_many = [{**t, 'id': f"{t['id']}-{i}"} for i in range(5) for t in small_trials()]
    for kwargs in ({'trials': too_many}, {'seed': -1},
                   {'settings_overrides': {'device': 'cuda'}},
                   {'settings_overrides': {'validation_interval': 201}}):
        with pytest.raises(ValueError):
            tuning.run_sweep(tmp_path / 'base', tmp_path / 'out', **kwargs)


def test_real_paired_student_seeds_keep_raw_split_and_teacher_frozen(base, tmp_path, monkeypatch):
    import torch
    from src import research_tuning as tuning
    from src import research_data as data
    from src.research_runner import train_stage
    from src.resume_checkpoint import load_checkpoint
    directory, deps, _ = base
    before = snapshot(directory)
    teacher_record = json.loads((directory / 'teacher.json').read_text())
    teacher_state = load_checkpoint(teacher_record['result']['best_checkpoint'])['model_state']
    packed = []
    real_pack = data.pack_split
    def pack(path, *args):
        assert Path(path).name != 'test.jsonl'
        packed.append(Path(path).name)
        return real_pack(path, *args)
    monkeypatch.setattr(data, 'pack_split', pack)
    initializations, teacher_seeds = [], []
    def adapter(model, adapter, seed):
        torch.manual_seed(seed)
        for parameter in model.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn_like(parameter))
        initializations.append((seed, tuning.initial_adapter_hash(model)))
        return model
    def runner(model, *args, **kwargs):
        assert kwargs['settings']['seed'] in (42, 43)
        teacher = kwargs['teacher']
        if teacher is not None:
            teacher_seeds.append(initializations[-1][0])
            assert not teacher.training
            assert all(not p.requires_grad for p in teacher.parameters())
            for key, value in teacher.state_dict().items():
                assert torch.equal(value, teacher_state[key])
        return train_stage(model, *args, **kwargs)
    deps = {**deps, 'adapter': adapter, 'runner': runner}
    plans = []
    for seed in (42, 43):
        output = tmp_path / f'seed-{seed}'
        args = dict(max_steps=3, dependencies=deps, trials=small_trials(), seed=seed,
                    settings_overrides=dict(validation_interval=1, patience=4,
                                            checkpoint_interval_steps=2))
        start = len(initializations)
        result = tuning.run_sweep(directory, output, **args)
        assert result['status'] == 'completed'
        assert [r['result']['final_step'] for r in result['trials']] == [3, 3]
        # Both independently initialized Students are actually hashed before training.
        hashes = [value for _, value in initializations[start:start + 2]]
        assert len(hashes) == 2 and hashes[0] == hashes[1]
        plan = json.loads((output / 'plan.json').read_text())
        plans.append(plan)
        assert plan['initial_adapter_sha256'] == hashes[0]
        assert plan['common_settings']['seed'] == seed
        assert plan['common_settings']['patience'] == 4
        for record in result['trials']:
            assert record['checkpoint_contract']['settings'] == record['contract']['settings']
            assert record['contract']['settings']['seed'] == seed
            assert record['contract']['settings']['checkpoint_interval_steps'] == 2
        saved = snapshot(output)
        def forbidden(*a, **kw):
            raise AssertionError('resume must not train')
        resume_args = {**args, 'dependencies': {**deps, 'runner': forbidden}}
        assert tuning.run_sweep(directory, output, resume=True, **resume_args) == result
        assert snapshot(output) == saved
        for changes in ({'seed': seed + 1}, {'trials': list(reversed(small_trials()))},
                        {'settings_overrides': {'validation_interval': 2}},
                        {'settings_overrides': {'patience': 5}},
                        {'settings_overrides': {'checkpoint_interval_steps': 1}}):
            with pytest.raises(ValueError, match='plan'):
                tuning.run_sweep(directory, output, resume=True, **{**resume_args, **changes})
            assert snapshot(output) == saved
    assert plans[0]['initial_adapter_sha256'] != plans[1]['initial_adapter_sha256']
    assert plans[0]['provenance'] == plans[1]['provenance']
    assert teacher_seeds == [42, 42]
    assert before == snapshot(directory)
    assert packed and set(packed) == {'train.jsonl', 'validation.jsonl'}


def test_service_forwards_literal_optional_arguments_and_legacy_namespaces(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import tuning_service as service
    from scripts import tune_research as cli
    from src import stage_records
    file = tmp_path / 'pair $not-shell; options.json'
    argv = service.command('base', tmp_path / 'out', 1000, python='/python', seed=43,
                           trials_file=file, validation_interval=200, patience=5,
                           checkpoint_interval=100)
    for flag, value in (('--seed', '43'), ('--trials-file', str(file)),
                        ('--validation-interval', '200'), ('--patience', '5'),
                        ('--checkpoint-interval', '100')):
        assert argv[argv.index(flag) + 1] == value
    assert '' not in argv and '--resume' not in argv
    defaults = service.command('base', tmp_path / 'out', 200, seed=42)
    assert defaults[-2:] == ['--max-steps', '200']
    assert '--seed' not in defaults
    calls = []
    monkeypatch.setattr(stage_records, 'reconcile_completed_stage', lambda *a: None)
    monkeypatch.setattr(service.os, 'execv', lambda *a: calls.append(a))
    args = SimpleNamespace(base_run=Path('base'), output_dir=tmp_path / 'out',
                           max_steps=1000, resume=False, seed=43, trials_file=file,
                           validation_interval=200, patience=5, checkpoint_interval=100)
    monkeypatch.setattr(cli, 'parse_args', lambda: args)
    service.main()
    assert calls[-1][1][1:] == argv[1:]
    for key in ('seed', 'trials_file', 'validation_interval', 'patience', 'checkpoint_interval'):
        delattr(args, key)
    service.main()
    assert calls[-1][1][-2:] == ['--max-steps', '1000']


def test_source_identity_covers_all_production_entrypoints():
    from src.research_tuning import _code_identity
    identity = _code_identity()
    assert {'src/research_tuning.py', 'scripts/tune_research.py',
            'scripts/tuning_service.py'} <= identity.keys()
