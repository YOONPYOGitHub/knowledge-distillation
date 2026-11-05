import copy
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    eos_token_id = 7
    special_tokens_map = {'eos_token': '<eos>'}
    def __init__(self):
        from types import SimpleNamespace
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: '{}')
    def get_vocab(self): return {str(i): i for i in range(8)}
    def encode(self, text, add_special_tokens=False): return [1, 2, 3, 4] * 4


def source(n):
    return [{'id': str(i), 'text': ('한국어 문서 테스트 데이터입니다. ' * 30) + str(i)} for i in range(n)]


def arguments(tmp_path, stage):
    return [stage, '--config', str(ROOT / 'configs/research_qwen35.yaml'), '--run-dir', str(tmp_path), '--max-documents', '30', '--seq-length', '8', '--max-steps', '2']


def test_prepare_exclusion_cli_freezes_policy_and_rejects_tampering(tmp_path):
    from scripts.run_research import main
    from src.research_data import digest
    rows = source(30)
    policy = {'source_ids': ['0'], 'text_sha256': []}
    registry = tmp_path / 'exposure.json'
    registry.write_text(json.dumps(policy))
    directory = tmp_path / 'run'
    args = arguments(directory, 'prepare') + ['--exposure-exclusion-file', str(registry)]
    deps = {'source': lambda n: rows, 'tokenizer': lambda spec: TinyTokenizer()}
    prepared = main(args, dependencies=deps)
    identity = prepared['raw']['identity']
    assert identity['exposure_exclusion'] == policy
    assert identity['exposure_exclusion_sha256'] == digest(policy)
    assert prepared['raw']['rejected']['exposed'] == 1
    for split in ('train', 'validation', 'test'):
        assert all(json.loads(line)['source_id'] != '0'
                   for line in (directory / 'raw' / f'{split}.jsonl').read_text().splitlines())
    assert main(args + ['--resume'], dependencies=deps) == prepared
    registry.write_text(json.dumps({'source_ids': ['1'], 'text_sha256': []}))
    with pytest.raises(ValueError, match='config|identity'):
        main(args + ['--resume'], dependencies=deps)


def test_exposure_registry_rejects_duplicate_json_keys_before_loading(tmp_path):
    from scripts.run_research import load_config, parse_args
    registry = tmp_path / 'exposure.json'
    registry.write_text('{"source_ids":["pilot"],"source_ids":[],"text_sha256":[]}')
    args = parse_args(arguments(tmp_path / 'run', 'prepare') +
                      ['--exposure-exclusion-file', str(registry)])
    with pytest.raises(ValueError, match='duplicate'):
        load_config(args)


def test_prepare_freezes_provenance_without_loading_models(tmp_path):
    from scripts.run_research import main
    def forbidden(*args, **kwargs): raise AssertionError('model download not allowed')
    deps = {'source': source, 'tokenizer': lambda spec: TinyTokenizer(), 'model': forbidden}
    result = main(arguments(tmp_path, 'prepare'), dependencies=deps)
    assert result['raw']['counts'] == {'train': 24, 'validation': 3, 'test': 3}
    assert result['tokenizer']['vocab_size'] == 8
    assert set(result['packing']) == {'train', 'validation'}
    assert not (tmp_path / 'test-results.json').exists()
    deps['source'] = forbidden
    assert main(arguments(tmp_path, 'prepare') + ['--resume'], dependencies=deps) == result



def tiny_dependencies(trace):
    import torch
    from types import SimpleNamespace
    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 6)
            self.head = torch.nn.Linear(6, 8)
        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))
    def model(spec, **kw):
        torch.manual_seed(17)
        value = TinyLM()
        trace.append(('load', spec['id']))
        return value
    return {'source': source, 'tokenizer': lambda spec: TinyTokenizer(), 'model': model,
            'adapter': lambda model, adapter, seed: model}


def cpu_args(tmp_path, stage):
    path = tmp_path.parent / (tmp_path.name + '-config.yaml')
    config = yaml.safe_load((ROOT / 'configs/research_qwen35.yaml').read_text())
    config['training'].update(device='cpu', validation_interval=1, min_delta=100)
    path.write_text(yaml.safe_dump(config))
    args = arguments(tmp_path, stage)
    args[args.index('--config') + 1] = str(path)
    return args


def test_teacher_selects_step_zero_and_ce_uses_only_fresh_student(tmp_path):
    from scripts.run_research import main
    trace = []
    deps = tiny_dependencies(trace)
    teacher = main(cpu_args(tmp_path, 'teacher'), dependencies=deps)
    assert teacher['result']['status'] == 'completed'
    assert teacher['result']['final_step'] == 2
    assert teacher['result']['best_step'] == 0
    assert teacher['teacher_selection'] == 'pretrained_step0_retained_no_validation_improvement'
    assert trace == [('load', 'Qwen/Qwen3.5-9B')]
    trace.clear()
    ce = main(cpu_args(tmp_path, 'ce'), dependencies=deps)
    assert ce['contract']['settings']['alpha'] == 1.0
    assert trace == [('load', 'Qwen/Qwen3.5-4B')]
    trace.clear()
    assert main(cpu_args(tmp_path, 'ce') + ['--resume'], dependencies=deps) == ce
    assert trace == []
    with pytest.raises(ValueError, match='resume'):
        main(cpu_args(tmp_path, 'ce'), dependencies=deps)


def test_kd_requires_frozen_teacher_and_pairs_ce_initialization_and_budget(tmp_path):
    import torch
    from scripts.run_research import main
    from src.research_runner import train_stage
    trace, initial, calls = [], {}, []
    deps = tiny_dependencies(trace)
    def runner(model, train, validation, optimizer, output_dir, **kwargs):
        name = Path(output_dir).name
        initial[name] = {k: v.clone() for k, v in model.state_dict().items()}
        calls.append(kwargs)
        if name == 'kd':
            assert not kwargs['teacher'].training
            assert not any(p.requires_grad for p in kwargs['teacher'].parameters())
        return train_stage(model, train, validation, optimizer, output_dir, **kwargs)
    deps['runner'] = runner
    with pytest.raises(ValueError, match='teacher'):
        main(cpu_args(tmp_path, 'kd'), dependencies=deps)
    assert trace == []
    main(cpu_args(tmp_path, 'teacher'), dependencies=deps)
    ce = main(cpu_args(tmp_path, 'ce'), dependencies=deps)
    kd = main(cpu_args(tmp_path, 'kd'), dependencies=deps)
    assert all(torch.equal(v, initial['kd'][k]) for k, v in initial['ce'].items())
    assert kd['result']['final_step'] == ce['result']['final_step'] == 2
    for key in ('max_steps', 'epochs', 'patience', 'min_delta', 'seed', 'validation_interval'):
        assert kd['contract']['settings'][key] == ce['contract']['settings'][key]
    assert kd['contract']['settings']['alpha'] == .5
    assert kd['contract']['identity']['teacher_selection']['checkpoint']
    assert kd['contract']['identity']['kd_support'] == 'renormalized_verified_tokenizer_ids; excludes_unassigned_head_padding; CE_uses_full_head'
    assert not (tmp_path / 'test-results.json').exists()


def test_all_freezes_selection_before_test_and_caches_final_results(tmp_path, monkeypatch):
    import scripts.run_research as cli
    trace = []
    deps = tiny_dependencies(trace)
    real_pack = cli.pack_split
    test_reads = []
    def guarded_pack(path, tokenizer, seq_length):
        if Path(path).name == 'test.jsonl':
            assert (tmp_path / 'selection.json').exists()
            selection = json.loads((tmp_path / 'selection.json').read_text())
            assert set(selection['stages']) == {'teacher', 'ce', 'kd'}
            test_reads.append(str(path))
        return real_pack(path, tokenizer, seq_length)
    monkeypatch.setattr(cli, 'pack_split', guarded_pack)
    with pytest.raises(ValueError, match='completed'):
        cli.main(cpu_args(tmp_path, 'evaluate'), dependencies=deps)
    assert test_reads == [] and trace == []
    result = cli.main(cpu_args(tmp_path, 'all'), dependencies=deps)
    assert set(result['metrics']) == {'student_pretrained', 'student_ce', 'student_kd', 'teacher_selected'}
    assert all(v['tokens'] > 0 and v['ppl'] > 0 for v in result['metrics'].values())
    assert len(test_reads) == 1
    assert result['interpretation'] == 'GPU smoke only; no effectiveness conclusion'
    trace.clear()
    assert cli.main(cpu_args(tmp_path, 'evaluate'), dependencies=deps) == result
    assert trace == [] and len(test_reads) == 1
    # Changing a training budget after looking at test requires an entirely new run.
    changed = cpu_args(tmp_path, 'all')
    changed[-1] = '3'
    with pytest.raises(ValueError, match='config'):
        cli.main(changed, dependencies=deps)
    # Mutating selection cannot silently reuse test metrics.
    selection = json.loads((tmp_path / 'selection.json').read_text())
    selection['config_hash'] = 'tampered'
    (tmp_path / 'selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='selection'):
        cli.main(cpu_args(tmp_path, 'evaluate'), dependencies=deps)


def test_cli_entrypoint_exposes_stages_offline():
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_research.py'), '--help'],
                            capture_output=True, text=True, env={**os.environ, 'HF_HUB_OFFLINE': '1'}, timeout=20)
    assert result.returncode == 0
    assert '{prepare,teacher,ce,kd,evaluate,all}' in result.stdout


def test_ce_cannot_skip_teacher_training(tmp_path):
    from scripts.run_research import main
    trace = []
    with pytest.raises(ValueError, match='teacher'):
        main(cpu_args(tmp_path, 'ce'), dependencies=tiny_dependencies(trace))
    assert trace == []


@pytest.mark.parametrize('section,key,value', [('training', 'temperature', float('nan')),
    ('training', 'teacher_learning_rate', -1), ('training', 'divergence', 'unknown'),
    ('adapter', 'rank', 0), ('adapter', 'dropout', 1.5), ('model', 'attention', 'flash_attention_2')])
def test_invalid_config_fails_before_model_or_corpus_load(tmp_path, section, key, value):
    from scripts.run_research import main
    config = yaml.safe_load((ROOT / 'configs/research_qwen35.yaml').read_text())
    config[section][key] = value
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(config))
    def forbidden(*args, **kwargs): raise AssertionError('discovery must not start on invalid configuration')
    with pytest.raises(ValueError):
        main(['prepare', '--config', str(path), '--run-dir', str(tmp_path / 'run')], dependencies={'source': forbidden})


def test_three_configs_are_pinned_and_cli_overrides_are_frozen(tmp_path):
    from scripts.run_research import parse_args, load_config, freeze_config, run_lock
    for name, teacher, student in [('qwen35', 'Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-4B'),
                                    ('gemma4', 'google/gemma-4-12B-it', 'google/gemma-4-E2B-it'),
                                    ('lfm25', 'LiquidAI/LFM2.5-24B-A2B', 'LiquidAI/LFM2.5-2.6B')]:
        args = parse_args(['prepare', '--config', str(ROOT / f'configs/research_{name}.yaml'), '--run-dir', str(tmp_path / name), '--max-steps', '2', '--max-documents', '30', '--seq-length', '8'])
        config = load_config(args)
        assert config['teacher']['id'] == teacher and config['student']['id'] == student
        assert all(len(config[r]['revision']) == 40 for r in ('teacher', 'student'))
        assert config['training']['max_steps'] == 2 and config['data']['seq_length'] == 8
        freeze_config(tmp_path / name, config)
        freeze_config(tmp_path / name, config)
        changed = copy.deepcopy(config)
        changed['training']['max_steps'] = 3
        with pytest.raises(ValueError, match='config'):
            freeze_config(tmp_path / name, changed)
        with run_lock(tmp_path / name):
            with pytest.raises(RuntimeError, match='locked'):
                with run_lock(tmp_path / name): pass
