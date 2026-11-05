"""Executable dispatch contract; no real heldout access."""
import subprocess
import sys
from pathlib import Path


def execution_sources():
    import hashlib
    root = Path(__file__).parents[1]
    paths = [root / name for name in (
        'scripts/evaluate_final.py', 'src/final_evaluation.py',
        'src/research_runner.py', 'src/research_models.py', 'src/research_data.py',
        'src/resume_checkpoint.py', 'src/numerical_policy.py')]
    return [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]


def test_executable_rejects_unknown_bundle_fields_before_imports(tmp_path):
    import hashlib
    import json
    payload = json.dumps(dict(protocol={}, selections={}, cohort={}, operation={},
        numerical_policy={}, exposure_adjudication='', exposure_sha256='0'*64,
        sources=[], commits={}, heldout={}, unexpected=True)).encode()
    path = tmp_path / 'unknown.json'
    path.write_bytes(payload)
    script = Path(__file__).parents[1] / 'scripts/evaluate_final.py'
    result = subprocess.run([sys.executable, str(script), '--bundle', str(path),
        '--bundle-sha256', hashlib.sha256(payload).hexdigest()],
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'invalid evaluation bundle schema' in result.stderr


def test_executable_rejects_incomplete_execution_sources_before_imports(tmp_path):
    import hashlib
    import json
    payload = json.dumps(dict(protocol={}, selections={}, cohort={}, operation={},
        numerical_policy={}, exposure_adjudication='', exposure_sha256='0'*64,
        sources=[], commits={}, heldout={})).encode()
    path = tmp_path / 'missing-sources.json'
    path.write_bytes(payload)
    script = Path(__file__).parents[1] / 'scripts/evaluate_final.py'
    result = subprocess.run([sys.executable, str(script), '--bundle', str(path),
        '--bundle-sha256', hashlib.sha256(payload).hexdigest()],
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'execution source coverage mismatch' in result.stderr


def test_executable_rejects_symlink_execution_source_before_imports(tmp_path):
    import hashlib
    import importlib.util
    import json
    import pytest
    script = Path(__file__).parents[1] / 'scripts/evaluate_final.py'
    spec = importlib.util.spec_from_file_location('unsafe_source_executable', script)
    executable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executable)
    root = tmp_path / 'snapshot'
    sources = []
    for original in execution_sources():
        relative = Path(original['path']).relative_to(script.parents[1])
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'synthetic source')
        sources.append(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    victim = root / 'src/final_evaluation.py'
    backing = tmp_path / 'backing.py'
    backing.write_bytes(victim.read_bytes())
    victim.unlink()
    victim.symlink_to(backing)
    executable.__file__ = str(root / 'scripts/evaluate_final.py')
    bundle = dict(protocol={}, selections={}, cohort={}, operation={},
        numerical_policy={}, exposure_adjudication='', exposure_sha256='0'*64,
        sources=sources, commits={}, heldout={})
    path = tmp_path / 'bundle.json'
    payload = json.dumps(bundle).encode()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match='unsafe evaluation source'):
        executable.execute_bundle(path, hashlib.sha256(payload).hexdigest(), device='cpu')


def test_final_evaluator_executable_exposes_frozen_input_contract():
    script = Path(__file__).parents[1] / 'scripts' / 'evaluate_final.py'
    assert script.is_file(), 'integrated evaluator executable missing'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    for option in ('--bundle', '--bundle-sha256', '--device'):
        assert option in result.stdout


def test_executable_rejects_duplicate_bundle_fields_before_dispatch(tmp_path):
    import hashlib
    script = Path(__file__).parents[1] / 'scripts/evaluate_final.py'
    payload = b'{"protocol":{},"protocol":{}}'
    bundle = tmp_path / 'duplicate.json'
    bundle.write_bytes(payload)
    result = subprocess.run([sys.executable, str(script), '--bundle', str(bundle),
                             '--bundle-sha256', hashlib.sha256(payload).hexdigest()],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'duplicate evaluation bundle field' in result.stderr


def test_executable_rejects_symlink_bundle_before_dispatch(tmp_path):
    import hashlib
    script = Path(__file__).parents[1] / 'scripts/evaluate_final.py'
    original = tmp_path / 'original.json'
    payload = b'{}'
    original.write_bytes(payload)
    linked = tmp_path / 'linked.json'
    linked.symlink_to(original)
    result = subprocess.run([sys.executable, str(script), '--bundle', str(linked),
                             '--bundle-sha256', hashlib.sha256(payload).hexdigest()],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'unsafe evaluation bundle' in result.stderr


def test_executable_rejects_oversized_bundle_before_dispatch(tmp_path):
    import hashlib
    script = Path(__file__).parents[1] / 'scripts/evaluate_final.py'
    payload = b' ' * (8 * 1024 * 1024 + 1)
    path = tmp_path / 'oversized.json'
    path.write_bytes(payload)
    result = subprocess.run([sys.executable, str(script), '--bundle', str(path),
                             '--bundle-sha256', hashlib.sha256(payload).hexdigest()],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'evaluation bundle exceeds byte limit' in result.stderr


def test_final_executable_rejects_untrusted_bundle_before_json_or_imports(tmp_path):
    script = Path(__file__).parents[1] / 'scripts' / 'evaluate_final.py'
    bundle = tmp_path / 'bundle.json'
    bundle.write_bytes(b'not JSON')
    result = subprocess.run([sys.executable, str(script), '--bundle', str(bundle),
                             '--bundle-sha256', '0' * 64],
                            capture_output=True, text=True, timeout=10)
    assert 'evaluation bundle SHA mismatch' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr
    assert result.returncode != 0


def test_executable_wires_authenticated_roster_and_bound_adapters(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import json
    from src import final_evaluation as fe
    script = Path(__file__).parents[1] / 'scripts' / 'evaluate_final.py'
    assert script.is_file(), 'integrated evaluator executable missing'
    spec = importlib.util.spec_from_file_location('executable', script)
    executable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executable)
    source = tmp_path / 'source.py'
    source.write_bytes(b'frozen source')
    manifest = b'{"files":{"test.jsonl":"unused synthetic raw"}}'
    exposure = b'synthetic exposure'
    bundle = dict(protocol={'models': [{'id': 'synthetic', 'model_id': 'fixture', 'model_revision': 'a'*40}]}, selections={}, cohort={'provenance': {'raw_manifest_hash': hashlib.sha256(manifest).hexdigest()}},
                  numerical_policy={}, exposure_adjudication=exposure.decode(),
                  exposure_sha256=hashlib.sha256(exposure).hexdigest(), operation={},
                  sources=execution_sources(),
                  commits={}, heldout=dict(raw_path='synthetic-not-opened', raw_manifest=manifest.decode(),
                                          tokenizer_spec={'id': 'fixture', 'revision': 'a'*40}, max_bytes=4096))
    path = tmp_path / 'bundle.json'
    path.write_text(json.dumps(bundle))
    seen = []
    monkeypatch.setattr(fe, 'load_final_roster_model', lambda entry, selections, commits, **kw: seen.append(('model', entry['id'])) or object())
    monkeypatch.setattr(fe, 'load_bound_final_heldout', lambda raw, **kw: seen.append(('heldout', raw)) or [])
    def run(protocol, selections, cohort, **kwargs):
        kwargs['load_model']({'id': 'synthetic'})
        kwargs['iter_batches']({'id': 'synthetic'})
        return {'synthetic': {'statistics': {'tokens': 511}}}
    monkeypatch.setattr(fe, 'run_authenticated_evaluation', run)
    result = executable.execute_bundle(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), device='cpu')
    assert result['synthetic']['statistics']['tokens'] == 511
    assert seen == [('model', 'synthetic'), ('heldout', 'synthetic-not-opened')]


def test_executable_rejects_substituted_manifest_before_model_dispatch(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import json
    import pytest
    from src import final_evaluation as fe
    script = Path(__file__).parents[1] / 'scripts' / 'evaluate_final.py'
    spec = importlib.util.spec_from_file_location('executable_rejection', script)
    executable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executable)
    bundle = dict(sources=execution_sources(), commits={}, selections={}, protocol={}, operation={},
                  numerical_policy={}, exposure_adjudication='', exposure_sha256='0'*64,
                  cohort={'provenance': {'raw_manifest_hash': '0'*64}},
                  heldout={'raw_manifest': '{}'})
    path = tmp_path / 'bundle.json'
    path.write_text(json.dumps(bundle))
    touched = []
    def run(*args, **kwargs):
        touched.append('orchestration')
        kwargs['load_model']({'id': 'synthetic'})
    monkeypatch.setattr(fe, 'run_authenticated_evaluation', run)
    monkeypatch.setattr(fe, 'load_final_roster_model', lambda *a, **k: touched.append('model'))
    with pytest.raises(ValueError, match='heldout raw manifest binding mismatch'):
        executable.execute_bundle(path, hashlib.sha256(path.read_bytes()).hexdigest(), device='cpu')
    assert touched == []


def test_execute_bundle_authenticates_scores_and_reuses_whole_roster(tmp_path, monkeypatch):
    """Real orchestration/publication; synthetic model and heldout boundaries only."""
    import hashlib
    import importlib.util
    import json
    import torch
    from src import final_evaluation as fe

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded

    executable = module('whole_executable', Path(__file__).parents[1] / 'scripts/evaluate_final.py')
    helper = module('whole_original_contracts', Path(__file__).with_name('test_final_real_contract_integration.py'))
    policy = dict(model_dtype='bfloat16', ce_dtype='float32', ce_reduction='shifted_valid_token_sum', ignore_index=-100, batch_size=1, sequence_length=512, deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False, allow_tf32=False, cublas_workspace_config=':4096:8')
    exposure = b'synthetic whole-runner fixture, not test authorization'
    seen = []

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        def forward(self, input_ids):
            return type('Output', (), {'logits': self.weight[input_ids]})()

    def load(entry, *args, **kwargs):
        seen.append(entry['id'])
        return Tiny()

    def batches(*args, **kwargs):
        ids = torch.zeros((1, 512), dtype=torch.long)
        return [(ids, ids.clone())]

    monkeypatch.setattr(fe, 'load_final_roster_model', load)
    monkeypatch.setattr(fe, 'load_bound_final_heldout', batches)

    def integrate(protocol, selections, cohort):
        from copy import deepcopy
        original_frozen = fe.freeze_cohort_protocol(protocol, selections, cohort)
        protocol, selections, cohort = deepcopy((protocol, selections, cohort))
        # Entirely synthetic manifest dependency, never a substitute real cohort.
        manifest = '{"files":{"test.jsonl":"synthetic-only"}}'
        manifest_sha = hashlib.sha256(manifest.encode()).hexdigest()
        cohort['provenance']['raw_manifest_hash'] = manifest_sha
        for entry in protocol['models']:
            if entry['id'] in selections and entry['id'] != 'teacher':
                contract = selections[entry['id']]['contract']
                contract['identity']['provenance']['raw_manifest_hash'] = manifest_sha
                retained = json.dumps(dict(version=1, step=3000, identity=contract['identity'], contract=contract)).encode()
                target = tmp_path / (entry['id'] + '-synthetic-manifest.json')
                target.write_bytes(retained)
                entry['checkpoint'] = dict(manifest_path=str(target), manifest_sha256=hashlib.sha256(retained).hexdigest())
        frozen = fe.freeze_evaluation_protocol(protocol, selections, cohort,
            numerical_policy=policy, exposure_adjudication_bytes=exposure,
            expected_exposure_sha256=hashlib.sha256(exposure).hexdigest())
        source = tmp_path / 'frozen.py'
        source.write_bytes(b'synthetic execution identity')
        paths = {entry['id']: str(tmp_path / (entry['id'] + '-result.json')) for entry in protocol['models']}
        operation = fe.freeze_evaluation_operation(protocol_bytes=frozen,
            source_sha256=hashlib.sha256(b''.join(Path(item['path']).read_bytes() for item in execution_sources())).hexdigest(),
            max_seconds=60, max_batches_per_model=1, output_paths=paths)
        bundle = dict(protocol=protocol, selections=selections, cohort=cohort,
            numerical_policy=policy, exposure_adjudication=exposure.decode(),
            exposure_sha256=hashlib.sha256(exposure).hexdigest(),
            operation=json.loads(operation), commits={},
            sources=execution_sources(),
            heldout=dict(raw_path='synthetic-only', raw_manifest=manifest,
                tokenizer_spec=dict(id=protocol['models'][1]['model_id'], revision=protocol['models'][1]['model_revision']), max_bytes=4096))
        path = tmp_path / 'whole-bundle.json'
        path.write_text(json.dumps(bundle))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Reject a self-consistently rehashed bundle with a foreign tokenizer
        # before orchestration/model/heldout access.
        import pytest
        foreign = deepcopy(bundle)
        foreign['heldout']['tokenizer_spec']['revision'] = '0' * 40
        path.write_text(json.dumps(foreign))
        with pytest.raises(ValueError, match='tokenizer binding mismatch'):
            executable.execute_bundle(path, hashlib.sha256(path.read_bytes()).hexdigest(), device='cpu')
        assert seen == []
        path.write_text(json.dumps(bundle))
        result = executable.execute_bundle(path, digest, device='cpu')
        assert set(result) == set(paths)
        assert len(seen) == 8
        for row in result.values():
            assert row['statistics']['tokens'] == 511
            assert abs(row['statistics']['ppl'] - 4) < 1e-5
        originals = {name: Path(path).read_bytes() for name, path in paths.items()}
        assert executable.execute_bundle(path, digest, device='cpu') == result
        assert len(seen) == 8
        assert originals == {name: Path(path).read_bytes() for name, path in paths.items()}
        return original_frozen

    monkeypatch.setattr(helper, 'freeze_cohort_protocol', integrate)
    helper.test_original_teacher_and_six_unmodified_student_contracts_bind(tmp_path)
