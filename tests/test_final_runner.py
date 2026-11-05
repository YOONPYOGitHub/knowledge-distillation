"""Runner resume must reuse results before loading models or heldout."""
import hashlib
import json
from src import final_evaluation
import pytest


def test_runner_rejects_duplicate_protocol_fields_before_reading_outputs():
    protocol = b'{"models":[{"id":"foreign"}],"models":[{"id":"ce-seed42"}]}'
    with pytest.raises(ValueError, match='roster invalid'):
        final_evaluation.reuse_completed_evaluation_roster(
            protocol_bytes=protocol, output_paths={'ce-seed42': '/missing.json'})


@pytest.mark.parametrize('protocol,paths', [
    (b'{"models":[]}', {}),
    (b'{"models":[{"id":"ce"},{"id":"ce"}]}', {'ce': '/missing.json'}),
    (b'{"models":[{"id":"ce"}]}', {'kd': '/missing.json'}),
    (b'{"models":null}', {}),
    (b'[]', {}),
    (b'{"models":[null]}', {}),
])
def test_runner_rejects_malformed_roster(protocol, paths):
    with pytest.raises(ValueError, match='roster invalid'):
        final_evaluation.reuse_completed_evaluation_roster(
            protocol_bytes=protocol, output_paths=paths)


@pytest.mark.parametrize('variant', ['missing', 'partial', 'foreign'])
def test_runner_rejects_unusable_output_without_rewriting(tmp_path, variant):
    protocol = b'{"models":[{"id":"ce"}]}'
    path = tmp_path / 'ce.json'
    payload = json.dumps({
        'version': 1, 'status': 'running' if variant == 'partial' else 'completed',
        'model_id': 'foreign' if variant == 'foreign' else 'ce',
        'protocol_sha256': hashlib.sha256(protocol).hexdigest(),
        'statistics': {'ce_sum': 10., 'tokens': 5},
    }).encode()
    if variant != 'missing':
        path.write_bytes(payload)
    with pytest.raises(ValueError):
        final_evaluation.reuse_completed_evaluation_roster(
            protocol_bytes=protocol, output_paths={'ce': str(path)})
    assert path.exists() == (variant != 'missing')
    if path.exists():
        assert path.read_bytes() == payload


def test_runner_reuses_completed_roster_without_scoring_or_writes(tmp_path):
    protocol = json.dumps({'models': [{'id': 'ce-seed42'}, {'id': 'kd-seed42'}]}).encode()
    outputs = {}
    before = {}
    for model_id in ('ce-seed42', 'kd-seed42'):
        path = tmp_path / (model_id + '.json')
        payload = json.dumps({
            'version': 1, 'status': 'completed', 'model_id': model_id,
            'protocol_sha256': hashlib.sha256(protocol).hexdigest(),
            'statistics': {'ce_sum': 10., 'tokens': 5},
        }).encode()
        path.write_bytes(payload)
        outputs[model_id] = str(path)
        before[model_id] = (payload, path.stat().st_mtime_ns)
    runner = getattr(final_evaluation, 'reuse_completed_evaluation_roster', None)
    assert callable(runner), 'runner completed-roster reuse missing'
    result = runner(protocol_bytes=protocol, output_paths=outputs)
    assert set(result) == set(outputs)
    for model_id, path in outputs.items():
        assert result[model_id]['retained_bytes'] == before[model_id][0]
        assert result[model_id]['statistics']['ce'] == 2.0
        from pathlib import Path
        assert Path(path).read_bytes() == before[model_id][0]
        assert Path(path).stat().st_mtime_ns == before[model_id][1]
