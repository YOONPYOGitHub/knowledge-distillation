"""Freeze executable evaluator limits before authorizing heldout access."""
import json
import pytest
from src import final_evaluation


def test_operational_contract_binds_source_budget_and_output_roster(tmp_path):
    freeze = getattr(final_evaluation, 'freeze_evaluation_operation', None)
    assert callable(freeze), 'operational evaluation contract missing'
    protocol = b'{"models":[{"id":"ce42"},{"id":"kd42"}]}'
    paths = {name: str(tmp_path / (name + '.json')) for name in ('ce42', 'kd42')}
    kwargs = dict(protocol_bytes=protocol, source_sha256='a' * 64,
                  max_seconds=3600, max_batches_per_model=2000, output_paths=paths)
    retained = freeze(**kwargs)
    assert retained == freeze(**kwargs)
    data = json.loads(retained)
    assert data['source_sha256'] == 'a' * 64
    assert data['max_seconds'] == 3600
    assert data['max_batches_per_model'] == 2000
    assert data['output_paths'] == paths
    assert not list(tmp_path.iterdir())
    for changed in ({'max_seconds': True}, {'max_batches_per_model': 0},
                    {'source_sha256': 'z' * 64}, {'output_paths': {'ce42': paths['ce42']}},
                    {'output_paths': {'ce42': paths['ce42'], 'kd42': paths['ce42']}},
                    {'output_paths': {'ce42': '../escape', 'kd42': paths['kd42']}}):
        with pytest.raises(ValueError):
            freeze(**(kwargs | changed))
