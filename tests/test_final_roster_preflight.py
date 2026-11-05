"""A corrupt later result must prevent all model loading/publication."""
import hashlib
import importlib.util
from pathlib import Path
import pytest
from src import final_evaluation as fe


def test_later_corrupt_output_rejected_before_any_model_load(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('preflight_fixture', Path(__file__).with_name('test_final_real_contract_integration.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    policy = dict(model_dtype='bfloat16', ce_dtype='float32', ce_reduction='shifted_valid_token_sum', ignore_index=-100, batch_size=1, sequence_length=512, deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False, allow_tf32=False, cublas_workspace_config=':4096:8')
    adjudication = b'CPU fixture only'
    metadata = dict(numerical_policy=policy, exposure_adjudication_bytes=adjudication, expected_exposure_sha256=hashlib.sha256(adjudication).hexdigest())
    loaded = []

    def integrate(protocol, selections, cohort):
        frozen = fe.freeze_evaluation_protocol(protocol, selections, cohort, **metadata)
        paths = {entry['id']: str(tmp_path / (entry['id'] + '-evaluation.json')) for entry in protocol['models']}
        corrupt = Path(list(paths.values())[-1])
        corrupt.write_bytes(b'broken result')
        source = b'fixture source'
        operation = fe.freeze_evaluation_operation(protocol_bytes=frozen, source_sha256=hashlib.sha256(source).hexdigest(), max_seconds=60, max_batches_per_model=1, output_paths=paths)
        def load(entry):
            loaded.append(entry['id'])
            raise AssertionError('model loaded before complete roster preflight')
        with pytest.raises(ValueError):
            fe.run_authenticated_evaluation(protocol, selections, cohort, operation_bytes=operation, source_bytes=source, load_model=load, iter_batches=lambda entry: iter(()), **metadata)
        assert loaded == []
        assert corrupt.read_bytes() == b'broken result'
        assert all(not Path(path).exists() for path in list(paths.values())[:-1])
        return frozen
    monkeypatch.setattr(helper, 'freeze_cohort_protocol', integrate)
    helper.test_original_teacher_and_six_unmodified_student_contracts_bind(tmp_path)
