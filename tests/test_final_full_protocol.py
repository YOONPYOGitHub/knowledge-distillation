"""Complete protocol integration using unmodified archived training contracts."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from copy import deepcopy

from src import final_evaluation


def test_authenticated_runner_reuses_real_full_protocol_roster(tmp_path, monkeypatch):
    helper_path = Path(__file__).with_name('test_final_real_contract_integration.py')
    spec = importlib.util.spec_from_file_location('runner_real_contract_fixture', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    adjudication = (Path(__file__).resolve().parent /
                    'fixtures/generic-final-metadata/GENERIC_CPT_EXPOSURE_ADJUDICATION.md').read_bytes()
    policy = {
        'model_dtype': 'bfloat16', 'ce_dtype': 'float32',
        'ce_reduction': 'shifted_valid_token_sum', 'ignore_index': -100,
        'batch_size': 1, 'sequence_length': 512,
        'deterministic_algorithms': True, 'cudnn_deterministic': True,
        'cudnn_benchmark': False, 'allow_tf32': False,
        'cublas_workspace_config': ':4096:8',
    }
    observed = []

    def integrate(protocol, selections, cohort):
        kwargs = dict(numerical_policy=policy,
                      exposure_adjudication_bytes=adjudication,
                      expected_exposure_sha256=hashlib.sha256(adjudication).hexdigest())
        retained = final_evaluation.freeze_evaluation_protocol(
            protocol, selections, cohort, **kwargs)
        paths = {}
        originals = {}
        for entry in json.loads(retained)['models']:
            model_id = entry['id']
            output_dir = tmp_path / 'completed-outputs'
            output_dir.mkdir(exist_ok=True)
            path = output_dir / (model_id + '.json')
            payload = json.dumps({
                'version': 1, 'status': 'completed', 'model_id': model_id,
                'protocol_sha256': hashlib.sha256(retained).hexdigest(),
                'statistics': {'ce_sum': 10., 'tokens': 5},
            }).encode()
            path.write_bytes(payload)
            paths[model_id] = str(path)
            originals[model_id] = (payload, path.stat().st_mtime_ns)
        runner = getattr(final_evaluation, 'reuse_authenticated_evaluation_roster', None)
        assert callable(runner), 'authenticated full protocol runner missing'
        results = runner(protocol, selections, cohort, output_paths=paths, **kwargs)
        assert set(results) == set(paths)
        for model_id, path in paths.items():
            assert results[model_id]['retained_bytes'] == originals[model_id][0]
            assert Path(path).stat().st_mtime_ns == originals[model_id][1]
        observed.append(len(results))
        return retained

    monkeypatch.setattr(helper, 'freeze_cohort_protocol', integrate)
    helper.test_original_teacher_and_six_unmodified_student_contracts_bind(tmp_path)
    assert observed == [8]


def test_full_protocol_binds_original_cohort_numerics_and_exposure(tmp_path, monkeypatch):
    helper_path = Path(__file__).with_name('test_final_real_contract_integration.py')
    spec = importlib.util.spec_from_file_location('original_contract_fixture', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
    adjudication = (root / 'GENERIC_CPT_EXPOSURE_ADJUDICATION.md').read_bytes()
    expected_sha = '1f2a6d352a26359febb6aad90b85e93a0d5e5a58fa5819b4bccf881821ea5379'
    assert hashlib.sha256(adjudication).hexdigest() == expected_sha
    policy = {
        'model_dtype': 'bfloat16', 'ce_dtype': 'float32',
        'ce_reduction': 'shifted_valid_token_sum', 'ignore_index': -100,
        'batch_size': 1, 'sequence_length': 512,
        'deterministic_algorithms': True, 'cudnn_deterministic': True,
        'cudnn_benchmark': False, 'allow_tf32': False,
        'cublas_workspace_config': ':4096:8',
    }
    freeze = getattr(final_evaluation, 'freeze_evaluation_protocol', None)
    assert callable(freeze), 'complete evaluation protocol integration missing'
    captured = []

    def integrate(protocol, selections, cohort):
        retained = freeze(protocol, selections, cohort, numerical_policy=policy,
                          exposure_adjudication_bytes=adjudication,
                          expected_exposure_sha256=expected_sha)
        output = json.loads(retained)
        assert output['numerical_policy'] == policy
        assert output['exposure_adjudication_sha256'] == expected_sha
        assert len(output['models']) == 8
        captured.append(retained)
        return retained

    monkeypatch.setattr(helper, 'freeze_cohort_protocol', integrate)
    helper.test_original_teacher_and_six_unmodified_student_contracts_bind(tmp_path)
    assert len(captured) == 1


@pytest.mark.parametrize('mutation', ['adjudication', 'numerical', 'cohort'])
def test_full_protocol_rejects_invalid_bound_inputs(tmp_path, monkeypatch, mutation):
    helper_path = Path(__file__).with_name('test_final_real_contract_integration.py')
    spec = importlib.util.spec_from_file_location('original_negative_fixture', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    policy = {
        'model_dtype': 'bfloat16', 'ce_dtype': 'float32',
        'ce_reduction': 'shifted_valid_token_sum', 'ignore_index': -100,
        'batch_size': 1, 'sequence_length': 512,
        'deterministic_algorithms': True, 'cudnn_deterministic': True,
        'cudnn_benchmark': False, 'allow_tf32': False,
        'cublas_workspace_config': ':4096:8',
    }
    adjudication = b'bounded historical exposure evidence'
    sha = hashlib.sha256(adjudication).hexdigest()
    observed = []

    def integrate(protocol, selections, cohort):
        original = deepcopy((protocol, selections, cohort))
        supplied_cohort = deepcopy(cohort)
        supplied_policy = deepcopy(policy)
        supplied_bytes = adjudication
        if mutation == 'adjudication':
            supplied_bytes += b' altered'
        elif mutation == 'numerical':
            supplied_policy['ce_dtype'] = 'float16'
        else:
            supplied_cohort['provenance']['teacher']['manifest_sha256'] = '0' * 64
        with pytest.raises(ValueError):
            final_evaluation.freeze_evaluation_protocol(
                protocol, selections, supplied_cohort, numerical_policy=supplied_policy,
                exposure_adjudication_bytes=supplied_bytes,
                expected_exposure_sha256=sha)
        def forbidden_output_access(*args, **kwargs):
            raise AssertionError('output access before protocol authentication')
        with monkeypatch.context() as guarded:
            guarded.setattr(final_evaluation, 'reuse_completed_evaluation_roster',
                            forbidden_output_access)
            with pytest.raises(ValueError):
                final_evaluation.reuse_authenticated_evaluation_roster(
                    protocol, selections, supplied_cohort, output_paths={},
                    numerical_policy=supplied_policy,
                    exposure_adjudication_bytes=supplied_bytes,
                    expected_exposure_sha256=sha)
        assert (protocol, selections, cohort) == original
        observed.append(mutation)
        # Let the unchanged real-contract fixture verify its normal path too.
        return final_evaluation.freeze_cohort_protocol(protocol, selections, cohort)

    monkeypatch.setattr(helper, 'freeze_cohort_protocol', integrate)
    helper.test_original_teacher_and_six_unmodified_student_contracts_bind(tmp_path)
    assert observed == [mutation]
