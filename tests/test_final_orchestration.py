"""Integrated CPU-only scoring/publication/reuse; no real heldout or GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from src import final_evaluation as fe


def test_authenticated_missing_roster_scores_publishes_and_reuses(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('original_fixture', Path(__file__).with_name('test_final_real_contract_integration.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    policy = dict(model_dtype='bfloat16', ce_dtype='float32', ce_reduction='shifted_valid_token_sum', ignore_index=-100, batch_size=1, sequence_length=512, deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False, allow_tf32=False, cublas_workspace_config=':4096:8')
    adjudication = b'CPU fixture adjudication, not actual heldout authorization'
    kwargs = dict(numerical_policy=policy, exposure_adjudication_bytes=adjudication, expected_exposure_sha256=hashlib.sha256(adjudication).hexdigest())
    seen = []

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        def forward(self, input_ids):
            return type('Output', (), {'logits': self.weight[input_ids]})()

    def integrate(protocol, selections, cohort):
        frozen = fe.freeze_evaluation_protocol(protocol, selections, cohort, **kwargs)
        paths = {entry['id']: str(tmp_path / (entry['id'] + '-evaluation.json')) for entry in protocol['models']}
        source = b'fixture source identity'
        operation = fe.freeze_evaluation_operation(protocol_bytes=frozen, source_sha256=hashlib.sha256(source).hexdigest(), max_seconds=60, max_batches_per_model=1, output_paths=paths)
        def load(entry):
            seen.append(entry['id'])
            return Tiny()
        def batches(entry):
            ids = torch.zeros((1, 512), dtype=torch.long)
            class CompleteStream:
                def __len__(self):
                    return 1
                def __iter__(self):
                    yield ids, ids.clone()
                    raise AssertionError('unauthorized overfetch')
            return CompleteStream()
        runner = getattr(fe, 'run_authenticated_evaluation', None)
        assert callable(runner), 'authenticated missing-model orchestration missing'
        result = runner(protocol, selections, cohort, operation_bytes=operation, source_bytes=source, load_model=load, iter_batches=batches, **kwargs)
        assert len(result) == 8
        assert len(seen) == 8
        for row in result.values():
            assert row['statistics']['tokens'] == 511
            assert abs(row['statistics']['ppl'] - 4) < 1e-5
        originals = {name: Path(path).read_bytes() for name, path in paths.items()}
        again = runner(protocol, selections, cohort, operation_bytes=operation, source_bytes=source, load_model=load, iter_batches=batches, **kwargs)
        assert again == result
        assert len(seen) == 8
        assert originals == {name: Path(path).read_bytes() for name, path in paths.items()}
        import pytest
        changed_source = b'different execution source'
        changed_operation = fe.freeze_evaluation_operation(protocol_bytes=frozen, source_sha256=hashlib.sha256(changed_source).hexdigest(), max_seconds=60, max_batches_per_model=1, output_paths=paths)
        with pytest.raises(ValueError):
            runner(protocol, selections, cohort, operation_bytes=changed_operation, source_bytes=changed_source, load_model=load, iter_batches=batches, **kwargs)
        assert len(seen) == 8
        assert originals == {name: Path(path).read_bytes() for name, path in paths.items()}
        return frozen
    monkeypatch.setattr(helper, 'freeze_cohort_protocol', integrate)
    helper.test_original_teacher_and_six_unmodified_student_contracts_bind(tmp_path)
