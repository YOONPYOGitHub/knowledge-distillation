"""Final evaluation policy is separate from historical training identities."""
import copy
import pytest
from src import final_evaluation


def test_freezes_explicit_evaluation_policy_and_rejects_drift():
    policy = {
        'model_dtype': 'bfloat16', 'ce_dtype': 'float32',
        'ce_reduction': 'shifted_valid_token_sum', 'ignore_index': -100,
        'batch_size': 1, 'sequence_length': 512,
        'deterministic_algorithms': True, 'cudnn_deterministic': True,
        'cudnn_benchmark': False, 'allow_tf32': False,
        'cublas_workspace_config': ':4096:8',
    }
    freeze = getattr(final_evaluation, 'freeze_evaluation_numerical_policy', None)
    assert callable(freeze), 'missing separate evaluation numerical policy gate'
    import json
    assert json.loads(freeze(policy)) == policy
    for key, value in policy.items():
        changed = copy.deepcopy(policy)
        changed[key] = not value if type(value) is bool else ('changed' if isinstance(value, str) else value + 1)
        with pytest.raises(ValueError, match='evaluation numerical policy'):
            freeze(changed)
    changed = copy.deepcopy(policy)
    changed['batch_size'] = True
    with pytest.raises(ValueError, match='evaluation numerical policy'):
        freeze(changed)
    with pytest.raises(ValueError, match='evaluation numerical policy'):
        freeze({**policy, 'extra': 1})
    with pytest.raises(ValueError, match='evaluation numerical policy'):
        freeze(None)


def test_applies_and_reads_back_evaluation_backend_policy(monkeypatch):
    import os
    import torch
    policy = {
        'model_dtype': 'bfloat16', 'ce_dtype': 'float32',
        'ce_reduction': 'shifted_valid_token_sum', 'ignore_index': -100,
        'batch_size': 1, 'sequence_length': 512,
        'deterministic_algorithms': True, 'cudnn_deterministic': True,
        'cudnn_benchmark': False, 'allow_tf32': False,
        'cublas_workspace_config': ':4096:8',
    }
    old = (torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
           torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        apply = getattr(final_evaluation, 'apply_evaluation_backend_policy', None)
        assert callable(apply), 'missing evaluation backend policy application/readback'
        assert apply(policy) == {
            'deterministic_algorithms': True, 'cudnn_deterministic': True,
            'cudnn_benchmark': False, 'allow_tf32': False,
            'cublas_workspace_config': ':4096:8',
        }
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.deterministic = old[2]
        torch.backends.cudnn.benchmark = old[3]
        torch.backends.cuda.matmul.allow_tf32 = old[4]


@pytest.mark.parametrize('failure', ['invalid_policy', 'initialized_cuda', 'readback', 'warn_only'])
def test_backend_policy_runtime_failure_paths(monkeypatch, failure):
    import json
    import os
    import torch
    from src import numerical_policy
    policy = json.loads(final_evaluation.freeze_evaluation_numerical_policy({
        'model_dtype': 'bfloat16', 'ce_dtype': 'float32',
        'ce_reduction': 'shifted_valid_token_sum', 'ignore_index': -100,
        'batch_size': 1, 'sequence_length': 512,
        'deterministic_algorithms': True, 'cudnn_deterministic': True,
        'cudnn_benchmark': False, 'allow_tf32': False,
        'cublas_workspace_config': ':4096:8',
    }))
    def snapshot():
        return (os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                torch.backends.cuda.matmul.allow_tf32)
    old = snapshot()
    try:
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
        before = snapshot()
        if failure == 'invalid_policy':
            policy['batch_size'] = 2
            with pytest.raises(ValueError, match='evaluation numerical policy'):
                final_evaluation.apply_evaluation_backend_policy(policy)
            assert snapshot() == before
        elif failure == 'initialized_cuda':
            monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
            with pytest.raises(RuntimeError, match='CUDA already initialized'):
                final_evaluation.apply_evaluation_backend_policy(policy)
            assert snapshot() == before
        else:
            # Only the effective-readback fault is injected; real configuration runs.
            if failure == 'readback':
                monkeypatch.setattr(torch, 'are_deterministic_algorithms_enabled', lambda: False)
            else:
                monkeypatch.setattr(torch, 'is_deterministic_algorithms_warn_only_enabled', lambda: True)
            with pytest.raises(RuntimeError, match='evaluation backend policy readback mismatch'):
                final_evaluation.apply_evaluation_backend_policy(policy)
    finally:
        torch.use_deterministic_algorithms(old[1], warn_only=old[2])
        torch.backends.cudnn.deterministic = old[3]
        torch.backends.cudnn.benchmark = old[4]
        torch.backends.cuda.matmul.allow_tf32 = old[5]
