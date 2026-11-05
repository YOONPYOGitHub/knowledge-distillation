import os
from unittest.mock import patch
import pytest
import torch


def test_numerical_policy_is_explicit_and_reproducible(monkeypatch):
    from src.numerical_policy import configure_numerics
    old=(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    try:
        with patch.object(torch.cuda,'is_initialized',return_value=False):
            policy=configure_numerics()
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        assert policy=={'deterministic_algorithms':True,'cudnn_deterministic':True,'cudnn_benchmark':False,'allow_tf32':False,'cublas_workspace_config':':4096:8'}
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.use_deterministic_algorithms(old[0]);torch.backends.cudnn.deterministic=old[1];torch.backends.cudnn.benchmark=old[2];torch.backends.cuda.matmul.allow_tf32=old[3]


def test_policy_refuses_wrong_workspace_after_cuda_initialized(monkeypatch):
    from src.numerical_policy import configure_numerics
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG','wrong')
    with patch.object(torch.cuda,'is_initialized',return_value=True),pytest.raises(RuntimeError,match='CUDA'):
        configure_numerics()
