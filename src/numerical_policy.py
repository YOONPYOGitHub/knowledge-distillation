"""Explicit deterministic numerical contract validated on the A100 replay probe."""
import os


def configure_numerics():
    import torch
    workspace=':4096:8'
    if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=workspace:
        raise RuntimeError('CUDA already initialized with incompatible CUBLAS workspace; start a fresh process')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=workspace
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    return {'deterministic_algorithms':True,'cudnn_deterministic':True,
            'cudnn_benchmark':False,'allow_tf32':False,
            'cublas_workspace_config':workspace}
