"""UI 전용 설정 (환경변수 로드)"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


# 프로젝트 루트 (knowledge-distillation/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# results/ 디렉토리
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"

# 환경변수
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEFAULT_RUN_ID = os.getenv("DEFAULT_RUN_ID", "").strip() or None
MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "2"))
DEVICE = os.getenv("DEVICE", "auto")


def resolve_device() -> str:
    """DEVICE 설정을 실제 torch device 문자열로 변환"""
    if DEVICE != "auto":
        return DEVICE
    # 지연 import (torch 로드 비용)
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
