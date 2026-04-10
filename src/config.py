"""하이퍼파라미터 & 경로 설정"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import yaml


def get_device() -> torch.device:
    """CUDA > MPS > CPU 순서로 디바이스 자동 감지"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class KDConfig:
    # --- 모델 ---
    teacher_model: str = "gpt2"
    student_model: str = "distilgpt2"

    # --- 데이터 ---
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_seq_length: int = 512

    # --- KD 하이퍼파라미터 ---
    temperature: float = 3.0
    alpha: float = 0.5  # CE vs KD 비율 (1.0 = CE만, 0.0 = KD만)

    # --- Teacher Fine-tuning ---
    teacher_epochs: int = 3
    teacher_learning_rate: float = 2e-5
    teacher_checkpoint: str = ""  # 비어있으면 pretrained 그대로, 경로 있으면 FT된 teacher 로드

    # --- 학습 ---
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0

    # --- 경로 ---
    output_dir: str = "results"
    run_id: str = ""  # 비어있으면 자동 생성 (YYYYMMDD_HHMMSS)

    # --- 디바이스 ---
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # --- 기타 ---
    seed: int = 42
    num_workers: int = 0  # DataLoader workers (MPS에서는 0 권장)
    fp16: bool = False  # CUDA 서버에서 활성화

    def __post_init__(self):
        if self.device == "auto":
            self.device = str(get_device())
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoints" / self.run_id

    @property
    def log_dir(self) -> Path:
        return Path(self.output_dir) / "logs" / self.run_id

    @property
    def figure_dir(self) -> Path:
        return Path(self.output_dir) / "figures" / self.run_id

    def ensure_dirs(self):
        """결과 저장 디렉토리 생성"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)


# --- 프리셋: 로컬(맥북) / 서버(GPU) ---

def local_config(**overrides) -> KDConfig:
    """맥북 로컬 개발/디버깅용 설정"""
    defaults = dict(
        teacher_model="gpt2",
        student_model="distilgpt2",
        batch_size=2,
        epochs=1,
        max_seq_length=128,
        num_workers=0,
        fp16=False,
    )
    defaults.update(overrides)
    return KDConfig(**defaults)


def server_config(**overrides) -> KDConfig:
    """GPU 서버 본 실험용 설정"""
    defaults = dict(
        teacher_model="gpt2-medium",
        student_model="distilgpt2",
        batch_size=8,
        epochs=3,
        max_seq_length=512,
        num_workers=4,
        fp16=True,
    )
    defaults.update(overrides)
    return KDConfig(**defaults)


def from_yaml(path: str, **overrides) -> KDConfig:
    """YAML 파일에서 설정 로드"""
    with open(path) as f:
        data = yaml.safe_load(f)
    data.update(overrides)
    return KDConfig(**data)
