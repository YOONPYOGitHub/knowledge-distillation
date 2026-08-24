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
    dataset_config: str | None = "wikitext-2-raw-v1"
    dataset_revision: str | None = None
    dataset_text_column: str = "text"
    dataset_streaming: bool = False
    dataset_max_samples: int = 0  # 0이면 전체 사용
    dataset_shuffle_buffer: int = 10_000
    validation_ratio: float = 0.01
    test_ratio: float = 0.01
    max_seq_length: int = 512

    # --- KD 하이퍼파라미터 ---
    temperature: float = 3.0
    alpha: float = 0.5  # CE vs KD 비율 (1.0 = CE만, 0.0 = KD만)
    kd_vocab_size: int = 0  # 동일 tokenizer의 실제 vocabulary 크기
    kd_reduction: str = "batchmean"  # "batchmean" 또는 "tokenmean"

    # --- Teacher Fine-tuning ---
    teacher_epochs: int = 3
    teacher_learning_rate: float = 2e-5
    teacher_checkpoint: str = ""  # 비어있으면 pretrained 그대로, 경로 있으면 FT된 teacher 로드
    teacher_lora_rank: int = 0  # 0이면 full fine-tuning
    teacher_lora_alpha: int = 32
    teacher_lora_dropout: float = 0.05
    teacher_lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # --- 학습 ---
    epochs: int = 3
    batch_size: int = 8
    max_train_steps: int = 0  # 0이면 전체 epoch 사용
    max_eval_steps: int = 0  # 0이면 전체 validation 사용
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    optimizer: str = "adamw"  # "adamw" 또는 메모리 절약형 "adafactor"
    gradient_checkpointing: bool = False

    # --- 경로 ---
    output_dir: str = "results"
    run_id: str = ""  # 비어있으면 자동 생성 (YYYYMMDD_HHMMSS)

    # --- 디바이스 ---
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    teacher_device: str = ""  # 비어있으면 device와 동일

    # --- 기타 ---
    seed: int = 42
    num_workers: int = 0  # DataLoader workers (MPS에서는 0 권장)
    fp16: bool = False  # CUDA 서버에서 활성화
    bf16: bool = False  # RTX 30 시리즈 이상에서 권장

    def __post_init__(self):
        if self.dataset_max_samples < 0:
            raise ValueError("dataset_max_samples must be zero or greater")
        if self.validation_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("validation_ratio and test_ratio must be greater than zero")
        if self.validation_ratio + self.test_ratio >= 1:
            raise ValueError("validation_ratio + test_ratio must be less than one")
        if self.dataset_streaming and self.dataset_max_samples == 0:
            raise ValueError("streaming datasets require dataset_max_samples")
        if self.dataset_shuffle_buffer <= 0:
            raise ValueError("dataset_shuffle_buffer must be greater than zero")
        if self.optimizer not in {"adamw", "adafactor"}:
            raise ValueError("optimizer must be 'adamw' or 'adafactor'")
        if self.max_train_steps < 0 or self.max_eval_steps < 0:
            raise ValueError("max_train_steps and max_eval_steps must be zero or greater")
        if self.teacher_lora_rank < 0:
            raise ValueError("teacher_lora_rank must be zero or greater")
        if self.kd_reduction not in {"batchmean", "tokenmean"}:
            raise ValueError("kd_reduction must be 'batchmean' or 'tokenmean'")
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be enabled")
        if self.device == "auto":
            self.device = str(get_device())
        if not self.teacher_device:
            self.teacher_device = self.device
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
