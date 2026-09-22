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
    # 청크마다 BOS 를 붙인다. Gemma 처럼 BOS 를 전제로 학습된 모델에 필요하다.
    prepend_bos: bool = False

    # --- KD 하이퍼파라미터 ---
    temperature: float = 3.0
    alpha: float = 0.5  # CE vs KD 비율 (1.0 = CE만, 0.0 = KD만)
    kd_vocab_size: int = 0  # 동일 tokenizer의 실제 vocabulary 크기
    kd_reduction: str = "batchmean"  # "batchmean" 또는 "tokenmean"
    kd_divergence: str = "forward_kl"  # "forward_kl" 또는 "reverse_kl"
    # KD 목표 분포를 Teacher 상위 K개 토큰으로 제한한다. 0이면 full vocabulary.
    # Gemma 처럼 vocabulary 가 큰 모델에서 KD 손실의 fp32 버퍼를 줄인다.
    kd_top_k: int = 0

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
    global_batch_size: int = 0  # Opt-in identical global batches across DDP world sizes
    training_seed: int | None = None  # Separate model/shuffle RNG from dataset split seed
    max_train_steps: int = 0  # 0이면 전체 epoch 사용
    max_eval_steps: int = 0  # 0이면 전체 validation 사용
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    optimizer: str = "adamw"  # "adamw" 또는 메모리 절약형 "adafactor"
    gradient_checkpointing: bool = False
    distill_student_checkpoint: str = ""  # KD 전용 Student 초기 checkpoint
    early_stopping_patience: int = 0  # 0이면 비활성화
    early_stopping_min_delta: float = 0.0

    # --- 경로 ---
    output_dir: str = "results"
    run_id: str = ""  # 비어있으면 자동 생성 (YYYYMMDD_HHMMSS)

    # --- 디바이스 ---
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    teacher_device: str = ""  # 비어있으면 device와 동일
    teacher_devices: list[str] = field(default_factory=list)  # One dedicated teacher GPU per rank
    # 한 장에 올라가지 않는 Teacher를 여러 GPU로 분할 (accelerate device_map). 예: "auto"
    teacher_device_map: str = ""
    # device_map 사용 시 GPU별 상한. 예: {"0": "22GiB", "1": "22GiB", "2": "0GiB", "3": "0GiB"}
    teacher_max_memory: dict = field(default_factory=dict)
    # 추론 전용 Teacher 양자화: "" | "int8" | "nf4" | "fp4".
    # 12B급 Teacher를 24GB 한 장에 올려 rank별 전용 Teacher 구성을 가능하게 한다.
    teacher_quantization: str = ""
    # Teacher fine-tuning 단계의 base 양자화 (QLoRA). LoRA rank가 필요하다.
    teacher_ft_quantization: str = ""

    # --- 기타 ---
    seed: int = 42
    num_workers: int = 0  # DataLoader workers (MPS에서는 0 권장)
    fp16: bool = False  # CUDA 서버에서 활성화
    bf16: bool = False  # RTX 30 시리즈 이상에서 권장
    # attention 구현 지정. Gemma 3 는 학습 시 "eager" 를 권장한다 (비우면 라이브러리 기본값).
    attn_implementation: str = ""

    def __post_init__(self):
        if self.batch_size <= 0 or self.global_batch_size < 0:
            raise ValueError("batch_size must be positive and global_batch_size nonnegative")
        if self.global_batch_size and self.kd_reduction != "tokenmean":
            raise ValueError("global_batch_size requires tokenmean KD for masked final batches")
        if len(set(self.teacher_devices)) != len(self.teacher_devices):
            raise ValueError("teacher_devices must be distinct")
        if self.teacher_device_map and self.teacher_devices:
            raise ValueError("teacher_device_map cannot be combined with per-rank teacher_devices")
        if self.teacher_max_memory and not self.teacher_device_map:
            raise ValueError("teacher_max_memory requires teacher_device_map")
        if self.teacher_quantization not in {"", "int8", "nf4", "fp4"}:
            raise ValueError("teacher_quantization must be '', 'int8', 'nf4', or 'fp4'")
        if self.teacher_ft_quantization not in {"", "int8", "nf4", "fp4"}:
            raise ValueError("teacher_ft_quantization must be '', 'int8', 'nf4', or 'fp4'")
        if self.teacher_ft_quantization and not self.teacher_lora_rank:
            # 양자화된 base 는 직접 학습할 수 없다. 학습되는 파라미터가 하나도 없게 된다.
            raise ValueError("teacher_ft_quantization requires teacher_lora_rank > 0 (QLoRA)")
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
        if self.kd_divergence not in {"forward_kl", "reverse_kl"}:
            raise ValueError("kd_divergence must be 'forward_kl' or 'reverse_kl'")
        if self.kd_top_k < 0:
            raise ValueError("kd_top_k must be zero or greater")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be zero or greater")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be zero or greater")
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
