"""Teacher/Student 모델 로드"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.optimization import Adafactor

from src.config import KDConfig


def model_dtype_kwargs(config: KDConfig, device: str | None = None) -> dict:
    """Return model loading dtype options for the target device."""
    target_device = device or config.device
    if config.bf16 and target_device.startswith("cuda"):
        return {"torch_dtype": torch.bfloat16}
    if config.fp16 and target_device.startswith("cuda"):
        return {"torch_dtype": torch.float16}
    return {}


def teacher_quantization_config(config: KDConfig, mode: str | None = None):
    """Teacher의 bitsandbytes 양자화 설정 (없으면 None)

    추론(KD/평가)에서는 Teacher가 gradient를 받지 않으므로 양자화해도 학습 안정성에
    영향이 없고, 12B급 Teacher가 24GB 한 장에 들어가 rank별 전용 Teacher 구성이 가능해진다.
    다만 Teacher logits 자체가 KD의 목표 분포이므로 양자화 오차는 증류 신호에 실린다.

    mode를 넘기면 그 값을 쓰고, 없으면 config.teacher_quantization을 따른다.
    Teacher fine-tuning(QLoRA)은 config.teacher_ft_quantization을 넘겨서 사용한다.
    """
    mode = config.teacher_quantization if mode is None else mode
    if not mode:
        return None

    from transformers import BitsAndBytesConfig

    if config.bf16:
        compute_dtype = torch.bfloat16
    elif config.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=mode,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_teacher(config: KDConfig) -> AutoModelForCausalLM:
    """Teacher 모델 로드 (추론 전용, 가중치 고정)

    teacher_checkpoint가 설정되어 있으면 fine-tuned 가중치를 로드.
    비어있으면 pretrained 그대로 사용.
    """
    kwargs = model_dtype_kwargs(config, config.teacher_device)
    quantization = teacher_quantization_config(config)

    if quantization is not None:
        # bitsandbytes 모델은 로드 시점에 배치가 끝나야 하며 이후 .to() 로 옮길 수 없다.
        kwargs["quantization_config"] = quantization
        kwargs["device_map"] = config.teacher_device_map or {"": config.teacher_device}
    elif config.teacher_device_map:
        # 양자화 없이 한 장에 안 들어가는 Teacher는 여러 GPU로 분할한다.
        kwargs["device_map"] = config.teacher_device_map

    placed_at_load = "device_map" in kwargs
    if placed_at_load and config.teacher_max_memory:
        kwargs["max_memory"] = {
            (int(key) if key.isdigit() else key): value
            for key, value in config.teacher_max_memory.items()
        }

    model = AutoModelForCausalLM.from_pretrained(config.teacher_model, **kwargs)

    # Fine-tuned checkpoint가 있으면 로드
    if config.teacher_checkpoint:
        checkpoint_path = Path(config.teacher_checkpoint)
        adapter_config = checkpoint_path / "adapter_config.json"
        if adapter_config.exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, checkpoint_path)
            if quantization is None:
                model = model.merge_and_unload()
                print(f"  ✅ Teacher LoRA adapter 로드 → {checkpoint_path}")
            else:
                # 양자화된 base 에 adapter 를 병합하면 가중치가 손상된다. 래퍼로 유지한다.
                print(f"  ✅ Teacher LoRA adapter 로드 (병합 없음, 양자화) → {checkpoint_path}")
        elif checkpoint_path.is_file():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"  ✅ Teacher FT 가중치 로드 → {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    if not placed_at_load:
        model.to(config.teacher_device)
    model.eval()

    # 가중치 고정 — 학습 시 Teacher는 업데이트하지 않음
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_student(config: KDConfig) -> AutoModelForCausalLM:
    """Student 모델 로드 (학습 대상)"""
    kwargs = model_dtype_kwargs(config)
    model = AutoModelForCausalLM.from_pretrained(config.student_model, **kwargs)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(config.device)
    model.train()
    return model


def create_optimizer(model, config: KDConfig, learning_rate: float):
    """Create the configured optimizer for a trainable model."""
    if config.optimizer == "adafactor":
        return Adafactor(
            model.parameters(),
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=config.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )


def model_info(model: AutoModelForCausalLM, name: str = "Model"):
    """모델 파라미터 수 및 메모리 정보 출력"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print(f"[{name}]")
    print(f"  파라미터: {total:,} (학습 가능: {trainable:,})")
    print(f"  메모리: {size_mb:.1f} MB")
    print(f"  디바이스: {next(model.parameters()).device}")
