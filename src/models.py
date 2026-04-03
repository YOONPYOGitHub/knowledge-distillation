"""Teacher/Student 모델 로드"""

import torch
from transformers import AutoModelForCausalLM

from src.config import KDConfig


def load_teacher(config: KDConfig) -> AutoModelForCausalLM:
    """Teacher 모델 로드 (추론 전용, 가중치 고정)"""
    kwargs = {}
    if config.fp16 and config.device == "cuda":
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(config.teacher_model, **kwargs)
    model.to(config.device)
    model.eval()

    # 가중치 고정 — 학습 시 Teacher는 업데이트하지 않음
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_student(config: KDConfig) -> AutoModelForCausalLM:
    """Student 모델 로드 (학습 대상)"""
    model = AutoModelForCausalLM.from_pretrained(config.student_model)
    model.to(config.device)
    model.train()
    return model


def model_info(model: AutoModelForCausalLM, name: str = "Model"):
    """모델 파라미터 수 및 메모리 정보 출력"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print(f"[{name}]")
    print(f"  파라미터: {total:,} (학습 가능: {trainable:,})")
    print(f"  메모리: {size_mb:.1f} MB")
    print(f"  디바이스: {next(model.parameters()).device}")
