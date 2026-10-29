"""Epoch 단위 학습 재개.

``resume_training: true`` 이면 각 학습 단계(teacher / baseline / distill)가 epoch 이 끝날 때마다
``<checkpoint_dir>/<stage>_last.pt`` 에 다음을 저장한다.

- 학습 가능한 파라미터 (LoRA Teacher 는 어댑터만, Student 는 전체)
- optimizer 상태
- epoch, best val CE, early stopping 카운터, history

같은 설정으로 다시 실행하면 다음 epoch 부터 이어간다. 학습 데이터 순서는 batch sampler 가
``seed + epoch`` 으로 정하므로 epoch 경계에서 이어가도 같은 순서를 본다. dropout 등 모델 내부
난수는 복원하지 않으므로 bit 단위로 같은 결과는 아니다.

설정이 바뀌었으면(지문 불일치) 이어가지 않고 처음부터 학습한다. 단계가 끝나면 파일을 지운다.
"""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import KDConfig
from src.distributed import barrier, is_main_process, unwrap_model

# 이어가기 판단에서 제외하는 필드. 장치 배치나 학습 길이는 바꿔서 이어가도 된다.
_FINGERPRINT_EXCLUDE = {
    "kd_vocab_size",  # distill 이 실행 중에 채운다
    "device",
    "teacher_device",
    "teacher_devices",
    "teacher_device_map",
    "teacher_max_memory",
    "num_workers",
    "resume_training",
    "epochs",
    "teacher_epochs",
    "early_stopping_patience",
}


def state_path(config: KDConfig, stage: str) -> Path:
    return config.checkpoint_dir / f"{stage}_last.pt"


def config_fingerprint(config: KDConfig) -> str:
    values = {k: v for k, v in asdict(config).items() if k not in _FINGERPRINT_EXCLUDE}
    encoded = json.dumps(values, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_state(config, stage, model, optimizer, epoch, best_val_loss,
               epochs_without_improvement, history) -> None:
    """epoch 종료 시점의 학습 상태를 저장한다. 모든 rank 가 호출한다."""
    if not config.resume_training:
        return
    if is_main_process():
        trainable = {
            name: param.detach().cpu()
            for name, param in unwrap_model(model).named_parameters()
            if param.requires_grad
        }
        path = state_path(config, stage)
        tmp = path.with_suffix(".tmp")
        torch.save({
            "fingerprint": config_fingerprint(config),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "trainable": trainable,
            "optimizer": optimizer.state_dict(),
        }, tmp)
        # 저장 도중 중단돼도 직전 epoch 의 상태가 남도록 원자적으로 바꾼다.
        os.replace(tmp, path)
    barrier()


def load_state(config, stage, model, optimizer) -> dict | None:
    """저장된 상태가 있으면 model/optimizer 에 복원하고 루프 상태를 돌려준다."""
    path = state_path(config, stage)
    if not config.resume_training or not path.is_file():
        return None

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["fingerprint"] != config_fingerprint(config):
        if is_main_process():
            print(f"  ⚠️  설정이 달라 이어가지 않고 처음부터 학습합니다: {path}")
        return None

    params = dict(unwrap_model(model).named_parameters())
    expected = {name for name, param in params.items() if param.requires_grad}
    if set(state["trainable"]) != expected:
        raise ValueError(f"재개 상태의 학습 파라미터가 현재 모델과 다릅니다: {path}")
    with torch.no_grad():
        for name, value in state["trainable"].items():
            params[name].copy_(value)

    optimizer.load_state_dict(state["optimizer"])
    # load_state_dict 는 상태 텐서를 파라미터 dtype 으로 바꾼다. Adafactor 는 bf16 파라미터의
    # 2차 모멘트를 fp32 로 들고 있으므로 저장된 dtype 을 그대로 되돌린다.
    ordered = [p for group in optimizer.param_groups for p in group["params"]]
    for index, values in state["optimizer"]["state"].items():
        target = optimizer.state[ordered[index]]
        for key, value in values.items():
            if torch.is_tensor(value):
                target[key] = value.to(ordered[index].device)

    if is_main_process():
        print(f"  ⏯  epoch {state['epoch']} 까지의 상태를 복원했습니다 → {path}")
    return state


def clear_state(config: KDConfig, stage: str) -> None:
    """단계가 끝나면 재개 상태를 지운다. 다음 실행은 처음부터 학습한다."""
    if config.resume_training and is_main_process():
        state_path(config, stage).unlink(missing_ok=True)
    barrier()
