"""지식 증류 학습 (Knowledge Distillation) — 핵심 모듈

KD Loss = α · CE(Student, Label) + (1-α) · T² · KL(Student ∥ Teacher)
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import KDConfig
from src.dataset import load_tokenizer, create_dataloaders
from src.distributed import (
    barrier,
    is_main_process,
    reduce_metrics,
    set_epoch,
    unwrap_model,
    wrap_ddp,
)
from src.models import create_optimizer, load_teacher, load_student, model_info


def kd_loss(student_logits, teacher_logits, labels, config: KDConfig):
    """Knowledge Distillation Loss 계산 (label shift 적용)

    Returns:
        total_loss, ce_loss, kd_loss_value (개별 추적용)
    """
    T = config.temperature
    alpha = config.alpha

    # Causal LM label shift: logits[t] → labels[t+1]
    shift_logits = student_logits[..., :-1, :].float().contiguous()
    shift_teacher = teacher_logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Hard Label Loss: Student vs 정답
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # Soft Label Loss: Student vs Teacher (Temperature 적용)
    kd_vocab_size = config.kd_vocab_size or min(
        shift_logits.size(-1), shift_teacher.size(-1)
    )
    if kd_vocab_size > min(shift_logits.size(-1), shift_teacher.size(-1)):
        raise ValueError(
            f"KD vocabulary size {kd_vocab_size} exceeds model logits: "
            f"student={shift_logits.size(-1)}, teacher={shift_teacher.size(-1)}"
        )
    teacher_log_soft = F.log_softmax(
        shift_teacher[..., :kd_vocab_size] / T, dim=-1
    )
    student_log_soft = F.log_softmax(
        shift_logits[..., :kd_vocab_size] / T, dim=-1
    )
    if config.kd_divergence == "reverse_kl":
        student_soft = student_log_soft.exp()
        token_kl = (
            student_soft * (student_log_soft - teacher_log_soft)
        ).sum(dim=-1)
    else:
        teacher_soft = teacher_log_soft.exp()
        token_kl = F.kl_div(
            student_log_soft, teacher_soft, reduction="none"
        ).sum(dim=-1)
    if config.kd_reduction == "tokenmean":
        valid_tokens = shift_labels != -100
        kd_loss_value = token_kl[valid_tokens].mean() * (T * T)
    else:
        kd_loss_value = token_kl.sum() / shift_logits.size(0) * (T * T)

    # Total Loss
    total_loss = alpha * ce_loss + (1 - alpha) * kd_loss_value

    return total_loss, ce_loss.item(), kd_loss_value.item()


def train_one_epoch(teacher, student, dataloader, optimizer, config: KDConfig):
    """1 에폭 학습"""
    student.train()
    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    kd_loss_sum = 0.0
    steps = 0

    progress = tqdm(
        dataloader,
        desc="Training",
        leave=False,
        disable=not is_main_process(),
    )
    teacher_device = next(teacher.parameters()).device
    for batch in progress:
        input_ids = batch["input_ids"].to(config.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(config.device, non_blocking=True)
        labels = batch["labels"].to(config.device, non_blocking=True)
        teacher_input_ids = batch["input_ids"].to(teacher_device, non_blocking=True)
        teacher_attention_mask = batch["attention_mask"].to(
            teacher_device, non_blocking=True
        )

        # Separate CUDA devices execute these queued forwards concurrently.
        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            )

        # Student forward
        student_outputs = student(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Loss 계산
        loss, ce, kd = kd_loss(
            student_outputs.logits,
            teacher_outputs.logits.to(config.device, non_blocking=True),
            labels,
            config,
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip)
        optimizer.step()

        total_loss_sum += loss.item()
        ce_loss_sum += ce
        kd_loss_sum += kd
        steps += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}", ce=f"{ce:.4f}", kd=f"{kd:.4f}"
        )
        if config.max_train_steps and steps >= config.max_train_steps:
            break

    return reduce_metrics({
        "total_loss": total_loss_sum / steps,
        "ce_loss": ce_loss_sum / steps,
        "kd_loss": kd_loss_sum / steps,
    }, config.device)


@torch.no_grad()
def validate(teacher, student, dataloader, config: KDConfig):
    """검증 (Validation)"""
    student.eval()
    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    steps = 0
    teacher_device = next(teacher.parameters()).device

    for batch in tqdm(
        dataloader,
        desc="Validation",
        leave=False,
        disable=not is_main_process(),
    ):
        input_ids = batch["input_ids"].to(config.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(config.device, non_blocking=True)
        labels = batch["labels"].to(config.device, non_blocking=True)
        teacher_input_ids = batch["input_ids"].to(teacher_device, non_blocking=True)
        teacher_attention_mask = batch["attention_mask"].to(
            teacher_device, non_blocking=True
        )

        teacher_outputs = teacher(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
        )
        student_outputs = student(
            input_ids=input_ids, attention_mask=attention_mask
        )

        loss, ce, kd = kd_loss(
            student_outputs.logits,
            teacher_outputs.logits.to(config.device, non_blocking=True),
            labels,
            config,
        )

        total_loss_sum += loss.item()
        ce_loss_sum += ce
        steps += 1
        if config.max_eval_steps and steps >= config.max_eval_steps:
            break

    student.train()
    return reduce_metrics({
        "val_total_loss": total_loss_sum / steps,
        "val_ce_loss": ce_loss_sum / steps,
    }, config.device)


def distill(config: KDConfig):
    """지식 증류 전체 파이프라인 실행"""
    print(f"=== Knowledge Distillation ===")
    print(f"Device: {config.device}")
    print(f"Teacher: {config.teacher_model} → Student: {config.student_model}")
    print(f"T={config.temperature}, α={config.alpha}, lr={config.learning_rate}")
    print(f"Epochs: {config.epochs}, Batch: {config.batch_size}, SeqLen: {config.max_seq_length}")
    print()

    config.ensure_dirs()

    # 토크나이저 & 데이터
    tokenizer = load_tokenizer(config.student_model)
    teacher_tokenizer = load_tokenizer(config.teacher_model)
    if teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError(
            "Teacher and student tokenizers must have identical vocabularies "
            "for logit-based knowledge distillation"
        )
    config.kd_vocab_size = len(tokenizer)
    loaders = create_dataloaders(config, tokenizer)
    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Val:   {len(loaders['validation'].dataset)} samples\n")

    # 모델 로드
    teacher = load_teacher(config)
    student = load_student(config)
    if config.distill_student_checkpoint:
        checkpoint_path = Path(config.distill_student_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Student initialization checkpoint not found: {checkpoint_path}"
            )
        student.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        print(f"  ✅ Student 초기 checkpoint 로드 → {checkpoint_path}")
    student = wrap_ddp(student, config.device)
    if is_main_process():
        model_info(teacher, "Teacher")
        model_info(student, "Student")
        print()

    # Optimizer
    optimizer = create_optimizer(student, config, config.learning_rate)

    # 학습 루프
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        set_epoch(loaders["train"], epoch)
        start = time.time()
        if is_main_process():
            print(f"--- Epoch {epoch}/{config.epochs} ---")

        train_metrics = train_one_epoch(
            teacher, student, loaders["train"], optimizer, config
        )
        val_metrics = validate(teacher, student, loaders["validation"], config)

        elapsed = time.time() - start
        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
        if is_main_process():
            history.append(metrics)

            print(
                f"  Train Loss: {metrics['total_loss']:.4f} "
                f"(CE: {metrics['ce_loss']:.4f}, KD: {metrics['kd_loss']:.4f})"
            )
            print(
                f"  Val Loss:   {metrics['val_total_loss']:.4f} "
                f"(CE: {metrics['val_ce_loss']:.4f})"
            )
            print(f"  Time: {elapsed:.1f}s")

        improved = (
            metrics["val_ce_loss"]
            < best_val_loss - config.early_stopping_min_delta
        )
        if improved:
            best_val_loss = metrics["val_ce_loss"]
            epochs_without_improvement = 0
            if is_main_process():
                save_path = config.checkpoint_dir / "student_kd_best.pt"
                torch.save(unwrap_model(student).state_dict(), save_path)
                print(f"  ✅ Best model saved → {save_path}")
        else:
            epochs_without_improvement += 1

        if is_main_process():
            print()
        barrier()
        if (
            config.early_stopping_patience
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            if is_main_process():
                print(
                    "  ⏹ Early stopping: validation CE did not improve "
                    f"for {epochs_without_improvement} epoch(s)."
                )
            break

    # 학습 로그 저장
    log_path = config.log_dir / "distill_history.json"
    if is_main_process():
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"학습 로그 저장 → {log_path}")
    barrier()

    return student, history


if __name__ == "__main__":
    from src.config import local_config

    config = local_config()
    student, history = distill(config)
