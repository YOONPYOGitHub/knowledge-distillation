"""지식 증류 학습 (Knowledge Distillation) — 핵심 모듈

KD Loss = α · CE(Student, Label) + (1-α) · T² · KL(Student ∥ Teacher)
"""

import json
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.config import KDConfig
from src.dataset import load_tokenizer, create_dataloaders
from src.models import load_teacher, load_student, model_info


def kd_loss(student_logits, teacher_logits, labels, config: KDConfig):
    """Knowledge Distillation Loss 계산

    Returns:
        total_loss, ce_loss, kd_loss_value (개별 추적용)
    """
    T = config.temperature
    alpha = config.alpha

    # Hard Label Loss: Student vs 정답
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    # Soft Label Loss: Student vs Teacher (Temperature 적용)
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)
    student_log_soft = F.log_softmax(student_logits / T, dim=-1)
    kd_loss_value = F.kl_div(
        student_log_soft, teacher_soft, reduction="batchmean"
    ) * (T * T)

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

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        # Teacher forward (no gradient)
        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids=input_ids, attention_mask=attention_mask
            )

        # Student forward
        student_outputs = student(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Loss 계산
        loss, ce, kd = kd_loss(
            student_outputs.logits, teacher_outputs.logits, labels, config
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

    return {
        "total_loss": total_loss_sum / steps,
        "ce_loss": ce_loss_sum / steps,
        "kd_loss": kd_loss_sum / steps,
    }


@torch.no_grad()
def validate(teacher, student, dataloader, config: KDConfig):
    """검증 (Validation)"""
    student.eval()
    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        teacher_outputs = teacher(
            input_ids=input_ids, attention_mask=attention_mask
        )
        student_outputs = student(
            input_ids=input_ids, attention_mask=attention_mask
        )

        loss, ce, kd = kd_loss(
            student_outputs.logits, teacher_outputs.logits, labels, config
        )

        total_loss_sum += loss.item()
        ce_loss_sum += ce
        steps += 1

    student.train()
    return {
        "val_total_loss": total_loss_sum / steps,
        "val_ce_loss": ce_loss_sum / steps,
    }


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
    loaders = create_dataloaders(config, tokenizer)
    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Val:   {len(loaders['validation'].dataset)} samples\n")

    # 모델 로드
    teacher = load_teacher(config)
    model_info(teacher, "Teacher")
    student = load_student(config)
    model_info(student, "Student")
    print()

    # Optimizer
    optimizer = AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 학습 루프
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        print(f"--- Epoch {epoch}/{config.epochs} ---")

        train_metrics = train_one_epoch(
            teacher, student, loaders["train"], optimizer, config
        )
        val_metrics = validate(teacher, student, loaders["validation"], config)

        elapsed = time.time() - start
        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
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

        # Best model 저장
        if metrics["val_total_loss"] < best_val_loss:
            best_val_loss = metrics["val_total_loss"]
            save_path = config.checkpoint_dir / "student_kd_best.pt"
            torch.save(student.state_dict(), save_path)
            print(f"  ✅ Best model saved → {save_path}")

        print()

    # 학습 로그 저장
    log_path = config.log_dir / "distill_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"학습 로그 저장 → {log_path}")

    return student, history


if __name__ == "__main__":
    from src.config import local_config

    config = local_config()
    student, history = distill(config)
