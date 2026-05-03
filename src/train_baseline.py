"""Student 자체 Fine-tuning (베이스라인) — KD 없이 CE Loss만으로 학습

distill.py와 동일 조건, Loss 함수만 다름 (공정 비교용)
"""

import json
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.config import KDConfig
from src.dataset import load_tokenizer, create_dataloaders
from src.models import load_student, model_info


def train_one_epoch(student, dataloader, optimizer, config: KDConfig):
    """1 에폭 학습 (CE Loss만)"""
    student.train()
    loss_sum = 0.0
    steps = 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        outputs = student(input_ids=input_ids, attention_mask=attention_mask)

        # Causal LM label shift: logits[t] → labels[t+1]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip)
        optimizer.step()

        loss_sum += loss.item()
        steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {"ce_loss": loss_sum / steps}


@torch.no_grad()
def validate(student, dataloader, config: KDConfig):
    """검증"""
    student.eval()
    loss_sum = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        outputs = student(input_ids=input_ids, attention_mask=attention_mask)

        # Causal LM label shift
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        loss_sum += loss.item()
        steps += 1

    student.train()
    return {"val_ce_loss": loss_sum / steps}


def train_baseline(config: KDConfig):
    """베이스라인 Fine-tuning 전체 파이프라인"""
    print(f"=== Baseline Fine-tuning (CE only) ===")
    print(f"Device: {config.device}")
    print(f"Student: {config.student_model}")
    print(f"lr={config.learning_rate}")
    print(f"Epochs: {config.epochs}, Batch: {config.batch_size}, SeqLen: {config.max_seq_length}")
    print()

    config.ensure_dirs()

    # 토크나이저 & 데이터
    tokenizer = load_tokenizer(config.student_model)
    loaders = create_dataloaders(config, tokenizer)
    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Val:   {len(loaders['validation'].dataset)} samples\n")

    # 모델 로드
    student = load_student(config)
    model_info(student, "Student (FT)")
    print()

    # Optimizer (distill.py와 동일)
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

        train_metrics = train_one_epoch(student, loaders["train"], optimizer, config)
        val_metrics = validate(student, loaders["validation"], config)

        elapsed = time.time() - start
        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
        history.append(metrics)

        print(f"  Train CE Loss: {metrics['ce_loss']:.4f}")
        print(f"  Val CE Loss:   {metrics['val_ce_loss']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Best model 저장
        if metrics["val_ce_loss"] < best_val_loss:
            best_val_loss = metrics["val_ce_loss"]
            save_path = config.checkpoint_dir / "student_ft_best.pt"
            torch.save(student.state_dict(), save_path)
            print(f"  ✅ Best model saved → {save_path}")

        print()

    # 학습 로그 저장
    log_path = config.log_dir / "baseline_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"학습 로그 저장 → {log_path}")

    return student, history
 

if __name__ == "__main__":
    from src.config import local_config

    config = local_config()
    student, history = train_baseline(config)
