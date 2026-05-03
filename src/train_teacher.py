"""Teacher Fine-tuning — KD 전에 Teacher를 target 도메인에 적응

Teacher가 target 데이터셋에서 Student FT보다 충분히 낮은 PPL을 달성해야
KD의 soft target이 의미 있는 dark knowledge를 전달할 수 있음.

파이프라인: train_teacher → distill → baseline → evaluate → compare
"""

import json
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.config import KDConfig
from src.dataset import load_tokenizer, create_dataloaders
from src.models import model_info

from transformers import AutoModelForCausalLM


def train_one_epoch(model, dataloader, optimizer, config: KDConfig):
    """1 에폭 학습 (CE Loss)"""
    model.train()
    loss_sum = 0.0
    steps = 0

    progress = tqdm(dataloader, desc="Teacher Training", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        loss_sum += loss.item()
        steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {"ce_loss": loss_sum / steps}


@torch.no_grad()
def validate(model, dataloader, config: KDConfig):
    """검증"""
    model.eval()
    loss_sum = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Teacher Validation", leave=False):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        loss_sum += loss.item()
        steps += 1

    return {"val_ce_loss": loss_sum / steps}


def train_teacher(config: KDConfig):
    """Teacher Fine-tuning 전체 파이프라인"""
    print(f"=== Teacher Fine-tuning ===")
    print(f"Device: {config.device}")
    print(f"Teacher: {config.teacher_model}")
    print(f"lr={config.teacher_learning_rate}, epochs={config.teacher_epochs}")
    print(f"Batch: {config.batch_size}, SeqLen: {config.max_seq_length}")
    print()

    config.ensure_dirs()

    # 토크나이저 & 데이터 (student 토크나이저 — GPT-2 계열 공유)
    tokenizer = load_tokenizer(config.student_model)
    loaders = create_dataloaders(config, tokenizer)
    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Val:   {len(loaders['validation'].dataset)} samples\n")

    # Teacher 모델 로드 (학습 모드)
    kwargs = {}
    if config.fp16 and config.device == "cuda":
        kwargs["torch_dtype"] = torch.float16

    teacher = AutoModelForCausalLM.from_pretrained(config.teacher_model, **kwargs)
    teacher.to(config.device)
    teacher.train()
    model_info(teacher, "Teacher (FT)")
    print()

    # Optimizer
    optimizer = AdamW(
        teacher.parameters(),
        lr=config.teacher_learning_rate,
        weight_decay=config.weight_decay,
    )

    # 학습 루프
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, config.teacher_epochs + 1):
        start = time.time()
        print(f"--- Teacher Epoch {epoch}/{config.teacher_epochs} ---")

        train_metrics = train_one_epoch(teacher, loaders["train"], optimizer, config)
        val_metrics = validate(teacher, loaders["validation"], config)

        elapsed = time.time() - start
        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
        history.append(metrics)

        print(f"  Train CE Loss: {metrics['ce_loss']:.4f}")
        print(f"  Val CE Loss:   {metrics['val_ce_loss']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Best model 저장
        if metrics["val_ce_loss"] < best_val_loss:
            best_val_loss = metrics["val_ce_loss"]
            save_path = config.checkpoint_dir / "teacher_ft_best.pt"
            torch.save(teacher.state_dict(), save_path)
            print(f"  ✅ Best teacher saved → {save_path}")

        print()

    # 학습 로그 저장
    log_path = config.log_dir / "teacher_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Teacher 학습 로그 저장 → {log_path}")

    # 메모리 해제 (이후 distill에서 checkpoint로 다시 로드)
    del teacher
    if config.device == "cuda":
        torch.cuda.empty_cache()

    return history
