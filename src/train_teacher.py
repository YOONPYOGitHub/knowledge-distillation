"""Teacher Fine-tuning — KD 전에 Teacher를 target 도메인에 적응

Teacher가 target 데이터셋에서 Student FT보다 충분히 낮은 PPL을 달성해야
KD의 soft target이 의미 있는 dark knowledge를 전달할 수 있음.

파이프라인: train_teacher → distill → baseline → evaluate → compare
"""

import json
import time

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
from src.models import create_optimizer, model_info

from transformers import AutoModelForCausalLM


def train_one_epoch(model, dataloader, optimizer, config: KDConfig):
    """1 에폭 학습 (CE Loss)"""
    model.train()
    loss_sum = 0.0
    steps = 0

    progress = tqdm(
        dataloader,
        desc="Teacher Training",
        leave=False,
        disable=not is_main_process(),
    )
    for batch in progress:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Causal LM label shift: logits[t] → labels[t+1]
        shift_logits = outputs.logits[..., :-1, :].float().contiguous()
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
        if config.max_train_steps and steps >= config.max_train_steps:
            break
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return reduce_metrics({"ce_loss": loss_sum / steps}, config.device)


@torch.no_grad()
def validate(model, dataloader, config: KDConfig):
    """검증"""
    model.eval()
    loss_sum = 0.0
    steps = 0

    for batch in tqdm(
        dataloader,
        desc="Teacher Validation",
        leave=False,
        disable=not is_main_process(),
    ):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].float().contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        loss_sum += loss.item()
        steps += 1
        if config.max_eval_steps and steps >= config.max_eval_steps:
            break

    return reduce_metrics({"val_ce_loss": loss_sum / steps}, config.device)


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
    if config.bf16 and config.device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.bfloat16
    elif config.fp16 and config.device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16

    teacher = AutoModelForCausalLM.from_pretrained(config.teacher_model, **kwargs)
    if config.teacher_lora_rank:
        from peft import LoraConfig, TaskType, get_peft_model

        teacher = get_peft_model(
            teacher,
            LoraConfig(
                r=config.teacher_lora_rank,
                lora_alpha=config.teacher_lora_alpha,
                lora_dropout=config.teacher_lora_dropout,
                target_modules=config.teacher_lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    if config.gradient_checkpointing:
        teacher.gradient_checkpointing_enable()
        teacher.config.use_cache = False
    teacher.to(config.device)
    teacher.train()
    teacher = wrap_ddp(teacher, config.device)
    if is_main_process():
        model_info(teacher, "Teacher (FT)")
        print()

    # Optimizer
    optimizer = create_optimizer(teacher, config, config.teacher_learning_rate)

    # 학습 루프
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.teacher_epochs + 1):
        set_epoch(loaders["train"], epoch)
        start = time.time()
        if is_main_process():
            print(f"--- Teacher Epoch {epoch}/{config.teacher_epochs} ---")

        train_metrics = train_one_epoch(teacher, loaders["train"], optimizer, config)
        val_metrics = validate(teacher, loaders["validation"], config)

        elapsed = time.time() - start
        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
        if is_main_process():
            history.append(metrics)

            print(f"  Train CE Loss: {metrics['ce_loss']:.4f}")
            print(f"  Val CE Loss:   {metrics['val_ce_loss']:.4f}")
            print(f"  Time: {elapsed:.1f}s")

        improved = (
            metrics["val_ce_loss"]
            < best_val_loss - config.early_stopping_min_delta
        )
        if improved:
            best_val_loss = metrics["val_ce_loss"]
            epochs_without_improvement = 0
            if is_main_process():
                if config.teacher_lora_rank:
                    save_path = config.checkpoint_dir / "teacher_ft_best_adapter"
                    unwrap_model(teacher).save_pretrained(save_path)
                else:
                    save_path = config.checkpoint_dir / "teacher_ft_best.pt"
                    torch.save(unwrap_model(teacher).state_dict(), save_path)
                print(f"  ✅ Best teacher saved → {save_path}")
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
    log_path = config.log_dir / "teacher_history.json"
    if is_main_process():
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Teacher 학습 로그 저장 → {log_path}")
    barrier()

    # 메모리 해제 (이후 distill에서 checkpoint로 다시 로드)
    del teacher
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()

    return history
