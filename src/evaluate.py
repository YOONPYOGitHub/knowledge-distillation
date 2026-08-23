"""개별 모델 평가 — Perplexity, 추론 속도, 메모리 측정"""

import json
import math
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from src.config import KDConfig
from src.dataset import load_tokenizer, create_dataloaders
from src.models import load_student, load_teacher, model_info


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device):
    """Perplexity 계산: exp(avg CE loss)"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Perplexity", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Causal LM label shift: logits[t] → labels[t+1]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        # 유효 토큰 수 (shift 후 기준)
        num_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return {"perplexity": perplexity, "avg_loss": avg_loss}


@torch.no_grad()
def evaluate_speed(model, dataloader, device, num_batches=50):
    """추론 속도 측정: tokens/sec, ms/token"""
    model.eval()
    total_tokens = 0

    # warmup
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    model(input_ids=input_ids, attention_mask=attention_mask)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model(input_ids=input_ids, attention_mask=attention_mask)
        total_tokens += attention_mask.sum().item()

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.time() - start
    tokens_per_sec = total_tokens / elapsed
    ms_per_token = (elapsed / total_tokens) * 1000

    return {
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token,
        "elapsed": elapsed,
    }


def get_model_size(model):
    """모델 파라미터 수 및 메모리"""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return {"total_params": total_params, "size_mb": size_mb}


def evaluate_model(model, name, dataloader, device):
    """단일 모델 전체 평가"""
    print(f"\n--- {name} 평가 ---")

    size = get_model_size(model)
    print(f"  파라미터: {size['total_params']:,} ({size['size_mb']:.1f} MB)")

    ppl = evaluate_perplexity(model, dataloader, device)
    print(f"  Perplexity: {ppl['perplexity']:.2f}")

    speed = evaluate_speed(model, dataloader, device)
    print(f"  속도: {speed['tokens_per_sec']:.0f} tokens/sec ({speed['ms_per_token']:.2f} ms/token)")

    return {
        "name": name,
        **size,
        **ppl,
        **speed,
    }


def evaluate_all(config: KDConfig):
    """4-Way 전체 평가: Teacher / Student(KD) / Student(FT) / Student(Base)"""
    print("=== 4-Way Model Evaluation ===")
    print(f"Device: {config.device}\n")

    config.ensure_dirs()

    tokenizer = load_tokenizer(config.student_model)
    loaders = create_dataloaders(config, tokenizer)
    test_loader = loaders["test"]
    print(f"Test set: {len(test_loader.dataset)} samples\n")

    results = []

    # 1. Teacher (FT checkpoint가 있으면 Teacher (FT)로 표시)
    teacher = load_teacher(config)
    teacher_name = "Teacher (FT)" if config.teacher_checkpoint else "Teacher"
    results.append(evaluate_model(teacher, teacher_name, test_loader, config.device))
    del teacher
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # 2. Student (KD) — 증류 학습된 모델
    kd_path = config.checkpoint_dir / "student_kd_best.pt"
    if kd_path.exists():
        student_kd = AutoModelForCausalLM.from_pretrained(config.student_model)
        student_kd.load_state_dict(torch.load(kd_path, map_location="cpu", weights_only=True))
        student_kd.to(config.device)
        student_kd.eval()
        results.append(evaluate_model(student_kd, "Student (KD)", test_loader, config.device))
        del student_kd
    else:
        print(f"\n⚠️ Student (KD) 체크포인트 없음: {kd_path}")

    # 3. Student (FT) — Fine-tuning만 한 모델
    ft_path = config.checkpoint_dir / "student_ft_best.pt"
    if ft_path.exists():
        student_ft = AutoModelForCausalLM.from_pretrained(config.student_model)
        student_ft.load_state_dict(torch.load(ft_path, map_location="cpu", weights_only=True))
        student_ft.to(config.device)
        student_ft.eval()
        results.append(evaluate_model(student_ft, "Student (FT)", test_loader, config.device))
        del student_ft
    else:
        print(f"\n⚠️ Student (FT) 체크포인트 없음: {ft_path}")

    # 4. Student (Base) — 추가 학습 없는 원본
    student_base = AutoModelForCausalLM.from_pretrained(config.student_model)
    student_base.to(config.device)
    student_base.eval()
    results.append(evaluate_model(student_base, "Student (Base)", test_loader, config.device))
    del student_base

    # 결과 저장
    results_path = config.log_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n평가 결과 저장 → {results_path}")

    return results


if __name__ == "__main__":
    from src.config import local_config

    config = local_config()
    results = evaluate_all(config)
