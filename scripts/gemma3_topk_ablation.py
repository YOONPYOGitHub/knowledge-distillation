"""top-K KD 가 full-vocab KD 와 얼마나 다른지 측정한다.

같은 Teacher/Student 상태와 같은 배치에서 KD 손실과 그 gradient 를 두 방식으로 계산해
비교한다. 학습을 돌리지 않고 근사 자체의 대가를 직접 재는 것이 목적이다.

  - kd 값 차이: 목표 분포를 자른 결과 손실 크기가 얼마나 달라지는가
  - gradient 코사인/상대 norm: Student 가 받는 학습 신호의 방향이 얼마나 보존되는가

gradient 방향이 보존되면 top-K 는 같은 방향으로 더 싸게 가는 것이고,
어긋나면 다른 목적함수를 최적화하는 것이다.

사용법:
    python scripts/gemma3_topk_ablation.py <config.yaml> [batches]
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import from_yaml
from src.dataset import create_dataloaders, load_tokenizer
from src.distill import kd_loss
from src.models import load_student, load_teacher


def grad_of(student, teacher_logits, batch, config, top_k):
    """KD 손실과 Student gradient 를 얻는다.

    1B 짜리 gradient 를 하나의 텐서로 이어붙여 double 로 올리면 GPU 에서 OOM 이 난다
    (1e9 x 8B x 2벌 = 16GB). 파라미터 단위로 CPU float32 사본을 들고 있다가
    비교 시점에 조각별로 double 누적한다.
    """
    config.kd_top_k = top_k
    student.zero_grad(set_to_none=True)
    outputs = student(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    total, ce, kd = kd_loss(outputs.logits, teacher_logits, batch["labels"], config)
    total.backward()
    grads = [
        p.grad.detach().float().cpu()
        for p in student.parameters() if p.grad is not None
    ]
    return grads, float(total), ce, kd


def compare_grads(full, topk):
    """코사인 유사도, norm 비, 상대 차이를 조각별 double 누적으로 계산한다."""
    dot = norm_full = norm_topk = norm_diff = 0.0
    for a, b in zip(full, topk):
        x, y = a.double(), b.double()
        dot += float((x * y).sum())
        norm_full += float(x.pow(2).sum())
        norm_topk += float(y.pow(2).sum())
        norm_diff += float((y - x).pow(2).sum())
    norm_full, norm_topk, norm_diff = (
        norm_full ** 0.5, norm_topk ** 0.5, norm_diff ** 0.5
    )
    denominator = max(norm_full * norm_topk, 1e-30)
    return {
        "grad_cosine": dot / denominator,
        "grad_norm_ratio": norm_topk / max(norm_full, 1e-30),
        "grad_rel_diff": norm_diff / max(norm_full, 1e-30),
    }


def main() -> int:
    config_path = sys.argv[1]
    max_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    # Teacher 는 KD 단계와 같은 조건(int8, 전용 GPU)으로 둔다.
    config = from_yaml(config_path, teacher_devices=[], teacher_device="cuda:2",
                       num_workers=0, batch_size=1)
    config.ensure_dirs()

    tokenizer = load_tokenizer(config.student_model)
    loaders = create_dataloaders(config, tokenizer, distributed=False)

    teacher = load_teacher(config)
    student = load_student(config)
    if config.distill_student_checkpoint and Path(config.distill_student_checkpoint).exists():
        state = torch.load(config.distill_student_checkpoint, map_location="cpu", weights_only=True)
        student.load_state_dict(state)
        print(f"KD 초기 Student 로드 → {config.distill_student_checkpoint}")
    student.to(config.device)

    teacher_device = next(teacher.parameters()).device
    rows = []

    for index, batch in enumerate(loaders["validation"]):
        if index >= max_batches:
            break
        moved = {k: v.to(config.device) for k, v in batch.items()}
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=batch["input_ids"].to(teacher_device),
                attention_mask=batch["attention_mask"].to(teacher_device),
            ).logits.to(config.device)

        full_grad, full_total, full_ce, full_kd = grad_of(
            student, teacher_logits, moved, config, 0
        )
        topk_grad, topk_total, topk_ce, topk_kd = grad_of(
            student, teacher_logits, moved, config, 128
        )

        stats = compare_grads(full_grad, topk_grad)
        del full_grad, topk_grad

        rows.append({
            "batch": index,
            "kd_full": full_kd, "kd_topk": topk_kd,
            "ce_full": full_ce, "ce_topk": topk_ce,
            "total_full": full_total, "total_topk": topk_total,
            **stats,
        })
        print(f"batch {index}: KD full {full_kd:.4f} / topK {topk_kd:.4f} "
              f"| grad cos {stats['grad_cosine']:.6f} "
              f"| |Δg|/|g| {stats['grad_rel_diff']:.4f}", flush=True)

    def mean(key):
        return sum(r[key] for r in rows) / len(rows)

    summary = {
        "config": config_path,
        "kd_top_k": 128,
        "batches": len(rows),
        "mean_kd_full": mean("kd_full"),
        "mean_kd_topk": mean("kd_topk"),
        "mean_ce_full": mean("ce_full"),
        "mean_grad_cosine": mean("grad_cosine"),
        "mean_grad_norm_ratio": mean("grad_norm_ratio"),
        "mean_grad_rel_diff": mean("grad_rel_diff"),
        "rows": rows,
    }
    out = config.log_dir / "topk_ablation.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\n=== top-K(128) vs full-vocab, {len(rows)} 배치 ===")
    print(f"  KD 손실     : full {summary['mean_kd_full']:.4f} → topK {summary['mean_kd_topk']:.4f}")
    print(f"  CE 손실     : {summary['mean_ce_full']:.4f} (top-K 와 무관해야 정상)")
    print(f"  gradient 코사인 : {summary['mean_grad_cosine']:.6f}")
    print(f"  |Δg|/|g|        : {summary['mean_grad_rel_diff']:.4f}")
    print(f"저장 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
