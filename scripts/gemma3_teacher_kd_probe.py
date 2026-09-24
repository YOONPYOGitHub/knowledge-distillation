"""KD 단계에서 실제로 쓴 Teacher 가 평가한 Teacher 와 얼마나 다른지 측정한다.

Gemma v3 의 Teacher 는 단계마다 base 정밀도가 다르다.

  - Teacher FT 학습: nf4 base 위 LoRA (QLoRA)
  - KD 목표 분포:   int8 base + 같은 LoRA 어댑터 (병합 없음)
  - 보고한 PPL:     bf16 base + 같은 LoRA 어댑터 (병합)

보고한 Teacher (FT) PPL 은 bf16 쪽이지만 Student 가 배운 것은 int8 쪽 분포다.
같은 test chunk 에서 두 Teacher 의 PPL 과 KD 목표 분포의 차이를 잰다.

  - PPL: 두 Teacher 각각
  - KL(bf16 || int8), T=2: KD 온도에서 목표 분포가 얼마나 달라지는가
  - top-1 일치율
  - top-K 겹침: top-K KD 가 목표로 삼는 상위 K개 토큰 집합이 얼마나 같은가

GPU: bf16 Teacher 를 cuda:0/1 에 분할, int8 Teacher 를 cuda:2 에 둔다.

사용법:
    python scripts/gemma3_teacher_kd_probe.py [config.yaml]

결과: results/gemma3/logs/<run_id>/teacher_kd_probe.json
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import from_yaml
from src.dataset import create_dataloaders, load_tokenizer
from src.models import load_teacher

INT8_DEVICE = "cuda:2"


def load_bf16_teacher(config_path):
    """평가 단계와 같은 bf16 Teacher (어댑터 병합, 2장 분할)."""
    config = from_yaml(
        config_path,
        teacher_quantization="",
        teacher_devices=[],
        teacher_device_map="auto",
        teacher_max_memory={"0": "22GiB", "1": "22GiB"},
        device="cuda:0",
        num_workers=0,
        global_batch_size=0,
    )
    return config, load_teacher(config)


def load_int8_teacher(config_path):
    """KD 단계와 같은 int8 Teacher (어댑터 병합 없음, 한 장)."""
    config = from_yaml(
        config_path,
        teacher_devices=[],
        teacher_device=INT8_DEVICE,
        device=INT8_DEVICE,
        num_workers=0,
        global_batch_size=0,
    )
    assert config.teacher_quantization == "int8", config.teacher_quantization
    return load_teacher(config)


@torch.no_grad()
def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/gemma3_12b_1b_3090x4_v3.yaml"

    config, bf16 = load_bf16_teacher(config_path)
    int8 = load_int8_teacher(config_path)
    bf16_device = next(bf16.parameters()).device
    top_k = config.kd_top_k or 128
    temperature = config.temperature

    tokenizer = load_tokenizer(config.student_model)
    test_loader = create_dataloaders(config, tokenizer, distributed=False)["test"]
    print(f"Test set: {len(test_loader.dataset)} chunks | T={temperature} | top-K={top_k}")

    nll_bf16 = nll_int8 = kl_sum = top1_same = overlap_sum = 0.0
    tokens = 0
    start = time.time()
    for batch in tqdm(test_loader, desc="Teacher probe"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].to(INT8_DEVICE)

        logits_a = bf16(
            input_ids=input_ids.to(bf16_device), attention_mask=attention_mask.to(bf16_device)
        ).logits.to(INT8_DEVICE)
        logits_b = int8(
            input_ids=input_ids.to(INT8_DEVICE), attention_mask=attention_mask.to(INT8_DEVICE)
        ).logits

        # Causal LM label shift. 유효 토큰 위치만 본다 (src/evaluate.py 와 같은 규칙).
        shift_labels = labels[..., 1:]
        valid = shift_labels != -100
        a = logits_a[..., :-1, :][valid].float()
        b = logits_b[..., :-1, :][valid].float()
        target = shift_labels[valid]

        nll_bf16 += F.cross_entropy(a, target, reduction="sum").item()
        nll_int8 += F.cross_entropy(b, target, reduction="sum").item()

        log_p = F.log_softmax(a / temperature, dim=-1)
        log_q = F.log_softmax(b / temperature, dim=-1)
        kl_sum += (log_p.exp() * (log_p - log_q)).sum().item()

        top1_same += (a.argmax(-1) == b.argmax(-1)).sum().item()

        top_a = a.topk(top_k, dim=-1).indices
        top_b = b.topk(top_k, dim=-1).indices
        in_b = torch.zeros_like(b, dtype=torch.bool).scatter_(-1, top_b, True)
        overlap_sum += in_b.gather(-1, top_a).float().mean(-1).sum().item()

        tokens += int(valid.sum().item())
        del logits_a, logits_b, a, b, log_p, log_q, in_b

    result = {
        "config": config_path,
        "chunks": len(test_loader.dataset),
        "tokens": tokens,
        "temperature": temperature,
        "top_k": top_k,
        "ppl_bf16_ft": math.exp(nll_bf16 / tokens),
        "ppl_int8_ft": math.exp(nll_int8 / tokens),
        "kl_T_bf16_to_int8": kl_sum / tokens,
        "top1_agree": top1_same / tokens,
        "topk_overlap": overlap_sum / tokens,
        "elapsed_s": round(time.time() - start, 1),
    }

    print(f"\nTeacher (FT) bf16 PPL : {result['ppl_bf16_ft']:.4f}  (평가 단계에서 보고한 Teacher)")
    print(f"Teacher (FT) int8 PPL : {result['ppl_int8_ft']:.4f}  (KD 목표 분포를 만든 Teacher)")
    print(f"KL(bf16 || int8), T={temperature} : {result['kl_T_bf16_to_int8']:.4f} nats/token")
    print(f"top-1 일치율          : {result['top1_agree']:.4f}")
    print(f"top-{top_k} 겹침         : {result['topk_overlap']:.4f}")

    out = config.log_dir / "teacher_kd_probe.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"\n저장 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
