"""Kanana 1.5 8B -> 2.1B 스모크 테스트 (RTX 3090 x 4)

4-way KD 를 돌리기 전에 Gemma 때 문제가 됐던 항목을 먼저 확인한다.

  1. tokenizer: Teacher / Student vocabulary 가 같은가, 문서마다 BOS 가 붙는가
  2. BOS: 청크 BOS 유무에 따라 Teacher / Student PPL 이 어떻게 달라지는가 (Gemma 는 순서가 뒤집혔다)
  3. 양자화: int8 Teacher 가 bf16 Teacher 와 얼마나 다른가 (KL(T=2), top-1 일치, PPL)
  4. KD 한 step: bf16 Teacher + 2.1B Student 가 3090 두 장에서 top-K KD forward/backward 를 통과하는가

tokenizer 가 계열마다 달라 PPL 을 Qwen / Gemma 와 직접 비교할 수 없으므로
tokenizer 와 무관한 bits-per-byte 도 함께 기록한다.

사용법:
  PYTHONPATH=. .venv-gemma/bin/python scripts/kanana_smoke.py configs/kanana_8b_2b_3090x4_eval.yaml

GPU 배치: Teacher bf16 cuda:0, Student cuda:1, Teacher int8 cuda:2.
"""

import argparse
import dataclasses
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

from src.config import from_yaml
from src.dataset import load_lm_dataset, load_tokenizer
from src.distill import kd_loss
from src.models import create_optimizer, teacher_quantization_config

TEACHER_BF16 = "cuda:0"
STUDENT = "cuda:1"
TEACHER_INT8 = "cuda:2"


def gb(n_bytes):
    return round(n_bytes / 1024**3, 2)


def check_tokenizer(config):
    t_tok = load_tokenizer(config.teacher_model)
    s_tok = load_tokenizer(config.student_model)
    t_cfg = AutoConfig.from_pretrained(config.teacher_model)
    s_cfg = AutoConfig.from_pretrained(config.student_model)
    ids = s_tok("대한민국의 수도는 서울특별시이다.")["input_ids"]
    result = {
        "same_vocab": t_tok.get_vocab() == s_tok.get_vocab(),
        "tokenizer_len": len(s_tok),
        "teacher_vocab_size": t_cfg.vocab_size,
        "student_vocab_size": s_cfg.vocab_size,
        "bos_token": s_tok.bos_token,
        "bos_token_id": s_tok.bos_token_id,
        "adds_bos_per_document": ids[0] == s_tok.bos_token_id,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["same_vocab"]:
        raise SystemExit("Teacher / Student vocabulary 가 달라 logit KD 를 할 수 없습니다")
    return s_tok, result


def test_loader(config, tokenizer, prepend_bos):
    cfg = dataclasses.replace(config, prepend_bos=prepend_bos)
    test = load_lm_dataset(cfg, tokenizer)["test"]
    return DataLoader(test, batch_size=config.batch_size, shuffle=False)


def count_bytes(loader, tokenizer):
    """예측 대상 토큰(각 청크의 둘째 토큰부터)이 나타내는 UTF-8 바이트 수."""
    special = set(tokenizer.all_special_ids)
    total = 0
    for batch in loader:
        for row in batch["input_ids"]:
            ids = [i for i in row[1:].tolist() if i not in special]
            total += len(tokenizer.decode(ids).encode("utf-8"))
    return total


@torch.no_grad()
def nll_and_logits(model, loader, device, keep_logits=False):
    """전체 test 의 NLL 합과 토큰 수. keep_logits 면 T=2 log-prob 을 CPU 에 모은다 (KL 비교용)."""
    total_nll, total_tokens, kept = 0.0, 0, []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids=input_ids, attention_mask=batch["attention_mask"].to(device)).logits
        shift = logits[:, :-1].float()
        labels = input_ids[:, 1:]
        total_nll += F.cross_entropy(shift.flatten(0, 1), labels.flatten(), reduction="sum").item()
        total_tokens += labels.numel()
        if keep_logits:
            kept.append(shift.cpu())
    return total_nll, total_tokens, kept


def ppl_row(nll, tokens, n_bytes):
    return {
        "ppl": round(math.exp(nll / tokens), 4),
        "bits_per_byte": round(nll / math.log(2) / n_bytes, 4),
        "tokens": tokens,
        "bytes": n_bytes,
    }


def load(model_name, device, config, quantization=None):
    kwargs = {"torch_dtype": torch.bfloat16}
    if quantization:
        kwargs["quantization_config"] = teacher_quantization_config(config, quantization)
        kwargs["device_map"] = {"": device}
    # 해당 장치의 CUDA context 가 만들어지기 전에는 메모리 통계를 초기화할 수 없다.
    torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not quantization:
        model.to(device)
    model.eval()
    return model, round(time.time() - start, 1), gb(torch.cuda.memory_allocated(device))


def free(*models):
    for m in models:
        del m
    gc.collect()
    torch.cuda.empty_cache()


def bos_and_quantization(config, tokenizer):
    loaders = {bos: test_loader(config, tokenizer, bos) for bos in (True, False)}
    n_bytes = {bos: count_bytes(loader, tokenizer) for bos, loader in loaders.items()}
    result = {"bos": {}, "quantization": {}}

    student, load_s, weights = load(config.student_model, STUDENT, config)
    result["student_weights_gb"] = weights
    for bos, loader in loaders.items():
        nll, tokens, _ = nll_and_logits(student, loader, STUDENT)
        result["bos"].setdefault(str(bos), {})["student"] = ppl_row(nll, tokens, n_bytes[bos])
    free(student)

    teacher, load_s, weights = load(config.teacher_model, TEACHER_BF16, config)
    result["quantization"]["bf16"] = {"device": TEACHER_BF16, "weights_gb": weights, "load_s": load_s}
    ref_logits = None
    for bos, loader in loaders.items():
        nll, tokens, kept = nll_and_logits(teacher, loader, TEACHER_BF16, keep_logits=bos)
        result["bos"][str(bos)]["teacher"] = ppl_row(nll, tokens, n_bytes[bos])
        if bos:
            ref_logits = kept
    result["quantization"]["bf16"]["peak_gb"] = gb(torch.cuda.max_memory_allocated(TEACHER_BF16))
    free(teacher)
    print(json.dumps(result["bos"], indent=2))

    # int8 Teacher 는 BOS 청크에서 bf16 Teacher 와 비교한다 (KD 에서 실제로 쓸 조건).
    teacher, load_s, weights = load(config.teacher_model, TEACHER_INT8, config, "int8")
    nll, tokens, kept = nll_and_logits(teacher, loaders[True], TEACHER_INT8, keep_logits=True)
    kl, agree, positions = 0.0, 0, 0
    for ref, q in zip(ref_logits, kept):
        p = F.log_softmax(ref / 2.0, dim=-1)
        r = F.log_softmax(q / 2.0, dim=-1)
        kl += F.kl_div(r, p, log_target=True, reduction="sum").item()
        agree += (ref.argmax(-1) == q.argmax(-1)).sum().item()
        positions += ref.shape[0] * ref.shape[1]
    result["quantization"]["int8"] = {
        "device": TEACHER_INT8,
        "weights_gb": weights,
        "peak_gb": gb(torch.cuda.max_memory_allocated(TEACHER_INT8)),
        "load_s": load_s,
        **ppl_row(nll, tokens, n_bytes[True]),
        "kl_T2_nats": round(kl / positions, 6),
        "top1_agree": round(agree / positions, 4),
    }
    free(teacher)
    del ref_logits, kept
    print(json.dumps(result["quantization"], indent=2))
    return result


def kd_step(config, tokenizer, top_k=128):
    """bf16 Teacher(cuda:0) + Student(cuda:1) 로 top-K KD 한 step 을 돌려 피크 메모리를 잰다."""
    cfg = dataclasses.replace(
        config, kd_top_k=top_k, kd_vocab_size=len(tokenizer), temperature=2.0, alpha=0.5,
        kd_reduction="tokenmean", kd_divergence="forward_kl", optimizer="adafactor",
    )
    loader = test_loader(cfg, tokenizer, prepend_bos=True)
    batch = next(iter(DataLoader(loader.dataset, batch_size=1)))

    teacher, _, _ = load(cfg.teacher_model, TEACHER_BF16, cfg)
    student, _, _ = load(cfg.student_model, STUDENT, cfg)
    student.gradient_checkpointing_enable()
    student.config.use_cache = False
    student.train()
    optimizer = create_optimizer(student, cfg, 1e-5)
    torch.cuda.reset_peak_memory_stats(STUDENT)

    with torch.no_grad():
        t_logits = teacher(input_ids=batch["input_ids"].to(TEACHER_BF16)).logits.to(STUDENT)
    s_logits = student(input_ids=batch["input_ids"].to(STUDENT)).logits
    loss, ce, kd = kd_loss(s_logits, t_logits, batch["labels"].to(STUDENT), cfg)
    loss.backward()
    optimizer.step()
    result = {
        "top_k": top_k,
        "loss": round(loss.item(), 4),
        "ce": round(float(ce), 4),
        "kd": round(float(kd), 4),
        "student_peak_gb": gb(torch.cuda.max_memory_allocated(STUDENT)),
        "teacher_peak_gb": gb(torch.cuda.max_memory_allocated(TEACHER_BF16)),
        "finite": math.isfinite(loss.item()),
    }
    free(teacher, student)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config = from_yaml(args.config)
    out = Path(args.out or Path(config.output_dir) / "logs" / config.run_id / "smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, tok = check_tokenizer(config)
    result = {"tokenizer": tok, **bos_and_quantization(config, tokenizer), "kd_step": kd_step(config, tokenizer)}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
