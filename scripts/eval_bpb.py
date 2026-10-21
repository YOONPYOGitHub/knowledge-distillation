"""여러 모델의 PPL 과 bits-per-byte 를 같은 test split 에서 잰다.

PPL 은 토큰 단위라 tokenizer 가 다른 계열끼리 비교할 수 없다. bits-per-byte 는 같은 원문 UTF-8 바이트 수로
나누므로 계열 간 절대 비교에 쓴다. 각 모델은 자기 tokenizer 로 test split 을 만든다.
config 의 데이터 조건(문서 수, revision, split seed, seq 길이, BOS)을 그대로 쓰므로,
v4 설정을 주면 4-way 평가와 같은 test 문서를 본다.

  --model "이름=모델ID"                  pretrained 모델
  --model "이름=모델ID@어댑터경로"         LoRA 어댑터를 병합한 모델 (Teacher FT)

예:
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. .venv-gemma/bin/python scripts/eval_bpb.py \
    configs/qwen3_8b_1.7b_3090x4_v4.yaml --out results/qwen3/logs/qwen-family-bpb/bpb.json \
    --model "Qwen3-8B Base=Qwen/Qwen3-8B-Base" --model "Qwen3-1.7B Base=Qwen/Qwen3-1.7B-Base"
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from src.config import from_yaml
from src.dataset import load_lm_dataset, load_tokenizer


def parse_model(spec):
    name, _, target = spec.partition("=")
    model_id, _, adapter = target.partition("@")
    return name.strip(), model_id.strip(), adapter.strip() or None


def count_bytes(dataset, tokenizer):
    """예측 대상 토큰(각 청크의 둘째 토큰부터)이 나타내는 UTF-8 바이트 수."""
    special = set(tokenizer.all_special_ids)
    total = 0
    for row in dataset["input_ids"]:
        row = row.tolist() if torch.is_tensor(row) else list(row)
        ids = [i for i in row[1:] if i not in special]
        total += len(tokenizer.decode(ids).encode("utf-8"))
    return total


@torch.no_grad()
def total_nll(model, loader, device):
    nll, tokens = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids=input_ids, attention_mask=batch["attention_mask"].to(device)).logits
        labels = input_ids[:, 1:]
        nll += F.cross_entropy(logits[:, :-1].float().flatten(0, 1), labels.flatten(), reduction="sum").item()
        tokens += labels.numel()
    return nll, tokens


def load_model(model_id, adapter, device):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    config = from_yaml(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(out.read_text()) if out.is_file() else {}
    result.setdefault("config", args.config)
    result.setdefault("models", {})
    splits = {}

    for spec in args.model:
        name, model_id, adapter = parse_model(spec)
        if name in result["models"]:
            print(f"건너뜀 (이미 측정): {name}", flush=True)
            continue
        tokenizer = load_tokenizer(model_id)
        if model_id not in splits:
            test = load_lm_dataset(config, tokenizer)["test"]
            splits[model_id] = (test, count_bytes(test, tokenizer), tokenizer)
        test, n_bytes, _ = splits[model_id]
        loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

        start = time.time()
        model = load_model(model_id, adapter, args.device)
        nll, tokens = total_nll(model, loader, args.device)
        row = {
            "model_id": model_id,
            "adapter": adapter,
            "ppl": round(math.exp(nll / tokens), 4),
            "bits_per_byte": round(nll / math.log(2) / n_bytes, 4),
            "tokens": tokens,
            "bytes": n_bytes,
            "chunks": len(test),
            "params": sum(p.numel() for p in model.parameters()),
            "seconds": round(time.time() - start, 1),
        }
        result["models"][name] = row
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"{name}: {json.dumps(row, ensure_ascii=False)}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # tokenizer 가 같은 모델끼리는 test 청크가 토큰 단위로 같아야 PPL 을 직접 비교할 수 있다.
    ids = {m: [r.tolist() if torch.is_tensor(r) else list(r) for r in s[0]["input_ids"]] for m, s in splits.items()}
    first = next(iter(ids))
    result["same_test_tokens_as_" + first] = {m: v == ids[first] for m, v in ids.items()}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
