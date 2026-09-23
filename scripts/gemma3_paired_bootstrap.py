"""Student 체크포인트끼리 test PPL 차이가 노이즈를 넘는지 paired bootstrap 으로 판정한다.

evaluate 단계는 test split 전체의 PPL 한 값만 낸다. 두 Student 의 PPL 차이가 작을 때
그 차이가 모델 차이인지 test chunk 구성에 따른 흔들림인지 구분할 수 없다.

같은 test chunk 에서 모델별 chunk 단위 NLL 합과 토큰 수를 구한 뒤, chunk 를 복원추출해
PPL 차이의 분포를 만든다. 두 모델이 같은 chunk 를 보므로(paired) chunk 난이도 차이는 상쇄된다.

사용법:
    python scripts/gemma3_paired_bootstrap.py [config.yaml] [resamples]

결과: results/gemma3/logs/<run_id>/paired_bootstrap.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import from_yaml
from src.dataset import create_dataloaders, load_tokenizer
from src.models import model_dtype_kwargs

CHECKPOINTS = Path("results/gemma3/checkpoints")

# (이름, 체크포인트). None 이면 pretrained Student.
MODELS = [
    ("Student (Base)", None),
    ("Student (FT)", CHECKPOINTS / "gemma3-12b-1b-v3" / "student_ft_best.pt"),
    ("Student (KD, top-K 128)", CHECKPOINTS / "gemma3-12b-1b-v3" / "student_kd_best.pt"),
    ("Student (KD, full-vocab)", CHECKPOINTS / "gemma3-12b-1b-v3-fullvocab" / "student_kd_best.pt"),
]

# (A, B): PPL(A) - PPL(B) 가 음수면 A 가 낫다.
PAIRS = [
    ("Student (KD, top-K 128)", "Student (FT)"),
    ("Student (KD, full-vocab)", "Student (FT)"),
    ("Student (KD, top-K 128)", "Student (KD, full-vocab)"),
    ("Student (FT)", "Student (Base)"),
]


@torch.no_grad()
def per_chunk_nll(model, dataloader, device):
    """chunk 별 NLL 합과 유효 토큰 수. src/evaluate.py 의 PPL 계산과 같은 shift/ignore 규칙."""
    model.eval()
    nll, tokens = [], []
    for batch in tqdm(dataloader, desc="NLL", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[..., :-1, :].float()
        shift_labels = labels[..., 1:]
        token_loss = F.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none"
        )
        nll.extend(token_loss.sum(dim=1).tolist())
        tokens.extend((shift_labels != -100).sum(dim=1).tolist())
    return np.array(nll, dtype=np.float64), np.array(tokens, dtype=np.int64)


def ppl(nll, tokens):
    return math.exp(nll.sum() / tokens.sum())


def paired_bootstrap(a, b, tokens, resamples, rng):
    """chunk 복원추출로 PPL(A) - PPL(B) 의 분포를 만든다."""
    n = len(tokens)
    index = rng.integers(0, n, size=(resamples, n))
    tok = tokens[index].sum(axis=1)
    diff = np.exp(a[index].sum(axis=1) / tok) - np.exp(b[index].sum(axis=1) / tok)
    low, high = np.percentile(diff, [2.5, 97.5])
    return {
        "observed": ppl(a, tokens) - ppl(b, tokens),
        "ci95": [float(low), float(high)],
        "p_a_better": float((diff < 0).mean()),
        "chunks_a_better": int((a / tokens < b / tokens).sum()),
        "chunks": int(n),
    }


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/gemma3_12b_1b_3090x4_v3.yaml"
    resamples = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    # 평가 단계와 같은 조건 (scripts/gemma3_stage.py 의 evaluate 설정).
    config = from_yaml(config_path, device="cuda:0", num_workers=0, global_batch_size=0)
    tokenizer = load_tokenizer(config.student_model)
    test_loader = create_dataloaders(config, tokenizer, distributed=False)["test"]
    print(f"Test set: {len(test_loader.dataset)} chunks")

    per_model = {}
    tokens = None
    for name, checkpoint in MODELS:
        if checkpoint is not None and not checkpoint.exists():
            print(f"⚠️  건너뜀 (체크포인트 없음): {name} → {checkpoint}")
            continue
        model = AutoModelForCausalLM.from_pretrained(config.student_model, **model_dtype_kwargs(config))
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(config.device)
        nll, chunk_tokens = per_chunk_nll(model, test_loader, config.device)
        del model
        torch.cuda.empty_cache()

        if tokens is None:
            tokens = chunk_tokens
        assert np.array_equal(tokens, chunk_tokens), "모델마다 test chunk 가 달라 paired 비교가 불가능하다"
        per_model[name] = nll
        print(f"  {name:<26} PPL {ppl(nll, tokens):.4f}")

    rng = np.random.default_rng(0)
    comparisons = []
    print(f"\npaired bootstrap ({resamples:,}회, chunk {len(tokens)}개 복원추출)")
    print(f"{'A':<26}{'B':<26}{'PPL(A)-PPL(B)':>14}{'95% CI':>22}{'P(A<B)':>9}{'chunk A<B':>11}")
    for a, b in PAIRS:
        if a not in per_model or b not in per_model:
            continue
        result = paired_bootstrap(per_model[a], per_model[b], tokens, resamples, rng)
        result.update({"a": a, "b": b})
        comparisons.append(result)
        low, high = result["ci95"]
        print(f"{a:<26}{b:<26}{result['observed']:>+14.4f}{f'[{low:+.4f}, {high:+.4f}]':>22}"
              f"{result['p_a_better']:>9.3f}{result['chunks_a_better']:>6}/{result['chunks']}")

    out = config.log_dir / "paired_bootstrap.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({
            "resamples": resamples,
            "chunks": int(len(tokens)),
            "tokens": int(tokens.sum()),
            "perplexity": {name: ppl(nll, tokens) for name, nll in per_model.items()},
            "comparisons": comparisons,
        }, handle, ensure_ascii=False, indent=2)
    print(f"\n저장 → {out}")
    print("CI 가 0 을 포함하면 두 모델의 PPL 차이는 test chunk 구성에 따른 흔들림과 구분되지 않는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
