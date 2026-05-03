"""Token Analysis — 각 position 의 next-token 분포 & KL divergence 계산

주어진 prompt 를 모든 모델에 teacher-forcing 방식으로 통과시켜:
- 각 position 의 top-k next-token 분포
- Teacher vs Student 의 per-position KL divergence (전체 어휘 기준)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ui.backend.model_registry import RunModels


@torch.no_grad()
def analyze_tokens(
    run_models: RunModels,
    model_ids: list[str],
    prompt: str,
    device: str,
    top_k: int = 10,
    temperature: float = 1.0,
    teacher_id: Optional[str] = "teacher_ft",
) -> dict:
    """
    반환 스키마:
      {
        "tokens": [str, ...],            # prompt 토큰 (decoded, 길이 N)
        "token_ids": [int, ...],
        "per_model": {
          model_id: {
            "positions": [
              { "top_tokens": [str], "top_probs": [float],
                "top_ids": [int], "actual_next": str | None }
            ]  # 길이 N-1 (마지막 position 은 next token 없음)
          }
        },
        "kl_divergence": {
          "<teacher>_vs_<student>": [float, ...]  # 길이 N-1
        },
        "temperature": float,
        "top_k": int,
      }
    """
    tokenizer = run_models.tokenizer
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]  # [1, N]
    n = input_ids.shape[1]

    if n < 2:
        return {
            "tokens": [tokenizer.decode([tid]) for tid in input_ids[0].tolist()],
            "token_ids": input_ids[0].tolist(),
            "per_model": {},
            "kl_divergence": {},
            "temperature": temperature,
            "top_k": top_k,
            "warning": "prompt 가 너무 짧습니다 (토큰 2개 이상 필요).",
        }

    tokens_decoded = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    actual_next_ids = input_ids[0, 1:].tolist()  # position i 의 정답 = input_ids[i+1]

    # 모델별 softmax 확률 저장 (KL 계산용)
    probs_per_model: dict[str, torch.Tensor] = {}  # [N-1, V]
    per_model_out: dict[str, dict] = {}

    t_eff = max(temperature, 1e-5)

    for mid in model_ids:
        model = run_models.get(mid)
        if model is None:
            per_model_out[mid] = {"error": f"모델 로드 안 됨: {mid}"}
            continue

        try:
            logits = model(input_ids=input_ids).logits  # [1, N, V]
            # 마지막 position 은 다음 토큰 예측 대상 없음 → [:, :-1, :]
            scaled = logits[0, :-1, :] / t_eff  # [N-1, V]
            probs = F.softmax(scaled, dim=-1)  # [N-1, V]
            probs_per_model[mid] = probs

            top_vals, top_idx = probs.topk(k=min(top_k, probs.shape[-1]), dim=-1)
            positions = []
            for pos in range(probs.shape[0]):
                ids = top_idx[pos].tolist()
                vals = top_vals[pos].tolist()
                positions.append({
                    "top_tokens": [tokenizer.decode([i]) for i in ids],
                    "top_ids": ids,
                    "top_probs": vals,
                    "actual_next": tokens_decoded[pos + 1],
                    "actual_next_id": actual_next_ids[pos],
                })
            per_model_out[mid] = {"positions": positions}
        except Exception as e:  # noqa: BLE001
            per_model_out[mid] = {"error": f"{type(e).__name__}: {e}"}

    # KL divergence: teacher || student
    kl_out: dict[str, list[float]] = {}
    if teacher_id and teacher_id in probs_per_model:
        t_probs = probs_per_model[teacher_id]
        t_log = torch.log(t_probs.clamp_min(1e-12))
        for sid, s_probs in probs_per_model.items():
            if sid == teacher_id:
                continue
            s_log = torch.log(s_probs.clamp_min(1e-12))
            # KL(t || s) = sum t * (log t - log s)
            kl = (t_probs * (t_log - s_log)).sum(dim=-1)  # [N-1]
            kl_out[f"{teacher_id}_vs_{sid}"] = kl.tolist()

    return {
        "tokens": tokens_decoded,
        "token_ids": input_ids[0].tolist(),
        "per_model": per_model_out,
        "kl_divergence": kl_out,
        "temperature": temperature,
        "top_k": top_k,
    }
