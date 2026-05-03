"""텍스트 생성 함수"""

from __future__ import annotations

import time
from typing import Optional

import torch

from ui.backend.model_registry import RunModels
from ui.backend.schemas import GenerationParams, ModelOutput


@torch.no_grad()
def generate_single(
    run_models: RunModels,
    model_id: str,
    prompt: str,
    params: GenerationParams,
    device: str,
) -> ModelOutput:
    """단일 모델 생성 (에러 시 error 필드 채움)"""
    model = run_models.get(model_id)
    if model is None:
        return ModelOutput(
            text="", num_tokens=0, elapsed_ms=0.0, tokens_per_sec=0.0,
            error=f"모델이 로드되지 않음: {model_id}",
        )
    tokenizer = run_models.tokenizer

    try:
        # 재현성을 위한 seed
        if params.seed is not None:
            torch.manual_seed(params.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(params.seed)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            repetition_penalty=params.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )
        if params.do_sample:
            gen_kwargs["temperature"] = max(params.temperature, 1e-5)
            gen_kwargs["top_p"] = params.top_p
            if params.top_k > 0:
                gen_kwargs["top_k"] = params.top_k

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        output_ids = model.generate(**inputs, **gen_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        # 생성된 부분만 디코드 (prompt 제외)
        new_tokens = output_ids[0][input_len:]
        num_tokens = int(new_tokens.numel())
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        tps = num_tokens / elapsed if elapsed > 0 else 0.0
        return ModelOutput(
            text=full_text,
            num_tokens=num_tokens,
            elapsed_ms=elapsed * 1000.0,
            tokens_per_sec=tps,
        )
    except Exception as e:  # noqa: BLE001
        return ModelOutput(
            text="", num_tokens=0, elapsed_ms=0.0, tokens_per_sec=0.0,
            error=f"{type(e).__name__}: {e}",
        )


def generate_compare(
    run_models: RunModels,
    model_ids: list[str],
    prompt: str,
    params: GenerationParams,
    device: str,
) -> dict[str, ModelOutput]:
    """여러 모델을 순차로 돌려 dict 반환"""
    return {
        mid: generate_single(run_models, mid, prompt, params, device)
        for mid in model_ids
    }
