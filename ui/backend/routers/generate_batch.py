"""Prompt 배치 모드 — 여러 prompt 를 한 번에 여러 모델로 생성.

요청 본문:
  { "run_id": ..., "model_ids": [...], "prompts": ["p1","p2",...], "params": {...} }

응답:
  { "items": [ { "prompt": ..., "outputs": {mid: ModelOutput} }, ... ] }
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ui.backend.generator import generate_compare
from ui.backend.model_registry import get_registry
from ui.backend.schemas import GenerationParams, ModelId, ModelOutput

router = APIRouter()


class BatchRequest(BaseModel):
    run_id: str
    model_ids: list[ModelId]
    prompts: list[str] = Field(min_length=1)
    params: GenerationParams = Field(default_factory=GenerationParams)


class BatchItem(BaseModel):
    prompt: str
    outputs: dict[str, ModelOutput]


class BatchResponse(BaseModel):
    run_id: str
    count: int
    items: list[BatchItem]


MAX_PROMPTS = 200


@router.post("/generate/batch", response_model=BatchResponse)
def generate_batch(req: BatchRequest):
    if len(req.prompts) > MAX_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"prompt 수가 너무 많습니다 ({len(req.prompts)} > {MAX_PROMPTS})",
        )
    # 빈 문자열 제거
    prompts = [p for p in req.prompts if p and p.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="유효한 prompt 가 없습니다.")

    registry = get_registry()
    try:
        run_models = registry.ensure_loaded(req.run_id, req.model_ids)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    items: list[BatchItem] = []
    for prompt in prompts:
        outputs = generate_compare(
            run_models=run_models,
            model_ids=req.model_ids,
            prompt=prompt,
            params=req.params,
            device=registry.device,
        )
        items.append(BatchItem(prompt=prompt, outputs=outputs))

    return BatchResponse(run_id=req.run_id, count=len(items), items=items)
