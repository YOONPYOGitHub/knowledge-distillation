"""/token-analysis — prompt 각 position 의 top-k 분포와 Teacher vs Student KL"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ui.backend.model_registry import get_registry
from ui.backend.schemas import ModelId
from ui.backend.token_analyzer import analyze_tokens

router = APIRouter()


class TokenAnalysisRequest(BaseModel):
    run_id: str
    model_ids: list[ModelId] = Field(
        default_factory=lambda: ["teacher_ft", "student_kd", "student_ft"]
    )
    prompt: str
    top_k: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=1.0, ge=0.1, le=5.0)
    teacher_id: ModelId | None = "teacher_ft"


@router.post("/token-analysis")
def token_analysis(req: TokenAnalysisRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt 가 비어 있습니다.")

    reg = get_registry()
    try:
        run_models = reg.ensure_loaded(req.run_id, req.model_ids)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"모델 로드 실패: {e}") from e

    return analyze_tokens(
        run_models=run_models,
        model_ids=req.model_ids,
        prompt=req.prompt,
        device=reg.device,
        top_k=req.top_k,
        temperature=req.temperature,
        teacher_id=req.teacher_id,
    )
