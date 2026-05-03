"""/generate — 여러 모델로 동시 생성"""

from fastapi import APIRouter, HTTPException

from ui.backend.generator import generate_compare
from ui.backend.model_registry import get_registry
from ui.backend.schemas import GenerateRequest, GenerateResponse

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    registry = get_registry()
    try:
        run_models = registry.ensure_loaded(req.run_id, req.model_ids)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    outputs = generate_compare(
        run_models=run_models,
        model_ids=req.model_ids,
        prompt=req.prompt,
        params=req.params,
        device=registry.device,
    )
    return GenerateResponse(run_id=req.run_id, prompt=req.prompt, outputs=outputs)
