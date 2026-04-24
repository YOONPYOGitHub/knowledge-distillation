"""/runs, /models/load — run 메타데이터 조회 & 모델 로드"""

from fastapi import APIRouter, HTTPException

from ui.backend.model_registry import get_registry
from ui.backend.results_reader import get_run_info, list_run_infos
from ui.backend.schemas import LoadRequest, LoadResponse

router = APIRouter()


@router.get("/runs")
def get_runs():
    return list_run_infos()


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    info = get_run_info(run_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"run 없음: {run_id}")
    return info


@router.post("/models/load", response_model=LoadResponse)
def load_models(req: LoadRequest):
    registry = get_registry()
    try:
        run_models = registry.ensure_loaded(req.run_id, req.models)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return LoadResponse(
        run_id=req.run_id,
        loaded=list(run_models.models.keys()),
        device=registry.device,
        memory_mb=registry.memory_mb(run_models),
    )
