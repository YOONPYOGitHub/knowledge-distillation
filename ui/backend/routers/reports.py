"""/runs/{run_id}/history, /runs/{run_id}/evaluation, /runs/{run_id}/figures

storage 추상화를 통해 로컬/Blob 모두 지원.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ui.backend.storage import get_storage

router = APIRouter()


@router.get("/runs/{run_id}/history")
def get_history(run_id: str):
    """teacher_history / distill_history / baseline_history 를 한 번에 반환"""
    storage = get_storage()
    if not storage.exists(f"logs/{run_id}/summary.json"):
        raise HTTPException(status_code=404, detail=f"run 없음: {run_id}")

    return {
        "teacher": storage.read_json(f"logs/{run_id}/teacher_history.json"),
        "distill": storage.read_json(f"logs/{run_id}/distill_history.json"),
        "baseline": storage.read_json(f"logs/{run_id}/baseline_history.json"),
    }


@router.get("/runs/{run_id}/evaluation")
def get_evaluation(run_id: str):
    """evaluation_results.json 반환 (4-Way PPL/속도/메모리)"""
    data = get_storage().read_json(f"logs/{run_id}/evaluation_results.json")
    if data is None:
        raise HTTPException(status_code=404, detail=f"evaluation_results 없음: {run_id}")
    return data


@router.get("/runs/{run_id}/figures")
def list_figures(run_id: str) -> list[dict]:
    """figures/{run_id}/ 의 PNG 목록"""
    items = get_storage().list_files(f"figures/{run_id}", suffix=".png")
    return [{"filename": it["filename"], "size_bytes": it["size_bytes"]} for it in items]


@router.get("/runs/{run_id}/figures/{filename}")
def get_figure(run_id: str, filename: str):
    """PNG 파일을 항상 백엔드를 통해 inline 반환.

    Storage 가 PE / 차단된 네트워크 뒤에 있을 경우 SAS redirect 는 외부 브라우저
    에서 도달 불가하므로, 백엔드가 VNet 내부에서 직접 읽어 전달한다.
    """
    if "/" in filename or ".." in filename or not filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="잘못된 파일명")

    storage = get_storage()
    rel = f"figures/{run_id}/{filename}"
    if not storage.exists(rel):
        raise HTTPException(status_code=404, detail="figure 없음")

    data = storage.read_bytes(rel)
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
