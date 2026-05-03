"""/runs/{run_id}/history, /runs/{run_id}/evaluation, /runs/{run_id}/figures

기존 엔드포인트는 runs.py 에 있고, 여기는 실험 리포트 전용으로 분리.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ui.backend.config import FIGURES_DIR, LOGS_DIR

router = APIRouter()


def _read_json(path: Path) -> list | dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/runs/{run_id}/history")
def get_history(run_id: str):
    """teacher_history / distill_history / baseline_history 를 한 번에 반환"""
    log_dir = LOGS_DIR / run_id
    if not log_dir.exists():
        raise HTTPException(status_code=404, detail=f"run 없음: {run_id}")

    return {
        "teacher": _read_json(log_dir / "teacher_history.json"),
        "distill": _read_json(log_dir / "distill_history.json"),
        "baseline": _read_json(log_dir / "baseline_history.json"),
    }


@router.get("/runs/{run_id}/evaluation")
def get_evaluation(run_id: str):
    """evaluation_results.json 반환 (4-Way PPL/속도/메모리)"""
    path = LOGS_DIR / run_id / "evaluation_results.json"
    data = _read_json(path)
    if data is None:
        raise HTTPException(status_code=404, detail=f"evaluation_results 없음: {run_id}")
    return data


@router.get("/runs/{run_id}/figures")
def list_figures(run_id: str) -> list[dict]:
    """results/figures/{run_id}/ 의 PNG 목록"""
    d = FIGURES_DIR / run_id
    if not d.exists():
        return []
    return [
        {"filename": p.name, "size_bytes": p.stat().st_size}
        for p in sorted(d.glob("*.png"))
    ]


@router.get("/runs/{run_id}/figures/{filename}")
def get_figure(run_id: str, filename: str):
    """PNG 파일 직접 반환"""
    # 경로 탈출 방지
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="잘못된 파일명")
    path = FIGURES_DIR / run_id / filename
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="figure 없음")
    return FileResponse(path, media_type="image/png")
