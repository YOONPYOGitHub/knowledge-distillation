"""FastAPI 엔트리 포인트

실행:
  uv run uvicorn ui.backend.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ui.backend.config import DEFAULT_RUN_ID
from ui.backend.model_registry import get_registry
from ui.backend.results_reader import latest_run_id
from ui.backend.routers import feedback as feedback_router
from ui.backend.routers import generate as generate_router
from ui.backend.routers import generate_batch as batch_router
from ui.backend.routers import generate_stream as stream_router
from ui.backend.routers import reports as reports_router
from ui.backend.routers import runs as runs_router
from ui.backend.routers import token_analysis as token_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 기본 run 사전 로드 (선택)
    target = DEFAULT_RUN_ID or latest_run_id()
    if target:
        log.info("기본 run 사전 로드 시도: %s", target)
        try:
            get_registry().ensure_loaded(
                target, ["teacher_ft", "student_kd", "student_ft"]
            )
            log.info("기본 run 로드 완료: %s", target)
        except Exception as e:  # noqa: BLE001
            log.warning("기본 run 로드 실패(%s): %s — 수동 로드 필요", target, e)
    else:
        log.info("사용 가능한 run 없음 — /runs 빈 배열 반환")
    yield


app = FastAPI(
    title="KD Comparison API",
    description="Teacher/Student(KD/FT/Base) 모델 비교 생성·평가 backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(runs_router.router, tags=["runs"])
app.include_router(reports_router.router, tags=["reports"])
app.include_router(feedback_router.router, tags=["feedback"])
app.include_router(generate_router.router, tags=["generate"])
app.include_router(stream_router.router, tags=["generate"])
app.include_router(batch_router.router, tags=["generate"])
app.include_router(token_router.router, tags=["token-analysis"])


@app.get("/health")
def health():
    reg = get_registry()
    return {
        "status": "ok",
        "device": reg.device,
        "cached_runs": list(reg._cache.keys()),  # noqa: SLF001
    }
