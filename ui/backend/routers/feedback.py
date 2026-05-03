"""/feedback — 세션 평가 결과를 JSONL로 저장"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ui.backend.config import RESULTS_DIR

router = APIRouter()

_QUAL_DIR = RESULTS_DIR / "qualitative"


class FeedbackEntry(BaseModel):
    run_id: str
    prompt: str
    params: dict = Field(default_factory=dict)
    outputs: dict = Field(default_factory=dict)  # model_id → {text, tokens_per_sec, ...}
    ratings: dict = Field(default_factory=dict)  # model_id → {fluency, coherence, factuality, creativity}
    preference: str | None = None                # "teacher_ft" | "student_kd" | ... | "tie"
    comment: str = ""
    mode: str = "compare"                        # "compare" | "blind"
    blind_mapping: dict | None = None            # blind 때 A/B/C → 실제 model_id


class SaveRequest(BaseModel):
    entries: list[FeedbackEntry]
    run_id: str
    filename_prefix: str = "session"             # session / blind


class SaveResponse(BaseModel):
    path: str
    count: int


@router.post("/feedback/save", response_model=SaveResponse)
def save_feedback(req: SaveRequest):
    if not req.entries:
        raise HTTPException(status_code=400, detail="저장할 항목이 없습니다.")

    target_dir = _QUAL_DIR / req.run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = target_dir / f"{req.filename_prefix}_{ts}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for entry in req.entries:
            record = entry.model_dump()
            record["saved_at"] = datetime.now().isoformat()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return SaveResponse(path=str(path), count=len(req.entries))


@router.get("/feedback/{run_id}")
def list_feedback(run_id: str) -> list[dict]:
    """해당 run 의 JSONL 파일 목록"""
    d: Path = _QUAL_DIR / run_id
    if not d.exists():
        return []
    return [
        {
            "filename": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        }
        for p in sorted(d.glob("*.jsonl"), reverse=True)
    ]


@router.get("/feedback/{run_id}/entries")
def read_feedback_entries(run_id: str, mode: str | None = None) -> list[dict]:
    """해당 run 의 모든 JSONL 파일을 합쳐서 entry 리스트로 반환.

    mode 필터: "compare" | "blind" | None(전체)
    """
    d: Path = _QUAL_DIR / run_id
    if not d.exists():
        return []

    entries: list[dict] = []
    for p in sorted(d.glob("*.jsonl")):
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec["_source_file"] = p.name
                    if mode and rec.get("mode") != mode:
                        continue
                    entries.append(rec)
        except (json.JSONDecodeError, OSError) as e:
            # 손상 파일은 건너뛰되 로그만
            entries.append({
                "_source_file": p.name,
                "_error": f"파싱 실패: {e}",
            })
    return entries

