"""/feedback — 세션 평가 결과를 JSONL로 저장 (storage 추상화 사용)"""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ui.backend.storage import get_storage

router = APIRouter()

_QUAL_PREFIX = "qualitative"


class FeedbackEntry(BaseModel):
    run_id: str
    prompt: str
    params: dict = Field(default_factory=dict)
    outputs: dict = Field(default_factory=dict)
    ratings: dict = Field(default_factory=dict)
    preference: str | None = None
    comment: str = ""
    mode: str = "compare"
    blind_mapping: dict | None = None


class SaveRequest(BaseModel):
    entries: list[FeedbackEntry]
    run_id: str
    filename_prefix: str = "session"


class SaveResponse(BaseModel):
    path: str
    count: int


@router.post("/feedback/save", response_model=SaveResponse)
def save_feedback(req: SaveRequest):
    if not req.entries:
        raise HTTPException(status_code=400, detail="저장할 항목이 없습니다.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rel_path = f"{_QUAL_PREFIX}/{req.run_id}/{req.filename_prefix}_{ts}.jsonl"

    lines = []
    for entry in req.entries:
        record = entry.model_dump()
        record["saved_at"] = datetime.now().isoformat()
        lines.append(json.dumps(record, ensure_ascii=False))
    content = "\n".join(lines) + "\n"

    get_storage().write_text(rel_path, content)
    return SaveResponse(path=rel_path, count=len(req.entries))


@router.get("/feedback/{run_id}")
def list_feedback(run_id: str) -> list[dict]:
    """해당 run 의 JSONL 파일 목록"""
    items = get_storage().list_files(f"{_QUAL_PREFIX}/{run_id}", suffix=".jsonl")
    return [
        {
            "filename": it["filename"],
            "path": f"{_QUAL_PREFIX}/{run_id}/{it['filename']}",
            "size_bytes": it["size_bytes"],
            "modified": it["modified"].isoformat() if it["modified"] else None,
        }
        for it in sorted(items, key=lambda x: x["filename"], reverse=True)
    ]


@router.get("/feedback/{run_id}/entries")
def read_feedback_entries(run_id: str, mode: str | None = None) -> list[dict]:
    """해당 run 의 모든 JSONL 파일을 합쳐서 entry 리스트로 반환.

    mode 필터: "compare" | "blind" | None(전체)
    """
    storage = get_storage()
    files = storage.list_files(f"{_QUAL_PREFIX}/{run_id}", suffix=".jsonl")

    entries: list[dict] = []
    for it in sorted(files, key=lambda x: x["filename"]):
        rel = f"{_QUAL_PREFIX}/{run_id}/{it['filename']}"
        text = storage.read_text(rel)
        if text is None:
            continue
        try:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_source_file"] = it["filename"]
                if mode and rec.get("mode") != mode:
                    continue
                entries.append(rec)
        except json.JSONDecodeError as e:
            entries.append({
                "_source_file": it["filename"],
                "_error": f"파싱 실패: {e}",
            })
    return entries
