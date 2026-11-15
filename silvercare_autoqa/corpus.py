"""온담 답의 근거 자료를 팀 지식 자료의 온전한 토막으로 바꿔 붙인다 (채점기가 잘린 발췌만 보지 않게).

왜: 온담 `/api/chat` 은 인용한 출처를 300자 안팎의 발췌로 돌려준다. 채점기는 그 발췌만 보고 "핵심 정보가 근거에 있었나"를
적으므로, 잘린 뒤에 있던 내용은 '근거에 없음'으로 읽힌다. 그러면 「지식 자료가 모자람」이 실제보다 많이 잡힌다.
평가 담당 PC 에는 팀 지식 자료(`knowledge.jsonl`)가 있으니, 인용한 출처를 그 자료의 토막과 맞춰 온전한 글로 바꾼다.

지키는 것
- 온담이 실제로 인용한 출처만 바꾼다. 인용하지 않은 자료를 더하지 않는다 (생성 모델이 못 본 글을 근거로 주지 않는다).
- 발췌의 앞부분이 그 토막에 글자 그대로 들어 있을 때만 바꾼다. 못 맞추면 발췌를 그대로 둔다.
- 질병관리청 실행 시 조회 자료(official_api)는 지식 자료에 없어 그대로 둔다.
- 결과 파일에는 지식 자료 원문(AI Hub 포함)이 들어간다. 레포에 올리지 않는다.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PROBE = 60          # 발췌 앞 60글자(공백 뺌)가 토막에 있어야 같은 글로 본다


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", "", s or "")


def load_knowledge(path: str | Path) -> dict[str, dict[str, list[dict]]]:
    by_url: dict[str, list[dict]] = defaultdict(list)
    by_title: dict[str, list[dict]] = defaultdict(list)
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        k = json.loads(ln)
        by_url[k.get("source_url") or ""].append(k)
        by_title[k.get("title") or ""].append(k)
    return {"url": by_url, "title": by_title}


def attach_corpus_excerpts(rows: list[dict], knowledge: dict[str, dict[str, list[dict]]]) -> dict[str, Any]:
    """응답 줄의 rag.sources[].excerpt 를 지식 자료의 온전한 토막으로 바꾼다 (rows 를 직접 고친다).
    돌려주는 값: 출처 수 · 바꾼 수 · 못 맞춘 수(종류별)."""
    stats = {"sources": 0, "replaced": 0, "kept": 0, "kept_by_kind": defaultdict(int), "rows_touched": 0}
    for r in rows:
        touched = False
        for s in (r.get("rag") or {}).get("sources") or []:
            stats["sources"] += 1
            exc = s.get("excerpt") or s.get("text") or ""
            probe = _norm(exc)[:PROBE]
            cands = knowledge["url"].get(s.get("url") or "", []) or knowledge["title"].get(s.get("title") or "", [])
            hit = next((k for k in cands if probe and probe in _norm((k.get("title") or "") + "\n" + (k.get("content") or ""))), None)
            if hit is None:
                stats["kept"] += 1
                stats["kept_by_kind"][s.get("evidence_kind") or "?"] += 1
                continue
            s["excerpt_api"] = exc                      # 온담이 돌려준 발췌는 남겨 둔다
            s["excerpt"] = hit["content"]
            s["excerpt_from"] = "team_corpus"
            stats["replaced"] += 1
            touched = True
        stats["rows_touched"] += touched
    stats["kept_by_kind"] = dict(stats["kept_by_kind"])
    return stats


def attach_file(responses: str | Path, knowledge_path: str | Path, out: str | Path) -> dict[str, Any]:
    rows = [json.loads(ln) for ln in Path(responses).read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    stats = attach_corpus_excerpts(rows, load_knowledge(knowledge_path))
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return stats
