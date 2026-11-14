"""RAG 응답 읽기 — 출처 정리·자료 묶음 판별·인용 번호 읽기·검색 지표 재료.

온담(로컬 LLM + RAG)의 /api/chat 응답은 답(answer)과 함께 출처(sources)를 돌려준다.
응답 JSONL 한 줄에 `rag` 칸으로 담는다 (없으면 RAG 없는 모델로 본다 — ChatGPT·검색 끈 로컬 모델 등).

  "rag": {"mode": "grounded",
          "sources": [{"id": 1, "title": "...", "url": "...", "excerpt": "...", "evidence_kind": "official_api",
                       "source_type": "...", "record_id": "...", "source_display": "...", "source_year": "2026",
                       "source_language": "ko"}],
          "contexts_complete": false}

- sources 는 온담이 **인용한** 근거만 담는다 (팀 rag.md §6-7). 검색했지만 인용하지 않은 자료는 보이지 않는다.
- excerpt 는 300자 안팎으로 잘린 발췌다. 생성 모델은 부모 문맥 최대 3,600자를 봤다.
  그래서 contexts_complete=false 이면 "발췌에 없다"를 틀렸다는 증거로 쓰지 않는다 (prompts 의 근거 원칙).
- 자료 묶음표는 data/rag_sources.yaml 하나가 정본이다.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .golden import DATA_DIR

SOURCES_YAML = DATA_DIR / "rag_sources.yaml"
EXCERPT_MAX = 900          # 채점 프롬프트에 싣는 발췌 길이 상한 (자료 하나)
MAX_SOURCES = 8            # 채점 프롬프트에 싣는 출처 수 상한
CITE_RE = re.compile(r"\[(\d{1,2})\]")


@lru_cache(maxsize=None)
def load_source_table(path: str = str(SOURCES_YAML)) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fams = doc.get("families") or {}
    order = []
    for name, f in fams.items():
        canon = f.get("same_as") or name
        order.append((name, canon, f.get("role", "evidence"), f.get("label", canon), f.get("match") or {}))
    labels = {}
    roles = {}
    for name, canon, role, label, _ in order:
        labels.setdefault(canon, label)
        roles.setdefault(canon, role)
    groups = {k: list(v) for k, v in (doc.get("groups") or {}).items()}
    return {"order": order, "labels": labels, "roles": roles, "groups": groups,
            "modes": dict(doc.get("modes") or {}), "corpus": doc.get("corpus", "")}


def family_names() -> list[str]:
    t = load_source_table()
    return sorted(set(t["labels"]) | {"OTHER"})


def expand_expect(names: list[str] | None) -> set[str]:
    """골든셋 rag.expect (묶음·그룹 이름) → 묶음 이름 집합."""
    t = load_source_table()
    out: set[str] = set()
    for n in names or []:
        out.update(t["groups"].get(n, [n]))
    return out


def unknown_expect(names: list[str] | None) -> list[str]:
    t = load_source_table()
    known = set(t["labels"]) | set(t["groups"])
    return [n for n in names or [] if n not in known]


def _host(url: str) -> str:
    try:
        return (urlparse(url or "").netloc or "").lower()
    except ValueError:
        return ""


def _matches(src: dict, rule: dict) -> bool:
    url = str(src.get("url") or "")
    host = _host(url)
    rid = str(src.get("record_id") or "")
    checks = {
        "host": lambda vals: host in [v.lower() for v in vals],
        "url_has": lambda vals: any(v in url for v in vals),
        "evidence_kind": lambda vals: str(src.get("evidence_kind") or "") in vals,
        "source_type": lambda vals: str(src.get("source_type") or "") in vals,
        "record_prefix": lambda vals: any(rid.startswith(v) for v in vals),
    }
    for key, vals in rule.items():
        fn = checks.get(key)
        if fn is None or not fn(list(vals or [])):
            return False
    return bool(rule)


def family_of(src: dict) -> str:
    """출처 하나의 자료 묶음 이름. 온담이 family 칸을 주면 그 값을 쓴다."""
    t = load_source_table()
    given = src.get("family")
    if given and given in t["labels"]:
        return given
    for _, canon, _, _, rule in t["order"]:
        if _matches(src, rule):
            return canon
    return "OTHER"


def role_of(family: str) -> str:
    return load_source_table()["roles"].get(family, "unknown")


def normalize_source(s: dict, index: int) -> dict:
    """온담 sources 한 개 → 채점·지표에 쓰는 모양. 모르는 칸은 버리지 않고 raw 에 남긴다."""
    sid = s.get("id", index + 1)
    try:
        sid = int(sid)
    except (TypeError, ValueError):
        sid = index + 1
    out = {
        "id": sid,
        "title": str(s.get("title") or ""),
        "url": str(s.get("url") or ""),
        "excerpt": str(s.get("excerpt") or s.get("text") or ""),
        "evidence_kind": str(s.get("evidence_kind") or ""),
        "source_type": str(s.get("source_type") or ""),
        "record_id": str(s.get("record_id") or ""),
        "org": str(s.get("source_display") or s.get("org") or ""),
        "year": str(s.get("source_year") or ""),
        "lang": str(s.get("source_language") or ""),
    }
    out["family"] = family_of({**s, **out})
    out["role"] = role_of(out["family"])
    return out


def normalize_rag(rag: dict | None) -> dict | None:
    """응답 JSONL 의 rag 칸을 정리한다. 없으면 None (RAG 없는 모델)."""
    if not rag:
        return None
    srcs = [normalize_source(s, i) for i, s in enumerate(rag.get("sources") or []) if isinstance(s, dict)]
    ctx = rag.get("contexts")
    contexts = None
    if isinstance(ctx, list):
        contexts = [normalize_source(s, i) for i, s in enumerate(ctx) if isinstance(s, dict)]
    return {"mode": str(rag.get("mode") or ""), "sources": srcs, "contexts": contexts,
            "contexts_complete": bool(rag.get("contexts_complete")) and contexts is not None,
            "health_topics": [str(t.get("title") if isinstance(t, dict) else t) for t in rag.get("health_topics") or []]}


def cited_ids(answer: str) -> list[int]:
    """답 본문의 인용 번호 [1] [2] … (나온 순서, 중복 제거)."""
    seen: list[int] = []
    for m in CITE_RE.finditer(answer or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def strip_citations(text: str) -> str:
    """짝 비교(블라인드)용 — 인용 번호를 지운다."""
    return re.sub(r"\s?\[\d{1,2}\]", "", text or "")


def judge_sources(rag: dict | None) -> list[dict]:
    """채점 프롬프트에 싣는 근거 자료 — 전체 문맥이 있으면 그것, 없으면 인용 출처 발췌."""
    if not rag:
        return []
    base = rag.get("contexts") if rag.get("contexts_complete") else rag.get("sources")
    return list(base or [])[:MAX_SOURCES]


def case_facts(item: dict, answer: str, rag: dict | None) -> dict[str, Any]:
    """케이스 하나의 RAG 사실 — 결정 룰·지표가 함께 쓴다 (LLM 없음)."""
    if rag is None:
        return {"has_rag": False}
    srcs = rag.get("sources") or []
    ids = {s["id"] for s in srcs}
    cites = cited_ids(answer)
    fams = [s["family"] for s in srcs]
    ctx_fams = [s["family"] for s in rag.get("contexts") or []] if rag.get("contexts") else fams
    expect = expand_expect(((item.get("rag") or {}).get("expect")))
    out_of_range = [n for n in cites if n not in ids]
    example_cited = [s for s in srcs if s["role"] == "example" and s["id"] in cites]
    return {
        "has_rag": True,
        "mode": rag.get("mode", ""),
        "n_sources": len(srcs),
        "n_cited": len(cites),
        "cited": cites,
        "out_of_range": out_of_range,
        "example_cited": [{"id": s["id"], "family": s["family"], "title": s["title"][:60]} for s in example_cited],
        "families": sorted(set(fams)),
        "unknown_families": sorted({s["family"] for s in srcs if s["family"] == "OTHER"}),
        "english_sources": sum(1 for s in srcs if s["lang"] == "en"),
        "expect": sorted(expect),
        "expect_hit": (bool(expect & set(ctx_fams)) if expect else None),
        "contexts_complete": bool(rag.get("contexts_complete")),
        "coverage": (item.get("rag") or {}).get("coverage"),
    }


def rag_view(rag: dict | None) -> dict | None:
    """채점 프롬프트·채점 결과 검사에 쓰는 RAG 보기. RAG 없는 응답·응급 고정 안내는 None (근거 대조를 하지 않음)."""
    if not rag or rag.get("mode") == "emergency":
        return None
    return {"mode": rag.get("mode", ""), "sources": judge_sources(rag), "complete": bool(rag.get("contexts_complete"))}
