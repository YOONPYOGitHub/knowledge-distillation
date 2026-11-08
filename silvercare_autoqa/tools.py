"""연결 전에 미리 돌려 두는 점검 도구 — 모델이 완성되기 전에도 쓸 수 있다.

- preflight        : 골든셋·팀 UI 백엔드(run·체크포인트)·Claude Code 로그인을 실행 전에 확인 (하나라도 실패하면 종료코드 2)
- check_responses  : 응답 JSONL 이 채점에 들어갈 수 있는지 (형식·문항 빠짐·중복·빈 답·프롬프트 섞임)
- contamination    : 골든셋 발화·모범 답안이 학습 데이터에 섞였는지 (섞이면 시험 문제를 미리 본 것)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .golden import load_golden, validate_golden
from .responders import APP_SYSTEM
from .textutil import nfc

# 응답에 이 문장이 보이면 시스템 프롬프트가 답에 섞인 것 (팀 UI 백엔드는 prompt 까지 붙여 돌려준다)
_LEAK_PROBES = [ln.strip("- ").strip()[:18] for ln in APP_SYSTEM.splitlines() if len(ln.strip()) > 20][:4]


# ───────────────────────────── preflight ─────────────────────────────

def preflight(golden_path: str, *, backend: str | None = None, run_id: str | None = None,
              model_ids: list[str] | None = None, judge: bool = False, claude_bin: str | None = None,
              timeout: int = 10) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    g = load_golden(golden_path)
    errs = validate_golden(g)
    checks.append(("골든셋 검증", not errs, f"{len(g['items'])}문항 · {g['golden_version']}" if not errs else errs[0]))
    if backend:
        import requests
        try:
            h = requests.get(backend.rstrip("/") + "/health", timeout=timeout)
            checks.append(("팀 UI 백엔드 /health", h.ok, f"HTTP {h.status_code}"))
        except Exception as e:  # noqa: BLE001
            checks.append(("팀 UI 백엔드 /health", False, f"연결 실패: {str(e)[:80]}"))
        if run_id:
            try:
                runs = requests.get(backend.rstrip("/") + "/runs", timeout=timeout).json()
                runs = runs.get("runs", runs) if isinstance(runs, dict) else runs
                info = next((r for r in runs if r.get("run_id") == run_id), None)
                checks.append((f"run '{run_id}' 있음", info is not None,
                               "있음" if info else f"없음 (있는 run: {[r.get('run_id') for r in runs][:6]})"))
                if info:
                    ck = info.get("checkpoints") or {}
                    for mid in model_ids or ["teacher_ft", "student_kd", "student_ft", "student_base"]:
                        ok = bool(ck.get(mid, mid == "student_base"))
                        checks.append((f"체크포인트 {mid}", ok, "있음" if ok else "없음 — 이 모델은 빼고 돌리거나 학습을 먼저"))
            except Exception as e:  # noqa: BLE001
                checks.append(("run 목록 /runs", False, str(e)[:80]))
    if judge:
        from .judge import claude_cli_call, find_claude
        try:
            exe = find_claude(claude_bin)
            out = claude_cli_call("너는 점검기다. JSON 만 낸다.", "ok 를 true 로 낸다.",
                                  {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                                  claude_bin=exe, timeout=180)
            checks.append(("Claude Code 채점기 (claude -p)", out.get("ok") is True, exe))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            hint = " — 터미널에서 claude 를 한 번 실행해 다시 로그인" if ("401" in msg or "auth" in msg.lower() or "expired" in msg.lower()) else ""
            checks.append(("Claude Code 채점기 (claude -p)", False, msg[:120] + hint))
    return checks


# ───────────────────────────── check_responses ─────────────────────────────

def check_responses(path: str, golden_path: str) -> dict[str, Any]:
    g = load_golden(golden_path)
    ids = set(g["by_id"])
    rows, problems = [], []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{i}행: JSON 아님 ({e})")
            continue
        miss = [k for k in ("model", "item_id", "response") if k not in r]
        if miss:
            problems.append(f"{i}행: 필드 없음 {miss}")
            continue
        if r["item_id"] not in ids:
            problems.append(f"{i}행: 골든셋에 없는 문항 {r['item_id']}")
        rows.append(r)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    report = {"n_rows": len(rows), "models": {}, "problems": problems}
    for m, rs in sorted(by_model.items()):
        c = Counter(r["item_id"] for r in rs)
        dup = sorted(k for k, v in c.items() if v > 1 and not all("sample" in r for r in rs if r["item_id"] == k))
        empty = [r["item_id"] for r in rs if not str(r["response"]).strip()]
        leak = [r["item_id"] for r in rs if any(p and p in str(r["response"]) for p in _LEAK_PROBES)]
        echo = [r["item_id"] for r in rs if re.search(r"어르신:|돌봄 AI:|### |<\|im_start\|>", str(r["response"]))]
        lens = sorted(len(str(r["response"])) for r in rs)
        report["models"][m] = {
            "n": len(rs), "missing_items": sorted(ids - set(c)), "duplicates": dup, "empty": empty,
            "system_prompt_leak": leak, "turn_marker_echo": echo,
            "median_chars": lens[len(lens) // 2] if lens else 0,
        }
        gen_err = [r["item_id"] for r in rs if (r.get("meta") or {}).get("error")]
        report["models"][m]["generation_error"] = gen_err
        if report["models"][m]["missing_items"]:
            problems.append(f"{m}: 골든셋 문항 {len(report['models'][m]['missing_items'])}개 빠짐 "
                            f"{report['models'][m]['missing_items'][:5]} — 빠진 채로 채점하면 응급 문항 없이 Pass 가 나올 수 있음")
        if gen_err:
            problems.append(f"{m}: 생성 오류가 있는 답 {gen_err[:5]} — 서버 문제를 빈 답으로 채점하지 않는다")
        if leak:
            problems.append(f"{m}: 시스템 프롬프트가 답에 섞임 {leak[:5]} — 응답 수집기가 prompt 를 떼지 못함")
        if dup:
            problems.append(f"{m}: 같은 문항 응답 중복 {dup[:5]}")
    report["ok"] = not problems
    return report


# ───────────────────────────── contamination ─────────────────────────────

def _grams(s: str, n: int = 8) -> set[str]:
    t = re.sub(r"\s+", "", nfc(s))
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


TEXT_SUFFIXES = (".jsonl", ".json", ".txt", ".csv", ".tsv")


def _read_text(f: Path, problems: list[str]) -> str | None:
    raw = f.read_bytes()
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    problems.append(f"글자를 읽지 못함(utf-8·cp949 아님): {f}")
    return None


def _iter_texts(paths: Iterable[str], problems: list[str]) -> Iterable[tuple[str, str]]:
    """학습 데이터 문장을 하나씩 낸다. 없는 경로·못 읽는 형식은 problems 에 적는다 (조용히 건너뛰면 '통과'로 보임, 감사 A06)."""
    for p in paths:
        P = Path(p)
        if not P.exists():
            problems.append(f"경로 없음: {p}")
            continue
        files = [P] if P.is_file() else sorted(f for f in P.rglob("*") if f.is_file())
        for f in files:
            if f.suffix.lower() not in TEXT_SUFFIXES:
                if f.suffix.lower() in (".parquet", ".arrow", ".gz", ".zip", ".xlsx", ".pt", ".bin"):
                    problems.append(f"검사하지 못하는 형식(먼저 jsonl·txt 로 풀어 넣기): {f}")
                continue
            text = _read_text(f, problems)
            if text is None:
                continue
            if f.suffix.lower() == ".json":
                try:
                    obj = json.loads(text)
                    yield from ((str(f), s) for s in _walk_strings(obj))
                    continue
                except json.JSONDecodeError:
                    pass
            for i, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    yield (f"{f}:{i}", line)


def _walk_strings(o: Any) -> Iterable[str]:
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _walk_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_strings(v)


def contamination(train_paths: list[str], golden_path: str, *, threshold: float = 0.6, extra: str | None = None) -> dict:
    """골든셋 발화·모범 답안의 8글자 묶음이 학습 문장에 threshold 이상 들어 있으면 오염 후보로 본다."""
    g = load_golden(golden_path)
    probes = []
    for it in g["items"]:
        probes.append((it["id"], "utterance", it["utterance"]))
        probes.append((it["id"], "reference_answer", it["reference_answer"]))
    if extra:
        doc = json.loads(Path(extra).read_text(encoding="utf-8"))
        for u in doc.get("utterances", []):
            probes.append((u["item_id"], f"heldout:{u['kind']}", u["utterance"]))
    pg = [(iid, kind, _grams(txt)) for iid, kind, txt in probes]
    hits = []
    n_lines = 0
    problems: list[str] = []
    for loc, line in _iter_texts(train_paths, problems):
        n_lines += 1
        lg = _grams(line)
        if not lg:
            continue
        for iid, kind, gset in pg:
            if not gset:
                continue
            cover = len(gset & lg) / len(gset)
            if cover >= threshold:
                hits.append({"item_id": iid, "kind": kind, "cover": round(cover, 2), "where": loc, "text": line[:120]})
    if n_lines == 0:
        problems.append("검사한 학습 문장이 0줄 — 0줄 검사는 통과가 아니다")
    return {"n_train_lines": n_lines, "threshold": threshold, "n_hits": len(hits), "problems": problems,
            "items_hit": sorted({h["item_id"] for h in hits}), "hits": hits[:200], "ok": not hits and not problems}
