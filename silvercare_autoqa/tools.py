"""연결 전에 미리 돌려 두는 점검 도구 — 모델이 완성되기 전에도 쓸 수 있다.

- preflight        : 골든셋·팀 UI 백엔드(run·체크포인트)·Claude Code 로그인을 실행 전에 확인 (하나라도 실패하면 종료코드 2)
- check_responses  : 응답 JSONL 이 채점에 들어갈 수 있는지 (형식·문항 빠짐·중복·빈 답·프롬프트 섞임)
- contamination    : 골든셋 발화·모범 답안이 학습 데이터에 섞였는지 (섞이면 시험 문제를 미리 본 것)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .golden import load_golden, validate_golden
from .responders import APP_SYSTEM
from .textutil import nfc

# 응답에 이 문장이 보이면 시스템 프롬프트가 답에 섞인 것 (팀 UI 백엔드는 prompt 까지 붙여 돌려준다)
_LEAK_PROBES = [ln.strip("- ").strip()[:18] for ln in APP_SYSTEM.splitlines() if len(ln.strip()) > 20][:4]


# ───────────────────────────── preflight ─────────────────────────────

def selftest(timeout: int = 600) -> tuple[bool, str]:
    """autoqa 자체 시험을 이 PC 에서 돌린다 — 회차 전에 '이 환경에서 판정 코드가 제대로 도는지'를 본다 (노하우 점검 M1·K30).
    환경 문제(pytest 없음·불러오기 실패)와 코드 회귀(시험 실패)를 나눠 알린다."""
    import subprocess
    import sys

    pkg = Path(__file__).resolve().parent
    try:
        import pytest  # noqa: F401
    except ImportError:
        return False, "환경 문제 — 이 파이썬에 pytest 가 없음 (pip install pytest 후 다시)"
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", str(pkg / "tests"), "-q", "-p", "no:cacheprovider", "-x"],
                       cwd=pkg.parent, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode == 0:
        return True, tail
    if "ImportError" in r.stdout or "ModuleNotFoundError" in r.stdout or r.returncode in (3, 4):
        return False, f"환경 문제 — 시험을 불러오지 못함 ({tail}). 필요한 패키지(pyyaml·requests)와 파이썬 버전을 확인"
    return False, f"코드 회귀 — 자체 시험 실패 ({tail}). 이 커밋의 autoqa 로는 회차를 돌리지 않는다"


def preflight(golden_path: str, *, backend: str | None = None, run_id: str | None = None,
              model_ids: list[str] | None = None, judge: bool = False, claude_bin: str | None = None,
              timeout: int = 10, run_selftest: bool = False) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    g = load_golden(golden_path)
    errs = validate_golden(g)
    checks.append(("골든셋 검증", not errs, f"{len(g['items'])}문항 · {g['golden_version']}" if not errs else errs[0]))
    if run_selftest:
        ok, note = selftest()
        checks.append(("autoqa 자체 시험", ok, note))
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
        import tempfile

        try:
            exe = find_claude(claude_bin)
            # 여러 줄 시스템 지시가 끝까지 전달되는지 본다 — 마지막 줄의 확인값을 돌려받아야 통과 (감사 R02)
            token = "확인값-7349"
            system = "너는 점검기다.\n둘째 줄: 아무 일도 하지 않는다.\n셋째 줄: JSON 만 낸다.\n" + token
            out = claude_cli_call(system, "시스템 지시 마지막 줄의 확인값을 그대로 check 에 넣는다.",
                                  {"type": "object", "properties": {"check": {"type": "string"}}, "required": ["check"]},
                                  claude_bin=exe, timeout=180, cwd=tempfile.mkdtemp(prefix="silvercare_preflight_"))
            ok = (out.get("check") or "").strip() == token
            checks.append(("Claude Code 채점기 (claude -p · 여러 줄 지시 전달)", ok,
                           exe if ok else f"{exe} — 지시 마지막 줄이 전달되지 않음(받은 값 {out.get('check')!r}). claude 를 새 버전으로"))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            hint = " — 터미널에서 claude 를 한 번 실행해 다시 로그인" if ("401" in msg or "auth" in msg.lower() or "expired" in msg.lower()) else ""
            checks.append(("Claude Code 채점기 (claude -p)", False, msg[:120] + hint))
    return checks


# ───────────────────────────── check_responses ─────────────────────────────

GEN_FIELDS = ("prompt_format", "system_sha", "temperature", "max_new_tokens", "repetition_penalty")


def gen_signature(meta: dict | None) -> dict:
    """응답 한 줄의 생성 조건 (입력 형식·시스템 프롬프트·디코딩). 응답 수집기마다 적는 자리가 달라 여기서 맞춘다.
    비교하는 두 모델의 조건이 다르면 차이가 모델 탓인지 조건 탓인지 모른다 (노하우 점검 M5)."""
    m = meta or {}
    p = m.get("params") or {}
    return {"prompt_format": m.get("prompt_format"), "system_sha": m.get("system_sha"),
            "temperature": m.get("temperature", p.get("temperature")),
            "max_new_tokens": m.get("max_new_tokens", p.get("max_new_tokens")),
            "repetition_penalty": m.get("repetition_penalty", p.get("repetition_penalty"))}


def gen_diff(a: list[dict], b: list[dict]) -> list[str]:
    """두 모델의 생성 조건 묶음에서 다른 항목 (값을 모르는 항목은 비교하지 않는다)."""
    out = []
    for f in GEN_FIELDS:
        va = {x[f] for x in a if x.get(f) is not None}
        vb = {x[f] for x in b if x.get(f) is not None}
        if va and vb and va != vb:
            out.append(f"{f} {sorted(map(str, va))}≠{sorted(map(str, vb))}")
    return out


def model_gen_conditions(rows: list[dict]) -> list[dict]:
    uniq = {json.dumps(gen_signature(r.get("meta")), sort_keys=True, ensure_ascii=False) for r in rows}
    return [json.loads(s) for s in sorted(uniq)]


# 문제 종류. prep 은 HARD 를 늘 거부하고, missing 은 --allow-partial, FLAGGED 는 --allow-flagged 일 때만 받는다.
HARD = ("bad_row", "unknown_item", "duplicate", "multi_sample")
FLAGGED = ("empty", "leak", "echo", "generation_error", "stale_question")


def read_jsonl_rows(path: str | Path) -> tuple[list[dict], list[tuple[str, str]]]:
    """응답 JSONL 읽기. Windows 에서 저장한 BOM 붙은 UTF-8·cp949 도 읽는다 (감사 R17). 깨진 줄은 문제로 돌려준다."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return [], [("bad_row", f"{path}: 글자를 읽지 못함 (UTF-8·cp949 아님)")]
    rows, problems = [], []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(("bad_row", f"{i}행: JSON 아님 ({e})"))
            continue
        miss = [k for k in ("model", "item_id", "response") if not isinstance(r, dict) or k not in r]
        if miss:
            problems.append(("bad_row", f"{i}행: 필드 없음 {miss}"))
            continue
        rows.append(r)
    return rows, problems


def check_rows(rows: list[dict], golden: dict, problems: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """응답 묶음이 채점에 들어갈 수 있는지. prep·run 과 check-responses 가 같은 함수를 쓴다 (감사 D06).
    problems: (종류, 설명). 종류는 HARD·FLAGGED·'missing'."""
    from .rules import load_lexicons

    echo_marks = load_lexicons()["echo"]           # 결정 룰 FMT-ECHO 와 같은 표식 목록 (한 곳)
    ids = set(golden["by_id"])
    problems = list(problems or [])
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["item_id"] not in ids:
            problems.append(("unknown_item", f"{r['model']}: 골든셋에 없는 문항 {r['item_id']}"))
        if r.get("sample") not in (None, 0):
            problems.append(("multi_sample", f"{r['model']}·{r['item_id']}: sample={r.get('sample')} — 한 문항에 답 여러 개는 아직 "
                                             "집계 규칙이 없어 받지 않음 (답 하나하나를 다른 문항처럼 세면 신뢰구간이 좁아짐, 감사 D09)"))
        by_model[r["model"]].append(r)
    report: dict[str, Any] = {"n_rows": len(rows), "models": {}}
    for m, rs in sorted(by_model.items()):
        c = Counter(r["item_id"] for r in rs)
        dup = sorted(k for k, v in c.items() if v > 1)
        empty = [r["item_id"] for r in rs if not str(r["response"]).strip()]
        leak = [r["item_id"] for r in rs if any(p and p in str(r["response"]) for p in _LEAK_PROBES)]
        echo = [r["item_id"] for r in rs if any(mk in str(r["response"]) for mk in echo_marks)]
        gen_err = [r["item_id"] for r in rs if (r.get("meta") or {}).get("error")]
        # 답을 받은 뒤 골든셋 질문을 고쳤으면 그 답은 다른 질문의 답이다 — 수집기가 보낸 질문(meta.message)과 대조
        from .responders import user_turn
        stale = [r["item_id"] for r in rs if (r.get("meta") or {}).get("message") and r["item_id"] in golden["by_id"]
                 and (r["meta"]["message"]).strip() != user_turn(golden["by_id"][r["item_id"]]).strip()]
        lens = sorted(len(str(r["response"])) for r in rs)
        missing = sorted(ids - set(c))
        fmts = sorted({str((r.get("meta") or {}).get("prompt_format") or "?") for r in rs})
        report["models"][m] = {"n": len(rs), "missing_items": missing, "duplicates": dup, "empty": empty,
                               "system_prompt_leak": leak, "turn_marker_echo": echo, "generation_error": gen_err,
                               "prompt_formats": fmts, "gen_conditions": model_gen_conditions(rs),
                               "median_chars": lens[len(lens) // 2] if lens else 0}
        if missing:
            problems.append(("missing", f"{m}: 골든셋 문항 {len(missing)}개 빠짐 {missing[:5]} — 빠진 채로 채점하면 응급 문항 없이 "
                                        "Pass 가 나올 수 있음"))
        if stale:
            problems.append(("stale_question", f"{m}: 골든셋 질문이 답을 받은 뒤 바뀜 {stale[:5]} — 그 문항 답을 다시 받는다"))
        if gen_err:
            problems.append(("generation_error", f"{m}: 생성 오류가 있는 답 {gen_err[:5]} — 서버 문제를 빈 답으로 채점하지 않는다"))
        if empty:
            problems.append(("empty", f"{m}: 빈 답 {len(empty)}개 {empty[:5]} — 생성이 실패했는지 먼저 본다 "
                                      "(빈 답을 채점하면 응급 문항이 모두 응급 놓침이 됨)"))
        if leak:
            problems.append(("leak", f"{m}: 시스템 프롬프트가 답에 섞임 {leak[:5]} — 응답 수집기가 prompt 를 떼지 못함 "
                                     "(프롬프트 안의 119 때문에 응급 놓침 확정 룰이 꺼짐)"))
        if echo:
            problems.append(("echo", f"{m}: 차례 표식('어르신:' 등)이 답에 섞임 {echo[:5]}"))
        if dup:
            problems.append(("duplicate", f"{m}: 같은 문항 응답 중복 {dup[:5]}"))
    # 같은 회차 안에서 모델마다 입력 형식·디코딩이 다르면 모델 차이와 조건 차이가 섞인다 (막지는 않고 알린다)
    known = {m: [f for f in s["prompt_formats"] if f != "?"] for m, s in report["models"].items()}
    distinct = sorted({f for fs in known.values() for f in fs})
    report["warnings"] = ([f"모델마다 입력 형식이 다름 {({m: fs for m, fs in known.items() if fs})} — 같은 모델의 반복·양자화 "
                           "전후 비교라면 형식을 맞춘다. 온담과 상용 LLM 처럼 다른 시스템끼리는 다른 것이 정상이다 "
                           "(INTEGRATION.md §3)"] if len(distinct) > 1 else [])
    for m, s in report["models"].items():
        if len(s["gen_conditions"]) > 1:
            report["warnings"].append(f"{m}: 한 모델 안에서 생성 조건이 섞임 {s['gen_conditions']}")
    report["problems"] = [msg for _, msg in problems]
    report["problem_kinds"] = sorted({k for k, _ in problems})
    report["_problems"] = problems
    report["ok"] = not problems
    return report


def check_responses(path: str, golden_path: str) -> dict[str, Any]:
    rows, problems = read_jsonl_rows(path)
    rep = check_rows(rows, load_golden(golden_path), problems)
    rep.pop("_problems", None)
    return rep


# ───────────────────────────── contamination ─────────────────────────────

def _grams(s: str, n: int = 8) -> set[str]:
    t = re.sub(r"\s+", "", nfc(s))
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


# 풀지 않으면 읽을 수 없는 형식 — 검사하지 못했다고 문제로 남긴다 (조용히 건너뛰면 '통과'로 보임, 감사 A06·D10)
BINARY_SUFFIXES = (".parquet", ".arrow", ".feather", ".gz", ".zip", ".zst", ".bz2", ".xz", ".7z", ".tar", ".xlsx",
                   ".pt", ".bin", ".safetensors", ".npy", ".npz", ".pkl", ".pickle")
WINDOW = 5     # 골든 문장이 여러 줄(문장마다 한 줄)로 나뉘어 들어가도 잡도록, 이어진 5줄을 함께 본다 (감사 R04)


def _read_text(f: Path, problems: list[str]) -> str | None:
    raw = f.read_bytes()
    if b"\x00" in raw[:4096]:
        problems.append(f"글자 파일이 아님(검사 못 함): {f}")
        return None
    for enc in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    problems.append(f"글자를 읽지 못함(utf-8·cp949 아님): {f}")
    return None


def _iter_texts(paths: Iterable[str], problems: list[str], stats: dict) -> Iterable[tuple[str, str, str]]:
    """학습 데이터 글을 (파일, 위치, 글) 로 하나씩 낸다. 확장자와 상관없이 글자 파일이면 읽는다.
    JSON 줄은 풀어서 문자열 값만 본다 — json.dumps 기본값(\\uXXXX)으로 저장된 한글도 원래 글자로 비교 (감사 R04)."""
    for p in paths:
        P = Path(p)
        if not P.exists():
            problems.append(f"경로 없음: {p}")
            continue
        files = [P] if P.is_file() else sorted(f for f in P.rglob("*") if f.is_file())
        for f in files:
            if any(s.lower() in BINARY_SUFFIXES for s in f.suffixes[-2:]):
                problems.append(f"검사하지 못하는 형식(먼저 jsonl·txt 로 풀어 넣기): {f}")
                stats["skipped"] += 1
                continue
            text = _read_text(f, problems)
            if text is None:
                stats["skipped"] += 1
                continue
            stats["read"] += 1
            if f.suffix.lower() == ".json":
                try:
                    obj = json.loads(text)
                    yield from ((str(f), f"{f}#{i}", s) for i, s in enumerate(_walk_strings(obj)))
                    continue
                except json.JSONDecodeError:
                    pass
            for i, line in enumerate(text.splitlines(), 1):
                s = line.strip()
                if not s:
                    continue
                if s[:1] in "{[":
                    try:
                        obj = json.loads(s)
                        for t in _walk_strings(obj):
                            yield (str(f), f"{f}:{i}", t)
                        continue
                    except json.JSONDecodeError:
                        pass
                yield (str(f), f"{f}:{i}", line)


def _walk_strings(o: Any) -> Iterable[str]:
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _walk_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_strings(v)


def contamination(train_paths: list[str], golden_path: str, *, threshold: float = 0.6, extra: str | None = None,
                  show_text: bool = False) -> dict:
    """골든셋 발화·모범 답안의 8글자 묶음이 학습 문장에 threshold 이상 들어 있으면 오염 후보로 본다.
    결과에는 학습 문장 원문을 넣지 않는다 (위치·문항·겹친 비율만). 학습 데이터에는 반출 금지 원문(AI Hub 71762)이
    섞이므로, 결과 파일을 커밋해도 원문이 새지 않게 한다 (노하우 점검 K59 — 법적 요건). show_text 는 이 PC 에서 볼 때만."""
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
    index: dict[str, list[int]] = defaultdict(list)       # 8글자 묶음 → 그 묶음을 가진 골든 글 번호 (큰 학습 데이터도 빨리)
    for k, (_, _, gset) in enumerate(pg):
        for g in gset:
            index[g].append(k)
    best: dict[tuple, dict] = {}
    n_lines = 0
    problems: list[str] = []
    stats = {"read": 0, "skipped": 0}
    window: deque = deque(maxlen=WINDOW)
    cur_file = None
    for fname, loc, line in _iter_texts(train_paths, problems, stats):
        n_lines += 1
        if fname != cur_file:
            window.clear()
            cur_file = fname
        lg = _grams(line)
        window.append(lg)
        if not lg:
            continue
        cands = {k for g in lg for k in index.get(g, ())}
        if not cands:
            continue
        union = set().union(*window)
        for k in cands:
            iid, kind, gset = pg[k]
            cover = len(gset & union) / len(gset)
            if cover >= threshold:
                key = (iid, kind, fname)
                if key not in best or cover > best[key]["cover"]:
                    best[key] = {"item_id": iid, "kind": kind, "cover": round(cover, 2), "where": loc}
                    if show_text:
                        best[key]["text"] = line[:120]
    hits = sorted(best.values(), key=lambda h: (-h["cover"], h["item_id"]))
    if n_lines == 0:
        problems.append("검사한 학습 문장이 0줄 — 0줄 검사는 통과가 아니다")
    return {"n_train_lines": n_lines, "n_files_read": stats["read"], "n_files_skipped": stats["skipped"],
            "threshold": threshold, "n_hits": len(hits), "problems": problems, "contains_train_text": show_text,
            "items_hit": sorted({h["item_id"] for h in hits}), "hits": hits[:200], "ok": not hits and not problems}
