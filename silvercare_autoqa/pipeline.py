"""회차 파이프라인 — prep → judge → import → verify → finalize.

run 폴더 구성
  meta.json                 회차 정보·평가기준 지문
  cases.jsonl               케이스(모델 × 문항) — 응답 원문과 결정 룰 결과를 동결
  packets/judge/*.json      채점 요청 (packet_id 는 모델·문항을 드러내지 않는 해시)
  results/judge/*.json      채점 결과 (claude-cli 또는 외부 실행기가 채움)
  packets/verify/*.json     반박 검증 요청 (Hold 로 이어지는 채점기 지적만)
  results/verify/*.json     반박 검증 결과
  scored.jsonl · metrics.json · report.md · report.html · defects.csv   finalize 산출물
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import fingerprint as fp_mod
from .golden import load_golden, validate_golden
from .judge import (DEFAULT_EFFORT, DEFAULT_MODEL, judge_validator_factory, validate_judgment, validate_verify,
                    verify_validator_factory)
from .metrics import model_metrics, paired_compare, teacher_compare
from .prompts import JUDGE_SYSTEM, VERIFY_SYSTEM, build_judge_user, build_verify_user, judge_schema, verify_schema
from .rules import det_checks
from .verdict import build_findings, verdict_of


def _jl_write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _jl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_responses(path: str | Path) -> list[dict]:
    """응답 JSONL: 한 줄 = {"model": "student_kd", "item_id": "SC-E01", "response": "...", "meta": {...}}"""
    rows = _jl_read(Path(path))
    for i, r in enumerate(rows):
        for k in ("model", "item_id", "response"):
            if k not in r:
                raise ValueError(f"{path}:{i + 1} '{k}' 필드 없음")
    return rows


def prep(golden_path: str | Path, responses: list[dict], run_dir: str | Path, *, judge_model: str = DEFAULT_MODEL,
         judge_effort: str = DEFAULT_EFFORT, label: str = "") -> dict:
    run = Path(run_dir)
    if (run / "cases.jsonl").exists():
        raise FileExistsError(f"{run} 에 이미 회차가 있음 — 새 폴더를 쓰거나 finalize 만 다시 돌린다")
    golden = load_golden(golden_path)
    errs = validate_golden(golden)
    if errs:
        raise ValueError("골든셋 검증 실패:\n" + "\n".join(errs[:20]))
    fp = fp_mod.compute(golden_path, judge_model, judge_effort)
    items = golden["by_id"]
    (run / "packets" / "judge").mkdir(parents=True, exist_ok=True)
    (run / "results" / "judge").mkdir(parents=True, exist_ok=True)
    (run / "packets" / "judge" / "JUDGE_SYSTEM.txt").write_text(JUDGE_SYSTEM, encoding="utf-8")
    schema = judge_schema()
    cases, seen = [], set()
    for r in responses:
        if r["item_id"] not in items:
            raise ValueError(f"골든셋에 없는 문항: {r['item_id']}")
        key = f"{r['model']}|{r['item_id']}|{r.get('sample', 0)}"
        if key in seen:
            raise ValueError(f"같은 모델·문항 응답이 두 번 있음: {key}")
        seen.add(key)
        pid = hashlib.sha256((fp["fingerprint"] + key).encode()).hexdigest()[:12]
        item = items[r["item_id"]]
        det = det_checks(item, r["response"]).to_dict()
        case = {"pid": pid, "key": key, "model": r["model"], "item_id": r["item_id"], "response": r["response"],
                "meta": r.get("meta") or {}, "det": det, "judged": not det["fatal_format"]}
        cases.append(case)
        if case["judged"]:
            hyb = [h for h in det["hits"] if h["kind"] == "hybrid"]
            pk = {"packet_id": pid, "kind": "judge", "system_file": "JUDGE_SYSTEM.txt",
                  "user": build_judge_user(pid, item, r["response"], hyb, golden["sources"]), "schema": schema}
            (run / "packets" / "judge" / f"{pid}.json").write_text(json.dumps(pk, ensure_ascii=False, indent=1),
                                                                    encoding="utf-8")
    _jl_write(run / "cases.jsonl", cases)
    models = sorted({c["model"] for c in cases})
    meta = {"label": label, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "golden_path": str(golden_path),
            "golden_version": golden["golden_version"], "golden_sha256": golden["sha256"],
            "judge_model": judge_model, "judge_effort": judge_effort, "fingerprint": fp["fingerprint"],
            "fingerprint_parts": fp["parts"], "models": models, "n_cases": len(cases),
            "n_judge_packets": sum(1 for c in cases if c["judged"])}
    (run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def _load(run: Path) -> tuple[dict, list[dict], dict, dict]:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    golden = load_golden(meta["golden_path"])
    if golden["sha256"] != meta["golden_sha256"]:
        raise RuntimeError("골든셋이 prep 이후 바뀜 — 같은 기준으로 다시 prep 해야 함")
    cases = _jl_read(run / "cases.jsonl")
    return meta, cases, golden, golden["by_id"]


def import_judge(run_dir: str | Path) -> dict:
    """채점 결과를 검증해 붙이고, Hold 로 이어지는 채점기 지적은 반박 검증 요청을 만든다."""
    run = Path(run_dir)
    meta, cases, golden, items = _load(run)
    vdir = run / "packets" / "verify"
    vdir.mkdir(parents=True, exist_ok=True)
    (run / "results" / "verify").mkdir(parents=True, exist_ok=True)
    (vdir / "VERIFY_SYSTEM.txt").write_text(VERIFY_SYSTEM, encoding="utf-8")
    vschema = verify_schema()
    stats = {"ok": 0, "error": 0, "missing": 0, "skipped_format": 0, "verify_packets": 0}
    for c in cases:
        c["judgment"], c["judge_errors"] = None, []
        if not c["judged"]:
            stats["skipped_format"] += 1
            continue
        rp = run / "results" / "judge" / f"{c['pid']}.json"
        if not rp.exists():
            c["judge_errors"] = ["채점 결과 파일 없음"]
            stats["missing"] += 1
            continue
        try:
            obj = json.loads(rp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            c["judge_errors"] = [f"JSON 해석 실패: {e}"]
            stats["error"] += 1
            continue
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        errs = validate_judgment(obj, c["pid"], items[c["item_id"]], c["response"], hids)
        if errs:
            c["judge_errors"] = errs
            stats["error"] += 1
            continue
        c["judgment"] = obj
        stats["ok"] += 1
        need = [f for f in build_findings(items[c["item_id"]], c["det"], obj) if f["needs_verify"]]
        if need:
            pk = {"packet_id": c["pid"], "kind": "verify", "system_file": "VERIFY_SYSTEM.txt",
                  "user": build_verify_user(c["pid"], items[c["item_id"]], c["response"], need), "schema": vschema,
                  "fids": [f["fid"] for f in need]}
            (vdir / f"{c['pid']}.json").write_text(json.dumps(pk, ensure_ascii=False, indent=1), encoding="utf-8")
            stats["verify_packets"] += 1
    _jl_write(run / "cases.jsonl", cases)
    meta["import_stats"] = stats
    (run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def finalize(run_dir: str | Path, *, teacher: str | None = None, pairs: list[tuple[str, str]] | None = None) -> dict:
    run = Path(run_dir)
    meta, cases, golden, items = _load(run)
    vstats = {"keep": 0, "refute": 0, "downgrade": 0, "missing": 0, "error": 0}
    for c in cases:
        decisions = None
        # 반박 검증이 필요한 지적은 요청 파일이 아니라 판정식에서 다시 계산한다
        # (요청 파일이 없는 사본에서 검증 누락이 '검증 불필요'로 보이던 빈틈을 막음)
        need = [f["fid"] for f in build_findings(items[c["item_id"]], c["det"], c.get("judgment"))
                if f["needs_verify"]] if c.get("judgment") else []
        if need:
            fids = need
            vr = run / "results" / "verify" / f"{c['pid']}.json"
            if vr.exists():
                obj = json.loads(vr.read_text(encoding="utf-8"))
                verr = validate_verify(obj, c["pid"], fids)
                if verr:
                    vstats["error"] += 1
                    c["verify_errors"] = verr
                else:
                    decisions = {d["fid"]: d for d in obj["decisions"]}
                    for d in obj["decisions"]:
                        vstats[d["decision"]] += 1
            else:
                vstats["missing"] += 1
        c["verify"] = decisions
        c["result"] = verdict_of(items[c["item_id"]], c["det"], c.get("judgment"), c.get("judge_errors"), decisions)
    _jl_write(run / "scored.jsonl", cases)

    by_model: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_model[c["model"]].append(c)
    out: dict[str, Any] = {"meta": meta, "verify_stats": vstats, "models": {}}
    for m, cs in sorted(by_model.items()):
        out["models"][m] = model_metrics(cs, items)
    if teacher and teacher in by_model:
        out["teacher"] = teacher
        out["teacher_compare"] = {m: teacher_compare(cs, by_model[teacher], items)
                                  for m, cs in by_model.items() if m != teacher}
    out["pairs"] = [paired_compare(by_model[a], by_model[b], a, b) for a, b in (pairs or [])
                    if a in by_model and b in by_model]
    # 회차 유효성 — 채점이 덜 됐거나 검증이 빠졌으면 수치를 공유하지 않는다
    n_judged = sum(1 for c in cases if c["judged"])
    n_err = sum(1 for c in cases if c["result"]["verdict"] == "error")
    n_unv = sum(1 for c in cases if c["result"].get("unverified"))
    out["validity"] = {
        "n_cases": len(cases), "n_judged": n_judged, "n_error": n_err,
        "error_rate": (n_err / n_judged) if n_judged else 0.0,
        "verify_missing": vstats["missing"], "verify_error": vstats["error"], "unverified_cases": n_unv,
        "ok": n_err == 0 and vstats["missing"] == 0 and vstats["error"] == 0 and n_unv == 0,
    }
    (run / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    from .report import write_reports
    write_reports(run, out, cases, golden)
    return out


def cli_judge(run_dir: str | Path, *, workers: int = 3, model: str | None = None, effort: str | None = None,
              claude_bin: str | None = None, phase: str = "judge") -> dict:
    """claude -p 로 채점(또는 반박 검증) 요청을 처리한다."""
    from .judge import run_packets_with_cli
    run = Path(run_dir)
    meta, cases, golden, items = _load(run)
    model = model or meta["judge_model"]
    effort = effort or meta["judge_effort"]
    if (model, effort) != (meta["judge_model"], meta["judge_effort"]):
        raise ValueError("prep 때와 다른 채점 모델·effort — 지문이 달라지므로 새 회차로 prep 해야 함")
    if phase == "judge":
        validator = judge_validator_factory({c["pid"]: c for c in cases}, items)
        return run_packets_with_cli(run / "packets" / "judge", run / "results" / "judge", validator,
                                    model=model, effort=effort, workers=workers, claude_bin=claude_bin)
    vindex = {}
    for p in (run / "packets" / "verify").glob("*.json"):
        vindex[p.stem] = json.loads(p.read_text(encoding="utf-8"))["fids"]
    return run_packets_with_cli(run / "packets" / "verify", run / "results" / "verify",
                                verify_validator_factory(vindex), model=model, effort=effort, workers=workers,
                                claude_bin=claude_bin)
