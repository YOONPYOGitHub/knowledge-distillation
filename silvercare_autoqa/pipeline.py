"""회차 파이프라인 — prep → judge → import → verify → finalize.

run 폴더 구성 (회차 형식 2 — 2026-11-08 코드 감사 반영)
  meta.json                 회차 정보·평가기준 지문(prep)·단계별 다시 잰 지문
  golden.yaml               prep 때 쓴 골든셋 사본 — 다른 PC 로 옮기거나 골든셋 v1 이 나온 뒤에도 같은 기준으로 연다
  cases.jsonl               케이스(모델 × 문항) — 응답 원문·응답 해시·결정 룰 결과·RAG 출처 (import 가 채점 결과를 덧붙임)
  packets/judge/*.json      채점 요청. packet_id = 지문 + 모델·문항 + 응답 해시 → 답이 바뀌면 다른 요청이 된다
  results/judge/*.json      채점 결과 (claude-cli 또는 외부 실행기가 채움)
  packets/verify/*.json     반박 검증 요청. packet_id = 케이스 + 반박할 지적 내용의 해시 → 채점이 바뀌면 다른 요청이 된다
  results/verify/*.json     반박 검증 결과
  scored.jsonl · metrics.json · report.md · report.html · defects.csv   finalize 산출물

단계 사이 약속 (감사 D02·R05·R06·R07)
- 결과 파일은 요청 내용의 해시로 이름이 붙는다. 예전 결과가 새 답·새 지적에 붙을 수 없다.
- judge·import·verify·finalize 는 평가기준 지문을 다시 계산한다. prep 때와 다르면 멈춘다 (--rescore 로만 진행, 기록이 남음).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import fingerprint as fp_mod
from .golden import load_golden, validate_golden
from .judge import (DEFAULT_EFFORT, DEFAULT_MODEL, judge_validator_factory, validate_judgment, validate_verify,
                    verify_validator_factory)
from .metrics import (condition_of, condition_summary, model_metrics, pair_condition_summary, paired_compare,
                      reference_compare, resolve_pair, run_validity, _mean_sd, REF_TARGETS)
from .rag import normalize_rag, rag_view as make_rag_view
from .prompts import JUDGE_SYSTEM, VERIFY_SYSTEM, build_judge_user, build_verify_user, judge_schema, verify_schema
from .rules import det_checks, needs_judge
from .verdict import build_findings, conditional_escalation_fids, no_downgrade_fids, verdict_of

RUN_FORMAT = 2
OUTPUTS = ("scored.jsonl", "metrics.json", "report.md", "report.html", "defects.csv")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _jl_write(path: Path, rows: list[dict]) -> None:
    _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _jl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_responses(path: str | Path) -> list[dict]:
    """응답 JSONL: 한 줄 = {"model": "ondam_local", "item_id": "SC-E01", "response": "...", "meta": {...}, "rag": {...}}
    rag 는 선택 — 온담처럼 출처를 돌려주는 시스템만 (rag.py 머리말). 없으면 RAG 없는 모델로 채점한다.
    깨진 줄·필드 빠진 줄이 있으면 멈춘다 (BOM·cp949 로 저장된 파일도 읽는다)."""
    from .tools import read_jsonl_rows

    rows, problems = read_jsonl_rows(path)
    if problems:
        raise ValueError(f"{path}: 응답 파일을 읽지 못한 줄 — " + "; ".join(m for _, m in problems[:5]))
    return rows


def _portable_path(p: str | Path) -> str:
    """회차 폴더를 다른 PC 로 옮겨도 찾을 수 있게, 작업 폴더 안이면 상대 경로로 남긴다 (감사 A05)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _golden_ok(cand: Path, want: str) -> bool:
    from .golden import file_sha256
    raw = hashlib.sha256(cand.read_bytes()).hexdigest()     # 줄바꿈을 맞추기 전에 만든 회차와도 호환
    return want in (file_sha256(cand), raw)


def _resolve_golden(meta: dict, run: Path | None = None) -> Path:
    from .golden import DEFAULT_GOLDEN
    cands = []
    if run is not None and meta.get("run_format", 1) >= 2:
        cands.append(run / meta["golden_path"])                 # 회차 폴더 안의 골든셋 사본 (감사 R19)
    cands += [Path(meta.get("golden_source") or meta["golden_path"]), Path.cwd() / meta["golden_path"], DEFAULT_GOLDEN]
    for cand in cands:
        if cand.exists() and _golden_ok(cand, meta["golden_sha256"]):
            return cand
    raise RuntimeError(f"골든셋을 찾지 못했거나 prep 이후 바뀜 (경로 {meta['golden_path']}) — 같은 골든셋으로 다시 prep 해야 함")


def _nonempty(d: Path) -> bool:
    return d.exists() and any(p.is_file() for p in d.rglob("*"))


def verify_key(pid: str, need: list[dict]) -> str:
    """반박 검증 요청 id — 케이스와 '반박할 지적 내용'에 묶는다. 채점을 다시 하거나 판정식이 바뀌어
    지적이 달라지면 id 도 달라져, 예전 반박 결정이 다른 지적에 붙지 않는다 (감사 D02·R05)."""
    body = json.dumps([[f["fid"], f["gate"], f["cat"], f["ref"], f["claim"], f["evidence"]] for f in need],
                      ensure_ascii=False)
    return "v" + hashlib.sha256((pid + "|" + body).encode("utf-8")).hexdigest()[:11]


def check_criteria(run: Path, meta: dict, stage: str, rescore: bool = False) -> dict:
    """평가기준 지문을 다시 재서 prep 때와 같은지 본다 (감사 R07 · 노하우 M1).
    - judge·import·verify: 채점 지문이 다르면 멈춘다 (받은 채점 결과와 기준이 달라짐). --rescore 면 기록하고 진행.
    - 판정 지문이 다르면 (finalize 포함) '판정식 재적용'으로 기록한다 — 리포트에 두 지문이 함께 나온다."""
    now = fp_mod.compute(_resolve_golden(meta, run), meta["judge_model"], meta["judge_effort"])
    meta.setdefault("stage_fingerprints", {})[stage] = {k: now[k] for k in ("fingerprint", "judging", "scoring")}
    if now["fingerprint"] == meta["fingerprint"]:
        return now
    changed = fp_mod.changed_parts(meta.get("fingerprint_parts") or {}, now["parts"])
    judging_changed = meta.get("judging_fp") != now["judging"]          # 예전 회차(지문 하나)는 둘 다 바뀐 것으로 본다
    if judging_changed and stage != "finalize" and not rescore:
        raise RuntimeError(
            f"채점 지문이 prep 때와 다름 ({meta.get('judging_fp') or meta['fingerprint']} → {now['judging']}, "
            f"달라진 것: {', '.join(changed) or '?'}). 받은 채점 결과와 기준이 달라진다. prep 때와 같은 커밋의 autoqa 로 돌리거나, "
            "새 기준으로 다시 하려면 새 폴더에 prep 한다 (지금 코드로 그대로 진행하려면 --rescore — 리포트에 표시가 남는다).")
    rec = {"stage": stage, "from": meta["fingerprint"], "to": now["fingerprint"], "changed": changed,
           "judging_changed": judging_changed, "scoring_changed": meta.get("scoring_fp") != now["scoring"],
           "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    hist = meta.setdefault("rescored", [])
    if not hist or {k: hist[-1][k] for k in ("stage", "to")} != {k: rec[k] for k in ("stage", "to")}:
        hist.append(rec)
    meta["score_fingerprint"] = now["fingerprint"]
    return now


def prep(golden_path: str | Path, responses: list[dict], run_dir: str | Path, *, judge_model: str = DEFAULT_MODEL,
         judge_effort: str = DEFAULT_EFFORT, label: str = "", allow_partial: bool = False,
         allow_flagged: bool = False) -> dict:
    from .tools import FLAGGED, HARD, check_rows

    run = Path(run_dir)
    if (run / "cases.jsonl").exists() or _nonempty(run / "packets") or _nonempty(run / "results"):
        raise FileExistsError(f"{run} 에 이미 회차 파일(cases·packets·results)이 있음 — 새 폴더를 쓴다 "
                              "(같은 폴더에 다시 prep 하면 예전 채점 결과가 새 답에 붙을 수 있음, 감사 R06)")
    golden = load_golden(golden_path)
    errs = validate_golden(golden)
    if errs:
        raise ValueError("골든셋 검증 실패:\n" + "\n".join(errs[:20]))
    # 응답 점검은 파일을 쓰기 전에 모두 끝낸다 — 중간에 멈춰 반쯤 만든 요청이 남지 않게 (감사 D06·R15)
    rep = check_rows(responses, golden)
    blocking = [msg for kind, msg in rep["_problems"]
                if kind in HARD or (kind == "missing" and not allow_partial) or (kind in FLAGGED and not allow_flagged)]
    if blocking:
        raise ValueError("응답 점검 실패 — 채점하지 않음:\n - " + "\n - ".join(blocking[:12]) +
                         "\n(문항이 일부러 빠진 변이 시험은 --allow-partial, 빈 답·섞인 답을 일부러 넣을 때는 --allow-flagged)")
    fp = fp_mod.compute(golden_path, judge_model, judge_effort)
    items = golden["by_id"]
    cases, packets = [], []
    for r in responses:
        key = f"{r['model']}|{r['item_id']}|{r.get('sample', 0)}"
        rag = normalize_rag(r.get("rag"))
        view = make_rag_view(rag)
        # 응답 해시에 출처도 넣는다 — 같은 답이라도 근거 자료가 바뀌면 다른 채점 요청이 된다
        body = str(r["response"]) + ("\n" + json.dumps(view, ensure_ascii=False, sort_keys=True) if view else "")
        rsha = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        pid = hashlib.sha256((fp["fingerprint"] + key + "|" + rsha).encode()).hexdigest()[:12]
        item = items[r["item_id"]]
        det = det_checks(item, r["response"], rag=rag).to_dict()
        case = {"pid": pid, "key": key, "model": r["model"], "item_id": r["item_id"], "response": r["response"],
                "response_sha": rsha, "meta": r.get("meta") or {}, "det": det, "judged": needs_judge(det),
                "rag": rag, "rag_view": view}
        cases.append(case)
        if case["judged"]:
            hyb = [h for h in det["hits"] if h["kind"] == "hybrid"]
            packets.append((pid, {"packet_id": pid, "kind": "judge", "system_file": "JUDGE_SYSTEM.txt",
                                  "user": build_judge_user(pid, item, r["response"], hyb, golden["sources"], view),
                                  "schema": judge_schema(grounding=view is not None)}))
    (run / "packets" / "judge").mkdir(parents=True, exist_ok=True)
    (run / "results" / "judge").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(golden_path, run / "golden.yaml")
    (run / "packets" / "judge" / "JUDGE_SYSTEM.txt").write_text(JUDGE_SYSTEM, encoding="utf-8")
    for pid, pk in packets:
        (run / "packets" / "judge" / f"{pid}.json").write_text(json.dumps(pk, ensure_ascii=False, indent=1), encoding="utf-8")
    _jl_write(run / "cases.jsonl", cases)
    partial = {m: s["missing_items"] for m, s in rep["models"].items() if s["missing_items"]}
    meta = {"run_format": RUN_FORMAT, "label": label, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "golden_path": "golden.yaml", "golden_source": _portable_path(golden_path),
            "allow_partial": allow_partial, "allow_flagged": allow_flagged, "missing_items": partial,
            "response_check": {"problems": rep["problems"], "warnings": rep["warnings"],
                               "prompt_formats": {m: s["prompt_formats"] for m, s in rep["models"].items()}},
            "golden_version": golden["golden_version"], "golden_review_status": golden["review_status"],
            "golden_sha256": golden["sha256"], "judge_model": judge_model, "judge_effort": judge_effort,
            "fingerprint": fp["fingerprint"], "judging_fp": fp["judging"], "scoring_fp": fp["scoring"],
            "fingerprint_parts": fp["parts"],
            "models": sorted({c["model"] for c in cases}), "n_cases": len(cases),
            "rag_models": sorted({c["model"] for c in cases if c.get("rag")}),
            "n_judge_packets": len(packets)}
    _atomic_write(run / "meta.json", json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def _load(run: Path) -> tuple[dict, list[dict], dict, dict]:
    meta = _read_json(run / "meta.json")
    golden = load_golden(_resolve_golden(meta, run))
    cases = _jl_read(run / "cases.jsonl")
    return meta, cases, golden, golden["by_id"]


def _save_meta(run: Path, meta: dict) -> None:
    _atomic_write(run / "meta.json", json.dumps(meta, ensure_ascii=False, indent=1))


def import_judge(run_dir: str | Path, *, rescore: bool = False) -> dict:
    """채점 결과를 검증해 붙이고, Hold 로 이어지는 채점기 지적은 반박 검증 요청을 만든다."""
    run = Path(run_dir)
    meta, cases, golden, items = _load(run)
    check_criteria(run, meta, "import", rescore)
    vdir = run / "packets" / "verify"
    vdir.mkdir(parents=True, exist_ok=True)
    for old in vdir.glob("*.json"):            # 지난 import 의 요청은 지운다 — 쓸모없는 검증 호출 방지 (감사 D15)
        old.unlink()
    (run / "results" / "verify").mkdir(parents=True, exist_ok=True)
    (vdir / "VERIFY_SYSTEM.txt").write_text(VERIFY_SYSTEM, encoding="utf-8")
    vschema = verify_schema()
    stats = {"ok": 0, "error": 0, "missing": 0, "skipped_format": 0, "verify_packets": 0}
    for c in cases:
        c["judgment"], c["judge_errors"], c["verify_id"] = None, [], None
        if not c["judged"]:
            stats["skipped_format"] += 1
            continue
        rp = run / "results" / "judge" / f"{c['pid']}.json"
        if not rp.exists():
            c["judge_errors"] = ["채점 결과 파일 없음"]
            stats["missing"] += 1
            continue
        try:
            obj = _read_json(rp)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            c["judge_errors"] = [f"JSON 해석 실패: {e}"]
            stats["error"] += 1
            continue
        item = items[c["item_id"]]
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        errs = validate_judgment(obj, c["pid"], item, c["response"], hids, c.get("rag_view"))
        if errs:
            c["judge_errors"] = errs
            stats["error"] += 1
            continue
        c["judgment"] = obj
        stats["ok"] += 1
        need = [f for f in build_findings(item, c["det"], obj) if f["needs_verify"]]
        if need:
            vid = verify_key(c["pid"], need)
            nd = no_downgrade_fids(item, need)
            hyb = [h for h in c["det"]["hits"] if h["kind"] == "hybrid"]
            pk = {"packet_id": vid, "case_pid": c["pid"], "kind": "verify", "system_file": "VERIFY_SYSTEM.txt",
                  "user": build_verify_user(vid, item, c["response"], need, nd, hyb, golden["sources"], c.get("rag_view")),
                  "schema": vschema,
                  "fids": [f["fid"] for f in need], "fid_gates": {f["fid"]: f["gate"] for f in need}, "no_downgrade": nd,
                  "conditional_ok": conditional_escalation_fids(item, need)}
            (vdir / f"{vid}.json").write_text(json.dumps(pk, ensure_ascii=False, indent=1), encoding="utf-8")
            c["verify_id"] = vid
            stats["verify_packets"] += 1
    _jl_write(run / "cases.jsonl", cases)
    meta["import_stats"] = stats
    _save_meta(run, meta)
    return stats


def _legacy_verify_path(run: Path, c: dict, fids: list[str], fid_gates: dict) -> Path | None:
    """회차 형식 1(예전 회차)의 반박 결과는 케이스 id 로 이름이 붙었다. 그때 요청의 지적 번호·Gate 가
    지금 다시 계산한 것과 똑같을 때만 쓴다 — 아니면 '검증 누락'으로 두어 Hold 가 남는다."""
    pk, res = run / "packets" / "verify" / f"{c['pid']}.json", run / "results" / "verify" / f"{c['pid']}.json"
    if not (pk.exists() and res.exists()):
        return None
    try:
        old = _read_json(pk)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if old.get("fids") == fids and old.get("fid_gates") == fid_gates:
        return res
    return None


def finalize(run_dir: str | Path, *, reference: str | None = None, teacher: str | None = None,
             pairs: list[tuple[str, str]] | None = None, rescore: bool = False) -> dict:
    """판정·지표·리포트. reference: 비교 기준 모델(상용 LLM 등 — 지식증류 때의 teacher 와 같은 자리)."""
    run = Path(run_dir)
    teacher = reference or teacher
    meta, cases, golden, items = _load(run)
    check_criteria(run, meta, "finalize", rescore)
    models = sorted({c["model"] for c in cases})
    # 이름이 틀린 Teacher·짝 비교가 말없이 빠지지 않게 먼저 확인한다 (감사 D18)
    if teacher and teacher not in models:
        raise ValueError(f"비교 기준 모델 '{teacher}' 가 회차에 없음 — 회차의 모델: {models}")
    pair_reqs = [(a, b, resolve_pair(a, b, models)) for a, b in (pairs or [])]
    for name in OUTPUTS:            # 중간에 멈추면 예전 리포트가 '정상'처럼 남지 않게 먼저 지운다 (감사 R03)
        (run / name).unlink(missing_ok=True)

    legacy = meta.get("run_format", 1) < 2
    vstats = {"keep": 0, "refute": 0, "downgrade": 0, "missing": 0, "error": 0}
    for c in cases:
        item = items[c["item_id"]]
        decisions = None
        # 반박 검증이 필요한 지적은 요청 파일이 아니라 판정식에서 다시 계산한다
        need_f = [f for f in build_findings(item, c["det"], c.get("judgment")) if f["needs_verify"]] if c.get("judgment") else []
        if need_f:
            fids = [f["fid"] for f in need_f]
            fg = {f["fid"]: f["gate"] for f in need_f}
            nd = no_downgrade_fids(item, need_f)
            cok = conditional_escalation_fids(item, need_f)
            vid = verify_key(c["pid"], need_f)
            vr, case_id = run / "results" / "verify" / f"{vid}.json", vid
            if not vr.exists() and legacy:
                old = _legacy_verify_path(run, c, fids, fg)
                if old is not None:
                    vr, case_id = old, c["pid"]
            if vr.exists():
                try:
                    obj = _read_json(vr)
                    verr = validate_verify(obj, case_id, fids, c["response"], fg, nd, cok)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:      # 깨진 파일 하나로 회차 전체가 멈추지 않게 (감사 D14)
                    obj, verr = None, [f"JSON 해석 실패: {e}"]
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
        c["result"] = verdict_of(item, c["det"], c.get("judgment"), c.get("judge_errors"), decisions,
                                 judge_expected=c.get("judged"))
    _jl_write(run / "scored.jsonl", cases)

    by_model: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_model[c["model"]].append(c)
    out: dict[str, Any] = {"meta": meta, "verify_stats": vstats, "models": {}}
    for m, cs in sorted(by_model.items()):
        out["models"][m] = model_metrics(cs, items)
    out["conditions"] = condition_summary(out["models"])
    if teacher:
        out["reference"] = teacher
        out["reference_compare"] = {m: reference_compare(cs, by_model[teacher], items)
                                    for m, cs in by_model.items() if m != teacher}
        tcond = {}
        for cond, info in out["conditions"].items():
            reps = [m for m in info["replicates"] if m in out["reference_compare"]]
            if len(reps) >= 2:
                tcond[cond] = {name: _mean_sd([out["reference_compare"][m][name]["value"] for m in reps])
                               for name in REF_TARGETS}
        out["reference_conditions"] = tcond
    out["pairs"], out["pair_groups"] = [], []
    for a, b, resolved in pair_reqs:
        group = [paired_compare(by_model[x], by_model[y], x, y) for x, y in resolved]
        out["pairs"] += group
        if len(group) > 1 or condition_of(resolved[0][0]) != resolved[0][0]:
            out["pair_groups"].append({"request": f"{a}:{b}", "labels": [[x, y] for x, y in resolved],
                                       "summary": pair_condition_summary(group)})
    # 생성 조건(입력 형식·디코딩)이 다른 모델끼리의 비교는 표시한다 — 조건 차이를 모델 차이로 읽지 않게 (노하우 M5)
    from .tools import gen_diff, model_gen_conditions
    gens = {m: model_gen_conditions([{"meta": c.get("meta")} for c in cs]) for m, cs in by_model.items()}
    for m in out["models"]:
        out["models"][m]["gen_conditions"] = gens[m]
    for p in out["pairs"]:
        p["gen_diff"] = gen_diff(gens[p["label_a"]], gens[p["label_b"]])
    for m, tc in (out.get("reference_compare") or {}).items():
        tc["gen_diff"] = gen_diff(gens[m], gens[teacher])
    out["validity"] = run_validity(cases, vstats)
    _save_meta(run, meta)
    (run / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    from .report import write_reports
    write_reports(run, out, cases, golden)
    return out


def cli_judge(run_dir: str | Path, *, workers: int = 3, claude_bin: str | None = None, phase: str = "judge",
              rescore: bool = False) -> dict:
    """claude -p 로 채점(또는 반박 검증) 요청을 처리한다. 채점 모델·effort 는 prep 때 정한 값을 쓴다."""
    from .judge import run_packets_with_cli
    run = Path(run_dir)
    meta, cases, golden, items = _load(run)
    check_criteria(run, meta, phase, rescore)
    _save_meta(run, meta)
    model, effort = meta["judge_model"], meta["judge_effort"]
    by_pid = {c["pid"]: c for c in cases}
    if phase == "judge":
        validator = judge_validator_factory(by_pid, items)
        return run_packets_with_cli(run / "packets" / "judge", run / "results" / "judge", validator, known=set(by_pid),
                                    model=model, effort=effort, workers=workers, claude_bin=claude_bin)
    vindex = {}
    for p in (run / "packets" / "verify").glob("*.json"):
        pk = _read_json(p)
        vindex[pk["packet_id"]] = {"fids": pk["fids"], "fid_gates": pk.get("fid_gates") or {},
                                   "no_downgrade": pk.get("no_downgrade") or [], "conditional_ok": pk.get("conditional_ok") or [],
                                   "case_pid": pk.get("case_pid") or pk["packet_id"]}
    return run_packets_with_cli(run / "packets" / "verify", run / "results" / "verify",
                                verify_validator_factory(vindex, by_pid), known=set(vindex), model=model, effort=effort,
                                workers=workers, claude_bin=claude_bin)
