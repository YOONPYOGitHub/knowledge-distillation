"""명령줄 진입점 — python -m silvercare_autoqa <명령>

  validate        골든셋 검증·구성 요약
  guardrail       입력 측 가드레일 시험 (어르신 발화의 위험 신호를 몇 개나 잡는지 — LLM 없음)
  generate        모델 응답 만들기 (kd-ui | openai | reference)
  prep            회차 만들기 (결정 룰 + 채점 요청 packet)
  judge           claude -p 로 채점 (기본 Sonnet 5.5 · effort high)
  import          채점 결과 검증·반입 + 반박 검증 요청 생성
  verify          claude -p 로 반박 검증
  finalize        판정·지표·리포트
  run             prep → judge → import → verify → finalize 한 번에 (claude -p 사용)
  mutants         변이 시험용 응답 만들기
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import golden as G


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def cmd_validate(a) -> int:
    g = G.load_golden(a.golden)
    errs = G.validate_golden(g)
    print(json.dumps(G.summary(g), ensure_ascii=False, indent=1))
    if errs:
        print("\n검증 실패:")
        for e in errs:
            print(" -", e)
        return 1
    # 모범 답안이 자기 문항의 결정 룰을 통과하는지 (골든셋 자체 점검)
    from .rules import det_checks
    bad = []
    for it in g["items"]:
        d = det_checks(it, it["reference_answer"])
        probs = [h.rule for h in d.hits if h.rule not in ("DET-OVERALARM",)]
        if probs or not d.kpi["within_3_sentences"] or not d.kpi["honorific_ok"]:
            bad.append((it["id"], probs, d.kpi["n_sentences"], d.kpi["honorific_ok"]))
    if bad:
        print("\n모범 답안 자체 점검 경고:")
        for b in bad:
            print(" -", b)
    print(f"\n골든셋 통과: {len(g['items'])}문항")
    return 0


def cmd_guardrail(a) -> int:
    from .rules import input_risk_flags
    g = G.load_golden(a.golden)
    if a.heldout:
        doc = json.loads(Path(a.heldout).read_text(encoding="utf-8"))
        cases = [(u["item_id"], "emergency" if u["kind"] != "routine" else "routine", u["kind"], u["utterance"])
                 for u in doc["utterances"]]
    else:
        cases = [(it["id"], it["risk"], it.get("variant"), it["utterance"]) for it in g["items"]]
    rows, tp, fn, fp, tn = [], 0, 0, 0, 0
    by_kind: dict[str, list[int]] = {}
    for iid, risk, kind, utt in cases:
        flags = input_risk_flags(utt)
        risky = risk == "emergency"
        if risky:
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][0] += bool(flags)
            by_kind[kind][1] += 1
        hit = bool(flags)
        if risky and hit:
            tp += 1
        elif risky:
            fn += 1
        elif hit:
            fp += 1
        else:
            tn += 1
        rows.append((iid, risk, kind, ",".join(flags) or "-", "✅" if hit == risky else "❌", utt[:40]))
    for r in rows:
        print(" | ".join(str(x) for x in r))
    print(f"\n응급 발화 감지 {tp}/{tp + fn} · 놓침 {fn} · 비응급 발화 오발동 {fp}/{fp + tn}")
    print("변형별 감지:", {k: f"{v[0]}/{v[1]}" for k, v in by_kind.items()})
    return 0


def cmd_generate(a) -> int:
    from . import responders as R
    g = G.load_golden(a.golden)
    items = [g["by_id"][i] for i in a.items.split(",")] if a.items else g["items"]
    if a.source == "reference":
        rows = R.reference_rows(g)
    elif a.source == "kd-ui":
        rows = R.kd_ui(items, a.base_url, a.run_id, a.model_ids.split(","), max_new_tokens=a.max_new_tokens)
    elif a.source == "openai":
        rows = R.openai_compat(items, a.base_url, a.model, a.label, max_tokens=a.max_new_tokens)
    elif a.source == "claude":
        rows = R.claude_teacher(items, a.label)
    else:
        raise SystemExit(f"알 수 없는 source {a.source}")
    R.write_jsonl(rows, a.out)
    print(f"응답 {len(rows)}건 → {a.out}")
    return 0


def cmd_prep(a) -> int:
    from .pipeline import load_responses, prep
    rows = []
    for p in a.responses:
        rows += load_responses(p)
    meta = prep(a.golden, rows, a.run, judge_model=a.judge_model, judge_effort=a.judge_effort, label=a.label)
    print(json.dumps({k: meta[k] for k in ("fingerprint", "models", "n_cases", "n_judge_packets")}, ensure_ascii=False))
    return 0


def cmd_judge(a) -> int:
    from .pipeline import cli_judge
    print(cli_judge(a.run, workers=a.workers, claude_bin=a.claude_bin, phase="judge"))
    return 0


def cmd_import(a) -> int:
    from .pipeline import import_judge
    print(import_judge(a.run))
    return 0


def cmd_verify(a) -> int:
    from .pipeline import cli_judge
    print(cli_judge(a.run, workers=a.workers, claude_bin=a.claude_bin, phase="verify"))
    return 0


def _pairs(s: str | None):
    if not s:
        return None
    return [tuple(p.split(":")) for p in s.split(",")]


def cmd_finalize(a) -> int:
    from .pipeline import finalize
    out = finalize(a.run, teacher=a.teacher, pairs=_pairs(a.pairs))
    for m, mm in out["models"].items():
        print(f"{m}: 최종 {mm['final']} · Pass/Fix/Hold {mm['verdicts']} · Gate {sum(mm['gate_counts'].values())}건 · 오류 {mm['n_error']}")
    print("유효성:", out["validity"])
    print("리포트:", Path(a.run) / "report.md")
    return 0 if out["validity"]["ok"] else 2


def cmd_run(a) -> int:
    from .pipeline import cli_judge, finalize, import_judge, load_responses, prep
    rows = []
    for p in a.responses:
        rows += load_responses(p)
    prep(a.golden, rows, a.run, judge_model=a.judge_model, judge_effort=a.judge_effort, label=a.label)
    print("채점:", cli_judge(a.run, workers=a.workers, claude_bin=a.claude_bin, phase="judge"))
    print("반입:", import_judge(a.run))
    print("반박 검증:", cli_judge(a.run, workers=a.workers, claude_bin=a.claude_bin, phase="verify"))
    return cmd_finalize(a)


def cmd_mutants(a) -> int:
    from .mutate import sample_mutants, subtle_requests
    from .responders import reference_rows, write_jsonl
    g = G.load_golden(a.golden)
    muts = sample_mutants(g, seed=a.seed)
    rows = reference_rows(g, "reference")
    for m in muts:
        rows.append({"model": m["mutant"], "item_id": m["item_id"], "response": m["response"],
                     "meta": {"expect": m["expect"], "codes": m["codes"], "note": m["note"]}})
    write_jsonl(rows, a.out)
    Path(a.subtle_out).write_text(json.dumps(subtle_requests(g), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"모범 답안 {len(g['items'])} + 결정적 변이 {len(muts)} → {a.out}")
    print(f"미묘한 변이 요청 {len(subtle_requests(g))}건 → {a.subtle_out} (LLM 이 응답을 채운다)")
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8()
    p = argparse.ArgumentParser(prog="silvercare_autoqa", description="실버케어 지식증류 모델 자동검수")
    p.add_argument("--golden", default=str(G.DEFAULT_GOLDEN))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    s = sub.add_parser("guardrail")
    s.add_argument("--heldout", help="처음 보는 발화 시험셋 (예: silvercare_autoqa/data/golden/guardrail_heldout_v0.json)")
    s.set_defaults(fn=cmd_guardrail)

    s = sub.add_parser("generate")
    s.add_argument("--source", choices=["reference", "kd-ui", "openai", "claude"], required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--items")
    s.add_argument("--base-url", default="http://127.0.0.1:8000")
    s.add_argument("--run-id")
    s.add_argument("--model-ids", default="teacher_ft,student_kd,student_ft,student_base")
    s.add_argument("--model", default="local")
    s.add_argument("--label", default="model")
    s.add_argument("--max-new-tokens", type=int, default=160)
    s.set_defaults(fn=cmd_generate)

    for name, fn in (("prep", cmd_prep), ("run", cmd_run)):
        s = sub.add_parser(name)
        s.add_argument("--responses", nargs="+", required=True)
        s.add_argument("--run", required=True)
        s.add_argument("--label", default="")
        s.add_argument("--judge-model", default="claude-sonnet-5-5")
        s.add_argument("--judge-effort", default="high")
        if name == "run":
            s.add_argument("--workers", type=int, default=3)
            s.add_argument("--claude-bin")
            s.add_argument("--teacher")
            s.add_argument("--pairs")
        s.set_defaults(fn=fn)

    for name, fn in (("judge", cmd_judge), ("verify", cmd_verify)):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--workers", type=int, default=3)
        s.add_argument("--claude-bin")
        s.set_defaults(fn=fn)

    s = sub.add_parser("import")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("finalize")
    s.add_argument("--run", required=True)
    s.add_argument("--teacher", help="Teacher 모델 이름 (예: teacher_ft) — 결론 일치율·안전 판단 보존율 계산")
    s.add_argument("--pairs", help="짝 비교 목록 (예: student_kd:student_ft)")
    s.set_defaults(fn=cmd_finalize)

    s = sub.add_parser("mutants")
    s.add_argument("--out", required=True)
    s.add_argument("--subtle-out", required=True)
    s.add_argument("--seed", type=int, default=20261108)
    s.set_defaults(fn=cmd_mutants)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
