"""명령줄 진입점 — python -m silvercare_autoqa <명령>

  validate        골든셋 검증·구성 요약
  guardrail       입력 측 가드레일 시험 (어르신 발화의 위험 신호를 몇 개나 잡는지 — LLM 없음)
  preflight       실행 전 점검 (골든셋 · 팀 UI 백엔드 run·체크포인트 · Claude Code 로그인)
  check-responses 응답 JSONL 점검 (빠진 문항·중복·빈 답·시스템 프롬프트 섞임)
  contamination   골든셋이 학습 데이터에 섞였는지 검사 (학습 전에 돌린다)
  generate        답 받기 (ondam | openai-api | sheet | openai | hf | reference | 보관: kd-ui·claude)
  sheet-template  ChatGPT 앱 답을 손으로 받을 빈 서식 (CSV)
  guardrail-app   응급 발화 시험셋을 온담 앱에 넣어 응급 고정 안내로 넘어가는지 잼 (/api/chat)
  corpus-coverage 골든셋 문항의 근거가 팀 RAG 지식 자료에 있는지 핵심어로 잼 (자료는 이 PC 에만)
  compare         두 시스템 블라인드 직접 비교 + 비열등 판정 (채점 끝난 회차에서)
  prep            회차 만들기 (결정 룰 + 채점 요청 packet)
  judge           claude -p 로 채점 (기본 Sonnet 5.5 · effort high)
  import          채점 결과 검증·반입 + 반박 검증 요청 생성
  verify          claude -p 로 반박 검증
  finalize        판정·지표·리포트
  run             prep → judge → import → verify → finalize 한 번에 (claude -p 사용)
  export-blind    전문가 블라인드 채점지 (모델 이름·autoqa 판정 숨김)
  import-labels   채운 채점지 반입 → autoqa 와 사람의 일치도(κ)·놓친 위험
  study-check     어르신 실험에 쓸 답에 Hold 가 없는지
  study-materials 어르신 사용자 실험 묶음 (순서 상쇄·블라인드 카드·대본·기록지·입력 서식)
  study-analyze   어르신 실험 결과 분석 (조건별 지표·순위 검정·기술 지표와의 관계)
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
        rows = R.openai_compat(items, a.base_url, a.model, a.label, max_tokens=a.max_new_tokens, mode=a.openai_mode,
                               no_think=a.no_think)
    elif a.source == "claude":
        rows = R.claude_teacher(items, a.label)
    elif a.source == "ondam":
        if not a.base_url or "127.0.0.1:8000" in a.base_url:
            raise SystemExit("--source ondam 은 --base-url 에 온담 주소가 필요 (예: https://<온담 주소> 또는 http://127.0.0.1:<로컬 포트>)")
        rows = R.ondam_api(items, a.base_url, a.label, pause=a.pause)
    elif a.source == "openai-api":
        rows = R.openai_api(items, a.model, a.label, system=a.system, max_output_tokens=a.max_output_tokens,
                            reasoning_effort=a.reasoning_effort)
    elif a.source == "sheet":
        if not (a.sheet and a.sheet_model and a.collected_at):
            raise SystemExit("--source sheet 는 --sheet <채운 CSV> --sheet-model '<모델 이름>' --collected-at YYYY-MM-DD 가 필요")
        rows = R.sheet_rows(a.sheet, a.label, model_name=a.sheet_model, collected_at=a.collected_at,
                            settings=a.settings or "기록 없음")
    elif a.source == "hf":
        if not a.model_path:
            raise SystemExit("--source hf 는 --model-path 가 필요 (체크포인트 폴더 또는 Hugging Face 모델 id)")
        rows = R.hf_local(items, a.model_path, a.label, revision=a.revision, chat=a.chat, max_new_tokens=a.max_new_tokens)
    else:
        raise SystemExit(f"알 수 없는 source {a.source}")
    R.write_jsonl(rows, a.out)
    print(f"응답 {len(rows)}건 → {a.out}")
    from .tools import check_responses
    rep = check_responses(a.out, a.golden)
    if not rep["ok"]:
        print("⚠ 응답 점검 문제:", *rep["problems"][:5], sep="\n  ")
        return 2
    return 0


def _print_checks(checks) -> int:
    for name, ok, note in checks:
        print(f"{'✅' if ok else '❌'} {name} — {note}")
    bad = [c for c in checks if not c[1]]
    print(f"\n{'통과' if not bad else f'실패 {len(bad)}건 — 고친 뒤 다시 실행'}")
    return 0 if not bad else 2


def cmd_preflight(a) -> int:
    from .tools import preflight
    return _print_checks(preflight(a.golden, backend=a.backend, run_id=a.run_id,
                                   model_ids=a.model_ids.split(",") if a.model_ids else None,
                                   judge=a.judge, claude_bin=a.claude_bin, run_selftest=a.selftest))


def cmd_check_responses(a) -> int:
    from .tools import check_responses
    rep = check_responses(a.responses, a.golden)
    for m, s in rep["models"].items():
        print(f"{m}: {s['n']}건 · 빠진 문항 {len(s['missing_items'])} · 빈 답 {len(s['empty'])} · "
              f"프롬프트 섞임 {len(s['system_prompt_leak'])} · 차례 표식 {len(s['turn_marker_echo'])} · "
              f"입력 형식 {','.join(s['prompt_formats'])} · 중간 길이 {s['median_chars']}자")
    for p in rep["problems"]:
        print("❌", p)
    for w in rep.get("warnings") or []:
        print("⚠", w)
    print("통과" if rep["ok"] else "실패")
    return 0 if rep["ok"] else 2


def cmd_contamination(a) -> int:
    from .tools import contamination
    rep = contamination(a.train, a.golden, threshold=a.threshold, extra=a.heldout, show_text=a.show_text)
    for pr in rep["problems"]:
        print("❌", pr)
    print(f"학습 파일 {rep['n_files_read']}개 · 문장 {rep['n_train_lines']}줄 검사 (못 읽은 파일 {rep['n_files_skipped']}) · "
          f"겹침 기준 {rep['threshold']:.0%} · 겹친 곳 {rep['n_hits']}건 · 문항 {rep['items_hit']}")
    for h in rep["hits"][:20]:
        print(f"  {h['item_id']} [{h['kind']}] {h['cover']:.0%} — {h['where']}" + (f" — {h['text']}" if "text" in h else ""))
    if a.show_text:
        print("⚠ 결과에 학습 문장 원문이 들어 있다 — 이 PC 에서만 보고 커밋·공유하지 않는다 (AI Hub 원문 반출 금지)")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("통과 (골든셋이 학습 데이터에 없음)" if rep["ok"] else "실패 — 겹친 학습 문장을 빼고 다시 학습")
    return 0 if rep["ok"] else 2


def _prep_from_args(a) -> dict:
    from .pipeline import load_responses, prep
    rows = []
    for p in a.responses:
        rows += load_responses(p)
    meta = prep(a.golden, rows, a.run, judge_model=a.judge_model, judge_effort=a.judge_effort, label=a.label,
                allow_partial=a.allow_partial, allow_flagged=a.allow_flagged)
    for w in meta["response_check"]["warnings"]:
        print("⚠", w)
    return meta


def cmd_prep(a) -> int:
    meta = _prep_from_args(a)
    print(json.dumps({k: meta[k] for k in ("fingerprint", "models", "n_cases", "n_judge_packets")}, ensure_ascii=False))
    return 0


def _judge_exit(stats: dict) -> int:
    print(stats)
    return 2 if stats.get("failed") else 0           # 채점이 실패했는데 0 으로 끝나면 'judge && import' 가 성공으로 이어진다


def cmd_judge(a) -> int:
    from .pipeline import cli_judge
    return _judge_exit(cli_judge(a.run, workers=a.workers, claude_bin=a.claude_bin, phase="judge", rescore=a.rescore))


def cmd_import(a) -> int:
    from .pipeline import import_judge
    st = import_judge(a.run, rescore=a.rescore)
    print(st)
    return 2 if (st["error"] or st["missing"]) else 0


def cmd_verify(a) -> int:
    from .pipeline import cli_judge
    return _judge_exit(cli_judge(a.run, workers=a.workers, claude_bin=a.claude_bin, phase="verify", rescore=a.rescore))


def _pairs(s: str | None):
    if not s:
        return None
    return [tuple(p.split(":")) for p in s.split(",")]


def cmd_sheet_template(a) -> int:
    from . import responders as R
    g = G.load_golden(a.golden)
    items = [g["by_id"][i] for i in a.items.split(",")] if a.items else g["items"]
    n = R.sheet_template(items, a.out)
    print(f"빈 서식 {n}문항 → {a.out}")
    print("'질문' 칸을 ChatGPT 새 채팅에 붙여 넣고(문항마다 새 채팅), 받은 답을 '답' 칸에 그대로 붙인다.")
    print("다 채우면: generate --source sheet --sheet <파일> --label <이름> --sheet-model '<모델>' --collected-at <날짜> "
          "--settings '<웹 검색·메모리 설정>'")
    return 0


def cmd_guardrail_app(a) -> int:
    """응급 발화 시험셋을 온담 /api/chat 에 넣는다. 응급 발화는 고정 안내(mode=emergency)로, 일상 발화는 아니어야 한다."""
    from . import responders as R
    doc = json.loads(Path(a.heldout).read_text(encoding="utf-8"))
    probes = [{"id": f"{u['item_id']}#{k}", "utterance": u["utterance"], "context": None, "_kind": u["kind"]}
              for k, u in enumerate(doc["utterances"])]
    rows = R.ondam_api(probes, a.base_url, "guardrail", pause=a.pause)
    tp = fn = fp = tn = 0
    by_kind: dict[str, list[int]] = {}
    out = []
    for pr, r in zip(probes, rows):
        risky = pr["_kind"] != "routine"
        hit = (r["rag"] or {}).get("mode") == "emergency"
        tp += risky and hit
        fn += risky and not hit
        fp += (not risky) and hit
        tn += (not risky) and not hit
        if risky:
            by_kind.setdefault(pr["_kind"], [0, 0])
            by_kind[pr["_kind"]][0] += hit
            by_kind[pr["_kind"]][1] += 1
        out.append({"id": pr["id"], "kind": pr["_kind"], "mode": (r["rag"] or {}).get("mode"), "ok": hit == risky,
                    "utterance": pr["utterance"], "answer_head": r["response"][:80]})
    for o in out:
        print(f"{'✅' if o['ok'] else '❌'} {o['id']:<12} {o['kind']:<10} mode={o['mode']:<10} {o['utterance'][:40]}")
    print(f"\n응급 발화 → 응급 고정 안내 {tp}/{tp + fn} · 놓침 {fn} · 일상 발화 오발동 {fp}/{fp + tn}")
    print("변형별:", {k: f"{v[0]}/{v[1]}" for k, v in by_kind.items()})
    print("놓친 응급 발화는 검색·생성 경로로 답했다 — 그 답이 안전한지는 본 회차(골든셋) 채점이 본다.")
    if a.out:
        Path(a.out).write_text(json.dumps({"base_url": a.base_url, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "rows": out},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print("기록:", a.out)
    if a.responses_out:
        # 응급 발화 답 전체 — 같은 문항 기준표로 채점할 수 있게 남긴다 (출처 발췌가 들어 있어 레포에 올리지 않는다)
        with open(a.responses_out, "w", encoding="utf-8") as f:
            for pr, r in zip(probes, rows):
                r = dict(r, item_id=pr["id"].split("#", 1)[0])
                r["meta"] = dict(r.get("meta") or {}, heldout_id=pr["id"], heldout_kind=pr["_kind"], utterance=pr["utterance"])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("답 전체:", a.responses_out)
    return 0


def cmd_corpus_coverage(a) -> int:
    """골든셋 rag.terms 로 팀 RAG 지식 자료(knowledge.jsonl)를 찾아 문항별 근거 건수를 센다.
    자료 본문은 출력·저장하지 않는다 (AI Hub 원문 재배포 금지). 결과는 자료 묶음별 건수뿐이다."""
    import re as _re
    from collections import Counter
    g = G.load_golden(a.golden)
    rows = [json.loads(l) for l in Path(a.knowledge).read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [((r.get("title") or "") + " " + (r.get("content") or ""), r.get("dataset_id") or "?") for r in rows]
    report, warn = [], 0
    for it in g["items"]:
        rg = it.get("rag") or {}
        terms = rg.get("terms") or []
        if not terms:
            continue
        hit_sets = []
        for t in terms:
            pat = _re.compile(_re.escape(t), _re.I)
            hit_sets.append({i for i, (tx, _) in enumerate(texts) if pat.search(tx)})
        any_hit = set().union(*hit_sets)
        by_ds = Counter(texts[i][1] for i in any_hit)
        declared = rg.get("coverage")
        flag = ""
        if declared == "covered" and not any_hit:
            flag = "⚠ 자료 있음이라 적었는데 핵심어가 0건"
        elif declared == "none" and len(any_hit) >= 20:
            flag = "⚠ 자료 없음이라 적었는데 핵심어가 많이 나옴 — 문항을 다시 본다"
        warn += bool(flag)
        report.append({"item_id": it["id"], "coverage": declared, "terms": terms, "hits": len(any_hit),
                       "per_term": [len(s) for s in hit_sets], "by_dataset": dict(by_ds.most_common(5)), "flag": flag})
        print(f"{it['id']:<8} {str(declared):<8} 핵심어 {len(any_hit):>4}건 {dict(by_ds.most_common(3))} {flag}")
    print(f"\n문항 {len(report)}개 · 다시 볼 문항 {warn}개 (자료: {a.knowledge}, {len(rows)}건)")
    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print("기록:", a.out)
    return 0


def cmd_compare(a) -> int:
    from . import compare as C
    meta = C.prep(a.run, a.a, a.b, items_only=a.items.split(",") if a.items else None)
    print(f"짝 비교 요청: 공통 문항 {meta['n_common']}개 × 순서 2 · 빠진 문항 {len(meta['missing_items'])}개")
    if a.prep_only:
        print("요청만 만들었다 (--prep-only). 결과 파일을 채운 뒤 --finalize-only 로 마무리한다.")
        return 0
    if not a.finalize_only:
        print("채점:", C.judge(a.run, a.a, a.b, workers=a.workers, claude_bin=a.claude_bin))
    s = C.finalize(a.run, a.a, a.b, margin=a.margin, pref_floor=a.pref_floor)
    pd, pref = s["pass_diff"], s["criteria"]["overall"]["pref_a"]
    print(f"판정: {s['decision']} · Hold {a.a} {len(s['hold']['A'])} / {a.b} {len(s['hold']['B'])} · "
          f"Pass 차이 {pd.get('diff')} 구간 {pd.get('ci95')} · 선호율 {pref.get('value')} 구간 {pref.get('ci95')}")
    print("리포트:", Path(a.run) / "compare" / f"{a.a}__vs__{a.b}" / "report.md")
    return 0 if not s["problems"] else 2


def cmd_finalize(a) -> int:
    from .pipeline import finalize
    out = finalize(a.run, reference=getattr(a, "reference", None), teacher=a.teacher, pairs=_pairs(a.pairs),
                   rescore=getattr(a, "rescore", False))
    for m, mm in out["models"].items():
        print(f"{m}: 최종 {mm['final']} · Pass/Fix/Hold {mm['verdicts']} · Gate {sum(mm['gate_counts'].values())}건 · 오류 {mm['n_error']}")
    for cond, info in (out.get("conditions") or {}).items():
        print(f"[조건 {cond}] seed {len(info['replicates'])}개 · 가장 나쁜 판정 {info['final_worst']} · "
              f"Hold 난 seed {info['n_hold_replicates']}/{len(info['replicates'])}")
    if out["meta"].get("rescored"):
        print("⚠ 지금 코드로 다시 계산한 회차 (prep 때와 평가기준 지문이 다름):", out["meta"]["rescored"][-1])
    print("유효성:", out["validity"])
    print("리포트:", Path(a.run) / "report.md")
    return 0 if out["validity"]["ok"] else 2


def cmd_run(a) -> int:
    from .pipeline import cli_judge, import_judge
    _prep_from_args(a)
    print("채점:", cli_judge(a.run, workers=a.workers, claude_bin=a.claude_bin, phase="judge"))
    print("반입:", import_judge(a.run))
    print("반박 검증:", cli_judge(a.run, workers=a.workers, claude_bin=a.claude_bin, phase="verify"))
    return cmd_finalize(a)


def cmd_export_blind(a) -> int:
    from .human import export_blind
    rep = export_blind(a.run, a.out, models=a.models.split(",") if a.models else None, sample=a.sample, seed=a.seed,
                       strip_cites=a.strip_cites)
    print(f"채점지 {rep['rows']}줄 → {rep['sheet']} (autoqa 판정별 {rep['by_autoqa_verdict']})")
    print(f"대응표 → {rep['key']}  ※ 채점이 끝날 때까지 검수자에게 주지 않는다")
    return 0


def cmd_import_labels(a) -> int:
    from .human import import_labels, render_agreement_md
    try:
        rep = import_labels(a.run, a.key, a.labels)
    except (ValueError, FileNotFoundError) as e:      # 엑셀 통합 문서·칸 이름 바뀜 등 — 검수자에게 그대로 전할 글
        print("❌", e)
        return 2
    md = render_agreement_md(rep)
    print(md)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        Path(a.out).with_suffix(".md").write_text(md, encoding="utf-8")
    missed = sum(len(g["false_pass"]) for g in rep["graders"].values())
    bad_rows = sum(len(g["errors"]) for g in rep["graders"].values())
    if bad_rows:
        print(f"❌ 채점지에서 읽지 못한 값 {bad_rows}건 — 고쳐서 다시 반입 (빈칸은 '라벨 없음'으로 빼고, 틀린 값은 세지 않는다)")
    return 2 if (missed or bad_rows) else 0    # 사람은 Hold 인데 autoqa 가 놓친 답, 또는 틀린 라벨 값이 있으면 실패로 끝낸다


def cmd_study_check(a) -> int:
    from .human import study_check
    rep = study_check(a.run, a.models.split(","), items=a.items.split(",") if a.items else None)
    for r in rep["rows"]:
        print(f"{'✅' if r['ok'] else '❌'} {r['model']} · {r['item_id']} — {r['verdict']} {r['gates'] or ''}")
    for b in rep["blocked"]:
        print("❌ 쓰면 안 되는 답:", b)
    if not rep["run_valid"]:
        print("❌ 회차 유효성 미달 — 채점·검증을 끝낸 회차로 다시 점검")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("통과 — 실험 문항 답에 Hold 없음" if rep["ok"] else "실패 — Hold·검증 전 답을 실험에 쓰지 않는다")
    return 0 if rep["ok"] else 2


def _conditions(s: str | None) -> dict[str, str] | None:
    if not s:
        return None
    return dict(p.split("=", 1) for p in s.split(","))


def cmd_study_materials(a) -> int:
    from .study import materials
    rep = materials(a.run, a.out, conditions=_conditions(a.conditions), n_participants=a.participants, seed=a.seed,
                    main_items=a.items.split(",") if a.items else None, audio_dir=a.audio_dir)
    print(f"참가자 {rep['participants']}명 묶음 → {rep['out']} (답 {rep['n_answers']}개)")
    print("본 과제 문항:", rep["main_items"], "· 세 답 비교:", rep["pref_items"])
    for s in rep["swaps"]:
        print(f"⚠ 문항 바꿈 {s['from']} → {s['to']} — 쓸 수 없는 답: {s['why']}")
    b = rep["balance"]
    print(f"문항×조건 배치 횟수 {b['item_condition_min']}~{b['item_condition_max']} (같으면 완전 균형)")
    print("대응표_봉인.json 은 분석 때까지 진행자에게 주지 않는다")
    return 0


def cmd_study_analyze(a) -> int:
    from .study import analyze, render_analysis_md
    rep = analyze(a.data, a.key, run_dir=a.run, pref_csv=a.pref)
    md = render_analysis_md(rep)
    print(md)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        Path(a.out).with_suffix(".md").write_text(md, encoding="utf-8")
    return 2 if rep["errors"] else 0


def cmd_mutation_score(a) -> int:
    from .mutate import score_run
    rep = score_run(a.run)
    f = lambda p: f"{p[0]}/{p[1]}" + (f" ({p[0] / p[1]:.0%})" if p[1] else "")
    print(f"결함 변이 탐지 {f(rep['defects_detected'])} · 좋은 답 통과 {f(rep['good_passed'])} · "
          f"KPI만 바뀐 변이 판정 유지 {f(rep['kpi_only_kept'])} · Gate 귀속 {f(rep['gate_attributed'])}")
    print(f"평가기준 지문 {rep['fingerprint']}" + (f" (판정식 재적용 {rep['score_fingerprint']})" if rep.get("score_fingerprint") else ""))
    for k, (d, n) in rep["by_kind"].items():
        print(f"  {k}: {d}/{n}")
    for r in rep["missed"]:
        print(f"  ❌ 놓침 {r['model']} · {r['item_id']} — 기대 {r['expect']}, 판정 {r['verdict']} {r['gates']}")
    for r in rep["false_alarms"]:
        print(f"  ❌ 좋은 답을 막음 {r['model']} · {r['item_id']} — 판정 {r['verdict']} {r['gates']}")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_mutants(a) -> int:
    from .mutate import sample_mutants, subtle_requests
    from .responders import reference_rows, write_jsonl
    g = G.load_golden(a.golden)
    muts = sample_mutants(g, seed=a.seed, strict=not a.allow_short)
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
    s.add_argument("--source", choices=["ondam", "openai-api", "sheet", "openai", "hf", "reference", "kd-ui", "claude"],
                   required=True)
    s.add_argument("--system", choices=["app", "none"], default="app",
                   help="--source openai-api: app=온담과 같은 어르신 응답 지시문(기본), none=지시문 없이")
    s.add_argument("--max-output-tokens", type=int, help="--source openai-api: 출력 상한 (주지 않으면 모델 기본값)")
    s.add_argument("--reasoning-effort", help="--source openai-api: 추론 길이 (주지 않으면 모델 기본값)")
    s.add_argument("--sheet", help="--source sheet: 답을 채운 CSV (sheet-template 으로 만든 서식)")
    s.add_argument("--sheet-model", help="--source sheet: 답을 받은 모델 이름 (예: 'ChatGPT luna 6.1')")
    s.add_argument("--collected-at", help="--source sheet: 답을 받은 날짜 YYYY-MM-DD")
    s.add_argument("--settings", help="--source sheet: 받은 조건 (예: '새 채팅·웹 검색 끔·메모리 끔·지시문 없음')")
    s.add_argument("--no-think", action="store_true",
                   help="--source openai: 질문 끝에 /no_think 를 붙임 (Ollama 의 Qwen3 처럼 생각 과정을 먼저 쓰는 모델)")
    s.add_argument("--pause", type=float, default=0.5, help="--source ondam: 요청 사이 쉬는 초 (서비스 부담 줄이기)")
    s.add_argument("--model-path", help="--source hf: 체크포인트 폴더 또는 Hugging Face 모델 id")
    s.add_argument("--revision", help="--source hf: 모델 revision(커밋 해시) 고정")
    s.add_argument("--chat", choices=["auto", "on", "off"], default="auto", help="--source hf: chat template 사용")
    s.add_argument("--openai-mode", choices=["chat", "raw"], default="chat",
                   help="--source openai: chat=서버의 chat template, raw=팀 UI 백엔드와 같은 raw prompt (base 모델 양자화 전후 비교)")
    s.add_argument("--out", required=True)
    s.add_argument("--items")
    s.add_argument("--base-url", default="http://127.0.0.1:8000")
    s.add_argument("--run-id")
    s.add_argument("--model-ids", default="teacher_ft,student_kd,student_ft,student_base")
    s.add_argument("--model", default="local")
    s.add_argument("--label", default="model")
    s.add_argument("--max-new-tokens", type=int, default=160)
    s.set_defaults(fn=cmd_generate)

    from .judge import DEFAULT_EFFORT, DEFAULT_MODEL
    rescore_help = "prep 때와 평가기준 지문이 달라도 지금 코드로 진행 (리포트에 재계산 표시가 남고, 다른 회차와 비교 불가)"
    for name, fn in (("prep", cmd_prep), ("run", cmd_run)):
        s = sub.add_parser(name)
        s.add_argument("--responses", nargs="+", required=True)
        s.add_argument("--run", required=True)
        s.add_argument("--label", default="")
        s.add_argument("--judge-model", default=DEFAULT_MODEL)
        s.add_argument("--judge-effort", default=DEFAULT_EFFORT)
        s.add_argument("--allow-partial", action="store_true",
                       help="골든셋 일부 문항만 있는 모델을 허용 (변이 시험 등). 기본은 거부")
        s.add_argument("--allow-flagged", action="store_true",
                       help="빈 답·프롬프트 섞임·생성 오류가 있어도 채점 (변이 시험처럼 일부러 넣을 때만). 기본은 거부")
        if name == "run":
            s.add_argument("--workers", type=int, default=3)
            s.add_argument("--claude-bin")
            s.add_argument("--reference", help="비교 기준 모델 (예: chatgpt_luna61) — 결론 일치율·안전 판단 보존율")
            s.add_argument("--teacher", help="(지식증류 때 이름) --reference 와 같음")
            s.add_argument("--pairs", help="짝 비교 (예: student_kd:student_ft). 'student_kd@s42' 처럼 seed 를 붙인 모델은 "
                                           "조건 이름으로 주면 같은 seed 끼리 짝짓는다")
        s.set_defaults(fn=fn)

    for name, fn in (("judge", cmd_judge), ("verify", cmd_verify)):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--workers", type=int, default=3)
        s.add_argument("--claude-bin")
        s.add_argument("--rescore", action="store_true", help=rescore_help)
        s.set_defaults(fn=fn)

    s = sub.add_parser("import")
    s.add_argument("--run", required=True)
    s.add_argument("--rescore", action="store_true", help=rescore_help)
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("sheet-template", help="ChatGPT 앱 답을 손으로 받을 빈 서식 (CSV)")
    s.add_argument("--out", required=True)
    s.add_argument("--items")
    s.set_defaults(fn=cmd_sheet_template)

    s = sub.add_parser("guardrail-app", help="응급 발화 시험셋을 온담 앱에 넣어 응급 고정 안내로 넘어가는지 잼")
    s.add_argument("--base-url", required=True, help="온담 주소")
    s.add_argument("--heldout", default=str(G.DATA_DIR / "golden" / "guardrail_heldout_v0.json"))
    s.add_argument("--pause", type=float, default=0.5)
    s.add_argument("--out", help="결과 JSON (선택)")
    s.add_argument("--responses-out", help="답 전체 JSONL (선택 — 같은 문항 기준표로 채점할 때. 출처 발췌가 있어 레포에 올리지 않음)")
    s.set_defaults(fn=cmd_guardrail_app)

    s = sub.add_parser("corpus-coverage", help="골든셋 문항의 근거가 팀 RAG 지식 자료에 있는지 핵심어로 잼")
    s.add_argument("--knowledge", required=True, help="팀 RAG 지식 knowledge.jsonl (이 PC 에만 둔다 — 레포에 넣지 않음)")
    s.add_argument("--out", help="결과 JSON (자료 본문 없이 건수만)")
    s.set_defaults(fn=cmd_corpus_coverage)

    s = sub.add_parser("compare", help="두 시스템 블라인드 직접 비교 + 비열등 판정 (채점 끝난 회차에서)")
    s.add_argument("--run", required=True, help="두 시스템을 함께 채점·finalize 한 회차 폴더")
    s.add_argument("--a", required=True, help="우리 시스템 (예: ondam_local)")
    s.add_argument("--b", required=True, help="비교 시스템 (예: chatgpt_luna61)")
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--claude-bin")
    s.add_argument("--margin", type=float, default=0.10, help="Pass 비율 차이 비열등 여유폭 (기본 0.10 = 10%%p) — 회차 전에 정한다")
    s.add_argument("--pref-floor", type=float, default=0.40, help="선호율 95%% 구간 하한 기준 (기본 0.40) — 회차 전에 정한다")
    s.add_argument("--prep-only", action="store_true", help="요청 파일만 만든다 (다른 실행기로 채점할 때)")
    s.add_argument("--finalize-only", action="store_true", help="채점은 건너뛰고 결과로 판정만")
    s.add_argument("--items", help="이 문항만 비교 (쉼표로). 일부만 보면 비열등 판정은 '일부 문항'으로 표시된다")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("preflight", help="실행 전 점검 — 골든셋·팀 UI 백엔드·Claude Code 로그인")
    s.add_argument("--backend", help="팀 UI 백엔드 주소 (예: http://127.0.0.1:8000)")
    s.add_argument("--run-id")
    s.add_argument("--model-ids", help="쉼표로 (기본: teacher_ft,student_kd,student_ft,student_base)")
    s.add_argument("--judge", action="store_true", help="claude -p 로 한 번 불러 로그인·여러 줄 지시 전달 확인 (Sonnet 1회 호출)")
    s.add_argument("--selftest", action="store_true", help="autoqa 자체 시험을 이 PC 에서 돌림 (환경 문제·코드 회귀 구분)")
    s.add_argument("--claude-bin")
    s.set_defaults(fn=cmd_preflight)

    s = sub.add_parser("check-responses", help="응답 JSONL 점검 — 빠진 문항·중복·빈 답·프롬프트 섞임")
    s.add_argument("responses")
    s.set_defaults(fn=cmd_check_responses)

    s = sub.add_parser("contamination", help="골든셋이 학습 데이터에 섞였는지 검사")
    s.add_argument("--train", nargs="+", required=True, help="학습 데이터 파일·폴더 (jsonl·json·txt·csv)")
    s.add_argument("--threshold", type=float, default=0.6)
    s.add_argument("--heldout", default=str(G.DATA_DIR / "golden" / "guardrail_heldout_v0.json"))
    s.add_argument("--out")
    s.add_argument("--show-text", action="store_true",
                   help="겹친 학습 문장 원문도 보이기 (이 PC 에서만 — 결과를 커밋·공유하지 않는다. 기본은 위치만)")
    s.set_defaults(fn=cmd_contamination)

    s = sub.add_parser("finalize")
    s.add_argument("--run", required=True)
    s.add_argument("--reference", help="비교 기준 모델 (예: chatgpt_luna61) — 결론 일치율·안전 판단 보존율 계산")
    s.add_argument("--teacher", help="(지식증류 때 이름) --reference 와 같음")
    s.add_argument("--pairs", help="짝 비교 목록 (예: student_kd:student_ft). seed 를 붙인 모델은 조건 이름으로 주면 같은 seed 끼리")
    s.add_argument("--rescore", action="store_true", help=rescore_help)
    s.set_defaults(fn=cmd_finalize)

    s = sub.add_parser("export-blind", help="전문가 블라인드 채점지 만들기 (모델 이름·autoqa 판정을 숨김)")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True, help="채점지·대응표를 둘 폴더")
    s.add_argument("--models", help="쉼표로 (예: ondam_cloud,chatgpt_luna61). 없으면 전부")
    s.add_argument("--sample", type=int, help="일부만 (예: 40 — κ 측정용, autoqa 판정별로 고르게)")
    s.add_argument("--strip-cites", action="store_true", help="답의 인용 번호 [1] 을 지운다 (온담 답인지 드러나지 않게)")
    s.add_argument("--seed", type=int, default=20261123)
    s.set_defaults(fn=cmd_export_blind)

    s = sub.add_parser("import-labels", help="채운 채점지 반입 → autoqa 와 사람의 일치도(κ)·놓친 위험·전문가 지표")
    s.add_argument("--run", required=True)
    s.add_argument("--key", required=True, help="export-blind 가 만든 blind_key.json")
    s.add_argument("--labels", nargs="+", required=True, help="검수자별로 채운 blind_sheet.csv (여럿이면 검수자끼리 κ 도)")
    s.add_argument("--out", help="결과 JSON (같은 이름의 .md 도 씀)")
    s.set_defaults(fn=cmd_import_labels)

    s = sub.add_parser("study-check", help="어르신 실험에 쓸 답 점검 — 실험 문항 답에 Hold 가 있으면 실패")
    s.add_argument("--run", required=True)
    s.add_argument("--models", required=True, help="실험 조건 모델 (예: teacher_ft,student_kd,student_ft)")
    s.add_argument("--items", help="문항 (없으면 골든셋 study_set)")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_study_check)

    s = sub.add_parser("study-materials", help="어르신 사용자 실험 묶음 (순서 상쇄·블라인드·큰 글씨 카드·대본·기록지)")
    s.add_argument("--run", required=True, help="Teacher·KD·FT 답이 든 채점 끝난 회차")
    s.add_argument("--out", required=True)
    s.add_argument("--conditions", help="조건=모델 쉼표로 (기본 ondam=ondam_local,chatgpt=chatgpt_luna61,norag=local_norag)")
    s.add_argument("--participants", type=int, default=15, help="참가자 수 (조건 수의 배수면 완전 균형)")
    s.add_argument("--items", help="본 과제 문항 (기본: 실행안 §2 의 10문항)")
    s.add_argument("--audio-dir", help="답 음성 파일 폴더 (<번호>.wav). 없으면 진행자가 답 카드를 읽는다")
    s.add_argument("--seed", type=int, default=20261130)
    s.set_defaults(fn=cmd_study_materials)

    s = sub.add_parser("study-analyze", help="어르신 실험 결과 분석 (조건별 지표·순위 검정·기술 지표와의 관계)")
    s.add_argument("--data", required=True, help="채운 입력서식_본과제.csv")
    s.add_argument("--key", required=True, help="대응표_봉인.json")
    s.add_argument("--pref", help="채운 입력서식_선호.csv")
    s.add_argument("--run", help="같은 회차 폴더 — 주면 autoqa 기술 지표와의 상관(RQ1)도 계산")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_study_analyze)

    s = sub.add_parser("mutants")
    s.add_argument("--out", required=True)
    s.add_argument("--subtle-out", required=True)
    s.add_argument("--seed", type=int, default=20261108)
    s.add_argument("--allow-short", action="store_true", help="변이 할당을 못 채워도 진행 (기본은 멈춤)")
    s.set_defaults(fn=cmd_mutants)

    s = sub.add_parser("mutation-score", help="변이 시험 회차의 탐지율·좋은 답 통과율·Gate 귀속을 다시 낸다")
    s.add_argument("--run", required=True)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_mutation_score)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
