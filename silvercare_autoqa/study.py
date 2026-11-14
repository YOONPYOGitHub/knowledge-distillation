"""어르신 사용자 실험 — 준비물 만들기와 결과 분석 (LLM 채점과 따로, 실제 타깃 고객에게 평가받는다).

실행안 정본 = docs/user_study/README.md. 이 모듈은 그 실행안의 준비물을 만들고 결과를 계산한다.
- materials : autoqa 회차의 세 조건 답(기본: 온담 로컬+RAG · ChatGPT luna 6.1 · 검색 끈 로컬)으로 참가자별 실험 묶음을
              만든다 (순서 상쇄 · 블라인드 · 큰 글씨 카드 · 진행 대본 · 관찰 기록지 · 입력 서식).
              Hold·검증 전 답은 어르신에게 들려주지 않는다 — 그 문항은 바꾼다.
- analyze   : 채운 입력 서식 → 조건별 행동 이해율·보기 정답률·만족도, Friedman·Wilcoxon(정확검정)·Holm 보정,
              주 비교(온담 − ChatGPT, 지식증류 때는 KD − Teacher) 차이의 신뢰구간, autoqa 기술 지표와 어르신 지표의 상관(RQ1).
통계는 의존성 없이 직접 계산한다 (팀원 PC 에 scipy 가 없어도 돈다). 리커트(5점)는 순서형이라 t-검정을 쓰지 않는다.
"""

from __future__ import annotations

import csv
import html
import itertools
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .golden import ACTIONS

# ───────────────────────────── 실행안의 기본값 (docs/user_study/README.md §2) ─────────────────────────────

# 2026-11-14: 고객이 가장 많이 묻는 영양제 질문 2개(먹는 약과 같이·여러 개 같이)를 일반 문항에 넣었다 (G01·P01 대신)
MAIN_ITEMS = ["SC-E01", "SC-E06", "SC-K01", "SC-E09",                   # 위험 4
              "SC-M01", "SC-K03", "SC-K06", "SC-S16", "SC-S23", "SC-B02"]  # 일반 6
PRACTICE_ITEM = "SC-G06"                                                 # 연습 (집계 안 함)
PREF_ITEMS = ["SC-K02", "SC-G04"]                                        # 세 답을 다 듣고 고르기
# Hold 가 나온 문항을 바꿀 후보 (같은 위험도끼리). 자살 관련(E10·E11)·민감 주제는 넣지 않는다.
SPARE = {"risk": ["SC-E08", "SC-E03", "SC-M02", "SC-K05", "SC-G05"],
         "routine": ["SC-G01", "SC-P01", "SC-G06", "SC-M05", "SC-G02"]}
# 조건 이름 = 회차의 모델 이름 (RAG 전환 2026-11-14). 모델 이름이 다르면 study-materials --conditions 로 바꾼다.
DEFAULT_CONDITIONS = {"ondam": "ondam_local", "chatgpt": "chatgpt_luna61", "norag": "local_norag"}
# 주 비교 — "온담이 ChatGPT 만큼 알아듣기 쉬운가": (앞 − 뒤) 행동 이해율 차이의 95% 하한 > −10%p 면 "차이가 거의 없음".
# 지식증류 때 묶음(kd·teacher)도 그대로 읽는다.
PRIMARY_PAIRS = (("ondam", "chatgpt"), ("kd", "teacher"))
MARGIN_VS_REFERENCE = 0.10
MARGIN_VS_TEACHER = MARGIN_VS_REFERENCE      # 예전 이름
TARGETS = {"행동 이해율": 0.90, "보기 정답률": 0.90, "만족도(100점)": 80.0}   # Proposal §5 시니어 실효성 목표

# 보기 문항의 선택지 — 답이 권한 행동의 긴급도 사다리 (어르신이 읽는 말)
LEVEL_TEXT = {
    4: "지금 바로 119에 전화하거나 응급실에 간다",
    3: "응급조치를 먼저 하고, 나아지지 않으면 119에 전화한다",
    2: "오늘이나 며칠 안에 병원에 간다",
    1: "약을 바꾸기 전에 약사나 의사에게 물어본다",
    0: "집에서 생활 습관을 조심하며 지켜본다",
}
ACTION_TEXT = {"crisis_line": "지금 바로 상담전화(109)나 119에 연락한다", "decline_redirect": "병명과 약은 의사에게 확인한다",
               "empathize": "이야기를 나누며 마음을 편히 한다"}


def action_text(action: str) -> str:
    return ACTION_TEXT.get(action) or LEVEL_TEXT[max(0, min(4, ACTIONS.get(action, 0)))]


def is_risk(item: dict) -> bool:
    return item["risk"] in ("emergency", "urgent")


# ───────────────────────────── 1. 배치 (순서 상쇄) ─────────────────────────────

def latin_schedule(items: list[str], conds: list[str], n_participants: int, seed: int) -> list[list[tuple[str, str]]]:
    """참가자마다 (문항, 조건) 순서. 조건은 문항마다 돌려 배치하고(참가자 p, 문항 k → 조건 (k+p) mod C),
    문항 순서는 참가자마다 섞는다. 참가자 수가 조건 수의 배수면 문항×조건이 모두 같은 횟수로 나온다."""
    rng = random.Random(seed)
    C = len(conds)
    out = []
    for p in range(n_participants):
        pairs = [(it, conds[(k + p) % C]) for k, it in enumerate(items)]
        rng.shuffle(pairs)
        out.append(pairs)
    return out


def schedule_balance(schedule: list[list[tuple[str, str]]]) -> dict:
    ic = Counter((it, c) for s in schedule for it, c in s)
    per_p = [Counter(c for _, c in s) for s in schedule]
    return {"item_condition_min": min(ic.values()), "item_condition_max": max(ic.values()),
            "per_participant": [dict(x) for x in per_p]}


# ───────────────────────────── 2. 준비물 ─────────────────────────────

def _load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    from .golden import load_golden
    from .pipeline import _resolve_golden

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8-sig"))
    cases = [json.loads(l) for l in (run_dir / "scored.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    items = load_golden(_resolve_golden(meta, run_dir))["by_id"]
    by = {(c["model"], c["item_id"]): c for c in cases}
    return meta, by, items


def _usable(c: dict | None) -> tuple[bool, str]:
    """어르신에게 들려줘도 되는 답인가 — Hold·채점 오류·검증 전·채점 빠짐은 안 된다 (위해 답변 0건)."""
    if c is None:
        return False, "답 없음"
    r = c["result"]
    if r["verdict"] in ("hold", "error"):
        return False, f"{r['verdict']} {r['gates']}"
    if r.get("unverified") or r.get("judge_missing"):
        return False, "검증 전·채점 빠짐"
    return True, ""


def choose_items(by: dict, items: dict, models: list[str], wanted: list[str]) -> tuple[list[str], list[dict]]:
    """모든 조건의 답이 쓸 수 있는 문항만 고른다. 안 되는 문항은 같은 위험도의 예비 문항으로 바꾸고 기록한다."""
    chosen, swaps = [], []
    used = set(wanted)
    for iid in wanted:
        bad = {m: why for m in models for ok, why in [_usable(by.get((m, iid)))] if not ok}
        if not bad:
            chosen.append(iid)
            continue
        pool = SPARE["risk" if is_risk(items[iid]) else "routine"]
        rep = next((s for s in pool if s not in used and s in items and
                    all(_usable(by.get((m, s)))[0] for m in models)), None)
        swaps.append({"from": iid, "to": rep, "why": bad})
        if rep:
            used.add(rep)
            chosen.append(rep)
    return chosen, swaps


def mcq_options(answer_action: str, rng: random.Random) -> tuple[list[str], int]:
    """보기 4개: 답이 권한 행동 + 긴급도가 다른 행동 3개. (정답 번호 1~4)"""
    lvl = max(0, min(4, ACTIONS.get(answer_action, 0)))
    others = [LEVEL_TEXT[l] for l in sorted(LEVEL_TEXT) if l != lvl]
    rng.shuffle(others)
    opts = [action_text(answer_action)] + others[:3]
    rng.shuffle(opts)
    return opts, opts.index(action_text(answer_action)) + 1


def _answer_action(c: dict, item: dict) -> str:
    return ((c.get("judgment") or {}).get("response_action")) or item["expected_action"]


def _scenario(item: dict) -> str:
    ctx = f" (기기 기록: {item['context']})" if item.get("context") else ""
    return f"{item['utterance']}{ctx}"


CSS = """
body{font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif;color:#111;background:#fff;margin:0}
.page{padding:28px 32px;page-break-after:always;min-height:90vh;box-sizing:border-box}
.big{font-size:34px;line-height:1.6}.mid{font-size:24px;line-height:1.6}.small{font-size:15px;color:#444}
.card{border:3px solid #111;border-radius:16px;padding:28px;margin:18px 0}
.opt{font-size:30px;border:2px solid #111;border-radius:12px;padding:16px 20px;margin:14px 0}
.faces{display:flex;justify-content:space-between;gap:10px;font-size:30px;text-align:center}
.faces div{border:2px solid #111;border-radius:12px;padding:12px 6px;flex:1}.faces b{font-size:56px;display:block}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #888;padding:6px;vertical-align:top}
.ans{border-left:6px solid #888;padding:10px 14px;background:#f4f4f4;font-size:18px;line-height:1.7}
h1{font-size:26px}h2{font-size:21px;margin-top:26px}
"""
FACES = [("😟", "전혀 아니다"), ("🙁", "아니다"), ("😐", "보통"), ("🙂", "그렇다"), ("😀", "매우 그렇다")]

# 진행 대본 첫머리 — 세션 전체 흐름 (실행안 docs/user_study/README.md §4 와 같은 순서·시간)
SESSION_FLOW = """<table><tr><th>순서</th><th>시간</th><th>할 일 (인쇄물)</th></tr>
<tr><td>0. 시작</td><td>5분</td><td>인사 · 동의서를 읽어 드리고 서명 · 녹음은 따로 묻는다 (동의서)</td></tr>
<tr><td>1. 사전 질문</td><td>3분</td><td>나이대·휴대폰 사용·건강 정보를 얻는 곳 (사전 질문지)</td></tr>
<tr><td>2. 연습</td><td>1분</td><td>연습 문항 1개로 순서를 익히신다 — 기록하지 않음</td></tr>
<tr><td>3. 본 과제</td><td>18분</td><td>이야기 {n_main}개 (아래 순서)</td></tr>
<tr><td>4. 답 비교</td><td>4분</td><td>이야기 {n_pref}개 — 답들을 다 듣고 가장 알아듣기 쉬운 답 고르기</td></tr>
<tr><td>5. 사후 인터뷰</td><td>6분</td><td>사후 인터뷰 질문지</td></tr>
<tr><td>6. 마무리</td><td>2분</td><td>마무리 안내 카드를 읽어 드림 · 감사 인사</td></tr></table>
<p><b>안전:</b> 실제로 지금 아픈 데를 말씀하시거나 힘들어하시면 바로 멈추고 안전 수칙 카드대로 한다.
조건 이름(어느 AI 답인지)은 이 대본에 없다 — 진행자도 모른다.</p>"""

TRIAL_STEPS = """<ol class='mid' style='font-size:18px'>
<li>이야기 카드를 보여 드리며 읽는다: "이런 상황이라고 생각해 보세요."</li>
<li>답을 한 번 들려준다 (음성 파일, 없으면 진행자가 또박또박 읽기 — 모든 답을 같은 사람이 같은 속도로). 다시 청하시면 한 번 더.</li>
<li>"그래서 지금 무엇을 하시면 될까요?" — 하신 말씀을 그대로 적는다 (관찰자).</li>
<li>보기 카드를 보여 드리며: "방금 답에서 하라고 한 것은 어느 것이었나요?"</li>
<li>얼굴 카드: "이 답이 마음에 드세요?"</li></ol>"""


def _doc(title: str, body: str) -> str:
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def _faces(q: str) -> str:
    cells = "".join(f"<div><b>{f}</b>{i}<br>{html.escape(w)}</div>" for i, (f, w) in enumerate(FACES, 1))
    return f"<div class='page'><p class='big'>{html.escape(q)}</p><div class='faces'>{cells}</div></div>"


def materials(run_dir: str | Path, out_dir: str | Path, *, conditions: dict[str, str] | None = None,
              n_participants: int = 15, seed: int = 20261130, main_items: list[str] | None = None,
              practice_item: str = PRACTICE_ITEM, pref_items: list[str] | None = None,
              audio_dir: str | Path | None = None) -> dict[str, Any]:
    """실험 묶음을 만든다. 쓸 수 없는 답(Hold 등)이 있으면 그 문항을 바꾸고, 바꿀 문항이 모자라면 멈춘다."""
    run, out = Path(run_dir), Path(out_dir)
    conditions = conditions or DEFAULT_CONDITIONS
    meta, by, items = _load_run(run)
    validity = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["validity"]
    if not validity["ok"]:
        raise ValueError("회차 유효성 미달(채점 오류·검증 누락) — 채점을 끝낸 회차로만 실험 묶음을 만든다")
    models = list(conditions.values())
    missing = sorted({m for m in models if not any(k[0] == m for k in by)})
    if missing:
        raise ValueError(f"회차에 없는 모델: {missing} (회차의 모델: {sorted({k[0] for k in by})})")
    main, swaps = choose_items(by, items, models, main_items or MAIN_ITEMS)
    lost = [s for s in swaps if not s["to"]]
    if lost:
        raise ValueError("바꿀 문항이 모자라 실험 묶음을 만들 수 없음 (어르신에게 Hold 답을 들려주지 않는다): " +
                         "; ".join(f"{s['from']} {s['why']}" for s in lost))
    pref, pswaps = choose_items(by, items, models, pref_items or PREF_ITEMS)
    pref = [p for p in pref if p not in main]
    practice_ok = all(_usable(by.get((m, practice_item)))[0] for m in models)
    if not practice_ok:
        raise ValueError(f"연습 문항 {practice_item} 의 답에 쓸 수 없는 답이 있음 — --practice 로 다른 문항을 고른다")
    labels = list(conditions)
    sched = latin_schedule(main, labels, n_participants, seed)
    rng = random.Random(seed + 1)
    audio = Path(audio_dir) if audio_dir else None
    out.mkdir(parents=True, exist_ok=True)
    key: dict[str, Any] = {"run": str(run), "fingerprint": meta.get("fingerprint"), "conditions": conditions,
                           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed, "main_items": main,
                           "pref_items": pref, "practice_item": practice_item, "swaps": swaps + pswaps, "trials": {}, "pref": {}}
    data_rows, tts_rows = [], []
    for p in range(n_participants):
        pid = f"P{p + 1:02d}"
        trials = [(f"{pid}-연습", practice_item, labels[p % len(labels)], "practice")] + \
            [(f"{pid}-{k:02d}", it, lab, "main") for k, (it, lab) in enumerate(sched[p], 1)]
        script, cards, obs = [], [], []
        for code, iid, lab, phase in trials:
            c = by[(conditions[lab], iid)]
            it = items[iid]
            opts, ans_no = mcq_options(_answer_action(c, it), rng)
            key["trials"][code] = {"participant": pid, "item_id": iid, "condition": lab, "model": conditions[lab],
                                   "pid": c["pid"], "phase": phase, "mcq_answer": ans_no,
                                   "answer_action": _answer_action(c, it)}
            tts_rows.append({"file": f"{code}.wav", "text": c["response"]})
            has_audio = audio and any((audio / f"{code}{ext}").exists() for ext in (".wav", ".mp3", ".m4a"))
            musts = it.get("must_include") or []
            must = " / ".join(m["text"] for m in musts if m.get("tier") == 1) or \
                " / ".join(m["text"] for m in musts) or "—"
            script.append(f"<h2>{html.escape(code)} {'(연습 — 집계 안 함)' if phase == 'practice' else ''}</h2>"
                          + (f"<p>음성 파일 <b>{html.escape(code)}</b> 를 들려준다. 파일이 없으면 아래 답을 읽는다.</p>" if has_audio else "")
                          + f"<div class='ans'>{html.escape(c['response'])}</div>"
                          f"<p class='small'>재진술 판정 기준 — 답이 권한 행동: <b>{html.escape(action_text(_answer_action(c, it)))}</b>"
                          f" · 답에 들어 있어야 할 말: {html.escape(must)} · 보기 정답 <b>{ans_no}번</b></p>")
            cards.append(f"<div class='page'><p class='small'>{html.escape(code)}</p><div class='card big'>"
                         f"{html.escape(_scenario(it))}</div></div>")
            cards.append(f"<div class='page'><p class='small'>{html.escape(code)} 보기</p><p class='mid'>방금 답에서 하라고 한 것은?</p>" +
                         "".join(f"<div class='opt'>{n} {html.escape(o)}</div>" for n, o in enumerate(opts, 1)) + "</div>")
            obs.append(f"<tr><td>{html.escape(code)}</td><td></td><td></td><td>맞음 · 부분 · 틀림 · 모름</td><td>① ② ③ ④</td>"
                       f"<td>1 2 3 4 5</td><td></td></tr>")
            if phase == "main":
                data_rows.append({"participant": pid, "trial": code, "item_id": iid, "restate_text": "", "restate_code": "",
                                  "restate_code_2": "", "mcq_choice": "", "satisfaction": "", "replay_count": "",
                                  "time_s": "", "note": ""})
        # 세 답을 다 듣고 고르기 (선호)
        pref_html = []
        for j, iid in enumerate(pref, 1):
            order = labels[p % len(labels):] + labels[:p % len(labels)]
            codes = []
            for n, lab in enumerate(order, 1):
                code = f"{pid}-선호{j}-{n}"
                c = by[(conditions[lab], iid)]
                key["pref"][code] = {"participant": pid, "item_id": iid, "condition": lab, "model": conditions[lab],
                                     "pid": c["pid"], "position": n}
                tts_rows.append({"file": f"{code}.wav", "text": c["response"]})
                codes.append((code, c["response"]))
            pref_html.append(f"<h2>선호 {j} — {html.escape(items[iid]['scenario'])}</h2><p>이야기를 읽고, 답 {len(order)}개를 차례로 들려준다. "
                             f"\"어느 답이 가장 알아듣기 쉬웠나요? 왜 그런가요?\"</p>" +
                             "".join(f"<p><b>{n}번 답</b> ({html.escape(cd)})</p><div class='ans'>{html.escape(t)}</div>"
                                     for n, (cd, t) in enumerate(codes, 1)))
            cards.append(f"<div class='page'><p class='small'>선호 {j}</p><div class='card big'>{html.escape(_scenario(items[iid]))}</div>"
                         f"<p class='mid'>가장 알아듣기 쉬운 답은? &nbsp; " + " · ".join(f"{n}번" for n in range(1, len(order) + 1)) +
                         "</p></div>")
        folder = out / pid
        folder.mkdir(exist_ok=True)
        flow = SESSION_FLOW.format(n_main=len(sched[p]), n_pref=len(pref))
        (folder / "진행대본.html").write_text(_doc(f"{pid} 진행 대본", f"<div style='padding:20px'><h1>{pid} 진행 대본</h1>{flow}"
                                                     f"<h1>본 과제 — 문항마다 같은 순서</h1>{TRIAL_STEPS}{''.join(script)}"
                                                     f"<h1>답 비교</h1>{''.join(pref_html)}</div>"), encoding="utf-8")
        (folder / "참가자카드.html").write_text(_doc(f"{pid} 카드", "".join(cards) + _faces("이 답이 마음에 드세요?")), encoding="utf-8")
        (folder / "관찰기록지.html").write_text(_doc(f"{pid} 관찰 기록지", (
            f"<div style='padding:20px'><h1>{pid} 관찰 기록지</h1><p>참가자 번호만 적는다 (이름 쓰지 않음). 날짜 ____ · 진행자 ____ · 관찰자 ____</p>"
            "<table><tr><th>번호</th><th>다시 들음</th><th>재진술 (말씀 그대로)</th><th>재진술 판정</th><th>보기</th><th>만족</th><th>표정·머뭇거림</th></tr>"
            + "".join(obs) + "</table><p>선호: 1번 ___ 2번 ___ · 이유 ________</p></div>")), encoding="utf-8")
    with (out / "입력서식_본과제.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data_rows[0]))
        w.writeheader()
        w.writerows(data_rows)
    with (out / "입력서식_선호.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["participant", "pref_set", "first_choice(1~3)", "reason"])
        for p in range(n_participants):
            for j in range(1, len(pref) + 1):
                w.writerow([f"P{p + 1:02d}", j, "", ""])
    with (out / "음성목록.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "text"])
        w.writeheader()
        w.writerows(tts_rows)
    (out / "대응표_봉인.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    # 어떤 회차든 똑같이 쓰는 인쇄물 (동의서·사전 질문지·사후 인터뷰·안전 수칙·마무리 안내·모집 안내문)
    printed = Path(__file__).resolve().parents[1] / "docs" / "user_study" / "print"
    if printed.exists():
        common = out / "공통_인쇄물"
        common.mkdir(exist_ok=True)
        for f in sorted(printed.glob("*.html")):
            (common / f.name).write_bytes(f.read_bytes())
    bal = schedule_balance(sched)
    return {"out": str(out), "participants": n_participants, "main_items": main, "pref_items": pref, "swaps": swaps + pswaps,
            "balance": bal, "n_answers": len(tts_rows)}


# ───────────────────────────── 3. 통계 (의존성 없음) ─────────────────────────────

def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def wilcoxon_exact(x: list[float], y: list[float]) -> dict:
    """짝지은 두 조건의 Wilcoxon 부호순위 검정 (차이 0 은 뺀다). n ≤ 20 이면 모든 부호 조합으로 정확한 p."""
    d = [a - b for a, b in zip(x, y) if a - b != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "W+": 0.0, "p": 1.0, "r_rb": 0.0}
    rk = _ranks([abs(v) for v in d])
    wp = sum(r for r, v in zip(rk, d) if v > 0)
    total = sum(rk)
    if n <= 20:
        mean = total / 2
        dist = Counter()
        for signs in itertools.product((0, 1), repeat=n):
            dist[sum(r for r, s in zip(rk, signs) if s)] += 1
        far = sum(cnt for w, cnt in dist.items() if abs(w - mean) >= abs(wp - mean) - 1e-9)
        p = far / 2 ** n
    else:
        mean, var = total / 2, sum(r * r for r in rk) / 4
        z = (wp - mean) / math.sqrt(var)
        p = math.erfc(abs(z) / math.sqrt(2))
    return {"n": n, "W+": wp, "p": min(1.0, p), "r_rb": (2 * wp - total) / total}     # 짝 순위 이연 상관 (−1~1)


def friedman(table: list[list[float]]) -> dict:
    """참가자 × 조건 표 → Friedman χ² (동점 보정), Kendall W. 조건 3개면 자유도 2 → p = exp(−χ²/2)."""
    n, k = len(table), len(table[0])
    R = [0.0] * k
    ties = 0.0
    for row in table:
        rk = _ranks(row)
        for j in range(k):
            R[j] += rk[j]
        for cnt in Counter(row).values():
            ties += cnt ** 3 - cnt
    chi = (12 / (n * k * (k + 1))) * sum(r * r for r in R) - 3 * n * (k + 1)
    corr = 1 - ties / (n * (k ** 3 - k))
    chi = chi / corr if corr > 0 else 0.0
    if k == 3:
        p = math.exp(-chi / 2)
    else:      # 자유도가 2 가 아니면 윌슨-힐퍼티 근사
        df = k - 1
        z = ((chi / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        p = 0.5 * math.erfc(z / math.sqrt(2))
    return {"n": n, "k": k, "chi2": chi, "p": p, "kendall_w": chi / (n * (k - 1)) if n else 0.0, "rank_sums": R}


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[name] = run
    return out


def bootstrap_diff(a: list[float], b: list[float], *, reps: int = 5000, seed: int = 7) -> tuple[float, float, float]:
    """짝지은 평균 차이(a−b)와 참가자를 다시 뽑아 만든 95% 신뢰구간."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return (float("nan"),) * 3
    sims = sorted(statistics.fmean(rng.choice(d) for _ in d) for _ in range(reps))
    return statistics.fmean(d), sims[int(0.025 * reps)], sims[int(0.975 * reps) - 1]


def spearman(x: list[float], y: list[float], *, reps: int = 2000, seed: int = 11) -> dict:
    """순위 상관과 순열 p (두 쪽)."""
    n = len(x)
    if n < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": n, "rho": None, "p": None}
    rx, ry = _ranks(x), _ranks(y)

    def rho(u, v):
        mu, mv = statistics.fmean(u), statistics.fmean(v)
        num = sum((a - mu) * (b - mv) for a, b in zip(u, v))
        den = math.sqrt(sum((a - mu) ** 2 for a in u) * sum((b - mv) ** 2 for b in v))
        return num / den if den else 0.0
    r0 = rho(rx, ry)
    rng = random.Random(seed)
    ge = 0
    for _ in range(reps):
        sh = ry[:]
        rng.shuffle(sh)
        ge += abs(rho(rx, sh)) >= abs(r0) - 1e-12
    return {"n": n, "rho": r0, "p": (ge + 1) / (reps + 1)}


# ───────────────────────────── 4. 분석 ─────────────────────────────

RESTATE = {"맞음": 1.0, "부분": 0.5, "틀림": 0.0, "모름": 0.0}


def _restate_score(row: dict) -> float | None:
    v = (row.get("restate_code") or "").strip()
    return RESTATE.get(v) if v else None


def analyze(data_csv: str | Path, key_json: str | Path, *, run_dir: str | Path | None = None,
            pref_csv: str | Path | None = None) -> dict[str, Any]:
    """채운 입력 서식 → 조건별 지표·검정·RQ1. 틀린 값은 errors 로 모은다 (빈칸은 '기록 없음'으로 뺀다)."""
    key = json.loads(Path(key_json).read_text(encoding="utf-8-sig"))
    rows = list(csv.DictReader(Path(data_csv).read_text(encoding="utf-8-sig").splitlines()))
    conds = list(key["conditions"])
    errors, trials = [], []
    for r in rows:
        k = key["trials"].get(r.get("trial", ""))
        if not k or k["phase"] != "main":
            continue
        rec = {"participant": k["participant"], "condition": k["condition"], "item_id": k["item_id"], "pid": k["pid"]}
        try:
            rec["restate"] = _restate_score(r)
            if (r.get("restate_code") or "").strip() and rec["restate"] is None:
                raise ValueError(f"재진술 판정 {r['restate_code']!r} 는 맞음·부분·틀림·모름")
            rec["restate2"] = RESTATE.get((r.get("restate_code_2") or "").strip())
            ch = (r.get("mcq_choice") or "").strip()
            rec["mcq"] = (int(ch) == k["mcq_answer"]) if ch else None
            if ch and not 1 <= int(ch) <= 4:
                raise ValueError(f"보기 {ch} 는 1~4")
            s = (r.get("satisfaction") or "").strip()
            rec["sat"] = int(s) if s else None
            if s and not 1 <= int(s) <= 5:
                raise ValueError(f"만족 {s} 는 1~5")
        except ValueError as e:
            errors.append(f"{r.get('trial')}: {e}")
            continue
        trials.append(rec)

    def per_participant(field: str, f=lambda v: float(v)) -> dict[str, dict[str, float]]:
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for t in trials:
            if t[field] is not None:
                acc[t["participant"]][t["condition"]].append(f(t[field]))
        return {p: {c: statistics.fmean(v) for c, v in cs.items()} for p, cs in acc.items()}

    metrics = {"행동 이해율": per_participant("restate"), "보기 정답률": per_participant("mcq"),
               "만족도(100점)": per_participant("sat", lambda v: (v - 1) / 4 * 100)}
    out: dict[str, Any] = {"n_trials": len(trials), "errors": errors, "conditions": key["conditions"], "metrics": {}}
    for name, pp in metrics.items():
        complete = [p for p, cs in pp.items() if all(c in cs for c in conds)]
        table = [[pp[p][c] for c in conds] for p in complete]
        summ = {c: {"mean": statistics.fmean(pp[p][c] for p in complete) if complete else None,
                    "median": statistics.median(pp[p][c] for p in complete) if complete else None} for c in conds}
        res: dict[str, Any] = {"n_participants": len(complete), "by_condition": summ,
                               "target": TARGETS.get(name), "meets_target": {c: (summ[c]["mean"] is not None and
                                                                                 summ[c]["mean"] >= TARGETS[name]) for c in conds}}
        if len(complete) >= 2 and len(conds) >= 2:
            if len(conds) >= 3:
                res["friedman"] = friedman(table)
            pw = {}
            for a, b in itertools.combinations(range(len(conds)), 2):
                pw[f"{conds[a]} vs {conds[b]}"] = wilcoxon_exact([row[a] for row in table], [row[b] for row in table])
            adj = holm({k_: v["p"] for k_, v in pw.items()})
            for k_ in pw:
                pw[k_]["p_holm"] = adj[k_]
            res["pairwise"] = pw
            pair = next(((x, y) for x, y in PRIMARY_PAIRS if x in conds and y in conds), None)
            if pair:
                a, b = conds.index(pair[0]), conds.index(pair[1])
                m, lo, hi = bootstrap_diff([row[a] for row in table], [row[b] for row in table])
                scale = 100.0 if name == "만족도(100점)" else 1.0
                res["primary_diff"] = {"pair": list(pair), "mean": m, "ci95": [lo, hi],
                                       "almost_same": (lo > -MARGIN_VS_REFERENCE * scale) if not math.isnan(lo) else None}
                if pair == ("kd", "teacher"):
                    res["kd_minus_teacher"] = res["primary_diff"]      # 예전 이름
        out["metrics"][name] = res
    # 재진술 판정 두 사람 일치도 (같은 답을 두 사람이 따로 판정)
    pairs = [(t["restate"], t["restate2"]) for t in trials if t["restate"] is not None and t["restate2"] is not None]
    if pairs:
        from .human import cohen_kappa
        out["restate_coder_kappa"] = {"n": len(pairs), "kappa": cohen_kappa([(str(a), str(b)) for a, b in pairs])}
    # RQ1 — autoqa 기술 지표와 어르신 지표의 관계 (답 하나 단위)
    if run_dir:
        cases = {json.loads(l)["pid"]: json.loads(l) for l in
                 (Path(run_dir) / "scored.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()}
        feats = {"행동 가능성(채점기 1~5)": lambda c: (c.get("judgment") or {}).get("actionability"),
                 "쉬운 표현 점수": lambda c: c["det"]["kpi"].get("plain_score"),
                 "첫 세 문장에 핵심": lambda c: int(bool((c.get("judgment") or {}).get("core_in_first3"))),
                 "3문장 이내": lambda c: int(bool(c["det"]["kpi"].get("within_3_sentences")))}
        rq1 = {}
        for fname, fn in feats.items():
            for oname, ofield in (("행동 이해", "restate"), ("만족", "sat")):
                xs, ys = [], []
                for t in trials:
                    c = cases.get(t["pid"])
                    if c is None or t[ofield] is None or fn(c) is None:
                        continue
                    xs.append(float(fn(c)))
                    ys.append(float(t[ofield]))
                rq1[f"{fname} ↔ {oname}"] = spearman(xs, ys)
        out["rq1"] = rq1
    if pref_csv:
        prow = list(csv.DictReader(Path(pref_csv).read_text(encoding="utf-8-sig").splitlines()))
        wins = Counter()
        for r in prow:
            ch = (r.get("first_choice(1~3)") or "").strip()
            if not ch:
                continue
            code = f"{r['participant']}-선호{r['pref_set']}-{ch}"
            k = key["pref"].get(code)
            if k:
                wins[k["condition"]] += 1
            else:
                errors.append(f"선호 {r['participant']} {r['pref_set']}: 번호 {ch} 를 찾지 못함")
        out["preference_first_choice"] = dict(wins)
    return out


def _fmt(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.0f}%" if pct else f"{v:.1f}"


def render_analysis_md(rep: dict) -> str:
    conds = list(rep["conditions"])
    L = ["# 어르신 사용자 실험 결과", "", "## 한눈에", ""]
    for name, m in rep["metrics"].items():
        pct = name != "만족도(100점)"
        means = " · ".join(f"{c} {_fmt(m['by_condition'][c]['mean'], pct)}" for c in conds)
        L.append(f"- **{name}** (참가자 {m['n_participants']}명): {means}")
    km = rep["metrics"].get("행동 이해율", {}).get("primary_diff")
    if km:
        x, y = km["pair"]
        L.append(f"- {x} − {y} 행동 이해율 {_fmt(km['mean'], True)} (95% 구간 {_fmt(km['ci95'][0], True)}~{_fmt(km['ci95'][1], True)}) → "
                 + ("**차이가 거의 없음** (하한이 −10%p 보다 위)" if km["almost_same"] else "차이가 거의 없다고 말할 수 없음"))
    if rep["errors"]:
        L.append(f"- ⚠ 읽지 못한 기록 {len(rep['errors'])}건 — 고친 뒤 다시 계산한다")
    L += ["", "## 1. 조건별 지표와 검정", "",
          "| 지표 | " + " | ".join(conds) + " | 목표 | Friedman p |", "|---|" + "---|" * (len(conds) + 2)]
    for name, m in rep["metrics"].items():
        pct = name != "만족도(100점)"
        fr = m.get("friedman")
        tgt = m["target"]
        L.append(f"| {name} | " + " | ".join(_fmt(m["by_condition"][c]["mean"], pct) for c in conds) +
                 f" | {_fmt(tgt, pct) if tgt is not None else '—'} | {('%.3f' % fr['p']) if fr else '—'} |")
    L += ["", "| 짝 비교 (행동 이해율) | 차이 0 뺀 참가자 | p (정확) | p (Holm) | 효과 크기 r |", "|---|---|---|---|---|"]
    for k, v in (rep["metrics"].get("행동 이해율", {}).get("pairwise") or {}).items():
        L.append(f"| {k} | {v['n']} | {v['p']:.3f} | {v['p_holm']:.3f} | {v['r_rb']:+.2f} |")
    if rep.get("restate_coder_kappa"):
        k = rep["restate_coder_kappa"]
        L += ["", f"재진술 판정 두 사람 일치도 κ = {_fmt(k['kappa'])} ({k['n']}건)."]
    if rep.get("rq1"):
        L += ["", "## 2. 기술 지표와 어르신 지표 (Proposal RQ1)", "", "| 관계 | 답 수 | 순위 상관 ρ | p |", "|---|---|---|---|"]
        for k, v in rep["rq1"].items():
            L.append(f"| {k} | {v['n']} | {_fmt(v['rho']) if v['rho'] is not None else '—'} | "
                     f"{('%.3f' % v['p']) if v['p'] is not None else '—'} |")
    if rep.get("preference_first_choice"):
        L += ["", "## 3. 세 답을 다 듣고 고른 답", "", "| 조건 | 가장 알아듣기 쉽다고 고른 횟수 |", "|---|---|"]
        for c in conds:
            L.append(f"| {c} | {rep['preference_first_choice'].get(c, 0)} |")
    L += ["", "읽는 법: 표의 값은 참가자별 평균의 평균이다. 조건 비교는 순위 검정(Friedman·Wilcoxon)으로 했다 — 5점 척도는 순서형이다.",
          "참가자가 12~15명이라 p 값만 보지 말고 차이의 방향과 신뢰구간을 같이 본다."]
    return "\n".join(L) + "\n"
