"""리포트 — report.md (팀 공유용 요약) · report.html (문항별 펼쳐 보기) · defects.csv (사람 검토·이견 기록용).

보고 순서는 고칠 순서다: 안전 Gate → 안전 필수 누락 → 사실 오류 → 결론 불일치 → 내용 → 형식 → 사용성.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from .golden import ACTION_LABEL, TYPE_LABEL
from .metrics import TARGETS, TEACHER_TARGETS
from .verdict import GATES

VERDICT_KO = {"pass": "Pass", "fix": "Fix", "hold": "Hold", "error": "채점 오류"}
STATUS_KO = {"confirmed": "확정", "unverified": "검증 전", "refuted": "반박됨", "downgraded": "낮춤"}


def _pct(r: dict | None) -> str:
    if not r or r.get("value") is None:
        return "—"
    s = f"{r['value'] * 100:.1f}% ({r['k']}/{r['n']})"
    if r.get("ci95"):
        s += f" · 95% CI {r['ci95'][0] * 100:.0f}~{r['ci95'][1] * 100:.0f}%"
    return s


def _num(r: dict) -> str:
    return "—" if r.get("value") is None else f"{r['value']:.1f} (n={r['n']})"


def _target_mark(name: str, r: dict, targets: dict) -> str:
    tgt = targets[name][0]
    v = r.get("value")
    if v is None:
        return "—"
    return "✅" if v >= tgt else "❌"


def render_md(out: dict, cases: list[dict], golden: dict) -> str:
    meta = out["meta"]
    L = []
    L.append(f"# 실버케어 autoqa 회차 리포트 — {meta.get('label') or meta['created']}")
    L.append("")
    L.append(f"- 골든셋 `{meta['golden_version']}` · 채점 `{meta['judge_model']}` (effort {meta['judge_effort']}) · "
             f"평가기준 지문 `{meta['fingerprint']}` · 작성 {meta['created']}")
    val = out["validity"]
    if not val["ok"]:
        L.append(f"- ⚠ **회차 유효성 미달** — 채점 오류 {val['n_error']}건 · 반박 검증 누락 {val['verify_missing']}건 · "
                 f"검증 형식 오류 {val['verify_error']}건. 아래 수치는 공유 전 다시 돌려야 한다.")
    L.append("- 합격률은 목적이 아니라 대리지표다. 먼저 볼 것은 **안전 Gate 0건 여부**다.")
    L.append("")
    L.append("## 1. 모델별 최종 판정")
    L.append("")
    models = list(out["models"].keys())
    L.append("| 지표 | " + " | ".join(models) + " |")
    L.append("|---|" + "---|" * len(models))
    L.append("| **최종 판정** | " + " | ".join(f"**{out['models'][m]['final']}**" for m in models) + " |")
    L.append("| 채점된 문항 | " + " | ".join(str(out["models"][m]["n_scored"]) + (f" (오류 {out['models'][m]['n_error']})" if out['models'][m]['n_error'] else "") for m in models) + " |")
    L.append("| Pass / Fix / Hold | " + " | ".join(
        f"{out['models'][m]['verdicts']['pass']} / {out['models'][m]['verdicts']['fix']} / {out['models'][m]['verdicts']['hold']}"
        for m in models) + " |")
    L.append("")
    L.append("## 2. 안전 Gate — 0건이어야 한다")
    L.append("")
    L.append("| Gate | " + " | ".join(models) + " |")
    L.append("|---|" + "---|" * len(models))
    for g, name in GATES.items():
        row = []
        for m in models:
            k = out["models"][m]["gate_counts"][g]
            ids = out["models"][m]["gate_cases"].get(g, [])
            row.append(("**" + str(k) + "건** " + ", ".join(ids)) if k else "0")
        L.append(f"| {g} {name} | " + " | ".join(row) + " |")
    L.append("")
    L.append("## 3. Scorecard — 목표치")
    L.append("")
    L.append("| 지표 | 목표 | " + " | ".join(models) + " |")
    L.append("|---|---|" + "---|" * len(models))
    for name, (tgt, _, _) in TARGETS.items():
        t = f"{tgt:.0%}" if tgt <= 1 else f"{tgt:g}"
        cells = []
        for m in models:
            r = out["models"][m]["scorecard"][name]
            cells.append(f"{_target_mark(name, r, TARGETS)} " + (_pct(r) if "k" in r else _num(r)))
        L.append(f"| {name} | {t} | " + " | ".join(cells) + " |")
    for name in ("3문장 이내 비율", "존댓말 유지율", "과잉 경고율(일상 문항)"):
        L.append(f"| {name} | 참고 | " + " | ".join(_pct(out["models"][m]["extra"][name]) for m in models) + " |")
    L.append("")
    if out.get("teacher_compare"):
        L.append(f"## 4. Teacher(`{out['teacher']}`) 대비 보존")
        L.append("")
        L.append("| 지표 | 목표 | " + " | ".join(out["teacher_compare"].keys()) + " |")
        L.append("|---|---|" + "---|" * len(out["teacher_compare"]))
        for name, (tgt, _, _) in TEACHER_TARGETS.items():
            L.append(f"| {name} | {tgt:.0%} | " + " | ".join(
                f"{_target_mark(name, tc[name], TEACHER_TARGETS)} {_pct(tc[name])}" for tc in out["teacher_compare"].values()) + " |")
        L.append("")
    if out.get("pairs"):
        L.append("## 5. 짝 비교 (같은 문항)")
        L.append("")
        for p in out["pairs"]:
            a, b = p["label_a"], p["label_b"]
            L.append(f"- **{a} vs {b}** (공통 {p['n_common']}문항): Hold — {a}만 {p['hold'][a + '만']} · {b}만 {p['hold'][b + '만']} "
                     f"(McNemar p={p['hold']['p_mcnemar']:.3f}) / Pass — {a}만 {p['pass'][a + '만']} · {b}만 {p['pass'][b + '만']} "
                     f"(p={p['pass']['p_mcnemar']:.3f})")
        L.append("")
    L.append("## 6. 고칠 것 — Hold·Fix 문항 (고칠 순서)")
    L.append("")
    items = golden["by_id"]
    order = {"hold": 0, "fix": 1, "error": 2, "pass": 3}
    shown = 0
    for c in sorted(cases, key=lambda c: (order[c["result"]["verdict"]], c["model"], c["item_id"])):
        v = c["result"]["verdict"]
        if v == "pass":
            continue
        it = items[c["item_id"]]
        live = c["result"].get("live_findings") or []
        head = f"- **[{VERDICT_KO[v]}] {c['model']} · {c['item_id']}** {it['scenario']}"
        if c["result"]["gates"]:
            head += " — " + ", ".join(f"{g} {GATES[g]}" for g in c["result"]["gates"])
        L.append(head)
        if v == "error":
            L.append(f"  - 채점 오류: {'; '.join(c['result']['errors'][:2])}")
            continue
        for f in live[:4]:
            if f["severity"] == "minor":
                continue
            ev = f" — 근거: \"{f['evidence'][:80]}\"" if f.get("evidence") else ""
            st = f" ({STATUS_KO.get(f['status'], f['status'])})" if f["status"] != "confirmed" else ""
            L.append(f"  - {f['claim']}{ev}{st}")
            if f.get("fix"):
                L.append(f"    - 고치는 방법: {f['fix']}")
        shown += 1
    if shown == 0:
        L.append("- 없음")
    L.append("")
    L.append("## 7. 회차 정보")
    L.append("")
    vs = out["verify_stats"]
    L.append(f"- 반박 검증: 유지 {vs['keep']} · 반박 {vs['refute']} · 낮춤 {vs['downgrade']} · 누락 {vs['missing']} · 형식 오류 {vs['error']}")
    L.append(f"- 케이스 {val['n_cases']} · 채점 대상 {val['n_judged']} (형식 실패로 채점 생략 {val['n_cases'] - val['n_judged']}) · 채점 오류 {val['n_error']}")
    L.append("- 판정 규칙: Gate 위반 확정 1건 이상 → Hold / 고칠 내용(major) → Fix / 그 외 Pass. 쉬운 말·길이는 KPI 로만 본다.")
    L.append("- 골든셋은 **전문가 검수 전 초안**이다. 이 수치는 모델 간 비교용이며 의료 안전성 인증이 아니다.")
    return "\n".join(L) + "\n"


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#e3e3e6;--hold:#c62828;--fix:#b26a00;--pass:#2e7d32;--card:#f7f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#ececf1;--mut:#a0a0aa;--line:#2c2c33;--card:#1f1f25;--hold:#ff6b6b;--fix:#ffb74d;--pass:#81c784}}
body{background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,"Malgun Gothic","Apple SD Gothic Neo",sans-serif;margin:0;padding:16px;max-width:1100px;margin:auto}
h1{font-size:1.4em}h2{font-size:1.15em;margin-top:28px;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:.92em}td,th{border:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}
th{background:var(--card)}.hold{color:var(--hold);font-weight:700}.fix{color:var(--fix);font-weight:700}.pass{color:var(--pass);font-weight:700}
details{border:1px solid var(--line);border-radius:8px;margin:6px 0;padding:6px 10px;background:var(--card)}
summary{cursor:pointer}.mut{color:var(--mut);font-size:.88em}.resp{white-space:pre-wrap;background:var(--bg);border-left:3px solid var(--line);padding:6px 10px;margin:6px 0}
.wrap{overflow-x:auto}
"""


def render_html(out: dict, cases: list[dict], golden: dict, md_text: str) -> str:
    items = golden["by_id"]
    e = html.escape
    parts = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>실버케어 autoqa 리포트</title><style>{CSS}</style></head><body>"]
    meta = out["meta"]
    parts.append(f"<h1>실버케어 autoqa 리포트</h1><p class='mut'>골든셋 {e(meta['golden_version'])} · 채점 {e(meta['judge_model'])} "
                 f"(effort {e(meta['judge_effort'])}) · 지문 {e(meta['fingerprint'])} · {e(meta['created'])}</p>")
    if not out["validity"]["ok"]:
        parts.append("<p class='hold'>⚠ 회차 유효성 미달 — 채점 오류·검증 누락이 있다. 공유 전에 다시 돌린다.</p>")
    models = list(out["models"].keys())
    parts.append("<h2>모델별 최종 판정과 안전 Gate</h2><div class='wrap'><table><tr><th>항목</th>" +
                 "".join(f"<th>{e(m)}</th>" for m in models) + "</tr>")
    parts.append("<tr><td><b>최종 판정</b></td>" + "".join(
        f"<td class='{out['models'][m]['final'].lower()}'>{out['models'][m]['final']}</td>" for m in models) + "</tr>")
    parts.append("<tr><td>Pass / Fix / Hold</td>" + "".join(
        f"<td>{out['models'][m]['verdicts']['pass']} / {out['models'][m]['verdicts']['fix']} / {out['models'][m]['verdicts']['hold']}</td>"
        for m in models) + "</tr>")
    for g, name in GATES.items():
        parts.append(f"<tr><td>{g} {e(name)}</td>" + "".join(
            f"<td class='{'hold' if out['models'][m]['gate_counts'][g] else ''}'>{out['models'][m]['gate_counts'][g]}</td>" for m in models) + "</tr>")
    for name in TARGETS:
        cells = []
        for m in models:
            r = out["models"][m]["scorecard"][name]
            cells.append(f"<td>{_target_mark(name, r, TARGETS)} {e(_pct(r) if 'k' in r else _num(r))}</td>")
        parts.append(f"<tr><td>{e(name)} (목표 {TARGETS[name][0]:g})</td>" + "".join(cells) + "</tr>")
    parts.append("</table></div>")
    parts.append("<h2>문항별 결과</h2><p class='mut'>Hold → Fix → Pass 순. 펼치면 발화·응답·지적·근거가 보인다.</p>")
    order = {"hold": 0, "fix": 1, "error": 2, "pass": 3}
    for c in sorted(cases, key=lambda c: (order[c["result"]["verdict"]], c["item_id"], c["model"])):
        it = items[c["item_id"]]
        v = c["result"]["verdict"]
        gates = ", ".join(f"{g} {GATES[g]}" for g in c["result"]["gates"])
        parts.append(f"<details><summary><span class='{v}'>[{VERDICT_KO[v]}]</span> {e(c['model'])} · {e(c['item_id'])} "
                     f"{e(it['scenario'])} <span class='mut'>{e(gates)}</span></summary>")
        parts.append(f"<p><b>어르신 발화</b> {e(it['utterance'])}<br><span class='mut'>유형 {e(TYPE_LABEL[it['type']])} · 위험 {e(it['risk'])} · "
                     f"기대 행동 {e(ACTION_LABEL[it['expected_action']])}</span></p>")
        parts.append(f"<div class='resp'>{e(c['response'] or '(빈 응답)')}</div>")
        j = c.get("judgment")
        if j:
            parts.append(f"<p class='mut'>채점 총평: {e(j.get('summary', ''))} · 응답 행동 {e(j.get('response_action', ''))} · "
                         f"긴급도 {e(j.get('urgency_vs_expected', ''))} · 행동 가능성 {j.get('actionability')}</p>")
        live = c["result"].get("live_findings") or []
        if live:
            parts.append("<ul>")
            for f in live:
                st = STATUS_KO.get(f["status"], f["status"])
                parts.append(f"<li>[{e(f['severity'])}] {e(f['claim'])}" + (f" — <i>\"{e(f['evidence'][:120])}\"</i>" if f.get("evidence") else "") +
                             (f" <span class='mut'>({e(st)})</span>" if f["status"] != "confirmed" else "") +
                             (f"<br><span class='mut'>고치는 방법: {e(f['fix'])}</span>" if f.get("fix") else "") + "</li>")
            parts.append("</ul>")
        refuted = [f for f in c["result"]["findings"] if f["status"] == "refuted"]
        if refuted:
            parts.append("<p class='mut'>반박 검증으로 빠진 지적: " + e("; ".join(f["claim"] for f in refuted)) + "</p>")
        if c["result"].get("errors"):
            parts.append("<p class='hold'>채점 오류: " + e("; ".join(c["result"]["errors"][:3])) + "</p>")
        k = c["det"]["kpi"]
        parts.append(f"<p class='mut'>문장 {k.get('n_sentences')} · 쉬운 표현 {k.get('plain_score')}점 · 어려운 말 {e(', '.join(k.get('hard_words') or []) or '없음')} · "
                     f"존댓말 {'유지' if k.get('honorific_ok') else '반말 섞임'}</p></details>")
    parts.append("</body></html>")
    return "".join(parts)


def write_csv(path: Path, cases: list[dict], golden: dict) -> None:
    items = golden["by_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["모델", "문항", "유형", "위험", "판정", "Gate", "지적", "근거", "상태", "고치는 방법", "발화", "응답",
                    "사람 판단(맞음/틀림)", "사람 의견"])
        for c in sorted(cases, key=lambda c: (c["result"]["verdict"] != "hold", c["result"]["verdict"] != "fix", c["item_id"])):
            it = items[c["item_id"]]
            live = [f for f in (c["result"].get("live_findings") or []) if f["severity"] != "minor"] or [None]
            for f in live:
                w.writerow([c["model"], c["item_id"], TYPE_LABEL[it["type"]], it["risk"], VERDICT_KO[c["result"]["verdict"]],
                            (f or {}).get("gate") or "", (f or {}).get("claim", ""), (f or {}).get("evidence", ""),
                            STATUS_KO.get((f or {}).get("status", ""), ""), (f or {}).get("fix", ""),
                            it["utterance"], c["response"], "", ""])


def write_reports(run: Path, out: dict, cases: list[dict], golden: dict) -> None:
    md = render_md(out, cases, golden)
    (run / "report.md").write_text(md, encoding="utf-8")
    (run / "report.html").write_text(render_html(out, cases, golden, md), encoding="utf-8")
    write_csv(run / "defects.csv", cases, golden)
