"""리포트 — report.md (팀 공유용 요약) · report.html (문항별 펼쳐 보기) · defects.csv (사람 검토·이견 기록용).

보고 순서는 고칠 순서다: 안전 Gate → 안전 필수 누락 → 사실 오류 → 결론 불일치 → 내용 → 형식 → 사용성.
요약 표는 summary_tables 한 곳에서 만들고 md·html 이 같은 표를 그린다 — 두 파일의 숫자가 어긋나지 않게 (감사 R23).
"""

from __future__ import annotations

import csv
import html
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


def _ms(s: dict, pct: bool = False) -> str:
    """평균±표준편차 (최소~최대)."""
    if not s or s.get("mean") is None:
        return "—"
    f = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.1f}")
    return f"{f(s['mean'])} ± {f(s['sd'])} ({f(s['min'])}~{f(s['max'])}, n={s['n']})"


def _target_mark(name: str, r: dict, targets: dict) -> str:
    tgt = targets[name][0]
    v = r.get("value")
    if v is None:
        return "—"
    return "✅" if v >= tgt else "❌"


def info_line(out: dict) -> str:
    meta = out["meta"]
    return (f"골든셋 `{meta['golden_version']}` ({meta.get('golden_review_status') or '검수 상태 미기록'}) · 채점 `{meta['judge_model']}` "
            f"(effort {meta['judge_effort']}) · 평가기준 지문 `{meta['fingerprint']}` · 작성 {meta['created']}")


def summary_lines(out: dict) -> list[str]:
    """첫 화면 요약 — 모델별 최종 판정, 회차 유효성, 다음에 할 일 (WRITING_GUIDE: 첫 화면에 답)."""
    mm = out["models"]
    L = ["최종 판정: " + " · ".join(
        f"{m} **{v['final']}**" + (f" (Gate {sum(v['gate_counts'].values())}건)" if sum(v["gate_counts"].values()) else "")
        for m, v in mm.items())]
    for cond, c in (out.get("conditions") or {}).items():
        L.append(f"seed 반복 {cond}: 가장 나쁜 seed **{c['final_worst']}** · Hold 난 seed {c['n_hold_replicates']}/{len(c['replicates'])}")
    L += header_lines(out)
    if any(v["final"] == "Hold" for v in mm.values()):
        L.append("할 일: §6 의 Hold 문항부터 고친다. Hold 답은 앱·사용자 실험에 쓰지 않는다.")
    elif any(v["final"] in ("Fix", "Incomplete") for v in mm.values()):
        L.append("할 일: §3 에서 목표에 못 미친 지표와 §6 의 Fix 문항을 본다.")
    return L


def header_lines(out: dict) -> list[str]:
    """경고 줄 — 회차를 공유하기 전에 꼭 봐야 하는 것만."""
    meta = out["meta"]
    L = []
    if meta.get("rescored"):
        r = meta["rescored"][-1]
        what = ("판정식 재적용 — 채점 결과는 prep 때 기준, 판정·지표는 지금 코드" if r.get("scoring_changed", True)
                else "채점 기준(프롬프트·룰)이 prep 뒤에 바뀜 — 이 회차의 채점은 prep 때 기준")
        L.append(f"⚠ **{what}** (prep 지문 `{r['from']}` → 지금 `{r['to']}`, 달라진 것: {', '.join(r['changed']) or '?'}). "
                 f"지문이 같은 회차끼리만 숫자를 비교한다.")
    val = out["validity"]
    if not val["ok"]:
        L.append(f"⚠ **회차 유효성 미달** — 채점 오류·빠짐 {val['n_error']}건 · 반박 검증 누락 {val['verify_missing']}건 · "
                 f"검증 형식 오류 {val['verify_error']}건 · 검증 전 Hold {val['unverified_cases']}건. 아래 수치는 공유 전 다시 돌려야 한다.")
    for w in (meta.get("response_check") or {}).get("warnings") or []:
        L.append(f"⚠ {w}")
    if out["validity"]["ok"] and not meta.get("rescored"):
        L.append("회차 유효성 정상 — 채점·반박 검증이 모두 끝났다.")
    return L


def summary_tables(out: dict) -> list[tuple[str, list[str], list[list[str]], str]]:
    """(제목, 머리줄, 행들, 설명) 목록 — md·html 이 같은 표를 그린다."""
    T = []
    models = list(out["models"].keys())
    mm = out["models"]
    rows = [["**최종 판정**"] + [f"**{mm[m]['final']}**" for m in models],
            ["판정된 문항"] + [str(mm[m]["n_judged"]) + (f" (채점 오류·빠짐 {mm[m]['n_error']})" if mm[m]["n_error"] else "")
                           for m in models],
            ["Pass / Fix / Hold"] + [f"{mm[m]['verdicts']['pass']} / {mm[m]['verdicts']['fix']} / {mm[m]['verdicts']['hold']}"
                                     for m in models]]
    T.append(("1. 모델별 최종 판정", ["지표"] + models, rows, ""))
    rows = []
    for g, name in GATES.items():
        row = [f"{g} {name}"]
        for m in models:
            k = mm[m]["gate_counts"][g]
            ids = mm[m]["gate_cases"].get(g, [])
            row.append(("**" + str(k) + "건** " + ", ".join(ids)) if k else "0")
        rows.append(row)
    T.append(("2. 안전 Gate — 0건이어야 한다", ["Gate"] + models, rows, ""))
    rows = []
    for name, (tgt, _, _) in TARGETS.items():
        t = f"{tgt:.0%}" if tgt <= 1 else f"{tgt:g}"
        rows.append([name, t] + [f"{_target_mark(name, mm[m]['scorecard'][name], TARGETS)} " +
                                 (_pct(mm[m]["scorecard"][name]) if "k" in mm[m]["scorecard"][name] else _num(mm[m]["scorecard"][name]))
                                 for m in models])
    for name in ("3문장 이내 비율", "존댓말 유지율", "과잉 경고율(일상 문항)"):
        rows.append([name, "참고"] + [_pct(mm[m]["extra"][name]) for m in models])
    T.append(("3. Scorecard — 목표치 (채점 결과가 있는 문항 기준)", ["지표", "목표"] + models, rows, ""))
    conds = out.get("conditions") or {}
    if conds:
        rows = []
        for cond, c in conds.items():
            rows.append([cond, ", ".join(f"{m.split('@', 1)[-1]}={f}" for m, f in c["finals"].items()), f"**{c['final_worst']}**",
                         f"{c['n_hold_replicates']}/{len(c['replicates'])}", _ms(c["hold_cases"]), _ms(c["pass_cases"]),
                         _ms(c["scorecard"]["필수 응답 준수율"], pct=True), _ms(c["scorecard"]["핵심 정보 보존율"], pct=True)])
        T.append(("3-1. seed 반복 — 조건별 (가장 나쁜 seed 가 조건 판정)",
                  ["조건", "seed별 판정", "조건 판정", "Hold 난 seed", "Hold 문항 수", "Pass 문항 수", "필수 응답 준수율", "핵심 정보 보존율"],
                  rows, "가장 좋은 seed 하나만 보고하지 않는다. 평균 ± 표준편차 (최소~최대, seed 수)."))
    if out.get("teacher_compare"):
        tc = out["teacher_compare"]
        rows = [[name, f"{tgt:.0%}"] + [f"{_target_mark(name, tc[m][name], TEACHER_TARGETS)} {_pct(tc[m][name])}" for m in tc]
                for name, (tgt, _, _) in TEACHER_TARGETS.items()]
        rows.append(["생성 조건", ""] + [("⚠ 다름: " + "; ".join(tc[m]["gen_diff"])) if tc[m].get("gen_diff") else "같음"
                                       for m in tc])
        T.append((f"4. Teacher(`{out['teacher']}`) 대비 보존", ["지표", "목표"] + list(tc), rows,
                  "채점 결과가 없는 문항은 분모에서 뺐다."))
        tcond = out.get("teacher_conditions") or {}
        if tcond:
            rows = [[cond] + [_ms(v[name], pct=True) for name in TEACHER_TARGETS] for cond, v in tcond.items()]
            T.append(("4-1. Teacher 대비 — 조건별 seed 평균", ["조건"] + list(TEACHER_TARGETS), rows, ""))
    if out.get("pairs"):
        rows = []
        for p in out["pairs"]:
            a, b = p["label_a"], p["label_b"]
            rows.append([f"{a} vs {b}", str(p["n_common"]), f"{a}만 {p['hold'][a + '만']} · {b}만 {p['hold'][b + '만']}",
                         f"{p['hold']['p_mcnemar']:.3f}", f"{a}만 {p['pass'][a + '만']} · {b}만 {p['pass'][b + '만']}",
                         f"{p['pass']['p_mcnemar']:.3f}",
                         ("⚠ 다름: " + "; ".join(p["gen_diff"])) if p.get("gen_diff") else "같음"])
        T.append(("5. 짝 비교 (같은 문항)", ["짝", "공통 문항", "Hold", "p(McNemar)", "Pass", "p(McNemar)", "생성 조건"], rows,
                  "생성 조건(입력 형식·디코딩)이 다르면 차이가 모델 때문인지 조건 때문인지 가릴 수 없다."))
        if out.get("pair_groups"):
            rows = [[g["request"], str(g["summary"]["n_pairs"]), _ms(g["summary"]["hold_diff"]), _ms(g["summary"]["pass_diff"])]
                    for g in out["pair_groups"]]
            T.append(("5-1. 짝 비교 — 조건별 (같은 seed 끼리)", ["조건 짝", "seed 짝 수", "Hold 차이(앞−뒤)", "Pass 차이(앞−뒤)"], rows,
                      "Hold 차이가 음수면 앞 조건의 Hold 가 적다."))
    vs, val = out["verify_stats"], out["validity"]
    T.append(("7. 회차 정보", ["항목", "값"], [
        ["기준", info_line(out)],
        ["반박 검증", f"유지 {vs['keep']} · 반박 {vs['refute']} · 낮춤 {vs['downgrade']} · 누락 {vs['missing']} · 형식 오류 {vs['error']}"],
        ["케이스", f"{val['n_cases']} · 채점 대상 {val['n_judged']} (빈 답이라 채점 생략 {val['n_cases'] - val['n_judged']}) · "
                  f"채점 오류·빠짐 {val['n_error']}"],
        ["판정 규칙", "Gate 위반 확정 1건 이상 → Hold, 고칠 내용(major) → Fix, 그 외 Pass"],
        ["읽는 법", "합격률은 대리지표다. 먼저 안전 Gate 0건인지 본다. 쉬운 말·길이는 KPI 로만 본다"],
        ["골든셋", f"{out['meta'].get('golden_review_status') or '검수 상태 미기록'} — 모델 비교용이고 의료 안전 인증이 아니다"],
    ], ""))
    return T


def _md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    L += ["| " + " | ".join(r) + " |" for r in rows]
    return L


def _major_live(c: dict, limit: int | None = None) -> list[dict]:
    """고칠 것 목록 — 사소한 지적을 먼저 빼고 자른다 (사소한 지적이 Fix 이유를 가리지 않게, 감사 D21)."""
    live = [f for f in c["result"].get("live_findings") or [] if f["severity"] != "minor"]
    return live[:limit] if limit else live


def render_md(out: dict, cases: list[dict], golden: dict) -> str:
    meta = out["meta"]
    L = [f"# 실버케어 autoqa 회차 리포트 — {meta.get('label') or meta['created']}", "", "## 한눈에", ""]
    L += [f"- {h}" for h in summary_lines(out)]
    L.append("")
    tables = summary_tables(out)
    for title, head, rows, note in tables[:-1]:
        L += [f"## {title}", ""] + _md_table(head, rows) + ([f"", f"{note}"] if note else []) + [""]
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
        head = f"- **[{VERDICT_KO[v]}] {c['model']} · {c['item_id']}** {it['scenario']}"
        if c["result"]["gates"]:
            head += " — " + ", ".join(f"{g} {GATES[g]}" for g in c["result"]["gates"])
        if c["result"].get("judge_missing"):
            head += " (채점 빠짐 — 결정 룰 확정분만)"
        L.append(head)
        if v == "error":
            L.append(f"  - 채점 오류: {'; '.join(c['result']['errors'][:2])}")
            continue
        for f in _major_live(c, 4):
            ev = f" — 근거: \"{f['evidence'][:80]}\"" if f.get("evidence") else ""
            st = f" ({STATUS_KO.get(f['status'], f['status'])})" if f["status"] != "confirmed" else ""
            L.append(f"  - {f['claim']}{ev}{st}")
            if f.get("fix"):
                L.append(f"    - 고치는 방법: {f['fix']}")
        shown += 1
    if shown == 0:
        L.append("- 없음")
    L.append("")
    title, head, rows, _ = tables[-1]
    L += [f"## {title}", ""] + _md_table(head, rows)
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
.wrap{overflow-x:auto}.warn{color:var(--hold)}
"""


def _cell(s: str) -> str:
    """md 표 칸의 **굵게** 를 html 로."""
    e = html.escape(s)
    parts = e.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def render_html(out: dict, cases: list[dict], golden: dict, md_text: str = "") -> str:
    items = golden["by_id"]
    e = html.escape
    parts = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>실버케어 autoqa 리포트</title><style>{CSS}</style></head><body>"]
    parts.append("<h1>실버케어 autoqa 리포트</h1><h2>한눈에</h2>")
    for h in summary_lines(out):
        parts.append(f"<p class='{'warn' if h.startswith('⚠') else ''}'>{_cell(h.replace('`', ''))}</p>")
    for title, head, rows, note in summary_tables(out):
        parts.append(f"<h2>{e(title)}</h2><div class='wrap'><table><tr>" + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>")
        for r in rows:
            parts.append("<tr>" + "".join(f"<td>{_cell(c)}</td>" for c in r) + "</tr>")
        parts.append("</table></div>" + (f"<p class='mut'>{e(note)}</p>" if note else ""))
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


def _gate_cell(f: dict | None) -> str:
    """defects.csv 의 Gate 칸 — Hold 로 센 지적만 Gate 번호를 적는다 (낮춤·반박은 표시를 달리해 Gate 집계와 맞춤, 감사 R10)."""
    if not f or not f.get("gate"):
        return ""
    if f["cat"] == "gate" and f["severity"] == "critical":
        return f["gate"]
    return f"{f['gate']}({STATUS_KO.get(f['status'], f['status'])})"


def _csv_safe(v: Any) -> str:
    """Excel 이 수식으로 읽지 않게 = + - @ 로 시작하는 칸 앞에 ' 를 붙인다 (감사 A16)."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


def write_csv(path: Path, cases: list[dict], golden: dict) -> None:
    items = golden["by_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["모델", "문항", "유형", "위험", "판정", "Gate", "지적", "근거", "상태", "고치는 방법", "발화", "응답",
                    "사람 판단(맞음/틀림)", "사람 의견"])
        for c in sorted(cases, key=lambda c: (c["result"]["verdict"] != "hold", c["result"]["verdict"] != "fix", c["item_id"])):
            it = items[c["item_id"]]
            live = _major_live(c) or [None]
            for f in live:
                w.writerow([_csv_safe(x) for x in (
                    c["model"], c["item_id"], TYPE_LABEL[it["type"]], it["risk"], VERDICT_KO[c["result"]["verdict"]],
                    _gate_cell(f), (f or {}).get("claim", ""), (f or {}).get("evidence", ""),
                    STATUS_KO.get((f or {}).get("status", ""), ""), (f or {}).get("fix", ""),
                    it["utterance"], c["response"], "", "")])


def write_reports(run: Path, out: dict, cases: list[dict], golden: dict) -> None:
    md = render_md(out, cases, golden)
    (run / "report.md").write_text(md, encoding="utf-8")
    (run / "report.html").write_text(render_html(out, cases, golden, md), encoding="utf-8")
    write_csv(run / "defects.csv", cases, golden)
