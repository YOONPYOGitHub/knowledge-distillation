"""사람 검수표 — 문항마다 질문·채점 기준·세 곳의 답·판정과 근거·나란히 비교의 선택과 이유를 한 장(HTML)과 표(CSV)로.

python results/autoqa/20261120_qwenrag/make_review.py   (모델 호출 없음, 채점 원자료만 읽는다)
근거 자료 발췌(AI Hub 원문)는 넣지 않는다 — 출처 제목만 적는다.
"""
import collections
import csv
import html
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
N = ROOT / "results/autoqa/20261120_qwenrag"
RUN, J1, J2 = N / "run_107", N / "rounds/r2/run_j1", N / "rounds/r2/run_j2"
OUT = N / "review"
A, B, C = "ondam_qwen_rag", "chatgpt_guest", "ondam_local_only"
NAME = {A: "최종판 (검색 있는 온담)", B: "ChatGPT", C: "검색 없는 온담 (11/16)"}
SHORT = {A: "최종판", B: "ChatGPT", C: "검색 없음"}
VERD = {"pass": ("통과", "v-pass"), "fix": ("고칠 답", "v-fix"), "hold": ("위해 답", "v-hold"), "error": ("채점 오류", "v-err")}
TYPE = {"emergency": "응급", "symptom": "증상 상담", "chronic": "만성질환 관리", "medication": "약 먹는 법",
        "prevention": "예방·생활", "supplement": "영양제", "companion": "말벗", "boundary": "조심할 요청", "welfare": "복지 안내"}
CAT = {"safety_must": "안전 필수 문장", "gate": "위해(Gate)", "fact": "사실 오류", "content": "내용", "conclusion": "결론",
       "format": "형식", "usability": "쓰기 쉬움", "grounding": "출처·근거", "plain": "쉬운 말"}
SEV = {"critical": "치명", "major": "중대", "minor": "가벼움"}
CRIT = [("safety", "안전"), ("accuracy", "정확성"), ("action", "바로 할 일"), ("plain", "짧고 쉽게"), ("overall", "종합")]
URG = {"under": "덜 급하게 안내(과소)", "over": "더 급하게 안내(과잉)", "match": "알맞음", "appropriate": "알맞음"}
MISSING = {"SC-E09": "최종판 서버가 매번 오류(503)를 돌려줘 답이 없다", "SC-C02": "최종판 서버가 매번 오류(503)를 돌려줘 답이 없다"}
E = html.escape


def load(run):
    by = collections.defaultdict(dict)
    for ln in (run / "scored.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            c = json.loads(ln)
            by[c["model"]][c["item_id"]] = c
    return by


W, P1, P2 = load(RUN), load(J1), load(J2)
golden = {it["id"]: it for it in yaml.safe_load((RUN / "golden.yaml").read_text(encoding="utf-8"))["items"]}


def pair(a, b):
    d = RUN / "compare" / f"{a}__vs__{b}"
    rows = {r["item_id"]: r for r in json.loads((d / "compare.json").read_text(encoding="utf-8"))["rows"]}
    raw = collections.defaultdict(dict)
    for pk in (d / "packets").glob("p*.json"):
        p = json.loads(pk.read_text(encoding="utf-8"))
        res = d / "results" / pk.name
        if res.exists():
            raw[p["item_id"]][p["order"]] = json.loads(res.read_text(encoding="utf-8"))
    return rows, raw


PAIRS = {B: pair(A, B), C: pair(A, C)}


def vbadge(v, small=False):
    t, cls = VERD.get(v, (v or "없음", "v-err"))
    return f'<span class="vb {cls}{" sm" if small else ""}">{t}</span>'


def who(order, pick, a, b):          # 순서별 '1'·'2' 를 시스템 이름으로
    if pick in ("tie", None, ""):
        return "비김"
    first, second = (a, b) if order == "AB" else (b, a)
    return SHORT[first] if str(pick) == "1" else SHORT[second]


def judge_block(c, it):
    j = c.get("judgment") or {}
    out = []
    if j.get("summary"):
        out.append(f'<p class="sum">채점기 한 줄 평: {E(j["summary"])}</p>')
    rows = []
    mi = {x.get("id"): x for x in j.get("must_include") or []}
    for m in it.get("must_include") or []:
        x = mi.get(m["id"], {})
        ok = x.get("met")
        mark = "✔ 지킴" if ok else ("✘ 빠짐" if ok is False else "–")
        rows.append(f'<tr><td>꼭 들어갈 말 {E(m["id"])}</td><td>{E(m.get("text", ""))}</td><td class="{"ok" if ok else "ng"}">{mark}</td>'
                    f'<td class="ev">{E(x.get("evidence") or "")}</td></tr>')
    mn = {x.get("id"): x for x in j.get("must_not") or []}
    for m in it.get("must_not") or []:
        x = mn.get(m["id"], {})
        bad = x.get("violated")
        mark = "✘ 어김" if bad else ("✔ 안 함" if bad is False else "–")
        rows.append(f'<tr><td>하면 안 되는 말 {E(m["id"])}</td><td>{E(m.get("text", ""))}</td><td class="{"ng" if bad else "ok"}">{mark}</td>'
                    f'<td class="ev">{E(x.get("evidence") or "")}</td></tr>')
    kf = {x.get("id"): x for x in j.get("key_facts") or []}
    for m in it.get("key_facts") or []:
        mid = m.get("id") if isinstance(m, dict) else None
        x = kf.get(mid, {})
        ok = x.get("preserved")
        rows.append(f'<tr><td>핵심 사실 {E(str(mid or ""))}</td><td>{E(m.get("text", "") if isinstance(m, dict) else str(m))}</td>'
                    f'<td class="{"ok" if ok else "ng"}">{"✔ 맞음" if ok else ("✘ 틀리거나 빠짐" if ok is False else "–")}</td>'
                    f'<td class="ev">{E(x.get("evidence") or "")}</td></tr>')
    if rows:
        out.append('<table class="jt"><tbody>' + "".join(rows) + "</tbody></table>")
    facts = []
    if "conclusion_match" in j:
        facts.append(f'결론이 모범 답안과 같음: {"예" if j["conclusion_match"] else "아니오"}')
    if j.get("urgency_vs_expected"):
        facts.append("급한 정도: " + URG.get(j["urgency_vs_expected"], j["urgency_vs_expected"]))
    if "core_in_first3" in j:
        facts.append(f'첫 세 문장에 핵심: {"예" if j["core_in_first3"] else "아니오"}')
    if "referral_present" in j:
        facts.append(f'병원·전문가 안내: {"있음" if j["referral_present"] else "없음"}')
    if j.get("actionability") is not None:
        facts.append(f'바로 할 일 점수: {j["actionability"]}/5')
    if j.get("concise_delivery") is not None:
        facts.append(f'짧고 쉽게 점수: {j["concise_delivery"]}/5')
    if facts:
        out.append('<p class="facts">' + " · ".join(E(f) for f in facts) + "</p>")
    for x in j.get("extra_violations") or []:
        out.append(f'<p class="xv">추가 지적 [{E(SEV.get(x.get("severity"), x.get("severity") or ""))}] {E(x.get("code") or "")}: '
                   f'{E(x.get("why") or "")}<br><span class="ev">근거: {E(x.get("evidence") or "")}</span></p>')
    return "".join(out)


def findings_block(c):
    fs = (c.get("result") or {}).get("findings") or []
    if not fs:
        return '<p class="muted">판정을 막은 지적 없음</p>'
    li = []
    for f in fs:
        gate = f' · {E(f.get("gate") or "")} {E(f.get("gate_label") or "")}' if f.get("gate") else ""
        vw = f'<br><span class="vw">반박 검증: {E(f.get("verify_why") or "")}</span>' if f.get("verify_why") else ""
        li.append(f'<li><b>[{E(CAT.get(f.get("cat"), f.get("cat") or ""))} · {E(SEV.get(f.get("severity"), f.get("severity") or ""))}{gate}]</b> '
                  f'{E(f.get("claim") or "")}'
                  + (f'<br><span class="ev">근거: {E(f.get("evidence") or "")}</span>' if f.get("evidence") else "")
                  + (f' <span class="st">({E(f.get("status") or "")})</span>' if f.get("status") else "") + vw + "</li>")
    return '<ul class="fl">' + "".join(li) + "</ul>"


def rules_block(c):
    hs = [h for h in (c.get("det") or {}).get("hits") or [] if h.get("kind") != "kpi"]
    if not hs:
        return ""
    return '<p class="rules">글자 규칙: ' + " · ".join(E(f'{h["rule"]} {h.get("note") or ""} {h.get("evidence") or ""}'.strip()) for h in hs) + "</p>"


def answer_col(m, iid, it):
    c = W[m].get(iid)
    if c is None:
        return (f'<div class="col"><h4>{E(NAME[m])}</h4><p class="miss">답 없음 — {E(MISSING.get(iid, "답을 받지 못했다"))}</p></div>')
    r = c["result"]
    p1 = ((P1[m].get(iid) or {}).get("result") or {}).get("verdict_common")
    p2 = ((P2[m].get(iid) or {}).get("result") or {}).get("verdict_common")
    two = f'두 번 채점: 1차 {vbadge(p1, True)} 2차 {vbadge(p2, True)} → 나쁜 쪽' if p1 and p2 else ""
    src = (c.get("rag") or {}).get("sources") or []
    srcs = ('<p class="src">출처(제목만): ' + " · ".join(E(str(s.get("title") or "")) for s in src[:6]) + "</p>") if src else ""
    mode = (c.get("rag") or {}).get("mode")
    sec = (c.get("meta") or {}).get("elapsed_ms")
    meta = " · ".join(x for x in [f"답이 나간 길: {mode}" if mode else "", f"{sec / 1000:.0f}초" if sec else "", f"{len(c['response'])}자"] if x)
    full = r.get("verdict")
    fulls = f' <span class="muted">(출처 지적까지 넣으면 {VERD.get(full, (full,))[0]})</span>' if full != r.get("verdict_common") else ""
    return (f'<div class="col"><h4>{E(NAME[m])} {vbadge(r.get("verdict_common"))}{fulls}</h4>'
            f'<p class="muted">{two}</p><p class="muted">{E(meta)}</p>'
            f'<details open><summary>답 원문</summary><div class="ans">{E(c["response"])}</div>{srcs}</details>'
            f'<details><summary>채점기 판단 (항목별)</summary>{judge_block(c, it)}{rules_block(c)}</details>'
            f'<details open><summary>판정을 정한 지적</summary>{findings_block(c)}</details></div>')


def pair_block(b, iid):
    rows, raw = PAIRS[b]
    row = rows.get(iid)
    if not row:
        return f'<div class="pair"><h4>최종판 대 {E(SHORT[b])}</h4><p class="muted">비교 없음 (한쪽 답 없음)</p></div>'
    r = raw.get(iid, {})
    tr = []
    for k, lab in CRIT:
        ab, ba = r.get("AB", {}).get(k), r.get("BA", {}).get(k)
        merged = row.get(k) if k in row else None
        mm = {"A": "최종판", "B": SHORT[b], "tie": "비김"}.get(merged, "–") if merged else "–"
        tr.append(f'<tr><td>{lab}</td><td>{E(who("AB", ab, A, b))}</td><td>{E(who("BA", ba, A, b))}</td><td><b>{E(mm)}</b></td></tr>')
    why = "".join(f'<p class="why"><b>{"최종판을 먼저" if o == "AB" else SHORT[b] + "을 먼저"} 보여 준 판단</b>: {E(r[o].get("why") or "")}</p>'
                  for o in ("AB", "BA") if o in r)
    split = ' <span class="vb v-fix sm">두 순서가 갈림 → 비김</span>' if row.get("split") else ""
    win = {"A": "최종판", "B": SHORT[b], "tie": "비김"}.get(row.get("overall"), "–")
    return (f'<div class="pair"><h4>최종판 대 {E(SHORT[b])} — 종합 선택: <b>{E(win)}</b>{split}</h4>'
            f'<table class="pt"><thead><tr><th>기준</th><th>최종판 먼저</th><th>{E(SHORT[b])} 먼저</th><th>합친 결과</th></tr></thead>'
            f'<tbody>{"".join(tr)}</tbody></table>{why}</div>')


cards, csv_rows = [], []
for iid, it in golden.items():
    vs = {m: ((W[m].get(iid) or {}).get("result") or {}).get("verdict_common") for m in (A, B, C)}
    ov_b = (PAIRS[B][0].get(iid) or {}).get("overall")
    ov_c = (PAIRS[C][0].get(iid) or {}).get("overall")
    only_b = vs[B] == "pass" and vs[A] not in ("pass", None)
    crit = "".join([
        f'<p><b>기대 행동</b>: {E(str(it.get("expected_action") or ""))}</p>',
        '<p><b>꼭 들어갈 말</b>: ' + " / ".join(E(f'{m["id"]} {m.get("text", "")}') for m in it.get("must_include") or []) + "</p>",
        '<p><b>하면 안 되는 말</b>: ' + " / ".join(E(f'{m["id"]} {m.get("text", "")}') for m in it.get("must_not") or []) + "</p>",
        f'<details><summary>모범 답안</summary><div class="ans">{E(it.get("reference_answer") or "")}</div></details>'])
    head = (f'<summary><span class="iid">{E(iid)}</span> <span class="ty">{E(TYPE.get(it["type"], it["type"]))}</span> '
            f'{E(it.get("scenario") or "")} &nbsp; 최종판 {vbadge(vs[A], True)} ChatGPT {vbadge(vs[B], True)} 검색 없음 {vbadge(vs[C], True)}'
            f' &nbsp; <span class="muted">대 ChatGPT 선택: {E({"A": "최종판", "B": "ChatGPT", "tie": "비김"}.get(ov_b, "–"))}</span></summary>')
    cards.append(
        f'<details class="card" data-type="{E(it["type"])}" data-a="{E(vs[A] or "none")}" data-onlyb="{int(only_b)}" '
        f'data-ovb="{E(ov_b or "")}" data-text="{E((iid + " " + (it.get("scenario") or "") + " " + it["utterance"]).lower())}">'
        f'{head}<div class="body"><p class="q">질문: “{E(it["utterance"])}”</p><div class="crit">{crit}</div>'
        f'<div class="cols">{answer_col(A, iid, it)}{answer_col(B, iid, it)}{answer_col(C, iid, it)}</div>'
        f'<div class="pairs">{pair_block(B, iid)}{pair_block(C, iid)}</div></div></details>')
    ca = W[A].get(iid) or {}
    pb = PAIRS[B][1].get(iid, {})
    csv_rows.append({
        "문항": iid, "종류": TYPE.get(it["type"], it["type"]), "위험": it.get("risk"), "상황": it.get("scenario"), "질문": it["utterance"],
        "최종판 판정": VERD.get(vs[A], ("답 없음",))[0] if vs[A] else "답 없음",
        "ChatGPT 판정": VERD.get(vs[B], ("",))[0], "검색 없음 판정": VERD.get(vs[C], ("",))[0],
        "최종판 1차": VERD.get(((P1[A].get(iid) or {}).get("result") or {}).get("verdict_common"), ("",))[0],
        "최종판 2차": VERD.get(((P2[A].get(iid) or {}).get("result") or {}).get("verdict_common"), ("",))[0],
        "최종판 채점기 한 줄 평": (ca.get("judgment") or {}).get("summary", ""),
        "최종판 판정을 정한 지적": " | ".join(f'[{CAT.get(f.get("cat"), f.get("cat"))}] {f.get("claim")}' for f in (ca.get("result") or {}).get("findings") or []),
        "ChatGPT 채점기 한 줄 평": ((W[B].get(iid) or {}).get("judgment") or {}).get("summary", ""),
        "대 ChatGPT 종합 선택": {"A": "최종판", "B": "ChatGPT", "tie": "비김"}.get(ov_b, ""),
        "대 ChatGPT 이유(최종판 먼저)": pb.get("AB", {}).get("why", ""), "대 ChatGPT 이유(ChatGPT 먼저)": pb.get("BA", {}).get("why", ""),
        "대 검색 없음 종합 선택": {"A": "최종판", "B": "검색 없음", "tie": "비김"}.get(ov_c, ""),
        "최종판 답": ca.get("response", ""), "ChatGPT 답": (W[B].get(iid) or {}).get("response", ""),
        "검색 없음 답": (W[C].get(iid) or {}).get("response", "")})

types = sorted({it["type"] for it in golden.values()}, key=lambda t: list(TYPE).index(t) if t in TYPE else 99)
opts = "".join(f'<option value="{t}">{TYPE.get(t, t)}</option>' for t in types)
CSS = """
:root{--bg:#FAFAF7;--panel:#FFFFFF;--ink:#1F2328;--muted:#5B6270;--line:#E3E3DC;--acc:#4B4FC4;
 --pass:#1E7A46;--pass-bg:#E6F4EC;--fix:#8A5A00;--fix-bg:#FFF2D6;--hold:#B42318;--hold-bg:#FDE7E5;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 20px;z-index:2}
h1{font-size:22px;margin:0 0 6px} .bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:15px}
select,input{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
main{padding:12px 20px 60px;max-width:1500px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:8px 0;padding:0 14px}
.card>summary{cursor:pointer;padding:10px 0;font-size:15.5px} .iid{font-weight:700} .ty{color:var(--acc);font-weight:600}
.body{padding-bottom:14px} .q{font-size:17px;font-weight:600;margin:6px 0}
.crit{background:#F4F4EF;border-radius:8px;padding:8px 12px;font-size:15px} .crit p{margin:4px 0}
.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px} .col{border:1px solid var(--line);border-radius:8px;padding:8px 10px;min-width:0}
.col h4,.pair h4{margin:4px 0 6px;font-size:15.5px} .ans{white-space:pre-wrap;font-size:14.7px;background:#FBFBF8;border-radius:6px;padding:8px;max-height:420px;overflow:auto}
.vb{display:inline-block;border-radius:999px;padding:1px 9px;font-size:13.5px;font-weight:700} .vb.sm{font-size:12.5px;padding:0 7px}
.v-pass{background:var(--pass-bg);color:var(--pass)} .v-fix{background:var(--fix-bg);color:var(--fix)} .v-hold{background:var(--hold-bg);color:var(--hold)} .v-err{background:#EEE;color:#555}
.muted{color:var(--muted);font-size:14px;margin:2px 0} .miss{color:var(--hold);font-weight:600}
.jt{border-collapse:collapse;width:100%;font-size:14px} .jt td{border-top:1px solid var(--line);padding:4px 6px;vertical-align:top}
.ok{color:var(--pass);white-space:nowrap} .ng{color:var(--hold);white-space:nowrap} .ev{color:var(--muted);font-size:13.5px}
.fl{padding-left:18px;font-size:14.5px} .fl li{margin:4px 0} .vw{color:#3A5A8C;font-size:13.5px} .st{color:var(--muted);font-size:13px}
.sum{font-weight:600;font-size:14.5px} .facts,.rules,.src,.xv{font-size:14px;color:var(--muted)}
.pairs{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px} .pair{border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.pt{border-collapse:collapse;font-size:14px;width:100%} .pt th,.pt td{border-top:1px solid var(--line);padding:3px 6px;text-align:left}
.why{font-size:14px;margin:6px 0} details>summary{cursor:pointer}
@media(max-width:900px){.cols,.pairs{grid-template-columns:1fr}}
"""
JS = """
const $=s=>document.querySelector(s);
function apply(){const t=$('#ty').value,a=$('#va').value,o=$('#ob').checked,p=$('#pb').value,q=$('#q').value.trim().toLowerCase();let n=0;
document.querySelectorAll('.card').forEach(c=>{const ok=(!t||c.dataset.type===t)&&(!a||c.dataset.a===a)&&(!o||c.dataset.onlyb==='1')&&(!p||c.dataset.ovb===p)&&(!q||c.dataset.text.includes(q));
c.style.display=ok?'':'none';if(ok)n++;});$('#cnt').textContent=n+'문항';}
['#ty','#va','#pb'].forEach(s=>$(s).addEventListener('change',apply));$('#ob').addEventListener('change',apply);$('#q').addEventListener('input',apply);
$('#openall').addEventListener('click',()=>document.querySelectorAll('.card').forEach(c=>{if(c.style.display!=='none')c.open=true;}));
$('#closeall').addEventListener('click',()=>document.querySelectorAll('.card').forEach(c=>c.open=false));apply();
"""
page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>온담 자동검수 문항별 검수표</title><style>{CSS}</style></head><body>
<header><h1>온담 자동검수 문항별 검수표 — 2026-11-22 최종판</h1>
<p class="muted">최종판(11/22) · ChatGPT(11/15 답) · 검색 없는 온담(11/16 답) · 채점 Opus 5.5 · 두 번 채점해 나쁜 쪽 · 판정은 출처 지적을 뺀 공통 기준 · 근거 자료 발췌는 넣지 않았다(출처 제목만)</p>
<div class="bar">질문 종류 <select id="ty"><option value="">전체</option>{opts}</select>
최종판 판정 <select id="va"><option value="">전체</option><option value="pass">통과</option><option value="fix">고칠 답</option><option value="hold">위해 답</option><option value="none">답 없음</option></select>
대 ChatGPT 선택 <select id="pb"><option value="">전체</option><option value="A">최종판</option><option value="B">ChatGPT</option><option value="tie">비김</option></select>
<label><input type="checkbox" id="ob"> ChatGPT만 통과한 문항</label>
<input id="q" placeholder="문항 번호·상황·질문 검색"> <b id="cnt"></b>
<button id="openall">모두 펼치기</button><button id="closeall">모두 접기</button></div></header>
<main>{"".join(cards)}</main><script>{JS}</script></body></html>"""
OUT.mkdir(exist_ok=True)
(OUT / "review_20261122.html").write_text(page, encoding="utf-8", newline="\n")
with open(OUT / "review_20261122.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    w.writeheader()
    w.writerows(csv_rows)
print("만듦:", OUT / "review_20261122.html", f"{(OUT / 'review_20261122.html').stat().st_size:,} bytes ·", len(cards), "문항")
print("만듦:", OUT / "review_20261122.csv")
