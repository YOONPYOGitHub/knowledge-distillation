"""autoqa 발표 자료(HTML 한 파일)를 만든다 — 숫자는 talk_data.json 에서만 읽는다.

만들려면:
    python docs/autoqa/talk/build_talk.py

- 온담을 고쳐 다시 잰 뒤에는 talk_data.json 만 바꾸고 다시 만든다 (본문에 숫자를 손으로 적지 않는다).
- 한 화면에 한 가지 말만 한다. 화면마다 위에서 아래로: 자리 표시 → 제목 → 그림 → 결론 한 줄 또는 출처.
- 색은 무채 세 단계 + 강조 한 가지. 온담·지금·결론에만 강조색을 쓴다.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "talk_data.json").read_text(encoding="utf-8"))
E = html.escape
LEDGER = HERE.parents[2] / "silvercare_autoqa" / "data" / "knowhow_ledger.yaml"


def ledger_counts() -> dict:
    """원칙 장부에서 상태별·묶음별 수를 센다 (applied + today = 들어 있는 것)."""
    import yaml
    doc = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    items = doc["items"]
    out = {s: sum(1 for i in items if i["status"] == s) for s in ("applied", "today", "next", "na")}
    out["total"] = len(items)
    out["families"] = []
    for key, name in doc["families"].items():
        mine = [i for i in items if i["family"] == key and i["status"] != "na"]
        out["families"].append({"name": name, "n": len(mine), "done": sum(1 for i in mine if i["status"] in ("applied", "today"))})
    return out

CSS = """
:root{
  --bg:#0A0A0C; --surface:#15161B; --surface-accent:#191A2A; --border:#2A2B33; --divider:rgba(244,244,247,.09);
  --text:#F4F4F7; --muted:#B4B6C0; --dim:#9EA0AC; --accent:#6C6FE0; --accent-text:#AEB0F6;
  --tie:#34363F; --other:#A9ABB6; --track:#22232B;
  --edge:clamp(1.25rem,5vw,4.5rem);
}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:"Segoe UI","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;
  font-size:18px;line-height:1.55;-webkit-font-smoothing:antialiased;word-break:keep-all;}
p,td,th,li{text-wrap:pretty;}
h1,h2,h3{text-wrap:balance;}
.deck{display:flex;flex-direction:column;}
.view{min-height:100vh;display:flex;flex-direction:column;justify-content:center;
  padding:clamp(2rem,5vh,3.5rem) var(--edge);gap:1.35rem;border-top:1px solid var(--border);}
.view:first-child{border-top:none;}
.kicker{margin:0;font-size:19px;font-weight:700;letter-spacing:.2em;color:var(--accent-text);}
.headline{margin:0;font-weight:800;line-height:1.18;color:var(--text);font-size:clamp(32px,4vw,52px);max-width:24ch;}
.headline.wide{max-width:30ch;}
.lead{margin:0;color:var(--muted);font-size:clamp(20px,1.9vw,24px);max-width:52ch;}
.takeaway{margin:0;font-weight:600;color:var(--accent-text);font-size:clamp(21px,2.2vw,26px);max-width:48ch;line-height:1.4;}
.source{margin:0;color:var(--dim);font-size:18px;}
.divider-motif{height:6px;width:120px;border-radius:999px;background:linear-gradient(90deg,var(--accent),var(--accent-text));}

.cover .headline{font-size:clamp(40px,5.6vw,72px);max-width:18ch;}
.cover .who{margin:0;color:var(--muted);font-size:22px;}

.card{background:var(--surface);border:1px solid var(--border);border-radius:18px;
  padding:clamp(1.1rem,2.2vw,1.6rem);display:flex;flex-direction:column;gap:.6rem;}
.card.is-focus{background:var(--surface-accent);border-color:var(--accent);}
.card .c-title{margin:0;font-size:clamp(22px,2.2vw,28px);font-weight:700;color:var(--text);}
.card.is-focus .c-title{color:var(--accent-text);}
.card .c-sub{margin:0;color:var(--muted);font-size:19px;}
.card .c-rule{height:1px;background:var(--divider);margin:.2rem 0;}
.card .c-label{margin:0;color:var(--dim);font-size:18px;}
.card .c-value{margin:0;color:var(--text);font-weight:700;font-size:21px;}
.pill{display:inline-block;padding:.28rem .85rem;border-radius:999px;font-size:18px;font-weight:600;align-self:flex-start;}
.pill-ghost{border:1px solid var(--border);color:var(--muted);}
.pill-accent{background:var(--accent);color:#fff;}
.pill.mini{font-size:14.7px;padding:2px 10px;margin-left:6px;vertical-align:middle;}
.pills{display:flex;flex-wrap:wrap;gap:.6rem;}

.grid{display:grid;gap:clamp(.9rem,1.8vw,1.5rem);}
.g2{grid-template-columns:repeat(2,1fr);} .g3{grid-template-columns:repeat(3,1fr);} .g4{grid-template-columns:repeat(4,1fr);}
@media(max-width:900px){.g2,.g3,.g4{grid-template-columns:1fr;}}
.grid>*{min-width:0;}

.numcard .n{font-size:clamp(36px,4.4vw,50px);font-weight:800;color:var(--accent-text);line-height:1;}
.stat .big{font-size:clamp(50px,6.6vw,84px);font-weight:800;line-height:1;color:var(--text);}
.stat.is-focus .big{color:var(--accent-text);}
.stat .cap{margin:0;color:var(--muted);font-size:19px;}
.stat .unit{font-size:clamp(20px,2.4vw,30px);font-weight:700;margin-left:.15em;color:var(--muted);}

.fromto{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:clamp(.75rem,2vw,1.5rem);}
.fromto .arrow{font-size:clamp(30px,3.6vw,44px);color:var(--accent-text);text-align:center;}
.col-head{margin:0 0 .5rem;font-size:18px;letter-spacing:.1em;}
.col-from{color:var(--muted);} .col-to{color:var(--accent-text);}
@media(max-width:900px){.fromto{grid-template-columns:1fr;}.fromto .arrow{transform:rotate(90deg);}}

/* 세 조건 점수판 — 이름을 숫자 바로 위에 붙이고, 세 카드의 줄(상태·조건·숫자·표시·한 줄)을 서로 맞춘다 */
.goal{display:grid;grid-template-rows:subgrid;grid-row:span 5;row-gap:.6rem;
  padding:clamp(.9rem,1.7vw,1.3rem) clamp(1rem,2vw,1.5rem);}
.goal .pill{justify-self:start;}
.goal .c-title{font-size:clamp(19px,1.75vw,23px);line-height:1.3;color:var(--text);}
.goal .c-sub{text-wrap:balance;} .goal .c-sub b{display:block;color:var(--text);font-weight:700;}
.vs{display:grid;grid-template-rows:auto auto;grid-auto-flow:column;grid-template-columns:1fr 1fr;gap:0 1rem;align-items:baseline;}
.vs.has-tie{grid-template-columns:1fr 1fr auto;}
.vs .who{margin:0;color:var(--muted);font-size:18px;line-height:1.4;white-space:nowrap;}
.vs .who .sw{margin-right:.4rem;vertical-align:-1px;}
.vs .num{margin:0;font-size:clamp(40px,4.6vw,62px);font-weight:800;line-height:1.05;color:var(--text);white-space:nowrap;}
.vs .num .unit{font-size:.36em;font-weight:700;margin-left:.12em;color:var(--muted);}
.vs .num.sm{font-size:clamp(24px,2.6vw,34px);color:var(--muted);}
.marks{display:grid;grid-template-columns:1fr 1fr;gap:0 1rem;align-items:center;min-height:12px;}
.marks.one{grid-template-columns:1fr;}
.marks .track{height:10px;} .marks .split{height:10px;} .marks .seg{padding:0;}
.dots{display:flex;gap:6px;}
.dot{width:12px;height:12px;border-radius:999px;background:var(--other);}
.dot.me{background:var(--accent);}

/* 막대 — 굵기 24px 이하, 값 쪽 끝만 둥글게, 붙은 조각 사이는 2px 틈 */
.bars{display:flex;flex-direction:column;gap:1rem;}
.brow{display:grid;grid-template-columns:minmax(7.5em,12em) 1fr minmax(6.5em,auto);align-items:center;gap:1rem;}
.brow .b-name{color:var(--text);font-weight:600;font-size:20px;}
.brow .b-val{color:var(--text);font-size:20px;font-weight:700;text-align:right;white-space:nowrap;}
.brow .b-val small{color:var(--muted);font-size:18px;font-weight:500;}
.track{height:22px;background:var(--track);border-radius:0 4px 4px 0;display:flex;}
.fill{height:100%;border-radius:0 4px 4px 0;background:var(--other);}
.fill.is-accent{background:var(--accent);}
.fill:hover,.seg:hover{filter:brightness(1.15);}
.split{display:flex;gap:2px;height:28px;}
.seg{display:flex;align-items:center;justify-content:flex-start;padding:0 .5rem;font-size:18px;font-weight:700;
  border-radius:4px;min-width:0;white-space:nowrap;overflow:visible;}
.seg.s-win{background:var(--accent);color:#fff;}
.seg.s-tie{background:var(--tie);color:var(--muted);}
.seg.s-loss{background:var(--other);color:#0A0A0C;}
.seg.s-pass{background:var(--tie);color:var(--text);}
.seg.s-fix{background:var(--other);color:#0A0A0C;}
.seg.s-hold{background:var(--accent);color:#fff;}
.legend{display:flex;flex-wrap:wrap;gap:1.2rem;color:var(--muted);font-size:18px;margin:0;padding:0;list-style:none;}
.legend li{display:flex;align-items:center;gap:.45rem;}
.brow .own{color:var(--dim);font-size:18px;font-weight:500;}
.brow .gain{color:var(--accent-text);font-weight:700;margin-right:.5em;}
.bars.tight{gap:.55rem;} .bars.tight .track{height:16px;}
.bars.names .brow{grid-template-columns:minmax(12em,19em) 1fr minmax(5em,auto);}
.fromnow{white-space:nowrap;}
.g3 .fromnow .now,.g3 .fromnow .goal{font-size:clamp(28px,3.1vw,42px);}
.grid.tight .card{padding:.9rem 1.3rem;gap:.3rem;} .grid.tight .stat .big{font-size:clamp(40px,4.6vw,60px);}
.takeaway{text-wrap:balance;}
.sw{width:14px;height:14px;border-radius:3px;display:inline-block;}
.sw.win,.sw.hold{background:var(--accent);} .sw.tie,.sw.pass{background:var(--tie);} .sw.loss,.sw.fix{background:var(--other);}
.seg.s-open{background:transparent;border:1.5px solid var(--other);color:var(--muted);} .sw.open{border:1.5px solid var(--other);}

details.tv{color:var(--muted);font-size:18px;}
details.tv summary{cursor:pointer;color:var(--dim);}
details.tv table{border-collapse:collapse;margin-top:.6rem;font-size:18px;}
details.tv th,details.tv td{border-bottom:1px solid var(--border);padding:.35rem .9rem .35rem 0;text-align:left;}
details.tv th{color:var(--dim);font-weight:600;}

.dgm{overflow-x:auto;border:1px solid var(--border);border-radius:18px;padding:10px;background:var(--surface);}
.dgm svg{width:100%;height:auto;min-width:1180px;display:block;}
.dgm svg text{font-family:"Segoe UI","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;}
.t-title{fill:var(--text);font-weight:700;} .t-body{fill:var(--muted);} .t-dim{fill:var(--dim);}
.t-on-accent{fill:#fff;font-weight:700;} .t-accent{fill:var(--accent-text);font-weight:700;}
.f-box{fill:var(--bg);stroke:var(--border);} .f-soft{fill:var(--track);} .f-accent{fill:var(--accent);}
.f-other{fill:var(--other);} .f-tie{fill:var(--tie);} .f-gate{fill:#23245C;stroke:var(--accent);}
.s-spine{stroke:var(--border);} .s-accent{stroke:var(--accent);} .s-muted{stroke:var(--dim);}
.f-none{fill:none;}

.fromnow{display:flex;align-items:baseline;gap:.5rem;flex-wrap:wrap;}
.fromnow .now{font-size:clamp(34px,4vw,52px);font-weight:800;color:var(--text);line-height:1;}
.fromnow .arr{font-size:clamp(24px,2.6vw,34px);color:var(--accent-text);}
.fromnow .goal{font-size:clamp(34px,4vw,52px);font-weight:800;color:var(--accent-text);line-height:1;}
.quote{border-left:4px solid var(--accent);padding:.2rem 0 .2rem 1.2rem;margin:0;}
.quote p{margin:0;color:var(--text);font-size:clamp(23px,2.4vw,29px);line-height:1.5;max-width:42ch;}
/* 흐름 그림은 창 높이에 맞춰 줄인다 — 넓고 낮은 창에서 그림이 화면 밖으로 길어지지 않게 (270px = 제목·결론 줄·틈) */
@media(min-width:901px){.dgm svg{max-height:calc(100vh - 2*clamp(2rem,5vh,3.5rem) - 270px);}}
/* 창을 다 채우지 않고 띄운 낮은 화면 — 한 화면 안에 들어오게 여백과 큰 글자를 조금 줄인다 */
@media(max-height:800px) and (min-width:901px){
  .view{padding:1.1rem var(--edge);gap:.85rem;}
  .headline{font-size:clamp(30px,3.4vw,44px);}
  .stat .big{font-size:clamp(44px,5.2vw,66px);}
  .card{padding:1rem 1.3rem;gap:.45rem;}
  .goal{padding:.9rem 1.3rem;row-gap:.4rem;} .vs .num{font-size:clamp(36px,4vw,54px);}
  .dgm svg{max-height:calc(100vh - 270px);}
}
/* 더 낮은 화면 (1280×720 화면에 창으로 띄운 때) */
@media(max-height:660px) and (min-width:901px){
  .view{padding:.7rem var(--edge);gap:.6rem;}
  .headline{font-size:clamp(26px,3vw,38px);}
  .dgm svg{max-height:calc(100vh - 235px);}
}
/* 좁은 화면(휴대폰) — 이름과 숫자를 윗줄에, 막대를 아랫줄에 둔다 */
@media(max-width:900px){
  .brow,.bars.names .brow{grid-template-columns:1fr auto;grid-template-areas:"name val" "bar bar";gap:.3rem .8rem;}
  .brow .b-name{grid-area:name;} .brow .b-val{grid-area:val;} .brow .track,.brow .split{grid-area:bar;}
  .bars{gap:1.1rem;} .legend{gap:.5rem 1rem;}
}
@media print{.view{min-height:auto;break-after:page;padding:2rem;}}
"""

JS = """
(function(){
  var views=[].slice.call(document.querySelectorAll('.view'));
  function cur(){var y=window.scrollY+window.innerHeight/3,i=0;views.forEach(function(v,k){if(v.offsetTop<=y)i=k;});return i;}
  function go(i,fast){i=Math.max(0,Math.min(views.length-1,i));views[i].scrollIntoView({behavior:fast?'auto':'smooth',block:'start'});}
  document.addEventListener('keydown',function(e){
    if(e.target&&/input|textarea|summary/i.test(e.target.tagName))return;
    if(['ArrowDown','ArrowRight','PageDown',' '].indexOf(e.key)>=0){e.preventDefault();go(cur()+1);}
    if(['ArrowUp','ArrowLeft','PageUp'].indexOf(e.key)>=0){e.preventDefault();go(cur()-1);}
    if(e.key==='Home'){e.preventDefault();go(0);} if(e.key==='End'){e.preventDefault();go(views.length-1);}
  });
  var q=new URLSearchParams(location.search);
  var jump=q.get('v'), qa=q.get('qa')||window.__QA;                   // window.__V·__QA 는 화면 점검 도구가 넣는다
  if(jump!==null){document.documentElement.style.scrollBehavior='auto';go(parseInt(jump,10)||0,true);}
  if(window.__V!==undefined){views.forEach(function(v,k){if(k!==window.__V)v.style.display='none';});}   // 점검: 그 화면만 남긴다
  if(qa){                     // 화면 점검용: 한 화면을 넘치는 뷰와 그림 밖으로 나간 글자를 적는다
    var out=[];
    views.forEach(function(v,k){
      if(v.scrollHeight>window.innerHeight+2)out.push('view '+k+' height '+v.scrollHeight+' > '+window.innerHeight);
      if(v.scrollWidth>v.clientWidth+2)out.push('view '+k+' wider than screen');
    });
    [].forEach.call(document.querySelectorAll('.seg'),function(s){if(s.scrollWidth>s.clientWidth+1)out.push('seg label overflow: '+s.textContent);});
    [].forEach.call(document.querySelectorAll('svg [data-box]'),function(g){
      var b=g.querySelector('rect').getBBox();
      [].forEach.call(g.querySelectorAll('text'),function(t){var tb=t.getBBox();
        if(tb.x<b.x+6||tb.x+tb.width>b.x+b.width-6)out.push('svg text outside box: '+t.textContent+' ['+Math.round(tb.x)+','+Math.round(tb.x+tb.width)+'] box ['+b.x+','+(b.x+b.width)+']');});
    });
    var pre=document.createElement('pre');pre.id='qa';
    pre.textContent=(out.length?out.join('\\n'):'QA OK')+'\\n(views '+views.length+', '+window.innerWidth+'x'+window.innerHeight+')';
    pre.style.cssText='font-size:18px;color:#F4F4F7;padding:24px;white-space:pre-wrap;margin:0';
    views.forEach(function(v){v.style.display='none';});document.body.insertBefore(pre,document.body.firstChild);
  }
})();
"""


# ───────────────────────── 조각 ─────────────────────────
def nb(s: str) -> str:
    """숫자와 단위 사이가 줄 끝에서 갈라지지 않게 붙는 공백으로 바꾼다."""
    return s.replace("{nb}", "&nbsp;")


def view(kicker: str | None, headline: str, body: str, *, takeaway: str = "", source: str = "", lead: str = "",
         cls: str = "", label: str = "", wide: bool = False) -> str:
    name = re.sub(r"<[^>]+>", " ", label or headline).replace("{nb}", " ")
    parts = [f'<section class="view {cls}" aria-label="{E(" ".join(name.split()))}">']
    if cls == "cover":
        parts.append('<span class="divider-motif" aria-hidden="true"></span>')
    if kicker:
        parts.append(f'<p class="kicker">{E(kicker)}</p>')
    parts.append(f'<h2 class="headline{" wide" if wide else ""}">{nb(headline)}</h2>')
    if lead:
        parts.append(f'<p class="lead">{nb(lead)}</p>')
    parts.append(body)
    if takeaway:
        parts.append(f'<p class="takeaway">{nb(takeaway)}</p>')
    if source:
        parts.append(f'<p class="source">{nb(source)}</p>')
    parts.append("</section>")
    return "\n".join(parts)


def card(title: str, sub: str = "", *, focus: bool = False, pill: str = "", label: str = "", value: str = "",
         num: str = "") -> str:
    cls = "card" + (" is-focus" if focus else "") + (" numcard" if num else "")
    out = [f'<article class="{cls}">']
    if num:
        out.append(f'<div class="n">{E(num)}</div>')
    if pill:
        out.append(f'<span class="pill {"pill-accent" if focus else "pill-ghost"}">{E(pill)}</span>')
    out.append(f'<h3 class="c-title">{nb(title)}</h3>')
    if sub:
        out.append(f'<p class="c-sub">{nb(sub)}</p>')
    if label:
        out.append(f'<div class="c-rule"></div><p class="c-label">{nb(label)}</p><p class="c-value">{nb(value)}</p>')
    out.append("</article>")
    return "".join(out)


def stat(big: str, cap: str, *, unit: str = "", focus: bool = False, note: str = "") -> str:
    u = f'<span class="unit">{E(unit)}</span>' if unit else ""
    extra = f'<p class="c-sub">{nb(note)}</p>' if note else ""
    return (f'<article class="card stat{" is-focus" if focus else ""}"><div class="big">{E(big)}{u}</div>'
            f'<p class="cap">{nb(cap)}</p>{extra}</article>')


def table_view(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{E(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{E(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<details class="tv"><summary>표로 보기</summary><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></details>'


def meter_rows(rows: list[dict]) -> str:
    """이름 · 막대 · 숫자. rows: name, k, n, accent"""
    out = ['<div class="bars" role="img" aria-label="' + E(" / ".join(f"{r['name']} {r['k']}/{r['n']}" for r in rows)) + '">']
    for r in rows:
        pct = 100 * r["k"] / r["n"]
        out.append(
            f'<div class="brow"><span class="b-name">{E(r["name"])}</span>'
            f'<div class="track"><div class="fill{" is-accent" if r.get("accent") else ""}" style="width:{pct:.1f}%" '
            f'title="{E(r["name"])} {r["k"]}/{r["n"]} ({pct:.0f}%)"></div></div>'
            f'<span class="b-val">{pct:.0f}% <small>{r["k"]}/{r["n"]}</small></span></div>')
    out.append("</div>")
    return "".join(out)


def split_row(name: str, parts: list[tuple[str, int, str]], total: int, right: str = "") -> str:
    """한 줄을 조각으로 나눈 막대. parts: (종류 class, 값, 이름). 좁은 조각에는 글자를 넣지 않는다."""
    segs = []
    for cls, v, lab in parts:
        if v <= 0:
            continue
        share = v / total
        txt = str(v) if share >= 0.075 else ""
        segs.append(f'<div class="seg {cls}" style="flex:{v} 1 0" title="{E(lab)} {v}/{total}">{txt}</div>')
    return (f'<div class="brow"><span class="b-name">{E(name)}</span><div class="split">{"".join(segs)}</div>'
            f'<span class="b-val">{right}</span></div>')


def legend(items: list[tuple[str, str]]) -> str:
    return '<ul class="legend">' + "".join(f'<li><span class="sw {c}"></span>{E(t)}</li>' for c, t in items) + "</ul>"


def versus(a: str, b: str, name_b: str, unit: str = "", tie: str = "") -> str:
    """온담 대 비교 상대 — 이름을 숫자 바로 위에 붙여 어느 쪽 숫자인지 한눈에 보이게 한다."""
    u = f'<span class="unit">{E(unit)}</span>' if unit else ""
    t = f'<p class="who">비김</p><p class="num sm">{E(tie)}</p>' if tie else ""
    return (f'<div class="vs{" has-tie" if tie else ""}">'
            f'<p class="who"><span class="sw win"></span>온담</p><p class="num">{E(a)}{u}</p>'
            f'<p class="who"><span class="sw loss"></span>{E(name_b)}</p><p class="num">{E(b)}{u}</p>{t}</div>')


def goal_card(met: bool, title: str, numbers: str, marks: str, why: str) -> str:
    """조건 하나 — 위에서 아래로: 넘었는지 · 조건 · 두 쪽 숫자 · 작은 표시 · 기준과 견준 한 줄."""
    pill = '<span class="pill pill-accent">넘었습니다</span>' if met else '<span class="pill pill-ghost">아직입니다</span>'
    return (f'<article class="card goal{" is-focus" if met else ""}">{pill}<h3 class="c-title">{nb(title)}</h3>'
            f'{numbers}{marks}<p class="c-sub">{nb(why)}</p></article>')


def pts(x: float) -> str:
    """비율 차이를 %p 로. 1%p 가 안 되는 값만 소수 한 자리까지 적는다."""
    v = abs(x) * 100
    return f"{v:.0f}%p" if v >= 0.95 else f"{v:.1f}%p"


def fromnow(now: str, goal: str) -> str:
    return (f'<div class="fromnow"><span class="now">{E(now)}</span><span class="arr" aria-hidden="true">→</span>'
            f'<span class="goal">{E(goal)}</span></div>')


# ───────────────────────── 흐름 그림 (SVG 하나) ─────────────────────────
def pipeline_svg() -> str:
    W, H = 1360, 470
    bw, gap, x0, top, bh = 148, 20, 18, 34, 300
    X = [x0 + i * (bw + gap) for i in range(8)]
    mid = top + bh / 2
    judge = D["meta"]["judge"]                       # 예: Claude Sonnet 5.5
    j1, j2 = judge.split(" ", 1) if " " in judge else (judge, "")
    o: list[str] = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="aspect-ratio:{W}/{H}" role="img" '
        f'aria-label="질문 109개에서 답 받기, 글자 규칙, 기준표 채점, 반박 검증, 판정, 나란히 비교를 거쳐 세 조건을 확인하는 흐름">',
        '<defs><marker id="mkflow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0 0 L10 5 L0 10 z" class="f-accent"/></marker>'
        '<marker id="mkback" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0 0 L10 5 L0 10 z" style="fill:var(--dim)"/></marker></defs>',
        f'<line x1="{x0}" y1="{mid}" x2="{W - x0}" y2="{mid}" class="s-spine" stroke-width="10" stroke-linecap="round"/>',
    ]

    def box(i: int, t1: str, t2: str = "") -> tuple[float, float]:
        x = X[i]
        o.append(f'<g data-box="{i}"><rect x="{x}" y="{top}" width="{bw}" height="{bh}" rx="16" class="f-box" stroke-width="1.5"/>')
        o.append(f'<text x="{x + 14}" y="{top + 34}" font-size="19" class="t-title">{E(t1)}</text>')
        if t2:
            o.append(f'<text x="{x + 14}" y="{top + 58}" font-size="19" class="t-title">{E(t2)}</text>')
        return x, top

    def caps(i: int, lines: list[str]) -> None:
        for k, ln in enumerate(lines):
            o.append(f'<text x="{X[i] + bw / 2}" y="{top + 214 + k * 22}" font-size="16" text-anchor="middle" class="t-body">{E(ln)}</text>')

    def badge(i: int, lines: list[str], model: bool) -> None:
        x, y, h = X[i] + 12, top + bh - 14 - (22 * len(lines) + 12), 22 * len(lines) + 12
        cls = 'style="fill:var(--surface-accent);stroke:var(--accent)"' if model else 'style="fill:var(--track);stroke:var(--border)"'
        o.append(f'<rect x="{x}" y="{y}" width="{bw - 24}" height="{h}" rx="12" {cls} stroke-width="1.2"/>')
        for k, ln in enumerate(lines):
            o.append(f'<text x="{X[i] + bw / 2}" y="{y + 23 + k * 22}" font-size="16" text-anchor="middle" '
                     f'class="{"t-accent" if model else "t-body"}">{E(ln)}</text>')

    def close() -> None:
        o.append("</g>")

    gy = top + 78          # 그림 영역 위쪽
    # 0 질문 묶음 — 겹친 종이
    x, _ = box(0, "질문 109개")
    for k in range(3):
        o.append(f'<rect x="{x + 34 + k * 9}" y="{gy + 4 + k * 9}" width="62" height="78" rx="6" class="f-soft" stroke="var(--border)" stroke-width="1.2"/>')
    for k in range(4):
        o.append(f'<line x1="{x + 62}" y1="{gy + 40 + k * 13}" x2="{x + 104}" y2="{gy + 40 + k * 13}" class="s-muted" stroke-width="2"/>')
    caps(0, ["필수답·금지답", "모범 답안"])
    close()
    # 1 답 받기 — 두 줄 화살표
    x, _ = box(1, "① 답 받기")
    for k, (lab, cls, tcls) in enumerate([("온담", "f-accent", "t-on-accent"), ("ChatGPT", "f-other", "")]):
        yy = gy + 6 + k * 52
        o.append(f'<path d="M{x + 16} {yy} h{bw - 52} l20 20 l-20 20 h-{bw - 52} z" class="{cls}"/>')
        style = "" if tcls else ' style="fill:#0A0A0C;font-weight:700"'
        o.append(f'<text x="{x + 26}" y="{yy + 26}" font-size="17" class="{tcls}"{style}>{lab}</text>')
    caps(1, ["같은 질문을", "두 곳에 묻는다"])
    close()
    # 2 글자 규칙 — 점검 목록
    x, _ = box(2, "② 글자 규칙")
    for k, lab in enumerate(["119 빠짐", "없는 출처", "형식 고장"]):
        yy = gy + 4 + k * 34
        o.append(f'<rect x="{x + 18}" y="{yy}" width="20" height="20" rx="4" class="f-none s-muted" stroke-width="2"/>')
        o.append(f'<path d="M{x + 22} {yy + 10} l5 5 l8 -10" class="f-none s-accent" stroke-width="2.5" stroke-linecap="round"/>')
        o.append(f'<text x="{x + 48}" y="{yy + 17}" font-size="17" class="t-body">{lab}</text>')
    badge(2, ["규칙으로 확정", "모델 안 씀"], False)
    close()
    # 3 기준표 채점 — 기준표 문서
    x, _ = box(3, "③ 기준표", "채점")
    o.append(f'<rect x="{x + 18}" y="{gy + 6}" width="{bw - 36}" height="92" rx="8" class="f-soft" stroke="var(--border)" stroke-width="1.2"/>')
    for k, lab in enumerate(["필수답", "금지답", "근거 자료"]):
        yy = gy + 30 + k * 26
        o.append(f'<circle cx="{x + 32}" cy="{yy - 6}" r="4" class="f-accent"/>')
        o.append(f'<text x="{x + 44}" y="{yy}" font-size="16" class="t-body">{lab}</text>')
    caps(3, ["두 번 채점"])
    badge(3, [j1, j2] if j2 else [j1], True)
    close()
    # 4 반박 검증 — 되돌아보는 고리
    x, _ = box(4, "④ 반박 검증")
    cx, cy = x + bw / 2, gy + 50
    o.append(f'<path d="M{cx + 34} {cy - 14} A36 36 0 1 0 {cx + 36} {cy + 8}" class="f-none s-accent" stroke-width="4" stroke-linecap="round" marker-end="url(#mkflow)"/>')
    o.append(f'<text x="{cx}" y="{cy + 7}" font-size="18" text-anchor="middle" class="t-title">Hold?</text>')
    o.append(f'<text x="{cx}" y="{top + 196}" font-size="16" text-anchor="middle" class="t-body">Hold 지적만 다시</text>')
    badge(4, [j1, j2] if j2 else [j1], True)
    close()
    # 5 판정 — 깊어지는 띠
    x, _ = box(5, "⑤ 판정")
    for k, (lab, cls, tcls) in enumerate([("Pass", "f-tie", "t-title"), ("Fix", "f-other", ""), ("Hold", "f-accent", "t-on-accent")]):
        yy = gy + 2 + k * 34
        o.append(f'<rect x="{x + 18 + k * 8}" y="{yy}" width="{bw - 36 - k * 8}" height="30" rx="6" class="{cls}"/>')
        style = "" if tcls else ' style="fill:#0A0A0C;font-weight:700"'
        o.append(f'<text x="{x + 30 + k * 8}" y="{yy + 21}" font-size="17" class="{tcls}"{style}>{lab}</text>')
    badge(5, ["규칙으로 확정", "모델 안 씀"], False)
    close()
    # 6 나란히 비교 — 이름 가린 두 답
    x, _ = box(6, "⑥ 나란히", "비교")
    for k, lab in enumerate(["답 1", "답 2"]):
        xx = x + 16 + k * 62
        o.append(f'<rect x="{xx}" y="{gy + 10}" width="54" height="66" rx="8" class="f-soft" stroke="var(--border)" stroke-width="1.2"/>')
        o.append(f'<text x="{xx + 27}" y="{gy + 50}" font-size="16" text-anchor="middle" class="t-body">{lab}</text>')
    o.append(f'<text x="{x + bw / 2}" y="{top + 196}" font-size="16" text-anchor="middle" class="t-body">이름 가림·두 번</text>')
    badge(6, [j1, j2] if j2 else [j1], True)
    close()
    # 7 출구 관문 — 진한 문
    x = X[7]
    o.append(f'<g data-box="7"><rect x="{x}" y="{top - 14}" width="{bw}" height="{bh + 28}" rx="16" class="f-gate" stroke-width="2"/>')
    o.append(f'<text x="{x + bw / 2}" y="{top + 24}" font-size="19" text-anchor="middle" class="t-on-accent">밀리지 않는다</text>')
    o.append(f'<text x="{x + bw / 2}" y="{top + 48}" font-size="16" text-anchor="middle" style="fill:var(--accent-text)">세 조건 모두</text>')
    tg = D["targets"]                             # 기준값은 비교 결과에 적힌 것 (세 조건 화면과 같은 값)
    for k, (a, b) in enumerate([("위해 답", f'{tg["hold"]["goal"]}건'), ("통과율 차이", f'{pts(tg["pass"]["margin"])} 안'),
                                ("고른 비율", f'{tg["pref"]["floor"] * 100:.0f}% 이상')]):
        yy = top + 100 + k * 70
        o.append(f'<line x1="{x + 24}" y1="{yy - 28}" x2="{x + bw - 24}" y2="{yy - 28}" style="stroke:rgba(244,244,247,.18)" stroke-width="1"/>')
        o.append(f'<text x="{x + bw / 2}" y="{yy}" font-size="16" text-anchor="middle" style="fill:var(--accent-text)">{a}</text>')
        o.append(f'<text x="{x + bw / 2}" y="{yy + 25}" font-size="19" text-anchor="middle" class="t-on-accent">{b}</text>')
    o.append("</g>")
    # 단계 사이 화살표 (등뼈 위)
    for i in range(7):
        xa = X[i] + bw + 3
        o.append(f'<line x1="{xa}" y1="{mid}" x2="{xa + gap - 7}" y2="{mid}" class="s-accent" stroke-width="3" marker-end="url(#mkflow)"/>')
    # 되돌아가는 길 (점선) — 관문 아래에서 ① 로
    yb = top + bh + 50
    o.append(f'<path d="M{X[7] + bw / 2} {top + bh + 20} V{yb} H{X[1] + bw / 2} V{top + bh + 6}" class="f-none s-muted" '
             f'stroke-width="2.5" stroke-dasharray="7 7" marker-end="url(#mkback)"/>')
    o.append(f'<text x="{(X[1] + X[7] + bw) / 2}" y="{yb + 28}" font-size="18" text-anchor="middle" class="t-body">'
             f'조건에 못 미치면 → 온담을 고친다 (지식 자료 · 지시문 · 응급 문구) → 같은 질문으로 다시 잰다</text>')
    # 범례
    ly = H - 14
    o.append(f'<rect x="{x0}" y="{ly - 14}" width="16" height="16" rx="4" style="fill:var(--surface-accent);stroke:var(--accent)"/>')
    o.append(f'<text x="{x0 + 24}" y="{ly}" font-size="16" class="t-dim">{E(judge)}가 판단</text>')
    o.append(f'<rect x="{x0 + 250}" y="{ly - 14}" width="16" height="16" rx="4" style="fill:var(--track);stroke:var(--border)"/>')
    o.append(f'<text x="{x0 + 274}" y="{ly}" font-size="16" class="t-dim">글자 규칙·판정 코드 (모델 안 씀)</text>')
    o.append("</svg>")
    return '<div class="dgm">' + "".join(o) + "</div>"


# ───────────────────────── 화면들 ─────────────────────────
def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}%"


def build() -> str:
    m, c, cmp, od, og = D["meta"], D["common"], D["compare"], D["ondam_full"], D["common"]["ondam"]
    cg = c["chatgpt"]
    n = m["n_common"]
    V: list[str] = []

    # 0 표지
    V.append(view(None, "온담의 답,<br>어르신이 그대로 따라 해도 될까", f'<p class="who">{E(m["team_line"])}</p>'
                  f'<p class="source">{E(m["date"])} 기준 · 자동검수(autoqa)</p>', cls="cover", label="표지"))

    # 1 질문을 바꾼다
    V.append(view("OPENING · 질문을 바꿉니다", "평균 점수가 아니라, 위험한 답 한{nb}건", f'''<div class="fromto">
<div><p class="col-head col-from">흔히 묻는 것</p>{card("평균 몇 점인가", "답 100개의 평균")}</div>
<div class="arrow" aria-hidden="true">→</div>
<div><p class="col-head col-to">우리가 묻는 것</p>{card("따라 해도 되는가", "위험한 답이 한 건이라도 있는가 · 바로 할 일이 보이는가", focus=True)}</div>
</div>''', takeaway="돌봄 AI는 잘한 답 99개보다 위험한 답 1개가 먼저입니다."))

    # 2 오늘 볼 것
    V.append(view("OPENING · 오늘 볼 것", "네 가지를 차례로 봅니다", '<div class="grid g2">'
                  + card("무엇을 어떻게 재나", "판정 세 갈래와 여섯 단계", num="01")
                  + card("채점기를 믿어도 되나", "채점기부터 시험했다", num="02")
                  + card("온담은 지금 어디인가", "ChatGPT와 같은 질문으로", num="03")
                  + card("무엇을 고치고 다시 재나", "고칠 곳 · 담당 · 목표", num="04") + "</div>"))

    # 3 판정 세 갈래
    gates = "".join(f'<span class="pill pill-ghost">{E(g)}</span>' for g in D["gates"])
    V.append(view("01 · 무엇을 어떻게 재나", "답 하나는 세 갈래로 나뉩니다", '<div class="grid g3">'
                  + card("Pass", "그대로 써도 된다", pill="통과")
                  + card("Fix", "고친 뒤 다시 잰다", pill="고칠 것")
                  + card("Hold", "어르신께 내보내지 않는다", pill="보류", focus=True) + "</div>"
                  + f'<p class="c-label">Hold가 되는 여섯 가지</p><div class="pills">{gates}</div>',
                  takeaway="여섯 가지 중 하나만 어겨도 Hold입니다.<br>평균 점수로 덮지 않습니다."))

    # 4 흐름
    V.append(view("01 · 무엇을 어떻게 재나", "여섯 단계를 지나야 숫자가 나옵니다", pipeline_svg(),
                  takeaway="마지막 문은 하나.<br>세 조건을 모두 넘어야 “밀리지 않는다”고 씁니다.", wide=True))

    # 5 채점기 시험
    v = D["validation"]
    V.append(view("02 · 채점기를 믿어도 되나", "채점기부터 시험했습니다", '<div class="grid g3">'
                  + stat(f'{v["mutants"][0]}/{v["mutants"][1]}', "일부러 심은 결함을 잡았다", focus=True)
                  + stat(f'{v["good"][0]}/{v["good"][1]}', "좋은 답을 결함으로 몰지 않았다")
                  + stat(f'{v["gate"][0]}/{v["gate"][1]}', "잡은 결함의 종류를 맞게 짚었다") + "</div>"
                  + (f'<p class="lead">{nb(v["note"])}</p>' if v.get("note") else ""),
                  source=v["source"]))

    # 6 실제 답으로 다시 의심
    rf, rj = D["rule_fixes"], D["rejudge"]
    cards = "".join(
        f'<article class="card"><p class="c-label">{E(r["name"])}</p>{fromnow(str(r["before"]) + r["unit"], str(r["after"]) + r["unit"])}'
        f'<p class="c-sub">{nb(r["note"])}</p></article>' for r in rf)
    V.append(view("02 · 채점기를 믿어도 되나", "실제 답을 넣어 보니, 채점 규칙도 틀렸습니다",
                  f'<div class="grid g3">{cards}</div>'
                  f'<p class="lead">같은 답을 두 번 채점하니 {rj["n"]}개 중 <b>{rj["flipped"]}개</b>의 판정이 갈렸습니다. '
                  f'그래서 두 번 채점하고 나쁜 쪽을 씁니다.</p>',
                  takeaway="채점기도 틀립니다.<br>잘못 잡은 것을 고치고, 두 번 채점합니다.",
                  source="규칙을 고치기 전과 뒤를 같은 답으로 대조했다 · 2026-11-15", wide=True))

    # 7 가져온 원칙 — 숫자는 장부(silvercare_autoqa/data/knowhow_ledger.yaml)에서 센다
    led = ledger_counts()
    done = led["applied"] + led["today"]
    frows = "".join(
        f'<div class="brow"><span class="b-name">{E(f["name"])}</span>'
        f'<div class="track"><div class="fill is-accent" style="width:{100 * f["done"] / f["n"]:.1f}%" '
        f'title="{E(f["name"])} {f["done"]}/{f["n"]}"></div></div>'
        f'<span class="b-val">{f["done"]}<small>/{f["n"]}</small></span></div>' for f in led["families"])
    V.append(view("02 · 채점기를 믿어도 되나", f'상용 서비스 검수의 원칙 {led["total"]}가지를 하나씩 대조했습니다',
                  '<div class="grid g3 tight">'
                  + stat(str(done), "이미 들어 있거나 오늘 넣었다", unit="가지", focus=True)
                  + stat(str(led["next"]), "넣을 때를 정해 두었다", unit="가지")
                  + stat(str(led["na"]), "우리 일에는 맞지 않는다", unit="가지") + "</div>"
                  + '<div class="bars tight">' + frows + "</div>",
                  source=D["knowhow"]["source"], wide=True))

    # 8 통과율·Hold
    V.append(view("03 · 온담은 지금 어디인가", f'같은 질문 {n}개,<br>통과는 {og["pass"]} 대 {cg["pass"]}',
                  meter_rows([{"name": "온담", "k": og["pass"], "n": n, "accent": True},
                              {"name": m["comparator"], "k": cg["pass"], "n": n}])
                  + '<div class="grid g2">'
                  + stat(str(og["hold"]), "온담의 위해 답 (Hold)", unit="건", focus=True, note=D["hold_names"]["ondam"])
                  + stat(str(cg["hold"]), m["comparator"] + "의 위해 답 (Hold)", unit="건", note=D["hold_names"]["chatgpt"])
                  + "</div>"
                  + table_view(["", "Pass", "Fix", "Hold"], [["온담", og["pass"], og["fix"], og["hold"]],
                                                           [m["comparator"], cg["pass"], cg["fix"], cg["hold"]]]),
                  takeaway=("통과는 온담이 더 많습니다.<br>위해 답은 두 쪽 모두 남았습니다." if og["pass"] >= cg["pass"]
                            else "통과는 아직 " + m["comparator"] + "가 더 많습니다."),
                  source=m["common_source"].format(n=n)))

    # 9 나란히
    rows, tv = [], []
    for cr in cmp["criteria"]:
        rows.append(split_row(cr["name"], [("s-win", cr["win"], "온담이 나음"), ("s-tie", cr["tie"], "비김"),
                                           ("s-loss", cr["loss"], m["comparator"] + "가 나음")], n,
                              f'{cr["win"]} <small>: {cr["loss"]}</small>'))
        tv.append([cr["name"], cr["win"], cr["tie"], cr["loss"]])
    V.append(view("03 · 온담은 지금 어디인가", "온담은 짧고 쉽게,<br>ChatGPT는 빠짐없이",
                  '<div class="bars">' + "".join(rows) + "</div>"
                  + legend([("win", "온담이 나음"), ("tie", "비김"), ("loss", m["comparator"] + "가 나음")])
                  + table_view(["기준", "온담이 나음", "비김", m["comparator"] + "가 나음"], tv),
                  source=cmp["source"].format(n=n), wide=True))

    # 10 ChatGPT 도 넘지 못한 기준
    cn = D["chatgpt_notes"]
    cc = "".join(f'<article class="card"><div class="stat"><div class="big">{E(x["value"])}</div></div>'
                 f'<h3 class="c-title">{nb(x["title"])}</h3><p class="c-sub">{nb(x["sub"])}</p></article>' for x in cn["cards"])
    V.append(view("03 · 온담은 지금 어디인가", "상용 서비스도 어르신 기준은 넘지 못했습니다", f'<div class="grid g3">{cc}</div>',
                  takeaway=cn["takeaway"], wide=True))

    # 11 밀리는 곳
    gp = "".join(f'<article class="card"><span class="pill pill-ghost">{E(g["owner"])}</span>'
                 f'<div class="stat"><div class="big">{E(g["big"])}<span class="unit">{E(g["unit"])}</span></div></div>'
                 f'<h3 class="c-title">{nb(g["title"])}</h3><p class="c-sub">{nb(g["sub"])}</p></article>' for g in D["gaps"])
    V.append(view("04 · 무엇을 고치고 다시 재나", "온담이 고칠 곳은 네 군데입니다", f'<div class="grid g4">{gp}</div>',
                  takeaway="고칠 곳마다 담당과 다시 잴 방법을 정했습니다.", wide=True))

    # 11-2 이것부터 고치면 — 고칠 순서를 계산으로
    ff = D["fixfirst"]
    N = ff["n"]
    own = ff["by_owner"]
    parts = [("s-pass", ff["pass"], "지금 통과")]
    names = [("데이터 담당", "s-win"), ("온담 담당", "s-fix"), ("앱 담당", "s-open")]
    seg_of = dict(names)
    for o, cls in names:
        if own.get(o):
            parts.append((cls, own[o], o + "이 고치면 통과"))
    whole = split_row(f"답 {N}개", parts, N, "")
    steps = [s for s in ff["steps"] if s["hold_cleared"] or s["gain"] >= 3][:6]      # Hold 가 풀리는 것은 한 문항이어도 싣는다
    srows = []
    for s in steps:
        prev = s["pass_after"] - s["gain"]
        bar = (f'<div class="split"><div class="seg s-pass" style="flex:{prev} 1 0" title="앞 순서까지 고쳤을 때 통과 {prev}"></div>'
               f'<div class="seg {seg_of.get(s["owner"], "s-win")}" style="flex:{s["gain"]} 1 0" title="이것까지 고치면 +{s["gain"]}">'
               f'{"+" + str(s["gain"]) if s["gain"] >= 10 else ""}</div>'
               f'<div style="flex:{N - s["pass_after"]} 1 0"></div></div>')
        hold = ' <span class="pill pill-accent mini">Hold 풀림</span>' if s["hold_cleared"] else ""
        srows.append(f'<div class="brow"><span class="b-name">{E(s["what"])} <small class="own">{E(s["owner"].replace(" 담당", ""))}</small>{hold}</span>'
                     f'{bar}<span class="b-val"><span class="gain">+{s["gain"]}</span> {s["pass_after"]}<small>/{N}</small></span></div>')
    V.append(view("04 · 무엇을 고치고 다시 재나", "무엇부터 고칠지 계산으로 냈습니다",
                  '<div class="bars tight names">' + whole + "</div>"
                  + legend([("pass", "지금 통과"), ("win", "데이터 담당이 고치면"), ("fix", "온담 담당이 고치면"),
                            ("open", "앱 담당이 고치면")])
                  + '<p class="c-label">위해 답부터, 그다음은 통과가 많이 느는 순서입니다</p><div class="bars tight names">' + "".join(srows) + "</div>",
                  source=D["fixfirst_source"], wide=True))

    # 12 세 조건 — 큰 숫자는 두 쪽의 개수(온담 대 비교 상대)로 적고, 기준과 견준 말은 카드마다 한 줄만 둔다.
    #    “질문을 바꿔 재면”= 95% 신뢰구간의 아래 끝. 넘었는지는 비교 결과에 적힌 판정을 그대로 쓴다.
    tg, name_b = D["targets"], m["comparator"]
    pd, pf = cmp["pass_diff"], cmp["pref"]
    ov = next(x for x in cmp["criteria"] if x["name"] == "종합")

    def dots(k: int, me: bool) -> str:
        return '<div class="dots">' + f'<span class="dot{" me" if me else ""}"></span>' * min(k, 12) + "</div>"

    def bar(k: int, me: bool) -> str:
        return (f'<div class="track"><div class="fill{" is-accent" if me else ""}" style="width:{100 * k / n:.1f}%" '
                f'title="{k}/{n}"></div></div>')

    def near(x: float, ref: float) -> str:       # 기준과 1%p 안쪽으로 붙으면 소수 한 자리까지 적는다
        return f"{x * 100:.1f}%" if abs(x - ref) < 0.0095 else f"{x * 100:.0f}%"

    # ① 위해 답
    hold_names = ", ".join(E(x) for x in D["hold_names"]["ondam"].split(" · "))
    c1 = goal_card(
        tg["hold"]["met"], "위해 답이 0건일 것", versus(str(og["hold"]), str(cg["hold"]), name_b, unit="건"),
        f'<div class="marks" role="img" aria-label="위해 답 온담 {og["hold"]}건, {E(name_b)} {cg["hold"]}건">'
        f'{dots(og["hold"], True)}{dots(cg["hold"], False)}</div>',
        f"{hold_names} <b>{og['hold']}건이 남았습니다</b>" if og["hold"]
        else f"온담의 위해 답이 없습니다 <b>기준 {tg['hold']['goal']}건을 지켰습니다</b>")
    # ② 통과율
    lo, mg = pd["lo"], tg["pass"]["margin"]
    lo_txt = pts(lo) if abs(abs(lo) - mg) >= 0.0095 else f"{abs(lo) * 100:.1f}%p"
    if lo >= 0:
        pass_why = f"질문을 바꿔 재도 뒤지지 않습니다 <b>기준 {pts(mg)} 안에 들었습니다</b>"
    elif tg["pass"]["met"]:
        pass_why = f"질문을 바꿔 재서 뒤지더라도 {lo_txt}까지 <b>기준 {pts(mg)} 안에 들었습니다</b>"
    else:
        pass_why = f"질문을 바꿔 재면 {lo_txt}까지 뒤질 수 있습니다 <b>기준은 {pts(mg)} 안입니다</b>"
    c2 = goal_card(
        tg["pass"]["met"], "통과율이 크게 뒤지지 않을 것", versus(str(og["pass"]), str(cg["pass"]), name_b, unit=f"/{n}"),
        f'<div class="marks" role="img" aria-label="통과 온담 {og["pass"]}/{n}, {E(name_b)} {cg["pass"]}/{n}">'
        f'{bar(og["pass"], True)}{bar(cg["pass"], False)}</div>', pass_why)
    # ③ 나란히 놓고 고르기
    fl = tg["pref"]["floor"]
    if tg["pref"]["met"]:
        pref_why = f"질문을 바꿔 재도 온담을 고른 비율이 {near(pf['lo'], fl)} <b>기준 {fl * 100:.0f}%를 넘었습니다</b>"
    else:
        pref_why = (f"질문을 바꿔 재면 온담을 고른 비율이 {near(pf['lo'], fl)}까지 "
                    f"<b>기준 {fl * 100:.0f}%에 {pts(fl - pf['lo'])} 모자랍니다</b>")
    segs = "".join(f'<div class="seg {c}" style="flex:{v} 1 0" title="{E(t)} {v}/{n}"></div>'
                   for c, v, t in (("s-win", ov["win"], "온담이 나음"), ("s-tie", ov["tie"], "비김"),
                                   ("s-loss", ov["loss"], name_b + "가 나음")) if v > 0)
    c3 = goal_card(
        tg["pref"]["met"], "나란히 놓으면 비슷하게 뽑힐 것",
        versus(str(ov["win"]), str(ov["loss"]), name_b, unit="번", tie=str(ov["tie"])),
        f'<div class="marks one" role="img" aria-label="온담이 나음 {ov["win"]}, 비김 {ov["tie"]}, '
        f'{E(name_b)}가 나음 {ov["loss"]}"><div class="split">{segs}</div></div>', pref_why)

    n_met = sum(bool(t["met"]) for t in tg.values())
    head13 = {0: "세 조건, 아직 하나도 넘지 못했습니다", 1: "세 조건 중 하나를 넘었습니다", 2: "세 조건 중 둘을 넘었습니다",
              3: "세 조건을 모두 넘었습니다"}[n_met]
    V.append(view("04 · 무엇을 고치고 다시 재나", head13, f'<div class="grid g3 goals">{c1}{c2}{c3}</div>',
                  lead=f"셋을 모두 넘어야 “{E(name_b)}에 밀리지 않는다”고 말합니다.",
                  takeaway="다음 — 온담을 보강한 뒤, 같은 질문으로 다시 채점합니다." if n_met < 3 else "",
                  source=D["targets_source"], wide=True))

    # 13 마무리
    V.append(view("CLOSING · 처음 질문으로", "어르신이 그대로 따라 해도 될까",
                  f'<blockquote class="quote"><p>{nb(D["closing"]["quote"])}</p></blockquote>',
                  takeaway=D["closing"]["takeaway"]))

    title = "온담 자동검수 발표 자료"
    return (f'<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{title}</title>\n'
            f'<style>{CSS}</style>\n</head>\n<body>\n<main class="deck">\n' + "\n\n".join(V)
            + f'\n</main>\n<script>{JS}</script>\n</body>\n</html>\n')


if __name__ == "__main__":
    out = HERE / "index.html"
    out.write_text(build(), encoding="utf-8", newline="\n")
    print("만듦:", out, f"({out.stat().st_size:,} bytes)")
