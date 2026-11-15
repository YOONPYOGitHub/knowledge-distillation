"""「이것부터 고치면」 — 고칠 것을 하나씩 뺐을 때 통과가 가장 많이 느는 순서를 낸다.

쇼핑 AI 검수에서 가져온 원칙이다. 많이 걸린 결함과 고치면 통과가 느는 결함은 다르다
(문구 34건을 고쳤는데 합격률이 그대로였던 일이 있었다). 모델을 부르지 않고 채점 결과만으로 계산한다.

- 고칠 단위 = (누가 고치나, 무엇을). 답 하나에 하나씩 붙인다. 나누는 규칙은 routing.route 한 곳에만 있다
  (리포트의 담당별 표와 같은 규칙이다. 두 곳에 따로 두면 같은 문항에 담당이 다르게 나온다).
- 순서는 탐욕으로 정한다: 지금까지 고친 것에 더해 통과가 가장 많이 느는 단위를 고른다. Hold 가 풀리는 단위가 먼저다.
- 분모는 그 모델이 낸 답 전체로 고정한다 (불합격 문항만 세면 비율이 부푼다).

계산이지 측정이 아니다. 실제로 고친 뒤에는 같은 질문으로 다시 받아 다시 채점한다 (고치다 다른 결함이 생길 수 있다).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import case_verdict
from .routing import OWNER_APP, OWNER_DATA, OWNER_ONDAM, partial_evidence, route  # noqa: F401 (담당 이름은 여기서도 쓴다)

Unit = tuple[str, str]          # (담당, 고칠 것)
NO_OWNER = "(담당 없음)"         # 출처를 돌려주지 않는 시스템 — 우리 팀이 고치는 대상이 아니다


def unit_of(case: dict, item: dict) -> Unit:
    """통과하지 못한 답 하나를 '누가 무엇을 고치나'로 옮긴다. 규칙은 routing.route 가 정한다. 추정이다."""
    r = route(case, item)
    if r is None:
        return (NO_OWNER, "통과하지 못한 답")
    return (r.owner, r.reason)


def fix_first(cases: list[dict], items: dict[str, dict], *, basis: str = "common") -> dict[str, Any]:
    """한 모델의 채점 결과에서 고칠 순서를 낸다. cases: scored.jsonl 의 그 모델 줄."""
    total = len(cases)
    rows = []
    for c in cases:
        v = case_verdict(c, basis)
        if v not in ("fix", "hold"):
            continue
        rows.append({"item_id": c["item_id"], "verdict": v, "unit": unit_of(c, items[c["item_id"]])})
    n_pass = sum(case_verdict(c, basis) == "pass" for c in cases)
    n_hold = sum(r["verdict"] == "hold" for r in rows)
    by_unit: dict[Unit, list[dict]] = defaultdict(list)
    for r in rows:
        by_unit[r["unit"]].append(r)
    # Hold 가 풀리는 단위 먼저, 그다음 통과가 많이 느는 순서. 답 하나에 단위가 하나라 한 번 정렬하면 탐욕 순서와 같다
    ordered = sorted(by_unit.items(), key=lambda kv: (-sum(r["verdict"] == "hold" for r in kv[1]), -len(kv[1]), kv[0]))
    order, cur_pass, cur_hold = [], n_pass, n_hold
    for (owner, what), rs in ordered:
        holds = sorted(r["item_id"] for r in rs if r["verdict"] == "hold")
        cur_pass += len(rs)
        cur_hold -= len(holds)
        order.append({"owner": owner, "what": what, "items": sorted(r["item_id"] for r in rs), "gain": len(rs),
                      "hold_cleared": holds, "pass_after": cur_pass, "hold_after": cur_hold})
    by_owner: dict[str, int] = defaultdict(int)
    for r in rows:
        by_owner[r["unit"][0]] += 1
    return {"basis": basis, "n": total, "pass": n_pass, "hold": n_hold, "not_pass": len(rows), "order": order,
            "by_owner": dict(sorted(by_owner.items())), "excerpt_cut": partial_evidence(cases)}


def run_fix_first(run_dir: str | Path, model: str, *, basis: str = "common") -> dict[str, Any]:
    from .pipeline import _load

    run = Path(run_dir)
    _, _, _, items = _load(run)
    p = run / "scored.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — finalize 까지 끝낸 회차를 준다")
    cases = [c for c in (json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()) if c["model"] == model]
    if not cases:
        raise ValueError(f"모델 '{model}' 의 답이 회차에 없음")
    out = fix_first(cases, items, basis=basis)
    out["model"] = model
    return out


def render_md(s: dict) -> str:
    n = s["n"]
    L = [f"# 이것부터 고치면 — {s['model']}", "", "## 한눈에", "",
         f"- 지금: 답 {n}개 중 Pass {s['pass']} · Hold {s['hold']} "
         f"({'출처 지적을 뺀 공통 기준' if s['basis'] == 'common' else '전체 판정'}).",
         "- 아래 순서로 고치면 통과가 가장 빨리 는다. 그 답이 통과했다고 치고 센 계산이다. 모델은 부르지 않았다.",
         "- 고친 뒤에는 같은 질문으로 답을 다시 받아 다시 채점한다. 고치다 다른 결함이 생길 수 있다.",
         "- 담당은 추정이다. 답이 나간 길(mode)·자료 범위·지적 종류로 나눴다. 규칙은 리포트의 담당별 표와 같다.",
         *([f"- ⚠ 답 {s['excerpt_cut']}개는 채점기가 근거 자료의 일부만 봤다. 「지식 자료에 근거가 없음」·「찾아오지 못함」은 "
            f"실제보다 많이 잡혔을 수 있다."] if s.get("excerpt_cut") else []),
         "", "## 고칠 순서", "",
         "| 순서 | 담당 | 고칠 것 | 문항 | 여기까지 고치면 Pass | 풀리는 Hold |", "|---|---|---|---|---|---|"]
    for i, o in enumerate(s["order"], 1):
        L.append(f"| {i} | {o['owner']} | {o['what']} | {o['gain']} | {o['pass_after']}/{n} | {', '.join(o['hold_cleared']) or '—'} |")
    L += ["", "## 담당별", "", "| 담당 | 고칠 문항 |", "|---|---|"]
    for o, k in s["by_owner"].items():
        L.append(f"| {o} | {k} |")
    L += ["", "## 문항 목록", ""]
    for i, o in enumerate(s["order"], 1):
        L.append(f"- {i}. {o['what']} ({o['owner']}): {', '.join(o['items'])}")
    return "\n".join(L) + "\n"
