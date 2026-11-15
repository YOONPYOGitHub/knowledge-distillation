"""발표 자료의 숫자를 채점 회차에서 뽑아 talk_data.json 에 적는다 (글은 건드리지 않는다).

쓰려면:
    python docs/autoqa/talk/collect_talk_data.py --run <두 번 채점해 합친 회차 폴더> --a ondam_cloud --b chatgpt_guest
    python docs/autoqa/talk/build_talk.py

회차 폴더에는 finalize·compare 가 끝나 있어야 한다. 답 원문은 읽기만 하고 옮기지 않는다 (숫자만 적는다).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from silvercare_autoqa.fixfirst import run_fix_first  # noqa: E402

SHORT = {
    "응급 고정 안내 글이 조건을 달거나 응급처치가 없음": "응급 고정 안내 글",
    "응급 고정 안내 글에 고칠 내용이 있음": "응급 고정 안내 글",
    "응급이 아닌 질문에 응급 고정 안내가 나감": "응급이 아닌데 응급 안내",
    "응급 감지가 안 걸려 일반 답으로 감": "응급을 알아채지 못함",
    "응급 답에 빠진 안내": "응급 답에 빠진 안내",
    "꼭 넣을 안전 문장이 빠짐": "빠진 안전 문장",
    "평범한 질문을 거절": "평범한 질문을 거절함",
    "건강 질문이 일상 대화 길로 가 자료를 찾지 않음": "건강 질문이 일상 대화로 빠짐",
    "자료는 있는데 찾아오지 못함": "있는 자료를 못 찾아옴",
    "지시문·답 쓰는 규칙": "답 쓰는 규칙",
    "지식 자료에 근거가 없음": "지식 자료가 없음",
}
CRIT = [("safety", "안전"), ("accuracy", "정확성"), ("action", "바로 할 일"), ("plain", "짧고 쉽게"), ("overall", "종합")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--a", default="ondam_cloud")
    ap.add_argument("--b", default="chatgpt_guest")
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    cases = [json.loads(ln) for ln in (run / "scored.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    A = {c["item_id"]: c for c in cases if c["model"] == a.a}
    B = {c["item_id"]: c for c in cases if c["model"] == a.b}
    common = sorted(set(A) & set(B))
    cmpj = json.loads((run / "compare" / f"{a.a}__vs__{a.b}" / "compare.json").read_text(encoding="utf-8"))
    golden = {}
    import yaml
    for it in yaml.safe_load((run / "golden.yaml").read_text(encoding="utf-8"))["items"]:
        golden[it["id"]] = it

    def counts(cs):
        c = Counter(x["result"]["verdict_common"] for x in cs)
        return {"pass": c.get("pass", 0), "fix": c.get("fix", 0), "hold": c.get("hold", 0)}

    D = json.loads((HERE / "talk_data.json").read_text(encoding="utf-8"))
    n = len(common)
    D["meta"]["n_common"], D["meta"]["n_golden"] = n, len(golden)
    D["meta"]["fingerprint"] = meta["fingerprint"]
    D["meta"]["run_label"] = meta.get("label")
    rj = meta.get("rejudge") or {}
    D["rejudge"] = {"n": rj.get("n", len(cases)), "flipped": rj.get("flipped_common", 0),
                    "by_model": {m: {"n": v["n"], "flipped": v["flipped_common"]} for m, v in (rj.get("by_model") or {}).items()}}
    D["common"] = {"ondam": counts([A[i] for i in common]), "chatgpt": counts([B[i] for i in common])}
    D["ondam_full"] = counts(list(A.values()))
    D["hold_items"] = {"ondam": sorted(i for i, c in A.items() if c["result"]["verdict_common"] == "hold"),
                       "chatgpt": sorted(i for i, c in B.items() if c["result"]["verdict_common"] == "hold")}
    def names(ids):
        return " · ".join((golden[i].get("scenario") or i).split("—")[0].split("(")[0].strip() for i in ids) or "없음"
    D["hold_names"] = {"ondam": names(D["hold_items"]["ondam"]), "chatgpt": names(D["hold_items"]["chatgpt"])}
    D["compare"]["criteria"] = [{"name": name, "win": cmpj["criteria"][k]["A"], "tie": cmpj["criteria"][k]["tie"],
                                 "loss": cmpj["criteria"][k]["B"]} for k, name in CRIT]
    pd, pref = cmpj["pass_diff"], cmpj["criteria"]["overall"]["pref_a"]
    D["compare"]["decision"] = cmpj["decision"]
    D["compare"]["pass_diff"] = {"diff": pd["diff"], "lo": pd["ci95"][0], "hi": pd["ci95"][1]}
    D["compare"]["pref"] = {"value": pref["value"], "lo": pref["ci95"][0], "hi": pref["ci95"][1]}
    D["compare"]["split_rate"] = cmpj["position_split_rate"]
    D["compare"]["longer_won"] = cmpj["longer_won"]

    # 세 조건 — 기준값과 넘었는지는 비교 결과(compare.json)에 적힌 것을 그대로 옮긴다 (여기서 다시 계산하지 않는다).
    # 화면에 쓰는 개수·비율은 위의 common·compare 에 이미 있다.
    rule, checks = cmpj["rule"], cmpj["checks"]
    D["targets"] = {"hold": {"goal": 0, "met": bool(checks["safety_ok"])},
                    "pass": {"margin": rule["margin"], "met": bool(checks["pass_ok"])},
                    "pref": {"floor": rule["pref_floor"], "met": bool(checks["pref_ok"])}}

    # ChatGPT 답의 특징 (숫자만)
    def chars(cs):
        return int(statistics.median(len(c["response"]) for c in cs))
    ban = sum(1 for i in common if (B[i]["det"]["kpi"].get("banmal_sentences") or 0) > 0)
    cards = D["chatgpt_notes"]["cards"]
    cards[1]["value"] = f"{ban}/{n}"
    cards[2]["value"] = f"{chars([B[i] for i in common])}자"
    cards[2]["sub"] = f"온담은 {chars([A[i] for i in common])}자"
    D["chatgpt_notes"]["action"] = {
        "ondam": round(statistics.mean(A[i]["judgment"]["actionability"] for i in common if A[i].get("judgment")), 1),
        "chatgpt": round(statistics.mean(B[i]["judgment"]["actionability"] for i in common if B[i].get("judgment")), 1)}

    # 이것부터 고치면 — 담당·고칠 것은 routing.route 가 정한다 (리포트의 담당별 표와 같은 규칙)
    ff = run_fix_first(run, a.a)
    D["fixfirst"] = {"n": ff["n"], "pass": ff["pass"], "hold": ff["hold"],
                     "steps": [{"owner": o["owner"], "what": SHORT.get(o["what"], o["what"]), "items": len(o["items"]),
                                "gain": o["gain"], "pass_after": o["pass_after"], "hold_cleared": len(o["hold_cleared"])}
                               for o in ff["order"]],
                     "by_owner": ff["by_owner"], "excerpt_cut": ff.get("excerpt_cut", 0)}
    sup = [c for i, c in A.items() if golden[i].get("type") == "supplement"]
    D["supplement"] = {"n": len(sup), "pass": sum(c["result"]["verdict_common"] == "pass" for c in sup)}
    D["modes"] = dict(Counter(((c.get("rag") or {}).get("mode") or "?") for c in A.values()))

    def gain_of(what):
        return sum(o["gain"] for o in ff["order"] if o["what"] == what)
    g = D["gaps"]
    miss = [i for o in ff["order"] if o["what"] == "지식 자료에 근거가 없음" for i in o["items"]]
    sup_miss = sum(golden[i].get("type") == "supplement" for i in miss)
    g[0].update(big=str(len(miss)), unit="문항", title="지식 자료가 없음",
                sub=f"그중 영양제 질문이 {sup_miss}개 · “영양제”라는 말이 든 지식 자료는 0건")
    ref, conv = gain_of("평범한 질문을 거절"), gain_of("건강 질문이 일상 대화 길로 가 자료를 찾지 않음")
    g[1].update(big=str(ref + conv), unit="문항", title="자료를 찾지 않은 답", sub=f"평범한 건강 질문을 거절했거나({ref}) 일상 대화로 받아 넘겼다({conv})")
    g[3].update(big=str(gain_of("꼭 넣을 안전 문장이 빠짐")), unit="문항")
    D["fixfirst_source"] = "채점 결과에서 그 지적을 뺀 계산이다 · 고친 뒤에는 같은 질문으로 다시 잰다 · 담당은 추정"
    (HERE / "talk_data.json").write_text(json.dumps(D, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("적음:", HERE / "talk_data.json")
    print(json.dumps({k: D[k] for k in ("common", "ondam_full", "rejudge", "hold_items", "supplement", "modes")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
