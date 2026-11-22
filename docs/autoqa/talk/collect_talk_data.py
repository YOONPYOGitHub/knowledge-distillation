"""발표 자료의 숫자를 채점 회차에서 뽑아 talk_data.json 에 적는다 (사람이 쓴 문장은 건드리지 않는다).

쓰려면:
    python docs/autoqa/talk/collect_talk_data.py --run <두 번 채점해 합친 회차 폴더> \
        --a ondam_local_rag --b chatgpt_guest --c ondam_local_only [--mutation <변이 시험 mutation_score.json>] \
        [--knowledge <팀 지식 자료 knowledge.jsonl>]
    python docs/autoqa/talk/build_talk.py

- A = 우리 시스템(지식 검색 있음), B = 비교 상대(ChatGPT), C = 같은 모델에서 검색만 뺀 것. 비교는 A 대 B · A 대 C 두 가지.
- 회차 폴더에는 finalize 와 두 compare 가 끝나 있어야 한다. 답 원문은 읽기만 하고 옮기지 않는다 (숫자만 적는다).
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

from silvercare_autoqa import routing as RT  # noqa: E402
from silvercare_autoqa.fixfirst import run_fix_first  # noqa: E402

SHORT = {
    RT.APP_FIXED_TEXT: "응급 고정 안내 글",
    RT.APP_FIXED_OTHER: "응급 고정 안내 글",
    RT.APP_FALSE_ALARM: "응급이 아닌데 응급 안내",
    RT.APP_MISSED: "응급을 알아채지 못함",
    RT.ONDAM_EMERGENCY_LINE: "응급 답에 빠진 안내",
    RT.ONDAM_SAFETY_LINE: "빠진 안전 문장",
    RT.ONDAM_REFUSAL: "평범한 질문을 거절함",
    RT.ONDAM_CONVERSATION: "건강 질문이 일상 대화로 빠짐",
    RT.ONDAM_SEARCH: "있는 자료를 못 찾아옴",
    RT.ONDAM_RULES: "답 쓰는 규칙",
    RT.DATA_MISSING: "지식 자료가 없음",
}
# 고칠 곳 카드 — 같은 손으로 고치는 것끼리 묶는다 (카드 제목, 한 줄 설명)
GAP_GROUP = {
    RT.DATA_MISSING: ("data", "지식 자료가 없음"),
    RT.ONDAM_REFUSAL: ("path", "자료를 찾지 않은 답"),
    RT.ONDAM_CONVERSATION: ("path", "자료를 찾지 않은 답"),
    RT.ONDAM_SAFETY_LINE: ("safety", "빠진 안전 문장"),
    RT.ONDAM_SEARCH: ("search", "있는 자료를 못 찾아옴"),
    RT.ONDAM_RULES: ("rules", "답 쓰는 규칙"),
    RT.ONDAM_EMERGENCY_LINE: ("emergency", "응급 답에 빠진 안내"),
    RT.APP_FIXED_TEXT: ("app", "응급 고정 안내"),
    RT.APP_FIXED_OTHER: ("app", "응급 고정 안내"),
    RT.APP_FALSE_ALARM: ("app", "응급 고정 안내"),
    RT.APP_MISSED: ("app", "응급을 알아채지 못함"),
}
TYPE_LABEL = {"emergency": "응급", "symptom": "증상 상담", "chronic": "만성질환 관리", "medication": "약 먹는 법",
              "prevention": "예방·생활", "supplement": "영양제", "companion": "말벗(일상 대화)",
              "boundary": "조심할 요청(진단·개인정보)", "welfare": "복지 안내"}
# 막힌 이유 — 채점 지적의 종류(cat)를 사람이 읽는 말로. 한 답에 여럿일 수 있다
REASON_KEY = {"fact": "fact", "content": "content", "conclusion": "conclusion", "safety_must": "safety",
              "gate": "gate", "format": "format"}
REASON_LABEL = {"fact": "사실과 다른 말", "content": "꼭 할 말이 빠지거나 뒤집힘", "conclusion": "결론이 모범 답안과 다름",
                "safety": "안전 안내가 빠짐", "gate": "위해 답", "format": "답 형식이 깨짐", "other": "그 밖의 이유"}
CRIT = [("safety", "안전"), ("accuracy", "정확성"), ("action", "바로 할 일"), ("plain", "짧고 쉽게"), ("overall", "종합")]
NAMES = {"ondam_qwen_rag": ("검색 있는 온담", "최종판 · Qwen + 지식 검색"),
         "ondam_local_rag": ("검색 있는 온담", "온디바이스 Qwen3.5-4B + 지식 검색"),
         "ondam_local_only": ("검색 없는 온담", "11/16 판 · Qwen3.5-4B, 지식 검색 없음"),
         "chatgpt_guest": ("ChatGPT", "로그인 안 한 무료 기본 · 11/15 에 받은 답")}
JUDGE_NAME = {"claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--a", default="ondam_local_rag")
    ap.add_argument("--b", default="chatgpt_guest")
    ap.add_argument("--c", default="ondam_local_only")
    ap.add_argument("--mutation", help="Opus 로 다시 잰 변이 시험 mutation_score.json")
    ap.add_argument("--knowledge", help="팀 지식 자료 knowledge.jsonl (‘영양제’가 든 토막 수를 센다)")
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    cases = [json.loads(ln) for ln in (run / "scored.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    by = {m: {c["item_id"]: c for c in cases if c["model"] == m} for m in (a.a, a.b, a.c)}
    A, B, C = by[a.a], by[a.b], by[a.c]
    common = sorted(set(A) & set(B) & set(C))
    import yaml
    golden = {it["id"]: it for it in yaml.safe_load((run / "golden.yaml").read_text(encoding="utf-8"))["items"]}

    def cmpj(x: str, y: str) -> dict:
        return json.loads((run / "compare" / f"{x}__vs__{y}" / "compare.json").read_text(encoding="utf-8"))

    def counts(cs) -> dict:
        c = Counter(x["result"]["verdict_common"] for x in cs)
        return {"pass": c.get("pass", 0), "fix": c.get("fix", 0), "hold": c.get("hold", 0)}

    def names(ids) -> str:
        return " · ".join((golden[i].get("scenario") or i).split("—")[0].split("(")[0].strip() for i in ids) or "없음"

    def holds(M) -> list[str]:
        return sorted(i for i in common if M[i]["result"]["verdict_common"] == "hold")

    def median_sec(M) -> float | None:
        el = [(M[i].get("meta") or {}).get("elapsed_ms") for i in common]
        el = [e for e in el if isinstance(e, (int, float))]
        return round(statistics.median(el) / 1000, 1) if el else None

    D = json.loads((HERE / "talk_data.json").read_text(encoding="utf-8"))
    n = len(common)
    D["meta"].update(n_common=n, n_golden=len(golden), fingerprint=meta["fingerprint"], run_label=meta.get("label"),
                     judge=JUDGE_NAME.get(meta["judge_model"], meta["judge_model"]), comparator=NAMES[a.b][0])
    rj = meta.get("rejudge") or {}
    D["rejudge"] = {"n": rj.get("n", len(cases)), "flipped": rj.get("flipped_common", 0),
                    "by_model": {m: {"n": v["n"], "flipped": v["flipped_common"]} for m, v in (rj.get("by_model") or {}).items()}}
    D["common"] = {"ondam": counts([A[i] for i in common]), "chatgpt": counts([B[i] for i in common]),
                   "nosearch": counts([C[i] for i in common])}
    D["ondam_full"] = counts(list(A.values()))
    D["hold_items"] = {"ondam": holds(A), "chatgpt": holds(B), "nosearch": holds(C)}
    D["hold_names"] = {k: names(v) for k, v in D["hold_items"].items()}
    D["systems"] = [{"key": m, "name": NAMES[m][0], "sub": NAMES[m][1], **counts([by[m][i] for i in common]),
                     "hold_names": names(holds(by[m])), "median_sec": median_sec(by[m]) if m != a.b else None,
                     "hold_emergency": sum(golden[i].get("risk") == "emergency" for i in holds(by[m]))}
                    for m in (a.a, a.c, a.b)]

    def pair(x: str, y: str) -> dict:
        j = cmpj(x, y)
        pd, pref = j["pass_diff"], j["criteria"]["overall"]["pref_a"]
        return {"criteria": [{"name": nm, "win": j["criteria"][k]["A"], "tie": j["criteria"][k]["tie"], "loss": j["criteria"][k]["B"]}
                             for k, nm in CRIT],
                "decision": j["decision"], "pass_diff": {"diff": pd["diff"], "lo": pd["ci95"][0], "hi": pd["ci95"][1]},
                "pref": {"value": pref["value"], "lo": pref["ci95"][0], "hi": pref["ci95"][1]},
                "split_rate": j["position_split_rate"], "longer_won": j["longer_won"], "n_pairs": j["n_complete"],
                "rule": j["rule"], "checks": j["checks"]}

    src_ab = D["compare"].get("source", "")
    D["compare"] = {**pair(a.a, a.b), "source": src_ab}
    D["compare_rag"] = {**pair(a.a, a.c), "source": src_ab}
    j = D["compare"]
    D["targets"] = {"hold": {"goal": 0, "met": bool(j["checks"]["safety_ok"])},
                    "pass": {"margin": j["rule"]["margin"], "met": bool(j["checks"]["pass_ok"])},
                    "pref": {"floor": j["rule"]["pref_floor"], "met": bool(j["checks"]["pref_ok"])}}

    # ChatGPT 답의 특징 (숫자만)
    def chars(M) -> int:
        return int(statistics.median(len(M[i]["response"]) for i in common))
    ban = sum(1 for i in common if (B[i]["det"]["kpi"].get("banmal_sentences") or 0) > 0)
    cn = D["chatgpt_notes"]
    cn["cards"] = [
        {"title": "ChatGPT의 위해 답", "sub": D["hold_names"]["chatgpt"], "value": f"{len(D['hold_items']['chatgpt'])}건"},
        {"title": "어르신께 반말로 답한 문항", "sub": "“몇 살이야?” “추천해.” 같은 말", "value": f"{ban}/{n}"},
        {"title": "답 하나의 길이 (가운데 값)", "sub": f"검색 있는 온담은 {chars(A)}자", "value": f"{chars(B)}자"},
    ]
    act = lambda M: round(statistics.mean(M[i]["judgment"]["actionability"] for i in common if M[i].get("judgment")), 1)  # noqa: E731
    cn["action"] = {"ondam": act(A), "chatgpt": act(B), "nosearch": act(C)}

    # 고칠 곳 — 담당·고칠 것은 routing.route 가 정한다 (리포트의 담당별 표와 같은 규칙)
    ff = run_fix_first(run, a.a)
    D["fixfirst"] = {"n": ff["n"], "pass": ff["pass"], "hold": ff["hold"],
                     "steps": [{"owner": o["owner"], "what": SHORT.get(o["what"], o["what"]), "items": len(o["items"]),
                                "gain": o["gain"], "pass_after": o["pass_after"], "hold_cleared": len(o["hold_cleared"])}
                               for o in ff["order"]],
                     "by_owner": ff["by_owner"], "excerpt_cut": ff.get("excerpt_cut", 0)}
    sup = [c for i, c in A.items() if golden[i].get("type") == "supplement"]
    D["supplement"] = {"n": len(sup), "pass": sum(c["result"]["verdict_common"] == "pass" for c in sup)}
    D["modes"] = dict(Counter(((c.get("rag") or {}).get("mode") or "?") for c in A.values()))

    groups: dict[str, dict] = {}
    for o in ff["order"]:
        key, title = GAP_GROUP.get(o["what"], ("other", SHORT.get(o["what"], o["what"])))
        g = groups.setdefault(key, {"owner": o["owner"], "title": title, "items": [], "hold": 0, "parts": Counter()})
        g["items"] += o["items"]
        g["hold"] += len(o["hold_cleared"])
        g["parts"][o["what"]] += len(o["items"])
    kn_supp = None
    if a.knowledge:
        kn_supp = sum(1 for ln in open(a.knowledge, encoding="utf-8") if "영양제" in ln or "건강기능식품" in ln)

    def sub_of(key: str, g: dict) -> str:
        if key == "data":
            s = f"그중 영양제 질문이 {sum(golden[i].get('type') == 'supplement' for i in g['items'])}개"
            return s + (f" · “영양제”라는 말이 든 지식 자료는 {kn_supp}건" if kn_supp is not None else "")
        if key == "path":
            return f"평범한 건강 질문을 거절했거나({g['parts'][RT.ONDAM_REFUSAL]}) 일상 대화로 받아 넘겼다({g['parts'][RT.ONDAM_CONVERSATION]})"
        if key == "rules" and g["hold"]:
            return f"약 먹는 법을 잘못 안내한 위해 답 {g['hold']}건 포함 · 결론·금지답·형식"
        if key == "app" and RT.APP_MISSED in g["parts"]:
            return "응급 질문인데 119 안내가 아예 없었다 — 앱의 응급 감지가 걸리지 않았다"
        return {"safety": "“약은 스스로 끊지 마세요” 같은 말", "search": "지식 자료에는 있는데 검색이 가져오지 못했다",
                "rules": "결론·금지답·형식 같은 답 쓰는 규칙",
                "emergency": "응급 증상인데 ‘지금 바로 119’ 대신 병원 상담이나 ‘~하면 119’로 답했다",
                "app": "앱이 정해 둔 응급 안내 글"}.get(key, "")
    ordered = sorted(groups.items(), key=lambda kv: (-(kv[1]["hold"] > 0), -len(kv[1]["items"])))
    D["gaps"] = [{"owner": g["owner"], "big": str(len(g["items"])), "unit": "문항", "title": g["title"], "sub": sub_of(k, g),
                  "hold": g["hold"]} for k, g in ordered]
    D["fixfirst_source"] = "채점 결과에서 그 지적을 뺀 계산이다 · 고친 뒤에는 같은 질문으로 다시 잰다 · 담당은 추정"

    # ChatGPT 대비 어디서·왜 밀리나 — ① 질문 종류별 통과 ② ChatGPT 는 통과, 우리는 못 넘은 답의 막힌 이유
    def is_pass(M, i):
        return M[i]["result"]["verdict_common"] == "pass"
    bt = []
    for t in sorted({golden[i]["type"] for i in common}):
        ids_t = [i for i in common if golden[i]["type"] == t]
        bt.append({"type": t, "label": TYPE_LABEL.get(t, t), "n": len(ids_t),
                   "a": sum(is_pass(A, i) for i in ids_t), "b": sum(is_pass(B, i) for i in ids_t)})
    D["by_type"] = sorted(bt, key=lambda x: (x["a"] - x["b"], (x["a"] - x["b"]) / x["n"]))   # 통과 수 차이가 큰 종류부터
    gap = [i for i in common if is_pass(B, i) and not is_pass(A, i)]
    reasons: dict[str, dict] = {}
    for i in gap:
        for key in sorted({REASON_KEY.get(f.get("cat"), "other") for f in RT._blocking(A[i]) if f.get("cat") != "grounding"}):
            r = reasons.setdefault(key, {"key": key, "label": REASON_LABEL[key], "items": []})
            r["items"].append(i)
    D["why_gap"] = {"n": len(gap), "n_reverse": sum(1 for i in common if is_pass(A, i) and not is_pass(B, i)),
                    "reasons": [{"key": r["key"], "label": r["label"], "count": len(r["items"]), "items": r["items"],
                                 "examples": [names([i]) for i in r["items"][:2]]}
                                for r in sorted(reasons.values(), key=lambda r: -len(r["items"]))]}

    if a.mutation:
        ms = json.loads(Path(a.mutation).read_text(encoding="utf-8"))
        D["validation"].update(mutants=ms["defects_detected"], good=ms["good_passed"], gate=ms["gate_attributed"],
                               missed=[f"{x['model']} {x['item_id']}" for x in ms.get("missed") or []],
                               false_alarms=[f"{x['model']} {x['item_id']}" for x in ms.get("false_alarms") or []])
    (HERE / "talk_data.json").write_text(json.dumps(D, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("적음:", HERE / "talk_data.json")
    print(json.dumps({k: D[k] for k in ("common", "rejudge", "hold_items", "supplement", "modes")}, ensure_ascii=False))
    print("검색 있음 대 ChatGPT:", D["compare"]["decision"], "· 검색 있음 대 검색 없음:", D["compare_rag"]["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
