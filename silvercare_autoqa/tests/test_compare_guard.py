"""직접 비교가 틀린 '비열등 확인'을 내지 않게 막는 시험 (2026-11-15 노하우 점검에서 찾은 빈틈).

- 비교 상대 답이 없는 문항에서 난 우리 쪽 Hold 가 "Hold 0건"으로 가려지면 안 된다.
- 골든셋 일부 문항만 견준 결과로 "비열등 확인"을 내면 안 된다.
- 본 회차(각자 채점)가 유효성 미달이면 판정을 미룬다.
- 심사자가 보는 글(문항 기준표·두 답)이 바뀌면 요청 번호가 바뀐다. 판정 계산을 고쳐도 심사 지문은 그대로다.
"""

import copy
import json

from silvercare_autoqa import compare as C
from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.pipeline import finalize, import_judge, prep
from silvercare_autoqa.responders import reference_rows

from .conftest import make_judgment


def _judge_all(run, items):
    for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if not c["judged"]:
            continue
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        j = make_judgment(items[c["item_id"]], c["response"], case_id=c["pid"],
                          det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids])
        (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")


def _tie_all(run, a, b):
    meta = C.prep(run, a, b)
    folder = run / "compare" / f"{a}__vs__{b}"
    for p in meta["pairs"]:
        for order in ("AB", "BA"):
            pid = f"p{p['key']}{order}"
            obj = {"case_id": pid, **{k: "tie" for k in C.CRITERIA}, "why": "-"}
            (folder / "results" / f"{pid}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return C.finalize(run, a, b)


def test_hold_on_unpaired_item_still_fails_safety(tmp_path, golden, items):
    ref = {r["item_id"]: r["response"] for r in reference_rows(golden)}
    rows = [{"model": "ours", "item_id": "SC-E01", "response": "좀 누워서 쉬어 보세요."}]     # 응급 놓침 — 상대 답이 없는 문항
    for i in ("SC-M08", "SC-K03"):
        rows += [{"model": "ours", "item_id": i, "response": ref[i]}, {"model": "theirs", "item_id": i, "response": ref[i]}]
    run = tmp_path / "run"
    prep(DEFAULT_GOLDEN, rows, run, allow_partial=True)
    _judge_all(run, items)
    import_judge(run)
    assert finalize(run)["validity"]["ok"]
    s = _tie_all(run, "ours", "theirs")
    assert s["hold"]["A"] == ["SC-E01"] and s["hold"]["A_paired"] == []
    assert not s["checks"]["safety_ok"] and s["decision"].startswith("안전 미달")
    assert "전체에서 셈" in (run / "compare" / "ours__vs__theirs" / "report.md").read_text(encoding="utf-8")


def test_partial_comparison_is_held_not_confirmed(tmp_path, golden, items):
    ref = {r["item_id"]: r["response"] for r in reference_rows(golden)}
    rows = []
    for i in ("SC-M08", "SC-K03", "SC-P02"):
        rows += [{"model": "ours", "item_id": i, "response": ref[i]}, {"model": "theirs", "item_id": i, "response": ref[i]}]
    run = tmp_path / "run"
    prep(DEFAULT_GOLDEN, rows, run, allow_partial=True)
    _judge_all(run, items)
    import_judge(run)
    finalize(run)
    s = _tie_all(run, "ours", "theirs")
    assert s["checks"]["safety_ok"] and not s["full"]
    assert s["decision"].startswith("판정 보류") and "3문항만" in s["decision"]


def test_full_comparison_can_confirm(tmp_path, golden, items):
    rows = []
    for r in reference_rows(golden):
        rows += [dict(r, model="ours"), dict(r, model="theirs")]
    run = tmp_path / "run"
    prep(DEFAULT_GOLDEN, rows, run)
    _judge_all(run, items)
    import_judge(run)
    assert finalize(run)["validity"]["ok"]
    s = _tie_all(run, "ours", "theirs")
    assert s["full"] and s["base_ok"] and s["n_pairs"] == len(golden["items"])
    assert s["decision"] == "비열등 확인"


def test_invalid_base_run_holds_the_decision(tmp_path, golden, items):
    """반박 검증이 빠진 본 회차로는 비열등을 말하지 않는다."""
    rows = []
    for r in reference_rows(golden):
        rows += [dict(r, model="ours"), dict(r, model="theirs")]
    run = tmp_path / "run"
    prep(DEFAULT_GOLDEN, rows, run)
    _judge_all(run, items)
    # theirs 의 응급 문항 하나를 '필수답 미충족'으로 채점 → Hold 지적이 생기지만 반박 검증 결과는 넣지 않는다
    for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if c["model"] == "theirs" and c["item_id"] == "SC-E03":
            it = items["SC-E03"]
            j = make_judgment(it, c["response"], case_id=c["pid"], met={m["id"]: False for m in it["must_include"]}, concl=False)
            (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    import_judge(run)
    out = finalize(run)
    assert not out["validity"]["ok"]
    s = _tie_all(run, "ours", "theirs")
    assert not s["base_ok"] and s["decision"].startswith("판정 보류 (본 회차 유효성 미달")


def test_pair_key_follows_what_the_judge_sees(items):
    fp = C.compare_fingerprint("m", "high")
    it = items["SC-M08"]
    k = C.pair_key(fp, it, "답 가", "답 나")
    assert k == C.pair_key(fp, it, "답 가", "답 나")
    assert k != C.pair_key(fp, it, "답 가", "답 다")                      # 답이 바뀌면
    changed = copy.deepcopy(it)
    changed["must_include"][0]["text"] += " (고침)"
    assert k != C.pair_key(fp, changed, "답 가", "답 나")                 # 문항 기준이 바뀌면
    assert k != C.pair_key(C.compare_fingerprint("m2", "high"), it, "답 가", "답 나")   # 채점 모델이 바뀌면


def test_compare_fingerprint_ignores_decision_code():
    import hashlib

    src = C.PAIR_SYSTEM + "|" + json.dumps(C.pair_schema(), ensure_ascii=False, sort_keys=True)
    assert C.compare_fingerprint("m", "high") == hashlib.sha256(f"{src}|m|high".encode("utf-8")).hexdigest()[:12]
    assert len(C.decision_fingerprint()) == 12
