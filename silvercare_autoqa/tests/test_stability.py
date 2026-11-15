"""같은 답을 두 번 채점해 나쁜 쪽 판정을 쓰는 기능 (stability.py) 시험 — 모의 채점기."""

import json
from pathlib import Path

import pytest

from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.pipeline import finalize, import_judge, prep
from silvercare_autoqa.responders import reference_rows
from silvercare_autoqa.stability import compare_runs, merge_worst, pick_worse

from .conftest import make_judgment

IDS = ["SC-E01", "SC-K06", "SC-M01", "SC-C01"]


def _rows(golden):
    return [r for r in reference_rows(golden) if r["item_id"] in IDS]


def _judge_run(run: Path, items, golden, bad: set[str]) -> None:
    """모의 채점: bad 문항은 필수답을 모두 미충족으로, 나머지는 모두 충족으로 채점하고 finalize 까지 끝낸다."""
    prep(DEFAULT_GOLDEN, _rows(golden), run, label=run.name, allow_partial=True)
    for c in (json.loads(l) for l in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines()):
        it = items[c["item_id"]]
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        rv = [{"hid": h, "confirmed": False, "why": "-"} for h in hids]
        if c["item_id"] in bad:
            j = make_judgment(it, c["response"], case_id=c["pid"], met={m["id"]: False for m in it["must_include"]},
                              concl=False, det_review=rv)
        else:
            j = make_judgment(it, c["response"], case_id=c["pid"], det_review=rv)
        (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    import_judge(run)
    for p in (run / "packets" / "verify").glob("*.json"):           # Hold 가 될 지적은 모두 '유지'로 반박 검증
        pk = json.loads(p.read_text(encoding="utf-8"))
        (run / "results" / "verify" / p.name).write_text(json.dumps(
            {"case_id": pk["packet_id"], "decisions": [{"fid": f, "decision": "keep", "evidence": "", "why": "-"} for f in pk["fids"]]},
            ensure_ascii=False), encoding="utf-8")
    assert finalize(run)["validity"]["ok"]


def _verdicts(run: Path) -> dict[str, str]:
    return {c["item_id"]: c["result"]["verdict"] for c in
            (json.loads(l) for l in (run / "scored.jsonl").read_text(encoding="utf-8").splitlines())}


def test_merge_takes_the_worse_verdict_per_item(tmp_path, golden, items):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "merged"
    _judge_run(a, items, golden, bad={"SC-K06"})       # 첫 채점: 일상 문항 하나가 Fix
    _judge_run(b, items, golden, bad={"SC-E01"})       # 두 번째 채점: 응급 문항이 Hold
    assert _verdicts(a)["SC-K06"] == "fix" and _verdicts(a)["SC-E01"] == "pass"
    assert _verdicts(b)["SC-E01"] == "hold" and _verdicts(b)["SC-K06"] == "pass"

    s = merge_worst(a, b, out)
    assert s["n"] == 4 and s["flipped_common"] == 2
    assert s["by_model"]["reference"]["items_flipped_common"] == ["SC-E01", "SC-K06"]
    res = finalize(out)
    v = _verdicts(out)
    assert v == {"SC-E01": "hold", "SC-K06": "fix", "SC-M01": "pass", "SC-C01": "pass"}      # 한 번이라도 나쁘면 나쁜 쪽
    assert res["validity"]["ok"] and res["models"]["reference"]["final"] == "Hold"
    assert "두 번 채점" in (out / "report.md").read_text(encoding="utf-8")
    assert json.loads((out / "stability.json").read_text(encoding="utf-8"))["summary"]["flipped_common"] == 2
    assert finalize(out)["models"]["reference"]["verdicts"] == res["models"]["reference"]["verdicts"]   # 다시 돌려도 같다


def test_merge_refuses_different_criteria_or_used_folder(tmp_path, golden, items):
    a, b = tmp_path / "a", tmp_path / "b"
    _judge_run(a, items, golden, bad=set())
    _judge_run(b, items, golden, bad=set())
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="이미 내용"):
        merge_worst(a, b, tmp_path / "used")
    meta = json.loads((b / "meta.json").read_text(encoding="utf-8"))
    meta["fingerprint"] = "0" * 16
    (b / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="지문이 다름"):
        compare_runs(a, b)


def test_copied_judgments_are_not_two_passes(tmp_path, golden, items):
    """한 번 채점한 결과를 복사한 두 회차는 '두 번 채점'으로 받지 않는다."""
    a, b = tmp_path / "a", tmp_path / "b"
    _judge_run(a, items, golden, bad=set())
    _judge_run(b, items, golden, bad=set())           # 모의 채점기는 같은 글을 쓰므로 결과 파일이 글자까지 같다
    with pytest.raises(ValueError, match="글자까지 같다"):
        merge_worst(a, b, tmp_path / "out")


def test_diff_runs_separates_judge_noise_from_changed_answers(tmp_path, golden, items):
    from silvercare_autoqa.stability import diff_runs, render_diff

    a, b = tmp_path / "a", tmp_path / "b"
    _judge_run(a, items, golden, bad={"SC-K06"})
    _judge_run(b, items, golden, bad=set())            # 같은 답·같은 기준인데 한 문항 판정이 다름 → 채점기 흔들림
    d = diff_runs(a, b)
    assert d["same_judging"] and d["same_scoring"]
    k = d["by_model"]["reference"]["kinds"]
    assert set(k) == {"judge"} and k["judge"]["changed"] == 1 and k["judge"]["better"] == ["SC-K06"]

    # 뒤 회차에서 한 문항의 답이 바뀜 → 그 문항만 '답 바뀜'으로 나뉜다
    c = tmp_path / "c"
    rows = _rows(golden)
    for r in rows:
        if r["item_id"] == "SC-K06":
            r["response"] = "국물은 남기고 건더기 위주로 드세요. 혈압약은 그대로 드시고 다음 진료 때 의사와 상의하세요."
    prep(DEFAULT_GOLDEN, rows, c, label="c", allow_partial=True)
    for case in (json.loads(l) for l in (c / "cases.jsonl").read_text(encoding="utf-8").splitlines()):
        it = items[case["item_id"]]
        hids = [h["hid"] for h in case["det"]["hits"] if h["kind"] == "hybrid"]
        j = make_judgment(it, case["response"], case_id=case["pid"], det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids])
        (c / "results" / "judge" / f"{case['pid']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    import_judge(c)
    finalize(c)
    d2 = diff_runs(a, c)
    k2 = d2["by_model"]["reference"]["kinds"]
    assert k2["answer"]["n"] == 1 and k2["answer"]["better"] == ["SC-K06"] and k2["judge"]["n"] == 3
    text = render_diff(d2)
    assert "답 바뀜" in text and "채점기 흔들림" in text and "SC-K06" in text

    # 기준이 다른 회차와 견주면 경고가 붙고, 답이 그대로인 문항은 '기준 바뀜'으로 나뉜다
    meta = json.loads((b / "meta.json").read_text(encoding="utf-8"))
    meta["judging_fp"], meta["fingerprint"] = "f" * 16, "e" * 16
    (b / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    d3 = diff_runs(a, b)
    assert not d3["same_judging"] and set(d3["by_model"]["reference"]["kinds"]) == {"criteria"}
    assert "채점 기준이 다르다" in render_diff(d3)


def test_pick_worse_rules():
    def case(v, vc=None, err=False):
        return {"result": {"verdict": v, "verdict_common": vc or v, "errors": ["x"] if err else []}}
    assert pick_worse(case("pass"), case("fix")) == "b"
    assert pick_worse(case("hold"), case("fix")) == "a"
    assert pick_worse(case("pass"), case("pass")) == "a"                     # 같으면 첫 채점
    assert pick_worse(case("fix", "pass"), case("fix", "fix")) == "b"        # 견줄 때 쓰는 공통 기준이 먼저
    assert pick_worse(case("fix", "pass"), case("pass", "pass")) == "a"      # 공통 기준이 같으면 전체 판정
    assert pick_worse(case("hold", err=True), case("pass")) == "b"           # 한쪽만 채점 오류면 채점된 쪽
    assert pick_worse(case("pass"), case("hold", err=True)) == "a"


def test_diff_names_evidence_change_apart_from_answer_change():
    """답 글은 그대로인데 채점기에 준 근거 자료와 기준이 함께 바뀐 문항을 '답이 바뀜'이라고 부르지 않는다."""
    from silvercare_autoqa.stability import KINDS
    assert "답 글은 그대로" in KINDS["evidence_criteria"] and KINDS["both"] == "답과 기준이 함께 바뀜"
