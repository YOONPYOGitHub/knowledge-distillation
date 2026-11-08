"""어르신 사용자 실험 준비물·분석 시험 — 순서 상쇄, Hold 답 차단, 통계 계산."""

import csv
import json
import math

import pytest

from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.pipeline import finalize, import_judge, prep
from silvercare_autoqa.responders import reference_rows
from silvercare_autoqa.study import (MAIN_ITEMS, analyze, bootstrap_diff, friedman, holm, latin_schedule, materials,
                                     render_analysis_md, schedule_balance, wilcoxon_exact)

from .conftest import make_judgment


def _run(tmp_path, golden, items, bad=None):
    """teacher·kd·ft 세 모델 모두 모범 답안. bad=(모델, 문항) 이면 그 답을 응급 놓침 답으로."""
    rows = []
    for m in ("teacher_ft", "student_kd", "student_ft"):
        rows += reference_rows(golden, m)
    for r in rows:
        if bad and (r["model"], r["item_id"]) == bad:
            r["response"] = "좀 누워서 쉬어 보세요."
    run = tmp_path / "run"
    prep(DEFAULT_GOLDEN, rows, run)
    for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(make_judgment(
            items[c["item_id"]], c["response"], case_id=c["pid"],
            det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids]), ensure_ascii=False), encoding="utf-8")
    import_judge(run)
    finalize(run)
    return run


def test_latin_schedule_is_balanced():
    sched = latin_schedule(MAIN_ITEMS, ["teacher", "kd", "ft"], 15, seed=1)
    b = schedule_balance(sched)
    assert b["item_condition_min"] == b["item_condition_max"] == 5          # 15명 → 문항×조건 5번씩
    assert all(sorted(pp.values()) in ([3, 3, 4], [3, 4, 3]) or sum(pp.values()) == 10 for pp in b["per_participant"])
    assert all(len({it for it, _ in s}) == 10 for s in sched)                # 같은 이야기를 두 번 듣지 않음


def test_materials_bundle_and_blinding(tmp_path, golden, items):
    run = _run(tmp_path, golden, items)
    rep = materials(run, tmp_path / "b", n_participants=3)
    assert rep["main_items"] == MAIN_ITEMS and not rep["swaps"]
    script = (tmp_path / "b" / "P01" / "진행대본.html").read_text(encoding="utf-8")
    for name in ("teacher_ft", "student_kd", "student_ft", "teacher", "Teacher"):
        assert name not in script, name                                    # 대본에 모델·조건 이름이 없음 (블라인드)
    key = json.loads((tmp_path / "b" / "대응표_봉인.json").read_text(encoding="utf-8"))
    assert len([t for t in key["trials"].values() if t["phase"] == "main"]) == 30
    assert (tmp_path / "b" / "공통_인쇄물" / "동의서.html").exists()


def test_hold_answer_is_swapped_out(tmp_path, golden, items):
    run = _run(tmp_path, golden, items, bad=("student_kd", "SC-E01"))          # KD 의 E01 대답이 Hold
    rep = materials(run, tmp_path / "b", n_participants=3)
    assert "SC-E01" not in rep["main_items"] and rep["swaps"][0]["from"] == "SC-E01" and rep["swaps"][0]["to"]
    key = json.loads((tmp_path / "b" / "대응표_봉인.json").read_text(encoding="utf-8"))
    assert all(t["item_id"] != "SC-E01" for t in key["trials"].values())   # 어르신께 Hold 대답을 들려주지 않음


def test_hold_without_spare_refuses(tmp_path, golden, items, monkeypatch):
    import silvercare_autoqa.study as S
    run = _run(tmp_path, golden, items, bad=("student_kd", "SC-E01"))
    monkeypatch.setitem(S.SPARE, "risk", [])
    with pytest.raises(ValueError, match="Hold"):
        materials(run, tmp_path / "b", n_participants=3)


def test_statistics_known_values():
    # Wilcoxon: 차이 [1,2,3,4,5] 모두 양수 → W+=15, 정확 p = 2/32 = 0.0625
    w = wilcoxon_exact([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert w["W+"] == 15 and abs(w["p"] - 0.0625) < 1e-9 and w["r_rb"] == 1.0
    # Friedman: 참가자 4명이 모두 같은 순서 → 순위합 4·8·12, χ²=8, p=exp(−4)
    f = friedman([[1, 2, 3]] * 4)
    assert abs(f["chi2"] - 8.0) < 1e-9 and abs(f["p"] - math.exp(-4)) < 1e-12 and abs(f["kendall_w"] - 1.0) < 1e-9
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h == {"a": 0.03, "c": 0.06, "b": 0.06}
    m, lo, hi = bootstrap_diff([1.0] * 10, [0.5] * 10)
    assert m == 0.5 and lo == 0.5 and hi == 0.5


def test_analyze_end_to_end(tmp_path, golden, items):
    run = _run(tmp_path, golden, items)
    materials(run, tmp_path / "b", n_participants=6)
    key = json.loads((tmp_path / "b" / "대응표_봉인.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((tmp_path / "b" / "입력서식_본과제.csv").read_text(encoding="utf-8-sig").splitlines()))
    for r in rows:
        t = key["trials"][r["trial"]]
        good = t["condition"] != "ft"                                         # FT 대답만 덜 알아듣는다고 가정
        r["restate_code"] = "맞음" if good else "틀림"
        r["restate_code_2"] = r["restate_code"]
        r["mcq_choice"] = str(t["mcq_answer"] if good else (t["mcq_answer"] % 4) + 1)
        r["satisfaction"] = "5" if good else "2"
    f = tmp_path / "filled.csv"
    with f.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rep = analyze(f, tmp_path / "b" / "대응표_봉인.json", run_dir=run)
    m = rep["metrics"]["행동 이해율"]
    assert m["by_condition"]["kd"]["mean"] == 1.0 and m["by_condition"]["ft"]["mean"] == 0.0
    assert m["pairwise"]["kd vs ft"]["p"] < 0.05 and m["kd_minus_teacher"]["almost_same"]
    assert rep["restate_coder_kappa"]["kappa"] == 1.0 and not rep["errors"] and rep["rq1"]
    md = render_analysis_md(rep)
    assert "## 한눈에" in md and "차이가 거의 없음" in md


def test_analyze_flags_bad_values(tmp_path, golden, items):
    run = _run(tmp_path, golden, items)
    materials(run, tmp_path / "b", n_participants=3)
    rows = list(csv.DictReader((tmp_path / "b" / "입력서식_본과제.csv").read_text(encoding="utf-8-sig").splitlines()))
    rows[0]["satisfaction"] = "7"
    rows[1]["restate_code"] = "아마"
    f = tmp_path / "bad.csv"
    with f.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert len(analyze(f, tmp_path / "b" / "대응표_봉인.json")["errors"]) == 2
