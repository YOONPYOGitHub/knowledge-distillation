"""사람 평가 연결 시험 — 블라인드 채점지 · 라벨 반입과 κ · 어르신 실험 답 점검."""

import csv
import json

from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.human import COLS, cohen_kappa, export_blind, import_labels, study_check
from silvercare_autoqa.pipeline import finalize, import_judge, prep
from silvercare_autoqa.responders import reference_rows

from .conftest import make_judgment


def _finalized_run(tmp_path, golden, items):
    rows = reference_rows(golden, "student_kd") + reference_rows(golden, "student_ft")
    for r in rows:
        if r["model"] == "student_ft" and r["item_id"] == "SC-E01":
            r["response"] = "좀 누워서 쉬어 보세요."                        # 응급 놓침 (결정 룰 확정)
    run = tmp_path / "run"
    prep(DEFAULT_GOLDEN, rows, run)
    for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        j = make_judgment(items[c["item_id"]], c["response"], case_id=c["pid"],
                          det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids])
        (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    import_judge(run)
    finalize(run)
    return run


def test_export_blind_hides_model_and_verdict(tmp_path, golden, items):
    run = _finalized_run(tmp_path, golden, items)
    rep = export_blind(run, tmp_path / "blind", sample=10)
    sheet = (tmp_path / "blind" / "blind_sheet.csv").read_text(encoding="utf-8-sig")
    assert rep["rows"] == 10 and rep["by_autoqa_verdict"].get("hold") == 1          # Hold 를 빠뜨리지 않고 뽑는다
    assert "student_kd" not in sheet and "student_ft" not in sheet               # 모델 이름 없음
    header, *data = list(csv.reader(sheet.splitlines()))
    assert header == COLS
    label_cols = range(COLS.index("응답") + 1, len(COLS))
    assert all(r[i] == "" for r in data for i in label_cols)                    # autoqa 판정·Gate 를 미리 채우지 않음
    key = json.loads((tmp_path / "blind" / "blind_key.json").read_text(encoding="utf-8"))
    assert len(key["rows"]) == 10 and all("model" in v for v in key["rows"].values())


def test_import_labels_kappa_and_false_pass(tmp_path, golden, items):
    run = _finalized_run(tmp_path, golden, items)
    export_blind(run, tmp_path / "blind", models=["student_kd", "student_ft"])
    key = json.loads((tmp_path / "blind" / "blind_key.json").read_text(encoding="utf-8"))
    rows = list(csv.reader((tmp_path / "blind" / "blind_sheet.csv").read_text(encoding="utf-8-sig").splitlines()))
    g1, verdict_col = COLS.index("G1 응급 놓침(예/아니오)"), COLS.index("종합 판정(Hold/Fix/Pass)")
    target = next(rid for rid, v in key["rows"].items() if v["model"] == "student_kd" and v["item_id"] == "SC-E02")
    for r in rows[1:]:
        k = key["rows"][r[0]]
        hold = k["autoqa_verdict"] == "hold" or r[0] == target                    # 사람이 한 건을 더 Hold 로 봄
        r[g1] = "예" if hold else "아니오"
        r[verdict_col] = "Hold" if hold else "Pass"
    labeled = tmp_path / "검수자A.csv"
    with labeled.open("w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    rep = import_labels(run, tmp_path / "blind" / "blind_key.json", [labeled])
    g = rep["graders"]["검수자A"]
    assert g["n_labeled"] == 100 and len(g["false_pass"]) == 1 and g["false_pass"][0]["item_id"] == "SC-E02"
    assert 0 < g["kappa_hold"] < 1
    assert g["expert_by_model"]["student_ft"]["응급 놓침(G1)"] == 1


def test_cohen_kappa():
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert cohen_kappa([("a", "b"), ("b", "a")]) == -1.0
    assert cohen_kappa([]) is None
    # 손으로 푼 2×2 표: 둘 다 Hold 4 · autoqa만 2 · 사람만 1 · 둘 다 아님 8 → po=0.8, pe=0.5333, κ=0.5714
    t = [("h", "h")] * 4 + [("h", "n")] * 2 + [("n", "h")] * 1 + [("n", "n")] * 8
    assert abs(cohen_kappa(t) - 0.5714) < 1e-3


def test_bad_label_values_are_errors(tmp_path, golden, items):
    run = _finalized_run(tmp_path, golden, items)
    export_blind(run, tmp_path / "blind", models=["student_kd"])
    rows = list(csv.reader((tmp_path / "blind" / "blind_sheet.csv").read_text(encoding="utf-8-sig").splitlines()))
    rows[1][COLS.index("G1 응급 놓침(예/아니오)")] = "아마도"
    rows[2][COLS.index("행동 가능성(1~5)")] = "7"
    f = tmp_path / "b.csv"
    with f.open("w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    rep = import_labels(run, tmp_path / "blind" / "blind_key.json", [f])
    assert len(rep["graders"]["b"]["errors"]) == 2 and rep["graders"]["b"]["n_labeled"] == 0


def test_study_check_blocks_hold_answers(tmp_path, golden, items):
    run = _finalized_run(tmp_path, golden, items)
    study = [iid for iid, it in items.items() if it.get("study_set")]
    rep = study_check(run, ["student_kd", "student_ft"], items=study + ["SC-E01"])
    assert not rep["ok"] and any(b["model"] == "student_ft" and b["item_id"] == "SC-E01" for b in rep["blocked"])
    assert study_check(run, ["student_kd"], items=study)["ok"]
