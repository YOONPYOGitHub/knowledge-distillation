"""사람 평가 연결 시험 — 블라인드 채점지 · 라벨 반입과 κ · 어르신 실험 답 점검."""

import csv
import json
import zipfile

import pytest

from silvercare_autoqa.cli import main
from silvercare_autoqa.gates import GATES
from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.human import (ACT_COL, COLS, GUIDE, REF_COL, VERDICT_COL, VERDICTS, _yn, cohen_kappa, export_blind,
                                     import_labels, read_labels, study_check)
from silvercare_autoqa.pipeline import finalize, import_judge, prep
from silvercare_autoqa.report import RISK_SHORT
from silvercare_autoqa.responders import reference_rows

from .conftest import make_judgment


def _finalized_run(tmp_path, golden, items, edits=None):
    rows = reference_rows(golden, "student_kd") + reference_rows(golden, "student_ft")
    for r in rows:
        if r["model"] == "student_ft" and r["item_id"] == "SC-E01":
            r["response"] = "좀 누워서 쉬어 보세요."                        # 응급 놓침 (결정 룰 확정)
        r["response"] = (edits or {}).get((r["model"], r["item_id"]), r["response"])
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
    assert g["n_labeled"] == 2 * len(items) and len(g["false_pass"]) == 1 and g["false_pass"][0]["item_id"] == "SC-E02"
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


# ── 첫 독자 리뷰(2026-11-08) H-1~H-10: 약사·간호사가 받는 채점지·안내 ──

def _filled_sheet(tmp_path, golden, items, models=("student_kd",)) -> tuple:
    """채점지를 만들고 Hold 답에는 G1 예·Hold, 나머지는 아니오·Pass 로 채운 줄을 돌려준다."""
    run = _finalized_run(tmp_path, golden, items)
    export_blind(run, tmp_path / "blind", models=list(models))
    key = json.loads((tmp_path / "blind" / "blind_key.json").read_text(encoding="utf-8"))
    rows = list(csv.reader((tmp_path / "blind" / "blind_sheet.csv").read_text(encoding="utf-8-sig").splitlines()))
    g1, vcol = COLS.index("G1 응급 놓침(예/아니오)"), COLS.index(VERDICT_COL)
    for r in rows[1:]:
        hold = key["rows"][r[0]]["autoqa_verdict"] == "hold"
        r[g1], r[vcol] = ("예", "Hold") if hold else ("아니오", "Pass")
    return run, tmp_path / "blind" / "blind_key.json", rows


def _write(path, rows, encoding="utf-8-sig"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, errors="replace", newline="") as fh:     # cp949 에 없는 글자는 응답 칸에만 있다
        csv.writer(fh).writerows(rows)
    return path


def test_h5_sheet_risk_column_is_korean(tmp_path, golden, items):
    run = _finalized_run(tmp_path, golden, items)
    export_blind(run, tmp_path / "blind", models=["student_kd"])
    rows = list(csv.DictReader((tmp_path / "blind" / "blind_sheet.csv").read_text(encoding="utf-8-sig").splitlines()))
    assert {r["위험 등급"] for r in rows} == {"응급", "긴급(오늘~며칠 안 진료)", "일상"} == set(RISK_SHORT.values())


def test_sheet_text_cells_do_not_open_as_excel_formulas(tmp_path, golden, items):
    """'엑셀로 열어 적어도 됩니다' 가 참이려면, '-' 로 시작하는 답이 엑셀에서 수식(#NAME?)이 되면 안 된다."""
    dash = "- " + items["SC-K03"]["reference_answer"]
    run = _finalized_run(tmp_path, golden, items, edits={("student_kd", "SC-K03"): dash})
    export_blind(run, tmp_path / "blind", models=["student_kd"])
    rows = list(csv.DictReader((tmp_path / "blind" / "blind_sheet.csv").read_text(encoding="utf-8-sig").splitlines()))
    assert next(r for r in rows if r["문항"] == "SC-K03")["응답"] == "'" + dash


def test_guide_is_plain_text_and_matches_the_sheet():
    assert not [l for l in GUIDE.splitlines() if l.lstrip().startswith("|")]          # 메모장에서 깨지는 마크다운 표 없음 (H-9)
    for must in ("blind_sheet.csv", "109", "혼자 판단", "여러 칸이 '예'", "CSV UTF-8", ".xlsx", "보류·수정·통과",
                 "지나친 조심", "응급 = ", "긴급 = ", "일상 = "):
        assert must in GUIDE, must
    for g, name in GATES.items():                       # 칸 이름과 같은 Gate 이름 (H-2·H-6)
        assert f"{g} {name}:" in GUIDE
    for word in ("emergency", "urgent", "routine", "|---"):
        assert word not in GUIDE
    # 안내문이 '된다'고 한 값은 실제로 읽힌다 (H-6·H-8)
    assert _yn("O") and _yn("1") and _yn("예") and _yn("X") is False and _yn("0") is False
    assert {VERDICTS[v] for v in ("보류", "수정", "통과", "Hold".lower())} == {"hold", "fix", "pass"}


def test_xlsx_gets_a_korean_save_as_csv_message(tmp_path, golden, items, capsys):
    run, key, rows = _filled_sheet(tmp_path, golden, items)
    xlsx = tmp_path / "검수자A.xlsx"
    with zipfile.ZipFile(xlsx, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
    with pytest.raises(ValueError, match="CSV UTF-8"):
        read_labels(xlsx)
    renamed = tmp_path / "검수자B.csv"                  # 확장자만 .csv 로 바꾼 엑셀 파일도 같은 안내
    renamed.write_bytes(xlsx.read_bytes())
    with pytest.raises(ValueError, match="엑셀 통합 문서"):
        read_labels(renamed)
    assert main(["import-labels", "--run", str(run), "--key", str(key), "--labels", str(xlsx)]) == 2
    out = capsys.readouterr().out
    assert "❌" in out and "CSV UTF-8" in out and "Traceback" not in out


def test_header_match_ignores_bom_spaces_and_case(tmp_path, golden, items):
    run, key, rows = _filled_sheet(tmp_path, golden, items)
    exact = import_labels(run, key, [_write(tmp_path / "a" / "exact.csv", rows)])["graders"]["exact"]
    messy = [[f" {h} ".replace("응급 놓침", "응급놓침").replace("Hold/Fix/Pass", "hold/fix/pass") for h in rows[0]]] + rows[1:]
    messy[0][0] = chr(0xFEFF) + "번호 "
    got = import_labels(run, key, [_write(tmp_path / "b" / "messy.csv", messy)])["graders"]["messy"]
    assert got["n_labeled"] == exact["n_labeled"] > 0 and got["kappa_hold"] == exact["kappa_hold"] and not got["errors"]
    cp949 = import_labels(run, key, [_write(tmp_path / "c" / "cp949.csv", rows, encoding="cp949")])["graders"]["cp949"]
    assert cp949["n_labeled"] == exact["n_labeled"]                 # 엑셀의 그냥 'CSV(쉼표로 분리)'


def test_missing_label_column_stops_with_column_name(tmp_path, golden, items):
    run, key, rows = _filled_sheet(tmp_path, golden, items)
    i = COLS.index(VERDICT_COL)
    cut = [r[:i] + r[i + 1:] for r in rows]
    with pytest.raises(ValueError, match="종합 판정"):              # 말없이 '라벨 없음'으로 넘어가지 않는다
        import_labels(run, key, [_write(tmp_path / "cut.csv", cut)])


def test_same_file_name_from_two_graders_is_not_merged(tmp_path, golden, items):
    run, key, rows = _filled_sheet(tmp_path, golden, items)
    a = _write(tmp_path / "약사" / "blind_sheet.csv", rows)
    b = _write(tmp_path / "간호사" / "blind_sheet.csv", rows)
    rep = import_labels(run, key, [a, b])
    assert set(rep["graders"]) == {"blind_sheet", "간호사/blind_sheet"}
    assert rep["between_graders"][0]["n_common"] == rep["graders"]["blind_sheet"]["n_labeled"] > 0


def test_actionability_accepts_excel_decimal_and_rejects_half_points(tmp_path, golden, items):
    run, key, rows = _filled_sheet(tmp_path, golden, items)
    act, ref = COLS.index(ACT_COL), COLS.index(REF_COL)
    rows[1][act], rows[1][ref] = "4.0", "예"
    rows[2][act] = "4.5"
    labels, errs = read_labels(_write(tmp_path / "act.csv", rows))
    assert labels[rows[1][0]]["actionability"] == 4 and labels[rows[1][0]]["referral"] is True
    assert len(errs) == 1 and "1~5 숫자 하나가 아님" in errs[0]
