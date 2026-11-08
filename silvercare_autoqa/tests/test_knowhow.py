"""쇼핑 AI 검수 노하우 이식 점검(2026-11-08) 보완 M1~M5·K59 시험. id 는 docs/autoqa/AUDIT_LOG.md §4."""

import json
from pathlib import Path

import pytest

from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.mutate import QUOTA, mutants_for, sample_mutants, score_run
from silvercare_autoqa.pipeline import finalize, import_judge, prep
from silvercare_autoqa.prompts import JUDGE_SYSTEM, RUBRIC_PRINCIPLES, SEVERITY, VERIFY_SYSTEM, build_verify_user, item_block
from silvercare_autoqa.responders import reference_rows
from silvercare_autoqa.rules import det_checks
from silvercare_autoqa.tools import contamination, gen_diff

from .conftest import make_judgment

ROOT = Path(__file__).resolve().parents[2]


# ── M2: 채점자와 반박 검증자가 같은 기준·같은 문항 블록 ──

def test_m2_judge_and_verifier_share_rubric(golden):
    assert RUBRIC_PRINCIPLES in JUDGE_SYSTEM and RUBRIC_PRINCIPLES in VERIFY_SYSTEM
    assert SEVERITY in JUDGE_SYSTEM and SEVERITY in VERIFY_SYSTEM
    it = golden["by_id"]["SC-E01"]
    hyb = [h for h in det_checks(it, "괜찮아요 지켜보세요.").to_dict()["hits"] if h["kind"] == "hybrid"]
    u = build_verify_user("v1", it, "응답", [{"fid": "F1", "gate_label": "응급 놓침", "claim": "c", "evidence": ""}],
                          ["F1"], hyb, golden["sources"])
    assert item_block(it, hyb, golden["sources"]) in u


# ── M4: 변이 시험 점수를 명령으로 다시 낸다 · 할당 부족은 멈춤 · 맹점 변이 ──

def test_m4_mutation_score_reproduces_committed_numbers():
    run = ROOT / "docs" / "autoqa" / "verification_20261108" / "verify_20261108"
    rep = score_run(run)
    assert rep["defects_detected"] == [77, 83] and rep["good_passed"] == [58, 58] and rep["gate_attributed"] == [49, 49]


def test_m4_quota_shortfall_and_blind_spot_mutants(golden):
    assert sample_mutants(golden, strict=True)                               # v0 는 할당을 채운다
    with pytest.raises(ValueError, match="할당 부족"):
        sample_mutants(golden, quota=dict(QUOTA, **{"D-COND-ESC": 999}), strict=True)
    ms = {m["mutant"]: m for m in mutants_for(golden["by_id"]["SC-E01"])}
    cond, neg = ms["D-COND-ESC"]["response"], ms["D-NEG-ESC"]["response"]
    assert "심해지면 119" in cond and "119까지 부를 필요는 없어요" in neg
    for r in (cond, neg):                                                     # 결정 룰은 이 맹점을 못 잡는다 → 채점기 몫
        assert det_checks(golden["by_id"]["SC-E01"], r).escalation_unmet_confirmed == []
    # 위기 문항은 '지금 바로 109' 가 주 연결이라 119 만 조건부로 바꾸면 결함이 아니다 (재검증에서 잘못 만든 변이)
    assert not [m for m in mutants_for(golden["by_id"]["SC-E11"]) if m["mutant"] == "D-COND-ESC"]


# ── M5: 생성 조건이 다른 모델끼리의 비교는 표시한다 ──

def test_m5_generation_conditions_flagged(tmp_path, golden, items):
    a = [dict(r, meta={"prompt_format": "raw", "temperature": 0.0, "max_new_tokens": 160, "repetition_penalty": 1.1})
         for r in reference_rows(golden, "student_kd")]
    b = [dict(r, meta={"prompt_format": "raw", "temperature": 0.0, "max_new_tokens": 160, "repetition_penalty": 1.0})
         for r in reference_rows(golden, "student_kd_q4")]
    run = tmp_path / "r"
    meta = prep(DEFAULT_GOLDEN, a + b, run)
    for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(make_judgment(
            items[c["item_id"]], c["response"], case_id=c["pid"],
            det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids]), ensure_ascii=False), encoding="utf-8")
    import_judge(run)
    out = finalize(run, pairs=[("student_kd", "student_kd_q4")])
    assert out["pairs"][0]["gen_diff"] and "repetition_penalty" in out["pairs"][0]["gen_diff"][0]
    assert "생성 조건" in (run / "report.md").read_text(encoding="utf-8") and "⚠ 다름" in (run / "report.md").read_text(encoding="utf-8")
    assert gen_diff([{"prompt_format": "raw"}], [{"prompt_format": None}]) == []            # 모르는 값은 비교하지 않음


# ── K59: 오염 검사 결과에 학습 문장 원문을 넣지 않는다 (AI Hub 원문 반출 금지) ──

def test_k59_contamination_output_has_no_train_text(tmp_path, golden):
    it = golden["by_id"]["SC-E01"]
    d = tmp_path / "t"
    d.mkdir()
    (d / "a.txt").write_text(it["reference_answer"] + "\n", encoding="utf-8")
    rep = contamination([str(d)], str(DEFAULT_GOLDEN))
    assert rep["hits"] and all("text" not in h for h in rep["hits"]) and not rep["contains_train_text"]
    rep2 = contamination([str(d)], str(DEFAULT_GOLDEN), show_text=True)
    assert all("text" in h for h in rep2["hits"]) and rep2["contains_train_text"]
