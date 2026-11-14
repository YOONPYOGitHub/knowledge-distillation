"""RAG 전환(2026-11-14) 시험 — 출처 묶음 판별·인용 결정 룰·근거 대조 채점 검사·판정·지표·직접 비교·수집기.

시험 자료의 출처 발췌는 모두 직접 지어 쓴 문장이다 (팀 RAG 자료·공공 자료 문장을 옮기지 않는다).
"""

import json

import pytest

from silvercare_autoqa import compare as C
from silvercare_autoqa import fingerprint as fp_mod
from silvercare_autoqa import golden as G
from silvercare_autoqa import rag as RAG
from silvercare_autoqa import responders as R
from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.judge import validate_judgment
from silvercare_autoqa.metrics import paired_diff_ci
from silvercare_autoqa.pipeline import finalize, import_judge, prep
from silvercare_autoqa.rules import det_checks
from silvercare_autoqa.verdict import verdict_of

from .conftest import make_judgment


def src(i, **kw):
    base = {"id": i, "title": f"자료{i}", "url": "", "excerpt": "지어 쓴 발췌 문장입니다.", "evidence_kind": "reference",
            "source_type": "", "record_id": "", "source_display": "", "source_year": "2026", "source_language": "ko"}
    base.update(kw)
    return base


KDCA = src(1, url="https://health.kdca.go.kr/healthinfo/view.do?cntnts_sn=6698", evidence_kind="official_api")
MEDLINE = src(2, url="https://medlineplus.gov/drugreactions.html", source_type="patient_education", source_language="en")
DIALOGUE = src(3, evidence_kind="dialogue", source_type="dialogue_case")


# ───────────────────────────── 자료 묶음 ─────────────────────────────

@pytest.mark.parametrize("s,fam", [
    (KDCA, "KDCA_API"),
    (src(9, evidence_kind="official_api"), "KDCA_API"),
    (src(9, url="https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71874", evidence_kind="dataset_knowledge"), "AIHUB_GUIDE"),
    (src(9, url="https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71875", evidence_kind="dataset_knowledge"), "AIHUB_ONLINE"),
    (src(9, evidence_kind="dataset_knowledge"), "AIHUB_KNOWLEDGE"),
    (MEDLINE, "MEDLINEPLUS"),
    (src(9, url="https://www.bokjiro.go.kr/x?wlfareInfoId=WLF00003191", record_id="WLF00003191"), "WELFARE_CENTRAL"),
    (src(9, url="https://www.bokjiro.go.kr/a.pdf#page=35", record_id="welfare-2026-26"), "WELFARE_SENIOR_2026"),
    (src(9, url="https://www.mohw.go.kr/board.es?list_no=1", record_id="section2:ab"), "MOHW_CARE"),
    (src(9, source_type="drug_label"), "OPENFDA_LABEL"),
    (src(9, source_type="drug_mechanism"), "FDA_CYP"),
    (DIALOGUE, "CASE_DIALOGUE"),
    (src(9, evidence_kind="qa_case"), "CASE_QA"),
    (src(9, url="https://example.org/x"), "OTHER"),
])
def test_family_of(s, fam):
    assert RAG.family_of(s) == fam


def test_roles_and_groups():
    assert RAG.role_of("CASE_DIALOGUE") == "example" and RAG.role_of("KDCA_API") == "evidence"
    assert {"MOHW_CARE", "WELFARE_CENTRAL"} <= RAG.expand_expect(["WELFARE"])
    assert RAG.unknown_expect(["KDCA_API", "NOPE"]) == ["NOPE"]


# ───────────────────────────── 결정 룰 ─────────────────────────────

def _rag(*sources, mode="grounded"):
    return RAG.normalize_rag({"mode": mode, "sources": list(sources)})


def test_cite_out_of_range_and_example_are_confirmed(items):
    it = items["SC-M08"]
    ans = "드시기 전에 약사에게 물어보세요.[1] 피가 잘 안 멈출 수 있어요.[4] 사례도 있어요.[3]"
    det = det_checks(it, ans, rag=_rag(KDCA, MEDLINE, DIALOGUE)).to_dict()
    rules = {h["rule"] for h in det["hits"]}
    assert {"RAG-CITE-RANGE", "RAG-CITE-EXAMPLE"} <= rules
    assert det["rag"]["out_of_range"] == [4] and det["rag"]["n_sources"] == 3
    r = verdict_of(it, det, make_judgment(it, ans), [], None)
    cats = {f["cat"] for f in r["live_findings"] if f["severity"] == "major"}
    assert r["verdict"] == "fix" and cats == {"grounding"}
    assert r["verdict_common"] == "pass"          # 출처 지적을 빼면 통과 — 상용 LLM 과 같은 잣대


def test_no_rag_answers_get_no_rag_hits(items):
    det = det_checks(items["SC-M08"], "약사에게 물어보세요.[1]").to_dict()
    assert det["rag"] == {"has_rag": False} and not [h for h in det["hits"] if h["rule"].startswith("RAG-")]


def test_kpi_no_source_and_emergency_mode(items):
    det = det_checks(items["SC-M08"], "약사에게 물어보세요.", rag=_rag()).to_dict()
    assert "RAG-NO-SOURCE" in {h["rule"] for h in det["hits"]}
    det = det_checks(items["SC-E01"], "지금 바로 119에 전화하세요.", rag=_rag(mode="grounded")).to_dict()
    assert "RAG-EMERG-MODE" in {h["rule"] for h in det["hits"]}
    det = det_checks(items["SC-E01"], "지금 바로 119에 전화하세요.", rag=_rag(mode="emergency")).to_dict()
    assert "RAG-EMERG-MODE" not in {h["rule"] for h in det["hits"]}


def test_expect_hit(items):
    it = dict(items["SC-M08"], rag={"coverage": "partial", "expect": ["KDCA_API"], "terms": ["아스피린"]})
    f = RAG.case_facts(it, "답[1]", _rag(KDCA))
    assert f["expect_hit"] is True
    f = RAG.case_facts(it, "답[1]", _rag(MEDLINE))
    assert f["expect_hit"] is False


# ───────────────────────────── 근거 대조 채점 검사 ─────────────────────────────

def _grounding(item, issues=(), cov="n/a", faith=5, in_ctx=True):
    return {"issues": list(issues), "key_facts_in_context": [{"id": k["id"], "in_context": in_ctx} for k in item["key_facts"]],
            "coverage_handling": cov, "faithfulness": faith}


def test_validate_grounding(items):
    it = items["SC-M08"]
    ans = "드시기 전에 약사에게 물어보세요.[1]"
    view = RAG.rag_view(_rag(KDCA))
    j = make_judgment(it, ans)
    assert any("grounding 이 없음" in e for e in validate_judgment(j, "c1", it, ans, [], view))
    j["grounding"] = _grounding(it)
    assert validate_judgment(j, "c1", it, ans, [], view) == []
    j["grounding"] = _grounding(it, [{"type": "contradicted", "evidence": "없는 문장", "source_id": 1, "why": "-"}])
    assert any("응답 원문에 없음" in e for e in validate_judgment(j, "c1", it, ans, [], view))
    j["grounding"] = _grounding(it, [{"type": "contradicted", "evidence": "약사에게 물어보세요", "source_id": 7, "why": "-"}])
    assert any("source_id 7" in e for e in validate_judgment(j, "c1", it, ans, [], view))
    j2 = make_judgment(it, ans)
    j2["grounding"] = _grounding(it)
    assert any("RAG 응답이 아님" in e for e in validate_judgment(j2, "c1", it, ans, [], None))


def test_grounding_findings_and_common_verdict(items):
    it = items["SC-M08"]
    ans = "드시기 전에 약사에게 물어보세요.[1] 오메가3는 하루 10g 드셔도 돼요.[1]"
    det = det_checks(it, ans, rag=_rag(KDCA)).to_dict()
    j = make_judgment(it, ans)
    j["grounding"] = _grounding(it, [{"type": "unsupported", "evidence": "오메가3는 하루 10g 드셔도 돼요", "source_id": 0,
                                      "why": "-"}])
    r = verdict_of(it, det, j, [], None)
    assert r["verdict"] == "pass"                 # 발췌가 잘린 채 본 '근거 없음'은 참고(minor)
    j["grounding"]["issues"][0]["type"] = "contradicted"
    r = verdict_of(it, det, j, [], None)
    assert r["verdict"] == "fix" and r["verdict_common"] == "pass"
    j["grounding"] = _grounding(it, cov="fabricated")
    assert verdict_of(it, det, j, [], None)["verdict"] == "fix"


def test_sources_reach_the_judge_prompt_with_boundary(tmp_path, golden):
    rows = [{"model": "ondam", "item_id": "SC-M08", "response": "약사에게 물어보세요.[1]",
             "rag": {"mode": "grounded", "sources": [DIALOGUE | {"id": 1}]}}]
    meta = prep(DEFAULT_GOLDEN, rows, tmp_path / "r", allow_partial=True)
    pk = json.loads(next((tmp_path / "r" / "packets" / "judge").glob("*.json")).read_text(encoding="utf-8"))
    assert "grounding" in pk["schema"]["properties"] and "<<<근거-" in pk["user"] and "사실 근거 아님" in pk["user"]
    assert meta["rag_models"] == ["ondam"]


# ───────────────────────────── 회차 끝까지 + 직접 비교 ─────────────────────────────

def _fake_judge_all(run, items):
    for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if not c["judged"]:
            continue
        it = items[c["item_id"]]
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        j = make_judgment(it, c["response"], case_id=c["pid"],
                          det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids])
        if c.get("rag_view") is not None:
            j["grounding"] = _grounding(it, in_ctx=False)
        (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")


def test_end_to_end_rag_vs_commercial(tmp_path, golden, items):
    ids = ["SC-M08", "SC-K03", "SC-P02", "SC-G01"]
    rows = []
    for i in ids:
        ref = items[i]["reference_answer"]
        rows.append({"model": "ondam_local", "item_id": i, "response": ref + "[1]",
                     "rag": {"mode": "grounded", "sources": [KDCA]}, "meta": {"elapsed_ms": 1500}})
        rows.append({"model": "chatgpt", "item_id": i, "response": ref, "meta": {"elapsed_ms": 900}})
    run = tmp_path / "run"
    prep(DEFAULT_GOLDEN, rows, run, allow_partial=True)
    _fake_judge_all(run, items)
    assert import_judge(run)["error"] == 0
    out = finalize(run, reference="chatgpt", pairs=[("ondam_local", "chatgpt")])
    om = out["models"]["ondam_local"]
    assert om["rag"]["n"] == 4 and om["rag"]["출처 붙은 답 비율"]["value"] == 1.0
    assert om["rag"]["실패 원인(핵심 정보)"].get("자료 밖 지식으로 답함", 0) > 0
    assert out["models"]["chatgpt"]["rag"] is None and om["latency"]["median_ms"] == 1500
    assert out["pairs"][0]["basis"] == "common" and out["pairs"][0]["pass_diff"]["n"] == 4
    md = (run / "report.md").read_text(encoding="utf-8")
    assert "3-2. RAG" in md and "4. 비교 모델(`chatgpt`) 대비" in md and "공통 기준" in md

    meta = C.prep(run, "ondam_local", "chatgpt")
    folder = run / "compare" / "ondam_local__vs__chatgpt"
    assert meta["n_common"] == 4 and len(list((folder / "packets").glob("p*.json"))) == 8
    pk = json.loads(next((folder / "packets").glob("p*AB.json")).read_text(encoding="utf-8"))
    assert "[1]" not in pk["user"] and "온담" not in pk["user"].split("■ 답 1")[1]
    for p in meta["pairs"]:              # 온담이 두 순서 모두 이김 → 승, 하나는 순서 따라 엇갈림 → 비김
        split = p["item_id"] == "SC-G01"
        for order in ("AB", "BA"):
            pid = f"p{p['key']}{order}"
            pick = "1" if order == "AB" else ("1" if split else "2")
            obj = {"case_id": pid, **{k: ("tie" if k == "safety" else pick) for k in C.CRITERIA}, "why": "-"}
            (folder / "results" / f"{pid}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    s = C.finalize(run, "ondam_local", "chatgpt")
    ov = s["criteria"]["overall"]
    assert (ov["A"], ov["tie"], ov["B"], ov["split"]) == (3, 1, 0, 1)
    assert s["checks"]["safety_ok"] and s["decision"] in ("비열등 확인", "비열등 미확인")
    assert (folder / "report.md").exists() and "미리 정한 규칙" in (folder / "report.md").read_text(encoding="utf-8")


def test_pair_validator_and_blinding():
    ok = {"case_id": "x", **{k: "tie" for k in C.CRITERIA}, "why": "-"}
    assert C.validate_pair(ok, "x") == []
    bad = dict(ok, safety="1", overall="2")
    assert any("안전" in e for e in C.validate_pair(bad, "x"))
    t = C.blind_text("**온담**이 알려드려요.[1] ChatGPT 와 달리[2] GPT-6.1 기준")
    assert "[1]" not in t and "**" not in t and "온담" not in t and "ChatGPT" not in t and "GPT-6.1" not in t


def test_paired_diff_ci():
    r = paired_diff_ci(10, 5, 5, 10)
    assert abs(r["diff"]) < 1e-9 and r["ci95"][0] < 0 < r["ci95"][1]
    r = paired_diff_ci(20, 8, 0, 2)
    assert r["diff"] > 0 and r["ci95"][0] > 0
    assert paired_diff_ci(0, 0, 0, 0)["diff"] is None


def test_preference_ci_is_reproducible():
    a = C.preference_ci([1, 0.5, 1, 0, 1, 1])
    b = C.preference_ci([1, 0.5, 1, 0, 1, 1])
    assert a == b and 0 <= a["ci95"][0] <= a["value"] <= a["ci95"][1] <= 1


# ───────────────────────────── 골든셋 칸·지문·수집기 ─────────────────────────────

def test_golden_v04_fields(golden):
    g = {**golden, "items": [dict(golden["by_id"]["SC-M08"])]}
    g["items"][0]["rag"] = {"coverage": "maybe", "expect": ["NOPE"], "terms": []}
    errs = G.validate_golden(g)
    assert any("rag.coverage" in e for e in errs) and any("NOPE" in e for e in errs)
    s = dict(golden["by_id"]["SC-M08"], id="SC-S99", type="supplement")
    errs = G.validate_golden({**golden, "items": [s]})
    assert any("customer_q" in e for e in errs)
    s["customer_q"] = {"group": 3, "label": G.CUSTOMER_GROUPS[3]}
    assert not [e for e in G.validate_golden({**golden, "items": [s]}) if "customer_q" in e]


def test_rag_files_are_in_both_fingerprints():
    names = lambda ps: {p.name for p in ps}
    assert {"rag.py", "rag_sources.yaml"} <= names(fp_mod.JUDGING) & names(fp_mod.SCORING)
    assert "compare.py" not in names(fp_mod.PARTS) and "responders.py" not in names(fp_mod.PARTS)


def test_sheet_round_trip(tmp_path, golden):
    items = [golden["by_id"]["SC-M08"], golden["by_id"]["SC-K03"]]
    p = tmp_path / "s.csv"
    assert R.sheet_template(items, p) == 2
    text = p.read_text(encoding="utf-8-sig").replace("SC-M08,", "SC-M08,").splitlines()
    filled = [text[0]] + [line.replace(",,", ",받은 답입니다.,", 1) for line in text[1:]]
    p.write_text("\n".join(filled) + "\n", encoding="utf-8-sig")
    rows = R.sheet_rows(p, "chatgpt", model_name="ChatGPT luna 6.1", collected_at="2026-11-20", settings="새 채팅")
    assert [r["item_id"] for r in rows] == ["SC-M08", "SC-K03"] and rows[0]["meta"]["source"] == "sheet"
    p.write_text(text[0] + "\n" + text[1] + "\n", encoding="utf-8-sig")
    with pytest.raises(SystemExit):
        R.sheet_rows(p, "chatgpt", model_name="x", collected_at="2026-11-20", settings="-")


class _Resp:
    def __init__(self, code, data):
        self.status_code, self._d, self.text = code, data, json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._d


def test_ondam_api_collector(monkeypatch, golden):
    import requests

    calls = []

    def fake_post(url, json=None, timeout=None, headers=None):
        calls.append(json["message"])
        if "혈압약" in json["message"] or "아스피린" in json["message"]:
            return _Resp(200, {"answer": "약사에게 물어보세요.[1]", "mode": "grounded", "sources": [KDCA],
                               "contexts": [KDCA | {"excerpt": "전체 문맥"}]})
        return _Resp(422, {"detail": "의료진과 상의해 주세요."})

    monkeypatch.setattr(requests, "post", fake_post)
    rows = R.ondam_api([golden["by_id"]["SC-M08"], golden["by_id"]["SC-K03"]], "https://example.test", "ondam",
                       pause=0, log=lambda *_: None)
    assert rows[0]["rag"]["contexts_complete"] is True and rows[0]["meta"]["mode"] == "grounded"
    assert rows[1]["meta"]["refusal"] == "http_422" and rows[1]["response"] == "의료진과 상의해 주세요."


def test_export_blind_can_strip_citations(tmp_path, golden, items):
    """견주는 전문가 채점지에서 인용 번호가 온담 답을 드러내지 않게 지울 수 있다."""
    from silvercare_autoqa.human import export_blind

    rows = [{"model": "ondam_cloud", "item_id": "SC-M08", "response": "약사에게 먼저 물어보세요.[1] 상자를 보여 주세요.[2]",
             "rag": {"mode": "grounded", "sources": [KDCA, MEDLINE]}},
            {"model": "chatgpt", "item_id": "SC-M08", "response": "약사에게 먼저 물어보세요. 상자를 보여 주세요."}]
    run = tmp_path / "run"
    prep(DEFAULT_GOLDEN, rows, run, allow_partial=True)
    _fake_judge_all(run, items)
    import_judge(run)
    finalize(run)
    export_blind(run, tmp_path / "b", strip_cites=True)
    sheet = (tmp_path / "b" / "blind_sheet.csv").read_text(encoding="utf-8-sig")
    assert "[1]" not in sheet and "[2]" not in sheet and "상자를 보여 주세요" in sheet


def test_stale_question_is_flagged(golden):
    from silvercare_autoqa.tools import check_rows

    it = golden["by_id"]["SC-M08"]
    ok = {"model": "ondam", "item_id": "SC-M08", "response": "약사에게 물어보세요.", "meta": {"message": R.user_turn(it)}}
    old = {"model": "ondam2", "item_id": "SC-M08", "response": "약사에게 물어보세요.", "meta": {"message": "예전 질문"}}
    rep = check_rows([ok, old], golden)
    assert "stale_question" in rep["problem_kinds"] and all("ondam2" in m for m in rep["problems"] if "바뀜" in m)


def test_health_question_on_conversation_path_counts_as_no_source(items):
    """온담이 건강 질문을 일상 대화 길(conversation — 검색 생략)로 보내면 출처 없음으로 센다 (2026-11-14 실측: 수면제 과다 복용 문항)."""
    det = det_checks(items["SC-E12"], "그러셨군요. 푹 쉬세요.", rag=_rag(mode="conversation")).to_dict()
    rules = {h["rule"] for h in det["hits"]}
    assert {"RAG-NO-SOURCE", "RAG-EMERG-MODE"} <= rules
    det = det_checks(items["SC-C04"], "날씨가 좋네요.", rag=_rag(mode="conversation")).to_dict()
    assert "RAG-NO-SOURCE" not in {h["rule"] for h in det["hits"]}      # 말동무 문항은 출처가 없어도 된다


def test_conditional_escalation_must_can_be_refuted_with_conditional_quote(items):
    """필수답이 '어지럽거나 피를 토하면 119'처럼 조건부면, 같은 꼴의 119 문장으로 G1 지적을 반박할 수 있다 (실측 SC-K07).
    '지금 바로 119' 필수답은 그대로 무조건 인용만 받는다."""
    from silvercare_autoqa.judge import validate_verify
    from silvercare_autoqa.verdict import conditional_escalation_fids

    it = items["SC-K07"]
    cond_must = next(m for m in it["must_include"] if m["gate"] == "escalation" and "면" in m["text"])
    f = [{"fid": "F1", "gate": "G1", "ref": cond_must["id"], "needs_verify": True}]
    assert conditional_escalation_fids(it, f) == ["F1"]
    resp = "지금 응급실에 가세요. 어지럽거나 쓰러질 것 같으면 119에 전화하세요."
    obj = {"case_id": "v1", "decisions": [{"fid": "F1", "decision": "refute", "evidence": "어지럽거나 쓰러질 것 같으면 119에 전화하세요",
                                           "why": "-"}]}
    assert validate_verify(obj, "v1", ["F1"], resp, {"F1": "G1"}, ["F1"], ["F1"]) == []
    assert any("조건 없는" in e for e in validate_verify(obj, "v1", ["F1"], resp, {"F1": "G1"}, ["F1"], []))
    e01 = items["SC-E01"]
    now_must = next(m for m in e01["must_include"] if m["gate"] == "escalation")
    assert conditional_escalation_fids(e01, [{"fid": "F1", "gate": "G1", "ref": now_must["id"], "needs_verify": True}]) == []


def test_length_is_support_score_not_a_verdict(items, golden):
    """답 길이는 판정이 아니라 보조 점수다 (2026-11-14 사용자 결정). 문장이 많아도 쉬운 표현 점수가 깎이지 않고,
    채점기의 간결 전달 점수는 모델 판정에 들어가지 않는다."""
    from silvercare_autoqa.metrics import TARGETS, SUPPORT_TARGETS

    long = "약사에게 물어보세요. 상자를 보여 주세요. 약 봉투도 가져가세요. 드시던 약은 그대로 드세요. 궁금하면 다시 물어보세요."
    d = det_checks(items["SC-M08"], long)
    assert d.kpi["n_sentences"] == 5 and d.kpi["plain_score"] == 100
    assert "간결 전달 점수(보조)" in SUPPORT_TARGETS and "간결 전달 점수(보조)" not in TARGETS
    bad = make_judgment(items["SC-M08"], long)
    bad["concise_delivery"] = 9
    assert any("concise_delivery" in e for e in validate_judgment(bad, "c1", items["SC-M08"], long, []))
