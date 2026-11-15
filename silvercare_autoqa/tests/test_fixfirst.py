"""「이것부터 고치면」(fixfirst.py) 시험 — 손으로 만든 채점 결과로 순서·분모를 본다. 모델을 부르지 않는다.
담당을 나누는 규칙 자체는 routing.py 와 test_routing.py 에 있다. 여기서는 두 곳이 같은 규칙을 쓰는지와 계산을 본다."""

from silvercare_autoqa import routing
from silvercare_autoqa.fixfirst import NO_OWNER, OWNER_APP, OWNER_DATA, OWNER_ONDAM, fix_first, render_md, unit_of


def _f(cat, severity="major", gate=None, ref=None):
    return {"cat": cat, "severity": severity, "gate": gate, "fid": "F", "claim": "-", "status": "confirmed", "ref": ref}


def _case(item_id, verdict, findings, *, mode="grounded", lost=(), in_ctx=None, rag=True, esc=False):
    gates = sorted({f["gate"] for f in findings if f.get("gate")})
    return {"model": "ondam", "item_id": item_id,
            "rag": {"mode": mode, "sources": [{"id": 1}], "contexts_complete": False} if rag else None,
            "det": {"rag": {"has_rag": rag, "mode": mode}, "hits": [], "escalation_present": esc},
            "judgment": {"key_facts": [{"id": k, "preserved": False} for k in lost],
                         "grounding": {"key_facts_in_context": [{"id": k, "in_context": v} for k, v in (in_ctx or {}).items()]}},
            "result": {"verdict": verdict, "verdict_common": verdict, "live_findings": findings, "gates": gates, "errors": []}}


ITEMS = {
    "A1": {"risk": "emergency", "type": "emergency", "rag": {"coverage": "partial"}},
    "A2": {"risk": "routine", "type": "supplement", "rag": {"coverage": "none"}},
    "A3": {"risk": "routine", "type": "supplement", "rag": {"coverage": "none"}},
    "A4": {"risk": "routine", "type": "chronic", "rag": {"coverage": "covered"},
           "must_include": [{"id": "I1", "gate": "med_safety"}]},
    "A5": {"risk": "emergency", "type": "symptom", "rag": {"coverage": "partial"}},
    "A6": {"risk": "routine", "type": "chronic", "rag": {"coverage": "covered"}},
    "A7": {"risk": "routine", "type": "companion", "rag": {"coverage": "covered"}},
}


def test_units_come_from_the_routing_rules():
    """고칠 순서와 리포트의 담당별 표가 같은 규칙을 쓴다 — 같은 답에 담당이 다르게 나오면 안 된다."""
    cases = [
        _case("A1", "hold", [_f("gate", "critical", "G1")], mode="emergency"),
        _case("A2", "fix", [_f("conclusion")]),
        _case("A4", "fix", [_f("safety_must", ref="I1")]),
        _case("A5", "hold", [_f("gate", "critical", "G1")], esc=True),
        _case("A6", "fix", [_f("conclusion")], mode="refusal"),
    ]
    for c in cases:
        r = routing.route(c, ITEMS[c["item_id"]])
        assert unit_of(c, ITEMS[c["item_id"]]) == (r.owner, r.reason)
    assert unit_of(cases[0], ITEMS["A1"])[0] == OWNER_APP
    assert unit_of(cases[1], ITEMS["A2"])[0] == OWNER_DATA
    assert unit_of(cases[2], ITEMS["A4"]) == (OWNER_ONDAM, routing.ONDAM_SAFETY_LINE)        # 안내 문장은 답 쓰는 규칙
    assert unit_of(cases[3], ITEMS["A5"]) == (OWNER_ONDAM, routing.ONDAM_EMERGENCY_LINE)     # 119 는 있는데 한 줄이 빠짐
    assert unit_of(_case("A6", "fix", [_f("conclusion")], rag=False), ITEMS["A6"])[0] == NO_OWNER


def test_emergency_answer_without_any_escalation_goes_to_the_app_owner():
    c = _case("A5", "hold", [_f("gate", "critical", "G1")], esc=False)
    assert unit_of(c, ITEMS["A5"]) == (OWNER_APP, routing.APP_MISSED)


def test_order_holds_first_and_fixed_denominator():
    cases = [
        _case("A1", "hold", [_f("gate", "critical", "G1")], mode="emergency"),
        _case("A2", "fix", [_f("conclusion")]),
        _case("A3", "fix", [_f("conclusion"), _f("fact")]),
        _case("A4", "fix", [_f("safety_must", ref="I1")]),
        _case("A6", "pass", []),
        _case("A7", "pass", [_f("content", "minor")]),                    # minor 는 통과를 막지 않는다
    ]
    s = fix_first(cases, ITEMS)
    assert (s["n"], s["pass"], s["hold"], s["not_pass"]) == (6, 2, 1, 4)
    assert s["order"][0]["owner"] == OWNER_APP and s["order"][0]["hold_cleared"] == ["A1"]     # Hold 가 풀리는 것이 먼저
    assert s["order"][1]["owner"] == OWNER_DATA and s["order"][1]["items"] == ["A2", "A3"]     # 그다음 통과가 많이 느는 것
    assert s["order"][-1]["pass_after"] == 6 and s["order"][-1]["hold_after"] == 0             # 다 고치면 전부 통과
    assert [o["pass_after"] for o in s["order"]] == sorted(o["pass_after"] for o in s["order"])
    assert sum(s["by_owner"].values()) == s["not_pass"]
    md = render_md({**s, "model": "ondam"})
    assert "6개 중 Pass 2" in md and "일부만 봤다" in md and "계산이다" in md


def test_full_basis_counts_citation_findings():
    cases = [_case("A6", "fix", [_f("grounding")])]
    cases[0]["result"]["verdict_common"] = "pass"
    assert fix_first(cases, ITEMS, basis="common")["not_pass"] == 0
    full = fix_first(cases, ITEMS, basis="full")
    assert full["not_pass"] == 1 and full["order"][0]["owner"] == OWNER_ONDAM
