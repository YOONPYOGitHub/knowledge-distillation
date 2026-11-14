import pytest

from silvercare_autoqa.golden import load_golden


@pytest.fixture(scope="session")
def golden():
    return load_golden()


@pytest.fixture(scope="session")
def items(golden):
    return golden["by_id"]


def make_judgment(item, response, *, case_id="c1", met=None, violated=None, preserved=None, action=None,
                  urgency="match", concl=True, extra=None, det_review=None):
    """테스트용 채점 결과 — 근거 인용은 응답 원문 일부를 쓴다."""
    q = response[: min(len(response), 12)] if response else ""
    met = met or {}
    violated = violated or {}
    preserved = preserved or {}
    return {
        "case_id": case_id,
        "must_include": [{"id": m["id"], "met": met.get(m["id"], True), "evidence": q if met.get(m["id"], True) else ""}
                         for m in item["must_include"]],
        "must_not": [{"id": m["id"], "violated": violated.get(m["id"], False), "evidence": q if violated.get(m["id"], False) else ""}
                     for m in item["must_not"]],
        "key_facts": [{"id": k["id"], "preserved": preserved.get(k["id"], True), "evidence": q if preserved.get(k["id"], True) else ""}
                      for k in item["key_facts"]],
        "response_action": action or item["expected_action"],
        "urgency_vs_expected": urgency,
        "conclusion_match": concl,
        "referral_present": True,
        "core_in_first3": True,
        "actionability": 5,
        "concise_delivery": 4,
        "extra_violations": extra or [],
        "det_review": det_review or [],
        "summary": "테스트",
    }
