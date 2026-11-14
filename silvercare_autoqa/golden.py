"""골든셋 로더·검증기.

골든셋 한 문항 = 어르신 발화 + 기대 행동 + 필수답(must_include) + 금지답(must_not) + 핵심 정보(key_facts)
+ 모범 답안 + 출처. 필드 정의와 판정 연결은 docs/autoqa/GOLDEN_DATA_CARD.md 가 정본이다.
v0.4(2026-11-14, RAG 전환)부터 문항마다 rag(팀 RAG 자료에 근거가 있는지·검색되면 좋은 자료 묶음·핵심어)를 붙이고,
고객이 많이 묻는 영양제 질문(customer_q 12묶음)과 돌봄·복지 문항을 더했다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from . import gates

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_GOLDEN = DATA_DIR / "golden" / "silvercare_golden_v0.yaml"

# 질문 유형 (1학기 평가계획 질문셋 6유형 + 말동무 + RAG 전환 뒤 영양제·돌봄 복지)
TYPES = ("emergency", "medication", "chronic", "symptom", "prevention", "boundary", "companion", "supplement", "welfare")
TYPE_LABEL = {
    "emergency": "응급·악화 신호",
    "medication": "약 복용·상호작용",
    "chronic": "만성질환 관리",
    "symptom": "일반 증상 상담",
    "prevention": "생활·예방",
    "boundary": "진단·처방 요구·개인정보",
    "companion": "말동무·정서",
    "supplement": "건강기능식품·영양제",
    "welfare": "돌봄·복지 서비스",
}
# 고객이 영양제에 대해 많이 묻는 질문 12묶음 (2026-11-14 사용자 조사)
CUSTOMER_GROUPS = {
    1: "나이·몸에 맞는 추천", 2: "고민별 성분 찾기", 3: "먹는 약과 같이 먹어도 되나", 4: "여러 개를 같이 먹어도 되나",
    5: "언제·어떻게 먹나", 6: "효과가 진짜인가", 7: "성분표 읽기·제품 비교", 8: "검진 결과를 들고 묻기",
    9: "먹고 나서 이상해요", 10: "음식으로 대신하기", 11: "가족 대신 고르기", 12: "어디서·얼마에 사나",
}
# rag.coverage — 팀 RAG 지식 자료에 이 문항 답의 근거가 있는가 (작성자가 자료를 찾아보고 정함, corpus-coverage 로 다시 잼)
COVERAGES = ("covered", "partial", "none")
COVERAGE_LABEL = {"covered": "자료 있음", "partial": "일부만", "none": "자료 없음"}
RISKS = ("emergency", "urgent", "routine")
RISK_LABEL = {"emergency": "지금 바로 119·응급실", "urgent": "오늘~며칠 안 진료", "routine": "일상 관리"}

# 기대 행동과 긴급도 (숫자가 클수록 급함). 응답 행동이 기대보다 낮으면 과소대응, 높으면 과잉대응.
ACTIONS = {
    "call_119": 4,
    "crisis_line": 4,
    "er_now": 4,
    "first_aid_then_119_if": 3,
    "see_doctor_soon": 2,
    "ask_pharmacist_doctor": 1,
    "decline_redirect": 1,
    "self_care": 0,
    "empathize": 0,
    "none": -1,
}
ACTION_LABEL = {
    "call_119": "지금 바로 119",
    "crisis_line": "위기 상담(109) 연결, 위급하면 119",
    "er_now": "지금 바로 응급실(또는 119)",
    "first_aid_then_119_if": "응급조치 후 나아지지 않으면 119",
    "see_doctor_soon": "오늘~며칠 안 진료",
    "ask_pharmacist_doctor": "약사·의사에게 확인",
    "decline_redirect": "진단·처방은 거절하고 전문가 안내",
    "self_care": "생활 관리 + 악화 시 진료 조건",
    "empathize": "공감 대화",
    "none": "행동 안내 없음",
}

# gate 종류는 gates.py 한 곳이 정본이다 (코드 감사 D07). 여기 이름은 예전 코드와의 호환용.
MUST_GATES = gates.MUST_KINDS
NOT_GATES = tuple(gates.NOT_KIND_TO_GATE)
# 1학기 평가계획 안전 Gate 와 연결되는 금지 종류 (content 는 Gate 아님)
SAFETY_NOT_GATES = gates.safety_not_kinds()

AIHUB_CATEGORIES = (
    "호흡기질환", "신장비뇨기질환", "순환기질환", "뇌신경정신질환", "유방내분비질환", "소아청소년질환",
    "근골격질환", "치과질환", "응급질환", "유전질환", "귀코목질환", "기타", "소화기질환", "여성질환",
    "눈질환", "감염성질환", "피부질환", "성형미용 및 재건", "종양혈액질환",
)
AIHUB_INTENTIONS = ("정의", "증상", "진단", "치료", "예방", "약물", "운동", "재활", "식이/생활")

REQUIRED = ("id", "type", "function", "risk", "scenario", "utterance", "expected_action",
            "must_include", "must_not", "key_facts", "reference_answer", "sources")


def file_sha256(path: Path) -> str:
    """줄바꿈(CRLF/LF)을 맞춘 뒤의 해시 — Windows·Linux 사이에 회차 폴더를 옮겨도 같은 값 (감사 A05)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_golden(path: Path | str = DEFAULT_GOLDEN) -> dict[str, Any]:
    """골든셋 YAML 을 읽어 {meta, sources, items, by_id} 로 돌려준다. 검증은 validate_golden 이 한다."""
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = doc.get("items") or []
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "schema_version": doc.get("schema_version"),
        "golden_version": doc.get("golden_version"),
        "review_status": doc.get("review_status") or "draft (전문가 검수 전)",
        "sources": doc.get("sources") or {},
        "items": items,
        "by_id": {it.get("id"): it for it in items},
    }


def _check_entries(item_id: str, key: str, entries: Any, prefix: str, gates: tuple, errors: list[str],
                   need_tier: bool = True) -> None:
    if not isinstance(entries, list):
        errors.append(f"{item_id}: {key} 는 목록이어야 함")
        return
    seen = set()
    for e in entries:
        if not isinstance(e, dict):
            errors.append(f"{item_id}: {key} 항목이 dict 가 아님")
            continue
        eid = e.get("id")
        if not eid or not str(eid).startswith(prefix):
            errors.append(f"{item_id}: {key} id '{eid}' 는 {prefix} 로 시작해야 함")
        if eid in seen:
            errors.append(f"{item_id}: {key} id '{eid}' 중복")
        seen.add(eid)
        if not str(e.get("text") or "").strip():
            errors.append(f"{item_id}: {key} {eid} text 비어 있음")
        if need_tier:
            if e.get("tier") not in (1, 2):
                errors.append(f"{item_id}: {key} {eid} tier 는 1 또는 2")
            if e.get("gate") not in gates:
                errors.append(f"{item_id}: {key} {eid} gate '{e.get('gate')}' 는 {gates} 중 하나")


def validate_golden(golden: dict[str, Any]) -> list[str]:
    """스키마·정합성 오류 목록. 비어 있으면 통과."""
    errors: list[str] = []
    ids = set()
    src_ids = set(golden["sources"].keys())
    for it in golden["items"]:
        iid = it.get("id", "<id 없음>")
        for k in REQUIRED:
            if k not in it:
                errors.append(f"{iid}: 필드 '{k}' 없음")
        if iid in ids:
            errors.append(f"{iid}: id 중복")
        ids.add(iid)
        if it.get("type") not in TYPES:
            errors.append(f"{iid}: type '{it.get('type')}' 알 수 없음")
        if it.get("risk") not in RISKS:
            errors.append(f"{iid}: risk '{it.get('risk')}' 알 수 없음")
        if it.get("expected_action") not in ACTIONS or it.get("expected_action") == "none":
            errors.append(f"{iid}: expected_action '{it.get('expected_action')}' 알 수 없음")
        if not str(it.get("utterance") or "").strip():
            errors.append(f"{iid}: utterance 비어 있음")
        if not str(it.get("reference_answer") or "").strip():
            errors.append(f"{iid}: reference_answer 비어 있음")
        _check_entries(iid, "must_include", it.get("must_include"), "I", MUST_GATES, errors)
        _check_entries(iid, "must_not", it.get("must_not"), "X", NOT_GATES, errors)
        _check_entries(iid, "key_facts", it.get("key_facts"), "K", (), errors, need_tier=False)
        if not it.get("must_include"):
            errors.append(f"{iid}: must_include 가 하나는 있어야 함")
        for s in it.get("sources") or []:
            if s not in src_ids:
                errors.append(f"{iid}: 출처 '{s}' 가 sources 목록에 없음")
        if it.get("type") != "companion" and not it.get("sources"):
            errors.append(f"{iid}: 건강 문항은 출처가 하나는 있어야 함")
        # 응급 문항은 tier 1 응급 연결(escalation) 필수답이 반드시 있어야 응급 놓침을 잴 수 있다
        if it.get("risk") == "emergency":
            if not any(m.get("tier") == 1 and m.get("gate") == "escalation" for m in it.get("must_include") or []):
                errors.append(f"{iid}: 응급 문항에 tier1 escalation 필수답이 없음")
            if ACTIONS.get(it.get("expected_action"), 0) < 3:
                errors.append(f"{iid}: 응급 문항의 기대 행동 긴급도가 낮음")
        _check_rag_fields(iid, it, errors)
        ai = it.get("aihub")
        if ai is not None:
            if ai.get("disease_category") not in AIHUB_CATEGORIES:
                errors.append(f"{iid}: aihub.disease_category '{ai.get('disease_category')}' 가 AI Hub 19분류에 없음")
            if ai.get("intention") not in AIHUB_INTENTIONS:
                errors.append(f"{iid}: aihub.intention '{ai.get('intention')}' 가 AI Hub 의도 목록에 없음")
    return errors


def _check_rag_fields(iid: str, it: dict, errors: list[str]) -> None:
    """v0.4 칸 — rag·customer_q·review_note. rag 는 예전 골든셋 호환을 위해 없어도 되지만 있으면 형식을 본다."""
    from .rag import unknown_expect     # rag.py 가 golden 의 DATA_DIR 을 쓰므로 함수 안에서 불러 순환을 피한다

    rg = it.get("rag")
    if rg is not None:
        if not isinstance(rg, dict):
            errors.append(f"{iid}: rag 는 사전이어야 함")
        else:
            if rg.get("coverage") not in COVERAGES:
                errors.append(f"{iid}: rag.coverage '{rg.get('coverage')}' 는 {COVERAGES} 중 하나")
            bad = unknown_expect(rg.get("expect"))
            if bad:
                errors.append(f"{iid}: rag.expect 의 {bad} 가 자료 묶음표(rag_sources.yaml)에 없음")
            if not isinstance(rg.get("expect", []), list) or not isinstance(rg.get("terms", []), list):
                errors.append(f"{iid}: rag.expect·rag.terms 는 목록")
    cq = it.get("customer_q")
    if it.get("type") == "supplement":
        if not isinstance(cq, dict) or cq.get("group") not in CUSTOMER_GROUPS:
            errors.append(f"{iid}: 영양제 문항은 customer_q.group(1~12)이 있어야 함")
        elif cq.get("label") != CUSTOMER_GROUPS[cq["group"]]:
            errors.append(f"{iid}: customer_q.label 은 '{CUSTOMER_GROUPS[cq['group']]}' 이어야 함")
    if "review_note" in it and not isinstance(it.get("review_note"), str):
        errors.append(f"{iid}: review_note 는 문자열")


def tier1_escalations(item: dict) -> list[dict]:
    return [m for m in item.get("must_include") or [] if m.get("tier") == 1 and m.get("gate") == "escalation"]


def summary(golden: dict[str, Any]) -> dict[str, Any]:
    """골든셋 구성 요약 (리포트·데이터 카드용)."""
    from collections import Counter

    items = golden["items"]
    return {
        "n_items": len(items),
        "by_type": dict(Counter(it["type"] for it in items)),
        "by_risk": dict(Counter(it["risk"] for it in items)),
        "by_variant": dict(Counter(it.get("variant", "표준어") for it in items)),
        "by_aihub_category": dict(Counter((it.get("aihub") or {}).get("disease_category", "(해당 없음)") for it in items)),
        "by_customer_group": dict(Counter((it.get("customer_q") or {}).get("group") for it in items
                                          if (it.get("customer_q") or {}).get("group"))),
        "by_rag_coverage": dict(Counter((it.get("rag") or {}).get("coverage", "(표시 없음)") for it in items)),
        "n_review_notes": sum(1 for it in items if str(it.get("review_note") or "").strip()),
        "study_set": [it["id"] for it in items if it.get("study_set")],
        "n_must_include": sum(len(it["must_include"]) for it in items),
        "n_must_not": sum(len(it["must_not"]) for it in items),
        "n_key_facts": sum(len(it["key_facts"]) for it in items),
        "review_status": golden.get("review_status") or "draft (전문가 검수 전)",
    }
