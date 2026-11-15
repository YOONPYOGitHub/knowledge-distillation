"""결함을 고칠 사람에게 — 판정이 Hold·Fix 인 답마다 「누가 고치나」를 추정한다 (노하우 점검 K53).

온담을 고칠 수 있는 사람은 셋이다.
- 앱 담당    : 응급 감지, 응급 고정 안내 글
- 온담 담당  : 지시문(생성 프롬프트), 답 쓰는 규칙, 질문을 어느 길(mode)로 보낼지
- 데이터 담당: RAG 지식 자료

채점 결과(scored.jsonl 한 줄)와 골든셋 문항만 읽는다. 모델을 부르지 않는다. 판정·지표를 바꾸지 않는다.
이 파일은 채점 지문·판정 지문에 들어가지 않는다 — 고쳐도 이미 잰 회차의 숫자가 그대로다.

추정이다. 담당이 틀릴 수 있으니 리포트와 defects.csv 에 「추정」이라고 적는다.

규칙은 위에서부터 보고 처음 맞는 것을 쓴다 (route).
  1. 앱 담당     — 응급 고정 안내로 나간 답 (mode=emergency)
  2. 앱 담당     — 응급 문항이 일반 답으로 나가 안전 Gate 를 어김
  3. 온담 담당   — 거절 (mode=refusal)
  4. 온담 담당   — 건강 질문이 일상 대화 길로 감 (mode=conversation)
  5. 온담 담당   — 응급이 아닌 문항의 안전 Gate 위반, 비었거나 깨진 답
  6. 데이터 담당 — 골든셋 자료 범위가 「없음」
  7. 찾은 근거에 답할 내용이 없었다 (답에 빠진 핵심 정보의 절반 넘게가 근거에도 없음, 또는 지어낸 답)
     - 골든셋 자료 범위가 「있음」이면 온담 담당 — 자료는 있는데 검색이 찾아오지 못했다
     - 그 밖(「일부」·표시 없음)이면 데이터 담당 — 자료가 모자란다
  8. 온담 담당   — 그 밖 (근거에 있는데 안 씀·틀린 인용·안전 문장 빠짐·결론 다름·틀린 사실)
출처를 돌려주지 않는 모델(상용 LLM 등)과 Pass·채점 오류에는 담당을 붙이지 않는다.

전체 문맥을 받지 못한 회차는 7번이 실제보다 많이 걸린다 (채점기가 인용한 출처의 발췌만 보아서). 리포트가 그 답 수를 함께 적는다.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, NamedTuple

from .gates import GATES

OWNER_APP, OWNER_ONDAM, OWNER_DATA = "앱 담당", "온담 담당", "데이터 담당"
OWNERS = (OWNER_APP, OWNER_ONDAM, OWNER_DATA)          # 리포트 표에 적는 순서

# 한 줄 이유 — 리포트는 같은 이유끼리 묶어 센다
APP_FIXED_TEXT = "응급 고정 안내 글이 조건을 달거나 응급처치가 없음"
APP_FIXED_OTHER = "응급 고정 안내 글에 고칠 내용이 있음"
APP_FALSE_ALARM = "응급이 아닌 질문에 응급 고정 안내가 나감"
APP_MISSED = "응급 감지가 안 걸려 일반 답으로 감"
ONDAM_EMERGENCY_LINE = "응급 답에 빠진 안내"
ONDAM_SAFETY_LINE = "꼭 넣을 안전 문장이 빠짐"
POLICY_GATES = ("referral", "escalation", "med_safety")    # 골든셋 필수답 종류 가운데 '안내 문장' — 지식이 아니라 답 쓰는 규칙
ONDAM_REFUSAL = "평범한 질문을 거절"
ONDAM_CONVERSATION = "건강 질문이 일상 대화 길로 가 자료를 찾지 않음"
ONDAM_SEARCH = "자료는 있는데 찾아오지 못함"
ONDAM_RULES = "지시문·답 쓰는 규칙"
DATA_MISSING = "지식 자료에 근거가 없음"

# 근거 대조 지적 종류(채점기 grounding.issues 의 type) 가운데 인용·답 쓰기 문제인 것 → 이유에 붙일 말
_CITE_ISSUES = (("example_as_evidence", "참고 사례를 사실 근거로 인용"), ("contradicted", "답이 근거 자료와 어긋남"),
                ("wrong_citation", "인용한 자료가 다른 내용"))
MAX_DETAILS = 3          # 이유 한 줄에 붙이는 사유 수 상한


class Route(NamedTuple):
    owner: str               # 담당
    reason: str              # 한 줄 이유 (위 상수 가운데 하나)
    detail: str = ""         # 어느 조건에 걸렸나 — 담당이 추정을 바로 확인할 수 있게

    @property
    def line(self) -> str:
        return f"{self.reason} ({self.detail})" if self.detail else self.reason


def _mode(case: dict) -> str:
    """답이 나간 길 — grounded(검색 후 답) · emergency(응급 고정 안내) · conversation(일상 대화) · refusal(거절)."""
    return str((case.get("rag") or {}).get("mode") or ((case.get("det") or {}).get("rag") or {}).get("mode") or "")


def _blocking(case: dict) -> list[dict]:
    """Pass 를 막은 지적 — 반박되지 않은 지적 가운데 참고(minor)를 뺀 것."""
    return [f for f in (case.get("result") or {}).get("live_findings") or [] if f.get("severity") != "minor"]


def _grounding(case: dict) -> dict:
    return (case.get("judgment") or {}).get("grounding") or {}


def _lost_key_facts(case: dict) -> tuple[list[str], list[str], list[str]]:
    """채점기가 '답에 없다'고 본 핵심 정보 → (전부, 찾은 근거에도 없던 것, 찾은 근거에는 있던 것).
    응급 고정 안내는 근거 대조를 하지 않아 뒤의 둘이 빈다."""
    in_ctx = {r.get("id"): r.get("in_context") for r in _grounding(case).get("key_facts_in_context") or []}
    lost = [r.get("id") for r in (case.get("judgment") or {}).get("key_facts") or [] if r.get("preserved") is False]
    return lost, [i for i in lost if in_ctx.get(i) is False], [i for i in lost if in_ctx.get(i) is True]


def _evidence_gap(case: dict, lost: list[str], absent: list[str]) -> str:
    """온담이 찾은 근거에 답할 내용이 없었다고 볼 사유 (없으면 '')."""
    if lost and len(absent) * 2 > len(lost):            # 답에 빠진 핵심 정보의 절반 넘게가 찾은 근거에도 없다
        return f"답에 빠진 핵심 정보 {len(lost)}개 중 {len(absent)}개가 찾은 근거에도 없음"
    g = _grounding(case)
    if g.get("coverage_handling") == "fabricated":
        return "찾은 근거에 답이 없는데 지어내어 단정"
    # '근거 없는 단정'은 전체 문맥을 받은 회차에서만 쓴다 — 발췌가 잘린 회차는 판정식도 참고(minor)로만 센다
    complete = bool((case.get("rag") or {}).get("contexts_complete"))
    if complete and any(i.get("type") == "unsupported" for i in g.get("issues") or []):
        return "찾은 근거에 없는 수치·사실을 단정"
    return ""


def _rule_details(case: dict, lost_in_context: bool) -> list[str]:
    """지시문·답 쓰는 규칙으로 고칠 사유 — 고칠 순서(안전 문장 → 사실 → 인용 → 결론 → 그 밖)대로."""
    cats = {f.get("cat") for f in _blocking(case)}
    hits = {h.get("rule") for h in (case.get("det") or {}).get("hits") or []}
    types = {i.get("type") for i in _grounding(case).get("issues") or []}
    if "RAG-CITE-EXAMPLE" in hits:
        types.add("example_as_evidence")                # 결정 룰이 확정한 사례 인용은 채점기 지적과 같은 말로 적는다
    out = []
    if "safety_must" in cats:
        out.append("안전 문장이 빠짐")
    if "fact" in cats:
        out.append("틀린 사실·연락처")
    if "RAG-CITE-RANGE" in hits:
        out.append("없는 출처 번호를 인용")
    out += [text for t, text in _CITE_ISSUES if t in types]
    if lost_in_context:
        out.append("근거에 있는 핵심 정보를 답에 안 씀")
    if "conclusion" in cats:
        out.append("결론이 모범 답안과 다름")
    if "content" in cats:
        out.append("금지답 위반 등 그 밖의 내용")
    if "format" in cats:
        out.append("답 형식")
    return out


def route(case: dict, item: dict) -> Route | None:
    """판정이 Hold·Fix 인 답 하나의 담당을 추정한다. 담당을 붙이지 않는 답은 None.
    case: scored.jsonl 한 줄 (pipeline.finalize 가 만든 것). item: 그 문항의 골든셋 항목."""
    res = case.get("result") or {}
    if res.get("verdict") not in ("hold", "fix"):
        return None             # Pass 는 고칠 것이 없다. 채점 오류는 다시 채점할 일이지 온담을 고칠 일이 아니다
    if case.get("rag") is None:
        return None             # 출처를 돌려주지 않는 모델(상용 LLM 등) — 우리 팀이 고치는 시스템이 아니다
    mode = _mode(case)
    gates = list(res.get("gates") or [])
    emergency_item = item.get("risk") == "emergency"

    # 1. 응급 고정 안내는 앱이 정해 둔 글이다 (검색도 모델 호출도 없다). 그 글의 결함은 지시문·자료로는 못 고친다
    if mode == "emergency":
        if not emergency_item:
            return Route(OWNER_APP, APP_FALSE_ALARM)
        if {"G1", "G4"} & set(gates):
            return Route(OWNER_APP, APP_FIXED_TEXT)
        return Route(OWNER_APP, APP_FIXED_OTHER)
    # 2. 응급 문항이 일반 답으로 갔고 그 답이 안전 Gate 를 어겼다
    if emergency_item and gates:
        # 답이 이미 119·응급실로 보냈으면 감지 문제가 아니다 — 답에 꼭 넣을 응급 안내 한 줄이 빠진 것이라 지시문으로 고친다
        # (예: "지금 바로 응급실" 은 있는데 "말이 어눌하거나 힘이 빠지면 바로 119" 가 없음)
        if (case.get("det") or {}).get("escalation_present"):
            return Route(OWNER_ONDAM, ONDAM_EMERGENCY_LINE, "119·응급실 안내는 있고 꼭 넣을 응급 안내가 빠짐")
        return Route(OWNER_APP, APP_MISSED)             # 응급 연결 말이 아예 없다 — 응급 감지가 걸렸다면 고정 안내가 나갔을 답이다
    # 3·4. 답이 나간 길이 틀렸다 — 길을 바로잡기 전에는 자료를 보태도 그 답에 닿지 않는다
    if mode == "refusal":
        return Route(OWNER_ONDAM, ONDAM_REFUSAL)
    conversation = mode == "conversation"
    if conversation and item.get("type") != "companion":
        return Route(OWNER_ONDAM, ONDAM_CONVERSATION)
    # 5. 자료가 있든 없든 나오면 안 되는 답 — 답 쓰는 규칙·생성 설정으로 막는다
    if gates:
        return Route(OWNER_ONDAM, ONDAM_RULES, "안전 Gate 위반: " + ", ".join(f"{g} {GATES.get(g, '')}".strip() for g in gates))
    if (case.get("det") or {}).get("fatal_format"):
        return Route(OWNER_ONDAM, ONDAM_RULES, "답이 비었거나 형식이 깨짐")
    # 5-2. 빠진 것이 "약은 스스로 끊지 마세요"·"약사에게 확인하세요"·"~면 바로 119" 같은 안내 문장뿐이면
    #      지식 자료를 보탤 일이 아니라 답 쓰는 규칙에 넣을 일이다 (자료 범위가 '없음'이어도 같다)
    musts = {m.get("id"): m.get("gate") for m in item.get("must_include") or []}
    core = [f for f in _blocking(case) if f.get("cat") != "grounding"]
    if core and all(f.get("cat") == "safety_must" and musts.get(f.get("ref")) in POLICY_GATES for f in core):
        return Route(OWNER_ONDAM, ONDAM_SAFETY_LINE)
    # 6·7. 지식 자료와 검색 — 일상 대화 길(말동무 문항)은 자료를 찾지 않으므로 보지 않는다
    lost, absent, in_context = _lost_key_facts(case)
    if not conversation:
        coverage = (item.get("rag") or {}).get("coverage")
        if coverage == "none":
            return Route(OWNER_DATA, DATA_MISSING, "골든셋 자료 범위 「없음」")
        gap = _evidence_gap(case, lost, absent)
        if gap:
            # 골든셋이 '자료 있음'이라고 적은 문항이면 자료를 보탤 일이 아니다 — 있는 자료를 검색이 못 가져온 것이다
            return Route(OWNER_ONDAM, ONDAM_SEARCH, gap) if coverage == "covered" else Route(OWNER_DATA, DATA_MISSING, gap)
    # 8. 그 밖은 지시문·답 쓰는 규칙
    return Route(OWNER_ONDAM, ONDAM_RULES, " · ".join(_rule_details(case, bool(in_context))[:MAX_DETAILS]))


def owner_of(case: dict, item: dict) -> tuple[str, str] | None:
    """(담당, 한 줄 이유). 담당을 붙이지 않는 답(Pass·채점 오류·출처를 돌려주지 않는 모델)은 None. 추정이다."""
    r = route(case, item)
    return (r.owner, r.line) if r else None


def owner_counts(cases: list[dict], items: dict[str, dict]) -> dict[str, dict[str, dict[str, Any]]]:
    """출처를 돌려주는 모델마다 담당별 Hold·Fix 문항을 모은다 (리포트의 담당별 표).
    {모델: {담당: {"hold": n, "fix": n, "items": [문항 id — Hold 먼저], "reasons": {한 줄 이유: 건수 — 많은 순}}}}
    고칠 문항이 없는 모델은 빈 사전이다. 출처를 돌려주지 않는 모델은 넣지 않는다."""
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for c in sorted(cases, key=lambda c: (c["model"], c["result"]["verdict"] != "hold", c["item_id"])):
        if c.get("rag") is None:
            continue
        per = found.setdefault(c["model"], {})
        r = route(c, items[c["item_id"]])
        if r is None:
            continue
        v = per.setdefault(r.owner, {"hold": 0, "fix": 0, "items": [], "reasons": Counter()})
        v[c["result"]["verdict"]] += 1
        if c["item_id"] not in v["items"]:
            v["items"].append(c["item_id"])
        v["reasons"][r.reason] += 1
    return {m: {o: {**per[o], "reasons": dict(per[o]["reasons"].most_common())} for o in OWNERS if o in per}
            for m, per in found.items()}


def partial_evidence(cases: list[dict]) -> int:
    """채점기가 근거 자료의 일부만 본 답 수 — 전체 문맥을 받지 못해 인용한 출처의 발췌만 본 답이다.
    이런 답은 '찾은 근거에 없음'이 실제보다 많이 잡힌다 (rag.py 머리말)."""
    return sum(1 for c in cases if (c.get("rag") or {}).get("sources") and not (c.get("rag") or {}).get("contexts_complete"))
