"""평가기준 지문 — 같은 기준으로 잰 회차끼리만 숫자를 비교한다 (쇼핑 AI 검수의 '평가기준 지문').

지문은 둘이다 (감사 R07·A14, 노하우 점검 M1).
- 채점 지문(judging): 채점기가 본 것 — 골든셋·용어집·결정 룰·프롬프트·채점 모델. 바뀌면 이미 받은 채점 결과와
  기준이 달라지므로 judge·import·verify 는 멈춘다 (--rescore 로만 진행, 기록이 남음).
- 판정 지문(scoring): 채점 결과를 판정·지표로 바꾸는 코드 — 판정식·지표·결과 검사·Gate 표·반박 연결.
  finalize 는 늘 지금 코드로 다시 계산하므로, 바뀌었으면 '판정식 재적용'으로 두 지문을 리포트에 함께 적는다.
보기만 바꾸는 report.py 는 어느 쪽에도 넣지 않는다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .golden import DATA_DIR

PKG = Path(__file__).resolve().parent
LEX = [DATA_DIR / "lexicons" / "guardrails.yaml", DATA_DIR / "lexicons" / "plain_language.yaml"]
JUDGING = [PKG / "rules.py", PKG / "prompts.py", PKG / "textutil.py", PKG / "gates.py", PKG / "golden.py", *LEX]
SCORING = [PKG / "verdict.py", PKG / "metrics.py", PKG / "judge.py", PKG / "gates.py", PKG / "golden.py",
           PKG / "pipeline.py", PKG / "textutil.py", PKG / "rules.py", *LEX]
PARTS = sorted(set(JUDGING) | set(SCORING))


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file(p: Path) -> str:
    return _h(p.read_bytes().replace(b"\r\n", b"\n"))


def compute(golden_path: str | Path, judge_model: str, judge_effort: str) -> dict:
    parts = {p.name: _file(p) for p in PARTS}
    parts["golden"] = _file(Path(golden_path))
    parts["judge"] = _h(f"{judge_model}|{judge_effort}".encode())

    def fp(names: list[str]) -> str:
        return _h("|".join(f"{k}={parts[k]}" for k in sorted(names)).encode())[:16]

    judging = fp([p.name for p in JUDGING] + ["golden", "judge"])
    scoring = fp([p.name for p in SCORING])
    return {"fingerprint": _h(f"{judging}|{scoring}".encode())[:16], "judging": judging, "scoring": scoring,
            "parts": parts}


def changed_parts(old: dict, new: dict) -> list[str]:
    """두 지문에서 달라진 부분 이름 (리포트·오류 문구용)."""
    a, b = old or {}, new or {}
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
