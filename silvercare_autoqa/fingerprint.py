"""평가기준 지문 — 골든셋·용어집·룰·프롬프트·판정식·채점 모델이 같은 회차끼리만 비교한다.

기준이 바뀐 회차를 섞어 비교하면 모델 차이와 기준 차이가 구분되지 않는다 (쇼핑 AI 검수의 '평가기준 지문').
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .golden import DATA_DIR

PKG = Path(__file__).resolve().parent
PARTS = [
    PKG / "rules.py", PKG / "verdict.py", PKG / "prompts.py", PKG / "metrics.py", PKG / "textutil.py",
    DATA_DIR / "lexicons" / "guardrails.yaml", DATA_DIR / "lexicons" / "plain_language.yaml",
]


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def compute(golden_path: str | Path, judge_model: str, judge_effort: str) -> dict:
    parts = {p.name: _h(p.read_bytes().replace(b"\r\n", b"\n")) for p in PARTS}
    parts["golden"] = _h(Path(golden_path).read_bytes().replace(b"\r\n", b"\n"))
    parts["judge"] = _h(f"{judge_model}|{judge_effort}".encode())
    joined = "|".join(f"{k}={v}" for k, v in sorted(parts.items()))
    return {"fingerprint": _h(joined.encode())[:16], "parts": parts}
