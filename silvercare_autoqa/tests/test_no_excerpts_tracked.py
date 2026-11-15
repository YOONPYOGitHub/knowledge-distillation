"""레포에 올린 회차 파일에 온담 근거 자료 발췌가 들어 있으면 안 된다 (노하우 K59 — 법으로 지켜야 하는 것).

온담 답의 출처 발췌에는 AI Hub 원문이 들어 있다. 다시 나눠 줄 수 없는 자료다.
회차 폴더(results/autoqa/, runs/)는 무시 목록에 있고, 골라서 올린 리포트·지표에도 발췌가 섞이지 않았는지 이 시험이 본다.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXCERPT = re.compile(r'"excerpt(?:_api)?"\s*:\s*"[^"]')          # 빈 값이 아닌 발췌


def _tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git 을 쓸 수 없음")
    if out.returncode != 0:
        pytest.skip("git 저장소가 아님")
    return [p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]


def test_no_source_excerpts_in_tracked_run_files():
    hits = []
    for rel in _tracked():
        if not (rel.endswith(".jsonl") or "/packets/" in rel or rel.endswith("scored.json")):
            continue
        p = ROOT / rel
        if not p.exists() or p.stat().st_size > 30_000_000:
            continue
        if EXCERPT.search(p.read_text(encoding="utf-8", errors="replace")):
            hits.append(rel)
    assert not hits, f"근거 자료 발췌가 든 파일이 레포에 올라가 있음: {hits[:5]}"


def test_run_folders_are_ignored():
    for rel in ("results/autoqa/x/run/cases.jsonl", "runs/m/scored.jsonl"):
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
        if r.returncode not in (0, 1):
            pytest.skip("git check-ignore 를 쓸 수 없음")
        assert r.returncode == 0, f"{rel} 이 무시 목록에 없음 — 회차 폴더에는 근거 발췌가 들어간다"
