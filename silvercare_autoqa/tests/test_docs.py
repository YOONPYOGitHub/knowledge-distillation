"""팀 공유 문서 기준 검사 — docs/autoqa/WRITING_GUIDE.md 를 기계가 확인한다.

사람이 읽는 문서가 다시 AI 말투·긴 문장·첫 화면 요약 없음으로 돌아가지 않게, 고칠 때마다 이 시험이 돈다
(`preflight --selftest` 도 이 시험을 돌린다). 검사 목록을 바꾸면 WRITING_GUIDE.md §6 도 같은 커밋에서 바꾼다.
"""

import html
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs" / "autoqa"
STUDY_DIR = ROOT / "docs" / "user_study"
MD_DOCS = sorted(p for p in DOCS_DIR.glob("*.md") if p.name not in ("CLAUDE.md", "AGENTS.md")) + \
    sorted(STUDY_DIR.glob("*.md")) + [ROOT / "silvercare_autoqa" / "README.md"]
PRINTS = sorted((STUDY_DIR / "print").glob("*.html"))      # 어르신·전문가가 읽는 인쇄물
README = ROOT / "README.md"
INDEX = DOCS_DIR / "index.html"

MAX_SENTENCE = 90        # 본문 문장 (글자 수). 기준 문서는 80자에서 나누라고 쓴다 — 시험은 여유 10자
MAX_CELL = 90            # 표 한 칸
MAX_NODE = 18            # Mermaid 칸 한 줄

# WRITING_GUIDE.md §3 '쓰지 않는 말'. 고유명사·인용이라 꼭 필요하면 팀에 묻고 이 목록과 기준 문서를 함께 고친다.
BANNED = [
    "포착", "가시화", "환류", "정합", "레버리지", "접점", "임팩트", "시너지", "트리거", "인에이블", "액션 아이템",
    "확보한다", "확보해", "확보하", "기여한다", "기여해", "기여하", "수행한다", "수행해", "수행하",
    "의 경우", "에 있어서", "앞단", "길목", "경보선", "칸 채우기", "접촉 부담", "숫자가 말", "라고 할 수 있다",
    "구조적", "직관적", "효과적", "체계적", "실질적", "본격적", "전반적", "결정적", "핵심적", "적극적", "지속적",
    "선제적", "근본적", "실효적", "획기적", "일관적", "명시적", "암묵적", "잠재적", "전략적", "통계적", "기술적",
    "안정적", "단계적", "구체적",
]


def _readme_part() -> str:
    s = README.read_text(encoding="utf-8")
    a, b = s.find("<!-- autoqa:start -->"), s.find("<!-- autoqa:end -->")
    assert a != -1 and b > a, "README 의 autoqa 부분 표식(<!-- autoqa:start/end -->)이 없음"
    return s[a:b]


def _sources() -> list[tuple[str, str]]:
    return [("README.md(autoqa)", _readme_part())] + [(str(p.relative_to(ROOT)), p.read_text(encoding="utf-8")) for p in MD_DOCS]


def _strip_code(md: str) -> tuple[str, list[str]]:
    """코드 블록·인라인 코드·주석을 빼고 본문만. Mermaid 블록은 따로 돌려준다."""
    mermaid = re.findall(r"```mermaid\n(.*?)```", md, flags=re.S)
    body = re.sub(r"```.*?```", "", md, flags=re.S)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"`[^`\n]*`", "", body)
    return body, mermaid


def _visible(text: str) -> str:
    t = re.sub(r"\]\([^)]*\)", "]", text)            # 링크 주소
    t = re.sub(r"https?://\S+", "", t)
    return t.replace("**", "")


def _html_text(src: str) -> str:
    src = re.sub(r"<(script|style)\b.*?</\1>", "", src, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", "\n", src))


def _sentences(line: str) -> list[str]:
    line = re.sub(r"^\s*([-*>]|\d+\.)\s+", "", line)
    return [s.strip() for s in re.split(r"(?<=[.!?。])\s+", line) if s.strip()]


def test_no_banned_phrases():
    bad = []
    texts = _sources() + [("docs/autoqa/index.html", _html_text(INDEX.read_text(encoding="utf-8")))] + \
        [(str(p.relative_to(ROOT)), _html_text(p.read_text(encoding="utf-8"))) for p in PRINTS]
    for name, src in texts:
        if name.endswith("WRITING_GUIDE.md"):
            continue                                   # 기준 문서는 쓰지 않는 말을 목록으로 보여 준다
        body, _ = _strip_code(src)
        for i, line in enumerate(body.splitlines(), 1):
            for w in BANNED:
                if w in line:
                    bad.append(f"{name}: '{w}' — {line.strip()[:70]}")
    assert not bad, "쓰지 않는 말 (WRITING_GUIDE.md §3):\n" + "\n".join(bad[:40])


def test_every_doc_starts_with_summary():
    missing = []
    for name, src in _sources():
        head = "\n".join(src.splitlines()[:30])
        if "## 한눈에" not in head:
            missing.append(name)
    idx = _html_text(INDEX.read_text(encoding="utf-8"))
    if "한눈에" not in "\n".join(l for l in idx.splitlines() if l.strip())[:600]:
        missing.append("docs/autoqa/index.html")
    assert not missing, "첫 30줄 안에 '## 한눈에' 가 없는 문서: " + ", ".join(missing)


def test_sentence_and_cell_length():
    long = []
    for name, src in _sources():
        body, _ = _strip_code(src)
        for line in body.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("|---"):
                continue
            if s.startswith("|"):
                for cell in [c.strip() for c in s.strip("|").split("|")]:
                    if len(_visible(cell)) > MAX_CELL:
                        long.append(f"{name} [표 {len(_visible(cell))}자] {cell[:60]}…")
                continue
            for sent in _sentences(_visible(s)):
                if len(sent) > MAX_SENTENCE:
                    long.append(f"{name} [문장 {len(sent)}자] {sent[:60]}…")
    assert not long, f"너무 긴 문장·표 칸 (문장 {MAX_SENTENCE}자·칸 {MAX_CELL}자 이하):\n" + "\n".join(long[:40])


def test_mermaid_nodes_short():
    long = []
    for name, src in _sources():
        _, blocks = _strip_code(src)
        for b in blocks:
            for label in re.findall(r'["|]([^"|\n]{2,})["|]', b):
                for part in re.split(r"<br\s*/?>", label):
                    if len(part.strip()) > MAX_NODE:
                        long.append(f"{name}: '{part.strip()}' ({len(part.strip())}자)")
    assert not long, f"Mermaid 칸 한 줄이 {MAX_NODE}자를 넘음 — <br/> 로 나누거나 줄인다:\n" + "\n".join(long[:30])


def test_relative_links_resolve():
    broken = []
    files = [(README, _readme_part())] + [(p, p.read_text(encoding="utf-8")) for p in MD_DOCS]
    for p, src in files:
        body, _ = _strip_code(src)
        for target in re.findall(r"\]\(([^)\s]+)\)", body):
            if re.match(r"(https?:|mailto:|#)", target):
                continue
            path = target.split("#", 1)[0]
            if path and not (p.parent / path).exists():
                broken.append(f"{p.relative_to(ROOT)} → {target}")
    assert not broken, "없는 파일을 가리키는 링크:\n" + "\n".join(broken)


@pytest.mark.parametrize("folder", ["docs/autoqa", "silvercare_autoqa"])
def test_agent_instruction_pairs_identical(folder):
    c, a = ROOT / folder / "CLAUDE.md", ROOT / folder / "AGENTS.md"
    assert c.exists() and a.exists(), f"{folder} 에 CLAUDE.md·AGENTS.md 짝이 있어야 함"
    assert c.read_text(encoding="utf-8") == a.read_text(encoding="utf-8"), f"{folder} 의 CLAUDE.md 와 AGENTS.md 가 다름"
    assert "WRITING_GUIDE.md" in c.read_text(encoding="utf-8")


def test_generated_report_and_grader_guide_follow_the_rule(tmp_path, golden, items):
    """사람이 읽는 출력(리포트·전문가 채점 안내)도 같은 말 검사를 지난다."""
    from silvercare_autoqa.golden import DEFAULT_GOLDEN
    from silvercare_autoqa.human import GUIDE
    from silvercare_autoqa.pipeline import finalize, import_judge, prep
    from silvercare_autoqa.responders import reference_rows

    from .conftest import make_judgment

    run = tmp_path / "r"
    prep(DEFAULT_GOLDEN, reference_rows(golden), run)
    for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(make_judgment(
            items[c["item_id"]], c["response"], case_id=c["pid"],
            det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids]), ensure_ascii=False), encoding="utf-8")
    import_judge(run)
    finalize(run)
    report = (run / "report.md").read_text(encoding="utf-8")
    assert "## 한눈에" in "\n".join(report.splitlines()[:12]), "리포트 첫머리에 '## 한눈에' 가 없음"
    for name, text in (("report.md", report), ("human.GUIDE", GUIDE)):
        hits = [w for w in BANNED if w in text]
        assert not hits, f"{name} 에 쓰지 않는 말: {hits}"
