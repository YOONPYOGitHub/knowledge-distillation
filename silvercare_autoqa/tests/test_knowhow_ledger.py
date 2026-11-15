"""쇼핑 AI 검수 원칙 대조 장부(data/knowhow_ledger.yaml)와 발표 자료가 어긋나지 않는지 본다.

장부는 '무엇을 넣었고 무엇을 미뤘나'를 하나씩 적은 표다. 발표 자료·감사 기록의 숫자는 이 장부에서 센다.
숫자를 손으로 고쳐 쓰면 장부와 어긋나므로, 장부의 모양과 문서에 적힌 숫자를 여기서 맞춰 본다.
"""

import hashlib
import html
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "silvercare_autoqa" / "data" / "knowhow_ledger.yaml"
TALK = ROOT / "docs" / "autoqa" / "talk" / "index.html"
STATUSES = {"applied", "today", "next", "na"}
# 공개 저장소에 적지 않는 이름 — 이름을 여기에도 적지 않으려고 (글자 수, 해시 앞 16자리)로만 둔다.
# 글에서 띄어쓰기를 빼고 소문자로 바꾼 뒤, 같은 길이로 잘라 해시를 견준다.
_PRIVATE = [(5, '0929e0f87c9842dd'), (6, 'cae2c32ee9aff5b3'), (7, '7c2a2bbba9f3c322'), (8, '0932db1f939308d8'), (10, '7f26d1513b6f1757')]


def _has_private_name(text: str) -> bool:
    flat = re.sub(r"\s+", "", text).lower()
    for n, h in _PRIVATE:
        for i in range(len(flat) - n + 1):
            if hashlib.sha256(flat[i:i + n].encode("utf-8")).hexdigest()[:16] == h:
                return True
    return False


def _ledger() -> dict:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


def test_ledger_shape():
    d = _ledger()
    items = d["items"]
    ids = [r["id"] for r in items]
    assert len(ids) == len(set(ids)) == 168
    assert sum(i.startswith("K") for i in ids) == 60 and sum(i.startswith("N") for i in ids) == 108
    for r in items:
        assert r["status"] in STATUSES, r["id"]
        assert r["family"] in d["families"], r["id"]
        assert r["principle"].strip(), r["id"]
        if r["status"] in ("today", "next", "na"):
            assert r["note"].strip(), f"{r['id']}: 넣은 곳·넣을 때·맞지 않는 이유를 적는다"


def test_ledger_has_no_company_names():
    assert not _has_private_name(LEDGER.read_text(encoding="utf-8"))


def test_docs_quote_the_ledger_counts():
    """감사 기록·한 장 요약·발표 자료에 적힌 '넣은 것 · 넣을 때를 정한 것 · 맞지 않는 것'이 장부와 같다."""
    c = Counter(r["status"] for r in _ledger()["items"])
    done, nxt, na = c["applied"] + c["today"], c["next"], c["na"]
    audit = (ROOT / "docs" / "autoqa" / "AUDIT_LOG.md").read_text(encoding="utf-8")
    assert f"들어 있는 것 {done} · 넣을 때를 정한 것 {nxt} · 이 프로젝트에 맞지 않는 것 {na}" in audit
    index = (ROOT / "docs" / "autoqa" / "index.html").read_text(encoding="utf-8")
    assert f">{done}<span class=\"lab\"> / 168</span>" in index and f"넣을 때를 정한 것 {nxt} · 맞지 않는 것 {na}" in index
    talk = html.unescape(re.sub(r"<[^>]+>", " ", TALK.read_text(encoding="utf-8")))
    assert re.search(rf"{done}\s+가지\s+이미 들어 있거나 오늘 넣었다", talk) and re.search(rf"{nxt}\s+가지\s+넣을 때를 정해 두었다", talk)


def test_talk_page_is_plain_and_self_contained():
    """발표 자료: 밖에서 불러오는 파일이 없고, 회사 이름과 쓰지 않는 말이 없다."""
    from .test_docs import BANNED

    src = TALK.read_text(encoding="utf-8")
    assert not re.search(r"""(?:src|href)\s*=\s*["']https?://""", src), "발표 자료는 인터넷 없이 열려야 한다"
    body = re.sub(r"<(script|style)\b.*?</\1>", "", src, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", body))
    assert not _has_private_name(text)
    bad = [w for w in BANNED if w in text]
    assert not bad, f"발표 자료에 쓰지 않는 말: {bad}"
    assert "{nb}" not in src
