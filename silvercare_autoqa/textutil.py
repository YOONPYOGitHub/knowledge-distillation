"""한국어 응답 문자열 처리 — 문장 나누기, 높임말 판별, 근거 인용 대조."""

from __future__ import annotations

import re
import unicodedata

_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
_TRAIL = re.compile(r"[\s\"'”’)\]}>~…·•\-\.!?。,:;😊🙂^]+$")
_POLITE_END = re.compile(r"(요|니다|니까|세요|죠|시오|소서)$")
_BANMAL_END = re.compile(r"(해|야|어|아|지|니|냐|자|래|라|다|네|군|거든|잖아|게|걸|까|나|렴|마|겨|줘|봐|둬|와|가|서|져|워|려|쳐|켜|꿔|대|여)$")
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_HAN = re.compile(r"[一-鿿]")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def sentences(text: str) -> list[str]:
    """마침표·물음표·느낌표 뒤 공백, 줄바꿈 기준으로 나눈다. 목록 기호는 떼지 않는다."""
    text = nfc(text).strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]


def sentence_ending(sent: str) -> str:
    """'polite' | 'banmal' | 'neutral' (명사형 목록·숫자로 끝나는 문장 등)."""
    s = _TRAIL.sub("", nfc(sent))
    if not s:
        return "neutral"
    if _POLITE_END.search(s):
        return "polite"
    if _BANMAL_END.search(s):
        return "banmal"
    return "neutral"


def hangul_ratio(text: str) -> float:
    t = nfc(text)
    h = len(_HANGUL.findall(t))
    other = len(_LATIN.findall(t)) + len(_HAN.findall(t))
    total = h + other
    return 1.0 if total == 0 else h / total


def _norm(s: str) -> str:
    s = nfc(s)
    return re.sub(r"[\s\"'“”‘’`.,!?。·…~\-()\[\]{}<>:;]+", "", s)


def norm_len(s: str) -> int:
    """공백·문장부호를 뺀 글자 수 (근거 인용 길이 검사용)."""
    return len(_norm(s or ""))


def grounded(evidence: str, response: str) -> bool:
    """근거 인용이 응답 원문에 실제로 있는지. 공백·문장부호는 무시한다.
    '…' 또는 '...' 로 이어 붙인 인용은 조각마다 확인한다."""
    ev = nfc(evidence or "").strip()
    if not ev:
        return False
    resp = _norm(response)
    parts = [p for p in re.split(r"…|\.\.\.|\s/\s", ev) if _norm(p)]
    if not parts:
        return False
    return all(_norm(p) in resp for p in parts)


def max_repeat(text: str) -> tuple[int, str]:
    """같은 문장이 몇 번 반복됐는지와 그 문장, 그리고 4어절 묶음 반복 중 큰 값."""
    sents = [_norm(s) for s in sentences(text)]
    best, best_s = 0, ""
    for s in set(sents):
        if len(s) >= 4:
            c = sents.count(s)
            if c > best:
                best, best_s = c, s
    toks = nfc(text).split()
    grams: dict[tuple, int] = {}
    for i in range(len(toks) - 3):
        g = tuple(toks[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    if grams:
        g, c = max(grams.items(), key=lambda kv: kv[1])
        if c > best:
            best, best_s = c, " ".join(g)
    return best, best_s
