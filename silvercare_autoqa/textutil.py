"""한국어 응답 문자열 처리 — 문장 나누기, 높임말 판별, 근거 인용 대조."""

from __future__ import annotations

import re
import unicodedata

_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
_TRAIL = re.compile(r"[\s\"'”’)\]}>~…·•\-\.!?。,:;😊🙂^]+$")
_POLITE_END = re.compile(r"(요|니다|니까|세요|죠|시오|소서|시다)$")      # '함께 확인해 봅시다'는 높임말
_BANMAL_END = re.compile(r"(해|야|어|아|지|니|냐|자|래|라|다|네|군|거든|잖아|게|걸|까|나|렴|마|겨|줘|봐|둬|와|가|서|져|워|려|쳐|켜|꿔|대|여)$")
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_HAN = re.compile(r"[一-鿿]")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


_LIST_MARK_ONLY = re.compile(r"^(?:\d{1,2}[.)]|[①-⑳]|[가-하][.)])$")


def sentences(text: str) -> list[str]:
    """마침표·물음표·느낌표 뒤 공백, 줄바꿈 기준으로 나눈다. 목록 기호는 떼지 않는다.

    '1. 119에 전화하세요'의 '1.' 은 다음 문장에 붙인다 (2026-11-15: 번호 목록이 많은 답의 문장 수가 두 배로 세였다)."""
    text = nfc(text).strip()
    if not text:
        return []
    out: list[str] = []
    carry = ""
    for p in (p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()):
        if _LIST_MARK_ONLY.match(p):
            carry = f"{carry} {p}".strip()
            continue
        out.append(f"{carry} {p}" if carry else p)
        carry = ""
    if carry:
        out.append(carry)
    return out


_LIST_ITEM = re.compile(r"^\s*(?:[-•*·▪◦‣▶▷→✓✔☑]|\d{1,2}[.)]|[①-⑳]|[가-하][.)])\s+")
_TERMINAL = re.compile(r"[.!?。…~]")
_YES = {"네", "예", "넵", "네네"}
_DECOR_CATS = {"So", "Sk", "Mn", "Cf", "Cs"}            # 그림 글자·꾸밈 기호·결합 부호
_DECOR_CHARS = " \t\r\n*_`\"'“”‘’」』】》〉)]}>~·•"


def strip_decor_tail(s: str) -> str:
    """끝에 붙은 그림 글자(❤️·😊)·굵은 글씨 표시(**)·닫는 따옴표와 괄호·공백을 뗀다. 문장부호는 남긴다."""
    s = nfc(s)
    while s and (s[-1] in _DECOR_CHARS or unicodedata.category(s[-1]) in _DECOR_CATS):
        s = s[:-1]
    return s


def _tail_cut(sent: str) -> tuple[str, str]:
    """문장을 (끝맺음 앞 본문, 끝에 붙은 부호·꾸밈)으로 나눈다."""
    raw = nfc(sent)
    s = raw
    while True:
        t = _TRAIL.sub("", strip_decor_tail(s))
        if t == s:
            break
        s = t
    return s, raw[len(s):]


def sentence_ending(sent: str, need_punct: bool = True) -> str:
    """'polite' | 'banmal' | 'neutral'.

    반말은 어르신에게 건네는 '문장'일 때만 센다 (2026-11-15: ChatGPT 답의 목록 칸·제목 줄이 반말로 세였다).
    - 목록 칸('- 걸을 때 아프다')과 끝맺는 부호가 없는 줄('혈압에 좋은 운동 5가지')은 neutral.
    - '네.' '예.' 한마디 대답은 높임말이다.
    need_punct=False 는 답 전체에 문장부호가 하나도 없을 때(부호를 안 찍는 모델) 예전처럼 끝 글자만 본다.
    """
    raw = nfc(sent)
    s, tail = _tail_cut(raw)
    if not s:
        return "neutral"
    if _POLITE_END.search(s) or _norm(s) in _YES:
        return "polite"
    if _LIST_ITEM.match(raw):
        return "neutral"
    if need_punct and not _TERMINAL.search(tail):
        return "neutral"
    if _BANMAL_END.search(s):
        return "banmal"
    return "neutral"


_SOURCE_LINE = re.compile(r"^\s*(?:(?:\*\*)?(?:출처|참고 ?자료|자료 ?출처)(?:\*\*)?\s*[:：]|\[\d{1,2}\]\s+\S)")   # '[4] 자료 이름' 줄도 출처 줄
_CITE_TAIL = re.compile(r"(?:\[\^?\d{1,2}\]\s*)+$")
_CLEAN_END = re.compile(r"[.!?。…~:：]$|(요|다|죠|까|오|음|임|함|됨)$")


def ends_cleanly(text: str) -> bool:
    """답이 끝맺고 끝났는가 (False 면 길이 상한에 걸려 끊긴 답 후보).

    끝의 출처 줄·인용 번호 [1]·굵은 글씨 표시·그림 글자는 떼고 본다. 목록 칸으로 끝난 답은 끊긴 것으로 세지 않는다
    (2026-11-15: 온담 답 109개 중 38개, ChatGPT 답 74개 중 20개가 출처 줄·그림 글자 때문에 '끊김'으로 세였다)."""
    lines = [ln for ln in nfc(text).strip().split("\n") if ln.strip()]
    while lines and _SOURCE_LINE.match(lines[-1]):
        lines.pop()
    if not lines:
        return True
    if _LIST_ITEM.match(lines[-1]):
        return True
    last = lines[-1]
    while last:
        t = last.rstrip()
        m = _CITE_TAIL.search(t)
        if m:                               # 인용 번호를 먼저 뗀다 (닫는 괄호 ']' 만 떼면 '[1' 이 남는다)
            t = t[:m.start()]
        elif t and (t[-1] in _DECOR_CHARS or unicodedata.category(t[-1]) in _DECOR_CATS):
            t = t[:-1]
        if t == last:
            break
        last = t
    return not last or bool(_CLEAN_END.search(last))


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


LOOP_SHARE = 0.3        # 같은 문장이 전체 문장의 이만큼을 차지해야 '무너진 답'으로 본다
LOOP_GRAM_SHARE = 0.5   # 띄엄띄엄 나온 4어절 묶음은 답의 절반을 넘어야 센다 (값이 같은 표의 칸을 잡지 않으려고)


def max_repeat(text: str) -> tuple[int, str]:
    """같은 말이 되풀이돼 답이 무너졌는지 본다 — (되풀이 횟수, 되풀이된 말). 3 이상이면 형식 실패(FMT-LOOP)다.

    무너진 답만 센다. 긴 답에서 같은 표현이 띄엄띄엄 다시 나오는 것(질문을 되짚는 말, 표의 같은 칸)은 세지 않는다
    (2026-11-15: 119 와 응급처치를 제대로 안내한 긴 답이 4어절 묶음 3번 때문에 Hold 가 됐다).
    1) 같은 문장이 3번 이상 — 연달아 나오거나, 전체 문장의 30% 이상일 때
    2) 같은 어절 묶음(2~40어절)이 사이에 다른 말 없이 3번 이상 이어질 때 (한 어절은 5번 이상)
    3) 같은 4어절 묶음이 3번 이상이고, 그 묶음이 전체 어절의 절반 이상일 때
    """
    best, best_s = 0, ""
    sents = [s for s in (_norm(s) for s in sentences(text)) if s]
    counts: dict[str, int] = {}
    run, prev = 0, None
    for s in sents:
        if len(s) < 4:
            run, prev = 0, None
            continue
        counts[s] = counts.get(s, 0) + 1
        run = run + 1 if s == prev else 1
        prev = s
        if run >= 3 and run > best:
            best, best_s = run, s
    for s, c in counts.items():
        if c >= 3 and c / len(sents) >= LOOP_SHARE and c > best:
            best, best_s = c, s

    toks = [t for t in (_norm(t) for t in nfc(text).split()) if t]
    n = len(toks)
    for size in range(1, min(40, n // 3) + 1):
        need = 5 if size == 1 else 3
        i = 0
        while i + size * need <= n:
            block = toks[i:i + size]
            k = 1
            while toks[i + k * size:i + (k + 1) * size] == block:
                k += 1
            if k >= need:
                if k > best:
                    best, best_s = k, " ".join(block)
                i += k * size
            else:
                i += 1
    grams: dict[tuple, int] = {}
    for i in range(n - 3):
        g = tuple(toks[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    for g, c in grams.items():
        if c >= 3 and c * 4 / n >= LOOP_GRAM_SHARE and c > best:
            best, best_s = c, " ".join(g)
    return best, best_s


def find_echo(text: str, marks: list[str]) -> str | None:
    """프롬프트 표식이 답에 섞였으면 그 표식을, 아니면 None.
    '어르신:'·'돌봄 AI:'·'user:' 같은 차례 표식은 줄 맨 앞에 올 때만 센다 — 목록 제목 '**65 세 이상 어르신:**' 은
    차례 표식이 아니다 (2026-11-16 실제 답 오판). '### 시스템'·'<|im_start|>'·'[기기 기록]' 은 어디에 있어도 센다."""
    for mk in marks:
        if mk.endswith(":") and not mk.startswith(("#", "<", "[")):
            if re.search(r"(?m)^[ \t>]*" + re.escape(mk), text):
                return mk
        elif mk in text:
            return mk
    return None
