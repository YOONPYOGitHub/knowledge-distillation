"""어르신 실험 대본·음성 목록 시험 — 어르신이 듣거나 진행자가 보면 안 되는 것이 남지 않는다 (노하우 점검 K50).

실행안은 "인용 번호는 소리로 읽지 않는다 · 진행자도 어느 답이 온담인지 모른다"고 적었다 (docs/user_study/README.md §2).
대본이나 음성 목록에 인용 번호 [1] · 굵은 글씨 표시 ** · 시스템 이름이 하나라도 남으면 이 시험이 실패한다.
답 앞뒤에 붙인 꾸밈말은 모두 지어 쓴 문장이다 (실제 답·출처 글을 옮기지 않는다).
"""

import csv
import io
import json
import re

from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.pipeline import finalize, import_judge, prep
from silvercare_autoqa.study import MAIN_ITEMS, PRACTICE_ITEM, PREF_ITEMS, materials, spoken_text

from .conftest import make_judgment

CONDITIONS = ("ondam_local", "chatgpt_luna61", "local_norag")
# 남으면 안 되는 시스템 이름 — 아래 꾸밈말에 모두 넣었다
NAMES = ("온담", "ChatGPT", "챗GPT", "GPT-6.1", "Claude", "클로드", "Qwen", "Kanana")
HEAD = "**먼저 말씀드려요.** 저는 온담이에요.[1] "
TAIL = " ChatGPT 와 챗GPT, GPT-6.1, Claude, 클로드, Qwen, Kanana 도 **같은 말**을 해요.[2][12]"
CITE = re.compile(r"\[\d+\]")


def _leftovers(text: str) -> list[str]:
    """글에 남은 인용 번호 · 굵은 글씨 표시 · 시스템 이름."""
    low = text.lower()
    return CITE.findall(text) + (["**"] if "**" in text else []) + [n for n in NAMES if n.lower() in low]


def _decorated_run(tmp_path, items):
    """세 조건의 실험 문항 답(모범 답안)에 인용 번호·굵은 글씨·시스템 이름을 붙인 채점 끝난 회차."""
    used = MAIN_ITEMS + [PRACTICE_ITEM] + PREF_ITEMS
    rows = [{"model": m, "item_id": i, "response": HEAD + items[i]["reference_answer"] + TAIL} for m in CONDITIONS for i in used]
    run = tmp_path / "run"
    prep(DEFAULT_GOLDEN, rows, run, allow_partial=True)
    for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(make_judgment(
            items[c["item_id"]], c["response"], case_id=c["pid"],
            det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids]), ensure_ascii=False), encoding="utf-8")
    import_judge(run)
    assert finalize(run)["validity"]["ok"]
    return run


def test_script_and_voice_list_carry_no_citation_bold_or_system_name(tmp_path, golden, items):
    assert all(_leftovers(HEAD + TAIL).count(x) for x in ["[1]", "[12]", "**"] + list(NAMES))     # 꾸밈말에 세 가지가 다 들어 있다
    run = _decorated_run(tmp_path, items)
    rep = materials(run, tmp_path / "b", n_participants=3)
    assert not rep["swaps"]                                               # 꾸민 답도 쓸 수 있는 답이다 (Hold 가 아니다)
    pages = sorted((tmp_path / "b").glob("P*/*.html"))                     # 진행 대본 · 참가자 카드 · 관찰 기록지
    assert len(pages) == 9
    for p in pages:
        text = p.read_text(encoding="utf-8")
        assert not _leftovers(text), (p.parent.name, p.name, _leftovers(text)[:6])     # 진행자가 읽고 어르신이 보는 글
    for p in sorted((tmp_path / "b").glob("P*/진행대본.html")):
        text = p.read_text(encoding="utf-8")
        assert "먼저 말씀드려요." in text and "같은 말" in text               # 답의 말은 그대로 남는다
    voice = list(csv.DictReader(io.StringIO((tmp_path / "b" / "음성목록.csv").read_text(encoding="utf-8-sig"), newline="")))
    assert len(voice) == rep["n_answers"] > 0
    for r in voice:                                                       # 앱 TTS 가 읽는 글
        assert not _leftovers(r["text"]), (r["file"], _leftovers(r["text"]))
        assert r["text"].startswith("먼저 말씀드려요.") and "같은 말" in r["text"]


def test_original_answer_stays_in_the_run_for_scoring(tmp_path, golden, items):
    """채점은 화면의 답 원문으로 한다. 다듬은 글은 대본·음성 목록에만 쓰고, 원문은 대응표의 pid 로 회차에서 찾는다."""
    run = _decorated_run(tmp_path, items)
    materials(run, tmp_path / "b", n_participants=3)
    cases = {c["pid"]: c for c in map(json.loads, (run / "scored.jsonl").read_text(encoding="utf-8").splitlines())}
    key = json.loads((tmp_path / "b" / "대응표_봉인.json").read_text(encoding="utf-8"))
    links = list(key["trials"].values()) + list(key["pref"].values())
    assert links and all(HEAD in cases[t["pid"]]["response"] and "[12]" in cases[t["pid"]]["response"] for t in links)


def test_spoken_text_drops_marks_and_keeps_words():
    t = spoken_text("**물을 자주 드세요.**[1] 온담이 내일 다시 여쭤볼게요.[2]")
    assert not _leftovers(t)
    assert t.startswith("물을 자주 드세요.") and t.endswith("내일 다시 여쭤볼게요.")
    assert spoken_text("약사에게 먼저 물어보세요.") == "약사에게 먼저 물어보세요."      # 뺄 것이 없으면 그대로
