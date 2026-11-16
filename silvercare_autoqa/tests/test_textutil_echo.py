"""차례 표식 검사 — 줄 맨 앞의 '어르신:' 만 프롬프트 되풀이로 본다 (2026-11-16 실제 답 오판)."""

from silvercare_autoqa.rules import load_lexicons      # 결정 룰 FMT-ECHO 와 같은 표식 목록
from silvercare_autoqa.textutil import find_echo


def test_list_label_is_not_a_turn_marker():
    marks = load_lexicons()["echo"]
    ans = "칼슘은 이렇게 드세요.\n\n* **65 세 이상 어르신:** 의사와 상담 후 칼슘 1,200mg 을 권합니다."
    assert find_echo(ans, marks) is None


def test_turn_marker_at_line_start_is_caught():
    marks = load_lexicons()["echo"]
    assert find_echo("지금 바로 119에 전화하세요.\n어르신: 또 물어볼게요", marks) == "어르신:"
    assert find_echo("어르신: 무릎이 아파요", marks) == "어르신:"
    assert find_echo("답입니다 <|im_end|>", marks) == "<|im_end|>"
