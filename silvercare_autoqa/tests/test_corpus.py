"""온담 답의 잘린 근거 발췌를 지식 자료의 온전한 토막으로 바꾸는 도구(corpus.py) 시험 — 지어낸 짧은 글만 쓴다."""

import json

from silvercare_autoqa.corpus import attach_corpus_excerpts, attach_file, load_knowledge

FULL = "고혈압 예방 안내 " + "혈압은 집에서 아침저녁으로 재고 적어 두면 진료에 도움이 됩니다. " * 12 + "소금은 하루 한 숟갈보다 적게 드세요."


def _knowledge(tmp_path):
    rows = [{"title": "고혈압 예방", "content": FULL, "source_url": "https://example.org/bp"},
            {"title": "다른 글", "content": "전혀 다른 내용입니다. " * 20, "source_url": "https://example.org/bp"}]
    p = tmp_path / "knowledge.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


def _row(excerpt, url="https://example.org/bp", kind="reference"):
    return {"model": "ondam", "item_id": "SC-K06", "response": "국물은 남기세요.[1]",
            "rag": {"mode": "grounded", "sources": [{"id": 1, "title": "고혈압 예방", "url": url, "excerpt": excerpt, "evidence_kind": kind}]}}


def test_replaces_only_when_the_excerpt_is_found_in_a_chunk(tmp_path):
    kn = load_knowledge(_knowledge(tmp_path))
    hit, miss = _row(FULL[:120]), _row("지식 자료에 없는 글입니다. 이 문장은 어디에도 없습니다. 그래서 그대로 둡니다. 끝까지 그대로입니다.")
    api = _row("질병관리청에서 바로 가져온 글입니다. 지식 자료에는 들어 있지 않은 본문이라 발췌를 그대로 둡니다.", url="https://kdca.example/1", kind="official_api")
    stats = attach_corpus_excerpts([hit, miss, api], kn)
    assert stats["sources"] == 3 and stats["replaced"] == 1 and stats["kept"] == 2
    assert stats["kept_by_kind"] == {"reference": 1, "official_api": 1}
    s = hit["rag"]["sources"][0]
    assert s["excerpt"] == FULL and s["excerpt_from"] == "team_corpus" and s["excerpt_api"] == FULL[:120]
    assert "excerpt_from" not in miss["rag"]["sources"][0] and miss["rag"]["sources"][0]["excerpt"].startswith("지식 자료에 없는")


def test_attach_file_round_trip_and_judge_sees_the_full_chunk(tmp_path, golden):
    from silvercare_autoqa.golden import DEFAULT_GOLDEN
    from silvercare_autoqa.pipeline import prep

    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text(json.dumps(_row(FULL[:120]), ensure_ascii=False) + "\n", encoding="utf-8")
    stats = attach_file(src, _knowledge(tmp_path), out)
    assert stats["replaced"] == 1
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    a, b = tmp_path / "short", tmp_path / "full"
    prep(DEFAULT_GOLDEN, [_row(FULL[:120])], a, allow_partial=True)
    prep(DEFAULT_GOLDEN, rows, b, allow_partial=True)
    pa = json.loads(next((a / "packets" / "judge").glob("*.json")).read_text(encoding="utf-8"))
    pb = json.loads(next((b / "packets" / "judge").glob("*.json")).read_text(encoding="utf-8"))
    assert "소금은 하루 한 숟갈보다 적게" not in pa["user"] and "소금은 하루 한 숟갈보다 적게" in pb["user"]
    assert pa["packet_id"] != pb["packet_id"]          # 근거 자료가 바뀌면 다른 채점 요청이다 (예전 채점이 붙지 않는다)
