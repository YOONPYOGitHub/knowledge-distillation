"""전문가 채점지 묶음 시험 — 노하우 점검(2026-11-15) K41·K42·K43·K21 에서 찾은 빈틈.

- K41 대응표에 응답 해시가 들어간다. 지문이 다른 회차에도 모델·문항·응답 해시가 같은 답이면 라벨을 옮겨 붙인다.
       답이 바뀐 줄은 붙이지 않는다. κ 값 옆에 읽는 말이 붙고, 읽는 법 한 줄이 첫머리에 나온다.
- K43 검수자가 '의견' 칸에 적은 줄이 모두 표에 나온다. '고칠 곳' 칸은 비어 있다.
- K21 autoqa 는 Fix 인데 사람은 Pass 인 줄을 세고 적는다. 출처 지적만으로 Fix 인 답은 따로 표시한다.
- K42 exclude_key 로 겹치지 않는 채점지 A·B 를 낸다.
시험에 쓰는 답은 골든셋 모범 답안이거나 지어 쓴 문장이다. 의견·출처 발췌도 지어 썼다.
"""

import csv
import io
import json
from pathlib import Path

import pytest
import yaml

from silvercare_autoqa.cli import main
from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.human import (COLS, FIX_WHERE_COL, KAPPA_GUIDE, VERDICT_COL, export_blind, import_labels, kappa_word,
                                     render_agreement_md)
from silvercare_autoqa.pipeline import finalize, import_judge, prep

from .conftest import make_judgment

MODELS = ("sys_a", "sys_b")
IDS = ["SC-E01", "SC-E02", "SC-K01", "SC-K03", "SC-M01", "SC-M08", "SC-G06", "SC-S16"]
HOLD = ("sys_a", "SC-E01")                                              # 응급 놓침 답 (결정 룰이 확정)
FIX = {("sys_a", "SC-K01"), ("sys_b", "SC-K03"), ("sys_b", "SC-M01")}    # 채점기가 필수답 빠짐·결론 다름으로 본 답
G1, VCOL, OPINION = COLS.index("G1 응급 놓침(예/아니오)"), COLS.index(VERDICT_COL), COLS.index("의견")
SOURCE = {"id": 1, "title": "자료1", "url": "", "excerpt": "지어 쓴 발췌 문장입니다.", "evidence_kind": "reference"}


def _run(folder, items, *, golden_path=DEFAULT_GOLDEN, fix=FIX, edits=None, extra_rows=()):
    """모델 둘 × 문항 여덟의 작은 채점 회차. Hold 1 · Fix 3 · 나머지 Pass 가 되게 채점 결과를 넣는다."""
    rows = [{"model": m, "item_id": i, "response": items[i]["reference_answer"]} for m in MODELS for i in IDS]
    for r in rows:
        if (r["model"], r["item_id"]) == HOLD:
            r["response"] = "좀 누워서 쉬어 보세요."
        r["response"] = (edits or {}).get((r["model"], r["item_id"]), r["response"])
    prep(golden_path, rows + list(extra_rows), folder, allow_partial=True)
    for line in (folder / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if not c["judged"]:
            continue
        it = items[c["item_id"]]
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        kw = {"met": {"I1": False}, "concl": False} if (c["model"], c["item_id"]) in fix else {}
        j = make_judgment(it, c["response"], case_id=c["pid"], det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids],
                          **kw)
        if c.get("rag_view") is not None:                               # 출처가 붙은 답은 근거 대조 칸이 있어야 한다
            j["grounding"] = {"issues": [], "key_facts_in_context": [{"id": k["id"], "in_context": False} for k in it["key_facts"]],
                              "coverage_handling": "n/a", "faithfulness": 5}
        (folder / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    assert import_judge(folder)["error"] == 0
    finalize(folder)
    return folder


def _golden_v1(tmp_path, utterances=None) -> Path:
    """기준을 고친 골든셋 흉내 — 주석 한 줄을 더해 평가기준 지문이 달라진다.
    utterances={문항: 새 발화} 를 주면 그 문항의 질문도 바꾼다."""
    text = Path(DEFAULT_GOLDEN).read_text(encoding="utf-8")
    if utterances:
        doc = yaml.safe_load(text)
        for it in doc["items"]:
            it["utterance"] = utterances.get(it["id"], it["utterance"])
        text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    p = tmp_path / "golden_v1.yaml"
    p.write_text(text + "\n# 시험용: 기준을 고친 골든셋\n", encoding="utf-8")
    return p


def _meta(run) -> dict:
    return json.loads((run / "meta.json").read_text(encoding="utf-8"))


def _cases(run) -> dict:
    lines = (run / "scored.jsonl").read_text(encoding="utf-8").splitlines()
    return {(c["model"], c["item_id"]): c for c in map(json.loads, lines)}


def _sheet(folder) -> tuple[dict, list[list[str]]]:
    key = json.loads((folder / "blind_key.json").read_text(encoding="utf-8"))
    rows = list(csv.reader(io.StringIO((folder / "blind_sheet.csv").read_text(encoding="utf-8-sig"), newline="")))
    return key, rows


def _same_as_autoqa(k: dict) -> str:
    return k["autoqa_verdict"]


def _hold_or_pass(k: dict) -> str:
    return "hold" if k["autoqa_verdict"] == "hold" else "pass"


def _fill(key, rows, human=_hold_or_pass):
    """줄마다 사람 라벨을 적는다. human(대응표 줄) → 'hold'·'fix'·'pass'."""
    for r in rows[1:]:
        v = human(key["rows"][r[0]])
        r[G1], r[VCOL] = "예" if v == "hold" else "아니오", v.capitalize()
    return rows


def _write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return path


def _case_set(key) -> set:
    return {(v["model"], v["item_id"]) for v in key["rows"].values()}


# ── K41: 응답 해시 · 라벨 옮겨 붙이기 · κ 읽는 법 ──

def test_key_has_the_response_sha_of_each_case(tmp_path, golden, items):
    run = _run(tmp_path / "run", items)
    export_blind(run, tmp_path / "A")
    key, _ = _sheet(tmp_path / "A")
    cases = _cases(run)
    assert len(key["rows"]) == len(cases) == 16
    assert all(v["response_sha"] and v["response_sha"] == cases[(v["model"], v["item_id"])]["response_sha"]
               for v in key["rows"].values())


def test_labels_move_to_a_rescored_run_only_when_the_answer_is_the_same(tmp_path, golden, items):
    run1 = _run(tmp_path / "run1", items)
    key_path = tmp_path / "A" / "blind_key.json"
    export_blind(run1, tmp_path / "A")
    key, rows = _sheet(tmp_path / "A")
    changed, refixed = ("sys_b", "SC-G06"), ("sys_a", "SC-M08")                     # 다음 회차에서 답이 바뀔 줄 · 판정이 바뀔 줄
    rid = {(v["model"], v["item_id"]): r for r, v in key["rows"].items()}
    for r in _fill(key, rows, _same_as_autoqa)[1:]:                                 # 사람 판정 = 채점지를 낸 회차의 autoqa 판정
        if r[0] in (rid[changed], rid[refixed]):
            r[OPINION] = "설명이 조금 길어요"
    labeled = _write(tmp_path / "약사.csv", rows)

    same = import_labels(run1, key_path, [labeled])                                 # 채점지를 낸 회차: 옮긴 라벨이 없다
    g = same["graders"]["약사"]
    assert same["moved"] is None and g["n_labeled"] == 16 and g["n_moved"] == 0 and not g["not_moved"]
    assert g["kappa_verdict3"] == 1.0 and "옮겨 붙인 라벨" not in render_agreement_md(same)

    # 기준을 고쳐 다시 채점한 회차 — 지문이 다르다. 답 하나가 바뀌었고, 다른 답 하나는 autoqa 판정이 Pass 에서 Fix 로 바뀌었다
    run2 = _run(tmp_path / "run2", items, golden_path=_golden_v1(tmp_path), fix=FIX | {refixed},
                edits={changed: "물을 조금 줄이고 일찍 주무세요."})
    assert _meta(run1)["fingerprint"] != _meta(run2)["fingerprint"]
    rep = import_labels(run2, key_path, [labeled])
    g = rep["graders"]["약사"]
    assert rep["moved"]["mark"] == "옮겨 붙인 라벨" and rep["moved"]["key_fingerprint"] == _meta(run1)["fingerprint"]
    assert rep["moved"]["run_fingerprint"] == _meta(run2)["fingerprint"]
    assert g["n_labeled"] == g["n_moved"] == 15 and not g["errors"]                 # 답이 바뀐 줄은 틀린 값으로 세지 않는다
    assert [(r["model"], r["item_id"], r["까닭"]) for r in g["not_moved"]] == [(*changed, "답이 바뀜")]
    # autoqa 판정은 이 회차의 것을 쓴다 — 새로 Fix 가 된 답이 사람 Pass 와 어긋난다
    assert g["kappa_verdict3"] < 1.0 and [(r["model"], r["item_id"]) for r in g["fix_pass"]] == [refixed]
    # 의견은 붙이지 못한 줄의 것도 남긴다. 그 줄은 이 회차에 같은 답이 없어 autoqa 판정이 비어 있다
    ops = {(o["model"], o["item_id"]): o["autoqa"] for o in g["opinions"]}
    assert ops == {refixed: "fix", changed: None}
    md = render_agreement_md(rep)
    assert "— 15건 (옮겨 붙인 라벨)" in md and "- 옮겨 붙인 라벨 15건 · 붙이지 못한 라벨 1건 (일치도에서 뺐다)" in md
    assert f"  - 답이 바뀜 1건: {g['not_moved'][0]['번호']}" in md
    assert _meta(run1)["fingerprint"] in md and _meta(run2)["fingerprint"] in md
    assert main(["import-labels", "--run", str(run2), "--key", str(key_path), "--labels", str(labeled)]) == 0

    # 응답 해시가 없는 예전 대응표는 지문이 다른 회차에 붙이지 않는다 (답이 같은지 알 수 없다)
    old = {**key, "rows": {no: {k: v for k, v in row.items() if k != "response_sha"} for no, row in key["rows"].items()}}
    old_path = tmp_path / "old_key.json"
    old_path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="지문이 다름"):
        import_labels(run2, old_path, [labeled])
    assert import_labels(run1, old_path, [labeled])["graders"]["약사"]["n_labeled"] == 16      # 낸 회차에는 그대로 붙는다


def test_labels_do_not_move_when_the_question_changed(tmp_path, golden, items):
    """답 글이 같아도 질문(발화)이 바뀐 문항의 라벨은 붙이지 않는다 — 검수자는 예전 질문을 보고 판정했다."""
    run1 = _run(tmp_path / "run1", items)
    export_blind(run1, tmp_path / "A")
    key, rows = _sheet(tmp_path / "A")
    assert all(len(v["question_sha"]) == 16 for v in key["rows"].values())
    labeled = _write(tmp_path / "약사.csv", _fill(key, rows, _same_as_autoqa))
    asked = items["SC-S16"]["utterance"] + " 같이 먹어도 되는지 궁금해요."
    run2 = _run(tmp_path / "run2", items, golden_path=_golden_v1(tmp_path, utterances={"SC-S16": asked}))
    assert all(_cases(run1)[k]["response_sha"] == c["response_sha"] for k, c in _cases(run2).items())      # 답은 모두 그대로다
    g = import_labels(run2, tmp_path / "A" / "blind_key.json", [labeled])["graders"]["약사"]
    assert g["n_moved"] == 14 and not g["errors"]
    assert sorted((r["model"], r["item_id"], r["까닭"]) for r in g["not_moved"]) == [
        ("sys_a", "SC-S16", "질문(발화·기기 기록)이 바뀜"), ("sys_b", "SC-S16", "질문(발화·기기 기록)이 바뀜")]


def test_kappa_has_a_word_next_to_it_and_one_reading_line(tmp_path, golden, items):
    words = [kappa_word(k) for k in (-0.2, 0.39, 0.4, 0.59, 0.6, 0.79, 0.8, 1.0)]
    assert words == ["낮음", "낮음", "보통", "보통", "높음", "높음", "매우 높음", "매우 높음"]
    assert kappa_word(None) == "" and kappa_word(0.598) == "높음"                   # 0.60 으로 찍히는 값은 0.6 으로 읽는다
    for part in ("0.4 미만은 낮음", "0.4~0.6 은 보통", "0.6~0.8 은 높음", "0.8 이상은 매우 높음"):
        assert part in KAPPA_GUIDE
    run = _run(tmp_path / "run", items)
    export_blind(run, tmp_path / "A")
    key, rows = _sheet(tmp_path / "A")
    (tmp_path / "간호사").mkdir()
    a = _write(tmp_path / "약사.csv", _fill(key, rows, _same_as_autoqa))
    b = _write(tmp_path / "간호사" / "간호사.csv", _fill(key, rows, _hold_or_pass))
    md = render_agreement_md(import_labels(run, tmp_path / "A" / "blind_key.json", [a, b]))
    assert KAPPA_GUIDE in md.split("\n## ", 1)[0]                                   # 첫 κ 값보다 앞, 첫 화면에 읽는 법
    assert md.count(KAPPA_GUIDE) == 1
    assert "**Hold 여부 κ 1.00 (매우 높음)** · 일치 16/16 · Hold/Fix/Pass κ 1.00 (매우 높음)" in md       # 약사
    assert "Hold/Fix/Pass κ 0.36 (낮음)" in md                                      # 간호사 — Fix 3건을 Pass 로 봄
    between = next(l for l in md.splitlines() if l.startswith("- 검수자끼리"))
    assert between.endswith("κ 0.36 (낮음)")
    assert "### 검수자 의견 0건" in md and "'의견' 칸에 적은 줄이 없다." in md


# ── K43: 의견 표 ──

def test_every_opinion_is_listed_with_an_empty_fix_column(tmp_path, golden, items):
    run = _run(tmp_path / "run", items)
    export_blind(run, tmp_path / "A")
    key, rows = _sheet(tmp_path / "A")
    rows = _fill(key, rows)
    rid = {(v["model"], v["item_id"]): r for r, v in key["rows"].items()}
    by = {r[0]: r for r in rows[1:]}
    passed, blank, broken = rid[("sys_b", "SC-E02")], rid[("sys_a", "SC-M08")], rid[("sys_b", "SC-S16")]
    by[passed][OPINION] = "말투가 조금 딱딱해요"                                    # 놓친 위험·과잉 Hold 가 아닌 줄의 의견
    by[blank][G1] = by[blank][VCOL] = ""                                            # 판정은 안 적고 의견만 적은 줄
    by[blank][OPINION] = "질문이 무슨 뜻인지\n모르겠어요 | 다시 봐 주세요"
    by[broken][G1], by[broken][OPINION] = "아마도", "기준이 너무 엄격해요"          # 값이 틀린 줄의 의견
    rep = import_labels(run, tmp_path / "A" / "blind_key.json", [_write(tmp_path / "약사.csv", rows)])
    g = rep["graders"]["약사"]
    ops = {o["번호"]: o for o in g["opinions"]}
    assert set(ops) == {passed, blank, broken} and all(o[FIX_WHERE_COL] == "" for o in ops.values())
    assert (ops[passed]["item_id"], ops[passed]["autoqa"], ops[passed]["사람"]) == ("SC-E02", "pass", "pass")
    assert ops[blank]["사람"] is None and ops[broken]["사람"] is None
    assert [o["번호"] for o in g["opinions"]] == sorted(ops)                        # 번호 순서
    assert g["n_labeled"] == 14 and len(g["errors"]) == 1                           # 의견만 적은 줄·틀린 줄은 일치도에 넣지 않는다
    md = render_agreement_md(rep)
    assert "### 검수자 의견 3건" in md
    assert f"| 번호 | 문항 | autoqa 판정 | 사람 판정 | 의견 | {FIX_WHERE_COL} |" in md and FIX_WHERE_COL == "고칠 곳"
    assert f"| {passed} | SC-E02 | Pass | Pass | 말투가 조금 딱딱해요 |  |" in md
    assert f"| {blank} | SC-M08 | Pass | — | 질문이 무슨 뜻인지 모르겠어요 \\| 다시 봐 주세요 |  |" in md      # 줄바꿈·세로줄이 표를 깨지 않는다
    assert f"| {broken} | SC-S16 | Pass | — | 기준이 너무 엄격해요 |  |" in md


# ── K21: autoqa 는 Fix, 사람은 Pass ──

def test_autoqa_fix_but_human_pass_is_counted_and_listed(tmp_path, golden, items):
    # 없는 출처 번호 [4] 를 인용한 답 — 출처 지적만으로 Fix 다 (출처 지적을 뺀 공통 기준 판정은 Pass)
    cited = {"model": "sys_a", "item_id": "SC-S23", "response": items["SC-S23"]["reference_answer"] + "[4]",
             "rag": {"mode": "grounded", "sources": [SOURCE]}}
    run = _run(tmp_path / "run", items, extra_rows=[cited])
    res = _cases(run)[("sys_a", "SC-S23")]["result"]
    assert res["verdict"] == "fix" and res["verdict_common"] == "pass"
    export_blind(run, tmp_path / "A")
    key, rows = _sheet(tmp_path / "A")
    agreed = ("sys_b", "SC-M01")                                                    # 이 답은 사람도 Fix 로 본다

    def human(k):
        return "fix" if (k["model"], k["item_id"]) == agreed else _hold_or_pass(k)

    rep = import_labels(run, tmp_path / "A" / "blind_key.json", [_write(tmp_path / "간호사.csv", _fill(key, rows, human))])
    g = rep["graders"]["간호사"]
    assert g["n_autoqa_fix"] == 4                                                   # Fix 3 + 출처 지적 1
    got = {(r["model"], r["item_id"]): r["출처 지적뿐"] for r in g["fix_pass"]}
    assert got == {("sys_a", "SC-K01"): False, ("sys_b", "SC-K03"): False, ("sys_a", "SC-S23"): True}
    md = render_agreement_md(rep)
    assert "**autoqa 는 Fix, 사람은 Pass 3건** (autoqa 가 Fix 로 본 답 4건 중) · 그중 출처·인용 지적뿐인 답 1건." in md
    k03 = next(r for r in g["fix_pass"] if r["item_id"] == "SC-K03")
    s23 = next(r for r in g["fix_pass"] if r["item_id"] == "SC-S23")
    assert f"  - Fix→Pass {k03['번호']} sys_b · SC-K03" in md
    assert f"  - Fix→Pass {s23['번호']} sys_a · SC-S23 (출처·인용 지적뿐" in md


# ── K42: 겹치지 않는 채점지 A·B ──

def test_exclude_key_gives_non_overlapping_sheets(tmp_path, golden, items):
    run = _run(tmp_path / "run", items)
    key_a, key_b, key_c = (tmp_path / n / "blind_key.json" for n in "ABC")
    a = export_blind(run, tmp_path / "A", sample=6)
    b = export_blind(run, tmp_path / "B", sample=6, exclude_key=key_a)
    ka, _ = _sheet(tmp_path / "A")
    kb, rows_b = _sheet(tmp_path / "B")
    assert a["rows"] == b["rows"] == 6 and a["excluded"] == 0 and b["excluded"] == kb["excluded"] == 6
    assert kb["excluded_keys"] == [str(key_a)]
    assert not _case_set(ka) & _case_set(kb)                                        # 같은 답이 두 채점지에 들어가지 않는다
    assert sorted(ka["rows"]) == [f"B{n:03d}" for n in range(1, 7)]
    assert [r[0] for r in rows_b[1:]] == [f"B{n:03d}" for n in range(7, 13)]        # 줄 번호는 앞 채점지 다음부터

    # 채점지 B 를 대응표 A 와 함께 넣어도 라벨이 다른 답에 붙지 않는다 — 모두 '대응표에 없는 줄'로 남는다
    filled_b = _write(tmp_path / "채점지B.csv", _fill(kb, rows_b))
    g = import_labels(run, key_a, [filled_b])["graders"]["채점지B"]
    assert g["n_labeled"] == 0 and g["unknown_rows"] == sorted(kb["rows"])
    assert import_labels(run, key_b, [filled_b])["graders"]["채점지B"]["n_labeled"] == 6

    c = export_blind(run, tmp_path / "C", exclude_key=[key_a, key_b])               # 앞 대응표 여럿을 함께 뺀다
    kc, _ = _sheet(tmp_path / "C")
    assert c["rows"] == 4 and c["excluded"] == 12 and min(kc["rows"]) == "B013"
    assert not _case_set(kc) & (_case_set(ka) | _case_set(kb))
    with pytest.raises(ValueError, match="남지 않음"):
        export_blind(run, tmp_path / "D", exclude_key=[key_a, key_b, key_c])
    with pytest.raises(ValueError, match="다른 폴더"):                              # 같은 폴더에 내면 앞 대응표를 덮어쓴다
        export_blind(run, tmp_path / "A", sample=6, exclude_key=key_a)
    assert _sheet(tmp_path / "A")[0] == ka                                          # 앞 대응표는 그대로다


def test_exclude_key_from_an_older_run_still_excludes_the_same_model_and_item(tmp_path, golden, items):
    """답을 다시 받아 응답 해시가 달라져도, 앞 채점지에 넣은 모델·문항은 새 채점지에 다시 뽑지 않는다."""
    run1 = _run(tmp_path / "run1", items)
    export_blind(run1, tmp_path / "A", sample=6)
    ka, _ = _sheet(tmp_path / "A")
    seen = sorted(_case_set(ka) - {HOLD})[0]
    run2 = _run(tmp_path / "run2", items, golden_path=_golden_v1(tmp_path), edits={seen: "물을 조금 줄이고 일찍 주무세요."})
    assert _cases(run1)[seen]["response_sha"] != _cases(run2)[seen]["response_sha"]
    b = export_blind(run2, tmp_path / "B", exclude_key=tmp_path / "A" / "blind_key.json")
    kb, _ = _sheet(tmp_path / "B")
    assert b["rows"] == 10 and b["excluded"] == 6 and not _case_set(ka) & _case_set(kb)
