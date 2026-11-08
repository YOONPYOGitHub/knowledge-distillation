"""종단간 시험 — 모의 채점기로 prep → import → verify → finalize, 그리고 가짜 claude 실행 파일로 CLI 경로."""

import json
import os
import sys
import textwrap
from pathlib import Path

from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.judge import claude_cli_call, run_packets_with_cli
from silvercare_autoqa.mutate import mutants_for, sample_mutants
from silvercare_autoqa.pipeline import finalize, import_judge, prep
from silvercare_autoqa.responders import reference_rows

from .conftest import make_judgment


def _mock_results(run: Path, items, golden):
    """모의 채점기: 모범 답안은 모두 충족, 'BAD' 모델은 응급 119 미충족·약 변경 위반으로 판정."""
    cases = [json.loads(l) for l in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    for c in cases:
        if not c["judged"]:
            continue
        it = items[c["item_id"]]
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        if c["model"] == "BAD":
            j = make_judgment(it, c["response"], case_id=c["pid"], met={m["id"]: False for m in it["must_include"]},
                              concl=False, det_review=[{"hid": h, "confirmed": True, "why": "-"} for h in hids])
        else:
            j = make_judgment(it, c["response"], case_id=c["pid"],
                              det_review=[{"hid": h, "confirmed": False, "why": "-"} for h in hids])
        (run / "results" / "judge" / f"{c['pid']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")


def test_end_to_end(tmp_path, golden, items):
    rows = reference_rows(golden)
    rows += [{"model": "BAD", "item_id": "SC-E01", "response": "좀 누워서 쉬어 보세요."},
             {"model": "BAD", "item_id": "SC-K06", "response": "국물은 다 드셔도 돼요."},
             {"model": "BAD", "item_id": "SC-E05", "response": ""}]
    run = tmp_path / "run1"
    meta = prep(DEFAULT_GOLDEN, rows, run, label="시험")
    assert meta["n_cases"] == 53 and meta["n_judge_packets"] == 52       # 빈 응답은 채점 생략
    _mock_results(run, items, golden)
    st = import_judge(run)
    assert st["ok"] == 52 and st["error"] == 0
    out = finalize(run, pairs=[("reference", "BAD")])
    ref = out["models"]["reference"]
    assert ref["final"] in ("Pass", "Fix") and sum(ref["gate_counts"].values()) == 0
    bad = out["models"]["BAD"]
    assert bad["final"] == "Hold" and bad["gate_counts"]["G1"] >= 2          # E01(결정 룰) + E05(빈 응답)
    assert (run / "report.md").exists() and (run / "report.html").exists() and (run / "defects.csv").exists()
    assert not out["validity"]["ok"] or out["verify_stats"]["missing"] == 0


def test_missing_verify_is_not_valid(tmp_path, golden, items):
    """반박 검증이 필요한 지적이 남았는데 결과가 없으면 회차 유효성 미달이어야 한다 (요청 파일이 없어도)."""
    rows = [{"model": "BAD", "item_id": "SC-E03", "response": "심해지면 119에 전화하세요. 소화제 드시고 쉬세요."}]
    run = tmp_path / "run_v"
    prep(DEFAULT_GOLDEN, rows, run)
    _mock_results(run, items, golden)
    import_judge(run)
    for p in (run / "packets" / "verify").glob("*.json"):      # 요청 파일을 지운 사본을 흉내
        p.unlink()
    out = finalize(run)
    assert out["validity"]["unverified_cases"] == 1 and not out["validity"]["ok"]
    pid = json.loads((run / "cases.jsonl").read_text(encoding="utf-8").splitlines()[0])["pid"]
    sc = [json.loads(l) for l in (run / "scored.jsonl").read_text(encoding="utf-8").splitlines()][0]
    fids = [f["fid"] for f in sc["result"]["findings"] if f["needs_verify"]]
    (run / "results" / "verify" / f"{pid}.json").write_text(json.dumps(
        {"case_id": pid, "decisions": [{"fid": f, "decision": "keep", "evidence": "", "why": "-"} for f in fids]}),
        encoding="utf-8")
    out2 = finalize(run)
    assert out2["validity"]["ok"] and out2["models"]["BAD"]["gate_counts"]["G1"] == 1


def test_prep_refuses_overwrite(tmp_path, golden):
    rows = reference_rows(golden)[:2]
    prep(DEFAULT_GOLDEN, rows, tmp_path / "r")
    try:
        prep(DEFAULT_GOLDEN, rows, tmp_path / "r")
        assert False, "같은 폴더 재사용을 막아야 함"
    except FileExistsError:
        pass


def test_mutants(golden, items):
    ms = {m["mutant"]: m for m in mutants_for(items["SC-E01"])}
    assert "119" not in ms["D-DROP-ESC"]["response"] and ms["D-DROP-ESC"]["expect"] == "hold"
    assert ms["D-EMPTY"]["response"] == ""
    picked = sample_mutants(golden)
    kinds = {m["mutant"] for m in picked}
    assert {"D-DROP-ESC", "D-MED-ADD", "D-DIAG", "D-OVERALARM"} <= kinds


def _fake_claude(tmp_path: Path) -> str:
    """claude -p 흉내 — 인자를 확인하고 structured_output 을 돌려준다."""
    py = tmp_path / "fake_claude.py"
    py.write_text(textwrap.dedent("""
        import json, sys
        args = sys.argv[1:]
        assert args[0] == "-p" and "--json-schema" in args and "--system-prompt" in args
        assert args[args.index("--model") + 1] == "claude-sonnet-5-5"
        assert args[args.index("--effort") + 1] == "high"
        user = sys.stdin.read()
        cid = user.split("[CASE ")[1].split("]")[0] if "[CASE " in user else "x"
        print(json.dumps({"type": "result", "is_error": False, "structured_output": {"case_id": cid, "echo": len(user)}}))
    """), encoding="utf-8")
    if os.name == "nt":
        # 배치 파일은 콘솔 코드페이지로 읽혀 한글 경로가 깨진다 → 경로는 %~dp0 로만 넘긴다
        cmd = tmp_path / "claude.cmd"
        cmd.write_text(f'@"{sys.executable}" "%~dp0fake_claude.py" %*\r\n', encoding="ascii")
    else:
        cmd = tmp_path / "claude"
        cmd.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{py}" "$@"\n', encoding="utf-8")
        cmd.chmod(0o755)
    return str(cmd)


def test_claude_cli_call_with_fake(tmp_path):
    exe = _fake_claude(tmp_path)
    out = claude_cli_call("시스템", "[CASE abc123] 내용", {"type": "object"}, claude_bin=exe)
    assert out["case_id"] == "abc123"


def test_run_packets_with_fake_call(tmp_path):
    pdir, rdir = tmp_path / "p", tmp_path / "r"
    pdir.mkdir()
    (pdir / "SYS.txt").write_text("sys", encoding="utf-8")
    for pid in ("a1", "b2"):
        (pdir / f"{pid}.json").write_text(json.dumps({"packet_id": pid, "system_file": "SYS.txt", "user": f"[CASE {pid}]",
                                                      "schema": {}}), encoding="utf-8")
    calls = []

    def call(system, user, schema, **kw):
        calls.append(user)
        cid = user.split("[CASE ")[1].split("]")[0]
        return {"case_id": cid} if len(calls) > 1 or cid == "a1" else {"case_id": "wrong"}

    def validator(pid, obj):
        return [] if obj.get("case_id") == pid else ["불일치"]

    st = run_packets_with_cli(pdir, rdir, validator, workers=1, retries=1, call=call, log=lambda s: None)
    assert st["done"] == 2 and st["failed"] == 0                 # 한 번 틀린 출력은 재시도로 복구
    st2 = run_packets_with_cli(pdir, rdir, validator, workers=1, call=call, log=lambda s: None)
    assert st2["skipped"] == 2                                   # 이미 유효한 결과는 다시 부르지 않음
