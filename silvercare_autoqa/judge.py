"""채점기 실행·검증.

실행 방식은 둘이다.
- claude-cli : 이 PC 의 Claude Code 를 `claude -p` 로 직접 부른다 (기본 Sonnet 5.5 · effort high).
- packets    : 채점 요청을 파일(packet)로 떨궈 두고, 다른 실행기(사람·다른 에이전트·서버)가 결과 파일을 채우면 읽어 들인다.
두 방식 모두 결과는 validate_judgment 한 곳을 지난다. 형식이 하나라도 틀리면 그 케이스는 judge_error 로
분모에서 빠지고 리포트에 따로 나온다 (fail-closed — 부분 응답을 만점으로 채우지 않는다).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .prompts import ACTION_ENUM, judge_schema, verify_schema, schema_json
from .textutil import grounded

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "high"

EXTRA_CODES = set(judge_schema()["properties"]["extra_violations"]["items"]["properties"]["code"]["enum"])


# ───────────────────────────── 검증 ─────────────────────────────

def _ids(entries: list[dict]) -> list[str]:
    return [e.get("id") for e in entries]


def validate_judgment(obj: Any, case_id: str, item: dict, response: str, hybrid_hids: list[str]) -> list[str]:
    """채점 결과 형식·근거 검사. 오류 문장 목록 (비면 통과)."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["결과가 JSON 객체가 아님"]
    if obj.get("case_id") != case_id:
        errs.append(f"case_id 불일치 ({obj.get('case_id')} != {case_id})")

    def check_list(key: str, flag: str, expected: list[str]):
        got = obj.get(key)
        if not isinstance(got, list):
            errs.append(f"{key} 가 목록이 아님")
            return
        got_ids = [g.get("id") if isinstance(g, dict) else None for g in got]
        if sorted(x for x in got_ids if x) != sorted(expected) or len(got_ids) != len(expected):
            errs.append(f"{key} id 집합 불일치 (기대 {sorted(expected)}, 받음 {got_ids})")
        for g in got:
            if not isinstance(g, dict):
                continue
            if not isinstance(g.get(flag), bool):
                errs.append(f"{key} {g.get('id')} 의 {flag} 가 boolean 이 아님")
                continue
            ev = g.get("evidence")
            if not isinstance(ev, str):
                errs.append(f"{key} {g.get('id')} evidence 가 문자열이 아님")
                continue
            if g[flag] and not grounded(ev, response):
                errs.append(f"{key} {g.get('id')} 근거 인용이 응답 원문에 없음: \"{ev[:40]}\"")

    check_list("must_include", "met", _ids(item.get("must_include") or []))
    check_list("must_not", "violated", _ids(item.get("must_not") or []))
    check_list("key_facts", "preserved", _ids(item.get("key_facts") or []))

    if obj.get("response_action") not in ACTION_ENUM:
        errs.append(f"response_action '{obj.get('response_action')}' 허용값 아님")
    if obj.get("urgency_vs_expected") not in ("match", "over", "under"):
        errs.append("urgency_vs_expected 허용값 아님")
    for b in ("conclusion_match", "referral_present", "core_in_first3"):
        if not isinstance(obj.get(b), bool):
            errs.append(f"{b} 가 boolean 이 아님")
    a = obj.get("actionability")
    if not (isinstance(a, int) and not isinstance(a, bool) and 1 <= a <= 5):
        errs.append("actionability 는 1~5 정수")

    ev_list = obj.get("extra_violations")
    if not isinstance(ev_list, list):
        errs.append("extra_violations 가 목록이 아님")
    else:
        for i, v in enumerate(ev_list):
            if not isinstance(v, dict):
                errs.append(f"extra_violations[{i}] 가 객체가 아님")
                continue
            if v.get("code") not in EXTRA_CODES:
                errs.append(f"extra_violations[{i}] code 허용값 아님")
            if v.get("severity") not in ("critical", "major", "minor"):
                errs.append(f"extra_violations[{i}] severity 허용값 아님")
            if not grounded(v.get("evidence") or "", response):
                errs.append(f"extra_violations[{i}] 근거 인용이 응답 원문에 없음")

    dr = obj.get("det_review")
    if not isinstance(dr, list):
        errs.append("det_review 가 목록이 아님")
    else:
        got = [d.get("hid") for d in dr if isinstance(d, dict)]
        if sorted(got) != sorted(hybrid_hids):
            errs.append(f"det_review 후보 집합 불일치 (기대 {sorted(hybrid_hids)}, 받음 {got})")
        for d in dr:
            if isinstance(d, dict) and not isinstance(d.get("confirmed"), bool):
                errs.append(f"det_review {d.get('hid')} confirmed 가 boolean 이 아님")
    if not isinstance(obj.get("summary"), str):
        errs.append("summary 가 문자열이 아님")
    return errs


def validate_verify(obj: Any, case_id: str, fids: list[str]) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["결과가 JSON 객체가 아님"]
    if obj.get("case_id") != case_id:
        errs.append(f"case_id 불일치 ({obj.get('case_id')} != {case_id})")
    ds = obj.get("decisions")
    if not isinstance(ds, list):
        return errs + ["decisions 가 목록이 아님"]
    got = [d.get("fid") for d in ds if isinstance(d, dict)]
    if sorted(got) != sorted(fids):
        errs.append(f"fid 집합 불일치 (기대 {sorted(fids)}, 받음 {got})")
    for d in ds:
        if isinstance(d, dict) and d.get("decision") not in ("keep", "refute", "downgrade"):
            errs.append(f"{d.get('fid')} decision 허용값 아님")
    return errs


# ───────────────────────────── 실행기 ─────────────────────────────

def find_claude(explicit: str | None = None) -> str:
    """claude 실행 파일 위치. 환경변수 SILVERCARE_CLAUDE_BIN > 인자 > PATH."""
    cand = os.environ.get("SILVERCARE_CLAUDE_BIN") or explicit or shutil.which("claude")
    if not cand:
        raise FileNotFoundError("claude 실행 파일을 찾지 못함 — Claude Code 설치·로그인 후 다시 실행")
    return cand


def _parse_cli_stdout(stdout: str) -> Any:
    data = json.loads(stdout)
    if isinstance(data, dict) and data.get("type") == "result":
        if data.get("is_error"):
            raise RuntimeError(f"claude 오류: {str(data.get('result'))[:200]}")
        if data.get("structured_output") is not None:
            return data["structured_output"]
        res = data.get("result")
        if isinstance(res, str):
            res = res.strip()
            if res.startswith("```"):
                res = res.strip("`")
                res = res[res.find("{"):]
            return json.loads(res[res.find("{"): res.rfind("}") + 1])
    return data


def claude_cli_call(system: str, user: str, schema: dict, *, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
                    claude_bin: str | None = None, timeout: int = 900, cwd: str | None = None) -> Any:
    """claude -p 한 번 호출 → 구조화 출력(dict). 도구는 모두 끈다."""
    cmd = [find_claude(claude_bin), "-p", "--model", model, "--effort", effort,
           "--output-format", "json", "--json-schema", schema_json(schema),
           "--tools", "", "--no-session-persistence", "--system-prompt", system]
    proc = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                          timeout=timeout, cwd=cwd)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"claude 종료코드 {proc.returncode}: {proc.stderr[:300]}")
    return _parse_cli_stdout(proc.stdout)


def run_packets_with_cli(packet_dir: Path, result_dir: Path, validator: Callable[[str, Any], list[str]], *,
                         model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, workers: int = 3,
                         retries: int = 1, claude_bin: str | None = None, log: Callable[[str], None] = print,
                         call: Callable[..., Any] = claude_cli_call) -> dict[str, int]:
    """packet 폴더의 요청을 claude -p 로 처리해 result 폴더에 쓴다. 이미 유효한 결과가 있으면 건너뛴다."""
    result_dir.mkdir(parents=True, exist_ok=True)
    work_dir = packet_dir.parent / "_cli_cwd"          # CLAUDE.md 같은 프로젝트 지침을 읽지 않게 빈 폴더에서 실행
    work_dir.mkdir(exist_ok=True)
    packets = sorted(packet_dir.glob("*.json"))
    stats = {"done": 0, "skipped": 0, "failed": 0}

    def one(p: Path) -> tuple[str, str]:
        pk = json.loads(p.read_text(encoding="utf-8"))
        out = result_dir / p.name
        if out.exists():
            try:
                if not validator(pk["packet_id"], json.loads(out.read_text(encoding="utf-8"))):
                    return pk["packet_id"], "skipped"
            except Exception:
                pass
        system = (packet_dir / pk["system_file"]).read_text(encoding="utf-8")
        user = pk["user"]
        last_err = ""
        for attempt in range(retries + 1):
            try:
                obj = call(system, user, pk["schema"], model=model, effort=effort, claude_bin=claude_bin,
                           cwd=str(work_dir))
                errs = validator(pk["packet_id"], obj)
                if not errs:
                    out.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
                    return pk["packet_id"], "done"
                last_err = "; ".join(errs[:3])
                user = pk["user"] + "\n\n[직전 출력이 검증에 실패했다 — 고쳐서 다시 낸다]\n" + last_err
            except Exception as e:  # noqa: BLE001 — 실패는 기록하고 다음 시도
                last_err = str(e)[:300]
                time.sleep(2)
        (result_dir / (p.stem + ".error.txt")).write_text(last_err, encoding="utf-8")
        return pk["packet_id"], "failed"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, p) for p in packets]
        for i, f in enumerate(as_completed(futs), 1):
            pid, status = f.result()
            stats[status] += 1
            if i % 10 == 0 or i == len(futs):
                log(f"  채점 {i}/{len(futs)} — 완료 {stats['done']} · 건너뜀 {stats['skipped']} · 실패 {stats['failed']}")
    return stats


def judge_validator_factory(cases: dict[str, dict], items: dict[str, dict]) -> Callable[[str, Any], list[str]]:
    def v(pid: str, obj: Any) -> list[str]:
        c = cases[pid]
        hids = [h["hid"] for h in c["det"]["hits"] if h["kind"] == "hybrid"]
        return validate_judgment(obj, pid, items[c["item_id"]], c["response"], hids)
    return v


def verify_validator_factory(vindex: dict[str, list[str]]) -> Callable[[str, Any], list[str]]:
    def v(pid: str, obj: Any) -> list[str]:
        return validate_verify(obj, pid, vindex[pid])
    return v
