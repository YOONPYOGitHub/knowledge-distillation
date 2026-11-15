"""같은 답을 두 번 채점해 나쁜 쪽 판정을 쓴다 (쇼핑 AI 검수 노하우 K45 — 채점기 재현성).

왜: 2026-11-14 에 같은 답 30개를 다시 채점하니 4개(13%)의 판정이 바뀌었다. 한 번 채점한 숫자로 두 시스템을
견주면 채점기의 흔들림이 시스템 차이처럼 보인다.

어떻게:
1. 같은 응답 파일로 회차를 두 번 만든다 (prep → judge → import → verify → finalize). 기준과 답이 같으면
   채점 요청 번호도 같다.
2. `merge-rejudge` 가 문항마다 더 나쁜 판정을 낸 쪽의 채점 결과를 골라 새 회차 폴더를 만든다.
3. 새 회차는 보통 회차와 같다. finalize·compare 를 그대로 돌린다. 리포트에 "두 번 채점" 줄이 붙는다.

나쁜 쪽을 쓰는 이유: 위해 답을 놓치는 쪽이 더 큰 잘못이라서다 (한 번이라도 Hold 가 나온 답은 Hold 로 둔다).
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RANK = {"pass": 0, "fix": 1, "hold": 2}
NAME = "stability.json"


def _jl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _rank(case: dict, basis: str) -> int | None:
    """판정 순위. 채점 오류로 판정이 없으면 None."""
    r = case.get("result") or {}
    if r.get("errors"):
        return None
    return RANK.get(r.get(basis))


def pick_worse(a: dict, b: dict) -> str:
    """두 채점 중 어느 쪽을 쓸지 — 'a' 또는 'b'.

    견줄 때 쓰는 공통 기준 판정이 더 나쁜 쪽, 같으면 전체 판정이 더 나쁜 쪽, 그것도 같으면 a.
    한쪽만 채점 오류면 채점된 쪽을 쓴다 (오류를 판정으로 세지 않는다)."""
    ka = (_rank(a, "verdict_common"), _rank(a, "verdict"))
    kb = (_rank(b, "verdict_common"), _rank(b, "verdict"))
    if None in kb:
        return "a"
    if None in ka:
        return "b"
    return "b" if kb > ka else "a"


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    """두 회차가 같은 기준·같은 답인지 확인하고, 문항마다 두 판정과 고른 쪽을 돌려준다."""
    A, B = Path(run_a), Path(run_b)
    for r in (A, B):
        if not (r / "scored.jsonl").exists():
            raise ValueError(f"{r} 에 scored.jsonl 이 없음 — 두 회차 모두 finalize 까지 끝내야 한다")
    ma = json.loads((A / "meta.json").read_text(encoding="utf-8"))
    mb = json.loads((B / "meta.json").read_text(encoding="utf-8"))
    if ma.get("fingerprint") != mb.get("fingerprint"):
        raise ValueError(f"두 회차의 평가기준 지문이 다름 ({ma.get('fingerprint')} / {mb.get('fingerprint')}) — "
                         "기준이 다르면 같은 답의 재채점이 아니다")
    ca = {c["pid"]: c for c in _jl(A / "scored.jsonl")}
    cb = {c["pid"]: c for c in _jl(B / "scored.jsonl")}
    if set(ca) != set(cb):
        only_a, only_b = len(set(ca) - set(cb)), len(set(cb) - set(ca))
        raise ValueError(f"두 회차의 채점 요청이 다름 (한쪽에만 있는 것 {only_a} / {only_b}) — 응답 파일이 같아야 한다")
    # 복사한 채점을 두 번 채점으로 세지 않는다 — 따로 부른 채점이면 요약 글이 한 글자도 안 다를 수 없다
    ja, jb = A / "results" / "judge", B / "results" / "judge"
    files = [pid for pid in ca if (ja / f"{pid}.json").exists() and (jb / f"{pid}.json").exists()]
    same = sum((ja / f"{pid}.json").read_bytes() == (jb / f"{pid}.json").read_bytes() for pid in files)
    if files and same == len(files):
        raise ValueError(f"두 회차의 채점 결과 파일 {same}개가 글자까지 같다 — 한 번 채점한 결과를 복사한 것은 두 번 채점이 아니다. "
                         "같은 응답으로 judge 를 따로 돌린 회차를 준다")
    rows = []
    for pid, a in ca.items():
        b = cb[pid]
        src = pick_worse(a, b)
        va, vb = (a.get("result") or {}), (b.get("result") or {})
        rows.append({"pid": pid, "model": a["model"], "item_id": a["item_id"],
                     "a": {"verdict": va.get("verdict"), "verdict_common": va.get("verdict_common"), "error": bool(va.get("errors"))},
                     "b": {"verdict": vb.get("verdict"), "verdict_common": vb.get("verdict_common"), "error": bool(vb.get("errors"))},
                     "picked": src,
                     "flipped": va.get("verdict") != vb.get("verdict"),
                     "flipped_common": va.get("verdict_common") != vb.get("verdict_common"),
                     "verify_id": (a if src == "a" else b).get("verify_id")})
    return {"meta_a": ma, "meta_b": mb, "rows": rows, "identical_results": same, "n_result_files": len(files)}


def summarize(rows: list[dict]) -> dict[str, Any]:
    """모델마다: 답 수, 판정이 갈린 답 수(전체·공통 기준), 어떻게 갈렸는지."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    out = {}
    for m, rs in sorted(by.items()):
        moves = Counter("↔".join(sorted([r["a"]["verdict_common"] or "?", r["b"]["verdict_common"] or "?"], key=lambda v: RANK.get(v, 9)))
                        for r in rs if r["flipped_common"])
        out[m] = {"n": len(rs), "flipped": sum(r["flipped"] for r in rs), "flipped_common": sum(r["flipped_common"] for r in rs),
                  "moves_common": dict(moves),
                  "items_flipped_common": sorted(r["item_id"] for r in rs if r["flipped_common"]),
                  "one_side_error": sum(r["a"]["error"] != r["b"]["error"] for r in rs)}
    n = len(rows)
    return {"n": n, "flipped": sum(r["flipped"] for r in rows), "flipped_common": sum(r["flipped_common"] for r in rows),
            "one_side_error": sum(r["a"]["error"] != r["b"]["error"] for r in rows), "by_model": out}


def merge_worst(run_a: str | Path, run_b: str | Path, out_dir: str | Path, *, label: str | None = None) -> dict[str, Any]:
    """두 채점 회차에서 문항마다 나쁜 쪽 채점 결과를 골라 새 회차 폴더를 만든다 (import 까지 끝낸 상태).

    돌려주는 값은 흔들림 요약이다. 이어서 finalize 를 돌리면 판정·지표·리포트가 나온다."""
    from .pipeline import import_judge

    A, B, OUT = Path(run_a), Path(run_b), Path(out_dir)
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError(f"{OUT} 에 이미 내용이 있음 — 새 폴더를 쓴다 (회차 폴더는 덮어쓰지 않는다)")
    cmp = compare_runs(A, B)
    rows, ma, mb = cmp["rows"], cmp["meta_a"], cmp["meta_b"]
    (OUT / "results" / "judge").mkdir(parents=True)
    (OUT / "results" / "verify").mkdir(parents=True)
    shutil.copytree(A / "packets" / "judge", OUT / "packets" / "judge")
    for name in ("golden.yaml", "cases.jsonl"):
        shutil.copy2(A / name, OUT / name)
    src = {"a": A, "b": B}
    for r in rows:
        jr = src[r["picked"]] / "results" / "judge" / f"{r['pid']}.json"
        if jr.exists():
            shutil.copy2(jr, OUT / "results" / "judge" / jr.name)
        if r["verify_id"]:
            vr = src[r["picked"]] / "results" / "verify" / f"{r['verify_id']}.json"
            if vr.exists():
                shutil.copy2(vr, OUT / "results" / "verify" / vr.name)
    summary = summarize(rows)
    meta = dict(ma)
    for k in ("import_stats", "rescored"):
        meta.pop(k, None)
    meta["label"] = label or f"{ma.get('label') or ma.get('created')}+재채점"
    meta["rejudge"] = {"policy": "worst_of_2", "runs": [ma.get("label") or str(A), mb.get("label") or str(B)], **summary}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / NAME).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    import_judge(OUT)
    return summary


def report_line(meta: dict) -> str | None:
    """리포트 첫 화면에 붙일 한 줄 (report.header_lines 가 부른다)."""
    rj = meta.get("rejudge")
    if not rj:
        return None
    err = f" 한 번만 채점된 답 {rj['one_side_error']}개(다른 한 번은 채점 오류)." if rj.get("one_side_error") else ""
    return (f"같은 답을 두 번 채점해 문항마다 나쁜 쪽 판정을 썼다. 두 번의 판정이 갈린 답 {rj['flipped_common']}/{rj['n']} "
            f"(출처 지적까지 넣으면 {rj['flipped']}/{rj['n']}).{err} 한 번 채점한 회차와는 숫자를 견주지 않는다.")


# ───────────────────────── 두 회차 견주기 — 차이의 원인부터 나눈다 (노하우 K32) ─────────────────────────
KINDS = {"judge": "채점기 흔들림", "criteria": "기준 바뀜", "evidence": "채점기에 준 근거 자료가 바뀜", "answer": "답 바뀜",
         "evidence_criteria": "근거 자료와 기준이 함께 바뀜 (답 글은 그대로)", "both": "답과 기준이 함께 바뀜"}


def _text_sha(case: dict) -> str:
    import hashlib
    return hashlib.sha256(str(case.get("response", "")).encode("utf-8")).hexdigest()[:16]


def diff_runs(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    """두 회차의 판정 차이를 원인별로 나눈다. A = 앞 회차, B = 뒤 회차.

    2026-11-15 에 손으로 견주다 "채점기가 하루 새 달라졌다"고 잘못 적었다 (다른 모델 답을 섞어 묶음).
    숫자 차이를 말하기 전에 (모델·문항)마다 답이 같은지, 채점 기준이 같은지부터 본다.
    - 답 같음 · 기준 같음 → 채점기 흔들림 (시스템이 달라진 것이 아니다)
    - 답 같음 · 기준 다름 → 기준 바뀜
    - 답 다름 · 기준 같음 → 답 바뀜 (온담을 고친 효과는 여기서만 읽는다)
    - 답 다름 · 기준 다름 → 원인을 나눌 수 없다. 좋아졌다·나빠졌다를 말하지 않는다
    """
    A, B = Path(run_a), Path(run_b)
    ma = json.loads((A / "meta.json").read_text(encoding="utf-8"))
    mb = json.loads((B / "meta.json").read_text(encoding="utf-8"))
    same_judging = ma.get("judging_fp") == mb.get("judging_fp") and ma.get("judging_fp") is not None
    same_scoring = ma.get("scoring_fp") == mb.get("scoring_fp")
    same_criteria = same_judging and same_scoring
    pa, pb = ma.get("fingerprint_parts") or {}, mb.get("fingerprint_parts") or {}
    changed_parts = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
    ca = {(c["model"], c["item_id"]): c for c in _jl(A / "scored.jsonl")}
    cb = {(c["model"], c["item_id"]): c for c in _jl(B / "scored.jsonl")}
    rows = []
    for key in sorted(set(ca) & set(cb)):
        a, b = ca[key], cb[key]
        same_text = _text_sha(a) == _text_sha(b)
        same_all = a.get("response_sha") == b.get("response_sha")      # 답 글 + 채점기에 준 근거 자료
        if same_text and same_all:
            kind = "judge" if same_criteria else "criteria"
        elif same_text:
            kind = "evidence" if same_criteria else "evidence_criteria"   # 답 글은 같고 근거 자료(발췌 길이 등)가 다름
        else:
            kind = "answer" if same_criteria else "both"
        va, vb = (a.get("result") or {}), (b.get("result") or {})
        rows.append({"model": key[0], "item_id": key[1], "kind": kind,
                     "a": va.get("verdict_common"), "b": vb.get("verdict_common"),
                     "a_full": va.get("verdict"), "b_full": vb.get("verdict"),
                     "changed": va.get("verdict_common") != vb.get("verdict_common")})
    by_model: dict[str, dict] = {}
    for m in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == m]
        kinds = {}
        for k in KINDS:
            ks = [r for r in rs if r["kind"] == k]
            if not ks:
                continue
            better = [r for r in ks if r["changed"] and RANK.get(r["b"], 9) < RANK.get(r["a"], 9)]
            worse = [r for r in ks if r["changed"] and RANK.get(r["b"], 9) > RANK.get(r["a"], 9)]
            kinds[k] = {"n": len(ks), "changed": sum(r["changed"] for r in ks),
                        "better": sorted(r["item_id"] for r in better), "worse": sorted(r["item_id"] for r in worse),
                        "new_hold": sorted(r["item_id"] for r in ks if r["b"] == "hold" and r["a"] != "hold"),
                        "cleared_hold": sorted(r["item_id"] for r in ks if r["a"] == "hold" and r["b"] != "hold")}
        n = len(rs)
        by_model[m] = {"n_common": n,
                       "pass_a": sum(r["a"] == "pass" for r in rs), "pass_b": sum(r["b"] == "pass" for r in rs),
                       "hold_a": sum(r["a"] == "hold" for r in rs), "hold_b": sum(r["b"] == "hold" for r in rs),
                       "kinds": kinds}
    return {"a": ma.get("label") or str(A), "b": mb.get("label") or str(B),
            "same_judging": same_judging, "same_scoring": same_scoring, "changed_parts": changed_parts,
            "fingerprints": {"a": ma.get("fingerprint"), "b": mb.get("fingerprint")},
            "only_in_a": sorted(f"{m}|{i}" for m, i in set(ca) - set(cb)),
            "only_in_b": sorted(f"{m}|{i}" for m, i in set(cb) - set(ca)),
            "passes": {"a": ma.get("rejudge", {}).get("policy", "single"), "b": mb.get("rejudge", {}).get("policy", "single")},
            "by_model": by_model, "rows": rows}


def render_diff(d: dict) -> str:
    """diff_runs 결과를 읽는 글로."""
    L = [f"# 두 회차 견주기 — {d['a']} → {d['b']}", "", "## 한눈에", ""]
    if d["same_judging"] and d["same_scoring"]:
        L.append(f"- 채점 기준이 같다 (지문 `{d['fingerprints']['a']}`). 판정 차이는 답이 바뀐 문항과 채점기 흔들림으로만 나뉜다.")
    else:
        L.append(f"- ⚠ 채점 기준이 다르다 (지문 `{d['fingerprints']['a']}` → `{d['fingerprints']['b']}`, 달라진 것: "
                 f"{', '.join(d['changed_parts']) or '?'}). 답이 그대로인 문항의 판정 차이에는 바뀐 기준과 채점기 흔들림이 섞여 있다. "
                 f"바뀐 문항을 하나씩 읽어 가른다.")
    if d["passes"]["a"] != d["passes"]["b"]:
        L.append("- ⚠ 한쪽은 한 번 채점, 다른 쪽은 두 번 채점해 나쁜 쪽을 쓴 회차다. 두 번 채점한 쪽이 늘 더 낮게 나온다.")
    if d["only_in_a"] or d["only_in_b"]:
        L.append(f"- 한쪽에만 있는 답: 앞 회차 {len(d['only_in_a'])}개 · 뒤 회차 {len(d['only_in_b'])}개 (아래 수치에서 뺐다).")
    for m, v in d["by_model"].items():
        L += ["", f"## {m} — 두 회차에 다 있는 문항 {v['n_common']}개", "",
              f"Pass {v['pass_a']} → {v['pass_b']} · Hold {v['hold_a']} → {v['hold_b']} (출처 지적을 뺀 공통 기준)", "",
              "| 원인 | 문항 | 판정이 바뀐 문항 | 좋아짐 | 나빠짐 | 새 Hold | 풀린 Hold |", "|---|---|---|---|---|---|---|"]
        for k, name in KINDS.items():
            s = v["kinds"].get(k)
            if s:
                L.append(f"| {name} | {s['n']} | {s['changed']} | {len(s['better'])} | {len(s['worse'])} | "
                         f"{', '.join(s['new_hold']) or '—'} | {', '.join(s['cleared_hold']) or '—'} |")
        if "both" in v["kinds"] or "evidence_criteria" in v["kinds"]:
            L += ["", "둘이 함께 바뀐 문항은 원인을 나눌 수 없다. 좋아졌다고도 나빠졌다고도 쓰지 않는다."]
        if "answer" in v["kinds"]:
            s = v["kinds"]["answer"]
            L += ["", f"답이 바뀌어 좋아진 문항: {', '.join(s['better']) or '없음'}", "",
                  f"답이 바뀌어 나빠진 문항: {', '.join(s['worse']) or '없음'}"]
    L += ["", "채점기는 같은 답도 다시 채점하면 판정이 일부 바뀐다. 「답 바뀜」의 좋아짐·나빠짐 수가 「채점기 흔들림」에서 "
          "바뀐 문항 수보다 작으면 효과라고 쓰지 않는다.", "",
          "좋아진 문항과 풀린 Hold 는 「이번 회차에 다시 안 나옴」이다. 「고쳐졌다」고 쓰려면 무엇을 고쳤는지(지식 자료·지시문·응급 문구)와 "
          "그 문항을 다시 확인한 기록을 함께 적는다."]
    return "\n".join(L) + "\n"
