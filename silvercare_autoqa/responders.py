"""응답 생성기 — 골든셋 발화를 모델에 넣고 응답 JSONL 을 만든다. 채점과 분리해 응답을 먼저 동결한다.

- kd-ui  : 팀 UI 백엔드 POST /generate/batch (teacher_ft / student_kd / student_ft / student_base 한 번에)
- openai : llama.cpp server · vLLM 같은 OpenAI 호환 /v1/chat/completions (양자화 GGUF 실측용)
- claude : 이 PC 의 Claude Code (Teacher 대용 기준 응답을 만들 때)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from .golden import DATA_DIR

APP_SYSTEM = (DATA_DIR / "app_system_prompt.txt").read_text(encoding="utf-8").strip()

RAW_TEMPLATE = "{system}\n\n{context}어르신: {utterance}\n돌봄 AI:"


def user_turn(item: dict) -> str:
    ctx = f"[기기 기록] {item['context']}\n" if item.get("context") else ""
    return f"{ctx}{item['utterance']}"


def raw_prompt(item: dict, system: str = APP_SYSTEM, template: str = RAW_TEMPLATE) -> str:
    ctx = f"[기기 기록] {item['context']}\n" if item.get("context") else ""
    return template.format(system=system, context=ctx, utterance=item["utterance"])


def _clean(text: str) -> str:
    """생성 모델이 다음 차례까지 이어 쓴 꼬리를 자른다 (base 모델 대비)."""
    for stop in ("\n어르신:", "\n사용자:", "\nUser:", "<|im_end|>", "<end_of_turn>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def kd_ui(items: Iterable[dict], base_url: str, run_id: str, model_ids: list[str], *, max_new_tokens: int = 160,
          temperature: float = 0.0, chunk: int = 50, timeout: int = 1800) -> list[dict]:
    import requests

    items = list(items)
    rows = []
    params = {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": 1.0, "top_k": 0,
              "repetition_penalty": 1.1, "do_sample": temperature > 0, "seed": 42}
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        body = {"run_id": run_id, "model_ids": model_ids, "prompts": [raw_prompt(it) for it in part], "params": params}
        r = requests.post(base_url.rstrip("/") + "/generate/batch", json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        for it, res in zip(part, data["items"]):
            for mid, o in res["outputs"].items():
                rows.append({"model": mid, "item_id": it["id"], "response": _clean(o.get("text") or ""),
                             "meta": {"source": "kd-ui", "run_id": run_id, "num_tokens": o.get("num_tokens"),
                                      "elapsed_ms": o.get("elapsed_ms"), "error": o.get("error"), "params": params}})
    return rows


def openai_compat(items: Iterable[dict], base_url: str, model: str, label: str, *, max_tokens: int = 160,
                  temperature: float = 0.0, api_key: str = "none", timeout: int = 300) -> list[dict]:
    import requests

    rows = []
    for it in items:
        body = {"model": model, "messages": [{"role": "system", "content": APP_SYSTEM},
                                             {"role": "user", "content": user_turn(it)}],
                "max_tokens": max_tokens, "temperature": temperature}
        t0 = time.time()
        r = requests.post(base_url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {api_key}"})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        rows.append({"model": label, "item_id": it["id"], "response": _clean(text),
                     "meta": {"source": "openai", "model": model, "elapsed_ms": (time.time() - t0) * 1000}})
    return rows


def claude_teacher(items: Iterable[dict], label: str = "teacher_claude", *, model: str = "claude-sonnet-5-5",
                   effort: str = "high", claude_bin: str | None = None) -> list[dict]:
    import subprocess

    from .judge import find_claude

    rows = []
    for it in items:
        cmd = [find_claude(claude_bin), "-p", "--model", model, "--effort", effort, "--output-format", "json",
               "--tools", "", "--no-session-persistence", "--system-prompt", APP_SYSTEM]
        proc = subprocess.run(cmd, input=user_turn(it), capture_output=True, text=True, encoding="utf-8", timeout=600)
        data = json.loads(proc.stdout)
        if data.get("is_error"):
            raise RuntimeError(str(data.get("result"))[:200])
        rows.append({"model": label, "item_id": it["id"], "response": (data.get("result") or "").strip(),
                     "meta": {"source": "claude-cli", "model": model, "effort": effort}})
    return rows


def reference_rows(golden: dict, label: str = "reference") -> list[dict]:
    """골든셋 모범 답안을 응답처럼 넣는다 — 채점기가 좋은 답을 통과시키는지 보는 기준선."""
    return [{"model": label, "item_id": it["id"], "response": it["reference_answer"], "meta": {"source": "golden"}}
            for it in golden["items"]]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
