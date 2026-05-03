"""FastAPI 호출 래퍼"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, Optional

import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
_TIMEOUT = httpx.Timeout(300.0, connect=5.0)


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE_URL, timeout=_TIMEOUT)


def health() -> Optional[dict]:
    try:
        with _client() as c:
            r = c.get("/health")
            r.raise_for_status()
            return r.json()
    except Exception:  # noqa: BLE001
        return None


def list_runs() -> list[dict]:
    with _client() as c:
        r = c.get("/runs")
        r.raise_for_status()
        return r.json()


def get_run(run_id: str) -> dict:
    with _client() as c:
        r = c.get(f"/runs/{run_id}")
        r.raise_for_status()
        return r.json()


def load_models(run_id: str, model_ids: list[str]) -> dict:
    with _client() as c:
        r = c.post(
            "/models/load",
            json={"run_id": run_id, "models": model_ids},
        )
        r.raise_for_status()
        return r.json()


def generate(
    run_id: str,
    model_ids: list[str],
    prompt: str,
    params: dict[str, Any],
) -> dict:
    with _client() as c:
        r = c.post(
            "/generate",
            json={
                "run_id": run_id,
                "model_ids": model_ids,
                "prompt": prompt,
                "params": params,
            },
        )
        r.raise_for_status()
        return r.json()


def save_feedback(
    run_id: str, entries: list[dict], filename_prefix: str = "session"
) -> dict:
    with _client() as c:
        r = c.post(
            "/feedback/save",
            json={
                "run_id": run_id,
                "entries": entries,
                "filename_prefix": filename_prefix,
            },
        )
        r.raise_for_status()
        return r.json()


def list_feedback(run_id: str) -> list[dict]:
    with _client() as c:
        r = c.get(f"/feedback/{run_id}")
        r.raise_for_status()
        return r.json()


def read_feedback_entries(run_id: str, mode: str | None = None) -> list[dict]:
    params = {"mode": mode} if mode else None
    with _client() as c:
        r = c.get(f"/feedback/{run_id}/entries", params=params)
        r.raise_for_status()
        return r.json()


def get_history(run_id: str) -> dict:
    with _client() as c:
        r = c.get(f"/runs/{run_id}/history")
        r.raise_for_status()
        return r.json()


def get_evaluation(run_id: str) -> list[dict]:
    with _client() as c:
        r = c.get(f"/runs/{run_id}/evaluation")
        r.raise_for_status()
        return r.json()


def list_figures(run_id: str) -> list[dict]:
    with _client() as c:
        r = c.get(f"/runs/{run_id}/figures")
        r.raise_for_status()
        return r.json()


def figure_url(run_id: str, filename: str) -> str:
    return f"{API_BASE_URL}/runs/{run_id}/figures/{filename}"


def token_analysis(
    run_id: str,
    model_ids: list[str],
    prompt: str,
    top_k: int = 10,
    temperature: float = 1.0,
    teacher_id: str | None = "teacher_ft",
) -> dict:
    with _client() as c:
        r = c.post("/token-analysis", json={
            "run_id": run_id,
            "model_ids": model_ids,
            "prompt": prompt,
            "top_k": top_k,
            "temperature": temperature,
            "teacher_id": teacher_id,
        })
        r.raise_for_status()
        return r.json()


def generate_stream(
    run_id: str,
    model_ids: list[str],
    prompt: str,
    params: dict[str, Any],
) -> Iterator[dict]:
    """SSE 이벤트를 순차 반환. yield: dict (type, model_id, ...)"""
    payload = {
        "run_id": run_id,
        "model_ids": model_ids,
        "prompt": prompt,
        "params": params,
    }
    with httpx.stream(
        "POST", f"{API_BASE_URL}/generate/stream",
        json=payload, timeout=_TIMEOUT,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue


def generate_batch(
    run_id: str,
    model_ids: list[str],
    prompts: list[str],
    params: dict[str, Any],
) -> dict:
    with _client() as c:
        r = c.post("/generate/batch", json={
            "run_id": run_id,
            "model_ids": model_ids,
            "prompts": prompts,
            "params": params,
        })
        r.raise_for_status()
        return r.json()
