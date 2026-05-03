"""SSE 기반 멀티모델 스트리밍 생성.

각 모델을 별도 스레드에서 `TextIteratorStreamer` 로 돌리고,
모든 스레드의 토큰을 공용 큐에 던져 한 개의 SSE 스트림으로 병합한다.

이벤트 포맷 (JSON line, `data: {...}\\n\\n`):
- {"type":"start", "model_id":..., "prompt_tokens":N}
- {"type":"token", "model_id":..., "text":"...", "index":i}
- {"type":"done",  "model_id":..., "num_tokens":N, "elapsed_ms":..., "tokens_per_sec":...}
- {"type":"error", "model_id":..., "error":"..."}
- {"type":"all_done"}
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Iterator

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer

from ui.backend.model_registry import get_registry
from ui.backend.schemas import GenerateRequest

router = APIRouter()


_SENTINEL = object()


def _stream_one(
    model, tokenizer, prompt: str, params, device: str,
    model_id: str, out_q: "queue.Queue",
) -> None:
    """단일 모델을 스트리밍으로 돌려 out_q 에 이벤트 push"""
    try:
        if params.seed is not None:
            torch.manual_seed(params.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(params.seed)

        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = int(enc["input_ids"].shape[1])

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            repetition_penalty=params.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
        )
        if params.do_sample:
            gen_kwargs["temperature"] = max(params.temperature, 1e-5)
            gen_kwargs["top_p"] = params.top_p
            if params.top_k > 0:
                gen_kwargs["top_k"] = params.top_k

        out_q.put({
            "type": "start",
            "model_id": model_id,
            "prompt_tokens": input_len,
        })

        t0 = time.time()

        def _run():
            with torch.no_grad():
                model.generate(**gen_kwargs)

        th = threading.Thread(target=_run, daemon=True)
        th.start()

        idx = 0
        for chunk in streamer:
            if not chunk:
                continue
            out_q.put({
                "type": "token",
                "model_id": model_id,
                "text": chunk,
                "index": idx,
            })
            idx += 1

        th.join()
        elapsed = time.time() - t0
        # streamer chunk 수를 토큰 수 근사로 사용 (문자 청크 단위라 실제와 소폭 차이)
        out_q.put({
            "type": "done",
            "model_id": model_id,
            "num_tokens": idx,
            "elapsed_ms": elapsed * 1000.0,
            "tokens_per_sec": idx / elapsed if elapsed > 0 else 0.0,
        })
    except Exception as e:  # noqa: BLE001
        out_q.put({
            "type": "error",
            "model_id": model_id,
            "error": f"{type(e).__name__}: {e}",
        })


def _event_stream(req: GenerateRequest) -> Iterator[str]:
    registry = get_registry()
    run_models = registry.ensure_loaded(req.run_id, req.model_ids)
    device = registry.device

    out_q: queue.Queue = queue.Queue()
    threads: list[threading.Thread] = []

    for mid in req.model_ids:
        model = run_models.get(mid)
        if model is None:
            yield f"data: {json.dumps({'type':'error','model_id':mid,'error':'model not loaded'})}\n\n"
            continue
        th = threading.Thread(
            target=_stream_one,
            args=(model, run_models.tokenizer, req.prompt, req.params,
                  device, mid, out_q),
            daemon=True,
        )
        th.start()
        threads.append(th)

    # 종료 조건: 각 모델에서 done/error 가 한 번씩 들어오면 끝
    pending = len(threads)
    done_models: set[str] = set()

    # 종료 감시 스레드
    def _watch():
        for th in threads:
            th.join()
        out_q.put(_SENTINEL)

    threading.Thread(target=_watch, daemon=True).start()

    while True:
        item = out_q.get()
        if item is _SENTINEL:
            break
        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        if item.get("type") in {"done", "error"}:
            done_models.add(item["model_id"])
            if len(done_models) >= pending:
                break

    yield f"data: {json.dumps({'type':'all_done'})}\n\n"


@router.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt 가 비어 있습니다.")
    return StreamingResponse(
        _event_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx buffering 방지
        },
    )
