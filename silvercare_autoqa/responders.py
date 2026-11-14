"""응답 생성기 — 골든셋 발화를 모델에 넣고 응답 JSONL 을 만든다. 채점과 분리해 응답을 먼저 동결한다.

RAG 전환(2026-11-14) 뒤 주로 쓰는 것
- ondam     : 온담 앱 POST /api/chat — 답과 출처(sources)를 함께 받는다. 앱 뒤의 모델이 Azure 든 로컬(Qwen·Kanana·Ollama)이든 같다
- openai-api: OpenAI 공식 API (ChatGPT luna 6.1 같은 상용 비교 모델). 키는 환경변수 OPENAI_API_KEY
- sheet     : ChatGPT 앱에서 손으로 받은 답을 채운 CSV (sheet-template 으로 빈 서식을 만든다)
- openai    : Ollama·llama.cpp·vLLM 같은 OpenAI 호환 서버 — 검색 없이 같은 로컬 모델에 바로 묻기(RAG 효과 비교)
지식증류 때 쓰던 것 (보관)
- kd-ui  : 팀 UI 백엔드 POST /generate/batch (teacher_ft / student_kd / student_ft / student_base 한 번에)
- claude : 이 PC 의 Claude Code (Teacher 대용 기준 응답을 만들 때)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from .golden import DATA_DIR

APP_SYSTEM = (DATA_DIR / "app_system_prompt.txt").read_text(encoding="utf-8").strip()
# 응답마다 남기는 생성 계약 — 입력 형식·시스템 프롬프트가 모델마다 다르면 모델 차이와 형식 차이가 섞인다
# (쇼핑 AI 검수의 '학습·평가 계약 드리프트' 교훈). check-responses·prep 이 모델 간 형식 차이를 알린다.
SYSTEM_SHA = __import__("hashlib").sha256(APP_SYSTEM.encode("utf-8")).hexdigest()[:12]

RAW_TEMPLATE = "{system}\n\n{context}어르신: {utterance}\n돌봄 AI:"


def user_turn(item: dict) -> str:
    ctx = f"[기기 기록] {item['context']}\n" if item.get("context") else ""
    return f"{ctx}{item['utterance']}"


def raw_prompt(item: dict, system: str = APP_SYSTEM, template: str = RAW_TEMPLATE) -> str:
    ctx = f"[기기 기록] {item['context']}\n" if item.get("context") else ""
    return template.format(system=system, context=ctx, utterance=item["utterance"])


ANSWER_MARK = "돌봄 AI:"


def _clean(text: str) -> str:
    """생성 모델이 다음 차례까지 이어 쓴 꼬리를 자른다 (base 모델 대비)."""
    for stop in ("\n어르신:", "\n사용자:", "\nUser:", "<|im_end|>", "<end_of_turn>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def strip_prompt(text: str, prompt: str) -> str:
    """팀 UI 백엔드는 prompt 까지 붙인 전체 텍스트를 돌려준다(generator.py 의 decode(output_ids[0])).
    prompt 를 떼고 답만 남긴다. tokenizer 가 공백을 바꿔 앞부분이 안 맞으면 마지막 답 표식 뒤를 쓴다.
    이걸 안 하면 응답 대신 시스템 프롬프트(119 안내 포함)를 채점하게 된다."""
    if text.startswith(prompt):
        return text[len(prompt):]
    k = text.rfind(ANSWER_MARK)
    if k != -1:
        return text[k + len(ANSWER_MARK):]
    return text


def kd_ui(items: Iterable[dict], base_url: str, run_id: str, model_ids: list[str], *, max_new_tokens: int = 160,
          temperature: float = 0.0, chunk: int = 50, timeout: int = 1800) -> list[dict]:
    import requests

    items = list(items)
    rows = []
    params = {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": 1.0, "top_k": 0,
              "repetition_penalty": 1.1, "do_sample": temperature > 0, "seed": 42}
    gen_errors: list[str] = []
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        prompts = [raw_prompt(it) for it in part]
        body = {"run_id": run_id, "model_ids": model_ids, "prompts": prompts, "params": params}
        r = requests.post(base_url.rstrip("/") + "/generate/batch", json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if len(data["items"]) != len(part):
            # 백엔드는 빈 prompt 를 버리므로 개수가 달라지면 문항 대응이 어긋난다 → 멈춘다
            raise RuntimeError(f"응답 개수 불일치: 보낸 {len(part)} · 받은 {len(data['items'])}")
        for it, p, res in zip(part, prompts, data["items"]):
            if res.get("prompt") not in (None, p):
                raise RuntimeError(f"{it['id']}: 응답이 다른 prompt 에 대한 것")
            for mid, o in res["outputs"].items():
                if o.get("error"):
                    gen_errors.append(f"{it['id']}/{mid}: {str(o['error'])[:80]}")
                    continue
                rows.append({"model": mid, "item_id": it["id"], "response": _clean(strip_prompt(o.get("text") or "", p)),
                             "meta": {"source": "kd-ui", "prompt_format": "raw", "system_sha": SYSTEM_SHA, "run_id": run_id,
                                      "num_tokens": o.get("num_tokens"), "elapsed_ms": o.get("elapsed_ms"), "params": params}})
    if gen_errors:
        # 모델 미적재·메모리 부족 같은 서버 문제를 '빈 답'으로 채점하면 응급 놓침이 대량으로 생긴다 → 멈춘다 (감사 A04)
        raise RuntimeError(f"백엔드 생성 오류 {len(gen_errors)}건 — 모델 적재·메모리를 확인하고 다시 실행: " + "; ".join(gen_errors[:5]))
    return rows


def openai_compat(items: Iterable[dict], base_url: str, model: str, label: str, *, max_tokens: int = 160,
                  temperature: float = 0.0, api_key: str = "none", timeout: int = 300, mode: str = "chat",
                  no_think: bool = False) -> list[dict]:
    """llama.cpp server · vLLM 같은 OpenAI 호환 서버 (양자화 GGUF 실측용).
    mode="chat": /v1/chat/completions — 서버가 모델의 chat template 을 쓴다 (대화형 모델).
    mode="raw" : /v1/completions 에 팀 UI 백엔드와 같은 raw prompt — base 모델의 양자화 전후를 같은 입력 형식으로 비교할 때.
    chat 모드는 추론(thinking)을 끄라고 함께 보낸다 (Qwen3 계열은 기본으로 생각 과정을 먼저 쓴다)."""
    import requests

    rows = []
    extras_ok = True          # 팀 UI 백엔드와 같은 반복 억제·추론 끄기 — 모르는 서버면 빼고, 뺐다고 기록한다
    rep = 1.1
    for it in items:
        t0 = time.time()
        headers = {"Authorization": f"Bearer {api_key}"}
        if mode == "raw":
            url = base_url.rstrip("/") + "/v1/completions"
            body = {"model": model, "prompt": raw_prompt(it), "max_tokens": max_tokens, "temperature": temperature,
                    "stop": ["\n어르신:"]}
            extras = {"repeat_penalty": rep, "repetition_penalty": rep}
        else:
            url = base_url.rstrip("/") + "/v1/chat/completions"
            body = {"model": model, "messages": [{"role": "system", "content": APP_SYSTEM},
                                                 {"role": "user", "content": user_turn(it) + (" /no_think" if no_think else "")}],
                    "max_tokens": max_tokens, "temperature": temperature}
            extras = {"repeat_penalty": rep, "repetition_penalty": rep, "chat_template_kwargs": {"enable_thinking": False}}
        r = requests.post(url, json={**body, **extras} if extras_ok else body, timeout=timeout, headers=headers)
        if r.status_code in (400, 422) and extras_ok:            # 이 옵션들을 모르는 서버 — 빼고 다시
            extras_ok = False
            r = requests.post(url, json=body, timeout=timeout, headers=headers)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        text = ch.get("text") if mode == "raw" else ch["message"].get("content")
        if text is None:
            raise RuntimeError(f"{it['id']}: 서버가 content 없이 답함 (도구 호출·거절 형식?) — 빈 답으로 채점하지 않음")
        rows.append({"model": label, "item_id": it["id"], "response": _clean(text),
                     "meta": {"source": "openai", "prompt_format": "raw" if mode == "raw" else "server_chat_template",
                              "system_sha": SYSTEM_SHA, "temperature": temperature, "max_new_tokens": max_tokens,
                              "repetition_penalty": rep if extras_ok else None,
                              "thinking_off_sent": mode != "raw" and extras_ok, "no_think_tag": no_think,
                              "model": model, "elapsed_ms": (time.time() - t0) * 1000}})
    return rows


def claude_teacher(items: Iterable[dict], label: str = "teacher_claude", *, model: str = "claude-sonnet-5-5",
                   effort: str = "high", claude_bin: str | None = None) -> list[dict]:
    import subprocess
    import tempfile

    from .judge import claude_base_cmd, clean_env

    rows = []
    work = tempfile.mkdtemp(prefix="silvercare_teacher_")      # 프로젝트 지침(CLAUDE.md)을 읽지 않게 빈 폴더에서 실행
    for it in items:
        cmd = claude_base_cmd(APP_SYSTEM, model=model, effort=effort, claude_bin=claude_bin, cwd=work) + \
            ["--output-format", "json"]
        proc = subprocess.run(cmd, input=user_turn(it), capture_output=True, text=True, encoding="utf-8", timeout=600,
                              env=clean_env(), cwd=work)
        data = json.loads(proc.stdout)
        if data.get("is_error"):
            raise RuntimeError(str(data.get("result"))[:200])
        rows.append({"model": label, "item_id": it["id"], "response": (data.get("result") or "").strip(),
                     "meta": {"source": "claude-cli", "prompt_format": "claude-cli", "system_sha": SYSTEM_SHA,
                              "model": model, "effort": effort}})
    return rows


def hf_local(items: Iterable[dict], model_path: str, label: str, *, revision: str | None = None, chat: str = "auto",
             max_new_tokens: int = 160, dtype: str = "bfloat16", device: str = "cuda",
             repetition_penalty: float = 1.1, log=print) -> list[dict]:
    """Hugging Face 형식 모델(팀 체크포인트 폴더 또는 공개 모델 id)을 직접 불러 답을 만든다.

    - chat="auto": tokenizer 에 chat template 이 있으면 쓰고(instruct·대화형 Student), 없으면 raw prompt(base 모델).
      팀 UI 백엔드는 chat template 을 쓰지 않으므로, 대화형 Student 는 이 경로로 받는 것이 맞다.
    - 탐욕 디코딩(do_sample=False) · 새로 만든 토큰만 디코드 · 추론(thinking) 모드는 끈다.
    """
    import time as _t

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = list(items)
    tok = AutoTokenizer.from_pretrained(model_path, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, torch_dtype=getattr(torch, dtype),
                                                 device_map=device)
    model.eval()
    use_chat = chat == "on" or (chat == "auto" and bool(getattr(tok, "chat_template", None)))
    # base 모델 tokenizer 에도 chat template 이 들어 있는 경우가 많다 — 고른 형식을 늘 알려서 4-way 조건이 섞이지 않게 (감사 R24)
    log(f"  입력 형식: {'chat template' if use_chat else 'raw prompt'} (--chat {chat}) — 같은 비교의 다른 모델과 같은지 확인")
    rows = []
    for k, it in enumerate(items, 1):
        if use_chat:
            msgs = [{"role": "system", "content": APP_SYSTEM}, {"role": "user", "content": user_turn(it)}]
            try:
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=False)
            except Exception:  # noqa: BLE001 — system 역할이나 enable_thinking 을 모르는 template
                try:
                    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                except Exception:  # noqa: BLE001
                    merged = [{"role": "user", "content": APP_SYSTEM + "\n\n" + user_turn(it)}]
                    ids = tok.apply_chat_template(merged, add_generation_prompt=True, return_tensors="pt")
            prompt_used = "chat_template"
        else:
            ids = tok(raw_prompt(it), return_tensors="pt").input_ids
            prompt_used = "raw"
        ids = ids.to(model.device)
        t0 = _t.time()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=repetition_penalty,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        new = out[0][ids.shape[1]:]
        text = _clean(tok.decode(new, skip_special_tokens=True))
        rows.append({"model": label, "item_id": it["id"], "response": text,
                     "meta": {"source": "hf-local", "model_path": model_path, "revision": revision, "prompt": prompt_used,
                              "prompt_format": prompt_used, "system_sha": SYSTEM_SHA, "temperature": 0.0,
                              "repetition_penalty": repetition_penalty,
                              "max_new_tokens": max_new_tokens, "num_new_tokens": int(new.numel()),
                              "elapsed_ms": round((_t.time() - t0) * 1000, 1), "hit_max_tokens": int(new.numel()) >= max_new_tokens}})
        if k % 10 == 0 or k == len(items):
            log(f"  생성 {k}/{len(items)}")
    return rows


def reference_rows(golden: dict, label: str = "reference") -> list[dict]:
    """골든셋 모범 답안을 응답처럼 넣는다 — 채점기가 좋은 답을 통과시키는지 보는 기준선."""
    return [{"model": label, "item_id": it["id"], "response": it["reference_answer"], "meta": {"source": "golden"}}
            for it in golden["items"]]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# ───────────────────────────── RAG 전환 뒤 수집기 ─────────────────────────────

def ondam_api(items: Iterable[dict], base_url: str, label: str, *, timeout: int = 150, pause: float = 0.5,
              retries: int = 2, log=print) -> list[dict]:
    """온담 앱 POST /api/chat 으로 답과 출처를 받는다 (한 문항씩, 대화 이력 없이).

    - 앱 뒤의 생성 모델(Azure·로컬 LLM)이 바뀌어도 받는 방법은 같다. 어떤 모델이었는지는 --label 과 기록으로 남긴다.
    - 422(제공자 안전 거절 등)는 어르신 화면에 뜨는 안내 문구를 답으로 남기고 meta.refusal 에 표시한다.
    - 그 밖의 서버 오류는 다시 시도하고, 끝내 실패하면 멈춘다 (서버 문제를 빈 답으로 채점하지 않는다 — 감사 A04).
    - 서비스에 부담을 주지 않게 요청 사이에 pause 초를 쉰다.
    """
    import requests

    url = base_url.rstrip("/") + "/api/chat"
    rows = []
    items = list(items)
    for k, it in enumerate(items, 1):
        body = {"message": user_turn(it), "history": []}
        last = ""
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                r = requests.post(url, json=body, timeout=timeout, headers={"User-Agent": "silvercare-autoqa"})
            except requests.RequestException as e:
                last = f"연결 실패: {e}"
                time.sleep(2 + 2 * attempt)
                continue
            elapsed = (time.time() - t0) * 1000
            if r.status_code == 200:
                data = r.json()
                answer = data.get("answer")
                if not isinstance(answer, str):
                    raise RuntimeError(f"{it['id']}: 응답에 answer 가 없음 — API 형식이 바뀌었는지 확인")
                rag = {"mode": data.get("mode"), "sources": data.get("sources") or [],
                       "health_topics": data.get("health_topics") or []}
                if isinstance(data.get("contexts"), list):          # 평가용으로 전체 문맥을 주면 함께 받는다
                    rag["contexts"] = data["contexts"]
                    rag["contexts_complete"] = True
                rows.append({"model": label, "item_id": it["id"], "response": answer.strip(), "rag": rag,
                             "meta": {"source": "ondam-api", "base_url": base_url, "prompt_format": "ondam-api",
                                      "message": body["message"], "mode": data.get("mode"), "elapsed_ms": round(elapsed, 1),
                                      "medical_caution": bool(data.get("medical_caution")),
                                      "display_notice": data.get("display_notice") or ""}})
                break
            if r.status_code == 422:
                try:
                    detail = r.json().get("detail")
                except ValueError:
                    detail = None
                text = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)[:300]
                rows.append({"model": label, "item_id": it["id"], "response": text or "",
                             "rag": {"mode": "refusal", "sources": []},
                             "meta": {"source": "ondam-api", "base_url": base_url, "prompt_format": "ondam-api",
                                      "message": body["message"], "mode": "refusal", "refusal": "http_422",
                                      "elapsed_ms": round(elapsed, 1)}})
                break
            last = f"HTTP {r.status_code}: {r.text[:120]}"
            time.sleep(2 + 2 * attempt)
        else:
            raise RuntimeError(f"{it['id']}: 온담 API 실패 — {last} (서비스 상태를 확인하고 다시 실행)")
        if k % 10 == 0 or k == len(items):
            log(f"  온담 {k}/{len(items)}")
        time.sleep(pause)
    return rows


def openai_api(items: Iterable[dict], model: str, label: str, *, system: str = "app",
               base_url: str = "https://api.openai.com/v1", api_key_env: str = "OPENAI_API_KEY",
               max_output_tokens: int | None = None, reasoning_effort: str | None = None, timeout: int = 300,
               log=print) -> list[dict]:
    """OpenAI 공식 API 로 상용 비교 모델의 답을 받는다 (예: ChatGPT luna 6.1 — 정확한 모델 이름은 계정에서 확인).

    - system="app": 온담과 같은 어르신 응답 지시문(app_system_prompt.txt)을 준다 — 같은 지시로 비교(기본).
      system="none": 지시문 없이 묻는다 — 어르신이 ChatGPT 에 그냥 물었을 때와 비교.
    - 온도·추론 길이는 모델 기본값을 쓴다 (새 모델은 temperature 를 못 바꾸는 경우가 있다). 바꾼 값은 기록에 남는다.
    - 키는 환경변수에서만 읽고 기록하지 않는다.
    """
    import os

    import requests

    key = os.environ.get(api_key_env)
    if not key:
        raise SystemExit(f"환경변수 {api_key_env} 가 없음 — OpenAI API 키를 이 PC 환경변수에 넣고 다시 실행 "
                         "(키가 없으면 sheet-template 으로 ChatGPT 앱 답을 손으로 받는다)")
    if system not in ("app", "none"):
        raise ValueError("system 은 app 또는 none")
    rows = []
    items = list(items)
    for k, it in enumerate(items, 1):
        msgs = ([{"role": "system", "content": APP_SYSTEM}] if system == "app" else []) + \
            [{"role": "user", "content": user_turn(it)}]
        body: dict = {"model": model, "messages": msgs}
        if max_output_tokens:
            body["max_completion_tokens"] = max_output_tokens
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        t0 = time.time()
        r = requests.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code != 200:
            raise RuntimeError(f"{it['id']}: OpenAI API HTTP {r.status_code} — {r.text[:200]}")
        data = r.json()
        ch = data["choices"][0]
        text = (ch.get("message") or {}).get("content")
        if text is None:
            raise RuntimeError(f"{it['id']}: content 없이 답함 (거절·도구 호출?) — 빈 답으로 채점하지 않음")
        rows.append({"model": label, "item_id": it["id"], "response": text.strip(),
                     "meta": {"source": "openai-api", "model": data.get("model") or model, "requested_model": model,
                              "message": user_turn(it),
                              "prompt_format": f"openai-chat:{system}",
                              "system_sha": SYSTEM_SHA if system == "app" else None,
                              "max_new_tokens": max_output_tokens, "reasoning_effort": reasoning_effort,
                              "finish_reason": ch.get("finish_reason"), "usage": data.get("usage"),
                              "collected_at": time.strftime("%Y-%m-%d"),
                              "elapsed_ms": round((time.time() - t0) * 1000, 1)}})
        if k % 10 == 0 or k == len(items):
            log(f"  {label} {k}/{len(items)}")
    return rows


SHEET_HEAD = ["item_id", "질문", "답", "메모"]


def sheet_template(items: Iterable[dict], path: str | Path) -> int:
    """손으로 답을 받을 빈 서식 (CSV, 엑셀에서 열림). '질문' 칸을 그대로 붙여 넣고 받은 답을 '답' 칸에 붙인다."""
    import csv

    items = list(items)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(SHEET_HEAD)
        for it in items:
            w.writerow([it["id"], user_turn(it), "", ""])
    return len(items)


def sheet_rows(path: str | Path, label: str, *, model_name: str, collected_at: str, settings: str) -> list[dict]:
    """채운 서식 → 응답 행. 빈 답이 있으면 멈춘다 (빠뜨린 문항을 '빈 답'으로 채점하지 않게)."""
    import csv

    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        head = {h.strip().lower(): h for h in rd.fieldnames or []}
        iid = head.get("item_id") or head.get("문항")
        ans = head.get("답") or head.get("answer") or head.get("응답")
        if not iid or not ans:
            raise SystemExit(f"{path}: 'item_id'·'답' 칸이 필요함 (받은 칸: {rd.fieldnames})")
        rows, empty = [], []
        for rec in rd:
            text = (rec.get(ans) or "").strip()
            if not text:
                empty.append(rec.get(iid))
                continue
            rows.append({"model": label, "item_id": (rec.get(iid) or "").strip(), "response": text,
                         "meta": {"source": "sheet", "model": model_name, "collected_at": collected_at,
                                  "settings": settings, "prompt_format": f"manual:{settings}"}})
    if empty:
        raise SystemExit(f"{path}: 답이 빈 문항 {len(empty)}개 {empty[:5]} — 다 채운 뒤 다시 실행")
    return rows
