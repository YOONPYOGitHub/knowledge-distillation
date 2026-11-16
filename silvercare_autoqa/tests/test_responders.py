"""응답 수집 연동 시험 — 팀 UI 백엔드(/generate/batch)와 OpenAI 호환 서버를 흉내 낸 가짜 서버로.

가짜 서버는 팀 코드의 실제 동작을 따른다.
- ui/backend/schemas.py: model_ids 는 teacher_ft·student_kd·student_ft·student_base, 생성 설정 범위 검사
- ui/backend/routers/generate_batch.py: prompt 최대 200개, 응답 items[{prompt, outputs{mid: ModelOutput}}]
- ui/backend/generator.py: text 는 prompt 까지 붙인 전체 디코드 결과 (decode(output_ids[0]))
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from silvercare_autoqa import responders as R

MODEL_IDS = {"teacher_ft", "student_kd", "student_ft", "student_base"}


class _Handler(BaseHTTPRequestHandler):
    calls: list = []
    drop_one = False
    fail_on = None            # 이 글이 든 질문이 오면 500 (온담 서버가 중간에 멈춘 경우)

    def log_message(self, *a):          # 시험 출력 조용히
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        _Handler.calls.append((self.path, body))
        if self.path == "/generate/batch":
            p = body["params"]
            ok = (body.get("run_id") and set(body["model_ids"]) <= MODEL_IDS and 1 <= len(body["prompts"]) <= 200
                  and 1 <= p["max_new_tokens"] <= 1024 and 0 <= p["temperature"] <= 2 and 0 <= p["top_p"] <= 1
                  and 0 <= p["top_k"] <= 200 and 1 <= p["repetition_penalty"] <= 2 and isinstance(p["do_sample"], bool))
            if not ok:
                return self._send(422, {"detail": "schema"})
            prompts = body["prompts"][:-1] if _Handler.drop_one else body["prompts"]
            items = []
            for pr in prompts:
                outs = {}
                for mid in body["model_ids"]:
                    gen = f" {mid} 답입니다. 지금 바로 119에 전화하세요.\n어르신: 또 물어볼게요"
                    # student_base 는 tokenizer 가 공백을 바꾼 경우를 흉내 (앞부분 불일치 → 답 표식으로 자름)
                    text = (pr.replace("\n\n", "\n") if mid == "student_base" else pr) + gen
                    outs[mid] = {"text": text, "num_tokens": 12, "elapsed_ms": 5.0, "tokens_per_sec": 2400.0, "error": None}
                items.append({"prompt": pr, "outputs": outs})
            return self._send(200, {"run_id": body["run_id"], "count": len(items), "items": items})
        if self.path.endswith("/api/chat"):            # 온담 앱 — 주소 뒤에 경로가 붙은 서비스(/local-rag)도 같은 모양
            if _Handler.fail_on and _Handler.fail_on in body["message"]:
                return self._send(500, {"detail": "down"})
            return self._send(200, {"answer": "지금 바로 119에 전화하세요.", "mode": "grounded",
                                    "sources": [{"id": 1, "title": "t", "url": "u", "excerpt": "e"}]})
        if self.path == "/v1/chat/completions":
            msgs = body["messages"]
            assert msgs[0]["role"] == "system" and msgs[0]["content"] == R.APP_SYSTEM
            return self._send(200, {"choices": [{"message": {"content": "지금 바로 119에 전화하세요.<|im_end|>"}}]})
        return self._send(404, {})


@pytest.fixture()
def server():
    _Handler.calls = []
    _Handler.drop_one = False
    _Handler.fail_on = None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_kd_ui_strips_prompt_and_tail(server, golden):
    items = [golden["by_id"][i] for i in ("SC-E01", "SC-M02", "SC-C04")]      # SC-M02 는 기기 기록 포함
    rows = R.kd_ui(items, server, "kanana-v5", sorted(MODEL_IDS), chunk=2)
    assert len(_Handler.calls) == 2                                          # 50개 단위가 아니라 chunk 대로 나눠 보냄
    assert len(rows) == 3 * 4
    for r in rows:
        assert r["response"] == f"{r['model']} 답입니다. 지금 바로 119에 전화하세요.", r
        assert "돌봄 말벗" not in r["response"] and "어르신:" not in r["response"]   # 시스템 프롬프트·다음 차례가 섞이지 않음
        assert r["meta"]["run_id"] == "kanana-v5" and r["meta"]["params"]["do_sample"] is False
    body = _Handler.calls[0][1]
    assert body["params"]["temperature"] == 0.0 and body["params"]["max_new_tokens"] == 160
    assert "[기기 기록]" in _Handler.calls[0][1]["prompts"][1]                 # SC-M02 기록이 입력에 들어감


def test_kd_ui_count_mismatch_stops(server, golden):
    _Handler.drop_one = True
    with pytest.raises(RuntimeError, match="개수 불일치"):
        R.kd_ui([golden["by_id"]["SC-E01"], golden["by_id"]["SC-E02"]], server, "x", ["student_kd"])


def test_openai_compat(server, golden):
    rows = R.openai_compat([golden["by_id"]["SC-M02"]], server, "qwen3.5-2b-q4", "student_kd_q4")
    assert rows[0]["response"] == "지금 바로 119에 전화하세요." and rows[0]["model"] == "student_kd_q4"
    msgs = _Handler.calls[0][1]["messages"]
    assert "[기기 기록]" in msgs[1]["content"] and golden["by_id"]["SC-M02"]["utterance"] in msgs[1]["content"]


def test_strip_prompt():
    p = R.raw_prompt({"utterance": "숨이 차요", "context": None})
    assert R.strip_prompt(p + " 119에 전화하세요.", p) == " 119에 전화하세요."
    assert R.strip_prompt("앞부분이 바뀐 글\n돌봄 AI: 답", p) == " 답"


def test_ondam_api_saves_each_answer_and_resumes(server, golden, tmp_path, monkeypatch):
    """검색이 느린 온담 서버: 받은 답을 한 문항씩 바로 저장하고, 멈춘 뒤에는 받은 문항을 다시 묻지 않고 이어 받는다."""
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    items = [golden["by_id"][i] for i in ("SC-E01", "SC-E02", "SC-E03")]
    part = tmp_path / "ondam.jsonl.part"
    _Handler.fail_on = items[2]["utterance"][:12]                    # 세 번째 문항에서 서버가 멈춘다
    with pytest.raises(RuntimeError, match="--resume"):
        R.ondam_api(items, server + "/local-rag", "ondam_local_rag", save_to=part, retries=0, log=lambda *_: None)
    saved = [json.loads(ln)["item_id"] for ln in part.read_text(encoding="utf-8").splitlines()]
    assert saved == ["SC-E01", "SC-E02"]                              # 멈추기 전에 받은 답은 남아 있다
    assert {p for p, _ in _Handler.calls} == {"/local-rag/api/chat"}  # 주소 뒤 경로 아래 /api/chat 을 부른다
    _Handler.fail_on, _Handler.calls = None, []
    rows = R.ondam_api(items, server + "/local-rag", "ondam_local_rag", save_to=part, resume=True, log=lambda *_: None)
    assert [r["item_id"] for r in rows] == ["SC-E01", "SC-E02", "SC-E03"]
    assert len(_Handler.calls) == 1                                    # 이미 받은 두 문항은 다시 묻지 않았다
    assert rows[2]["rag"]["mode"] == "grounded" and rows[2]["rag"]["sources"][0]["excerpt"] == "e"


def test_ondam_api_keep_going_skips_only_the_failing_item(server, golden, tmp_path, monkeypatch):
    """한 문항에서만 서버 오류가 계속되면(11/16 SC-E12 → 503) 그 문항만 실패 목록에 두고 나머지는 받는다. 빈 답으로 채우지 않는다."""
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    items = [golden["by_id"][i] for i in ("SC-E01", "SC-E02", "SC-E03")]
    _Handler.fail_on = items[1]["utterance"][:12]                    # 가운데 문항만 오류
    failed: list = []
    rows = R.ondam_api(items, server, "ondam_local_rag", save_to=tmp_path / "x.part", retries=0, failed=failed,
                       log=lambda *_: None)
    assert [r["item_id"] for r in rows] == ["SC-E01", "SC-E03"]
    assert [f["item_id"] for f in failed] == ["SC-E02"] and "500" in failed[0]["error"]
