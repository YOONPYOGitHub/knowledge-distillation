"""사전 점검 도구 시험 — 응답 점검(프롬프트 섞임), 학습 데이터 오염 검사, 실행 전 점검(가짜 백엔드)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from silvercare_autoqa.golden import DEFAULT_GOLDEN
from silvercare_autoqa.responders import APP_SYSTEM, raw_prompt
from silvercare_autoqa.tools import check_responses, contamination, preflight


def test_check_responses_catches_prompt_leak(tmp_path, golden):
    it = golden["by_id"]["SC-E01"]
    rows = [{"model": "ok", "item_id": i["id"], "response": i["reference_answer"]} for i in golden["items"]]
    rows += [{"model": "leaky", "item_id": "SC-E01", "response": raw_prompt(it)},         # 프롬프트 통째로 섞인 답
             {"model": "leaky", "item_id": "SC-E02", "response": ""}]
    p = tmp_path / "r.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    rep = check_responses(str(p), str(DEFAULT_GOLDEN))
    assert rep["models"]["ok"]["missing_items"] == [] and not rep["models"]["ok"]["system_prompt_leak"]
    assert rep["models"]["leaky"]["system_prompt_leak"] == ["SC-E01"]
    assert rep["models"]["leaky"]["empty"] == ["SC-E02"] and not rep["ok"]


def test_contamination(tmp_path, golden):
    it = golden["by_id"]["SC-M01"]
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps({"q": "무릎이 아파요", "a": "병원에 가 보세요"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert contamination([str(clean)], str(DEFAULT_GOLDEN), extra=None)["ok"]
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "x.json").write_text(json.dumps({"data": [{"question": it["utterance"] + " 꼭 알려 주세요"}]}, ensure_ascii=False),
                                  encoding="utf-8")
    rep = contamination([str(dirty)], str(DEFAULT_GOLDEN), extra=None)
    assert not rep["ok"] and rep["items_hit"] == ["SC-M01"]


class _B(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            body = {"status": "ok"}
        elif self.path == "/runs":
            body = [{"run_id": "kanana-v4", "checkpoints": {"teacher_ft": True, "student_kd": True, "student_ft": False}}]
        else:
            self.send_response(404)
            self.end_headers()
            return
        b = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def test_preflight_backend():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _B)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        checks = dict((n, ok) for n, ok, _ in preflight(str(DEFAULT_GOLDEN), backend=url, run_id="kanana-v4"))
        assert checks["골든셋 검증"] and checks["팀 UI 백엔드 /health"] and checks["run 'kanana-v4' 있음"]
        assert checks["체크포인트 student_kd"] and not checks["체크포인트 student_ft"] and checks["체크포인트 student_base"]
        missing = dict((n, ok) for n, ok, _ in preflight(str(DEFAULT_GOLDEN), backend=url, run_id="없는-run"))
        assert not missing["run '없는-run' 있음"]
    finally:
        srv.shutdown()
    down = dict((n, ok) for n, ok, _ in preflight(str(DEFAULT_GOLDEN), backend="http://127.0.0.1:9", timeout=2))
    assert not down["팀 UI 백엔드 /health"]


def test_leak_probes_exist():
    from silvercare_autoqa.tools import _LEAK_PROBES
    assert len(_LEAK_PROBES) >= 3 and all(p in APP_SYSTEM for p in _LEAK_PROBES)
