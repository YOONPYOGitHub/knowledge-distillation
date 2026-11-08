#!/usr/bin/env bash
# 4-way(Teacher FT · Student KD · Student FT · Student Base) 답을 팀 UI 백엔드에서 받아 autoqa 로 채점한다.
# GPU 서버(VESSL 등)에서 실행한다. 이 PC 에 Claude Code 로그인이 있으면 채점까지, 없으면 채점 요청 파일까지 만든다.
#
#   bash scripts/autoqa_4way.sh <run_id> [backend_url] [model_ids]
#   예) bash scripts/autoqa_4way.sh kanana-8b-2b-v4
#
# 결과: results/autoqa/<날짜>_<run_id>/{responses.jsonl, run/report.md, run/report.html, ...}
set -euo pipefail

RUN_ID="${1:?run_id 를 넣으세요 (python -m silvercare_autoqa preflight --backend URL 로 목록 확인)}"
BASE="${2:-http://127.0.0.1:8000}"
MODELS="${3:-teacher_ft,student_kd,student_ft,student_base}"
PY="${PYTHON:-python}"
OUT="results/autoqa/$(date +%Y%m%d_%H%M)_${RUN_ID}"
mkdir -p "$OUT"

# Teacher·짝 비교는 모델 목록에서 정한다 — 목록을 바꿔도 비교가 말없이 빠지지 않게 (감사 R18·D18)
read -r TEACHER PAIRS < <("$PY" - "$MODELS" <<'EOF'
import sys
ms = sys.argv[1].split(",")
t = next((m for m in ms if m.startswith("teacher")), "")
pairs = [f"{a}:{b}" for a, b in (("student_kd", "student_ft"), ("student_kd", "student_base")) if a in ms and b in ms]
print(t or "-", ",".join(pairs) or "-")
EOF
)
FIN_ARGS=()
[ "$TEACHER" != "-" ] && FIN_ARGS+=(--teacher "$TEACHER")
[ "$PAIRS" != "-" ] && FIN_ARGS+=(--pairs "$PAIRS")

# 1) 백엔드가 안 떠 있으면 띄우고, 이 스크립트가 띄운 것은 끝날 때 끈다 (GPU 메모리를 계속 잡지 않게)
BACKEND_PID=""
cleanup() { if [ -n "$BACKEND_PID" ]; then kill "$BACKEND_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
if ! curl -sf "$BASE/health" >/dev/null 2>&1; then
  PORT="$("$PY" -c 'import sys, urllib.parse as u; print(u.urlparse(sys.argv[1]).port or 8000)' "$BASE")"
  echo "[1/5] 팀 UI 백엔드 시작 (포트 $PORT) → $OUT/backend.log"
  nohup "$PY" -m uvicorn ui.backend.app:app --host 127.0.0.1 --port "$PORT" >"$OUT/backend.log" 2>&1 &
  BACKEND_PID=$!
  for _ in $(seq 1 60); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 5; done
fi

# 2) 실행 전 점검 — 골든셋·백엔드·체크포인트·autoqa 자체 시험. 하나라도 실패하면 멈춘다
echo "[2/5] 실행 전 점검"
"$PY" -m silvercare_autoqa preflight --backend "$BASE" --run-id "$RUN_ID" --model-ids "$MODELS" --selftest

# 3) 골든셋 50문항 답 받기 (탐욕 디코딩, 160토큰, raw prompt)
echo "[3/5] 답 받기"
"$PY" -m silvercare_autoqa generate --source kd-ui --base-url "$BASE" --run-id "$RUN_ID" --model-ids "$MODELS" --out "$OUT/responses.jsonl"

# 4) 응답 점검 (빠진 문항·빈 답·시스템 프롬프트 섞임·생성 조건) — prep 도 같은 점검을 하지만 기록을 남긴다
"$PY" -m silvercare_autoqa check-responses "$OUT/responses.jsonl" | tee "$OUT/check_responses.txt"

# 5) 채점 — Claude Code 로그인이 되어 있으면 끝까지, 아니면 요청 파일까지
if JUDGE_CHECK="$("$PY" -m silvercare_autoqa preflight --judge 2>&1)"; then
  echo "[5/5] 채점·반박 검증·리포트 (claude -p, Sonnet 5.5 high)"
  "$PY" -m silvercare_autoqa run --responses "$OUT/responses.jsonl" --run "$OUT/run" ${FIN_ARGS[@]+"${FIN_ARGS[@]}"}
  echo "리포트: $OUT/run/report.md"
else
  echo "$JUDGE_CHECK" | grep -E "❌" || true
  echo "[5/5] 이 서버에서는 채점기(claude -p)를 쓸 수 없음 → 채점 요청 파일만 만든다"
  "$PY" -m silvercare_autoqa prep --responses "$OUT/responses.jsonl" --run "$OUT/run"
  echo "다음: $OUT/run 폴더를 Claude Code 가 있는 PC 로 옮겨 (같은 커밋의 autoqa 로)"
  echo "  python -m silvercare_autoqa judge --run <폴더> && python -m silvercare_autoqa import --run <폴더> &&"
  echo "  python -m silvercare_autoqa verify --run <폴더> && python -m silvercare_autoqa finalize --run <폴더> ${FIN_ARGS[*]+${FIN_ARGS[*]}}"
fi
