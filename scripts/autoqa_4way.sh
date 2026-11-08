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

# 1) 백엔드가 안 떠 있으면 띄운다
if ! curl -sf "$BASE/health" >/dev/null 2>&1; then
  echo "[1/5] 팀 UI 백엔드 시작 → $OUT/backend.log"
  nohup $PY -m uvicorn ui.backend.app:app --host 127.0.0.1 --port "${BASE##*:}" >"$OUT/backend.log" 2>&1 &
  for _ in $(seq 1 60); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 5; done
fi

# 2) 실행 전 점검 — 하나라도 실패하면 멈춘다
echo "[2/5] 실행 전 점검"
$PY -m silvercare_autoqa preflight --backend "$BASE" --run-id "$RUN_ID" --model-ids "$MODELS"

# 3) 골든셋 50문항 답 받기 (탐욕 디코딩, 160토큰)
echo "[3/5] 답 받기"
$PY -m silvercare_autoqa generate --source kd-ui --base-url "$BASE" --run-id "$RUN_ID" --model-ids "$MODELS" --out "$OUT/responses.jsonl"

# 4) 응답 점검 (빠진 문항·빈 답·시스템 프롬프트 섞임) — generate 가 이미 하지만 기록을 남긴다
$PY -m silvercare_autoqa check-responses "$OUT/responses.jsonl" | tee "$OUT/check_responses.txt"

# 5) 채점 — Claude Code 로그인이 되어 있으면 끝까지, 아니면 요청 파일까지
if $PY -m silvercare_autoqa preflight --judge >/dev/null 2>&1; then
  echo "[5/5] 채점·반박 검증·리포트 (claude -p, Sonnet 5.5 high)"
  $PY -m silvercare_autoqa run --responses "$OUT/responses.jsonl" --run "$OUT/run" \
    --teacher teacher_ft --pairs student_kd:student_ft,student_kd:student_base
  echo "리포트: $OUT/run/report.md"
else
  echo "[5/5] 이 서버에는 Claude Code 로그인이 없음 → 채점 요청 파일만 만든다"
  $PY -m silvercare_autoqa prep --responses "$OUT/responses.jsonl" --run "$OUT/run"
  echo "다음: $OUT/run 폴더를 Claude Code 가 있는 PC 로 옮겨"
  echo "  python -m silvercare_autoqa judge --run <폴더> && python -m silvercare_autoqa import --run <폴더> &&"
  echo "  python -m silvercare_autoqa verify --run <폴더> && python -m silvercare_autoqa finalize --run <폴더> --teacher teacher_ft --pairs student_kd:student_ft"
fi
