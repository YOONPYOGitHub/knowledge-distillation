#!/usr/bin/env bash
# 온담(로컬 LLM + RAG) 답과 상용 LLM(ChatGPT luna 6.1) 답을 같은 골든셋으로 각각 채점하고, 블라인드로 직접 비교한다.
# Claude Code 가 로그인된 PC(채점기 claude -p)에서 실행한다. 온담·비교 모델은 주소·키로 부른다.
#
#   ONDAM_URL=https://<온담 주소> LABEL=ondam_local \
#   COMPARE_MODEL=<OpenAI 모델 이름> bash scripts/autoqa_rag_compare.sh
#
# 환경변수
#   ONDAM_URL      (필수) 온담 주소. 로컬로 띄웠으면 http://127.0.0.1:<포트>
#   LABEL          온담 답의 이름 (기본 ondam_local). 앱 뒤 모델을 바꿔 여러 번 잴 때 이름을 바꾼다 (예: ondam_qwen4b)
#   COMPARE_MODEL  OpenAI 공식 API 모델 이름 — 키는 OPENAI_API_KEY 환경변수 (계정에서 정확한 이름 확인)
#   COMPARE_SHEET  API 대신 ChatGPT 앱 답을 손으로 채운 CSV (sheet-template 으로 만든 서식). 둘 중 하나는 필수
#   COMPARE_LABEL  비교 모델 답의 이름 (기본 chatgpt_luna61)
#   COMPARE_SYSTEM app(온담과 같은 어르신 지시문, 기본) | none(지시문 없이)
#   NORAG_URL·NORAG_MODEL  (선택) 같은 로컬 모델을 검색 없이 부를 OpenAI 호환 주소·모델 (Ollama 예: http://127.0.0.1:11434)
#   NO_THINK=1     (선택) 검색 끈 로컬 모델이 Qwen3 처럼 생각 과정을 먼저 쓰면 /no_think 를 붙인다
#
# 결과: results/autoqa/<날짜_시각>_<LABEL>/{responses/*.jsonl, run/report.md, run/compare/<A>__vs__<B>/report.md}
set -euo pipefail

: "${ONDAM_URL:?ONDAM_URL 에 온담 주소를 넣으세요}"
LABEL="${LABEL:-ondam_local}"
CMP_LABEL="${COMPARE_LABEL:-chatgpt_luna61}"
CMP_SYSTEM="${COMPARE_SYSTEM:-app}"
PY="${PYTHON:-python}"
OUT="results/autoqa/$(date +%Y%m%d_%H%M)_${LABEL}"
mkdir -p "$OUT/responses"

echo "[1/6] 실행 전 점검 — 골든셋·autoqa 자체 시험·채점기 로그인"
"$PY" -m silvercare_autoqa validate >/dev/null
"$PY" -m silvercare_autoqa preflight --judge --selftest

echo "[2/6] 온담 답 받기 ($ONDAM_URL)"
"$PY" -m silvercare_autoqa generate --source ondam --base-url "$ONDAM_URL" --label "$LABEL" --out "$OUT/responses/ondam.jsonl"

echo "[3/6] 비교 모델 답 받기"
if [ -n "${COMPARE_SHEET:-}" ]; then
  "$PY" -m silvercare_autoqa generate --source sheet --sheet "$COMPARE_SHEET" --label "$CMP_LABEL" \
    --sheet-model "${COMPARE_MODEL:-ChatGPT luna 6.1}" --collected-at "${COLLECTED_AT:?손으로 받은 날짜 COLLECTED_AT=YYYY-MM-DD}" \
    --settings "${COMPARE_SETTINGS:-기록 없음}" --out "$OUT/responses/compare.jsonl"
else
  : "${COMPARE_MODEL:?COMPARE_MODEL(OpenAI API) 또는 COMPARE_SHEET(손으로 받은 CSV) 중 하나가 필요}"
  "$PY" -m silvercare_autoqa generate --source openai-api --model "$COMPARE_MODEL" --system "$CMP_SYSTEM" \
    --label "$CMP_LABEL" --out "$OUT/responses/compare.jsonl"
fi

RESP=("$OUT/responses/ondam.jsonl" "$OUT/responses/compare.jsonl")
PAIRS="$LABEL:$CMP_LABEL"
if [ -n "${NORAG_URL:-}" ]; then
  echo "[3+/6] 같은 로컬 모델, 검색 없이 ($NORAG_URL)"
  NT=()
  [ "${NO_THINK:-0}" = "1" ] && NT=(--no-think)
  "$PY" -m silvercare_autoqa generate --source openai --base-url "$NORAG_URL" --model "${NORAG_MODEL:?NORAG_MODEL 필요}" \
    --label local_norag --max-new-tokens 600 ${NT[@]+"${NT[@]}"} --out "$OUT/responses/norag.jsonl"
  RESP+=("$OUT/responses/norag.jsonl")
  PAIRS="$PAIRS,$LABEL:local_norag"
fi
for f in "${RESP[@]}"; do "$PY" -m silvercare_autoqa check-responses "$f" | tee -a "$OUT/check_responses.txt"; done

echo "[4/6] 각각 채점 — 결정 룰 → 채점기 → 반박 검증 → 판정 (claude -p, Sonnet 5.5 high)"
"$PY" -m silvercare_autoqa run --responses "${RESP[@]}" --run "$OUT/run" --reference "$CMP_LABEL" --pairs "$PAIRS"

echo "[5/6] 블라인드 직접 비교 + 비열등 판정 ($LABEL vs $CMP_LABEL)"
"$PY" -m silvercare_autoqa compare --run "$OUT/run" --a "$LABEL" --b "$CMP_LABEL"

echo "[6/6] 끝"
echo "각자 채점: $OUT/run/report.md"
echo "직접 비교: $OUT/run/compare/${LABEL}__vs__${CMP_LABEL}/report.md"
