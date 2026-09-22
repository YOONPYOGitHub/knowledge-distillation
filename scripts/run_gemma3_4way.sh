#!/usr/bin/env bash
# Gemma 3 12B -> 1B 4-way, Qwen v3 설정 기준. RTX 3090 x 4.
#
# v3 는 KD 초기값으로 Student SFT 체크포인트를 쓰므로 baseline 이 distill 보다 먼저다.
# main.py 의 --step 은 고정 순서로 실행되므로 단계마다 따로 호출한다.
#
# 한 단계가 실패해도 다음 단계를 시도한다. 앞 단계 산출물이 없으면
# scripts/gemma3_stage.py 가 조건을 낮춰서(pretrained Teacher / Student) 진행한다.

set -u
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/gemma3_12b_1b_3090x4_v3.yaml}
PY="$PWD/.venv-gemma/bin/python"
TORCHRUN="$PWD/.venv-gemma/bin/torchrun --nproc_per_node=2 --master_port=29571"
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

RESULTS=()

run_stage() {
  local name=$1; shift
  echo ""
  echo "=================================================================="
  echo " $name  ($(date '+%H:%M:%S'))"
  echo "=================================================================="
  local start=$SECONDS
  if "$@"; then
    RESULTS+=("✅ $name ($(( (SECONDS - start) / 60 ))분)")
  else
    RESULTS+=("❌ $name ($(( (SECONDS - start) / 60 ))분 후 실패)")
  fi
}

run_stage "STEP 1/4  Teacher FT (QLoRA nf4)" \
  $TORCHRUN scripts/gemma3_stage.py "$CONFIG" train_teacher

run_stage "STEP 2/4  Student SFT (baseline)" \
  $TORCHRUN scripts/gemma3_stage.py "$CONFIG" baseline

run_stage "STEP 3/4  KD (top-K reverse KL, Teacher int8)" \
  $TORCHRUN scripts/gemma3_stage.py "$CONFIG" distill

run_stage "STEP 4/4  Evaluate (Teacher bf16 분할)" \
  $PY scripts/gemma3_stage.py "$CONFIG" evaluate

run_stage "STEP 4/4  Compare" \
  $PY scripts/gemma3_stage.py "$CONFIG" compare

echo ""
echo "=================================================================="
echo " 요약  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=================================================================="
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "PIPELINE_DONE"
