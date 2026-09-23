#!/usr/bin/env bash
# Gemma 3 12B -> 1B 4-way, Qwen v3 설정 기준. RTX 3090 x 4.
#
# v3 는 KD 초기값으로 Student SFT 체크포인트를 쓰므로 baseline 이 distill 보다 먼저다.
# main.py 의 --step 은 고정 순서로 실행되므로 단계마다 따로 호출한다.
#
# 한 단계가 실패해도 다음 단계를 시도한다. 앞 단계 산출물이 없으면
# scripts/gemma3_stage.py 가 조건을 낮춰서(pretrained Teacher / Student) 진행한다.
#
# 사용법:
#   scripts/run_gemma3_4way.sh [config] [시작단계]
#
# 시작단계를 주면 그 앞 단계는 건너뛴다. 중단된 실행을 이어갈 때 쓴다.
# 앞 단계의 체크포인트(Teacher FT 어댑터 / student_ft_best.pt)가 남아 있어야 한다.
#   scripts/run_gemma3_4way.sh configs/gemma3_12b_1b_3090x4_v3.yaml distill

set -u
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/gemma3_12b_1b_3090x4_v3.yaml}
START=${2:-train_teacher}
PY="$PWD/.venv-gemma/bin/python"
TORCHRUN="$PWD/.venv-gemma/bin/torchrun --nproc_per_node=2 --master_port=29571"
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

case "$START" in
  train_teacher|baseline|distill|evaluate|compare) ;;
  *) echo "❌ 시작단계가 잘못됐습니다: $START (train_teacher|baseline|distill|evaluate|compare)"; exit 2 ;;
esac

RESULTS=()
STARTED=0

# 시작단계에 도달하기 전이면 건너뛴다. 한 번 도달하면 이후는 모두 실행한다.
should_run() {
  [ "$STARTED" = "1" ] && return 0
  [ "$1" = "$START" ] && { STARTED=1; return 0; }
  return 1
}

run_stage() {
  local stage=$1 name=$2; shift 2
  if ! should_run "$stage"; then
    echo "⏭️  건너뜀: $name (시작단계 $START 이전)"
    RESULTS+=("⏭️ $name (건너뜀)")
    return
  fi
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

run_stage train_teacher "STEP 1/4  Teacher FT (QLoRA nf4)" \
  $TORCHRUN scripts/gemma3_stage.py "$CONFIG" train_teacher

run_stage baseline "STEP 2/4  Student SFT (baseline)" \
  $TORCHRUN scripts/gemma3_stage.py "$CONFIG" baseline

run_stage distill "STEP 3/4  KD (top-K reverse KL, Teacher int8)" \
  $TORCHRUN scripts/gemma3_stage.py "$CONFIG" distill

run_stage evaluate "STEP 4/4  Evaluate (Teacher bf16 분할)" \
  $PY scripts/gemma3_stage.py "$CONFIG" evaluate

run_stage compare "STEP 4/4  Compare" \
  $PY scripts/gemma3_stage.py "$CONFIG" compare

echo ""
echo "=================================================================="
echo " 요약  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=================================================================="
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "PIPELINE_DONE"
