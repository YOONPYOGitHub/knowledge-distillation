#!/usr/bin/env bash
# top-K(128) KD 대조 실험 — full-vocab KD 를 같은 조건으로 한 번 더 돌린다.
#
# 본 실행(gemma3-12b-1b-v3)이 끝나기를 기다린 뒤,
#   1. Teacher FT 어댑터와 Student SFT 초기값을 재사용해 full-vocab KD 를 학습
#   2. 4-way 평가 + 비교
#   3. 같은 배치에서 top-K 와 full-vocab 의 KD 손실/gradient 를 직접 비교 (정적 측정)
#
# train_teacher / baseline 은 다시 돌리지 않는다. kd_top_k 하나만 다른 A/B 가 목적이다.

set -u
cd "$(dirname "$0")/.."

MAIN_LOG=results/gemma3/logs/gemma3-12b-1b-v3/pipeline.log
MAIN_CKPT=results/gemma3/checkpoints/gemma3-12b-1b-v3
AB_CKPT=results/gemma3/checkpoints/gemma3-12b-1b-v3-fullvocab
CONFIG=configs/gemma3_12b_1b_3090x4_v3_fullvocab.yaml

PY="$PWD/.venv-gemma/bin/python"
TORCHRUN="$PWD/.venv-gemma/bin/torchrun --nproc_per_node=2 --master_port=29575"
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

echo "본 실행 종료를 기다립니다..."
until grep -q "PIPELINE_DONE" "$MAIN_LOG" 2>/dev/null; do sleep 60; done
echo "본 실행 종료 확인 ($(date '+%H:%M:%S'))"

if [ ! -f "$MAIN_CKPT/student_ft_best.pt" ]; then
  echo "❌ Student SFT 체크포인트가 없어 A/B 를 진행할 수 없습니다: $MAIN_CKPT"
  exit 1
fi

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

# 4-way 평가가 Student (FT) 행을 채우려면 같은 SFT 체크포인트가 이 run 의 디렉터리에도 보여야 한다.
mkdir -p "$AB_CKPT"
ln -sf "$PWD/$MAIN_CKPT/student_ft_best.pt" "$AB_CKPT/student_ft_best.pt"

run_stage "A/B 1/3  KD (full-vocab reverse KL)" \
  $TORCHRUN scripts/gemma3_stage.py "$CONFIG" distill

run_stage "A/B 2/3  Evaluate + Compare (full-vocab)" \
  bash -c "$PY scripts/gemma3_stage.py '$CONFIG' evaluate && $PY scripts/gemma3_stage.py '$CONFIG' compare"

# 같은 Student 상태에서 두 방식의 KD 손실과 gradient 를 직접 비교한다.
run_stage "A/B 3/3  top-K vs full-vocab 정적 측정" \
  $PY scripts/gemma3_topk_ablation.py configs/gemma3_12b_1b_3090x4_v3.yaml 16

run_stage "A/B 요약표" \
  $PY scripts/gemma3_compare_runs.py

echo ""
echo "=================================================================="
echo " A/B 요약  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=================================================================="
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "AB_DONE"
