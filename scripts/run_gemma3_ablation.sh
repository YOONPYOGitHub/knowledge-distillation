#!/usr/bin/env bash
# v3 요인 분리 ablation — KD 초기값(FT/Base) × divergence(reverse/forward KL).
#
# v3(FT 초기값 + reverse KL)은 이미 있으므로 나머지 세 칸을 순서대로 돌린다.
# Teacher FT 어댑터와 Student SFT 는 v3 실행의 것을 재사용한다.
#
#   1. 실행마다 KD 학습 -> 4-way 평가 -> 비교
#   2. 네 KD Student 를 같은 test chunk 에서 paired bootstrap
#   3. 2×2 요약표
#
# 사용법: scripts/run_gemma3_ablation.sh [시작할 변형: ft_fwd|base_rev|base_fwd]

set -u
cd "$(dirname "$0")/.."

V3_CKPT=results/gemma3/checkpoints/gemma3-12b-1b-v3
VARIANTS=(ft_fwd base_rev base_fwd)
START=${1:-ft_fwd}

PY="$PWD/.venv-gemma/bin/python"
TORCHRUN="$PWD/.venv-gemma/bin/torchrun --nproc_per_node=2 --master_port=29577"
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

if [ ! -f "$V3_CKPT/student_ft_best.pt" ] || [ ! -d "$V3_CKPT/teacher_ft_best_adapter" ]; then
  echo "❌ v3 체크포인트가 없어 ablation 을 진행할 수 없습니다: $V3_CKPT"
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

started=0
for variant in "${VARIANTS[@]}"; do
  [ "$variant" = "$START" ] && started=1
  [ $started -eq 0 ] && continue

  config=configs/gemma3_12b_1b_3090x4_v3_abl_${variant}.yaml
  ckpt=results/gemma3/checkpoints/gemma3-12b-1b-v3-abl-${variant//_/-}

  # 4-way 평가가 Student (FT) 행을 채우려면 같은 SFT 체크포인트가 이 run 의 디렉터리에도 보여야 한다.
  mkdir -p "$ckpt"
  ln -sf "$PWD/$V3_CKPT/student_ft_best.pt" "$ckpt/student_ft_best.pt"

  run_stage "ABL $variant  KD" \
    $TORCHRUN scripts/gemma3_stage.py "$config" distill

  run_stage "ABL $variant  Evaluate + Compare" \
    bash -c "$PY scripts/gemma3_stage.py '$config' evaluate && $PY scripts/gemma3_stage.py '$config' compare"
done

run_stage "ABL  paired bootstrap" \
  $PY scripts/gemma3_paired_bootstrap.py configs/gemma3_12b_1b_3090x4_v3.yaml 10000 ablation

run_stage "ABL  2×2 요약표" \
  $PY scripts/gemma3_ablation_summary.py

echo ""
echo "=================================================================="
echo " ablation 요약  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=================================================================="
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "ABLATION_DONE"
