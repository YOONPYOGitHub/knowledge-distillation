#!/usr/bin/env bash
# Gemma 3 12B -> 1B v4 러너. RTX 3090 x 4.
#
# v4 KD 는 Base 초기값이라 Student SFT 에 의존하지 않는다. 그래서 두 학습을 동시에 돌린다.
#   1. Teacher FT (GPU 0,1)  ‖  Student SFT (GPU 2,3)
#   2. KD (Student GPU 0,1 / Teacher GPU 2,3, 양자화는 config 의 teacher_quantization)
#   3. 평가 -> 비교 -> paired bootstrap
# 순서대로 돌리는 run_gemma3_4way.sh 보다 Student SFT 시간(수 시간)만큼 짧다.
#
# 다시 실행하면 이어간다. 끝난 단계는 산출물로 판단해 건너뛰고,
# 중간에 끊긴 학습 단계는 resume_training 으로 마지막 epoch 다음부터 이어간다.
# 한 단계라도 실패하면 조건을 낮추지 않고 멈춘다.
#
# 결과 경로는 config 의 output_dir / run_id 를 따르므로 다른 계열(Kanana 등)의 v4 설정에도 쓴다.
#
# 사용법:
#   setsid nohup scripts/run_gemma3_v4.sh >> results/gemma3/logs/gemma3-12b-1b-v4/pipeline.log 2>&1 &
#   setsid nohup scripts/run_gemma3_v4.sh configs/kanana_8b_2b_3090x4_v4.yaml \
#     >> results/kanana/logs/kanana-8b-2b-v4/pipeline.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/gemma3_12b_1b_3090x4_v4.yaml}
RUN_ID=$(awk '/^run_id:/ {print $2}' "$CONFIG")
OUTPUT_DIR=$(awk '/^output_dir:/ {print $2}' "$CONFIG")
LOGS=$OUTPUT_DIR/logs/$RUN_ID
CKPT=$OUTPUT_DIR/checkpoints/$RUN_ID

PY="$PWD/.venv-gemma/bin/python"
TORCHRUN="$PWD/.venv-gemma/bin/torchrun --nproc_per_node=2"
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
# 2.1B Student 전체 학습에서 할당 단편화로 OOM 이 난다 (사용 11.6GB, 예약만 된 빈 공간 11.7GB).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOGS"
echo ""
echo "############ v4 러너 시작 — $(date '+%Y-%m-%d %H:%M:%S') ############"

banner() {
  echo ""
  echo "=================================================================="
  echo " $1  ($(date '+%H:%M:%S'))"
  echo "=================================================================="
}

fail() {
  echo "❌ $1 실패로 멈춥니다. 원인을 해결한 뒤 같은 명령으로 다시 실행하면 이어갑니다."
  echo "PIPELINE_FAILED"
  exit 1
}

# 학습 단계는 history 가 저장되고 재개 파일이 지워졌으면 끝난 것이다.
trained() {
  [ -f "$LOGS/$1_history.json" ] && [ ! -f "$CKPT/$1_last.pt" ]
}

# --- 1. Teacher FT ‖ Student SFT ---
pids=()
if trained teacher; then
  echo "⏭️  Teacher FT 완료됨"
else
  banner "STEP 1a  Teacher FT (QLoRA nf4, GPU 0,1)"
  CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --master_port=29581 \
    scripts/gemma3_stage.py "$CONFIG" train_teacher >> "$LOGS/teacher.log" 2>&1 &
  pids+=("$!:Teacher FT")
fi
if trained baseline; then
  echo "⏭️  Student SFT 완료됨"
else
  banner "STEP 1b  Student SFT (GPU 2,3)"
  CUDA_VISIBLE_DEVICES=2,3 $TORCHRUN --master_port=29582 \
    scripts/gemma3_stage.py "$CONFIG" baseline >> "$LOGS/baseline.log" 2>&1 &
  pids+=("$!:Student SFT")
fi
status=0
for entry in "${pids[@]}"; do
  pid=${entry%%:*}; name=${entry#*:}
  if wait "$pid"; then echo "✅ $name ($(date '+%H:%M:%S'))"; else echo "❌ $name"; status=1; fi
done
[ $status -eq 0 ] || fail "STEP 1"
trained teacher && trained baseline || fail "STEP 1 (산출물 없음)"

# --- 2. KD ---
if trained distill; then
  echo "⏭️  KD 완료됨"
else
  banner "STEP 2  KD (Student GPU 0,1 / Teacher GPU 2,3)"
  $TORCHRUN --master_port=29583 scripts/gemma3_stage.py "$CONFIG" distill \
    >> "$LOGS/distill.log" 2>&1 || fail "STEP 2 KD"
  echo "✅ KD ($(date '+%H:%M:%S'))"
fi

# --- 3. 평가 / 비교 / bootstrap ---
banner "STEP 3  평가 + 비교 + paired bootstrap"
$PY scripts/gemma3_stage.py "$CONFIG" evaluate > "$LOGS/evaluate.log" 2>&1 || fail "평가"
$PY scripts/gemma3_stage.py "$CONFIG" compare >> "$LOGS/evaluate.log" 2>&1 || fail "비교"
$PY scripts/gemma3_paired_bootstrap.py "$CONFIG" 10000 run > "$LOGS/bootstrap.log" 2>&1 || fail "bootstrap"
echo "✅ 평가 / 비교 / bootstrap ($(date '+%H:%M:%S'))"

echo ""
echo "PIPELINE_DONE  $(date '+%Y-%m-%d %H:%M:%S')"
