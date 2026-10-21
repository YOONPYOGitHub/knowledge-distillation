#!/usr/bin/env bash
# v4 실험 한 벌을 tmux 안에서 돌리는 실행 스크립트.
#   1. 자동 commit·push 감시(scripts/v4_git_watcher.py)를 켠다 (이미 켜져 있으면 건너뛴다)
#   2. config 의 Teacher / Student 모델을 받는다 (받아 둔 것은 건너뛴다)
#   3. v4 러너(scripts/run_gemma3_v4.sh)를 실행한다. 끝난 단계는 건너뛰고 끊긴 단계는 epoch 단위로 이어간다
#
# 워크스페이스를 재시작한 뒤에도 같은 명령을 다시 실행하면 된다.
#
# 사용법:
#   tmux new -d -s qwen3 "bash scripts/run_v4_session.sh configs/qwen3_8b_1.7b_3090x4_v4.yaml 'Qwen3 v4' 'Qwen3 8B -> 1.7B'; exec bash"
set -u
cd "$(dirname "$0")/.."

CONFIG=$1
LABEL=$2
TITLE=$3
RUN_ID=$(awk '/^run_id:/ {print $2}' "$CONFIG")
OUTPUT_DIR=$(awk '/^output_dir:/ {print $2}' "$CONFIG")
LOG=$OUTPUT_DIR/logs/$RUN_ID/pipeline.log
mkdir -p "$(dirname "$LOG")"

if ! pgrep -f "[v]4_git_watcher.py $CONFIG" >/dev/null; then
  setsid nohup .venv-gemma/bin/python scripts/v4_git_watcher.py "$CONFIG" --label "$LABEL" --title "$TITLE" \
    >> "$HOME/v4_git_watcher_$RUN_ID.log" 2>&1 < /dev/null &
fi

{
  echo "### model download $(date '+%Y-%m-%d %H:%M:%S')"
  .venv-gemma/bin/python - "$CONFIG" <<'PY'
import sys
import yaml
from huggingface_hub import snapshot_download
config = yaml.safe_load(open(sys.argv[1]))
for key in ("teacher_model", "student_model"):
    print(config[key], "->", snapshot_download(config[key]))
PY
  scripts/run_gemma3_v4.sh "$CONFIG"
} 2>&1 | tee -a "$LOG"
