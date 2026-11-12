#!/bin/bash
# Run after every workspace restart: bash /root/runs/fineweb4way/after_restart.sh
set -u
R=/root/runs/fineweb4way
REPO=/root/knowledge-distillation
command -v tmux >/dev/null || { apt-get update -qq >/dev/null && apt-get install -y -qq tmux >/dev/null; }
command -v tmux >/dev/null || { echo 'tmux install failed'; exit 1; }
[ -f /usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf ] || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fonts-nanum >/dev/null  # Korean chart font
rm -f "$R/.models_verified"          # the container disk was wiped; verify the cache again
echo "$(date -d '+24 hours' +%FT%T)+00:00" > "$R/workspace_stop.txt"
python3 - "$R/workspace_stop.txt" <<'PY'
import datetime as dt, sys
p = sys.argv[1]
t = dt.datetime.fromisoformat(open(p).read().strip()).astimezone(dt.timezone(dt.timedelta(hours=9)))
open(p, 'w').write(t.isoformat(timespec='seconds'))
PY
bash "$R/start_progress.sh"
tmux kill-session -t models 2>/dev/null
tmux new -d -s models "$R/ensure_models.sh; exec bash"
echo "restart services up; models are being restored"
# Every family continues from wherever it stopped.
for pair in 'qwen3 0' 'kanana 1' 'gemma3 2,3'; do
  set -- $pair
  tmux kill-session -t "a-$1" 2>/dev/null
  tmux new -d -s "a-$1" "until [ -f $R/.models_verified ]; do sleep 30; done; $R/chain_after.sh $1 $2; exec bash"
done
