#!/bin/bash
# usage: chain_round1.sh <family> <cuda devices>
# Waits for the family's Teacher, probes KD memory briefly, then runs Round 1 (default trials).
# Only ENOSPC from the shared volume is retried; every other failure stops the chain.
F=$1; G=$2
R=/root/runs/fineweb4way
L=$R/$F/logs
cd /root/knowledge-distillation || exit 1
source /root/runs/fineweb4way/env.sh
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$G

until grep -q 'teacher end' "$L/teacher.status" 2>/dev/null; do sleep 60; done
if ! tail -1 "$L/teacher.status" | grep -q 'rc=0'; then
  echo "[chain $F] teacher failed" | tee -a "$L/chain.status"; exit 1
fi

# Keep only the selected and final Teacher checkpoints.
python3 - "$R/$F/base/teacher" "$L/teacher.json" <<'PY' | tee -a "$L/chain.status"
import json, os, shutil, sys
teacher, log = sys.argv[1:3]
result = json.load(open(log))['result']
keep = {os.path.basename(result['best_checkpoint']), os.path.basename(result['last_checkpoint'])}
root = os.path.join(teacher, 'checkpoints')
removed = 0
for name in os.listdir(root):
    if name not in keep:
        shutil.rmtree(os.path.join(root, name))
        removed += 1
print(f'[prune teacher] kept {sorted(keep)} removed {removed}')
PY

# Short CE+KD probe: fails early here rather than hours into Round 1.
P=$R/smoke/$F-kd-probe
for ATTEMPT in $(seq 1 6); do
  rm -rf "$P"; mkdir -p "$P"
  cat > "$P/trials.json" <<'J'
[{"id":"ce-1","method":"ce","learning_rate":8e-5,"alpha":1.0,"temperature":1.0,"divergence":"forward_kl"},
 {"id":"kd-1","method":"kd","learning_rate":8e-5,"alpha":0.7,"temperature":1.0,"divergence":"forward_kl"}]
J
  ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$G" | tr '\n' ' ' >> "$L/kd_probe_gpu.txt"; echo >> "$L/kd_probe_gpu.txt"; sleep 2; done ) & M=$!
  S=$(date +%s)
  .venv-gemma/bin/python scripts/tune_research.py --base-run "$R/$F/base" --output-dir "$P/sweep" --trials-file "$P/trials.json" \
    --seed 42 --max-steps 20 --validation-interval 20 --checkpoint-interval 20 --patience 3 > "$L/kd_probe.json" 2> "$L/kd_probe.err"
  RC=$?; kill $M
  [ "$RC" = 0 ] && break
  grep -q 'Errno 28' "$L/kd_probe.err" || break
  echo "[kd probe retry $F attempt=$ATTEMPT shared volume full] $(date -u +%FT%TZ)" | tee -a "$L/interruptions.txt"
  sleep 300
done
echo "[kd probe $F rc=$RC seconds=$(( $(date +%s) - S ))] peak MiB sample: $(sort -t' ' -k2 -n "$L/kd_probe_gpu.txt" | tail -1)" | tee -a "$L/chain.status"
rm -rf "$P"
[ "$RC" = 0 ] || exit 1

echo "[round1 start $F] $(date -u +%FT%TZ)" | tee -a "$L/chain.status"
ARGS=()
for ATTEMPT in $(seq 1 12); do
  .venv-gemma/bin/python scripts/tune_research.py --base-run "$R/$F/base" --output-dir "$R/$F/round1" \
    --seed 42 --max-steps 3000 --validation-interval 200 --checkpoint-interval 100 --patience 3 "${ARGS[@]}" >> "$L/round1.json" 2>> "$L/round1.err"
  RC=$?
  [ "$RC" = 0 ] && break
  tail -c 2000 "$L/round1.err" | grep -q 'Errno 28' || break
  echo "[round1 retry $F attempt=$ATTEMPT shared volume full] $(date -u +%FT%TZ)" | tee -a "$L/interruptions.txt"
  sleep 300
  ARGS=(--resume)
done
echo "[round1 end $F rc=$RC] $(date -u +%FT%TZ)" | tee -a "$L/chain.status"
exit $RC
