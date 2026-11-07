#!/bin/bash
# usage: stage_teacher.sh <family> <gpus> [--resume]
# /root sits on a volume shared with other workspaces, so writes can fail with ENOSPC
# while this workspace has room. Only that error is retried, always from the last checkpoint.
F=$1; G=$2; shift 2
cd /root/knowledge-distillation
source /root/runs/fineweb4way/env.sh
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$G
R=/root/runs/fineweb4way; L=$R/$F/logs; mkdir -p $L
( while true; do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $G | tr '\n' ' ')" >> $L/teacher_gpu.txt; sleep 30; done ) & M=$!
echo "[teacher start $F gpu$G] $(date -u +%FT%TZ)" | tee -a $L/teacher.status
ARGS=("$@")
for ATTEMPT in $(seq 1 12); do
  .venv-gemma/bin/python scripts/run_research.py teacher --config configs/fineweb_$F.yaml --run-dir $R/$F/base --exposure-exclusion-file $R/exposure-exclusion.json "${ARGS[@]}" >> $L/teacher.json 2>> $L/teacher.err
  RC=$?
  [ "$RC" = 0 ] && break
  tail -c 2000 $L/teacher.err | grep -q 'Errno 28' || break
  echo "[teacher retry $F attempt=$ATTEMPT shared volume full] $(date -u +%FT%TZ)" | tee -a $L/interruptions.txt
  sleep 300
  ARGS=(--resume)
done
kill $M
echo "[teacher end $F rc=$RC] $(date -u +%FT%TZ)" | tee -a $L/teacher.status
exit $RC
