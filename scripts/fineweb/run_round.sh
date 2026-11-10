#!/bin/bash
# usage: run_round.sh <family> <gpus> <round name> <trials file> [--seed N]
# Runs one tuning round. Resumes an existing round; retries only ENOSPC from the shared volume.
F=$1; G=$2; NAME=$3; TRIALS=$4; shift 4
R=/root/runs/fineweb4way; L=$R/$F/logs
cd /root/knowledge-distillation || exit 1
source $R/env.sh
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$G
until [ -f $R/.models_verified ]; do sleep 30; done
echo "[$NAME start $F] $(date -u +%FT%TZ)" | tee -a $L/chain.status
# A run that failed before freezing its plan leaves only a lock file; start that round afresh.
if [ -d $R/$F/$NAME ] && [ ! -f $R/$F/$NAME/plan.json ] && [ -z "$(ls -A $R/$F/$NAME | grep -v '^.run.lock$')" ]; then
  rm -rf $R/$F/$NAME
fi
for ATTEMPT in $(seq 1 12); do
  ARGS=()
  [ -f $R/$F/$NAME/plan.json ] && ARGS=(--resume)
  .venv-gemma/bin/python scripts/tune_research.py --base-run $R/$F/base --output-dir $R/$F/$NAME --trials-file "$TRIALS" \
    --max-steps 3000 --validation-interval 200 --checkpoint-interval 100 --patience 3 "$@" "${ARGS[@]}" >> $L/$NAME.json 2>> $L/$NAME.err
  RC=$?
  [ "$RC" = 0 ] && break
  tail -c 2000 $L/$NAME.err | grep -q 'Errno 28' || break
  echo "[$NAME retry $F attempt=$ATTEMPT shared volume full] $(date -u +%FT%TZ)" | tee -a $L/interruptions.txt
  sleep 300
done
echo "[$NAME end $F rc=$RC] $(date -u +%FT%TZ)" | tee -a $L/chain.status
exit $RC
