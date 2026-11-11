#!/bin/bash
# Full common scoring under docs/small-model-kd-v1/COMMON_SCORING_RULE.md, after the mechanics check passes.
# Four GPU workers share one job queue; finished scores are kept, so a rerun after a restart resumes.
R=/root/runs/fineweb4way; C=$R/common; Q=$C/queue; OUT=$C/scores
cd /root/knowledge-distillation; source $R/env.sh; export TOKENIZERS_PARALLELISM=false

until grep -q '^rc=' $C/mechanics.log 2>/dev/null; do sleep 60; done
if ! grep -q '^rc=0' $C/mechanics.log || [ "$(ls $C/mechanics/*.json 2>/dev/null | wc -l)" != 4 ]; then
  echo "[common] mechanics check failed; full scoring not started" | tee -a $C/status.log; exit 1
fi
echo "[common] mechanics passed; full scoring start $(date -u +%FT%TZ)" | tee -a $C/status.log
mkdir -p $Q $OUT
MODELS="pretrained-student ce-seed42 kd-seed42 ce-seed43 kd-seed43 ce-seed44 kd-seed44"
JOBS=""
for f in qwen35; do for m in $MODELS; do JOBS="$JOBS colleague:$f:$m"; done; done   # slowest first
for f in gemma4 lfm25; do for m in $MODELS; do JOBS="$JOBS colleague:$f:$m"; done; done
for f in gemma3 qwen3 kanana; do for m in $MODELS; do JOBS="$JOBS ours:$f:$m"; done; done

worker() {
  local gpu=$1
  for job in $JOBS; do
    IFS=: read impl fam model <<< "$job"
    local out=$OUT/$fam--$model.json
    [ -f $out ] && continue
    mkdir $Q/$fam--$model 2>/dev/null || continue          # another worker owns this job
    local py=/root/venv-t5/bin/python; [ $impl = ours ] && py=.venv-gemma/bin/python
    CUDA_VISIBLE_DEVICES=$gpu $py $R/common_score.py $impl $fam $model $out.part >> $C/worker$gpu.log 2>&1 \
      && mv $out.part $out && echo "[common] $fam $model done gpu$gpu $(date -u +%FT%TZ)" >> $C/status.log \
      || { echo "[common] $fam $model FAILED gpu$gpu" | tee -a $C/status.log; rm -rf $Q/$fam--$model; }
  done
}
rm -rf $Q; mkdir -p $Q
for g in 0 1 2 3; do worker $g & done
wait
n=$(ls $OUT/*.json 2>/dev/null | wc -l)
echo "[common] scoring end: $n/42 scores $(date -u +%FT%TZ)" | tee -a $C/status.log
[ "$n" = 42 ] || exit 1
.venv-gemma/bin/python $R/common_analyze.py | tee -a $C/status.log
