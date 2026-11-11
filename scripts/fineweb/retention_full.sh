#!/bin/bash
# Retention evaluation under docs/small-model-kd-v1/RETENTION_RULE.md. Four GPU workers share one queue;
# finished scores are kept, so a rerun after a restart resumes.
R=/root/runs/fineweb4way; T=$R/retention; Q=$T/queue; OUT=$T/scores
cd /root/knowledge-distillation; source $R/env.sh; export TOKENIZERS_PARALLELISM=false
mkdir -p $OUT; rm -rf $Q; mkdir -p $Q
echo "[retention] start $(date -u +%FT%TZ)" | tee -a $T/status.log
MODELS="pretrained-student ce-seed42 kd-seed42 ce-seed43 kd-seed43 ce-seed44 kd-seed44"
JOBS=""
for f in qwen35 gemma4 lfm25; do for m in $MODELS; do JOBS="$JOBS colleague:$f:$m"; done; done
for f in gemma3 qwen3 kanana; do for m in $MODELS; do JOBS="$JOBS ours:$f:$m"; done; done
worker() {
  local gpu=$1
  for job in $JOBS; do
    IFS=: read impl fam model <<< "$job"
    local out=$OUT/$fam--$model.json
    [ -f $out ] && continue
    mkdir $Q/$fam--$model 2>/dev/null || continue
    local py=/root/venv-t5/bin/python; [ $impl = ours ] && py=.venv-gemma/bin/python
    CUDA_VISIBLE_DEVICES=$gpu $py $R/retention_score.py $impl $fam $model $out.part >> $T/worker$gpu.log 2>&1 \
      && mv $out.part $out && echo "[retention] $fam $model done gpu$gpu $(date -u +%FT%TZ)" >> $T/status.log \
      || { echo "[retention] $fam $model FAILED gpu$gpu" | tee -a $T/status.log; rm -rf $Q/$fam--$model; }
  done
}
for g in 0 1 2 3; do worker $g & done
wait
n=$(ls $OUT/*.json 2>/dev/null | wc -l)
echo "[retention] end: $n/42 $(date -u +%FT%TZ)" | tee -a $T/status.log
[ "$n" = 42 ] && .venv-gemma/bin/python $R/retention_analyze.py | tee -a $T/status.log
