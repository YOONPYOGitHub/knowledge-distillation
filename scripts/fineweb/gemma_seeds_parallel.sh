#!/bin/bash
# Check that Gemma 3 training gives identical results on GPUs 0,1 and 2,3, then run seeds 43 and 44.
# Identical -> seed 43 on 2,3 and seed 44 on 0,1 in parallel. Otherwise -> both on 2,3, one after the other.
R=/root/runs/fineweb4way; F=gemma3; L=$R/$F/logs
cd /root/knowledge-distillation || exit 1
source $R/env.sh
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false

until [ -f $R/.models_verified ]; do sleep 30; done
while pgrep -f '[c]olleague_restore_check.py|[d]ownload_verify.py' >/dev/null; do sleep 30; done
echo "[determinism check start] $(date -u +%FT%TZ)" | tee -a $L/chain.status

P=$R/smoke/gemma3-determinism; rm -rf $P; mkdir -p $P
python3 - "$R/$F/seed_trials.json" "$P/trials.json" <<'PY'
import json, sys
kd = [t for t in json.load(open(sys.argv[1])) if t['method'] == 'kd'][0]
ce = {'id': 'ce-1', 'method': 'ce', 'learning_rate': kd['learning_rate'], 'alpha': 1.0,
      'temperature': 1.0, 'divergence': 'forward_kl'}
json.dump([ce, kd], open(sys.argv[2], 'w'))
PY
for pair in a:0,1 b:2,3; do
  name=${pair%%:*}; gpus=${pair#*:}
  CUDA_VISIBLE_DEVICES=$gpus .venv-gemma/bin/python scripts/tune_research.py --base-run $R/$F/base \
    --output-dir $P/$name --trials-file $P/trials.json --seed 44 --max-steps 20 --validation-interval 20 \
    --checkpoint-interval 20 --patience 3 > $P/$name.json 2> $P/$name.err &
done
wait
CUDA_VISIBLE_DEVICES= .venv-gemma/bin/python - "$P" <<'PY' | tee -a $L/chain.status
import json, sys, torch
from pathlib import Path
from src.resume_checkpoint import load_checkpoint
P = Path(sys.argv[1])
same = True
for trial in ('ce-1', 'kd-1'):
    try:
        ra = json.loads((P / 'a/trials' / trial / 'result.json').read_text())['result']
        rb = json.loads((P / 'b/trials' / trial / 'result.json').read_text())['result']
        sa = load_checkpoint(P / 'a/trials' / trial)['model_state']
        sb = load_checkpoint(P / 'b/trials' / trial)['model_state']
    except Exception as exc:
        print(f'[determinism {trial}] could not compare: {type(exc).__name__}: {exc}')
        same = False
        continue
    hist = [(h['step'], h['ce']) for h in ra['history']] == [(h['step'], h['ce']) for h in rb['history']]
    weights = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    print(f"[determinism {trial}] validation history identical={hist} weights bit-identical={weights} "
          f"(GPU0,1 CE {ra['history'][-1]['ce']:.10f} / GPU2,3 CE {rb['history'][-1]['ce']:.10f})")
    same = same and hist and weights
print('[determinism result] ' + ('IDENTICAL' if same else 'DIFFERENT'))
PY
rm -rf $P

if tail -1 $L/chain.status | grep -q IDENTICAL; then
  echo "[gemma3 seeds] parallel: seed43 on 2,3 and seed44 on 0,1" | tee -a $L/chain.status
  $R/run_round.sh $F 2,3 seed43 $R/$F/seed_trials.json --seed 43 &
  $R/run_round.sh $F 0,1 seed44 $R/$F/seed_trials.json --seed 44 &
  wait
else
  echo "[gemma3 seeds] sequential on 2,3" | tee -a $L/chain.status
  $R/chain_after.sh $F 2,3
fi
[ -f $R/$F/seed43/selection.json ] && [ -f $R/$F/seed44/selection.json ] && \
  echo "[chain-after gemma3] training complete $(date -u +%FT%TZ)" | tee -a $L/chain.status
