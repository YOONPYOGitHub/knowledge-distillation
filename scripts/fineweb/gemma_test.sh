#!/bin/bash
# After Gemma 3 seeds 43 and 44 finish: reproduce training validation with the evaluator,
# then freeze and push the test protocol, score test once, and push the tables.
# Any mismatch stops before test is opened.
R=/root/runs/fineweb4way; F=gemma3; L=$R/$F/logs
REPO=/root/knowledge-distillation
cd $REPO || exit 1
source $R/env.sh
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2,3
until [ -f $R/$F/seed43/selection.json ] && [ -f $R/$F/seed44/selection.json ]; do sleep 60; done
while pgrep -f "[t]une_research.py.*gemma3" >/dev/null; do sleep 30; done
[ -f $R/$F/final_selection.json ] || { echo "[gemma3 test] no final selection" | tee -a $L/chain.status; exit 1; }

echo "[gemma3 test] validation reproduction check $(date -u +%FT%TZ)" | tee -a $L/chain.status
.venv-gemma/bin/python scripts/fineweb/evaluate_test.py check gemma3 teacher ce-seed43 kd-seed43 > $L/test_check.log 2>&1
python3 - "$L/test_check.log" <<'PY' | tee -a $L/chain.status
import re, sys
lines = [l for l in open(sys.argv[1], errors='replace') if 'training log validation PPL' in l]
ok = len(lines) == 3
for line in lines:
    m = re.search(r'validation PPL ([0-9.]+).*training log validation PPL ([0-9.]+)', line)
    ok = ok and m is not None and abs(float(m.group(1)) - float(m.group(2))) < 1e-3
    print('[gemma3 check] ' + line.strip())
print('[gemma3 check] ' + ('REPRODUCED' if ok else 'MISMATCH'))
PY
tail -1 $L/chain.status | grep -q REPRODUCED || exit 1

CUDA_VISIBLE_DEVICES= .venv-gemma/bin/python scripts/fineweb/evaluate_test.py freeze gemma3 | tee -a $L/chain.status || exit 1
for i in 1 2 3; do
  git add -- results/small-model-v1/test-protocol/gemma3.json && \
  git commit -q -m "eval: freeze the Gemma 3 test scoring protocol

Settings were chosen on validation and frozen before this commit. The evaluator
reproduced the Gemma 3 training validation PPL before test was opened.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- results/small-model-v1/test-protocol/gemma3.json && break
  sleep 20
done
git push -q origin feature/fineweb-small-model-kd || { echo "[gemma3 test] protocol push failed" | tee -a $L/chain.status; exit 1; }

.venv-gemma/bin/python scripts/fineweb/evaluate_test.py score gemma3 > $L/test_score.log 2>&1
RC=$?
echo "[gemma3 test score rc=$RC] $(date -u +%FT%TZ)" | tee -a $L/chain.status
[ "$RC" = 0 ] || exit 1
python3 scripts/fineweb/make_tables.py | tee -a $L/chain.status
for i in 1 2 3; do
  git add -- results/small-model-v1 && git commit -q -m "results: Gemma 3 test scores under the frozen protocol

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- results/small-model-v1 && break
  sleep 20
done
git push -q origin feature/fineweb-small-model-kd && echo "[gemma3 test] results pushed" | tee -a $L/chain.status
