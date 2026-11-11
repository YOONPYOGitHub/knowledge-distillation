#!/bin/bash
# Common scoring needs Student weights only; the large Teachers are not downloaded.
R=/root/runs/fineweb4way
cd /root/knowledge-distillation; source $R/env.sh
.venv-gemma/bin/python $R/download_verify.py Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e kakaocorp/kanana-1.5-2.1b-instruct-2505@7df4bc35ccd610e451809d7106e1c3cf82bfd44c google/gemma-3-1b-it@dcc83ea841ab6100d6b47a070329e1ba4cf78752 2>&1 | tee -a $R/download_verify.log
[ "${PIPESTATUS[0]}" = 0 ] && touch $R/.models_verified || echo "[student models not verified]" | tee -a $R/download_verify.log
