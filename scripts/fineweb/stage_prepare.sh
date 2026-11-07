#!/bin/bash
F=$1
cd /root/knowledge-distillation
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
R=/root/runs/fineweb4way; L=$R/$F/logs; mkdir -p $L
.venv-gemma/bin/python scripts/run_research.py prepare --config configs/fineweb_$F.yaml --run-dir $R/$F/base --exposure-exclusion-file $R/exposure-exclusion.json > $L/prepare.json 2> $L/prepare.err
RC=$?
echo "[prepare end $F rc=$RC] $(date -u +%FT%TZ)" | tee -a $L/prepare.status
exit $RC
