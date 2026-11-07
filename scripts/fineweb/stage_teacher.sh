#!/bin/bash
# usage: stage_teacher.sh <family> <gpu> [--resume]
F=$1; G=$2; shift 2
cd /root/knowledge-distillation
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$G
R=/root/runs/fineweb4way; L=$R/$F/logs; mkdir -p $L
( while true; do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $G)" >> $L/teacher_gpu.txt; sleep 30; done ) & M=$!
echo "[teacher start $F gpu$G] $(date -u +%FT%TZ)" | tee -a $L/teacher.status
.venv-gemma/bin/python scripts/run_research.py teacher --config configs/fineweb_$F.yaml --run-dir $R/$F/base --exposure-exclusion-file $R/exposure-exclusion.json "$@" >> $L/teacher.json 2>> $L/teacher.err
RC=$?; kill $M
echo "[teacher end $F rc=$RC] $(date -u +%FT%TZ)" | tee -a $L/teacher.status
