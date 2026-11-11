#!/bin/bash
# Restore the colleague base models, then check the common scorer on 20 documents per environment.
R=/root/runs/fineweb4way
cd /root/knowledge-distillation; source $R/env.sh; export TOKENIZERS_PARALLELISM=false
/root/venv-t5/bin/python $R/download_verify.py Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7 LiquidAI/LFM2.5-2.6B@654f9463ce32b05d0429d76fe1f580b27d4c1ac0 || exit 1
until [ -f $R/.models_verified ]; do sleep 30; done
T=$R/common/mechanics; rm -rf $T; mkdir -p $T
CUDA_VISIBLE_DEVICES=0 .venv-gemma/bin/python $R/common_score.py ours qwen3 kd-seed44 $T/qwen3.json --limit 20 &
CUDA_VISIBLE_DEVICES=1 /root/venv-t5/bin/python $R/common_score.py colleague qwen35 kd-seed42 $T/qwen35.json --limit 20 &
CUDA_VISIBLE_DEVICES=2 /root/venv-t5/bin/python $R/common_score.py colleague gemma4 ce-seed43 $T/gemma4.json --limit 20 &
CUDA_VISIBLE_DEVICES=3 /root/venv-t5/bin/python $R/common_score.py colleague lfm25 kd-seed42 $T/lfm25.json --limit 20 &
wait
ls $T
