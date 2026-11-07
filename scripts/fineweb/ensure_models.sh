#!/bin/bash
R=/root/runs/fineweb4way
cd /root/knowledge-distillation
source "$R/env.sh"
.venv-gemma/bin/python "$R/download_verify.py" \
  Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218 \
  Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  kakaocorp/kanana-1.5-8b-instruct-2505@c963a5f4f6496c749f94064a20b33028b0db9f19 \
  kakaocorp/kanana-1.5-2.1b-instruct-2505@7df4bc35ccd610e451809d7106e1c3cf82bfd44c \
  google/gemma-3-12b-it@96b6f1eccf38110c56df3a15bffe176da04bfd80 \
  google/gemma-3-1b-it@dcc83ea841ab6100d6b47a070329e1ba4cf78752 \
  2>&1 | tee -a "$R/download_verify.log"
if [ "${PIPESTATUS[0]}" != 0 ]; then
  echo '[models not verified; Teachers not started]' | tee -a "$R/download_verify.log"; exit 1
fi
touch "$R/.models_verified"
