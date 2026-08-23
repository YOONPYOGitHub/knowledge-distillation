#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="${WORK_DIR:-/mnt/knowledge-distillation}"
VENV_DIR="${VENV_DIR:-/mnt/kd-venv}"
HF_HOME="${HF_HOME:-/mnt/huggingface}"
CONFIG="${1:-configs/h100_qwen_korean_smoke.yaml}"
shift || true

export HF_HOME
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

cd "${WORK_DIR}"
exec "${VENV_DIR}/bin/python" main.py "${CONFIG}" "$@"