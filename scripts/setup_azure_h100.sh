#!/usr/bin/env bash

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/YOONPYOGitHub/knowledge-distillation.git}"
BRANCH="${BRANCH:-feature/azure-h100-korean-kd}"
WORK_DIR="${WORK_DIR:-/mnt/knowledge-distillation}"
VENV_DIR="${VENV_DIR:-/mnt/kd-venv}"
HF_HOME="${HF_HOME:-/mnt/huggingface}"

export HF_HOME
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

mkdir -p /mnt/kd-results "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

if [[ -d "${WORK_DIR}/.git" ]]; then
    git -C "${WORK_DIR}" fetch origin "${BRANCH}"
    git -C "${WORK_DIR}" checkout "${BRANCH}"
    git -C "${WORK_DIR}" reset --hard "origin/${BRANCH}"
else
    git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${WORK_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch
"${VENV_DIR}/bin/python" -m pip install -r "${WORK_DIR}/requirements.txt"

"${VENV_DIR}/bin/python" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY