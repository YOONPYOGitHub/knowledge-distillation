#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="${WORK_DIR:-/mnt/knowledge-distillation}"
VENV_DIR="${VENV_DIR:-/mnt/kd-venv}"
CONFIG="${1:-configs/h100_qwen7b_korean_t4_tokenmean.yaml}"

cd "${WORK_DIR}"

mapfile -t paths < <("${VENV_DIR}/bin/python" - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
from src.config import from_yaml

config = from_yaml(sys.argv[1])
reference_dir = Path("/var/lib/kd-results/qwen7b-4way")
reference_run = "qwen7b-korean-4way-v1"
print(config.checkpoint_dir / "student_kd_best.pt")
print(config.log_dir / "distill_history.json")
print(config.log_dir / "evaluation_results.json")
print(config.log_dir / "summary.json")
print(reference_dir / "checkpoints" / reference_run / "student_ft_best.pt")
print(reference_dir / "logs" / reference_run / "teacher_history.json")
print(reference_dir / "logs" / reference_run / "baseline_history.json")
PY
)

student_kd_checkpoint="${paths[0]}"
distill_history="${paths[1]}"
evaluation_results="${paths[2]}"
summary="${paths[3]}"
reference_ft_checkpoint="${paths[4]}"
reference_teacher_history="${paths[5]}"
reference_baseline_history="${paths[6]}"

mkdir -p "$(dirname "${student_kd_checkpoint}")" "$(dirname "${distill_history}")"

if [[ ! -f "${student_kd_checkpoint}" || ! -f "${distill_history}" ]]; then
    bash scripts/run_azure_h100.sh "${CONFIG}" --step distill
fi

if [[ ! -f "${reference_ft_checkpoint}" || ! -f "${reference_teacher_history}" || ! -f "${reference_baseline_history}" ]]; then
    echo "Reference FT artifacts are missing" >&2
    exit 1
fi

ln -sfn "${reference_ft_checkpoint}" "$(dirname "${student_kd_checkpoint}")/student_ft_best.pt"
ln -sfn "${reference_teacher_history}" "$(dirname "${distill_history}")/teacher_history.json"
ln -sfn "${reference_baseline_history}" "$(dirname "${distill_history}")/baseline_history.json"

if [[ ! -f "${evaluation_results}" ]]; then
    bash scripts/run_azure_h100.sh "${CONFIG}" --step evaluate
fi

if [[ ! -f "${summary}" ]]; then
    bash scripts/run_azure_h100.sh "${CONFIG}" --step compare
fi