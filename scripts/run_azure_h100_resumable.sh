#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="${WORK_DIR:-/mnt/knowledge-distillation}"
VENV_DIR="${VENV_DIR:-/mnt/kd-venv}"
CONFIG="${1:-configs/h100_qwen7b_korean_4way.yaml}"

cd "${WORK_DIR}"

mapfile -t paths < <("${VENV_DIR}/bin/python" - "${CONFIG}" <<'PY'
import sys
from src.config import from_yaml

config = from_yaml(sys.argv[1])
print(config.teacher_checkpoint)
print(config.checkpoint_dir / "student_kd_best.pt")
print(config.checkpoint_dir / "student_ft_best.pt")
print(config.log_dir / "teacher_history.json")
print(config.log_dir / "distill_history.json")
print(config.log_dir / "baseline_history.json")
print(config.log_dir / "evaluation_results.json")
print(config.log_dir / "summary.json")
PY
)

teacher_checkpoint="${paths[0]}"
student_kd_checkpoint="${paths[1]}"
student_ft_checkpoint="${paths[2]}"
teacher_history="${paths[3]}"
distill_history="${paths[4]}"
baseline_history="${paths[5]}"
evaluation_results="${paths[6]}"
summary="${paths[7]}"

run_step() {
    echo "=== Running $1 ==="
    bash scripts/run_azure_h100.sh "${CONFIG}" --step "$1"
}

if [[ ! -e "${teacher_checkpoint}" || ! -f "${teacher_history}" ]]; then
    run_step train_teacher
else
    echo "=== Skipping completed train_teacher ==="
fi

if [[ ! -f "${student_kd_checkpoint}" || ! -f "${distill_history}" ]]; then
    run_step distill
else
    echo "=== Skipping completed distill ==="
fi

if [[ ! -f "${student_ft_checkpoint}" || ! -f "${baseline_history}" ]]; then
    run_step baseline
else
    echo "=== Skipping completed baseline ==="
fi

if [[ ! -f "${evaluation_results}" ]]; then
    run_step evaluate
else
    echo "=== Skipping completed evaluate ==="
fi

if [[ ! -f "${summary}" ]]; then
    run_step compare
else
    echo "=== Skipping completed compare ==="
fi