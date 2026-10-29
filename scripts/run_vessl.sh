#!/usr/bin/env bash
# Two student ranks + two dedicated teachers. Default action only checks readiness.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
CONFIG="${1:-configs/vessl_qwen7b_korean_3090x4.yaml}"
STEP="${2:-check}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export KD_DATA_CACHE="${KD_DATA_CACHE:-${HOME}/.cache/kd-data}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${WORLD_SIZE:-1}" != 1 ]]; then
    echo "Run this launcher directly; it creates the correct number of student ranks." >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Remote Python environment not found: ${PYTHON_BIN}" >&2
    exit 2
fi

"${PYTHON_BIN}" - "${CONFIG}" "${STEP}" <<'PY'
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import from_yaml

config = from_yaml(sys.argv[1])
step = sys.argv[2]
if step not in {"check", "smoke", "distill", "evaluate", "compare"}:
    raise SystemExit("Usage: bash scripts/run_vessl.sh [config] [check|smoke|distill|evaluate|compare]")
print(json.dumps(asdict(config), indent=2, ensure_ascii=False))
print(f"torch={torch.__version__}, CUDA={torch.version.cuda}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(all)')}")
problems = []
nproc = len(config.teacher_devices) or 1
if config.global_batch_size and config.batch_size * nproc != config.global_batch_size:
    problems.append("batch_size * student ranks must equal global_batch_size")
if config.teacher_devices and set(config.teacher_devices) & {f"cuda:{i}" for i in range(nproc)}:
    problems.append("Teacher devices must not overlap student devices")
if not torch.cuda.is_available():
    problems.append("CUDA is unavailable")
else:
    devices = ({f"cuda:{i}" for i in range(nproc)} | set(config.teacher_devices)
               if config.teacher_devices else {config.device, config.teacher_device})
    for name in sorted(devices):
        device = torch.device(name)
        if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
            problems.append(f"Explicit, available CUDA device required: {name}")
            continue
        with torch.cuda.device(device):
            print(f"{name}: {torch.cuda.get_device_name(device)}, bf16={torch.cuda.is_bf16_supported()}")
            if config.bf16 and not torch.cuda.is_bf16_supported():
                problems.append(f"BF16 unsupported on {name}")

if step == "smoke":
    if torch.cuda.device_count() < 4:
        problems.append("Four visible GPUs are required for the paired smoke test")
    if problems:
        raise SystemExit("; ".join(problems))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node=2", "scripts/vessl_3090x4_smoke.py"]
    raise SystemExit(subprocess.run(command, timeout=180).returncode)

teacher = Path(config.teacher_checkpoint)
adapter_weights = [teacher / "adapter_model.safetensors", teacher / "adapter_model.bin"]
if not (teacher.is_file() or ((teacher / "adapter_config.json").is_file() and any(p.is_file() for p in adapter_weights))):
    problems.append(f"Missing original Azure Teacher checkpoint: {teacher}")
if not config.distill_student_checkpoint or not Path(config.distill_student_checkpoint).is_file():
    problems.append(f"Missing original Azure Student FT checkpoint: {config.distill_student_checkpoint}")
if step == "evaluate":
    for name in ("student_kd_best.pt", "student_ft_best.pt"):
        if not (config.checkpoint_dir / name).is_file():
            problems.append(f"Missing 4-way comparison checkpoint: {config.checkpoint_dir / name}")
if step == "compare" and not (config.log_dir / "evaluation_results.json").is_file():
    problems.append("Run evaluate before compare")
for problem in problems:
    print(f"BLOCKED: {problem}")
if step == "check":
    print("READY" if not problems else "NOT READY: no training started")
    raise SystemExit(0)
if problems:
    raise SystemExit("Preflight blocked execution; no model downloads or training started")

from main import run_pipeline

# Training random seeds were NOT initialized by the Azure runner. Explicit seeding
# here is a documented portability improvement, not a bitwise v3 reproduction.
from transformers import set_seed

set_seed(config.training_seed if config.training_seed is not None else config.seed)
if step == "distill" and (config.checkpoint_dir / "student_kd_best.pt").exists():
    raise SystemExit("Existing run checkpoint: choose a NEW run_id before retraining")
if step == "evaluate":
    # Same evaluation batch as H100; evaluate the full test set on a single GPU.
    config.batch_size = config.global_batch_size or config.batch_size
    config.teacher_device = config.device
    config.teacher_devices = []
config.ensure_dirs()
if step == "distill":
    # Four-way evaluation must use the same FT checkpoint used to initialize KD.
    ft_link = config.checkpoint_dir / "student_ft_best.pt"
    source = Path(config.distill_student_checkpoint).resolve()
    if ft_link.exists() and ft_link.resolve() != source:
        raise SystemExit(f"Conflicting FT comparison checkpoint: {ft_link}")
    if not ft_link.exists():
        ft_link.symlink_to(source)
with (config.log_dir / f"config_{step}.json").open("w") as file:
    json.dump(asdict(config), file, indent=2)
if step == "distill" and config.teacher_devices:
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={nproc}", "main.py", sys.argv[1], "--step", "distill"]
    raise SystemExit(subprocess.run(command).returncode)
run_pipeline(config, steps=[step])
PY