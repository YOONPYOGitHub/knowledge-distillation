#!/usr/bin/env python3
"""Keep a Spot H100 experiment running until all resumable stages complete."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/h100_qwen7b_korean_4way.yaml")
    parser.add_argument("--subscription", default="66679423-9d1a-4f45-8ae3-3078b8b62e99")
    parser.add_argument("--resource-group", default="rg-ai")
    parser.add_argument("--vm-name", default="vm-test-100")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument(
        "--runner",
        default="scripts/run_azure_h100_resumable.sh",
        choices=(
            "scripts/run_azure_h100_resumable.sh",
            "scripts/run_azure_h100_kd_retry.sh",
            "scripts/run_azure_h100_v4.sh",
        ),
    )
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def stamp(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def az_base(args: argparse.Namespace) -> list[str]:
    return [
        "az",
        "vm",
        "run-command",
        "invoke",
        "--subscription",
        args.subscription,
        "--resource-group",
        args.resource_group,
        "--name",
        args.vm_name,
        "--command-id",
        "RunShellScript",
    ]


def power_state(args: argparse.Namespace) -> str:
    result = run(
        [
            "az",
            "vm",
            "get-instance-view",
            "--subscription",
            args.subscription,
            "--resource-group",
            args.resource_group,
            "--name",
            args.vm_name,
            "--query",
            "instanceView.statuses[?starts_with(code, 'PowerState')].code | [0]",
            "--output",
            "tsv",
        ],
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def start_vm(args: argparse.Namespace) -> bool:
    result = run(
        [
            "az",
            "vm",
            "start",
            "--subscription",
            args.subscription,
            "--resource-group",
            args.resource_group,
            "--name",
            args.vm_name,
        ],
        check=False,
    )
    if result.returncode:
        stamp(f"VM start failed; retrying later: {result.stderr.strip()}")
        return False
    stamp("VM started")
    return True


def remote(args: argparse.Namespace, script: str, check: bool = True) -> str:
    result = run(
        az_base(args)
        + ["--scripts", script, "--query", "value[0].message", "--output", "tsv"],
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout if result.returncode == 0 else ""


def bootstrap(args: argparse.Namespace) -> None:
    branch = os.environ.get("BRANCH", "feature/azure-h100-korean-kd")
    script = f"""
set -eu
    repo=/var/lib/knowledge-distillation
if [ -d "${{repo}}/.git" ]; then
  git -C "${{repo}}" fetch origin '{branch}'
  git -C "${{repo}}" checkout '{branch}'
  git -C "${{repo}}" reset --hard 'origin/{branch}'
else
  git clone --branch '{branch}' --single-branch \
    https://github.com/YOONPYOGitHub/knowledge-distillation.git "${{repo}}"
fi
if [ ! -x /var/lib/kd-venv/bin/python ] || \
    ! /var/lib/kd-venv/bin/python -c 'import torch, peft, transformers, datasets' >/dev/null 2>&1; then
  bash "${{repo}}/scripts/setup_azure_h100.sh"
fi
mkdir -p /var/lib/kd-results
"""
    remote(args, f"bash -c {shell_quote(script)}")
    stamp("VM environment ready")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def inspect(args: argparse.Namespace, config_path: str) -> tuple[str, str]:
    config = from_yaml(config_path)
    summary = config.log_dir / "summary.json"
    log_file = Path(config.output_dir) / "jobs" / f"{config.run_id}.log"
    pid_file = log_file.with_suffix(".pid")
    script = f"""
set -eu
summary='{summary}'
log_file='{log_file}'
pid_file='{pid_file}'
if [ -f "${{summary}}" ]; then
  echo STATE=complete
elif [ -f "${{pid_file}}" ]; then
  pid=$(tr -d '"' < "${{pid_file}}")
    command=$(ps -p "${{pid}}" -o args= 2>/dev/null || true)
    if printf '%s' "${{command}}" | grep -Eq 'run_azure_h100(_kd_retry|_resumable)?\.sh|/mnt/kd-venv/bin/python main\.py|/var/lib/kd-venv/bin/python main\.py'; then
    echo STATE=running
  else
    echo STATE=stopped
  fi
else
  echo STATE=stopped
fi
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
if [ -f "${{log_file}}" ]; then
  python3 -c 'import sys; from pathlib import Path; text=Path(sys.argv[1]).read_bytes()[-32768:].decode("utf-8",errors="replace"); print("\\n".join(text.splitlines()[-8:]))' "${{log_file}}"
fi
"""
    output = remote(args, f"bash -c {shell_quote(script)}", check=False)
    state = "unknown"
    for line in output.splitlines():
        if line.startswith("STATE="):
            state = line.removeprefix("STATE=")
            break
    return state, output


def launch(args: argparse.Namespace, config_path: str) -> None:
    config = from_yaml(config_path)
    jobs_dir = Path(config.output_dir) / "jobs"
    log_file = jobs_dir / f"{config.run_id}.log"
    pid_file = log_file.with_suffix(".pid")
    script = f"""
set -eu
mkdir -p '{jobs_dir}'
cd /var/lib/knowledge-distillation
nohup env HF_HOME=/var/lib/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false \
    bash '{args.runner}' '{config_path}' \
  >> '{log_file}' 2>&1 < /dev/null &
pid=$!
echo "${{pid}}" > '{pid_file}'
echo "Started PID=${{pid}} LOG={log_file}"
"""
    output = remote(args, f"bash -c {shell_quote(script)}")
    stamp(output.strip().splitlines()[-1])


def sync_results(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/sync_azure_h100_results.py",
        args.config,
        "--subscription",
        args.subscription,
        "--resource-group",
        args.resource_group,
        "--vm-name",
        args.vm_name,
    ]
    result = run(command)
    print(result.stdout, end="")


def main() -> int:
    args = parse_args()
    os.chdir(ROOT_DIR)
    if args.interval <= 0 or args.max_restarts < 0:
        raise SystemExit("interval must be positive and max-restarts non-negative")

    restarts = 0
    while True:
        state = power_state(args)
        stamp(f"VM state: {state}")
        if state != "PowerState/running":
            if args.status_only:
                return 1
            if restarts >= args.max_restarts:
                raise RuntimeError("Maximum VM restart attempts reached")
            restarts += 1
            if not start_vm(args):
                time.sleep(args.interval)
                continue

        try:
            job_state, output = inspect(args, args.config)
            print(output, end="", flush=True)
            if job_state == "complete":
                sync_results(args)
                stamp("Experiment complete and artifacts synced")
                return 0
            if args.status_only:
                return 0
            if job_state != "running":
                bootstrap(args)
                launch(args, args.config)
        except RuntimeError as error:
            stamp(f"Azure operation failed; retrying: {error}")

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())