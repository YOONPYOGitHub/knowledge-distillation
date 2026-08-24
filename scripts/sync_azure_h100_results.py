#!/usr/bin/env python3
"""Download small H100 run artifacts and regenerate charts locally."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.compare import (
    plot_perplexity,
    plot_speed,
    plot_training_curves,
    plot_val_loss_curves,
)
from src.config import from_yaml


ARTIFACTS = (
    "teacher_history.json",
    "distill_history.json",
    "baseline_history.json",
    "evaluation_results.json",
    "summary.json",
    "notes.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Local YAML config path")
    parser.add_argument("--subscription", default="66679423-9d1a-4f45-8ae3-3078b8b62e99")
    parser.add_argument("--resource-group", default="rg-ai")
    parser.add_argument("--vm-name", default="vm-test-100")
    return parser.parse_args()


def run_az(args: argparse.Namespace, script: str) -> str:
    command = [
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
        "--scripts",
        script,
        "--query",
        "value[0].message",
        "--output",
        "tsv",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def download_artifacts(args: argparse.Namespace, remote_dir: Path, local_dir: Path) -> None:
    names = " ".join(f"'{name}'" for name in ARTIFACTS)
    script = f"""
set -eu
cd '{remote_dir}'
echo ARCHIVE_BASE64_BEGIN
tar -czf - {names} | base64 -w0
echo
echo ARCHIVE_BASE64_END
"""
    output = run_az(args, script)
    start = output.find("ARCHIVE_BASE64_BEGIN")
    end = output.find("ARCHIVE_BASE64_END")
    if start < 0 or end < 0:
        raise RuntimeError(f"Artifact markers missing from Azure output: {output[-1000:]}")
    encoded = output[start + len("ARCHIVE_BASE64_BEGIN") : end].strip()
    archive = base64.b64decode(encoded)

    local_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = tar.getmembers()
        unexpected = {member.name for member in members} - set(ARTIFACTS)
        if unexpected or any(not member.isfile() for member in members):
            raise RuntimeError(f"Unexpected archive members: {unexpected}")
        for member in members:
            if member.name == "notes.md" and (local_dir / member.name).exists():
                continue
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read {member.name}")
            (local_dir / member.name).write_bytes(source.read())


def validate_and_plot(config_path: str, run_id: str) -> None:
    log_dir = Path("results/logs") / run_id
    for name in ARTIFACTS[:-1]:
        with (log_dir / name).open(encoding="utf-8") as file:
            json.load(file)

    config = from_yaml(config_path, output_dir="results", run_id=run_id, device="cpu")
    with (log_dir / "evaluation_results.json").open(encoding="utf-8") as file:
        results = json.load(file)

    figure_dir = config.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_perplexity(results, str(figure_dir / "perplexity_comparison.png"))
    plot_speed(results, str(figure_dir / "inference_speed.png"))
    plot_training_curves(config, str(figure_dir / "training_loss.png"))
    plot_val_loss_curves(config, str(figure_dir / "val_loss.png"))


def main() -> int:
    args = parse_args()
    os.chdir(ROOT_DIR)
    config = from_yaml(args.config)
    remote_dir = config.log_dir
    local_dir = Path("results/logs") / config.run_id
    download_artifacts(args, remote_dir, local_dir)
    validate_and_plot(args.config, config.run_id)
    print(f"Logs: {local_dir}")
    print(f"Figures: results/figures/{config.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())