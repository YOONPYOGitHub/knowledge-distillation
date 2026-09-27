#!/usr/bin/env python3
"""Deploy/run/collect this isolated H100 experiment through authenticated Azure CLI.

Does not start/stop VMs, modify network rules, or overwrite the original checkout.
No cloud credentials are stored here; az uses the user's existing login.
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/var/lib/knowledge-distillation-h100-tuning-20260920"
RESULTS = "/var/lib/kd-results/h100-tuning-20260920"
LOCAL_RESULTS = ROOT / "results/logs/h100-tuning-20260920"


def az_json(args, command):
    result = subprocess.run(
        ["az", *command, "--subscription", args.subscription, "-o", "json"],
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else {}


def vm_state(args):
    vm = az_json(args, ["vm", "get-instance-view", "--resource-group", args.resource_group,
                        "--name", args.vm_name])
    power = next((item["code"] for item in vm.get("instanceView", {}).get("statuses", [])
                  if item["code"].startswith("PowerState/")), "PowerState/unknown")
    return {"power": power, "location": vm["location"], "priority": vm.get("priority"),
            "eviction_policy": vm.get("evictionPolicy")}


def job_state(args, name):
    job = az_json(args, ["vm", "run-command", "show", "--resource-group", args.resource_group,
                        "--vm-name", args.vm_name, "--run-command-name", name,
                        "--expand", "instanceView"])
    return {"name": name, "provisioning": job.get("provisioningState"),
            "execution": job.get("instanceView", {})}


def save_job(data):
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    path = LOCAL_RESULTS / "managed_job.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def status(args):
    result = {"vm": vm_state(args)}
    path = LOCAL_RESULTS / "managed_job.json"
    if path.exists():
        job = json.loads(path.read_text())
        result["job"] = job_state(args, job["name"])
        write_path = LOCAL_RESULTS / "managed_status.json"
        write_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def invoke(args, script):
    command = ["az", "vm", "run-command", "invoke", "--subscription", args.subscription,
               "--resource-group", args.resource_group, "--name", args.vm_name,
               "--command-id", "RunShellScript", "--scripts", script,
               "--query", "value[].message", "-o", "tsv"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return result.stdout


def deploy(args):
    paths = [ROOT / "main.py", ROOT / "configs/h100_qwen7b_korean_tuning.yaml",
             ROOT / "scripts/tune_h100.py", ROOT / "tests/test_h100_tuning.py"]
    paths += sorted((ROOT / "src").glob("*.py"))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=str(path.relative_to(ROOT)))
    data = archive.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    encoded = base64.b64encode(data).decode()
    script = f"""set -eu
test ! -e {shlex.quote(REMOTE)}
mkdir -p {shlex.quote(REMOTE)}
printf '%s' '{encoded}' | base64 -d | tar -xzf - -C {shlex.quote(REMOTE)}
printf '%s\\n' '{digest}' > {shlex.quote(REMOTE)}/deployment.sha256
cd {shlex.quote(REMOTE)}
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
/var/lib/kd-venv/bin/python -m unittest discover -s tests -v
/var/lib/kd-venv/bin/python -m compileall -q main.py src scripts
echo DEPLOY_AND_TESTS_PASSED
"""
    output = invoke(args, script)
    print(output, flush=True)
    if "DEPLOY_AND_TESTS_PASSED" not in output:
        raise RuntimeError("Azure deployment/tests failed; do not launch training")


def run(args):
    phase = args.phase
    if phase not in {"prepare", "smoke", "lr", "alpha", "temperature", "final", "all"}:
        raise ValueError("Unknown phase")
    vm = vm_state(args)
    if vm["power"] != "PowerState/running":
        raise RuntimeError(f"VM is {vm['power']}; possible Spot stop. No automatic VM start.")
    job_path = LOCAL_RESULTS / "managed_job.json"
    if job_path.exists():
        previous = json.loads(job_path.read_text())
        previous_state = job_state(args, previous["name"])
        execution = previous_state["execution"].get("executionState")
        if execution not in {"Succeeded", "Failed", "Canceled", "TimedOut"}:
            raise RuntimeError(f"Previous job is {execution or 'provisioning'}; use status, not run")
    name = "kd-tuning-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    print(f"Submitting managed H100 job {name}: phase={phase}", flush=True)
    script = f"""set -u
cd {shlex.quote(REMOTE)} || exit 1
export PYTHONPATH="$PWD" HF_HOME=/var/lib/huggingface
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
mkdir -p {shlex.quote(RESULTS)}/runner-logs
log={shlex.quote(RESULTS)}/runner-logs/{name}.log
flock -n /var/lib/kd-results/h100-tuning.lock timeout 5000 /var/lib/kd-venv/bin/python scripts/tune_h100.py --phase {phase} > "$log" 2>&1
ret=$?
tail -c 3000 "$log"
printf '\\nTUNING_EXIT_CODE=%s\\n' "$ret"
exit "$ret"
"""
    record = {"name": name, "phase": phase, "vm_name": args.vm_name,
              "resource_group": args.resource_group, "subscription": args.subscription,
              "location": vm["location"], "remote_log": f"{RESULTS}/runner-logs/{name}.log"}
    # Persist name before submission: a disconnected CLI must not lead to duplicate jobs.
    save_job(record)
    az_json(args, ["vm", "run-command", "create", "--resource-group", args.resource_group,
                   "--vm-name", args.vm_name, "--location", vm["location"],
                   "--run-command-name", name, "--async-execution", "true",
                   "--timeout-in-seconds", "5100", "--script", script])
    print(json.dumps(job_state(args, name), indent=2), flush=True)
    print("Submitted. Query with action=status; do NOT invoke another training job to monitor progress.", flush=True)


def collect(args):
    # Azure Run Command returns only a small output tail: transfer compressed chunks.
    name = args.artifact
    if not re.fullmatch(r"[a-zA-Z0-9_./-]+", name) or ".." in Path(name).parts:
        raise ValueError("Invalid relative artifact path")
    if not name.endswith((".json", ".log")):
        raise ValueError("Only small logs and JSON can be collected")
    remote = f"{RESULTS}/{name}"
    size_output = invoke(args, f"set -eu; printf 'SIZE='; stat -c %s {shlex.quote(remote)}")
    match = re.search(r"SIZE=(\d+)", size_output)
    if not match or int(match[1]) > 200000:
        raise ValueError("Missing or oversized artifact")
    size = int(match[1])
    content = bytearray()
    for offset in range(0, size, 1800):
        output = invoke(args, f"set -eu; echo DATA_BEGIN; dd if={shlex.quote(remote)} bs=1 skip={offset} count=1800 status=none | base64 -w0; echo; echo DATA_END")
        encoded = output.split("DATA_BEGIN\n", 1)[1].split("\nDATA_END", 1)[0]
        content.extend(base64.b64decode(encoded))
    if len(content) != size:
        raise RuntimeError("Incomplete artifact transfer")
    if name.endswith(".json"):
        json.loads(content)
    destination = ROOT / "results/logs/h100-tuning-20260920" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    print(f"Collected {name}: {size} bytes -> {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["deploy", "run", "status", "collect"])
    parser.add_argument("--phase", default="prepare")
    parser.add_argument("--artifact", default="test_results.json")
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--vm-name", required=True)
    args = parser.parse_args()
    {"deploy": deploy, "run": run, "status": status, "collect": collect}[args.action](args)


if __name__ == "__main__":
    main()