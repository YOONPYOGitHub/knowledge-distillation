#!/usr/bin/env python3
"""Read-only H100 artifact snapshots, local reports, and completion watcher.

No checkpoint/model transfers, VM restarts, or training submissions. Unlike the
old fixed-file sync, accepts both provisional and final tuning artifacts.
"""

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import azure_h100_tuning as azure
from src.tuning_report import render_tuning_report

SUITE = "h100-tuning-20260920"
ALLOWED = {".json", ".log", ".md", ".yaml", ".csv"}
MAX_UNPACKED = 20 * 1024 ** 2


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def extract_snapshot(payload: bytes, log_dir: Path) -> int:
    """Only regular, bounded, relative text artifacts; reject links/traversal."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = log_dir.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 512 or sum(m.size for m in members) > MAX_UNPACKED:
            raise ValueError("Oversized artifact snapshot")
        parsed = []
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or path.suffix not in ALLOWED or "data" in path.parts
                    or not path.parts or member.name in names):
                raise ValueError(f"Unsafe artifact: {member.name}")
            names.add(member.name)
            destination = log_dir.joinpath(*path.parts)
            if not destination.resolve().is_relative_to(root):
                raise ValueError("Local artifact path escapes results directory")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Missing archive contents")
            contents = source.read()
            if path.suffix == ".json":
                json.loads(contents)
            parsed.append((destination, contents))
        # Validate the complete archive before writing anything.
        for destination, contents in parsed:
            if destination.name == "notes.md" and destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            if temporary.is_symlink():
                raise ValueError("Temporary artifact path must not be a symlink")
            temporary.write_bytes(contents)
            temporary.replace(destination)
    return len(parsed)


def snapshot(args):
    """Serialize manual and automatic snapshots for the same local destination."""
    directory = Path(args.local_results).expanduser().resolve() / "logs" / SUITE
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "sync.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another artifact snapshot is still running")
        return _snapshot(args)


def _snapshot(args):
    remote_archive = "/tmp/kd-tuning-snapshot-" + uuid.uuid4().hex + ".tgz"
    # Construct a stable snapshot, not a tar stream of changing live files.
    build = f"""set -eu
/var/lib/kd-venv/bin/python - <<'PY'
from pathlib import Path
import hashlib, io, json, tarfile
root = Path({azure.RESULTS!r})
archive_path = Path({remote_archive!r})
assert root.is_dir(), 'Result directory missing'
total = 0
count = 0
with tarfile.open(archive_path, 'w:gz') as archive:
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if ('data' in relative.parts or path.suffix not in {ALLOWED!r}
                or path.is_symlink() or not path.is_file()):
            continue
        size = path.stat().st_size
        if size > 2 * 1024 ** 2:
            raise ValueError('Individual artifact exceeds 2 MiB')
        data = path.read_bytes()
        if path.suffix == '.json':
            json.loads(data)
        total += len(data)
        count += 1
        assert total <= {MAX_UNPACKED} and count <= 512, 'Too many artifact bytes/files'
        info = tarfile.TarInfo(relative.as_posix())
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
data = archive_path.read_bytes()
print('SNAPSHOT_META=' + json.dumps(dict(size=len(data), sha256=hashlib.sha256(data).hexdigest(), files=count)))
PY
"""
    output = azure.invoke(args, build)
    match = re.search(r"SNAPSHOT_META=(\{[^\n]+\})", output)
    if not match:
        raise RuntimeError("Snapshot creation failed: " + output[-800:])
    meta = json.loads(match[1])
    if not 0 < meta["size"] <= MAX_UNPACKED:
        raise ValueError("Unexpected compressed snapshot size")
    data = bytearray()
    # 2400 bytes -> 3200 Base64 chars, safely below Azure's 4KB output tail.
    for offset in range(0, meta["size"], 2400):
        output = azure.invoke(args, f"set -eu; echo SNAPSHOT_BEGIN; dd if={shlex.quote(remote_archive)} bs=1 skip={offset} count=2400 status=none | base64 -w0; echo; echo SNAPSHOT_END")
        match = re.search(r"SNAPSHOT_BEGIN\n([A-Za-z0-9+/=\r\n]*)\nSNAPSHOT_END", output)
        if not match:
            raise RuntimeError("Snapshot chunk truncated or unavailable")
        data.extend(base64.b64decode(match[1], validate=False))
    if len(data) != meta["size"] or hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise ValueError("Snapshot integrity check failed")
    results_root = Path(args.local_results).expanduser().resolve()
    log_dir = results_root / "logs" / SUITE
    files = extract_snapshot(bytes(data), log_dir)
    summary = render_tuning_report(results_root, SUITE)
    atomic_json(log_dir / "sync_status.json", {
        "synced_at": datetime.now(timezone.utc).isoformat(), "remote": azure.RESULTS,
        "archive_sha256": meta["sha256"], "files": files,
        "test_results_present": (log_dir / "test_results.json").is_file(),
        "figures": str(results_root / "figures" / SUITE),
    })
    cleanup = azure.invoke(args, f"set -eu; rm -- {shlex.quote(remote_archive)}; echo SNAPSHOT_REMOVED")
    if "SNAPSHOT_REMOVED" not in cleanup:
        print("Warning: remote temporary snapshot cleanup unconfirmed", flush=True)
    print(f"SYNCED {files} files -> {log_dir}; figures -> {results_root / 'figures' / SUITE}", flush=True)
    return summary


def watch(args):
    root = Path(args.local_results).expanduser().resolve() / "logs" / SUITE
    root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with (root / "watch.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An artifact watcher is already running for this destination")
        print(f"WATCHING H100 completion -> {root}; no VM restart or training launch", flush=True)
        # Allow an initial manual snapshot to finish; completed jobs still sync immediately.
        last_sync = time.monotonic()
        deadline = time.monotonic() + args.max_hours * 3600
        while not stop.is_set() and time.monotonic() < deadline:
            try:
                state = azure.status(args)
                atomic_json(root / "watch_status.json", {"checked_at": datetime.now(timezone.utc).isoformat(), **state})
                power = state["vm"]["power"]
                execution = state.get("job", {}).get("execution", {})
                phase = execution.get("executionState", "Unknown")
                if power != "PowerState/running":
                    print(f"WAITING: {power}; possible Spot stop; local artifacts retained", flush=True)
                elif phase == "Succeeded" and execution.get("exitCode") == 0:
                    snapshot(args)
                    if not (root / "test_results.json").is_file():
                        raise RuntimeError("Job ended without final test_results.json; not complete")
                    atomic_json(root / "sync_complete.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
                                                               "job": state["job"]["name"], "result": "success"})
                    print("H100_RESULTS_SYNC_COMPLETE", flush=True)
                    return
                elif phase in {"Failed", "Canceled", "TimedOut"}:
                    snapshot(args)
                    print(f"H100_JOB_{phase.upper()}: partial artifacts saved; inspect logs before retrying", flush=True)
                    return
                elif time.monotonic() - last_sync >= args.snapshot_every:
                    snapshot(args)
                    last_sync = time.monotonic()
            except (subprocess.CalledProcessError, RuntimeError, ValueError, OSError) as exc:
                message = str(exc)
                if isinstance(exc, subprocess.CalledProcessError):
                    message = (exc.stderr or "Azure command failed")[-800:]
                print(f"WATCH_RETRY: {message}", flush=True)
                atomic_json(root / "sync_error.json", {"time": datetime.now(timezone.utc).isoformat(), "error": message})
            stop.wait(args.interval)
        print("WATCH_STOPPED: local artifacts retained; training not affected", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sync", "watch"])
    parser.add_argument("--local-results", type=Path, required=True)
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--vm-name", required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--snapshot-every", type=float, default=300)
    parser.add_argument("--max-hours", type=float, default=6)
    args = parser.parse_args()
    if min(args.interval, args.snapshot_every, args.max_hours) <= 0:
        parser.error("Watcher intervals and duration must be positive")
    {"sync": snapshot, "watch": watch}[args.action](args)


if __name__ == "__main__":
    main()