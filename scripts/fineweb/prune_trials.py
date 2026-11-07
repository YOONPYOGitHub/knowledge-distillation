#!/usr/bin/env python3
"""Free disk from finished Student trials, keeping only their selected and final checkpoints.

A trial is touched only when its result.json says completed and both kept checkpoints exist.
Running trials are left alone, so resuming stays safe. Run with --once to check a single time.
"""
import json
import shutil
import sys
import time
from pathlib import Path

RUNS = Path('/root/runs/fineweb4way')
POLL_SECONDS = 600


def prune_trial(trial, log=print):
    root = trial / 'checkpoints'
    try:
        result = json.loads((trial / 'result.json').read_text(encoding='utf-8')).get('result', {})
    except (OSError, ValueError):
        return 0
    if result.get('status') != 'completed' or not root.is_dir():
        return 0
    keep = {Path(result.get('best_checkpoint') or '').name, Path(result.get('last_checkpoint') or '').name}
    if not all(name and (root / name).is_dir() for name in keep):
        return 0
    removed = 0
    for path in root.iterdir():
        if path.is_dir() and path.name not in keep:
            shutil.rmtree(path)
            removed += 1
    if removed:
        log(f'{trial}: kept {sorted(keep)} removed {removed}')
    return removed


def check(runs=RUNS):
    return sum(prune_trial(trial) for trial in runs.glob('*/*/trials/*') if trial.is_dir())


if __name__ == '__main__':
    while True:
        check()
        sys.stdout.flush()
        if '--once' in sys.argv:
            break
        time.sleep(POLL_SECONDS)
