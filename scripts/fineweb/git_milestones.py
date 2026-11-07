#!/usr/bin/env python3
"""Commit and push small-model KD results at meaningful checkpoints only.

Milestones:
  teachers      all three Teachers finished and were selected on validation
  <family> <r>  a tuning round froze its validation-only selection (selection.json)
  <family> seeds  the selected CE and KD pair finished for seeds 43 and 44

Only metric summaries under results/small-model-v1 are committed. Raw text, token IDs,
and weights never are. Run with --once to check a single time.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('/root/knowledge-distillation')
RUNS = Path('/root/runs/fineweb4way')
OUT = REPO / 'results' / 'small-model-v1'
DONE = RUNS / '.milestones.json'
FAILED = RUNS / '.git_push_failed'
FAMILIES = ('qwen3', 'gemma3', 'kanana')
ROUNDS = ('round1', 'round2')
BRANCH = 'feature/fineweb-small-model-kd'
POLL_SECONDS = 600
TRAILER = '\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
RESULT_KEYS = ('status', 'stop_reason', 'best_step', 'final_step', 'best_validation',
               'initial_validation', 'history')


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def scrub(value):
    """Drop machine paths from metadata before it leaves the workspace."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(RUNS) + '/', '').replace(str(REPO) + '/', '')
    return value


def result_summary(result):
    return {k: result[k] for k in RESULT_KEYS if k in result}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scrub(value), indent=2, sort_keys=True, ensure_ascii=False) + '\n',
                    encoding='utf-8')


def teacher_summary(family):
    status = RUNS / family / 'logs' / 'teacher.status'
    record = load(RUNS / family / 'logs' / 'teacher.json')
    if not status.exists() or 'rc=0' not in status.read_text().splitlines()[-1] or not record:
        return None
    result = record.get('result', {})
    if result.get('status') != 'completed':
        return None
    config = load(RUNS / family / 'base' / 'config.json') or {}
    return dict(family=family, teacher={k: config.get('teacher', {}).get(k) for k in ('id', 'revision')},
                student={k: config.get('student', {}).get(k) for k in ('id', 'revision')},
                result=result_summary(result))


def round_summary(family, name):
    directory = RUNS / family / name
    selection = load(directory / 'selection.json')
    plan = load(directory / 'plan.json')
    if not selection or not plan:
        return None
    trials = []
    for trial in plan['trials']:
        record = load(directory / 'trials' / trial['id'] / 'result.json') or {}
        trials.append(dict(trial=trial, result=result_summary(record.get('result', {}))))
    return dict(family=family, round=name, trials=trials, selection=selection)


def seeds_summary(family):
    rows = {}
    for seed in (43, 44):
        directory = RUNS / family / f'seed{seed}'
        plan = load(directory / 'plan.json')
        if not plan or not load(directory / 'selection.json'):
            return None
        rows[seed] = [dict(trial=t, result=result_summary(
            (load(directory / 'trials' / t['id'] / 'result.json') or {}).get('result', {})))
            for t in plan['trials']]
    return dict(family=family, seeds=rows)


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)


def publish(message):
    """Commit results/small-model-v1 only. Returns True once the commit is on the remote."""
    git('add', '--', str(OUT))
    if git('diff', '--cached', '--quiet', '--', str(OUT)).returncode != 0:
        commit = git('commit', '-m', message + TRAILER, '--', str(OUT))
        if commit.returncode != 0:
            FAILED.write_text(commit.stderr[-500:])
            return False
    push = git('push', 'origin', BRANCH)
    if push.returncode != 0:
        FAILED.write_text(push.stderr[-500:])
        print(push.stderr[-500:], flush=True)
        return False
    FAILED.unlink(missing_ok=True)
    return True


def check():
    done = set(load(DONE) or [])

    def finish(key, message):
        if publish(message):
            done.add(key)
            DONE.write_text(json.dumps(sorted(done)))
            print(f'{datetime.now(timezone.utc):%FT%TZ} pushed {key}', flush=True)

    if 'teachers' not in done:
        summaries = {f: teacher_summary(f) for f in FAMILIES}
        if all(summaries.values()):
            for family, summary in summaries.items():
                write(OUT / family / 'teacher.json', summary)
            lines = [f'{f}: validation PPL {s["result"]["best_validation"]["ppl"]:.3f} '
                     f'(step {s["result"]["best_step"]})' for f, s in summaries.items()]
            finish('teachers', 'results: small-model Teachers selected on validation\n\n' + '\n'.join(lines))

    for family in FAMILIES:
        for name in ROUNDS:
            key = f'{family}/{name}'
            summary = None if key in done else round_summary(family, name)
            if summary:
                write(OUT / family / f'{name}.json', summary)
                finish(key, f'results({family}): {name} validation-only selection')
        key = f'{family}/seeds'
        summary = None if key in done else seeds_summary(family)
        if summary:
            write(OUT / family / 'seeds.json', summary)
            finish(key, f'results({family}): selected CE and KD for seeds 43 and 44')


def main():
    while True:
        try:
            check()
        except Exception as exc:  # keep watching across a transient failure, but make it visible
            FAILED.write_text(f'{type(exc).__name__}: {exc}'[:500])
            print(f'{type(exc).__name__}: {exc}', flush=True)
        if '--once' in sys.argv:
            return
        time.sleep(POLL_SECONDS)


main()
