#!/usr/bin/env python3
"""Hourly KD progress report. --peek prints without updating the speed baseline."""
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys

R = '/root/runs/fineweb4way'
STATE = f'{R}/.progress_state.json'
KST = dt.timezone(dt.timedelta(hours=9))
MAX_STEPS = 3000
SKIP_DIRS = ('checkpoints', 'raw', 'logs')
sys.stdout.reconfigure(encoding='utf-8')


def load(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def groups():
    return [g for g in sorted(os.listdir(R)) if g != 'smoke' and os.path.isdir(f'{R}/{g}')]


def log_files(group, suffix):
    d = f'{R}/{group}/logs'
    return [f'{d}/{f}' for f in sorted(os.listdir(d)) if f.endswith(suffix)] if os.path.isdir(d) else []


def stage_rows(group):
    """Every directory holding latest.json is one training stage (Teacher or trial)."""
    root, rows = f'{R}/{group}', []
    for path, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if 'latest.json' not in files:
            continue
        name = os.path.relpath(path, root)
        ckpt = (load(f'{path}/latest.json') or {}).get('checkpoint', '')
        m = re.search(r'step-0*(\d+)', ckpt)
        rec = load(f'{path}/result.json')
        if rec is None and name.endswith('teacher'):
            rec = load(f'{root}/logs/teacher.json')
        res = rec.get('result', {}) if isinstance(rec, dict) else {}
        rows.append(dict(key=f'{group}/{name}', step=int(m.group(1)) if m else 0,
                         status=res.get('status', 'running'),
                         best=(res.get('best_validation') or {}).get('ppl'),
                         best_step=res.get('best_step')))
    return rows


def plan_rows(group):
    out = []
    for path, dirs, files in os.walk(f'{R}/{group}'):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS + ('trials',)]
        if 'plan.json' not in files:
            continue
        trials = load(f'{path}/plan.json', {}).get('trials', [])
        finished = sum(1 for t in trials
                       if (load(f'{path}/trials/{t["id"]}/result.json') or {})
                       .get('result', {}).get('status') == 'completed')
        out.append((os.path.relpath(path, R), finished, len(trials),
                    os.path.exists(f'{path}/selection.json')))
    return out


def main():
    now = dt.datetime.now(KST)
    prev = load(STATE, {})
    warnings = []

    gpu = sh('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu '
             '--format=csv,noheader,nounits').splitlines()
    gpu_rows = [[x.strip() for x in g.split(',')] for g in gpu]
    busy = sum(1 for g in gpu_rows if int(g[3]) > 0 or int(g[1]) > 200)
    procs = [p for p in sh('ps -eo etimes,cmd').splitlines()
             if re.search(r'scripts/(run_research|tune_research)\.py', p)]
    if not procs:
        warnings.append('실행 중인 학습 프로세스가 없습니다.')

    rows, plans = [], []
    for group in groups():
        rows += stage_rows(group)
        plans += plan_rows(group)
        for err in log_files(group, '.err'):
            with open(err, errors='replace') as f:
                tail = f.read()[-3000:]
            if 'Traceback' in tail or re.search(r'\b\w*Error\b:', tail):
                warnings.append(f'오류 기록: {os.path.relpath(err, R)}')
        for status in log_files(group, '.status'):
            with open(status, errors='replace') as f:
                last = (f.read().splitlines() or [''])[-1]
            m = re.search(r'rc=(\d+)', last)
            if m and m.group(1) != '0':
                warnings.append(f'비정상 종료: {last}')

    state, active, done = {}, [], []
    for r in rows:
        state[r['key']] = dict(step=r['step'], ts=now.timestamp())
        if r['status'] == 'completed':
            done.append(r)
            continue
        old = prev.get(r['key'])
        r['speed'] = None
        if old and now.timestamp() - old['ts'] > 60 and r['step'] >= old['step']:
            elapsed = now.timestamp() - old['ts']
            r['speed'] = (r['step'] - old['step']) / (elapsed / 3600)
            if r['step'] == old['step'] and elapsed > 3000:
                warnings.append(f'1시간 가까이 진행 없음: {r["key"]} step {r["step"]}')
        active.append(r)
        if not procs:
            warnings.append(f'미완료 상태로 멈춤, 재개 필요: {r["key"]} step {r["step"]}')

    usage = shutil.disk_usage('/root')
    pct = usage.used / usage.total * 100
    if pct >= 80:
        warnings.append(f'디스크 사용률 {pct:.0f}%: 체크포인트 정리가 필요합니다.')
    left = None
    try:
        with open(f'{R}/workspace_stop.txt') as f:
            left = (dt.datetime.fromisoformat(f.read().strip()) - now).total_seconds() / 3600
        if left < 2:
            warnings.append(f'워크스페이스 정지까지 약 {left:.1f}시간: 재시작을 준비하세요.')
    except (OSError, ValueError):
        pass

    if os.path.exists(f'{R}/.git_push_failed'):
        warnings.append('GitHub push 실패: git_milestones.log를 확인하세요.')
    warnings = list(dict.fromkeys(warnings))
    lines = [f'[KD {now:%m/%d %H:%M}] GPU {busy}/{len(gpu_rows)} 사용, 진행 중 {len(active)}개, '
             f'완료 {len(done)}개, 경고 {len(warnings)}개', '']
    if left is not None:
        lines.append(f'워크스페이스 예상 정지까지: 약 {left:.1f}시간')
    lines.append(f'디스크: {pct:.0f}% 사용 ({usage.free / 1e9:.0f}GB 남음)')
    lines += ['', 'GPU'] + [f'  {g[0]}번: 메모리 {g[1]}/{g[2]} MiB, 사용률 {g[3]}%' for g in gpu_rows]
    lines += ['', '진행 중']
    for r in active:
        extra = ''
        if r.get('speed'):
            extra = (f', {r["speed"]:.0f} update/시간, '
                     f'남은 시간 약 {(MAX_STEPS - r["step"]) / r["speed"]:.1f}시간')
        lines.append(f'  {r["key"]}: step {r["step"]}/{MAX_STEPS}{extra}')
    if not active:
        lines.append('  없음')
    if plans:
        lines += ['', '후보 탐색']
        lines += [f'  {p}: {f}/{t} 완료' + (', 선택 확정' if s else '') for p, f, t, s in plans]
    lines += ['', '완료']
    lines += [f'  {r["key"]}: validation PPL {r["best"]:.3f} (step {r["best_step"]})'
              if r['best'] else f'  {r["key"]}' for r in done] or ['  없음']
    lines += ['', '경고'] + ([f'  {w}' for w in warnings] or ['  없음'])
    print('\n'.join(lines))
    if '--peek' not in sys.argv:
        with open(STATE, 'w', encoding='utf-8') as f:
            json.dump(state, f)


main()
