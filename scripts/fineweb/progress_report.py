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


DIRECTIONS = (
    ('실행 중인 학습', '마지막 단계의 로그를 확인하고, 끊긴 단계를 체크포인트에서 재개합니다.'),
    ('미완료 상태로 멈춤', '원인을 확인한 뒤 --resume으로 마지막 체크포인트에서 재개합니다.'),
    ('오류 기록', '로그에서 원인을 찾아 고치고 검증한 뒤 재개합니다. 공간 부족 외 오류는 자동 재개되지 않습니다.'),
    ('비정상 종료', '로그에서 원인을 찾아 고치고 검증한 뒤 재개합니다.'),
    ('1시간 가까이 진행 없음', '프로세스와 GPU 상태를 확인하고, 멈췄다면 종료 후 체크포인트에서 재개합니다.'),
    ('디스크 사용률', '완료된 trial의 중간 체크포인트 정리가 도는지 확인하고, 불필요한 파일을 정리합니다.'),
    ('공유 볼륨 여유', '학습은 공간이 나면 자동 재개됩니다. 계속되면 GPU 센터에 볼륨 정리를 요청합니다.'),
    ('워크스페이스 정지', '웹 콘솔에서 재시작한 뒤 새 SSH 정보로 보고와 학습을 재개합니다.'),
    ('GitHub push 실패', 'git_milestones.log를 확인합니다. 10분마다 다시 시도합니다.'),
)


def direction(warning):
    for prefix, text in DIRECTIONS:
        if warning.startswith(prefix):
            return text
    return '원인을 확인해 고칩니다.'


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


def log_lines(group, name):
    try:
        with open(f'{R}/{group}/logs/{name}', errors='replace') as f:
            return f.read().splitlines()
    except OSError:
        return []


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
    preparing = bool(sh('pgrep -f download_verify.py'))
    if not procs and not preparing:
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
        if not procs and not preparing:
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
    vfs = os.statvfs('/root')
    if vfs.f_ffree < 10000000:
        warnings.append(f'공유 볼륨 여유 약 {vfs.f_ffree * 512 / 1e9:.1f}GB: 쓰기 실패가 날 수 있습니다.')
    warnings = list(dict.fromkeys(warnings))
    lines = [f'[KD {now:%m/%d %H:%M}] GPU {busy}/{len(gpu_rows)} 사용, 진행 중 {len(active)}개, '
             f'완료 {len(done)}개, 경고 {len(warnings)}개',
             (f'문제: {warnings[0]} / 해결 방향: {direction(warnings[0])}' if warnings else '특이사항 없음'),
             '']
    if left is not None:
        lines.append(f'워크스페이스 예상 정지까지: 약 {left:.1f}시간')
    lines.append(f'디스크: {pct:.0f}% 사용 ({usage.free / 1e9:.0f}GB 남음)')
    if preparing:
        lines.append('모델 다운로드·검증 중: 끝나면 Teacher가 자동으로 시작됩니다.')
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
    lines += ['', '문제와 해결 방향']
    for w in warnings:
        lines += [f'  현재 상황: {w}', f'  해결 방향: {direction(w)}']
    if not warnings:
        lines.append('  없음')
    history = [line for group in groups() for line in log_lines(group, 'interruptions.txt')]
    if history:
        lines += ['', '재개 완료된 중단 기록'] + [f'  {line}' for line in history]
    print('\n'.join(lines))
    if '--peek' not in sys.argv:
        with open(STATE, 'w', encoding='utf-8') as f:
            json.dump(state, f)


main()
