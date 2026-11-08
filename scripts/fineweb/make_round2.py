#!/usr/bin/env python3
"""Build Round 2 trials from Round 1 by the rule frozen in docs/small-model-kd-v1/ROUND2_RULE.md.

usage: make_round2.py <family>   writes /root/runs/fineweb4way/<family>/round2_trials.json
"""
import json
import sys
from pathlib import Path

R = Path('/root/runs/fineweb4way')
GRID = {2e-5: (1e-5, 1.5e-5, 3e-5, 4e-5), 5e-5: (3e-5, 4e-5, 6e-5, 8e-5),
        1e-4: (7.5e-5, 1.25e-4, 1.5e-4, 2.5e-4), 2e-4: (1.5e-4, 2.5e-4, 3e-4, 4e-4)}


def main(family):
    round1 = R / family / 'round1'
    selection = json.loads((round1 / 'selection.json').read_text())['methods']
    trials = {t['id']: t for t in json.loads((round1 / 'plan.json').read_text())['trials']}
    ce, kd = trials[selection['ce']['trial_id']], trials[selection['kd']['trial_id']]
    rates = next(v for k, v in GRID.items() if abs(k - ce['learning_rate']) < 1e-12)
    teacher_ppl = json.loads((R / family / 'logs' / 'teacher.json').read_text())['result']['best_validation']['ppl']
    ce_ppl = selection['ce']['best_validation']['ppl']
    alpha = 0.7 if teacher_ppl < ce_ppl else 0.9
    out = [dict(id=f'ce-{i + 1}', method='ce', learning_rate=lr, alpha=1.0, temperature=1.0,
                divergence='forward_kl') for i, lr in enumerate(rates)]
    out += [dict(id=f'kd-{i + 1}', method='kd', learning_rate=lr, alpha=alpha, temperature=kd['temperature'],
                 divergence=kd['divergence']) for i, lr in enumerate(rates)]
    path = R / family / 'round2_trials.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    print(f'{family}: round1 best CE {ce["id"]} lr={ce["learning_rate"]} PPL {ce_ppl:.3f}; '
          f'best KD {kd["id"]} T={kd["temperature"]} {kd["divergence"]}; Teacher PPL {teacher_ppl:.3f} '
          f'-> round2 lr={list(rates)} KD alpha={alpha}')


main(sys.argv[1])
