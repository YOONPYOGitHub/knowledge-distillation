#!/usr/bin/env python3
"""Pick the final CE and KD settings across Round 1 and Round 2 and write the seed trials file.

Rule (docs/small-model-kd-v1/ROUND2_RULE.md): per method, the lowest validation CE across
both rounds; ties go to Round 1, then the lower trial ID. Validation only.
usage: select_final.py <family>   writes <family>/final_selection.json and <family>/seed_trials.json
"""
import json
import sys
from pathlib import Path

R = Path('/root/runs/fineweb4way')


def main(family):
    candidates = []
    for order, name in enumerate(('round1', 'round2')):
        directory = R / family / name
        if not (directory / 'selection.json').exists():
            raise SystemExit(f'{family}/{name} has no frozen selection')
        for trial in json.loads((directory / 'plan.json').read_text())['trials']:
            result = json.loads((directory / 'trials' / trial['id'] / 'result.json').read_text())['result']
            if result['status'] == 'completed':
                candidates.append((trial['method'], result['best_validation']['ce'], order, trial['id'], name,
                                   trial, result))
    chosen, seeds = {}, []
    for method in ('ce', 'kd'):
        method_rows = sorted((c for c in candidates if c[0] == method), key=lambda c: (c[1], c[2], c[3]))
        _, ce, _, trial_id, name, trial, result = method_rows[0]
        chosen[method] = dict(round=name, trial_id=trial_id, trial=trial, best_step=result['best_step'],
                              best_validation=result['best_validation'])
        seeds.append({**trial, 'id': f'{method}-1'})
    (R / family / 'final_selection.json').write_text(json.dumps(chosen, indent=1) + '\n')
    (R / family / 'seed_trials.json').write_text(json.dumps(seeds, indent=1) + '\n')
    for method, c in chosen.items():
        t = c['trial']
        print(f"{family} {method}: {c['round']}/{c['trial_id']} lr={t['learning_rate']} alpha={t['alpha']} "
              f"T={t['temperature']} {t['divergence']} validation PPL {c['best_validation']['ppl']:.3f}")


main(sys.argv[1])
