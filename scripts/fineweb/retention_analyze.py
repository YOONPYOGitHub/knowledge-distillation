#!/usr/bin/env python3
"""Analyse the retention scores under docs/small-model-kd-v1/RETENTION_RULE.md."""
import csv
import json
from pathlib import Path

import numpy as np

R = Path('/root/runs/fineweb4way')
OUT = Path('/root/knowledge-distillation/results/small-model-v1/retention')
FAMILIES = ('qwen35', 'gemma4', 'lfm25', 'qwen3', 'gemma3', 'kanana')
STUDY = {'qwen35': 'generic-kd-v1', 'gemma4': 'generic-kd-v1', 'lfm25': 'generic-kd-v1',
         'qwen3': 'small-model-v1', 'gemma3': 'small-model-v1', 'kanana': 'small-model-v1'}
SEEDS = (42, 43, 44)
GROUPS = {'kobest': ('kobest_boolq', 'kobest_copa', 'kobest_hellaswag', 'kobest_sentineg', 'kobest_wic'),
          'kmmlu': ('kmmlu_Health', 'kmmlu_Social-Welfare', 'kmmlu_Psychology')}
REPLICATES = 10000


def verdict(lo, hi):
    return 'up' if lo > 0 else 'down' if hi < 0 else 'no change'


def main():
    scores, tasks = {}, None
    for fam in FAMILIES:
        for model in ['pretrained-student'] + [f'{m}-seed{s}' for m in ('ce', 'kd') for s in SEEDS]:
            d = json.loads((R / 'retention' / 'scores' / f'{fam}--{model}.json').read_text())
            if d['limited_to'] is not None:
                raise SystemExit(f'{fam} {model}: incomplete')
            tasks = tasks or d['tasks']
            scores[fam, model] = np.array(d['correct'], dtype=np.float64)
    tasks = np.array(tasks)
    rng = np.random.default_rng(20261112)
    task_idx = {t: np.where(tasks == t)[0] for t in np.unique(tasks)}
    draws = {t: idx[rng.integers(0, len(idx), size=(REPLICATES, len(idx)))] for t, idx in task_idx.items()}

    def acc(arr, task_list, boot):
        if boot:
            return np.mean([arr[draws[t]].mean(axis=1) for t in task_list], axis=0)
        return float(np.mean([arr[task_idx[t]].mean() for t in task_list]))

    rows = []
    units = [(t, (t,)) for t in sorted(task_idx)] + [(g, ts) for g, ts in GROUPS.items()]
    for fam in FAMILIES:
        for unit, task_list in units:
            pre = scores[fam, 'pretrained-student']
            ce = [scores[fam, f'ce-seed{s}'] for s in SEEDS]
            kd = [scores[fam, f'kd-seed{s}'] for s in SEEDS]
            point = dict(pre=acc(pre, task_list, False), ce=np.mean([acc(a, task_list, False) for a in ce]),
                         kd=np.mean([acc(a, task_list, False) for a in kd]))
            b_pre = acc(pre, task_list, True)
            b_ce = np.mean([acc(a, task_list, True) for a in ce], axis=0)
            b_kd = np.mean([acc(a, task_list, True) for a in kd], axis=0)
            row = dict(study=STUDY[fam], family=fam, unit=unit, pretrained=point['pre'], ce=point['ce'], kd=point['kd'])
            for name, diff, boot in (('ce_vs_pre', point['ce'] - point['pre'], b_ce - b_pre),
                                     ('kd_vs_pre', point['kd'] - point['pre'], b_kd - b_pre),
                                     ('kd_vs_ce', point['kd'] - point['ce'], b_kd - b_ce)):
                lo, hi = np.percentile(boot, [2.5, 97.5])
                row.update({f'{name}_pp': 100 * diff, f'{name}_lo': 100 * lo, f'{name}_hi': 100 * hi,
                            f'{name}_verdict': verdict(lo, hi)})
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'retention.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r['unit'] in GROUPS:
            print(f"{r['family']:7} {r['unit']:7} pre {100 * r['pretrained']:5.1f} CE {100 * r['ce']:5.1f} "
                  f"KD {100 * r['kd']:5.1f} | CE-pre {r['ce_vs_pre_pp']:+5.1f} ({r['ce_vs_pre_verdict']}) "
                  f"KD-pre {r['kd_vs_pre_pp']:+5.1f} ({r['kd_vs_pre_verdict']}) "
                  f"KD-CE {r['kd_vs_ce_pp']:+5.1f} [{r['kd_vs_ce_lo']:+.1f},{r['kd_vs_ce_hi']:+.1f}] ({r['kd_vs_ce_verdict']})")


main()
