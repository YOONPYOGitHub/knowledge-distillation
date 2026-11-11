#!/usr/bin/env python3
"""Analyse the 42 common scores under docs/small-model-kd-v1/COMMON_SCORING_RULE.md and write the tables."""
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

R = Path('/root/runs/fineweb4way')
REPO = Path('/root/knowledge-distillation')
OUT = REPO / 'results' / 'small-model-v1' / 'common'
FAMILIES = ('qwen35', 'gemma4', 'lfm25', 'qwen3', 'gemma3', 'kanana')
STUDY = {'qwen35': 'generic-kd-v1', 'gemma4': 'generic-kd-v1', 'lfm25': 'generic-kd-v1',
         'qwen3': 'small-model-v1', 'gemma3': 'small-model-v1', 'kanana': 'small-model-v1'}
MODELS = ('pretrained-student', 'ce-seed42', 'ce-seed43', 'ce-seed44', 'kd-seed42', 'kd-seed43', 'kd-seed44')
MARGINS = {'primary': 0.01, 'sensitivity': 0.005}
REPLICATES = 10000


def main():
    plan = json.loads((R / 'common' / 'coverage.json').read_text())
    keep = [i for i, d in enumerate(plan['documents']) if not d['excluded']]
    nbytes = np.array([plan['documents'][i]['covered_bytes'] for i in keep], dtype=np.float64)
    bits = {}
    for fam in FAMILIES:
        for model in MODELS:
            score = json.loads((R / 'common' / 'scores' / f'{fam}--{model}.json').read_text())
            if score['limited_to'] is not None or score['totals']['bytes'] != int(nbytes.sum()):
                raise SystemExit(f'{fam} {model}: incomplete coverage')
            bits[fam, model] = np.array([score['documents'][i][0] for i in keep]) / math.log(2)

    rng = np.random.default_rng(20261111)
    idx = rng.integers(0, len(keep), size=(REPLICATES, len(keep)))
    byte_sums = nbytes[idx].sum(axis=1)

    def boot(fam, method):
        return np.mean([bits[fam, f'{method}-seed{s}'][idx].sum(axis=1) / byte_sums for s in (42, 43, 44)], axis=0)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'scores.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['study', 'family', 'model_id', 'bits_per_byte', 'bits', 'bytes'])
        for (fam, model), b in bits.items():
            writer.writerow([STUDY[fam], fam, model, b.sum() / nbytes.sum(), b.sum(), int(nbytes.sum())])

    point = {(fam, m): np.mean([bits[fam, f'{m}-seed{s}'].sum() / nbytes.sum() for s in (42, 43, 44)])
             for fam in FAMILIES for m in ('ce', 'kd')}
    boots = {fam: boot(fam, 'kd') for fam in FAMILIES}
    rows = []
    for a, b in itertools.combinations(FAMILIES, 2):
        rel = boots[a] / boots[b] - 1
        lo, hi = np.percentile(rel, [2.5, 97.5])
        row = dict(a=a, b=b, kd_bpb_a=point[a, 'kd'], kd_bpb_b=point[b, 'kd'],
                   relative_difference=point[a, 'kd'] / point[b, 'kd'] - 1, ci_low=lo, ci_high=hi)
        for name, m in MARGINS.items():
            row[f'verdict_{name}'] = ('tie' if lo > -m and hi < m else
                                      f'{a} better' if hi < -m else f'{b} better' if lo > m else 'undecided')
        rows.append(row)
    with open(OUT / 'pairwise.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print('family  study            pretrained   CE mean   KD mean   KD vs CE')
    for fam in sorted(FAMILIES, key=lambda f: point[f, 'kd']):
        pre = bits[fam, 'pretrained-student'].sum() / nbytes.sum()
        print(f"{fam:7} {STUDY[fam]:16} {pre:10.4f} {point[fam, 'ce']:9.4f} {point[fam, 'kd']:9.4f} "
              f"{100 * (point[fam, 'kd'] / point[fam, 'ce'] - 1):+8.2f}%")
    for r in rows:
        print(f"{r['a']} vs {r['b']}: {100 * r['relative_difference']:+.2f}% "
              f"[{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}] "
              f"1%: {r['verdict_primary']}; 0.5%: {r['verdict_sensitivity']}")


main()
