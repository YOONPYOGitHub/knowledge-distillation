#!/usr/bin/env python3
"""Build results.csv and paired_effects.csv in the generic-kd-v1 format from the write-once test outputs."""
import csv
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results' / 'small-model-v1'
ROLES = {'teacher': 'Teacher', 'pretrained-student': 'pretrainedStudent'}
FIELDS = ['family', 'role', 'model_id', 'seed', 'status', 'split', 'ce_sum', 'tokens', 'ce', 'ppl',
          'protocol_sha256', 'original_path', 'original_sha256', 'published_path']


def main():
    rows, by_family = [], {}
    for protocol_file in sorted((OUT / 'test-protocol').glob('*.json')):
        family = protocol_file.stem
        protocol_sha = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
        scored = {}
        for model in json.loads(protocol_file.read_text())['models']:
            path = OUT / 'final' / f"{family}--{model['model_id']}.json"
            if not path.exists():
                break
            record = json.loads(path.read_text())
            if record['protocol_sha256'] != protocol_sha or record['status'] != 'completed':
                raise SystemExit(f'{path.name}: protocol or status mismatch')
            stats = record['statistics']
            ce = stats['ce_sum'] / stats['tokens']
            model_id = model['model_id']
            role = ROLES.get(model_id) or ('StudentCE' if model_id.startswith('ce-') else 'StudentKD')
            rel = str(path.relative_to(REPO))
            rows.append(dict(family=family, role=role, model_id=model_id, seed=model['seed'] or '',
                             status='completed', split='test', ce_sum=stats['ce_sum'], tokens=stats['tokens'],
                             ce=ce, ppl=math.exp(ce), protocol_sha256=protocol_sha, original_path=rel,
                             original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), published_path=rel))
            scored[model_id] = math.exp(ce)
        else:
            by_family[family] = scored
    with open(OUT / 'results.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / 'paired_effects.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['family', 'seed', 'ce_ppl', 'kd_ppl', 'kd_relative_ppl_change_percent',
                         'pretrained_to_ce_ppl_decrease_percent', 'selection_seed_reused'])
        for family, ppl in by_family.items():
            for seed in (42, 43, 44):
                ce, kd = ppl[f'ce-seed{seed}'], ppl[f'kd-seed{seed}']
                writer.writerow([family, seed, ce, kd, 100 * (kd / ce - 1),
                                 100 * (1 - ce / ppl['pretrained-student']), seed == 42])
                print(f'{family} seed{seed}: CE {ce:.4f} KD {kd:.4f} KD vs CE {100 * (kd / ce - 1):+.2f}% '
                      f'pretrained->CE {100 * (1 - ce / ppl["pretrained-student"]):.1f}%')
    print(f'{len(rows)} result rows; complete families: {sorted(by_family)}')


main()
