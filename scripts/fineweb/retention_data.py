#!/usr/bin/env python3
"""Build the retention benchmark items (KoBEST test, KMMLU Health/Social-Welfare/Psychology test).

Each item: task, id, context, choices, gold, metric. Prompts follow the lm-evaluation-harness
KoBEST templates; KMMLU uses a lettered four-choice prompt. Written to retention/items.jsonl.
The source text is not committed; only the item hashes and counts are.
"""
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path('/root/runs/fineweb4way/retention')
KOBEST = ('boolq', 'copa', 'hellaswag', 'sentineg', 'wic')
KMMLU = ('Health', 'Social-Welfare', 'Psychology')


def kobest(task, row, index):
    if task == 'boolq':
        return f"{row['paragraph']} 질문: {row['question']} 답변: ", ['아니오', '예'], row['label'], 'acc'
    if task == 'copa':
        connector = {'원인': ' 왜냐하면', '결과': ' 그래서'}[row['question'].strip()]
        return row['premise'].rstrip('.') + connector, [' ' + row['alternative_1'], ' ' + row['alternative_2']], \
            row['label'], 'acc_norm'
    if task == 'hellaswag':
        return row['context'], [' ' + row[f'ending_{i}'] for i in range(1, 5)], row['label'], 'acc_norm'
    if task == 'sentineg':
        return f"문장: {row['sentence']} 긍부정:", [' 부정', ' 긍정'], row['label'], 'acc'
    if task == 'wic':
        return (f"문장1: {row['context_1']} 문장2: {row['context_2']} 두 문장에서 {row['word']}가 같은 뜻으로 쓰였나?",
                ['아니오', '예'], row['label'], 'acc')
    raise ValueError(task)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for task in KOBEST:
        for i, row in enumerate(load_dataset('skt/kobest_v1', task, split='test')):
            context, choices, gold, metric = kobest(task, row, i)
            items.append(dict(task=f'kobest_{task}', id=i, context=context, choices=choices, gold=int(gold), metric=metric))
    for subject in KMMLU:
        for i, row in enumerate(load_dataset('HAERAE-HUB/KMMLU', subject, split='test')):
            context = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n정답:"
            items.append(dict(task=f'kmmlu_{subject}', id=i, context=context, choices=[' A', ' B', ' C', ' D'],
                              gold=int(row['answer']) - 1, metric='acc'))
    payload = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items)
    (OUT / 'items.jsonl').write_text(payload, encoding='utf-8')
    counts = {}
    for item in items:
        counts[item['task']] = counts.get(item['task'], 0) + 1
    summary = dict(items=len(items), counts=counts, items_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                   sources={'kobest': 'skt/kobest_v1 (CC BY-SA 4.0), test split',
                            'kmmlu': 'HAERAE-HUB/KMMLU (CC BY-ND 4.0), test split, Health/Social-Welfare/Psychology'})
    (OUT / 'items-summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


main()
