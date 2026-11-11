#!/usr/bin/env python3
"""Zero-shot multiple-choice scoring for the retention evaluation.

usage: retention_score.py <ours|colleague> <family> <model_id> <out.json> [--limit N]
Each choice is scored by the summed log-probability of its tokens after the context.
acc picks the highest sum; acc_norm divides by the choice's UTF-8 byte length.
No BOS, as in training. Contexts are cut from the left to fit 1024 tokens.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

R = Path('/root/runs/fineweb4way')
ITEMS = R / 'retention' / 'items.jsonl'
MAX_TOKENS = 1024

spec = importlib.util.spec_from_file_location('common_score', R / 'common_score.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def encode_pair(tokenizer, context, choice):
    ctx = tokenizer(context, add_special_tokens=False)['input_ids']
    full = tokenizer(context + choice, add_special_tokens=False)['input_ids']
    cont = full[len(ctx):] if full[:len(ctx)] == ctx else tokenizer(choice, add_special_tokens=False)['input_ids']
    seq = (ctx + cont)[-MAX_TOKENS:]
    return seq, len(cont)


def main():
    impl, family, model_id, out = sys.argv[1:5]
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    payload = ITEMS.read_bytes()
    items = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
    model, tokenizer, source = (common.load_ours if impl == 'ours' else common.load_colleague)(family, model_id)
    import torch
    model.eval()
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode():
        for n, item in enumerate(items):
            if limit is not None and n >= limit:
                break
            pairs = [encode_pair(tokenizer, item['context'], c) for c in item['choices']]
            width = max(len(s) for s, _ in pairs)
            batch = torch.tensor([s + [0] * (width - len(s)) for s, _ in pairs], device=device)
            logp = torch.log_softmax(model(input_ids=batch).logits.float(), dim=-1)
            sums = []
            for row, (seq, k) in enumerate(pairs):
                start = len(seq) - k
                targets = torch.tensor(seq[start:], device=device)
                sums.append(float(logp[row, start - 1:len(seq) - 1].gather(1, targets[:, None]).sum()))
            norms = [s / max(1, len(c.encode('utf-8'))) for s, c in zip(sums, item['choices'])]
            pick = max(range(len(sums)), key=(norms if item['metric'] == 'acc_norm' else sums).__getitem__)
            rows.append(int(pick == item['gold']))
    result = dict(version=1, impl=impl, family=family, model_id=model_id, source=source,
                  items_sha256=hashlib.sha256(payload).hexdigest(), limited_to=limit, max_tokens=MAX_TOKENS,
                  tasks=[i['task'] for i in items[:len(rows)]], correct=rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result) + '\n')
    print(f'{impl} {family} {model_id}: {len(rows)} items, accuracy {sum(rows) / len(rows):.4f}'
          + (' (mechanics check)' if limit else ''))


if __name__ == '__main__':
    main()
