#!/usr/bin/env python3
"""Score one Student on the shared validation byte coverage and record bits per document.

usage: common_score.py <ours|colleague> <family> <model_id> <out.json> [--limit N]
model_id: pretrained-student, ce-seed42..44, kd-seed42..44

Every model predicts exactly the same UTF-8 bytes (coverage.json). Long documents use a
sliding window: each target token is scored once with up to 511 tokens of context, at least
256 when the document is long enough. No BOS, as in training. FP32 log-softmax.
--limit scores only the first N documents (a mechanics check, not a result).
"""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

R = Path('/root/runs/fineweb4way')
PLAN = R / 'common' / 'coverage.json'
RAW = R / 'qwen3' / 'base' / 'raw'
WINDOW, STRIDE = 512, 256


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_ours(family, model_id):
    repo = Path('/root/knowledge-distillation')
    sys.path.insert(0, str(repo))
    from src.numerical_policy import configure_numerics
    configure_numerics()
    from transformers import AutoTokenizer
    from scripts.run_research import make_model
    from src.research_models import load_model, restore_best
    config = json.loads((R / family / 'base' / 'config.json').read_text())
    source = dict(model=config['student'])
    if model_id == 'pretrained-student':
        model = load_model(config['student'], device='cuda', **config['model'])
    else:
        method, seed = model_id.split('-seed')
        seed = int(seed)
        if seed == 42:
            chosen = json.loads((R / family / 'final_selection.json').read_text())[method]
            record = json.loads((R / family / chosen['round'] / 'trials' / chosen['trial_id'] / 'result.json').read_text())
        else:
            record = json.loads((R / family / f'seed{seed}' / 'trials' / f'{method}-1' / 'result.json').read_text())
        cfg = copy.deepcopy(config)
        cfg['training']['seed'] = seed
        model = make_model('student', cfg, {})
        checkpoint = record['result']['best_checkpoint']
        restore_best(model, checkpoint, contract=record['checkpoint_contract'], freeze=True)
        source.update(checkpoint=str(Path(checkpoint).relative_to(R)),
                      manifest_sha256=sha((Path(checkpoint) / 'manifest.json').read_bytes()))
    spec = config['student']
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], local_files_only=True)
    return model, tokenizer, source


def load_colleague(family, model_id):
    code = Path('/root/colleague/code/source')
    sys.path.insert(0, str(code))
    from src.numerical_policy import configure_numerics
    configure_numerics()
    import yaml
    import zipfile
    from transformers import AutoTokenizer
    from src.research_models import load_model, attach_lora, restore_best
    config = yaml.safe_load((code / 'configs' / f'research_{family}.yaml').read_text())
    spec = config['student']
    source = dict(model=spec)
    model = load_model(spec, device='cuda', dtype=config['model']['dtype'], attention=config['model']['attention'])
    if model_id != 'pretrained-student':
        packages = sorted(Path('/root/colleague-ckpts').glob(f'{family}-{model_id}-*.zip'))
        if len(packages) != 1:
            raise SystemExit(f'expected one package for {family} {model_id}, found {len(packages)}')
        ckpt = Path('/root/colleague/ckpt') / f'{family}-{model_id}'
        if not (ckpt / 'state.pt').exists():
            zipfile.ZipFile(packages[0]).extractall(ckpt)
        manifest = json.loads((ckpt / 'manifest.json').read_text())
        model = attach_lora(model, config['adapter'], seed=int(model_id.split('-seed')[1]))
        restore_best(model, ckpt, contract=manifest['contract'], freeze=True)
        source.update(package=packages[0].name, manifest_sha256=sha((ckpt / 'manifest.json').read_bytes()))
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'])
    return model, tokenizer, source


def main():
    impl, family, model_id, out = sys.argv[1:5]
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    payload = (RAW / 'validation.jsonl').read_bytes()
    if sha(payload) != plan['summary']['validation_sha256']:
        raise SystemExit('validation file differs from the coverage plan')
    model, tokenizer, source = (load_ours if impl == 'ours' else load_colleague)(family, model_id)
    import torch
    import transformers
    model.eval()
    device = next(model.parameters()).device
    texts = [json.loads(line)['text'] for line in payload.decode('utf-8').splitlines()]
    rows, total_nats, total_bytes = [], 0.0, 0
    for index, (text, doc) in enumerate(zip(texts, plan['documents'])):
        if limit is not None and index >= limit:
            break
        if doc['excluded']:
            rows.append(None)
            continue
        expect = doc['tokenizers'][family]
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        if sha(json.dumps(ids).encode()) != expect['ids_sha256']:
            raise SystemExit(f'document {index}: tokenization differs from the coverage plan')
        n, pos, nats, count = len(ids), expect['first_scored_index'], 0.0, 0
        with torch.inference_mode():
            while pos < n:
                end = min(n, pos + STRIDE)
                ctx = max(0, end - WINDOW)
                window = torch.tensor([ids[ctx:end]], device=device)
                logp = torch.log_softmax(model(input_ids=window).logits[0].float(), dim=-1)
                targets = torch.tensor(ids[pos:end], device=device)
                nats += float(-logp[pos - ctx - 1:end - ctx - 1].gather(1, targets[:, None]).sum())
                count += end - pos
                pos = end
        if count != expect['scored_tokens'] or not math.isfinite(nats):
            raise SystemExit(f'document {index}: scored {count} tokens, plan says {expect["scored_tokens"]}')
        rows.append([nats, count])
        total_nats += nats
        total_bytes += doc['covered_bytes']
    bits = total_nats / math.log(2)
    result = dict(version=1, impl=impl, family=family, model_id=model_id, source=source, split='validation',
                  coverage_sha256=sha(plan_bytes), window=WINDOW, stride=STRIDE, limited_to=limit,
                  torch=torch.__version__, transformers=transformers.__version__,
                  totals=dict(bits=bits, bytes=total_bytes, bits_per_byte=bits / total_bytes,
                              documents=sum(r is not None for r in rows)),
                  documents=rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result) + '\n')
    print(f"{impl} {family} {model_id}: {result['totals']['documents']} documents scored, "
          f"{total_bytes} bytes" + (' (mechanics check)' if limit else ''))


main()
