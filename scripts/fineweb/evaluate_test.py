#!/usr/bin/env python3
"""One-time test scoring of the small-model 4-way roster, using generic-kd-v1's CE statistics.

  freeze <family>                         write the scoring protocol (commit it before scoring)
  check  <family> <model_id>...           score on VALIDATION only, to compare with training logs
  score  <family>                         score the frozen roster on TEST once; outputs are write-once

The CE arithmetic and test packing come unchanged from src/final_evaluation.py
(feature/azure-h100-kd-tuning dbdaa7d): FP32 shifted CE sum / valid tokens, seq512 blocks,
add_special_tokens=False, EOS per document, tail dropped.
"""
import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RUNS = Path('/root/runs/fineweb4way')
OUT = REPO / 'results' / 'small-model-v1'
MAX_BYTES = 16 * 1024 * 1024
ORDER = ('teacher', 'pretrained-student', 'ce-seed42', 'kd-seed42', 'ce-seed43', 'kd-seed43',
         'ce-seed44', 'kd-seed44')
POLICY = {'model_dtype': 'bfloat16', 'ce_dtype': 'float32', 'ce_reduction': 'shifted_valid_token_sum',
          'ignore_index': -100, 'batch_size': 1, 'sequence_length': 512,
          'deterministic_algorithms': True, 'cudnn_deterministic': True, 'cudnn_benchmark': False,
          'allow_tf32': False, 'cublas_workspace_config': ':4096:8'}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def roster(family):
    """Every scored model with the record that pins its selected checkpoint."""
    teacher = load(RUNS / family / 'base' / 'teacher.json')
    selection = load(RUNS / family / 'final_selection.json')
    records = {'teacher': ('teacher', None, teacher), 'pretrained-student': ('student', None, None)}
    for method in ('ce', 'kd'):
        chosen = selection[method]
        records[f'{method}-seed42'] = ('student', 42, load(
            RUNS / family / chosen['round'] / 'trials' / chosen['trial_id'] / 'result.json'))
        for seed in (43, 44):
            records[f'{method}-seed{seed}'] = ('student', seed, load(
                RUNS / family / f'seed{seed}' / 'trials' / f'{method}-1' / 'result.json'))
    return [(model_id, *records[model_id]) for model_id in ORDER]


def protocol_path(family):
    return OUT / 'test-protocol' / f'{family}.json'


def freeze(family):
    config = load(RUNS / family / 'base' / 'config.json')
    published = load(REPO / 'results/published/generic-kd-v1/cohort-manifest.json')['files']
    local = load(RUNS / family / 'base/raw/manifest.json')['files']
    if local != published:
        raise SystemExit('local cohort differs from the published generic-kd-v1 cohort')
    models = []
    for model_id, role, seed, record in roster(family):
        row = dict(model_id=model_id, role=role, seed=seed, model=config[role])
        if record is not None:
            if record['result']['status'] != 'completed':
                raise SystemExit(f'{model_id} is not completed')
            checkpoint = Path(record['result']['best_checkpoint'])
            row.update(checkpoint=str(checkpoint.relative_to(RUNS)), best_step=record['result']['best_step'],
                       checkpoint_manifest_sha256=sha_file(checkpoint / 'manifest.json'),
                       checkpoint_contract_sha256=sha_bytes(canonical(record['checkpoint_contract'])))
        models.append(row)
    import torch, transformers, peft
    protocol = dict(
        version=1, family=family, split='test', test_sha256=published['test.jsonl'],
        cohort_manifest='results/published/generic-kd-v1/cohort-manifest.json',
        numerical_policy=POLICY, models=models,
        final_selection_sha256=sha_file(RUNS / family / 'final_selection.json'),
        code=dict(evaluator_sha256=sha_file(__file__),
                  final_evaluation_sha256=sha_file(REPO / 'src/final_evaluation.py'),
                  torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__),
        rules=['Settings were chosen on validation only and frozen before this protocol.',
               'Each model is scored on test once. Outputs are write-once.',
               'No setting is changed and no training continues after test scoring.'])
    path = protocol_path(family)
    if path.exists():
        raise SystemExit(f'{path} already exists; a protocol is frozen once')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'froze {path.relative_to(REPO)} sha256={sha_file(path)}')


def build(model_id, role, seed, record, config):
    from scripts.run_research import make_model
    from src.research_models import load_model, restore_best
    if record is None:
        return load_model(config[role], device=config['training']['device'], **config['model'])
    cfg = copy.deepcopy(config)
    cfg['training']['seed'] = seed if seed is not None else cfg['training']['seed']
    model = make_model(role, cfg, {})
    restore_best(model, record['result']['best_checkpoint'], contract=record['checkpoint_contract'], freeze=True)
    return model


def check_model(model, device):
    """evaluate_final_batch's model checks, for the PEFT version in this environment.

    Same rule: a BF16 backbone whose only non-BF16 weights are vanilla LoRA A/B in FP32.
    This PEFT has no LoraLayer.lora_variant, so vanilla means DoRA is off for that adapter.
    A Teacher split over two GPUs skips only the same-device check.
    """
    import torch
    from peft.tuners.lora.layer import Linear as LoraLinear
    allowed = set()
    for module in model.modules():
        if isinstance(module, LoraLinear) and module.get_base_layer().weight.dtype == torch.bfloat16:
            for name, adapters in (('A', module.lora_A), ('B', module.lora_B)):
                for key, adapter in adapters.items():
                    if isinstance(adapter, torch.nn.Linear) and not module.use_dora.get(key, False):
                        allowed.add(id(adapter.weight))
    bad = [p for p in model.parameters() if p.is_floating_point() and p.dtype != torch.bfloat16
           and (p.dtype != torch.float32 or id(p) not in allowed)]
    if bad:
        raise ValueError('model must use a bfloat16 backbone with only FP32 vanilla LoRA A/B weights')
    if getattr(model, 'hf_device_map', None) is None and any(
            t.device != torch.device(device) for t in (*model.parameters(), *model.buffers())):
        raise ValueError('model/input device mismatch')


def score_model(model, batches):
    import torch
    from src.final_evaluation import final_token_ce_statistics, aggregate_final_token_ce
    check_model(model, batches[0][0].device)
    rows = []
    model.eval()
    for input_ids, labels in batches:
        if (tuple(input_ids.shape) != (1, 512) or tuple(labels.shape) != (1, 512)
                or bool(((labels < 0) & (labels != -100)).any())):
            raise ValueError('batch contract mismatch')
        with torch.inference_mode():
            rows.append(final_token_ce_statistics(model(input_ids=input_ids).logits, labels))
    return aggregate_final_token_ce(rows), rows


def batches_for(family, config, split, expected_sha):
    from transformers import AutoTokenizer
    from src.final_evaluation import load_final_heldout_batches
    spec = config['student']
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'],
                                              trust_remote_code=False, local_files_only=True)
    return load_final_heldout_batches(RUNS / family / 'base' / 'raw' / f'{split}.jsonl', expected_sha256=expected_sha,
                                      tokenizer=tokenizer, max_bytes=MAX_BYTES, device='cuda')


def run(family, split, wanted):
    from src.final_evaluation import apply_evaluation_backend_policy
    apply_evaluation_backend_policy(POLICY)   # before CUDA starts
    import torch
    config = load(RUNS / family / 'base' / 'config.json')
    files = load(RUNS / family / 'base/raw/manifest.json')['files']
    if split == 'test':
        path = protocol_path(family)
        committed = subprocess.run(['git', 'show', f'HEAD:{path.relative_to(REPO)}'], cwd=REPO,
                                   capture_output=True).stdout
        if not committed or committed != path.read_bytes():
            raise SystemExit('protocol is not committed unchanged; commit it before scoring')
        protocol = load(path)
        protocol_sha = sha_file(path)
        if protocol['code']['evaluator_sha256'] != sha_file(__file__):
            raise SystemExit('evaluator changed after the protocol was frozen')
        if protocol['final_selection_sha256'] != sha_file(RUNS / family / 'final_selection.json'):
            raise SystemExit('final selection changed after the protocol was frozen')
        pinned = {m['model_id']: m for m in protocol['models']}
    batches = batches_for(family, config, split, files[f'{split}.jsonl'])
    print(f'{family} {split}: {len(batches)} blocks', flush=True)
    for model_id, role, seed, record in roster(family):
        if model_id not in wanted:
            continue
        if split == 'test':
            out = OUT / 'final' / f'{family}--{model_id}.json'
            if out.exists():
                print(f'{model_id}: already scored, kept as is', flush=True)
                continue
            if record is not None and (
                    sha_file(Path(record['result']['best_checkpoint']) / 'manifest.json')
                    != pinned[model_id]['checkpoint_manifest_sha256']):
                raise SystemExit(f'{model_id}: checkpoint differs from the frozen protocol')
            with open(OUT / 'test-protocol' / 'test_access.log', 'a', encoding='utf-8') as log:
                log.write(f'{dt.datetime.now(dt.timezone.utc):%FT%TZ} {family} {model_id} scored once\n')
        model = build(model_id, role, seed, record, config)
        stats, rows = score_model(model, batches)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        line = f"{model_id}: {split} PPL {stats['ppl']:.4f} (CE {stats['ce']:.6f}, {stats['tokens']} tokens)"
        if record is not None and split == 'validation':
            line += f"; training log validation PPL {record['result']['best_validation']['ppl']:.4f}"
        print(line, flush=True)
        if split == 'test':
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(dict(model_id=model_id, protocol_sha256=protocol_sha,
                                           statistics={'ce_sum': stats['ce_sum'], 'tokens': stats['tokens']},
                                           status='completed', version=1), sort_keys=True) + '\n')
            blocks = OUT / 'blocks' / f'{family}--{model_id}.json'
            blocks.parent.mkdir(parents=True, exist_ok=True)
            blocks.write_text(json.dumps([[r['ce_sum'], r['tokens']] for r in rows]) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('freeze', 'check', 'score'))
    parser.add_argument('family', choices=('qwen3', 'kanana', 'gemma3'))
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.family)
    elif args.action == 'check':
        run(args.family, 'validation', set(args.models or ORDER))
    else:
        run(args.family, 'test', set(ORDER))


main()
