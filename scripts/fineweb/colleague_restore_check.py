#!/usr/bin/env python3
"""Restore one generic-kd-v1 Student with the colleague's own code and rescore validation.

Passes only if our validation CE matches the best_validation stored in the checkpoint.
usage: colleague_restore_check.py <unpacked checkpoint dir> <family config name>
"""
import json
import sys
from pathlib import Path

SOURCE = Path('/root/colleague/code/source')
sys.path.insert(0, str(SOURCE))
RUNS = Path('/root/runs/fineweb4way')


def main(ckpt, config_name):
    import yaml
    from src.numerical_policy import configure_numerics
    configure_numerics()
    import torch
    from transformers import AutoTokenizer
    from src.research_models import load_model, attach_lora, restore_best
    from src.resume_checkpoint import load_checkpoint
    from src.final_evaluation import load_final_heldout_batches, final_token_ce_statistics, aggregate_final_token_ce

    config = yaml.safe_load((SOURCE / 'configs' / config_name).read_text())
    manifest = json.loads((Path(ckpt) / 'manifest.json').read_text())
    state = load_checkpoint(ckpt, expected_contract=manifest['contract'])
    recorded = state['best_validation']
    print('checkpoint step', state['step'], 'recorded best_validation', recorded, flush=True)

    spec = config['student']
    model = load_model(spec, device='cuda', dtype=config['model']['dtype'], attention=config['model']['attention'])
    model = attach_lora(model, config['adapter'], seed=42)
    restore_best(model, ckpt, contract=manifest['contract'], freeze=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False)
    raw = RUNS / 'qwen3' / 'base' / 'raw'
    sha = json.loads((raw / 'manifest.json').read_text())['files']['validation.jsonl']
    batches = load_final_heldout_batches(raw / 'validation.jsonl', expected_sha256=sha, tokenizer=tokenizer,
                                         max_bytes=16 * 1024 * 1024, device='cuda')
    rows = []
    with torch.inference_mode():
        for input_ids, labels in batches:
            rows.append(final_token_ce_statistics(model(input_ids=input_ids).logits, labels))
    ours = aggregate_final_token_ce(rows)
    print(f"ours: validation CE {ours['ce']:.6f} PPL {ours['ppl']:.4f} tokens {ours['tokens']} blocks {len(batches)}")
    diff = abs(ours['ce'] - recorded['ce'])
    tokens_match = ours['tokens'] == recorded.get('tokens', ours['tokens'])
    print(f"recorded CE {recorded['ce']:.6f}; |difference| {diff:.2e}; tokens match {tokens_match}")
    print('RESTORE CHECK', 'PASSED' if diff < 1e-3 and tokens_match else 'FAILED')


main(sys.argv[1], sys.argv[2])
