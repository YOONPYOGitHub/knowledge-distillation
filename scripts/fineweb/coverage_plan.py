#!/usr/bin/env python3
"""Common byte coverage for scoring six tokenizers on the same validation documents.

For each document, find the first character position at or after MIN_PREFIX_BYTES where
every tokenizer starts a token cleanly (the previous token ends exactly there). Every model
then scores only the tokens from that position to the end of the document, so all models
predict exactly the same UTF-8 bytes. Documents with no common boundary are excluded for all.

Writes coverage.json: per document text_sha256, start char/byte, covered bytes, and per
tokenizer the scored token count and a hash of the token IDs, so each scorer can verify it
tokenizes identically in its own environment.
"""
import hashlib
import json
import sys
from pathlib import Path

RAW = Path('/root/runs/fineweb4way/qwen3/base/raw')
MIN_PREFIX_BYTES = 64
TOKENIZERS = {
    'qwen3': ('Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
    'kanana': ('kakaocorp/kanana-1.5-2.1b-instruct-2505', '7df4bc35ccd610e451809d7106e1c3cf82bfd44c'),
    'gemma3': ('google/gemma-3-1b-it', 'dcc83ea841ab6100d6b47a070329e1ba4cf78752'),
    'qwen35': ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
    'gemma4': ('google/gemma-4-E2B-it', '3e22461f65e89153144f8adb70e3b8c2cc9845a7'),
    'lfm25': ('LiquidAI/LFM2.5-2.6B', '654f9463ce32b05d0429d76fe1f580b27d4c1ac0'),
}


def clean_starts(tokenizer, text):
    """Character positions where a token begins exactly where the previous one ended."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets, ids = enc['offset_mapping'], enc['input_ids']
    starts, prev_end = {}, 0
    for i, (start, end) in enumerate(offsets):
        if start == prev_end and end > start:
            starts.setdefault(start, i)
        prev_end = max(prev_end, end)
    if prev_end != len(text):
        raise ValueError('offsets do not reach the end of the document')
    return starts, ids


def main(out_path):
    from transformers import AutoTokenizer
    sha = json.loads((RAW / 'manifest.json').read_text())['files']['validation.jsonl']
    payload = (RAW / 'validation.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha:
        raise SystemExit('validation file differs from the frozen cohort')
    from transformers import PreTrainedTokenizerFast

    def load(repo, rev):
        try:
            return AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=False)
        except Exception:  # Transformers 5 rejects Kanana's model config; the tokenizer file alone is enough
            return PreTrainedTokenizerFast.from_pretrained(repo, revision=rev)
    toks = {k: load(i, r) for k, (i, r) in TOKENIZERS.items()}
    docs, skipped = [], 0
    for line in payload.decode('utf-8').splitlines():
        text = json.loads(line)['text']
        info = {k: clean_starts(t, text) for k, t in toks.items()}
        common = set.intersection(*(set(s) for s, _ in info.values()))
        candidates = sorted(p for p in common if len(text[:p].encode('utf-8')) >= MIN_PREFIX_BYTES)
        if not candidates:
            skipped += 1
            docs.append(dict(text_sha256=hashlib.sha256(text.encode()).hexdigest(), excluded=True))
            continue
        start = candidates[0]
        row = dict(text_sha256=hashlib.sha256(text.encode()).hexdigest(), excluded=False, start_char=start,
                   start_byte=len(text[:start].encode('utf-8')),
                   covered_bytes=len(text[start:].encode('utf-8')), tokenizers={})
        for k, (starts, ids) in info.items():
            first = starts[start]
            row['tokenizers'][k] = dict(first_scored_index=first, total_tokens=len(ids),
                                        scored_tokens=len(ids) - first,
                                        ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        docs.append(row)
    kept = [d for d in docs if not d['excluded']]
    summary = dict(min_prefix_bytes=MIN_PREFIX_BYTES, documents=len(docs), excluded=skipped,
                   covered_bytes=sum(d['covered_bytes'] for d in kept),
                   scored_tokens={k: sum(d['tokenizers'][k]['scored_tokens'] for d in kept) for k in TOKENIZERS},
                   median_start_byte=sorted(d['start_byte'] for d in kept)[len(kept) // 2],
                   tokenizers={k: dict(id=i, revision=r) for k, (i, r) in TOKENIZERS.items()},
                   validation_sha256=sha)
    Path(out_path).write_text(json.dumps(dict(summary=summary, documents=docs)) + '\n')
    print(json.dumps(summary, indent=1))


main(sys.argv[1])
