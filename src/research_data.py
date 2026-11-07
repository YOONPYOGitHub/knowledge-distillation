"""Bounded Korean corpus preparation; exact dedup and heuristic PII only."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

SPLITS = ('train', 'validation', 'test')
DATASET_REVISION = 'af9c13333eb981300149d5ca60a8e9d659b276b9'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n')


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stream_source(max_documents, *, loader=None):
    if loader is None:
        from datasets import load_dataset
        loader = load_dataset
    return loader('HuggingFaceFW/fineweb-2', 'kor_Hang', revision=DATASET_REVISION,
                  split='train', streaming=True).shuffle(seed=42, buffer_size=1000).take(max_documents)


def pack_split(path, tokenizer, seq_length):
    """Pack only within one document split, appending EOS after every document."""
    import torch
    if seq_length < 2 or tokenizer.eos_token_id is None:
        raise ValueError('seq_length >= 2 and EOS token required')
    pending, blocks, documents, total = [], [], 0, 0
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            ids = tokenizer.encode(json.loads(line)['text'], add_special_tokens=False) + [tokenizer.eos_token_id]
            documents += 1
            total += len(ids)
            pending.extend(ids)
            while len(pending) >= seq_length:
                tensor = torch.tensor(pending[:seq_length], dtype=torch.long)
                blocks.append({'input_ids': tensor, 'labels': tensor.clone(), 'attention_mask': torch.ones_like(tensor)})
                del pending[:seq_length]
    if not blocks:
        raise ValueError('empty packed split; increase documents or reduce seq_length')
    return blocks, dict(documents=documents, total_tokens=total, packed_tokens=total - len(pending),
                        dropped_tail_tokens=len(pending), blocks=len(blocks))


def validate_exposure_identity(identity):
    """Validate exclusion schema and digest before accessing the source."""
    if 'exposure_exclusion' in identity or 'exposure_exclusion_sha256' in identity:
        policy = identity.get('exposure_exclusion')
        if not isinstance(policy, dict) or set(policy) != {'source_ids', 'text_sha256'}:
            raise ValueError('invalid exposure exclusion policy')
        for key, values in policy.items():
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError('invalid exposure exclusion entries')
            if key == 'text_sha256' and any(re.fullmatch('[0-9a-f]{64}', v) is None for v in values):
                raise ValueError('invalid exposure exclusion text hash')
        if identity.get('exposure_exclusion_sha256') != digest(policy):
            raise ValueError('exposure exclusion SHA mismatch or missing')


def select_feasibility_documents(directory, *, train_documents, validation_documents):
    """Select diagnostic prefixes in their original splits; never access test."""
    if any(type(n) is not int or n < 1 for n in (train_documents, validation_documents)):
        raise ValueError('document counts must be a positive integer')
    splits = {}
    for split, count in (('train', train_documents), ('validation', validation_documents)):
        records = []
        with (Path(directory) / f'{split}.jsonl').open(encoding='utf-8') as stream:
            for _ in range(count):
                line = next(stream, None)
                if line is None:
                    raise ValueError(f'insufficient {split} documents')
                row = json.loads(line)
                if (not isinstance(row, dict)
                        or not isinstance(row.get('source_id'), str) or not row['source_id']
                        or not isinstance(row.get('text'), str) or not row['text']
                        or row.get('text_sha256') != hashlib.sha256(row['text'].encode('utf-8')).hexdigest()):
                    raise ValueError(f'invalid feasibility record in {split}')
                records.append(row)
        splits[split] = records
    return {'rule': 'canonical_split_file_prefix_no_resplit_v1', 'splits': splits}


def prepare_feasibility_subset(canonical, directory, tokenizer, seq_length, *, train_documents, validation_documents):
    """Freeze a diagnostic-only selection before packing; keep canonical raw intact."""
    canonical, directory = Path(canonical), Path(directory)
    manifest_bytes = (canonical / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    for split in ('train', 'validation'):
        name = f'{split}.jsonl'
        if file_hash(canonical / name) != manifest['files'][name]:
            raise ValueError('canonical split hash mismatch: ' + name)
    selected = select_feasibility_documents(canonical, train_documents=train_documents,
                                            validation_documents=validation_documents)
    selection = {'canonical_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                 'rule': selected['rule'], 'purpose': 'diagnostic_only_not_comparison_cohort',
                 'splits': {s: [{k: row[k] for k in ('source_id', 'text_sha256')} for row in rows]
                            for s, rows in selected['splits'].items()}}
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / 'feasibility-selection.json', selection)
    datasets, packing = {}, {}
    for split, rows in selected['splits'].items():
        path = directory / 'raw' / f'{split}.jsonl'
        atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        datasets[split], packing[split] = pack_split(path, tokenizer, seq_length)
    return {'selection': selection, 'datasets': datasets, 'packing': packing}


def prepare_raw(source, directory, identity):
    """Freeze document splits before packing; no claim of official heldout data."""
    validate_exposure_identity(identity)
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('raw data identity mismatch')
        for name, expected in manifest['files'].items():
            if not (directory / name).exists() or file_hash(directory / name) != expected:
                raise ValueError('raw data hash mismatch: ' + name)
        return manifest
    if any((directory / f'{s}.jsonl').exists() for s in SPLITS):
        raise ValueError('incomplete raw preparation: use a new run directory')
    accepted, seen = [], set()
    exclusion = identity.get('exposure_exclusion', {})
    excluded_ids = set(exclusion.get('source_ids', []))
    excluded_hashes = set(exclusion.get('text_sha256', []))
    rejected = dict(short=0, language=0, pii=0, duplicate=0)
    if exclusion:
        rejected['exposed'] = 0
    for index, row in enumerate(source):
        if index >= identity['max_documents']:
            break
        text = str(row.get('text', '')).strip()
        if len(text) < 200:
            rejected['short'] += 1
            continue
        if len(re.findall('[가-힣]', text)) / max(1, len(text)) < .2:
            rejected['language'] += 1
            continue
        if re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)(?:\+82[- .]?)?0?1[016789][- .]?\d{3,4}[- .]?\d{4}(?!\d)', text):
            rejected['pii'] += 1
            continue
        sha = hashlib.sha256(text.encode()).hexdigest()
        source_id = str(row.get('id', row.get('url', index)))
        if source_id in excluded_ids or sha in excluded_hashes:
            rejected['exposed'] += 1
            continue
        if sha in seen:
            rejected['duplicate'] += 1
            continue
        seen.add(sha)
        accepted.append({'source_id': str(row.get('id', row.get('url', index))), 'text_sha256': sha, 'text': text})
    if len(accepted) < 10:
        raise ValueError('at least 10 accepted documents required')
    accepted.sort(key=lambda row: digest([identity['seed'], row['text_sha256']]))
    holdout = max(1, len(accepted) // 10)
    groups = dict(train=accepted[2 * holdout:], validation=accepted[:holdout], test=accepted[holdout:2 * holdout])
    for split, records in groups.items():
        atomic_text(directory / f'{split}.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    manifest = {'identity': identity, 'counts': {s: len(v) for s, v in groups.items()}, 'rejected': rejected,
                'split_policy': 'deterministic hash-ranked 80/10/10 from source train; not official test',
                'cleaning': 'exact text SHA256 dedup; heuristic email/Korean mobile exclusion, not complete PII/minhash cleaning',
                'files': {f'{s}.jsonl': file_hash(directory / f'{s}.jsonl') for s in SPLITS}}
    manifest['fingerprint'] = digest(manifest)
    atomic_json(manifest_path, manifest)
    return manifest
