#!/usr/bin/env python3
"""Read-only aggregate verification; stdlib only. No corpus or weights required."""
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

PROTOCOLS = {
    'qwen35': 'fe0a525699e8c26318565e94fe47e156c9d210449a111ef8dd00c4d6c2d9b5b1',
    'gemma4': '141c71995cf098c50a8cd791551055d729be7252a22ab7d1ac8f6b34d118c1bf',
    'lfm25': 'c6223d65a4601522671da8cb2f011d72ba036b801d65ad8c4109bce37c3ee645',
}
CANONICAL = '53e4e887ec898c2cce985a8027f0dec33360761ba16f705a3c95d244887f904a'
RESULT_PREFIX = 'results/published/generic-kd-v1/'
DOC_PREFIX = 'docs/published/generic-kd-v1/'


def validate_result(data, protocol):
    """Reject non-aggregate schema; return token-weighted CE and PPL."""
    expected = {'version', 'model_id', 'status', 'protocol_sha256', 'statistics'}
    if set(data) != expected or type(data['version']) is not int or data['version'] != 1:
        raise ValueError('unexpected aggregate schema')
    if data['status'] != 'completed' or not isinstance(data['model_id'], str):
        raise ValueError('not completed or invalid model identity')
    if not re.fullmatch(r'[0-9a-f]{64}', protocol) or data['protocol_sha256'] != protocol:
        raise ValueError('protocol mismatch')
    stats = data['statistics']
    if set(stats) != {'ce_sum', 'tokens'}:
        raise ValueError('unexpected statistics fields')
    n, total = stats['tokens'], stats['ce_sum']
    if type(n) is not int or n <= 0 or type(total) not in (int, float):
        raise ValueError('invalid token count or CE sum')
    if not math.isfinite(total) or total < 0:
        raise ValueError('nonfinite or negative CE sum')
    ce = total / n
    try:
        ppl = math.exp(ce)
    except OverflowError as exc:
        raise ValueError('PPL overflow') from exc
    return ce, ppl


def verify(results_dir):
    """Verify only the two publication directories. Never open provenance paths."""
    results_dir = Path(results_dir).resolve()
    root = results_dir.parents[2]
    docs_dir = root / DOC_PREFIX
    manifest_path = results_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def close(actual, expected, label):
        actual, expected = float(actual), float(expected)
        require(math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                'metric mismatch: ' + label)

    def safe_path(relative):
        p = PurePosixPath(relative)
        require(not p.is_absolute() and '..' not in p.parts
                and str(p) == relative
                and relative.startswith((RESULT_PREFIX, DOC_PREFIX)),
                'outside publication: ' + relative)
        target = root / relative
        require(not any(x.is_symlink() for x in [target, *target.parents]
                        if x != root and root in x.parents), 'symlink refused')
        require(target.is_file() and target.stat().st_size <= 2_000_000,
                'missing or oversized publication leaf')
        return target

    inventory = {}
    for entry in manifest['files']:
        name = entry['path']
        require(name not in inventory and name != RESULT_PREFIX + 'manifest.json',
                'duplicate inventory or self inclusion')
        target = safe_path(name)
        data = target.read_bytes()
        require(len(data) == entry['bytes']
                and hashlib.sha256(data).hexdigest() == entry['sha256'],
                'SHA/bytes mismatch: ' + name)
        inventory[name] = entry
    actual_paths = set()
    for directory in (results_dir, docs_dir):
        for path in directory.rglob('*'):
            require(not path.is_symlink(), 'publication symlink refused')
            if path.is_file() and path != manifest_path:
                actual_paths.add(path.relative_to(root).as_posix())
    require(actual_paths == set(inventory), 'publication inventory mismatch')

    with (results_dir / 'results.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    expected_models = {'teacher', 'pretrained-student'} | {
        f'{arm}-seed{seed}' for arm in ('ce', 'kd') for seed in (42, 43, 44)}
    expected_keys = {(family, model) for family in PROTOCOLS for model in expected_models}
    metrics = {}
    for row in rows:
        key = (row['family'], row['model_id'])
        require(key in expected_keys and key not in metrics, 'unexpected or duplicate result')
        family, model = key
        require(row['published_path'] == f'final/{family}--{model}.json',
                'unstable published result path')
        target = safe_path(RESULT_PREFIX + row['published_path'])
        payload = target.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        require(digest == row['original_sha256'], 'original result SHA mismatch')
        mappings = [x for x in manifest['provenance']
                    if x['published_path'] == RESULT_PREFIX + row['published_path']]
        require(len(mappings) == 1 and mappings[0]['source_path'] == row['original_path']
                and mappings[0]['source_sha256'] == digest
                and mappings[0]['published_sha256'] == digest,
                'source/result provenance mismatch')
        data = json.loads(payload)
        ce, ppl = validate_result(data, PROTOCOLS[family])
        require(data['model_id'] == model and row['protocol_sha256'] == PROTOCOLS[family]
                and row['status'] == 'completed' and row['split'] == 'test',
                'result identity/status mismatch')
        role = ('Teacher' if model == 'teacher' else 'pretrainedStudent'
                if model == 'pretrained-student' else 'StudentCE'
                if model.startswith('ce-') else 'StudentKD')
        seed = model.split('seed')[-1] if '-seed' in model else ''
        require(row['role'] == role and row['seed'] == seed, 'role/seed mismatch')
        require(int(row['tokens']) == data['statistics']['tokens'], 'token mismatch')
        close(row['ce_sum'], data['statistics']['ce_sum'], 'CE sum')
        close(row['ce'], ce, 'CE')
        close(row['ppl'], ppl, 'PPL')
        metrics[key] = ppl
    require(set(metrics) == expected_keys and len(rows) == 24, 'expected exactly24 results')
    for family in PROTOCOLS:
        tokens = {int(x['tokens']) for x in rows if x['family'] == family}
        require(len(tokens) == 1, 'within-family token mismatch')

    with (results_dir / 'paired_effects.csv').open(newline='', encoding='utf-8') as f:
        pairs = list(csv.DictReader(f))
    seen = set()
    for pair in pairs:
        key = pair['family'], int(pair['seed'])
        family, seed = key
        require(family in PROTOCOLS and seed in (42, 43, 44) and key not in seen,
                'unexpected or duplicate pair')
        seen.add(key)
        ce, kd = metrics[family, f'ce-seed{seed}'], metrics[family, f'kd-seed{seed}']
        pretrained = metrics[family, 'pretrained-student']
        close(pair['ce_ppl'], ce, 'paired CE PPL')
        close(pair['kd_ppl'], kd, 'paired KD PPL')
        close(pair['kd_relative_ppl_change_percent'], (kd / ce - 1) * 100, 'KD effect')
        close(pair['pretrained_to_ce_ppl_decrease_percent'],
              (1 - ce / pretrained) * 100, 'adaptation effect')
        require(pair['selection_seed_reused'] == str(seed == 42), 'seed reuse mismatch')
    require(len(seen) == 9, 'expected exactly9 pairs')

    cohort_bytes = (results_dir / 'cohort-manifest.json').read_bytes()
    require(hashlib.sha256(cohort_bytes).hexdigest() == CANONICAL, 'canonical SHA mismatch')
    cohort = json.loads(cohort_bytes)
    require(cohort['counts'] == {'train': 6814, 'validation': 851, 'test': 851}
            and cohort['identity']['subset'] == 'kor_Hang', 'cohort identity mismatch')
    selections = json.loads((results_dir / 'selected-metadata.json').read_bytes())['selections']
    selected = {}
    for item in selections:
        key = item['family'], item['model_key']
        require(key in expected_keys and key[1] != 'pretrained-student' and key not in selected,
                'unexpected or duplicate selection')
        require(0 < item['selected_step'] <= 3000, 'selected step outside declared budget')
        for field in ('contract_sha256', 'checkpoint_manifest_sha256',
                      'checkpoint_complete_sha256', 'cohort_identity_sha256'):
            require(bool(re.fullmatch(r'[0-9a-f]{64}', item[field])), 'malformed metadata SHA')
        selected[key] = item
    require(len(selected) == 21, 'expected exactly21 selections')
    for family in PROTOCOLS:
        for seed in (42, 43, 44):
            ce_item, kd_item = (selected[family, f'{arm}-seed{seed}'] for arm in ('ce', 'kd'))
            require(ce_item['settings']['seed'] == kd_item['settings']['seed'] == seed,
                    'selected seed mismatch')
            require(ce_item['identity_scalars']['initial_adapter_sha256']
                    == kd_item['identity_scalars']['initial_adapter_sha256'],
                    'paired initialization mismatch')
    census = json.loads((results_dir / 'run-census.json').read_bytes())
    require(census['classification_counts'] == {'common_student': 60, 'formal_teacher': 3,
            'historical_pilot': 16, 'historical_diagnostic': 10}, 'reported census mismatch')
    return {'status': 'verified_local_aggregates_only', 'results': len(rows),
            'pairs': len(seen), 'selected_metadata': len(selected),
            'inventory_files_excluding_manifest': len(inventory),
            'inventory_bytes_excluding_manifest': sum(x['bytes'] for x in inventory.values()),
            'canonical_sha256': CANONICAL}


def main():
    """CLI audit guard: publication-only reads; no writes/network/subprocesses."""
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    allowed = (directory, (root / DOC_PREFIX).resolve())

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if not isinstance(path, (str, bytes, Path)):
                raise PermissionError('file-descriptor open refused')
            if isinstance(path, bytes):
                path = path.decode()
            target = Path(path).resolve()
            if not any(target == base or base in target.parents for base in allowed):
                raise PermissionError('read outside publication refused')
            # os flags: WRONLY=1, RDWR=2; creation/truncation also imply writes.
            if (mode and any(x in mode for x in 'wax+')) or (flags and flags & 3):
                raise PermissionError('writes refused')
        if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.exec',
                             'os.spawn', 'os.remove', 'os.rename', 'os.mkdir',
                             'os.rmdir', 'os.link', 'os.symlink', 'os.truncate')):
            raise PermissionError('external or mutating operation refused')

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    try:
        print(json.dumps(verify(directory), indent=2, sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        print('verification failed: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
