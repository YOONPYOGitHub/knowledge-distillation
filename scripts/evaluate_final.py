"""Frozen-bundle final evaluation entrypoint; no training or selection."""
import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def execute_bundle(path, expected_sha256, *, device='cuda'):
    bundle_path = Path(path).absolute()
    if any(parent.is_symlink() for parent in (bundle_path, *bundle_path.parents)):
        raise ValueError('unsafe evaluation bundle')
    try:
        fd = os.open(bundle_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError('unsafe evaluation bundle') from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('unsafe evaluation bundle')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            payload = stream.read(8 * 1024 * 1024 + 1)
        if len(payload) > 8 * 1024 * 1024:
            raise ValueError('evaluation bundle exceeds byte limit')
    finally:
        os.close(fd)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('evaluation bundle SHA mismatch')
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate evaluation bundle field')
            result[key] = value
        return result
    bundle = json.loads(payload, object_pairs_hook=unique_fields)
    expected_fields = {'protocol', 'selections', 'cohort', 'operation',
                       'numerical_policy', 'exposure_adjudication',
                       'exposure_sha256', 'sources', 'commits', 'heldout'}
    if type(bundle) is not dict or set(bundle) != expected_fields:
        raise ValueError('invalid evaluation bundle schema')
    root = Path(__file__).resolve().parents[1]
    required = [root / name for name in (
        'scripts/evaluate_final.py', 'src/final_evaluation.py',
        'src/research_runner.py', 'src/research_models.py', 'src/research_data.py',
        'src/resume_checkpoint.py', 'src/numerical_policy.py')]
    declared = bundle['sources']
    if (type(declared) is not list
            or any(type(item) is not dict or set(item) != {'path', 'sha256'}
                   or type(item['path']) is not str or type(item['sha256']) is not str
                   for item in declared)
            or [item['path'] for item in declared] != [str(path) for path in required]):
        raise ValueError('execution source coverage mismatch')
    sources = []
    for item in bundle['sources']:
        source_path = Path(item['path'])
        if any(parent.is_symlink() for parent in (source_path, *source_path.parents)):
            raise ValueError('unsafe evaluation source')
        try:
            source_fd = os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as exc:
            raise ValueError('unsafe evaluation source') from exc
        try:
            if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                raise ValueError('unsafe evaluation source')
            with os.fdopen(source_fd, 'rb', closefd=False) as stream:
                data = stream.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError('evaluation source exceeds byte limit')
        finally:
            os.close(source_fd)
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('evaluation source SHA mismatch')
        sources.append(data)
    source_bytes = b''.join(sources)
    sys.path.insert(0, str(root))
    from src import final_evaluation as fe
    commits = {name: {'marker_bytes': item['marker'].encode('utf-8'),
                      'manifest_bytes': item['manifest'].encode('utf-8'),
                      'max_bytes': item['max_bytes']}
               for name, item in bundle['commits'].items()}
    heldout = bundle['heldout']
    manifest_bytes = heldout['raw_manifest'].encode('utf-8')
    if (len(manifest_bytes) > 1024 * 1024
            or hashlib.sha256(manifest_bytes).hexdigest() != bundle['cohort']['provenance']['raw_manifest_hash']):
        raise ValueError('heldout raw manifest binding mismatch')
    student_models = {(entry['model_id'], entry['model_revision'])
                      for entry in bundle['protocol']['models']
                      if entry['id'] != 'teacher'}
    tokenizer = heldout['tokenizer_spec']
    if student_models != {(tokenizer['id'], tokenizer['revision'])}:
        raise ValueError('tokenizer binding mismatch')
    def load(entry):
        return fe.load_final_roster_model(entry, bundle['selections'], commits,
                                         device=device)
    def batches(entry):
        return fe.load_bound_final_heldout(
            heldout['raw_path'], raw_manifest_bytes=heldout['raw_manifest'].encode('utf-8'),
            expected_manifest_sha256=bundle['cohort']['provenance']['raw_manifest_hash'],
            tokenizer_spec=heldout['tokenizer_spec'], max_bytes=heldout['max_bytes'],
            device=device)
    return fe.run_authenticated_evaluation(
        bundle['protocol'], bundle['selections'], bundle['cohort'],
        operation_bytes=json.dumps(bundle['operation'], sort_keys=True,
                                   separators=(',', ':'), allow_nan=False).encode('utf-8'),
        source_bytes=source_bytes, load_model=load, iter_batches=batches,
        numerical_policy=bundle['numerical_policy'],
        exposure_adjudication_bytes=bundle['exposure_adjudication'].encode('utf-8'),
        expected_exposure_sha256=bundle['exposure_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    result = execute_bundle(args.bundle, args.bundle_sha256, device=args.device)
    print(json.dumps({name: row['statistics'] for name, row in result.items()},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
