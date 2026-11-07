"""Immutable local checkpoint generations; COMPLETE is the commit record."""
import hashlib
import json
import os
import uuid
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path, value):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def save_checkpoint(output_dir, state, contract):
    output_dir = Path(output_dir)
    generations = output_dir / 'checkpoints'
    generations.mkdir(parents=True, exist_ok=True)
    generation = generations / f"step-{state['step']:08d}-{uuid.uuid4().hex}"
    generation.mkdir()
    state = dict(state)
    if state.get('best_checkpoint') is None:
        state['best_checkpoint'] = str(generation.relative_to(output_dir))
    with open(generation / 'state.pt', 'wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {'version': 1, 'step': state['step'], 'contract': contract,
                'identity': contract.get('identity', {}),
                'files': {'state.pt': {'sha256': sha256_file(generation / 'state.pt'),
                                       'size': (generation / 'state.pt').stat().st_size}}}
    _write_json(generation / 'manifest.json', manifest)
    _write_json(generation / 'COMPLETE', {'manifest_sha256': sha256_file(generation / 'manifest.json')})
    _fsync_directory(generation)
    _fsync_directory(generations)
    pointer = output_dir / f'.latest-{uuid.uuid4().hex}.tmp'
    _write_json(pointer, {'checkpoint': str(generation.relative_to(output_dir)),
                          'manifest_sha256': sha256_file(generation / 'manifest.json')})
    os.replace(pointer, output_dir / 'latest.json')
    _fsync_directory(output_dir)
    return str(generation.resolve())


def load_checkpoint(path, *, expected_contract=None):
    """Verify before deserializing. Only load trusted engine-owned checkpoints."""
    path = Path(path).resolve()
    if path.name == 'latest.json':
        path = path.parent
    pointer_hash = None
    if (path / 'latest.json').is_file():
        pointer = json.loads((path / 'latest.json').read_text())
        pointer_hash = pointer['manifest_sha256']
        target = (path / pointer['checkpoint']).resolve()
        if target.parent != (path / 'checkpoints').resolve():
            raise ValueError('checkpoint pointer escapes generation directory')
        path = target
    if not (path / 'COMPLETE').is_file():
        raise ValueError('checkpoint is incomplete: COMPLETE missing')
    complete = json.loads((path / 'COMPLETE').read_text())
    manifest_hash = sha256_file(path / 'manifest.json')
    if manifest_hash != complete['manifest_sha256'] or (pointer_hash is not None and manifest_hash != pointer_hash):
        raise ValueError('checkpoint manifest hash mismatch')
    manifest = json.loads((path / 'manifest.json').read_text())
    if expected_contract is not None and manifest['contract'] != expected_contract:
        raise ValueError('checkpoint identity/settings/data contract mismatch')
    if manifest['files'].keys() != {'state.pt'}:
        raise ValueError('unexpected checkpoint weight payloads')
    for name, metadata in manifest['files'].items():
        if sha256_file(path / name) != metadata['sha256'] or (path / name).stat().st_size != metadata['size']:
            raise ValueError('checkpoint payload hash/size mismatch')
    state = torch.load(path / 'state.pt', map_location='cpu', weights_only=True)
    if state['step'] != manifest['step']:
        raise ValueError('checkpoint step mismatch')
    return state
