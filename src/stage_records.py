"""Read-only reconciliation of stage summaries against durable engine state.

Checkpoint hashes establish internal integrity, not authenticity: as with the
engine loader, only use trusted, engine-owned local checkpoint directories.
"""
import json
from pathlib import Path

from .resume_checkpoint import load_checkpoint


def _real_path(path):
    if path.is_symlink():
        raise ValueError('Symlinked stage/checkpoint path')
    return path


def _generation(output, reference):
    if (not isinstance(reference, str) or Path(reference).is_absolute() or
            len(Path(reference).parts) != 2 or Path(reference).parts[0] != 'checkpoints' or
            any(part in ('.', '..') for part in reference.split('/'))):
        raise ValueError('Unsafe checkpoint generation path')
    path = _real_path(output / reference)
    if path.resolve().parent != output / 'checkpoints':
        raise ValueError('Checkpoint path escapes generation directory')
    for name in ('manifest.json', 'COMPLETE', 'state.pt'):
        _real_path(path / name)
    if (path / 'latest.json').exists() or (path / 'latest.json').is_symlink():
        raise ValueError('Unexpected nested generation pointer path')
    return path


def reconcile_completed_stage(directory, stage, *, contract=None):
    """Reconstruct a completed result without writing or repairing any artifact.

    Call under the run's existing lock if another process could change latest.
    Only latest must be completed: an earlier selected best is normally running.
    Missing/malformed/inconsistent evidence is an error, never a cached success.
    """
    if stage not in ('teacher', 'ce', 'kd'):
        raise ValueError('Unknown research stage')
    try:
        return _reconcile(Path(directory).resolve(), stage, contract)
    except (OSError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError(stage + ' stage must be completed first; evidence missing or malformed') from exc


def _reconcile(directory, stage, contract):
    output = _real_path(directory / stage)
    _real_path(output / 'checkpoints')
    _real_path(output / 'latest.json')
    record = json.loads(_real_path(directory / f'{stage}.json').read_text())
    if contract is not None and record['contract'] != json.loads(json.dumps(contract, allow_nan=False)):
        raise ValueError(stage + ' stage contract mismatch')
    if (set(record['contract']) != {'settings', 'identity'} or
            any(record['checkpoint_contract'].get(key) != value
                for key, value in record['contract'].items())):
        raise ValueError(stage + ' checkpoint contract does not match stage contract')
    pointer = json.loads((output / 'latest.json').read_text())
    latest_path = _generation(output, pointer['checkpoint'])
    latest = load_checkpoint(output, expected_contract=record['checkpoint_contract'])
    if latest['status'] != 'completed':
        raise ValueError(stage + ' latest durable checkpoint is not completed')
    best_path = _generation(output, latest['best_checkpoint'])
    selected = load_checkpoint(best_path, expected_contract=record['checkpoint_contract'])
    if (selected['step'] != latest['best_step'] or
            selected['best_step'] != selected['step'] or
            selected['best_validation'] != latest['best_validation']):
        raise ValueError(stage + ' selected best step/metrics mismatch')
    selected_history = selected['history']
    metrics = {k: v for k, v in selected_history[-1].items() if k != 'step'}
    if (selected_history != latest['history'][:len(selected_history)] or
            selected_history[-1]['step'] != selected['step'] or
            metrics != latest['best_validation']):
        raise ValueError(stage + ' selected best history mismatch')
    result = {
        'status': latest['status'], 'final_step': latest['step'],
        'best_step': latest['best_step'],
        'initial_validation': {k: v for k, v in latest['history'][0].items() if k != 'step'},
        'best_validation': latest['best_validation'],
        'best_checkpoint': str(best_path),
        'last_checkpoint': str(latest_path),
        'stop_reason': latest['stop_reason'], 'history': latest['history'],
    }
    if record['result'] != result:
        raise ValueError(stage + ' result differs from durable checkpoint')
    return {**record, 'result': result}
