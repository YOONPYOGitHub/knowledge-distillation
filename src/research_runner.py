"""Download-free causal-LM stage engine. Validation only; never accepts a test set."""
import hashlib
import json
import math
import random
import signal
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .resume_checkpoint import load_checkpoint, save_checkpoint


@contextmanager
def _termination_flag():
    flag = {'stop': False}

    def request_stop(signum, frame):
        flag['stop'] = True

    owns_handler = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if owns_handler else None
    if owns_handler:
        signal.signal(signal.SIGTERM, request_stop)
    try:
        yield flag
    finally:
        if owns_handler:
            signal.signal(signal.SIGTERM, previous)


def _data_hash(dataset):
    digest = hashlib.sha256()
    digest.update(str(len(dataset)).encode())
    for index in range(len(dataset)):
        row = dataset[index]
        for key in ('input_ids', 'labels', 'attention_mask'):
            digest.update(key.encode())
            if key not in row:
                digest.update(b'absent')
                continue
            value = torch.as_tensor(row[key]).detach().cpu().contiguous()
            digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _batch(dataset, indices, device):
    rows = [dataset[i] for i in indices]
    return {key: torch.stack([torch.as_tensor(row[key]) for row in rows]).to(device)
            for key in ('input_ids', 'labels', 'attention_mask') if key in rows[0]}


def _ce(logits, labels):
    logits, labels = logits[:, :-1].float(), labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction='sum'), int((labels != -100).sum())


def _forward(model, batch):
    return model(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask')).logits


def _distillation_loss(student, teacher, labels, settings):
    size = settings['vocab_size'] or student.shape[-1]
    if student.shape[:-1] != teacher.shape[:-1]:
        raise ValueError('teacher/student token shapes mismatch')
    if size > min(student.shape[-1], teacher.shape[-1]):
        raise ValueError('KD vocab_size exceeds teacher/student logits')
    if not settings['vocab_size'] and student.shape[-1] != teacher.shape[-1]:
        raise ValueError('full-vocabulary KD requires aligned teacher/student vocabulary')
    temperature = settings['temperature']
    log_s = (student[:, :-1, :size].float() / temperature).log_softmax(-1)
    log_t = (teacher[:, :-1, :size].float() / temperature).log_softmax(-1)
    if settings['divergence'] == 'forward_kl':
        terms = log_t.exp() * (log_t - log_s)
    else:
        terms = log_s.exp() * (log_s - log_t)
    return terms.sum(-1)[labels[:, 1:] != -100].mean() * temperature ** 2


def _validate(model, dataset, settings):
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(dataset), settings['batch_size']):
            batch = _batch(dataset, range(start, min(start + settings['batch_size'], len(dataset))), settings['device'])
            loss, count = _ce(_forward(model, batch), batch['labels'])
            total += loss.item()
            tokens += count
    if not tokens:
        raise ValueError('validation has no supervised tokens')
    ce = total / tokens
    if not math.isfinite(ce) or ce > math.log(float.fromhex('0x1.fffffffffffffp+1023')):
        raise FloatingPointError('validation CE/perplexity is not finite')
    return {'ce': ce, 'ppl': math.exp(ce), 'tokens': tokens}


def _capture_rng():
    np_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def _model_state(model):
    # Frozen base weights are reconstructed by the caller using identity revisions.
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    names.update(name for name, _ in model.named_buffers())
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def _restore_model(model, state):
    expected = _model_state(model)
    if expected.keys() != state.keys():
        raise ValueError('model trainable/buffer keys mismatch')
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f'model shape/dtype mismatch: {name}')
    model.load_state_dict(state, strict=False)


def _settings(value, teacher):
    defaults = {'batch_size': 1, 'epochs': 1, 'patience': 0, 'min_delta': 0.0,
                'alpha': 1.0, 'temperature': 1.0, 'divergence': 'forward_kl',
                'vocab_size': 0, 'device': 'cpu', 'checkpoint_interval_steps': 100}
    allowed = set(defaults) | {'seed', 'max_steps', 'validation_interval'}
    if set(value) - allowed:
        raise ValueError(f'unsupported settings: {sorted(set(value) - allowed)}')
    value = {**defaults, **value}
    for key in ('seed', 'max_steps', 'validation_interval', 'batch_size', 'epochs',
                'patience', 'vocab_size', 'checkpoint_interval_steps'):
        minimum = 0 if key in ('seed', 'patience', 'vocab_size') else 1
        if type(value.get(key)) is not int or value[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    for key in ('alpha', 'temperature', 'min_delta'):
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)) or not math.isfinite(value[key]):
            raise ValueError(f'{key} must be finite numeric')
    if not 0 <= value['alpha'] <= 1 or value['temperature'] <= 0 or value['min_delta'] < 0:
        raise ValueError('invalid alpha, temperature, or min_delta')
    if value['divergence'] not in ('forward_kl', 'reverse_kl'):
        raise ValueError('unsupported divergence')
    if value['alpha'] < 1 and teacher is None:
        raise ValueError('KD requires a frozen teacher')
    if not isinstance(value['device'], str):
        raise ValueError('device must be a string')
    return value


def _optimizer_contract(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        if any(id(p) not in names for p in group['params']):
            raise ValueError('optimizer contains parameters outside student model')
        groups.append({**{k: v for k, v in group.items() if k != 'params'},
                       'params': [names[id(p)] for p in group['params']]})
    return {'class': type(optimizer).__module__ + '.' + type(optimizer).__qualname__, 'groups': groups}


def train_stage(model, train_dataset, validation_dataset, optimizer, output_dir: Path, *,
                teacher=None, settings: dict, identity: dict, resume: bool = False) -> dict:
    """Train at safe optimizer boundaries; resume never silently starts afresh.

    max_steps and epochs are independent upper bounds. All checkpoints contain only
    trainable parameters and buffers, not the frozen base or an external teacher.
    """
    settings = _settings(settings, teacher)
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError('training and validation datasets must not be empty')
    if teacher is not None:
        student_parameters = {id(p) for p in model.parameters()}
        if teacher is model or any(id(p) in student_parameters for p in teacher.parameters()):
            raise ValueError('external teacher must not share student parameters')
    output_dir = Path(output_dir).resolve()
    contract = json.loads(json.dumps({'settings': settings, 'identity': identity,
                          'optimizer': _optimizer_contract(model, optimizer),
                          'data': {'train': _data_hash(train_dataset),
                                   'validation': _data_hash(validation_dataset)}},
                                    sort_keys=True, allow_nan=False))
    state = load_checkpoint(output_dir, expected_contract=contract) if resume else None
    if state is not None:
        best_target = (output_dir / state['best_checkpoint']).resolve()
        if best_target.parent != (output_dir / 'checkpoints').resolve():
            raise ValueError('best checkpoint path escapes generation directory')
        selected = load_checkpoint(best_target, expected_contract=contract)
        if selected['step'] != state['best_step']:
            raise ValueError('best checkpoint step mismatch')
    if not resume and output_dir.exists():
        raise ValueError('output directory already exists; explicit resume=True required')
    model.to(settings['device'])
    if teacher is not None:
        teacher.to(settings['device'])
        teacher.requires_grad_(False)
        teacher.eval()
        for param in teacher.parameters():
            param.grad = None
    if state is not None:
        _restore_model(model, state['model_state'])
        optimizer.load_state_dict(state['optimizer'])
        _restore_rng(state['rng'])
        step, cursor = state['step'], state['cursor']
        history, best = state['history'], state['best_validation']
        best_step, best_path = state['best_step'], str(best_target)
        stale = state['stale']
        pointer = json.loads((output_dir / 'latest.json').read_text())
        last_path = str((output_dir / pointer['checkpoint']).resolve())
    else:
        random.seed(settings['seed'])
        np.random.seed(settings['seed'])
        torch.manual_seed(settings['seed'])
        step, cursor, stale = 0, {'epoch': 0, 'batch': 0}, 0
        best = _validate(model, validation_dataset, settings)
        history = [{'step': 0, **best}]
        best_step, best_path, last_path = 0, None, None
    initial = {k: v for k, v in history[0].items() if k != 'step'}

    def summary(status, reason):
        return {'status': status, 'final_step': step, 'best_step': best_step,
                'initial_validation': initial, 'best_validation': best,
                'best_checkpoint': best_path, 'last_checkpoint': last_path,
                'stop_reason': reason, 'history': history}

    def checkpoint(status, reason, improved=False):
        nonlocal best_path, last_path
        payload = {'step': step, 'model_state': _model_state(model),
                   'optimizer': optimizer.state_dict(), 'rng': _capture_rng(),
                   'history': history, 'cursor': dict(cursor), 'stale': stale,
                   'best_step': best_step, 'best_validation': best,
                   'status': status, 'stop_reason': reason,
                   'best_checkpoint': None if improved else str(Path(best_path).relative_to(output_dir))}
        last_path = save_checkpoint(output_dir, payload, contract)
        if improved:
            best_path = last_path

    if state is not None and state['status'] == 'completed':
        return summary('completed', state['stop_reason'])
    with _termination_flag() as termination:
        if state is None:
            checkpoint('running', '', improved=True)
        reason = 'epochs'
        for epoch in range(cursor['epoch'], settings['epochs']):
            generator = torch.Generator().manual_seed(settings['seed'] + epoch)
            order = torch.randperm(len(train_dataset), generator=generator).tolist()
            starts = list(range(0, len(order), settings['batch_size']))
            first = cursor['batch'] if epoch == cursor['epoch'] else 0
            for batch_index in range(first, len(starts)):
                start = starts[batch_index]
                model.train()
                batch = _batch(train_dataset, order[start:start + settings['batch_size']], settings['device'])
                optimizer.zero_grad(set_to_none=True)
                logits = _forward(model, batch)
                loss, count = _ce(logits, batch['labels'])
                if not count:
                    raise ValueError('training batch has no supervised tokens')
                loss = loss / count
                if settings['alpha'] < 1:
                    with torch.no_grad():
                        teacher_logits = _forward(teacher, batch)
                    kd = _distillation_loss(logits, teacher_logits, batch['labels'], settings)
                    loss = settings['alpha'] * loss + (1 - settings['alpha']) * kd
                if not torch.isfinite(loss):
                    raise FloatingPointError('training loss is not finite')
                loss.backward()
                optimizer.step()
                step += 1
                cursor = {'epoch': epoch, 'batch': batch_index + 1}
                if cursor['batch'] == len(starts):
                    cursor = {'epoch': epoch + 1, 'batch': 0}
                final = step >= settings['max_steps'] or cursor['epoch'] >= settings['epochs']
                improved = False
                validated = step % settings['validation_interval'] == 0 or final
                if validated:
                    value = _validate(model, validation_dataset, settings)
                    history.append({'step': step, **value})
                    improved = value['ce'] < best['ce'] - settings['min_delta']
                    if improved:
                        best, best_step, stale = value, step, 0
                    else:
                        stale += 1
                if validated and settings['patience'] > 0 and stale >= settings['patience']:
                    final, reason = True, 'early_stopping'
                elif final:
                    reason = 'max_steps' if step >= settings['max_steps'] else 'epochs'
                interrupted = termination['stop'] and not final
                if interrupted or final or improved or validated or step % settings['checkpoint_interval_steps'] == 0:
                    checkpoint('completed' if final else ('interrupted' if interrupted else 'running'),
                               reason if final else ('sigterm' if interrupted else ''), improved)
                if interrupted:
                    return summary('interrupted', 'sigterm')
                if final:
                    return summary('completed', reason)
        raise RuntimeError('training budget contains no batches')
