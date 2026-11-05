#!/usr/bin/env python3
"""Controlled Teacher CPT -> paired pretrained Student CE/KD -> frozen test.

The default 20-step run is a GPU feasibility smoke, not efficacy evidence.
Run each stage separately to release GPU memory between processes.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research_data import atomic_json, prepare_raw, stream_source, pack_split, digest, DATASET_REVISION, validate_exposure_identity
from src.research_models import validate_revision, verify_tokenizers


def load_tokenizer(spec):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False, use_fast=True)


def prepare(config, directory, dependencies):
    raw_identity = dict(dataset='HuggingFaceFW/fineweb-2', subset='kor_Hang', revision=DATASET_REVISION,
                        max_documents=config['data']['max_documents'], seed=42)
    if 'exposure_exclusion' in config['data']:
        policy = config['data']['exposure_exclusion']
        raw_identity.update(exposure_exclusion=policy, exposure_exclusion_sha256=digest(policy))
    validate_exposure_identity(raw_identity)
    raw_dir = directory / 'raw'
    source = () if (raw_dir / 'manifest.json').exists() else dependencies.get('source', stream_source)(raw_identity['max_documents'])
    raw = prepare_raw(source, raw_dir, raw_identity)
    tokenizer_factory = dependencies.get('tokenizer', load_tokenizer)
    teacher_tokenizer, tokenizer = (tokenizer_factory(config[role]) for role in ('teacher', 'student'))
    token_identity = verify_tokenizers(teacher_tokenizer, tokenizer)
    datasets, packing = {}, {}
    for split in ('train', 'validation'):
        datasets[split], packing[split] = pack_split(raw_dir / f'{split}.jsonl', tokenizer, config['data']['seq_length'])
    prepared = dict(raw=raw, tokenizer=token_identity, packing=packing,
                    seq_length=config['data']['seq_length'], teacher=config['teacher'], student=config['student'])
    prepared['fingerprint'] = digest(prepared)
    path = directory / 'prepared.json'
    if path.exists() and json.loads(path.read_text()) != prepared:
        raise ValueError('prepared tokenizer/data identity mismatch')
    atomic_json(path, prepared)
    return prepared, tokenizer, datasets


def main(argv=None, *, dependencies=None):
    args = parse_args(argv)
    dependencies = dependencies or {}
    config = load_config(args)
    directory = args.run_dir.resolve()
    with run_lock(directory):
        freeze_config(directory, config)
        prepared, tokenizer, datasets = prepare(config, directory, dependencies)
        if args.stage == 'prepare':
            return prepared
        if args.stage in ('teacher', 'ce', 'kd'):
            return train_one(args.stage, config, directory, prepared, datasets, dependencies, args.resume)
        if args.stage == 'all':
            for stage in ('teacher', 'ce', 'kd'):
                record = train_one(stage, config, directory, prepared, datasets, dependencies, args.resume)
                release_memory()
                if record['result']['status'] != 'completed':
                    return record
        return evaluate(config, directory, prepared, tokenizer, dependencies)


def run_feasibility(config, canonical, directory, *, train_documents, validation_documents, dependencies=None):
    """Fresh diagnostic-only Teacher/CE/KD; never prepare or evaluate test."""
    from src.research_data import prepare_feasibility_subset
    dependencies = dependencies or {}
    directory = Path(directory)
    factory = dependencies.get('tokenizer', load_tokenizer)
    teacher_tokenizer, tokenizer = (factory(config[role]) for role in ('teacher', 'student'))
    token_identity = verify_tokenizers(teacher_tokenizer, tokenizer)
    subset = prepare_feasibility_subset(canonical, directory, tokenizer, config['data']['seq_length'],
                                        train_documents=train_documents, validation_documents=validation_documents)
    prepared = dict(raw=subset['selection'], tokenizer=token_identity, packing=subset['packing'],
                    seq_length=config['data']['seq_length'], teacher=config['teacher'], student=config['student'])
    prepared['fingerprint'] = digest(prepared)
    with run_lock(directory):
        freeze_config(directory, config)
        atomic_json(directory / 'prepared.json', prepared)
        records = {}
        for stage in ('teacher', 'ce', 'kd'):
            records[stage] = train_one(stage, config, directory, prepared, subset['datasets'], dependencies, False)
            release_memory()
            if records[stage]['result']['status'] != 'completed':
                break
        return records


def reload_feasibility_subset(config, directory, expected_hashes, *, dependencies=None):
    """Reload only authenticated frozen train/validation; never rebuild a split."""
    from src.research_data import file_hash
    directory = Path(directory)
    names = {'config.json', 'prepared.json', 'feasibility-selection.json',
             'raw/train.jsonl', 'raw/validation.jsonl'}
    if set(expected_hashes) != names:
        raise ValueError('recovery requires exact frozen subset hash roster')
    for name in sorted(names):
        if file_hash(directory / name) != expected_hashes[name]:
            raise ValueError('frozen subset hash mismatch: ' + name)
    if json.loads((directory / 'config.json').read_text()) != config:
        raise ValueError('frozen config mismatch')
    prepared = json.loads((directory / 'prepared.json').read_text())
    selection = json.loads((directory / 'feasibility-selection.json').read_text())
    if prepared['raw'] != selection or prepared['fingerprint'] != digest({k: v for k, v in prepared.items() if k != 'fingerprint'}):
        raise ValueError('frozen prepared identity mismatch')
    factory = (dependencies or {}).get('tokenizer', load_tokenizer)
    tokenizers = [factory(config[role]) for role in ('teacher', 'student')]
    if verify_tokenizers(*tokenizers) != prepared['tokenizer']:
        raise ValueError('frozen tokenizer mismatch')
    if any(prepared[role] != config[role] for role in ('teacher', 'student')) or prepared['seq_length'] != config['data']['seq_length']:
        raise ValueError('frozen model/sequence mismatch')
    datasets = {}
    for split in ('train', 'validation'):
        datasets[split], packing = pack_split(directory / 'raw' / f'{split}.jsonl', tokenizers[1], prepared['seq_length'])
        if packing != prepared['packing'][split]:
            raise ValueError('frozen packing mismatch: ' + split)
    return prepared, datasets


def resume_feasibility(config, directory, expected_hashes, *, dependencies=None):
    """Resume the existing diagnostic contract without source reconstruction."""
    dependencies = dependencies or {}
    directory = Path(directory)
    with run_lock(directory):
        prepared, datasets = reload_feasibility_subset(config, directory, expected_hashes,
                                                       dependencies=dependencies)
        records = {}
        for stage in ('teacher', 'ce', 'kd'):
            records[stage] = train_one(stage, config, directory, prepared, datasets, dependencies, True)
            release_memory()
            if records[stage]['result']['status'] != 'completed':
                break
        return records


def release_memory():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_metrics(model, dataset, settings):
    import math
    import torch
    import torch.nn.functional as F
    model.eval()
    model.requires_grad_(False)
    total, tokens = 0., 0
    with torch.no_grad():
        for start in range(0, len(dataset), settings['batch_size']):
            rows = dataset[start:start + settings['batch_size']]
            batch = {key: torch.stack([row[key] for row in rows]).to(settings['device']) for key in rows[0]}
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits[:, :-1].float()
            labels = batch['labels'][:, 1:]
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction='sum').item()
            tokens += int((labels != -100).sum())
    if tokens == 0:
        raise ValueError('test contains no supervised tokens')
    ce = total / tokens
    return dict(ce=ce, ppl=math.exp(ce), tokens=tokens)


def evaluate(config, directory, prepared, tokenizer, dependencies):
    from src.research_data import file_hash
    from src.research_models import restore_best
    teacher = read_completed(directory, 'teacher', contract=stage_contract('teacher', config, prepared))
    records = {'teacher': teacher, 'ce': read_completed(directory, 'ce', contract=stage_contract('ce', config, prepared)),
               'kd': read_completed(directory, 'kd', contract=stage_contract('kd', config, prepared, teacher))}
    selection = {'config_hash': digest(config), 'data_fingerprint': prepared['fingerprint'],
                 'stages': {stage: {'best_checkpoint': r['result']['best_checkpoint'],
                            'best_step': r['result']['best_step'], 'contract_hash': digest(r['checkpoint_contract']),
                            'manifest_sha256': file_hash(Path(r['result']['best_checkpoint']) / 'manifest.json')}
                            for stage, r in records.items()}}
    path = directory / 'selection.json'
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise ValueError('frozen selection mismatch')
    else:
        atomic_json(path, selection)
    output = directory / 'test-results.json'
    if output.exists():
        result = json.loads(output.read_text())
        if result['selection_hash'] != digest(selection):
            raise ValueError('test results selection mismatch')
        return result
    # No heldout tokenization or model evaluation occurs until every selection freezes.
    test, packing = pack_split(directory / 'raw/test.jsonl', tokenizer, config['data']['seq_length'])
    metrics = {}
    for label, stage in [('student_pretrained', None), ('student_ce', 'ce'), ('student_kd', 'kd'), ('teacher_selected', 'teacher')]:
        model = make_model('teacher' if stage == 'teacher' else 'student', config, dependencies)
        if stage is not None:
            restore_best(model, records[stage]['result']['best_checkpoint'], contract=records[stage]['checkpoint_contract'], freeze=True)
        metrics[label] = evaluate_metrics(model, test, config['training'])
        del model
        release_memory()
    result = {'selection_hash': digest(selection), 'metrics': metrics, 'packing': packing,
              'split_policy': prepared['raw']['split_policy'], 'interpretation': 'GPU smoke only; no effectiveness conclusion'}
    atomic_json(output, result)
    return result


def stage_contract(stage, config, prepared, teacher_record=None):
    role = 'teacher' if stage == 'teacher' else 'student'
    training = config['training']
    settings = {key: training[key] for key in ('seed', 'device', 'batch_size', 'max_steps', 'epochs',
                'validation_interval', 'patience', 'min_delta', 'checkpoint_interval_steps', 'temperature', 'divergence')}
    settings['alpha'] = training['alpha'] if stage == 'kd' else 1.0
    settings['vocab_size'] = prepared['tokenizer']['vocab_size']
    identity = {'stage': stage, 'model': config[role], 'tokenizer_hash': prepared['tokenizer']['fingerprint'],
                'data_fingerprint': prepared['fingerprint'], 'adapter': config['adapter'],
                'initialization': 'direct_pretrained_with_paired_seeded_lora', 'optimizer': 'AdamW',
                'learning_rate': training[role + '_learning_rate'], 'weight_decay': .01,
                'model_loading': config['model']}
    if teacher_record is not None:
        identity['kd_support'] = 'renormalized_verified_tokenizer_ids; excludes_unassigned_head_padding; CE_uses_full_head'
        identity['teacher_selection'] = {'checkpoint': teacher_record['result']['best_checkpoint'],
                                         'contract_hash': digest(teacher_record['contract'])}
    return dict(settings=settings, identity=identity)


def read_completed(directory, stage, *, contract=None):
    from src.stage_records import reconcile_completed_stage
    return reconcile_completed_stage(directory, stage, contract=contract)


def make_model(role, config, dependencies):
    from src.research_models import load_model, attach_lora
    model = dependencies.get('model', load_model)(config[role], device=config['training']['device'], **config['model'])
    return dependencies.get('adapter', attach_lora)(model, config['adapter'], seed=config['training']['seed'])


def train_one(stage, config, directory, prepared, datasets, dependencies, resume):
    import torch
    from src.research_runner import train_stage
    selected_teacher = read_completed(directory, 'teacher', contract=stage_contract('teacher', config, prepared)) if stage in ('ce', 'kd') else None
    teacher_record = selected_teacher if stage == 'kd' else None
    contract = stage_contract(stage, config, prepared, teacher_record)
    record_path = directory / f'{stage}.json'
    if record_path.exists():
        if not resume:
            raise ValueError('completed stage exists; use --resume')
        return read_completed(directory, stage, contract=contract)
    if (directory / 'selection.json').exists():
        raise ValueError('selection is frozen; training cannot be repeated')
    role = 'teacher' if stage == 'teacher' else 'student'
    model = make_model(role, config, dependencies)
    teacher = None
    if teacher_record is not None:
        from src.research_models import restore_best
        teacher = make_model('teacher', config, dependencies)
        restore_best(teacher, teacher_record['result']['best_checkpoint'],
                     contract=teacher_record['checkpoint_contract'], freeze=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=contract['identity']['learning_rate'], weight_decay=.01)
    output_dir = directory / stage
    result = dependencies.get('runner', train_stage)(model, datasets['train'], datasets['validation'], optimizer,
                  output_dir, teacher=teacher, settings=contract['settings'], identity=contract['identity'],
                  resume=resume and (output_dir / 'latest.json').exists())
    record = dict(contract=contract, result=result)
    if stage == 'teacher':
        record['teacher_selection'] = ('pretrained_step0_retained_no_validation_improvement' if result['best_step'] == 0
                                       else 'validation_selected_additionally_trained_teacher')
    if result['status'] == 'completed':
        from src.resume_checkpoint import load_checkpoint
        load_checkpoint(result['best_checkpoint'])
        record['checkpoint_contract'] = json.loads((Path(result['best_checkpoint']) / 'manifest.json').read_text())['contract']
        if any(record['checkpoint_contract'].get(key) != value for key, value in contract.items()):
            raise ValueError('runner checkpoint contract mismatch')
        atomic_json(record_path, record)
    return record



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'teacher', 'ce', 'kd', 'evaluate', 'all'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--max-documents', type=int)
    parser.add_argument('--seq-length', type=int)
    parser.add_argument('--exposure-exclusion-file', type=Path)
    return parser.parse_args(argv)


def load_config(args):
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    if args.exposure_exclusion_file is not None:
        def unique_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate exposure exclusion JSON field: ' + key)
                result[key] = value
            return result
        config['data']['exposure_exclusion'] = json.loads(
            args.exposure_exclusion_file.read_text(), object_pairs_hook=unique_fields)
    for role in ('teacher', 'student'):
        validate_revision(config[role]['revision'])
    for argument, section, key in [('max_steps', 'training', 'max_steps'), ('max_documents', 'data', 'max_documents'), ('seq_length', 'data', 'seq_length')]:
        value = getattr(args, argument)
        if value is not None:
            config[section][key] = value
    training = config['training']
    for key in ('batch_size', 'max_steps', 'epochs', 'validation_interval', 'patience', 'checkpoint_interval_steps'):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if config['data']['max_documents'] < 10 or config['data']['seq_length'] < 2:
        raise ValueError('at least 10 documents and sequence length >= 2 required')
    import math
    for key in ('temperature', 'alpha', 'min_delta', 'teacher_learning_rate', 'student_learning_rate'):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + ' must be finite numeric')
    if any(training[role + '_learning_rate'] <= 0 for role in ('teacher', 'student')):
        raise ValueError('learning rates must be positive')
    if training['divergence'] not in ('forward_kl', 'reverse_kl'):
        raise ValueError('unsupported divergence')
    adapter = config['adapter']
    if type(adapter['rank']) is not int or adapter['rank'] < 1 or not math.isfinite(adapter['alpha']) or adapter['alpha'] <= 0:
        raise ValueError('invalid LoRA rank/alpha')
    if not math.isfinite(adapter['dropout']) or not 0 <= adapter['dropout'] < 1:
        raise ValueError('invalid LoRA dropout')
    if config['model']['attention'] not in ('sdpa', 'eager') or config['model']['dtype'] not in ('bfloat16', 'float32'):
        raise ValueError('smoke requires sdpa/eager and bfloat16/float32')
    if training['min_delta'] < 0 or not 0 <= training['alpha'] <= 1 or training['temperature'] <= 0:
        raise ValueError('invalid selection/KD settings')
    return config


def freeze_config(directory, config):
    directory = Path(directory)
    path = directory / 'config.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('frozen run config mismatch; choose a new run-dir, especially after test evaluation')
    else:
        atomic_json(path, config)


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / '.run.lock', 'a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('run directory is locked by another process') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True, ensure_ascii=False, indent=2))
