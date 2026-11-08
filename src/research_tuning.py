"""Frozen, balanced validation-only pilot tuning on an existing research run.

Never tokenizes or evaluates test data. The original smoke test was already viewed;
this pilot does not create a new blind test or establish effectiveness.
"""

import copy
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

from .research_data import atomic_json, atomic_text, digest, file_hash

INTERPRETATION = ('Validation-only smoke/pilot; no effectiveness conclusion. '
                  'Original test already viewed; not a blind test. '
                  'Equal per-method search budget, not a causal hyperparameter ablation.')


def default_trials():
    ce = [dict(id=f'ce-{i + 1}', method='ce', learning_rate=lr, alpha=1.,
               temperature=1., divergence='forward_kl')
          for i, lr in enumerate((2e-5, 5e-5, 1e-4, 2e-4))]
    kd = [dict(id=f'kd-{i + 1}', method='kd', learning_rate=lr, alpha=alpha,
               temperature=temperature, divergence=divergence)
          for i, (lr, alpha, temperature, divergence) in enumerate((
              (1e-4, .7, 1., 'forward_kl'), (5e-5, .7, 2., 'forward_kl'),
              (5e-5, .7, 1., 'reverse_kl'), (2e-5, .9, 1., 'reverse_kl')))]
    return ce + kd


def sweep_settings(training, max_steps, vocab_size, *, seed=42, settings_overrides=None):
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    overrides = {} if settings_overrides is None else settings_overrides
    allowed = {'validation_interval', 'patience', 'checkpoint_interval_steps'}
    if not isinstance(overrides, dict) or overrides.keys() - allowed:
        raise ValueError('invalid shared settings overrides')
    for key, value in overrides.items():
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
        if key != 'patience' and value > max_steps:
            raise ValueError(f'{key} must not exceed max_steps')
    return dict(seed=seed, device=training['device'], batch_size=training['batch_size'],
                max_steps=max_steps, epochs=100, min_delta=1e-4, vocab_size=vocab_size,
                **{'validation_interval': min(20, max_steps), 'patience': 3,
                   'checkpoint_interval_steps': min(20, max_steps), **overrides})


def initial_adapter_hash(model):
    """Hash actual trainable tensors, preserving names, dtype, shape and bytes."""
    import torch
    h = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_base(base_run, dependencies):
    """Read-only equivalent of prepare(): only train/validation are packed."""
    from .research_data import DATASET_REVISION, prepare_raw, pack_split
    from .research_models import verify_tokenizers, validate_revision
    from scripts.run_research import load_tokenizer, read_completed, stage_contract
    config = json.loads((base_run / 'config.json').read_text())
    for role in ('teacher', 'student'):
        validate_revision(config[role]['revision'])
    if config['training']['seed'] != 42 or config['adapter'] != dict(rank=8, alpha=16, dropout=.05):
        raise ValueError('fixed seed42/rank8/alpha16/dropout.05 required')
    # Require frozen files: never stream, repair, or write into the base run.
    for name in ('prepared.json', 'raw/manifest.json', 'raw/train.jsonl',
                 'raw/validation.jsonl', 'raw/test.jsonl'):
        if not (base_run / name).is_file():
            raise ValueError('missing frozen base artifact: ' + name)
    raw_identity = dict(dataset='HuggingFaceFW/fineweb-2', subset='kor_Hang',
                        revision=DATASET_REVISION, max_documents=config['data']['max_documents'], seed=42)
    if 'exposure_exclusion' in config['data']:
        policy = config['data']['exposure_exclusion']
        raw_identity.update(exposure_exclusion=policy, exposure_exclusion_sha256=digest(policy))
    raw = prepare_raw((), base_run / 'raw', raw_identity)
    factory = dependencies.get('tokenizer', load_tokenizer)
    token_identity = verify_tokenizers(factory(config['teacher']), tokenizer := factory(config['student']))
    datasets, packing = {}, {}
    for split in ('train', 'validation'):
        datasets[split], packing[split] = pack_split(base_run / f'raw/{split}.jsonl', tokenizer,
                                                    config['data']['seq_length'])
    prepared = dict(raw=raw, tokenizer=token_identity, packing=packing,
                    seq_length=config['data']['seq_length'], teacher=config['teacher'], student=config['student'])
    prepared['fingerprint'] = digest(prepared)
    if json.loads((base_run / 'prepared.json').read_text()) != prepared:
        raise ValueError('frozen prepared identity mismatch')
    teacher = read_completed(base_run, 'teacher', contract=stage_contract('teacher', config, prepared))
    return config, prepared, datasets, teacher


def _code_identity():
    root = Path(__file__).resolve().parents[1]
    return {name: file_hash(root / name) for name in (
        'src/research_tuning.py', 'src/numerical_policy.py', 'src/research_runner.py', 'src/research_models.py',
        'src/research_data.py', 'src/resume_checkpoint.py', 'src/stage_records.py', 'scripts/run_research.py',
        'scripts/tune_research.py', 'scripts/tuning_service.py')}


def _validate_trials(trials):
    if not isinstance(trials, list) or not 2 <= len(trials) <= 8:
        raise ValueError('trials must be a balanced list of 2 to 8 trials (at most 4 per method)')
    required = {'id', 'method', 'learning_rate', 'alpha', 'temperature', 'divergence'}
    ids = set()
    counts = dict(ce=0, kd=0)
    for trial in trials:
        if not isinstance(trial, dict) or trial.keys() != required:
            raise ValueError('invalid trial fields')
        if not isinstance(trial['id'], str) or not re.fullmatch(r'[a-z][a-z0-9-]*', trial['id']) or trial['id'] in ids:
            raise ValueError('invalid/duplicate trial ID')
        ids.add(trial['id'])
        if not isinstance(trial['method'], str) or trial['method'] not in counts:
            raise ValueError('trial method must be ce or kd')
        counts[trial['method']] += 1
        for key in ('learning_rate', 'alpha', 'temperature'):
            try:
                finite = not isinstance(trial[key], bool) and isinstance(trial[key], (int, float)) and math.isfinite(trial[key])
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError('trial parameters must be finite numeric')
        if trial['learning_rate'] <= 0 or trial['temperature'] <= 0 or not 0 <= trial['alpha'] <= 1:
            raise ValueError('invalid trial learning_rate/temperature/alpha')
        if trial['divergence'] not in ('forward_kl', 'reverse_kl'):
            raise ValueError('invalid trial divergence')
        if (trial['method'] == 'ce' and trial['alpha'] != 1) or (trial['method'] == 'kd' and trial['alpha'] == 1):
            raise ValueError('trial method/alpha mismatch')
    if counts['ce'] != counts['kd'] or not counts['ce']:
        raise ValueError('trials must have equal nonzero CE/KD counts')


def _freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'frozen {path.name} mismatch')
    else:
        atomic_json(path, value)


def _trial_contract(plan, trial):
    settings = {**plan['common_settings'], **{k: trial[k] for k in ('alpha', 'temperature', 'divergence')}}
    identity = dict(stage=trial['method'], trial_id=trial['id'], trial=trial,
                    plan_hash=digest(plan), model=plan['student'], adapter=plan['adapter'],
                    model_loading=plan['model_loading'], provenance=plan['provenance'],
                    initial_adapter_sha256=plan['initial_adapter_sha256'],
                    numerical_policy=plan['numerical_policy'],
                    initialization='direct_pretrained_with_paired_seeded_lora',
                    optimizer='AdamW', learning_rate=trial['learning_rate'], weight_decay=.01)
    if trial['method'] == 'kd':
        identity['kd_support'] = 'renormalized_verified_tokenizer_ids; excludes_unassigned_head_padding; CE_uses_full_head'
    return dict(settings=settings, identity=identity)


def _write_if_changed(path, text):
    if not path.exists() or path.read_bytes() != text.encode('utf-8'):
        atomic_text(path, text)


def _checkpoint_result(trial_dir, expected_contract):
    """Rebuild trustworthy metrics from verified durable latest + selected best."""
    from .resume_checkpoint import load_checkpoint
    state = load_checkpoint(trial_dir, expected_contract=expected_contract)
    best_path = (trial_dir / state['best_checkpoint']).resolve()
    if best_path.parent != (trial_dir / 'checkpoints').resolve():
        raise ValueError('best checkpoint escapes trial directory')
    best = load_checkpoint(best_path, expected_contract=expected_contract)
    if best['step'] != state['best_step'] or best['best_validation'] != state['best_validation']:
        raise ValueError('selected best checkpoint mismatch')
    pointer = json.loads((trial_dir / 'latest.json').read_text())
    return dict(status=state['status'], final_step=state['step'], best_step=state['best_step'],
        initial_validation={k: v for k, v in state['history'][0].items() if k != 'step'},
        best_validation=state['best_validation'], best_checkpoint=str(best_path),
        last_checkpoint=str((trial_dir / pointer['checkpoint']).resolve()),
        stop_reason=state['stop_reason'], history=state['history'])


def _report(output, plan, records, status):
    report = dict(status=status, plan_hash=digest(plan), trials=records,
                  test_evaluated=False, interpretation=INTERPRETATION)
    _write_if_changed(output / 'report.json', json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + '\n')
    stream = io.StringIO()
    fields = ['trial_id', 'method', 'learning_rate', 'alpha', 'temperature', 'divergence',
              *plan['common_settings'], 'weight_decay', 'adapter_rank', 'adapter_alpha', 'adapter_dropout',
              'status', 'final_step', 'best_step', 'best_validation_ce', 'best_validation_ppl', 'stop_reason']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    by_id = {r['trial']['id']: r for r in records}
    for trial in plan['trials']:
        row = dict(trial_id=trial['id'], **{k: v for k, v in trial.items() if k != 'id'},
                   **plan['common_settings'], weight_decay=.01, adapter_rank=plan['adapter']['rank'],
                   adapter_alpha=plan['adapter']['alpha'], adapter_dropout=plan['adapter']['dropout'], status='pending')
        if trial['id'] in by_id:
            result = by_id[trial['id']]['result']
            row.update({k: result[k] for k in ('status', 'final_step', 'best_step', 'stop_reason')})
            row.update(best_validation_ce=result['best_validation']['ce'],
                       best_validation_ppl=result['best_validation']['ppl'])
        writer.writerow(row)
    _write_if_changed(output / 'trials.csv', stream.getvalue())
    lines = ['# Validation-only fixed-data / fixed-Teacher sweep', '', INTERPRETATION,
             '', f'Status: {status}. Completed {sum(r["result"]["status"] == "completed" for r in records)} / {len(plan["trials"])} trials.',
             '', 'Selection: minimum validation CE per method; ties use trial ID. No first-crossing stop.',
             '', '| Trial | Method | LR | alpha (CE weight) | T | KL | Best val PPL | Best / final step | Status |',
             '|---|---|---|---|---|---|---|---|---|']
    for record in records:
        t, r = record['trial'], record['result']
        lines.append(f'| {t["id"]} | {t["method"]} | {t["learning_rate"]} | {t["alpha"]} | {t["temperature"]} | {t["divergence"]} | {r["best_validation"]["ppl"]:.8g} | {r["best_step"]} / {r["final_step"]} | {r["status"]} |')
    _write_if_changed(output / 'summary.md', '\n'.join(lines) + '\n')
    return report


def run_sweep(base_run, output_dir, *, resume=False, max_steps=200, dependencies=None,
              trials=None, seed=42, settings_overrides=None):
    """Run a bounded balanced validation-only round (default: eight trials).

    Selected-pair Student seed replication is conditional on the SAME frozen
    Teacher checkpoint and raw seed42 split; it does not replicate Teacher fitting.
    alpha is the CE weight: alpha*CE + (1-alpha)*KL, as in train_stage.
    """
    trials = copy.deepcopy(default_trials() if trials is None else trials)
    _validate_trials(trials)
    sweep_settings({'device': 'cpu', 'batch_size': 1}, max_steps, 0,
                   seed=seed, settings_overrides=settings_overrides)
    from .numerical_policy import configure_numerics
    numerical_policy = configure_numerics()
    import torch
    from scripts.run_research import make_model, run_lock, release_memory
    from .research_models import restore_best
    from .research_runner import train_stage, _optimizer_contract, _data_hash
    dependencies = dependencies or {}
    base_run, output = Path(base_run).resolve(), Path(output_dir).resolve()
    if base_run == output or base_run in output.parents or output in base_run.parents:
        raise ValueError('output must be separate from base run')
    if output.exists() and not resume:
        raise ValueError('output exists; explicit resume required')
    if resume and not (output / 'plan.json').is_file():
        raise ValueError('resume requires an existing plan')
    config, prepared, datasets, teacher_record = load_base(base_run, dependencies)
    base_config = config
    config = copy.deepcopy(base_config)
    config['training']['seed'] = seed
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        model = make_model('student', config, dependencies)
        initial_hash = initial_adapter_hash(model)
        kd_vocab = prepared['tokenizer']['vocab_size']
        head = model.get_output_embeddings() if hasattr(model, 'get_output_embeddings') else None
        rows = getattr(head, 'weight', None)
        if rows is not None and rows.shape[0] < kd_vocab:
            # Gemma 3 1B scores 262144 IDs while its tokenizer lists one more (an image
            # placeholder). KD may only cover IDs the Student can predict, and only if
            # the data never uses the rest.
            largest = max(int(row['input_ids'].max()) for split in ('train', 'validation')
                          for row in datasets[split])
            if largest >= rows.shape[0]:
                raise ValueError('data uses token IDs the student cannot predict')
            kd_vocab = int(rows.shape[0])
        teacher_path = Path(teacher_record['result']['best_checkpoint'])
        plan = dict(version=1, trials=trials, numerical_policy=numerical_policy,
                    common_settings=sweep_settings(config['training'], max_steps, kd_vocab,
                                                   seed=seed, settings_overrides=settings_overrides),
                    student=config['student'], adapter=config['adapter'], model_loading=config['model'],
                    initial_adapter_sha256=initial_hash, code=_code_identity(),
                    provenance=dict(base_config_hash=digest(base_config), prepared_fingerprint=prepared['fingerprint'],
                        raw_manifest_hash=file_hash(base_run / 'raw/manifest.json'),
                        teacher=dict(model=config['teacher'], contract_hash=digest(teacher_record['checkpoint_contract']),
                                     manifest_sha256=file_hash(teacher_path / 'manifest.json'),
                                     best_step=teacher_record['result']['best_step'])),
                    selection_policy='minimum best_validation.ce per method; ties lexicographic trial ID',
                    interpretation=INTERPRETATION, test_evaluated=False)
        _freeze(output / 'plan.json', plan)
        data_contract = {split: _data_hash(datasets[split]) for split in ('train', 'validation')}
        durable = {}
        expected_contracts = {}
        # Preflight ALL extant checkpoints before any training/report writes.
        for trial in plan['trials']:
            contract = _trial_contract(plan, trial)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=trial['learning_rate'], weight_decay=.01)
            expected = json.loads(json.dumps(dict(**contract, optimizer=_optimizer_contract(model, optimizer),
                                                  data=data_contract), sort_keys=True, allow_nan=False))
            expected_contracts[trial['id']] = expected
            trial_dir = output / 'trials' / trial['id']
            if trial_dir.exists():
                result = _checkpoint_result(trial_dir, expected)
                durable[trial['id']] = result
                record_path = trial_dir / 'result.json'
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    if any(record.get(k) != v for k, v in dict(trial=trial, contract=contract,
                        checkpoint_contract=expected, plan_hash=digest(plan)).items()):
                        raise ValueError('trial result contract/plan mismatch')
                    if record['result']['status'] == 'completed' and record['result'] != result:
                        raise ValueError('completed result/checkpoint mismatch')
        if (output / 'selection.json').exists() and any(
            trial['id'] not in durable or durable[trial['id']]['status'] != 'completed'
            for trial in plan['trials']
        ):
            raise ValueError('selection is frozen; cannot restart missing/incomplete trials')
        records = []
        for trial in plan['trials']:
            contract = _trial_contract(plan, trial)
            trial_dir = output / 'trials' / trial['id']
            result = durable.get(trial['id'])
            if result is None or result['status'] != 'completed':
                if model is None:
                    model = make_model('student', config, dependencies)
                if initial_adapter_hash(model) != initial_hash:
                    raise ValueError('student initial adapter hash mismatch')
                teacher = None
                if trial['method'] == 'kd':
                    teacher = make_model('teacher', base_config, dependencies)
                    restore_best(teacher, teacher_path, contract=teacher_record['checkpoint_contract'], freeze=True)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                               lr=trial['learning_rate'], weight_decay=.01)
                returned = dependencies.get('runner', train_stage)(model, datasets['train'], datasets['validation'],
                    optimizer, trial_dir, teacher=teacher, settings=contract['settings'], identity=contract['identity'],
                    resume=resume and trial_dir.exists())
                result = _checkpoint_result(trial_dir, expected_contracts[trial['id']])
                if result != returned:
                    raise ValueError('runner result/checkpoint mismatch')
                del teacher, optimizer
                model = None
                release_memory()
            record = dict(trial=trial, contract=contract, checkpoint_contract=expected_contracts[trial['id']],
                          plan_hash=digest(plan), result=result)
            if result['status'] == 'completed':
                record_path = trial_dir / 'result.json'
                if record_path.exists() and json.loads(record_path.read_text())['result']['status'] != 'completed':
                    atomic_json(record_path, record)
                else:
                    _freeze(record_path, record)
            else:
                atomic_json(trial_dir / 'result.json', record)
            records.append(record)
            if result['status'] != 'completed':
                return _report(output, plan, records, result['status'])
        methods = {}
        for method in ('ce', 'kd'):
            best = min((r for r in records if r['trial']['method'] == method),
                       key=lambda r: (r['result']['best_validation']['ce'], r['trial']['id']))
            methods[method] = dict(trial_id=best['trial']['id'], best_validation=best['result']['best_validation'],
                best_step=best['result']['best_step'], best_checkpoint=best['result']['best_checkpoint'],
                manifest_sha256=file_hash(Path(best['result']['best_checkpoint']) / 'manifest.json'))
        _freeze(output / 'selection.json', dict(plan_hash=digest(plan), methods=methods,
                selection_policy=plan['selection_policy'], test_evaluated=False))
        return _report(output, plan, records, 'completed')
