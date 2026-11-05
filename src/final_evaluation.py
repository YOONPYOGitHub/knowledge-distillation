"""Heldout-free preflight for an explicit frozen checkpoint list."""
import hashlib
import errno
import os
import stat
from copy import deepcopy
from pathlib import Path


def freeze_evaluation_protocol(protocol, selections, cohort, *, numerical_policy,
                               exposure_adjudication_bytes,
                               expected_exposure_sha256):
    """Bind existing metadata gates; does not authorize or execute test scoring."""
    import json
    if (type(exposure_adjudication_bytes) is not bytes
            or not 0 < len(exposure_adjudication_bytes) <= 65536
            or type(expected_exposure_sha256) is not str
            or len(expected_exposure_sha256) != 64
            or hashlib.sha256(exposure_adjudication_bytes).hexdigest()
            != expected_exposure_sha256):
        raise ValueError('exposure adjudication binding mismatch')
    numerical = json.loads(freeze_evaluation_numerical_policy(numerical_policy))
    frozen = json.loads(freeze_cohort_protocol(protocol, selections, cohort))
    return json.dumps({**frozen, 'numerical_policy': numerical,
                       'exposure_adjudication_sha256': expected_exposure_sha256},
                      sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def publish_completed_evaluation_output(path, *, protocol_bytes, model_id, statistics):
    """Publish validated bytes atomically without replacing any existing target.

    Caller must authenticate the frozen protocol before authorizing scoring.
    """
    import json
    import secrets
    payload = json.dumps({
        'version': 1, 'status': 'completed', 'model_id': model_id,
        'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
        'statistics': statistics,
    }, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    validate_completed_evaluation_output(
        payload, protocol_bytes=protocol_bytes, expected_model_id=model_id)
    if (type(path) is not str or not path or '\x00' in path
            or not Path(path).is_absolute() or '..' in Path(path).parts):
        raise ValueError('completed evaluation output path invalid')
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    temporary = '.kd-result-' + secrets.token_hex(16)
    created = False
    try:
        for component in Path(path).parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600, dir_fd=parent)
        created = True
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, Path(path).name, src_dir_fd=parent,
                    dst_dir_fd=parent, follow_symlinks=False)
            os.fsync(parent)
        finally:
            os.unlink(temporary, dir_fd=parent)
            created = False
    finally:
        if created:
            os.unlink(temporary, dir_fd=parent)
        os.close(parent)
    return payload


def freeze_evaluation_operation(*, protocol_bytes, source_sha256, max_seconds,
                                max_batches_per_model, output_paths):
    """Bind operational limits without accessing outputs or heldout data.

    Metadata authentication and actual source verification remain caller duties.
    """
    import json
    try:
        protocol = json.loads(protocol_bytes)
        ids = [entry['id'] for entry in protocol['models']]
        if (type(protocol_bytes) is not bytes or not ids
                or any(type(name) is not str or not name for name in ids)
                or len(set(ids)) != len(ids)
                or type(source_sha256) is not str or len(source_sha256) != 64
                or any(c not in '0123456789abcdef' for c in source_sha256)
                or type(max_seconds) is not int or max_seconds <= 0
                or type(max_batches_per_model) is not int or max_batches_per_model <= 0
                or type(output_paths) is not dict or set(output_paths) != set(ids)):
            raise ValueError('evaluation operation invalid')
        paths = list(output_paths.values())
        if (any(type(path) is not str or not path or '\x00' in path
                or not Path(path).is_absolute() or '..' in Path(path).parts
                for path in paths) or len(set(paths)) != len(paths)):
            raise ValueError('evaluation operation output paths invalid')
    except (TypeError, KeyError, UnicodeError, RecursionError) as error:
        raise ValueError('evaluation operation invalid') from error
    return json.dumps({
        'version': 1, 'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
        'source_sha256': source_sha256, 'max_seconds': max_seconds,
        'max_batches_per_model': max_batches_per_model, 'output_paths': output_paths,
    }, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def run_authenticated_evaluation(protocol, selections, cohort, *, operation_bytes,
                                 source_bytes, load_model, iter_batches, **metadata):
    """Authenticate then score missing outputs through caller-supplied adapters.

    Adapters must verify/load selected checkpoints and authorize the heldout
    iterator. This integration alone is not a deployed model/data adapter.
    Runtime checks are cooperative; a service must enforce the wall-clock cap.
    """
    import json
    import time
    retained = freeze_evaluation_protocol(protocol, selections, cohort, **metadata)
    operation = json.loads(operation_bytes)
    expected = freeze_evaluation_operation(
        protocol_bytes=retained, source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        max_seconds=operation['max_seconds'],
        max_batches_per_model=operation['max_batches_per_model'],
        output_paths=operation['output_paths'])
    if operation_bytes != expected:
        raise ValueError('evaluation operation/source binding mismatch')
    # Result identity includes stable execution source, but not per-invocation
    # budgets or output paths: safe bounded resumes can retain original bytes.
    retained = json.dumps({**json.loads(retained),
                           'execution_source_sha256': operation['source_sha256']},
                          sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode('utf-8')
    deadline = time.monotonic() + operation['max_seconds']
    results = {}
    # Validate every existing result before any model or heldout adapter runs.
    # Only an absent leaf is a resumable missing result; corruption fails closed.
    for entry in json.loads(retained)['models']:
        try:
            read_completed_evaluation_output(
                operation['output_paths'][entry['id']],
                protocol_bytes=retained, expected_model_id=entry['id'])
        except ValueError as error:
            if not isinstance(error.__cause__, FileNotFoundError):
                raise
    backend_applied = False
    for entry in json.loads(retained)['models']:
        model_id = entry['id']
        path = operation['output_paths'][model_id]
        try:
            payload, statistics = read_completed_evaluation_output(
                path, protocol_bytes=retained, expected_model_id=model_id)
        except ValueError as error:
            if not isinstance(error.__cause__, FileNotFoundError):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError('evaluation runtime budget exhausted')
            if not backend_applied:
                apply_evaluation_backend_policy(metadata['numerical_policy'])
                backend_applied = True
            model = load_model(entry)
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError('evaluation runtime budget exhausted')
                def bounded_batches():
                    stream = iter_batches(entry)
                    for batch in complete_evaluation_batches(
                            stream, max_batches=operation['max_batches_per_model']):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('evaluation runtime budget exhausted')
                        yield batch
                statistics = score_final_model_batches(model, bounded_batches())
                if time.monotonic() >= deadline:
                    raise TimeoutError('evaluation runtime budget exhausted')
                payload = publish_completed_evaluation_output(
                    path, protocol_bytes=retained, model_id=model_id,
                    statistics={key: statistics[key] for key in ('ce_sum', 'tokens')})
                payload, statistics = read_completed_evaluation_output(
                    path, protocol_bytes=retained, expected_model_id=model_id)
            finally:
                del model
        results[model_id] = {'retained_bytes': payload, 'statistics': statistics}
    return results


def load_bound_final_heldout(raw_path, *, raw_manifest_bytes,
                             expected_manifest_sha256, tokenizer_spec,
                             max_bytes, device='cpu'):
    """Connect independently retained manifest identity to pinned local packing.

    The runner must obtain expected_manifest_sha256 and tokenizer_spec from its
    authenticated frozen cohort/model contract, not from untrusted raw metadata.
    """
    import json
    from .research_models import validate_revision
    if (type(raw_manifest_bytes) is not bytes
            or len(raw_manifest_bytes) > 1024 * 1024
            or hashlib.sha256(raw_manifest_bytes).hexdigest() != expected_manifest_sha256):
        raise ValueError('heldout raw manifest binding mismatch')
    manifest = json.loads(raw_manifest_bytes)
    expected_raw = manifest['files']['test.jsonl']
    if (type(expected_raw) is not str or len(expected_raw) != 64
            or any(c not in '0123456789abcdef' for c in expected_raw)):
        raise ValueError('heldout raw hash invalid')
    validate_revision(tokenizer_spec['revision'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_spec['id'], revision=tokenizer_spec['revision'],
        trust_remote_code=False, local_files_only=True)
    return load_final_heldout_batches(
        raw_path, expected_sha256=expected_raw, tokenizer=tokenizer,
        max_bytes=max_bytes, device=device)


def load_final_heldout_batches(raw_path, *, expected_sha256, tokenizer, max_bytes, device='cpu'):
    """Pack authenticated retained raw bytes with the existing EOS/tail policy.

    Caller must bind tokenizer and raw SHA to the frozen authorized protocol.
    Returns an exactly sized batch list on the requested device; CPU by default.
    This is not a deployed scoring runner.
    """
    import json
    import torch
    descriptor = os.open(raw_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError('heldout raw size/type invalid')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            payload = stream.read(max_bytes + 1)
    finally:
        os.close(descriptor)
    if len(payload) > max_bytes or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('heldout raw SHA/size mismatch')
    if tokenizer.eos_token_id is None:
        raise ValueError('heldout packing requires EOS')
    pending, batches = [], []
    for line in payload.decode('utf-8').splitlines():
        pending.extend(tokenizer.encode(json.loads(line)['text'], add_special_tokens=False))
        pending.append(tokenizer.eos_token_id)
        while len(pending) >= 512:
            ids = torch.tensor([pending[:512]], dtype=torch.long, device=device)
            batches.append((ids, ids.clone()))
            del pending[:512]
    if not batches:
        raise ValueError('heldout packed split empty')
    return batches


def complete_evaluation_batches(batches, *, max_batches):
    """Require a metadata-sized complete stream before opening its iterator."""
    count = len(batches)
    if count > max_batches:
        raise ValueError('evaluation batch budget exceeded')
    iterator = iter(batches)
    for _ in range(count):
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise ValueError('evaluation complete stream truncated') from error
        yield batch


def score_final_model_batches(model, batches):
    """Score a caller-authorized stream; no checkpoint loading or output writes.

    Protocol authentication and heldout authorization belong to the caller.
    This primitive does not establish deployment/evaluation readiness.
    """
    return aggregate_final_token_ce(
        evaluate_final_batch(model, input_ids, labels)
        for input_ids, labels in batches)


def reuse_authenticated_evaluation_roster(protocol, selections, cohort, *, output_paths,
                                         numerical_policy, exposure_adjudication_bytes,
                                         expected_exposure_sha256):
    """Authenticate the complete metadata contract before reading saved outputs.

    This only reuses completed files; it never loads models or scores heldout.
    """
    retained = freeze_evaluation_protocol(
        protocol, selections, cohort, numerical_policy=numerical_policy,
        exposure_adjudication_bytes=exposure_adjudication_bytes,
        expected_exposure_sha256=expected_exposure_sha256)
    return reuse_completed_evaluation_roster(
        protocol_bytes=retained, output_paths=output_paths)


def reuse_completed_evaluation_roster(*, protocol_bytes, output_paths):
    """Reuse an entirely completed roster; not authorization for new scoring.

    Missing/partial/foreign files fail closed through the bounded reader. The
    caller must independently authenticate the complete frozen protocol first.
    """
    import json
    if type(protocol_bytes) is not bytes or not protocol_bytes:
        raise ValueError('completed evaluation roster protocol invalid')
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate completed evaluation roster field')
            result[key] = value
        return result

    try:
        protocol = json.loads(protocol_bytes, object_pairs_hook=unique_fields)
        models = protocol['models']
        ids = [model['id'] for model in models]
        if (type(models) is not list or not ids
                or any(type(model_id) is not str or not model_id for model_id in ids)
                or len(ids) != len(set(ids))
                or type(output_paths) is not dict or set(output_paths) != set(ids)):
            raise ValueError('completed evaluation roster mismatch')
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as error:
        raise ValueError('completed evaluation roster invalid') from error
    results = {}
    for model_id in ids:
        retained, statistics = read_completed_evaluation_output(
            output_paths[model_id], protocol_bytes=protocol_bytes,
            expected_model_id=model_id)
        results[model_id] = {'retained_bytes': retained, 'statistics': statistics}
    return results


def read_completed_evaluation_output(path, *, protocol_bytes, expected_model_id):
    """Read bounded existing result bytes without following any symlink or writing."""
    if (type(path) is not str or not path or '\x00' in path
            or not Path(path).is_absolute() or '..' in Path(path).parts):
        raise ValueError('completed evaluation output path invalid')
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in Path(path).parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(Path(path).name,
                             os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
    except OSError as error:
        raise ValueError('completed evaluation output unavailable or unsafe') from error
    finally:
        os.close(parent)
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode) or not 0 < information.st_size <= 65536:
            raise ValueError('completed evaluation output must be bounded regular file')
        retained = os.read(descriptor, 65537)
    finally:
        os.close(descriptor)
    statistics = validate_completed_evaluation_output(
        retained, protocol_bytes=protocol_bytes, expected_model_id=expected_model_id)
    return retained, statistics


def validate_completed_evaluation_output(output_bytes, *, protocol_bytes,
                                         expected_model_id):
    """Validate retained result bytes before reuse; never access heldout data."""
    import json
    if (type(output_bytes) is not bytes or len(output_bytes) > 65536
            or type(protocol_bytes) is not bytes or not protocol_bytes
            or type(expected_model_id) is not str or not expected_model_id):
        raise ValueError('completed evaluation output inputs invalid')
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate completed evaluation output field')
            result[key] = value
        return result

    try:
        output = json.loads(output_bytes, object_pairs_hook=unique_fields)
        if (type(output) is not dict
                or set(output) != {'version', 'status', 'model_id',
                                   'protocol_sha256', 'statistics'}
                or type(output['version']) is not int or output['version'] != 1
                or output['status'] != 'completed'
                or output['model_id'] != expected_model_id
                or output['protocol_sha256'] != hashlib.sha256(protocol_bytes).hexdigest()):
            raise ValueError('completed evaluation output binding mismatch')
        return aggregate_final_token_ce([output['statistics']])
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError('completed evaluation output invalid') from error


def freeze_exposure_registry(registry, *, expected_cohort,
                             expected_raw_manifest_sha256):
    """Serialize declared exposure metadata; this does not prove non-access."""
    import json
    if (not isinstance(expected_cohort, str) or not expected_cohort
            or not isinstance(expected_raw_manifest_sha256, str)
            or len(expected_raw_manifest_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_raw_manifest_sha256)
            or not isinstance(registry, dict)
            or set(registry) != {'version', 'cohort', 'raw_manifest_sha256', 'events'}
            or type(registry['version']) is not int or registry['version'] != 1
            or registry['cohort'] != expected_cohort
            or registry['raw_manifest_sha256'] != expected_raw_manifest_sha256
            or type(registry['events']) is not list or registry['events']):
        raise ValueError('exposure registry mismatch or prior exposure requires adjudication')
    return json.dumps(registry, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def freeze_evaluation_numerical_policy(policy):
    """Freeze evaluation-only settings without rewriting training contracts.

    Schema validation is not proof these settings were applied at runtime.
    """
    import json
    expected = {
        'model_dtype': 'bfloat16', 'ce_dtype': 'float32',
        'ce_reduction': 'shifted_valid_token_sum', 'ignore_index': -100,
        'batch_size': 1, 'sequence_length': 512,
        'deterministic_algorithms': True, 'cudnn_deterministic': True,
        'cudnn_benchmark': False, 'allow_tf32': False,
        'cublas_workspace_config': ':4096:8',
    }
    if (not isinstance(policy, dict) or set(policy) != set(expected)
            or any(type(policy[key]) is not type(value) or policy[key] != value
                   for key, value in expected.items())):
        raise ValueError('evaluation numerical policy mismatch')
    return json.dumps(expected, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def apply_evaluation_backend_policy(policy):
    """Apply and inspect backend flags only; model execution is a separate gate."""
    freeze_evaluation_numerical_policy(policy)
    from src.numerical_policy import configure_numerics
    import torch
    configure_numerics()
    actual = {
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    }
    if (any(actual[key] != policy[key] for key in actual)
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError('evaluation backend policy readback mismatch')
    return actual


def evaluate_final_batch(model, input_ids, labels):
    """Execute one frozen-shape BF16 model batch under inference mode."""
    import torch
    if (not isinstance(input_ids, torch.Tensor)
            or input_ids.dtype != torch.long or tuple(input_ids.shape) != (1, 512)
            or not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.long or tuple(labels.shape) != (1, 512)
            or labels.device != input_ids.device):
        raise ValueError('final evaluation batch contract mismatch')
    # BF16 is the backbone/forward policy, not a promise that PEFT's trained
    # adapter storage is BF16. get_peft_model promotes vanilla LoRA A/B weights
    # to FP32; exact checkpoint restoration intentionally preserves that dtype.
    # Recognize actual PEFT Linear parameter objects, never name substrings.
    non_bf16 = [parameter for parameter in model.parameters()
                if parameter.is_floating_point() and parameter.dtype != torch.bfloat16]
    if non_bf16:
        from peft.tuners.lora.layer import Linear as LoraLinear
        allowed = set()
        for module in model.modules():
            if (isinstance(module, LoraLinear)
                    and not module.lora_variant
                    and module.get_base_layer().weight.dtype == torch.bfloat16):
                for adapters in (module.lora_A, module.lora_B):
                    for adapter in adapters.values():
                        if isinstance(adapter, torch.nn.Linear):
                            allowed.add(id(adapter.weight))
        if any(parameter.dtype != torch.float32 or id(parameter) not in allowed
               for parameter in non_bf16):
            raise ValueError('final evaluation model must use bfloat16 backbone '
                             'with only native FP32 PEFT LoRA A/B weights')
    if any(tensor.device != input_ids.device
           for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError('final evaluation model/input device mismatch')
    if (bool(((labels < 0) & (labels != -100)).any())
            or not bool((labels[:, 1:] != -100).any())):
        raise ValueError('final evaluation invalid labels')
    model.eval()
    with torch.inference_mode():
        output = model(input_ids=input_ids)
        return final_token_ce_statistics(output.logits, labels)


def final_token_ce_statistics(logits, labels):
    """Return shifted FP32 CE sum and valid-token count, never batch means."""
    import torch
    import torch.nn.functional as functional
    if (not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or logits.ndim != 3 or labels.ndim != 2
            or logits.shape[:2] != labels.shape or logits.shape[-1] == 0
            or not logits.is_floating_point() or labels.dtype != torch.long
            or logits.device != labels.device):
        raise ValueError('invalid token CE inputs')
    if bool(((labels != -100) & ((labels < 0) | (labels >= logits.shape[-1]))).any()):
        raise ValueError('invalid token CE inputs')
    if not bool(torch.isfinite(logits.float()).all()):
        raise ValueError('nonfinite logits')
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    count = int((shifted_labels != -100).sum().item())
    if count == 0:
        raise ValueError('zero valid tokens')
    total = functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1), ignore_index=-100, reduction='sum')
    if not bool(torch.isfinite(total)):
        raise ValueError('nonfinite token CE')
    return {'ce_sum': float(total.item()), 'tokens': count}


def aggregate_final_token_ce(records):
    """Combine CE sufficient statistics with token, not batch, weighting."""
    import math
    try:
        records = list(records)
        if not records or any(
                not isinstance(record, dict) or set(record) != {'ce_sum', 'tokens'}
                or type(record['tokens']) is not int or record['tokens'] <= 0
                or type(record['ce_sum']) not in (int, float)
                or not math.isfinite(record['ce_sum']) or record['ce_sum'] < 0
                for record in records):
            raise ValueError('invalid token CE statistics')
        total = math.fsum(record['ce_sum'] for record in records)
        tokens = sum(record['tokens'] for record in records)
        ce = total / tokens
        ppl = math.exp(ce)
    except (TypeError, OverflowError) as error:
        raise ValueError('invalid token CE statistics') from error
    if not all(math.isfinite(value) for value in (total, ce, ppl)):
        raise ValueError('invalid token CE statistics')
    return {'ce_sum': total, 'tokens': tokens, 'ce': ce, 'ppl': ppl}


def verify_model_roster(entries, expected_ids):
    """Reject omitted, extra or duplicate models without reading any files."""
    if (not isinstance(expected_ids, list) or not expected_ids
            or any(not isinstance(item, str) or not item for item in expected_ids)
            or len(set(expected_ids)) != len(expected_ids)):
        raise ValueError('expected model roster invalid')
    if not isinstance(entries, list) or any(
            not isinstance(entry, dict) or not isinstance(entry.get('id'), str)
            or not entry['id'] for entry in entries):
        raise ValueError('model roster mismatch')
    actual = [entry['id'] for entry in entries]
    if len(actual) != len(set(actual)) or set(actual) != set(expected_ids):
        raise ValueError('model roster mismatch')
    return deepcopy(entries)


def freeze_final_protocol(protocol):
    """Bind the Qwen final eight-model roster to verified manifest identities.

    This is a heldout-free preflight, not permission to load test or models.
    """
    import json
    if (not isinstance(protocol, dict)
            or set(protocol) != {'version', 'family', 'models'}
            or type(protocol['version']) is not int or protocol['version'] != 1
            or protocol['family'] not in ('qwen35', 'gemma4', 'lfm25')):
        raise ValueError('final protocol schema invalid')
    expected = ['teacher', 'pretrained-student'] + [
        f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]
    models = verify_model_roster(protocol['models'], expected)
    models.sort(key=lambda model: expected.index(model['id']))
    checkpoints = []
    for model in models:
        if set(model) != {'id', 'model_id', 'model_revision', 'checkpoint'}:
            raise ValueError('final model schema invalid')
        revision = model['model_revision']
        if (not isinstance(revision, str) or len(revision) != 40
                or any(character not in '0123456789abcdef' for character in revision)):
            raise ValueError('final model revision invalid')
        expected_identity = (
            ('Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a')
            if model['id'] == 'teacher' else
            ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'))
        if protocol['family'] == 'gemma4':
            expected_identity = (
                ('google/gemma-4-12B-it', '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7')
                if model['id'] == 'teacher' else
                ('google/gemma-4-E2B-it', '3e22461f65e89153144f8adb70e3b8c2cc9845a7'))
        if protocol['family'] == 'lfm25':
            expected_identity = (
                ('LiquidAI/LFM2.5-24B-A2B', '3fcfdbfa1a1b6d8166846a6323e69f1e1254440d')
                if model['id'] == 'teacher' else
                ('LiquidAI/LFM2.5-2.6B', '654f9463ce32b05d0429d76fe1f580b27d4c1ac0'))
        if (model['model_id'], revision) != expected_identity:
            raise ValueError('final model identity mismatch')
        checkpoint = model['checkpoint']
        if model['id'] == 'pretrained-student':
            if checkpoint is not None:
                raise ValueError('pretrained student cannot have checkpoint')
        else:
            if not isinstance(checkpoint, dict) or set(checkpoint) != {
                    'manifest_path', 'manifest_sha256'}:
                raise ValueError('final model checkpoint required')
            checkpoints.append({'id': model['id'], **checkpoint})
    verify_checkpoint_list(checkpoints)
    return json.dumps({'version': 1, 'family': protocol['family'], 'models': models},
                      sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def verify_student_cohort_binding(contract, cohort):
    """Bind Student train/validation and Teacher provenance; no file access.

    The caller must supply independently frozen cohort metadata. This primitive
    alone neither authenticates that metadata nor authorizes heldout access.
    """
    import json
    try:
        if (not isinstance(contract, dict) or not isinstance(cohort, dict)
                or set(cohort) != {'data', 'provenance'}
                or not isinstance(cohort['data'], dict)
                or set(cohort['data']) != {'train', 'validation'}
                or not isinstance(cohort['provenance'], dict)
                or set(cohort['provenance']) != {
                    'base_config_hash', 'prepared_fingerprint', 'raw_manifest_hash', 'teacher'}):
            raise ValueError('cohort dependency schema invalid')
        provenance = cohort['provenance']
        teacher = provenance['teacher']
        if (not isinstance(teacher, dict)
                or set(teacher) != {'best_step', 'contract_hash', 'manifest_sha256', 'model'}
                or type(teacher.get('best_step')) is not int or teacher['best_step'] < 0
                or teacher.get('model') not in (
                    {'id': 'Qwen/Qwen3.5-9B',
                     'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'},
                    {'id': 'google/gemma-4-12B-it',
                     'revision': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'},
                    {'id': 'LiquidAI/LFM2.5-24B-A2B',
                     'revision': '3fcfdbfa1a1b6d8166846a6323e69f1e1254440d'})):
            raise ValueError('cohort dependency teacher invalid')
        hashes = [*cohort['data'].values(), provenance['base_config_hash'],
                  provenance['prepared_fingerprint'], provenance['raw_manifest_hash'],
                  teacher.get('contract_hash'), teacher.get('manifest_sha256')]
        if any(not isinstance(value, str) or len(value) != 64
               or any(char not in '0123456789abcdef' for char in value)
               for value in hashes):
            raise ValueError('cohort dependency hash invalid')
        identity = contract.get('identity')
        if not isinstance(identity, dict):
            raise ValueError('cohort dependency identity invalid')
        canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False,
                                            separators=(',', ':'))
        if (canonical(contract.get('data')) != canonical(cohort['data'])
                or canonical(identity.get('provenance')) != canonical(cohort['provenance'])):
            raise ValueError('cohort dependency mismatch')
    except (TypeError, OverflowError, RecursionError) as error:
        raise ValueError('cohort dependency invalid') from error
    return deepcopy(contract)


def verify_teacher_cohort_binding(identity, cohort):
    """Bind selected Teacher prepared identity to independently frozen cohort."""
    if not isinstance(identity, dict):
        raise ValueError('cohort dependency teacher identity invalid')
    # Reuse strict cohort schema/hash checks without accessing any corpus files.
    verify_student_cohort_binding(
        {'data': cohort.get('data') if isinstance(cohort, dict) else None,
         'identity': {'provenance': cohort.get('provenance')
                      if isinstance(cohort, dict) else None}}, cohort)
    if (identity.get('data_fingerprint') != cohort['provenance']['prepared_fingerprint']
            or identity.get('model') != cohort['provenance']['teacher']['model']
            or identity.get('stage') != 'teacher'):
        raise ValueError('cohort dependency teacher mismatch')
    return deepcopy(identity)


def verify_teacher_roster_contract(contract):
    """Validate frozen Teacher model/stage/seed metadata without file access."""
    if not isinstance(contract, dict):
        raise ValueError('teacher roster contract invalid')
    identity, settings = contract.get('identity'), contract.get('settings')
    if (not isinstance(identity, dict) or not isinstance(settings, dict)
            or identity.get('model') not in (
                {'id': 'Qwen/Qwen3.5-9B',
                 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'},
                {'id': 'google/gemma-4-12B-it',
                 'revision': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'},
                {'id': 'LiquidAI/LFM2.5-24B-A2B',
                 'revision': '3fcfdbfa1a1b6d8166846a6323e69f1e1254440d'})
            or identity.get('stage') != 'teacher'
            or type(settings.get('seed')) is not int or settings['seed'] != 42):
        raise ValueError('teacher roster contract identity mismatch')
    return deepcopy(contract)


def verify_student_roster_contract(model_id, contract):
    """Check a real Student contract's model, method and training seed.

    This standalone metadata gate does not authorize checkpoint or test loading.
    """
    allowed = {f'{arm}-seed{seed}': (arm, seed)
               for seed in (42, 43, 44) for arm in ('ce', 'kd')}
    if not isinstance(model_id, str) or model_id not in allowed or not isinstance(contract, dict):
        raise ValueError('student roster contract invalid')
    arm, seed = allowed[model_id]
    identity, settings = contract.get('identity'), contract.get('settings')
    if not isinstance(identity, dict) or not isinstance(settings, dict):
        raise ValueError('student roster contract invalid')
    model, trial = identity.get('model'), identity.get('trial')
    if model == {'id': 'google/gemma-4-E2B-it',
                 'revision': '3e22461f65e89153144f8adb70e3b8c2cc9845a7'}:
        selected_trial = arm + '-1'
    elif model == {'id': 'Qwen/Qwen3.5-4B',
                   'revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'}:
        selected_trial = {'ce': 'ce-2', 'kd': 'kd-1'}[arm]
    elif model == {'id': 'LiquidAI/LFM2.5-2.6B',
                   'revision': '654f9463ce32b05d0429d76fe1f580b27d4c1ac0'}:
        selected_trial = arm + '-2'
    else:
        raise ValueError('student roster contract identity mismatch')
    if (identity.get('stage') != arm or not isinstance(trial, dict)
            or trial.get('method') != arm
            or trial.get('id') != selected_trial
            or identity.get('trial_id') != selected_trial
            or type(settings.get('seed')) is not int or settings['seed'] != seed):
        raise ValueError('student roster contract identity mismatch')
    return deepcopy(contract)


def freeze_selected_protocol(protocol, selections):
    """Bind the complete roster to selected contracts before heldout access.

    Every binding consumes retained SHA-verified bytes, never a new path read.
    This remains metadata preflight, not a payload-loading authorization.
    """
    import json
    frozen = json.loads(freeze_final_protocol(protocol))
    checkpoints = [{'id': model['id'], **model['checkpoint']}
                   for model in frozen['models'] if model['checkpoint'] is not None]
    if (not isinstance(selections, dict)
            or set(selections) != {entry['id'] for entry in checkpoints}
            or any(not isinstance(selection, dict)
                   or set(selection) != {'step', 'contract'}
                   for selection in selections.values())):
        raise ValueError('selection roster invalid')
    selections = deepcopy(selections)
    retained = read_verified_checkpoint_manifests(checkpoints)
    for model_id, payload in retained.items():
        selection = selections[model_id]
        roster_model = next(model for model in frozen['models'] if model['id'] == model_id)
        if selection['contract'].get('identity', {}).get('model') != {
                'id': roster_model['model_id'], 'revision': roster_model['model_revision']}:
            raise ValueError('roster contract model mismatch')
        if model_id == 'teacher':
            verify_teacher_roster_contract(selection['contract'])
        else:
            verify_student_roster_contract(model_id, selection['contract'])
        bind_checkpoint_manifest(payload, expected_step=selection['step'],
                                 expected_contract=selection['contract'])
    return json.dumps({**frozen, 'selections': selections}, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def freeze_cohort_protocol(protocol, selections, cohort):
    """Bind every selected contract to independently authenticated cohort metadata.

    Metadata preflight only; numerical/exposure/payload gates remain separate.
    """
    import json
    frozen = json.loads(freeze_selected_protocol(protocol, selections))
    cohort = deepcopy(cohort)
    for model_id, selection in frozen['selections'].items():
        contract = selection['contract']
        if model_id == 'teacher':
            verify_teacher_cohort_binding(contract['identity'], cohort)
            dependency = cohort['provenance']['teacher']
            teacher_model = next(model for model in frozen['models'] if model['id'] == 'teacher')
            if teacher_model['checkpoint']['manifest_sha256'] != dependency['manifest_sha256']:
                raise ValueError('cohort dependency selected teacher manifest mismatch')
            if selection['step'] != dependency['best_step']:
                raise ValueError('cohort dependency selected teacher step mismatch')
            import hashlib
            contract_sha = hashlib.sha256(json.dumps(
                contract, sort_keys=True, separators=(',', ':'), allow_nan=False
            ).encode('utf-8')).hexdigest()
            if contract_sha != dependency['contract_hash']:
                raise ValueError('cohort dependency selected teacher contract mismatch')
        else:
            verify_student_cohort_binding(contract, cohort)
    return json.dumps({**frozen, 'cohort': cohort}, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def bind_checkpoint_manifest(payload, *, expected_step, expected_contract):
    """Bind already SHA-verified bytes to a frozen selection; no file access.

    This validates metadata only, not COMPLETE, payloads or loading safety.
    """
    import json

    def reject_constant(value):
        raise ValueError('selected checkpoint nonfinite JSON')

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('selected checkpoint duplicate JSON field')
            result[key] = value
        return result

    if (not isinstance(payload, bytes) or type(expected_step) is not int
            or expected_step < 0 or not isinstance(expected_contract, dict)
            or not isinstance(expected_contract.get('identity'), dict)
            or not expected_contract['identity']):
        raise ValueError('selected checkpoint binding inputs invalid')
    if len(payload) > 1024 * 1024:
        raise ValueError('selected checkpoint size exceeds limit')
    def finite_float(value):
        import math
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('selected checkpoint nonfinite JSON')
        return parsed

    try:
        manifest = json.loads(payload, object_pairs_hook=unique_object,
                              parse_constant=reject_constant, parse_float=finite_float)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('selected checkpoint JSON invalid') from error
    def same_typed_value(actual, expected):
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(
                same_typed_value(actual[key], value) for key, value in expected.items())
        if isinstance(expected, list):
            return len(actual) == len(expected) and all(
                same_typed_value(left, right) for left, right in zip(actual, expected))
        return actual == expected

    if (not isinstance(manifest, dict)
            or type(manifest.get('version')) is not int or manifest['version'] != 1
            or type(manifest.get('step')) is not int or manifest['step'] != expected_step
            or not same_typed_value(manifest.get('contract'), expected_contract)
            or not same_typed_value(manifest.get('identity'), expected_contract.get('identity', {}))):
        raise ValueError('selected checkpoint step/contract/identity mismatch')
    return manifest


def verify_complete_marker(marker_bytes, manifest_bytes):
    """Bind retained COMPLETE bytes to actual manifest bytes; no payload access."""
    import json
    if (not isinstance(marker_bytes, bytes) or not isinstance(manifest_bytes, bytes)
            or len(marker_bytes) > 4096):
        raise ValueError('COMPLETE marker inputs invalid')
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate COMPLETE field')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('nonfinite COMPLETE value')

    def finite_float(value):
        import math
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('nonfinite COMPLETE value')
        return parsed

    try:
        marker = json.loads(marker_bytes, object_pairs_hook=unique_object,
                            parse_constant=reject_constant, parse_float=finite_float)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('COMPLETE marker JSON invalid') from error
    if (not isinstance(marker, dict)
            or marker.get('manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest()):
        raise ValueError('COMPLETE manifest SHA mismatch')


def bind_checkpoint_commit(marker_bytes, manifest_bytes, *, expected_step, expected_contract):
    """Bind retained commit bytes to selection; does not authorize payload loading."""
    verify_complete_marker(marker_bytes, manifest_bytes)
    return bind_checkpoint_manifest(manifest_bytes, expected_step=expected_step,
                                    expected_contract=expected_contract)


def read_verified_checkpoint_payload(generation_dir, manifest, *, max_bytes):
    """Retain bounded SHA/size-verified bytes without deserialization.

    Caller must first bind retained COMPLETE/manifest to the frozen selection.
    Every path component is opened without following symlinks.
    """
    if (type(max_bytes) is not int or max_bytes <= 0
            or not isinstance(manifest, dict)
            or not isinstance(manifest.get('files'), dict)
            or set(manifest['files']) != {'state.pt'}):
        raise ValueError('checkpoint payload schema invalid')
    metadata = manifest['files']['state.pt']
    if (not isinstance(metadata, dict) or set(metadata) != {'size', 'sha256'}
            or type(metadata['size']) is not int
            or not 0 < metadata['size'] <= max_bytes
            or not isinstance(metadata['sha256'], str)
            or len(metadata['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in metadata['sha256'])):
        raise ValueError('checkpoint payload metadata/limit invalid')
    if (not isinstance(generation_dir, str) or not generation_dir
            or '\x00' in generation_dir or not Path(generation_dir).is_absolute()
            or '..' in Path(generation_dir).parts):
        raise ValueError('checkpoint payload path invalid')
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in Path(generation_dir).parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open('state.pt', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError('checkpoint payload must be regular non-symlink') from error
        raise
    finally:
        os.close(parent)
    with os.fdopen(descriptor, 'rb') as stream:
        information = os.fstat(stream.fileno())
        if not stat.S_ISREG(information.st_mode):
            raise ValueError('checkpoint payload must be regular non-symlink')
        if information.st_size != metadata['size']:
            raise ValueError('checkpoint payload hash/size mismatch')
        payload = stream.read(metadata['size'] + 1)
    if (len(payload) != metadata['size']
            or hashlib.sha256(payload).hexdigest() != metadata['sha256']):
        raise ValueError('checkpoint payload hash/size mismatch')
    return payload


def load_selected_checkpoint(generation_dir, marker_bytes, manifest_bytes, *,
                             expected_step, expected_contract, max_bytes):
    """Load CPU state only after retained commit, selection and payload binding.

    The caller still owns authoritative roster/cohort/protocol authorization.
    This does not read heldout data or permit arbitrary pickle fallback.
    """
    manifest = bind_checkpoint_commit(marker_bytes, manifest_bytes,
                                      expected_step=expected_step,
                                      expected_contract=expected_contract)
    payload = read_verified_checkpoint_payload(generation_dir, manifest, max_bytes=max_bytes)
    return load_retained_checkpoint_payload(payload)


def load_final_roster_model(entry, selections, commits, *, device='cuda'):
    """Dispatch an authenticated roster through retained selected commits.

    Callers must authenticate the full roster before invoking this adapter.
    The unadapted Student never reads a checkpoint or attaches LoRA.
    """
    if entry['id'] != 'pretrained-student':
        import hashlib
        selection = selections[entry['id']]
        commit = commits[entry['id']]
        checkpoint = entry['checkpoint']
        if hashlib.sha256(commit['manifest_bytes']).hexdigest() != checkpoint['manifest_sha256']:
            raise ValueError('roster retained manifest mismatch')
        contract = selection['contract']
        model_spec = {'id': entry['model_id'], 'revision': entry['model_revision']}
        return load_final_selected_model(
            str(Path(checkpoint['manifest_path']).parent),
            commit['marker_bytes'], commit['manifest_bytes'],
            expected_step=selection['step'], expected_contract=contract,
            max_bytes=commit['max_bytes'], model_spec=model_spec,
            adapter=contract['identity']['adapter'], seed=contract['settings']['seed'],
            device=device)
    if entry['checkpoint'] is not None:
        raise ValueError('pretrained roster checkpoint must be absent')
    from .research_models import load_model
    model = load_model({'id': entry['model_id'], 'revision': entry['model_revision']},
                       device=device, dtype='bfloat16', attention='sdpa')
    model.requires_grad_(False)
    model.eval()
    return model


def load_final_selected_model(generation_dir, marker_bytes, manifest_bytes, *,
                              expected_step, expected_contract, max_bytes,
                              model_spec, adapter, seed, device='cuda'):
    """Reconstruct base/LoRA and restore a bound selected checkpoint.

    Construction arguments must match the original selected contract. No heldout
    reads occur here; commit/payload verification precedes base model loading.
    """
    identity = expected_contract.get('identity', {})
    settings = expected_contract.get('settings', {})
    if (model_spec != identity.get('model') or adapter != identity.get('adapter')
            or type(seed) is not int or seed != settings.get('seed')):
        raise ValueError('construction contract mismatch')
    from .research_models import load_model, attach_lora
    from .research_runner import _restore_model
    checkpoint = load_selected_checkpoint(
        generation_dir, marker_bytes, manifest_bytes, expected_step=expected_step,
        expected_contract=expected_contract, max_bytes=max_bytes)
    model = load_model(model_spec, device=device, dtype='bfloat16', attention='sdpa')
    model = attach_lora(model, adapter, seed=seed)
    _restore_model(model, checkpoint['model_state'])
    model.requires_grad_(False)
    model.eval()
    return model


def load_retained_checkpoint_payload(payload):
    """Deserialize retained bytes on CPU without arbitrary pickle fallback.

    Standalone loading primitive only: caller must verify commit, selection and
    payload before use. Unsupported legacy objects fail closed.
    """
    import io
    import torch
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)


def verify_checkpoint_list(entries):
    """Read checkpoint manifests only; never load models or heldout datasets."""
    read_verified_checkpoint_manifests(entries)
    return entries


def read_verified_checkpoint_manifests(entries):
    """Retain hash-verified bytes for parsing without a second path read.

    Payload/COMPLETE verification and safe model loading remain separate gates.
    """
    if not isinstance(entries, list) or not entries:
        raise ValueError('checkpoint list must be nonempty')
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('id'), str) or not entry['id']:
            raise ValueError('checkpoint model id required')
        if entry['id'] in seen:
            raise ValueError('duplicate checkpoint model id')
        seen.add(entry['id'])
        if set(entry) != {'id', 'manifest_path', 'manifest_sha256'}:
            raise ValueError('checkpoint entry schema invalid')
        path = entry['manifest_path']
        if (not isinstance(path, str) or not path or '\x00' in path
                or not Path(path).is_absolute() or len(Path(path).parts) < 2
                or '..' in Path(path).parts
                or any(0xD800 <= ord(character) <= 0xDFFF for character in path)):
            raise ValueError('checkpoint entry manifest path invalid')
        digest = entry['manifest_sha256']
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in '0123456789abcdef' for character in digest)):
            raise ValueError('checkpoint entry SHA invalid')
    verified = {}
    for entry in entries:
        parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        try:
            parts = Path(entry['manifest_path']).parts[1:]
            for component in parts[:-1]:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=parent)
                os.close(parent)
                parent = child
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=parent)
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError('checkpoint manifest must be regular non-symlink') from error
            raise
        finally:
            os.close(parent)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError('checkpoint manifest must be regular non-symlink')
        with os.fdopen(descriptor, 'rb') as manifest:
            if os.fstat(manifest.fileno()).st_size > 1024 * 1024:
                raise ValueError('checkpoint manifest size exceeds limit')
            payload = manifest.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError('checkpoint manifest size exceeds limit')
            actual = hashlib.sha256(payload).hexdigest()
        if actual != entry['manifest_sha256']:
            raise ValueError('checkpoint manifest SHA mismatch')
        verified[entry['id']] = payload
    return verified
