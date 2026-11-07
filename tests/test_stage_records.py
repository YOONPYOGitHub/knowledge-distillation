"""Real CPU TinyLM checkpoints; no model downloads or efficacy claims."""
import copy
import json
from pathlib import Path

import pytest
import torch

from src.research_runner import train_stage
from src.resume_checkpoint import load_checkpoint, save_checkpoint, sha256_file
from test_research_runner import TinyLM, examples


@pytest.fixture
def completed(tmp_path, request):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        model = TinyLM()
        # A declared selection threshold retains the genuine initial best.
        settings = dict(seed=42, batch_size=2, max_steps=2, epochs=2,
                        validation_interval=1, patience=0, min_delta=100.0,
                        checkpoint_interval_steps=1, alpha=1.0, temperature=1.0,
                        divergence='forward_kl', vocab_size=0, device='cpu')
        settings.update(getattr(request, 'param', {}))
        identity = {'stage': 'teacher', 'base': 'tiny', 'revision': 'fixed'}
        result = train_stage(model, examples(6), examples(3),
                             torch.optim.AdamW(model.parameters(), lr=0.03),
                             tmp_path/'teacher', settings=settings, identity=identity)
        checkpoint_contract = json.loads((Path(result['best_checkpoint'])/'manifest.json').read_text())['contract']
        record = dict(contract=dict(settings=settings, identity=identity),
                      checkpoint_contract=checkpoint_contract, result=result,
                      teacher_selection=('pretrained_step0_retained_no_validation_improvement'
                                         if result['best_step'] == 0 else
                                         'validation_selected_additionally_trained_teacher'))
        (tmp_path/'teacher.json').write_text(json.dumps(record))
        yield tmp_path, record
    finally:
        torch.set_num_threads(threads)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_new_plan_fingerprints_stage_record_guard():
    from src.research_tuning import _code_identity
    module = Path(__file__).resolve().parents[1]/'src/stage_records.py'
    assert _code_identity()['src/stage_records.py'] == sha256_file(module)


def test_public_cli_rejects_stale_completed_record(completed):
    from scripts.run_research import read_completed
    root, record = completed
    record['result']['final_step'] += 123
    (root/'teacher.json').write_text(json.dumps(record))
    before = snapshot(root)
    with pytest.raises(ValueError, match='result'):
        read_completed(root, 'teacher', contract=record['contract'])
    assert snapshot(root) == before


def test_forged_completed_result_is_rejected_without_writes(completed):
    root, record = completed
    forged = copy.deepcopy(record)
    forged['result'].update(final_step=200, best_step=123,
                            best_validation={'ce': 0, 'ppl': 1, 'tokens': 1})
    (root/'teacher.json').write_text(json.dumps(forged))
    # The selected payload itself is genuine; the unverified summary is forged.
    assert load_checkpoint(record['result']['best_checkpoint'])['step'] == 0
    from src.stage_records import reconcile_completed_stage
    before = snapshot(root)
    with pytest.raises(ValueError, match='result'):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before


def test_genuine_earlier_running_best_is_valid_and_read_only(completed):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    selected = load_checkpoint(record['result']['best_checkpoint'])
    assert selected['status'] == 'running'
    assert selected['step'] == 0 < record['result']['final_step']
    before = snapshot(root)
    assert reconcile_completed_stage(root, 'teacher', contract=record['contract']) == record
    assert snapshot(root) == before


@pytest.mark.parametrize('status', ['running', 'interrupted'])
def test_latest_must_be_durably_completed_even_if_record_matches(completed, status):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    latest = load_checkpoint(root/'teacher')
    latest.update(status=status, stop_reason='sigterm' if status == 'interrupted' else '')
    path = save_checkpoint(root/'teacher', latest, record['checkpoint_contract'])
    record['result'].update(status=status, stop_reason=latest['stop_reason'], last_checkpoint=path)
    (root/'teacher.json').write_text(json.dumps(record))
    before = snapshot(root)
    with pytest.raises(ValueError, match='completed'):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before


@pytest.mark.parametrize('damage', ['missing', 'payload', 'marker'])
def test_selected_best_is_verified_before_accepting_record(completed, damage):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    best = Path(record['result']['best_checkpoint'])
    if damage == 'missing':
        (best/'COMPLETE').unlink()
    elif damage == 'payload':
        (best/'state.pt').write_bytes(b'corrupt selected payload')
    else:
        (best/'COMPLETE').write_text(json.dumps({'manifest_sha256': '0'*64}))
    before = snapshot(root)
    with pytest.raises(ValueError, match='COMPLETE|hash'):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before


@pytest.mark.parametrize('field', ['best_step', 'best_validation', 'history'])
def test_durable_selected_best_must_agree_with_latest(completed, field):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    latest = load_checkpoint(root/'teacher')
    if field == 'best_step':
        latest[field] = latest['step']
        record['result'][field] = latest[field]
    elif field == 'best_validation':
        latest[field] = {k: v for k, v in latest['history'][-1].items() if k != 'step'}
        assert latest[field] != record['result'][field]
        record['result'][field] = latest[field]
    else:
        # Deliberately inconsistent provenance; these are not numerical claims.
        latest['history'][0]['tokens'] += 1
        record['result']['history'] = latest['history']
        record['result']['initial_validation']['tokens'] += 1
    record['result']['last_checkpoint'] = save_checkpoint(root/'teacher', latest, record['checkpoint_contract'])
    (root/'teacher.json').write_text(json.dumps(record))
    before = snapshot(root)
    with pytest.raises(ValueError, match='selected|best|history'):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before


@pytest.mark.parametrize('changed', ['expected', 'stage'])
def test_stage_and_checkpoint_contracts_must_agree(completed, changed):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    expected = copy.deepcopy(record['contract'])
    if changed == 'expected':
        expected['identity']['revision'] = 'changed'
    else:
        record['contract']['settings']['seed'] += 1
        (root/'teacher.json').write_text(json.dumps(record))
        expected = None
    before = snapshot(root)
    with pytest.raises(ValueError, match='contract'):
        reconcile_completed_stage(root, 'teacher', contract=expected)
    assert snapshot(root) == before


@pytest.mark.parametrize('reference', ['absolute', 'relative', 'symlink'])
def test_selected_reference_cannot_use_verified_outside_generation(completed, reference):
    import shutil
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    best = Path(record['result']['best_checkpoint'])
    outside = root/'outside-best'
    shutil.copytree(best, outside)
    latest = load_checkpoint(root/'teacher')
    if reference == 'absolute':
        latest['best_checkpoint'] = str(outside)
    elif reference == 'relative':
        latest['best_checkpoint'] = '../outside-best'
    else:
        link = root/'teacher'/'checkpoints'/'outside-link'
        link.symlink_to(outside, target_is_directory=True)
        latest['best_checkpoint'] = 'checkpoints/outside-link'
    record['result']['best_checkpoint'] = str(outside)
    record['result']['last_checkpoint'] = save_checkpoint(root/'teacher', latest, record['checkpoint_contract'])
    (root/'teacher.json').write_text(json.dumps(record))
    before = snapshot(root)
    with pytest.raises(ValueError, match='path|generation|symlink'):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before


@pytest.mark.parametrize('target', ['teacher', 'teacher/checkpoints', 'teacher.json',
                                   'teacher/latest.json', 'selected-payload'])
def test_symlinked_stage_sources_are_rejected(completed, target):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    path = (Path(record['result']['best_checkpoint'])/'state.pt'
            if target == 'selected-payload' else root/target)
    outside = root/('external-' + path.name)
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=outside.is_dir())
    before = snapshot(root)
    with pytest.raises(ValueError, match='path|generation|symlink'):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before


@pytest.mark.parametrize('completed', [{'min_delta': 0.0}], indirect=True)
def test_genuine_completed_best_can_be_the_latest_generation(completed):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    assert record['result']['best_step'] == record['result']['final_step']
    assert record['result']['best_checkpoint'] == record['result']['last_checkpoint']
    assert reconcile_completed_stage(root, 'teacher') == record


@pytest.mark.parametrize('field', ['status', 'final_step', 'best_step', 'initial_validation',
                                 'best_validation', 'best_checkpoint', 'last_checkpoint',
                                 'stop_reason', 'history', 'extra'])
def test_every_result_field_is_compared(completed, field):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    record['result'][field] = 'forged'
    (root/'teacher.json').write_text(json.dumps(record))
    before = snapshot(root)
    with pytest.raises(ValueError, match='result'):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before


@pytest.mark.parametrize('missing', ['settings', 'identity'])
def test_incomplete_stage_contract_is_rejected(completed, missing):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    del record['contract'][missing]
    (root/'teacher.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='contract'):
        reconcile_completed_stage(root, 'teacher')


@pytest.mark.parametrize('missing', ['teacher.json', 'teacher/latest.json', 'latest-state'])
def test_missing_required_source_fails_closed_as_value_error(completed, missing):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    path = Path(record['result']['last_checkpoint'])/'state.pt' if missing == 'latest-state' else root/missing
    path.unlink()
    before = snapshot(root)
    with pytest.raises(ValueError):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before


def test_latest_pointer_hash_is_verified(completed):
    from src.stage_records import reconcile_completed_stage
    root, _ = completed
    path = root/'teacher'/'latest.json'
    pointer = json.loads(path.read_text())
    pointer['manifest_sha256'] = '0'*64
    path.write_text(json.dumps(pointer))
    before = snapshot(root)
    with pytest.raises(ValueError, match='hash'):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before


@pytest.mark.parametrize('stage', ['../teacher', '/teacher', 'unknown'])
def test_invalid_stage_is_rejected_before_reading(tmp_path, stage):
    from src.stage_records import reconcile_completed_stage
    with pytest.raises(ValueError, match='stage'):
        reconcile_completed_stage(tmp_path, stage)
    assert list(tmp_path.iterdir()) == []


def test_running_step_zero_cannot_be_promoted_by_forged_completed_json(completed):
    from src.stage_records import reconcile_completed_stage
    root, record = completed
    best = Path(record['result']['best_checkpoint'])
    pointer = {'checkpoint': str(best.relative_to(root/'teacher')),
               'manifest_sha256': sha256_file(best/'manifest.json')}
    (root/'teacher'/'latest.json').write_text(json.dumps(pointer))
    record['result'].update(final_step=200, best_step=123,
                            best_validation={'ce': 0, 'ppl': 1, 'tokens': 1})
    (root/'teacher.json').write_text(json.dumps(record))
    assert load_checkpoint(root/'teacher')['status'] == 'running'
    before = snapshot(root)
    with pytest.raises(ValueError, match='completed'):
        reconcile_completed_stage(root, 'teacher')
    assert snapshot(root) == before
