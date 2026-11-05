"""Cohort binding must be enforced by the full selected-roster preflight."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from src import final_evaluation


def test_full_protocol_rejects_consistently_substituted_student_cohort(tmp_path):
    root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
    teacher = json.loads((root / 'final-teacher-identity-20261026.json').read_text())
    cohort = json.loads((root / 'final-cohort-auth-20261026T1205.json').read_text())['cohort']
    models, selections = [], {}
    for model_id in ['teacher', 'pretrained-student'] + [
            f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]:
        is_teacher = model_id == 'teacher'
        checkpoint = None
        if model_id != 'pretrained-student':
            if is_teacher:
                contract = {key: teacher[key] for key in ('identity', 'settings')}
                cohort['provenance']['teacher']['contract_hash'] = hashlib.sha256(
                    json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            else:
                arm, seed = model_id.split('-seed')
                folder = {'42': 'round2', '43': 'seed43', '44': 'seed44'}[seed]
                archive = json.loads((root / f'{folder}-originals-transfer/originals.json').read_text())
                trial = {'ce': 'ce-2', 'kd': 'kd-1'}[arm]
                contract = deepcopy(archive[f'trials/{trial}/result.json']['checkpoint_contract'])
                contract['identity']['provenance']['teacher'] = deepcopy(cohort['provenance']['teacher'])
            payload = json.dumps({'version': 1, 'step': 3000,
                                  'identity': contract['identity'], 'contract': contract}).encode()
            if is_teacher:
                cohort['provenance']['teacher']['manifest_sha256'] = hashlib.sha256(payload).hexdigest()
            path = tmp_path.resolve() / f'{model_id}.json'
            path.write_bytes(payload)
            checkpoint = {'manifest_path': str(path), 'manifest_sha256': hashlib.sha256(payload).hexdigest()}
            selections[model_id] = {'step': 3000, 'contract': contract}
        models.append({'id': model_id,
                       'model_id': 'Qwen/Qwen3.5-9B' if is_teacher else 'Qwen/Qwen3.5-4B',
                       'model_revision': ('c202236235762e1c871ad0ccb60c8ee5ba337b9a' if is_teacher
                                          else '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
                       'checkpoint': checkpoint})
    protocol = {'version': 1, 'family': 'qwen35', 'models': models}
    freeze = getattr(final_evaluation, 'freeze_cohort_protocol', None)
    assert callable(freeze), 'full protocol cohort gate missing'
    frozen = json.loads(freeze(protocol, selections, cohort))
    assert frozen['cohort'] == cohort
    teacher_path = Path(next(model['checkpoint']['manifest_path'] for model in models
                             if model['id'] == 'teacher'))
    original_teacher = teacher_path.read_bytes()
    changed_payload = original_teacher + b'\n'
    teacher_path.write_bytes(changed_payload)
    changed_protocol = deepcopy(protocol)
    next(model for model in changed_protocol['models'] if model['id'] == 'teacher')['checkpoint']['manifest_sha256'] = hashlib.sha256(changed_payload).hexdigest()
    with pytest.raises(ValueError, match='cohort dependency'):
        freeze(changed_protocol, selections, cohort)
    teacher_path.write_bytes(original_teacher)
    wrong_selection = deepcopy(selections)
    wrong_selection['teacher']['step'] = 3001
    wrong_manifest = json.loads(original_teacher)
    wrong_manifest['step'] = 3001
    wrong_payload = json.dumps(wrong_manifest).encode()
    teacher_path.write_bytes(wrong_payload)
    wrong_protocol = deepcopy(protocol)
    next(model for model in wrong_protocol['models'] if model['id'] == 'teacher')['checkpoint']['manifest_sha256'] = hashlib.sha256(wrong_payload).hexdigest()
    with pytest.raises(ValueError, match='cohort dependency'):
        freeze(wrong_protocol, wrong_selection, cohort)
    teacher_path.write_bytes(original_teacher)
    wrong_teacher_selection = deepcopy(selections)
    wrong_teacher_selection['teacher']['contract']['settings']['learning_rate'] = 0.123
    wrong_teacher_manifest = json.loads(original_teacher)
    wrong_teacher_manifest['contract'] = wrong_teacher_selection['teacher']['contract']
    wrong_teacher_payload = json.dumps(wrong_teacher_manifest).encode()
    teacher_path.write_bytes(wrong_teacher_payload)
    wrong_teacher_protocol = deepcopy(protocol)
    wrong_teacher_sha = hashlib.sha256(wrong_teacher_payload).hexdigest()
    next(model for model in wrong_teacher_protocol['models'] if model['id'] == 'teacher')['checkpoint']['manifest_sha256'] = wrong_teacher_sha
    wrong_teacher_cohort = deepcopy(cohort)
    wrong_teacher_cohort['provenance']['teacher']['manifest_sha256'] = wrong_teacher_sha
    for model_id, selection in wrong_teacher_selection.items():
        if model_id != 'teacher':
            selection['contract']['identity']['provenance']['teacher']['manifest_sha256'] = wrong_teacher_sha
            model = next(model for model in wrong_teacher_protocol['models'] if model['id'] == model_id)
            manifest = json.loads(Path(model['checkpoint']['manifest_path']).read_bytes())
            manifest['contract'] = selection['contract']
            manifest['identity'] = selection['contract']['identity']
            updated = json.dumps(manifest).encode()
            Path(model['checkpoint']['manifest_path']).write_bytes(updated)
            model['checkpoint']['manifest_sha256'] = hashlib.sha256(updated).hexdigest()
    with pytest.raises(ValueError, match='cohort dependency.*contract'):
        freeze(wrong_teacher_protocol, wrong_teacher_selection, wrong_teacher_cohort)
    teacher_path.write_bytes(original_teacher)
    # Restore the synthetic Student manifests for the independent cohort probes.
    for model_id, selection in selections.items():
        if model_id != 'teacher':
            model = next(model for model in protocol['models'] if model['id'] == model_id)
            Path(model['checkpoint']['manifest_path']).write_bytes(json.dumps({
                'version': 1, 'step': 3000, 'identity': selection['contract']['identity'],
                'contract': selection['contract']}).encode())
    for model_id in selections:
        wrong = deepcopy(selections)
        contract = wrong[model_id]['contract']
        if model_id == 'teacher':
            contract['identity']['data_fingerprint'] = '0' * 64
        else:
            contract['data']['train'] = '0' * 64
        payload = json.dumps({'version': 1, 'step': 3000,
                              'identity': contract['identity'], 'contract': contract}).encode()
        path = Path(next(model['checkpoint']['manifest_path'] for model in models if model['id'] == model_id))
        original = path.read_bytes()
        path.write_bytes(payload)
        changed_protocol = deepcopy(protocol)
        next(model for model in changed_protocol['models'] if model['id'] == model_id)['checkpoint']['manifest_sha256'] = hashlib.sha256(payload).hexdigest()
        with pytest.raises(ValueError, match='cohort dependency'):
            freeze(changed_protocol, wrong, cohort)
        path.write_bytes(original)
