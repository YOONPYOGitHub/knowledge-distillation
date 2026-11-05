"""Regression of original contracts, not evidence of remote payload readiness."""
import hashlib
import json
from pathlib import Path

from src.final_evaluation import freeze_cohort_protocol


def test_original_teacher_and_six_unmodified_student_contracts_bind(tmp_path):
    root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
    cohort = json.loads((root / 'final-cohort-auth-20261026T1205.json').read_text())['cohort']
    teacher_path = root / 'final-teacher-original-manifest-20261026.json'
    original_teacher = teacher_path.read_bytes()
    teacher = json.loads(original_teacher)
    assert hashlib.sha256(original_teacher).hexdigest() == cohort['provenance']['teacher']['manifest_sha256']
    models, selections, original_contracts = [], {}, {}
    for model_id in ['teacher', 'pretrained-student'] + [
            f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]:
        checkpoint = None
        is_teacher = model_id == 'teacher'
        if model_id != 'pretrained-student':
            if is_teacher:
                contract = teacher['contract']
                path = teacher_path
                payload = original_teacher
            else:
                arm, seed = model_id.split('-seed')
                folder = {'42': 'round2', '43': 'seed43', '44': 'seed44'}[seed]
                archive = json.loads((root / f'{folder}-originals-transfer/originals.json').read_text())
                trial = {'ce': 'ce-2', 'kd': 'kd-1'}[arm]
                contract = archive[f'trials/{trial}/result.json']['checkpoint_contract']
                # Construct metadata envelopes only; never rewrite archived contracts.
                payload = json.dumps({'version': 1, 'step': 3000,
                                      'identity': contract['identity'], 'contract': contract}).encode()
                path = tmp_path.resolve() / f'{model_id}.json'
                path.write_bytes(payload)
            original_contracts[model_id] = json.dumps(contract, sort_keys=True)
            checkpoint = {'manifest_path': str(path), 'manifest_sha256': hashlib.sha256(payload).hexdigest()}
            selections[model_id] = {'step': 3000, 'contract': contract}
        models.append({'id': model_id,
                       'model_id': 'Qwen/Qwen3.5-9B' if is_teacher else 'Qwen/Qwen3.5-4B',
                       'model_revision': ('c202236235762e1c871ad0ccb60c8ee5ba337b9a' if is_teacher
                                          else '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
                       'checkpoint': checkpoint})
    frozen = json.loads(freeze_cohort_protocol(
        {'version': 1, 'family': 'qwen35', 'models': models}, selections, cohort))
    assert frozen['cohort'] == cohort
    assert {key: json.dumps(value['contract'], sort_keys=True)
            for key, value in frozen['selections'].items()} == original_contracts
    assert teacher_path.read_bytes() == original_teacher
