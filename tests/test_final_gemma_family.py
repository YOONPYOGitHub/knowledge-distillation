"""Gemma final roster admission; no checkpoint payload or corpus access."""
import json
import hashlib
from src.final_evaluation import freeze_final_protocol, verify_teacher_roster_contract
from copy import deepcopy
import pytest


def test_gemma_student_contract_binds_selected_trial_and_pinned_model():
    from src.final_evaluation import verify_student_roster_contract
    for arm in ('ce', 'kd'):
        for seed in (42, 43, 44):
            contract = {'identity': {'stage': arm, 'model': {
                'id': 'google/gemma-4-E2B-it',
                'revision': '3e22461f65e89153144f8adb70e3b8c2cc9845a7'},
                'trial_id': arm + '-1', 'trial': {'id': arm + '-1', 'method': arm}},
                'settings': {'seed': seed}}
            assert verify_student_roster_contract(f'{arm}-seed{seed}', contract) == contract
            for path, value in [(('model', 'revision'), '0' * 40),
                                (('trial', 'id'), arm + '-2'),
                                (('trial_id',), arm + '-2')]:
                wrong = deepcopy(contract)
                target = wrong['identity']
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with pytest.raises(ValueError, match='student roster contract'):
                    verify_student_roster_contract(f'{arm}-seed{seed}', wrong)


def test_gemma_cohort_teacher_accepts_exact_pinned_identity():
    from src.final_evaluation import verify_student_cohort_binding, verify_teacher_cohort_binding
    provenance = {'base_config_hash': 'a' * 64, 'prepared_fingerprint': 'b' * 64,
                  'raw_manifest_hash': 'c' * 64, 'teacher': {
                      'best_step': 3000, 'contract_hash': 'd' * 64,
                      'manifest_sha256': 'e' * 64, 'model': {
                          'id': 'google/gemma-4-12B-it',
                          'revision': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'}}}
    cohort = {'data': {'train': 'f' * 64, 'validation': '0' * 64}, 'provenance': provenance}
    contract = {'data': cohort['data'], 'identity': {'provenance': provenance}}
    assert verify_student_cohort_binding(contract, cohort) == contract
    identity = {'stage': 'teacher', 'model': provenance['teacher']['model'],
                'data_fingerprint': provenance['prepared_fingerprint']}
    assert verify_teacher_cohort_binding(identity, cohort) == identity
    wrong = deepcopy(cohort)
    wrong['provenance']['teacher']['model']['revision'] = '0' * 40
    with pytest.raises(ValueError, match='cohort dependency'):
        verify_student_cohort_binding(contract, wrong)


def test_gemma_teacher_contract_accepts_only_pinned_identity():
    contract = {'identity': {'stage': 'teacher', 'model': {
        'id': 'google/gemma-4-12B-it',
        'revision': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'}},
        'settings': {'seed': 42}}
    assert verify_teacher_roster_contract(contract) == contract
    for field, value in [('revision', '0' * 40), ('id', 'google/gemma-4-E2B-it')]:
        wrong = deepcopy(contract)
        wrong['identity']['model'][field] = value
        with pytest.raises(ValueError, match='teacher roster contract'):
            verify_teacher_roster_contract(wrong)


def test_selected_roster_rejects_cross_family_teacher_contract(tmp_path):
    from src.final_evaluation import freeze_selected_protocol
    models, selections = [], {}
    for key in ['teacher', 'pretrained-student'] + [
            f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]:
        teacher = key == 'teacher'
        model = {'id': 'Qwen/Qwen3.5-9B' if teacher else 'Qwen/Qwen3.5-4B',
                 'revision': ('c202236235762e1c871ad0ccb60c8ee5ba337b9a' if teacher
                              else '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')}
        checkpoint = None
        if key != 'pretrained-student':
            if teacher:
                identity = {'stage': 'teacher', 'model': {
                    'id': 'google/gemma-4-12B-it',
                    'revision': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'}}
                seed = 42
            else:
                arm, seed = key.split('-seed')
                trial_id = 'ce-2' if arm == 'ce' else 'kd-1'
                identity = {'stage': arm, 'model': model, 'trial_id': trial_id,
                            'trial': {'id': trial_id, 'method': arm}}
            contract = {'identity': identity, 'settings': {'seed': int(seed)}}
            payload = json.dumps({'version': 1, 'step': 3000,
                                  'identity': identity, 'contract': contract}).encode()
            path = tmp_path.resolve() / (key + '.json')
            path.write_bytes(payload)
            checkpoint = {'manifest_path': str(path),
                          'manifest_sha256': hashlib.sha256(payload).hexdigest()}
            selections[key] = {'step': 3000, 'contract': contract}
        models.append({'id': key, 'model_id': model['id'],
                       'model_revision': model['revision'], 'checkpoint': checkpoint})
    with pytest.raises(ValueError, match='roster.*model'):
        freeze_selected_protocol({'version': 1, 'family': 'qwen35', 'models': models}, selections)


def test_frozen_gemma_roster_accepts_exact_pinned_family(tmp_path):
    models = []
    for key in ['teacher', 'pretrained-student'] + [
            f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]:
        teacher = key == 'teacher'
        manifest = tmp_path.resolve() / (key + '.json')
        manifest.write_bytes(b'{}')
        models.append({'id': key,
            'model_id': 'google/gemma-4-12B-it' if teacher else 'google/gemma-4-E2B-it',
            'model_revision': ('707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7' if teacher
                               else '3e22461f65e89153144f8adb70e3b8c2cc9845a7'),
            'checkpoint': None if key == 'pretrained-student' else {
                'manifest_path': str(manifest),
                'manifest_sha256': hashlib.sha256(b'{}').hexdigest()}})
    protocol = {'version': 1, 'family': 'gemma4', 'models': models}
    assert json.loads(freeze_final_protocol(protocol)) == protocol
