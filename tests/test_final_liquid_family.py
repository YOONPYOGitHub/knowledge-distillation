"""Liquid exact pinned roster admission; synthetic manifests only."""
import hashlib
import json
from copy import deepcopy
import pytest
from src.final_evaluation import freeze_final_protocol


def test_liquid_teacher_contract_accepts_only_pinned_identity():
    from src.final_evaluation import verify_teacher_roster_contract
    contract = {'identity': {'stage': 'teacher', 'model': {
        'id': 'LiquidAI/LFM2.5-24B-A2B',
        'revision': '3fcfdbfa1a1b6d8166846a6323e69f1e1254440d'}},
        'settings': {'seed': 42}}
    assert verify_teacher_roster_contract(contract) == contract
    wrong = deepcopy(contract)
    wrong['identity']['model']['revision'] = '0' * 40
    with pytest.raises(ValueError, match='teacher roster contract'):
        verify_teacher_roster_contract(wrong)


def test_liquid_student_contract_binds_exact_selected_pair():
    from src.final_evaluation import verify_student_roster_contract
    for arm in ('ce', 'kd'):
        for seed in (42, 43, 44):
            contract = {'identity': {'stage': arm, 'model': {
                'id': 'LiquidAI/LFM2.5-2.6B',
                'revision': '654f9463ce32b05d0429d76fe1f580b27d4c1ac0'},
                'trial_id': arm + '-2', 'trial': {'id': arm + '-2', 'method': arm}},
                'settings': {'seed': seed}}
            assert verify_student_roster_contract(f'{arm}-seed{seed}', contract) == contract
            for key in ('trial_id', 'revision', 'seed'):
                wrong = deepcopy(contract)
                if key == 'trial_id':
                    wrong['identity'][key] = arm + '-1'
                elif key == 'revision':
                    wrong['identity']['model'][key] = '0' * 40
                else:
                    wrong['settings'][key] = seed + 1
                with pytest.raises(ValueError, match='student roster contract'):
                    verify_student_roster_contract(f'{arm}-seed{seed}', wrong)


def test_liquid_cohort_binds_exact_teacher_provenance():
    from src.final_evaluation import verify_student_cohort_binding, verify_teacher_cohort_binding
    provenance = {'base_config_hash': 'a' * 64, 'prepared_fingerprint': 'b' * 64,
                  'raw_manifest_hash': 'c' * 64, 'teacher': {
                      'best_step': 3000, 'contract_hash': 'd' * 64,
                      'manifest_sha256': 'e' * 64, 'model': {
                          'id': 'LiquidAI/LFM2.5-24B-A2B',
                          'revision': '3fcfdbfa1a1b6d8166846a6323e69f1e1254440d'}}}
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


def test_liquid_roster_accepts_only_exact_pinned_family(tmp_path):
    models = []
    for key in ['teacher', 'pretrained-student'] + [
            f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]:
        teacher = key == 'teacher'
        manifest = tmp_path.resolve() / (key + '.json')
        manifest.write_bytes(b'{}')
        models.append({'id': key,
            'model_id': 'LiquidAI/LFM2.5-24B-A2B' if teacher else 'LiquidAI/LFM2.5-2.6B',
            'model_revision': ('3fcfdbfa1a1b6d8166846a6323e69f1e1254440d' if teacher
                               else '654f9463ce32b05d0429d76fe1f580b27d4c1ac0'),
            'checkpoint': None if key == 'pretrained-student' else {
                'manifest_path': str(manifest),
                'manifest_sha256': hashlib.sha256(b'{}').hexdigest()}})
    protocol = {'version': 1, 'family': 'lfm25', 'models': models}
    assert json.loads(freeze_final_protocol(protocol)) == protocol
    wrong = deepcopy(protocol)
    wrong['models'][0]['model_revision'] = '0' * 40
    with pytest.raises(ValueError, match='model identity'):
        freeze_final_protocol(wrong)
