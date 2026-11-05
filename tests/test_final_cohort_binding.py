"""Heldout-free binding against archived common-corpus contracts."""


def test_teacher_cohort_binding_rejects_substituted_prepared_fingerprint():
    import pytest
    root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
    archive = json.loads((root / 'round2-originals-transfer/originals.json').read_text())
    cohort = {'data': deepcopy(archive['trials/ce-2/result.json']['checkpoint_contract']['data']),
              'provenance': deepcopy(archive['plan.json']['provenance'])}
    teacher = json.loads((root / 'final-teacher-identity-20261026.json').read_text())
    identity = teacher['identity']
    verify = getattr(final_evaluation, 'verify_teacher_cohort_binding', None)
    assert callable(verify), 'teacher cohort dependency verifier missing'
    assert verify(identity, cohort) == identity
    for invalid in (None, True, '', '0' * 64, 'g' * 64):
        wrong = deepcopy(identity)
        wrong['data_fingerprint'] = invalid
        with pytest.raises(ValueError, match='cohort dependency'):
            verify(wrong, cohort)

import json
from copy import deepcopy
from pathlib import Path
from src import final_evaluation


def test_student_cohort_binding_rejects_substituted_dependencies():
    root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
    archive = json.loads((root / 'round2-originals-transfer/originals.json').read_text())
    contract = archive['trials/ce-2/result.json']['checkpoint_contract']
    cohort = {'data': deepcopy(contract['data']),
              'provenance': deepcopy(archive['plan.json']['provenance'])}
    verify = getattr(final_evaluation, 'verify_student_cohort_binding', None)
    assert callable(verify), 'student cohort dependency verifier missing'
    assert verify(contract, cohort) == contract
    paths = [('data', 'train'), ('data', 'validation'),
             ('identity', 'provenance', 'raw_manifest_hash'),
             ('identity', 'provenance', 'prepared_fingerprint'),
             ('identity', 'provenance', 'teacher', 'manifest_sha256'),
             ('identity', 'provenance', 'teacher', 'contract_hash'),
             ('identity', 'provenance', 'teacher', 'best_step'),
             ('identity', 'provenance', 'teacher', 'model', 'revision')]
    import pytest
    for path in paths:
        wrong = deepcopy(contract)
        target = wrong
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = True if path[-1] == 'best_step' else '0' * 64
        with pytest.raises(ValueError, match='cohort dependency'):
            verify(wrong, cohort)
    for malformed in (None, {}, {'data': {}, 'provenance': {}}):
        with pytest.raises(ValueError, match='cohort dependency'):
            verify(contract, malformed)


def test_student_cohort_binding_rejects_matching_malformed_teacher_identity():
    import pytest
    root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
    archive = json.loads((root / 'round2-originals-transfer/originals.json').read_text())
    original = archive['trials/ce-2/result.json']['checkpoint_contract']
    mutations = [
        ('best_step', True), ('best_step', -1), ('best_step', 3000.0),
        ('best_step', None), ('model', {}),
        ('model', {'id': 'Other/Teacher', 'revision': 'a' * 40}),
        ('model', {'id': 'Qwen/Qwen3.5-9B', 'revision': 'a' * 40}),
        ('unexpected', 'extra'),
    ]
    for field, value in mutations:
        contract = deepcopy(original)
        cohort = {'data': contract['data'], 'provenance': contract['identity']['provenance']}
        cohort['provenance']['teacher'][field] = value
        with pytest.raises(ValueError, match='cohort dependency'):
            final_evaluation.verify_student_cohort_binding(contract, cohort)


def test_student_cohort_binding_rejects_matching_malformed_frozen_hashes():
    import pytest
    root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
    archive = json.loads((root / 'round2-originals-transfer/originals.json').read_text())
    original = archive['trials/ce-2/result.json']['checkpoint_contract']
    paths = [('data', 'train'), ('data', 'validation'),
             ('provenance', 'base_config_hash'), ('provenance', 'prepared_fingerprint'),
             ('provenance', 'raw_manifest_hash'),
             ('provenance', 'teacher', 'contract_hash'),
             ('provenance', 'teacher', 'manifest_sha256')]
    for path in paths:
        for invalid in (None, True, 17, '', 'a' * 63, 'g' * 64, 'A' * 64):
            contract = deepcopy(original)
            cohort = {'data': contract['data'], 'provenance': contract['identity']['provenance']}
            target = cohort
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = invalid
            with pytest.raises(ValueError, match='cohort dependency'):
                final_evaluation.verify_student_cohort_binding(contract, cohort)
