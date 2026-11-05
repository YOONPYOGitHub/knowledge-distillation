"""Real archived contracts: heldout-free roster identity regression."""
import json
from copy import deepcopy
from pathlib import Path
import unittest

from src import final_evaluation


class TeacherRosterContractTests(unittest.TestCase):
    def test_freeze_rejects_consistently_substituted_teacher_contract(self):
        import hashlib
        import tempfile
        root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
        recovered = json.loads((root / 'final-teacher-identity-20261026.json').read_text())
        archive = json.loads((root / 'round2-originals-transfer/originals.json').read_text())
        teacher_contract = {key: recovered[key] for key in ('identity', 'settings')}
        with tempfile.TemporaryDirectory() as directory:
            models, selections = [], {}
            for model_id in ['teacher', 'pretrained-student'] + [
                    f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]:
                teacher = model_id == 'teacher'
                checkpoint = None
                if model_id != 'pretrained-student':
                    if teacher:
                        contract = deepcopy(teacher_contract)
                        contract['identity']['stage'] = 'ce'
                    else:
                        arm, seed = model_id.split('-seed')
                        trial = {'ce': 'ce-2', 'kd': 'kd-1'}[arm]
                        contract = deepcopy(archive[f'trials/{trial}/result.json']['checkpoint_contract'])
                        contract['settings']['seed'] = int(seed)
                    payload = json.dumps({'version': 1, 'step': 3000,
                        'identity': contract['identity'], 'contract': contract}).encode()
                    path = Path(directory).resolve() / (model_id + '.json')
                    path.write_bytes(payload)
                    checkpoint = {'manifest_path': str(path),
                                  'manifest_sha256': hashlib.sha256(payload).hexdigest()}
                    selections[model_id] = {'step': 3000, 'contract': contract}
                models.append({'id': model_id,
                    'model_id': 'Qwen/Qwen3.5-9B' if teacher else 'Qwen/Qwen3.5-4B',
                    'model_revision': ('c202236235762e1c871ad0ccb60c8ee5ba337b9a' if teacher
                                       else '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
                    'checkpoint': checkpoint})
            protocol = {'version': 1, 'family': 'qwen35', 'models': models}
            with self.assertRaisesRegex(ValueError, 'teacher roster contract'):
                final_evaluation.freeze_selected_protocol(protocol, selections)
            # The same full roster must accept the genuine Teacher contract.
            payload = json.dumps({'version': 1, 'step': 3000,
                'identity': teacher_contract['identity'], 'contract': teacher_contract}).encode()
            path = Path(directory).resolve() / 'teacher.json'
            path.write_bytes(payload)
            models[0]['checkpoint']['manifest_sha256'] = hashlib.sha256(payload).hexdigest()
            selections['teacher']['contract'] = teacher_contract
            self.assertIsInstance(final_evaluation.freeze_selected_protocol(protocol, selections), bytes)

    def test_real_teacher_contract_rejects_model_stage_seed_substitution(self):
        verify = getattr(final_evaluation, 'verify_teacher_roster_contract', None)
        self.assertTrue(callable(verify), 'teacher roster contract verifier missing')
        root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
        recovered = json.loads((root / 'final-teacher-identity-20261026.json').read_text())
        contract = {key: recovered[key] for key in ('identity', 'settings')}
        self.assertEqual(verify(contract), contract)
        for field, value in (('model_id', 'Qwen/Qwen3.5-4B'),
                             ('revision', '0' * 40), ('stage', 'ce'),
                             ('seed', 43), ('seed', True)):
            wrong = deepcopy(contract)
            if field == 'model_id':
                wrong['identity']['model']['id'] = value
            elif field == 'revision':
                wrong['identity']['model']['revision'] = value
            elif field == 'stage':
                wrong['identity']['stage'] = value
            else:
                wrong['settings']['seed'] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, 'teacher roster contract'):
                    verify(wrong)


class StudentRosterContractTests(unittest.TestCase):
    def test_freeze_rejects_consistently_swapped_real_student_contract(self):
        import hashlib
        import tempfile
        root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
        archive = json.loads((root / 'round2-originals-transfer/originals.json').read_text())
        ce = archive['trials/ce-2/result.json']['checkpoint_contract']
        kd = archive['trials/kd-1/result.json']['checkpoint_contract']
        with tempfile.TemporaryDirectory() as directory:
            models, selections = [], {}
            for model_id in ['teacher', 'pretrained-student'] + [
                    f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]:
                teacher = model_id == 'teacher'
                checkpoint = None
                if model_id != 'pretrained-student':
                    recovered = json.loads((root / 'final-teacher-identity-20261026.json').read_text())
                    contract = {key: recovered[key] for key in ('identity', 'settings')} if teacher else deepcopy(
                        ce if model_id.startswith('ce') else kd)
                    if not teacher:
                        contract['settings']['seed'] = int(model_id[-2:])
                    # Attacker supplies matching manifest AND selection for wrong arm.
                    if model_id == 'ce-seed42':
                        contract = deepcopy(kd)
                    payload = json.dumps({'version': 1, 'step': 3000,
                        'identity': contract['identity'], 'contract': contract}).encode()
                    path = Path(directory).resolve() / (model_id + '.json')
                    path.write_bytes(payload)
                    checkpoint = {'manifest_path': str(path),
                                  'manifest_sha256': hashlib.sha256(payload).hexdigest()}
                    selections[model_id] = {'step': 3000, 'contract': contract}
                models.append({'id': model_id,
                    'model_id': 'Qwen/Qwen3.5-9B' if teacher else 'Qwen/Qwen3.5-4B',
                    'model_revision': ('c202236235762e1c871ad0ccb60c8ee5ba337b9a' if teacher
                                       else '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
                    'checkpoint': checkpoint})
            with self.assertRaisesRegex(ValueError, 'roster contract'):
                final_evaluation.freeze_selected_protocol(
                    {'version': 1, 'family': 'qwen35', 'models': models}, selections)

    def test_real_student_contract_rejects_unselected_trial_id(self):
        root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
        archive = json.loads((root / 'round2-originals-transfer/originals.json').read_text())
        for arm, selected, wrong_trial in (('ce', 'ce-2', 'ce-1'), ('kd', 'kd-1', 'kd-2')):
            contract = archive[f'trials/{selected}/result.json']['checkpoint_contract']
            for field in ('trial_id', 'trial.id', 'both'):
                wrong = deepcopy(contract)
                if field in ('trial_id', 'both'):
                    wrong['identity']['trial_id'] = wrong_trial
                if field in ('trial.id', 'both'):
                    wrong['identity']['trial']['id'] = wrong_trial
                with self.subTest(arm=arm, field=field):
                    with self.assertRaisesRegex(ValueError, 'roster contract'):
                        final_evaluation.verify_student_roster_contract(f'{arm}-seed42', wrong)

    def test_real_student_contracts_reject_cross_seed_and_arm_substitution(self):
        verify = getattr(final_evaluation, 'verify_student_roster_contract', None)
        self.assertTrue(callable(verify), 'student roster contract verifier missing')
        root = Path(__file__).resolve().parent / 'fixtures/generic-final-metadata'
        contracts = {}
        for seed, folder in ((42, 'round2-originals-transfer'),
                             (43, 'seed43-originals-transfer'),
                             (44, 'seed44-originals-transfer')):
            archive = json.loads((root / folder / 'originals.json').read_text())
            for arm, trial in (('ce', 'ce-2'), ('kd', 'kd-1')):
                model_id = f'{arm}-seed{seed}'
                contracts[model_id] = archive[f'trials/{trial}/result.json']['checkpoint_contract']
        for model_id, contract in contracts.items():
            with self.subTest(valid=model_id):
                self.assertEqual(verify(model_id, contract), contract)
            for other_id, other in contracts.items():
                if other_id != model_id:
                    with self.subTest(target=model_id, substituted=other_id):
                        with self.assertRaisesRegex(ValueError, 'roster contract'):
                            verify(model_id, other)
            wrong = deepcopy(contract)
            wrong['identity']['model']['revision'] = '0' * 40
            with self.assertRaisesRegex(ValueError, 'roster contract'):
                verify(model_id, wrong)
