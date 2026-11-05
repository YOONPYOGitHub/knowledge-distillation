"""Final evaluation preflight must not read heldout data."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


class FrozenCheckpointListTests(unittest.TestCase):
    def test_verifies_every_explicit_checkpoint_before_returning(self):
        self.assertIsNotNone(importlib.util.find_spec('src.final_evaluation'),
                             'explicit checkpoint-list preflight is missing')
        from src.final_evaluation import verify_checkpoint_list
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_bytes(b'{"step":3000}')
            entry = {'id': 'ce-seed42', 'manifest_path': str(path),
                     'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(verify_checkpoint_list([entry]), [entry])

    def test_rejects_empty_list_and_duplicate_or_missing_model_ids(self):
        from src.final_evaluation import verify_checkpoint_list
        for entries in ([], [{'manifest_path': 'not-read'}],
                        [{'id': 'same'}, {'id': 'same'}]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                verify_checkpoint_list(entries)

    def test_rejects_invalid_entry_schema_before_reading_any_file(self):
        from unittest.mock import patch
        from src.final_evaluation import verify_checkpoint_list
        valid = {'id': 'ce-seed42', 'manifest_path': '/unread/manifest.json',
                 'manifest_sha256': 'a' * 64}
        invalid = [
            {**valid, 'manifest_path': None},
            {**valid, 'manifest_path': 'relative/manifest.json'},
            {**valid, 'manifest_path': '/safe/../other/manifest.json'},
            {**valid, 'manifest_sha256': 'bad'},
            {**valid, 'manifest_sha256': 'A' * 64},
            {**valid, 'extra': True},
            {key: value for key, value in valid.items() if key != 'manifest_sha256'},
        ]
        for entry in invalid:
            with self.subTest(entry=entry), patch('src.final_evaluation.os.open', side_effect=AssertionError('unexpected file access')) as read:
                with self.assertRaisesRegex(ValueError, 'checkpoint entry'):
                    verify_checkpoint_list([valid, {**entry, 'id': 'kd-seed42'}])
                read.assert_not_called()

    def test_rejects_incomplete_expected_roster_before_manifest_access(self):
        from unittest.mock import patch
        import src.final_evaluation as evaluation
        with patch.object(Path, 'read_bytes') as read:
            verifier = getattr(evaluation, 'verify_model_roster', None)
            self.assertTrue(callable(verifier), 'exact model roster verifier missing')
            with self.assertRaisesRegex(ValueError, 'model roster mismatch'):
                verifier([{'id': 'ce-seed42'}],
                         ['ce-seed42', 'kd-seed42', 'teacher', 'pretrained-student'])
            read.assert_not_called()

    def test_roster_result_is_detached_from_caller_mutations(self):
        from src.final_evaluation import verify_model_roster
        entries = [{'id': 'teacher', 'identity': {'revision': 'frozen'}}]
        result = verify_model_roster(entries, ['teacher'])
        entries[0]['identity']['revision'] = 'changed'
        entries.append({'id': 'unexpected'})
        self.assertEqual(result, [{'id': 'teacher', 'identity': {'revision': 'frozen'}}])

    def test_roster_accepts_complete_ids_and_rejects_malformed_rosters(self):
        from src.final_evaluation import verify_model_roster
        self.assertEqual(verify_model_roster([{'id': 'kd'}, {'id': 'ce'}], ['ce', 'kd']),
                         [{'id': 'kd'}, {'id': 'ce'}])
        invalid = [([], ['ce']), ([{'id': 'ce'}, {'id': 'ce'}], ['ce']),
                   ([{'id': 'ce'}, {'id': 'extra'}], ['ce']),
                   ([None], ['ce']), ([{'id': ''}], ['ce']),
                   ([{'id': 1}], ['ce']), ([{'id': 'ce'}], ['ce', 'ce']),
                   ([{'id': 'ce'}], []), ([{'id': 'ce'}], [None]),
                   (None, ['ce']), ([{'id': 'ce'}], 'ce')]
        for entries, expected in invalid:
            with self.subTest(entries=entries, expected=expected), self.assertRaises(ValueError):
                verify_model_roster(entries, expected)

    def test_rejects_symlink_manifest_even_when_target_hash_matches(self):
        from src.final_evaluation import verify_checkpoint_list
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target.json'
            target.write_bytes(b'{"step":3000}')
            link = Path(directory) / 'manifest.json'
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'regular non-symlink'):
                verify_checkpoint_list([{'id': 'ce-seed42', 'manifest_path': str(link),
                    'manifest_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}])

    def test_rejects_symlink_parent_even_when_manifest_hash_matches(self):
        from src.final_evaluation import verify_checkpoint_list
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / 'real'
            target.mkdir()
            manifest = target / 'manifest.json'
            manifest.write_bytes(b'{"step":3000}')
            link = root / 'linked'
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'regular non-symlink'):
                verify_checkpoint_list([{'id': 'ce-seed42',
                    'manifest_path': str(link / 'manifest.json'),
                    'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}])

    def test_rejects_directory_manifest_as_nonregular(self):
        from src.final_evaluation import verify_checkpoint_list
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'regular non-symlink'):
                verify_checkpoint_list([{'id': 'ce-seed42', 'manifest_path': directory,
                                        'manifest_sha256': '0' * 64}])

    def test_rejects_fifo_manifest_without_waiting_for_writer(self):
        import json
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.fifo'
            os.mkfifo(path)
            entry = {'id': 'ce-seed42', 'manifest_path': str(path),
                     'manifest_sha256': '0' * 64}
            program = (
                'import json,sys; '
                'from src.final_evaluation import verify_checkpoint_list; '
                'verify_checkpoint_list(json.loads(sys.argv[1]))'
            )
            try:
                result = subprocess.run([sys.executable, '-c', program, json.dumps([entry])],
                                        capture_output=True, text=True, timeout=2)
            except subprocess.TimeoutExpired:
                self.fail('FIFO manifest blocked waiting for writer')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('checkpoint manifest must be regular non-symlink', result.stderr)

    def test_rejects_oversized_manifest_before_hashing(self):
        from src.final_evaluation import verify_checkpoint_list
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            payload = b'x' * (1024 * 1024 + 1)
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, 'manifest size'):
                verify_checkpoint_list([{'id': 'ce-seed42', 'manifest_path': str(path),
                    'manifest_sha256': hashlib.sha256(payload).hexdigest()}])

    def test_freezes_exact_eight_model_protocol_with_verified_checkpoints(self):
        import json
        import src.final_evaluation as evaluation
        freeze = getattr(evaluation, 'freeze_final_protocol', None)
        self.assertTrue(callable(freeze), 'integrated frozen protocol missing')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'manifest.json'
            path.write_bytes(b'{"step":3000}')
            checkpoint = {'manifest_path': str(path),
                          'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            ids = ['teacher', 'pretrained-student'] + [
                f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]
            protocol = {'version': 1, 'family': 'qwen35', 'models': [
                {'id': model_id,
                 'model_id': 'Qwen/Qwen3.5-9B' if model_id == 'teacher' else 'Qwen/Qwen3.5-4B',
                 'model_revision': ('c202236235762e1c871ad0ccb60c8ee5ba337b9a'
                                    if model_id == 'teacher'
                                    else '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
                 'checkpoint': None if model_id == 'pretrained-student' else dict(checkpoint)}
                for model_id in ids]}
            try:
                frozen = freeze(protocol)
            except ValueError as error:
                self.fail(f'official model identity binding is missing: {error}')
            self.assertIsInstance(frozen, bytes)
            self.assertEqual(json.loads(frozen), protocol)
            reversed_protocol = {**protocol, 'models': list(reversed(protocol['models']))}
            self.assertEqual(freeze(reversed_protocol), frozen)
            from copy import deepcopy
            for key, value in [('model_id', 'Qwen/Qwen3.5-4B'),
                               ('model_revision', 'f' * 40)]:
                wrong = deepcopy(protocol)
                wrong['models'][0][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'model identity mismatch'):
                    freeze(wrong)
            protocol['models'][0]['checkpoint']['manifest_sha256'] = '0' * 64
            self.assertEqual(json.loads(frozen)['models'][0]['checkpoint'], checkpoint)
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                freeze(protocol)
            with self.assertRaisesRegex(ValueError, 'model roster mismatch'):
                freeze({**protocol, 'models': protocol['models'][:-1]})

    def test_rejects_root_manifest_before_any_file_access(self):
        from unittest.mock import patch
        import src.final_evaluation as evaluation
        entries = [
            {'id': 'valid', 'manifest_path': '/unread/manifest.json',
             'manifest_sha256': 'a' * 64},
            {'id': 'invalid', 'manifest_path': '/', 'manifest_sha256': 'a' * 64}]
        with patch.object(evaluation.os, 'open', side_effect=AssertionError('unexpected file access')) as opened:
            with self.assertRaisesRegex(ValueError, 'manifest path invalid'):
                evaluation.verify_checkpoint_list(entries)
            opened.assert_not_called()

    def test_rejects_surrogate_path_before_any_file_access(self):
        from unittest.mock import patch
        import src.final_evaluation as evaluation
        entries = [
            {'id': 'valid', 'manifest_path': '/unread/manifest.json',
             'manifest_sha256': 'a' * 64},
            {'id': 'invalid', 'manifest_path': '/unread/\ud800/manifest.json',
             'manifest_sha256': 'a' * 64}]
        with patch.object(evaluation.os, 'open', side_effect=AssertionError('unexpected file access')) as opened:
            with self.assertRaisesRegex(ValueError, 'manifest path invalid'):
                evaluation.verify_checkpoint_list(entries)
            opened.assert_not_called()

    def test_retains_exact_verified_bytes_without_second_path_read(self):
        import src.final_evaluation as evaluation
        reader = getattr(evaluation, 'read_verified_checkpoint_manifests', None)
        self.assertTrue(callable(reader), 'verified manifest byte retention missing')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'manifest.json'
            payload = b'{"step":3000,"contract":{"seed":42}}'
            path.write_bytes(payload)
            entries = [{'id': 'ce-seed42', 'manifest_path': str(path),
                        'manifest_sha256': hashlib.sha256(payload).hexdigest()}]
            verified = reader(entries)
            path.write_bytes(b'changed after verification')
            self.assertEqual(verified, {'ce-seed42': payload})
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                reader(entries)

    def test_binds_retained_manifest_to_exact_selected_step_and_contract(self):
        import json
        from copy import deepcopy
        import src.final_evaluation as evaluation
        binder = getattr(evaluation, 'bind_checkpoint_manifest', None)
        self.assertTrue(callable(binder), 'selected manifest contract binding missing')
        contract = {'identity': {'model': {'id': 'frozen', 'revision': 'pinned'}},
                    'seed': 42, 'method': 'ce'}
        manifest = {'version': 1, 'step': 3000, 'contract': contract,
                    'identity': contract['identity'], 'files': {}}
        payload = json.dumps(manifest).encode()
        self.assertEqual(binder(payload, expected_step=3000, expected_contract=contract), manifest)
        for key, value in [('step', 2999), ('step', True), ('version', True),
                           ('contract', {**contract, 'seed': 43}), ('identity', {})]:
            changed = deepcopy(manifest)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'selected checkpoint'):
                binder(json.dumps(changed).encode(), expected_step=3000, expected_contract=contract)
        for invalid in (b'{"step":3000,"step":2999}', b'{"step":NaN}', b'[]', b'not-json'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'selected checkpoint'):
                binder(invalid, expected_step=3000, expected_contract=contract)
        contract['seed'] = 43
        self.assertEqual(json.loads(payload)['contract']['seed'], 42)

    def test_binding_rejects_boolean_integer_contract_alias(self):
        import json
        from src.final_evaluation import bind_checkpoint_manifest
        contract = {'identity': {'revision': 'pinned'}, 'settings': {'enabled': True}}
        manifest = {'version': 1, 'step': 3000,
                    'contract': {**contract, 'settings': {'enabled': 1}},
                    'identity': contract['identity']}
        with self.assertRaisesRegex(ValueError, 'selected checkpoint'):
            bind_checkpoint_manifest(json.dumps(manifest).encode(),
                                     expected_step=3000, expected_contract=contract)

    def test_binding_rejects_json_exponent_overflow(self):
        from src.final_evaluation import bind_checkpoint_manifest
        payload = (b'{"version":1,"step":0,"identity":{"revision":"pinned"},'
                   b'"contract":{"identity":{"revision":"pinned"},"x":1e999}}')
        with self.assertRaisesRegex(ValueError, 'selected checkpoint'):
            bind_checkpoint_manifest(payload, expected_step=0,
                expected_contract={'identity': {'revision': 'pinned'}, 'x': float('inf')})

    def test_binding_rejects_nested_array_with_controlled_error(self):
        import json
        from src.final_evaluation import bind_checkpoint_manifest
        payload = b'[' * 2000 + b'0' + b']' * 2000
        # Parser depth behavior is interpreter-dependent, not a schema limit.
        try:
            json.loads(payload)
        except RecursionError:
            expected_error = 'selected checkpoint JSON invalid'
        else:
            expected_error = 'selected checkpoint step/contract/identity mismatch'
        with self.assertRaisesRegex(ValueError, expected_error):
            bind_checkpoint_manifest(payload, expected_step=0,
                                     expected_contract={'identity': {'revision': 'pinned'}})

    def test_binding_normalizes_parser_recursion_error(self):
        from unittest.mock import patch
        from src.final_evaluation import bind_checkpoint_manifest
        # Inject only the parser failure; exercise the real boundary handler.
        with patch('json.loads', side_effect=RecursionError('parser depth')):
            with self.assertRaisesRegex(ValueError, 'selected checkpoint JSON invalid'):
                bind_checkpoint_manifest(b'{}', expected_step=0,
                    expected_contract={'identity': {'revision': 'pinned'}})

    def test_binding_requires_nonempty_frozen_identity(self):
        import json
        from src.final_evaluation import bind_checkpoint_manifest
        for contract in ({'seed': 42}, {'identity': {}}, {'identity': None}):
            manifest = {'version': 1, 'step': 3000, 'contract': contract,
                        'identity': contract.get('identity', {})}
            with self.subTest(contract=contract), self.assertRaisesRegex(ValueError, 'selected checkpoint'):
                bind_checkpoint_manifest(json.dumps(manifest).encode(),
                                         expected_step=3000, expected_contract=contract)

    def test_binding_rejects_oversized_json_before_parsing(self):
        from src.final_evaluation import bind_checkpoint_manifest
        with self.assertRaisesRegex(ValueError, 'selected checkpoint size'):
            bind_checkpoint_manifest(b' ' * (1024 * 1024 + 1), expected_step=3000,
                                     expected_contract={'identity': {'revision': 'pinned'}})

    def test_freezes_selected_roster_only_after_binding_every_retained_contract(self):
        import json
        from copy import deepcopy
        import src.final_evaluation as evaluation
        freeze = getattr(evaluation, 'freeze_selected_protocol', None)
        self.assertTrue(callable(freeze), 'selection-bound roster freeze missing')
        ids = ['teacher', 'pretrained-student'] + [
            f'{arm}-seed{seed}' for seed in (42, 43, 44) for arm in ('ce', 'kd')]
        with tempfile.TemporaryDirectory() as directory:
            models, selections = [], {}
            for model_id in ids:
                teacher = model_id == 'teacher'
                checkpoint = None
                if model_id != 'pretrained-student':
                    contract = {'identity': {
                        'model': {'id': 'Qwen/Qwen3.5-9B',
                                  'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'},
                        'stage': 'teacher'}, 'settings': {'seed': 42}}
                    if not teacher:
                        arm, seed = model_id.split('-seed')
                        contract = {'identity': {
                            'model': {'id': 'Qwen/Qwen3.5-4B',
                                      'revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'},
                            'stage': arm,
                            'trial_id': {'ce': 'ce-2', 'kd': 'kd-1'}[arm],
                            'trial': {'method': arm, 'id': {'ce': 'ce-2', 'kd': 'kd-1'}[arm]}},
                            'settings': {'seed': int(seed)}}
                    manifest = {'version': 1, 'step': 3000, 'contract': contract,
                                'identity': contract['identity']}
                    payload = json.dumps(manifest).encode()
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
            frozen = freeze(protocol, selections)
            self.assertEqual(json.loads(frozen)['selections'], selections)
            for model_id in selections:
                wrong = deepcopy(selections)
                wrong[model_id]['contract']['seed'] = 43
                with self.subTest(model_id=model_id), self.assertRaisesRegex(ValueError, 'selected checkpoint'):
                    freeze(protocol, wrong)
            for wrong in ({}, {**selections, 'unexpected': selections['teacher']}):
                with self.assertRaisesRegex(ValueError, 'selection roster'):
                    freeze(protocol, wrong)

    def test_complete_marker_rejects_nonfinite_json(self):
        from src.final_evaluation import verify_complete_marker
        manifest = b'{}'
        digest = hashlib.sha256(manifest).hexdigest()
        for value in ('NaN', 'Infinity', '-Infinity', '1e9999'):
            marker = ('{"manifest_sha256":"' + digest + '","extra":' + value + '}').encode()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'COMPLETE marker JSON invalid'):
                verify_complete_marker(marker, manifest)

    def test_complete_marker_rejects_duplicate_fields(self):
        from src.final_evaluation import verify_complete_marker
        manifest = b'{}'
        digest = hashlib.sha256(manifest).hexdigest()
        marker = ('{"manifest_sha256":"wrong","manifest_sha256":"' + digest + '"}').encode()
        with self.assertRaisesRegex(ValueError, 'COMPLETE marker JSON invalid'):
            verify_complete_marker(marker, manifest)

    def test_complete_marker_binds_actual_retained_manifest_bytes(self):
        import json
        import src.final_evaluation as evaluation
        verifier = getattr(evaluation, 'verify_complete_marker', None)
        self.assertTrue(callable(verifier), 'COMPLETE marker verification missing')
        manifest = b'{"version":1,"step":3000}'
        marker = json.dumps({'manifest_sha256': hashlib.sha256(manifest).hexdigest()}).encode()
        self.assertIsNone(verifier(marker, manifest))
        with self.assertRaisesRegex(ValueError, 'COMPLETE'):
            verifier(marker, manifest + b' ')

    def test_rejects_changed_manifest(self):
        from src.final_evaluation import verify_checkpoint_list
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                verify_checkpoint_list([{'id': 'ce-seed42', 'manifest_path': str(path),
                                         'manifest_sha256': '0' * 64}])


if __name__ == '__main__':
    unittest.main()
