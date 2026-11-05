"""Commit preflight binds retained marker and selected manifest bytes."""
import hashlib
import json
import unittest


class CommitBindingTests(unittest.TestCase):
    def test_requires_complete_and_selected_contract_together(self):
        import src.final_evaluation as evaluation
        verify = getattr(evaluation, 'bind_checkpoint_commit', None)
        self.assertTrue(callable(verify), 'integrated commit binding missing')
        contract = {'identity': {'stage': 'ce'}, 'settings': {'seed': 42}}
        manifest = {'version': 1, 'step': 3000, 'contract': contract,
                    'identity': contract['identity']}
        payload = json.dumps(manifest).encode()
        marker = json.dumps({'manifest_sha256': hashlib.sha256(payload).hexdigest()}).encode()
        self.assertEqual(verify(marker, payload, expected_step=3000,
                                expected_contract=contract), manifest)
        with self.assertRaisesRegex(ValueError, 'COMPLETE manifest SHA'):
            verify(b'{"manifest_sha256":"bad"}', payload,
                   expected_step=3000, expected_contract=contract)
        with self.assertRaisesRegex(ValueError, 'step/contract/identity'):
            verify(marker, payload, expected_step=2000, expected_contract=contract)
