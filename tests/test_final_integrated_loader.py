"""Heldout-free integrated commit-to-CPU loading contract."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
import src.final_evaluation as evaluation


class IntegratedLoaderTests(unittest.TestCase):
    def test_selected_commit_loads_only_bound_payload(self):
        loader = getattr(evaluation, 'load_selected_checkpoint', None)
        self.assertTrue(callable(loader), 'integrated selected checkpoint loader missing')
        contract = {'identity': {'stage': 'ce'}, 'settings': {'seed': 42}}
        buffer = io.BytesIO()
        torch.save({'adapter': {'weight': torch.tensor([2.0])}}, buffer)
        payload = buffer.getvalue()
        manifest = json.dumps({'version': 1, 'step': 3000, 'contract': contract,
                              'identity': contract['identity'], 'files': {'state.pt': {
                                  'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}}).encode()
        marker = json.dumps({'manifest_sha256': hashlib.sha256(manifest).hexdigest()}).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'state.pt').write_bytes(payload)
            loaded = loader(str(root), marker, manifest, expected_step=3000,
                            expected_contract=contract, max_bytes=len(payload))
            self.assertEqual(loaded['adapter']['weight'].device.type, 'cpu')
            self.assertEqual(loaded['adapter']['weight'].item(), 2.0)
            # A failed selection must be rejected before even touching payload.
            (root / 'state.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'step/contract/identity'):
                loader(str(root), marker, manifest, expected_step=2999,
                       expected_contract=contract, max_bytes=len(payload))
