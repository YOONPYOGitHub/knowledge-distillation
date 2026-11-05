"""Payload preflight never deserializes or opens heldout data."""
import hashlib
import tempfile
from pathlib import Path
import unittest


class PayloadTests(unittest.TestCase):
    def test_loads_retained_tensor_bytes_with_weights_only_on_cpu(self):
        import io
        import torch
        import src.final_evaluation as evaluation
        loader = getattr(evaluation, 'load_retained_checkpoint_payload', None)
        self.assertTrue(callable(loader), 'safe retained-byte loader missing')
        buffer = io.BytesIO()
        torch.save({'adapter': {'weight': torch.tensor([1.0, 2.0])}}, buffer)
        loaded = loader(buffer.getvalue())
        self.assertEqual(loaded['adapter']['weight'].device.type, 'cpu')
        self.assertTrue(torch.equal(loaded['adapter']['weight'], torch.tensor([1.0, 2.0])))

    def test_retains_size_and_sha_verified_payload_bytes(self):
        import src.final_evaluation as evaluation
        verify = getattr(evaluation, 'read_verified_checkpoint_payload', None)
        self.assertTrue(callable(verify), 'bounded retained payload gate missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            payload = b'trusted-checkpoint-bytes'
            (root / 'state.pt').write_bytes(payload)
            manifest = {'files': {'state.pt': {
                'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}}
            self.assertEqual(verify(str(root), manifest, max_bytes=1024), payload)
            (root / 'state.pt').write_bytes(b'x' * len(payload))
            with self.assertRaisesRegex(ValueError, 'payload hash/size'):
                verify(str(root), manifest, max_bytes=1024)
