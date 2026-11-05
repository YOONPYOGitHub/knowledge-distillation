"""Adversarial regression for retained payload gates; never reads real data."""
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest

from src.final_evaluation import read_verified_checkpoint_payload, load_retained_checkpoint_payload


class PayloadAdversarialTests(unittest.TestCase):
    def fixture(self, root, payload=b'checkpoint'):
        (root / 'state.pt').write_bytes(payload)
        return {'files': {'state.pt': {'size': len(payload),
                'sha256': hashlib.sha256(payload).hexdigest()}}}

    def test_rejects_invalid_size_types_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for size in (True, 0, -1, 1.0, '10', 1025):
                manifest = self.fixture(root)
                manifest['files']['state.pt']['size'] = size
                with self.subTest(size=size), self.assertRaises(ValueError):
                    read_verified_checkpoint_payload(str(root), manifest, max_bytes=1024)

    def test_rejects_file_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = self.fixture(root)
            (root / 'state.pt').rename(root / 'real.pt')
            (root / 'state.pt').symlink_to(root / 'real.pt')
            with self.assertRaisesRegex(ValueError, 'non-symlink'):
                read_verified_checkpoint_payload(str(root), manifest, max_bytes=1024)

    def test_rejects_ancestor_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'real').mkdir()
            manifest = self.fixture(root / 'real')
            (root / 'alias').symlink_to(root / 'real', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'non-symlink'):
                read_verified_checkpoint_payload(str(root / 'alias'), manifest, max_bytes=1024)

    def test_rejects_fifo_without_waiting_for_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = self.fixture(root)
            (root / 'state.pt').unlink()
            os.mkfifo(root / 'state.pt')
            with self.assertRaisesRegex(ValueError, 'regular'):
                read_verified_checkpoint_payload(str(root), manifest, max_bytes=1024)

    def test_retained_bytes_survive_path_replacement_before_loading(self):
        import torch
        buffer = io.BytesIO()
        torch.save({'weight': torch.tensor([7])}, buffer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = self.fixture(root, buffer.getvalue())
            retained = read_verified_checkpoint_payload(str(root), manifest, max_bytes=8192)
            (root / 'state.pt').write_bytes(b'corrupt replacement')
            loaded = load_retained_checkpoint_payload(retained)
            self.assertEqual(loaded['weight'].tolist(), [7])

    def test_rejects_truncated_serialization(self):
        import torch
        buffer = io.BytesIO()
        torch.save({'weight': torch.tensor([7])}, buffer)
        with self.assertRaises(Exception):
            load_retained_checkpoint_payload(buffer.getvalue()[:50])

    def test_arbitrary_pickle_cannot_execute(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory).resolve() / 'must-not-exist'
            class Attack:
                def __reduce__(self):
                    return os.system, ('touch ' + str(marker),)
            buffer = io.BytesIO()
            torch.save(Attack(), buffer)
            with self.assertRaisesRegex(Exception, 'Weights only load failed'):
                load_retained_checkpoint_payload(buffer.getvalue())
            self.assertFalse(marker.exists())
