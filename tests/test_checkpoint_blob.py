"""Offline tests of checkpoint generation commit and verified restore."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import checkpoint_blob as cb


def generation(root, name='step-00000001-test', step=1):
    p = root / name
    p.mkdir()
    state = b'known checkpoint bytes'
    (p/'state.pt').write_bytes(state)
    manifest = {'step': step, 'files': {'state.pt': {'size': len(state), 'sha256': hashlib.sha256(state).hexdigest()}}}
    (p/'manifest.json').write_text(json.dumps(manifest))
    (p/'COMPLETE').write_text(json.dumps({'manifest_sha256': cb.digest(p/'manifest.json')}))
    return p


class CheckpointBlobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = cb.LocalStore(self.root/'remote')

    def test_wrong_complete_manifest_hash_cannot_replace_remote_latest(self):
        first = generation(self.root)
        cb.publish(first, self.store)
        before = (self.root/'remote'/'latest.json').read_bytes()
        second = generation(self.root, 'step-00000002-test', 2)
        (second/'COMPLETE').write_text(json.dumps({'manifest_sha256': '0' * 64}))
        with self.assertRaisesRegex(ValueError, 'COMPLETE|manifest'):
            cb.publish(second, self.store)
        self.assertEqual((self.root/'remote'/'latest.json').read_bytes(), before)
        self.assertFalse((self.root/'remote'/'generations'/second.name).exists())

    def test_wrong_remote_complete_hash_refuses_restore_install(self):
        g = generation(self.root)
        cb.publish(g, self.store)
        before = (self.root/'remote'/'latest.json').read_bytes()
        marker = self.root/'remote'/'generations'/g.name/'COMPLETE'
        marker.write_text(json.dumps({'manifest_sha256': '0' * 64}))
        with self.assertRaisesRegex(ValueError, 'COMPLETE|manifest'):
            cb.restore(self.store, self.root/'restored')
        self.assertFalse((self.root/'restored').exists())
        self.assertEqual((self.root/'remote'/'latest.json').read_bytes(), before)

    def test_invalid_complete_json_refuses_publish_and_restore(self):
        g = generation(self.root)
        cb.publish(g, self.store)
        before = (self.root/'remote'/'latest.json').read_bytes()
        remote_marker = self.root/'remote'/'generations'/g.name/'COMPLETE'
        for marker in ('complete', '[]', 'null', '{}', '{"manifest_sha256": 42}'):
            with self.subTest(marker=marker):
                (g/'COMPLETE').write_text(marker)
                remote_marker.write_text(marker)
                with self.assertRaises(ValueError):
                    cb.publish(g, self.store)
                with self.assertRaises(ValueError):
                    cb.restore(self.store, self.root/'restored')
                self.assertFalse((self.root/'restored').exists())
                self.assertEqual((self.root/'remote'/'latest.json').read_bytes(), before)

    def test_publish_readback_and_restore(self):
        g = generation(self.root)
        result = cb.publish(g, self.store)
        self.assertEqual(result['step'], 1)
        restored = cb.restore(self.store, self.root/'restored')
        self.assertEqual((restored/'state.pt').read_bytes(), (g/'state.pt').read_bytes())
        self.assertTrue((restored/'COMPLETE').is_file())

    def test_failed_readback_does_not_replace_latest(self):
        first = generation(self.root)
        cb.publish(first, self.store)
        before = (self.root/'remote'/'latest.json').read_bytes()
        second = generation(self.root, 'step-00000002-test', 2)
        real_get = self.store.get
        def failing_get(key, destination):
            if 'step-00000002' in key:
                raise OSError('injected download failure')
            return real_get(key, destination)
        self.store.get = failing_get
        with self.assertRaises(OSError):
            cb.publish(second, self.store)
        self.assertEqual((self.root/'remote'/'latest.json').read_bytes(), before)
        self.assertFalse((self.root/'remote'/'generations'/second.name/'COMPLETE').exists())

    def test_corrupt_local_generation_never_publishes(self):
        g = generation(self.root)
        (g/'state.pt').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            cb.publish(g, self.store)
        self.assertFalse((self.root/'remote'/'latest.json').exists())

    def test_corrupt_remote_restore_does_not_publish_destination(self):
        g = generation(self.root)
        cb.publish(g, self.store)
        (self.root/'remote'/'generations'/g.name/'state.pt').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            cb.restore(self.store, self.root/'restored')
        self.assertFalse((self.root/'restored').exists())

    def test_reject_manifest_traversal_and_symlink(self):
        g = generation(self.root)
        manifest = json.loads((g/'manifest.json').read_text())
        manifest['files']['../escape'] = manifest['files']['state.pt']
        (g/'manifest.json').write_text(json.dumps(manifest))
        (g/'COMPLETE').write_text(json.dumps({'manifest_sha256': cb.digest(g/'manifest.json')}))
        with self.assertRaisesRegex(ValueError, 'Unsafe checkpoint path'):
            cb.publish(g, self.store)
        manifest['files'].pop('../escape')
        (g/'manifest.json').write_text(json.dumps(manifest))
        (g/'COMPLETE').write_text(json.dumps({'manifest_sha256': cb.digest(g/'manifest.json')}))
        state = g/'state.pt'
        target = self.root/'external'; target.write_bytes(state.read_bytes())
        state.unlink(); state.symlink_to(target)
        with self.assertRaises(ValueError):
            cb.publish(g, self.store)

    def test_late_old_generation_cannot_move_remote_latest_backwards(self):
        old = generation(self.root)
        new = generation(self.root, 'step-00000002-test', 2)
        cb.publish(new, self.store)
        result = cb.publish(old, self.store)
        self.assertEqual(result['step'], 2)
        self.assertEqual(json.loads((self.root/'remote'/'latest.json').read_text())['step'], 2)

    def test_azure_download_streams_without_xattr_dependent_azcopy(self):
        import io
        from unittest.mock import patch
        store=cb.AzCopyStore('testaccount','kd-backups','test-prefix')
        destination=self.root/'download.pt'
        with patch('urllib.request.urlopen', side_effect=[io.BytesIO(b'{"access_token":"fixture-token"}'),io.BytesIO(b'payload')]), patch('subprocess.run',side_effect=AssertionError('download must not invoke azcopy')):
            store.get('state.pt',destination)
        self.assertEqual(destination.read_bytes(),b'payload')

    def test_stage_restore_includes_previous_best_generation(self):
        from src.resume_checkpoint import save_checkpoint, load_checkpoint
        stage=self.root/'stage'
        contract={'identity':{'run':'fixture'}}
        best=save_checkpoint(stage,{'step':0,'model_state':{},'best_checkpoint':None},contract)
        latest=save_checkpoint(stage,{'step':2,'model_state':{},'best_checkpoint':str(Path(best).relative_to(stage))},contract)
        pointer=cb.publish_stage(stage,self.store)
        self.assertEqual(pointer['step'],2)
        restored=cb.restore_stage(self.store,self.root/'restored-stage')
        state=load_checkpoint(restored,expected_contract=contract)
        self.assertEqual(state['step'],2)
        self.assertEqual(load_checkpoint(restored/state['best_checkpoint'],expected_contract=contract)['step'],0)
        self.assertEqual(Path(latest).name,Path(restored/json.loads((restored/'latest.json').read_text())['checkpoint']).name)

    def test_restore_refuses_existing_destination(self):
        cb.publish(generation(self.root), self.store)
        destination=self.root/'existing'; destination.mkdir()
        with self.assertRaises(FileExistsError):
            cb.restore(self.store, destination)

if __name__ == '__main__':
    unittest.main()
