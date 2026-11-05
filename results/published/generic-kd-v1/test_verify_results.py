"""Authored validator fixtures only: these are NOT experimental results."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

MODULE = Path(__file__).with_name('verify_results.py')


def load_verifier():
    if not MODULE.is_file():
        raise AssertionError('missing aggregate verifier implementation')
    spec = importlib.util.spec_from_file_location('public_kd_verifier', MODULE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BundleTests(unittest.TestCase):
    def copied_bundle(self, base):
        """Copy only declared PUBLIC leaves. No original provenance paths opened."""
        original_root = MODULE.resolve().parents[3]
        manifest = json.loads(MODULE.with_name('manifest.json').read_bytes())
        for entry in manifest['files']:
            source = original_root / entry['path']
            target = base / entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        directory = base / 'results/published/generic-kd-v1'
        shutil.copyfile(MODULE.with_name('manifest.json'), directory / 'manifest.json')
        return directory

    def repin_fixture(self, directory, changed):
        """Test-only repin to exercise arithmetic beyond the outer hash check."""
        target = directory / 'manifest.json'
        manifest = json.loads(target.read_bytes())
        relative = changed.relative_to(directory.parents[2]).as_posix()
        entry = next(x for x in manifest['files'] if x['path'] == relative)
        data = changed.read_bytes()
        entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        target.write_text(json.dumps(manifest), encoding='utf-8')

    def test_isolated_portable_cli_is_read_only(self):
        with tempfile.TemporaryDirectory(prefix='.authored-public-copy-', dir=MODULE.parent) as tmp:
            directory = self.copied_bundle(Path(tmp) / 'isolated')
            before = {p.relative_to(Path(tmp)).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(tmp).rglob('*') if p.is_file()}
            process = subprocess.run([sys.executable, '-B', str(directory / 'verify_results.py')],
                                     cwd=tmp, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)['pairs'], 9)
            after = {p.relative_to(Path(tmp)).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(tmp).rglob('*') if p.is_file()}
            self.assertEqual(before, after)

    def test_authored_corrupt_publication_copies_are_rejected(self):
        mod = load_verifier()
        for case in ('byte_hash', 'csv_ppl', 'paired_effect', 'duplicate_pair',
                     'outside_path', 'duplicate_inventory', 'missing_result', 'extra_file'):
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                    prefix='.authored-invalid-copy-', dir=MODULE.parent) as tmp:
                directory = self.copied_bundle(Path(tmp) / 'isolated')
                if case == 'byte_hash':
                    target = directory / 'final/qwen35--ce-seed42.json'
                    target.write_bytes(target.read_bytes() + b' ')
                elif case in ('csv_ppl', 'paired_effect', 'duplicate_pair'):
                    name = 'results.csv' if case == 'csv_ppl' else 'paired_effects.csv'
                    target = directory / name
                    with target.open(newline='', encoding='utf-8') as f:
                        records = list(csv.DictReader(f))
                    if case == 'duplicate_pair':
                        records.append(dict(records[0]))
                    else:
                        field = 'ppl' if case == 'csv_ppl' else 'kd_relative_ppl_change_percent'
                        records[0][field] = '999'
                    with target.open('w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=list(records[0]))
                        writer.writeheader()
                        writer.writerows(records)
                    self.repin_fixture(directory, target)
                elif case in ('outside_path', 'duplicate_inventory'):
                    target = directory / 'manifest.json'
                    manifest = json.loads(target.read_bytes())
                    if case == 'outside_path':
                        manifest['files'][0]['path'] = '../not-public.json'
                    else:
                        manifest['files'].append(dict(manifest['files'][0]))
                    target.write_text(json.dumps(manifest), encoding='utf-8')
                elif case == 'missing_result':
                    (directory / 'final/qwen35--ce-seed42.json').unlink()
                else:
                    (directory / 'unlisted-fixture.txt').write_text('authored invalid fixture')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    mod.verify(directory)

    def test_completed_publication(self):
        mod = load_verifier()
        self.assertTrue(callable(getattr(mod, 'verify', None)), 'missing bundle verification')
        result = mod.verify(MODULE.parent)
        self.assertEqual(result['results'], 24)
        self.assertEqual(result['pairs'], 9)
        self.assertEqual(result['selected_metadata'], 21)


class AggregateTests(unittest.TestCase):
    def test_authored_fixture_metrics_and_rejections(self):
        mod = load_verifier()
        fixture = {'version': 1, 'model_id': 'authored-fixture',
                   'status': 'completed', 'protocol_sha256': 'a' * 64,
                   'statistics': {'ce_sum': 20.0, 'tokens': 10}}
        self.assertEqual(mod.validate_result(fixture, 'a' * 64), (2.0, math.exp(2)))
        for changed in [dict(fixture, version=True),
                        dict(fixture, status='running'),
                        dict(fixture, protocol_sha256='b' * 64),
                        dict(fixture, statistics={'ce_sum': 20, 'tokens': 0}),
                        dict(fixture, statistics={'ce_sum': float('nan'), 'tokens': 10}),
                        dict(fixture, input_ids=[1, 2])]:
            with self.assertRaises(ValueError):
                mod.validate_result(changed, 'a' * 64)


if __name__ == '__main__':
    unittest.main()
