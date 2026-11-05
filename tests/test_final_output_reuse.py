"""Completed outputs must bind the frozen protocol before reuse."""
import hashlib
import json
import pytest
from src import final_evaluation


def test_filesystem_reuse_retains_exact_completed_bytes_without_modification(tmp_path):
    protocol = b'frozen-protocol'
    output = json.dumps({
        'version': 1, 'status': 'completed', 'model_id': 'seed42-ce',
        'protocol_sha256': hashlib.sha256(protocol).hexdigest(),
        'statistics': {'ce_sum': 10.0, 'tokens': 5},
    }).encode()
    path = tmp_path / 'completed.json'
    path.write_bytes(output)
    before = path.stat()
    retained, statistics = final_evaluation.read_completed_evaluation_output(
        str(path), protocol_bytes=protocol, expected_model_id='seed42-ce')
    assert retained == output
    assert statistics['tokens'] == 5
    assert statistics['ce'] == 2.0
    assert path.read_bytes() == output
    assert path.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize('kind', ['symlink', 'parent_symlink', 'directory', 'fifo',
                                  'oversize', 'partial', 'foreign', 'missing'])
def test_filesystem_reuse_fails_closed_without_replacing_existing_result(tmp_path, kind):
    import os
    protocol = b'frozen-protocol'
    path = tmp_path / 'result.json'
    if kind == 'symlink':
        target = tmp_path / 'target'
        target.write_bytes(b'{}')
        path.symlink_to(target)
    elif kind == 'parent_symlink':
        target = tmp_path / 'target'
        target.mkdir()
        (target / 'result.json').write_bytes(b'{}')
        link = tmp_path / 'link'
        link.symlink_to(target, target_is_directory=True)
        path = link / 'result.json'
    elif kind == 'directory':
        path.mkdir()
    elif kind == 'fifo':
        os.mkfifo(path)
    elif kind == 'oversize':
        path.write_bytes(b' ' * 65537)
    elif kind == 'partial':
        path.write_bytes(b'{')
    elif kind == 'foreign':
        path.write_bytes(json.dumps({
            'version': 1, 'status': 'completed', 'model_id': 'foreign',
            'protocol_sha256': hashlib.sha256(protocol).hexdigest(),
            'statistics': {'ce_sum': 10., 'tokens': 5},
        }).encode())
    existed = path.exists()
    before = path.lstat() if existed else None
    with pytest.raises(ValueError, match='completed evaluation output'):
        final_evaluation.read_completed_evaluation_output(
            str(path), protocol_bytes=protocol, expected_model_id='seed42-ce')
    assert path.exists() == existed
    if existed:
        assert path.lstat().st_ino == before.st_ino
        assert path.lstat().st_mtime_ns == before.st_mtime_ns


def test_filesystem_reuse_closes_descriptor_when_fstat_fails(tmp_path, monkeypatch):
    import errno
    import os
    path = tmp_path / 'result.json'
    path.write_bytes(b'{}')
    opened = []
    original_open = os.open
    original_fstat = os.fstat
    def record_open(name, flags, *args, **kwargs):
        descriptor = original_open(name, flags, *args, **kwargs)
        if name == 'result.json':
            opened.append(descriptor)
        return descriptor
    def fail_fstat(descriptor):
        # Inject only into this test's result FD, not pytest's capture streams.
        if descriptor in opened:
            raise OSError(errno.EIO, 'injected fstat failure')
        return original_fstat(descriptor)
    monkeypatch.setattr(final_evaluation.os, 'open', record_open)
    monkeypatch.setattr(final_evaluation.os, 'fstat', fail_fstat)
    try:
        with pytest.raises(OSError, match='injected fstat failure'):
            final_evaluation.read_completed_evaluation_output(
                str(path), protocol_bytes=b'protocol', expected_model_id='seed42-ce')
        assert len(opened) == 1
        with pytest.raises(OSError) as closed:
            original_fstat(opened[0])
        assert closed.value.errno == errno.EBADF
    finally:
        for descriptor in opened:
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_completed_output_reuse_rejects_duplicate_fields_at_every_depth():
    protocol = b'protocol-v1'
    digest = hashlib.sha256(protocol).hexdigest()
    prefix = ('{"version":1,"status":"completed","model_id":"seed42-ce",'
              '"protocol_sha256":"' + digest + '",')
    for suffix in (
        '"statistics":{"ce_sum":999,"ce_sum":10,"tokens":5}}',
        '"status":"running","status":"completed",'
        '"statistics":{"ce_sum":10,"tokens":5}}',
    ):
        with pytest.raises(ValueError, match='completed evaluation output'):
            final_evaluation.validate_completed_evaluation_output(
                (prefix + suffix).encode(), protocol_bytes=protocol,
                expected_model_id='seed42-ce')


@pytest.mark.parametrize('change', [
    {'status': 'running'}, {'version': True}, {'statistics': None},
    {'statistics': {}}, {'statistics': {'ce_sum': 1., 'tokens': 0}},
    {'statistics': {'ce_sum': float('nan'), 'tokens': 5}},
    {'statistics': {'ce_sum': 10., 'tokens': True}},
    {'statistics': {'ce_sum': -1., 'tokens': 5}},
    {'statistics': {'ce_sum': 10., 'tokens': 5, 'extra': 1}},
])
def test_completed_output_reuse_rejects_partial_or_malformed_results(change):
    protocol = b'protocol-v1'
    record = {'version': 1, 'status': 'completed', 'model_id': 'seed42-ce',
              'protocol_sha256': hashlib.sha256(protocol).hexdigest(),
              'statistics': {'ce_sum': 10., 'tokens': 5}}
    record.update(change)
    with pytest.raises(ValueError, match='completed evaluation output'):
        final_evaluation.validate_completed_evaluation_output(
            json.dumps(record).encode(), protocol_bytes=protocol,
            expected_model_id='seed42-ce')


@pytest.mark.parametrize('output', [b'', b'{', b'\xff', b' ' * 65537])
def test_completed_output_reuse_rejects_invalid_or_oversize_bytes(output):
    with pytest.raises(ValueError, match='completed evaluation output'):
        final_evaluation.validate_completed_evaluation_output(
            output, protocol_bytes=b'protocol-v1', expected_model_id='seed42-ce')


def test_completed_output_reuse_binds_protocol_and_exact_model():
    protocol = b'{"cohort":"frozen-v1"}'
    output = json.dumps({
        'version': 1, 'status': 'completed', 'model_id': 'seed42-ce',
        'protocol_sha256': hashlib.sha256(protocol).hexdigest(),
        'statistics': {'ce_sum': 10.0, 'tokens': 5},
    }).encode()
    reused = final_evaluation.validate_completed_evaluation_output(
        output, protocol_bytes=protocol, expected_model_id='seed42-ce')
    assert reused == {'ce_sum': 10.0, 'tokens': 5, 'ce': 2.0,
                      'ppl': pytest.approx(7.38905609893065)}
    with pytest.raises(ValueError, match='completed evaluation output'):
        final_evaluation.validate_completed_evaluation_output(
            output, protocol_bytes=b'changed', expected_model_id='seed42-ce')
    with pytest.raises(ValueError, match='completed evaluation output'):
        final_evaluation.validate_completed_evaluation_output(
            output, protocol_bytes=protocol, expected_model_id='seed43-ce')
