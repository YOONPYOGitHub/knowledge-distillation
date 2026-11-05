"""Completed results are exclusively published and reusable, never overwritten."""
import hashlib
import json
import pytest
from src import final_evaluation


def test_completed_publication_is_exclusive_and_validated(tmp_path):
    path = str(tmp_path / 'ce.json')
    protocol = b'{"models":[{"id":"ce"}]}'
    publish = getattr(final_evaluation, 'publish_completed_evaluation_output', None)
    assert callable(publish), 'missing exclusive completed-result publication'
    retained = publish(path, protocol_bytes=protocol, model_id='ce',
                       statistics={'ce_sum': 10., 'tokens': 5})
    saved, statistics = final_evaluation.read_completed_evaluation_output(
        path, protocol_bytes=protocol, expected_model_id='ce')
    assert retained == saved
    assert statistics['ce'] == 2.
    assert json.loads(saved)['protocol_sha256'] == hashlib.sha256(protocol).hexdigest()
    with pytest.raises(FileExistsError):
        publish(path, protocol_bytes=protocol, model_id='ce',
                statistics={'ce_sum': 11., 'tokens': 5})
    assert open(path, 'rb').read() == saved


def test_publication_write_failure_leaves_no_partial_result(tmp_path, monkeypatch):
    path = tmp_path / 'ce.json'
    def failed_fsync(descriptor):
        raise OSError('injected durability failure')
    monkeypatch.setattr(final_evaluation.os, 'fsync', failed_fsync)
    with pytest.raises(OSError, match='durability failure'):
        final_evaluation.publish_completed_evaluation_output(
            str(path), protocol_bytes=b'{}', model_id='ce',
            statistics={'ce_sum': 10., 'tokens': 5})
    assert list(tmp_path.iterdir()) == []


def test_publication_rejects_symlink_parent_without_touching_target(tmp_path):
    target = tmp_path / 'target'
    target.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        final_evaluation.publish_completed_evaluation_output(
            str(link / 'ce.json'), protocol_bytes=b'{}', model_id='ce',
            statistics={'ce_sum': 10., 'tokens': 5})
    assert list(target.iterdir()) == []


def test_publication_never_replaces_symlink_destination(tmp_path):
    original = tmp_path / 'original'
    original.write_bytes(b'untouched')
    destination = tmp_path / 'ce.json'
    destination.symlink_to(original)
    with pytest.raises(FileExistsError):
        final_evaluation.publish_completed_evaluation_output(
            str(destination), protocol_bytes=b'{}', model_id='ce',
            statistics={'ce_sum': 10., 'tokens': 5})
    assert destination.is_symlink()
    assert original.read_bytes() == b'untouched'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['ce.json', 'original']
