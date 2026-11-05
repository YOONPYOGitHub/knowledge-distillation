"""Sized authorized streams must be checked before reading beyond a budget."""
import pytest
from src import final_evaluation as fe


def test_complete_stream_budget_rejected_before_any_read():
    reads = []
    class Stream:
        def __len__(self):
            return 2
        def __iter__(self):
            reads.append('opened')
            yield object()
            yield object()
    traverse = getattr(fe, 'complete_evaluation_batches', None)
    assert callable(traverse), 'missing exact complete traversal'
    with pytest.raises(ValueError, match='batch budget'):
        list(traverse(Stream(), max_batches=1))
    assert reads == []


def test_complete_stream_reads_exact_declared_count_without_extra_next():
    class Stream:
        def __len__(self):
            return 1
        def __iter__(self):
            yield 'only'
            raise AssertionError('unauthorized extra next')
    assert list(fe.complete_evaluation_batches(Stream(), max_batches=1)) == ['only']


def test_complete_stream_rejects_truncated_declared_count():
    class Stream:
        def __len__(self):
            return 2
        def __iter__(self):
            yield 'only'
    with pytest.raises(ValueError, match='truncated'):
        list(fe.complete_evaluation_batches(Stream(), max_batches=2))
