"""Exposure declarations are metadata, not proof of historical non-access."""
import copy
import json
import pytest


@pytest.mark.parametrize('change', [
    {'cohort': 'different-cohort'}, {'raw_manifest_sha256': 'b' * 64},
    {'version': True}, {'version': 2}, {'extra': 'undeclared'},
    {'events': [{'split': 'test'}]}, {'events': None},
    {'raw_manifest_sha256': 'not-a-hash'},
])
def test_exposure_gate_rejects_unbound_or_prior_exposure(change):
    registry = {'version': 1, 'cohort': 'common-korean-v1',
                'raw_manifest_sha256': 'a' * 64, 'events': []}
    registry.update(change)
    with pytest.raises(ValueError, match='exposure registry'):
        final_evaluation.freeze_exposure_registry(
            registry, expected_cohort='common-korean-v1',
            expected_raw_manifest_sha256='a' * 64)

def test_exposure_gate_rejects_nonempty_list_with_false_truthiness():
    class FalseList(list):
        def __bool__(self):
            return False

    registry = {'version': 1, 'cohort': 'common-korean-v1',
                'raw_manifest_sha256': 'a' * 64,
                'events': FalseList([{'split': 'test'}])}
    with pytest.raises(ValueError, match='exposure registry'):
        final_evaluation.freeze_exposure_registry(
            registry, expected_cohort='common-korean-v1',
            expected_raw_manifest_sha256='a' * 64)


from src import final_evaluation


def test_freeze_exposure_registry_binds_cohort_without_mutation():
    registry = {'version': 1, 'cohort': 'common-korean-v1',
                'raw_manifest_sha256': 'a' * 64, 'events': []}
    before = copy.deepcopy(registry)
    frozen = final_evaluation.freeze_exposure_registry(
        registry, expected_cohort='common-korean-v1',
        expected_raw_manifest_sha256='a' * 64)
    assert json.loads(frozen) == registry
    assert registry == before
    assert isinstance(frozen, bytes)
