import json
from pathlib import Path

import pytest


def test_packing_preserves_eos_boundaries_and_accounts_tail(tmp_path):
    from src.research_data import pack_split
    class Tokenizer:
        eos_token_id = 9
        def encode(self, text, add_special_tokens=False):
            return [int(x) for x in text.split()]
    path = tmp_path / 'raw.jsonl'
    path.write_text('\n'.join(json.dumps({'text': text}) for text in ['1 2', '3 4 5']))
    dataset, metadata = pack_split(path, Tokenizer(), 3)
    assert [x['input_ids'].tolist() for x in dataset] == [[1, 2, 9], [3, 4, 5]]
    assert all((x['labels'] == x['input_ids']).all() for x in dataset)
    assert metadata == {'documents': 2, 'total_tokens': 7, 'packed_tokens': 6, 'dropped_tail_tokens': 1, 'blocks': 2}
    with pytest.raises(ValueError, match='empty'):
        pack_split(path, Tokenizer(), 100)


def test_stream_is_bounded_and_pinned():
    from src.research_data import stream_source, DATASET_REVISION
    calls = []
    class Stream:
        def shuffle(self, **kw): calls.append(('shuffle', kw)); return self
        def take(self, n): calls.append(('take', n)); return iter([])
    def loader(*args, **kwargs):
        calls.append((args, kwargs)); return Stream()
    list(stream_source(12, loader=loader))
    assert calls[0] == (('HuggingFaceFW/fineweb-2', 'kor_Hang'), {'revision': DATASET_REVISION, 'split': 'train', 'streaming': True})
    assert calls[1:] == [('shuffle', {'seed': 42, 'buffer_size': 1000}), ('take', 12)]


def rows(n=30):
    return [{'id': str(i), 'text': ('안녕하세요 한국어 연구 문서입니다. ' * 25) + str(i)} for i in range(n)]


def test_exposed_pilot_documents_are_excluded_by_id_or_text_hash(tmp_path):
    import hashlib
    from src.research_data import prepare_raw, digest
    source = rows()
    exclusion = {'source_ids': ['0'], 'text_sha256': [hashlib.sha256(source[1]['text'].encode()).hexdigest()]}
    identity = {'revision': 'a' * 40, 'max_documents': 30, 'seed': 42,
                'exposure_exclusion': exclusion, 'exposure_exclusion_sha256': digest(exclusion)}
    manifest = prepare_raw(source, tmp_path, identity)
    records = [json.loads(line) for split in ('train', 'validation', 'test')
               for line in (tmp_path / f'{split}.jsonl').read_text().splitlines()]
    assert not any(r['source_id'] == '0' or r['text_sha256'] in exclusion['text_sha256'] for r in records)
    assert len(records) == 28
    assert manifest['rejected']['exposed'] == 2
    assert manifest['identity']['exposure_exclusion_sha256'] == digest(exclusion)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('policy,sha', [
    ({'source_ids': ['0'], 'text_sha256': []}, None),
    ({'source_ids': ['0'], 'text_sha256': []}, '0' * 64),
    ({'source_ids': '0', 'text_sha256': []}, 'computed'),
    ({'source_ids': [], 'text_sha256': ['invalid']}, 'computed'),
    ({'source_ids': [], 'text_sha256': [], 'unexpected': True}, 'computed'),
    (None, '0' * 64),
])
def test_exposure_policy_fails_closed_before_cache_reuse(tmp_path, cached, policy, sha):
    from src.research_data import prepare_raw, digest
    identity = {'revision': 'a' * 40, 'max_documents': 30, 'seed': 42,
                'exposure_exclusion': policy}
    if sha is not None:
        identity['exposure_exclusion_sha256'] = digest(policy) if sha == 'computed' else sha
    if cached:
        (tmp_path / 'manifest.json').write_text(json.dumps({'identity': identity, 'files': {}}))
    with pytest.raises(ValueError, match='exposure exclusion'):
        prepare_raw(rows(), tmp_path, identity)
    assert not (tmp_path / 'train.jsonl').exists()


@pytest.mark.parametrize('policy', [None, [], {'source_ids': [], 'text_sha256': ['bad']},
                                   {'source_ids': 'pilot', 'text_sha256': []},
                                   {'source_ids': [], 'text_sha256': [], 'extra': True}])
def test_malformed_exposure_registry_rejected_before_source_access(tmp_path, policy):
    from scripts.run_research import prepare
    accessed = []
    def source(_):
        accessed.append(True)
        raise RuntimeError('source was accessed before validation')
    config = {'data': {'max_documents': 30, 'exposure_exclusion': policy}}
    with pytest.raises(ValueError, match='exposure exclusion'):
        prepare(config, tmp_path, {'source': source})
    assert accessed == []
    assert not tmp_path.joinpath('raw').exists()


def test_frozen_split_deduplicates_filters_and_refuses_changed_identity(tmp_path):
    from src.research_data import prepare_raw
    source = rows() + [rows()[0], {'id': 'pii', 'text': rows()[0]['text'] + ' me@example.com'}]
    identity = {'revision': 'a' * 40, 'max_documents': 32, 'seed': 42}
    manifest = prepare_raw(source, tmp_path, identity)
    assert manifest['counts'] == {'train': 24, 'validation': 3, 'test': 3}
    assert manifest['rejected']['duplicate'] == 1
    assert manifest['rejected']['pii'] == 1
    hashes = []
    for split in ('train', 'validation', 'test'):
        hashes.extend(json.loads(x)['text_sha256'] for x in (tmp_path / f'{split}.jsonl').read_text().splitlines())
    assert len(hashes) == len(set(hashes)) == 30
    assert prepare_raw(iter([]), tmp_path, identity) == manifest
    with pytest.raises(ValueError, match='identity'):
        prepare_raw(iter([]), tmp_path, {**identity, 'seed': 1})
    (tmp_path / 'test.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='hash'):
        prepare_raw(iter([]), tmp_path, identity)
