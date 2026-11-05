"""Real retained-byte packing fixture; never uses campaign heldout."""
import hashlib
import json
import torch
from src import final_evaluation as fe
from src.research_data import pack_split


def test_authenticated_heldout_adapter_matches_existing_complete_packing(tmp_path):
    path = tmp_path / 'test.jsonl'
    path.write_text(''.join(json.dumps({'text': text}) + '\n' for text in ['a' * 600, 'b' * 600]))
    class Tokenizer:
        eos_token_id = 3
        def encode(self, text, add_special_tokens=False):
            return [1 if letter == 'a' else 2 for letter in text]
    tokenizer = Tokenizer()
    expected, metadata = pack_split(path, tokenizer, 512)
    loader = getattr(fe, 'load_final_heldout_batches', None)
    assert callable(loader), 'concrete authenticated heldout adapter missing'
    batches = loader(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tokenizer=tokenizer, max_bytes=4096)
    assert len(batches) == metadata['blocks'] == 2
    for (ids, labels), block in zip(batches, expected):
        assert ids.shape == labels.shape == (1, 512)
        assert torch.equal(ids[0], block['input_ids'])
        assert torch.equal(labels[0], block['labels'])


def test_heldout_adapter_places_complete_batches_on_requested_device(tmp_path):
    path = tmp_path / 'test.jsonl'
    path.write_text(json.dumps({'text': 'a' * 1024}) + '\n')
    class Tokenizer:
        eos_token_id = 3
        def encode(self, text, add_special_tokens=False):
            return [1] * len(text)
    batches = fe.load_final_heldout_batches(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        tokenizer=Tokenizer(), max_bytes=4096, device='meta')
    assert len(batches) == 2
    assert all(ids.device.type == labels.device.type == 'meta' for ids, labels in batches)


def test_heldout_sha_mismatch_never_accesses_tokenizer(tmp_path):
    import pytest
    path = tmp_path / 'test.jsonl'
    path.write_text(json.dumps({'text': 'a' * 1024}) + '\n')
    class ForbiddenTokenizer:
        @property
        def eos_token_id(self):
            raise AssertionError('unauthenticated tokenizer access')
    with pytest.raises(ValueError, match='SHA/size mismatch'):
        fe.load_final_heldout_batches(path, expected_sha256='0' * 64,
                                      tokenizer=ForbiddenTokenizer(), max_bytes=4096)
