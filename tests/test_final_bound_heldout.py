"""Concrete manifest-to-tokenizer-to-packed-stream slice, synthetic raw only."""
import hashlib
import json
import sys
from types import SimpleNamespace
from src import final_evaluation as fe


def test_bound_heldout_loads_pinned_tokenizer_after_manifest_authentication(tmp_path, monkeypatch):
    raw = tmp_path / 'test.jsonl'
    raw.write_text(json.dumps({'text': 'a' * 1024}) + '\n')
    manifest = json.dumps({'files': {'test.jsonl': hashlib.sha256(raw.read_bytes()).hexdigest()}}).encode()
    spec = {'id': 'fixture/student', 'revision': 'a' * 40}
    calls = []
    class Tokenizer:
        eos_token_id = 3
        def encode(self, text, add_special_tokens=False):
            return [1] * len(text)
    def load(name, **kwargs):
        calls.append((name, kwargs))
        return Tokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    loader = getattr(fe, 'load_bound_final_heldout', None)
    assert callable(loader), 'manifest-bound concrete heldout loader missing'
    batches = loader(raw, raw_manifest_bytes=manifest,
                     expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                     tokenizer_spec=spec, max_bytes=4096, device='cpu')
    assert len(batches) == 2
    assert batches[0][0].shape == (1, 512)
    assert calls == [('fixture/student', {'revision': 'a' * 40, 'trust_remote_code': False, 'local_files_only': True})]
