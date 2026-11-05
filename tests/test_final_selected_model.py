"""Concrete selected model path with real retained CPU checkpoint bytes."""
import hashlib
import io
import json
import torch
from src import final_evaluation as fe


def test_selected_model_restores_verified_state_before_freezing(tmp_path, monkeypatch):
    from src import research_models
    model = torch.nn.Linear(2, 1, dtype=torch.bfloat16)
    state = {name: torch.full_like(value, 2) for name, value in model.state_dict().items()}
    stream = io.BytesIO()
    torch.save({'model_state': state}, stream)
    payload = stream.getvalue()
    contract = {'identity': {'stage': 'ce',
                            'model': {'id': 'fixture', 'revision': 'a' * 40},
                            'adapter': {'rank': 8, 'alpha': 16, 'dropout': .05}},
                'settings': {'seed': 42}}
    manifest = json.dumps({'version': 1, 'step': 3000, 'contract': contract,
                          'identity': contract['identity'], 'files': {'state.pt': {
                              'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}}).encode()
    marker = json.dumps({'manifest_sha256': hashlib.sha256(manifest).hexdigest()}).encode()
    (tmp_path / 'state.pt').write_bytes(payload)
    calls = []
    # Only network/base loading and PEFT construction are replaced; commit,
    # deserialization, trainable/buffer restore and freezing are real.
    def base(spec, **kwargs):
        calls.append(('base', spec, kwargs))
        return model
    def adapter(value, config, *, seed):
        calls.append(('adapter', config, seed))
        return value
    monkeypatch.setattr(research_models, 'load_model', base)
    monkeypatch.setattr(research_models, 'attach_lora', adapter)
    loader = getattr(fe, 'load_final_selected_model', None)
    assert callable(loader), 'concrete selected model adapter missing'
    restored = loader(str(tmp_path), marker, manifest, expected_step=3000,
                      expected_contract=contract, max_bytes=len(payload),
                      model_spec={'id': 'fixture', 'revision': 'a' * 40},
                      adapter={'rank': 8, 'alpha': 16, 'dropout': .05}, seed=42, device='cpu')
    assert restored is model
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
    assert calls[0][2] == {'device': 'cpu', 'dtype': 'bfloat16', 'attention': 'sdpa'}
    assert calls[1][2] == 42


def test_selected_model_rejects_foreign_construction_before_payload(monkeypatch):
    import pytest
    contract = {'identity': {'stage': 'ce',
                            'model': {'id': 'fixture', 'revision': 'a' * 40},
                            'adapter': {'rank': 8, 'alpha': 16, 'dropout': .05}},
                'settings': {'seed': 42}}
    def forbidden(*args, **kwargs):
        pytest.fail('payload accessed before construction binding')
    monkeypatch.setattr(fe, 'load_selected_checkpoint', forbidden)
    original = dict(model_spec=contract['identity']['model'],
                    adapter=contract['identity']['adapter'], seed=42)
    for change in ({'model_spec': {'id': 'foreign', 'revision': 'a' * 40}},
                   {'adapter': {'rank': 16, 'alpha': 16, 'dropout': .05}},
                   {'seed': 43}):
        with pytest.raises(ValueError, match='construction contract mismatch'):
            fe.load_final_selected_model('/unread', b'', b'', expected_step=3000,
                                         expected_contract=contract, max_bytes=1,
                                         **{**original, **change})
