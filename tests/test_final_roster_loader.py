"""Concrete model adapter for an already authenticated frozen roster."""
import torch
from src import final_evaluation as fe


def test_pretrained_roster_branch_loads_no_adapter_or_checkpoint(monkeypatch):
    from src import research_models
    entry = {'id': 'pretrained-student', 'model_id': 'Qwen/Qwen3.5-4B',
             'model_revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
             'checkpoint': None}
    model = torch.nn.Linear(2, 1, dtype=torch.bfloat16)
    calls = []
    def base(spec, **kwargs):
        calls.append((spec, kwargs))
        return model
    def forbidden(*args, **kwargs):
        raise AssertionError('pretrained branch read checkpoint or attached LoRA')
    monkeypatch.setattr(research_models, 'load_model', base)
    monkeypatch.setattr(research_models, 'attach_lora', forbidden)
    monkeypatch.setattr(fe, 'load_final_selected_model', forbidden)
    loader = getattr(fe, 'load_final_roster_model', None)
    assert callable(loader), 'concrete roster model adapter missing'
    assert loader(entry, {}, {}, device='cpu') is model
    assert calls == [({'id': entry['model_id'], 'revision': entry['model_revision']},
                      {'device': 'cpu', 'dtype': 'bfloat16', 'attention': 'sdpa'})]
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_selected_roster_dispatches_original_contract_and_retained_commit(monkeypatch):
    entry = {'id': 'ce-seed42', 'model_id': 'fixture',
             'model_revision': 'a' * 40, 'checkpoint': {
                 'manifest_path': '/frozen/generation/manifest.json',
                 'manifest_sha256': 'b' * 64}}
    contract = {'identity': {'model': {'id': 'fixture', 'revision': 'a' * 40},
                             'adapter': {'rank': 8}}, 'settings': {'seed': 42}}
    selection = {'step': 3000, 'contract': contract}
    import hashlib
    manifest = b'retained manifest bytes'
    entry['checkpoint']['manifest_sha256'] = hashlib.sha256(manifest).hexdigest()
    commit = {'marker_bytes': b'retained marker', 'manifest_bytes': manifest,
              'max_bytes': 1024}
    seen = []
    sentinel = object()
    def selected(*args, **kwargs):
        seen.append((args, kwargs))
        return sentinel
    monkeypatch.setattr(fe, 'load_final_selected_model', selected)
    assert fe.load_final_roster_model(entry, {entry['id']: selection},
                                     {entry['id']: commit}, device='cpu') is sentinel
    assert seen == [(('/frozen/generation', b'retained marker', manifest), {
        'expected_step': 3000, 'expected_contract': contract, 'max_bytes': 1024,
        'model_spec': contract['identity']['model'], 'adapter': {'rank': 8},
        'seed': 42, 'device': 'cpu'})]


def test_selected_roster_manifest_mismatch_never_dispatches(monkeypatch):
    import pytest
    entry = {'id': 'ce-seed42', 'checkpoint': {
        'manifest_path': '/frozen/generation/manifest.json',
        'manifest_sha256': '0' * 64}}
    def forbidden(*args, **kwargs):
        raise AssertionError('mismatched retained commit dispatched')
    monkeypatch.setattr(fe, 'load_final_selected_model', forbidden)
    with pytest.raises(ValueError, match='retained manifest mismatch'):
        fe.load_final_roster_model(entry, {'ce-seed42': {}},
                                  {'ce-seed42': {'manifest_bytes': b'substituted'}},
                                  device='cpu')


def test_pretrained_roster_checkpoint_rejected_before_base_load(monkeypatch):
    import pytest
    from src import research_models
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid pretrained entry loaded a model')
    monkeypatch.setattr(research_models, 'load_model', forbidden)
    with pytest.raises(ValueError, match='checkpoint must be absent'):
        fe.load_final_roster_model({'id': 'pretrained-student', 'checkpoint': {}},
                                  {}, {}, device='cpu')
