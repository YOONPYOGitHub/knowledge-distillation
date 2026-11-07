import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('kind,class_name', [('qwen3_5','Qwen3_5'), ('gemma4','Gemma4'), ('gemma4_unified','Gemma4Unified')])
def test_real_official_text_loader_restores_prefixed_local_weights(tmp_path, kind, class_name):
    import torch
    import transformers as t
    from safetensors.torch import save_file
    from src.research_models import load_model
    kwargs = dict(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=8, layer_types=['full_attention'])
    if kind == 'gemma4': kwargs.update(hidden_size_per_layer_input=0, vocab_size_per_layer_input=16)
    text = getattr(t, class_name + 'TextConfig')(**kwargs)
    original = getattr(t, class_name + 'ForCausalLM')(text)
    outer = getattr(t, class_name + 'Config')(text_config=text.to_dict())
    outer.save_pretrained(tmp_path)
    weights = {k.replace('model.', 'model.language_model.', 1) if k.startswith('model.') else k: v.clone() for k, v in original.state_dict().items()}
    save_file(weights, tmp_path / 'model.safetensors', metadata={'format': 'pt'})
    loaded = load_model({'id': str(tmp_path), 'revision': 'a' * 40}, device='cpu', dtype='float32')
    assert not any('vision' in n or 'audio' in n for n, _ in loaded.named_parameters())
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in original.state_dict().items())
    assert loaded(torch.tensor([[1, 2, 3]])).logits.shape == (1, 3, 16)


def test_adapter_initialization_is_paired_and_checkpoint_restoration_is_strict(tmp_path):
    import torch
    import transformers as t
    from src.research_models import attach_lora, restore_best
    from src.resume_checkpoint import save_checkpoint
    config = t.Lfm2Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, layer_types=['full_attention'])
    torch.manual_seed(1)
    base = t.Lfm2ForCausalLM(config)
    import copy
    first, second = [attach_lora(copy.deepcopy(base), {'rank': 2, 'alpha': 4, 'dropout': .05}, seed=42) for _ in range(2)]
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    names = {n for n, p in first.named_parameters() if p.requires_grad} | {n for n, _ in first.named_buffers()}
    state = {k: v.clone() for k, v in first.state_dict().items() if k in names}
    path = save_checkpoint(tmp_path, {'step': 0, 'model_state': state}, {'test': 1})
    restore_best(second, path, contract={'test': 1}, freeze=True)
    assert not any(p.requires_grad for p in second.parameters())
    with pytest.raises(ValueError, match='contract'):
        restore_best(first, path, contract={'test': 2})
    (Path(path) / 'state.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        restore_best(first, path, contract={'test': 1})


class Tokenizer:
    eos_token_id = 3
    special_tokens_map = {'eos_token': '<eos>'}
    def __init__(self, vocab=None, normalizer='NFC'):
        self.vocab = vocab or {'한국': 0, '어': 1, '문장': 2, '<eos>': 3}
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps({'normalizer': normalizer}))
    def get_vocab(self): return self.vocab
    def encode(self, text, add_special_tokens=False): return [0, 1]


def test_tokenizer_contract_checks_full_map_and_normalizer():
    from src.research_models import verify_tokenizers
    report = verify_tokenizers(Tokenizer(), Tokenizer())
    assert report['vocab_size'] == 4
    assert len(report['fingerprint']) == 64
    with pytest.raises(ValueError, match='token'):
        verify_tokenizers(Tokenizer(), Tokenizer({'한국': 1, '어': 0, '문장': 2, '<eos>': 3}))
    with pytest.raises(ValueError, match='token'):
        verify_tokenizers(Tokenizer(), Tokenizer(normalizer='NFKC'))


def test_official_text_class_selection_fails_closed():
    from src.research_models import text_loader_spec
    expected = {'qwen3_5': 'Qwen3_5ForCausalLM', 'gemma4': 'Gemma4ForCausalLM',
                'gemma4_unified': 'Gemma4UnifiedForCausalLM', 'lfm2': 'Lfm2ForCausalLM', 'lfm2_moe': 'Lfm2MoeForCausalLM'}
    for kind, name in expected.items():
        cfg = SimpleNamespace(model_type=kind, text_config=SimpleNamespace(model_type=kind + '_text'))
        cls, text, mapping = text_loader_spec(cfg)
        assert cls == name
        assert text is (cfg if kind.startswith('lfm') else cfg.text_config)
        assert bool(mapping) == (not kind.startswith('lfm'))
    with pytest.raises(ValueError, match='unsupported'):
        text_loader_spec(SimpleNamespace(model_type='unknown'))
