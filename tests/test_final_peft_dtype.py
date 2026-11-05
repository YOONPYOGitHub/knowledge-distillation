"""Exercise the real pinned-loader -> PEFT -> selected restore -> scoring seam."""
import hashlib
import io
import json

import pytest
import torch
import transformers as t

from src import final_evaluation as fe
from src.research_models import attach_lora, load_model


def selected_peft_model(tmp_path):
    config = t.Qwen3_5TextConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=1, num_attention_heads=2,
                               num_key_value_heads=1, head_dim=8,
                               layer_types=['full_attention'])
    torch.manual_seed(7)
    t.Qwen3_5ForCausalLM(config).save_pretrained(tmp_path / 'base')
    spec = {'id': str(tmp_path / 'base'), 'revision': 'a' * 40}
    adapter = {'rank': 2, 'alpha': 4, 'dropout': .05}
    original = attach_lora(load_model(spec, device='cpu'), adapter, seed=42)
    names = {n for n, p in original.named_parameters() if p.requires_grad}
    names.update(n for n, _ in original.named_buffers())
    state = {k: v.clone() for k, v in original.state_dict().items() if k in names}
    # Nonzero FP32 trained weights must not be rounded/cast to make a guard pass.
    for name in state:
        if 'lora_B' in name:
            state[name].fill_(0.01234567)
    stream = io.BytesIO()
    torch.save({'model_state': state}, stream)
    payload = stream.getvalue()
    contract = {'identity': {'stage': 'ce', 'model': spec, 'adapter': adapter},
                'settings': {'seed': 42}}
    manifest = json.dumps({'version': 1, 'step': 3000, 'contract': contract,
                          'identity': contract['identity'], 'files': {'state.pt': {
                              'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}}).encode()
    marker = json.dumps({'manifest_sha256': hashlib.sha256(manifest).hexdigest()}).encode()
    generation = tmp_path / 'generation'
    generation.mkdir()
    (generation / 'state.pt').write_bytes(payload)
    loaded = fe.load_final_selected_model(str(generation), marker, manifest,
        expected_step=3000, expected_contract=contract, max_bytes=len(payload),
        model_spec=spec, adapter=adapter, seed=42, device='cpu')
    return loaded, state


def test_real_selected_peft_fp32_adapters_score_without_casting(tmp_path):
    model, state = selected_peft_model(tmp_path)
    assert any(p.dtype == torch.float32 for p in model.parameters())
    inputs = torch.arange(512).remainder(16).reshape(1, 512)
    result = fe.evaluate_final_batch(model, inputs, inputs.clone())
    assert result['tokens'] == 511
    assert result['ce_sum'] > 0
    assert all(torch.equal(model.state_dict()[name], value) for name, value in state.items())
    with torch.inference_mode():
        logits = model(input_ids=inputs).logits
    assert logits.dtype == torch.bfloat16
    oracle = (torch.logsumexp(logits[:, :-1].float(), -1)
              - logits[:, :-1].float().gather(-1, inputs[:, 1:, None]).squeeze(-1)).sum()
    assert result['ce_sum'] == pytest.approx(oracle.item(), rel=1e-6)


@pytest.mark.parametrize('bad', ['base', 'spoof', 'adapter_half'])
def test_peft_exception_does_not_allow_unrelated_dtypes(tmp_path, bad):
    model, _ = selected_peft_model(tmp_path)
    if bad == 'base':
        model.get_input_embeddings().to(torch.float32)
    elif bad == 'spoof':
        model.register_parameter('lora_A_spoof', torch.nn.Parameter(torch.ones(1)))
    else:
        next(m for m in model.modules() if hasattr(m, 'lora_A') and m.lora_A).lora_A['default'].to(torch.float16)
    inputs = torch.zeros((1, 512), dtype=torch.long)
    with pytest.raises(ValueError, match='bfloat16'):
        fe.evaluate_final_batch(model, inputs, inputs.clone())
