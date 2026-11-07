"""Pinned official text-only Transformers loading and shared-tokenizer gates."""
import json
import re
from .research_data import digest

KOREAN_PROBES = ('안녕하세요. 오늘 기분은 어떠세요?', '약은 식사 후에 복용하세요.', '서울의 기온은 23.5°C입니다.', '한글과 English, 숫자 123!')


def verify_tokenizers(teacher, student):
    def signature(tokenizer):
        vocab = tokenizer.get_vocab()
        if set(vocab.values()) != set(range(len(vocab))):
            raise ValueError('token IDs must be contiguous')
        return {'vocab': vocab, 'backend': json.loads(tokenizer.backend_tokenizer.to_str()),
                'special_tokens': {str(k): str(v) for k, v in tokenizer.special_tokens_map.items()},
                'eos_token_id': tokenizer.eos_token_id,
                'probes': [tokenizer.encode(text, add_special_tokens=False) for text in KOREAN_PROBES]}
    left, right = signature(teacher), signature(student)
    if left != right:
        raise ValueError('teacher/student tokenizer token map, backend, special tokens or Korean encoding mismatch')
    return {'fingerprint': digest(left), 'vocab_size': len(left['vocab']), 'korean_probes': left['probes']}


def text_loader_spec(config):
    kind = config.model_type
    classes = {'qwen3_5': 'Qwen3_5ForCausalLM', 'qwen3_5_text': 'Qwen3_5ForCausalLM',
               'gemma4': 'Gemma4ForCausalLM', 'gemma4_text': 'Gemma4ForCausalLM',
               'gemma4_unified': 'Gemma4UnifiedForCausalLM', 'gemma4_unified_text': 'Gemma4UnifiedForCausalLM',
               'lfm2': 'Lfm2ForCausalLM', 'lfm2_moe': 'Lfm2MoeForCausalLM',
               'qwen3': 'Qwen3ForCausalLM', 'llama': 'LlamaForCausalLM',
               'gemma3': 'Gemma3ForCausalLM', 'gemma3_text': 'Gemma3ForCausalLM'}
    if kind not in classes:
        raise ValueError('unsupported official text model: ' + kind)
    nested = getattr(config, 'text_config', None) if not kind.startswith('lfm') else None
    # Published multimodal checkpoints store text tensors under model.language_model.
    mapping = {r'^model\.language_model\.': 'model.', r'^language_model\.model\.': 'model.',
               r'^language_model\.lm_head\.': 'lm_head.'} if nested is not None else None
    return classes[kind], nested if nested is not None else config, mapping


def load_model(spec, *, device='cuda', dtype='bfloat16', attention='sdpa'):
    import torch
    import transformers as t
    validate_revision(spec['revision'])
    config = t.AutoConfig.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False)
    name, text, mapping = text_loader_spec(config)
    if tuple(int(x) for x in t.__version__.split('.')[:2]) < (4, 52):
        # Older Transformers strips the language_model prefix itself. Renaming first
        # makes it skip every tensor, so the mapping only applies from 4.52 on.
        mapping = None
    cls = getattr(t, name, None)
    if cls is None:
        raise RuntimeError(f'{name} unavailable; install compatible Transformers 5')
    placement = {}
    if spec.get('device_map'):
        placement['device_map'] = spec['device_map']
        if spec.get('max_memory'):
            placement['max_memory'] = {int(k): v for k, v in spec['max_memory'].items()}
    model, info = cls.from_pretrained(spec['id'], revision=spec['revision'], config=text,
                                     trust_remote_code=False, torch_dtype=getattr(torch, dtype),
                                     attn_implementation=attention, key_mapping=mapping,
                                     output_loading_info=True, **placement)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('text backbone was not completely restored: ' + str(info))
    unexpected = info.get('unexpected_keys', [])
    if any(not any(part in key for part in ('vision', 'visual', 'audio', 'embed_vision', 'multi_modal', 'mm_projector')) for key in unexpected):
        raise ValueError('unrecognized checkpoint tensors: ' + str(unexpected[:10]))
    model.config.use_cache = False
    return model if placement else model.to(device)


def attach_lora(model, adapter, *, seed):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=adapter['rank'], lora_alpha=adapter['alpha'],
                                           lora_dropout=adapter['dropout'], target_modules='all-linear',
                                           bias='none', task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model


def restore_best(model, path, *, contract, freeze=False):
    from .resume_checkpoint import load_checkpoint
    state = load_checkpoint(path, expected_contract=contract)['model_state']
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    names.update(name for name, _ in model.named_buffers())
    expected = {k: v for k, v in model.state_dict().items() if k in names}
    if state.keys() != expected.keys():
        raise ValueError('checkpoint trainable/buffer keys mismatch')
    for key, tensor in state.items():
        if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
            raise ValueError('checkpoint shape/dtype mismatch: ' + key)
    model.load_state_dict(state, strict=False)
    if freeze:
        model.requires_grad_(False)
        model.eval()
    return model


def validate_revision(revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('revision must be a pinned 40-character SHA')
