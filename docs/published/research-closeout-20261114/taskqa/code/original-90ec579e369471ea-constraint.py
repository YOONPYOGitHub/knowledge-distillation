"""CPU-only canonical JSON language for a future symmetric decoding arm.

No model runtime, corpus reader, output repair or inference authorization.
The language narrows outputs; it does not relax the frozen scorer.
"""
import json
import string
import unicodedata


def prefix_status(text):
    """Return partial/complete/invalid, with empty-answer iff abstaining."""
    if len(text) > 4096:
        return 'invalid'
    head = '{"answer":"'
    if len(text) < len(head):
        return 'partial' if head.startswith(text) else 'invalid'
    if not text.startswith(head):
        return 'invalid'
    i = len(head)
    while i < len(text):
        char = text[i]
        if unicodedata.category(char) in ['Cc', 'Cs']:
            return 'invalid'
        if char == '"':
            try:
                answer = json.loads(text[len('{"answer":'):i+1])
            except (ValueError, TypeError):
                return 'invalid'
            if any(unicodedata.category(c) in ['Cc', 'Cs'] for c in answer):
                return 'invalid'
            normalized = ''.join(' ' if unicodedata.category(c)[0] in 'PS' else c
                                 for c in unicodedata.normalize('NFKC', answer)).strip()
            if answer and not normalized:
                return 'invalid'
            suffix = ',"abstain":' + ('true}' if not answer else 'false}')
            if i + 1 + len(suffix) > 4096:
                return 'invalid'
            remaining = text[i+1:]
            if remaining == suffix:
                return 'complete'
            return 'partial' if suffix.startswith(remaining) else 'invalid'
        if char == '\\':
            i += 1
            if i == len(text):
                return _partial_status(text, ('"', '\\', '"a'))
            escape = text[i]
            if escape in ['"', '\\', '/']:
                i += 1
                continue
            if escape != 'u':
                return 'invalid'
            digits = text[i+1:i+5]
            if any(c not in string.hexdigits for c in digits):
                return 'invalid'
            if len(digits) < 4:
                low = int(digits.ljust(4, '0'), 16)
                high = int(digits.ljust(4, 'f'), 16)
                if any(start <= low <= high <= end for start,end in
                       [(0,31),(127,159),(0xD800,0xDFFF)]):
                    return 'invalid'
                if len(text) < 4000:
                    return 'partial'
                choices = []
                for value in range(low, high+1):
                    char = chr(value)
                    if unicodedata.category(char) in ['Cc', 'Cs']:
                        continue
                    if not choices:
                        choices.append(format(value, '04x')[len(digits):])
                    content = ''.join(' ' if unicodedata.category(c)[0] in 'PS' else c
                                      for c in unicodedata.normalize('NFKC', char)).strip()
                    if content:
                        choices.append(format(value, '04x')[len(digits):])
                        break
                return _partial_status(text, choices)
            if unicodedata.category(chr(int(digits, 16))) in ['Cc', 'Cs']:
                return 'invalid'
            i += 5
            continue
        i += 1
    return _partial_status(text)


def _partial_status(text, fillers=('',)):
    if len(text) < 4000:
        return 'partial'
    tails = ['","abstain":true}', '","abstain":false}', 'a","abstain":false}']
    return ('partial' if any(prefix_status(text + fill + tail) == 'complete'
                             for fill in fillers for tail in tails) else 'invalid')


def _utf8_can_extend(text, pending):
    """Require a real completion of a pending codepoint at tight bounds."""
    if not pending:
        return prefix_status(text) != 'invalid'
    if len(text) < 3999:
        return prefix_status(text + '가') != 'invalid'
    import itertools
    width = (2 if pending[0] < 0xE0 else 3 if pending[0] < 0xF0 else 4)
    for tail in itertools.product(range(0x80, 0xC0), repeat=width-len(pending)):
        try:
            char = (pending + bytes(tail)).decode('utf-8', 'strict')
        except UnicodeDecodeError:
            continue
        if prefix_status(text + char) != 'invalid':
            return True
    return False


def allowed_token_ids(prefix, vocab_bytes, eos_ids, special_ids):
    """Slow reference mask; measure real-vocabulary CPU cost before adoption."""
    import codecs
    try:
        text, consumed = codecs.utf_8_decode(prefix, 'strict', False)
    except UnicodeDecodeError:
        return []
    pending = prefix[consumed:]
    status = prefix_status(text)
    if status == 'complete' and not pending:
        return sorted(eos_ids)
    if status == 'invalid' or not _utf8_can_extend(text, pending):
        return []
    allowed = []
    for ident, piece in vocab_bytes.items():
        if ident in special_ids or ident in eos_ids or not piece:
            continue
        candidate = prefix + piece
        try:
            decoded, consumed = codecs.utf_8_decode(candidate, 'strict', False)
        except UnicodeDecodeError:
            continue
        if prefix_status(decoded) == 'invalid':
            continue
        # The exact parser check above already validates empty-pending text.
        # Only incomplete UTF-8 needs the unchanged extension feasibility check.
        if consumed < len(candidate) and not _utf8_can_extend(decoded, candidate[consumed:]):
            continue
        allowed.append(ident)
    return allowed


class PrefixCallback:
    """Single-example greedy Transformers prefix_allowed_tokens_fn adapter."""
    def __init__(self, prompt_ids, vocab_bytes, eos_ids, special_ids,
                 max_new_tokens, logits_size, excluded_ids=()):
        self.prompt_ids = list(prompt_ids)
        self.vocab_bytes = dict(vocab_bytes)
        self.eos_ids = set(eos_ids)
        self.special_ids = set(special_ids)
        self.max_new_tokens = max_new_tokens
        self.logits_size = logits_size
        self.excluded_ids = set(excluded_ids)
        if (not self.prompt_ids or not 1 <= max_new_tokens <= 256
                or not self.eos_ids or not self.eos_ids <= self.special_ids
                or self.excluded_ids & (set(self.vocab_bytes) | self.special_ids)
                or set(self.vocab_bytes) | self.special_ids | self.excluded_ids
                   != set(range(logits_size))):
            raise ValueError('unbound EOS, token budget or uncovered logits vocabulary')

    def __call__(self, batch_id, input_ids):
        ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
        if batch_id != 0 or ids[:len(self.prompt_ids)] != self.prompt_ids:
            raise ValueError('single-example prompt identity mismatch')
        generated = ids[len(self.prompt_ids):]
        if len(generated) >= self.max_new_tokens:
            raise ValueError('generation token budget exhausted before EOS')
        if any(i not in self.vocab_bytes or i in self.special_ids or i in self.eos_ids
               for i in generated):
            raise ValueError('generated unknown/special/EOS token before callback')
        prefix = b''.join(self.vocab_bytes[i] for i in generated)
        allowed = allowed_token_ids(prefix, self.vocab_bytes,
                                    self.eos_ids, self.special_ids)
        if not allowed or (len(generated) == self.max_new_tokens-1
                           and not set(allowed) <= self.eos_ids):
            raise ValueError('empty mask or no terminal EOS room; no fallback')
        return allowed


class ConstrainedModel:
    """Generation-only adapter; original caller keeps unmodified raw traces."""
    def __init__(self, model, vocab_bytes, eos_ids, special_ids, excluded_ids):
        self.model = model
        self.vocab_bytes = vocab_bytes
        self.eos_ids = set(eos_ids)
        self.special_ids = set(special_ids)
        self.excluded_ids = set(excluded_ids)

    @property
    def device(self):
        return self.model.device

    def generate(self, **kwargs):
        prompt_ids = kwargs['input_ids'].tolist()[0]
        callback = PrefixCallback(prompt_ids, self.vocab_bytes, self.eos_ids,
                                  self.special_ids, kwargs['max_new_tokens'],
                                  self.model.get_output_embeddings().weight.shape[0],
                                  excluded_ids=self.excluded_ids)
        observation = kwargs.pop('generation_observation', None)
        if observation is not None:
            strict_callback = callback
            def callback(batch_id, input_ids):
                observation.observe_prefix(input_ids)
                try:
                    return strict_callback(batch_id, input_ids)
                except Exception:
                    # No token was selected/appended after this observed prefix.
                    observation.callback_failed = True
                    raise
        return self.model.generate(**kwargs, prefix_allowed_tokens_fn=callback)


def bytelevel_vocab(tokenizer_json):
    """Invert the pinned canonical ByteLevel byte-to-Unicode alphabet."""
    expected = {'type':'ByteLevel','add_prefix_space':False,
                'trim_offsets':False,'use_regex':False}
    if tokenizer_json['decoder'] != expected or tokenizer_json['model']['type'] != 'BPE':
        raise ValueError('unsupported decoder; no silent vocabulary adaptation')
    visible = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    codepoints = list(visible)
    missing = [b for b in range(256) if b not in visible]
    visible += missing
    codepoints += list(range(256,256+len(missing)))
    inverse = {chr(c):b for b,c in zip(visible,codepoints)}
    result = {ident:bytes(inverse[c] for c in piece)
              for piece,ident in tokenizer_json['model']['vocab'].items()}
    for token in tokenizer_json.get('added_tokens', []):
        if token['special']:
            continue
        if any(token.get(k, False) for k in ['normalized','lstrip','rstrip']):
            raise ValueError('unsupported added-token normalization/whitespace policy')
        piece = token['content'].encode('utf-8')
        if token['id'] in result and result[token['id']] != piece:
            raise ValueError('added-token ID collision')
        result[token['id']] = piece
    return result
