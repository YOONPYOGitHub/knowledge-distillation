"""Real tiny-model evaluation execution policy, without any heldout access."""
import pytest
import torch
from src import final_evaluation


class TinyModel(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 4, dtype=dtype)
        self.observed = None

    def forward(self, input_ids):
        self.observed = (self.training, torch.is_inference_mode_enabled(),
                         self.embedding.weight.dtype, tuple(input_ids.shape))
        return type('Output', (), {'logits': self.embedding(input_ids)})()


def test_final_batch_enforces_eval_inference_and_bfloat16():
    model = TinyModel().train()
    inputs = torch.arange(512, dtype=torch.long).remainder(4).reshape(1, 512)
    evaluate = getattr(final_evaluation, 'evaluate_final_batch', None)
    assert callable(evaluate), 'missing final model execution policy'
    result = evaluate(model, inputs, inputs.clone())
    assert model.observed == (False, True, torch.bfloat16, (1, 512))
    assert result['tokens'] == 511
    logits = model.embedding(inputs).float()[:, :-1]
    targets = inputs[:, 1:]
    expected = (torch.logsumexp(logits, dim=-1)
                - logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).sum()
    assert result['ce_sum'] == pytest.approx(float(expected.item()), rel=1e-6)


def test_model_device_mismatch_rejected_before_forward():
    model = TinyModel().to('meta').train()
    inputs = torch.zeros((1, 512), dtype=torch.long)
    with pytest.raises(ValueError, match='final evaluation'):
        final_evaluation.evaluate_final_batch(model, inputs, inputs.clone())
    assert model.observed is None
    assert model.training


@pytest.mark.parametrize('case', ['all_ignored', 'illegal_negative'])
def test_invalid_labels_rejected_before_forward(case):
    model = TinyModel().train()
    inputs = torch.zeros((1, 512), dtype=torch.long)
    labels = inputs.clone()
    if case == 'all_ignored':
        labels[:, 1:] = -100
    else:
        labels[:, 1] = -2
    with pytest.raises(ValueError):
        final_evaluation.evaluate_final_batch(model, inputs, labels)
    assert model.observed is None
    assert model.training


@pytest.mark.parametrize('case', ['shape', 'input_dtype', 'label_dtype', 'label_shape', 'non_tensor', 'model_dtype'])
def test_final_batch_rejects_invalid_contract_before_model_execution(case):
    model = TinyModel(dtype=torch.float32 if case == 'model_dtype' else torch.bfloat16).train()
    inputs = torch.zeros((1, 512), dtype=torch.long)
    labels = inputs.clone()
    if case == 'shape':
        inputs = inputs[:, :511]
    elif case == 'input_dtype':
        inputs = inputs.float()
    elif case == 'label_dtype':
        labels = labels.float()
    elif case == 'label_shape':
        labels = labels[:, :511]
    elif case == 'non_tensor':
        inputs = [[0] * 512]
    with pytest.raises(ValueError, match='final evaluation'):
        final_evaluation.evaluate_final_batch(model, inputs, labels)
    assert model.observed is None
    assert model.training
