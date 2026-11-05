"""Real CPU forward scoring over multiple heldout-shaped batches, not GPU evidence."""
import torch
from src import final_evaluation


def test_model_scoring_runner_aggregates_real_forwards_once_per_batch():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
            self.calls = 0

        def forward(self, input_ids):
            assert not self.training
            assert not torch.is_grad_enabled()
            self.calls += 1
            return type('Output', (), {'logits': self.weight[input_ids]})()

    model = Tiny()
    ids = torch.zeros((1, 512), dtype=torch.long)
    first_labels = torch.full_like(ids, -100)
    first_labels[0, 1:3] = 1
    second_labels = torch.full_like(ids, -100)
    second_labels[0, 1] = 2
    batches = [(ids, first_labels), (ids, second_labels)]
    runner = getattr(final_evaluation, 'score_final_model_batches', None)
    assert callable(runner), 'missing real model batch scoring runner'
    result = runner(model, iter(batches))
    assert model.calls == 2
    assert result['tokens'] == 3
    assert abs(result['ce'] - float(torch.log(torch.tensor(4.)))) < 1e-6
    assert abs(result['ppl'] - 4.) < 1e-6
