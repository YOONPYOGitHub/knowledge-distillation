"""Synthetic heldout-free token-weighted final-evaluation statistics."""
import unittest
import torch
import torch.nn.functional as functional
import src.final_evaluation as evaluation


class FinalTokenCETests(unittest.TestCase):
    def test_rejects_ce_overflow_from_finite_fp32_logits(self):
        logits = torch.tensor([[[3e38, -3e38], [0., 0.]]])
        with self.assertRaisesRegex(ValueError, 'nonfinite token CE'):
            evaluation.final_token_ce_statistics(logits, torch.tensor([[0, 1]]))

    def test_aggregation_normalizes_noniterable_and_integer_overflow_errors(self):
        for records in (None, 7, [{'ce_sum': 10**400, 'tokens': 10**400}]):
            with self.subTest(kind=type(records).__name__), self.assertRaisesRegex(
                    ValueError, 'invalid token CE statistics'):
                evaluation.aggregate_final_token_ce(records)

    def test_aggregation_weights_tokens_not_batch_means(self):
        aggregate = getattr(evaluation, 'aggregate_final_token_ce', None)
        self.assertTrue(callable(aggregate), 'token CE aggregation missing')
        result = aggregate(iter([{'ce_sum': 2.0, 'tokens': 1}, {'ce_sum': 12.0, 'tokens': 3}]))
        self.assertEqual(result['tokens'], 4)
        self.assertEqual(result['ce_sum'], 14.0)
        self.assertEqual(result['ce'], 3.5)
        self.assertAlmostEqual(result['ppl'], __import__('math').exp(3.5))

    def test_aggregation_rejects_invalid_statistics(self):
        for records in ([], [None], [{'ce_sum': 1., 'tokens': 0}],
                        [{'ce_sum': 1., 'tokens': True}],
                        [{'ce_sum': 1., 'tokens': 1.5}],
                        [{'ce_sum': -1., 'tokens': 1}],
                        [{'ce_sum': float('nan'), 'tokens': 1}],
                        [{'ce_sum': float('inf'), 'tokens': 1}],
                        [{'ce_sum': True, 'tokens': 1}],
                        [{'ce_sum': 1000., 'tokens': 1}],
                        [{'ce_sum': 1e308, 'tokens': 1}] * 2,
                        [{'tokens': 1}]):
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, 'invalid token CE statistics'):
                evaluation.aggregate_final_token_ce(records)

    def test_rejects_invalid_tensor_contract(self):
        cases = [
            (torch.zeros(1, 3, 4), torch.zeros(1, 3)),
            (torch.zeros(1, 3, 4), torch.tensor([[0, -1, 2]])),
            (torch.zeros(1, 3, 4), torch.tensor([[0, 4, 2]])),
            (torch.zeros(1, 3, 4), torch.zeros(2, 3, dtype=torch.long)),
            (torch.zeros(3, 4), torch.zeros(1, 3, dtype=torch.long)),
            (torch.zeros(1, 3, 4, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long)),
        ]
        for logits, labels in cases:
            with self.subTest(shapes=(logits.shape, labels.shape), dtype=labels.dtype):
                with self.assertRaisesRegex(ValueError, 'invalid token CE inputs'):
                    evaluation.final_token_ce_statistics(logits, labels)

    def test_rejects_nonfinite_logits_even_when_position_is_masked(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            logits = torch.zeros(1, 3, 4)
            logits[0, 1, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonfinite logits'):
                evaluation.final_token_ce_statistics(logits, torch.tensor([[0, 1, -100]]))

    def test_rejects_all_masked_shifted_targets(self):
        with self.assertRaisesRegex(ValueError, 'zero valid tokens'):
            evaluation.final_token_ce_statistics(torch.zeros(1, 3, 4), torch.full((1, 3), -100))

    def test_shifted_masked_statistics_use_fp32_token_sum(self):
        statistic = getattr(evaluation, 'final_token_ce_statistics', None)
        self.assertTrue(callable(statistic), 'final token CE statistics missing')
        logits = torch.tensor([[[2., 0., -1.], [0., 3., 1.], [1., 0., 4.], [0., 0., 0.]]], dtype=torch.bfloat16)
        labels = torch.tensor([[2, 0, -100, 2]])
        expected = functional.cross_entropy(logits[:, [0, 2], :].float().reshape(-1, 3), torch.tensor([0, 2]), reduction='sum').item()
        result = statistic(logits, labels)
        self.assertEqual(result['tokens'], 2)
        self.assertAlmostEqual(result['ce_sum'], expected, places=7)
        self.assertEqual(set(result), {'ce_sum', 'tokens'})
