"""Offline regressions for top-K truncated KD loss."""

import unittest

import torch

from src.config import KDConfig
from src.distill import kd_loss


def _config(**overrides):
    base = dict(
        temperature=2.0, alpha=0.5, kd_reduction="tokenmean",
        kd_divergence="reverse_kl", device="cpu",
    )
    base.update(overrides)
    return KDConfig(**base)


class TopKKdTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vocab = 64
        self.student = torch.randn(2, 9, self.vocab, dtype=torch.float32, requires_grad=True)
        self.teacher = torch.randn(2, 9, self.vocab, dtype=torch.float32)
        self.labels = torch.randint(0, self.vocab, (2, 9))
        self.labels[0, 3] = -100  # 무시 토큰이 섞여도 tokenmean 이 맞아야 한다

    def test_full_vocab_top_k_matches_untruncated(self):
        """K가 vocabulary 전체면 top-K 경로는 기존 full-vocab 결과와 같아야 한다."""
        full = kd_loss(self.student, self.teacher, self.labels, _config(kd_top_k=0))
        topk = kd_loss(self.student, self.teacher, self.labels, _config(kd_top_k=self.vocab))
        for a, b in zip(full[1:], topk[1:]):
            self.assertAlmostEqual(a, b, places=5)
        self.assertTrue(torch.allclose(full[0], topk[0], atol=1e-6))

    def test_k_larger_than_vocab_is_clamped(self):
        clamped = kd_loss(self.student, self.teacher, self.labels, _config(kd_top_k=self.vocab * 10))
        full = kd_loss(self.student, self.teacher, self.labels, _config(kd_top_k=0))
        self.assertAlmostEqual(clamped[2], full[2], places=5)

    def test_truncation_changes_the_kd_term_but_not_ce(self):
        """top-K 는 KD 항만 바꾼다. CE 는 full vocabulary 그대로여야 한다."""
        full = kd_loss(self.student, self.teacher, self.labels, _config(kd_top_k=0))
        topk = kd_loss(self.student, self.teacher, self.labels, _config(kd_top_k=8))
        self.assertAlmostEqual(full[1], topk[1], places=6)  # CE 동일
        self.assertNotAlmostEqual(full[2], topk[2], places=3)  # KD 는 달라진다

    def test_gradients_flow_to_selected_tokens(self):
        loss, _, _ = kd_loss(self.student, self.teacher, self.labels, _config(kd_top_k=4))
        loss.backward()
        self.assertIsNotNone(self.student.grad)
        self.assertTrue(torch.isfinite(self.student.grad).all())
        self.assertGreater(self.student.grad.abs().sum().item(), 0.0)

    def test_forward_kl_also_supports_truncation(self):
        full = kd_loss(self.student, self.teacher, self.labels,
                       _config(kd_divergence="forward_kl", kd_top_k=0))
        topk = kd_loss(self.student, self.teacher, self.labels,
                       _config(kd_divergence="forward_kl", kd_top_k=self.vocab))
        self.assertAlmostEqual(full[2], topk[2], places=5)

    def test_selected_indices_follow_the_teacher(self):
        """선택은 Teacher 분포 기준이어야 한다 — Student 가 바뀌어도 대상 토큰은 그대로."""
        other_student = torch.randn(2, 9, self.vocab)
        a = kd_loss(self.student, self.teacher, self.labels, _config(kd_top_k=4))
        b = kd_loss(other_student, self.teacher, self.labels, _config(kd_top_k=4))
        # Teacher 가 같으므로 목표 분포(그리고 그 엔트로피)는 동일하다. KD 값 자체는 달라도
        # Teacher 를 바꾸면 반드시 달라져야 한다.
        shifted_teacher = self.teacher.flip(-1)
        c = kd_loss(self.student, shifted_teacher, self.labels, _config(kd_top_k=4))
        self.assertNotAlmostEqual(a[2], c[2], places=3)
        self.assertNotAlmostEqual(a[2], b[2], places=3)

    def test_rejects_negative_k(self):
        with self.assertRaises(ValueError):
            KDConfig(kd_top_k=-1, device="cpu")


if __name__ == "__main__":
    unittest.main()
