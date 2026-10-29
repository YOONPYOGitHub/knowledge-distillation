"""Offline regressions for BOS-prefixed packing and quantized Teacher configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.config import KDConfig
from src.dataset import load_lm_dataset


class _FakeTokenizer:
    """토크나이저 대역 — 문서당 BOS 로 시작하는 토큰열을 돌려준다."""

    name_or_path = "fake/tokenizer"

    def __init__(self, bos_token_id=2):
        self.bos_token_id = bos_token_id
        self.pad_token = "<pad>"

    def __call__(self, texts, **kwargs):
        # 문서마다 BOS + 고유 토큰 40개
        return {
            "input_ids": [
                [self.bos_token_id] + list(range(100 + i * 40, 140 + i * 40))
                for i in range(len(texts))
            ]
        }


def _dataset_stub():
    from datasets import Dataset, DatasetDict

    def make(n, offset):
        return Dataset.from_dict({"text": [f"문서 {i + offset}" for i in range(n)]})

    # _load_raw_dataset 을 대체하므로 이미 split 된 형태를 돌려준다
    return DatasetDict({"train": make(16, 0), "validation": make(2, 16), "test": make(2, 18)})


class BosPackingTests(unittest.TestCase):
    def _pack(self, prepend_bos, seq_len=8):
        config = KDConfig(
            dataset_name="stub", dataset_config=None, dataset_streaming=False,
            dataset_max_samples=0, max_seq_length=seq_len, prepend_bos=prepend_bos,
            validation_ratio=0.1, test_ratio=0.1, device="cpu",
        )
        with patch("src.dataset._load_raw_dataset", return_value=_dataset_stub()):
            return load_lm_dataset(config, _FakeTokenizer())

    def test_every_chunk_starts_with_bos_when_enabled(self):
        packed = self._pack(prepend_bos=True)
        for split in ["train", "validation", "test"]:
            for row in packed[split]:
                self.assertEqual(row["input_ids"][0].item(), 2)
                self.assertEqual(len(row["input_ids"]), 8)
                self.assertEqual(len(row["attention_mask"]), 8)
                # labels 는 input_ids 와 같은 길이여야 shift 후 정렬이 맞는다
                self.assertEqual(len(row["labels"]), 8)

    def test_packing_is_unchanged_when_disabled(self):
        packed = self._pack(prepend_bos=False)
        rows = [row["input_ids"].tolist() for row in packed["train"]]
        self.assertTrue(all(len(row) == 8 for row in rows))
        # BOS 를 끼워넣지 않으므로 대부분의 청크는 문서 중간에서 시작한다
        self.assertFalse(all(row[0] == 2 for row in rows))

    def test_bos_requires_a_bos_token(self):
        config = KDConfig(
            dataset_name="stub", dataset_config=None, dataset_streaming=False,
            max_seq_length=8, prepend_bos=True, device="cpu",
        )
        with patch("src.dataset._load_raw_dataset", return_value=_dataset_stub()):
            with self.assertRaises(ValueError):
                load_lm_dataset(config, _FakeTokenizer(bos_token_id=None))


class TeacherQuantizationConfigTests(unittest.TestCase):
    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            KDConfig(teacher_quantization="int3", device="cpu")
        with self.assertRaises(ValueError):
            KDConfig(teacher_ft_quantization="int3", device="cpu")

    def test_qlora_requires_lora_rank(self):
        """양자화된 base 는 직접 학습할 수 없으므로 LoRA rank 가 없으면 학습 대상이 0개다."""
        with self.assertRaises(ValueError):
            KDConfig(teacher_ft_quantization="nf4", teacher_lora_rank=0, device="cpu")
        KDConfig(teacher_ft_quantization="nf4", teacher_lora_rank=16, device="cpu")

    def test_device_map_conflicts_with_per_rank_teachers(self):
        with self.assertRaises(ValueError):
            KDConfig(
                teacher_device_map="auto", teacher_devices=["cuda:2", "cuda:3"],
                device="cpu",
            )

    def test_quantization_builds_expected_bnb_config(self):
        from src.models import teacher_quantization_config

        self.assertIsNone(teacher_quantization_config(KDConfig(device="cpu")))

        int8 = teacher_quantization_config(KDConfig(device="cpu", teacher_quantization="int8"))
        self.assertTrue(int8.load_in_8bit)

        nf4 = teacher_quantization_config(
            KDConfig(device="cpu", bf16=True, teacher_quantization="nf4")
        )
        self.assertTrue(nf4.load_in_4bit)
        self.assertEqual(nf4.bnb_4bit_quant_type, "nf4")
        self.assertTrue(nf4.bnb_4bit_use_double_quant)

        # mode 인자가 config 값보다 우선한다 (QLoRA 경로가 이 방식으로 호출한다)
        override = teacher_quantization_config(
            KDConfig(device="cpu", teacher_quantization="int8"), "nf4"
        )
        self.assertTrue(override.load_in_4bit)


if __name__ == "__main__":
    unittest.main()
