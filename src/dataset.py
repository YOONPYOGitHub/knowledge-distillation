"""데이터 로드 & 전처리 — WikiText-2 데이터셋"""

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import KDConfig


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """토크나이저 로드 (GPT-2 계열은 pad_token 설정 필요)"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_wikitext(config: KDConfig, tokenizer: AutoTokenizer):
    """WikiText-2 데이터셋 로드 및 토크나이징 (Packing 방식)

    전체 텍스트를 연결한 뒤 max_seq_length 단위로 chunking.
    패딩 없이 모든 토큰이 유효한 학습/평가 대상이 됨.

    Returns:
        dict: {"train": Dataset, "validation": Dataset, "test": Dataset}
    """
    raw_dataset = load_dataset(config.dataset_name, config.dataset_config)
    seq_len = config.max_seq_length

    def tokenize_and_pack(examples):
        # 전체 텍스트를 연결하여 토크나이즈
        concatenated = tokenizer(
            examples["text"],
            return_attention_mask=False,
        )["input_ids"]

        # 모든 토큰을 하나로 이어붙이기
        all_ids = []
        for ids in concatenated:
            all_ids.extend(ids)

        # seq_len 단위로 chunking (나머지는 버림)
        total_length = (len(all_ids) // seq_len) * seq_len
        all_ids = all_ids[:total_length]

        result = {
            "input_ids": [all_ids[i : i + seq_len] for i in range(0, total_length, seq_len)],
            "attention_mask": [[1] * seq_len for _ in range(0, total_length, seq_len)],
            "labels": [all_ids[i : i + seq_len] for i in range(0, total_length, seq_len)],
        }
        return result

    tokenized_dataset = raw_dataset.map(
        tokenize_and_pack,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def create_dataloaders(config: KDConfig, tokenizer: AutoTokenizer) -> dict:
    """Train/Validation/Test DataLoader 생성

    Returns:
        dict: {"train": DataLoader, "validation": DataLoader, "test": DataLoader}
    """
    dataset = load_wikitext(config, tokenizer)

    loaders = {}
    for split in ["train", "validation", "test"]:
        loaders[split] = DataLoader(
            dataset[split],
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            pin_memory=(config.device == "cuda"),
        )

    return loaders
