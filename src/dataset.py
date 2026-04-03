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
    """WikiText-2 데이터셋 로드 및 토크나이징

    Returns:
        dict: {"train": Dataset, "validation": Dataset, "test": Dataset}
    """
    raw_dataset = load_dataset(config.dataset_name, config.dataset_config)

    def tokenize_and_chunk(examples):
        # 빈 문자열 필터링 (WikiText-2에 빈 줄이 많음)
        texts = [t for t in examples["text"] if t.strip()]
        if not texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        # Causal LM: labels = input_ids (다음 토큰 예측)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = raw_dataset.map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )
    tokenized_dataset.set_format("torch")

    # 빈 샘플 필터링
    for split in tokenized_dataset:
        tokenized_dataset[split] = tokenized_dataset[split].filter(
            lambda x: len(x["input_ids"]) > 0
        )

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
