"""Causal language-modeling dataset loading and token packing."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

from src.config import KDConfig
from src.distributed import barrier, is_distributed, is_main_process, rank, world_size


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """토크나이저 로드 (GPT-2 계열은 pad_token 설정 필요)"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _split_train_dataset(raw_dataset: DatasetDict, config: KDConfig) -> DatasetDict:
    """Create validation and test splits when a dataset only provides train."""
    if "validation" in raw_dataset and "test" in raw_dataset:
        return raw_dataset

    if set(raw_dataset) != {"train"}:
        raise ValueError(
            "Dataset must provide train/validation/test splits or only a train split"
        )

    holdout_ratio = config.validation_ratio + config.test_ratio
    split = raw_dataset["train"].train_test_split(
        test_size=holdout_ratio,
        seed=config.seed,
    )
    holdout = split["test"].train_test_split(
        test_size=config.test_ratio / holdout_ratio,
        seed=config.seed,
    )
    return DatasetDict(
        {
            "train": split["train"],
            "validation": holdout["train"],
            "test": holdout["test"],
        }
    )


def _load_raw_dataset(config: KDConfig) -> DatasetDict:
    load_args = [config.dataset_name]
    if config.dataset_config:
        load_args.append(config.dataset_config)
    raw_dataset = load_dataset(
        *load_args,
        revision=config.dataset_revision,
        streaming=config.dataset_streaming,
    )

    if config.dataset_streaming:
        shuffled_train = raw_dataset["train"].shuffle(
            seed=config.seed,
            buffer_size=config.dataset_shuffle_buffer,
        )
        records = list(shuffled_train.take(config.dataset_max_samples))
        if len(records) < config.dataset_max_samples:
            raise ValueError(
                f"Dataset returned {len(records)} records; "
                f"requested {config.dataset_max_samples}"
            )
        raw_dataset = DatasetDict({"train": Dataset.from_list(records)})
    elif config.dataset_max_samples:
        train = raw_dataset["train"]
        sample_count = min(config.dataset_max_samples, len(train))
        raw_dataset["train"] = train.select(range(sample_count))

    return _split_train_dataset(raw_dataset, config)


def load_lm_dataset(config: KDConfig, tokenizer: AutoTokenizer):
    """Load and tokenize a causal language-modeling dataset with packing.

    전체 텍스트를 연결한 뒤 max_seq_length 단위로 chunking.
    패딩 없이 모든 토큰이 유효한 학습/평가 대상이 됨.

    Returns:
        dict: {"train": Dataset, "validation": Dataset, "test": Dataset}
    """
    raw_dataset = _load_raw_dataset(config)
    if config.dataset_text_column not in raw_dataset["train"].column_names:
        raise ValueError(
            f"Text column {config.dataset_text_column!r} not found; "
            f"available columns: {raw_dataset['train'].column_names}"
        )
    seq_len = config.max_seq_length

    def tokenize_and_pack(examples):
        # 전체 텍스트를 연결하여 토크나이즈
        concatenated = tokenizer(
            examples[config.dataset_text_column],
            return_attention_mask=False,
            verbose=False,
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
        keep_in_memory=config.dataset_streaming,
    )
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def _distributed_dataset(config: KDConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """Prepare a dataset once per node and share it across local DDP ranks."""
    cache_key = json.dumps(
        {
            "dataset": config.dataset_name,
            "config": config.dataset_config,
            "revision": config.dataset_revision,
            "text_column": config.dataset_text_column,
            "streaming": config.dataset_streaming,
            "max_samples": config.dataset_max_samples,
            "shuffle_buffer": config.dataset_shuffle_buffer,
            "validation_ratio": config.validation_ratio,
            "test_ratio": config.test_ratio,
            "sequence_length": config.max_seq_length,
            "seed": config.seed,
            "tokenizer": tokenizer.name_or_path,
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(cache_key).hexdigest()[:16]
    cache_root = Path(os.environ.get("KD_DATA_CACHE", "/tmp/kd-data-cache"))
    cache_path = cache_root / digest
    success_marker = cache_path / "_SUCCESS"

    if is_main_process() and not success_marker.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_root / f".{digest}.tmp"
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        dataset = load_lm_dataset(config, tokenizer)
        dataset.save_to_disk(temporary_path)
        (temporary_path / "_SUCCESS").touch()
        if cache_path.exists():
            shutil.rmtree(cache_path)
        temporary_path.rename(cache_path)

    barrier()
    dataset = load_from_disk(cache_path)
    dataset.set_format("torch")
    return dataset


def create_dataloaders(config: KDConfig, tokenizer: AutoTokenizer) -> dict:
    """Train/Validation/Test DataLoader 생성

    Returns:
        dict: {"train": DataLoader, "validation": DataLoader, "test": DataLoader}
    """
    dataset = (
        _distributed_dataset(config, tokenizer)
        if is_distributed()
        else load_lm_dataset(config, tokenizer)
    )

    loaders = {}
    for split in ["train", "validation", "test"]:
        sampler = (
            DistributedSampler(
                dataset[split],
                num_replicas=world_size(),
                rank=rank(),
                shuffle=(split == "train"),
                seed=config.seed,
            )
            if is_distributed()
            else None
        )
        loaders[split] = DataLoader(
            dataset[split],
            batch_size=config.batch_size,
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.device.startswith("cuda"),
        )

    return loaders
