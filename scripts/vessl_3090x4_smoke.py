#!/usr/bin/env python3
"""Offline four-GPU parity test: tiny Qwen, no downloads or real checkpoints.

Run with two torchrun ranks. Compare paired DDP updates with an equivalent
single-student global batch, including a final batch with one zero-token rank.
This is NOT a 7B VRAM, throughput, or perplexity benchmark.
"""

import copy
import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.batching import GlobalBatchSampler, MaskedDataset
from src.config import KDConfig
from src.distill import kd_loss, train_one_epoch, validate
from src.distributed import (
    cleanup_distributed, normalize_batch_loss, rank, setup_distributed, unwrap_model, wrap_ddp,
)
from src.models import create_optimizer


def run_case(config, dtype):
    torch.manual_seed(123)
    architecture = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, use_cache=False, attention_dropout=0.0,
    )
    teacher = Qwen2ForCausalLM(architecture).to(config.teacher_device, dtype=dtype).eval()
    teacher.requires_grad_(False)
    student = Qwen2ForCausalLM(architecture).to(config.device, dtype=dtype)
    student.gradient_checkpointing_enable()
    reference = copy.deepcopy(student)
    student = wrap_ddp(student, config.device, fp32_reduce=True)
    optimizer = create_optimizer(student, config, config.learning_rate)
    reference_optimizer = create_optimizer(reference, config, config.learning_rate)

    generator = torch.Generator().manual_seed(99)
    records = []
    for index in range(5):
        ids = torch.randint(0, 64, (8,), generator=generator)
        labels = ids.clone()
        if index % 2:
            labels[-2:] = -100  # Exercise unequal VALID token counts as well.
        records.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels))
    local_loader = DataLoader(
        MaskedDataset(records),
        batch_sampler=GlobalBatchSampler(5, 2, 2, rank()),
    )
    reference_loader = DataLoader(records, batch_sampler=GlobalBatchSampler(5, 2))
    max_gradient_error = 0.0
    max_parameter_error = 0.0
    for local, global_batch in zip(local_loader, reference_loader):
        local_labels = local["labels"].to(config.device)
        with torch.no_grad():
            teacher_logits = teacher(local["input_ids"].to(config.teacher_device)).logits
            global_teacher_logits = teacher(global_batch["input_ids"].to(config.teacher_device)).logits
        logits = student(local["input_ids"].to(config.device)).logits
        loss, ce, kd = kd_loss(logits, teacher_logits.to(config.device), local_labels, config)
        loss, metrics = normalize_batch_loss(
            loss, {"total_loss": loss.item(), "ce": ce, "kd": kd}, local_labels, config,
        )
        reference_logits = reference(global_batch["input_ids"].to(config.device)).logits
        reference_loss, _, _ = kd_loss(
            reference_logits, global_teacher_logits.to(config.device),
            global_batch["labels"].to(config.device), config,
        )
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        loss.backward()
        reference_loss.backward()
        if dtype == torch.float32:
            assert abs(metrics["total_loss"] - reference_loss.item()) < 2e-5
        for actual, expected in zip(unwrap_model(student).parameters(), reference.parameters()):
            assert actual.grad is not None and torch.isfinite(actual.grad).all()
            error = (actual.grad.float() - expected.grad.float()).abs().max().item()
            max_gradient_error = max(max_gradient_error, error)
            if dtype == torch.float32:
                torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=3e-4)
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), config.gradient_clip)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(unwrap_model(student).parameters(), reference.parameters()):
            error = (actual.float() - expected.float()).abs().max().item()
            max_parameter_error = max(max_parameter_error, error)
            if dtype == torch.float32:
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-4)

    # Also exercise the production epoch/validation code, not just its helpers.
    history = train_one_epoch(teacher, student, local_loader, optimizer, config)
    history.update(validate(teacher, student, local_loader, config))
    assert all(math.isfinite(value) for value in history.values()), history
    assert all(parameter.grad is None for parameter in teacher.parameters())
    vector = torch.cat([p.detach().flatten() for p in unwrap_model(student).parameters()])
    peer = vector.clone()
    dist.broadcast(peer, src=0)
    torch.testing.assert_close(vector, peer, atol=0, rtol=0)
    print(
        f"rank={rank()} student={config.device} teacher={config.teacher_device} "
        f"{dtype} {config.kd_divergence}: PASS; max_grad_diff={max_gradient_error:.3g} "
        f"max_weight_diff={max_parameter_error:.3g}", flush=True,
    )


def main():
    config = KDConfig(
        device="cuda:0", teacher_devices=["cuda:2", "cuda:3"],
        batch_size=1, global_batch_size=2, training_seed=42,
        kd_reduction="tokenmean", kd_divergence="reverse_kl",
        optimizer="adafactor", learning_rate=1e-5,
    )
    setup_distributed(config)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for divergence in ("forward_kl", "reverse_kl"):
            config.kd_divergence = divergence
            run_case(config, torch.float32)
        run_case(config, torch.bfloat16)
        if rank() == 0:
            print("PAIRED_4GPU_PARITY_PASSED (tiny models only; full-model fit/speed unverified)")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()