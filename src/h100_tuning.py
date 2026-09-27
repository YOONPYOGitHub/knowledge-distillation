"""Single-device, fixed-budget tuning on parent-prepared train/validation data.

No scheduler, warmup, early stopping, test-split access, or implicit resume.
The parent owns tokenizer compatibility, data freezing, and unique trial paths.
"""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import set_seed

from src.config import KDConfig
from src.distill import kd_loss
from src.models import create_optimizer, load_student, load_teacher


def _ce_sum(logits, labels):
    shifted = logits[:, :-1, :].float()
    return F.cross_entropy(
        shifted.reshape(-1, shifted.size(-1)), labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="sum",
    )


@torch.no_grad()
def evaluate_ce(model, dataloader, device) -> dict:
    """Return full-validation next-token CE, weighted by unmasked token count."""
    was_training = model.training
    total, tokens = 0.0, 0
    try:
        model.eval()
        for batch in dataloader:
            labels = batch["labels"].to(device, non_blocking=True)
            inputs = {k: v.to(device, non_blocking=True)
                      for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            loss = _ce_sum(logits, labels).item()
            if not math.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            total += loss
            tokens += int((labels[:, 1:] != -100).sum().item())
        if not tokens:
            raise ValueError("Validation contains no valid next-token labels")
        ce = total / tokens
        try:
            ppl = math.exp(ce)
        except OverflowError as exc:
            raise ValueError("Nonfinite validation perplexity") from exc
        if not math.isfinite(ce) or not math.isfinite(ppl):
            raise ValueError("Nonfinite validation metrics")
        return {"ce": ce, "ppl": ppl, "tokens": tokens}
    finally:
        model.train(was_training)


def make_loader(dataset, batch_size, seed=None, num_workers=0) -> DataLoader:
    """Shuffle only when seeded; use a private generator, not the global RNG."""
    generator = torch.Generator().manual_seed(0 if seed is None else seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=seed is not None,
        generator=generator, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(model, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def run_trial(config: KDConfig, dataset: dict, trial_dir: Path, training_seed: int,
              validation_checks_per_epoch: int = 4,
              matched_step: int | None = None) -> dict:
    """Train from original FT weights; never overwrite or resume a trial directory.

    Epoch/step limits are the only stopping criteria. Configured early stopping,
    warmup, and max_eval_steps are deliberately ignored (full validation).
    """
    trial_dir = Path(trial_dir).absolute()
    trial_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    step, history = 0, []
    try:
        _atomic_json(trial_dir / "config.json", {
            **asdict(config), "training_seed": training_seed,
            "validation_checks_per_epoch": validation_checks_per_epoch,
            "matched_step": matched_step,
            "runner": {"batch_size": 2, "optimizer": "adafactor", "scheduler": None,
                       "warmup_steps": 0, "early_stopping": False, "max_eval_steps": 0},
        })
        initial_checkpoint = Path(config.distill_student_checkpoint).resolve()
        if not config.distill_student_checkpoint or not initial_checkpoint.is_file():
            raise FileNotFoundError("Original student FT checkpoint must be a file")
        if not config.teacher_checkpoint or not Path(config.teacher_checkpoint).exists():
            raise FileNotFoundError("Teacher checkpoint must exist")
        epochs = getattr(config, "epochs", 2)
        if epochs < 1 or validation_checks_per_epoch < 1:
            raise ValueError("Epochs and validation checks must be positive")
        if not 0 <= config.alpha <= 1 or config.batch_size != 2:
            raise ValueError("Require alpha in [0, 1] and fixed batch_size=2")
        if config.optimizer != "adafactor":
            raise ValueError("Tuning requires the fixed-learning-rate Adafactor optimizer")
        device = torch.device(config.device)
        teacher_device = torch.device(config.teacher_device or config.device)
        if config.alpha < 1 and teacher_device != device:
            default_index = torch.cuda.current_device() if device.type == "cuda" else None
            if (teacher_device.type != device.type or
                    (teacher_device.index if teacher_device.index is not None else default_index) !=
                    (device.index if device.index is not None else default_index)):
                raise ValueError("Teacher and student must use the same device")
        steps_per_epoch = math.ceil(len(dataset["train"]) / config.batch_size)
        budget = min(epochs * steps_per_epoch, config.max_train_steps or math.inf)
        if not steps_per_epoch or (matched_step is not None and not 0 <= matched_step <= budget):
            raise ValueError("Empty training split or matched_step outside training budget")
        set_seed(training_seed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        student = load_student(config)
        student.load_state_dict(torch.load(initial_checkpoint, map_location="cpu", weights_only=True))
        student.config.use_cache = False
        teacher = None
        if config.alpha < 1:
            teacher = load_teacher(config)
            teacher.config.use_cache = False
            teacher.eval()
            teacher.requires_grad_(False)
        optimizer = create_optimizer(student, config, config.learning_rate)
        validation_loader = make_loader(dataset["validation"], config.batch_size,
                                        num_workers=config.num_workers)
        best_validation, matched_validation, matched_checkpoint = None, None, None
        best_step, best_epoch, best_checkpoint = 0, 0.0, str(initial_checkpoint)

        def validate(epoch):
            nonlocal best_validation, best_step, best_epoch, best_checkpoint
            nonlocal matched_validation, matched_checkpoint
            metrics = evaluate_ce(student, validation_loader, device)
            if best_validation is None:
                best_validation = metrics
            elif metrics["ce"] < best_validation["ce"]:
                path = trial_dir / "best.pt"
                _save_checkpoint(student, path)
                best_validation, best_step, best_epoch = metrics, step, float(epoch)
                best_checkpoint = str(path)
            if matched_step == step:
                path = trial_dir / "matched.pt"
                _save_checkpoint(student, path)
                matched_validation, matched_checkpoint = metrics, str(path)
            history.append({"epoch": float(epoch), "step": step, **metrics,
                            "elapsed_seconds": time.perf_counter() - started})
            _atomic_json(trial_dir / "validation_history.json", history)
            print(f"validation step={step} epoch={epoch:.3f} "
                  f"ce={metrics['ce']:.6f} ppl={metrics['ppl']:.4f}", flush=True)
            return metrics

        initial_validation = final_validation = validate(0.0)
        for epoch in range(1, epochs + 1):
            loader = make_loader(dataset["train"], config.batch_size,
                                 seed=training_seed + epoch, num_workers=config.num_workers)
            interval = math.ceil(len(loader) / validation_checks_per_epoch)
            student.train()
            for batch_index, batch in enumerate(loader, 1):
                labels = batch["labels"].to(device, non_blocking=True)
                inputs = {k: v.to(device, non_blocking=True)
                          for k, v in batch.items() if k != "labels"}
                valid_tokens = (labels[:, 1:] != -100).sum()
                if not valid_tokens.item():
                    raise ValueError("Training batch contains no valid next-token labels")
                optimizer.zero_grad(set_to_none=True)
                logits = student(**inputs).logits
                if teacher is None:
                    loss = _ce_sum(logits, labels) / valid_tokens
                else:
                    with torch.no_grad():
                        teacher_logits = teacher(**inputs).logits
                    loss, _, _ = kd_loss(logits, teacher_logits, labels, config)
                    del teacher_logits
                if not torch.isfinite(loss).item():
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip,
                                               error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                del logits, loss
                step += 1
                if (batch_index % interval == 0 or batch_index == len(loader)
                        or step == budget or step == matched_step):
                    final_validation = validate(epoch - 1 + batch_index / len(loader))
                if step >= budget:
                    break
            if step >= budget:
                break
        peak_memory_gb = 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        result = {
            "initial_validation": initial_validation, "best_validation": best_validation,
            "best_step": best_step, "best_epoch": best_epoch, "best_checkpoint": best_checkpoint,
            "final_validation": final_validation, "final_step": step, "history": history,
            "elapsed_seconds": time.perf_counter() - started, "peak_memory_gb": peak_memory_gb,
            "matched_validation": matched_validation, "matched_checkpoint": matched_checkpoint,
        }
        _atomic_json(trial_dir / "result.json", result)
        return result
    except BaseException as exc:
        _atomic_json(trial_dir / "error.json", {
            "error_type": type(exc).__name__, "step": step,
            "message": "Trial failed; inspect the local exception (details omitted for safety).",
        })
        raise