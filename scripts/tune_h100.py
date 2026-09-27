#!/usr/bin/env python3
"""Bounded H100 search: 3 LRs -> 2 alphas -> 2 temperatures -> CE control.

Model selection uses ONLY full validation. Test is opened only by final, after
selection.json has been frozen. Every trial restarts from the original FT weights.
Completed trials can be reused; incomplete trials fail rather than silently resume.
"""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from datasets import load_from_disk
from src.config import KDConfig, from_yaml
from src.dataset import load_lm_dataset, load_tokenizer
from src.h100_tuning import evaluate_ce, make_loader, run_trial
from src.models import load_student


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(config, root):
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["base_config"] != asdict(config):
            raise RuntimeError("Existing suite has a different config; choose a new output_dir")
        return manifest
    if shutil.disk_usage(root.parent).free < 35 * 1024 ** 3:
        raise RuntimeError("Need at least 35 GiB free for bounded candidate checkpoints")
    init = Path(config.distill_student_checkpoint)
    adapter = Path(config.teacher_checkpoint) / "adapter_model.safetensors"
    if not init.is_file() or not adapter.is_file():
        raise FileNotFoundError("Original Azure FT checkpoints required")
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config.student_model)
    teacher_tokenizer = load_tokenizer(config.teacher_model)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("KD vocabularies must match")
    print("Materializing frozen Korean Wikipedia split (no model training)", flush=True)
    dataset = load_lm_dataset(config, tokenizer)
    data_path = root / "data"
    if data_path.exists():
        raise RuntimeError("Incomplete data preparation exists; use a new suite directory")
    dataset.save_to_disk(str(data_path))
    splits = {}
    for split, values in dataset.items():
        digest = hashlib.sha256()
        for record in values:
            digest.update(record["input_ids"].numpy().tobytes())
        splits[split] = {"blocks": len(values), "input_ids_sha256": digest.hexdigest()}
    manifest = {
        "base_config": asdict(config), "training_seed": 42,
        "validation_checks_per_epoch": 4, "selection_metric": "validation token-weighted FP32 CE",
        "vocabulary_size": len(tokenizer), "splits": splits,
        "initial_ft_sha256": sha256(init), "teacher_adapter_sha256": sha256(adapter),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "datasets", "peft", "accelerate", "numpy")},
        "source_sha256": {name: sha256(ROOT / name) for name in
                          ("src/h100_tuning.py", "src/distill.py", "scripts/tune_h100.py")},
        "gpu": torch.cuda.get_device_name(0),
        "note": "Old H100 training RNG was not fixed; this is a new controlled comparison.",
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def trial(config, root, manifest, name, lr, alpha, temperature, matched_step=None):
    directory = root / "trials" / name
    config = replace(config, learning_rate=lr, alpha=alpha, temperature=temperature,
                     kd_vocab_size=manifest["vocabulary_size"], run_id=name)
    payload = {"config": asdict(config), "training_seed": 42, "matched_step": matched_step,
               "data": str(root / "data"), "trial_dir": str(directory)}
    payload_path = root / "plans" / f"{name}.json"
    if payload_path.exists() and json.loads(payload_path.read_text()) != payload:
        raise RuntimeError(f"Trial plan changed: {name}")
    write_json(payload_path, payload)
    result_path = directory / "result.json"
    if not result_path.exists():
        if directory.exists():
            raise RuntimeError(f"Incomplete trial {name}; inspect its log, do not overwrite")
        if shutil.disk_usage(root).free < 8 * 1024 ** 3:
            raise RuntimeError("Insufficient free disk for next checkpoint")
        print(f"START {name}: lr={lr} alpha={alpha} T={temperature}", flush=True)
        log_path = root / "logs" / f"{name}.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("x") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(payload_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1500)
    result = json.loads(result_path.read_text())
    result.update(name=name, learning_rate=lr, alpha=alpha, temperature=temperature)
    # All candidates must start at the same measured FT validation score.
    baseline = root / "initial_validation.json"
    if not baseline.exists():
        write_json(baseline, result["initial_validation"])
    if abs(json.loads(baseline.read_text())["ce"] - result["initial_validation"]["ce"]) > 1e-6:
        raise RuntimeError("Initial FT validation mismatch across trials")
    print(f"DONE {name}: best_ce={result['best_validation']['ce']:.6f} "
          f"best_step={result['best_step']} epoch={result['best_epoch']:.3f} "
          f"seconds={result['elapsed_seconds']:.1f} peak_GiB={result['peak_memory_gb']:.2f}", flush=True)
    return result


def brief(result):
    return {key: result[key] for key in (
        "name", "learning_rate", "alpha", "temperature", "best_validation", "best_step",
        "best_epoch", "best_checkpoint", "final_step", "elapsed_seconds", "peak_memory_gb")}


def load_phase(root, phase):
    path = root / f"{phase}.json"
    if not path.exists():
        raise RuntimeError(f"Complete the {phase} phase first")
    return json.loads(path.read_text())


def choose(results):
    return min(results, key=lambda item: item["best_validation"]["ce"])


def search(config, root, manifest, phase):
    if (root / "selection.json").exists():
        raise RuntimeError("Selection already frozen; do not tune after viewing the test set")
    if phase == "lr":
        results = [trial(config, root, manifest, name, lr, 0.5, 2.0) for name, lr in
                   (("lr10e-6", 1e-5), ("lr5e-6", 5e-6), ("lr2e-6", 2e-6))]
    elif phase == "alpha":
        previous = load_phase(root, "lr")
        best = choose(previous)
        results = previous + [trial(config, root, manifest, f"alpha{int(alpha*10)}", best["learning_rate"], alpha, 2.0)
                              for alpha in (0.3, 0.7)]
    else:
        previous = load_phase(root, "alpha")
        best = choose(previous)
        results = previous + [trial(config, root, manifest, f"temperature{int(temp)}",
                                   best["learning_rate"], best["alpha"], temp) for temp in (1.0, 4.0)]
    compact = [brief(result) for result in results]
    write_json(root / f"{phase}.json", compact)
    print("PHASE_WINNER " + json.dumps(choose(compact)), flush=True)


def finalize(config, root, manifest):
    selection_path = root / "selection.json"
    if not selection_path.exists():
        candidates = load_phase(root, "temperature")
        best = choose(candidates)
        # Same FT initial weights, LR, deterministic data order and two-epoch budget.
        ce = trial(config, root, manifest, "ce_only_control", best["learning_rate"], 1.0,
                   best["temperature"], matched_step=best["best_step"])
        write_json(selection_path, {"winner": best, "ce_control": brief(ce),
                                   "ce_at_kd_step": ce["matched_validation"],
                                   "ce_at_kd_checkpoint": ce["matched_checkpoint"],
                                   "candidates": candidates, "test_used_for_selection": False})
    if (root / "test_results.json").exists():
        print((root / "test_results.json").read_text(), flush=True)
        return
    selection = json.loads(selection_path.read_text())
    dataset = load_from_disk(str(root / "data"))
    dataset.set_format("torch")
    test_loader = make_loader(dataset["test"], config.batch_size, num_workers=config.num_workers)
    checkpoints = {
        "student_ft_initial": config.distill_student_checkpoint,
        "student_kd_v3_reference": "/var/lib/kd-results/qwen7b-reverse-v3/checkpoints/qwen7b-korean-reverse-v3/student_kd_best.pt",
        "student_kd_selected": selection["winner"]["best_checkpoint"],
        "student_ce_control_best": selection["ce_control"]["best_checkpoint"],
        "student_ce_at_kd_step": selection["ce_at_kd_checkpoint"],
    }
    student = load_student(config)
    student.config.use_cache = False
    measurements = {}
    for name, checkpoint in checkpoints.items():
        student.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        measurements[name] = evaluate_ce(student, test_loader, config.device)
        print(f"FINAL_TEST {name}: " + json.dumps(measurements[name]), flush=True)
    summary = {"selected_parameters": selection["winner"], "test": measurements,
               "selection": "Validation-only, test evaluated once after selection freeze",
               "caution": "Single training seed, small Wikipedia holdout; not a general capability benchmark."}
    write_json(root / "test_results.json", summary)


def smoke(config, root):
    """Real pretrained models, two updates, a few train/val blocks, no test access."""
    manifest = prepare(config, root)
    directory = root / "full_model_smoke"
    if (directory / "result.json").exists():
        print("FULL_MODEL_SMOKE already completed", flush=True)
        return
    data = load_from_disk(str(root / "data"))
    data.set_format("torch")
    subset = {"train": data["train"].select(range(min(8, len(data["train"])))),
              "validation": data["validation"].select(range(min(8, len(data["validation"]))))}
    trial_config = replace(config, epochs=1, max_train_steps=2, learning_rate=5e-6,
                           kd_vocab_size=manifest["vocabulary_size"])
    result = run_trial(trial_config, subset, directory, training_seed=42)
    if result["final_step"] != 2:
        raise RuntimeError("Full model smoke did not complete two optimizer steps")
    print("FULL_MODEL_SMOKE_PASSED " + json.dumps(brief(dict(result, name="smoke", learning_rate=5e-6,
                                                            alpha=config.alpha, temperature=config.temperature))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h100_qwen7b_korean_tuning.yaml")
    parser.add_argument("--phase", choices=["prepare", "smoke", "lr", "alpha", "temperature", "final", "all"], default="prepare")
    parser.add_argument("--worker")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.worker:
        payload = json.loads(Path(args.worker).read_text())
        data = load_from_disk(payload["data"])
        data.set_format("torch")
        # Do not expose the held-out test split to the trial runner.
        run_trial(KDConfig(**payload["config"]), {key: data[key] for key in ("train", "validation")},
                  Path(payload["trial_dir"]), payload["training_seed"], matched_step=payload["matched_step"])
        return
    config = from_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("H100 CUDA runtime required")
    root = Path(config.output_dir)
    manifest = prepare(config, root)
    phases = ["smoke", "lr", "alpha", "temperature", "final"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "prepare":
            continue
        if phase == "smoke":
            smoke(config, root)
            gc.collect()
            torch.cuda.empty_cache()
        elif phase == "final":
            finalize(config, root, manifest)
        else:
            search(config, root, manifest, phase)


if __name__ == "__main__":
    main()