"""Render local H100 tuning artifacts without loading models or contacting Azure.

Generated JSON/CSV/Markdown lives alongside the raw logs; PNGs live under
``results_root/figures/suite_id``. Missing training artifacts are normal during
collection. Invalid or inconsistent final test/selection artifacts raise before
publishing anything. The returned dictionary is the persisted summary, including
absolute paths in ``files``. Individual outputs are atomically replaced, not the
entire report bundle. Calls should be serialized by the caller.
"""

import csv
import io
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_COLORS = ("#2563EB", "#059669", "#D97706")
_PARAMETERS = ("learning_rate", "alpha", "temperature")
_CSV_FIELDS = (
    "name", "status", *_PARAMETERS, "initial_validation_ce", "best_validation_ce",
    "best_validation_ppl", "final_validation_ce", "best_step", "best_epoch",
    "best_checkpoint", "final_step", "elapsed_seconds", "peak_memory_gb",
)
_LIMITATIONS = (
    "Selection uses validation only; test is never used to rank candidates.",
    "Validation CE is token-weighted FP32 next-token cross-entropy, not training loss.",
    "Training loss and inference speed were not measured in these artifacts; "
    "no train-loss or inference-speed figures are generated.",
    "Observed elapsed time includes model loading, validation, and checkpoint saving; "
    "it is not pure training time or inference speed. Incomplete trials show only "
    "the last recorded elapsed time, not a live duration.",
    "Single training seed and small Wikipedia holdout: not a general capability benchmark.",
    "Artifact status does not prove a training process is currently alive.",
)


def _component(value):
    if (not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", value)
            or value == "." or ".." in value):
        raise ValueError("suite_id and trial names must be safe single path components")
    return value


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _metric(value):
    return (isinstance(value, dict) and _number(value.get("ce")) and value["ce"] >= 0
            and _number(value.get("ppl")) and value["ppl"] > 0
            and _integer(value.get("tokens")) and value["tokens"] > 0)


def _read(path, expected, warnings, *, strict=False):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        # Reject NaN/Infinity, including overflowing JSON exponents, recursively.
        json.dumps(value, allow_nan=False)
        if not isinstance(value, expected):
            raise ValueError(f"expected {expected.__name__}")
        return value
    except FileNotFoundError:
        return None
    except (ValueError, UnicodeError) as exc:
        message = f"Invalid artifact {path.name}: {exc}"
        if strict:
            raise ValueError(message) from exc
        warnings.append(message)
        return None


@contextmanager
def _temporary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def _text(path, content, *, preserve=False):
    with _temporary(path) as temporary:
        temporary.write_text(content, encoding="utf-8")
        if preserve:
            # Publish a complete template without ever replacing user notes.
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        else:
            temporary.replace(path)


def _json(path, value):
    _text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _history(value, name, warnings):
    rows = []
    for item in value:
        if (not _metric(item) or not _number(item.get("epoch")) or item["epoch"] < 0
                or not _integer(item.get("step"))):
            warnings.append(f"Skipped invalid validation history row in {name}")
            continue
        row = {key: item[key] for key in ("epoch", "step", "ce", "ppl", "tokens")}
        if _number(item.get("elapsed_seconds")) and item["elapsed_seconds"] >= 0:
            row["elapsed_seconds"] = item["elapsed_seconds"]
        rows.append(row)
    return sorted(rows, key=lambda row: (row["step"], row["epoch"]))


def _collect(root, selection, warnings):
    """Union plans, phase briefs, selection, and trials, including partial trials."""
    briefs = {}
    for phase in ("lr", "alpha", "temperature"):
        for item in _read(root / f"{phase}.json", list, warnings) or []:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                briefs.setdefault(_component(item["name"]), {}).update(item)
    for item in ((selection or {}).get("candidates", [])
                 + [(selection or {}).get("winner"), (selection or {}).get("ce_control")]):
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            briefs.setdefault(_component(item["name"]), {}).update(item)
    plans = {p.stem: p for p in (root / "plans").glob("*.json")}
    directories = {p.name: p for p in (root / "trials").glob("*") if p.is_dir()}
    trials = []
    for name in sorted(briefs.keys() | plans.keys() | directories.keys()):
        _component(name)
        directory = root / "trials" / name
        plan = _read(plans[name], dict, warnings) if name in plans else {}
        plan_config = (plan or {}).get("config", {})
        config = _read(directory / "config.json", dict, warnings)
        result = _read(directory / "result.json", dict, warnings) or {}
        recorded = _read(directory / "validation_history.json", list, warnings)
        embedded = result.get("history", [])
        history = _history(recorded or [], name, warnings)
        fallback = _history(embedded if isinstance(embedded, list) else [], name, warnings)
        if len(fallback) > len(history):
            history = fallback
        data = {**briefs.get(name, {}),
                **(plan_config if isinstance(plan_config, dict) else {}),
                **(config or {}), **result}
        complete = (all(_metric(result.get(key)) for key in
                        ("initial_validation", "best_validation", "final_validation"))
                    and _integer(result.get("final_step")))
        if complete:
            status = "completed"
        elif (directory / "error.json").exists() or (directory / "result.json").exists():
            status = "partial"
        elif name in directories:
            status = "running"
        else:
            status = "partial" if name in briefs else "pending"
        entry = {"name": name, "status": status, "completed": complete, "history": history}
        for key in (*_PARAMETERS, "best_step", "best_epoch", "best_checkpoint",
                    "final_step", "elapsed_seconds", "peak_memory_gb"):
            if key in data:
                entry[key] = data[key]
        for key in ("initial_validation", "best_validation", "final_validation"):
            if _metric(data.get(key)):
                entry[key] = data[key]
        initial = next((row for row in history if row["step"] == 0), None)
        if initial and "initial_validation" not in entry:
            entry["initial_validation"] = {key: initial[key] for key in ("ce", "ppl", "tokens")}
        best = min(history, key=lambda row: row["ce"], default=None)
        if best and ("best_validation" not in entry or best["ce"] < entry["best_validation"]["ce"]):
            entry.update(best_validation={key: best[key] for key in ("ce", "ppl", "tokens")},
                         best_step=best["step"], best_epoch=best["epoch"])
            if not complete:
                entry.pop("best_checkpoint", None)
        if not complete:
            elapsed = [row["elapsed_seconds"] for row in history if "elapsed_seconds" in row]
            if elapsed:
                entry["elapsed_seconds"] = max(elapsed)
        trials.append(entry)
    return trials


def _final_artifacts(root, warnings):
    selection = _read(root / "selection.json", dict, warnings, strict=True)
    test = _read(root / "test_results.json", dict, warnings, strict=True)
    if selection is not None:
        winner = selection.get("winner")
        if (selection.get("test_used_for_selection") is not False
                or not isinstance(winner, dict) or not isinstance(winner.get("name"), str)
                or not _metric(winner.get("best_validation"))
                or not all(_number(winner.get(key)) for key in _PARAMETERS)
                or not isinstance(selection.get("candidates", []), list)):
            raise ValueError("Invalid validation-only selection.json")
        _component(winner["name"])
    if test is not None:
        if selection is None or test.get("selected_parameters") != selection["winner"]:
            raise ValueError("test_results.json selected_parameters do not match selection.json winner")
        measurements = test.get("test")
        if (not isinstance(measurements, dict) or not measurements
                or not all(isinstance(name, str) and name and _metric(metrics)
                           for name, metrics in measurements.items())):
            raise ValueError("test_results.json must contain actual finite test metrics and positive tokens")
    return selection, test


def _standard_history(trial):
    return [{"epoch": row["epoch"], "step": row["step"], "val_ce_loss": row["ce"],
             "val_perplexity": row["ppl"], "tokens": row["tokens"],
             **({"elapsed_seconds": row["elapsed_seconds"]} if "elapsed_seconds" in row else {})}
            for row in (trial or {}).get("history", [])]


@contextmanager
def _chart(path):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            yield fig, ax
            fig.tight_layout()
            with _temporary(path) as temporary:
                fig.savefig(temporary, format="png", dpi=150)
                temporary.replace(path)
        finally:
            plt.close(fig)


def _color(trial, selected_name):
    if trial.get("alpha") == 1 or trial["name"] == "ce_only_control":
        return _COLORS[2]
    return _COLORS[1] if trial["name"] == selected_name else _COLORS[0]


def _trial_label(trial):
    suffix = "" if trial["completed"] else f" [{trial['status']}]"
    return trial["name"] + suffix


def _plots(figures, trials, initial, selected_name, evaluations, provisional):
    paths = {}
    subtitle = "Provisional - validation only" if provisional else "Validation-only selection"
    histories = [trial for trial in trials if trial["history"]]
    if histories:
        path = figures / "val_loss.png"
        with _chart(path) as (_, ax):
            for index, trial in enumerate(histories):
                rows = trial["history"]
                ax.plot([row["epoch"] for row in rows], [row["ce"] for row in rows],
                        label=_trial_label(trial), color=_color(trial, selected_name),
                        linestyle=("-", "--", "-.", ":")[index % 4],
                        marker=("o", "s", "^", "D")[index % 4], markersize=3)
            ax.set(xlabel="Epoch", ylabel="Validation CE (token-weighted)",
                   title=f"Recorded validation histories\n{subtitle}")
            ax.legend(fontsize=8, ncol=2)
        paths[path.name] = str(path)
    candidates = [trial for trial in trials if _metric(trial.get("best_validation"))]
    if candidates:
        path = figures / "parameter_search.png"
        with _chart(path) as (_, ax):
            ax.bar(range(len(candidates)), [trial["best_validation"]["ce"] for trial in candidates],
                   color=[_color(trial, selected_name) for trial in candidates])
            ax.set_xticks(range(len(candidates)), [_trial_label(trial) for trial in candidates],
                          rotation=30, ha="right", fontsize=8)
            if initial:
                ax.axhline(initial["ce"], color=_COLORS[2], linestyle="--", label="Initial FT validation CE")
                ax.legend(fontsize=8)
            ax.set(ylabel="Best recorded validation CE (lower is better)",
                   title=f"Parameter search (includes initial step 0)\n{subtitle}")
        paths[path.name] = str(path)
    timed = [trial for trial in trials if _number(trial.get("elapsed_seconds"))
             and trial["elapsed_seconds"] >= 0]
    if timed:
        path = figures / "training_time.png"
        with _chart(path) as (_, ax):
            ax.bar(range(len(timed)), [trial["elapsed_seconds"] for trial in timed],
                   color=[_color(trial, selected_name) for trial in timed])
            ax.set_xticks(range(len(timed)), [_trial_label(trial) for trial in timed],
                          rotation=30, ha="right", fontsize=8)
            ax.set(ylabel="Observed elapsed seconds (last recorded for incomplete trials)",
                   title="Observed elapsed time - NOT pure training / inference speed\n"
                         "Includes model loading, validation, checkpoint saving")
        paths[path.name] = str(path)
    if evaluations:
        path = figures / "perplexity_comparison.png"
        labels = {"student_ft_initial": "Initial FT", "student_kd_v3_reference": "KD v3 reference",
                  "student_kd_selected": "Selected KD", "student_ce_control_best": "CE control best",
                  "student_ce_at_kd_step": "CE at KD step"}
        with _chart(path) as (_, ax):
            bars = ax.bar(range(len(evaluations)), [row["perplexity"] for row in evaluations],
                          color=[_COLORS[index % 3] for index in range(len(evaluations))])
            ax.set_xticks(range(len(evaluations)),
                          [labels.get(row["name"], row["name"].encode("ascii", "replace").decode())
                           for row in evaluations], rotation=15, ha="right", fontsize=9)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
            ax.margins(y=0.15)
            ax.set(ylabel="Test perplexity (lower is better)",
                   title="Held-out test perplexity - evaluated after selection freeze")
        paths[path.name] = str(path)
    # Do not leave obsolete test or training plots when rerendering a partial snapshot.
    for name in ("val_loss.png", "parameter_search.png", "training_time.png", "perplexity_comparison.png"):
        if name not in paths:
            (figures / name).unlink(missing_ok=True)
    return paths


def _display(value):
    if value is None:
        return "not recorded"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _report(summary, root):
    provisional = summary["provisional"]
    lines = [f"# H100 tuning report: {summary['suite_id']}", "",
             f"**Status: {summary['status']}**; completed: {str(summary['completed']).lower()}.", "",
             ("**PROVISIONAL - validation-only. No final test results are available.**" if provisional
              else "Final test metrics are available; candidate selection was frozen using validation only."),
             "", "## Current parameters and validation", ""]
    selected = summary["selected_parameters"]
    if selected:
        lines += [f"{'Provisional leader' if not summary['selection_frozen'] else 'Frozen winner'}: "
                  f"**{_display(selected['name'])}**.", "",
                  ", ".join(f"{key}={_display(selected.get(key))}" for key in _PARAMETERS) + ".", "",
                  f"Best validation CE: {_display(selected.get('best_validation', {}).get('ce'))}; "
                  f"step: {_display(selected.get('best_step'))}; epoch: {_display(selected.get('best_epoch'))}.", ""]
        if selected.get("best_step") == 0:
            lines += ["Best checkpoint is the original FT at step 0; this is not evidence of KD improvement.", ""]
    else:
        lines += ["No measured KD candidate is available yet.", ""]
    lines += ["| Trial | Status | LR | Alpha | Temperature | Best validation CE | Best step | Elapsed s | Peak GiB |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for trial in summary["trials"]:
        cells = [trial["name"], trial["status"], *(trial.get(key) for key in _PARAMETERS),
                 trial.get("best_validation", {}).get("ce"), trial.get("best_step"),
                 trial.get("elapsed_seconds"), trial.get("peak_memory_gb")]
        lines.append("| " + " | ".join(_display(cell) for cell in cells) + " |")
    if not summary["trials"]:
        lines += ["", "No trial artifacts recorded yet."]
    lines += ["", "## Held-out test", ""]
    if summary["evaluation_results"]:
        lines += ["| Model | Test CE | Test perplexity | Tokens |", "|---|---:|---:|---:|"]
        for row in summary["evaluation_results"]:
            lines.append("| " + " | ".join(_display(row[key]) for key in
                                           ("name", "avg_loss", "perplexity", "tokens")) + " |")
    else:
        lines += ["Not available. Validation metrics are not substituted for test results."]
    lines += ["", "## Evidence and figures", ""]
    for path in summary["sources"]:
        lines.append(f"- [{path}]({path})")
    for name, path in summary["figures"].items():
        relative = Path(os.path.relpath(path, root)).as_posix()
        lines += ["", f"![{name}]({relative})"]
    lines += ["", "## Measurement limitations", ""]
    lines += [f"- {note}" for note in summary["limitations"]]
    manifest = summary["manifest"]
    lines += ["", "## Reproducibility", "",
              f"Training seed: {_display(manifest.get('training_seed'))}.", "",
              "Frozen configuration, split counts/hashes, and versions (when available):", "",
              "```json", json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), "```"]
    if summary["warnings"]:
        lines += ["", "## Artifact warnings", ""] + [f"- {warning}" for warning in summary["warnings"]]
    lines += ["", "User notes are preserved in [notes.md](notes.md).", ""]
    return "\n".join(lines)


def render_tuning_report(results_root: Path, suite_id: str = "h100-tuning-20260920") -> dict:
    """Render a reproducible snapshot; never infer completion from trial scores alone.

    ``status`` is completed, running (unfinished local trial), or partial.
    A partial/running report is explicitly provisional. ``files`` and ``figures``
    contain only generated/current outputs. Missing artifacts are tolerated;
    invalid final selection or test evidence raises ValueError before writes.
    """
    _component(suite_id)
    results_root = Path(results_root).resolve()
    root, figures = results_root / "logs" / suite_id, results_root / "figures" / suite_id
    warnings = []
    selection, test = _final_artifacts(root, warnings)
    manifest = _read(root / "manifest.json", dict, warnings) or {}
    trials = _collect(root, selection, warnings)
    candidates = [trial for trial in trials if _metric(trial.get("best_validation"))
                  and trial.get("alpha") != 1 and trial["name"] != "ce_only_control"]
    leader = min(candidates, key=lambda trial: trial["best_validation"]["ce"], default=None)
    selected = selection["winner"] if selection else (
        {key: value for key, value in leader.items() if key != "history"} if leader else None)
    selected_name = selected["name"] if selected else None
    selected_trial = next((trial for trial in trials if trial["name"] == selected_name), None)
    ce_name = (selection or {}).get("ce_control", {}).get("name")
    ce_trial = next((trial for trial in trials if trial["name"] == ce_name), None)
    if ce_trial is None:
        ce_trial = next((trial for trial in trials if trial.get("alpha") == 1
                         or trial["name"] == "ce_only_control"), None)
    initial = _read(root / "initial_validation.json", dict, warnings)
    if not _metric(initial):
        initial = next((trial["initial_validation"] for trial in trials
                        if "initial_validation" in trial), None)
    evaluations = [{"name": name, "perplexity": metrics["ppl"],
                    "avg_loss": metrics["ce"], "tokens": metrics["tokens"]}
                   for name, metrics in (test or {}).get("test", {}).items()]
    completed = test is not None and selection is not None
    status = "completed" if completed else (
        "running" if any(trial["status"] == "running" for trial in trials) else "partial")
    sources = sorted(str(path.relative_to(root).as_posix()) for pattern in
                     ("manifest.json", "initial_validation.json", "lr.json", "alpha.json",
                      "temperature.json", "selection.json", "test_results.json", "plans/*.json",
                      "trials/*/config.json", "trials/*/validation_history.json", "trials/*/result.json",
                      "trials/*/error.json") for path in root.glob(pattern))
    summary = {
        "suite_id": suite_id, "status": status, "completed": completed,
        "provisional": not completed, "validation_only": not completed,
        "selection_frozen": selection is not None, "selection": selection,
        "selected_parameters": selected, "initial_validation": initial,
        "completed_trials": sum(trial["completed"] for trial in trials),
        "trials": trials, "evaluation_results": evaluations, "manifest": manifest,
        "limitations": list(_LIMITATIONS), "warnings": warnings, "sources": sources,
        "files": {}, "figures": {},
    }
    if test and test.get("caution"):
        summary["limitations"].append(str(test["caution"]))
    summary["figures"] = _plots(figures, trials, initial, selected_name, evaluations, not completed)
    files = summary["files"]
    for name in ("summary.json", "parameter_search.csv", "distill_history.json",
                 "baseline_history.json", "notes.md", "report.md"):
        files[name] = str(root / name)
    files.update(summary["figures"])
    if evaluations:
        files["evaluation_results.json"] = str(root / "evaluation_results.json")
        _json(root / "evaluation_results.json", evaluations)
    else:
        (root / "evaluation_results.json").unlink(missing_ok=True)
    _json(root / "distill_history.json", _standard_history(selected_trial))
    _json(root / "baseline_history.json", _standard_history(ce_trial))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for trial in trials:
        if not trial["completed"]:
            continue
        row = {key: trial[key] for key in _CSV_FIELDS if key in trial}
        row.update(initial_validation_ce=trial["initial_validation"]["ce"],
                   best_validation_ce=trial["best_validation"]["ce"],
                   best_validation_ppl=trial["best_validation"]["ppl"],
                   final_validation_ce=trial["final_validation"]["ce"])
        writer.writerow(row)
    _text(root / "parameter_search.csv", buffer.getvalue())
    _text(root / "notes.md", f"# Tuning notes: {suite_id}\n\n"
          "User-maintained notes; this file is never overwritten by the renderer.\n\n"
          "See [report.md](report.md) for the current automated report.\n", preserve=True)
    _text(root / "report.md", _report(summary, root))
    _json(root / "summary.json", summary)
    return summary