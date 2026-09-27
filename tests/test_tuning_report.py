"""Download-free tests for local tuning reports (stdlib and Matplotlib only)."""

import copy
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import tuning_report as report


SUITE = "h100-tuning-20260920"


def metric(ce):
    return {"ce": ce, "ppl": math.exp(ce), "tokens": 128}


class TuningReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.logs = self.root / "logs" / SUITE
        self.figures = self.root / "figures" / SUITE

    def write(self, relative, value):
        path = self.logs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def load(self, relative):
        return json.loads((self.logs / relative).read_text(encoding="utf-8"))

    def trial(self, name, ce, *, alpha=0.5, complete=True, temperature=2.0):
        config = {"learning_rate": 5e-6, "alpha": alpha, "temperature": temperature,
                  "training_seed": 42, "run_id": name}
        history = [{"epoch": epoch, "step": step, **metric(loss), "elapsed_seconds": seconds}
                   for epoch, step, loss, seconds in
                   ((0.0, 0, 2.0, 10.0), (0.5, 4, ce, 20.0), (2.0, 16, ce + 0.1, 40.0))]
        result = {"initial_validation": metric(2.0), "best_validation": metric(ce),
                  "final_validation": metric(ce + 0.1), "best_step": 4, "best_epoch": 0.5,
                  "best_checkpoint": f"/remote/{name}/best.pt", "final_step": 16,
                  "elapsed_seconds": 45.0, "peak_memory_gb": 24.5, "history": history,
                  "matched_validation": metric(ce), "matched_checkpoint": f"/remote/{name}/matched.pt"}
        self.write(f"plans/{name}.json", {"config": config, "training_seed": 42})
        self.write(f"trials/{name}/config.json", config)
        self.write(f"trials/{name}/validation_history.json", history if complete else history[:2])
        if complete:
            self.write(f"trials/{name}/result.json", result)
        return {"name": name, **{key: config[key] for key in report._PARAMETERS},
                **{key: result[key] for key in
                   ("best_validation", "best_step", "best_epoch", "best_checkpoint", "final_step",
                    "elapsed_seconds", "peak_memory_gb")}}

    def completed_fixture(self):
        self.write("manifest.json", {"base_config": {"student_model": "example/student", "epochs": 2},
                                     "training_seed": 42, "splits": {"validation": {"blocks": 8}}})
        first = self.trial("lr5e-6", 1.8)
        winner = self.trial("temperature4", 1.6, temperature=4.0)
        ce = self.trial("ce_only_control", 1.7, alpha=1.0, temperature=4.0)
        self.write("initial_validation.json", metric(2.0))
        self.write("lr.json", [first])
        self.write("alpha.json", [first])
        self.write("temperature.json", [first, winner])
        selection = {"winner": winner, "ce_control": ce, "ce_at_kd_step": metric(1.7),
                     "candidates": [first, winner], "test_used_for_selection": False}
        self.write("selection.json", selection)
        self.write("test_results.json", {
            "selected_parameters": winner,
            "test": {name: metric(loss) for name, loss in
                     (("student_ft_initial", 2.1), ("student_kd_v3_reference", 2.0),
                      ("student_kd_selected", 1.65), ("student_ce_control_best", 1.75),
                      ("student_ce_at_kd_step", 1.8))},
            "selection": "Validation-only, test evaluated once after selection freeze",
            "caution": "Single seed; not a capability benchmark.",
        })
        return selection

    def render(self):
        return report.render_tuning_report(self.root)

    def assert_png(self, name):
        path = self.figures / name
        self.assertGreater(path.stat().st_size, 1000)
        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_completed_report_has_real_metrics_paths_and_all_pngs(self):
        selection = self.completed_fixture()
        before = {path: path.read_bytes() for path in self.logs.rglob("*.json")}
        summary = self.render()
        self.assertEqual(summary, self.load("summary.json"))
        self.assertTrue(summary["completed"])
        self.assertFalse(summary["provisional"])
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["selected_parameters"], selection["winner"])
        self.assertEqual(summary["completed_trials"], 3)
        for path in summary["files"].values():
            self.assertTrue(Path(path).is_file(), path)
        for name in ("val_loss.png", "parameter_search.png", "training_time.png", "perplexity_comparison.png"):
            self.assert_png(name)
        evaluations = self.load("evaluation_results.json")
        self.assertEqual(len(evaluations), 5)
        for row in evaluations:
            self.assertEqual(set(row), {"name", "perplexity", "avg_loss", "tokens"})
            raw = self.load("test_results.json")["test"][row["name"]]
            self.assertEqual(row["perplexity"], raw["ppl"])
            self.assertEqual(row["avg_loss"], raw["ce"])
        with (self.logs / "parameter_search.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertNotIn(b"\r", (self.logs / "parameter_search.csv").read_bytes())
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "completed" for row in rows))
        for name, trial in (("distill_history.json", "temperature4"),
                            ("baseline_history.json", "ce_only_control")):
            history = self.load(name)
            raw = self.load(f"trials/{trial}/validation_history.json")
            self.assertEqual([row["val_ce_loss"] for row in history], [row["ce"] for row in raw])
            self.assertEqual([row["epoch"] for row in history], [row["epoch"] for row in raw])
            self.assertTrue(all("ce_loss" not in row and "train_loss" not in row for row in history))
        markdown = (self.logs / "report.md").read_text()
        self.assertIn("temperature=4", markdown)
        self.assertIn("test is never used to rank", markdown)
        self.assertIn("not pure training time", markdown)
        self.assertIn("Training loss and inference speed were not measured", markdown)
        for relative in re.findall(r"\]\(([^)]+)\)", markdown):
            self.assertFalse(Path(relative).is_absolute())
            self.assertTrue((self.logs / relative).is_file(), relative)
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_partial_histories_include_completed_running_and_pending_trials(self):
        self.trial("finished", 1.9)
        self.trial("still_running", 1.8, complete=False)
        self.write("plans/pending.json", {"config": {"learning_rate": 2e-6, "alpha": 0.3, "temperature": 2}})
        summary = self.render()
        self.assertFalse(summary["completed"])
        self.assertTrue(summary["provisional"])
        self.assertTrue(summary["validation_only"])
        self.assertEqual(summary["status"], "running")
        self.assertEqual(summary["selected_parameters"]["name"], "still_running")
        by_name = {trial["name"]: trial for trial in summary["trials"]}
        self.assertEqual(by_name["pending"]["status"], "pending")
        self.assertEqual(len(by_name["still_running"]["history"]), 2)
        self.assertEqual(by_name["still_running"]["elapsed_seconds"], 20)
        self.assertEqual(summary["completed_trials"], 1)
        self.assert_png("val_loss.png")
        self.assert_png("parameter_search.png")
        self.assert_png("training_time.png")
        self.assertFalse((self.logs / "evaluation_results.json").exists())
        self.assertFalse((self.figures / "perplexity_comparison.png").exists())
        self.assertIn("PROVISIONAL - validation-only", (self.logs / "report.md").read_text())
        with (self.logs / "parameter_search.csv").open(newline="") as stream:
            self.assertEqual([row["name"] for row in csv.DictReader(stream)], ["finished"])

    def test_notes_preserved_report_refreshed_and_figures_closed_on_repeated_calls(self):
        self.trial("candidate", 1.8, complete=False)
        self.render()
        note = self.logs / "notes.md"
        note.write_text("# User decisions\nKeep these notes. 한글 메모\n", encoding="utf-8")
        before, modified = note.read_bytes(), note.stat().st_mtime_ns
        (self.logs / "report.md").write_text("outdated report")
        for index in range(3):
            history = self.load("trials/candidate/validation_history.json")
            history[-1]["ce"] -= 0.01
            self.write("trials/candidate/validation_history.json", history)
            summary = self.render()
            self.assertEqual(note.read_bytes(), before)
            self.assertEqual(note.stat().st_mtime_ns, modified)
            self.assertEqual(report.plt.get_fignums(), [])
            self.assertNotIn("outdated report", (self.logs / "report.md").read_text())
            self.assertAlmostEqual(summary["selected_parameters"]["best_validation"]["ce"], 1.79 - index * 0.01)

    def test_empty_artifacts_are_graceful_and_do_not_invent_metrics(self):
        summary = self.render()
        self.assertEqual(summary["status"], "partial")
        self.assertFalse(summary["completed"])
        self.assertEqual(summary["trials"], [])
        self.assertEqual(summary["figures"], {})
        self.assertEqual(self.load("distill_history.json"), [])
        self.assertEqual(self.load("baseline_history.json"), [])
        self.assertFalse((self.logs / "evaluation_results.json").exists())
        self.assertIn("No trial artifacts", (self.logs / "report.md").read_text())

    def test_test_selection_mismatch_or_test_leakage_cannot_publish_completion(self):
        self.completed_fixture()
        original_selection, original_test = self.load("selection.json"), self.load("test_results.json")
        for mutation in ("name", "alpha", "best_checkpoint", "used_test", "missing_flag", "missing_selection"):
            with self.subTest(mutation=mutation):
                selection, test = copy.deepcopy(original_selection), copy.deepcopy(original_test)
                if mutation in ("name", "alpha", "best_checkpoint"):
                    test["selected_parameters"][mutation] = "wrong"
                elif mutation == "used_test":
                    selection["test_used_for_selection"] = True
                elif mutation == "missing_flag":
                    selection.pop("test_used_for_selection")
                self.write("selection.json", selection)
                self.write("test_results.json", test)
                if mutation == "missing_selection":
                    (self.logs / "selection.json").unlink()
                with self.assertRaises(ValueError):
                    self.render()
                self.assertFalse((self.logs / "summary.json").exists())
                self.assertFalse((self.logs / "evaluation_results.json").exists())

    def test_invalid_test_metrics_and_nonfinite_json_are_rejected(self):
        self.completed_fixture()
        original = self.load("test_results.json")
        for value in ({}, {"student_kd_selected": {"ce": 1.2, "ppl": 3.3, "tokens": 0}},
                      {"student_kd_selected": {"ce": 1.2, "tokens": 12}}):
            bad = copy.deepcopy(original)
            bad["test"] = value
            self.write("test_results.json", bad)
            with self.assertRaises(ValueError):
                self.render()
        for text in ('{"test": NaN}', '{"test": 1e999}', '{"test":'):
            (self.logs / "test_results.json").write_text(text)
            with self.assertRaises(ValueError):
                self.render()
        self.assertFalse((self.logs / "summary.json").exists())

    def test_lost_test_artifact_removes_stale_evaluation_and_perplexity_plot(self):
        self.completed_fixture()
        self.render()
        (self.logs / "test_results.json").unlink()
        summary = self.render()
        self.assertFalse(summary["completed"])
        self.assertEqual(summary["status"], "partial")
        self.assertNotIn("evaluation_results.json", summary["files"])
        self.assertFalse((self.logs / "evaluation_results.json").exists())
        self.assertFalse((self.figures / "perplexity_comparison.png").exists())

    def test_phase_briefs_plans_and_embedded_history_are_supported(self):
        brief = self.trial("candidate", 1.8)
        (self.logs / "trials/candidate/validation_history.json").unlink()
        (self.logs / "trials/candidate/config.json").unlink()
        self.write("lr.json", [brief])
        self.write("alpha.json", [brief])
        summary = self.render()
        self.assertEqual(len(summary["trials"]), 1)
        self.assertEqual(summary["trials"][0]["learning_rate"], 5e-6)
        self.assertEqual(len(self.load("distill_history.json")), 3)
        self.assert_png("val_loss.png")
        (self.logs / "trials/candidate/result.json").unlink()
        summary = self.render()
        self.assertEqual(summary["completed_trials"], 0)
        self.assertEqual(summary["trials"][0]["best_validation"], brief["best_validation"])
        self.assert_png("parameter_search.png")

    def test_invalid_partial_artifact_warns_without_discarding_valid_history(self):
        self.trial("broken", 1.8, complete=False)
        (self.logs / "trials/broken/result.json").write_text('{"elapsed_seconds": NaN}')
        summary = self.render()
        self.assertFalse(summary["completed"])
        self.assertEqual(summary["trials"][0]["status"], "partial")
        self.assertTrue(summary["warnings"])
        self.assert_png("val_loss.png")
        self.assertNotIn("NaN", (self.logs / "summary.json").read_text())

    def test_initial_ft_best_is_not_claimed_as_kd_improvement(self):
        self.trial("candidate", 2.2)
        result = self.load("trials/candidate/result.json")
        result.update(best_validation=metric(2.0), best_step=0, best_epoch=0.0,
                      best_checkpoint="/remote/original_ft.pt")
        self.write("trials/candidate/result.json", result)
        self.render()
        self.assertIn("not evidence of KD improvement", (self.logs / "report.md").read_text())

    def test_suite_id_validation_rejects_traversal_before_writes(self):
        for suite in ("", ".", "..", "../bad", "/absolute", "bad/child", "bad\\child", "x..y", "bad\n"):
            with self.subTest(suite=suite), self.assertRaises(ValueError):
                report.render_tuning_report(self.root, suite)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_png_write_is_atomic_and_closes_figure(self):
        self.trial("candidate", 1.8, complete=False)
        self.render()
        original = (self.figures / "val_loss.png").read_bytes()
        with patch("matplotlib.figure.Figure.savefig", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                self.render()
        self.assertEqual((self.figures / "val_loss.png").read_bytes(), original)
        self.assertEqual(report.plt.get_fignums(), [])
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_import_is_headless_and_does_not_import_model_libraries(self):
        code = ("import sys; from src import tuning_report; "
                "assert tuning_report.matplotlib.get_backend().lower() == 'agg'; "
                "assert not {'torch', 'transformers', 'datasets'} & sys.modules.keys()")
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()