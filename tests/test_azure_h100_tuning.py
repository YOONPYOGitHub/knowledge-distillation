"""Offline Azure control-plane tests: no cloud requests or GPU work."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import azure_h100_tuning as control


class AzureControlTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(subscription="sub", resource_group="rg", vm_name="vm", phase="all")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(control, "LOCAL_RESULTS", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_spot_deallocated_does_not_submit_or_start_vm(self):
        with patch.object(control, "vm_state", return_value={"power": "PowerState/deallocated"}), \
                patch.object(control, "az_json") as az:
            with self.assertRaisesRegex(RuntimeError, "No automatic VM start"):
                control.run(self.args)
        az.assert_not_called()
        self.assertFalse((self.root / "managed_job.json").exists())

    def test_running_job_blocks_duplicate_submission(self):
        control.save_job({"name": "existing"})
        with patch.object(control, "vm_state", return_value={"power": "PowerState/running"}), \
                patch.object(control, "job_state", return_value={"execution": {"executionState": "Running"}}), \
                patch.object(control, "az_json") as az:
            with self.assertRaisesRegex(RuntimeError, "use status"):
                control.run(self.args)
        az.assert_not_called()

    def test_managed_submission_uses_actual_region_and_persists_identity(self):
        observed = []

        def submit(args, command):
            record = json.loads((self.root / "managed_job.json").read_text())
            self.assertIn(record["name"], command)
            observed.append(command)
            return {}

        with patch.object(control, "vm_state", return_value={"power": "PowerState/running", "location": "koreacentral"}), \
                patch.object(control, "job_state", return_value={"execution": {"executionState": "Running"}}), \
                patch.object(control, "az_json", side_effect=submit), redirect_stdout(io.StringIO()):
            control.run(self.args)
        self.assertEqual(len(observed), 1)
        command = observed[0]
        self.assertEqual(command[:3], ["vm", "run-command", "create"])
        self.assertEqual(command[command.index("--location") + 1], "koreacentral")
        self.assertEqual(command[command.index("--async-execution") + 1], "true")
        self.assertNotIn("invoke", command)
        script = command[command.index("--script") + 1]
        self.assertIn("flock -n", script)
        self.assertIn("timeout 5000", script)

    def test_status_is_read_only_and_reports_vm_power_separately(self):
        control.save_job({"name": "job"})
        with patch.object(control, "vm_state", return_value={"power": "PowerState/deallocated", "priority": "Spot"}), \
                patch.object(control, "job_state", return_value={"execution": {"executionState": "Failed"}}), \
                patch.object(control, "az_json") as az, redirect_stdout(io.StringIO()):
            result = control.status(self.args)
        az.assert_not_called()
        self.assertEqual(result["vm"]["power"], "PowerState/deallocated")
        self.assertEqual(result["job"]["execution"]["executionState"], "Failed")
        self.assertTrue((self.root / "managed_status.json").is_file())


if __name__ == "__main__":
    unittest.main()