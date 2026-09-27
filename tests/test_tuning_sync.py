"""Archive safety and output-integrity tests; no Azure requests."""

import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.sync_h100_tuning_results import extract_snapshot, snapshot


def archive(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as output:
        for name, content in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


class SnapshotTests(unittest.TestCase):
    def test_extract_nested_json_and_keep_user_notes(self):
        payload = archive({"trials/lr5e-6/result.json": b'{"best_step": 1}', "notes.md": b"remote note"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.md").write_text("user note")
            self.assertEqual(extract_snapshot(payload, root), 2)
            self.assertEqual(json.loads((root / "trials/lr5e-6/result.json").read_text()), {"best_step": 1})
            self.assertEqual((root / "notes.md").read_text(), "user note")

    def test_reject_traversal_and_large_model_files(self):
        for name in ("../outside.json", "/tmp/outside.json", "trials/a/best.pt", "data/state.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    extract_snapshot(archive({name: b"{}"}), Path(directory))

    def test_invalid_json_prevents_any_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                extract_snapshot(archive({"valid.json": b"{}", "bad.json": b"{"}), root)
            self.assertFalse((root / "valid.json").exists())

    def test_reject_symlink_members_and_local_symlink_escape(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as output:
            member = tarfile.TarInfo("evil.json")
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/outside"
            output.addfile(member)
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            with self.assertRaises(ValueError):
                extract_snapshot(buffer.getvalue(), root)
            (root / "trials").symlink_to(outside)
            with self.assertRaises(ValueError):
                extract_snapshot(archive({"trials/escape.json": b"{}"}), root)

    def test_snapshot_download_verifies_checksum_and_renders(self):
        payload = archive({"manifest.json": b'{"source": "test"}'})
        meta = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "files": 1}

        def invoke(args, script):
            if "SNAPSHOT_META=" in script:
                return "SNAPSHOT_META=" + json.dumps(meta)
            if "SNAPSHOT_BEGIN" in script:
                offset = int(re.search(r"skip=(\d+)", script)[1])
                return "SNAPSHOT_BEGIN\n" + base64.b64encode(payload[offset:offset + 2400]).decode() + "\nSNAPSHOT_END"
            return "SNAPSHOT_REMOVED"

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(local_results=Path(directory))
            with patch("scripts.sync_h100_tuning_results.azure.invoke", side_effect=invoke), \
                    patch("scripts.sync_h100_tuning_results.render_tuning_report", return_value={"status": "partial"}) as render:
                self.assertEqual(snapshot(args), {"status": "partial"})
            render.assert_called_once()
            state = json.loads((Path(directory) / "logs/h100-tuning-20260920/sync_status.json").read_text())
            self.assertFalse(state["test_results_present"])
            self.assertEqual(state["archive_sha256"], meta["sha256"])


if __name__ == "__main__":
    unittest.main()