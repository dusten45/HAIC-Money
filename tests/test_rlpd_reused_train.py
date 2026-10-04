"""Synthetic safety checks for the separate reused-TRAIN RLPD operator."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import diagnose_rlpd_reused_train as operator


class ReusedTrainOperatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "experiments").mkdir()
        (self.root / "runs").mkdir()
        self.patch_root = patch.object(operator, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def test_single_run_destination_is_exclusive(self):
        with self.assertRaises(ValueError):
            operator.located("../runs/previous", "runs")
        with self.assertRaises(ValueError):
            operator.located("runs/../previous", "runs")
        with self.assertRaises(ValueError):
            operator.located("experiments/study.json", "runs")
        output = operator.located("runs/new-attempt", "runs")
        output.mkdir(exist_ok=False)
        with self.assertRaises(FileExistsError):
            output.mkdir(exist_ok=False)

    def test_source_files_allow_root_entries_but_not_symlinks(self):
        (self.root / "agent.py").write_text("old frozen source", encoding="utf-8")
        self.assertEqual(operator.source_file("agent.py"), self.root / "agent.py")
        with self.assertRaises(ValueError):
            operator.source_file("../agent.py")
        (self.root / "link.py").symlink_to(self.root / "agent.py")
        with self.assertRaises(ValueError):
            operator.source_file("link.py")

    def test_attempt_receipt_writes_exclusively(self):
        output = self.root / "runs" / "attempt.json"
        operator.write_json(output, {"status": "attempting"})
        self.assertIn("attempting", output.read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            operator.write_json(output, {"status": "complete"})

    def test_unpinned_protocol_never_creates_run_directory(self):
        with self.assertRaises(ValueError):
            operator.drive("experiments/missing.json", "0" * 64, "runs/new-attempt")
        self.assertFalse((self.root / "runs" / "new-attempt").exists())


if __name__ == "__main__":
    unittest.main()
