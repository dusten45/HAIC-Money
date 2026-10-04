"""Synthetic path and exclusive-write checks for G1 protocol freezing."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import freeze_rlpd_g1_newhost as freeze


class FreezeG1NewHostTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "experiments").mkdir()
        (self.root / "runs").mkdir()
        root_patch = patch.object(freeze, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_unsafe_and_symlinked_paths_never_open(self):
        for path in ("../experiments/protocol.json", "experiments/../protocol.json",
                     "/tmp/protocol.json", "runs/protocol.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                freeze.within(path, first="experiments")
        (self.root / "experiments/linked.json").symlink_to(self.root / "runs")
        with self.assertRaises(ValueError):
            freeze.within("experiments/linked.json", first="experiments")

    def test_receipts_are_exclusive_and_hashed(self):
        relative = "experiments/g1-freeze-test.json"
        expected = freeze.write_exclusive(relative, {"status": "frozen"})
        self.assertEqual(expected, freeze.digest(self.root / relative))
        self.assertEqual(json.loads((self.root / relative).read_text())["status"], "frozen")
        with self.assertRaises(FileExistsError):
            freeze.write_exclusive(relative, {"status": "replaced"})

    def test_missing_claims_do_not_freeze_a_new_protocol(self):
        (self.root / "experiments/train-seed-claims").mkdir()
        with self.assertRaises(FileNotFoundError):
            freeze.claims_sha(freeze.cells())
        self.assertFalse((self.root / "experiments/rlpd-g1-newhost-20260929-v1.json").exists())


if __name__ == "__main__":
    unittest.main()
