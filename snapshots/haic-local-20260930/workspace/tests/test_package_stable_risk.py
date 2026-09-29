"""Exact-byte package checks for the measured road-clearance candidate."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from training.package_stable_risk import build_package, digest


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip"
SOURCE = ROOT / "haic_agent/stable_risk_envelope_runtime.py"


class StableRiskPackageTests(unittest.TestCase):
    def test_exact_candidate_wraps_frozen_control(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.zip"
            manifest = build_package(CONTROL, SOURCE, output)
            with zipfile.ZipFile(CONTROL) as control, zipfile.ZipFile(output) as candidate:
                self.assertEqual(set(candidate.namelist()), set(control.namelist()) | {
                    "haic_agent/stable_risk_envelope_runtime.py"
                })
                self.assertEqual(candidate.read("policy.pt"), control.read("policy.pt"))
                entry = candidate.read("agent.py").decode("utf-8")
                self.assertIn("StableRiskEnvelopeAgent(ObstacleFullRoadGuardAgent(base))", entry)
                self.assertEqual(candidate.read("haic_agent/stable_risk_envelope_runtime.py"), SOURCE.read_bytes())
            self.assertEqual(manifest["archive_sha256"], digest(output.read_bytes()))
            self.assertEqual(json.loads((output.parent / "package_manifest.json").read_text(encoding="utf-8")), manifest)

    def test_changed_source_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "candidate.py"
            altered.write_bytes(SOURCE.read_bytes() + b"\n# altered\n")
            with self.assertRaisesRegex(ValueError, "differs from confirmation"):
                build_package(CONTROL, altered, Path(directory) / "submission.zip")


if __name__ == "__main__":
    unittest.main()
