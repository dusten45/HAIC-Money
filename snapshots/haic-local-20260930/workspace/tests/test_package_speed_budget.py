import tempfile
import unittest
import zipfile
from pathlib import Path

from training.package_speed_budget import build_package


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/haic-research-v2/speed-coupled-preview-candidate-20260929/submission-speed-coupled-preview.zip"
SOURCE = ROOT / "haic_agent/high_speed_pivot_runtime.py"


class SpeedBudgetPackageTests(unittest.TestCase):
    def test_exact_base_and_controller_are_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.zip"
            manifest = build_package(BASE, SOURCE, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read("haic_agent/high_speed_pivot_runtime.py"), SOURCE.read_bytes())
                self.assertIn(b'HighSpeedPivotAgent(speed, "speed_budget")', archive.read("agent.py"))
                self.assertEqual(archive.read("policy.pt"), zipfile.ZipFile(BASE).read("policy.pt"))
            self.assertEqual(manifest["official_submission"], False)


if __name__ == "__main__":
    unittest.main()
