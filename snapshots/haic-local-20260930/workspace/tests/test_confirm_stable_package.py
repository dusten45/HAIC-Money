"""Package comparison counts finish-line completion rather than progress."""

from pathlib import Path
import unittest

from training.confirm_stable_package import check_archive, summarize


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/haic-research-v2/stable-risk-frozen-package-20260928/submission.zip"


class StablePackageConfirmationTests(unittest.TestCase):
    def test_full_progress_without_finish_is_incomplete(self):
        common = {"collisions": 0, "damage": 0.0, "invalid_actions": 0,
                  "import_create_s": 1.0, "reset_s": .01, "peak_rss_bytes": 1000,
                  "act_max_ms": 10.0, "act_p95_ms": 5.0, "risk_clear_decisions": 0}
        rows = [dict(common, completed=True, lapTimeMs=22000, progress=.99),
                dict(common, completed=False, lapTimeMs=None, progress=1.0)]
        summary = summarize(rows)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["mean_incomplete_progress"], 1.0)
        self.assertEqual(summary["median_finished_lap_ms"], 22000)

    def test_exact_candidate_archive_passes_static_audit(self):
        result = check_archive(CANDIDATE)
        self.assertEqual(result["files"], 17)
        self.assertLess(result["archive_bytes"], 500_000_000)


if __name__ == "__main__":
    unittest.main()
