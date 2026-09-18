import tempfile
import unittest
from pathlib import Path

from local_simulator.schema import MapSpec


class TestRunLog(unittest.TestCase):
    def _run_api(self):
        try:
            from local_simulator.logging import load_run_log, save_run_log
            from local_simulator.policies import BaselinePolicy
            from local_simulator.simulation import run_episode
        except ImportError as error:
            self.fail(f"run logging API is missing: {error}")
        return load_run_log, save_run_log, BaselinePolicy, run_episode

    def test_baseline_run_has_schema_and_steps(self):
        _, _, BaselinePolicy, run_episode = self._run_api()
        spec = MapSpec(1, 42, "custom_only", (), 2, 4)

        result = run_episode(spec, BaselinePolicy())

        self.assertEqual(result.schema_version, 1)
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].action, (0.0, 1.0, 0.0))
        self.assertGreater(len(result.track.points), 0)
        self.assertIn("lap_time_ms", result.summary)

    def test_log_round_trip(self):
        load_run_log, save_run_log, BaselinePolicy, run_episode = self._run_api()
        spec = MapSpec(1, 42, "custom_only", (), 1, 4)
        result = run_episode(spec, BaselinePolicy())

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "run.json"
            save_run_log(result, path)
            loaded = load_run_log(path)

        self.assertEqual(loaded.summary, result.summary)
        self.assertEqual(loaded.steps, result.steps)
        self.assertEqual(loaded.track, result.track)


if __name__ == "__main__":
    unittest.main()
