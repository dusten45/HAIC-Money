import json
import tempfile
import unittest
from pathlib import Path


class TestSimulatorCli(unittest.TestCase):
    def test_map_and_run_commands_create_artifacts(self):
        from local_simulator.logging import load_run_log
        from local_simulator.map import main as map_main
        from local_simulator.run import main as run_main

        with tempfile.TemporaryDirectory() as root:
            map_path = Path(root) / "map.json"
            run_path = Path(root) / "run.json"
            map_main(
                [
                    "--track-id",
                    "1",
                    "--seed",
                    "42",
                    "--output",
                    str(map_path),
                ]
            )
            run_main(
                [
                    "--map",
                    str(map_path),
                    "--output",
                    str(run_path),
                    "--max-steps",
                    "1",
                ]
            )

            self.assertTrue(map_path.exists())
            self.assertTrue(run_path.exists())
            self.assertEqual(load_run_log(run_path).schema_version, 1)

            map_payload = json.loads(map_path.read_text(encoding="utf-8"))
            self.assertGreater(len(map_payload["preview"]["track"]["points"]), 0)
            self.assertEqual(len(map_payload["preview"]["official_obstacles"]), 6)

    def test_map_command_accepts_custom_obstacles(self):
        from local_simulator.map import main as map_main

        with tempfile.TemporaryDirectory() as root:
            map_path = Path(root) / "custom-map.json"
            map_main(
                [
                    "--track-id",
                    "1",
                    "--seed",
                    "42",
                    "--obstacle-mode",
                    "custom_only",
                    "--obstacle",
                    "0.42,-0.25,1.2",
                    "--output",
                    str(map_path),
                ]
            )

            payload = json.loads(map_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["obstacle_mode"], "custom_only")
            self.assertEqual(payload["obstacles"][0]["progress"], 0.42)
            self.assertEqual(len(payload["preview"]["official_obstacles"]), 0)
            self.assertEqual(len(payload["preview"]["custom_obstacles"]), 1)


if __name__ == "__main__":
    unittest.main()
