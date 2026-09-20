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
            self.assertEqual(load_run_log(run_path).schema_version, 2)

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

    def test_custom_map_command_creates_geometry_preview(self):
        from local_simulator.map import main as map_main

        with tempfile.TemporaryDirectory() as root:
            map_path = Path(root) / "custom-track-0001.json"
            map_main(
                [
                    "--kind",
                    "custom",
                    "--map-id",
                    "custom-track-0001",
                    "--design-seed",
                    "90421",
                    "--template",
                    "hairpin",
                    "--output",
                    str(map_path),
                ]
            )

            payload = json.loads(map_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["map_kind"], "custom")
            self.assertEqual(payload["map_id"], "custom-track-0001")
            self.assertGreaterEqual(len(payload["geometry"]["centerline"]), 12)
            self.assertIn("track", payload["preview"])
            self.assertEqual(payload["preview"]["official_obstacles"], [])

    def test_custom_design_seeds_write_different_centerlines(self):
        from local_simulator.map import main as map_main

        with tempfile.TemporaryDirectory() as root:
            first_path = Path(root) / "first.json"
            second_path = Path(root) / "second.json"
            common = ["--kind", "custom", "--template", "oval"]
            map_main(common + ["--design-seed", "1", "--output", str(first_path)])
            map_main(common + ["--design-seed", "2", "--output", str(second_path)])

            first = json.loads(first_path.read_text(encoding="utf-8"))
            second = json.loads(second_path.read_text(encoding="utf-8"))
            self.assertNotEqual(
                first["geometry"]["centerline"],
                second["geometry"]["centerline"],
            )

    def test_technical_custom_map_cli_writes_recipe_metadata(self):
        from local_simulator.map import main as map_main

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "technical.json"
            result = map_main(
                [
                    "--kind", "custom",
                    "--template", "technical",
                    "--design-seed", "42",
                    "--output", str(path),
                ]
            )
            self.assertEqual(result, 0)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["generator"]["template"], "technical")
            self.assertEqual(payload["generator"]["generator_version"], 3)
            self.assertGreaterEqual(payload["generator"]["corner_count"], 9)


if __name__ == "__main__":
    unittest.main()
