import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.benchmark_corridor import _runtime_agent, load_direct_site_episodes, parse_args
from haic_research.config import load_config


class TestCorridorBenchmark(unittest.TestCase):
    def test_load_direct_site_map_preserves_added_obstacles(self):
        payload = {
            "schema_version": 1,
            "map_kind": "official",
            "map_id": "official-track1-seed42-plus-obstacle",
            "track_id": 1,
            "seed": 42,
            "obstacle_mode": "official_plus_custom",
            "obstacles": [{"progress": 0.145, "lateral": 0.6, "radius": 1.2}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            episodes = load_direct_site_episodes([path])

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].site_map.obstacle_mode, "official_plus_custom")
        self.assertEqual(episodes[0].site_map.obstacles[0].progress, 0.145)
        self.assertEqual(episodes[0].seed, 42)

    def test_cli_accepts_multiple_direct_site_maps(self):
        with patch(
            "sys.argv",
            ["benchmark_corridor", "--site-map", "first.json", "--site-map", "second.json"],
        ):
            args = parse_args()

        self.assertEqual(args.site_maps, [Path("first.json"), Path("second.json")])

    def test_registered_diagnostic_accepts_apex_and_matched_controls(self):
        with patch("sys.argv", ["benchmark_corridor", "--profile", "selected_full_road_guard",
                                "--profile", "fast_pedal_stable", "--profile", "apex_line"]):
            args = parse_args()
        config = load_config(Path(__file__).resolve().parents[1])

        self.assertEqual(args.profiles,
                         ["selected_full_road_guard", "fast_pedal_stable", "apex_line"])
        self.assertTrue(set(args.profiles).issubset(set(config.command_profiles[
            "benchmark_corridor_diagnostic"]["arguments"]["profile"]["choices"])))

    def test_apex_diagnostic_loads_the_exact_historical_policies(self):
        agent = _runtime_agent("apex_line")
        self.assertEqual(type(agent).__name__, "ApexLineAgent")

    def test_anticipatory_bend_is_registered_and_wraps_selected_control(self):
        with patch("sys.argv", ["benchmark_corridor", "--profile", "selected_full_road_guard",
                                "--profile", "anticipatory_bend"]):
            args = parse_args()
        config = load_config(Path(__file__).resolve().parents[1])
        self.assertIn("anticipatory_bend", args.profiles)
        self.assertIn("anticipatory_bend", config.command_profiles[
            "benchmark_corridor_diagnostic"]["arguments"]["profile"]["choices"])
        agent = _runtime_agent("anticipatory_bend")
        self.assertEqual(type(agent).__name__, "AnticipatoryBendAgent")
        self.assertEqual(type(agent.base).__name__, "ObstacleFullRoadGuardAgent")


if __name__ == "__main__":
    unittest.main()
