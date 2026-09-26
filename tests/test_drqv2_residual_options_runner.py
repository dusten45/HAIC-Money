import tempfile
import unittest
from pathlib import Path

import numpy as np

from haic.algorithms.drq_v2 import ResidualOption, select_greedy_option
from scripts.drqv2_residual_options import (
    exploration_values,
    load_protocol,
    reserved_geometry_seeds,
    sha256_file,
    summarize_cells,
    validate_base_actor,
)


ROOT = Path(__file__).resolve().parents[1]


class TestResidualOptionRunner(unittest.TestCase):
    def test_protocol_development_seeds_exclude_known_drq_partitions(self):
        protocol_path = ROOT / "experiments/drqv2-residual-options-pilot-v1.json"
        protocol = load_protocol(protocol_path)
        exclusions, source_hashes = reserved_geometry_seeds(protocol, ROOT)
        seeds = protocol["development_screen"]["geometry_seeds"]
        self.assertEqual(len(seeds), 8)
        self.assertTrue(set(protocol["development_screen"]["track_ids"]).isdisjoint(seeds))
        self.assertTrue(set(seeds).isdisjoint(exclusions - set(seeds)))
        self.assertEqual(len(source_hashes), len(protocol["exclusion_protocols"]))

    def test_v2_reuses_only_its_declared_consumed_development_seeds(self):
        protocol_path = ROOT / "experiments/drqv2-residual-options-pilot-v2.json"
        protocol = load_protocol(protocol_path)
        exclusions, _ = reserved_geometry_seeds(protocol, ROOT)
        seeds = set(protocol["development_screen"]["geometry_seeds"])
        self.assertEqual(seeds, set(protocol["reused_development_geometry_seeds"]))
        self.assertTrue(seeds.issubset(exclusions))
        self.assertEqual(protocol["training"]["intervention_margin_q_minus_keep"], 3.4834041595458984)

    def test_exploration_values_force_exactly_the_requested_option(self):
        for option in ResidualOption:
            values = exploration_values(option)
            self.assertIs(select_greedy_option(values), option)
            self.assertEqual(np.count_nonzero(values == 1e6), 1)

    def test_base_actor_validation_pins_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.bin"
            path.write_bytes(b"frozen actor test")
            digest = sha256_file(path)
            self.assertEqual(validate_base_actor(path, digest), digest)
            with self.assertRaises(ValueError):
                validate_base_actor(path, "0" * 64)

    def test_summary_counts_finishes_and_reports_finisher_time_only(self):
        summary = summarize_cells([
            {"finished": True, "progress": 1.0, "raw_reward": 100.0, "damage": 0.1, "lap_time_ms": 1000},
            {"finished": False, "progress": 0.4, "raw_reward": -5.0, "damage": 0.2, "lap_time_ms": None},
        ])
        self.assertEqual(summary["cells"], 2)
        self.assertEqual(summary["finish_count"], 1)
        self.assertEqual(summary["mean_lap_time_ms_finishers"], 1000.0)
        self.assertAlmostEqual(summary["mean_progress"], 0.7)


if __name__ == "__main__":
    unittest.main()
