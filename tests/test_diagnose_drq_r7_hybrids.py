"""Synthetic-only contracts for the frozen r6 hybrid diagnostic."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from drq_v2 import DrQActor, DrQCritic
from scripts.diagnose_drq_source_states import _state_sha
from scripts.diagnose_drq_r7_hybrids import (
    N, actor_actions, axis_stats, check_step, critic_values, reference_index, run,
)


class TestFrozenHybrids(unittest.TestCase):
    def test_encoder_and_downstream_swap_and_endpoint_parity(self):
        torch.manual_seed(8)
        source, variant = DrQActor(), DrQActor()
        with torch.no_grad():
            variant.encoder.linear[1].bias.add_(0.4)
            variant.trunk[0].bias.add_(0.3)
            variant.policy.bias.add_(0.5)
        before = [tensor.clone() for tensor in (*source.parameters(), *variant.parameters())]
        pixels = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
        with torch.inference_mode():
            result = actor_actions(source, variant, pixels)
            self.assertTrue(torch.allclose(result[:, 0], source(pixels)))
            self.assertTrue(torch.allclose(result[:, 1], variant(pixels)))
            self.assertTrue(torch.allclose(result[:, 2], torch.tanh(
                variant.policy(variant.trunk(source.encoder(pixels))))))
            self.assertTrue(torch.allclose(result[:, 3], torch.tanh(
                source.policy(source.trunk(variant.encoder(pixels))))))
        self.assertEqual(result.shape, (2, 4, 3))
        self.assertFalse(torch.allclose(result[:, 2], result[:, 3]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, (*source.parameters(), *variant.parameters()))))

    def test_per_axis_threshold_is_inclusive_and_uses_twenty_decisions(self):
        diffs = np.zeros((N, 3), dtype=np.float32)
        diffs[0, 0] = 0.1
        diffs[1, 0] = 0.2
        diffs[:, 2] = 0.11
        metrics = axis_stats(diffs)
        self.assertEqual(metrics["steer"]["ge_0_10_count"], 2)
        self.assertEqual(metrics["brake"]["ge_0_10_count"], N)
        self.assertEqual(metrics["gas"]["ge_0_10_count"], 0)
        self.assertAlmostEqual(metrics["steer"]["mean"], .015, places=7)
        self.assertEqual(metrics["steer"]["median"], 0)
        with self.assertRaises(ValueError):
            axis_stats(diffs[:-1])
        with self.assertRaises(ValueError):
            axis_stats(np.full((N, 3), np.nan))

    def test_recovered_frame_hash_must_preserve_float32_input_bytes(self):
        frame = np.full((4, 84, 84), .5, dtype=np.float32)
        self.assertNotEqual(_state_sha(frame), _state_sha(frame.astype(np.uint8)))
        self.assertEqual(_state_sha(frame), _state_sha(frame.copy()))

    def test_critic_hybrids_match_corresponding_q_encoder_and_head(self):
        torch.manual_seed(9)
        source = [DrQCritic(), DrQCritic()]
        variant = [DrQCritic(), DrQCritic()]
        pixels = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
        actions = torch.rand((2, 4, 3)) * 2 - 1
        with torch.inference_mode():
            measured = critic_values(source, variant, pixels, actions)
            self.assertEqual(set(measured), {"source", "r6", "source_encoder_r6_head",
                                             "r6_encoder_source_head"})
            for i in range(2):
                src_features = source[i].encoder(pixels)
                r6_features = variant[i].encoder(pixels)
                for j in range(4):
                    self.assertTrue(np.allclose(measured["source"][..., i, j],
                        source[i].q(torch.cat((src_features, actions[:, j]), -1)).squeeze(-1).numpy()))
                    self.assertTrue(np.allclose(measured["r6"][..., i, j],
                        variant[i].q(torch.cat((r6_features, actions[:, j]), -1)).squeeze(-1).numpy()))
                    self.assertTrue(np.allclose(measured["source_encoder_r6_head"][..., i, j],
                        variant[i].q(torch.cat((src_features, actions[:, j]), -1)).squeeze(-1).numpy()))
                    self.assertTrue(np.allclose(measured["r6_encoder_source_head"][..., i, j],
                        source[i].q(torch.cat((r6_features, actions[:, j]), -1)).squeeze(-1).numpy()))
        with self.assertRaisesRegex(ValueError, "twin critic"):
            critic_values(source[:1], variant, pixels, actions)

    def test_full_step_parity_refuses_telemetry_or_terminal_mismatch(self):
        trace = {"reward": np.array([1], dtype=np.float32),
                 "progress": np.array([.25], dtype=np.float32),
                 "damage": np.array([0], dtype=np.float32),
                 "terminated": np.array([False]), "truncated": np.array([False]),
                 "terminal": np.array([False]), "finished": np.array([False]),
                 "collision": np.array([False]), "finish_qualified": np.array([False]),
                 "retirement": np.array([""]), "off_track_counter": np.array([1]),
                 "finish_crossing_time_s": np.array([np.nan])}
        info = {"progress": .25, "damage": 0., "finished": False}
        with patch("scripts.diagnose_drq_r7_hybrids.EpisodeCollector._is_terminal", return_value=False):
            check_step(trace, 0, 1., False, False, info, 1)
            with self.assertRaisesRegex(ValueError, "progress"):
                check_step(trace, 0, 1., False, False, {**info, "progress": .26}, 1)
            with self.assertRaisesRegex(ValueError, "off_track_counter"):
                check_step(trace, 0, 1., False, False, info, 2)
            with self.assertRaisesRegex(ValueError, "finish time"):
                check_step(trace, 0, 1., False, False, {**info, "finish_time_s": 1.}, 1)

    def test_reference_index_requires_all_eleven_early_cells(self):
        selected = [{"source_learner_seed": 0 if i < 6 else 1, "geometry_seed": i, "steps": 1200}
                    for i in range(11)]
        with self.assertRaisesRegex(ValueError, "incomplete v2"):
            reference_index({"driven_decisions": 5998, "new_policy_rollouts": 0,
                             "same_source_observation_records": 1998, "states": []}, selected)
        rows = [{"source_seed": s["source_learner_seed"], "geometry_seed": s["geometry_seed"],
                 "decision": d, "variant": v} for s in selected for d in range(1, N + 1)
                for v in ("uniform", "failure_weighted", "easy_retention")]
        rows.append(dict(rows[0]))
        rows.extend([{**rows[0], "decision": d} for d in range(21, 1998 - len(rows) + 21)])
        with self.assertRaisesRegex(ValueError, "duplicate v2"):
            reference_index({"driven_decisions": 5998, "new_policy_rollouts": 0,
                             "same_source_observation_records": 1998, "states": rows}, selected)

    def test_missing_pinned_replay_aborts_before_output_or_environment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_base = root / "runs" / "20260926-drqv2-retention-r7" / "hybrid"
            output = output_base / "first20-v1"
            with patch("scripts.diagnose_drq_r7_hybrids.OUTPUT_BASE", output_base), \
                 patch("scripts.diagnose_drq_r7_hybrids._preflight") as preflight, \
                 patch("scripts.diagnose_drq_r7_hybrids.build_environment") as environment:
                with self.assertRaises(FileNotFoundError):
                    run(root, output)
                preflight.assert_not_called()
                environment.assert_not_called()
                self.assertFalse(output_base.exists())


if __name__ == "__main__":
    unittest.main()
