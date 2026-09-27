"""Synthetic-only tests; never reset a real driving environment."""

import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts.audit_drq_final_source_coverage import (
    BASE, PAIRED_DRIFT_CONTRACT, PHASES, action_distribution, actor_encoder_drift,
    encoder_action_drift_summary, neighbor_counts, phase_plan, reconstruct, replay_valid_starts,
    representative_actions, run, sparse_parity,
)
from scripts.diagnose_drq_geometry_regression import digest
from scripts.diagnose_drq_source_states import _state_sha


class TestOfflineFinalSourceCoverage(unittest.TestCase):
    def test_phase_windows_overlap_and_missing_recovery_is_explicit(self):
        n = 100
        progress = np.linspace(0, 1, n, dtype=np.float32)
        corner = np.zeros(n)
        corner[7], corner[42], corner[76] = .4, .5, .3
        labels, detail = phase_plan(n, progress, np.zeros(n, dtype=np.int16), corner, np.zeros(n))
        self.assertEqual(labels["early"], list(range(1, 21)))
        self.assertEqual(labels["middle"], list(range(40, 61)))
        self.assertEqual(labels["late"], list(range(75, 91)))
        self.assertEqual(detail["representatives"]["major_corner"], [43])
        self.assertEqual(detail["representatives"]["recovery_proxy"], [])
        self.assertIn(91, labels["finish_approach"])
        self.assertEqual(len(set(labels["late"]) & set(labels["finish_approach"])), 10)
        self.assertEqual(set(labels), set(PHASES))

    def test_recovery_counter_and_heading_are_observational_not_outcomes(self):
        counter = np.zeros(30, dtype=np.int16)
        counter[1:4] = 1
        heading = np.zeros(30)
        heading[9] = -.4
        labels, detail = phase_plan(30, np.zeros(30), counter, np.zeros(30), heading)
        self.assertEqual(labels["recovery_proxy"], [2, 5, 10])
        self.assertEqual(detail["representatives"]["recovery_proxy"], [2, 5, 10])
        self.assertEqual((detail["counter_rising"], detail["counter_falling"],
                          detail["heading_proxy_count"]), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            phase_plan(30, np.zeros(30), counter, np.full(30, np.nan), heading)

    def test_full_scan_covers_all_terminal_safe_starts_not_arbitrary_sequences(self):
        valid = np.ones(100000, dtype=bool)
        valid[[0, 25, 50, 75, 100, 125]] = False
        state = {"capacity": 100000, "n_step": 3, "gamma": .99, "action_dim": 3,
                 "size": 100000, "next_sequence": 131072}
        with patch("scripts.audit_drq_final_source_coverage.valid_starts", return_value=(valid, None)):
            starts = replay_valid_starts(state)
        self.assertEqual(len(starts), 99994)
        self.assertTrue(np.array_equal(starts, np.flatnonzero(valid) + 31072))
        self.assertEqual(starts[0], 31073)
        self.assertEqual(starts[-1], 131071)
        with patch("scripts.audit_drq_final_source_coverage.valid_starts", return_value=(valid[:-1], None)):
            with self.assertRaisesRegex(ValueError, "99994"):
                replay_valid_starts(state)

    def test_distribution_is_descriptive_and_rejects_bad_actions(self):
        source = np.zeros((2, 3), dtype=np.float32)
        stored = np.asarray([[0, .1, 0], [.2, 0, 0]], dtype=np.float32)
        result = action_distribution(source, stored)
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["native_linf_mean"], .15, places=6)
        self.assertEqual(result["near_0_10_n"], 1)
        with self.assertRaisesRegex(ValueError, "distribution"):
            action_distribution(source, np.full((2, 3), np.nan))

    def test_full_pool_and_actual_sampled_neighbor_counts_have_separate_denominators(self):
        cosine = np.asarray([.97, .96, .90, .99], dtype=np.float32)
        stored = np.asarray([[0, 0, 0], [.2, 0, 0], [0, 0, 0], [.05, 0, 0]], dtype=np.float32)
        weights = {"uniform": np.asarray([2, 0, 4, 1]),
                   "failure_weighted": np.asarray([0, 3, 0, 1])}
        full, arms = neighbor_counts(cosine, stored, np.zeros(3, dtype=np.float32), weights)
        self.assertEqual(full, {"valid_starts_n": 4, "feature_neighbors_cosine_ge_0_95_n": 3,
                                "feature_and_stored_action_near_0_10_n": 2})
        self.assertEqual(arms["uniform"], {
            "sampled_source_unique_starts_n": 3, "sampled_source_slots_n": 7,
            "feature_neighbors_cosine_ge_0_95_unique_n": 2,
            "feature_neighbors_cosine_ge_0_95_slots_n": 3,
            "feature_and_stored_action_near_0_10_unique_n": 2,
            "feature_and_stored_action_near_0_10_slots_n": 3,
        })
        self.assertEqual(arms["failure_weighted"]["feature_neighbors_cosine_ge_0_95_slots_n"], 4)
        self.assertEqual(arms["failure_weighted"]["feature_and_stored_action_near_0_10_slots_n"], 1)
        with self.assertRaisesRegex(ValueError, "sampled source weights"):
            neighbor_counts(cosine, stored, np.zeros(3, dtype=np.float32), {"uniform": -weights["uniform"]})

    def test_representative_actor_forward_is_singleton_even_when_batch_changes_output(self):
        tensor = torch.zeros((3, 4, 84, 84), dtype=torch.float32)
        actor = Mock(side_effect=lambda batch: torch.zeros((len(batch), 3)) if len(batch) == 1
                     else torch.ones((len(batch), 3)))
        self.assertTrue(np.array_equal(representative_actions(actor, tensor), np.zeros((3, 3))))
        self.assertEqual([call.args[0].shape for call in actor.call_args_list], [(1, 4, 84, 84)] * 3)
        self.assertTrue(torch.equal(actor(tensor), torch.ones((3, 3))))
        with self.assertRaisesRegex(ValueError, "representative source observation"):
            representative_actions(actor, tensor[:, 0])

    def test_encoder_drift_identical_and_shifted_same_float32_batch(self):
        source = np.zeros((2, 256), dtype=np.float32)
        source[0, 0] = 1.0
        source[1, 1] = 1.0
        identical = actor_encoder_drift(source, source.copy())
        np.testing.assert_allclose(identical["mse"], [0.0, 0.0], atol=0)
        np.testing.assert_allclose(identical["cosine"], [1.0, 1.0], atol=1e-15)
        np.testing.assert_allclose(identical["source_normalized_l2"], [0.0, 0.0], atol=0)
        changed = source.copy()
        changed[0, 0] = 2.0
        changed[1, 0] = 1.0
        drift = actor_encoder_drift(source, changed)
        np.testing.assert_allclose(drift["mse"], [1 / 256, 1 / 256], atol=1e-15)
        np.testing.assert_allclose(drift["cosine"], [1.0, 1 / np.sqrt(2)], atol=1e-15)
        np.testing.assert_allclose(drift["source_normalized_l2"], [1.0, 1.0], atol=1e-15)
        with self.assertRaisesRegex(ValueError, "undefined cosine"):
            actor_encoder_drift(source, np.zeros_like(source))
        with self.assertRaisesRegex(ValueError, "paired actor encoder"):
            actor_encoder_drift(source, np.zeros((2, 128)))

    def test_paired_phase_aggregate_is_descriptive_with_explicit_zero_count(self):
        rows = [{"r7b_vs_source_native_linf": {"uniform": action},
                 "r7b_actor_encoder_drift": {"uniform": {"mse": mse, "cosine": cosine,
                                                         "source_normalized_l2": normalized}}}
                for action, mse, cosine, normalized in
                ((.1, .0, 1.0, .0), (.3, .5, .5, 1.0), (.5, 1.0, .0, 2.0))]
        result = encoder_action_drift_summary(rows, "uniform")
        self.assertEqual(result["paired_representative_n"], 3)
        self.assertAlmostEqual(result["action_native_linf"]["median"], .3)
        self.assertAlmostEqual(result["encoder_mse"]["mean"], .5)
        self.assertAlmostEqual(result["encoder_cosine"]["median"], .5)
        self.assertAlmostEqual(result["encoder_source_normalized_l2"]["mean"], 1.0)
        self.assertEqual(encoder_action_drift_summary([], "uniform"), {
            "paired_representative_n": 0,
            **{field: {"mean": None, "median": None} for field in
               ("action_native_linf", "encoder_mse", "encoder_cosine", "encoder_source_normalized_l2")},
        })
        self.assertIn("no causal relationship", PAIRED_DRIFT_CONTRACT["interpretation"])
        self.assertIn("threshold tuning", PAIRED_DRIFT_CONTRACT["interpretation"])
        self.assertIn("identical ordered batched float32 tensor", PAIRED_DRIFT_CONTRACT["features"])

    def test_sparse_road_parity_requires_every_field(self):
        trace = {"sparse_nearest_point_index": np.array([2]),
                 "sparse_nearest_point_fraction": np.array([.5], dtype=np.float32),
                 "sparse_nearest_distance_m": np.array([1.], dtype=np.float32),
                 "sparse_speed_m_s": np.array([3.], dtype=np.float32)}
        measured = {"nearest_point_index": 2, "nearest_point_fraction": .5,
                    "nearest_distance_m": 1., "speed_m_s": 3.}
        with patch("scripts.audit_drq_final_source_coverage._road_sample", return_value=measured):
            sparse_parity(Mock(), trace, 0, 0)
            with self.assertRaisesRegex(ValueError, "nearest_distance_m"):
                sparse_parity(Mock(), {**trace, "sparse_nearest_distance_m": np.array([2.])}, 0, 0)

    def test_preflight_only_writes_nothing_and_input_failure_never_resets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / BASE / "offline-coverage"
            with patch("scripts.audit_drq_final_source_coverage.preflight", return_value={}), \
                 patch("scripts.audit_drq_final_source_coverage.build_environment") as env:
                self.assertEqual(run(root, output, preflight_only=True)["environment_resets"], 0)
                self.assertFalse((root / BASE).exists())
                env.assert_not_called()
            with patch("scripts.audit_drq_final_source_coverage.preflight", side_effect=ValueError("bad hash")), \
                 patch("scripts.audit_drq_final_source_coverage.build_environment") as env:
                with self.assertRaisesRegex(ValueError, "bad hash"):
                    run(root, output)
                self.assertFalse((root / BASE).exists())
                env.assert_not_called()
            output.mkdir(parents=True)
            with patch("scripts.audit_drq_final_source_coverage.preflight") as preflight:
                with self.assertRaisesRegex(ValueError, "NEW directory"):
                    run(root, output, preflight_only=True)
                preflight.assert_not_called()

    def test_new_drift_directory_preserves_existing_primary_receipt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / BASE / "offline-coverage" / "result.json"
            old.parent.mkdir(parents=True)
            old.write_text("sealed primary receipt\n", encoding="utf-8")
            drift = root / BASE / "offline-coverage-drift"
            with patch("scripts.audit_drq_final_source_coverage.preflight", return_value={}):
                status = run(root, drift, preflight_only=True)
                self.assertEqual(status["paired_primary_representative_frames"], 187)
                with self.assertRaisesRegex(ValueError, "NEW directory"):
                    run(root, old.parent, preflight_only=True)
            self.assertEqual(old.read_text(encoding="utf-8"), "sealed primary receipt\n")
            self.assertFalse(drift.exists())

    def test_last_episode_parity_failure_precedes_every_source_replay_scan(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / BASE / "offline-coverage"
            steps = [541, 518, 627, 580, 637, 627, 450, 457, 474, 553, 534]
            roads = [1, 2, 3, 4, 5, 6, 3, 4, 7, 8, 9]
            self.assertEqual(sum(steps), 5998)
            selected = [{"source_learner_seed": int(j >= 6), "geometry_seed": roads[j],
                         "steps": steps[j], "trace_path": f"traces/{j}.npz"} for j in range(11)]
            context = {"r7": {"diagnostic_cache": {"path": "first20.npz"}},
                       "protocol": {"source_actors": [{"source_seed": seed} for seed in (0, 1)]},
                       "selected": selected, "inputs": {"road": {road: {} for road in roads}},
                       "refs": {}}
            archive = nullcontext({"observations": [None] * 11,
                                   "native_action": np.zeros((650, 3), dtype=np.float32),
                                   "official_action": np.zeros((650, 3), dtype=np.float32)})
            def parity_result(_, source, *_args):
                if source is selected[-1]:
                    raise ValueError("final episode telemetry mismatch")
                return ({"source_seed": source["source_learner_seed"], "geometry_seed": source["geometry_seed"],
                         "steps": source["steps"], "phase_representatives": {p: [1] for p in PHASES}},
                        {1: np.zeros((4, 84, 84), dtype=np.float32)})

            with patch("scripts.audit_drq_final_source_coverage.preflight", return_value=context), \
                 patch("scripts.audit_drq_final_source_coverage.verified", return_value=root / "first20.npz"), \
                 patch("scripts.audit_drq_final_source_coverage.np.load", return_value=archive), \
                 patch("scripts.audit_drq_final_source_coverage._model", return_value={}), \
                 patch("scripts.audit_drq_final_source_coverage.reconstruct", side_effect=parity_result) as reconstruct_mock, \
                 patch("scripts.audit_drq_final_source_coverage.score_seed") as score:
                with self.assertRaisesRegex(ValueError, "final episode telemetry mismatch"):
                    run(root, output)
                self.assertEqual(reconstruct_mock.call_count, 11)
                score.assert_not_called()
                self.assertFalse(output.exists())

    def test_reconstruct_only_applies_archived_official_actions_and_aborts_on_parity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/traces/cell.npz"
            trace_path.parent.mkdir(parents=True)
            frame = np.zeros((4, 84, 84), dtype=np.float32)
            native = np.zeros((20, 3), dtype=np.float32)
            official = np.tile(np.asarray([.25, .5, 0.], dtype=np.float32), (20, 1))
            arrays = {"reward": np.zeros(20, dtype=np.float32),
                      "progress": np.linspace(0, 1, 20, dtype=np.float32),
                      "off_track_counter": np.zeros(20, dtype=np.int16),
                      "native_action": native, "official_action": official,
                      "actor_sha256": np.full(20, "actor"),
                      "source_actor_sha256": np.full(20, "actor"),
                      "checkpoint_sha256": np.full(20, "checkpoint"),
                      "source_checkpoint_sha256": np.full(20, "checkpoint"),
                      "sparse_step": np.array([0, 1, 20], dtype=np.int32),
                      "finished": np.asarray([False] * 19 + [True])}
            np.savez_compressed(trace_path, **arrays)
            source = {"trace_path": "traces/cell.npz", "trace_sha256": digest(trace_path),
                      "source_learner_seed": 0, "geometry_seed": 123, "steps": 20, "family": "synthetic"}
            adapter = Mock()
            adapter.to_official.return_value = official[0]
            spec = Mock()
            spec.validate.side_effect = lambda value: value
            model = {"actor_sha256": "actor", "checkpoint_sha256": "checkpoint",
                     "actor": Mock(side_effect=lambda value: torch.zeros((len(value), 3))),
                     "adapter": adapter, "observation_spec": spec}
            refs = {(0, 123, i, variant): {"observation_sha256": _state_sha(frame),
                     "source_trace_sha256": source["trace_sha256"], "source_native_action": [0, 0, 0],
                     "road_pre_action": {"nearest_point_index": 0, "road_relative_heading_rad": 0}}
                    for i in range(1, 21) for variant in ("uniform", "failure_weighted", "easy_retention")}

            def environment(*_):
                env = Mock()
                env.unwrapped.track = [[0, 0, 0, 0], [0, 0, 1, 0]]
                env.reset.return_value = (frame, {"seed": 123, "track_id": 1})
                calls = 0

                def step(action):
                    nonlocal calls
                    self.assertTrue(np.array_equal(action, official[calls]))
                    calls += 1
                    return frame, 0., calls == 20, False, {"seed": 123, "track_id": 1,
                                                             "finished": calls == 20}

                env.step.side_effect = step
                return env

            with patch("scripts.audit_drq_final_source_coverage.build_environment", side_effect=environment) as build, \
                 patch("scripts.audit_drq_final_source_coverage._road_hash", return_value="road"), \
                 patch("scripts.audit_drq_final_source_coverage._road_state", return_value={
                     "road_relative_heading_rad": 0., "nearest_point_index": 0}), \
                 patch("scripts.audit_drq_final_source_coverage._find_wrapper", return_value=Mock(
                     max_off_track_steps=100, warmup_steps=50, off_track_counter=0)), \
                 patch("scripts.audit_drq_final_source_coverage.sparse_parity"), \
                 patch("scripts.audit_drq_final_source_coverage.check_step") as check:
                cell, frames = reconstruct(root, source, {"road_coordinate_sha256": "road"}, model, refs,
                                           np.stack([frame] * 20))
                self.assertEqual(cell["steps"], 20)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(check.call_count, 20)
                self.assertEqual(cell["phase_counts"]["recovery_proxy"], 0)
                self.assertTrue(frames)
                with patch("scripts.audit_drq_final_source_coverage.check_step", side_effect=ValueError("reward mismatch")):
                    with self.assertRaisesRegex(ValueError, "reward mismatch"):
                        reconstruct(root, source, {"road_coordinate_sha256": "road"}, model, refs,
                                    np.stack([frame] * 20))


if __name__ == "__main__":
    unittest.main()
