import hashlib
import unittest

import numpy as np

from common_adapter import ActionSpec, ObservationSpec
from haic.algorithms.rlpd.g0_diagnostic import (
    G0Cell,
    G0Decision,
    G0EventRules,
    G0Identity,
    G0Telemetry,
    classify_g0_episode,
    summarize_g0_pairs,
    validate_g0_trace,
)


class TestG0DiagnosticPreflight(unittest.TestCase):
    def setUp(self):
        self.payload = b"synthetic-actor-only"
        self.observation_spec = ObservationSpec()
        self.action_spec = ActionSpec()
        self.identity = G0Identity(
            actor_sha256=hashlib.sha256(self.payload).hexdigest(),
            source_sha256="a" * 64,
            protocol_sha256="b" * 64,
            observation_fingerprint=self.observation_spec.fingerprint,
            action_fingerprint=self.action_spec.fingerprint,
            action_mode="exported_tanh_mean",
            reset_contract="explicit-reset-and-four-frame-history",
        )
        self.cell = G0Cell("TRAIN", 3101, 1)
        self.decision = G0Decision(
            step=0,
            observation=np.zeros((4, 84, 84), dtype=np.float32),
            proposed_native_action=np.array([0.25, -0.5, 0], dtype=np.float32),
            executed_native_action=np.array([0.25, -0.5, 0], dtype=np.float32),
            applied_official_action=np.array([0.25, 0.25, 0.5], dtype=np.float32),
        )

    def validate(self, *, decisions=None, **overrides):
        arguments = dict(
            identity=self.identity,
            actor_payload=self.payload,
            observation_spec=self.observation_spec,
            action_spec=self.action_spec,
            cell=self.cell,
            allowed_train_cells={(1, 3101)},
            decisions=[self.decision] if decisions is None else decisions,
        )
        arguments.update(overrides)
        return validate_g0_trace(**arguments)

    def decision_with(self, **changes):
        return G0Decision(**{**self.decision.__dict__, **changes})

    def test_accepts_hash_bound_train_trace_and_native_to_official_pedals(self):
        second = self.decision_with(step=1)
        self.assertEqual(self.validate(decisions=[self.decision, second]), 2)

    def test_rejects_identity_mode_hash_and_adapter_drift(self):
        with self.assertRaisesRegex(ValueError, "actor bytes"):
            self.validate(actor_payload=b"different-actor")
        with self.assertRaisesRegex(ValueError, "adapters"):
            self.validate(observation_spec=ObservationSpec(control_plane_fingerprint="changed"))
        hwc = ObservationSpec(channel_order="HWC")
        hwc_identity = G0Identity(**{
            **self.identity.__dict__, "observation_fingerprint": hwc.fingerprint,
        })
        with self.assertRaisesRegex(ValueError, "CHW"):
            self.validate(identity=hwc_identity, observation_spec=hwc)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            G0Identity(**{**self.identity.__dict__, "action_mode": "stochastic_training"})
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            G0Identity(**{**self.identity.__dict__, "source_sha256": "bad"})

    def test_rejects_nontrain_and_unallocated_cells(self):
        for cell in (G0Cell("blind", 3101, 1), G0Cell("TRAIN", 3102, 1), G0Cell("TRAIN", 3101, 2)):
            with self.subTest(cell=cell), self.assertRaisesRegex(ValueError, "TRAIN-only"):
                self.validate(cell=cell)

    def test_train_allowlist_follows_protocol_track_then_geometry_order(self):
        protocol = {"training_cells": [{"track_id": 1, "geometry_seed": 3101}]}
        allowed = {
            (row["track_id"], row["geometry_seed"])
            for row in protocol["training_cells"]
        }
        self.assertEqual(self.validate(allowed_train_cells=allowed), 1)
        with self.assertRaisesRegex(ValueError, "TRAIN-only"):
            self.validate(allowed_train_cells={(3101, 1)})

    def test_rejects_substituted_actions_and_mismapped_pedals(self):
        with self.assertRaisesRegex(ValueError, "proposed and executed"):
            self.validate(decisions=[self.decision_with(
                executed_native_action=np.array([0.5, -0.5, 0], dtype=np.float32)
            )])
        with self.assertRaisesRegex(ValueError, "applied action"):
            self.validate(decisions=[self.decision_with(
                applied_official_action=np.array([0.25, 0.5, 0.25], dtype=np.float32)
            )])
        with self.assertRaisesRegex(ValueError, "declared coordinates"):
            self.validate(decisions=[self.decision_with(
                applied_official_action=np.array([0.25, 0.25, 0.5], dtype=np.float64)
            )])

    def test_rejects_missing_or_malformed_observations_and_trace_steps(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.validate(decisions=[])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.validate(decisions=[self.decision, self.decision_with(step=2)])
        with self.assertRaisesRegex(ValueError, "observation"):
            self.validate(decisions=[self.decision_with(
                observation=np.zeros((3, 84, 84), dtype=np.float32)
            )])


class TestG0EventClassification(unittest.TestCase):
    def setUp(self):
        self.rules = G0EventRules(
            max_decisions=2000,
            negative_reward_limit=100,
            stall_window=3,
            tile_window=3,
            directed_delta_epsilon=0.01,
        )

    def row(self, step, **changes):
        values = dict(
            step=step,
            summed_reward=1.0,
            new_tiles=1,
            directed_delta=1.0,
            centerline_far=False,
            contact=False,
            damage=0.0,
            progress=0.2,
            finish_qualified=False,
            finish_phase="not_qualified",
            finished=False,
            terminated=False,
            truncated=False,
            out_of_bounds=False,
            retire_reason=None,
        )
        values.update(changes)
        return G0Telemetry(**values)

    def test_negative_reward_threshold_and_counter_reset_do_not_mean_grass(self):
        rows = [self.row(i, summed_reward=-0.1) for i in range(101)]
        rows[-1] = self.row(100, summed_reward=-0.1, terminated=True, retire_reason="off_track")
        result = classify_g0_episode(rows, rules=self.rules, end_reason="retired")
        self.assertEqual(result.outcome, "off_track")
        self.assertEqual(result.final_negative_streak, 101)
        self.assertEqual(result.first_event_status, "none")
        premature = [self.row(i, summed_reward=-0.1) for i in range(100)]
        premature[-1] = self.row(99, summed_reward=-0.1, terminated=True, retire_reason="off_track")
        with self.assertRaisesRegex(ValueError, "negative-reward threshold"):
            classify_g0_episode(premature, rules=self.rules, end_reason="retired")
        late = [self.row(i, summed_reward=-0.1) for i in range(102)]
        late[-1] = self.row(101, summed_reward=-0.1, terminated=True, retire_reason="off_track")
        with self.assertRaisesRegex(ValueError, "continued past"):
            classify_g0_episode(late, rules=self.rules, end_reason="retired")

        reset = [self.row(0, summed_reward=-1), self.row(1, summed_reward=0)]
        reset.append(self.row(2, summed_reward=-1))
        result = classify_g0_episode(reset, rules=self.rules, end_reason="collection_censored")
        self.assertEqual(result.final_negative_streak, 1)

    def test_contact_precedes_reward_retirement_without_causal_claim(self):
        rows = [self.row(i, summed_reward=-1) for i in range(101)]
        rows[3] = self.row(3, summed_reward=-1, contact=True, damage=0.2)
        for i in range(4, 101):
            rows[i] = self.row(i, summed_reward=-1, damage=0.2)
        rows[-1] = self.row(100, summed_reward=-1, damage=0.2, terminated=True, retire_reason="off_track")
        result = classify_g0_episode(rows, rules=self.rules, end_reason="retired")
        self.assertEqual(result.first_observed_step, 3)
        self.assertEqual(result.first_observed_events, ("observed_contact",))
        self.assertEqual(result.outcome, "off_track")

    def test_qualified_high_progress_is_not_finish_and_finish_wins_over_truncation(self):
        rows = [self.row(i, summed_reward=-1, progress=1.0, finish_qualified=True) for i in range(101)]
        rows[-1] = self.row(
            100, summed_reward=-1, progress=1.0, finish_qualified=True,
            terminated=True, retire_reason="off_track",
        )
        result = classify_g0_episode(rows, rules=self.rules, end_reason="retired")
        self.assertTrue(result.qualified_nonfinish)
        self.assertEqual(result.unassessed_precursors, ("finish_phase",))
        self.assertEqual(result.outcome, "off_track")
        finish = self.row(0, progress=1.0, finish_qualified=True, finished=True, truncated=True)
        result = classify_g0_episode([finish], rules=self.rules, end_reason="finished")
        self.assertEqual(result.outcome, "finished")
        self.assertFalse(result.qualified_nonfinish)

    def test_missing_early_telemetry_does_not_force_first_cause(self):
        rows = [self.row(0, centerline_far=None), self.row(1, contact=True)]
        result = classify_g0_episode(rows, rules=self.rules, end_reason="collection_censored")
        self.assertEqual(result.outcome, "collection_censored")
        self.assertEqual(result.first_observed_step, 1)
        self.assertEqual(result.first_event_status, "unknown")
        self.assertEqual(result.missing_steps, (0,))
        with self.assertRaisesRegex(ValueError, "censoring is not a terminal failure"):
            classify_g0_episode([self.row(0, terminated=True)], rules=self.rules, end_reason="collection_censored")

    def test_directed_motion_is_inapplicable_when_centerline_proxy_is_far(self):
        result = classify_g0_episode(
            [self.row(0, centerline_far=True, directed_delta=None)],
            rules=self.rules, end_reason="collection_censored",
        )
        self.assertEqual(result.first_observed_events, ("centerline_distance_exceeds_proxy",))
        self.assertEqual(result.first_event_status, "observed")
        self.assertEqual(result.missing_steps, ())

    def test_tied_precursors_and_successful_parent_controls_remain_visible(self):
        rows = [self.row(i, new_tiles=0, directed_delta=0.0) for i in range(3)]
        rows[-1] = self.row(2, new_tiles=0, directed_delta=0.0, finished=True, terminated=True)
        result = classify_g0_episode(rows, rules=self.rules, end_reason="finished")
        self.assertEqual(result.first_event_status, "mixed")
        self.assertEqual(result.first_observed_events, ("low_directed_motion", "no_new_tiles"))
        self.assertEqual(result.outcome, "finished")

    def test_invalid_rules_and_cumulative_progress_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "stall_window"):
            G0EventRules(2000, 100, 0, 3, 0.01)
        with self.assertRaisesRegex(ValueError, "progress decreased"):
            classify_g0_episode(
                [self.row(0, progress=0.6), self.row(1, progress=0.5)],
                rules=self.rules, end_reason="collection_censored",
            )

    def test_fractional_damage_and_collision_driven_crash(self):
        rows = [self.row(i, contact=True, damage=round(0.2 * (i + 1), 1)) for i in range(5)]
        rows[-1] = self.row(4, contact=True, damage=1.0, terminated=True, retire_reason="crash")
        result = classify_g0_episode(rows, rules=self.rules, end_reason="retired")
        self.assertEqual(result.outcome, "crash")
        self.assertEqual(result.first_observed_events, ("observed_contact",))
        with self.assertRaisesRegex(ValueError, "final contact"):
            classify_g0_episode(
                [self.row(0, terminated=True, retire_reason="crash")],
                rules=self.rules, end_reason="retired",
            )
        with self.assertRaisesRegex(ValueError, "cannot decrease"):
            classify_g0_episode(
                [self.row(0, contact=True, damage=0.2), self.row(1, damage=0.0)],
                rules=self.rules, end_reason="collection_censored",
            )

    def test_nonboolean_terminal_flags_and_missing_phase_are_uncertain(self):
        with self.assertRaisesRegex(ValueError, "finished must be a boolean"):
            classify_g0_episode(
                [self.row(0, finished="false", truncated=True)],
                rules=self.rules, end_reason="finished",
            )
        result = classify_g0_episode(
            [self.row(0, finish_phase=None), self.row(1, contact=True)],
            rules=self.rules, end_reason="collection_censored",
        )
        self.assertEqual(result.missing_steps, (0,))
        self.assertEqual(result.first_event_status, "unknown")
        result = classify_g0_episode(
            [self.row(0, damage=0.2, contact=True), self.row(1, damage=None),
             self.row(2, damage=0.4, contact=False)],
            rules=self.rules, end_reason="collection_censored",
        )
        self.assertEqual(result.missing_steps, (1,))

    def test_measured_playfield_exit_is_distinct_from_reward_off_track(self):
        result = classify_g0_episode(
            [self.row(0, summed_reward=-100, terminated=True, out_of_bounds=True)],
            rules=self.rules, end_reason="out_of_bounds",
        )
        self.assertEqual(result.outcome, "out_of_bounds")
        with self.assertRaisesRegex(ValueError, "measured playfield exit"):
            classify_g0_episode(
                [self.row(0, terminated=True)], rules=self.rules, end_reason="out_of_bounds",
            )


class TestG0GeometryPairSummary(unittest.TestCase):
    def row(self, seed, track, actor, outcome, road="b" * 64):
        return {
            "partition": "TRAIN", "track_id": track, "geometry_seed": seed,
            "actor_id": actor, "actor_sha256": ("a" if actor == "seed11" else "c") * 64,
            "road_centerline_sha256": road, "steps": 120,
            "summary": {
                "outcome": outcome, "first_event_status": "observed",
                "first_observed_events": ["observed_contact"],
                "missing_steps": [], "qualified_nonfinish": False,
            },
        }

    def test_complete_pairs_count_geometry_clusters_not_duplicate_roads(self):
        cells = [(1, 41000), (2, 41001)]
        rows = [
            self.row(41000, 1, "seed11", "finished"),
            self.row(41000, 1, "seed50", "off_track"),
            self.row(41001, 2, "seed11", "collection_censored"),
            self.row(41001, 2, "seed50", "finished"),
        ]
        summary = summarize_g0_pairs(rows, cells=cells, actor_ids=("seed11", "seed50"))
        self.assertEqual(summary["geometry_clusters"], 2)
        self.assertEqual(summary["diagnostic_episodes"], 4)
        self.assertEqual(summary["outcomes_by_actor"]["seed11"]["collection_censored"], 1)
        self.assertEqual(summary["first_events_by_actor"]["seed50"]["observed_contact"], 2)
        self.assertEqual(summary["missing_decisions_by_actor"]["seed11"], 0)
        with self.assertRaisesRegex(ValueError, "same road bytes"):
            summarize_g0_pairs(
                [rows[0], self.row(41000, 1, "seed50", "off_track", road="d" * 64), *rows[2:]],
                cells=cells, actor_ids=("seed11", "seed50"),
            )
        with self.assertRaisesRegex(ValueError, "complete, distinct paired"):
            summarize_g0_pairs(rows[:3], cells=cells, actor_ids=("seed11", "seed50"))


if __name__ == "__main__":
    unittest.main()
