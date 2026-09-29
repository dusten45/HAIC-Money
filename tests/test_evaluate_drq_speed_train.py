import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_drq_speed_train as speed


class DummyModel:
    def reset(self, observation):
        self.observation = observation

    def act(self, observation):
        return np.array([0.1, 0.9, 0.2], dtype=np.float32)


class DummyEnv:
    def __init__(self):
        self.unwrapped = self
        self.t = 4.0
        self.track_id = 1
        self.track_seed = 42
        self.track = [(0.0, 0.0, float(index), 0.0) for index in range(24)]
        specs = [type("Obstacle", (), {"position": (float(index), 0.0), "radius": 1.2})()
                 for index in range(6)]
        self.track_variables = type("TrackVariables", (), {"obstacles": specs})()
        self.obstacles = [None] * 6
        self.actions = []

    def reset(self):
        return np.zeros((4, 84, 84), dtype=np.float32), {"track_id": 1, "seed": 42}

    def step(self, action):
        self.actions.append(action.copy())
        return (np.zeros((4, 84, 84), dtype=np.float32), 1.0, True, False,
                {"finished": True, "finish_time_s": 5.0, "progress": 1.0,
                 "damage": 0.0, "retire_reason": None})


class SpeedStudyTests(unittest.TestCase):
    def test_frozen_control_and_treatment_use_one_official_mapping(self):
        base, treated = DummyEnv(), DummyEnv()
        control, baseline_trace = speed.run_episode(DummyModel(), base, 1, 42, "control", 0, 2000)
        treatment, treated_trace = speed.run_episode(DummyModel(), treated, 1, 42,
                                                      "light_brake_release", 0, 2000)
        self.assertTrue(control["finished"])
        self.assertEqual(control["lap_time_ms"], 1000)
        self.assertEqual(control["interventions"], 0)
        self.assertEqual(treatment["interventions"], 1)
        self.assertEqual(control["road_sha256"], treatment["road_sha256"])
        self.assertEqual(control["obstacle_sha256"], treatment["obstacle_sha256"])
        self.assertAlmostEqual(float(treated.actions[0][2]), 0.05, places=6)
        np.testing.assert_array_equal(base.actions[0][:2], treated.actions[0][:2])
        self.assertEqual(control["action_trace_sha256"], hashlib.sha256(baseline_trace.tobytes()).hexdigest())
        self.assertEqual(treatment["action_trace_sha256"], hashlib.sha256(treated_trace.tobytes()).hexdigest())

    def test_lap_deltas_are_mutual_finish_only_and_repeats_do_not_double_count(self):
        cells, episodes = [], []
        for track in (1, 2, 3):
            cells.append({"track_id": track, "geometry_seed": 42})
            for role, lap in (("control", 45000), ("light_brake_release", 40000)):
                for repeat in (0, 1):
                    episodes.append({"track_id": track, "geometry_seed": 42,
                                     "role": role, "repeat": repeat, "finished": True,
                                     "steps": 500, "lap_time_ms": lap,
                                     "action_trace_sha256": role + str(track)})
        summary = speed.summarize(episodes, cells)
        self.assertEqual(summary["mutual_finishes"], 3)
        self.assertEqual(summary["mean_paired_lap_delta_ms"], -5000)
        self.assertTrue(summary["speed_signal"])
        episodes[3]["finished"] = False
        episodes[3]["lap_time_ms"] = None
        with self.assertRaisesRegex(ValueError, "repeat changed"):
            speed.summarize(episodes, cells)
        episodes[2]["finished"] = False
        episodes[2]["lap_time_ms"] = None
        rejected = speed.summarize(episodes, cells)
        self.assertEqual(rejected["by_track"]["1"]["lost_finishes"], 1)
        self.assertEqual(rejected["mutual_finishes"], 2)
        self.assertFalse(rejected["speed_signal"])
        episodes[1]["progress"] = 0.25
        with self.assertRaisesRegex(ValueError, "repeat changed"):
            speed.summarize(episodes, cells)
        episodes[1].pop("progress")
        episodes[2]["road_sha256"] = episodes[3]["road_sha256"] = "another-road"
        with self.assertRaisesRegex(ValueError, "paired policies did not start"):
            speed.summarize(episodes, cells)

    def test_preflight_fails_closed_on_unseen_or_drifted_reuse_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "experiments/train-seed-claims").mkdir()
            actor = root / "actor.pt"
            actor.write_bytes(b"immutable-test-actor")
            actor_sha = speed.sha256(actor)
            sources = {}
            for name in speed.REQUIRED_SOURCES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode("utf-8"))
                sources[name] = speed.sha256(path)
            diagnostic = root / "diagnostic.jsonl"
            diagnostic.write_text(json.dumps({"role": "unchanged-source", "actor_sha256": actor_sha,
                                              "track_id": 1, "geometry_seed": 42, "repeat": 0}) + "\n")
            pilot = root / "pilot.json"
            pilot.write_text(json.dumps({"base_actor_sha256": actor_sha, "cells": [
                {"track_id": track, "geometry_seed": 42,
                 "control": {"track_id": track, "geometry_seed": 42, "done": True}}
                for track in (2, 3)]}))
            r6_path = root / "experiments/drqv2-geometry-mix-v1-r6.json"
            r6_path.write_text(json.dumps({"diagnostic_pool": {"track_ids": [1],
                                                            "frame_skip": 4, "geometry_seeds": [42]}}))
            r6_manifest = root / "runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json"
            r6_manifest.parent.mkdir(parents=True)
            r6_manifest.write_text(json.dumps({"files_sha256": {"episodes.jsonl": speed.sha256(diagnostic)}}))
            pilot_path = root / "experiments/drqv2-residual-options-pilot-v1.json"
            pilot_path.write_text(json.dumps({"environment": {"obstacles": True, "frame_skip": 4,
                                                               "max_decisions_per_episode": 2000},
                                              "development_screen": {
                                                  "geometry_seeds": list(range(4000000001, 4000000009))}}))
            pilot_manifest = root / "runs/20260924-drqv2-residual-options-pilot/iteration-1-retry/manifest.completed.json"
            pilot_manifest.parent.mkdir(parents=True)
            pilot_manifest.write_text(json.dumps({"status": "completed", "protocol_sha256": speed.sha256(pilot_path)}))
            historic = {path.relative_to(root).as_posix(): speed.sha256(path)
                        for path in (r6_path, r6_manifest, pilot_path, pilot_manifest)}
            audit = root / speed.AUDIT_PATH
            audit.write_text(json.dumps({"format": "haic-drq-speed-cell-audit-v1", "status": "reuse_only",
                                         "actor_sha256": actor_sha,
                                         "cells": [{"track_id": track, "geometry_seed": 42} for track in (1, 2, 3)],
                                         "active_claim_overlap": [], "protected_overlap": [],
                                         "partial_reset_overlap": []}))
            spec = {
                "format": speed.FORMAT, "actor": {"path": "actor.pt", "sha256": actor_sha},
                "roles": ["control", "light_brake_release"], "repeats": 2,
                "max_steps": 2000, "frame_skip": 4, "obstacles": True, "reward_shaping": False,
                "reuse_only": True, "acceptance": speed.ACCEPTANCE, "source_sha256": sources,
                "reuse_evidence": {
                    "diagnostic_track1": {"path": "diagnostic.jsonl", "sha256": speed.sha256(diagnostic)},
                    "residual_tracks2_3": {"path": "pilot.json", "sha256": speed.sha256(pilot)},
                },
                "cell_audit": {"path": speed.AUDIT_PATH, "sha256": speed.sha256(audit)},
                "cells": [{"track_id": track, "geometry_seed": 42} for track in (1, 2, 3)],
            }
            protocol = root / "experiments" / "speed.json"
            with (patch.object(speed, "ROOT", root),
                  patch.object(speed, "SOURCE_ACTOR_SHA", actor_sha),
                  patch.object(speed, "EXPECTED_CELLS", {(track, 42) for track in (1, 2, 3)}),
                  patch.object(speed, "EXPECTED_EVIDENCE", spec["reuse_evidence"]),
                  patch.object(speed, "HISTORICAL_PINS", historic)):
                protocol.write_text(json.dumps(spec))
                _, receipt = speed.preflight(protocol)
                self.assertEqual(receipt["planned_episodes"], 12)
                claim = root / "experiments/train-seed-claims/seed-42.json"
                claim.write_text(json.dumps({"format": "haic-train-seed-claim-v1", "geometry_seed": 42}))
                with self.assertRaisesRegex(ValueError, "active shared claim overlaps"):
                    speed.preflight(protocol)
                claim.unlink()
                original_evidence = spec["reuse_evidence"]
                impersonation = root / "forged-diagnostic.jsonl"
                impersonation.write_bytes(diagnostic.read_bytes())
                spec["reuse_evidence"] = {**original_evidence, "diagnostic_track1": {
                    "path": "forged-diagnostic.jsonl", "sha256": speed.sha256(impersonation)}}
                protocol.write_text(json.dumps(spec))
                with self.assertRaisesRegex(ValueError, "not the frozen historical source set"):
                    speed.preflight(protocol)
                spec["reuse_evidence"] = original_evidence
                spec["cells"][1]["geometry_seed"] = 43
                protocol.write_text(json.dumps(spec))
                with self.assertRaisesRegex(ValueError, "predeclared reused-development cohort"):
                    speed.preflight(protocol)
                spec["cells"][1]["geometry_seed"] = 42
                pilot.write_text("{}")
                protocol.write_text(json.dumps(spec))
                with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                    speed.preflight(protocol)


if __name__ == "__main__":
    unittest.main()
