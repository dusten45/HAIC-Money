"""Synthetic-only tests; no real roads or existing artifact outputs are opened."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from haic.algorithms.drq_v2.teacher_replay import TeacherDataset, TeacherEpisode
from scripts.analyze_drq_geometry_failures import (
    R3,
    R3_RUN,
    STUDIES,
    Inputs,
    PREFIX,
    PREFIX_RUN,
    _actions,
    _evaluation,
    _geometry,
    _outcome,
    _path,
    _teacher,
    aggregate,
    analyze,
    main,
)


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pool(track: int, seed: int) -> dict:
    return {"track_ids": [track], "seeds": [seed], "repeats": 2}


class GeometryAnalysisAggregationTests(unittest.TestCase):
    def test_actions_are_executed_official_axes_not_native_pedals(self):
        actions = [[1, 1, .2], [-1, .4, .3], [0, .2, 0]]
        stats, digest = _actions(actions, 3)
        self.assertEqual(stats["decisions"], 3)
        self.assertAlmostEqual(stats["steering_saturated_fraction_abs_ge_0_95"], 2 / 3)
        self.assertAlmostEqual(stats["steering_delta_abs_mean"], 1.5)
        self.assertAlmostEqual(stats["simultaneous_gas_brake_fraction_gt_0_1"], 2 / 3)
        self.assertAlmostEqual(stats["gas_mean"], (1 + .4 + .2) / 3)
        self.assertEqual(digest, hashlib.sha256(np.asarray(actions, dtype=np.float32).tobytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "trace SHA-256"):
            _actions(actions, 3, "a" * 64)

    def test_missing_per_step_fields_remain_null_not_fabricated(self):
        row = _outcome({"steps": 12, "finished": False, "progress": .94,
                        "retire_reason": "off_track"})
        self.assertIsNone(row["max_progress"])
        self.assertIsNone(row["damage"])
        self.assertIsNone(row["action_stats"])
        self.assertTrue(row["high_progress_nonfinish"])
        self.assertTrue(row["ever_reached_ge_0_9_nonfinish_when_observable"])
        self.assertFalse(_outcome({"steps": 3, "finished": True, "progress": .97,
                                   "retire_reason": None})["high_progress_nonfinish"])
        high = _outcome({"steps": 3, "finished": False, "progress": 1.,
                         "retire_reason": "off_track"})
        self.assertFalse(high["finished"])
        self.assertTrue(high["high_progress_nonfinish"])
        transient = _outcome({"steps": 3, "finished": False, "progress": .2,
                              "max_progress": .96, "retire_reason": "off_track"})
        self.assertFalse(transient["high_progress_nonfinish"])
        self.assertTrue(transient["ever_reached_ge_0_9_nonfinish_when_observable"])
        self.assertIsNone(_outcome({"steps": 3, "finished": False, "progress": .2})[
            "ever_reached_ge_0_9_nonfinish_when_observable"])

    def test_road_seed_groups_obstacle_variants_and_exact_repeats(self):
        base = {"actor_sha256": "a" * 64, "study": "pad", "partition": "screen",
                "progress": .3, "max_progress": None, "steps": 5, "damage": 0.,
                "retire_reason": "off_track", "terminal_class": "off_track",
                "finished": False, "high_progress_nonfinish": False,
                "ever_reached_ge_0_9_nonfinish_when_observable": None,
                "action_stats": None, "action_trace_sha256": "f" * 64}
        rows = [dict(base, track_id=101, geometry_seed=10),
                dict(base, track_id=101, geometry_seed=10),
                dict(base, track_id=102, geometry_seed=10, finished=True,
                     retire_reason=None, terminal_class="finished", progress=.97),
                dict(base, track_id=101, geometry_seed=11, progress=.15,
                     action_trace_sha256="b" * 64)]
        result = aggregate(rows)
        self.assertEqual(result["summary"]["episodes"], 4)
        self.assertEqual(result["unique_road_seeds"], 2)
        self.assertEqual(result["unique_track_seed_obstacle_cells"], 3)
        self.assertEqual(result["by_road_seed"][0]["outcome"], "mixed")
        self.assertEqual(result["repeated_identical_source_trajectories"][0]["count"], 2)
        self.assertEqual(result["summary"]["failure_terminal_progress_bins"]["0.3-0.4"], 2)


class GeometryAnalysisSafetyTests(unittest.TestCase):
    def test_raw_track_coordinate_hash_and_reset_use_only_consumed_variant(self):
        theta = np.linspace(0, 2 * math.pi, 120, endpoint=False)
        track = np.column_stack((theta, theta + math.pi / 2,
                                 60 * np.cos(theta), 60 * np.sin(theta)))

        class FakeCarRacing:
            resets = []

            def __init__(self, *, continuous, render_mode):
                self.track = track

            def reset(self, *, seed, options):
                self.resets.append((seed, options["track_id"]))

            def close(self):
                return None

        with patch("core.vendor.car_racing.CarRacing", FakeCarRacing):
            result = _geometry({11: 101})
        self.assertEqual(FakeCarRacing.resets, [(11, 101)])
        self.assertEqual(result["11"]["road_coordinate_sha256"], hashlib.sha256(
            np.ascontiguousarray(np.asarray(track, dtype="<f8")[:, 2:4]).tobytes()).hexdigest())
        self.assertEqual(result["11"]["signature"], result["11"]["descriptor"]["signature"])
        self.assertIn("centerline_connected", result["11"]["structure"])

    def test_rejects_blind_paths_and_symlinked_inputs_before_opening(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = root / "evaluations/screen/manifest.json"
            _json(allowed, {})
            for bad in ("evaluations/blind/manifest.json", "runs/unopened-blind.json",
                        "../outside.json", "/tmp/blind.json"):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "blind|untrusted"):
                    _path(root, bad)
            with self.assertRaisesRegex(ValueError, "exact source allowlist"):
                _path(root, "evaluations/screen/manifest.json", expected="runs/safe/manifest.json")
            link = root / "safe.json"
            link.symlink_to(allowed)
            with self.assertRaisesRegex(ValueError, "exact source allowlist"):
                _path(root, "safe.json", expected="evaluations/screen/manifest.json")
            outside = root / "escape.json"
            outside.symlink_to("/etc/hosts")
            with self.assertRaisesRegex(ValueError, "escapes repository"):
                _path(root, "escape.json", expected="escape.json")
            _json(root / "evaluations/blind-sealed/episodes.jsonl", {})
            (root / "safe.json").unlink()
            (root / "safe.json").symlink_to(root / "evaluations/blind-sealed/episodes.jsonl")
            with self.assertRaisesRegex(ValueError, "blind symlink"):
                _path(root, "safe.json", expected="safe.json")

    def test_rejects_blind_partition_and_unknown_actor_seed_before_episode_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name, run = STUDIES["pad"]
            protocol = {"partitions": {"screen": _pool(101, 11),
                                       "confirmation": _pool(201, 21),
                                       "blind": _pool(301, 31)}}
            source_dir = root / "runs" / run / "control-seed0"
            directory = source_dir / "evaluations" / f"20260925T010203000000Z_{name}-screen"
            _json(directory / "manifest.json", {
                "partition": "blind", "protocol_sha256": "a" * 64,
                "protocol": f"{name}-screen", "cell_matrix": protocol["partitions"]["screen"],
            })
            pointer = {"evaluation_dir": str(directory), "candidate_id": "expected"}
            with self.assertRaisesRegex(ValueError, "manifest/partition"):
                _evaluation(Inputs(root), protocol, "a" * 64, name, run, 0, "b" * 64,
                            "screen", pointer)
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest.update(partition="screen", frame_skip=4, max_steps=2000)
            _json(directory / "manifest.json", manifest)
            _json(directory / "determinism.json", [{"track_id": 101, "seed": 11,
                                                    "candidate_id": "expected", "repeats": 2,
                                                    "audited": True, "matches_canonical": True}])
            for cell, candidate in (((101, 31), "expected"), ((101, 11), "unknown")):
                bad = {"track_id": cell[0], "seed": cell[1], "candidate_id": candidate,
                       "repeat": 0, "status": "ok"}
                (directory / "episodes.jsonl").write_text(json.dumps(bad) + "\n")
                with self.subTest(cell=cell, candidate=candidate), self.assertRaisesRegex(ValueError, "unknown actor, seed"):
                    _evaluation(Inputs(root), protocol, "a" * 64, name, run, 0, "b" * 64,
                                "screen", pointer)


class GeometryAnalysisFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.actors = ["a" * 64, "b" * 64]
        self.r3 = {"name": R3, "study_id": R3, "frame_skip": 4, "max_steps": 2000,
                   "partitions": {"screen": _pool(102, 22), "confirmation": _pool(202, 32),
                                  "blind": _pool(302, 42)},
                   "training_pools": {"teacher_training": {"track_ids": [1],
                                                           "seeds": [50], "sampler_seed": 917001}},
                   "reserved_training_seeds": [42], "source_actors": [
                       {"learner_seed": learner, "actor_sha256": actor,
                        "actor_path": f"runs/{STUDIES['pad'][1]}/control-seed{learner}/checkpoints/step-000131072/actor.pt",
                        "checkpoint_sha256": str(learner) * 64}
                       for learner, actor in enumerate(self.actors)]}
        _json(self.root / "experiments" / f"{R3}.json", self.r3)
        for study, (name, run) in STUDIES.items():
            protocol = {"name": name, "frame_skip": 4, "max_steps": 2000,
                        "partitions": {"screen": _pool(101, 11),
                                       "confirmation": _pool(201, 21 if study == "pad" else 23),
                                       "blind": _pool(301, 41 if study == "pad" else 43)}}
            _json(self.root / "experiments" / f"{name}.json", protocol)
            for learner, actor in enumerate(self.actors):
                base = self.root / "runs" / run / f"control-seed{learner}"
                selected = {"candidate_id": actor[:12], "archive_sha256": actor,
                            "algorithm": "drq-v2", "eligible": True, "cpu_reload_matches": True,
                            "export_metadata": {"config": {"augmentation_pad": 4}}}
                dirs = {}
                for partition in ("screen", "confirmation"):
                    matrix = protocol["partitions"][partition]
                    directory = base / "evaluations" / f"20260925T010203000000Z_{name}-{partition}"
                    dirs[partition] = directory
                    _json(directory / "manifest.json", {"partition": partition,
                          "protocol": f"{name}-{partition}", "frame_skip": 4, "max_steps": 2000,
                          "protocol_sha256": _sha(self.root / "experiments" / f"{name}.json"),
                          "cell_matrix": matrix, "diagnostic_only": False})
                    _json(directory / "determinism.json", [{"candidate_id": selected["candidate_id"],
                          "track_id": matrix["track_ids"][0], "seed": matrix["seeds"][0],
                          "audited": True, "matches_canonical": True, "repeats": 2}])
                    with (directory / "episodes.jsonl").open("w") as handle:
                        for repeat in (0, 1):
                            result = {"candidate_id": selected["candidate_id"], "status": "ok",
                                      "loaded_archive_sha256": actor,
                                      "track_id": matrix["track_ids"][0], "seed": matrix["seeds"][0],
                                      "repeat": repeat, "steps": 1, "progress": .94, "damage": 0.,
                                      "finished": False, "termination_class": "off_track",
                                      "retire_reason": "off_track", "actions": [[1, .8, .2]]}
                            handle.write(json.dumps(result) + "\n")
                _json(base / "selection.json", {"selected": True,
                      "protocol_sha256": _sha(self.root / "experiments" / f"{name}.json"),
                      "actor_sha256": actor, "cpu_result": selected,
                      "evaluation_dir": str(dirs["screen"])})
                _json(base / "confirmation.json", {"partition": "confirmation",
                      "protocol_sha256": _sha(self.root / "experiments" / f"{name}.json"),
                      "ranked": [selected], "evaluation_dir": str(dirs["confirmation"]),
                      "diagnostic_only": False})
        for learner in (0, 1):
            actor = self.actors[learner]
            base = self.root / R3_RUN / "teacher-data" / f"learner-{learner}"
            ledger = {"episode_id": 0, "complete": True, "track_id": 1,
                      "geometry_seed": 50, "steps": 2, "finished": False,
                      "progress": .4, "damage": 0., "retire_reason": "off_track"}
            frames = np.zeros((3, 84, 84), dtype=np.uint8)
            episode = TeacherEpisode(frames=frames, actions=np.zeros((2, 3), dtype=np.float32),
                                     applied_actions=np.asarray([[0., .5, .5], [0., .5, .5]], dtype=np.float32),
                                     rewards=[-1., -1.], progress=[.6, .4], damage=[0., 0.],
                                     retire_reasons=[None, "off_track"], terminated=[False, True],
                                     truncated=[False, False], finished=[False, False],
                                     episode_id=0, source_id=f"source-learner-{learner}",
                                     source_actor_sha256=actor, geometry_id="50", track_id=1,
                                     metadata=ledger)
            dataset = TeacherDataset([episode])
            digest = dataset.seal()
            base.mkdir(parents=True, exist_ok=True)
            dataset_path = base / "teacher-dataset.npz"
            dataset_path.write_bytes(dataset.to_bytes())
            partial = {"episode_id": 1, "complete": False, "track_id": 1,
                       "geometry_seed": 50, "steps": 1, "finished": False,
                       "progress": .1, "damage": 0., "retire_reason": None}
            _json(base / "collection-result.json", {
                "phase": "teacher-data-collection", "study_id": R3,
                "study_protocol_sha256": _sha(self.root / "experiments" / f"{R3}.json"),
                "learner_seed": learner, "source_actor_sha256": actor,
                "source_checkpoint_sha256": str(learner) * 64,
                "training_pool": self.r3["training_pools"]["teacher_training"],
                "dataset_path": dataset_path.relative_to(self.root).as_posix(),
                "dataset_file_sha256": _sha(dataset_path), "dataset_digest": digest,
                "episode_summaries": [ledger, partial], "incomplete_episode": partial,
                "complete_episode_count": 1, "complete_transition_count": 2,
                "decisions": 3, "decision_cap": 3, "finished_geometry_seeds": [],
                "coverage_pass": False,
            })
        hashes = {name: _sha(self.root / "experiments" / f"{name}.json")
                  for name in [R3, *(name for name, _ in STUDIES.values())]}
        protocol_pins = patch.dict("scripts.analyze_drq_geometry_failures.FROZEN_PROTOCOL_SHA256", hashes)
        source_pins = patch.dict("scripts.analyze_drq_geometry_failures.FROZEN_SOURCE_SHA256",
                                 dict(enumerate(self.actors)))
        protocol_pins.start()
        source_pins.start()
        self.addCleanup(protocol_pins.stop)
        self.addCleanup(source_pins.stop)

    def test_end_to_end_synthetic_r3_and_selected_receipts_are_deterministic(self):
        first = analyze(self.root)
        second = analyze(self.root)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "complete")
        self.assertIsNone(first["road_geometry"])
        self.assertEqual(first["cohorts"]["pad-source0-screen"]["summary"]["episodes"], 1)
        self.assertEqual(first["cohorts"]["r3-source0-teacher_training"]["summary"]["mean_max_progress_when_available"],
                         float(np.float32(.6)))
        self.assertEqual(first["cohorts"]["r3-source0-teacher_training"]["incomplete_budget_interrupted_episode"]["steps"], 1)
        self.assertNotIn("blind", " ".join(first["inputs_sha256_exact_allowlist"]))
        output = self.root / "analysis.json"
        main(["--repo-root", str(self.root), "--output", str(output)])
        self.assertEqual(json.loads(output.read_text()), first)
        with self.assertRaises(SystemExit):
            main(["--repo-root", str(self.root), "--output", str(output)])

    def test_opt_in_static_descriptors_use_only_previously_consumed_cells(self):
        descriptor = {"abs_curvature_p90_per_m": .03, "sharp_entry_count": 3,
                      "rapid_reversals_65m": 1, "longest_straight_before_sharp_m": 20.,
                      "perimeter_m": 900.}
        geometry = {str(seed): {"descriptor": descriptor, "static_validation": {}}
                    for seed in (11, 21, 23, 50)}
        with patch("scripts.analyze_drq_geometry_failures._geometry", return_value=geometry) as reset:
            report = analyze(self.root, describe_geometry=True)
        reset.assert_called_once_with({11: 101, 21: 201, 23: 201, 50: 1})
        self.assertEqual(report["road_geometry"], geometry)
        self.assertEqual(report["structural_hypotheses_vs_success"][0]["status"],
                         "descriptive_hypothesis_not_a_causal_test")

    def test_unknown_source_or_blind_seed_in_collection_is_rejected(self):
        base = self.root / R3_RUN / "teacher-data" / "learner-0" / "collection-result.json"
        original = json.loads(base.read_text())
        record = dict(original, source_actor_sha256="c" * 64)
        _json(base, record)
        with self.assertRaisesRegex(ValueError, "source/pool"):
            analyze(self.root)
        _json(base, original)
        record = dict(original, training_pool={"track_ids": [1], "seeds": [42], "sampler_seed": 917001})
        _json(base, record)
        with self.assertRaisesRegex(ValueError, "source/pool"):
            analyze(self.root)

    def test_frozen_protocol_pin_rejects_replaced_partition_before_opening_episodes(self):
        name, _ = STUDIES["pad"]
        path = self.root / "experiments" / f"{name}.json"
        modified = json.loads(path.read_text())
        modified["partitions"]["screen"]["seeds"] = [42]
        _json(path, modified)
        with self.assertRaisesRegex(ValueError, "protocol SHA-256"):
            analyze(self.root)

    def test_claimed_determinism_cannot_hide_a_changed_repeat(self):
        name, run = STUDIES["pad"]
        directory = (self.root / "runs" / run / "control-seed0" / "evaluations"
                     / f"20260925T010203000000Z_{name}-screen")
        episodes = directory / "episodes.jsonl"
        rows = [json.loads(line) for line in episodes.read_text().splitlines()]
        rows[1]["progress"] = .92
        episodes.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "reloaded source trajectory"):
            analyze(self.root)

    def test_sealed_teacher_file_must_match_recorded_hash(self):
        source = self.r3["source_actors"][0]
        inputs = Inputs(self.root)
        protocol_hash = _sha(self.root / "experiments" / f"{R3}.json")
        rows, provenance = _teacher(inputs, self.r3, protocol_hash, 0, source)
        self.assertEqual(rows[0]["max_progress"], float(np.float32(.6)))
        self.assertEqual(provenance["distinct_finished_road_seeds"], [])
        dataset = self.root / R3_RUN / "teacher-data" / "learner-0" / "teacher-dataset.npz"
        dataset.write_bytes(dataset.read_bytes() + b"altered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            _teacher(Inputs(self.root), self.r3, protocol_hash, 0, source)

    def test_optional_prefix_is_only_consumed_training_and_uses_actual_pose(self):
        protocol_path = self.root / "experiments" / f"{PREFIX}.json"
        _json(protocol_path, {"name": PREFIX, "base_driver": {"actor_sha256": self.actors[1]},
              "environment": {"partition": "training-only causal mechanism diagnostic",
                              "track_ids": [1], "geometry_seeds": [60]},
              "procedure": {"predeclared_branch_points_after_decisions": [1]}})
        with patch.dict("scripts.analyze_drq_geometry_failures.FROZEN_PROTOCOL_SHA256",
                        {PREFIX: _sha(protocol_path)}):
            self._check_prefix(protocol_path)

    def _check_prefix(self, protocol_path: Path):
        base = self.root / PREFIX_RUN
        _json(base / "manifest.completed.json", {"protocol_sha256": _sha(protocol_path),
              "base_actor_sha256": self.actors[1],
              "status": "completed_training_only_single_intervention_diagnostic"})
        (base / "source-trajectories.jsonl").write_text(json.dumps({
            "track_id": 1, "geometry_seed": 60, "official_actions": [[0, .5, .5]],
            "step_signatures": [{"info": {"progress": .6},
                                 "state": {"position": [12., -8.], "off_track_counter": 10}}],
            "outcome": {"decisions": 1, "finished": False, "progress": .4,
                        "damage": 0., "retire_reason": "off_track"}}) + "\n")
        (base / "branch-trajectories.jsonl").write_text(json.dumps({
            "cell_key": "id1-seed60-after1", "head": "v1", "status": "selected_KEEP; no branch needed",
            "branch_after_decisions": 1}) + "\n")
        report = analyze(self.root, include_prefix=True)
        prefix = report["optional_training_only_prefix"]
        baseline = prefix["baseline_only"]["canonical_episode_rows"][0]
        self.assertEqual(baseline["terminal_position_xy_when_available"], [12., -8.])
        self.assertEqual(baseline["max_progress"], .6)
        self.assertEqual(prefix["intervention_contexts_not_independent_roads"], 1)
        self.assertNotIn("blind", " ".join(report["inputs_sha256_exact_allowlist"]))
        invalid = json.loads(protocol_path.read_text())
        invalid["environment"]["geometry_seeds"] = [42]
        _json(protocol_path, invalid)
        with patch.dict("scripts.analyze_drq_geometry_failures.FROZEN_PROTOCOL_SHA256",
                        {PREFIX: _sha(protocol_path)}):
            with self.assertRaisesRegex(ValueError, "reserved/unknown"):
                analyze(self.root, include_prefix=True)


if __name__ == "__main__":
    unittest.main()
