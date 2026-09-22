import hashlib
import importlib
import json
import os
import sys
import time
import tempfile
import unittest
import zipfile
import pickle
from types import SimpleNamespace
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import torch

import evaluate_policy
from agent import (
    DRQ_ACTOR_FORMAT,
    DrQActor,
    DREAMERV3_ACTOR_FORMAT,
    DreamerV3Encoder,
    DreamerV3RSSM,
    DreamerV3Actor,
    DreamerV3ExportedActor,
)
from common_adapter import ActionSpec, ObservationSpec
from evaluate_policy import (
    MAX_PROCESS_RSS_BYTES,
    PROTOCOLS,
    candidate_summary,
    determinism_audit,
    discover_candidates,
    evaluate_cell,
    evaluate_model,
    load_protocol_spec,
    parse_int_list,
    parse_worker_output,
    previous_evaluation_metadata,
    ranking_key,
    run_checkpoint_protocol,
    run_isolated_cell,
    snapshot_candidates,
    snapshot_runtime,
    terminal_class,
    timed_policy_call,
    validate_cpu_runtime,
    validate_protocol_request,
    parse_args,
)


class TestEvaluatePolicy(unittest.TestCase):
    def drq_candidate(self, root):
        actor = root / "actor.pt"
        torch.save({
            "format": DRQ_ACTOR_FORMAT,
            "config": {"observation_shape": (4, 84, 84), "action_dim": 3,
                       "feature_dim": 16, "hidden_dim": 16},
            "observation_spec": asdict(ObservationSpec()),
            "action_spec": asdict(ActionSpec()),
            "state_dict": DrQActor(16, 16).state_dict(),
        }, actor)
        (root / "config.json").write_text(json.dumps({
            "config": {"algorithm": "drq-v2", "max_steps": 2, "frame_skip": 4},
        }))
        return actor

    def dreamerv3_candidate(self, root):
        actor = root / "actor.pt"
        cfg = {
            "embed_dim": 64,
            "hidden_dim": 64,
            "num_categoricals": 8,
            "num_classes": 8,
            "unimix": 0.01,
        }
        enc = DreamerV3Encoder(4, 64)
        rssm = DreamerV3RSSM(3, 64, 64, 8, 8, 0.01)
        act = DreamerV3Actor(64 + 64, 3)
        model = DreamerV3ExportedActor(enc, rssm, act)
        torch.save({
            "format": DREAMERV3_ACTOR_FORMAT,
            "config": cfg,
            "observation_spec": asdict(ObservationSpec()),
            "action_spec": asdict(ActionSpec()),
            "state_dict": model.state_dict(),
        }, actor)
        (root / "config.json").write_text(json.dumps({
            "config": {"algorithm": "dreamerv3", "max_steps": 2, "frame_skip": 4},
        }))
        return actor

    def protocol_spec(self):
        return {
            "name": "unit-protocol", "max_steps": 2, "frame_skip": 4,
            "partitions": {
                partition: {"track_ids": [index + 1], "seeds": [index], "repeats": 2}
                for index, partition in enumerate(("screen", "confirmation", "blind"))
            },
        }

    def episode(self, repeat=0):
        return {
            "status": "ok", "candidate_id": "candidate", "track_id": 1, "seed": 0,
            "repeat": repeat, "steps": 2, "reward": 1.0, "progress": 0.5,
            "finished": False, "lap_time_ms": None, "damage": 0.0,
            "termination_class": "max_steps", "action_trace_sha256": "a" * 64,
            "process_initialization_seconds": .1, "agent_reset_seconds": .001,
            "max_action_seconds": .001, "peak_rss_bytes": 128 * 1024 * 1024,
        }

    def test_blind_protocol_is_disjoint_from_training_and_development_sets(self):
        blind_seeds = set(PROTOCOLS["checkpoint-v1-blind"]["seeds"])
        development_seeds = set(PROTOCOLS["checkpoint-v1-screen"]["seeds"])
        development_seeds.update(PROTOCOLS["checkpoint-v1-confirmation"]["seeds"])

        self.assertTrue(blind_seeds.isdisjoint(development_seeds))
        self.assertTrue(blind_seeds.isdisjoint({42, 777, 1337, 2024}))

    def test_import_does_not_mutate_training_process_environment(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "keep-visible", "OMP_NUM_THREADS": "7"}):
            importlib.reload(evaluate_policy)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "keep-visible")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "7")
    def test_parse_int_list_requires_unique_bounded_values(self):
        self.assertEqual(parse_int_list("1,2,3", "--track-ids", 1), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_int_list("1,1", "--track-ids", 1)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            parse_int_list("0", "--track-ids", 1)

    def test_evaluate_model_records_per_track_and_episodes(self):
        episode = {
            "steps": 100,
            "reward": 50.0,
            "progress": 0.5,
            "finished": False,
            "lap_time_ms": None,
            "damage": 0.0,
            "retire_reason": "off_track",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.zip"
            path.touch()
            with patch("evaluate_policy.PPO.load", return_value=object()) as load:
                with patch("evaluate_policy.evaluate", return_value=[episode]) as assess:
                    report = evaluate_model(path, [1, 2], [10, 11], 2000, 4)

        load.assert_called_once_with(str(path), device="cpu")
        self.assertEqual(assess.call_count, 4)
        self.assertEqual(report["summary"]["n_episodes"], 4)
        self.assertEqual(set(report["by_track"]), {"1", "2"})
        self.assertEqual(report["summary"]["termination_reasons"], {"off_track": 4})

    def test_discovery_deduplicates_policy_aliases_and_finds_checkpoint_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "ppo_baseline_10_steps.zip"
            alias = run_dir / "best_model.zip"
            for path in (checkpoint, alias):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("policy.pth", b"same-policy")
            normalizer = checkpoint_dir / "ppo_baseline_vecnormalize_10_steps.pkl"
            with normalizer.open("wb") as handle:
                pickle.dump(SimpleNamespace(norm_obs=False), handle)
            with (run_dir / "best_model_vecnormalize.pkl").open("wb") as handle:
                pickle.dump(SimpleNamespace(norm_obs=False), handle)

            candidates = discover_candidates([checkpoint, alias])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["aliases"], [str(checkpoint), str(alias)])
        self.assertEqual(candidates[0]["vecnormalize_path"], str(normalizer))

    def test_dreamerv3_candidate_metadata_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = self.dreamerv3_candidate(root)
            candidates = discover_candidates([actor], run_dir=root)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["algorithm"], "dreamerv3")
            self.assertEqual(candidates[0]["export_metadata"]["format"], DREAMERV3_ACTOR_FORMAT)

    def test_terminal_class_preserves_official_outcomes(self):
        self.assertEqual(terminal_class(True, False, True, {}), "finished")
        self.assertEqual(terminal_class(False, True, False, {"retire_reason": "crash"}), "crash")
        self.assertEqual(terminal_class(False, True, False, {}), "out_of_bounds")
        self.assertEqual(terminal_class(False, False, True, {}), "max_steps")

    def test_single_repeat_is_unaudited_and_progress_precedes_lap_time(self):
        episode = {"candidate_id": "candidate", "track_id": 2, "seed": 20, "repeat": 0}
        audits, non_reproducible, unaudited = determinism_audit([episode])

        self.assertFalse(audits[0]["audited"])
        self.assertEqual(non_reproducible, set())
        self.assertEqual(unaudited, {"candidate"})

        def candidate(progress, lap_time):
            metrics = {"n_episodes": 1, "finish_rate": 1.0, "avg_progress": progress,
                       "avg_lap_time_ms": lap_time}
            return {"eligible": True, "by_track": {"2": metrics}, "summary": metrics,
                    "operational_failures": 0}

        self.assertGreater(ranking_key(candidate(0.8, 20_000)), ranking_key(candidate(0.2, 1_000)))

    def test_candidate_snapshot_is_independent_of_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source = directory_path / "source.zip"
            source.write_bytes(b"original")
            vecnormalize_source = directory_path / "source.vecnormalize.pkl"
            vecnormalize_source.write_bytes(b"statistics")
            candidate = {
                "candidate_id": "candidate",
                "source_path": str(source),
                "archive_sha256": hashlib.sha256(b"original").hexdigest(),
                "vecnormalize_path": str(vecnormalize_source),
                "vecnormalize_sha256": hashlib.sha256(b"statistics").hexdigest(),
            }
            output = directory_path / "output"
            output.mkdir()

            snapshot_candidates(output, [candidate])
            source.write_bytes(b"mutated")

            self.assertEqual((output / "candidates/candidate.zip").read_bytes(), b"original")
            self.assertEqual(candidate["evaluation_archive_path"], "candidates/candidate.zip")
            self.assertEqual(
                candidate["evaluation_vecnormalize_path"],
                "candidates/candidate.vecnormalize.pkl",
            )

    def test_runtime_snapshot_includes_train_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = snapshot_runtime(Path(directory))
            self.assertEqual(worker.name, "evaluate_policy.py")
            self.assertTrue((worker.parent / "tracking.py").is_file())
            self.assertTrue((worker.parent / "action_smoothing.py").is_file())
            self.assertTrue((worker.parent / "agent.py").is_file())

    def test_drq_discovery_without_sb3_normalizer_and_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = self.drq_candidate(root)
            (root / "config.json").write_text(json.dumps({"config": {
                "algorithm": "drq-v2", "max_steps": 2, "frame_skip": 4,
            }}))
            with patch("evaluate_policy.PPO.load", side_effect=AssertionError("SB3 load")):
                candidate, = discover_candidates([actor], run_dir=root)
            self.assertEqual(candidate["algorithm"], "drq-v2")
            self.assertIsNone(candidate["vecnormalize_path"])
            self.assertEqual(candidate["run_max_steps"], 2)
            self.assertEqual(candidate["export_metadata"]["format"], DRQ_ACTOR_FORMAT)
            self.assertEqual(candidate["export_spec_fingerprints"]["action_spec"], ActionSpec().fingerprint)
            output = root / "snapshot"
            output.mkdir()
            original_hash = candidate["archive_sha256"]
            snapshot_candidates(output, [candidate])
            actor.write_bytes(b"mutated")
            snapshot = output / candidate["evaluation_archive_path"]
            self.assertEqual(snapshot.suffix, ".pt")
            self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), original_hash)
            self.assertTrue((output / candidate["evaluation_run_config_path"]).is_file())

    def test_custom_protocol_rejects_overlap_and_single_repeat_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            spec = self.protocol_spec()
            path.write_text(json.dumps(spec))
            self.assertEqual(load_protocol_spec(path), spec)
            spec["partitions"]["blind"]["seeds"] = [0]
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                load_protocol_spec(path)
            spec = self.protocol_spec()
            spec["partitions"]["screen"]["repeats"] = 1
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "two independent"):
                load_protocol_spec(path)

    def test_screen_only_protocol_is_reserved_for_harness_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            spec = self.protocol_spec()
            spec["partitions"] = {"screen": spec["partitions"]["screen"]}
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "partitions"):
                load_protocol_spec(path)
            spec["purpose"] = "harness-smoke"
            path.write_text(json.dumps(spec))
            self.assertEqual(load_protocol_spec(path), spec)

    def test_reserved_training_seeds_are_optional_and_preserve_partition_union(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            for extra in ({}, {"reserved_training_seeds": []},
                          {"reserved_training_seeds": [42]},
                          {"reserved_training_seeds": [0, 1, 2, 42, 2**32 - 1]}):
                with self.subTest(extra=extra):
                    spec = {**self.protocol_spec(), **extra}
                    path.write_text(json.dumps(spec))
                    loaded = load_protocol_spec(path)
                    self.assertEqual(loaded, spec)
                    partition_seeds = {
                        seed for matrix in loaded["partitions"].values() for seed in matrix["seeds"]
                    }
                    self.assertEqual(partition_seeds, {0, 1, 2})
                    self.assertEqual(
                        partition_seeds | set(loaded.get("reserved_training_seeds", [])),
                        {0, 1, 2} | set(extra.get("reserved_training_seeds", [])),
                    )
            spec["partitions"]["blind"]["seeds"] = [0]
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                load_protocol_spec(path)

    def test_reserved_training_seeds_reject_invalid_values_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            for values in (None, True, 42, "42", {"seed": 42}, [True], [False],
                           [0.0], ["42"], [None], [[]], [{}], [-1], [2**32], [42, 42]):
                with self.subTest(values=values):
                    spec = {**self.protocol_spec(), "reserved_training_seeds": values}
                    path.write_text(json.dumps(spec))
                    with self.assertRaisesRegex(ValueError, "reserved_training_seeds.*unique uint32"):
                        load_protocol_spec(path)

    def test_explicit_run_rejects_missing_config_or_unrelated_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = self.drq_candidate(root)
            (root / "config.json").unlink()
            with self.assertRaisesRegex(ValueError, "config.json provenance"):
                discover_candidates([actor], run_dir=root)
            other = root / "unrelated"
            other.mkdir()
            with self.assertRaisesRegex(ValueError, "belong"):
                discover_candidates([actor], run_dir=other)

    def test_custom_protocol_publishes_repeat_selection_summary_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = self.drq_candidate(root)
            spec_path = root / "protocol.json"
            spec = {**self.protocol_spec(), "reserved_training_seeds": [0, 1, 2, 42, 2**32 - 1]}
            spec_path.write_text(json.dumps(spec))
            args = SimpleNamespace(
                model=[actor], run_dir=root, legacy_model=None, protocol_file=spec_path,
                partition="screen", output=root / "pointer.json", max_steps=2,
                frame_skip=4, evaluations_dir=root / "evaluations", timeout_seconds=30, workers=2,
            )

            def worker(candidate, track_id, seed, repeat, **_arguments):
                if repeat == 0:
                    time.sleep(.01)
                return {**self.episode(repeat), "candidate_id": candidate["candidate_id"],
                        "track_id": track_id, "seed": seed, "runtime": {"test": True}}

            with patch("evaluate_policy.run_isolated_cell", side_effect=worker) as run:
                result_dir = run_checkpoint_protocol(args, "ad-hoc")
            pointer = json.loads(args.output.read_text())
            ranked = json.loads((result_dir / "summary.json").read_text())
            self.assertEqual(run.call_count, 2)
            self.assertEqual([call.args[1:3] for call in run.call_args_list], [(1, 0), (1, 0)])
            self.assertEqual(pointer["ranked"], ranked)
            self.assertEqual(Path(pointer["evaluation_dir"]), result_dir)
            self.assertEqual(pointer["partition"], "screen")
            self.assertEqual(pointer["protocol_sha256"], hashlib.sha256(spec_path.read_bytes()).hexdigest())
            self.assertTrue(ranked[0]["eligible"])
            self.assertTrue(ranked[0]["determinism_audited"])
            self.assertTrue(ranked[0]["cpu_reload_matches"])
            self.assertEqual(ranked[0]["summary"]["n_episodes"], 1)
            self.assertEqual((result_dir / "protocol_spec.json").read_bytes(), spec_path.read_bytes())
            with self.assertRaises(FileExistsError):
                run_checkpoint_protocol(args, "ad-hoc")

    def test_blind_rejects_multiple_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "protocol.json"
            spec.write_text(json.dumps(self.protocol_spec()))
            with patch("sys.argv", ["evaluate_policy.py", "--model", "one.pt", "--model", "two.pt",
                                     "--run-dir", str(root), "--protocol-file", str(spec),
                                     "--partition", "blind"]):
                args = parse_args()
            with self.assertRaisesRegex(ValueError, "one already-selected"):
                validate_protocol_request(args)

    def test_duplicate_repeats_and_failed_reload_cannot_be_selected(self):
        candidate = {"candidate_id": "candidate", "expected_cells": 1, "expected_results": 2}
        episodes = [self.episode(), self.episode()]
        _, non_reproducible, unaudited = determinism_audit(episodes)
        summary = candidate_summary(candidate, episodes, non_reproducible, unaudited)
        self.assertFalse(summary["eligible"])
        failed = {**self.episode(1), "status": "exception"}
        episodes = [self.episode(), failed]
        _, non_reproducible, unaudited = determinism_audit(episodes)
        summary = candidate_summary(candidate, episodes, non_reproducible, unaudited)
        self.assertFalse(summary["eligible"])
        self.assertFalse(summary["cpu_reload_matches"])
        self.assertEqual(summary["operational_failures"], 1)

    def test_resource_violation_in_second_repeat_blocks_selection(self):
        candidate = {"candidate_id": "candidate", "algorithm": "drq-v2",
                     "expected_cells": 1, "expected_results": 2}
        for key, value in (
            ("process_initialization_seconds", 10.01), ("agent_reset_seconds", 5.01),
            ("max_action_seconds", 5.01), ("peak_rss_bytes", MAX_PROCESS_RSS_BYTES + 1),
            ("max_action_seconds", float("nan")),
        ):
            with self.subTest(resource=key, value=value):
                episodes = [self.episode(), {**self.episode(1), key: value}]
                _, non_reproducible, unaudited = determinism_audit(episodes)
                summary = candidate_summary(candidate, episodes, non_reproducible, unaudited)
                self.assertFalse(summary["eligible"])
                self.assertFalse(summary["cpu_reload_matches"])
                self.assertEqual(summary["operational_failures"], 1)

    def test_confirmation_and_blind_bind_immutable_predecessor_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = self.drq_candidate(root)
            spec_path = root / "protocol.json"
            spec_path.write_text(json.dumps(self.protocol_spec()))
            args = SimpleNamespace(
                model=[actor], run_dir=root, legacy_model=None, protocol_file=spec_path,
                partition="screen", output=root / "screen.json", previous_evaluation=None,
                max_steps=2, frame_skip=4, evaluations_dir=root / "evaluations", timeout_seconds=30,
            )

            def worker(candidate, track_id, seed, repeat, **_arguments):
                return {**self.episode(repeat), "candidate_id": candidate["candidate_id"],
                        "track_id": track_id, "seed": seed, "finished": True,
                        "lap_time_ms": 80, "progress": 1.0}

            with patch("evaluate_policy.run_isolated_cell", side_effect=worker):
                screen_dir = run_checkpoint_protocol(args, "ad-hoc")
                screen_pointer = args.output
                args.partition = "confirmation"
                args.output = root / "confirmation.json"
                with self.assertRaisesRegex(ValueError, "previous-evaluation"):
                    run_checkpoint_protocol(args, "ad-hoc")
                args.previous_evaluation = screen_pointer
                confirmation_dir = run_checkpoint_protocol(args, "ad-hoc")
                self.assertEqual(
                    (confirmation_dir / "previous_evaluation.json").read_bytes(), screen_pointer.read_bytes()
                )
                confirmation_pointer = args.output
                args.partition = "blind"
                args.output = root / "blind.json"
                with self.assertRaisesRegex(ValueError, "preceding partition"):
                    run_checkpoint_protocol(args, "ad-hoc")
                args.previous_evaluation = confirmation_pointer
                run_checkpoint_protocol(args, "ad-hoc")
            pointer = json.loads(screen_pointer.read_text())
            candidate = pointer["ranked"][0]
            with self.assertRaisesRegex(ValueError, "protocol"):
                previous_evaluation_metadata(screen_pointer, "confirmation", "wrong-hash", candidate)
            with self.assertRaisesRegex(ValueError, "exact actor"):
                previous_evaluation_metadata(screen_pointer, "confirmation", pointer["protocol_sha256"],
                                             {**candidate, "archive_sha256": "wrong-actor"})
            pointer["ranked"][0]["summary"]["finish_rate"] = 0.0
            screen_pointer.write_text(json.dumps(pointer))
            with self.assertRaisesRegex(ValueError, "immutable"):
                previous_evaluation_metadata(screen_pointer, "confirmation", pointer["protocol_sha256"], candidate)
            (screen_dir / "summary.json").write_text(json.dumps(pointer["ranked"]))
            with self.assertRaisesRegex(ValueError, "nonzero completion"):
                previous_evaluation_metadata(screen_pointer, "confirmation", pointer["protocol_sha256"], candidate)

    def test_selection_order_is_finish_rate_progress_completed_lap(self):
        def ranked(finish, progress, lap):
            return {"eligible": True, "operational_failures": 0, "by_track": {},
                    "summary": {"finish_rate": finish, "avg_progress": progress,
                                "avg_lap_time_ms": lap}}
        self.assertGreater(ranking_key(ranked(.5, .2, 100)), ranking_key(ranked(0, 1, None)))
        self.assertGreater(ranking_key(ranked(.5, .8, 100)), ranking_key(ranked(.5, .2, 10)))
        self.assertGreater(ranking_key(ranked(.5, .8, 10)), ranking_key(ranked(.5, .8, 100)))

    def test_diagnostic_confirmation_requires_explicit_custom_confirmation(self):
        for options in (
            [], ["--protocol-file", "protocol.json", "--partition", "screen"],
            ["--protocol-file", "protocol.json", "--partition", "blind"],
        ):
            with self.subTest(options=options), patch(
                "sys.argv", ["evaluate_policy.py", "--diagnostic-confirmation", *options]
            ):
                args = parse_args()
                with self.assertRaisesRegex(ValueError, "requires custom partition=confirmation"):
                    validate_protocol_request(args)
                with self.assertRaisesRegex(ValueError, "requires custom partition=confirmation"):
                    run_checkpoint_protocol(args, "ad-hoc")

    def test_diagnostic_confirmation_only_relaxes_screen_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = self.drq_candidate(root)
            spec_path = root / "protocol.json"
            spec_path.write_text(json.dumps(self.protocol_spec()))
            args = SimpleNamespace(
                model=[actor], run_dir=root, legacy_model=None, protocol_file=spec_path,
                partition="screen", output=root / "screen.json", previous_evaluation=None,
                max_steps=2, frame_skip=4, evaluations_dir=root / "evaluations", timeout_seconds=30,
                diagnostic_confirmation=False,
            )

            def worker(candidate, track_id, seed, repeat, **_arguments):
                episode = {**self.episode(repeat), "candidate_id": candidate["candidate_id"],
                           "track_id": track_id, "seed": seed}
                if scenario == "nondeterministic" and repeat == 1:
                    episode["action_trace_sha256"] = "b" * 64
                if scenario == "resource_failure" and repeat == 1:
                    episode["peak_rss_bytes"] = MAX_PROCESS_RSS_BYTES + 1
                if args.partition == "confirmation":
                    episode.update(finished=True, lap_time_ms=80, progress=1.0)
                return episode

            for scenario in ("nondeterministic", "resource_failure", "zero_finish"):
                args.partition = "screen"
                args.diagnostic_confirmation = False
                args.previous_evaluation = None
                args.output = root / f"{scenario}-screen.json"
                with patch("evaluate_policy.run_isolated_cell", side_effect=worker):
                    run_checkpoint_protocol(args, "ad-hoc")
                args.previous_evaluation = args.output
                args.partition = "confirmation"
                args.output = root / f"{scenario}-confirmation.json"
                with self.assertRaisesRegex(ValueError, "nonzero completion"):
                    run_checkpoint_protocol(args, "ad-hoc")
                args.diagnostic_confirmation = True
                if scenario != "zero_finish":
                    with self.assertRaisesRegex(ValueError, "operational success"):
                        run_checkpoint_protocol(args, "ad-hoc")
                    continue
                screen_pointer = json.loads(args.previous_evaluation.read_text())
                self.assertFalse(screen_pointer["diagnostic_only"])
                with self.assertRaisesRegex(ValueError, "protocol"):
                    previous_evaluation_metadata(
                        args.previous_evaluation, "confirmation", "changed-protocol",
                        screen_pointer["ranked"][0], diagnostic_confirmation=True,
                    )
                with self.assertRaisesRegex(ValueError, "exact actor"):
                    previous_evaluation_metadata(
                        args.previous_evaluation, "confirmation", screen_pointer["protocol_sha256"],
                        {"archive_sha256": "changed-actor"}, diagnostic_confirmation=True,
                    )
                args.model = [actor, actor]
                with self.assertRaisesRegex(ValueError, "one already-selected actor"):
                    run_checkpoint_protocol(args, "ad-hoc")
                args.model = [actor]
                with patch("evaluate_policy.run_isolated_cell", side_effect=worker):
                    result_dir = run_checkpoint_protocol(args, "ad-hoc")
            pointer_bytes = args.output.read_text()
            manifest_bytes = (result_dir / "manifest.json").read_text()
            pointer = json.loads(pointer_bytes)
            candidate = pointer["ranked"][0]
            self.assertTrue(pointer["diagnostic_only"])
            self.assertTrue(json.loads(manifest_bytes)["diagnostic_only"])
            self.assertTrue(candidate["diagnostic_only"])
            self.assertTrue(candidate["eligible"])
            self.assertEqual(candidate["summary"]["finish_rate"], 1.0)
            self.assertEqual(json.loads((result_dir / "summary.json").read_text()), pointer["ranked"])
            self.assertIn("DIAGNOSTIC ONLY: NON-PROMOTING", (result_dir / "report.md").read_text())
            self.assertIn("failed screen remains failed", (result_dir / "report.md").read_text())
            for location in ("pointer", "manifest", "summary"):
                with self.subTest(diagnostic_marker=location):
                    pointer = json.loads(pointer_bytes)
                    manifest = json.loads(manifest_bytes)
                    pointer["diagnostic_only"] = location == "pointer"
                    manifest["diagnostic_only"] = location == "manifest"
                    pointer["ranked"][0]["diagnostic_only"] = location == "summary"
                    args.output.write_text(json.dumps(pointer))
                    (result_dir / "summary.json").write_text(json.dumps(pointer["ranked"]))
                    (result_dir / "manifest.json").write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                        previous_evaluation_metadata(
                            args.output, "blind", pointer["protocol_sha256"], candidate,
                        )

    def test_isolated_worker_uses_explicit_interpreter_and_frozen_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = self.drq_candidate(root)
            candidate, = discover_candidates([actor])
            candidate["_worker_model_path"] = str(actor)
            explicit_python = (
                r"C:\cpu-venv\bin\python.exe"
                if sys.platform == "win32"
                else "/cpu-venv/bin/python"
            )
            args = SimpleNamespace(python=explicit_python, max_steps=2,
                                   frame_skip=4, timeout_seconds=30)
            completed = SimpleNamespace(returncode=0, stdout=json.dumps(self.episode()), stderr="")
            with patch.dict("os.environ", {"PYTHONPATH": "/unsafe/repository"}):
                with patch("evaluate_policy.subprocess.run", return_value=completed) as run:
                    result = run_isolated_cell(candidate, 1, 0, 0, args, root / "evaluate_policy.py")
            self.assertEqual(run.call_args.args[0][0], explicit_python)
            self.assertEqual(run.call_args.kwargs["cwd"], root)
            self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])
            self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(result["status"], "ok")

    def test_drq_cell_resets_agent_records_trace_and_computes_official_lap(self):
        class Environment:
            def __init__(self):
                self.unwrapped = self
                self.t = 1.0
                self.closed = False

            def reset(self):
                return np.zeros((4, 84, 84), dtype=np.float32), {}

            def step(self, action):
                self.t += .08
                return np.zeros((4, 84, 84), dtype=np.float32), 1.0, False, True, {
                    "finished": True, "finish_time_s": self.t, "progress": 1.0,
                    "damage": 0.2,
                }

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            actor = self.drq_candidate(Path(directory))
            env = Environment()
            with patch("evaluate_policy.build_env", return_value=env):
                with patch("agent.Agent.reset", autospec=True) as reset, patch(
                    "evaluate_policy.peak_rss_bytes", return_value=128 * 1024 * 1024
                ):
                    result = evaluate_cell(actor, 1, 0, 2, 4)
            self.assertEqual(reset.call_count, 2)
            self.assertEqual(result["lap_time_ms"], 80)
            self.assertFalse(result["terminated"])
            self.assertTrue(result["truncated"])
            self.assertEqual(len(result["actions"]), 1)
            self.assertEqual(len(result["action_trace_sha256"]), 64)
            self.assertGreater(result["peak_rss_bytes"], 0)
            self.assertGreater(result["max_action_seconds"], 0)
            self.assertTrue(env.closed)
            with patch("evaluate_policy.build_env", return_value=Environment()):
                with patch("evaluate_policy.peak_rss_bytes", return_value=MAX_PROCESS_RSS_BYTES + 1):
                    with self.assertRaises(MemoryError):
                        evaluate_cell(actor, 1, 0, 2, 4)

    def test_policy_call_budget_and_malformed_worker_output(self):
        with self.assertRaises(TimeoutError):
            timed_policy_call(time.sleep, .05, seconds=.001)
        for output in ('null', '[]', '{"status": "ok"}', 'not json'):
            self.assertEqual(parse_worker_output(output)["error_type"], "WorkerOutputError")

    def test_cpu_runtime_validation_rejects_cuda_build_or_wrong_torch(self):
        runtime = {
            "python_version": [3, 11], "sys_platform": "linux", "torch_cuda": None,
            "cuda_available": False, "torch_threads": 1, "torch_interop_threads": 1,
            "packages": {"torch": "2.1.0+cpu", "numpy": "1.26.0",
                         "gymnasium": "0.29.1", "opencv-python": "4.8.1.78"},
        }
        validate_cpu_runtime(runtime)
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_cpu_runtime({**runtime, "torch_cuda": "12.1"})
        runtime["packages"]["torch"] = "2.11.0+cpu"
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_cpu_runtime(runtime)

    def test_runtime_preflight_validates_protocol_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            for spec, expected in ((self.protocol_spec(), 0), ({"name": "invalid"}, 1)):
                path.write_text(json.dumps(spec))
                with patch("sys.argv", ["evaluate_policy.py", "--check-runtime", "--protocol-file", str(path)]):
                    with patch("evaluate_policy.runtime_metadata", return_value={}), patch("evaluate_policy.validate_cpu_runtime"):
                        with patch("torch.set_num_interop_threads"), patch("builtins.print"):
                            self.assertEqual(evaluate_policy.main(), expected)



if __name__ == "__main__":
    unittest.main()
