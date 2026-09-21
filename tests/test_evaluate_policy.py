import hashlib
import tempfile
import unittest
import zipfile
import pickle
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from evaluate_policy import (
    PROTOCOLS,
    determinism_audit,
    discover_candidates,
    evaluate_model,
    parse_int_list,
    ranking_key,
    snapshot_candidates,
    snapshot_runtime,
    terminal_class,
)


class TestEvaluatePolicy(unittest.TestCase):
    def test_blind_protocol_is_disjoint_from_training_and_development_sets(self):
        blind_seeds = set(PROTOCOLS["checkpoint-v1-blind"]["seeds"])
        development_seeds = set(PROTOCOLS["checkpoint-v1-screen"]["seeds"])
        development_seeds.update(PROTOCOLS["checkpoint-v1-confirmation"]["seeds"])

        self.assertTrue(blind_seeds.isdisjoint(development_seeds))
        self.assertTrue(blind_seeds.isdisjoint({42, 777, 1337, 2024}))
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



if __name__ == "__main__":
    unittest.main()
