import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from haic_agent.dynamics import LatentDynamicsEnsemble
from haic_agent.networks import VisualActorCritic
from training.evaluate_closed_loop import (
    DEFAULT_HELD_OUT_EPISODES,
    aggregate_episode_results,
    choose_planner_from_tune,
    validate_episode_splits,
)
from training.package_submission import (
    build_submission,
    validate_checkpoints,
    validate_planner_settings,
    validate_submission_archive,
)
from training.package_submission import resolve_package_selection


ROOT = Path(__file__).resolve().parents[1]


class TestClosedLoopEvaluationContract(unittest.TestCase):
    def test_held_out_manifest_has_ten_unique_episodes_and_no_split_leakage(self):
        self.assertGreaterEqual(len(DEFAULT_HELD_OUT_EPISODES), 10)
        self.assertEqual(len(DEFAULT_HELD_OUT_EPISODES), len(set(DEFAULT_HELD_OUT_EPISODES)))
        validate_episode_splits(((1, 101),), ((2, 201),), DEFAULT_HELD_OUT_EPISODES)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_episode_splits(((1, 101),), ((1, 101),), DEFAULT_HELD_OUT_EPISODES)

    def test_tune_result_only_enables_planner_when_rank_improves(self):
        ppo_only = {"completion_rate": 0.0, "median_finished_lap_time_ms": None, "mean_progress": 0.2}
        not_better = {"completion_rate": 0.0, "median_finished_lap_time_ms": None, "mean_progress": 0.2}
        better = {"completion_rate": 0.0, "median_finished_lap_time_ms": None, "mean_progress": 0.3}

        self.assertFalse(choose_planner_from_tune(ppo_only, not_better))
        self.assertTrue(choose_planner_from_tune(ppo_only, better))

    def test_summary_keeps_dnf_episodes_and_latency_percentiles(self):
        summary = aggregate_episode_results(
            [
                {
                    "mode": "ppo_only",
                    "completed": False,
                    "lapTimeMs": None,
                    "progress": 0.4,
                    "act_latency_ms": [1.0, 3.0],
                },
                {
                    "mode": "ppo_only",
                    "completed": True,
                    "lapTimeMs": 1234,
                    "progress": 1.0,
                    "act_latency_ms": [2.0, 4.0],
                },
            ]
        )
        metrics = summary["by_mode"]["ppo_only"]
        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["completed_episodes"], 1)
        self.assertEqual(metrics["median_finished_lap_time_ms"], 1234.0)
        self.assertEqual(metrics["act_latency_ms"]["p50"], 2.5)


class TestInferenceOnlySubmissionArchive(unittest.TestCase):
    def test_corrupt_checkpoints_raise_the_packaging_error_contract(self):
        for index, contents in enumerate((b"", b"not a checkpoint")):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                directory_path = Path(directory)
                policy = directory_path / f"policy-{index}.pt"
                dynamics = directory_path / f"dynamics-{index}.pt"
                policy.write_bytes(contents)
                dynamics.write_bytes(contents)

                with self.assertRaisesRegex(ValueError, "strict-loadable CPU checkpoints"):
                    validate_checkpoints(policy, dynamics)

    def test_invalid_planner_values_are_rejected_even_when_planner_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            policy = directory_path / "policy.pt"
            dynamics = directory_path / "dynamics.pt"
            archive = directory_path / "invalid-submission.zip"
            torch.save({"model_state": VisualActorCritic().state_dict()}, policy)
            torch.save({"model": LatentDynamicsEnsemble().state_dict()}, dynamics)

            with self.assertRaisesRegex(ValueError, "planner settings"):
                build_submission(
                    source_root=ROOT,
                    policy_checkpoint=policy,
                    dynamics_checkpoint=dynamics,
                    archive_path=archive,
                    planner_enabled=False,
                    planner_settings={
                        "horizon": 0,
                        "population": 16,
                        "iterations": 2,
                        "candidate_batch_size": 8,
                        "uncertainty_cost": 1.0,
                    },
                )

            self.assertFalse(archive.exists())

    def test_planner_settings_reject_unbounded_dimensions_and_total_work(self):
        baseline = {
            "horizon": 4,
            "population": 16,
            "iterations": 2,
            "candidate_batch_size": 8,
            "uncertainty_cost": 1.0,
        }
        oversized = (
            {"horizon": 10_000_000},
            {"population": 10_000_000},
            {"iterations": 10_000_000},
            {"candidate_batch_size": 10_000_000},
            {"horizon": 32, "population": 256, "iterations": 8, "candidate_batch_size": 64},
        )
        for changes in oversized:
            with self.subTest(changes=changes):
                settings = {**baseline, **changes}
                with self.assertRaisesRegex(ValueError, "planner settings"):
                    validate_planner_settings(settings)

    def test_package_selection_uses_summary_or_safely_disables_planner(self):
        disabled, default_settings = resolve_package_selection(None)
        self.assertFalse(disabled)
        self.assertEqual(default_settings["horizon"], 4)
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            summary.write_text(json.dumps({
                "package_planner_enabled": False,
                "selected_planner_settings": {"horizon": 3, "population": 8, "iterations": 1,
                                              "candidate_batch_size": 8, "uncertainty_cost": 0.5},
            }))
            enabled, settings = resolve_package_selection(summary)
            self.assertFalse(enabled)
            self.assertEqual(settings["population"], 8)
            enabled_override, _ = resolve_package_selection(summary, planner_override=True)
            self.assertTrue(enabled_override)

    def test_archive_has_only_inference_code_and_cpu_smoke_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            policy = directory_path / "policy.pt"
            dynamics = directory_path / "dynamics.pt"
            archive = directory_path / "submission.zip"
            torch.save({"model_state": VisualActorCritic().state_dict()}, policy)
            torch.save({"model": LatentDynamicsEnsemble().state_dict()}, dynamics)

            result = build_submission(
                source_root=ROOT,
                policy_checkpoint=policy,
                dynamics_checkpoint=dynamics,
                archive_path=archive,
                smoke_test=True,
                planner_enabled=False,
            )

            self.assertTrue(result["smoke"]["finite_action"])
            self.assertFalse(result["planner_enabled"])
            self.assertEqual(result["planner_settings"]["horizon"], 4)
            self.assertNotIn("haic_agent/corridor_agent.py", result["layout"]["files"])
            self.assertNotIn("controller_mode", result["layout"])
            self.assertEqual(result["layout"]["runtime_policy"], "trained_visual_actor")
            with zipfile.ZipFile(archive) as zipped:
                names = set(zipped.namelist())
            self.assertIn("agent.py", names)
            self.assertIn("policy.pt", names)
            self.assertIn("dynamics.pt", names)
            self.assertNotIn("haic_agent/corridor_agent.py", names)
            self.assertFalse(any(name.startswith("training/") for name in names))
            self.assertFalse(any(".venv" in name or "labels" in name for name in names))
            manifest = validate_submission_archive(archive)
            self.assertEqual(manifest["root_agent"], "agent.py")
            self.assertFalse(manifest["planner_enabled"])
            self.assertTrue(manifest["strict_checkpoint_loading"])

    def test_package_api_cannot_select_corridor_runtime_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            policy = directory_path / "policy.pt"
            dynamics = directory_path / "dynamics.pt"
            archive = directory_path / "corridor.zip"
            torch.save({"model_state": VisualActorCritic().state_dict()}, policy)
            torch.save({"model": LatentDynamicsEnsemble().state_dict()}, dynamics)

            with self.assertRaises(TypeError):
                build_submission(
                    source_root=ROOT,
                    policy_checkpoint=policy,
                    dynamics_checkpoint=dynamics,
                    archive_path=archive,
                    smoke_test=False,
                    planner_enabled=False,
                    controller_mode="corridor",
                )


if __name__ == "__main__":
    unittest.main()
