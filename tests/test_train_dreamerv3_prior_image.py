"""Synthetic runner contracts only; no real training or environment is opened."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import train_dreamerv3_prior_image as runner
from tests import test_train_dreamerv3_reused_train as baseline_tests


class _Replay:
    size = 3
    memory_bytes = 180
    is_terminal = np.array([False, False, True])


class _Agent:
    calls: list[bool] = []

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed
        self.actor = torch.nn.Linear(1, 1)
        self.critic = torch.nn.Linear(1, 1)
        self.critic_target = torch.nn.Linear(1, 1)
        self.wm_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        self.wm_optimizer = torch.optim.Adam([self.wm_parameter], lr=0.01)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.01)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.01)
        self.gradient_steps = 0
        self.environment_steps = 0

    def wm_step(self):
        self.wm_parameter.grad = torch.ones_like(self.wm_parameter)
        self.wm_optimizer.step()

    def update(self, *, model_only=False):
        self.calls.append(model_only)
        if not model_only:
            raise AssertionError("actor update forbidden")
        self.wm_step()
        self.gradient_steps += 1
        return {"loss_wm": 1.0}

    def save_checkpoint(self, path, *, run_metadata=None):
        torch.save({"run_metadata": run_metadata, "environment_steps": self.environment_steps,
                    "gradient_steps": self.gradient_steps,
                    "wm_optimizer": self.wm_optimizer.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict()}, path)


class PriorRunnerFixture(unittest.TestCase):
    put = baseline_tests.TestReusedTrainOffline.put
    freeze = baseline_tests.TestReusedTrainOffline.freeze
    episode_bytes = staticmethod(baseline_tests.TestReusedTrainOffline.episode_bytes)

    def setUp(self):
        # Use the original synthetic dual-arm fixture, not its real sealed archives.
        baseline_tests.TestReusedTrainOffline.setUp(self)
        self.cgroup.update(limit_bytes=16 * 1024**3, used_bytes=4 * 1024**3,
                           available_bytes=12 * 1024**3)
        self.protocol_name = runner.BASE_PROTOCOL_PATH
        self.collection["study_id"] = "dreamerv3-reused-train-diagnostic-v1"
        self.put(self.collection_name, self.collection)
        self.protocol["study_id"] = self.collection["study_id"]
        self.protocol["collection_protocol"]["sha256"] = runner.original._sha256(
            self.root / self.collection_name,
        )
        for arm in ("random", "teacher"):
            row = self.protocol["datasets"][arm]
            receipt = json.loads((self.root / row["receipt_path"]).read_text())
            receipt["study_id"] = self.collection["study_id"]
            receipt["protocol_sha256"] = self.protocol["collection_protocol"]["sha256"]
            self.put(row["receipt_path"], receipt)
            row["receipt_sha256"] = runner.original._sha256(self.root / row["receipt_path"])
        self.protocol["learner"]["updates"] = 256
        self.protocol["learner"]["seeds"] = [0, 1]
        self.protocol["learner"]["config"].update(seq_len=32, burnin_steps=8)
        self.protocol["resources"].update(max_cgroup_memory_bytes=16 * 1024**3,
                                          min_cgroup_available_bytes=8 * 1024**3)
        self.base_sha = self.freeze()
        base_patch = patch.object(runner, "BASE_PROTOCOL_SHA256", self.base_sha)
        base_patch.start()
        self.addCleanup(base_patch.stop)
        self.output_name = f"{runner.RUN_ROOT}/teacher-seed-0"
        (self.root / runner.RUN_ROOT).mkdir(parents=True)
        self.prior_protocol_name = "experiments/dreamerv3-reused-train-prior-image-v1.json"
        helper_path = self.root / runner.HELPER_SOURCE
        helper_path.parent.mkdir(parents=True, exist_ok=True)
        helper_path.write_text("# synthetic helper SHA identity\n", encoding="ascii")
        helper_patch = patch.object(runner, "HELPER_SHA256", runner.original._sha256(helper_path))
        helper_patch.start()
        self.addCleanup(helper_patch.stop)
        script_path = self.root / runner.RUNNER_SOURCE
        shutil.copyfile(runner.__file__, script_path)
        self.prior_protocol = {
            "format": runner.FORMAT, "purpose": runner.PURPOSE,
            "study_id": "dreamerv3-reused-train-diagnostic-v1",
            "base_offline_protocol": {"path": runner.BASE_PROTOCOL_PATH, "sha256": self.base_sha},
            "source_sha256": {
                **self.protocol["source_sha256"],
                runner.RUNNER_SOURCE: runner.original._sha256(script_path),
                runner.HELPER_SOURCE: runner.original._sha256(helper_path),
            },
            "learner": {
                "base_updates": 256, "aux_updates": 64, "aux_every_base_updates": 4,
                "prior_horizon": 8, "prior_weight": 0.25,
                "anchor_mode": "deterministic-first-last-full-horizon",
                "aux_rng_mode": "fork-torch-numpy-python",
                "aux_rng_seed_base": 260926000, "aux_rng_seed_stride": 1000,
            },
            "resources": copy.deepcopy(self.protocol["resources"]),
            "output_root": runner.RUN_ROOT,
        }
        self.events = []
        _Agent.calls = []
        self.agent_patch = patch.object(runner, "DreamerV3Agent", _Agent)
        self.replay_patch = patch.object(runner, "replay_from_dataset_bytes", return_value=(
            _Replay(), {"decisions": 3, "distinct_finished_cells": 1, "episodes": 1},
        ))
        self.helper_patch = patch.object(runner, "_load_aux_update", return_value=self.aux)
        for active in (self.agent_patch, self.replay_patch, self.helper_patch):
            active.start()
            self.addCleanup(active.stop)

    def freeze_prior(self):
        self.put(self.prior_protocol_name, self.prior_protocol)
        return runner.original._sha256(self.root / self.prior_protocol_name)

    def preflight_prior(self, *, arm="teacher", seed=0, output=None):
        return runner.preflight(self.prior_protocol_name, self.freeze_prior(), arm, seed,
                                output or self.output_name, repo_root=self.root)

    def run_prior(self, *, arm="teacher", seed=0, output=None):
        return runner.train_arm(self.prior_protocol_name, self.freeze_prior(), arm, seed,
                                output or self.output_name, repo_root=self.root)

    def aux(self, agent, *, horizon, weight):
        self.events.append((agent.gradient_steps, horizon, weight))
        agent.wm_step()
        return {"aux_loss": 0.25, "frame_mse": 1.0, "anchor_count": 2.0,
                "target_count": 16.0, "effective_horizon": 8.0}


class PriorRunnerTests(PriorRunnerFixture):
    def test_exact_base_and_aux_schedule_and_separate_receipt_hashes(self):
        result = self.run_prior()
        self.assertEqual(_Agent.calls, [True] * 256)
        self.assertEqual(self.events, [(step, 8, 0.25) for step in range(4, 257, 4)])
        self.assertEqual((result["base_model_only_updates"], result["aux_world_model_optimizer_steps"],
                          result["world_model_optimizer_steps"], result["environment_steps"]),
                         (256, 64, 320, 0))
        self.assertFalse(result["matched_pure_256_step_model"])
        self.assertTrue(result["actor_critic_target_unchanged"])
        self.assertTrue(result["actor_critic_optimizers_unchanged"])
        output = self.root / self.output_name
        for field, sha_field in (("base_metrics_path", "base_metrics_sha256"),
                                 ("aux_metrics_path", "aux_metrics_sha256"),
                                 ("checkpoint_path", "checkpoint_sha256")):
            self.assertEqual(runner.original._sha256(output / result[field]), result[sha_field])
        base_rows = [json.loads(row) for row in (output / result["base_metrics_path"]).read_text().splitlines()]
        aux_rows = [json.loads(row) for row in (output / result["aux_metrics_path"]).read_text().splitlines()]
        self.assertEqual([row["base_update"] for row in base_rows], list(range(1, 257)))
        self.assertEqual([row["after_base_update"] for row in aux_rows], list(range(4, 257, 4)))
        checkpoint = torch.load(output / result["checkpoint_path"], weights_only=False)
        self.assertEqual(checkpoint["gradient_steps"], 256)
        self.assertEqual(checkpoint["run_metadata"]["world_model_optimizer_steps"], 320)
        self.assertEqual(checkpoint["run_metadata"]["base_offline_protocol_sha256"], self.base_sha)
        self.assertEqual(checkpoint["run_metadata"]["source_sha256"], result["source_sha256"])
        self.assertEqual(checkpoint["run_metadata"]["base_metrics_sha256"], result["base_metrics_sha256"])
        self.assertEqual(checkpoint["run_metadata"]["aux_metrics_sha256"], result["aux_metrics_sha256"])
        self.assertFalse(checkpoint["actor_optimizer"]["state"])
        self.assertFalse(checkpoint["critic_optimizer"]["state"])
        with patch.object(runner.original, "_archive_bytes", side_effect=AssertionError("overwrite read")):
            with self.assertRaises(FileExistsError):
                self.run_prior()

    def test_all_four_arm_seed_paths_are_isolated_and_source_identical(self):
        for arm in ("random", "teacher"):
            for seed in (0, 1):
                with self.subTest(arm=arm, seed=seed):
                    output = f"{runner.RUN_ROOT}/{arm}-seed-{seed}"
                    checked = self.preflight_prior(arm=arm, seed=seed, output=output)
                    self.assertEqual(checked["prior_output"], self.root / output)
                    self.assertEqual(checked["datasets"][arm]["row"]["stored_decisions"], 3)

    def test_source_and_protocol_sha_drift_fail_before_archive_or_training(self):
        cases = (
            ("base reference", lambda: self.prior_protocol["base_offline_protocol"].update(sha256="0" * 64)),
            ("original source", lambda: self.prior_protocol["source_sha256"].update({"dreamer_v3.py": "0" * 64})),
            ("new helper", lambda: self.prior_protocol["source_sha256"].update({runner.HELPER_SOURCE: "0" * 64})),
            ("new runner", lambda: self.prior_protocol["source_sha256"].update({runner.RUNNER_SOURCE: "0" * 64})),
            ("collector receipt", lambda: self.protocol["datasets"]["random"].update(receipt_sha256="0" * 64)),
        )
        for title, mutate in cases:
            with self.subTest(title=title):
                original_prior = copy.deepcopy(self.prior_protocol)
                original_base = copy.deepcopy(self.protocol)
                mutate()
                if title == "collector receipt":
                    self.freeze()
                    runner.BASE_PROTOCOL_SHA256 = runner.original._sha256(self.root / self.protocol_name)
                    self.prior_protocol["base_offline_protocol"]["sha256"] = runner.BASE_PROTOCOL_SHA256
                with patch.object(runner.original, "_archive_bytes", side_effect=AssertionError("archive read")):
                    with self.assertRaises(ValueError):
                        self.run_prior()
                self.assertFalse((self.root / self.output_name).exists())
                self.prior_protocol = original_prior
                self.protocol = original_base
                self.freeze()
                runner.BASE_PROTOCOL_SHA256 = self.base_sha

    def test_executing_helper_must_match_file_path_and_sha(self):
        self.helper_patch.stop()
        helper = self.root / runner.HELPER_SOURCE
        update = lambda agent, *, horizon, weight: {}
        module = SimpleNamespace(__file__=str(helper), prior_image_aux_update=update)
        with patch.object(runner.importlib, "import_module", return_value=module):
            self.assertIs(runner._load_aux_update(
                self.root, self.prior_protocol["source_sha256"][runner.HELPER_SOURCE],
            ), update)
            with self.assertRaisesRegex(ValueError, "executing prior-image helper"):
                runner._load_aux_update(self.root, "0" * 64)
            module.__file__ = str(self.root / runner.RUNNER_SOURCE)
            with self.assertRaisesRegex(ValueError, "executing prior-image helper"):
                runner._load_aux_update(self.root, self.prior_protocol["source_sha256"][runner.HELPER_SOURCE])

    def test_on_disk_helper_drift_rejected_before_archive_read(self):
        digest = self.freeze_prior()
        (self.root / runner.HELPER_SOURCE).write_text("# source changed after freeze\n", encoding="ascii")
        with patch.object(runner.original, "_archive_bytes", side_effect=AssertionError("archive read")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner.train_arm(self.prior_protocol_name, digest, "teacher", 0,
                                 self.output_name, repo_root=self.root)
        self.assertFalse((self.root / self.output_name).exists())
        self.assertEqual(_Agent.calls, [])

    def test_prior_protocol_sha_drift_fails_before_archive_read(self):
        digest = self.freeze_prior()
        self.prior_protocol["learner"]["prior_weight"] = 0.5
        self.freeze_prior()
        with patch.object(runner.original, "_archive_bytes", side_effect=AssertionError("archive read")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner.train_arm(self.prior_protocol_name, digest, "teacher", 0,
                                 self.output_name, repo_root=self.root)
        self.assertEqual(_Agent.calls, [])
        self.assertFalse((self.root / self.output_name).exists())

    def test_outdated_helper_digest_cannot_be_frozen_even_if_file_matches(self):
        with patch.object(runner, "HELPER_SHA256", "0" * 64):
            with patch.object(runner.original, "_archive_bytes", side_effect=AssertionError("archive read")):
                with self.assertRaisesRegex(ValueError, "corrected helper"):
                    self.run_prior()
        self.assertFalse((self.root / self.output_name).exists())

    def test_aux_rng_is_deterministic_and_does_not_advance_base_streams(self):
        base_draws = []
        first_base_torch_state = []
        helper_draws = []
        original_update = _Agent.update

        def traced_update(agent, *, model_only=False):
            if not first_base_torch_state:
                first_base_torch_state.append(torch.get_rng_state().clone())
            base_draws.append((random.random(), np.random.random(), torch.rand(1).item()))
            return original_update(agent, model_only=model_only)

        def random_aux(agent, *, horizon, weight):
            helper_draws.append((random.random(), np.random.random(), torch.rand(1).item()))
            return self.aux(agent, horizon=horizon, weight=weight)

        saved_python = random.getstate()
        saved_numpy = np.random.get_state()
        saved_torch = torch.get_rng_state().clone()
        try:
            random.seed(123)
            np.random.seed(123)
            torch.random.default_generator.manual_seed(123)
            with (patch.object(_Agent, "update", traced_update),
                  patch.object(runner, "_load_aux_update", return_value=random_aux)):
                self.run_prior()
            random.seed(123)
            np.random.seed(123)
            torch.set_rng_state(first_base_torch_state[0])
            expected = [(random.random(), np.random.random(), torch.rand(1).item())
                        for _ in range(256)]
            self.assertEqual(base_draws, expected)
            self.assertEqual(len(helper_draws), 64)
            for index, draw in enumerate(helper_draws, 1):
                seed = runner.AUX_RNG_SEED_BASE + index
                np_expected = np.random.RandomState(seed).random_sample()
                py_expected = random.Random(seed).random()
                with torch.random.fork_rng(devices=[], enabled=True):
                    torch.random.default_generator.manual_seed(seed)
                    torch_expected = torch.rand(1).item()
                self.assertEqual(draw, (py_expected, np_expected, torch_expected))
        finally:
            random.setstate(saved_python)
            np.random.set_state(saved_numpy)
            torch.set_rng_state(saved_torch)

    def test_new_output_parent_created_only_after_resource_preflight(self):
        (self.root / runner.RUN_ROOT).rmdir()
        self.cgroup["available_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "cgroup memory"):
            self.run_prior()
        self.assertFalse((self.root / runner.RUN_ROOT).exists())
        self.cgroup["available_bytes"] = 12 * 1024**3
        self.run_prior()
        self.assertTrue((self.root / self.output_name / "training-result.json").is_file())

    def test_wrong_schedule_seed_output_or_collision_never_trains(self):
        self.prior_protocol["learner"]["aux_updates"] = 63
        with self.assertRaisesRegex(ValueError, "256 base plus 64"):
            self.run_prior()
        self.prior_protocol["learner"]["aux_updates"] = 64
        for seed, output in ((7, self.output_name), (0, "runs/synthetic-offline/teacher-seed-23"),
                             (0, f"{runner.RUN_ROOT}/teacher-seed-1")):
            with self.subTest(seed=seed, output=output):
                with self.assertRaises(ValueError):
                    self.preflight_prior(seed=seed, output=output)
        (self.root / self.output_name).mkdir()
        with patch.object(runner.original, "_archive_bytes", side_effect=AssertionError("archive read")):
            with self.assertRaises(FileExistsError):
                self.run_prior()
        self.assertEqual(_Agent.calls, [])

    def test_resource_floor_and_archive_cap_fail_before_training(self):
        for change in (lambda: self.cgroup.update(available_bytes=7 * 1024**3),
                       lambda: self.protocol["resources"].update(max_archive_bytes=1)):
            with self.subTest(change=change):
                original_cgroup = self.cgroup.copy()
                original_base = copy.deepcopy(self.protocol)
                original_prior = copy.deepcopy(self.prior_protocol)
                change()
                if self.protocol != original_base:
                    self.prior_protocol["resources"] = copy.deepcopy(self.protocol["resources"])
                    self.freeze()
                    runner.BASE_PROTOCOL_SHA256 = runner.original._sha256(self.root / self.protocol_name)
                    self.prior_protocol["base_offline_protocol"]["sha256"] = runner.BASE_PROTOCOL_SHA256
                with patch.object(runner.original, "_archive_bytes", side_effect=AssertionError("archive read")):
                    with self.assertRaises(ValueError):
                        self.run_prior()
                self.assertEqual(_Agent.calls, [])
                self.assertFalse((self.root / self.output_name).exists())
                self.protocol, self.prior_protocol = original_base, original_prior
                self.cgroup.clear()
                self.cgroup.update(original_cgroup)
                self.freeze()
                runner.BASE_PROTOCOL_SHA256 = self.base_sha

    def test_aux_must_step_once_and_keep_actor_critic_optimizer_untouched(self):
        def no_step(agent, *, horizon, weight):
            return {"aux_loss": 1.0, "frame_mse": 1.0, "anchor_count": 2.0,
                    "target_count": 16.0, "effective_horizon": 8.0}

        with patch.object(runner, "_load_aux_update", return_value=no_step):
            with self.assertRaisesRegex(ValueError, "aux update changed"):
                self.run_prior()
        abort = json.loads((self.root / self.output_name / "abort.json").read_text())
        self.assertEqual((abort["completed_base_model_only_updates"],
                          abort["completed_aux_world_model_optimizer_steps"],
                          abort["observed_world_model_optimizer_steps"]), (4, 0, 4))

    def test_aux_rejects_last_transition_instead_of_full_latest_horizon(self):
        def short_late_anchor(agent, *, horizon, weight):
            result = self.aux(agent, horizon=horizon, weight=weight)
            return {**result, "target_count": 9.0}

        with patch.object(runner, "_load_aux_update", return_value=short_late_anchor):
            with self.assertRaisesRegex(ValueError, "aux update changed"):
                self.run_prior()
        self.assertEqual(json.loads((self.root / self.output_name / "abort.json").read_text())[
            "completed_aux_world_model_optimizer_steps"], 0)
        self.assertTrue(runner._aux_targets({"anchor_count": 2.0, "target_count": 7.0,
                                             "effective_horizon": 6.0}, 1))
        self.assertTrue(runner._aux_targets({"anchor_count": 1.0, "target_count": 1.0,
                                             "effective_horizon": 1.0}, 1))

    def test_aux_actor_or_optimizer_mutation_aborts_without_success_receipt(self):
        def bad_actor(agent, *, horizon, weight):
            result = self.aux(agent, horizon=horizon, weight=weight)
            with torch.no_grad():
                agent.actor.weight.add_(0.01)
            return result

        with patch.object(runner, "_load_aux_update", return_value=bad_actor):
            with self.assertRaisesRegex(ValueError, "aux update changed"):
                self.run_prior()
        self.assertFalse((self.root / self.output_name / "training-result.json").exists())
        self.assertEqual(json.loads((self.root / self.output_name / "abort.json").read_text())[
            "completed_aux_world_model_optimizer_steps"], 0)
        self.output_name = f"{runner.RUN_ROOT}/teacher-seed-1"

        def bad_optimizer(agent, *, horizon, weight):
            result = self.aux(agent, horizon=horizon, weight=weight)
            agent.actor_optimizer.param_groups[0]["lr"] = 0.02
            return result

        with patch.object(runner, "_load_aux_update", return_value=bad_optimizer):
            with self.assertRaisesRegex(ValueError, "aux update changed"):
                self.run_prior(seed=1)

    def test_base_actor_mutation_is_rejected_before_aux(self):
        original_update = _Agent.update

        def bad_base(agent, *, model_only=False):
            metrics = original_update(agent, model_only=model_only)
            with torch.no_grad():
                agent.critic_target.weight.add_(0.01)
            return metrics

        with patch.object(_Agent, "update", bad_base):
            with self.assertRaisesRegex(ValueError, "base update changed"):
                self.run_prior()
        abort = json.loads((self.root / self.output_name / "abort.json").read_text())
        self.assertEqual(abort["completed_base_model_only_updates"], 0)
        self.assertEqual(abort["completed_aux_world_model_optimizer_steps"], 0)

    def test_nonfinite_aux_metrics_or_training_resource_drop_preserves_abort(self):
        def nan_aux(agent, *, horizon, weight):
            metrics = self.aux(agent, horizon=horizon, weight=weight)
            return {**metrics, "aux_loss": float("nan")}

        with patch.object(runner, "_load_aux_update", return_value=nan_aux):
            with self.assertRaisesRegex(ValueError, "aux update changed"):
                self.run_prior()
        self.assertEqual((self.root / self.output_name / "aux-update-metrics.jsonl").read_text(), "")
        self.output_name = f"{runner.RUN_ROOT}/teacher-seed-1"
        current = self.cgroup.copy()
        low = dict(current, available_bytes=1)
        with patch.object(runner.original, "_cgroup_memory", side_effect=[current, current, current, current, low]):
            with self.assertRaisesRegex(ValueError, "resource floor"):
                self.run_prior(seed=1)
        self.assertFalse((self.root / self.output_name / "training-result.json").exists())
        self.assertTrue((self.root / self.output_name / "abort.json").exists())


if __name__ == "__main__":
    unittest.main()
