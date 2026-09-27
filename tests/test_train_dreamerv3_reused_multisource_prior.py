"""Synthetic mixed-source prior-interaction tests; no real archives or driving."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import Transition
from dreamer_v3 import DreamerV3Agent, DreamerV3Config, Uint8SequenceReplay
from haic.algorithms.dreamer_v3.prior_image import prior_image_aux_update
from scripts import train_dreamerv3_reused_multisource_prior as runner
from tests import test_train_dreamerv3_prior_image as prior_tests
from tests import test_train_dreamerv3_reused_multisource as mixed_tests


class _Agent(prior_tests._Agent):
    def __init__(self, config, seed):
        super().__init__(config, seed)
        for name in ("encoder", "rssm", "decoder", "reward_head", "continue_head"):
            setattr(self, name, torch.nn.Linear(1, 1))


class PriorMixedFixture(unittest.TestCase):
    # Reuse the existing fake source0/source1 TRAIN archive and receipt builder.
    put_bytes = mixed_tests.TestMultiSourceModelOnly.put_bytes
    put = mixed_tests.TestMultiSourceModelOnly.put
    file_sha = mixed_tests.TestMultiSourceModelOnly.file_sha
    ref = mixed_tests.TestMultiSourceModelOnly.ref
    reference = mixed_tests.TestMultiSourceModelOnly.reference
    actor = mixed_tests.TestMultiSourceModelOnly.actor
    dataset_bytes = mixed_tests.TestMultiSourceModelOnly.dataset_bytes
    freeze = mixed_tests.TestMultiSourceModelOnly.freeze

    def setUp(self):
        mixed_tests.TestMultiSourceModelOnly.setUp(self)
        self.base_output = self.output
        self.output = "runs/synthetic-multisource/prior-interaction-v1/learner-0"
        (self.root / "runs/synthetic-multisource/prior-interaction-v1").mkdir()
        for name, value in (("RUN_ROOT", "runs/synthetic-multisource/prior-interaction-v1"),
                            ("BASE_UPDATES", 8), ("AUX_UPDATES", 2),
                            ("EMBED_DIM", 64), ("HIDDEN_DIM", 32)):
            patched = patch.object(runner, name, value)
            patched.start()
            self.addCleanup(patched.stop)
        patched_updates = patch.object(runner.mixed, "UPDATES", 8)
        patched_updates.start()
        self.addCleanup(patched_updates.stop)
        self.ref_config = json.loads((self.root / self.refs["base_model_protocol"][0]).read_text())
        self.ref_config["learner"]["config"].update(seq_len=32, burnin_steps=8)
        self.ref("base_model_protocol", self.ref_config)
        self.protocol["base_model_protocol"] = self.reference("base_model_protocol")
        self.protocol["learner"]["config"].update(seq_len=32, burnin_steps=8)
        self.protocol["learner"]["updates"] = 8
        self.mixed_sha = self.freeze()
        sha_patch = patch.object(runner, "MIXED_PROTOCOL_SHA256", self.mixed_sha)
        sha_patch.start()
        self.addCleanup(sha_patch.stop)
        checked = runner.mixed.preflight(runner.MIXED_PROTOCOL_PATH, self.mixed_sha, 0,
                                         self.base_output, repo_root=self.root)
        _, lineage, audit = runner.mixed.materialize(checked)
        lineage_bytes = (json.dumps({"format": runner.mixed.FORMAT + "-lineage",
                                     "range_convention": "inclusive", "episodes": lineage,
                                     "transition_count": 18}, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n").encode()
        self.lineage_sha = hashlib.sha256(lineage_bytes).hexdigest()
        lineage_patch = patch.object(runner, "LINEAGE_SHA256", self.lineage_sha)
        lineage_patch.start()
        self.addCleanup(lineage_patch.stop)
        self.baseline_results = []
        for seed in (0, 1):
            prefix = f"runs/synthetic-multisource/learner-{seed}"
            self.put_bytes(f"{prefix}/lineage.json", lineage_bytes)
            self.put(f"{prefix}/training-result.json", {
                "format": runner.mixed.FORMAT + "-result", "status": "complete", "seed": seed,
                "protocol_sha256": self.mixed_sha, "sources": self.protocol["datasets"],
                "source_audits": audit, "lineage_sha256": self.lineage_sha,
                "lineage_path": f"{prefix}/lineage.json", "model_only_updates": 8,
                "environment_steps": 0, "actor_trained": False, "fresh_claim": False,
                "promotion_eligible": False, "actor_critic_target_optimizers_unchanged": True,
                "decisions": 18, "episodes": 6,
            })
            self.baseline_results.append({
                "seed": seed, "result_path": f"{prefix}/training-result.json",
                "result_sha256": self.file_sha(f"{prefix}/training-result.json"),
                "lineage_path": f"{prefix}/lineage.json", "lineage_sha256": self.lineage_sha,
            })
        baseline_patch = patch.object(runner, "BASELINE_RESULTS", self.baseline_results)
        baseline_patch.start()
        self.addCleanup(baseline_patch.stop)
        self.source_hashes = dict(self.protocol["source_sha256"])
        for name, original in ((runner.RUNNER_SOURCE, runner.__file__),
                               (runner.prior.RUNNER_SOURCE, runner.prior.__file__),
                               (runner.prior.HELPER_SOURCE, runner.prior.ROOT / runner.prior.HELPER_SOURCE)):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, dest)
            self.source_hashes[name] = self.file_sha(name)
        self.prior_protocol = {
            "format": runner.FORMAT, "purpose": runner.PURPOSE, "study_id": runner.STUDY_ID,
            "mixed_protocol": {"path": runner.MIXED_PROTOCOL_PATH, "sha256": self.mixed_sha},
            **{key: self.reference(key) for key in runner.mixed.REFS},
            "source_sha256": self.source_hashes, "cells": self.protocol["cells"],
            "datasets": self.protocol["datasets"], "expected_lineage_sha256": self.lineage_sha,
            "baseline_results": self.baseline_results,
            "output_root": runner.RUN_ROOT,
            "learner": {
                "seeds": [0, 1], "config": self.protocol["learner"]["config"],
                "base_updates": 8, "aux_updates": 2, "aux_every_base_updates": 4,
                "prior_horizon": 8, "prior_weight": 0.25,
                "anchor_mode": runner.prior.ANCHOR_MODE, "aux_rng_mode": runner.prior.AUX_RNG_MODE,
                "aux_rng_seed_base": runner.prior.AUX_RNG_SEED_BASE,
                "aux_rng_seed_stride": runner.prior.AUX_RNG_SEED_STRIDE,
            },
            "resources": {**self.protocol["resources"],
                          "min_cgroup_available_bytes": 12 * 1024**3},
        }
        self.events = []
        _Agent.calls = []
        for target, name, value in ((runner, "DreamerV3Agent", _Agent),
                                    (runner.prior, "_load_aux_update", lambda _root, _sha: self.aux),
                                    (runner, "_oom_kills", lambda: 1)):
            patched = patch.object(target, name, value)
            patched.start()
            self.addCleanup(patched.stop)

    def aux(self, agent, *, horizon, weight):
        self.events.append((agent.seed, agent.gradient_steps, horizon, weight))
        agent.wm_step()
        return {"aux_loss": 0.25, "frame_mse": 1.0, "anchor_count": 2.0,
                "target_count": 4.0, "effective_horizon": 3.0}

    def freeze_prior(self):
        self.put(runner.PROTOCOL_PATH, self.prior_protocol)
        return self.file_sha(runner.PROTOCOL_PATH)

    def preflight_prior(self, *, seed=0):
        return runner.preflight(runner.PROTOCOL_PATH, self.freeze_prior(), seed, self.output,
                                repo_root=self.root)

    def run_prior(self, *, seed=0):
        return runner.train(runner.PROTOCOL_PATH, self.freeze_prior(), seed, self.output,
                            repo_root=self.root)


class PriorMixedTests(PriorMixedFixture):
    def test_synthetic_replay_preserves_distinct_source_actors_and_rejects_cross_episode_windows(self):
        checked = self.preflight_prior()
        replay, lineage, audit = runner.mixed.materialize(checked)
        self.assertEqual([row["source_id"] for row in lineage],
                         ["drq-source-0", "drq-source-1"] * 3)
        self.assertEqual([row["road"]["geometry_seed"] for row in lineage],
                         [111, 111, 222, 222, 333, 333])
        self.assertEqual([(row["sequence_id_start"], row["sequence_id_end"])
                          for row in lineage], [(i * 3, i * 3 + 2) for i in range(6)])
        self.assertEqual({row["actor_sha256"] for row in lineage},
                         {source["source_actor_sha256"] for source in self.sources.values()})
        self.assertEqual((audit["source0"]["decisions"], audit["source1"]["decisions"]), (9, 9))
        for i in range(6):
            self.assertTrue(replay.is_first[i * 3])
            self.assertTrue(replay.is_last[i * 3 + 2])
        with patch.object(Uint8SequenceReplay, "sample_sequence",
                          return_value={"sequence_ids": torch.tensor([[2, 3]])}):
            with self.assertRaisesRegex(ValueError, "crosses source/episode"):
                replay.sample_sequence(1, 2)

    def test_exact_schedule_unchanged_source_pair_lineage_and_checkpoint(self):
        old_root = runner.mixed.OUTPUT_ROOT
        old_sha = runner.mixed.single._sha256(Path(runner.mixed.__file__))
        for seed in (0, 1):
            self.output = f"{runner.RUN_ROOT}/learner-{seed}"
            result = self.run_prior(seed=seed)
            self.assertEqual(result["environment_steps"], 0)
            self.assertEqual((result["base_model_only_updates"], result["aux_world_model_optimizer_steps"],
                              result["world_model_optimizer_steps"]), (8, 2, 10))
            self.assertEqual(result["sources"], self.protocol["datasets"])
            self.assertEqual(result["lineage_sha256"], self.lineage_sha)
            self.assertEqual(result["decisions"], 18)
            self.assertEqual(result["episodes"], 6)
            self.assertEqual(result["distinct_training_roads"], 3)
            self.assertFalse(result["actor_trained"])
            self.assertFalse(result["fresh_claim"])
            self.assertFalse(result["matched_pure_256_step_model"])
            output = self.root / self.output
            self.assertEqual(self.file_sha(result["lineage_path"]), self.lineage_sha)
            for name, digest in (("base_metrics_path", "base_metrics_sha256"),
                                 ("aux_metrics_path", "aux_metrics_sha256"),
                                 ("checkpoint_path", "checkpoint_sha256")):
                self.assertEqual(self.file_sha(f"{self.output}/{result[name]}"), result[digest])
            checkpoint = torch.load(output / result["checkpoint_path"], weights_only=False)
            metadata = checkpoint["run_metadata"]
            self.assertEqual(metadata["mixed_protocol"]["sha256"], self.mixed_sha)
            self.assertEqual(metadata["sources"], self.protocol["datasets"])
            self.assertEqual(metadata["pure_256_baselines"], self.baseline_results)
            self.assertEqual(metadata["source_sha256"], self.source_hashes)
            self.assertEqual(metadata["lineage_sha256"], self.lineage_sha)
            self.assertEqual(checkpoint["gradient_steps"], 8)
            self.assertFalse(checkpoint["actor_optimizer"]["state"])
            self.assertFalse(checkpoint["critic_optimizer"]["state"])
            base_rows = [json.loads(line) for line in (output / result["base_metrics_path"]).read_text().splitlines()]
            aux_rows = [json.loads(line) for line in (output / result["aux_metrics_path"]).read_text().splitlines()]
            self.assertEqual([row["base_update"] for row in base_rows], list(range(1, 9)))
            self.assertEqual([row["after_base_update"] for row in aux_rows], [4, 8])
            self.assertEqual(runner.mixed.OUTPUT_ROOT, old_root)
        self.assertEqual(_Agent.calls, [True] * 16)
        self.assertEqual(self.events, [(seed, i, 8, 0.25) for seed in (0, 1) for i in (4, 8)])
        self.assertEqual(runner.mixed.single._sha256(Path(runner.mixed.__file__)), old_sha)

    def test_preflight_strict_source_pair_resources_protocol_schedule_and_no_output(self):
        cases = (
            ("source actor", lambda p: p["datasets"]["source1"].update(source_id="drq-source-0")),
            ("archive", lambda p: p["datasets"]["source0"].update(archive_sha256="0" * 64)),
            ("catalog", lambda p: p["catalog"].update(sha256="0" * 64)),
            ("old runner", lambda p: p["source_sha256"].update({
                "scripts/train_dreamerv3_reused_multisource.py": "0" * 64})),
            ("helper", lambda p: p["source_sha256"].update({runner.prior.HELPER_SOURCE: "0" * 64})),
            ("prior runner", lambda p: p["source_sha256"].update({runner.prior.RUNNER_SOURCE: "0" * 64})),
            ("new wrapper", lambda p: p["source_sha256"].update({runner.RUNNER_SOURCE: "0" * 64})),
            ("seed list", lambda p: p["learner"].update(seeds=[0])),
            ("seed bool", lambda p: p["learner"].update(seeds=[False, 1])),
            ("model config", lambda p: p["learner"]["config"].update(seq_len=16)),
            ("aux budget", lambda p: p["learner"].update(aux_updates=3)),
            ("float budget", lambda p: p["learner"].update(base_updates=8.0)),
            ("cgroup floor", lambda p: p["resources"].update(min_cgroup_available_bytes=8 * 1024**3)),
            ("float floor", lambda p: p["resources"].update(min_cgroup_available_bytes=float(12 * 1024**3))),
            ("lineage", lambda p: p.update(expected_lineage_sha256="0" * 64)),
            ("old result", lambda p: p["baseline_results"][1].update(result_sha256="0" * 64)),
        )
        for title, mutate in cases:
            with self.subTest(title=title):
                saved = copy.deepcopy(self.prior_protocol)
                mutate(self.prior_protocol)
                with patch.object(runner.mixed, "materialize", side_effect=AssertionError("archive opened")):
                    with self.assertRaises(ValueError):
                        self.run_prior()
                self.assertFalse((self.root / self.output).exists())
                self.prior_protocol = saved

    def test_scoped_output_root_restored_on_success_failure_and_single_thread_guard(self):
        old_root = runner.mixed.OUTPUT_ROOT
        self.preflight_prior()
        self.assertEqual(runner.mixed.OUTPUT_ROOT, old_root)
        with patch.object(runner.mixed, "preflight", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                self.preflight_prior()
        self.assertEqual(runner.mixed.OUTPUT_ROOT, old_root)
        self.assertFalse((self.root / self.output).exists())
        with patch.object(runner.threading, "active_count", return_value=2):
            with self.assertRaisesRegex(ValueError, "single-threaded"):
                self.preflight_prior()
        self.assertEqual(runner.mixed.OUTPUT_ROOT, old_root)

    def test_changed_helper_or_protocol_or_catalog_rejected_before_output(self):
        digest = self.freeze_prior()
        helper = self.root / runner.prior.HELPER_SOURCE
        helper.write_text("# changed executable\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            runner.train(runner.PROTOCOL_PATH, digest, 0, self.output, repo_root=self.root)
        self.assertFalse((self.root / self.output).exists())
        shutil.copyfile(runner.prior.ROOT / runner.prior.HELPER_SOURCE, helper)
        self.prior_protocol["learner"]["prior_weight"] = 0.5
        self.freeze_prior()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            runner.train(runner.PROTOCOL_PATH, digest, 0, self.output, repo_root=self.root)
        self.assertFalse((self.root / self.output).exists())
        self.prior_protocol["learner"]["prior_weight"] = 0.25
        catalog = self.root / self.refs["catalog"][0]
        catalog.write_text("{}", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.run_prior()
        self.assertFalse((self.root / self.output).exists())

    def test_either_frozen_pure_256_result_or_lineage_drift_rejected_before_output(self):
        for row in self.baseline_results:
            for name in (row["result_path"], row["lineage_path"]):
                with self.subTest(seed=row["seed"], path=name):
                    path = self.root / name
                    original_bytes = path.read_bytes()
                    path.write_bytes(original_bytes + b" ")
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        self.run_prior()
                    self.assertFalse((self.root / self.output).exists())
                    path.write_bytes(original_bytes)

    def test_rng_fork_restores_base_torch_numpy_python_streams(self):
        seen_base, seen_aux, first_state = [], [], []
        original_update = _Agent.update

        def traced(agent, *, model_only=False):
            if not first_state:
                first_state.append(torch.get_rng_state().clone())
            seen_base.append((random.random(), np.random.random(), torch.rand(1).item()))
            return original_update(agent, model_only=model_only)

        def traced_aux(agent, *, horizon, weight):
            seen_aux.append((random.random(), np.random.random(), torch.rand(1).item()))
            return self.aux(agent, horizon=horizon, weight=weight)

        state_py, state_np, state_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(456)
            np.random.seed(456)
            torch.manual_seed(456)
            with (patch.object(_Agent, "update", traced),
                  patch.object(runner.prior, "_load_aux_update", return_value=traced_aux)):
                self.run_prior()
            random.seed(456)
            np.random.seed(456)
            torch.set_rng_state(first_state[0])
            self.assertEqual(seen_base, [(random.random(), np.random.random(), torch.rand(1).item())
                                         for _ in range(8)])
            for i, draw in enumerate(seen_aux, 1):
                seed = runner.prior.AUX_RNG_SEED_BASE + i
                with torch.random.fork_rng(devices=[], enabled=True):
                    torch.manual_seed(seed)
                    expected_torch = torch.rand(1).item()
                self.assertEqual(draw, (random.Random(seed).random(),
                                        np.random.RandomState(seed).random_sample(), expected_torch))
        finally:
            random.setstate(state_py)
            np.random.set_state(state_np)
            torch.set_rng_state(state_torch)

    def test_archive_drift_lineage_drift_and_output_collision_abort_without_update(self):
        archive = self.root / self.sources["source1"]["archive_path"]
        original_bytes = archive.read_bytes()
        with archive.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "archive size|archive SHA|archive changed"):
            self.run_prior()
        abort = json.loads((self.root / self.output / "abort.json").read_text())
        self.assertEqual(abort["completed_base_model_only_updates"], 0)
        self.assertEqual(abort["phase"], "source and archive verification")
        self.assertFalse((self.root / self.output / "training-result.json").exists())
        archive.write_bytes(original_bytes)
        self.output = f"{runner.RUN_ROOT}/learner-1"
        original_materialize = runner.mixed.materialize

        def changed_lineage(checked):
            replay, lineage, audit = original_materialize(checked)
            lineage[0]["original_episode_id"] = "wrong"
            return replay, lineage, audit

        with patch.object(runner.mixed, "materialize", side_effect=changed_lineage):
            with self.assertRaisesRegex(ValueError, "episode/sequence lineage differs"):
                self.run_prior(seed=1)
        self.assertEqual(_Agent.calls, [])
        with patch.object(runner.mixed.single, "_archive_bytes", side_effect=AssertionError("archive read")):
            with self.assertRaises(FileExistsError):
                self.run_prior(seed=1)

    def test_aux_targets_optimizer_steps_nonfinite_actor_and_cgroup_abort(self):
        def no_step(agent, *, horizon, weight):
            return {"aux_loss": 0.25, "frame_mse": 1.0, "anchor_count": 2.0,
                    "target_count": 4.0, "effective_horizon": 3.0}

        with patch.object(runner.prior, "_load_aux_update", return_value=no_step):
            with self.assertRaisesRegex(ValueError, "aux update"):
                self.run_prior()
        abort = json.loads((self.root / self.output / "abort.json").read_text())
        self.assertEqual((abort["completed_base_model_only_updates"],
                          abort["completed_aux_world_model_optimizer_steps"],
                          abort["observed_world_model_optimizer_steps"]), (4, 0, 4))
        self.assertFalse((self.root / self.output / "training-result.json").exists())
        self.output = f"{runner.RUN_ROOT}/learner-1"

        def bad_aux(agent, *, horizon, weight):
            metrics = self.aux(agent, horizon=horizon, weight=weight)
            return {**metrics, "target_count": 3.0, "aux_loss": float("nan")}

        with patch.object(runner.prior, "_load_aux_update", return_value=bad_aux):
            with self.assertRaisesRegex(ValueError, "aux update"):
                self.run_prior(seed=1)
        self.assertFalse((self.root / self.output / "training-result.json").exists())
        self.assertEqual((self.root / self.output / "aux-update-metrics.jsonl").read_text(), "")

    def test_actor_drift_and_memory_or_oom_counter_stop_without_success(self):
        update = _Agent.update

        def bad_actor(agent, *, model_only=False):
            result = update(agent, model_only=model_only)
            with torch.no_grad():
                agent.actor.weight.add_(1)
            return result

        with patch.object(_Agent, "update", bad_actor):
            with self.assertRaisesRegex(ValueError, "actor/critic"):
                self.run_prior()
        self.assertFalse((self.root / self.output / "training-result.json").exists())
        self.output = f"{runner.RUN_ROOT}/learner-1"
        low = {**self.cgroup, "available_bytes": 11 * 1024**3}
        with patch.object(runner.mixed.single, "_cgroup_memory", return_value=low):
            with self.assertRaisesRegex(ValueError, "12 GiB"):
                self.run_prior(seed=1)
        self.assertFalse((self.root / self.output).exists())
        with patch.object(runner, "_oom_kills", side_effect=[1, 1, 2]):
            with self.assertRaisesRegex(ValueError, "oom_kill"):
                self.run_prior(seed=1)
        abort = json.loads((self.root / self.output / "abort.json").read_text())
        self.assertEqual(abort["completed_base_model_only_updates"], 0)

    def test_mid_update_cgroup_drop_and_aux_actor_drift_abort(self):
        update = _Agent.update

        def memory_drop(agent, *, model_only=False):
            result = update(agent, model_only=model_only)
            self.cgroup["available_bytes"] = 11 * 1024**3
            return result

        with patch.object(_Agent, "update", memory_drop):
            with self.assertRaisesRegex(ValueError, "12 GiB"):
                self.run_prior()
        abort = json.loads((self.root / self.output / "abort.json").read_text())
        self.assertEqual(abort["completed_base_model_only_updates"], 0)
        self.assertFalse((self.root / self.output / "training-result.json").exists())
        self.cgroup["available_bytes"] = 28 * 1024**3
        self.output = f"{runner.RUN_ROOT}/learner-1"

        def actor_drift(agent, *, horizon, weight):
            result = self.aux(agent, horizon=horizon, weight=weight)
            with torch.no_grad():
                agent.critic_target.weight.add_(0.01)
            return result

        with patch.object(runner.prior, "_load_aux_update", return_value=actor_drift):
            with self.assertRaisesRegex(ValueError, "aux update"):
                self.run_prior(seed=1)
        abort = json.loads((self.root / self.output / "abort.json").read_text())
        self.assertEqual(abort["completed_base_model_only_updates"], 4)
        self.assertEqual(abort["completed_aux_world_model_optimizer_steps"], 0)
        self.assertFalse((self.root / self.output / "training-result.json").exists())

    def test_real_small_rssm_h8_last_full_anchor_and_terminal_successor(self):
        replay = Uint8SequenceReplay(capacity=48)
        for step in range(40):
            frame = np.full((4, 84, 84), step + 10, np.uint8)
            replay.add(Transition(
                observation=frame, next_observation=np.full((4, 84, 84), step + 11, np.uint8),
                action=np.array([step / 100, 0, 0], np.float32), reward=0.0,
                terminated=False, truncated=step == 39, terminal=step == 39,
                episode_id=1, step=step,
            ))
        config = DreamerV3Config(device="cpu", embed_dim=32, hidden_dim=32,
                                 num_categoricals=4, num_classes=4, batch_size=1,
                                 burnin_steps=8, seq_len=32, replay_capacity=48,
                                 reset_start_fraction=1.0, terminal_window_fraction=0.0,
                                 short_episode_fraction=0.0)
        torch.set_num_threads(1)
        agent = DreamerV3Agent(config, seed=0)
        agent.replay = replay
        snapshot = runner.mixed._snapshot(agent)
        metrics = prior_image_aux_update(agent, horizon=8, weight=0.25)
        self.assertEqual((metrics["anchor_count"], metrics["target_count"],
                          metrics["effective_horizon"]), (2.0, 16.0, 8.0))
        self.assertTrue(runner.prior._aux_targets(metrics, 1))
        self.assertEqual(agent.gradient_steps, 0)
        self.assertEqual(agent.environment_steps, 0)
        self.assertTrue(runner.mixed._unchanged(agent, snapshot))
        self.assertEqual(replay._boundary_observations[39][-1, 0, 0], 50)


if __name__ == "__main__":
    unittest.main()
