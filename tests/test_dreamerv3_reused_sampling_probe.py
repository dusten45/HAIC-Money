"""Synthetic, simulator-free sampling, anchoring and source-freeze contracts."""

from __future__ import annotations

import json
import shutil
import unittest
from unittest import mock

import numpy as np
import torch

from haic.algorithms.drq_v2.teacher_replay import TeacherEpisode
from scripts.diagnose import dreamerv3_reused_sampling_probe as probe
from scripts.diagnose import dreamerv3_reused_train_budget_probe as budget
from scripts.diagnose import dreamerv3_reused_train_open_loop as scorer
from tests.test_dreamerv3_reused_action_probe import (
    ProbeFixture as ActionFixture, _pattern, _sensitive_modules,
)
from tests.test_dreamerv3_reused_train_open_loop import (
    _Continue, _Decoder, _Encoder, _Reward, _RSSM, _episode, _sha,
)


def _zero_modules():
    return _Encoder(), _RSSM(), _Decoder(), _Reward(), _Continue()


class _StochasticRSSM(_RSSM):
    def step_prior(self, h, z, action):
        return torch.randn_like(h) * 0.2, z, None, None


class _StochasticDecoder:
    def __call__(self, state):
        return state[:, :1].reshape(-1, 1, 1, 1).expand(-1, 1, 84, 84) * 0.3


def _stochastic_modules():
    return _Encoder(), _StochasticRSSM(), _StochasticDecoder(), _Reward(), _Continue()


class _PrefixEncoder:
    def __call__(self, stacks):
        return stacks[:, -1].mean(dim=(1, 2)).unsqueeze(-1)


class _PrefixRSSM(_RSSM):
    def observe_sequence(self, embeds, actions, first):
        states = torch.zeros((1, embeds.shape[1], 2))
        states[0, :, 0] = torch.cumsum(embeds[0, :, 0], dim=0)
        return states, None, None


class _PrefixDecoder:
    def __call__(self, state):
        return (state[:, :1] * 0.001).reshape(-1, 1, 1, 1).expand(-1, 1, 84, 84)


def _prefix_modules():
    return _PrefixEncoder(), _PrefixRSSM(), _PrefixDecoder(), _Reward(), _Continue()


class WindowTests(unittest.TestCase):
    def test_zero_residual_matches_frozen_anchor_at_every_step(self):
        episode = _pattern(_episode(length=41, road=3910800002, arm="random",
                                    study_id=scorer.DEVELOPMENT_ID, attempt=0))
        old = scorer._score_episode(episode, _zero_modules(), 0.25, 0.2, 42)
        result = probe._episode(episode, _zero_modules(), 42, old)
        self.assertEqual([w["context_anchor_decision"] for w in result["windows"]], [8, 9])
        self.assertEqual((result["window_count"], result["window_label_uses"],
                          result["unique_scored_decisions"], result["duplicate_label_uses"]), (2, 64, 33, 31))
        self.assertEqual([(result["horizons"][name]["label_uses"],
                           result["horizons"][name]["unique_scored_decisions"],
                           result["horizons"][name]["duplicate_label_uses"])
                          for name in ("t1", "t1_8", "t9_32")], [(2, 2, 0), (16, 9, 7), (48, 25, 23)])
        for window in result["windows"]:
            for index, row in enumerate(window["per_step"]):
                self.assertEqual((row["target_decision"], row["target_observation"]),
                                 (window["context_anchor_decision"] + index,
                                  window["context_anchor_decision"] + index + 1))
                anchor_image = episode.observation(window["context_anchor_decision"])[-1]
                target_image = episode.observation(row["target_observation"])[-1]
                self.assertAlmostEqual(row["shifted_repeat_mse"],
                                       float(np.mean((anchor_image - target_image) ** 2)))
                self.assertEqual(row["single_prior_mse"], row["shifted_repeat_mse"])
                self.assertEqual(row["predictive_mean_mse"], row["shifted_repeat_mse"])
            for horizon in window["horizons"].values():
                metrics = horizon["metrics"]
                self.assertEqual(metrics["single_prior_mse"], metrics["shifted_repeat_mse"])
                self.assertEqual(metrics["predictive_mean_mse"], metrics["shifted_repeat_mse"])
                self.assertEqual(metrics["sampled_path_mse"], [metrics["shifted_repeat_mse"]] * 8)
                self.assertEqual(metrics["sampled_mse_std"], 0)

    def test_mean_image_mse_is_not_the_mean_of_eight_mses(self):
        episode = _episode(length=41, road=3910800002, arm="teacher",
                           study_id=scorer.DEVELOPMENT_ID, attempt=0)
        reference = scorer._score_episode(episode, _stochastic_modules(), 0.25, 0.2, 19)
        result = probe._episode(episode, _stochastic_modules(), 19, reference)
        for window in result["windows"]:
            bucket = window["horizons"]["t1"]["metrics"]
            self.assertEqual(len(bucket["sampled_path_mse"]), 8)
            self.assertAlmostEqual(bucket["sampled_mse_mean"],
                                   sum(bucket["sampled_path_mse"]) / 8)
            self.assertGreater(bucket["sampled_mse_std"], 0)
            self.assertGreater(bucket["sampled_mse_mean"], bucket["predictive_mean_mse"])
        self.assertEqual(result, probe._episode(episode, _stochastic_modules(), 19, reference))
        with self.assertRaisesRegex(ValueError, "one sampled prior"):
            probe._episode(episode, _stochastic_modules(), 20, reference)

    def test_logged_action_to_next_observation_and_no_future_posterior(self):
        episode = _pattern(_episode(length=90, road=3910800002, arm="random",
                                    study_id=scorer.DEVELOPMENT_ID, attempt=0))
        reference = scorer._score_episode(episode, _sensitive_modules(), 0.25, 0.2, 123)
        modules = _sensitive_modules()
        encoder, rssm = modules[:2]
        encoded = []
        native_encode = encoder.__call__

        def encode(stacks):
            encoded.extend(float(stack[-1, 0, 0]) for stack in stacks)
            return native_encode(stacks)

        # A future-conditioned posterior is not permitted in the claimed prior.
        rssm.step_post = mock.Mock(side_effect=AssertionError("future posterior used"))
        class RecordingEncoder:
            def __call__(self, stacks):
                return encode(stacks)

        result = probe._episode(episode, (RecordingEncoder(), *modules[1:]), 123, reference)
        self.assertEqual(len(encoded), 59)
        self.assertAlmostEqual(max(encoded), 58 / 255)
        context, first = rssm.prefixes[0]
        np.testing.assert_array_equal(context.numpy()[0], episode.actions[:58])
        self.assertEqual((len(context[0]), first[0, 0].item(), int(first.sum())), (58, True, 1))
        self.assertEqual(len(rssm.draws), 512)
        for window_index, anchor in enumerate((8, 58)):
            expected = episode.actions[anchor:anchor + 32]
            for path in range(8):
                begin = window_index * 256 + path * 32
                np.testing.assert_array_equal(torch.cat(rssm.prior_actions[begin:begin + 32]).numpy(), expected)
            first_path = rssm.draws[window_index * 256:window_index * 256 + 32]
            self.assertTrue(all(draw != first_path[0] for draw in first_path[1:3]))
            self.assertEqual(result["windows"][window_index]["per_step"][0]["target_observation"], anchor + 1)

        # Alter a target strictly after the terminal anchor: no observed-prefix
        # embeddings/actions (or posterior state) can depend on that future frame.
        frames = episode.frames.copy()
        frames[59] = 240
        altered = TeacherEpisode(
            frames=frames, actions=episode.actions, applied_actions=episode.applied_actions,
            rewards=episode.rewards, progress=episode.progress, damage=episode.damage,
            retire_reasons=episode.retire_reasons, terminated=episode.terminated,
            truncated=episode.truncated, finished=episode.finished, terminal=episode.terminal,
            episode_id=episode.episode_id, source_id=episode.source_id,
            source_actor_sha256=episode.source_actor_sha256, geometry_id=episode.geometry_id,
            track_id=episode.track_id, metadata=episode.metadata)
        alt_ref = scorer._score_episode(altered, _sensitive_modules(), 0.25, 0.2, 123)
        alt_modules = _sensitive_modules()
        changed = probe._episode(altered, alt_modules, 123, alt_ref)
        self.assertEqual(torch.cat(rssm.prior_actions).tolist(),
                         torch.cat(alt_modules[1].prior_actions).tolist())
        self.assertEqual(rssm.draws, alt_modules[1].draws)
        self.assertNotEqual(result["windows"][1]["per_step"][0]["single_prior_mse"],
                            changed["windows"][1]["per_step"][0]["single_prior_mse"])

    def test_future_observation_does_not_reanchor_an_earlier_prior(self):
        episode = _episode(length=90, road=3910800002, arm="random",
                           study_id=scorer.DEVELOPMENT_ID, attempt=0)
        old = scorer._score_episode(episode, _prefix_modules(), 0.25, 0.2, 55)
        result = probe._episode(episode, _prefix_modules(), 55, old)
        frames = episode.frames.copy()
        frames[9] = 150  # After reset anchor 8, but in terminal anchor 58's observed prefix.
        altered = TeacherEpisode(
            frames=frames, actions=episode.actions, applied_actions=episode.applied_actions,
            rewards=episode.rewards, progress=episode.progress, damage=episode.damage,
            retire_reasons=episode.retire_reasons, terminated=episode.terminated,
            truncated=episode.truncated, finished=episode.finished, terminal=episode.terminal,
            episode_id=episode.episode_id, source_id=episode.source_id,
            source_actor_sha256=episode.source_actor_sha256, geometry_id=episode.geometry_id,
            track_id=episode.track_id, metadata=episode.metadata)
        changed_ref = scorer._score_episode(altered, _prefix_modules(), 0.25, 0.2, 55)
        changed = probe._episode(altered, _prefix_modules(), 55, changed_ref)
        # The reset forecast at t=10 and its unchanged target o10 are identical.
        self.assertEqual(result["windows"][0]["per_step"][1]["single_prior_mse"],
                         changed["windows"][0]["per_step"][1]["single_prior_mse"])
        # The terminal anchor legitimately includes the modified observed prefix.
        self.assertNotEqual(result["windows"][1]["per_step"][0]["single_prior_mse"],
                            changed["windows"][1]["per_step"][0]["single_prior_mse"])


class Fixture(ActionFixture):
    def setUp(self):
        super().setUp()
        source = self.root / probe.SOURCE
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(probe.__file__, source)
        self.sampling_sha = _sha(source)
        self.sampling_output = self.root / probe.OUTPUT
        self.sampling_protocol_patch = mock.patch.object(probe, "SCORE_PROTOCOL_SHA256", self.budget_protocol_sha)
        self.sampling_result_patch = mock.patch.object(probe, "SCORE_RESULT_SHA256",
                                                       _sha(self.root / probe.SCORE_RESULT))
        for patch in (self.sampling_protocol_patch, self.sampling_result_patch):
            patch.start()
            self.addCleanup(patch.stop)

    def preflight_sampling(self):
        return probe.preflight(self.sampling_output, self.sampling_sha, repo_root=self.root)


class ProvenanceTests(Fixture):
    def test_preflight_never_opens_zip_and_fixed_output(self):
        with mock.patch.object(scorer, "_load_dataset", side_effect=AssertionError("ZIP opened")), \
             mock.patch.object(scorer, "_dataset_bytes", side_effect=AssertionError("ZIP opened")):
            checked = self.preflight_sampling()
        self.assertEqual(checked["v2_result"]["status"], "iterative_tuning_descriptive_only")
        self.assertFalse(self.sampling_output.exists())
        with self.assertRaisesRegex(ValueError, "fixed new sampling-v1"):
            probe.preflight(self.root / budget.RUN_ROOT / "other", self.sampling_sha, repo_root=self.root)

    def test_source_sha_drift_score_receipt_and_collision_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            probe.preflight(self.sampling_output, "f" * 64, repo_root=self.root)
        with mock.patch.object(probe, "SCORE_RESULT_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.preflight_sampling()
        with mock.patch.object(probe, "SCORE_PROTOCOL_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.preflight_sampling()
        checked = self.preflight_sampling()
        source = self.root / probe.SOURCE
        source_bytes = source.read_bytes()
        source.write_bytes(source_bytes + b"# drift\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            probe._recheck(checked)
        self.assertFalse(self.sampling_output.exists())
        with self.assertRaisesRegex(ValueError, "executing sampling probe differs"):
            probe.preflight(self.sampling_output, _sha(source), repo_root=self.root)
        source.write_bytes(source_bytes)
        self.sampling_output.mkdir()
        with self.assertRaises(FileExistsError):
            self.preflight_sampling()

    def test_oversized_archive_checkpoint_and_cgroup_rejected_before_zip(self):
        with mock.patch.object(scorer, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            dev = self.root / self.development["random"]["archive_path"]
            archive_bytes = dev.read_bytes()
            dev.write_bytes(b"x" * (self.protocol["resources"]["max_archive_bytes"] + 1))
            with self.assertRaisesRegex(ValueError, "development must use a separate arm archive"):
                self.preflight_sampling()
            dev.write_bytes(archive_bytes)
        with mock.patch.object(scorer, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            checkpoint = self.root / self.budget_training["random"][0]["checkpoint_path"]
            checkpoint_bytes = checkpoint.read_bytes()
            checkpoint.write_bytes(b"x" * (self.protocol["resources"]["max_checkpoint_bytes"] + 1))
            with self.assertRaisesRegex(ValueError, "oversized input"):
                self.preflight_sampling()
            checkpoint.write_bytes(checkpoint_bytes)
        with mock.patch.object(scorer, "_cgroup", return_value={"limit_bytes": 128 * 1024**3,
                                                                 "available_bytes": 0, "used_bytes": 128 * 1024**3}):
            with self.assertRaisesRegex(ValueError, "cgroup"):
                self.preflight_sampling()

    def test_synthetic_report_has_eight_models_and_independent_denominators(self):
        with mock.patch.object(scorer, "_checkpoint_modules", side_effect=lambda *args, **kwargs: _sensitive_modules()), \
             mock.patch.object(budget, "_modules", side_effect=lambda *args, **kwargs: _sensitive_modules()):
            report = probe.score(self.sampling_output, self.sampling_sha, repo_root=self.root)
        self.assertEqual(report["status"], "iterative_tuning_descriptive_only")
        self.assertFalse(report["fresh_claim"])
        self.assertFalse(report["promotion_eligible"])
        self.assertEqual(report["sampling"]["prior_paths"], 8)
        for stratum in report["strata"].values():
            self.assertEqual(len(stratum["models"]), 8)
            self.assertEqual([model["updates"] for model in stratum["models"]], [64] * 4 + [256] * 4)
            for model in stratum["models"]:
                aggregate = model["aggregate"]
                self.assertEqual((aggregate["independent_episode_count"], aggregate["independent_road_count"],
                                  aggregate["window_count"], aggregate["window_label_uses"],
                                  aggregate["unique_scored_decisions"], aggregate["duplicate_label_uses"]),
                                 (4, 4, 8, 256, 132, 124))
                self.assertEqual(len(model["roads"]), 4)
                for road in model["roads"]:
                    self.assertEqual((road["independent_episode_count"], road["window_count"]), (1, 2))
                for episode in model["episodes"]:
                    self.assertEqual(len(episode["windows"]), 2)
                    self.assertEqual(episode["horizons"]["t1_8"]["label_uses"], 16)
                    self.assertEqual(episode["horizons"]["t9_32"]["label_uses"], 48)
        self.assertEqual(json.loads((self.sampling_output / "sampling-result.json").read_text()), report)
        with self.assertRaises(FileExistsError):
            self.preflight_sampling()


if __name__ == "__main__":
    unittest.main()
