"""Synthetic, simulator-free action intervention and frozen-source contracts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
from unittest import mock

import numpy as np
import torch

from common_adapter import ActionAdapter, ActionSpec
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset, TeacherEpisode
from scripts.diagnose import dreamerv3_reused_action_probe as probe
from scripts.diagnose import dreamerv3_reused_train_budget_probe as budget
from scripts.diagnose import dreamerv3_reused_train_open_loop as scorer
from tests import test_dreamerv3_reused_train_budget_probe as budget_tests
from tests.test_dreamerv3_reused_train_open_loop import (
    _Continue, _Decoder, _Encoder, _Reward, _RSSM, _episode, _sha,
)


class _ActionRSSM:
    hidden_dim = stoch_dim = 1

    def __init__(self):
        self.prefixes = []
        self.draws = []
        self.prior_actions = []

    def observe_sequence(self, embeds, actions, first):
        self.prefixes.append((actions.clone(), first.clone()))
        states = torch.zeros((1, embeds.shape[1], 2))
        states[0, 1:, 0] = torch.cumsum(actions[0, :, 0] * 0.1, dim=0)
        return states, None, None

    def step_prior(self, h, z, action):
        noise = torch.rand_like(h) * 0.001
        self.draws.append(float(noise.item()))
        self.prior_actions.append(action.clone())
        return h + action[:, :1] * 0.1 + noise, z, None, None


class _ActionDecoder:
    def __call__(self, state):
        return state[:, :1].reshape(-1, 1, 1, 1).expand(-1, 1, 84, 84) * 0.02


class _ActionReward:
    def pred(self, state):
        return state[:, 0]


class _ActionContinue:
    def __call__(self, state):
        return state[:, :1]


def _sensitive_modules():
    return _Encoder(), _ActionRSSM(), _ActionDecoder(), _ActionReward(), _ActionContinue()


def _insensitive_modules():
    return _Encoder(), _RSSM(), _Decoder(), _Reward(), _Continue()


def _pattern(episode):
    t = np.arange(episode.steps, dtype=np.float32)
    actions = np.stack((np.where(t.astype(int) % 3 == 0, 0.9, -0.4),
                        np.where(t.astype(int) % 5 == 0, -0.7, 0.3),
                        np.where(t.astype(int) % 7 == 0, 0.6, -0.2)), axis=1).astype(np.float32)
    return _with_actions(episode, actions)


def _with_actions(episode, actions):
    adapter = ActionAdapter(ActionSpec())
    applied = np.stack([adapter.to_official(action, clip=False) for action in actions])
    return TeacherEpisode(
        frames=episode.frames, actions=actions, applied_actions=applied,
        rewards=episode.rewards, progress=episode.progress, damage=episode.damage,
        retire_reasons=episode.retire_reasons, terminated=episode.terminated,
        truncated=episode.truncated, finished=episode.finished, terminal=episode.terminal,
        episode_id=episode.episode_id, source_id=episode.source_id,
        source_actor_sha256=episode.source_actor_sha256, geometry_id=episode.geometry_id,
        track_id=episode.track_id, metadata=episode.metadata,
    )


class WindowTests(unittest.TestCase):
    def test_future_only_rotation_preserves_context_and_stochastic_draws(self):
        episode = _pattern(_episode(length=90, road=3910800002, arm="random",
                                    study_id=scorer.DEVELOPMENT_ID, attempt=0))
        modules = _sensitive_modules()
        original_score = scorer._score_episode(episode, modules, 0.25, 0.2, 123)
        modules = _sensitive_modules()
        result = probe._episode(episode, modules, 123, original_score)
        self.assertEqual(len(modules[1].prefixes), 1)
        context, first = modules[1].prefixes[0]
        np.testing.assert_array_equal(context.numpy()[0], episode.actions[:58])
        self.assertTrue(first[0, 0])
        self.assertEqual(int(first.sum()), 1)
        self.assertEqual(len(modules[1].draws), 192)
        for offset in (0, 96):
            anchor = 8 if offset == 0 else 58
            expected = episode.actions[anchor:anchor + 32]
            for condition_offset, variant in ((0, expected), (32, np.roll(expected, 11, axis=0)),
                                              (64, np.zeros_like(expected))):
                observed = torch.cat(modules[1].prior_actions[
                    offset + condition_offset:offset + condition_offset + 32]).numpy()
                np.testing.assert_array_equal(observed, variant)
            self.assertEqual(modules[1].draws[offset:offset + 32],
                             modules[1].draws[offset + 32:offset + 64])
            self.assertEqual(modules[1].draws[offset:offset + 32],
                             modules[1].draws[offset + 64:offset + 96])
        for window in result["windows"]:
            self.assertGreater(window["conditions"]["shifted"]["changed_native_action_uses"], 0)
            self.assertEqual(window["conditions"]["zero"]["changed_native_action_uses"], 32)
            self.assertGreater(window["conditions"]["shifted"]["prediction_diff_vs_logged"]["reward_prediction_abs"], 0)
            self.assertGreater(window["conditions"]["zero"]["prediction_diff_vs_logged"]["terminal_probability_abs"], 0)
            self.assertTrue(all(value == 0 for value in
                                window["conditions"]["logged"]["prediction_diff_vs_logged"].values()))
        self.assertEqual((result["unique_scored_decisions"], result["duplicate_label_uses"]), (64, 0))
        self.assertFalse(result["action_insensitive_small_response"])
        before = result["windows"][0]["conditions"]["logged"]["errors"]["reward_mse"]
        changed_actions = episode.actions.copy()
        changed_actions[1, 0] += 0.1
        changed_episode = _with_actions(episode, changed_actions)
        changed_reference = scorer._score_episode(changed_episode, _sensitive_modules(), 0.25, 0.2, 123)
        changed = probe._episode(changed_episode, _sensitive_modules(), 123, changed_reference)
        self.assertNotEqual(before, changed["windows"][0]["conditions"]["logged"]["errors"]["reward_mse"])

    def test_action_insensitive_rssm_and_nontrivial_actions(self):
        episode = _pattern(_episode(length=41, road=3910800002, arm="teacher",
                                    study_id=scorer.DEVELOPMENT_ID, attempt=0))
        reference = scorer._score_episode(episode, _insensitive_modules(), 0.25, 0.2, 543)
        result = probe._episode(episode, _insensitive_modules(), 543, reference)
        self.assertTrue(result["action_insensitive_small_response"])
        self.assertEqual([window["context_anchor_decision"] for window in result["windows"]], [8, 9])
        self.assertEqual((result["unique_scored_decisions"], result["duplicate_label_uses"],
                          result["unique_terminal_positive_labels"]), (33, 31, 1))
        for variant in ("shifted", "zero"):
            self.assertGreater(result["conditions"][variant]["changed_native_action_uses"], 0)
            for section in ("delta_error_vs_logged", "prediction_diff_vs_logged"):
                self.assertTrue(all(value == 0 for value in result["conditions"][variant][section].values()))

    def test_identical_inputs_cannot_be_called_insensitive(self):
        episode = _episode(length=41, road=3910800002, arm="random",
                           study_id=scorer.DEVELOPMENT_ID, attempt=0)
        reference = scorer._score_episode(episode, _insensitive_modules(), 0.25, 0.2, 1)
        result = probe._episode(episode, _insensitive_modules(), 1, reference)
        self.assertIsNone(result["action_insensitive_small_response"])
        self.assertEqual(result["conditions"]["zero"]["changed_native_action_uses"], 0)


class ProbeFixture(budget_tests.BudgetProbeFixture):
    def _archive(self, relative: str, *, arm: str, study_id: str, roads: tuple[int, ...], length: int):
        if study_id != self.dev_id:
            return super()._archive(relative, arm=arm, study_id=study_id, roads=roads, length=length)
        source_id = "uniform-native-random-seed-7392" if arm == "random" else "teacher-source"
        actor_hash = self.dev_random_hash if arm == "random" else "b" * 64
        episodes = [_pattern(_episode(length=length, road=road, arm=arm, study_id=study_id,
                                      attempt=index, source_id=source_id, actor_hash=actor_hash))
                    for index, road in enumerate(roads)]
        dataset = TeacherDataset(episodes)
        dataset.seal()
        return dataset.digest, self._write(relative, dataset.to_bytes())

    def _make_original_score(self):
        with mock.patch.object(budget_tests, "_fake_modules", side_effect=_sensitive_modules):
            super()._make_original_score()

    def setUp(self):
        super().setUp()
        source = self.root / probe.SOURCE
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(probe.__file__, source)
        self.source_sha = _sha(source)
        self.action_output = self.root / probe.OUTPUT
        with mock.patch.object(budget, "_modules", side_effect=lambda *args, **kwargs: _sensitive_modules()):
            budget.score(self.budget_protocol_path, self.budget_protocol_sha, self.budget_output, repo_root=self.root)
        self.probe_protocol_patch = mock.patch.object(probe, "V2_SCORE_PROTOCOL_SHA256", self.budget_protocol_sha)
        self.probe_result_patch = mock.patch.object(probe, "V2_SCORE_RESULT_SHA256",
                                                    _sha(self.root / probe.V2_SCORE_RESULT))
        for patch in (self.probe_protocol_patch, self.probe_result_patch):
            patch.start()
            self.addCleanup(patch.stop)

    def preflight_action(self):
        return probe.preflight(self.action_output, self.source_sha, repo_root=self.root)


class ProvenanceAndReportTests(ProbeFixture):
    def test_exact_pins_no_archive_read_in_preflight_and_hard_output(self):
        with mock.patch.object(scorer, "_load_dataset", side_effect=AssertionError("archive opened")):
            checked = self.preflight_action()
        self.assertEqual(checked["v2_result"]["status"], "iterative_tuning_descriptive_only")
        self.assertFalse(self.action_output.exists())
        with self.assertRaisesRegex(ValueError, "fixed action-sensitivity sibling"):
            probe.preflight(self.root / budget.RUN_ROOT / "other-output", self.source_sha, repo_root=self.root)

    def test_forged_source_protocol_result_and_output_collision(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            probe.preflight(self.action_output, "f" * 64, repo_root=self.root)
        with mock.patch.object(probe, "V2_SCORE_PROTOCOL_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.preflight_action()
        with mock.patch.object(probe, "V2_SCORE_RESULT_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.preflight_action()
        self.action_output.mkdir()
        with self.assertRaises(FileExistsError):
            self.preflight_action()

    def test_oversized_archive_and_score_result_rejected_before_archive_load(self):
        dev_file = self.root / self.development["random"]["archive_path"]
        data = dev_file.read_bytes()
        cap = self.protocol["resources"]["max_archive_bytes"]
        dev_file.write_bytes(b"x" * (cap + 1))
        with mock.patch.object(scorer, "_load_dataset", side_effect=AssertionError("archive opened")):
            with self.assertRaisesRegex(ValueError, "development must use a separate arm archive"):
                self.preflight_action()
        dev_file.write_bytes(data)
        v2_file = self.root / probe.V2_SCORE_RESULT
        v2_file.write_bytes(b"x" * (1024 * 1024 + 1))
        with mock.patch.object(scorer, "_load_dataset", side_effect=AssertionError("archive opened")):
            with self.assertRaisesRegex(ValueError, "oversized input"):
                self.preflight_action()

    def test_full_synthetic_report_labels_roads_models_and_no_promotion(self):
        with mock.patch.object(scorer, "_checkpoint_modules", side_effect=lambda *args, **kwargs: _sensitive_modules()), \
             mock.patch.object(budget, "_modules", side_effect=lambda *args, **kwargs: _sensitive_modules()):
            report = probe.score(self.action_output, self.source_sha, repo_root=self.root)
        self.assertEqual(report["format"], probe.FORMAT)
        self.assertEqual(report["status"], "iterative_tuning_descriptive_only")
        self.assertFalse(report["fresh_claim"])
        self.assertFalse(report["promotion_eligible"])
        self.assertEqual(report["sampling"]["shift_roll"], 11)
        self.assertEqual(set(report["strata"]), {"random", "teacher"})
        for stratum in report["strata"].values():
            self.assertEqual(len(stratum["models"]), 8)
            self.assertEqual([row["updates"] for row in stratum["models"]], [64] * 4 + [256] * 4)
            self.assertEqual(stratum["descriptive_model_mean"]["model_count"], 8)
            self.assertEqual(stratum["descriptive_model_mean"]["independent_episode_count"], 4)
            self.assertEqual(len(stratum["roads_descriptive_model_mean"]), 4)
            for road in stratum["roads_descriptive_model_mean"]:
                self.assertEqual(road["independent_episode_count"], 1)
                self.assertEqual(road["model_count"], 8)
            for model in stratum["models"]:
                aggregate = model["aggregate"]
                self.assertEqual((aggregate["independent_episode_count"], aggregate["independent_road_count"],
                                  aggregate["window_count"], aggregate["window_label_uses"],
                                  aggregate["unique_scored_decisions"], aggregate["duplicate_label_uses"],
                                  aggregate["unique_terminal_positive_labels"]), (4, 4, 8, 256, 132, 124, 4))
                self.assertGreater(aggregate["conditions"]["shifted"]["prediction_diff_vs_logged"]["reward_prediction_abs"], 0)
                for episode in model["episodes"]:
                    self.assertEqual(len(episode["windows"]), 2)
                    self.assertEqual(episode["conditions"]["logged"]["changed_native_action_uses"], 0)
        self.assertEqual(json.loads((self.action_output / "action-result.json").read_text()), report)
        with self.assertRaises(FileExistsError):
            self.preflight_action()


if __name__ == "__main__":
    unittest.main()
