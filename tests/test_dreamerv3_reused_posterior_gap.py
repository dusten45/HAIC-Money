"""Simulator-free posterior/forecast alignment and frozen-provenance contracts."""

from __future__ import annotations

import copy
import json
import shutil
import unittest
from unittest import mock

import numpy as np
import torch

from haic.algorithms.drq_v2.teacher_replay import TeacherEpisode
from scripts.diagnose import dreamerv3_reused_posterior_gap as gap
from scripts.diagnose import dreamerv3_reused_train_budget_probe as budget
from scripts.diagnose import dreamerv3_reused_train_open_loop as scorer
from tests import test_dreamerv3_reused_train_budget_probe as budget_tests
from tests.test_dreamerv3_reused_train_open_loop import _Continue, _Reward, _episode, _sha


class _Encoder:
    def __init__(self):
        self.reads = []

    def __call__(self, stack):
        self.reads.extend(float(value) for value in stack[:, -1, 0, 0])
        latest = stack[:, -1].mean(dim=(-2, -1))
        return torch.stack((latest, latest), dim=-1)


class _RSSM:
    hidden_dim = stoch_dim = 1

    def __init__(self):
        self.prefixes = []
        self.prior_calls = []
        self.post_calls = []

    def observe_sequence(self, embeds, actions, first):
        self.prefixes.append((embeds.shape[1], actions.clone(), first.clone()))
        h = z = torch.zeros((1, 1))
        previous = torch.zeros((1, 3))
        states = []
        for t in range(embeds.shape[1]):
            h, z, _, _ = self.step_post(h, z, previous, embeds[:, t])
            states.append(torch.cat((h, z), dim=-1))
            if t < actions.shape[1]:
                previous = actions[:, t]
        return torch.stack(states, dim=1), None, None

    def step_prior(self, h, z, action):
        self.prior_calls.append((h.clone(), z.clone(), action.clone(), len(self.post_calls)))
        next_h = h + 0.1 * action[:, :1]
        return next_h, torch.zeros_like(z), torch.zeros((1, 1, 2)), None

    def step_post(self, h, z, action, embed):
        self.post_calls.append((h.clone(), z.clone(), action.clone(), embed.clone()))
        next_h = h + 0.1 * action[:, :1]
        next_z = embed[:, :1]
        logits = torch.stack((torch.zeros_like(next_z), next_z * 8.0), dim=-1)
        return next_h, next_z, logits, None


class _Decoder:
    def __call__(self, state):
        return (state[:, 1:2] * 0.1).view(1, 1, 1, 1).expand(1, 1, 84, 84)


def _modules():
    return _Encoder(), _RSSM(), _Decoder(), _Reward(), _Continue()


class _StochasticRSSM(_RSSM):
    def step_prior(self, h, z, action):
        next_h, _, logits, _ = super().step_prior(h, z, action)
        return next_h, torch.rand_like(z) * 0.01, logits, None


def _pattern(episode):
    actions = episode.actions.copy()
    actions[:, 0] = (np.arange(episode.steps) % 5 + 1).astype(np.float32) / 10
    applied = episode.applied_actions.copy()
    applied[:, 0] = actions[:, 0]
    return TeacherEpisode(
        frames=episode.frames, actions=actions, applied_actions=applied,
        rewards=episode.rewards, progress=episode.progress, damage=episode.damage,
        retire_reasons=episode.retire_reasons, terminated=episode.terminated,
        truncated=episode.truncated, finished=episode.finished, terminal=episode.terminal,
        episode_id=episode.episode_id, source_id=episode.source_id,
        source_actor_sha256=episode.source_actor_sha256, geometry_id=episode.geometry_id,
        track_id=episode.track_id, metadata=episode.metadata,
    )


def _changed_frame(episode, index):
    frames = episode.frames.copy()
    frames[index] = 250
    return TeacherEpisode(
        frames=frames, actions=episode.actions, applied_actions=episode.applied_actions,
        rewards=episode.rewards, progress=episode.progress, damage=episode.damage,
        retire_reasons=episode.retire_reasons, terminated=episode.terminated,
        truncated=episode.truncated, finished=episode.finished, terminal=episode.terminal,
        episode_id=episode.episode_id, source_id=episode.source_id,
        source_actor_sha256=episode.source_actor_sha256, geometry_id=episode.geometry_id,
        track_id=episode.track_id, metadata=episode.metadata,
    )


class AlignmentTests(unittest.TestCase):
    def episode(self, length=90):
        return _pattern(_episode(length=length, road=3910800002, arm="random",
                                 study_id=scorer.DEVELOPMENT_ID, attempt=0))

    def score(self, episode, modules):
        reference = scorer._score_episode(episode, _modules(), 0.25, 0.2, 123)
        return gap._episode(episode, modules, 123, reference)

    def test_preaction_state_native_action_successor_and_full_prefix_parity(self):
        ep = self.episode()
        modules = _modules()
        result = self.score(ep, modules)
        encoder, rssm = modules[:2]
        self.assertEqual(len(rssm.prefixes), 1)
        size, prefix, first = rssm.prefixes[0]
        self.assertEqual(size, 59)  # terminal anchor is 90 - 32 = 58
        np.testing.assert_array_equal(prefix[0].numpy(), ep.actions[:58])
        self.assertEqual((int(first[0, 0]), int(first.sum())), (1, 1))
        self.assertEqual(len(rssm.prior_calls), 128)  # 32 free + 32 one-step at each anchor
        expected_h = 0.1 * float(ep.actions[:8, 0].sum())
        for index in (0, 32):
            h, z, action, _ = rssm.prior_calls[index]
            self.assertAlmostEqual(float(h.item()), expected_h, places=6)
            self.assertAlmostEqual(float(z.item()), 8 / 255, places=6)
            np.testing.assert_array_equal(action.numpy()[0], ep.actions[8])
        for offset in range(32):
            h, z, action, _ = rssm.prior_calls[32 + offset]
            self.assertAlmostEqual(float(h.item()),
                                   0.1 * float(ep.actions[:8 + offset, 0].sum()), places=5)
            self.assertAlmostEqual(float(z.item()), (8 + offset) / 255, places=6)
            np.testing.assert_array_equal(action.numpy()[0], ep.actions[8 + offset])
            self.assertAlmostEqual(float(rssm.post_calls[59 + offset][3][0, 0]),
                                   (9 + offset) / 255, places=6)
        np.testing.assert_allclose(encoder.reads[:59], [i / 255 for i in range(59)], rtol=1e-6)
        # The only free-running target accesses are image-error comparisons, not encoder input.
        np.testing.assert_allclose(encoder.reads[59:91], [i / 255 for i in range(9, 41)], rtol=1e-6)
        for window in result["windows"]:
            self.assertEqual(window["posterior_target_conditioned"]["target_frames"], 32)
            self.assertEqual(window["one_step_prior_teacher_forced_context"]["target_frames"], 32)
            self.assertEqual(window["same_transition_raw_logits_kl"]["transition_uses"], 32)
            self.assertGreater(window["same_transition_raw_logits_kl"]["mean_nats"], 0)
            self.assertNotEqual(window["posterior_target_conditioned"]["image_mse"],
                                window["one_step_prior_teacher_forced_context"]["image_mse"])
            for horizon in ("1", "8", "32"):
                self.assertEqual(window["free_running_prior"][horizon]["prefix"]["target_frames"], int(horizon))
            self.assertAlmostEqual(window["free_running_prior"]["32"]["prefix"]["image_mse"],
                                   scorer._score_episode(ep, _modules(), 0.25, 0.2, 123)["windows"][
                                       0 if window["kind"] == "reset" else 1]["metrics"]["image_mse"])
        self.assertEqual((result["unique_scored_decisions"], result["duplicate_label_uses"]), (64, 0))

    def test_stochastic_prior_rng_keeps_both_frozen_windows_aligned(self):
        ep = self.episode()

        def modules():
            return _Encoder(), _StochasticRSSM(), _Decoder(), _Reward(), _Continue()

        reference = scorer._score_episode(ep, modules(), 0.25, 0.2, 123)
        result = gap._episode(ep, modules(), 123, reference)
        for scored, pinned in zip(result["windows"], reference["windows"], strict=True):
            self.assertAlmostEqual(scored["free_running_prior"]["32"]["prefix"]["image_mse"],
                                   pinned["metrics"]["image_mse"], places=7)
            self.assertAlmostEqual(scored["free_running_prior"]["32"]["prefix"]["baseline_mse"],
                                   pinned["metrics"]["shifted_repeat_mse"], places=7)

    def test_future_frame_affects_posterior_but_not_anchored_prior_inputs(self):
        episode = self.episode()
        changed = _changed_frame(episode, 59)  # first result frame of terminal anchor 58
        before_modules, after_modules = _modules(), _modules()
        before = self.score(episode, before_modules)
        after = self.score(changed, after_modules)
        # Reset/terminal free-running state sees only observations <= its anchor.
        for index in (0, 64):
            for a, b in zip(before_modules[1].prior_calls[index][:3],
                            after_modules[1].prior_calls[index][:3], strict=True):
                torch.testing.assert_close(a, b)
        # At the terminal anchor the posterior updates using o[59], after the prior forecast.
        self.assertNotEqual(before["windows"][1]["posterior_target_conditioned"]["image_mse"],
                            after["windows"][1]["posterior_target_conditioned"]["image_mse"])
        self.assertNotEqual(float(before_modules[1].post_calls[-32][3][0, 0]),
                            float(after_modules[1].post_calls[-32][3][0, 0]))
        self.assertEqual(float(before_modules[1].prior_calls[64][1].item()),
                         float(after_modules[1].prior_calls[64][1].item()))

    def test_same_transition_logits_and_target_conditioned_not_forecast(self):
        self.assertAlmostEqual(gap._logits_kl(torch.tensor([[[0., 0.]]]),
                                               torch.tensor([[[0., 2.]]])), 0.3278133255, places=7)
        with self.assertRaisesRegex(ValueError, "same-transition"):
            gap._logits_kl(torch.zeros(1, 1, 2), torch.zeros(1, 2, 2))
        ep = self.episode(41)
        reference = scorer._score_episode(ep, _modules(), 0.25, 0.2, 123)
        wrong = copy.deepcopy(reference)
        wrong["windows"][0]["label_uses"] = 31
        with self.assertRaisesRegex(ValueError, "window differs"):
            gap._episode(ep, _modules(), 123, wrong)
        wrong = copy.deepcopy(reference)
        wrong["windows"][0]["metrics"]["image_mse"] += 0.1
        with self.assertRaisesRegex(ValueError, "pinned original score"):
            gap._episode(ep, _modules(), 123, wrong)
        self.assertEqual((reference["unique_scored_decisions"], reference["duplicate_label_uses"]), (33, 31))

    def test_clamp_counts_and_unclamped_signed_residual(self):
        previous = torch.full((1, 1, 84, 84), 0.05)
        target = torch.full_like(previous, 0.25)
        row = gap._image_row(previous, torch.full_like(previous, -0.1), target, previous)
        self.assertEqual((row["clipped_below_zero_pixels"], row["clipped_above_one_pixels"],
                          row["clipped_frames"]), (84 * 84, 0, 1))
        self.assertAlmostEqual(row["signed_residual_mean"], -0.25)
        self.assertAlmostEqual(row["absolute_residual_mean"], 0.25)
        high = gap._image_row(previous + 0.94, torch.full_like(previous, 0.1), target, previous)
        self.assertEqual(high["clipped_above_one_pixels"], 84 * 84)
        self.assertEqual(gap._image_summary([row, high])["target_frames"], 2)


class Fixture(budget_tests.BudgetProbeFixture):
    def _archive(self, relative, *, arm, study_id, roads, length):
        if study_id != self.dev_id:
            return super()._archive(relative, arm=arm, study_id=study_id, roads=roads, length=length)
        from haic.algorithms.drq_v2.teacher_replay import TeacherDataset
        source_id = "uniform-native-random-seed-7392" if arm == "random" else "teacher-source"
        actor_hash = self.dev_random_hash if arm == "random" else "b" * 64
        dataset = TeacherDataset([_pattern(_episode(length=length, road=road, arm=arm,
                                                     study_id=study_id, attempt=i,
                                                     source_id=source_id, actor_hash=actor_hash))
                                  for i, road in enumerate(roads)])
        dataset.seal()
        return dataset.digest, self._write(relative, dataset.to_bytes())

    def _make_original_score(self):
        with mock.patch.object(budget_tests, "_fake_modules", side_effect=_modules):
            super()._make_original_score()

    def setUp(self):
        super().setUp()
        for source in (gap.SOURCE, gap.ACTION_SOURCE):
            path = self.root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(gap.__file__ if source == gap.SOURCE
                            else (gap.ROOT / source), path)
        self.source_sha = _sha(self.root / gap.SOURCE)
        self.gap_output = self.root / gap.OUTPUT
        with mock.patch.object(budget, "_modules", side_effect=lambda *a, **kw: _modules()):
            budget.score(self.budget_protocol_path, self.budget_protocol_sha,
                         self.budget_output, repo_root=self.root)
        self._patch(gap, "BUDGET_PROTOCOL_SHA256", self.budget_protocol_sha)
        self._patch(gap, "BUDGET_RESULT_SHA256", _sha(self.root / gap.BUDGET_RESULT))
        self.gap_protocol = {
            "format": gap.FORMAT, "purpose": gap.PURPOSE, "study_id": self.dev_id,
            "budget_score_protocol": {"path": "experiments/dreamerv3-reused-train-budget-probe-v1.json",
                                      "sha256": gap.BUDGET_PROTOCOL_SHA256},
            "budget_score_result": {"path": gap.BUDGET_RESULT, "sha256": gap.BUDGET_RESULT_SHA256},
            "source_sha256": {**self.budget_protocol["source_sha256"],
                              gap.ACTION_SOURCE: gap.ACTION_SOURCE_SHA256, gap.SOURCE: self.source_sha},
            "sampling": copy.deepcopy(gap.SAMPLING), "resources": copy.deepcopy(self.protocol["resources"]),
        }
        self.gap_path = self.root / gap.PROTOCOL
        self.freeze_gap()

    def _patch(self, target, name, value):
        patch = mock.patch.object(target, name, value)
        patch.start()
        self.addCleanup(patch.stop)

    def freeze_gap(self):
        self.gap_sha = self._json(gap.PROTOCOL, self.gap_protocol)

    def preflight_gap(self):
        return gap.preflight(self.gap_path, self.gap_sha, self.source_sha,
                             self.gap_output, repo_root=self.root)


class ProvenanceTests(Fixture):
    def test_real_rssm_modules_on_synthetic_episode_only(self):
        row = self.training["random"][0]
        path = self.root / row["checkpoint_path"]
        def modules():
            return scorer._checkpoint_modules(
                path, arm="random", seed=0,
                dataset_digest=self.offline["datasets"]["random"]["dataset_digest"],
                config=self.config)

        episode = _pattern(_episode(length=41, road=self.development_roads[0], arm="random",
                                    study_id=self.dev_id, attempt=0,
                                    source_id="uniform-native-random-seed-7392",
                                    actor_hash=self.dev_random_hash))
        reference = scorer._score_episode(episode, modules(), 0.25, 0.2, 123)
        result = gap._episode(episode, modules(), 123, reference)
        self.assertEqual(result["summary"]["same_transition_raw_logits_kl"]["transition_uses"], 64)
        self.assertEqual(result["summary"]["posterior_target_conditioned"]["target_frames"], 64)

    def test_preflight_is_zip_free_exact_schema_source_and_collision(self):
        with mock.patch.object(scorer, "_load_dataset", side_effect=AssertionError("archive read")):
            self.assertEqual(self.preflight_gap()["gap_protocol_sha256"], self.gap_sha)
        self.assertFalse(self.gap_output.exists())
        with self.assertRaisesRegex(ValueError, "fixed posterior-gap"):
            gap.preflight(self.gap_path, self.gap_sha, self.source_sha,
                          self.root / budget.RUN_ROOT / "arbitrary", repo_root=self.root)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            gap.preflight(self.gap_path, "f" * 64, self.source_sha, self.gap_output, repo_root=self.root)
        self.gap_output.mkdir()
        with self.assertRaises(FileExistsError):
            self.preflight_gap()

    def test_new_source_pin_checked_before_checkpoint_load(self):
        self.gap_protocol["source_sha256"][gap.SOURCE] = "f" * 64
        self.freeze_gap()
        with mock.patch.object(budget, "preflight", side_effect=AssertionError("checkpoint read")):
            with self.assertRaisesRegex(ValueError, "SHA map"):
                self.preflight_gap()

    def test_exact_count_source_drift_archive_size_and_cgroup_guards(self):
        self.gap_protocol["source_sha256"][gap.SOURCE] = "f" * 64
        self.freeze_gap()
        with self.assertRaisesRegex(ValueError, "source|SHA map"):
            self.preflight_gap()
        self.gap_protocol["source_sha256"][gap.SOURCE] = self.source_sha
        self.gap_protocol["sampling"]["horizons"] = [1, 32]
        self.freeze_gap()
        with self.assertRaisesRegex(ValueError, "schema/sampling"):
            self.preflight_gap()
        self.gap_protocol["sampling"] = copy.deepcopy(gap.SAMPLING)
        self.freeze_gap()
        path = self.root / gap.BUDGET_RESULT
        result = json.loads(path.read_text())
        result["strata"]["random"]["scored_episode_count"] = 3
        self._patch(gap, "BUDGET_RESULT_SHA256", self._json(gap.BUDGET_RESULT, result))
        self.gap_protocol["budget_score_result"]["sha256"] = gap.BUDGET_RESULT_SHA256
        self.freeze_gap()
        with self.assertRaisesRegex(ValueError, "four independent scored|score identity"):
            self.preflight_gap()

    def test_oversized_archive_and_cgroup_guard_before_archive_decode(self):
        archive = self.root / self.development["random"]["archive_path"]
        archive.write_bytes(b"x" * (self.protocol["resources"]["max_archive_bytes"] + 1))
        with mock.patch.object(scorer, "_load_dataset", side_effect=AssertionError("ZIP read")):
            with self.assertRaisesRegex(ValueError, "archive"):
                self.preflight_gap()
        with mock.patch.object(scorer, "_cgroup", return_value={
            "limit_bytes": 2**30, "available_bytes": 0, "used_bytes": 2**30,
        }), mock.patch.object(scorer, "_load_dataset", side_effect=AssertionError("ZIP read")):
            with self.assertRaisesRegex(ValueError, "cgroup"):
                self.preflight_gap()

    def test_post_compute_source_drift_cannot_create_output(self):
        actual = gap._episode
        touched = False

        def drift(*args, **kwargs):
            nonlocal touched
            row = actual(*args, **kwargs)
            if not touched:
                touched = True
                with (self.root / gap.SOURCE).open("ab") as stream:
                    stream.write(b"# drift\n")
            return row

        with mock.patch.object(gap, "_episode", side_effect=drift), \
             mock.patch.object(scorer, "_checkpoint_modules", side_effect=lambda *a, **kw: _modules()), \
             mock.patch.object(budget, "_modules", side_effect=lambda *a, **kw: _modules()):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                gap.score(self.gap_path, self.gap_sha, self.source_sha,
                          self.gap_output, repo_root=self.root)
        self.assertFalse(self.gap_output.exists())

    def test_read_only_full_synthetic_report_eight_models_and_denominators(self):
        with mock.patch.object(scorer, "_checkpoint_modules", side_effect=lambda *a, **kw: _modules()), \
             mock.patch.object(budget, "_modules", side_effect=lambda *a, **kw: _modules()):
            report = gap.score(self.gap_path, self.gap_sha, self.source_sha,
                               self.gap_output, repo_root=self.root)
        self.assertEqual(report["status"], "iterative_tuning_descriptive_only")
        self.assertFalse(report["fresh_claim"])
        self.assertFalse(report["promotion_eligible"])
        for stratum in report["strata"].values():
            self.assertEqual(len(stratum["models"]), 8)
            for model in stratum["models"]:
                self.assertEqual(len(model["roads"]), 4)
                agg = model["aggregate"]
                self.assertEqual((agg["independent_episode_count"], agg["independent_road_count"],
                                  agg["window_count"], agg["window_label_uses"],
                                  agg["unique_scored_decisions"], agg["duplicate_label_uses"],
                                  agg["unique_terminal_positive_labels"]), (4, 4, 8, 256, 132, 124, 4))
                self.assertEqual(agg["same_transition_raw_logits_kl"]["transition_uses"], 256)
                self.assertEqual(agg["posterior_target_conditioned"]["pixel_uses"], 256 * 84 * 84)
                self.assertEqual(agg["free_running_prior"]["1"]["endpoint"]["target_frames"], 8)
                for ep in model["episodes"]:
                    self.assertEqual(len(ep["windows"]), 2)
        self.assertEqual(json.loads((self.gap_output / "posterior-gap-result.json").read_text()), report)


if __name__ == "__main__":
    unittest.main()
