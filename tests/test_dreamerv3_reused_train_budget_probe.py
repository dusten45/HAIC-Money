"""Simulator-free provenance and paired-window contracts for the budget probe."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock

import torch

from scripts.diagnose import dreamerv3_reused_train_budget_probe as probe
from scripts.diagnose import dreamerv3_reused_train_open_loop as scorer
from tests.test_dreamerv3_reused_train_open_loop import (
    ScorerFixture, _Continue, _Decoder, _Encoder, _Reward, _RSSM, _sha,
)


def _fake_modules():
    return _Encoder(), _RSSM(), _Decoder(), _Reward(), _Continue()


class BudgetProbeFixture(ScorerFixture):
    def setUp(self):
        super().setUp()
        self.budget_output = self.root / probe.RUN_ROOT / "scoring-v2"
        self.budget_protocol_path = self.root / "experiments/dreamerv3-reused-train-budget-probe-v1.json"
        self.probe_sha_patch = mock.patch.object(probe, "V2_OFFLINE_SHA256", "0" * 64)
        self.v1_sha_patch = mock.patch.object(probe, "ORIGINAL_SCORE_PROTOCOL_SHA256", "0" * 64)
        self.result_sha_patch = mock.patch.object(probe, "ORIGINAL_SCORE_RESULT_SHA256", "0" * 64)
        for patch in (self.probe_sha_patch, self.v1_sha_patch, self.result_sha_patch):
            patch.start()
            self.addCleanup(patch.stop)
        self._make_original_score()
        self._make_second_loop()

    def _make_original_score(self):
        self.protocol["scoring"]["latent_seed"] = 120926
        for arm in ("random", "teacher"):
            for seed, row in enumerate(self.training[arm]):
                checkpoint_path = self.root / row["checkpoint_path"]
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                payload.update(actor={"weight": torch.full((2,), float(seed + 1))},
                               critic={"weight": torch.full((2,), float(seed + 2))},
                               critic_target={"weight": torch.full((2,), float(seed + 3))},
                               actor_optimizer={"state": {}, "param_groups": []},
                               critic_optimizer={"state": {}, "param_groups": []})
                torch.save(payload, checkpoint_path)
                row["checkpoint_sha256"] = _sha(checkpoint_path)
                result = json.loads((self.root / row["result_path"]).read_text())
                result["checkpoint_sha256"] = row["checkpoint_sha256"]
                row["result_sha256"] = self._json(row["result_path"], result)
        self.freeze()
        original_output = self.root / probe.ORIGINAL_SCORE_RESULT_PATH
        with mock.patch.object(scorer, "_checkpoint_modules", side_effect=lambda *args, **kwargs: _fake_modules()):
            scorer.score(self.protocol_path, self.protocol_sha, original_output.parent, repo_root=self.root)
        probe.ORIGINAL_SCORE_PROTOCOL_SHA256 = self.protocol_sha
        probe.ORIGINAL_SCORE_RESULT_SHA256 = _sha(original_output)

    def _make_second_loop(self):
        self.v2_offline = copy.deepcopy(self.offline)
        self.v2_offline["learner"]["updates"] = 256
        self.v2_offline_sha = self._json(probe.V2_OFFLINE_PATH, self.v2_offline)
        probe.V2_OFFLINE_SHA256 = self.v2_offline_sha
        source_path = self.root / probe.PROBE_SOURCE
        source_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(probe.__file__, source_path)
        sources = {**self.sources, probe.PROBE_SOURCE: _sha(source_path)}
        self.budget_training = {}
        for arm in ("random", "teacher"):
            self.budget_training[arm] = []
            for seed, reference in enumerate(self.training[arm]):
                prefix = f"{probe.TRAIN_ROOT}/{arm}-seed-{seed}"
                original_payload = torch.load(self.root / reference["checkpoint_path"],
                                              map_location="cpu", weights_only=False)
                checkpoint = copy.deepcopy(original_payload)
                checkpoint["gradient_steps"] = 256
                checkpoint["run_metadata"]["offline_protocol_sha256"] = self.v2_offline_sha
                checkpoint_path = f"{prefix}/world-model-checkpoint.pt"
                checkpoint_file = self.root / checkpoint_path
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, checkpoint_file)
                v1_result = json.loads((self.root / reference["result_path"]).read_text())
                v2_result = copy.deepcopy(v1_result)
                v2_result.update(model_only_updates=256, offline_protocol_sha256=self.v2_offline_sha,
                                 checkpoint_sha256=_sha(checkpoint_file))
                result_path = f"{prefix}/training-result.json"
                result_sha = self._json(result_path, v2_result)
                self.budget_training[arm].append({"seed": seed, "result_path": result_path,
                                                  "result_sha256": result_sha, "checkpoint_path": checkpoint_path,
                                                  "checkpoint_sha256": _sha(checkpoint_file)})
        self.budget_protocol = {
            "format": probe.FORMAT, "purpose": probe.PURPOSE, "study_id": self.dev_id,
            "offline_protocol": {"path": probe.V2_OFFLINE_PATH, "sha256": self.v2_offline_sha},
            "original_score_protocol": {"path": probe.ORIGINAL_SCORE_PROTOCOL_PATH,
                                        "sha256": self.protocol_sha},
            "original_score_result": {"path": probe.ORIGINAL_SCORE_RESULT_PATH,
                                      "sha256": probe.ORIGINAL_SCORE_RESULT_SHA256},
            "training": self.budget_training,
            "development_collection_protocol": copy.deepcopy(self.protocol["development_collection_protocol"]),
            "development": copy.deepcopy(self.development),
            "source_sha256": sources, "scoring": copy.deepcopy(self.protocol["scoring"]),
            "resources": copy.deepcopy(self.protocol["resources"]),
        }
        self.freeze_probe()

    def freeze_probe(self):
        self.budget_protocol_sha = self._json("experiments/dreamerv3-reused-train-budget-probe-v1.json",
                                               self.budget_protocol)

    def preflight_probe(self):
        return probe.preflight(self.budget_protocol_path, self.budget_protocol_sha,
                               self.budget_output, repo_root=self.root)

    def refresh_result(self, arm="random", seed=0):
        row = self.budget_training[arm][seed]
        result = json.loads((self.root / row["result_path"]).read_text())
        result["checkpoint_sha256"] = row["checkpoint_sha256"]
        row["result_sha256"] = self._json(row["result_path"], result)
        self.freeze_probe()
        return row, result


class ProvenanceTests(BudgetProbeFixture):
    def test_pins_exact_first_and_second_loop_and_budget_only_change(self):
        checked = self.preflight_probe()
        self.assertEqual(checked["offline"]["learner"]["updates"], 256)
        self.assertEqual(checked["checked_v1"]["offline"]["learner"]["updates"], 64)
        self.assertFalse(self.budget_output.exists())
        self.v2_offline["learner"]["config"]["batch_size"] = 5
        self.budget_protocol["offline_protocol"]["sha256"] = self._json(probe.V2_OFFLINE_PATH,
                                                                          self.v2_offline)
        probe.V2_OFFLINE_SHA256 = self.budget_protocol["offline_protocol"]["sha256"]
        self.freeze_probe()
        with self.assertRaisesRegex(ValueError, "beyond learner.updates"):
            self.preflight_probe()

    def test_old_vs_v2_executable_source_pins_and_original_score_sha(self):
        self.budget_protocol["source_sha256"]["dreamer_v3.py"] = "c" * 64
        self.freeze_probe()
        with self.assertRaisesRegex(ValueError, "source pins"):
            self.preflight_probe()
        self.budget_protocol["source_sha256"]["dreamer_v3.py"] = self.sources["dreamer_v3.py"]
        self.budget_protocol["original_score_result"]["sha256"] = "e" * 64
        self.freeze_probe()
        with self.assertRaisesRegex(ValueError, "frozen reference"):
            self.preflight_probe()
        self.budget_protocol["original_score_result"]["sha256"] = probe.ORIGINAL_SCORE_RESULT_SHA256
        self.freeze_probe()
        (self.root / probe.ORIGINAL_SCORE_RESULT_PATH).write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.preflight_probe()

    def test_second_loop_cannot_change_training_executable_hash_or_seed(self):
        self.v2_offline["source_sha256"]["dreamer_v3.py"] = "d" * 64
        self.budget_protocol["offline_protocol"]["sha256"] = self._json(probe.V2_OFFLINE_PATH,
                                                                          self.v2_offline)
        probe.V2_OFFLINE_SHA256 = self.budget_protocol["offline_protocol"]["sha256"]
        self.freeze_probe()
        with self.assertRaisesRegex(ValueError, "beyond learner.updates"):
            self.preflight_probe()
        self.v2_offline["source_sha256"]["dreamer_v3.py"] = self.offline["source_sha256"]["dreamer_v3.py"]
        self.v2_offline["learner"]["seeds"] = [0, 2]
        self.budget_protocol["offline_protocol"]["sha256"] = self._json(probe.V2_OFFLINE_PATH,
                                                                          self.v2_offline)
        probe.V2_OFFLINE_SHA256 = self.budget_protocol["offline_protocol"]["sha256"]
        self.freeze_probe()
        with self.assertRaisesRegex(ValueError, "beyond learner.updates"):
            self.preflight_probe()

    def test_result_budget_mismatch_and_forged_actor_are_rejected(self):
        row, result = self.refresh_result()
        result["model_only_updates"] = 64
        row["result_sha256"] = self._json(row["result_path"], result)
        self.freeze_probe()
        with self.assertRaisesRegex(ValueError, "model_only_updates"):
            self.preflight_probe()
        result["model_only_updates"] = 256
        result["actor_trained"] = True
        row["result_sha256"] = self._json(row["result_path"], result)
        self.freeze_probe()
        with self.assertRaisesRegex(ValueError, "actor_trained"):
            self.preflight_probe()

    def test_checkpoint_gradient_actor_weights_and_optimizer_updates_are_rejected(self):
        row, result = self.refresh_result()
        checkpoint_file = self.root / row["checkpoint_path"]
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        payload["gradient_steps"] = 64
        torch.save(payload, checkpoint_file)
        row["checkpoint_sha256"] = _sha(checkpoint_file)
        self.refresh_result()
        with self.assertRaisesRegex(ValueError, "checkpoint metadata"):
            self.preflight_probe()
        payload["gradient_steps"] = 256
        payload["actor"]["weight"] += 1
        torch.save(payload, checkpoint_file)
        row["checkpoint_sha256"] = _sha(checkpoint_file)
        self.refresh_result()
        with self.assertRaisesRegex(ValueError, "actor differs"):
            self.preflight_probe()
        payload["actor"]["weight"] -= 1
        payload["actor_optimizer"]["state"] = {1: {"step": 1}}
        torch.save(payload, checkpoint_file)
        row["checkpoint_sha256"] = _sha(checkpoint_file)
        self.refresh_result()
        with self.assertRaisesRegex(ValueError, "optimizer has update state"):
            self.preflight_probe()

    def test_duplicate_output_and_changed_development_archive_are_rejected(self):
        self.budget_output.mkdir()
        with self.assertRaises(FileExistsError):
            self.preflight_probe()
        self.budget_output.rmdir()
        self.budget_protocol["development"]["random"]["archive_sha256"] = "f" * 64
        self.freeze_probe()
        with self.assertRaisesRegex(ValueError, "development archives"):
            self.preflight_probe()


class PairedScoreTests(BudgetProbeFixture):
    def test_exact_reset_terminal_windows_and_zero_paired_deltas(self):
        with (mock.patch.object(probe, "_modules", side_effect=lambda *args, **kwargs: _fake_modules()),
              mock.patch.object(scorer, "_score_episode", wraps=scorer._score_episode) as scored):
            report = probe.score(self.budget_protocol_path, self.budget_protocol_sha,
                                 self.budget_output, repo_root=self.root)
        self.assertEqual([call.args[-1] for call in scored.call_args_list],
                         [120926 + (source_arm == "teacher") * 1_000_003 + index * 1009 + seed
                          for source_arm in ("random", "teacher")
                          for model_arm in ("random", "teacher")
                          for seed in (0, 1) for index in range(4)])
        self.assertEqual(report["status"], "iterative_tuning_descriptive_only")
        self.assertFalse(report["promotion_eligible"])
        self.assertFalse(report["fresh_claim"])
        self.assertEqual(report["sampling"]["latent_seed"], 120926)
        for stratum in report["strata"].values():
            for model in stratum["models"]:
                aggregate = model["aggregate"]
                self.assertEqual((aggregate["episode_count"], aggregate["window_count"],
                                  aggregate["window_label_uses"], aggregate["unique_scored_decisions"],
                                  aggregate["duplicate_label_uses"]), (4, 8, 256, 132, 124))
                self.assertTrue(all(value == 0 for value in aggregate["paired_delta_vs_64"].values()))
                self.assertEqual(len(model["roads"]), 4)
                for episode in model["episodes"]:
                    self.assertEqual([(window["kind"], window["context_anchor_decision"],
                                       window["first_target_decision"], window["last_target_decision"])
                                      for window in episode["windows"]],
                                     [("reset", 8, 8, 39), ("terminal", 9, 9, 40)])
                    self.assertEqual((episode["unique_scored_decisions"], episode["duplicate_label_uses"],
                                      episode["unique_terminal_positive_labels"]), (33, 31, 1))
                    for window in episode["windows"]:
                        self.assertTrue(all(value == 0 for value in window["paired_delta_vs_64"].values()))
        self.assertEqual(json.loads((self.budget_output / "score-result.json").read_text()), report)
        with self.assertRaises(FileExistsError):
            self.preflight_probe()

    def test_pairing_fails_closed_on_mutated_window_and_baseline(self):
        old = json.loads((self.root / probe.ORIGINAL_SCORE_RESULT_PATH).read_text())
        model = old["strata"]["random"]["models"][0]
        current = copy.deepcopy(model)
        current["episodes"][0]["windows"][1]["context_anchor_decision"] = 8
        with self.assertRaisesRegex(ValueError, "window differs"):
            probe._pair(current, model)
        current = copy.deepcopy(model)
        current["aggregate"]["metrics_episode_mean"]["shifted_repeat_mse"] += 0.1
        with self.assertRaisesRegex(ValueError, "baseline differs"):
            probe._pair(current, model)


if __name__ == "__main__":
    unittest.main()
