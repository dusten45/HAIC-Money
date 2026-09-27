"""Synthetic checkpoint/ZIP provenance and paired prior-image scoring contracts."""

from __future__ import annotations

import copy
import json
import shutil
import unittest
from unittest import mock

import torch

from scripts.diagnose import dreamerv3_reused_prior_image_score as score
from scripts.diagnose import dreamerv3_reused_train_budget_probe as budget
from scripts.diagnose import dreamerv3_reused_train_open_loop as original
from tests.test_dreamerv3_reused_train_budget_probe import BudgetProbeFixture, _fake_modules
from tests.test_dreamerv3_reused_train_open_loop import _sha


class PriorScoreFixture(BudgetProbeFixture):
    def setUp(self):
        super().setUp()
        with mock.patch.object(budget, "_modules", side_effect=lambda *args, **kwargs: _fake_modules()):
            budget.score(self.budget_protocol_path, self.budget_protocol_sha,
                         self.budget_output, repo_root=self.root)
        self.prior_output = self.root / budget.RUN_ROOT / "scoring-prior-v1"
        self.prior_protocol_path = self.root / score.SCORE_PROTOCOL_PATH
        self.budget_result_sha = _sha(self.root / score.BUDGET_RESULT_PATH)
        for key, value in (("BUDGET_PROTOCOL_SHA256", self.budget_protocol_sha),
                           ("BUDGET_RESULT_SHA256", self.budget_result_sha)):
            patch = mock.patch.object(score, key, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.anchor = json.loads((self.root / score.BUDGET_RESULT_PATH).read_text())[
            "strata"]["teacher"]["models"][0]["aggregate"]["metrics_episode_mean"]["shifted_repeat_mse"]
        anchor_patch = mock.patch.object(score, "ANCHOR_MSE", self.anchor)
        gate_patch = mock.patch.object(score, "GATE", {**score.GATE, "teacher_shifted_repeat_mse": self.anchor})
        anchor_patch.start()
        gate_patch.start()
        self.addCleanup(anchor_patch.stop)
        self.addCleanup(gate_patch.stop)
        self._make_prior()

    def _make_prior(self):
        self.prior_sources = dict(self.v2_offline["source_sha256"])
        for name in ("scripts/train_dreamerv3_prior_image.py", "haic/algorithms/dreamer_v3/prior_image.py"):
            self.prior_sources[name] = self._write(name, b"# synthetic pinned prior source\n")
        self.prior = {
            "format": "haic-dreamerv3-reused-prior-image-offline-v1",
            "purpose": "reused-TRAIN-prior-image-auxiliary-diagnostic", "study_id": self.train_id,
            "base_offline_protocol": copy.deepcopy(self.budget_protocol["offline_protocol"]),
            "source_sha256": self.prior_sources,
            "learner": {"base_updates": 256, "aux_updates": 64, "aux_every_base_updates": 4,
                        "prior_horizon": 8, "prior_weight": 0.25,
                        "anchor_mode": "deterministic-first-last-full-horizon",
                        "aux_rng_mode": "fork-torch-numpy-python", "aux_rng_seed_base": 260926000,
                        "aux_rng_seed_stride": 1000},
            "resources": {key: self.budget_protocol["resources"][key] for key in (
                "max_archive_bytes", "max_cgroup_memory_bytes", "min_cgroup_available_bytes")},
            "output_root": score.TRAIN_ROOT,
        }
        prior_sha = self._json(score.PRIOR_PROTOCOL_PATH, self.prior)
        patch = mock.patch.object(score, "PRIOR_PROTOCOL_SHA256", prior_sha)
        patch.start()
        self.addCleanup(patch.stop)
        scorer_path = self.root / score.SOURCE
        scorer_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(score.__file__, scorer_path)
        self.score_sources = {**self.budget_protocol["source_sha256"], **self.prior_sources,
                              score.SOURCE: _sha(scorer_path)}
        self.prior_training = {}
        for arm in ("random", "teacher"):
            self.prior_training[arm] = []
            train = self.v2_offline["datasets"][arm]
            for seed, original_row in enumerate(self.budget_training[arm]):
                prefix = f"{score.TRAIN_ROOT}/{arm}-seed-{seed}"
                checkpoint_name = f"{prefix}/world-model-checkpoint.pt"
                baseline = torch.load(self.root / original_row["checkpoint_path"],
                                      map_location="cpu", weights_only=False)
                payload = copy.deepcopy(baseline)
                base_name, aux_name = (f"{prefix}/{kind}-update-metrics.jsonl" for kind in ("base", "aux"))
                base_sha = self._write(base_name, b"".join(json.dumps({
                    "base_update": index, "metrics": {"loss_wm": 0.5}}, sort_keys=True).encode() + b"\n"
                    for index in range(1, 257)))
                aux_sha = self._write(aux_name, b"".join(json.dumps({
                    "aux_update": index, "after_base_update": index * 4,
                    "metrics": {"aux_loss": 0.5, "frame_mse": 0.25, "anchor_count": 8,
                                "target_count": 64, "effective_horizon": 8}}, sort_keys=True).encode() + b"\n"
                    for index in range(1, 65)))
                payload["run_metadata"] = {
                    "purpose": self.prior["purpose"], "study_id": self.train_id, "arm": arm, "seed": seed,
                    "prior_protocol_sha256": prior_sha,
                    "base_offline_protocol_sha256": self.v2_offline_sha,
                    "source_sha256": self.prior_sources,
                    "collection_receipt_sha256": train["receipt_sha256"],
                    "archive_sha256": train["archive_sha256"], "dataset_digest": train["dataset_digest"],
                    "base_metrics_sha256": base_sha, "aux_metrics_sha256": aux_sha,
                    "base_model_only_updates": 256, "aux_world_model_optimizer_steps": 64,
                    "world_model_optimizer_steps": 320, "prior_horizon": 8, "prior_weight": 0.25,
                    "anchor_mode": self.prior["learner"]["anchor_mode"],
                    "aux_rng_mode": self.prior["learner"]["aux_rng_mode"],
                    "aux_rng_seed_base": 260926000, "aux_rng_seed_stride": 1000,
                    "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                    "promotion_eligible": False,
                }
                checkpoint_file = self.root / checkpoint_name
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_file)
                result = {
                    "format": "haic-dreamerv3-reused-prior-image-result-v1", "status": "complete",
                    "purpose": self.prior["purpose"], "study_id": self.train_id, "arm": arm, "seed": seed,
                    "prior_protocol_sha256": prior_sha,
                    "base_offline_protocol_path": budget.V2_OFFLINE_PATH,
                    "base_offline_protocol_sha256": self.v2_offline_sha,
                    "collection_protocol_sha256": self.v2_offline["collection_protocol"]["sha256"],
                    "source_sha256": self.prior_sources,
                    "collection_receipt_path": train["receipt_path"],
                    "collection_receipt_sha256": train["receipt_sha256"],
                    "checkpoint_path": "world-model-checkpoint.pt", "checkpoint_sha256": _sha(checkpoint_file),
                    "base_metrics_path": "base-update-metrics.jsonl", "base_metrics_sha256": base_sha,
                    "aux_metrics_path": "aux-update-metrics.jsonl", "aux_metrics_sha256": aux_sha,
                    "collection_decisions_spent": train["stored_decisions"],
                    "environment_steps": 0, "base_model_only_updates": 256,
                    "aux_world_model_optimizer_steps": 64, "world_model_optimizer_steps": 320,
                    "aux_after_every_base_updates": 4, "prior_horizon": 8, "prior_weight": 0.25,
                    "anchor_mode": self.prior["learner"]["anchor_mode"],
                    "aux_rng_mode": self.prior["learner"]["aux_rng_mode"],
                    "aux_rng_seed_base": 260926000, "aux_rng_seed_stride": 1000,
                    "actor_critic_target_unchanged": True, "actor_critic_optimizers_unchanged": True,
                    "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                    "promotion_eligible": False, "matched_pure_256_step_model": False,
                    **{key: train[key] for key in ("archive_path", "archive_sha256", "dataset_digest",
                                                    "source_id", "source_actor_sha256")},
                }
                result_name = f"{prefix}/training-result.json"
                result_sha = self._json(result_name, result)
                self.prior_training[arm].append({"seed": seed, "result_path": result_name,
                    "result_sha256": result_sha, "checkpoint_path": checkpoint_name,
                    "checkpoint_sha256": _sha(checkpoint_file)})
        self.score_protocol = {
            "format": score.FORMAT, "purpose": score.PURPOSE, "study_id": self.dev_id,
            "prior_protocol": {"path": score.PRIOR_PROTOCOL_PATH, "sha256": prior_sha},
            "budget_score_protocol": {"path": score.BUDGET_PROTOCOL_PATH,
                                      "sha256": self.budget_protocol_sha},
            "budget_score_result": {"path": score.BUDGET_RESULT_PATH,
                                    "sha256": self.budget_result_sha},
            "training": self.prior_training, "source_sha256": self.score_sources,
            "local_gate": copy.deepcopy(score.GATE),
        }
        self.freeze_prior_score()

    def freeze_prior_score(self):
        self.prior_score_sha = self._json(score.SCORE_PROTOCOL_PATH, self.score_protocol)

    def preflight_prior(self):
        return score.preflight(self.prior_protocol_path, self.prior_score_sha,
                               self.prior_output, repo_root=self.root)

    def refresh_result(self, arm="random", seed=0):
        row = self.prior_training[arm][seed]
        result = json.loads((self.root / row["result_path"]).read_text())
        result["checkpoint_sha256"] = row["checkpoint_sha256"]
        row["result_sha256"] = self._json(row["result_path"], result)
        self.freeze_prior_score()
        return row, result


class ProvenanceTests(PriorScoreFixture):
    def test_preflight_pins_four_aux_pairs_original_score_and_dev_bytes_without_zip_decode(self):
        with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP decode")):
            checked = self.preflight_prior()
        self.assertEqual([row["row"]["seed"] for row in checked["prior_training"]["teacher"]], [0, 1])
        self.assertFalse(self.prior_output.exists())
        self.assertEqual(checked["score_protocol"]["local_gate"]["teacher_shifted_repeat_mse"], self.anchor)

    def test_wrong_aux_count_and_edited_step_stream_fail_before_zip(self):
        row, result = self.refresh_result()
        result["aux_world_model_optimizer_steps"] = 63
        row["result_sha256"] = self._json(row["result_path"], result)
        self.freeze_prior_score()
        with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP decode")):
            with self.assertRaisesRegex(ValueError, "aux_world_model_optimizer_steps"):
                self.preflight_prior()
        result["aux_world_model_optimizer_steps"] = 64
        row["result_sha256"] = self._json(row["result_path"], result)
        self.freeze_prior_score()
        stream = self.root / score.TRAIN_ROOT / "random-seed-0" / "aux-update-metrics.jsonl"
        stream.write_bytes(stream.read_bytes().splitlines(keepends=True)[0])
        result["aux_metrics_sha256"] = _sha(stream)
        row["result_sha256"] = self._json(row["result_path"], result)
        self.freeze_prior_score()
        with self.assertRaisesRegex(ValueError, "wrong optimizer-step metrics count"):
            self.preflight_prior()

    def test_actor_drift_gradient_step_and_environment_step_fail_without_scoring(self):
        row, result = self.refresh_result("teacher", 0)
        path = self.root / row["checkpoint_path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["actor"]["weight"] += 1
        torch.save(payload, path)
        row["checkpoint_sha256"] = _sha(path)
        self.refresh_result("teacher", 0)
        with self.assertRaisesRegex(ValueError, "actor differs"):
            self.preflight_prior()
        payload["actor"]["weight"] -= 1
        payload["gradient_steps"] = 320
        torch.save(payload, path)
        row["checkpoint_sha256"] = _sha(path)
        self.refresh_result("teacher", 0)
        with self.assertRaisesRegex(ValueError, "gradient steps"):
            self.preflight_prior()
        payload["gradient_steps"] = 256
        payload["environment_steps"] = 1
        torch.save(payload, path)
        row["checkpoint_sha256"] = _sha(path)
        self.refresh_result("teacher", 0)
        with self.assertRaisesRegex(ValueError, "gradient steps"):
            self.preflight_prior()

    def test_checkpoint_optimizer_or_aux_metadata_drift_rejected(self):
        row, _ = self.refresh_result("random", 1)
        path = self.root / row["checkpoint_path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["run_metadata"]["aux_world_model_optimizer_steps"] = 63
        torch.save(payload, path)
        row["checkpoint_sha256"] = _sha(path)
        self.refresh_result("random", 1)
        with self.assertRaisesRegex(ValueError, "run_metadata"):
            self.preflight_prior()
        payload["run_metadata"]["aux_world_model_optimizer_steps"] = 64
        payload["actor_optimizer"]["state"] = {1: {"step": 1}}
        torch.save(payload, path)
        row["checkpoint_sha256"] = _sha(path)
        self.refresh_result("random", 1)
        with self.assertRaisesRegex(ValueError, "actor_optimizer differs"):
            self.preflight_prior()

    def test_aux_source_hash_and_original_result_sha_fail_closed(self):
        self.score_protocol["source_sha256"]["dreamer_v3.py"] = "f" * 64
        self.freeze_prior_score()
        with self.assertRaisesRegex(ValueError, "source SHA maps"):
            self.preflight_prior()
        self.score_protocol["source_sha256"]["dreamer_v3.py"] = self.score_sources["dreamer_v3.py"]
        self.score_protocol["budget_score_result"]["sha256"] = "f" * 64
        self.freeze_prior_score()
        with self.assertRaisesRegex(ValueError, "frozen reference"):
            self.preflight_prior()

    def test_window_and_training_baseline_drift_fail_before_archive_decode(self):
        path = self.root / score.BUDGET_RESULT_PATH
        frozen = json.loads(path.read_text())
        frozen["strata"]["random"]["models"][0]["episodes"][0]["windows"][1]["context_anchor_decision"] = 8
        new_sha = self._json(score.BUDGET_RESULT_PATH, frozen)
        self.score_protocol["budget_score_result"]["sha256"] = new_sha
        with mock.patch.object(score, "BUDGET_RESULT_SHA256", new_sha), mock.patch.object(
                original, "_load_dataset", side_effect=AssertionError("ZIP decode")):
            self.freeze_prior_score()
            with self.assertRaisesRegex(ValueError, "window identity"):
                self.preflight_prior()
        frozen["strata"]["random"]["models"][0]["episodes"][0]["windows"][1]["context_anchor_decision"] = 9
        frozen["strata"]["teacher"]["models"][2]["training_baselines"]["constant_reward"] = 999
        new_sha = self._json(score.BUDGET_RESULT_PATH, frozen)
        self.score_protocol["budget_score_result"]["sha256"] = new_sha
        with mock.patch.object(score, "BUDGET_RESULT_SHA256", new_sha), mock.patch.object(
                original, "_load_dataset", side_effect=AssertionError("ZIP decode")):
            self.freeze_prior_score()
            with self.assertRaisesRegex(ValueError, "baseline changed"):
                self.preflight_prior()

    def test_output_collision_and_bounded_archive_hash(self):
        self.prior_output.mkdir()
        with self.assertRaises(FileExistsError):
            self.preflight_prior()
        self.prior_output.rmdir()
        path = self.root / self.development["random"]["archive_path"]
        path.write_bytes(path.read_bytes() + b"tampered")
        with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP decode")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.preflight_prior()

    def test_bounded_synthetic_zip_and_cgroup_floor(self):
        archive = self.root / self.development["random"]["archive_path"]
        with self.assertRaisesRegex(ValueError, "ZIP entries"):
            original._dataset_bytes(archive, _sha(archive), archive.stat().st_size,
                                    256, self.budget_protocol["resources"]["max_decisions_per_archive"])
        with mock.patch.object(original, "_cgroup", return_value={
                "limit_bytes": 128 * 1024**3, "available_bytes": 0}):
            with self.assertRaisesRegex(ValueError, "cgroup lacks"):
                self.preflight_prior()


class PairingTests(PriorScoreFixture):
    def test_fake_archive_scoring_reuses_exact_seeds_windows_and_no_training_or_environment(self):
        with mock.patch.object(score, "_modules", side_effect=lambda *args: _fake_modules()), mock.patch.object(
                original, "_score_episode", wraps=original._score_episode) as scored, mock.patch.object(
                original, "_load_dataset", wraps=original._load_dataset) as loaded:
            report = score.score(self.prior_protocol_path, self.prior_score_sha,
                                 self.prior_output, repo_root=self.root)
        self.assertEqual(loaded.call_count, 2)
        self.assertEqual([call.args[-1] for call in scored.call_args_list],
                         [120926 + (source == "teacher") * 1_000_003 + index * 1009 + seed
                          for source in ("random", "teacher") for arm in ("random", "teacher")
                          for seed in (0, 1) for index in range(4)])
        self.assertEqual(report["status"], "iterative_tuning_descriptive_only")
        self.assertFalse(report["fresh_claim"] or report["student_actor_trained"] or report["promotion_eligible"])
        self.assertFalse(report["local_engineering_gate"]["passed"])
        for stratum in report["strata"].values():
            for model in stratum["models"]:
                self.assertEqual(len(model["episodes"]), len(model["roads"]))
                self.assertEqual(model["aggregate"]["window_label_uses"], 256)
                self.assertTrue(all(value == 0 for value in model["aggregate"]["paired_delta_vs_256"].values()))
                for episode in model["episodes"]:
                    self.assertEqual([(row["kind"], row["context_anchor_decision"], row["label_uses"])
                                      for row in episode["windows"]], [("reset", 8, 32), ("terminal", 9, 32)])
        self.assertEqual(json.loads((self.prior_output / "score-result.json").read_text()), report)
        with self.assertRaises(FileExistsError):
            self.preflight_prior()

    def test_original_input_drift_after_scoring_prevents_success_output(self):
        frozen_result = self.root / budget.ORIGINAL_SCORE_RESULT_PATH
        count = 0
        score_episode = original._score_episode

        def drift(*args):
            nonlocal count
            count += 1
            if count == 1:
                frozen_result.write_text("{}\n")
            return score_episode(*args)

        with mock.patch.object(score, "_modules", side_effect=lambda *args: _fake_modules()), mock.patch.object(
                original, "_score_episode", side_effect=drift):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                score.score(self.prior_protocol_path, self.prior_score_sha,
                            self.prior_output, repo_root=self.root)
        self.assertFalse(self.prior_output.exists())

    def test_pair_rejects_baseline_or_window_drift(self):
        frozen = json.loads((self.root / score.BUDGET_RESULT_PATH).read_text())["strata"]["random"]["models"][0]
        current = copy.deepcopy(frozen)
        current["aggregate"].pop("paired_delta_vs_64")
        for section in ("roads", "episodes"):
            for row in current[section]:
                row.pop("paired_delta_vs_64")
                if section == "episodes":
                    for window in row["windows"]:
                        window.pop("paired_delta_vs_64")
        current["aggregate"]["metrics_episode_mean"]["shifted_repeat_mse"] += 0.01
        with self.assertRaisesRegex(ValueError, "baseline differs"):
            score._pair(current, frozen)
        current["aggregate"]["metrics_episode_mean"]["shifted_repeat_mse"] -= 0.01
        current["episodes"][0]["windows"][1]["context_anchor_decision"] = 8
        with self.assertRaisesRegex(ValueError, "window identity"):
            score._pair(current, frozen)

    def test_gate_requires_both_teacher_seeds_both_strata_and_teacher_anchor(self):
        frozen = json.loads((self.root / score.BUDGET_RESULT_PATH).read_text())
        strata = copy.deepcopy(frozen["strata"])
        for source in ("random", "teacher"):
            for seed in (0, 1):
                model = strata[source]["models"][2 + seed]
                image = model["aggregate"]["metrics_episode_mean"]["image_mse"]
                model["aggregate"]["original_256_metrics"] = {"image_mse": image}
                model["aggregate"]["metrics_episode_mean"]["image_mse"] = min(image, self.anchor) - 0.001
        self.assertTrue(score._gate(strata)["passed"])
        strata["random"]["models"][3]["aggregate"]["metrics_episode_mean"]["image_mse"] = (
            strata["random"]["models"][3]["aggregate"]["original_256_metrics"]["image_mse"])
        self.assertFalse(score._gate(strata)["passed"])
        strata["random"]["models"][3]["aggregate"]["metrics_episode_mean"]["image_mse"] -= 0.001
        strata["teacher"]["models"][2]["aggregate"]["metrics_episode_mean"]["image_mse"] = self.anchor
        self.assertFalse(score._gate(strata)["passed"])


if __name__ == "__main__":
    unittest.main()
