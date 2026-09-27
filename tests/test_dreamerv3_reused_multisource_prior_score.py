"""Synthetic ZIPs/checkpoints only; never open real development archives."""

from __future__ import annotations

import copy
import json
import shutil
import unittest
from unittest import mock

import torch

from scripts.diagnose import dreamerv3_reused_multisource_open_loop as mixed
from scripts.diagnose import dreamerv3_reused_multisource_prior_score as score
from scripts.diagnose import dreamerv3_reused_train_open_loop as original
from tests.test_dreamerv3_reused_multisource_open_loop import SyntheticFixture, _digest, _modules


class SyntheticPriorFixture(SyntheticFixture):
    def setUp(self):
        super().setUp()
        for label in ("source0", "source1"):
            protocol_path = f"experiments/dreamerv3-reused-train-{label}-synthetic-v1.json"
            protocol_sha = self.put_json(protocol_path, {"source": label, "synthetic_only": True})
            self.train[f"{label}_protocol"] = {"path": protocol_path, "sha256": protocol_sha}
            receipt = self.train_receipts[label]
            receipt["protocol_sha256"] = protocol_sha
            receipt["r6_protocol_sha256"] = self.r6_sha
            receipt["source_checkpoint_sha256"] = self.train["datasets"][label]["source_checkpoint_sha256"]
            row = self.train["datasets"][label]
            row["receipt_sha256"] = self.put_json(row["receipt_path"], receipt)
        self.train["resources"] = {"max_archive_bytes": 4 * 1024**2,
                                   "max_total_uncompressed_bytes": 16 * 1024**2,
                                   "max_cgroup_memory_bytes": 32 * 1024**3,
                                   "min_cgroup_available_bytes": 8 * 1024**3,
                                   "min_disk_available_bytes": 2 * 1024**3}
        self.train_sha = self.put_json(mixed.TRAIN_PATH, self.train)
        self.patch_value("TRAIN_SHA256", self.train_sha)
        self.protocol["training_protocol"]["sha256"] = self.train_sha
        for row in self.training:
            path = self.root / row["checkpoint_path"]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["run_metadata"]["protocol_sha256"] = self.train_sha
            payload["run_metadata"]["sources"] = self.train["datasets"]
            torch.save(payload, path)
            row["checkpoint_sha256"] = _digest(path)
            result = json.loads((self.root / row["result_path"]).read_text())
            result["checkpoint_sha256"] = row["checkpoint_sha256"]
            result["protocol_sha256"] = self.train_sha
            result["sources"] = self.train["datasets"]
            result["source_audits"] = {name: {"source_id": source["source_id"],
                "source_actor_sha256": source["source_actor_sha256"],
                "archive_sha256": source["archive_sha256"], "dataset_digest": source["dataset_digest"],
                "decisions": source["stored_decisions"], "episodes": 12,
                "distinct_finished_cells": 0,
                "covered_cells": sorted([[1, road] for road in self.roads])}
                for name, source in self.train["datasets"].items()}
            result["union_finished_roads"] = []
            row["result_sha256"] = self.put_json(row["result_path"], result)
        self.patch_value("RESULT_SHAS", tuple(row["result_sha256"] for row in self.training))
        self.patch_value("CHECKPOINT_SHAS", tuple(row["checkpoint_sha256"] for row in self.training))
        for label, arm in (("source0", "teacher"), ("random", "random")):
            row = self.development[label]
            digest, archive_sha = self.archive(row["archive_path"], row["source_id"],
                                               row["source_actor_sha256"], self.dev_roads,
                                               self.dev["study_id"], arm, 72)
            receipt = json.loads((self.root / row["receipt_path"]).read_text())
            receipt.update(dataset_digest=digest, archive_sha256=archive_sha,
                           stored_decisions=288, decisions_spent=288)
            for episode in receipt["episode_rows"]:
                episode["decisions"] = 72
            row.update(dataset_digest=digest, archive_sha256=archive_sha)
            row["receipt_sha256"] = self.put_json(row["receipt_path"], receipt)
        self.freeze()
        with mock.patch.object(mixed, "_checkpoint", return_value=_modules(0)):
            mixed.score(mixed.PROTOCOL_PATH, self.protocol_sha, mixed.OUTPUT, repo_root=self.root)
        self.output = self.root / "runs/synthetic-multisource/prior-interaction-score-v1"
        self.patch_prior("RUN_ROOT", "runs/synthetic-multisource/prior-interaction-v1")
        self.patch_prior("OUTPUT", "runs/synthetic-multisource/prior-interaction-score-v1")
        self.patch_prior("PRIOR_SHA256", "0" * 64)
        self.patch_prior("PURE_PROTOCOL_SHA256", self.protocol_sha)
        self.patch_prior("PURE_RESULT_PATH", mixed.OUTPUT + "/score-result.json")
        self.patch_prior("PURE_RESULT_SHA256", _digest(self.root / mixed.OUTPUT / "score-result.json"))
        self.patch_prior("LINEAGE_SHA256", mixed.LINEAGE_SHA256)
        self.patch_prior("RESULT_SHAS", ("0" * 64, "0" * 64))
        self.patch_prior("CHECKPOINT_SHAS", ("0" * 64, "0" * 64))
        self.prior_sources = dict(self.train["source_sha256"])
        for name in ("scripts/train_dreamerv3_reused_multisource_prior.py",
                     "scripts/train_dreamerv3_prior_image.py",
                     "haic/algorithms/dreamer_v3/prior_image.py"):
            self.prior_sources[name] = self.put_bytes(name, b"# synthetic unexecuted H8 training source\n")
        self.prior = {"format": "haic-dreamerv3-reused-train-multisource-prior-v1",
                      "purpose": "reused-TRAIN-prior-image-interaction-diagnostic",
                      "study_id": "dreamerv3-reused-train-multisource-prior-v1",
                      "mixed_protocol": self.protocol["training_protocol"],
                      "r6_protocol": self.train["r6_protocol"],
                      "source0_protocol": self.train["source0_protocol"],
                      "source1_protocol": self.train["source1_protocol"],
                      "cells": self.train["cells"], "datasets": self.train["datasets"],
                      "baseline_results": [{key: row[key] for key in (
                          "seed", "result_path", "result_sha256", "lineage_path", "lineage_sha256")}
                          for row in self.training],
                      "expected_lineage_sha256": mixed.LINEAGE_SHA256,
                      "output_root": score.RUN_ROOT, "source_sha256": self.prior_sources,
                      "resources": {**self.train["resources"], "min_cgroup_available_bytes": 12 * 1024**3},
                      "learner": {"seeds": [0, 1], "config": self.config, "base_updates": 256,
                                  "aux_updates": 64, "aux_every_base_updates": 4,
                                  "prior_horizon": 8, "prior_weight": 0.25,
                                  "anchor_mode": "deterministic-first-last-full-horizon",
                                  "aux_rng_mode": "fork-torch-numpy-python",
                                  "aux_rng_seed_base": 260926000, "aux_rng_seed_stride": 1000}}
        prior_sha = self.put_json(score.PRIOR_PATH, self.prior)
        self.patch_prior("PRIOR_SHA256", prior_sha)
        source_path = self.root / score.SOURCE
        source_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(score.__file__, source_path)
        sources = {**self.protocol["source_sha256"], **self.prior_sources,
                   score.SOURCE: _digest(source_path)}
        self.prior_training = []
        for seed, baseline_row in enumerate(self.training):
            prefix = f"{score.RUN_ROOT}/learner-{seed}"
            lineage_path = prefix + "/lineage.json"
            self.put_bytes(lineage_path, (self.root / baseline_row["lineage_path"]).read_bytes())
            step_shas = {}
            for kind, count in (("base", 256), ("aux", 64)):
                path = prefix + f"/{kind}-update-metrics.jsonl"
                stream = b"".join((json.dumps({"base_update": i, "metrics": {"loss_wm": 0.5}}
                    if kind == "base" else {"aux_update": i, "after_base_update": i * 4,
                    "metrics": {"aux_loss": 0.5, "frame_mse": 0.25, "anchor_count": 8,
                                "target_count": 64, "effective_horizon": 8}}, sort_keys=True) + "\n").encode()
                    for i in range(1, count + 1))
                step_shas[kind] = self.put_bytes(path, stream)
            metadata = {"purpose": self.prior["purpose"], "study_id": self.prior["study_id"],
                        "seed": seed, "protocol_sha256": prior_sha,
                        "mixed_protocol": self.prior["mixed_protocol"],
                        "pure_256_baselines": self.prior["baseline_results"],
                        "source_sha256": self.prior_sources, "sources": self.train["datasets"],
                        "lineage_path": lineage_path, "lineage_sha256": mixed.LINEAGE_SHA256,
                        "base_metrics_sha256": step_shas["base"], "aux_metrics_sha256": step_shas["aux"],
                        "base_model_only_updates": 256, "aux_world_model_optimizer_steps": 64,
                        "world_model_optimizer_steps": 320, "aux_after_every_base_updates": 4,
                        "prior_horizon": 8, "prior_weight": 0.25,
                        "anchor_mode": self.prior["learner"]["anchor_mode"],
                        "aux_rng_mode": self.prior["learner"]["aux_rng_mode"],
                        "aux_rng_seed_base": 260926000, "aux_rng_seed_stride": 1000,
                        "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                        "promotion_eligible": False, "matched_pure_256_step_model": False}
            payload = torch.load(self.root / baseline_row["checkpoint_path"],
                                 map_location="cpu", weights_only=False)
            payload["run_metadata"] = metadata
            checkpoint_path = prefix + "/world-model-checkpoint.pt"
            torch.save(payload, self.root / checkpoint_path)
            ckpt_sha = _digest(self.root / checkpoint_path)
            base_result = json.loads((self.root / baseline_row["result_path"]).read_text())
            result = {"format": self.prior["format"] + "-result", "status": "complete",
                      **metadata, "checkpoint_path": "world-model-checkpoint.pt",
                      "checkpoint_sha256": ckpt_sha, "base_metrics_path": "base-update-metrics.jsonl",
                      "aux_metrics_path": "aux-update-metrics.jsonl", "environment_steps": 0,
                      "actor_critic_target_optimizers_unchanged": True,
                      "distinct_training_roads": 12, "episodes": 24, "decisions": 24,
                      "source_audits": base_result["source_audits"],
                      "union_finished_roads": base_result["union_finished_roads"]}
            result_path = prefix + "/training-result.json"
            self.prior_training.append({"seed": seed, "result_path": result_path,
                "result_sha256": self.put_json(result_path, result), "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": ckpt_sha, "lineage_path": lineage_path,
                "lineage_sha256": mixed.LINEAGE_SHA256})
        self.patch_prior("RESULT_SHAS", tuple(row["result_sha256"] for row in self.prior_training))
        self.patch_prior("CHECKPOINT_SHAS", tuple(row["checkpoint_sha256"] for row in self.prior_training))
        self.score_protocol = {"format": score.FORMAT, "purpose": score.PURPOSE,
                               "study_id": score.STUDY_ID,
                               "prior_protocol": {"path": score.PRIOR_PATH, "sha256": prior_sha},
                               "pure_256_score_protocol": {"path": mixed.PROTOCOL_PATH,
                                                           "sha256": self.protocol_sha},
                               "pure_256_score_result": {"path": mixed.OUTPUT + "/score-result.json",
                                                         "sha256": score.PURE_RESULT_SHA256},
                               "training": copy.deepcopy(self.prior_training),
                               "development": copy.deepcopy(self.development),
                               "source_sha256": sources, "scoring": self.protocol["scoring"],
                               "resources": copy.deepcopy(self.protocol["resources"]),
                               "local_gate": copy.deepcopy(score.GATE)}
        self.freeze_score()

    def patch_prior(self, name, value):
        patcher = mock.patch.object(score, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def freeze_score(self):
        self.score_sha = self.put_json(score.PROTOCOL_PATH, self.score_protocol)

    def preflight_prior(self):
        self.freeze_score()
        return score.preflight(score.PROTOCOL_PATH, self.score_sha, score.OUTPUT, repo_root=self.root)

    def score_prior(self):
        self.freeze_score()
        return score.score(score.PROTOCOL_PATH, self.score_sha, score.OUTPUT, repo_root=self.root)

    def refresh_training(self, seed=0):
        row = self.prior_training[seed]
        name = row["result_path"]
        result = json.loads((self.root / name).read_text())
        result["checkpoint_sha256"] = row["checkpoint_sha256"]
        row["result_sha256"] = self.put_json(name, result)
        self.score_protocol["training"] = copy.deepcopy(self.prior_training)
        self.patch_prior("RESULT_SHAS", tuple(entry["result_sha256"] for entry in self.prior_training))
        self.patch_prior("CHECKPOINT_SHAS", tuple(entry["checkpoint_sha256"] for entry in self.prior_training))
        return row, result


class PriorScoreContracts(SyntheticPriorFixture):
    def test_preflight_pins_both_models_lineage_and_same_development_without_zip_decode(self):
        with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            checked = self.preflight_prior()
        self.assertEqual([part["row"]["seed"] for part in checked["training"]], [0, 1])
        self.assertEqual(checked["baseline"]["training_decisions"], 24)
        self.assertFalse(self.output.exists())

    def test_score_reuses_original_seeds_windows_and_three_paired_proxy_deltas(self):
        with mock.patch.object(score, "_checkpoint", return_value=_modules(0)), mock.patch.object(
                original, "_score_episode", wraps=original._score_episode) as scorer, mock.patch.object(
                original, "_load_dataset", wraps=original._load_dataset) as load:
            report = self.score_prior()
        self.assertEqual(load.call_count, 2)
        self.assertEqual([call.args[-1] for call in scorer.call_args_list],
                         [120927 + source * 1_000_003 + index * 1009 + seed
                          for source in (0, 1) for seed in (0, 1) for index in range(4)])
        self.assertEqual(report["status"], "iterative_tuning_descriptive_only")
        self.assertFalse(report["local_engineering_gate"]["passed"])
        self.assertFalse(report["promotion_eligible"] or report["student_actor_trained"])
        for stratum in report["strata"].values():
            self.assertEqual(stratum["collected_independent_roads"], 4)
            for model in stratum["models"]:
                self.assertEqual(len(model["roads"]), 4)
                self.assertEqual(model["aggregate"]["window_label_uses"], 256)
                self.assertEqual(model["aggregate"]["unique_terminal_positive_labels"], 4)
                self.assertEqual(model["aggregate"]["paired_delta_vs_pure_256"],
                                 {key: 0 for key in score.PREDICTIONS})
                for episode in model["episodes"]:
                    self.assertEqual([row["kind"] for row in episode["windows"]], ["reset", "terminal"])
                    self.assertEqual(len(episode["windows"][0]["paired_delta_vs_pure_256"]), 3)
        self.assertEqual(json.loads((self.output / "score-result.json").read_text()), report)
        with self.assertRaises(FileExistsError):
            self.preflight_prior()

    def test_gate_is_strict_for_each_seed_in_each_stratum(self):
        strata = copy.deepcopy(self.preflight_prior()["frozen"]["strata"])
        for label in score.STRATA:
            for seed in (0, 1):
                model = strata[label]["models"][seed]
                metrics = model["aggregate"]["metrics_episode_mean"]
                model["aggregate"]["pure_256_metrics"] = copy.deepcopy(metrics)
                metrics["image_mse"] = min(metrics["image_mse"], metrics["shifted_repeat_mse"]) - 0.0001
        self.assertTrue(score._gate(strata)["passed"])
        model = strata["random"]["models"][1]
        model["aggregate"]["metrics_episode_mean"]["image_mse"] = model["aggregate"]["pure_256_metrics"]["image_mse"]
        self.assertFalse(score._gate(strata)["passed"])
        model["aggregate"]["metrics_episode_mean"]["image_mse"] = 0
        model = strata["source0"]["models"][0]
        model["aggregate"]["metrics_episode_mean"]["image_mse"] = model["aggregate"]["pure_256_metrics"]["shifted_repeat_mse"]
        self.assertFalse(score._gate(strata)["passed"])

    def test_actor_and_optimizer_mutation_rejected_with_recomputed_sha_before_zip(self):
        row = self.prior_training[0]
        path = self.root / row["checkpoint_path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["actor"][next(iter(payload["actor"]))].add_(1)
        torch.save(payload, path)
        row["checkpoint_sha256"] = _digest(path)
        self.refresh_training()
        with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            with self.assertRaisesRegex(ValueError, "seeded actor/critic"):
                self.preflight_prior()
        payload["actor"][next(iter(payload["actor"]))].sub_(1)
        payload["actor_optimizer"]["state"] = {1: {"step": 1}}
        torch.save(payload, path)
        row["checkpoint_sha256"] = _digest(path)
        self.refresh_training()
        with self.assertRaisesRegex(ValueError, "seeded actor/critic"):
            self.preflight_prior()

    def test_wrong_base_aux_or_environment_counts_fail_before_zip(self):
        row, result = self.refresh_training()
        for field, wrong in (("base_model_only_updates", 255),
                             ("aux_world_model_optimizer_steps", 63),
                             ("world_model_optimizer_steps", 319), ("environment_steps", 1)):
            old = result[field]
            result[field] = wrong
            row["result_sha256"] = self.put_json(row["result_path"], result)
            self.score_protocol["training"] = copy.deepcopy(self.prior_training)
            self.patch_prior("RESULT_SHAS", tuple(entry["result_sha256"] for entry in self.prior_training))
            with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
                with self.assertRaisesRegex(ValueError, field):
                    self.preflight_prior()
            result[field] = old

    def test_checkpoint_counter_metadata_or_source_actor_forgery_rejected(self):
        row = self.prior_training[1]
        path = self.root / row["checkpoint_path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["gradient_steps"] = 320
        torch.save(payload, path)
        row["checkpoint_sha256"] = _digest(path)
        self.refresh_training(1)
        with self.assertRaisesRegex(ValueError, "checkpoint metadata/config/model-only counters"):
            self.preflight_prior()
        payload["gradient_steps"] = 256
        payload["run_metadata"]["sources"]["source1"]["source_actor_sha256"] = "f" * 64
        torch.save(payload, path)
        row["checkpoint_sha256"] = _digest(path)
        self.refresh_training(1)
        with self.assertRaisesRegex(ValueError, "checkpoint metadata/config/model-only counters"):
            self.preflight_prior()

    def test_wrong_lineage_ranges_and_sidecar_sha_rejected_before_zip(self):
        row = self.prior_training[0]
        ledger = json.loads((self.root / row["lineage_path"]).read_text())
        ledger["episodes"][1]["sequence_id_start"] = 0
        row["lineage_sha256"] = self.put_json(row["lineage_path"], ledger)
        self.score_protocol["training"] = copy.deepcopy(self.prior_training)
        with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            with self.assertRaisesRegex(ValueError, "fixed SHA pair"):
                self.preflight_prior()
        row["lineage_sha256"] = mixed.LINEAGE_SHA256
        self.score_protocol["training"] = copy.deepcopy(self.prior_training)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.preflight_prior()

    def test_wrong_sources_protocol_and_original_result_sha_rejected(self):
        self.score_protocol["source_sha256"]["dreamer_v3.py"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "source SHA map"):
            self.preflight_prior()
        self.score_protocol["source_sha256"]["dreamer_v3.py"] = self.prior_sources["dreamer_v3.py"]
        self.score_protocol["pure_256_score_result"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "pure_256_score_result"):
            self.preflight_prior()
        self.score_protocol["pure_256_score_result"]["sha256"] = score.PURE_RESULT_SHA256
        self.score_protocol["development"]["random"]["source_actor_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "development"):
            self.preflight_prior()

    def test_catalog_has_bounded_four_mib_cap_without_weakening_other_refs(self):
        self.assertEqual(score._reference_cap("catalog"), 4 * 1024**2)
        self.assertEqual(score._reference_cap("historical_train_summary"), 1024**2)
        self.assertEqual(score._reference_cap("r6_protocol"), 1024**2)
        name = "experiments/synthetic-catalog.json"
        digest = self.put_bytes(name, b" " * (1024**2 + 1))
        self.assertEqual(mixed._pin(self.root, name, digest,
                                    max_bytes=score._reference_cap("catalog")), self.root / name)
        with self.assertRaisesRegex(ValueError, "oversized"):
            mixed._pin(self.root, name, digest,
                       max_bytes=score._reference_cap("historical_train_summary"))

    def test_collection_source_or_training_receipt_tamper_fails_before_zip(self):
        collector = self.root / "scripts/diagnose/collect_dreamerv3_reused_train.py"
        collector.write_bytes(b"# changed collector\n")
        with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.preflight_prior()
        collector.write_bytes(b"# synthetic collector, never invoked\n")
        receipt = self.root / self.train["datasets"]["source1"]["receipt_path"]
        receipt.write_text("{}\n")
        with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.preflight_prior()

    def test_baseline_and_window_mutations_fail_with_recomputed_frozen_sha_before_zip(self):
        name = score.PURE_RESULT_PATH
        frozen = json.loads((self.root / name).read_text())
        window = frozen["strata"]["source0"]["models"][0]["episodes"][0]["windows"][1]
        window["context_anchor_decision"] -= 1
        digest = self.put_json(name, frozen)
        self.score_protocol["pure_256_score_result"]["sha256"] = digest
        with mock.patch.object(score, "PURE_RESULT_SHA256", digest), mock.patch.object(
                original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            with self.assertRaisesRegex(ValueError, "window identity"):
                self.preflight_prior()
        window["context_anchor_decision"] += 1
        window["metrics"]["shifted_repeat_mse"] += 0.001
        digest = self.put_json(name, frozen)
        self.score_protocol["pure_256_score_result"]["sha256"] = digest
        with mock.patch.object(score, "PURE_RESULT_SHA256", digest):
            with self.assertRaisesRegex(ValueError, "episode/window baseline"):
                self.preflight_prior()
        window["metrics"]["shifted_repeat_mse"] -= 0.001
        frozen["gate"]["passed"] = True
        digest = self.put_json(name, frozen)
        self.score_protocol["pure_256_score_result"]["sha256"] = digest
        with mock.patch.object(score, "PURE_RESULT_SHA256", digest):
            with self.assertRaisesRegex(ValueError, "frozen image gate"):
                self.preflight_prior()

    def test_pair_rejects_window_label_or_baseline_mutation(self):
        frozen = self.preflight_prior()["frozen"]["strata"]["random"]["models"][0]
        current = {"learner_seed": 0, "pure_256_checkpoint_sha256": frozen["checkpoint_sha256"],
                   "aggregate": copy.deepcopy(frozen["aggregate"]), "roads": copy.deepcopy(frozen["roads"]),
                   "episodes": copy.deepcopy(frozen["episodes"])}
        current["episodes"][0]["windows"][1]["context_anchor_decision"] -= 1
        with self.assertRaisesRegex(ValueError, "window identity"):
            score._pair(current, frozen)
        current = {"learner_seed": 0, "pure_256_checkpoint_sha256": frozen["checkpoint_sha256"],
                   "aggregate": copy.deepcopy(frozen["aggregate"]), "roads": copy.deepcopy(frozen["roads"]),
                   "episodes": copy.deepcopy(frozen["episodes"])}
        current["aggregate"]["metrics_episode_mean"]["shifted_repeat_mse"] += 0.001
        with self.assertRaisesRegex(ValueError, "baseline differs"):
            score._pair(current, frozen)
        current = {"learner_seed": 0, "pure_256_checkpoint_sha256": frozen["checkpoint_sha256"],
                   "aggregate": copy.deepcopy(frozen["aggregate"]), "roads": copy.deepcopy(frozen["roads"]),
                   "episodes": copy.deepcopy(frozen["episodes"])}
        current["episodes"][0]["windows"][1]["terminal_positive_label_uses"] = 0
        with self.assertRaisesRegex(ValueError, "window or label identity"):
            score._pair(current, frozen)

    def test_path_collision_cgroup_checkpoint_archive_and_zip_bounds(self):
        with mock.patch.object(original, "_load_dataset", side_effect=AssertionError("ZIP opened")):
            self.output.mkdir()
            with self.assertRaises(FileExistsError):
                self.preflight_prior()
            self.output.rmdir()
            with mock.patch.object(original, "_cgroup", return_value={
                    "limit_bytes": 32 * 1024**3, "available_bytes": 1}):
                with self.assertRaisesRegex(ValueError, "cgroup lacks"):
                    self.preflight_prior()
            self.score_protocol["resources"]["max_checkpoint_bytes"] = 1
            with self.assertRaisesRegex(ValueError, "resources"):
                self.preflight_prior()
            self.score_protocol["resources"]["max_checkpoint_bytes"] = self.protocol["resources"]["max_checkpoint_bytes"]
            archive = self.root / self.development["random"]["archive_path"]
            archive.write_bytes(archive.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.preflight_prior()
        row = self.development["source0"]
        resources = dict(self.protocol["resources"], max_uncompressed_bytes=1)
        with self.assertRaisesRegex(ValueError, "ZIP entries"):
            original._dataset_bytes(self.root / row["archive_path"], row["archive_sha256"],
                                    resources["max_archive_bytes"], resources["max_uncompressed_bytes"],
                                    resources["max_decisions_per_archive"])

    def test_path_traversal_and_post_score_input_drift_do_not_create_output(self):
        self.score_protocol["training"][0]["result_path"] = "runs/../protected/score.json"
        with self.assertRaisesRegex(ValueError, "fixed SHA pair"):
            self.preflight_prior()
        self.score_protocol["training"][0]["result_path"] = self.prior_training[0]["result_path"]
        target = self.root / self.development["random"]["receipt_path"]
        original_scorer = original._score_episode
        calls = 0

        def drift(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                target.write_text("{}\n")
            return original_scorer(*args)

        with mock.patch.object(score, "_checkpoint", return_value=_modules(0)), mock.patch.object(
                original, "_score_episode", side_effect=drift):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.score_prior()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
