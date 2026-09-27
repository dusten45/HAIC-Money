"""Synthetic, source-tagged episodes only; never reset a road or read real archives."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter
from dreamer_v3 import DreamerV3Agent, DreamerV3Config, Uint8SequenceReplay
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset, TeacherEpisode
from scripts import train_dreamerv3_reused_multisource as runner


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class TestMultiSourceModelOnly(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roads = (111, 222, 333)
        self.families = ("family-a", "family-b", "family-c")
        self.cells = [{"track_id": 1, "geometry_seed": road} for road in self.roads]
        self.output = "runs/synthetic-multisource/learner-0"
        self.cgroup = {"limit_bytes": 32 * 1024**3, "used_bytes": 4 * 1024**3,
                       "available_bytes": 28 * 1024**3}
        patched_output = patch.object(runner, "OUTPUT_ROOT", "runs/synthetic-multisource")
        patched_output.start()
        self.addCleanup(patched_output.stop)
        for name, value in (("ROADS", self.roads), ("FAMILIES", self.families),
                            ("REPLAY_CAPACITY", 32), ("UPDATES", 2)):
            patched = patch.object(runner, name, value)
            patched.start()
            self.addCleanup(patched.stop)
        for name, value in (("_disk_available", 16 * 1024**3),):
            patched = patch.object(runner, name, return_value=value)
            patched.start()
            self.addCleanup(patched.stop)
        patched = patch.object(runner.single, "_cgroup_memory", return_value=self.cgroup)
        patched.start()
        self.addCleanup(patched.stop)
        (self.root / "experiments").mkdir()
        (self.root / "runs/synthetic-multisource").mkdir(parents=True)
        self.refs = {}
        for key, (name, _) in runner.REFS.items():
            self.refs[key] = [name, "0" * 64]
        self.sources = json.loads(json.dumps(runner.SOURCES))
        for label, row in self.sources.items():
            row["source_actor_sha256"] = ("a" if label == "source0" else "b") * 64
            row["source_checkpoint_sha256"] = ("c" if label == "source0" else "d") * 64
            row["stored_decisions"] = 9
            archive = self.dataset_bytes(row, label)
            self.put_bytes(row["archive_path"], archive)
            row["archive_sha256"] = sha(archive)
        pinned_sources = patch.dict(runner.SOURCES, self.sources, clear=True)
        pinned_sources.start()
        self.addCleanup(pinned_sources.stop)
        pinned_refs = patch.dict(runner.REFS, self.refs, clear=True)
        pinned_refs.start()
        self.addCleanup(pinned_refs.stop)
        r6 = {"study_id": "drqv2-geometry-mix-v1-r6",
              "training_pool": {"partition": "TRAIN", "track_ids": [1, 2, 3, 4],
                                "geometry_seeds": list(self.roads)},
              "diagnostic_pool": {"partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": [444]},
              "catalog_sha256": "0" * 64}
        self.ref("r6_protocol", r6)
        self.ref("catalog", {"format": "haic-drq-training-geometry-catalog-v1", "train": [
            {"geometry_seed": road, "family": family} for road, family in zip(self.roads, self.families)]})
        r6["catalog_sha256"] = self.refs["catalog"][1]
        self.ref("r6_protocol", r6)
        self.ref("historical_train_summary", {
            "format": "haic-drq-training-geometry-diagnostic-summary-v1",
            "catalog_sha256": self.refs["catalog"][1], "geometry": [
                {"geometry_seed": road, "partition": "train", "family": family,
                 "by_source": {"1": {"finished": road != 111}}}
                for road, family in zip(self.roads, self.families)]})
        config = asdict(DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32, num_categoricals=4, num_classes=4,
            batch_size=1, seq_len=4, burnin_steps=1, short_episode_fraction=1.0,
            replay_capacity=8192, imagination_horizon=2, warmup_steps=0))
        self.ref("base_model_protocol", {"learner": {"config": config}})
        self.s0 = {"format": "haic-dreamerv3-reused-train-diagnostic-v1", "purpose": runner.PURPOSE,
                   "study_id": "source0-collection", "r6_protocol": self.reference("r6_protocol"),
                   "cells": self.cells, "episode_schedule": [0, 1, 2], "frame_skip": 4,
                   "max_steps": 2000, "budgets": {"teacher_decision_cap": 24000},
                   "source_actor": self.actor("source0")}
        self.ref("source0_protocol", self.s0)
        self.s1 = {"format": "haic-dreamerv3-reused-train-source1-v1", "purpose": runner.PURPOSE,
                   "study_id": "source1-collection", "r6_protocol": self.reference("r6_protocol"),
                   "cells": self.cells, "episode_schedule": [0, 1, 2], "frame_skip": 4,
                   "max_steps": 2000, "budgets": {"teacher_decision_cap": 24000},
                   "source_actor": self.actor("source1"),
                   "source0_protocol": self.reference("source0_protocol"),
                   "source0_teacher_receipt": {"path": self.sources["source0"]["receipt_path"],
                                               "sha256": "0" * 64},
                   "catalog": self.reference("catalog"),
                   "historical_train_summary": self.reference("historical_train_summary")}
        for label in ("source0", "source1"):
            row = self.sources[label]
            finished = [111, 222] if label == "source0" else [222, 333]
            receipt = {
                "format": "haic-dreamerv3-reused-train-collection-result-v1",
                "status": "completed", "arm": "teacher", "purpose": runner.PURPOSE,
                "study_id": self.s0["study_id"] if label == "source0" else self.s1["study_id"],
                "protocol_sha256": self.refs[f"{label}_protocol"][1],
                "r6_protocol_sha256": self.refs["r6_protocol"][1],
                "source_id": row["source_id"], "source_actor_sha256": row["source_actor_sha256"],
                "source_checkpoint_sha256": row["source_checkpoint_sha256"],
                "archive_sha256": row["archive_sha256"], "dataset_digest": row["dataset_digest"],
                "stored_decisions": 9, "decisions_spent": 9, "dataset_path": "support-dataset.npz",
                "distinct_finished_geometries": finished,
                "allowed_cells": self.cells, "excluded_seeds": [444], "schedule_exhausted": True,
                "schedule_attempts": 3, "complete_episode_count": 3, "partial_decisions": 0,
                "unresolved_decision_calls": 0, "decision_cap": 24000,
                "episode_rows": [
                    {"attempt": i, "episode_id": i, "status": "complete", "complete": True,
                     "terminal": road in finished, "finished": road in finished,
                     "terminated": False, "truncated": True, "decisions": 3,
                     "track_id": 1, "geometry_seed": road}
                    for i, road in enumerate(self.roads)]}
            self.put(row["receipt_path"], receipt)
            row["receipt_sha256"] = self.file_sha(row["receipt_path"])
            if label == "source1":
                self.source1_receipt = receipt
        self.s1["source0_teacher_receipt"]["sha256"] = self.sources["source0"]["receipt_sha256"]
        self.ref("source1_protocol", self.s1)
        source1_receipt_path = self.sources["source1"]["receipt_path"]
        self.source1_receipt["protocol_sha256"] = self.refs["source1_protocol"][1]
        self.put(source1_receipt_path, self.source1_receipt)
        self.sources["source1"]["receipt_sha256"] = self.file_sha(source1_receipt_path)
        self.ref("source1_root_receipt", {
            "protocol_sha256": self.refs["source1_protocol"][1], "status": "completed",
            "catalog": self.reference("catalog"),
            "historical_source1_reference": self.reference("historical_train_summary"),
            "teacher": self.source1_receipt, "union_finished_geometries": list(self.roads),
            "source1_finished_geometries": [222, 333],
            "source0_reference": {"original_learner_gate_passed": False,
                                  "protocol_path": self.refs["source0_protocol"][0],
                                  "protocol_sha256": self.refs["source0_protocol"][1],
                                  "teacher_receipt_path": self.sources["source0"]["receipt_path"],
                                  "teacher_receipt_sha256": self.sources["source0"]["receipt_sha256"],
                                  "finished_geometries": [111, 222]},
            "support_gate": {"passed": True, "authorizes_training": False,
                             "merges_archives": False, "requires_complete_twelve_attempts": True}})
        source_hashes = {}
        for name in runner.SOURCE_PATHS:
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runner.RUNTIME_SOURCES[name], dest)
            source_hashes[name] = self.file_sha(name)
        self.protocol = {"format": runner.FORMAT, "purpose": runner.PURPOSE,
                         "study_id": runner.STUDY_ID,
                         **{key: self.reference(key) for key in runner.REFS},
                         "source_sha256": source_hashes, "cells": self.cells,
                         "datasets": self.sources, "learner": {"seeds": [0, 1],
                                                                  "updates": 2,
                                                                  "config": {**config, "replay_capacity": 32}},
                         "resources": {"max_archive_bytes": 8 * 1024**2,
                                       "max_total_uncompressed_bytes": 8 * 1024**2,
                                       "max_cgroup_memory_bytes": 32 * 1024**3,
                                       "min_cgroup_available_bytes": 8 * 1024**3,
                                       "min_disk_available_bytes": 2 * 1024**3}}
        self.freeze()

    def put_bytes(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put(self, name, value):
        self.put_bytes(name, json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))

    def file_sha(self, name):
        return runner.single._sha256(self.root / name)

    def ref(self, key, value):
        name = self.refs[key][0]
        self.put(name, value)
        self.refs[key][1] = self.file_sha(name)

    def reference(self, key):
        name, digest = self.refs[key]
        return {"path": name, "sha256": digest}

    def actor(self, label):
        row = self.sources[label]
        return {"source_id": row["source_id"], "actor_sha256": row["source_actor_sha256"],
                "checkpoint_sha256": row["source_checkpoint_sha256"]}

    def dataset_bytes(self, row, label):
        episodes = []
        finished = {111, 222} if label == "source0" else {222, 333}
        for i, road in enumerate(self.roads):
            frames = np.stack([np.full((84, 84), (i + 1) * 20 + step + (label == "source1") * 80,
                                       dtype=np.uint8) for step in range(4)])
            action = np.asarray([0.25 if label == "source0" else -0.25, -0.4, 0.15], dtype=np.float32)
            actions = np.stack([action] * 3)
            episodes.append(TeacherEpisode(
                frames=frames, actions=actions, applied_actions=np.stack([
                    ActionAdapter().to_official(action, clip=False) for action in actions]),
                rewards=[0.1, 0.2, 0.3], terminated=[False] * 3,
                truncated=[False, False, True], terminal=[False, False, road in finished],
                finished=[False, False, road in finished], episode_id=i,
                source_id=row["source_id"], source_actor_sha256=row["source_actor_sha256"],
                geometry_id=str(road), track_id=1, complete=True))
        dataset = TeacherDataset(episodes)
        row["dataset_digest"] = dataset.seal()
        return dataset.to_bytes()

    def freeze(self):
        self.put(runner.PROTOCOL_PATH, self.protocol)
        return self.file_sha(runner.PROTOCOL_PATH)

    def checked(self):
        return runner.preflight(runner.PROTOCOL_PATH, self.freeze(), 0, self.output, repo_root=self.root)

    def train(self, seed=0):
        return runner.train(runner.PROTOCOL_PATH, self.freeze(), seed, self.output, repo_root=self.root)

    def test_interleaving_source_lineage_and_terminal_successor_are_exact(self):
        checked = self.checked()
        replay, lineage, audit = runner.materialize(checked)
        self.assertEqual((replay.size, replay.total_steps), (18, 18))
        self.assertEqual([row["road"]["geometry_seed"] for row in lineage],
                         [111, 111, 222, 222, 333, 333])
        self.assertEqual([row["source_id"] for row in lineage],
                         ["drq-source-0", "drq-source-1"] * 3)
        self.assertEqual([(row["sequence_id_start"], row["sequence_id_end"])
                          for row in lineage], [(i * 3, i * 3 + 2) for i in range(6)])
        self.assertEqual([row["original_episode_id"] for row in lineage],
                         ["0", "0", "1", "1", "2", "2"])
        self.assertEqual({row["actor_sha256"] for row in lineage},
                         {source["source_actor_sha256"] for source in self.sources.values()})
        self.assertEqual(audit["source0"]["decisions"], 9)
        self.assertEqual(audit["source1"]["decisions"], 9)
        for i, entry in enumerate(lineage):
            start, end = entry["sequence_id_start"], entry["sequence_id_end"]
            self.assertTrue(replay.is_first[start])
            self.assertTrue(replay.is_last[end])
            self.assertEqual(bool(replay.is_terminal[end]), entry["road"]["geometry_seed"] in (
                (111, 222) if i % 2 == 0 else (222, 333)))
            self.assertEqual(replay._boundary_observations[end][-1, 0, 0],
                             replay.observations[end, -1, 0, 0] + 1)
            self.assertEqual(float(replay.actions[start, 0]), 0.25 if i % 2 == 0 else -0.25)
        batch = replay.sample_sequence(2, 4, short_episode_fraction=1.0)
        self.assertEqual(batch["actions"].shape, (2, 3, 3))
        self.assertTrue(batch["is_last"][:, -1].all().item())
        for ids in batch["sequence_ids"].tolist():
            self.assertEqual(ids[-1] // 3, ids[0] // 3)
        with patch.object(Uint8SequenceReplay, "sample_sequence",
                          return_value={"sequence_ids": torch.tensor([[2, 3]])}):
            with self.assertRaisesRegex(ValueError, "crosses source/episode"):
                replay.sample_sequence(1, 2)

    def test_model_only_two_updates_no_environment_or_policy_calls_and_sha_sidecar(self):
        updates = []
        real = DreamerV3Agent.update

        def spy(agent, *, model_only=False):
            updates.append(model_only)
            return real(agent, model_only=model_only)

        with patch.object(DreamerV3Agent, "update", spy), \
             patch.object(DreamerV3Agent, "observe", side_effect=AssertionError("environment observe")), \
             patch.object(DreamerV3Agent, "reset_episode", side_effect=AssertionError("environment reset")), \
             patch.object(DreamerV3Agent, "act", side_effect=AssertionError("policy act")):
            result = self.train()
        self.assertEqual(updates, [True, True])
        self.assertEqual(result["model_only_updates"], 2)
        self.assertEqual(result["environment_steps"], 0)
        self.assertEqual(result["distinct_training_roads"], 3)
        self.assertEqual(result["episodes"], 6)
        self.assertFalse(result["promotion_eligible"])
        output = self.root / self.output
        self.assertEqual(self.file_sha(result["lineage_path"]), result["lineage_sha256"])
        lineage = json.loads((self.root / result["lineage_path"]).read_text())
        self.assertEqual(lineage["range_convention"], "inclusive")
        self.assertEqual(lineage["transition_count"], 18)
        self.assertEqual(len(lineage["episodes"]), 6)
        checkpoint = torch.load(output / result["checkpoint_path"], map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["run_metadata"]["lineage_sha256"], result["lineage_sha256"])
        self.assertEqual(checkpoint["run_metadata"]["lineage_path"], result["lineage_path"])
        self.assertEqual(checkpoint["gradient_steps"], 2)
        self.assertEqual(checkpoint["environment_steps"], 0)
        self.assertEqual(checkpoint["trainer_state"], None)
        self.assertEqual(checkpoint["actor_optimizer"]["state"], {})
        self.assertEqual(checkpoint["critic_optimizer"]["state"], {})
        self.assertEqual([json.loads(line)["update"] for line in
                          (output / "update-metrics.jsonl").read_text().splitlines()], [1, 2])

    def test_protocol_fixed_seeds_budget_capacity_config_and_source_pair(self):
        for change, expected in (
            (lambda: self.protocol["learner"].update(seeds=[0]), "exact two learner"),
            (lambda: self.protocol["learner"].update(updates=3), "exact two learner"),
            (lambda: self.protocol["learner"]["config"].update(embed_dim=512), "config"),
            (lambda: self.protocol["learner"]["config"].update(replay_capacity=16), "config"),
            (lambda: self.protocol["datasets"]["source1"].update(source_id="drq-source-0"), "pairs"),
            (lambda: self.protocol["datasets"].update(random={}), "pairs"),
        ):
            with self.subTest(expected=expected):
                saved = json.loads(json.dumps(self.protocol))
                change()
                with patch.object(runner.single, "_archive_bytes", side_effect=AssertionError("archive load")):
                    with self.assertRaisesRegex(ValueError, expected):
                        self.checked()
                self.assertFalse((self.root / self.output).exists())
                self.protocol = saved

    def test_combined_capacity_rejected_even_when_each_archive_fits(self):
        checked = self.checked()
        checked["config"].replay_capacity = 16
        with self.assertRaisesRegex(ValueError, "silently overwrite"):
            runner.materialize(checked)

    def test_source_drift_rejected_by_own_bridge_before_join(self):
        checked = self.checked()
        item = checked["checked"]["source1"]
        row = item["row"]
        false_row = dict(row, source_id="drq-source-0")
        raw = self.dataset_bytes(false_row, "source1")
        self.put_bytes(row["archive_path"], raw)
        row["archive_sha256"] = sha(raw)
        row["dataset_digest"] = false_row["dataset_digest"]
        item["size"] = len(raw)
        with self.assertRaisesRegex(ValueError, "source actor"):
            runner.materialize(checked)

    def test_archive_sha_and_resource_failure_produce_abort_receipt(self):
        checked = self.checked()
        archive_path = self.root / self.sources["source1"]["archive_path"]
        with archive_path.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "archive size|archive SHA-256"):
            self.train()
        abort = json.loads((self.root / self.output / "abort.json").read_text())
        self.assertEqual(abort["completed_model_only_updates"], 0)
        self.assertEqual(abort["phase"], "archive verification")
        self.assertFalse((self.root / self.output / "world-model-checkpoint.pt").exists())

    def test_preflight_rejects_resource_pins_before_outputs(self):
        for change, expected in (
            (lambda: self.cgroup.update(available_bytes=1), "cgroup memory"),
            (lambda: self.protocol["resources"].update(max_archive_bytes=1), "archive path/size"),
            (lambda: self.protocol["resources"].update(min_disk_available_bytes=17 * 1024**3), "disk headroom"),
            (lambda: self.protocol["resources"].update(min_cgroup_available_bytes=1), "floors"),
        ):
            with self.subTest(expected=expected):
                saved = json.loads(json.dumps(self.protocol))
                cgroup = self.cgroup.copy()
                change()
                with patch.object(runner.single, "_archive_bytes", side_effect=AssertionError("archive read")):
                    with self.assertRaisesRegex(ValueError, expected):
                        self.checked()
                self.assertFalse((self.root / self.output).exists())
                self.protocol = saved
                self.cgroup.clear()
                self.cgroup.update(cgroup)

    def test_train_membership_partial_receipt_and_path_exclusion(self):
        self.protocol["cells"][0]["geometry_seed"] = 444
        with self.assertRaisesRegex(ValueError, "twelve fixed TRAIN"):
            self.checked()
        self.protocol["cells"][0]["geometry_seed"] = 111
        receipt_name = self.sources["source0"]["receipt_path"]
        receipt = json.loads((self.root / receipt_name).read_text())
        receipt["partial_decisions"] = 1
        self.put(receipt_name, receipt)
        self.sources["source0"]["receipt_sha256"] = self.file_sha(receipt_name)
        self.protocol["datasets"] = self.sources
        self.s1["source0_teacher_receipt"]["sha256"] = self.sources["source0"]["receipt_sha256"]
        self.ref("source1_protocol", self.s1)
        self.protocol["source1_protocol"] = self.reference("source1_protocol")
        with self.assertRaisesRegex(ValueError, "collection receipt"):
            self.checked()
        self.assertFalse((self.root / self.output).exists())
        self.output = "runs/synthetic-multisource/confirm-seed-0"
        with self.assertRaisesRegex(ValueError, "fixed TRAIN-only"):
            self.checked()

    def test_symlink_archive_and_catalog_drift_rejected_without_output(self):
        archive = self.root / self.sources["source0"]["archive_path"]
        target = archive.with_name("sealed-copy.npz")
        archive.rename(target)
        archive.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink path"):
            self.checked()
        self.assertFalse((self.root / self.output).exists())
        archive.unlink()
        target.rename(archive)
        with (self.root / self.refs["catalog"][0]).open("ab") as stream:
            stream.write(b"drift")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.checked()
        self.assertFalse((self.root / self.output).exists())

    def test_uncompressed_cap_and_wrong_road_abort_before_updates(self):
        checked = self.checked()
        checked["uncompressed_cap"] = 1
        with self.assertRaisesRegex(ValueError, "uncompressed archive"):
            runner.materialize(checked)
        checked = self.checked()
        source = checked["checked"]["source1"]
        original = source["receipt"]["episode_rows"][0]["geometry_seed"]
        source["receipt"]["episode_rows"][0]["geometry_seed"] = 999
        with self.assertRaisesRegex(ValueError, "road/terminal/receipt drift"):
            runner.materialize(checked)
        source["receipt"]["episode_rows"][0]["geometry_seed"] = original

    def test_optimizer_mutation_and_mid_update_resource_drop_keep_abort(self):
        actual = DreamerV3Agent.update

        def mutate_optimizer(agent, *, model_only=False):
            metrics = actual(agent, model_only=model_only)
            agent.actor_optimizer.param_groups[0]["lr"] = 0.7
            return metrics

        with patch.object(DreamerV3Agent, "update", mutate_optimizer):
            with self.assertRaisesRegex(ValueError, "optimizer or parameters"):
                self.train()
        self.assertEqual(json.loads((self.root / self.output / "abort.json").read_text())[
            "completed_model_only_updates"], 0)
        self.output = "runs/synthetic-multisource/learner-1"
        low = {**self.cgroup, "available_bytes": 1}
        with patch.object(runner.single, "_cgroup_memory",
                          side_effect=[self.cgroup, self.cgroup, low]):
            with self.assertRaisesRegex(ValueError, "training floor"):
                self.train(seed=1)
        self.assertEqual(json.loads((self.root / self.output / "abort.json").read_text())[
            "completed_model_only_updates"], 0)

    def test_nonfinite_or_actor_mutation_abort_without_success_receipt(self):
        actual = DreamerV3Agent.update

        def bad(agent, *, model_only=False):
            metrics = actual(agent, model_only=model_only)
            metrics["loss_wm"] = float("nan")
            return metrics

        with patch.object(DreamerV3Agent, "update", bad):
            with self.assertRaisesRegex(ValueError, "model-only metrics"):
                self.train()
        self.assertFalse((self.root / self.output / "training-result.json").exists())
        self.assertEqual(json.loads((self.root / self.output / "abort.json").read_text())[
            "completed_model_only_updates"], 0)
        self.output = "runs/synthetic-multisource/learner-1"

        def mutate(agent, *, model_only=False):
            metrics = actual(agent, model_only=model_only)
            with torch.no_grad():
                next(agent.critic_target.parameters()).add_(0.01)
            return metrics

        with patch.object(DreamerV3Agent, "update", mutate):
            with self.assertRaisesRegex(ValueError, "optimizer or parameters"):
                self.train(seed=1)
        self.assertFalse((self.root / self.output / "training-result.json").exists())

    def test_output_collision_before_archive_read(self):
        (self.root / self.output).mkdir()
        with patch.object(runner.single, "_archive_bytes", side_effect=AssertionError("archive loaded")):
            with self.assertRaises(FileExistsError):
                self.train()


if __name__ == "__main__":
    unittest.main()
