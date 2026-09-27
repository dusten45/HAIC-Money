"""Synthetic final-source DrQ evaluator contracts; no real actor or road is run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter, ObservationSpec
from scripts import diagnose_drq_final_replay as diagnostic
from scripts import diagnose_drq_geometry_mix as r6
from scripts import diagnose_drq_retention_r7 as r7
from tests.test_drq_geometry_mix_diagnostics import FakeActor, FakeEnvironment


def _json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class FinalReplayDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "runs").mkdir()
        (self.root / "experiments").mkdir()
        (self.root / diagnostic.RUN_ROOT).mkdir()
        FakeEnvironment.instances = []
        FakeEnvironment.reset_count = 0
        FakeEnvironment.diverge_on_repeat = False
        self.rows = [{"geometry_seed": seed, "family": r6.EXPECTED_FAMILIES[index % 6],
                      "road_coordinate_sha256": r6._road_hash(FakeEnvironment.track_for_seed(seed))}
                     for index, seed in enumerate(r6.DIAGNOSTIC_SEEDS)]
        self.source_by_seed = {seed: {
            "actor_sha256": hashlib.sha256(f"source-actor-{seed}".encode()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(f"source-checkpoint-{seed}".encode()).hexdigest(),
            "checkpoint_path": self.root / "runs" / f"source-{seed}" / "checkpoint.pt",
        } for seed in (0, 1)}
        self.sources = {}
        self.controls = {}
        for seed in (0, 1):
            for index, row in enumerate(self.rows):
                road = row["geometry_seed"]
                source = self.source_by_seed[seed]
                self.sources[(seed, road)] = {
                    "actor_sha256": source["actor_sha256"],
                    "checkpoint_sha256": source["checkpoint_sha256"],
                    "finished": index < (6 if seed == 0 else 5),
                    "trace_sha256": hashlib.sha256(f"source-{seed}-{road}".encode()).hexdigest(),
                }
                for variant in r6.VARIANTS:
                    self.controls[(seed, road, variant)] = {
                        "finished": index in (0, 6), "family": row["family"],
                        "trace_sha256": hashlib.sha256(f"control-{seed}-{road}-{variant}".encode()).hexdigest(),
                    }

    def _protocol(self):
        return {
            "format": diagnostic.FORMAT, "study_id": diagnostic.STUDY_ID,
            "run_root": diagnostic.RUN_ROOT.as_posix(),
            "r6_protocol_path": r6.PROTOCOL_PATH.as_posix(),
            "r6_protocol_sha256": r7.R6_PROTOCOL_SHA,
            "r7_protocol_path": r7.PROTOCOL.as_posix(),
            "r7_protocol_sha256": diagnostic.R7_PROTOCOL_SHA,
            "r6_manifest_sha256": r7.R6_MANIFEST_SHA,
            "r7_manifest_sha256": diagnostic.R7_MANIFEST_SHA,
            "lambda_preserve": .5,
            "catalog_sha256": r6.CATALOG_SHA256,
            "diagnostic_cache": {"sha256": "e" * 64},
            "replay_contract": {"batch_size": 64, "source_rows": 32, "online_rows": 32,
                                "source_kind": "final_source_policy"},
            "runs": [{"source_seed": seed, "variant": variant,
                      "condition": diagnostic.CONDITION,
                      "run_dir": (diagnostic.RUN_ROOT / f"learner-{seed}-{variant}-{diagnostic.CONDITION}").as_posix(),
                      "rng_seeds": {"actor_rng_seed": 104000 + seed}}
                     for seed, variant in sorted(diagnostic.ROLES)],
        }

    @staticmethod
    def _actor_loader(path, *, device):
        if device != "cpu":
            raise AssertionError("actor must load on CPU")
        state = torch.load(path, map_location="cpu", weights_only=False)
        return FakeActor(float(state["state_dict"]["policy.weight"][0])), ActionAdapter(), ObservationSpec()

    def _roles(self):
        roles = {}
        for index, (seed, variant) in enumerate(sorted(diagnostic.ROLES)):
            name = f"{variant}-{diagnostic.CONDITION}-seed{seed}"
            source = self.source_by_seed[seed]
            roles[name] = {
                "role": f"{variant}-{diagnostic.CONDITION}", "arm": f"{variant}-{diagnostic.CONDITION}",
                "variant": variant, "condition": diagnostic.CONDITION,
                "source_learner_seed": seed, "source_actor_sha256": source["actor_sha256"],
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "source_replay_sha256": "1" * 64, "pool_receipt_sha256": "2" * 64,
                "actor_path": self.root / "runs" / name / "actor.pt",
                "actor_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "actor_weights_sha256": "3" * 64,
                "checkpoint_path": self.root / "runs" / name / "checkpoint.pt",
                "checkpoint_sha256": "4" * 64,
                "checkpoint_manifest_path": self.root / "runs" / name / "checkpoint.manifest.json",
                "checkpoint_manifest_sha256": "5" * 64, "checkpoint_online_step": 32768,
                "catalog_path": self.root / "runs" / name / "checkpoint-catalog.json",
                "catalog_sha256": "6" * 64,
                "result_path": self.root / "runs" / name / "result.json", "result_sha256": "7" * 64,
                "sample_trace_sha256": "8" * 64,
                "actor": FakeActor(-0.7 + 0.08 * index),
                "action_adapter": ActionAdapter(), "observation_spec": ObservationSpec(),
            }
        return roles

    def test_exact_new_grid_rejects_old_paths_ratio_and_duplicate_roles(self):
        protocol = self._protocol()
        self.assertEqual(set(diagnostic._grid(protocol)), diagnostic.ROLES)
        protocol["runs"][-1] = dict(protocol["runs"][0])
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "duplicate"):
            diagnostic._grid(protocol)
        protocol = self._protocol()
        protocol["runs"][0]["run_dir"] = "runs/20260926-drqv2-retention-r7/learner-0-uniform-r7b"
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "new root"):
            diagnostic._grid(protocol)
        protocol = self._protocol()
        protocol["replay_contract"]["source_rows"] = 31
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "32:32"):
            diagnostic._grid(protocol)
        protocol = self._protocol()
        protocol["lambda_preserve"] = .25
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "weight"):
            diagnostic._grid(protocol)

    def test_study_preflight_requires_original_r7b_rng_cache_and_evaluator_source_hashes(self):
        protocol = self._protocol()
        code = {}
        for name in ("scripts/diagnose_drq_final_replay.py",
                     "scripts/diagnose_drq_geometry_mix.py",
                     "scripts/diagnose_drq_retention_r7.py",
                     "scripts/audit_drq_final_replay_samples.py"):
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(name.encode())
            code[name] = r6.sha256_file(path)
        protocol["code_sha256"] = code
        old = {"format": r7.FORMAT, "study_id": "drqv2-retention-r7",
               "r6_protocol_sha256": r7.R6_PROTOCOL_SHA,
               "diagnostic_manifest_sha256": r7.R6_MANIFEST_SHA,
               "lambda_preserve": .5, "diagnostic_cache": protocol["diagnostic_cache"],
               "runs": [{**row, "condition": "r7b", "rng_seeds": dict(row["rng_seeds"])}
                        for row in protocol["runs"]]}
        frozen = {"format": r6.PROTOCOL_FORMAT,
                  "diagnostics": {"geometry_seeds": list(r6.DIAGNOSTIC_SEEDS)}}
        path = self.root / "experiments/study.json"
        original_pinned = diagnostic._read_pinned

        def pinned(root, relative, digest, label):
            if relative == r7.PROTOCOL.as_posix():
                return old
            if relative == r6.PROTOCOL_PATH.as_posix():
                return frozen
            return original_pinned(root, relative, digest, label)

        with patch.object(diagnostic, "_read_pinned", side_effect=pinned), patch.object(
            r6, "_validate_study_contract",
        ):
            _json(path, protocol)
            self.assertEqual(diagnostic._protocol(self.root, path)[0], protocol)
            protocol["runs"][0]["rng_seeds"]["actor_rng_seed"] = 99
            _json(path, protocol)
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "r7b source RNG"):
                diagnostic._protocol(self.root, path)
            protocol["runs"][0]["rng_seeds"]["actor_rng_seed"] = 104000
            _json(path, protocol)
            (self.root / "scripts/diagnose_drq_final_replay.py").write_bytes(b"changed evaluator")
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "pinned executable source changed"):
                diagnostic._protocol(self.root, path)

    def test_pool_receipt_binds_new_sha_source_and_excludes_diagnostics(self):
        protocol = self._protocol()
        protocol["source_replay"] = {}
        old_replay = {}
        collection = {"format": "haic-drq-final-source-collection-v1",
                      "study_id": diagnostic.STUDY_ID, "run_root": diagnostic.RUN_ROOT.as_posix(),
                      "r6_protocol_sha256": r7.R6_PROTOCOL_SHA,
                      "r7_protocol_sha256": diagnostic.R7_PROTOCOL_SHA,
                      "catalog_sha256": r6.CATALOG_SHA256, "collection_noise_std": .05,
                      "sources": {}}
        for seed in (0, 1):
            source = self.source_by_seed[seed]
            pool = self.root / "runs" / f"pool-{seed}" / "replay.pt"
            pool.parent.mkdir()
            pool.write_bytes(f"sealed final source pool {seed}".encode())
            pool_sha = r6.sha256_file(pool)
            old_replay[str(seed)] = {"episode_ledger_sha256": hashlib.sha256(
                f"original-ledger-{seed}".encode()).hexdigest()}
            collection["sources"][str(seed)] = {
                "schedule_sha256": "9" * 64,
                "original_ledger_sha256": old_replay[str(seed)]["episode_ledger_sha256"],
                "source_actor_sha256": source["actor_sha256"],
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "collection_rng_seeds": {"action_noise_seed": 9000 + seed},
            }
        for seed in (0, 1):
            source = self.source_by_seed[seed]
            pool = self.root / "runs" / f"pool-{seed}" / "replay.pt"
            pool_sha = r6.sha256_file(pool)
            collection_sha = _json(pool.parent / "collection_protocol.json", collection)
            receipt = pool.parent / "result.json"
            receipt_sha = _json(receipt, {
                "format": "haic-drq-final-source-pool-v1",
                "completed": True, "source_seed": seed,
                "source_actor_sha256": source["actor_sha256"],
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "pool_path": pool.relative_to(self.root).as_posix(),
                "pool_sha256": pool_sha, "partition": "TRAIN",
                "excluded_diagnostic_roads": True, "decisions": 100000,
                "capacity": 100000, "valid_n_step_starts": 99990,
                "catalog_sha256": r6.CATALOG_SHA256,
                "schedule_sha256": "9" * 64,
                "original_ledger_sha256": old_replay[str(seed)]["episode_ledger_sha256"],
                "collection_protocol_sha256": collection_sha, "collection_noise_std": .05,
                "collection_rng_seeds": collection["sources"][str(seed)]["collection_rng_seeds"],
            })
            protocol["source_replay"][str(seed)] = {
                "pool_path": pool.relative_to(self.root).as_posix(), "pool_sha256": pool_sha,
                "receipt_path": receipt.relative_to(self.root).as_posix(),
                "receipt_sha256": receipt_sha,
            }
        original_pinned = diagnostic._read_pinned

        def pinned(root, relative, digest, label):
            if relative == r7.PROTOCOL.as_posix():
                return {"source_replay": old_replay}
            return original_pinned(root, relative, digest, label)

        with patch.object(diagnostic, "_read_pinned", side_effect=pinned), patch.object(
            diagnostic, "_verify_pool_ledgers",
        ) as validate_ledgers:
            pools = diagnostic._pools(self.root, protocol, self.source_by_seed,
                                      {row["geometry_seed"] for row in self.rows})
            self.assertEqual(set(pools), {0, 1})
            self.assertEqual(validate_ledgers.call_count, 2)
            pool_path = self.root / pools[0]["pool_path"]
            pool_path.write_bytes(b"tampered pool")
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "SHA-256 mismatch"):
                diagnostic._pools(self.root, protocol, self.source_by_seed, set())
            pool_path.write_bytes(b"sealed final source pool 0")
            receipt_path = self.root / pools[0]["receipt_path"]
            receipt = json.loads(receipt_path.read_text())
            receipt["original_ledger_sha256"] = "f" * 64
            protocol["source_replay"]["0"]["receipt_sha256"] = _json(receipt_path, receipt)
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "schedule/original"):
                diagnostic._pools(self.root, protocol, self.source_by_seed,
                                  {self.rows[0]["geometry_seed"]})

    def test_pool_ledgers_verify_original_reset_pair_and_every_step(self):
        original = self.root / "runs/original/episodes.jsonl"
        original.parent.mkdir()
        original.write_text("".join(json.dumps(row) + "\n" for row in (
            {"event": "reset", "episode_id": 0, "track_id": 1, "seed": 23},
            {"event": "reset", "episode_id": 1, "track_id": 2, "seed": 47},
        )))
        output = self.root / "runs/collection"
        output.mkdir()
        episode_path = output / "episodes.jsonl"
        episodes = [
            {"event": "reset", "episode_id": 0, "original_episode_id": 1,
             "source_seed": 0, "schedule_index": 0, "track_id": 2, "seed": 47,
             "partition": "TRAIN"},
            {"event": "end", "episode_id": 0, "original_episode_id": 1,
             "source_seed": 0, "schedule_index": 0, "track_id": 2, "seed": 47},
        ]
        episode_path.write_text("".join(json.dumps(row) + "\n" for row in episodes))
        step_path = output / "steps.jsonl"
        steps = [{"decision": i + 1, "sequence_id": i, "episode_id": 0,
                  "original_episode_id": 1, "schedule_index": 0, "source_seed": 0,
                  "track_id": 2, "geometry_seed": 47, "partition": "TRAIN"} for i in range(3)]
        step_path.write_text("".join(json.dumps(row) + "\n" for row in steps))
        receipt = {"episode_ledger_path": episode_path.relative_to(self.root).as_posix(),
                   "episode_ledger_sha256": r6.sha256_file(episode_path),
                   "step_ledger_path": step_path.relative_to(self.root).as_posix(),
                   "step_ledger_sha256": r6.sha256_file(step_path),
                   "geometry_seeds": [47],
                   "scheduled_episodes_consumed": 1}
        old = {"episode_ledger_path": original.relative_to(self.root).as_posix(),
               "episode_ledger_sha256": r6.sha256_file(original)}
        diagnostic._verify_pool_ledgers(self.root, receipt, old, 0, {99}, decisions=3)
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "TRAIN-DIAGNOSTIC|original source TRAIN"):
            diagnostic._verify_pool_ledgers(self.root, receipt, old, 0, {47}, decisions=3)
        steps[1]["geometry_seed"] = 99
        step_path.write_text("".join(json.dumps(row) + "\n" for row in steps))
        receipt["step_ledger_sha256"] = r6.sha256_file(step_path)
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "step 2"):
            diagnostic._verify_pool_ledgers(self.root, receipt, old, 0, set(), decisions=3)

    def test_independent_semantic_sample_receipt_required_before_actor_or_reset(self):
        pools = {seed: {"pool_sha256": hashlib.sha256(f"pool{seed}".encode()).hexdigest(),
                        "receipt_sha256": hashlib.sha256(f"receipt{seed}".encode()).hexdigest()}
                 for seed in (0, 1)}
        records = {key: {"result_sha256": hashlib.sha256(f"result{key}".encode()).hexdigest(),
                         "checkpoint_sha256": hashlib.sha256(f"checkpoint{key}".encode()).hexdigest(),
                         "sample_trace_sha256": hashlib.sha256(f"trace{key}".encode()).hexdigest()}
                   for key in diagnostic.ROLES}
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "missing"):
            diagnostic._sample_audit(self.root, "f" * 64, pools, records)
        receipt = {"format": "haic-drq-final-source-sample-audit-v1", "passed": True,
                   "no_environment_resets": True, "protocol_sha256": "f" * 64,
                   "source_pool": {str(seed): {**pool, "verified_source_decisions": 100000,
                                               "valid_n_step_starts": 99990}
                                   for seed, pool in pools.items()},
                   "runs": [{"source_seed": seed, "variant": variant,
                             "run_dir": (diagnostic.RUN_ROOT / f"learner-{seed}-{variant}-final_source").as_posix(),
                             "result_sha256": record["result_sha256"],
                             "final_checkpoint_sha256": record["checkpoint_sha256"],
                             "final_sample_trace_sha256": record["sample_trace_sha256"],
                             "updates": 22768, "batch_size": 64,
                             "source_rows": 728576, "online_rows": 728576}
                            for (seed, variant), record in sorted(records.items())]}
        path = self.root / diagnostic.SAMPLE_AUDIT
        expected_sha = _json(path, receipt)
        self.assertEqual(diagnostic._sample_audit(self.root, "f" * 64, pools, records), expected_sha)
        receipt["runs"][0]["final_sample_trace_sha256"] = "0" * 64
        _json(path, receipt)
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "semantic sample audit differs"):
            diagnostic._sample_audit(self.root, "f" * 64, pools, records)
        self.assertEqual(FakeEnvironment.reset_count, 0)

    def test_completed_treatment_checks_result_catalog_manifest_and_sampling(self):
        protocol = self._protocol()
        key = (0, "uniform")
        directory = diagnostic.RUN_ROOT / "learner-0-uniform-final_source"
        dest = self.root / directory
        dest.mkdir()
        protocol_sha = _json(self.root / "experiments/study.json", protocol)
        (dest / "study_protocol.json").write_bytes((self.root / "experiments/study.json").read_bytes())
        pool = {"pool_sha256": "a" * 64, "receipt_sha256": "b" * 64}
        source = self.source_by_seed[0]
        _json(dest / "run-config.json", {
            "format": "haic-drq-retention-run-config-v1", "protocol_sha256": protocol_sha,
            "source_seed": key[0], "variant": key[1], "condition": diagnostic.CONDITION,
            "source_actor_sha256": source["actor_sha256"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_replay_sha256": pool["pool_sha256"],
            "source_pool_receipt_sha256": pool["receipt_sha256"],
            "source_rows_per_batch": 32, "online_rows_per_batch": 32,
            "lambda_preserve": .5, "encoder_update": True,
            "diagnostic_cache_sha256": protocol["diagnostic_cache"]["sha256"],
            "optimizer": "Adam", "actor_lr": 1e-4, "critic_lr": 1e-4,
            "rng_seeds": next(run["rng_seeds"] for run in protocol["runs"]
                              if (run["source_seed"], run["variant"]) == key),
        })
        (dest / "step-metrics.jsonl").write_bytes(b"synthetic online step ledger")
        (dest / "drift.jsonl").write_bytes(b"synthetic drift ledger")
        candidates = []
        for step in (16384, 32768):
            folder = dest / "checkpoints" / f"step-{step:09d}"
            folder.mkdir(parents=True)
            actor = folder / "actor.pt"
            torch.save({"format": "haic-drq-v2-actor-v1",
                        "config": {"augmentation_pad": 4, "steering_logit_l2": 0.0},
                        "state_dict": FakeActor(0.1 + step / 327680).state_dict()}, actor)
            checkpoint = folder / "checkpoint.pt"
            checkpoint.write_bytes(f"{step} checkpoint".encode())
            _json(folder / "checkpoint.manifest.json", {
                "algorithm": "drq-v2", "extra": {"environment_steps": 131072 + step,
                "study_id": diagnostic.STUDY_ID, "study_protocol_sha256": protocol_sha,
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "source_replay_sha256": pool["pool_sha256"],
                "catalog_sha256": r6.CATALOG_SHA256},
            })
            trace = dest / f"replay-sample-trace-step-{step:09d}.npz"
            rows = step - 10000
            tags = np.broadcast_to(np.asarray([0] * 32 + [1] * 32, dtype=np.uint8), (rows, 64))
            indices = np.broadcast_to(np.arange(64, dtype=np.int64), (rows, 64))
            np.savez_compressed(trace, source=tags, source_indices=indices,
                                episode_id=np.zeros((rows, 64), dtype=np.int64),
                                protocol_sha256=np.frombuffer(protocol_sha.encode("ascii"), dtype=np.uint8))
            candidates.append({
                "checkpoint_online_step": step, "study_gradient_steps": rows,
                "actor_path": actor.relative_to(self.root).as_posix(),
                "actor_sha256": r6.sha256_file(actor),
                "checkpoint_path": checkpoint.relative_to(self.root).as_posix(),
                "checkpoint_sha256": r6.sha256_file(checkpoint),
                "sample_trace_path": trace.relative_to(self.root).as_posix(),
                "sample_trace_sha256": r6.sha256_file(trace),
            })
        _json(dest / "checkpoint-catalog.json", {
            "format": r7.CATALOG_FORMAT, "study_protocol_sha256": protocol_sha,
            "source_seed": 0, "variant": "uniform", "condition": diagnostic.CONDITION,
            "candidates": candidates,
        })
        result = {"format": r7.RESULT_FORMAT, "study_id": diagnostic.STUDY_ID,
                  "completed": True, "study_protocol_sha256": protocol_sha,
                  "source_seed": 0, "variant": "uniform", "condition": diagnostic.CONDITION,
                  "source_actor_sha256": source["actor_sha256"],
                  "source_checkpoint_sha256": source["checkpoint_sha256"],
                  "source_replay_sha256": pool["pool_sha256"],
                  "additional_online_steps": 32768, "study_gradient_steps": 22768,
                  "source_samples": 728576, "online_samples": 728576,
                  "online_replay_manifest_sha256": r6.sha256_file(dest / "step-metrics.jsonl"),
                  "diagnostic_trace_sha256": r6.sha256_file(dest / "drift.jsonl"),
                  "candidates": candidates}
        result_path = dest / "result.json"
        _json(result_path, result)
        with patch.object(r7, "_verify_step_ledger"):
            record = diagnostic._candidate(self.root, directory, key, protocol, protocol_sha,
                                           source, pool, {123})
            self.assertEqual(record["checkpoint_online_step"], 32768)
            r6._validate_actor(record["actor_path"], record["actor_sha256"],
                               record["actor_weights_sha256"], loader=self._actor_loader)
            for field, invalid in (("completed", False), ("condition", "r7b"),
                                   ("study_gradient_steps", 22767),
                                   ("source_replay_sha256", source["checkpoint_sha256"])):
                with self.subTest(field=field):
                    _json(result_path, {**result, field: invalid})
                    with self.assertRaisesRegex(diagnostic.DiagnosticError, "full-budget"):
                        diagnostic._candidate(self.root, directory, key, protocol, protocol_sha,
                                              source, pool, {123})
            _json(result_path, result)
            (dest / "checkpoints/step-000032768/checkpoint.pt").write_bytes(b"tampered checkpoint")
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "SHA-256 mismatch"):
                diagnostic._candidate(self.root, directory, key, protocol, protocol_sha,
                                      source, pool, {123})

    def test_paired_gate_kept_and_gained_and_distinct_roads(self):
        episodes = []
        for seed, variant in sorted(diagnostic.ROLES):
            for index, row in enumerate(self.rows):
                road = row["geometry_seed"]
                # Nine old wins retained, two source failures gained; same two
                # road IDs repeat across actor seeds, not independent roads.
                new_finish = index < (6 if seed == 0 else 3) or index in (6, 7)
                for repeat in (0, 1):
                    episodes.append({
                        "source_learner_seed": seed, "geometry_seed": road,
                        "variant": variant, "condition": diagnostic.CONDITION,
                        "repeat": repeat, "track_id": 1, "family": row["family"],
                        "finished": new_finish,
                        "source_actor_sha256": self.source_by_seed[seed]["actor_sha256"],
                        "source_checkpoint_sha256": self.source_by_seed[seed]["checkpoint_sha256"],
                        "trace_sha256": "a" * 64,
                        "trace_deterministic_sha256": "b" * 64,
                        "raw_reward_sum": 1.0, "max_progress": .5,
                        "repeat_trace_agrees_with_repeat0": True,
                    })
        summary, paired = diagnostic._paired_summary(episodes, self.rows, self.sources,
                                                      self.controls, {"protocol_sha256": "f" * 64})
        self.assertEqual((summary["observed_episode_count"], len(paired["pairs"])), (192, 96))
        groups = [row for row in summary["paired_source_and_r7b"]
                  if row["source_seed"] is None and row["family"] is None]
        self.assertEqual(len(groups), 3)
        for group in groups:
            self.assertEqual((group["paired_actor_road_cells"], group["source_success_cells"],
                              group["source_failure_cells"], group["distinct_reused_road_count"]),
                             (32, 11, 21, 16))
            self.assertEqual(group["source_to_treatment"], {
                "both_finished": 9, "lost": 2, "gained": 4, "both_failed": 17,
            })
            self.assertEqual(group["retention_cells"], {"K": 9, "L": 2, "G": 4, "neither": 17})
            self.assertEqual(group["distinct_gained_roads"], 2)
            self.assertTrue(group["fixed_gate_pass"])
        self.assertTrue(summary["any_treatment_passes_fixed_gate"])
        episodes.pop()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "192 episodes"):
            diagnostic._paired_summary(episodes, self.rows, self.sources, self.controls, {})

    def test_trace_file_hash_and_deterministic_digest_both_required(self):
        directory = diagnostic.RUN_ROOT
        trace = self.root / directory / "traces" / "seed-1" / "repeat-0.npz"
        trace.parent.mkdir(parents=True)
        arrays = {name: np.zeros((1,), dtype=np.float32) for name in r6.TRACE_ARRAYS}
        np.savez_compressed(trace, **arrays)
        relative = trace.relative_to(self.root / directory).as_posix()
        digest = r6._trace_digest(arrays)
        file_sha = r6.sha256_file(trace)
        diagnostic._trace(self.root, directory, {"trace_path": relative,
                           "trace_sha256": file_sha, "trace_deterministic_sha256": digest},
                          {relative: file_sha})
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "digest"):
            diagnostic._trace(self.root, directory, {"trace_path": relative,
                               "trace_sha256": file_sha, "trace_deterministic_sha256": "f" * 64},
                              {relative: file_sha})
        trace.write_bytes(b"tampered")
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "SHA-256 mismatch"):
            diagnostic._trace(self.root, directory, {"trace_path": relative,
                               "trace_sha256": file_sha, "trace_deterministic_sha256": digest},
                              {relative: file_sha})

    def test_old_source_and_r7b_controls_verify_all_archived_trace_files_before_reset(self):
        source_directory = self.root / r7.R6_DIAGNOSTIC
        control_directory = self.root / diagnostic.R7_DIAGNOSTIC
        source_directory.mkdir(parents=True)
        control_directory.mkdir(parents=True)
        evaluator = self.root / "scripts/diagnose_drq_retention_r7.py"
        evaluator.parent.mkdir()
        evaluator.write_bytes(b"synthetic immutable comparator")
        source_files = {}
        source_episodes = []
        old_r6 = {}

        def trace(directory, name, seed, repeat):
            file = directory / name
            file.parent.mkdir(parents=True, exist_ok=True)
            arrays = {field: np.asarray([seed], dtype=np.int32) for field in r6.TRACE_ARRAYS}
            arrays["repeat_index"] = np.asarray([repeat], dtype=np.int8)
            np.savez_compressed(file, **arrays)
            return r6.sha256_file(file), r6._trace_digest(arrays)

        for row in self.rows:
            road = row["geometry_seed"]
            for seed in (0, 1):
                for repeat in (0, 1):
                    relative = f"traces/seed-{road}/unchanged-source-seed{seed}/repeat-{repeat}.npz"
                    digest, deterministic = trace(source_directory, relative, seed, repeat)
                    source_files[relative] = digest
                    source_episodes.append({"arm": "unchanged-source", "source_learner_seed": seed,
                                            "geometry_seed": road, "repeat": repeat,
                                            "trace_path": relative, "trace_sha256": digest,
                                            "trace_deterministic_sha256": deterministic})
                    if repeat == 0:
                        self.sources[(seed, road)]["trace_sha256"] = digest
                for variant in r6.VARIANTS:
                    old_r6[(seed, road, variant)] = {"finished": False,
                                                    "trace_sha256": hashlib.sha256(
                                                        f"old-r6-{seed}-{road}-{variant}".encode()).hexdigest()}
        source_path = source_directory / "episodes.jsonl"
        source_path.write_text("".join(json.dumps(row) + "\n" for row in source_episodes))
        _json(source_directory / "manifest.json", {"files_sha256": source_files})
        source_manifest_sha = r6.sha256_file(source_directory / "manifest.json")

        control_files = {}
        control_episodes = []
        paired_rows = []
        selected = {}
        for variant, (kept, gained) in {"uniform": (4, 5), "failure_weighted": (1, 7),
                                          "easy_retention": (5, 5)}.items():
            cells = sorted(self.sources)
            selected[variant] = set([key for key in cells if self.sources[key]["finished"]][:kept])
            selected[variant].update([key for key in cells if not self.sources[key]["finished"]][:gained])
        for row in self.rows:
            road = row["geometry_seed"]
            for seed in (0, 1):
                for variant in r6.VARIANTS:
                    for condition in ("r7a", "r7b"):
                        outcome = condition == "r7b" and (seed, road) in selected[variant]
                        canonical = None
                        for repeat in (0, 1):
                            relative = (f"traces/seed-{road}/{variant}-{condition}-seed{seed}"
                                        f"/repeat-{repeat}.npz")
                            if condition == "r7b":
                                digest, deterministic = trace(control_directory, relative, seed, repeat)
                            else:
                                digest = hashlib.sha256(relative.encode()).hexdigest()
                                deterministic = hashlib.sha256(f"same-{seed}-{road}".encode()).hexdigest()
                            control_files[relative] = digest
                            control_episodes.append({
                                "source_learner_seed": seed, "geometry_seed": road,
                                "variant": variant, "condition": condition, "repeat": repeat,
                                "family": row["family"], "track_id": 1,
                                "protocol_sha256": diagnostic.R7_PROTOCOL_SHA,
                                "catalog_sha256": r6.CATALOG_SHA256,
                                "canonical_repeat": repeat == 0,
                                "repeat_trace_agrees_with_repeat0": True,
                                "trace_path": relative, "trace_sha256": digest,
                                "trace_deterministic_sha256": deterministic,
                                "source_actor_sha256": self.source_by_seed[seed]["actor_sha256"],
                                "source_checkpoint_sha256": self.source_by_seed[seed]["checkpoint_sha256"],
                                "finished": outcome,
                            })
                            if repeat == 0:
                                canonical = digest
                        parent = self.sources[(seed, road)]
                        old = old_r6[(seed, road, variant)]
                        paired_rows.append({
                            "source_seed": seed, "geometry_seed": road, "variant": variant,
                            "condition": condition, "family": row["family"], "track_id": 1,
                            "source_finished": parent["finished"], "r6_finished": old["finished"],
                            "r7_finished": outcome,
                            "source_trace_sha256": parent["trace_sha256"],
                            "r6_trace_sha256": old["trace_sha256"],
                            "r7_trace_sha256": canonical,
                            "source_to_r7": r7.r6_transition(parent["finished"], outcome),
                        })
        episodes_path = control_directory / "episodes.jsonl"
        episodes_path.write_text("".join(json.dumps(row) + "\n" for row in control_episodes))
        control_files["episodes.jsonl"] = r6.sha256_file(episodes_path)
        paired_sha = _json(control_directory / "paired.json", {
            "format": r7.DIAGNOSTIC_FORMAT, "paired_cells": 192,
            "catalog_sha256": r6.CATALOG_SHA256,
            "protocol_sha256": diagnostic.R7_PROTOCOL_SHA, "pairs": paired_rows,
        })
        control_files["paired.json"] = paired_sha
        control_files["summary.json"] = _json(control_directory / "summary.json", {"role": "mock"})
        self.assertEqual(len(control_files), 387)
        manifest_sha = _json(control_directory / "manifest.json", {
            "format": r7.DIAGNOSTIC_FORMAT, "role": "TRAIN-DIAGNOSTIC",
            "ranked": False, "score_selection": False, "protocol_sha256": diagnostic.R7_PROTOCOL_SHA,
            "r6_manifest_sha256": source_manifest_sha, "catalog_sha256": r6.CATALOG_SHA256,
            "episode_count": 384, "evaluator_source_sha256": r6.sha256_file(evaluator),
            "files_sha256": control_files,
        })
        (control_directory / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n")
        with patch.object(r7, "_r6_reference", return_value=(
            self.sources, old_r6, {"r6_manifest_sha256": source_manifest_sha},
        )), patch.object(r7, "R6_MANIFEST_SHA", source_manifest_sha), patch.object(
            diagnostic, "R7_MANIFEST_SHA", manifest_sha,
        ), patch.object(diagnostic, "R7_EPISODES_SHA", control_files["episodes.jsonl"]), patch.object(
            diagnostic, "R7_PAIRED_SHA", paired_sha,
        ):
            sources, controls, lineage = diagnostic._controls(self.root, self.rows, {})
            self.assertEqual((len(sources), len(controls)), (32, 96))
            self.assertEqual(lineage["r7_manifest_sha256"], manifest_sha)
            self.assertEqual(sum(control["finished"] for key, control in controls.items()
                                 if key[2] == "failure_weighted"), 8)
            bad = control_directory / control_episodes[3]["trace_path"]
            if not bad.exists():
                bad = control_directory / next(row["trace_path"] for row in control_episodes
                                               if row["condition"] == "r7b")
            bad.write_bytes(b"tampered historical control trace")
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "SHA-256 mismatch"):
                diagnostic._controls(self.root, self.rows, {})
        self.assertEqual(FakeEnvironment.reset_count, 0)

    def test_preflight_never_resets_or_writes_and_output_refuses_overwrite(self):
        output = diagnostic.RUN_ROOT / "train-diagnostic"
        with patch.object(diagnostic, "_prepare", side_effect=diagnostic.DiagnosticError("missing receipt")):
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "missing receipt"):
                diagnostic.run_diagnostic(root=self.root, protocol_path=Path("experiments/study.json"),
                                          output_root=output, preflight_only=True,
                                          environment_factory=lambda *_: self.fail("environment created"))
        self.assertEqual(FakeEnvironment.reset_count, 0)
        self.assertFalse((self.root / output).exists())
        lineage = {"protocol_sha256": "f" * 64, "catalog_sha256": r6.CATALOG_SHA256,
                   "runtime": {"device": "cpu"}}
        with patch.object(diagnostic, "_prepare", return_value=(
            self.rows, self._roles(), self.sources, self.controls, lineage,
        )):
            preflight = diagnostic.run_diagnostic(
                root=self.root, protocol_path=Path("experiments/study.json"),
                output_root=output, preflight_only=True,
                environment_factory=lambda *_: self.fail("preflight created environment"))
            self.assertEqual(preflight["environment_resets"], 0)
            self.assertFalse((self.root / output).exists())
            result = diagnostic.run_diagnostic(
                root=self.root, protocol_path=Path("experiments/study.json"),
                output_root=output, environment_factory=FakeEnvironment)
            self.assertEqual(result["observed_episode_count"], 192)
            self.assertEqual(FakeEnvironment.reset_count, 192)
            self.assertTrue(all(env.closed for env in FakeEnvironment.instances))
            records = [json.loads(line) for line in (self.root / output / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 192)
            self.assertTrue(all(row["repeat_trace_agrees_with_repeat0"] for row in records))
            manifest = json.loads((self.root / output / "manifest.json").read_text())
            self.assertEqual(len(manifest["files_sha256"]), 195)
            self.assertEqual((self.root / output / "manifest.sha256").read_text(),
                             f"{r6.sha256_file(self.root / output / 'manifest.json')}  manifest.json\n")
            with self.assertRaises(FileExistsError):
                diagnostic.run_diagnostic(root=self.root, protocol_path=Path("experiments/study.json"),
                                          output_root=output, preflight_only=True,
                                          environment_factory=FakeEnvironment)
            self.assertEqual(FakeEnvironment.reset_count, 192)

    def test_synthetic_repeat_divergence_stops_before_paired_summary(self):
        output = diagnostic.RUN_ROOT / "train-diagnostic"
        lineage = {"protocol_sha256": "f" * 64, "catalog_sha256": r6.CATALOG_SHA256}
        FakeEnvironment.diverge_on_repeat = True
        with patch.object(diagnostic, "_prepare", return_value=(
            self.rows, self._roles(), self.sources, self.controls, lineage,
        )), patch.object(diagnostic, "_paired_summary", side_effect=AssertionError("must not pair")):
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "diverged"):
                diagnostic.run_diagnostic(root=self.root, protocol_path=Path("experiments/study.json"),
                                          output_root=output, environment_factory=FakeEnvironment)
        self.assertEqual(FakeEnvironment.reset_count, 2)


if __name__ == "__main__":
    unittest.main()
