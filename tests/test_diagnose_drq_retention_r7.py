"""Synthetic r7 diagnostic contracts; never load a real road, checkpoint or actor."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter, ObservationSpec
from scripts import diagnose_drq_geometry_mix as r6
from scripts import diagnose_drq_retention_r7 as diagnostic
from tests.test_drq_geometry_mix_diagnostics import FakeActor, FakeEnvironment


def _json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class RetentionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "experiments").mkdir()
        (self.root / "runs").mkdir()
        (self.root / diagnostic.RUN_ROOT).mkdir()
        FakeEnvironment.instances = []
        FakeEnvironment.reset_count = 0
        FakeEnvironment.diverge_on_repeat = False
        self.rows = [{"geometry_seed": seed, "family": r6.EXPECTED_FAMILIES[index % 6],
                      "road_coordinate_sha256": r6._road_hash(FakeEnvironment.track_for_seed(seed))}
                     for index, seed in enumerate(r6.DIAGNOSTIC_SEEDS)]
        self.sources = {seed: {"actor_sha256": hashlib.sha256(f"source-actor-{seed}".encode()).hexdigest(),
                               "checkpoint_sha256": hashlib.sha256(f"source-checkpoint-{seed}".encode()).hexdigest()}
                        for seed in (0, 1)}
        self.source_rows, self.old_rows = self._historic_rows()

    def _historic_rows(self):
        source = {}
        old = {}
        for seed in (0, 1):
            for index, row in enumerate(self.rows):
                road = row["geometry_seed"]
                source[(seed, road)] = {
                    "finished": index < (5 if seed == 0 else 6),
                    "family": row["family"], "actor_sha256": self.sources[seed]["actor_sha256"],
                    "checkpoint_sha256": self.sources[seed]["checkpoint_sha256"],
                    "trace_sha256": hashlib.sha256(f"source-{seed}-{road}".encode()).hexdigest(),
                }
                for variant in r6.VARIANTS:
                    finish = index in ({"uniform": (9,), "failure_weighted": (0, 9),
                                        "easy_retention": (0, 10)}[variant] if seed == 0 else
                                       {"uniform": (), "failure_weighted": (8, 12),
                                        "easy_retention": (8,)}[variant])
                    old[(seed, road, variant)] = {
                        "finished": finish, "family": row["family"],
                        "trace_sha256": hashlib.sha256(f"r6-{seed}-{road}-{variant}".encode()).hexdigest(),
                    }
        return source, old

    def _run_protocol(self):
        protocol = {
            "format": diagnostic.FORMAT, "run_root": diagnostic.RUN_ROOT.as_posix(),
            "r6_protocol_path": r6.PROTOCOL_PATH.as_posix(),
            "r6_protocol_sha256": diagnostic.R6_PROTOCOL_SHA,
            "lambda_preserve": 0.5,
            "diagnostic_cache": {"path": "runs/hybrid/first20.npz", "sha256": "a" * 64},
            "runs": [{"source_seed": seed, "variant": variant, "condition": condition,
                      "run_dir": (diagnostic.RUN_ROOT / f"learner-{seed}-{variant}-{condition}").as_posix(),
                      "rng_seeds": {"study_seed": 9000 + seed}}
                     for seed, variant, condition in sorted(diagnostic.ROLES)],
        }
        sha = _json(self.root / diagnostic.PROTOCOL, protocol)
        return protocol, sha

    @staticmethod
    def _loader(path, *, device):
        if device != "cpu":
            raise AssertionError("actor must run on CPU")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return FakeActor(float(payload["state_dict"]["policy.weight"][0])), ActionAdapter(), ObservationSpec()

    def _receipts(self, protocol_sha):
        protocol = json.loads((self.root / diagnostic.PROTOCOL).read_text())
        for index, (seed, variant, condition) in enumerate(sorted(diagnostic.ROLES)):
            run = diagnostic.RUN_ROOT / f"learner-{seed}-{variant}-{condition}"
            folder = self.root / run
            folder.mkdir()
            (folder / "study_protocol.json").write_bytes((self.root / diagnostic.PROTOCOL).read_bytes())
            _json(folder / "run-config.json", {
                "format": "haic-drq-retention-run-config-v1", "protocol_sha256": protocol_sha,
                "source_seed": seed, "variant": variant, "condition": condition,
                "source_actor_sha256": self.sources[seed]["actor_sha256"],
                "source_replay_sha256": self.sources[seed]["checkpoint_sha256"],
                "source_rows_per_batch": 32, "online_rows_per_batch": 32,
                "lambda_preserve": 0.5 if condition == "r7b" else 0.0,
                "diagnostic_cache_sha256": protocol["diagnostic_cache"]["sha256"],
                "rng_seeds": next(row["rng_seeds"] for row in protocol["runs"]
                                  if (row["source_seed"], row["variant"], row["condition"])
                                  == (seed, variant, condition)),
            })
            (folder / "step-metrics.jsonl").write_bytes(b"synthetic ledger")
            (folder / "drift.jsonl").write_bytes(b"synthetic drift")
            candidates = []
            for step in (16384, diagnostic.FINAL_STEP):
                checkpoint_dir = run / "checkpoints" / f"step-{step:09d}"
                actor = FakeActor(-0.5 + (index + 1) * 0.05)
                export = self.root / checkpoint_dir / "actor.pt"
                export.parent.mkdir(parents=True)
                torch.save({"format": "haic-drq-v2-actor-v1",
                            "config": {"augmentation_pad": 4, "steering_logit_l2": 0.0},
                            "state_dict": actor.state_dict()}, export)
                ckpt = self.root / checkpoint_dir / "checkpoint.pt"
                ckpt.write_bytes(f"checkpoint-{seed}-{variant}-{condition}-{step}".encode())
                manifest = checkpoint_dir / "checkpoint.manifest.json"
                manifest_sha = _json(self.root / manifest, {
                    "algorithm": "drq-v2", "extra": {
                        "environment_steps": 131072 + step,
                        "study_id": "drqv2-retention-r7", "study_protocol_sha256": protocol_sha,
                        "source_checkpoint_sha256": self.sources[seed]["checkpoint_sha256"],
                        "source_replay_sha256": self.sources[seed]["checkpoint_sha256"],
                        "catalog_sha256": r6.CATALOG_SHA256,
                    },
                })
                trace = run / f"replay-sample-trace-step-{step:09d}.npz"
                (self.root / trace).write_bytes(f"trace-{index}-{step}".encode())
                candidates.append({
                    "checkpoint_online_step": step,
                    "study_gradient_steps": step - 10000,
                    "actor_path": checkpoint_dir.joinpath("actor.pt").as_posix(),
                    "actor_sha256": r6.sha256_file(export),
                    "checkpoint_path": checkpoint_dir.joinpath("checkpoint.pt").as_posix(),
                    "checkpoint_sha256": r6.sha256_file(ckpt),
                    "sample_trace_path": trace.as_posix(),
                    "sample_trace_sha256": r6.sha256_file(self.root / trace),
                })
            _json(folder / "checkpoint-catalog.json", {
                "format": diagnostic.CATALOG_FORMAT,
                "study_protocol_sha256": protocol_sha, "source_seed": seed,
                "variant": variant, "condition": condition, "candidates": candidates,
            })
            _json(folder / "result.json", {
                "format": diagnostic.RESULT_FORMAT, "study_id": "drqv2-retention-r7", "completed": True,
                "study_protocol_sha256": protocol_sha, "source_seed": seed,
                "variant": variant, "condition": condition,
                "additional_online_steps": 32768, "study_gradient_steps": 22768,
                "source_samples": 22768 * 32, "online_samples": 22768 * 32,
                "source_actor_sha256": self.sources[seed]["actor_sha256"],
                "source_checkpoint_sha256": self.sources[seed]["checkpoint_sha256"],
                "source_replay_sha256": self.sources[seed]["checkpoint_sha256"],
                "online_replay_manifest_sha256": r6.sha256_file(folder / "step-metrics.jsonl"),
                "diagnostic_trace_sha256": r6.sha256_file(folder / "drift.jsonl"),
                "candidates": candidates,
            })

    def _reference_fixture(self):
        files = {"summary.json": "f" * 64}
        episodes = []
        pairs = []
        counts = defaultdict(Counter)
        families = defaultdict(Counter)
        for row in self.rows:
            road, family = row["geometry_seed"], row["family"]
            for seed in (0, 1):
                for role in ("unchanged-source", *r6.VARIANTS):
                    parent = (self.source_rows[(seed, road)] if role == "unchanged-source"
                              else self.old_rows[(seed, road, role)])
                    for repeat in (0, 1):
                        trace = f"traces/seed-{road}/{role}-seed{seed}/repeat-{repeat}.npz"
                        files[trace] = hashlib.sha256(trace.encode()).hexdigest()
                        episodes.append({
                            "source_learner_seed": seed, "geometry_seed": road, "arm": role,
                            "repeat": repeat, "track_id": 1, "family": family,
                            "protocol_sha256": diagnostic.R6_PROTOCOL_SHA,
                            "catalog_sha256": r6.CATALOG_SHA256,
                            "trace_path": trace, "trace_sha256": files[trace],
                            "trace_deterministic_sha256": parent["trace_sha256"],
                            "canonical_repeat": repeat == 0,
                            "repeat_trace_agrees_with_repeat0": True,
                            "finished": parent["finished"],
                            "actor_sha256": self.sources[seed]["actor_sha256"] if role == "unchanged-source" else "r6",
                        })
                    if role != "unchanged-source":
                        source_path = f"traces/seed-{road}/unchanged-source-seed{seed}/repeat-0.npz"
                        candidate_path = f"traces/seed-{road}/{role}-seed{seed}/repeat-0.npz"
                        transition = diagnostic.r6_transition(self.source_rows[(seed, road)]["finished"],
                                                              parent["finished"])
                        counts[role][transition] += 1
                        families[(role, family)][transition] += 1
                        pairs.append({"source_seed": seed, "geometry_seed": road, "variant": role,
                                      "family": family, "track_id": 1, "transition": transition,
                                      "source_finished": self.source_rows[(seed, road)]["finished"],
                                      "candidate_finished": parent["finished"],
                                      "source_trace_sha256": files[source_path],
                                      "candidate_trace_sha256": files[candidate_path]})
        diagnostic_root = self.root / diagnostic.R6_DIAGNOSTIC
        diagnostic_root.mkdir(parents=True)
        raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes).encode()
        (diagnostic_root / "episodes.jsonl").write_bytes(raw)
        episode_sha = hashlib.sha256(raw).hexdigest()
        self.assertEqual(len(files), 257)
        files["episodes.jsonl"] = episode_sha
        manifest_sha = _json(diagnostic_root / "manifest.json", {
            "protocol_sha256": diagnostic.R6_PROTOCOL_SHA, "episode_count": 256,
            "role": "TRAIN-DIAGNOSTIC", "ranked": False, "score_selection": False,
            "catalog": {"catalog_sha256": r6.CATALOG_SHA256}, "files_sha256": files,
        })
        (diagnostic_root / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n")
        names = ("lost", "gained", "both_finished", "both_failed")
        paired_sha = _json(self.root / diagnostic.R6_PAIRED, {
            "protocol_sha256": diagnostic.R6_PROTOCOL_SHA,
            "catalog_sha256": r6.CATALOG_SHA256,
            "diagnostic_manifest_sha256": manifest_sha,
            "episodes_sha256": episode_sha, "pairs": pairs,
            "variant_transitions": {variant: {name: counts[variant][name] for name in names}
                                    for variant in r6.VARIANTS},
            "family_transitions": [{"variant": variant, "family": family,
                                    "n": sum(families[(variant, family)].values()),
                                    **{name: families[(variant, family)][name] for name in names}}
                                   for variant in r6.VARIANTS for family in r6.EXPECTED_FAMILIES],
        })
        return manifest_sha, episode_sha, paired_sha

    def test_exact_r7_grid_rejects_duplicate_reused_and_wrong_protocol(self):
        protocol, _ = self._run_protocol()
        self.assertEqual(set(diagnostic._grid(protocol)), diagnostic.ROLES)
        protocol["runs"][11] = dict(protocol["runs"][0])
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "duplicate"):
            diagnostic._grid(protocol)
        protocol, _ = self._run_protocol()
        protocol["runs"][0]["run_dir"] = "runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-uniform"
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "directory"):
            diagnostic._grid(protocol)
        protocol, _ = self._run_protocol()
        protocol["r6_protocol_sha256"] = "b" * 64
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "lineage"):
            diagnostic._grid(protocol)

    def test_complete_receipts_and_exact_budget_fail_closed(self):
        protocol, protocol_sha = self._run_protocol()
        self._receipts(protocol_sha)
        key = sorted(diagnostic.ROLES)[0]
        run = diagnostic.RUN_ROOT / f"learner-{key[0]}-{key[1]}-{key[2]}"
        with patch.object(diagnostic, "_verify_step_ledger"):
            value = diagnostic._candidate(self.root, run, key, protocol_sha, self.sources[key[0]],
                                          protocol, set(r6.DIAGNOSTIC_SEEDS))
        self.assertEqual(value["checkpoint_online_step"], 32768)
        result_path = self.root / run / "result.json"
        receipt = json.loads(result_path.read_text())
        for field, value in (("completed", False), ("study_gradient_steps", 22767),
                             ("additional_online_steps", 32767), ("condition", "r7b"),
                             ("online_samples", 1)):
            changed = {**receipt, field: value}
            _json(result_path, changed)
            with self.subTest(field=field), patch.object(diagnostic, "_verify_step_ledger"), self.assertRaisesRegex(diagnostic.DiagnosticError, "completed exact-budget"):
                diagnostic._candidate(self.root, run, key, protocol_sha, self.sources[key[0]],
                                      protocol, set(r6.DIAGNOSTIC_SEEDS))
        _json(result_path, receipt)
        checkpoint = self.root / run / "checkpoints/step-000032768/checkpoint.pt"
        checkpoint.write_bytes(b"tampered")
        with patch.object(diagnostic, "_verify_step_ledger"), self.assertRaisesRegex(diagnostic.DiagnosticError, "SHA-256 mismatch"):
            diagnostic._candidate(self.root, run, key, protocol_sha, self.sources[key[0]],
                                  protocol, set(r6.DIAGNOSTIC_SEEDS))

    def test_original_r6_pairs_validate_source_and_every_outcome(self):
        manifest, episodes, paired = self._reference_fixture()
        with patch.multiple(diagnostic, R6_MANIFEST_SHA=manifest, R6_EPISODES_SHA=episodes,
                            R6_PAIRED_SHA=paired):
            source, old, lineage = diagnostic._r6_reference(
                self.root, {"diagnostics": {"geometry_seeds": list(r6.DIAGNOSTIC_SEEDS)}}, self.rows,
            )
            self.assertEqual(len(source), 32)
            self.assertEqual(len(old), 96)
            self.assertEqual(sum(row["finished"] for row in source.values()), 11)
            self.assertEqual(lineage["r6_paired_regression_sha256"], paired)
            manifest_path = self.root / diagnostic.R6_DIAGNOSTIC / "manifest.json"
            content = json.loads(manifest_path.read_text())
            content["episode_count"] = 255
            _json(manifest_path, content)
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "SHA-256 mismatch"):
                diagnostic._r6_reference(
                    self.root, {"diagnostics": {"geometry_seeds": list(r6.DIAGNOSTIC_SEEDS)}}, self.rows,
                )

    def test_r7_actor_export_must_reload_with_identical_cpu_actions(self):
        protocol, protocol_sha = self._run_protocol()
        self._receipts(protocol_sha)
        key = sorted(diagnostic.ROLES)[0]
        run = diagnostic.RUN_ROOT / f"learner-{key[0]}-{key[1]}-{key[2]}"
        with patch.object(diagnostic, "_verify_step_ledger"):
            record = diagnostic._candidate(self.root, run, key, protocol_sha,
                                           self.sources[key[0]], protocol, set(r6.DIAGNOSTIC_SEEDS))
        r6._validate_actor(record["actor_path"], record["actor_sha256"],
                           record["actor_weights_sha256"], loader=self._loader)
        calls = 0

        def diverging_loader(path, *, device):
            nonlocal calls
            calls += 1
            actor, adapter, spec = self._loader(path, device=device)
            if calls == 2:
                actor.weight += 0.2
            return actor, adapter, spec

        with self.assertRaisesRegex(r6.DiagnosticError, "reload"):
            r6._validate_actor(record["actor_path"], record["actor_sha256"],
                               record["actor_weights_sha256"], loader=diverging_loader)

    def test_paired_seed_family_lists_and_exact_denominators(self):
        episodes = []
        for seed, variant, condition in diagnostic.ROLES:
            for row in self.rows:
                road = row["geometry_seed"]
                for repeat in (0, 1):
                    episodes.append({"source_learner_seed": seed, "geometry_seed": road,
                                     "variant": variant, "condition": condition, "repeat": repeat,
                                     "track_id": 1, "family": row["family"], "finished": road == self.rows[0]["geometry_seed"],
                                     "source_actor_sha256": self.sources[seed]["actor_sha256"],
                                     "source_checkpoint_sha256": self.sources[seed]["checkpoint_sha256"],
                                     "trace_sha256": "a" * 64, "max_progress": 0.9,
                                     "raw_reward_sum": 3.0, "repeat_trace_agrees_with_repeat0": True})
        summary, paired = diagnostic._paired_summary(
            episodes, self.rows, self.source_rows, self.old_rows, {"protocol_sha256": "f" * 64},
        )
        self.assertEqual(len(paired["pairs"]), 192)
        self.assertEqual(summary["observed_episode_count"], 384)
        top = next(item for item in summary["paired_source_and_r6"]
                   if item["variant"] == "uniform" and item["condition"] == "r7a"
                   and item["source_seed"] is None and item["family"] is None)
        self.assertEqual((top["paired_cells"], top["source_finishes"], top["r6_finishes"], top["r7_finishes"]),
                         (32, 11, 1, 2))
        self.assertEqual(top["source_to_r7"], {"lost": 9, "gained": 0,
                                                "both_finished": 2, "both_failed": 21})
        self.assertEqual(top["r6_to_r7"], {"lost": 1, "gained": 2,
                                            "both_finished": 0, "both_failed": 29})
        by_seed_family = next(item for item in summary["paired_source_and_r6"]
                              if item["variant"] == "uniform" and item["condition"] == "r7a"
                              and item["source_seed"] == 0 and item["family"] == self.rows[0]["family"])
        self.assertEqual(by_seed_family["source_success_roads"],
                         [self.rows[0]["geometry_seed"]])
        self.assertEqual(by_seed_family["kept_source_success_roads"], [self.rows[0]["geometry_seed"]])
        episodes.pop()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "incomplete"):
            diagnostic._paired_summary(episodes, self.rows, self.source_rows, self.old_rows, {})

    def test_preflight_aborts_before_reset_when_protocol_or_result_absent(self):
        _, protocol_sha = self._run_protocol()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "missing"):
            diagnostic.run_diagnostic(root=self.root, protocol_sha256=protocol_sha,
                                      output_root=diagnostic.RUN_ROOT / "train-diagnostic",
                                      environment_factory=lambda *_: self.fail("simulator was reset"))
        self.assertFalse((self.root / diagnostic.RUN_ROOT / "train-diagnostic").exists())
        self.assertEqual(FakeEnvironment.reset_count, 0)

    def test_all_twelve_receipts_are_required_before_any_reset(self):
        frozen_r6 = {"format": r6.PROTOCOL_FORMAT,
                     "environment": {"geometry_seeds": [700000000 + i for i in range(120)]},
                     "diagnostics": {"geometry_seeds": list(r6.DIAGNOSTIC_SEEDS)}}
        fake_r6_sha = _json(self.root / r6.PROTOCOL_PATH, frozen_r6)
        catalog_protocol_sha = _json(self.root / r6.CATALOG_PROTOCOL_PATH, {"format": "synthetic"})
        cache = self.root / "runs/hybrid/first20.npz"
        cache.parent.mkdir()
        cache.write_bytes(b"synthetic float32 cache")
        hybrid = self.root / "runs/hybrid/result.json"
        hybrid.write_bytes(b"synthetic hybrid receipt")
        probe = self.root / "runs/gradient-scale-v1.json"
        probe.write_bytes(b"synthetic gradient probe")
        source_replay = {}
        source_records = {}
        for seed in (0, 1):
            checkpoint = self.root / "runs" / f"synthetic-source-{seed}" / "checkpoint.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(f"source-checkpoint-{seed}".encode())
            self.assertEqual(r6.sha256_file(checkpoint), self.sources[seed]["checkpoint_sha256"])
            ledger = checkpoint.parent / "episodes.jsonl"
            ledger.write_bytes(b"synthetic source episode ledger")
            source_replay[str(seed)] = {
                "checkpoint_path": checkpoint.relative_to(self.root).as_posix(),
                "checkpoint_sha256": r6.sha256_file(checkpoint),
                "episode_ledger_path": ledger.relative_to(self.root).as_posix(),
                "episode_ledger_sha256": r6.sha256_file(ledger),
            }
            source_records[seed] = {**self.sources[seed], "checkpoint_path": checkpoint}

        with patch.object(diagnostic, "R6_PROTOCOL_SHA", fake_r6_sha), patch.object(
            r6, "CATALOG_PROTOCOL_SHA256", catalog_protocol_sha,
        ), patch.object(r6, "_validate_study_contract"), patch.object(
            r6, "_validate_cpu21", return_value={"device": "cpu"},
        ), patch.object(diagnostic, "_runtime_sources", return_value={"scripts/diagnose_drq_geometry_mix.py": "f" * 64}
        ), patch.object(r6, "_catalog_rows", return_value=(self.rows, {"catalog_sha256": r6.CATALOG_SHA256})), patch.object(
            r6, "_source_records", return_value=source_records,
        ), patch.object(diagnostic, "_r6_reference", return_value=(
            self.source_rows, self.old_rows, {"r6_protocol_sha256": fake_r6_sha},
        )), patch.object(diagnostic, "_verify_step_ledger"):
            protocol, _ = self._run_protocol()
            protocol.update({
                "study_id": "drqv2-retention-r7",
                "r6_contract": "inherit-learner-budget-geometry-environment-verbatim",
                "catalog_sha256": r6.CATALOG_SHA256,
                "diagnostic_manifest_sha256": diagnostic.R6_MANIFEST_SHA,
                "interpretation": {"no_post_result_lambda_or_threshold_change": True},
                "source_replay": source_replay,
                "gradient_probe": {"path": probe.relative_to(self.root).as_posix(),
                                   "sha256": r6.sha256_file(probe)},
                "diagnostic_cache": {"path": cache.relative_to(self.root).as_posix(),
                                     "sha256": r6.sha256_file(cache),
                                     "hybrid_result_path": hybrid.relative_to(self.root).as_posix(),
                                     "hybrid_result_sha256": r6.sha256_file(hybrid)},
            })
            protocol_sha = _json(self.root / diagnostic.PROTOCOL, protocol)
            self._receipts(protocol_sha)
            missing = self.root / diagnostic.RUN_ROOT / "learner-1-uniform-r7b/result.json"
            raw = missing.read_bytes()
            missing.unlink()
            output = diagnostic.RUN_ROOT / "train-diagnostic"
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "receipt is missing"):
                diagnostic.run_diagnostic(
                    root=self.root, protocol_sha256=protocol_sha, output_root=output,
                    environment_factory=lambda *_: self.fail("no real or fake reset is allowed"),
                    actor_loader=self._loader,
                )
            self.assertEqual(FakeEnvironment.reset_count, 0)
            self.assertFalse((self.root / output).exists())
            missing.write_bytes(raw)
            rows, roles, source, prior, lineage = diagnostic._prepare(self.root, protocol_sha, self._loader)
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(roles), 12)
            self.assertEqual(len(source), 32)
            self.assertEqual(len(prior), 96)
            self.assertEqual(lineage["protocol_sha256"], protocol_sha)
            self.assertEqual(FakeEnvironment.reset_count, 0)

    def test_changed_original_cpu_runtime_fails_closed(self):
        original_sources = diagnostic.RUNTIME_SOURCES
        file = self.root / "train.py"
        file.write_bytes(b"frozen")
        with patch.object(diagnostic, "RUNTIME_SOURCES", ("train.py",)):
            self.assertEqual(diagnostic._runtime_sources(
                self.root, {"source_sha256": {"train.py": r6.sha256_file(file)}},
            ), {"train.py": r6.sha256_file(file)})
            file.write_bytes(b"changed")
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "source hash changed"):
                diagnostic._runtime_sources(
                    self.root, {"source_sha256": {"train.py": hashlib.sha256(b"frozen").hexdigest()}},
                )
        self.assertEqual(len(original_sources), 17)

    def test_step_ledger_rejects_omission_wrong_update_and_non_train_road(self):
        path = self.root / "runs" / "synthetic-ledger.jsonl"
        road = self.rows[0]["geometry_seed"]
        lines = [{"additional_online_step": step, "gradient_steps": max(0, step - 1),
                  "geometry_seed": road, "track_id": 1, "source_samples": max(0, step - 1) * 32,
                  "online_samples": max(0, step - 1) * 32}
                 for step in range(1, 5)]
        def save():
            path.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")

        save()
        diagnostic._verify_step_ledger(path, {road}, max_step=4, warmup=1)
        lines[2]["gradient_steps"] = 0
        save()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "row 3"):
            diagnostic._verify_step_ledger(path, {road}, max_step=4, warmup=1)
        lines[2]["gradient_steps"] = 2
        lines[2]["geometry_seed"] = 99999
        save()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "TRAIN-only"):
            diagnostic._verify_step_ledger(path, {road}, max_step=4, warmup=1)
        lines[2]["geometry_seed"] = road
        lines.pop()
        save()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "truncated"):
            diagnostic._verify_step_ledger(path, {road}, max_step=4, warmup=1)

    def test_synthetic_full_matrix_repeat_hashes_and_immutable_output(self):
        roles = {}
        for index, (seed, variant, condition) in enumerate(sorted(diagnostic.ROLES)):
            name = f"{variant}-{condition}-seed{seed}"
            value = self.sources[seed]
            roles[name] = {
                "role": f"{variant}-{condition}", "arm": f"{variant}-{condition}",
                "variant": variant, "condition": condition, "source_learner_seed": seed,
                "source_actor_sha256": value["actor_sha256"],
                "source_checkpoint_sha256": value["checkpoint_sha256"],
                "actor_path": self.root / "runs" / name / "actor.pt", "actor_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "actor_weights_sha256": hashlib.sha256(f"w{name}".encode()).hexdigest(),
                "checkpoint_path": self.root / "runs" / name / "checkpoint.pt",
                "checkpoint_sha256": hashlib.sha256(f"c{name}".encode()).hexdigest(),
                "checkpoint_manifest_path": self.root / "runs" / name / "checkpoint.manifest.json",
                "checkpoint_manifest_sha256": hashlib.sha256(f"m{name}".encode()).hexdigest(),
                "checkpoint_online_step": 32768, "catalog_path": self.root / "runs" / name / "checkpoint-catalog.json",
                "catalog_sha256": "d" * 64, "result_path": self.root / "runs" / name / "result.json",
                "result_sha256": "e" * 64,
                "actor": FakeActor(-0.7 + 0.08 * index), "action_adapter": ActionAdapter(),
                "observation_spec": ObservationSpec(),
            }
        lineage = {"protocol_sha256": "f" * 64, "catalog_sha256": r6.CATALOG_SHA256,
                   "runtime": {"device": "cpu"}}
        output = diagnostic.RUN_ROOT / "train-diagnostic"
        with patch.object(diagnostic, "_prepare",
                          return_value=(self.rows, roles, self.source_rows, self.old_rows, lineage)):
            result = diagnostic.run_diagnostic(
                root=self.root, protocol_sha256="f" * 64, output_root=output,
                environment_factory=FakeEnvironment,
            )
            self.assertEqual(result["observed_episode_count"], 384)
            self.assertEqual(FakeEnvironment.reset_count, 384)
            self.assertTrue(all(env.closed for env in FakeEnvironment.instances))
            out = self.root / output
            records = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 384)
            self.assertEqual(len({row["cell_id"] for row in records}), 384)
            self.assertEqual({row["track_id"] for row in records}, {1})
            self.assertEqual({row["warmup_raw_noop_actions"] for row in records}, {50})
            self.assertTrue(all(row["repeat_trace_agrees_with_repeat0"] for row in records))
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(len(manifest["files_sha256"]), 387)
            self.assertEqual((out / "manifest.sha256").read_text(),
                             f"{r6.sha256_file(out / 'manifest.json')}  manifest.json\n")
            with np.load(out / records[0]["trace_path"], allow_pickle=False) as trace:
                self.assertEqual(set(trace.files), set(r6.TRACE_ARRAYS))
            with self.assertRaises(FileExistsError):
                diagnostic.run_diagnostic(root=self.root, protocol_sha256="f" * 64,
                                          output_root=output, environment_factory=FakeEnvironment)
            self.assertEqual(FakeEnvironment.reset_count, 384)

    def test_repeat_divergence_stops_without_paired_claim(self):
        role = {"actor": FakeActor(0.1), "action_adapter": ActionAdapter(),
                "observation_spec": ObservationSpec()}
        # Exercise the original collector's repeat digest: repeat index is excluded,
        # but any changed reward must change the digest and fail the r7 run.
        env = FakeEnvironment(self.rows[0]["geometry_seed"], r6.MAX_STEPS)
        env.diverge_on_repeat = True
        shared = {
            **role, "role": "uniform-r7a", "arm": "uniform-r7a", "variant": "uniform",
            "condition": "r7a", "source_learner_seed": 0,
            "actor_path": self.root / "runs/actor.pt", "actor_sha256": "a" * 64,
            "actor_weights_sha256": "b" * 64, "checkpoint_path": self.root / "runs/checkpoint.pt",
            "checkpoint_sha256": "c" * 64, "checkpoint_manifest_path": self.root / "runs/manifest.json",
            "checkpoint_manifest_sha256": "d" * 64, "checkpoint_online_step": 32768,
            "source_actor_sha256": self.sources[0]["actor_sha256"],
            "source_checkpoint_sha256": self.sources[0]["checkpoint_sha256"],
        }
        _, first = r6._collect_episode(env, shared, self.rows[0], 0, episode_id=0,
                                       max_steps=r6.MAX_STEPS, root=self.root)
        _, second = r6._collect_episode(env, shared, self.rows[0], 1, episode_id=0,
                                        max_steps=r6.MAX_STEPS, root=self.root)
        self.assertNotEqual(r6._trace_digest(first), r6._trace_digest(second))
        env.close()


if __name__ == "__main__":
    unittest.main()
