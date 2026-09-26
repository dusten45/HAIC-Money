"""Synthetic geometry-mix diagnostics tests; no real environment is created."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter, ObservationSpec
from scripts import diagnose_drq_geometry_mix as diagnostic


def _json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class FakeActor:
    def __init__(self, value: float):
        self.weight = torch.tensor([value], dtype=torch.float32)

    def state_dict(self):
        return {"policy.weight": self.weight}

    def act(self, observation, deterministic=True):
        if deterministic is not True:
            raise AssertionError("diagnostic policy action was stochastic")
        return np.asarray([float(self.weight[0]), 0.25, -0.5], dtype=np.float32)


class FakeEnvironment:
    instances: list["FakeEnvironment"] = []
    reset_count = 0
    diverge_on_repeat = False

    @staticmethod
    def track_for_seed(seed: int) -> np.ndarray:
        angle = np.arange(32, dtype=np.float64) * (2.0 * np.pi / 32)
        radius = 20.0 + (seed % 100) * 0.001
        return np.column_stack((
            np.zeros_like(angle), np.zeros_like(angle),
            radius * np.cos(angle), radius * np.sin(angle),
        ))

    def __init__(self, geometry_seed: int, max_steps: int):
        self.track_seed = geometry_seed
        self.track_id = 1
        self.max_steps = max_steps
        self.unwrapped = self
        self.env = None
        self.track = self.track_for_seed(geometry_seed)
        self.car = SimpleNamespace(hull=SimpleNamespace(
            position=self.track[0, 2:4].copy(), linearVelocity=np.asarray([3.0, 0.0]),
        ))
        self.off_track_counter = 0
        self.max_off_track_steps = 100
        self.warmup_steps = 50
        self.closed = False
        self.reset_calls = 0
        self.step_number = 0
        self.instances.append(self)

    def reset(self, *, seed=None, options=None):
        if seed is not None or options is not None:
            raise AssertionError("collector must directly reset the fixed HaicTrack with seed=None")
        self.reset_calls += 1
        FakeEnvironment.reset_count += 1
        self.step_number = 0
        self.off_track_counter = 0
        self.car.hull.position = self.track[0, 2:4].copy()
        self.car.hull.linearVelocity = np.asarray([3.0, 0.0])
        return np.zeros((4, 84, 84), dtype=np.float32), {
            "seed": self.track_seed, "track_id": self.track_id,
        }

    def step(self, official_action):
        action = np.asarray(official_action)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise AssertionError("invalid official action")
        self.step_number += 1
        self.car.hull.position = self.track[self.step_number % len(self.track), 2:4].copy()
        self.car.hull.linearVelocity = np.asarray([3.0 + self.step_number, 0.0])
        mode = self.track_seed % 3
        finished = mode == 0
        crashed = mode == 1
        off_track = mode == 2
        reward = 1.0 if not (self.diverge_on_repeat and self.reset_calls == 2) else 2.0
        self.off_track_counter = 101 if off_track else 0
        terminated = crashed or off_track
        truncated = finished
        return np.zeros((4, 84, 84), dtype=np.float32), reward, terminated, truncated, {
            "seed": self.track_seed,
            "track_id": 1,
            "progress": 0.75 if finished else 0.25,
            "damage": 0.5 if crashed else 0.0,
            "collision": crashed,
            "retire_reason": "crash" if crashed else "off_track" if off_track else None,
            "finished": finished,
            "finish_qualified": finished,
            "finish_time_s": 0.04 if finished else None,
            "finish_qualified_time_s": 0.02 if finished else None,
        }

    def close(self):
        self.closed = True


class GeometryMixDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "runs").mkdir()
        (self.root / "experiments").mkdir()
        FakeEnvironment.instances = []
        FakeEnvironment.reset_count = 0
        FakeEnvironment.diverge_on_repeat = False
        self.patches = [
            patch.object(diagnostic, "PROTOCOL_PATH", Path("experiments/drqv2-geometry-mix-v1-r6.json")),
            patch.object(diagnostic, "CATALOG_PATH", Path("runs/synthetic/catalog.json")),
            patch.object(diagnostic, "CATALOG_PROTOCOL_PATH", Path("experiments/catalog-protocol.json")),
            patch.object(diagnostic, "_validate_cpu21", return_value={"device": "cpu", "runtime": "mock"}),
            patch.object(diagnostic, "load_exported_actor", self._load_actor),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    @staticmethod
    def _actor_payload(value: float) -> dict:
        actor = FakeActor(value)
        return {
            "format": "haic-drq-v2-actor-v1",
            "config": {"augmentation_pad": 4, "steering_logit_l2": 0.0},
            "state_dict": actor.state_dict(),
        }

    @staticmethod
    def _load_actor(path: Path, *, device: str):
        if device != "cpu":
            raise AssertionError("models must be loaded on CPU")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = float(payload["state_dict"]["policy.weight"][0])
        return FakeActor(value), ActionAdapter(), ObservationSpec()

    def _write_actor(self, relative: str, value: float) -> tuple[str, str]:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._actor_payload(value), path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        weights_sha = diagnostic._state_dict_sha256(payload["state_dict"])
        return relative, weights_sha

    def _inputs(self, *, blind_seed: int | None = None, duplicate_run: bool = False,
                duplicate_geometry: bool = False,
                diagnostic_seeds: list[int] | None = None) -> tuple[Path, str]:
        generation_actors = []
        source_records = []
        expected_sources = {}
        for seed in (0, 1):
            actor_rel = f"runs/source-{seed}/checkpoints/step-000131072/actor.pt"
            actor_path, weight_sha = self._write_actor(actor_rel, -0.8 + seed * 0.2)
            actor_sha = diagnostic.sha256_file(self.root / actor_path)
            checkpoint_rel = f"runs/source-{seed}/checkpoints/step-000131072/checkpoint.pt"
            manifest_rel = f"runs/source-{seed}/checkpoints/step-000131072/checkpoint.manifest.json"
            checkpoint_path = self.root / checkpoint_rel
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(f"source-checkpoint-{seed}".encode())
            manifest = {"algorithm": "drq-v2", "extra": {"environment_steps": 131072}}
            manifest_sha = _json(self.root / manifest_rel, manifest)
            source = {
                "learner_seed": seed,
                "actor_path": actor_path,
                "actor_sha256": actor_sha,
                "actor_weights_sha256": weight_sha,
                "checkpoint_path": checkpoint_rel,
                "checkpoint_sha256": diagnostic.sha256_file(checkpoint_path),
                "checkpoint_manifest_path": manifest_rel,
                "checkpoint_manifest_sha256": manifest_sha,
                "source_checkpoint_path": checkpoint_rel,
                "source_checkpoint_sha256": diagnostic.sha256_file(checkpoint_path),
                "source_actor_path": actor_path,
                "source_actor_sha256": actor_sha,
                "source_actor_weights_sha256": weight_sha,
                "weight_only_fork": True,
            }
            source_records.append(source)
            generation_actors.append({key: source[key] for key in (
                "learner_seed", "actor_path", "actor_sha256", "actor_weights_sha256",
            )})
            expected_sources[seed] = {
                "actor_path": actor_path,
                "actor_sha256": actor_sha,
                "actor_weights_sha256": weight_sha,
            }

        generation_protocol = {
            "format": "haic-drq-training-geometry-protocol-v1",
            "source_actors": generation_actors,
            "exclusion_seed_ids": {
                "reserved": [], "heldout": [],
                "blind": [blind_seed] if blind_seed is not None else [99999999],
            },
        }
        generation_path = self.root / diagnostic.CATALOG_PROTOCOL_PATH
        generation_sha = _json(generation_path, generation_protocol)
        diagnostic.CATALOG_PROTOCOL_SHA256 = generation_sha

        ids = list(diagnostic.DIAGNOSTIC_SEEDS if diagnostic_seeds is None else diagnostic_seeds)
        families = list(diagnostic.EXPECTED_FAMILIES)
        rows = []
        for index, seed in enumerate(diagnostic.DIAGNOSTIC_SEEDS):
            track = FakeEnvironment.track_for_seed(seed)
            rows.append({
                "geometry_seed": seed,
                "track_id": 1,
                "family": families[index % len(families)],
                "stage": "diagnostic",
                "road_coordinate_sha256": diagnostic._road_hash(track),
                "signature_sha256": hashlib.sha256(f"signature-{seed}".encode()).hexdigest(),
                "verification": {"regenerated_coordinate_hash_match": True},
            })
        if duplicate_geometry:
            rows[-1] = dict(rows[0])
        catalog = {
            "format": diagnostic.CATALOG_FORMAT,
            "protocol_sha256": generation_sha,
            "train": [{"geometry_seed": 123, "track_id": 1, "family": "train-only-unused"}],
            "train_diagnostic": rows,
            "seed_audit": {"passed": True, "matched_collisions": []},
            "scan_path": "runs/synthetic/scan.json",
            "scan_sha256": "a" * 64,
        }
        catalog_path = self.root / diagnostic.CATALOG_PATH
        catalog_sha = _json(catalog_path, catalog)
        diagnostic.CATALOG_SHA256 = catalog_sha

        run_records = []
        for seed in (0, 1):
            for variant_index, variant in enumerate(diagnostic.VARIANTS):
                run_dir = Path("runs/20260925-drqv2-geometry-mix-v1-r6") / f"learner-{seed}-{variant}"
                run_records.append({"source_seed": seed, "variant": variant, "run_dir": run_dir.as_posix()})
        if duplicate_run:
            run_records[-1] = dict(run_records[0])

        protocol = {
            "format": diagnostic.PROTOCOL_FORMAT,
            "run_root": "runs/20260925-drqv2-geometry-mix-v1-r6",
            "catalog": {
                "path": diagnostic.CATALOG_PATH.as_posix(),
                "sha256": catalog_sha,
                "protocol_path": diagnostic.CATALOG_PROTOCOL_PATH.as_posix(),
                "protocol_sha256": generation_sha,
            },
            "catalog_sha256": catalog_sha,
            "catalog_protocol_path": diagnostic.CATALOG_PROTOCOL_PATH.as_posix(),
            "catalog_protocol_sha256": generation_sha,
            "environment": {
                "partition": "TRAIN",
                "geometry_seeds": [700000000 + index for index in range(120)],
                "track_ids": [1, 2, 3, 4],
                "frame_skip": 4,
                "reward_shaping": False,
                "reward_normalization": False,
            },
            "learner": {
                "source_environment_steps": 131072,
                "padding": 4,
                "steering_logit_l2": 0.0,
            },
            "budgets": {
                "additional_online_steps": 32768,
                "expected_gradient_updates": 22768,
                "warmup_steps": 10000,
                "checkpoint_steps": [16384, 32768],
            },
            "replay": {"mode": "online-only", "online_rows": 64, "teacher_rows": 0},
            "variants": {variant: {"score_selection": False} for variant in diagnostic.VARIANTS},
            "diagnostics": {
                "format": "haic-drq-geometry-mix-diagnostic-v1",
                "role": "training_diagnostic",
                "partition": "TRAIN-DIAGNOSTIC",
                "geometry_seeds": ids,
                "track_ids": [1],
                "repeats": 2,
                "max_steps": 1200,
                "frame_skip": 4,
                "raw_reward": True,
                "ranked": False,
                "source_actors_included": True,
                "no_official_performance_claim": True,
                "candidate_runs": [
                    {key: record[key] for key in ("source_seed", "variant", "run_dir")}
                    for record in run_records
                ],
                "runtime": {"verified_interpreter": "/tmp/kilo/haic-cpu21/bin/python"},
                "source_actors": source_records,
            },
            "runs": run_records,
        }
        protocol_path = self.root / diagnostic.PROTOCOL_PATH
        protocol_sha = _json(protocol_path, protocol)
        for record in run_records:
            seed, variant = record["source_seed"], record["variant"]
            if variant not in diagnostic.VARIANTS:
                continue
            run_dir = self.root / record["run_dir"]
            step_dir = run_dir / "checkpoints/step-000032768"
            actor_rel, actor_weights_sha = self._write_actor(
                f"{record['run_dir']}/checkpoints/step-000032768/actor.pt",
                0.1 + seed * 0.3 + diagnostic.VARIANTS.index(variant) * 0.07,
            )
            actor_path = self.root / actor_rel
            actor_sha = diagnostic.sha256_file(actor_path)
            checkpoint_rel = f"{record['run_dir']}/checkpoints/step-000032768/checkpoint.pt"
            manifest_rel = f"{record['run_dir']}/checkpoints/step-000032768/checkpoint.manifest.json"
            checkpoint_path = self.root / checkpoint_rel
            checkpoint_path.write_bytes(f"candidate-checkpoint-{seed}-{variant}".encode())
            source = source_records[seed]
            manifest_sha = _json(self.root / manifest_rel, {
                "algorithm": "drq-v2",
                "extra": {
                    "environment_steps": 131072 + 32768,
                    "study_protocol_sha256": protocol_sha,
                    "source_seed": seed,
                    "variant": variant,
                    "source_actor_sha256": source["actor_sha256"],
                    "source_checkpoint_sha256": source["checkpoint_sha256"],
                    "checkpoint_online_step": 32768,
                    "catalog_sha256": catalog_sha,
                },
            })
            candidate = {
                "source_seed": seed,
                "variant": variant,
                "checkpoint_online_step": 32768,
                "study_gradient_steps": 22768,
                "resume_allowed": False,
                "actor_path": actor_rel,
                "actor_sha256": actor_sha,
                "actor_weights_sha256": actor_weights_sha,
                "checkpoint_path": checkpoint_rel,
                "checkpoint_sha256": diagnostic.sha256_file(checkpoint_path),
                "checkpoint_manifest_path": manifest_rel,
                "checkpoint_manifest_sha256": manifest_sha,
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "source_actor_sha256": source["actor_sha256"],
                "study_protocol_sha256": protocol_sha,
            }
            checkpoint_catalog = {
                "format": diagnostic.CHECKPOINT_CATALOG_FORMAT,
                "study_protocol_sha256": protocol_sha,
                "source_seed": seed,
                "source_actor_sha256": source["actor_sha256"],
                "variant": variant,
                "catalog_sha256": catalog_sha,
                "candidates": [candidate],
            }
            _json(run_dir / "checkpoint-catalog.json", checkpoint_catalog)
        diagnostic.SOURCE_ACTORS = expected_sources
        return protocol_path, protocol_sha

    def _run(self, inputs, *, output_name="diagnostic", environment_factory=None):
        return diagnostic.run_diagnostic(
            root=self.root,
            protocol_path=inputs[0],
            protocol_sha256=inputs[1],
            output_root=self.root / "runs" / output_name,
            environment_factory=environment_factory or (lambda seed, steps: FakeEnvironment(seed, steps)),
            actor_loader=self._load_actor,
        )

    def test_complete_grid_repeats_and_descriptive_outputs(self):
        inputs = self._inputs()
        summary = self._run(inputs)
        out = self.root / "runs/diagnostic"
        rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
        self.assertEqual(summary["observed_episode_count"], 16 * 8 * 2)
        self.assertEqual(len(rows), 256)
        self.assertEqual(len({row["cell_id"] for row in rows}), 256)
        self.assertEqual({row["repeat"] for row in rows}, {0, 1})
        self.assertEqual({row["track_id"] for row in rows}, {1})
        self.assertEqual({row["geometry_seed"] for row in rows}, set(diagnostic.DIAGNOSTIC_SEEDS))
        self.assertEqual(len({row["episode_id"] for row in rows}), 16 * 8)
        self.assertEqual({row["collector_episode_id"] for row in rows}, {0})
        self.assertEqual({row["role"] for row in rows}, {
            "unchanged-source", *diagnostic.VARIANTS,
        })
        self.assertEqual(
            {(row["role"], row["source_learner_seed"]) for row in rows},
            {("unchanged-source", seed) for seed in (0, 1)}
            | {(variant, seed) for variant in diagnostic.VARIANTS for seed in (0, 1)},
        )
        self.assertTrue(all(row["repeat_trace_agrees_with_repeat0"] for row in rows))
        self.assertEqual({row["termination_class"] for row in rows}, {"finished", "crash", "off_track"})
        self.assertEqual({row["warmup_raw_noop_actions"] for row in rows}, {50})
        self.assertEqual({row["warmup_raw_frames"] for row in rows}, {200})
        self.assertEqual(len(summary["family_arm_seed"]), 6 * 8)
        self.assertIn("no significance test", summary["sample_interpretation"])
        self.assertFalse(summary["ranked"])
        self.assertEqual(FakeEnvironment.reset_count, 256)
        self.assertEqual(len(FakeEnvironment.instances), 128)
        self.assertTrue(all(env.closed for env in FakeEnvironment.instances))
        with self.assertRaises(FileExistsError):
            self._run(inputs)
        self.assertEqual(FakeEnvironment.reset_count, 256)
        for row in rows[:2]:
            with np.load(out / row["trace_path"], allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(diagnostic.TRACE_ARRAYS))
                self.assertEqual(len(archive["native_action"]), row["steps"])
                self.assertEqual(set(archive["repeat_index"].tolist()), {row["repeat"]})
                self.assertEqual(set(archive["actor_sha256"].tolist()), {row["actor_sha256"]})
                self.assertEqual(set(archive["checkpoint_sha256"].tolist()), {row["checkpoint_sha256"]})
                self.assertEqual(set(archive["source_actor_sha256"].tolist()), {row["source_actor_sha256"]})
                self.assertEqual(set(archive["source_checkpoint_sha256"].tolist()), {row["source_checkpoint_sha256"]})
        first_cell = rows[0]
        sibling = next(row for row in rows if row["cell_id"].rsplit(":repeat-", 1)[0]
                       == first_cell["cell_id"].rsplit(":repeat-", 1)[0] and row["repeat"] == 1)
        with np.load(out / first_cell["trace_path"], allow_pickle=False) as repeat0, np.load(
            out / sibling["trace_path"], allow_pickle=False,
        ) as repeat1:
            for name in diagnostic.TRACE_ARRAYS:
                if name != "repeat_index":
                    np.testing.assert_array_equal(repeat0[name], repeat1[name])
        manifest_sha = (out / "manifest.sha256").read_text().split()[0]
        self.assertEqual(manifest_sha, diagnostic.sha256_file(out / "manifest.json"))

    def test_blind_reused_and_wrong_exact_cells_reject_before_environment(self):
        for options, message in (
            ({"blind_seed": diagnostic.DIAGNOSTIC_SEEDS[0]}, "blind overlap"),
            ({"duplicate_run": True}, "duplicate candidate role"),
            ({"duplicate_geometry": True}, "reused geometry cell"),
            ({"diagnostic_seeds": list(diagnostic.DIAGNOSTIC_SEEDS[:-1]) + [999999]}, "wrong exact geometry IDs"),
        ):
            with self.subTest(message=message):
                inputs = self._inputs(**options)
                made = []

                def environment_factory(seed, steps):
                    made.append(seed)
                    return FakeEnvironment(seed, steps)

                with self.assertRaises(ValueError):
                    self._run(inputs, output_name=f"reject-{message.replace(' ', '-')}",
                              environment_factory=environment_factory)
                self.assertEqual(made, [])
                self.assertEqual(FakeEnvironment.reset_count, 0)

    def test_protocol_catalog_actor_and_checkpoint_hashes_fail_before_reset(self):
        inputs = self._inputs()
        protocol_path = inputs[0]
        original = protocol_path.read_bytes()
        protocol_path.write_bytes(original + b" ")
        made = []
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self._run(inputs, output_name="reject-mutated-protocol",
                      environment_factory=lambda seed, steps: made.append(seed))
        self.assertEqual(made, [])
        protocol_path.write_bytes(original)

        catalog_path = self.root / diagnostic.CATALOG_PATH
        original_catalog = catalog_path.read_bytes()
        catalog_path.write_bytes(original_catalog + b" ")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self._run(inputs, output_name="reject-mutated-catalog",
                      environment_factory=lambda seed, steps: made.append(seed))
        self.assertEqual(made, [])
        catalog_path.write_bytes(original_catalog)

        source_checkpoint = self.root / "runs/source-0/checkpoints/step-000131072/checkpoint.pt"
        original_source_checkpoint = source_checkpoint.read_bytes()
        source_checkpoint.write_bytes(original_source_checkpoint + b"changed")
        with self.assertRaisesRegex(ValueError, "source actor/checkpoint/manifest bytes changed"):
            self._run(inputs, output_name="reject-mutated-source-checkpoint",
                      environment_factory=lambda seed, steps: made.append(seed))
        self.assertEqual(made, [])
        source_checkpoint.write_bytes(original_source_checkpoint)

        run_dir = self.root / "runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-uniform"
        actor_path = run_dir / "checkpoints/step-000032768/actor.pt"
        original_actor = actor_path.read_bytes()
        actor_path.write_bytes(original_actor + b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self._run(inputs, output_name="reject-mutated-actor",
                      environment_factory=lambda seed, steps: made.append(seed))
        self.assertEqual(made, [])
        actor_path.write_bytes(original_actor)

        checkpoint_path = run_dir / "checkpoints/step-000032768/checkpoint.pt"
        original_checkpoint = checkpoint_path.read_bytes()
        checkpoint_path.write_bytes(original_checkpoint + b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self._run(inputs, output_name="reject-mutated-checkpoint",
                      environment_factory=lambda seed, steps: made.append(seed))
        self.assertEqual(made, [])
        checkpoint_path.write_bytes(original_checkpoint)

        payload_path = run_dir / "checkpoints/step-000032768/actor.pt"
        torch.save({"format": "unsupported", "config": {}, "state_dict": {}}, payload_path)
        new_actor_sha = diagnostic.sha256_file(payload_path)
        catalog_path = run_dir / "checkpoint-catalog.json"
        checkpoint_catalog = json.loads(catalog_path.read_text())
        checkpoint_catalog["candidates"][0]["actor_sha256"] = new_actor_sha
        _json(catalog_path, checkpoint_catalog)
        with self.assertRaisesRegex(ValueError, "unsupported or non-control"):
            self._run(inputs, output_name="reject-model-format",
                      environment_factory=lambda seed, steps: made.append(seed))
        self.assertEqual(made, [])

    def test_repeat_trace_disagreement_aborts_without_outcome_filtering(self):
        inputs = self._inputs()
        FakeEnvironment.diverge_on_repeat = True
        made = []

        def environment_factory(seed, steps):
            made.append(seed)
            return FakeEnvironment(seed, steps)

        with self.assertRaisesRegex(ValueError, "repeat trace diverged"):
            self._run(inputs, output_name="repeat-mismatch", environment_factory=environment_factory)
        self.assertEqual(made, [min(diagnostic.DIAGNOSTIC_SEEDS)])
        self.assertEqual(FakeEnvironment.reset_count, 2)
        self.assertFalse((self.root / "runs/repeat-mismatch/episodes.jsonl").exists())

    def test_fixed_diagnostic_boundaries_and_direct_twelve_hundred_time_limit(self):
        for change in (
            {"max_steps": 1199}, {"frame_skip": 3}, {"repeats": 1},
            {"track_ids": [1, 2]}, {"raw_reward": False}, {"partition": "confirmation"},
        ):
            inputs = self._inputs()
            protocol = json.loads(inputs[0].read_text())
            protocol["diagnostics"].update(change)
            protocol_sha = _json(inputs[0], protocol)
            with self.subTest(change=change):
                made = []
                with self.assertRaises(ValueError):
                    self._run((inputs[0], protocol_sha), output_name=f"boundary-{len(made)}",
                              environment_factory=lambda seed, steps: made.append(seed))
                self.assertEqual(made, [])

        calls = {}
        fake_train = ModuleType("train")

        def fake_haic_track(**kwargs):
            calls["constructor"] = kwargs
            return object()

        fake_train.HaicTrack = fake_haic_track

        def time_limit(raw, *, max_episode_steps):
            calls["raw"] = raw
            calls["max_episode_steps"] = max_episode_steps
            return raw

        with patch.dict(sys.modules, {"train": fake_train}), patch(
            "gymnasium.wrappers.TimeLimit", side_effect=time_limit,
        ):
            diagnostic.build_environment(7654321, 1200)
        self.assertEqual(calls["constructor"], {
            "track_id": 1, "seed": 7654321, "max_steps": 1200,
            "frame_skip": 4, "obstacles": True,
        })
        self.assertEqual(calls["max_episode_steps"], 1200)


if __name__ == "__main__":
    unittest.main()
