"""Synthetic-only training geometry diagnostics; never opens an official road."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter, ObservationSpec, file_sha256
from scripts import diagnose_drq_training_geometry as diagnostic


def _dump(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return file_sha256(path)


class FakeActor:
    def __init__(self, value: float, *, nondeterministic: bool = False):
        self.weight = torch.tensor([value], dtype=torch.float32)
        self.nondeterministic = nondeterministic

    def state_dict(self):
        return {"layer.weight": self.weight}

    def act(self, observation, deterministic=True):
        if deterministic is not True:
            raise AssertionError("exploratory actor call")
        steer = float(self.weight[0]) / 4
        return np.asarray([steer + (0.01 if self.nondeterministic else 0.0), -0.2, -1.0],
                          dtype=np.float32)


class FakeEnvironment:
    made: list["FakeEnvironment"] = []
    resets = 0
    mode = "timeout"

    @staticmethod
    def track_for_seed(seed):
        angle = np.arange(32, dtype=np.float64) * (2 * np.pi / 32)
        return [(float(theta), 0.0, float((10 + seed * 0.001) * np.cos(theta)),
                 float((10 + seed * 0.001) * np.sin(theta))) for theta in angle]

    def __init__(self, track_id, seed, max_steps, frame_skip, reward_shaping,
                 obstacles, collision_penalty):
        if (frame_skip, reward_shaping, obstacles, collision_penalty) != (4, False, True, 0.0):
            raise AssertionError("modified environment contract")
        self.track_id, self.track_seed, self.max_steps = track_id, seed, max_steps
        self.unwrapped = self
        self.step_number = 0
        self.off_track_counter = 0
        self.max_off_track_steps = 100
        self.t = 0.0
        self.track = self.track_for_seed(seed)
        self.car = SimpleNamespace(hull=SimpleNamespace(position=(self.track[0][2], self.track[0][3]),
                                                  linearVelocity=(3.0, 0.0)))
        self.finish_line_tracker = SimpleNamespace(
            departed_start_area=False, crossing_from_back=False,
            candidate_crossing_time_s=None, previous_longitudinal=0.0,
            previous_lateral=0.0,
        )
        self.closed = False
        self.made.append(self)

    def reset(self, *, seed=None, options=None):
        FakeEnvironment.resets += 1
        self.step_number = 0
        self.off_track_counter = 0
        return np.zeros((4, 84, 84), dtype=np.float32), {
            "seed": self.track_seed, "track_id": self.track_id,
        }

    def step(self, official_action):
        if np.asarray(official_action).shape != (3,):
            raise AssertionError("official action has the wrong shape")
        np.testing.assert_allclose(np.asarray(official_action)[1:], [0.4, 0.0])
        self.step_number += 1
        self.t += 0.04
        reward = -1.0 if self.step_number <= 2 else 1.0
        self.off_track_counter = self.off_track_counter + 1 if reward < 0 else 0
        self.car.hull.position = (self.track[self.step_number % len(self.track)][2],
                                  self.track[self.step_number % len(self.track)][3])
        self.car.hull.linearVelocity = (3.0 + self.step_number, 0.0)
        mode = FakeEnvironment.mode
        finished = mode == "finish" and self.step_number == 3
        retire = mode if mode in {"crash", "off_track"} and self.step_number == 3 else None
        terminated = retire is not None
        truncated = finished or self.step_number >= self.max_steps
        self.finish_line_tracker.departed_start_area = True
        if finished:
            self.finish_line_tracker.previous_longitudinal = 0.5
        return np.zeros((4, 84, 84), dtype=np.float32), reward, terminated, truncated, {
            "seed": self.track_seed, "track_id": self.track_id,
            "progress": 1.0 if self.step_number == 3 else 0.1 * self.step_number,
            "damage": float(self.step_number == 2),
            "collision": self.step_number == 2,
            "retire_reason": retire,
            "finished": finished,
            "finish_qualified": finished,
            "finish_qualified_time_s": self.t - 0.02 if finished else None,
            "finish_time_s": self.t if finished else None,
        }

    def close(self):
        self.closed = True


def _fake_loader(path: Path, *, device: str):
    if device != "cpu":
        raise AssertionError("actor was not reloaded on CPU")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = FakeActor(float(payload["state_dict"]["layer.weight"][0]))
    return actor, ActionAdapter(), ObservationSpec()


class DrQTrainingGeometryDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("runs", "experiments"):
            (self.root / name).mkdir()
        FakeEnvironment.made = []
        FakeEnvironment.resets = 0
        FakeEnvironment.mode = "timeout"
        self.ids = [3001, 3002, 3003]
        self.source_hashes = {}
        self.sources = []
        for learner_seed in (0, 1):
            actor_path = self.root / "runs" / f"source-{learner_seed}" / "actor.pt"
            actor_path.parent.mkdir()
            actor = FakeActor(float(learner_seed))
            torch.save({"format": "haic-drq-v2-actor-v1", "config": {
                "augmentation_pad": 4, "steering_logit_l2": 0.0,
            }, "state_dict": actor.state_dict()}, actor_path)
            export_sha = file_sha256(actor_path)
            weight_sha = diagnostic._weights_sha256(actor)
            self.source_hashes[learner_seed] = (export_sha, weight_sha)
            self.sources.append({
                "learner_seed": learner_seed,
                "actor_path": str(actor_path.relative_to(self.root)),
                "actor_sha256": export_sha,
                "actor_weights_sha256": weight_sha,
            })

    def _payloads(self):
        rows = []
        for index, seed in enumerate(self.ids):
            signature = {
                "turn_sequence": [0.1], "relative_width_sequence": [1.0],
                "perimeter_m": float(seed), "width_median_m": 40.0 / 3.0,
            }
            signature_sha = hashlib.sha256(json.dumps(
                signature, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            rows.append({
                "geometry_seed": seed, "track_id": 1,
                "family": "synthetic-bend", "direction": "left",
                "stage": "representative" if index < 2 else "diagnostic",
                "road_coordinate_sha256": diagnostic._road_sha256(FakeEnvironment.track_for_seed(seed)),
                "signature_sha256": signature_sha,
                "cyclic_signature_sha256": signature_sha,
                "signature": signature,
                "features": {"perimeter_m": 10, "signature_sha256": signature_sha, "signature": signature},
                "structure": {"centerline_connected": True},
                "verification": {"regenerated_coordinate_hash_match": True},
            })
        proposed = list(range(3100, 3356)) + self.ids
        proposed_sha = hashlib.sha256(json.dumps(proposed, separators=(",", ":")).encode()).hexdigest()
        catalog = {
            "format": diagnostic.CATALOG_FORMAT,
            "protocol_sha256": "a" * 64, "analysis_sha256": "b" * 64,
            "scan_path": "runs/synthetic/scan.json", "scan_sha256": "c" * 64,
            "train": rows[:2], "train_diagnostic": rows[2:],
            "seed_audit": {
                "format": "haic-drq-training-seed-audit-v1", "passed": True,
                "proposed_seeds": proposed, "proposed_seeds_sha256": proposed_sha,
                "matched_collisions": [],
                "blind_data_access": "none; protocol blind seed IDs are exclusion-only",
            },
        }
        protocol = {
            "format": diagnostic.PROTOCOL_FORMAT,
            "catalog_protocol_sha256": "a" * 64,
            "role": "training_diagnostic", "score_selection": False,
            "frame_skip": 4, "max_steps": 4,
            "raw_reward": True, "obstacles": True,
            "source_actors": self.sources,
            "reserved_training_seeds": [9999, 9998],
            "partitions": {
                "train": {"seeds": self.ids[:2]},
                "train_diagnostic": {"seeds": self.ids[2:]},
                "blind": {"seeds": [9999]},
                "confirmation": {"seeds": [9998]},
            },
        }
        return catalog, protocol

    def _inputs(self, catalog=None, protocol=None):
        default_catalog, default_protocol = self._payloads()
        catalog = default_catalog if catalog is None else catalog
        protocol = default_protocol if protocol is None else protocol
        catalog_path = self.root / "runs" / "catalog.json"
        catalog_sha = _dump(catalog_path, catalog)
        protocol["catalog_sha256"] = catalog_sha
        protocol_path = self.root / "experiments" / "diagnostic.json"
        protocol_sha = _dump(protocol_path, protocol)
        return catalog_path, catalog_sha, protocol_path, protocol_sha

    def _run(self, inputs, *, name="diagnostics", shard_index=0, shard_count=1):
        return diagnostic.run_shard(
            root=self.root, catalog_path=inputs[0], catalog_sha256=inputs[1],
            protocol_path=inputs[2], protocol_sha256=inputs[3],
            output_root=self.root / "runs" / name,
            shard_index=shard_index, shard_count=shard_count,
        )

    def _patch(self, *, loader=_fake_loader):
        stack = ExitStack()
        stack.enter_context(patch.dict(diagnostic.SOURCE_HASHES, self.source_hashes, clear=True))
        stack.enter_context(patch.object(diagnostic, "TRAIN_TARGET", 2))
        stack.enter_context(patch.object(diagnostic, "DIAGNOSTIC_TARGET", 1))
        stack.enter_context(patch.object(diagnostic, "build_env", FakeEnvironment))
        stack.enter_context(patch.object(diagnostic, "load_exported_actor", loader))
        return stack

    def test_canonical_seed_lineage_and_compressed_traces(self):
        inputs = self._inputs()
        with self._patch():
            manifest = self._run(inputs)
        self.assertEqual(manifest["geometry_ids"], self.ids)
        self.assertEqual(manifest["cell_count"], 6)
        self.assertEqual(manifest["catalog_sha256"], inputs[1])
        self.assertEqual(manifest["protocol_sha256"], inputs[3])
        folder = self.root / "runs" / "diagnostics" / "shard-0000-of-0001"
        self.assertEqual((folder / "manifest.sha256").read_text().split()[0],
                         file_sha256(folder / "manifest.json"))
        lines = [json.loads(line) for line in (folder / "cells.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 6)
        self.assertEqual(len({item["cell_id"] for item in lines}), 6)
        for result in lines:
            self.assertEqual(result["role"], "training_diagnostic")
            self.assertFalse(result["ranked"])
            self.assertEqual(result["geometry_id"], str(result["geometry_seed"]))
            self.assertIn(f"source-{result['source_learner_seed']}", result["cell_id"])
            self.assertEqual(result["source_actor_sha256"], self.source_hashes[result["source_learner_seed"]][0])
            self.assertEqual(result["trace_sha256"], manifest["file_sha256"][result["trace_path"]])
            self.assertEqual(result["collision_actions"], 1)
            self.assertEqual(result["max_progress"], 1.0)
            self.assertEqual(result["terminal_progress"], 0.4)
            self.assertEqual(result["first_road_sample"]["step"], 0)
            self.assertEqual(result["last_road_sample"]["step"], 4)
            with np.load(folder / result["trace_path"], allow_pickle=False) as trace:
                self.assertEqual(trace["reward"].shape, (4,))
                np.testing.assert_array_equal(trace["sparse_step"], [0, 1, 4])
                np.testing.assert_allclose(trace["official_action"][:, 1], 0.4)
                np.testing.assert_allclose(trace["native_action"][:, 1], -0.2)
                np.testing.assert_array_equal(trace["off_track_counter"], [1, 2, 0, 0])
        self.assertTrue(all(env.closed for env in FakeEnvironment.made))

    def test_cpu_reload_is_deterministic_and_run_results_repeat(self):
        inputs = self._inputs()
        with self._patch():
            first = self._run(inputs, name="first")
            second = self._run(inputs, name="second")
        self.assertEqual(first["geometry_ids"], second["geometry_ids"])
        one = (self.root / "runs" / "first" / "shard-0000-of-0001" / "cells.jsonl").read_bytes()
        two = (self.root / "runs" / "second" / "shard-0000-of-0001" / "cells.jsonl").read_bytes()
        self.assertEqual(one, two)

        load_count = [0]

        def nondeterministic_reload(path, *, device):
            load_count[0] += 1
            actor, adapter, spec = _fake_loader(path, device=device)
            actor.nondeterministic = load_count[0] % 2 == 0
            return actor, adapter, spec

        FakeEnvironment.resets = 0
        with self._patch(loader=nondeterministic_reload):
            with self.assertRaisesRegex(ValueError, "nondeterministic"):
                self._run(inputs, name="reload-mismatch")
        self.assertEqual(FakeEnvironment.resets, 0)

    def test_all_partitions_are_validated_before_any_reset(self):
        for alteration, message in (
            (lambda c, p: p["partitions"].update(blond={"seeds": [456]}), "unknown"),
            (lambda c, p: p["partitions"]["train"]["seeds"].append(456), "differs"),
            (lambda c, p: c["train_diagnostic"].append(dict(c["train"][0])), "duplicate"),
            (lambda c, p: c["train"][0].update(geometry_seed=9999), "reserved"),
            (lambda c, p: c["train"][0].update(partition="blind"), "row-blind"),
            (lambda c, p: c.update(blind=[c["train"][0]]), "partition"),
            (lambda c, p: p["reserved_training_seeds"].remove(9999), "blind"),
        ):
            with self.subTest(message=message):
                catalog, protocol = self._payloads()
                alteration(catalog, protocol)
                inputs = self._inputs(catalog, protocol)
                FakeEnvironment.resets = 0
                with self._patch():
                    with self.assertRaises(ValueError):
                        self._run(inputs, name=f"reject-{message}")
                self.assertEqual(FakeEnvironment.resets, 0)

    def test_changed_empty_actor_protocol_and_catalog_fail_before_reset(self):
        inputs = self._inputs()
        with self._patch():
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self._run((inputs[0], "", inputs[2], inputs[3]), name="missing-catalog-hash")
            self.assertEqual(FakeEnvironment.resets, 0)
            for index, contents in ((0, b""), (2, b"{}")):
                with self.subTest(index=index):
                    original = inputs[index].read_bytes()
                    inputs[index].write_bytes(contents)
                    try:
                        with self.assertRaises(ValueError):
                            self._run(inputs, name=f"changed-{index}")
                        self.assertEqual(FakeEnvironment.resets, 0)
                    finally:
                        inputs[index].write_bytes(original)
            source = self.root / self.sources[0]["actor_path"]
            original = source.read_bytes()
            source.write_bytes(b"corrupt")
            try:
                with self.assertRaisesRegex(ValueError, "export changed"):
                    self._run(inputs, name="changed-source")
                self.assertEqual(FakeEnvironment.resets, 0)
            finally:
                source.write_bytes(original)

    def test_actor_weights_and_pad4_must_match_before_any_reset(self):
        for mutate in (
            lambda p: p["source_actors"][0].update(actor_weights_sha256="f" * 64),
            lambda p: p["source_actors"][0].update(actor_sha256="f" * 64),
        ):
            catalog, protocol = self._payloads()
            mutate(protocol)
            inputs = self._inputs(catalog, protocol)
            with self._patch():
                with self.assertRaises(ValueError):
                    self._run(inputs, name="reject-source")
            self.assertEqual(FakeEnvironment.resets, 0)

        source = self.root / self.sources[0]["actor_path"]
        payload = torch.load(source, map_location="cpu", weights_only=False)
        payload["config"]["augmentation_pad"] = 1
        torch.save(payload, source)
        self.sources[0]["actor_sha256"] = file_sha256(source)
        self.source_hashes[0] = (self.sources[0]["actor_sha256"], self.source_hashes[0][1])
        inputs = self._inputs()
        with self._patch():
            with self.assertRaisesRegex(ValueError, "pad-4"):
                self._run(inputs, name="reject-pad1")
        self.assertEqual(FakeEnvironment.resets, 0)

    def test_reset_road_hash_mismatch_closes_env_without_actor_step(self):
        inputs = self._inputs()

        class WrongRoad(FakeEnvironment):
            def reset(self, *, seed=None, options=None):
                observation, info = super().reset(seed=seed, options=options)
                self.track[0] = (*self.track[0][:2], 100.0, 100.0)
                return observation, info

        with self._patch(), patch.object(diagnostic, "build_env", WrongRoad):
            with self.assertRaisesRegex(ValueError, "road coordinates differ"):
                self._run(inputs, name="reject-road-hash")
        self.assertEqual(FakeEnvironment.resets, 1)
        self.assertTrue(FakeEnvironment.made[-1].closed)
        self.assertEqual(FakeEnvironment.made[-1].step_number, 0)

    def test_reset_refuses_uncataloged_seed_before_actor_step(self):
        inputs = self._inputs()

        class WrongSeed(FakeEnvironment):
            def reset(self, *, seed=None, options=None):
                observation, info = super().reset(seed=seed, options=options)
                return observation, {**info, "seed": 9999}

        with self._patch(), patch.object(diagnostic, "build_env", WrongSeed):
            with self.assertRaisesRegex(ValueError, "outside the frozen catalog"):
                self._run(inputs, name="reject-reset-seed")
        self.assertEqual(FakeEnvironment.resets, 1)
        self.assertTrue(FakeEnvironment.made[-1].closed)
        self.assertEqual(FakeEnvironment.made[-1].step_number, 0)

    def test_shards_are_disjoint_and_existing_shard_is_immutable(self):
        inputs = self._inputs()
        with self._patch():
            first = self._run(inputs, shard_index=0, shard_count=2)
            second = self._run(inputs, shard_index=1, shard_count=2)
            with self.assertRaises(FileExistsError):
                self._run(inputs, shard_index=0, shard_count=2)
        self.assertFalse(set(first["geometry_ids"]) & set(second["geometry_ids"]))
        self.assertEqual(set(first["geometry_ids"]) | set(second["geometry_ids"]), set(self.ids))
        self.assertEqual(first["cell_count"] + second["cell_count"], 2 * len(self.ids))
        self.assertEqual(FakeEnvironment.resets, 2 * len(self.ids))

    def test_timeout_crash_offtrack_and_qualified_finish_are_distinct(self):
        inputs = self._inputs()
        with self._patch():
            for mode in ("timeout", "crash", "off_track", "finish"):
                FakeEnvironment.mode = mode
                self._run(inputs, name=mode)
                folder = self.root / "runs" / mode / "shard-0000-of-0001"
                result = json.loads((folder / "cells.jsonl").read_text().splitlines()[0])
                self.assertEqual(result["termination_class"], "finished" if mode == "finish" else mode)
                self.assertEqual(result["timeout_at_max_steps"], mode == "timeout")
                self.assertEqual(result["steps"], 4 if mode == "timeout" else 3)
                self.assertEqual(result["finish_line"]["qualified"], mode == "finish")
                self.assertEqual(result["finish_line"]["crossed"], mode == "finish")
                if mode == "finish":
                    self.assertIsNotNone(result["lap_time_ms"])
                else:
                    self.assertIsNone(result["lap_time_ms"])


if __name__ == "__main__":
    unittest.main()
