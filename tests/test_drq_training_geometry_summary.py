"""Synthetic-only tests for the sealed four-shard training geometry summary."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import diagnose_drq_training_geometry as diagnostic
from scripts import summarize_drq_training_geometry as summary


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return _hash(path)


class TrainingGeometrySummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ids = (3001, 3002, 3003, 3004)
        self.catalog_path = self.root / "runs" / "synthetic" / "catalog.json"
        self.catalog_path.parent.mkdir(parents=True)
        self.protocol_path = self.root / "experiments" / "synthetic-protocol.json"
        self.protocol_path.parent.mkdir()
        self.diagnostics = self.root / "runs" / "synthetic" / "diagnostics"
        self.diagnostics.mkdir()
        self.output = self.root / "experiments" / "training-summary.json"
        self.actors = []
        self.source_hashes = {}
        for source in (0, 1):
            actor_path = self.root / "runs" / "synthetic" / f"actor-{source}.pt"
            actor_path.write_bytes(f"fake-source-{source}".encode("ascii"))
            digest = _hash(actor_path)
            weight_digest = hashlib.sha256(f"fake-weights-{source}".encode()).hexdigest()
            self.source_hashes[source] = (digest, weight_digest)
            self.actors.append({
                "learner_seed": source, "actor_path": actor_path.relative_to(self.root).as_posix(),
                "actor_sha256": digest, "actor_weights_sha256": weight_digest,
            })
        catalog_rows = []
        for index, seed in enumerate(self.ids):
            signature = {"turn_sequence": [index + 0.1], "perimeter_m": 100 + index}
            signature_sha = hashlib.sha256(json.dumps(
                signature, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            catalog_rows.append({
                "geometry_seed": seed, "track_id": 1,
                "family": "early-bend" if index < 2 else "tight-turn", "direction": "left",
                "stage": "diagnostic" if index == 3 else "representative",
                "road_coordinate_sha256": hashlib.sha256(f"road-{seed}".encode()).hexdigest(),
                "signature_sha256": signature_sha, "cyclic_signature_sha256": signature_sha,
                "signature": signature, "features": {"signature_sha256": signature_sha, "signature": signature},
                "structure": {"centerline_connected": True, "centerline_self_intersections": 0,
                              "finish_static_only": True, "nonadjacent_road_overlap_warning_pairs": 3},
                "verification": {"regenerated_coordinate_hash_match": True, "raw_reset": True,
                                 "one_valid_raw_step": True, "fresh_tracker_after_reset": True,
                                 "finish_plausibility": "static_only_not_a_successful_lap"},
            })
        proposed = list(self.ids) + list(range(10000, 10256))
        catalog = {
            "format": diagnostic.CATALOG_FORMAT, "protocol_sha256": "a" * 64,
            "analysis_sha256": "b" * 64, "scan_path": "runs/synthetic/scan.json", "scan_sha256": "c" * 64,
            "train": catalog_rows[:3], "train_diagnostic": catalog_rows[3:],
            "seed_audit": {
                "format": "haic-drq-training-seed-audit-v1", "passed": True,
                "matched_collisions": [], "proposed_seeds": proposed,
                "proposed_seeds_sha256": hashlib.sha256(json.dumps(proposed, separators=(",", ":")).encode()).hexdigest(),
                "blind_data_access": "none; protocol blind seed IDs are exclusion-only",
            },
        }
        self.catalog_sha = _dump(self.catalog_path, catalog)
        protocol = {
            "format": diagnostic.PROTOCOL_FORMAT, "role": "training_diagnostic",
            "score_selection": False, "frame_skip": 4, "max_steps": 4, "raw_reward": True,
            "obstacles": True, "source_actors": self.actors,
            "catalog_path": self.catalog_path.relative_to(self.root).as_posix(),
            "catalog_sha256": self.catalog_sha, "catalog_protocol_sha256": "a" * 64,
            "catalog_scan_path": catalog["scan_path"], "catalog_scan_sha256": catalog["scan_sha256"],
            "reserved_training_seeds": [9000, 9001],
            "partitions": {
                "train": {"seeds": list(self.ids[:3])},
                "train_diagnostic": {"seeds": [self.ids[3]]},
                "blind": {"seeds": [9000]}, "confirmation": {"seeds": [9001]},
            },
        }
        self.protocol_sha = _dump(self.protocol_path, protocol)
        self.rows = {item["geometry_seed"]: {**item, "partition": (
            "train_diagnostic" if item["geometry_seed"] == self.ids[3] else "train"
        )} for item in catalog_rows}
        for index, seed in enumerate(self.ids):
            shard = self.shard(index)
            traces = shard / "traces"
            traces.mkdir(parents=True)
            cells = [self._cell(self.rows[seed], source, shard) for source in (0, 1)]
            (shard / "cells.jsonl").write_text(
                "".join(json.dumps(cell, sort_keys=True) + "\n" for cell in cells), encoding="utf-8",
            )
            inventory = {cell["trace_path"]: cell["trace_sha256"] for cell in cells}
            inventory["cells.jsonl"] = _hash(shard / "cells.jsonl")
            manifest = {
                "format": diagnostic.MANIFEST_FORMAT, "role": "training_diagnostic",
                "ranked": False, "score_selection": False,
                "performance_scope": "internal training-only", "created_at_utc": "2026-09-25T00:00:00Z",
                "catalog_path": protocol["catalog_path"], "catalog_sha256": self.catalog_sha,
                "catalog_protocol_sha256": catalog["protocol_sha256"],
                "analysis_sha256": catalog["analysis_sha256"], "scan_path": catalog["scan_path"],
                "scan_sha256": catalog["scan_sha256"],
                "protocol_path": self.protocol_path.relative_to(self.root).as_posix(),
                "protocol_sha256": self.protocol_sha, "source_actors": self.actors,
                "shard_index": index, "shard_count": 4, "geometry_ids": [seed],
                "cell_count": 2, "frame_skip": 4, "max_steps": 4,
                "sparse_stride": diagnostic.SPARSE_STRIDE,
                "schema": {"cell": diagnostic.RESULT_FORMAT,
                           "trace_arrays": {name: "synthetic test array" for name in summary.TRACE_DTYPES}},
                "file_sha256": inventory,
            }
            _dump(shard / "manifest.json", manifest)
            self._seal(index)

    def shard(self, index: int) -> Path:
        return self.diagnostics / f"shard-{index:04d}-of-0004"

    def _seal(self, index: int) -> None:
        manifest = self.shard(index) / "manifest.json"
        (self.shard(index) / "manifest.sha256").write_text(
            f"{_hash(manifest)}  manifest.json\n", encoding="ascii",
        )

    def _cells(self, index: int) -> list[dict]:
        return [json.loads(line) for line in (self.shard(index) / "cells.jsonl").read_text().splitlines()]

    def _replace_cells(self, index: int, cells: list[dict]) -> None:
        shard = self.shard(index)
        (shard / "cells.jsonl").write_text("".join(json.dumps(cell, sort_keys=True) + "\n" for cell in cells))
        manifest = json.loads((shard / "manifest.json").read_text())
        manifest["file_sha256"]["cells.jsonl"] = _hash(shard / "cells.jsonl")
        _dump(shard / "manifest.json", manifest)
        self._seal(index)

    def _cell(self, row: dict, source: int, shard: Path) -> dict:
        seed = row["geometry_seed"]
        finished = seed == self.ids[0] or (seed == self.ids[1] and source == 1)
        terminal_progress = {self.ids[0]: 1.0, self.ids[1]: 0.06,
                             self.ids[2]: 0.28, self.ids[3]: 0.08}[seed]
        if finished:
            terminal_progress = 1.0
        reward = np.asarray([-1, -1, 1, 1], dtype=np.float32)
        progress = np.asarray([0.02, 0.03, 0.04, terminal_progress], dtype=np.float32)
        damage = np.asarray([0, 0, 0.1, 0.1], dtype=np.float32)
        steer = -0.2 if source == 0 else 0.3
        official = np.tile([steer, 0.4, 0], (4, 1)).astype(np.float32)
        native = np.tile([steer, -0.2, -1], (4, 1)).astype(np.float32)
        fraction = np.asarray([0, 0.01, terminal_progress / 2], dtype=np.float32)
        arrays = {
            "reward": reward, "progress": progress, "damage": damage,
            "native_action": native, "official_action": official,
            "collision_action": np.asarray([False, False, True, False]),
            "off_track_counter": np.asarray([1, 2, 0, 0], dtype=np.int16),
            "sparse_step": np.asarray([0, 1, 4], dtype=np.int32),
            "sparse_nearest_point_index": np.asarray([0, 1, 5], dtype=np.int32),
            "sparse_nearest_point_fraction": fraction,
            "sparse_nearest_distance_m": np.asarray([0.1, 0.1, 0.2], dtype=np.float32),
            "sparse_speed_m_s": np.asarray([0, 1, 2], dtype=np.float32),
        }
        trace_path = f"traces/seed-{seed}-source-{source}.npz"
        np.savez_compressed(shard / trace_path, **arrays)
        sparse = [{
            "step": int(arrays["sparse_step"][index]),
            "nearest_point_index": int(arrays["sparse_nearest_point_index"][index]),
            "nearest_point_fraction": float(fraction[index]),
            "nearest_distance_m": float(arrays["sparse_nearest_distance_m"][index]),
            "speed_m_s": float(arrays["sparse_speed_m_s"][index]),
        } for index in range(3)]

        def axes(values):
            return {name: {
                "mean": float(values[:, index].mean()), "std": float(values[:, index].std()),
                "min": float(values[:, index].min()), "max": float(values[:, index].max()),
                "near_boundary_fraction": 0.0,
            } for index, name in enumerate(("steer", "gas", "brake"))}

        return {
            "format": diagnostic.RESULT_FORMAT, "role": "training_diagnostic", "ranked": False,
            "partition": row["partition"], "geometry_seed": seed, "track_id": 1,
            "geometry_id": str(seed), "cell_id": f"train-geometry:{row['partition']}:seed-{seed}:track-1:source-{source}",
            "source_learner_seed": source, "source_actor_sha256": self.actors[source]["actor_sha256"],
            "source_actor_weights_sha256": self.actors[source]["actor_weights_sha256"],
            "road_coordinate_sha256": row["road_coordinate_sha256"],
            "signature_sha256": row["signature_sha256"], "family": row["family"],
            "stage": row["stage"], "steps": 4, "max_steps": 4,
            "raw_reward_sum": float(reward.sum()), "finished": finished,
            "terminal_progress": float(progress[-1]), "max_progress": float(progress.max()),
            "terminal_damage": float(damage[-1]), "max_damage": float(damage.max()),
            "collision_actions": 1, "collision_semantics": summary.COLLISION_SEMANTICS,
            "terminated": False, "truncated": True, "terminal": finished,
            "retire_reason": None, "termination_class": "finished" if finished else "timeout",
            "timeout_at_max_steps": not finished,
            "off_track_counter_final": 0, "off_track_counter_max": 2,
            "off_track_semantics": summary.OFFTRACK_SEMANTICS,
            "official_action_axes": axes(official), "native_action_axes": axes(native),
            "steering_throttle_time_bins": [], "sparse_road_samples": sparse,
            "first_road_sample": sparse[0], "last_road_sample": sparse[-1],
            "finish_line": {"crossed": finished, "qualified": finished,
                            "crossing_time_s": 0.16 if finished else None,
                            "qualified_time_s": 0.14 if finished else None},
            "lap_time_ms": 160 if finished else None,
            "trace_path": trace_path, "trace_sha256": _hash(shard / trace_path),
            "catalog_sha256": self.catalog_sha, "protocol_sha256": self.protocol_sha,
        }

    def _summarize(self, **changes):
        inputs = {
            "root": self.root, "catalog_path": self.catalog_path, "catalog_sha256": self.catalog_sha,
            "protocol_path": self.protocol_path, "protocol_sha256": self.protocol_sha,
            "diagnostics_root": self.diagnostics, "output": self.output,
        }
        inputs.update(changes)
        with ExitStack() as stack:
            stack.enter_context(patch.object(summary, "CATALOG_PATH", self.catalog_path.relative_to(self.root).as_posix()))
            stack.enter_context(patch.object(summary, "CATALOG_SHA256", self.catalog_sha))
            stack.enter_context(patch.object(summary, "PROTOCOL_PATH", self.protocol_path.relative_to(self.root).as_posix()))
            stack.enter_context(patch.object(summary, "PROTOCOL_SHA256", self.protocol_sha))
            stack.enter_context(patch.object(summary, "DIAGNOSTICS_ROOT", self.diagnostics.relative_to(self.root).as_posix()))
            stack.enter_context(patch.object(diagnostic, "TRAIN_TARGET", 3))
            stack.enter_context(patch.object(diagnostic, "DIAGNOSTIC_TARGET", 1))
            stack.enter_context(patch.dict(diagnostic.SOURCE_HASHES, self.source_hashes, clear=True))
            return summary.summarize(**inputs)

    def assert_rejected(self, message: str) -> None:
        with self.assertRaisesRegex((ValueError, OSError), message):
            self._summarize()
        self.assertFalse(self.output.exists(), "partial summary must never be written")

    def test_complete_paired_summary_keeps_difficult_roads(self):
        result = self._summarize()
        self.assertEqual(result["geometry_count"], 4)
        self.assertEqual(result["cell_count"], 8)
        self.assertFalse(result["ranked"])
        self.assertEqual(result["pathological_count"], 0)
        self.assertEqual([item["finishes"] for item in result["geometry"]], [2, 1, 0, 0])
        self.assertEqual([item["provisional_difficulty"] for item in result["geometry"]], [
            "too_easy", "useful_boundary", "difficult_but_learnable", "unresolved_difficult",
        ])
        self.assertEqual(result["geometry"][-1]["finish_reachability"], "unknown_no_finish_observed")
        self.assertEqual(result["geometry"][1]["source_contrast"]["source1_only_finishes"], 1)
        self.assertEqual(result["families"]["tight-turn"]["provisional_difficulty_counts"], {
            "difficult_but_learnable": 1, "unresolved_difficult": 1,
        })
        self.assertEqual(result["families"]["tight-turn"]["partition_geometry_counts"], {
            "train": 1, "train_diagnostic": 1,
        })
        self.assertAlmostEqual(result["families"]["early-bend"]["by_source"]["1"]["mean_of_cell_action_means"][
            "official_action_axes"]["steer"], 0.3)
        self.assertIn("nearest_point_fraction", result["geometry"][-1]["by_source"]["0"]["last_road_sample"])
        self.assertEqual(json.loads(self.output.read_text())["difficulty_counts"], result["difficulty_counts"])
        with self.assertRaises(FileExistsError):
            self._summarize()

    def test_requires_exact_frozen_sha_pins_and_paths(self):
        with self.assertRaisesRegex(ValueError, "catalog hash"):
            self._summarize(catalog_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "protocol hash"):
            self._summarize(protocol_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "catalog path"):
            self._summarize(catalog_path=self.root / "runs" / "different.json")
        with self.assertRaisesRegex(ValueError, "diagnostics root"):
            self._summarize(diagnostics_root=self.root / "runs" / "different")
        original = self.protocol_path.read_bytes()
        self.protocol_path.write_bytes(original + b" ")
        self.assert_rejected("protocol is empty or changed")

    def test_missing_manifest_and_missing_or_changed_seal(self):
        (self.shard(0) / "manifest.json").unlink()
        self.assert_rejected("incomplete")

    def test_static_failure_is_invalid_not_actor_based_pathology(self):
        self.rows[self.ids[0]]["structure"]["centerline_connected"] = False
        with self.assertRaisesRegex(ValueError, "static/sanity invalid"):
            summary._static_sanity(list(self.rows.values()))
        self.rows[self.ids[0]]["structure"]["centerline_connected"] = True
        summary._static_sanity(list(self.rows.values()))

    def test_changed_seal_and_missing_trace_fail_closed(self):
        seal = self.shard(0) / "manifest.sha256"
        original = seal.read_bytes()
        seal.write_bytes(b"0" * 64 + b"  manifest.json\n")
        self.assert_rejected("manifest.sha256")
        seal.write_bytes(original)
        (self.shard(1) / "traces" / f"seed-{self.ids[1]}-source-1.npz").unlink()
        self.assert_rejected("extra/unsealed")

    def test_changed_manifest_without_resealing_is_rejected(self):
        manifest = self.shard(0) / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        self.assert_rejected("manifest.sha256")

    def test_changed_trace_even_with_unchanged_row_fails_inventory(self):
        trace = self.shard(0) / "traces" / f"seed-{self.ids[0]}-source-0.npz"
        trace.write_bytes(trace.read_bytes() + b"corrupt")
        self.assert_rejected("inventory file changed")

    def test_changed_trace_even_when_inventory_and_row_resealed_fails_semantics(self):
        trace = self.shard(0) / "traces" / f"seed-{self.ids[0]}-source-0.npz"
        with np.load(trace, allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in npz.files}
        arrays["off_track_counter"][1] = 0
        np.savez_compressed(trace, **arrays)
        cells = self._cells(0)
        cells[0]["trace_sha256"] = _hash(trace)
        self._replace_cells(0, cells)
        manifest = json.loads((self.shard(0) / "manifest.json").read_text())
        manifest["file_sha256"][cells[0]["trace_path"]] = _hash(trace)
        _dump(self.shard(0) / "manifest.json", manifest)
        self._seal(0)
        self.assert_rejected("negative-reward decisions")

    def test_wrong_actor_seed_and_duplicates_even_after_resealing(self):
        for key, value, error in (
            ("source_learner_seed", 9, "seed/actor"),
            ("geometry_seed", self.ids[1], "seed/actor"),
            ("cell_id", "duplicate", "cell cell_id"),
        ):
            with self.subTest(field=key):
                cells = self._cells(0)
                original = cells[0][key]
                cells[0][key] = value
                self._replace_cells(0, cells)
                self.assert_rejected(error)
                cells[0][key] = original
                self._replace_cells(0, cells)
        cells = self._cells(0)
        cells[1] = dict(cells[0])
        self._replace_cells(0, cells)
        self.assert_rejected("duplicate cell")

    def test_actor_provenance_must_match_export_and_cell(self):
        actor = self.root / self.actors[1]["actor_path"]
        actor.write_bytes(b"changed-export")
        self.assert_rejected("source actor export changed")

    def test_resealed_manifest_provenance_cannot_change(self):
        manifest = json.loads((self.shard(2) / "manifest.json").read_text())
        manifest["analysis_sha256"] = "f" * 64
        _dump(self.shard(2) / "manifest.json", manifest)
        self._seal(2)
        self.assert_rejected("analysis_sha256")

    def test_blind_and_leaked_partitions_are_excluded_without_reading_blind_artifacts(self):
        cells = self._cells(0)
        cells[0]["geometry_seed"] = 9000
        cells[0]["partition"] = "blind"
        self._replace_cells(0, cells)
        self.assert_rejected("seed/actor")
        cells = self._cells(0)
        cells[0]["geometry_seed"] = self.ids[0]
        self._replace_cells(0, cells)
        self.assert_rejected("cell partition")

    def test_extra_uninventoried_file_and_partial_shard_fail_closed(self):
        (self.shard(1) / "traces" / "blind-road.npz").write_bytes(b"never-read")
        self.assert_rejected("extra/unsealed")
        (self.shard(1) / "traces" / "blind-road.npz").unlink()
        (self.shard(3) / "manifest.sha256").unlink()
        self.assert_rejected("incomplete")


if __name__ == "__main__":
    unittest.main()
