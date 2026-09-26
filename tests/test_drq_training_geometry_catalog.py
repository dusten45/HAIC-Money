from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from haic.algorithms.drq_v2.geometry_features import describe_track
from scripts.generate_drq_training_geometry import (
    CATALOG_FORMAT,
    CatalogFailure,
    _select,
    _road_hash,
    generate_catalog,
    run_catalog,
)


SEEDS = list(range(2_000, 2_270))


def _ring(seed: int, *, radius_offset: float = 0.0) -> np.ndarray:
    group, slot = divmod(seed - SEEDS[0], 45)
    radius = 20.0 + 2.0 * group + 0.03 * slot + radius_offset
    theta = np.linspace(0, 2 * math.pi, 120, endpoint=False)
    return np.column_stack((theta, theta + math.pi / 2, radius * np.cos(theta), radius * np.sin(theta)))


class FakeRoadEnv:
    instances: list[FakeRoadEnv] = []

    def __init__(self, *, identical: bool = False, near_seed: int | None = None):
        self.identical = identical
        self.near_seed = near_seed
        self.calls: list[tuple[str, int | None]] = []
        self.unwrapped = self
        self.action_space = SimpleNamespace(contains=lambda action: (
            isinstance(action, np.ndarray) and action.shape == (3,)
            and np.isfinite(action).all() and -1 <= action[0] <= 1
            and 0 <= action[1] <= 1 and 0 <= action[2] <= 1
        ))
        self.closed = False
        self.instances.append(self)

    def reset(self, *, seed: int, options: dict[str, int]):
        assert options == {"track_id": 1}
        self.calls.append(("reset", seed))
        actual = 2_000 if self.identical else seed
        offset = -0.0299 if seed == self.near_seed else 0.0
        self.track = _ring(actual, radius_offset=offset)
        beta, x, y = self.track[0, 1:4]
        self.car = SimpleNamespace(hull=SimpleNamespace(position=(x, y)))
        self.finish_line_tracker = SimpleNamespace(
            center=(x, y), forward=(-math.sin(beta), math.cos(beta)),
            half_width=40 / 6, half_depth=3.5 / 4,
            qualification_ratio=0.95, qualified_time_s=None, finish_time_s=None,
        )
        self.finish_time_s = None
        self.new_lap = False
        return np.zeros((96, 96, 3), dtype=np.uint8), {}

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)
        self.calls.append(("step", None))
        return np.zeros((96, 96, 3), dtype=np.uint8), -0.1, False, False, {"finished": False}

    def close(self):
        self.closed = True


class GeometryCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("experiments", "runs", "evaluations"):
            (self.root / directory).mkdir()
        FakeRoadEnv.instances.clear()
        analysis = self.root / "experiments" / "analysis.json"
        analysis.write_text(json.dumps({"format": "haic-drq-consumed-geometry-failures-v1",
                                        "status": "complete", "road_geometry": {}}), encoding="utf-8")
        self.protocol = {
            "format": "haic-drq-training-geometry-protocol-v1",
            "candidate_seeds": SEEDS.copy(), "train_target": 120, "diagnostic_target": 16,
            "near_duplicate_threshold": 1e-5, "near_duplicate_allow_mirror": True,
            "family_rules": [
                {"name": f"curve-{group}", "direction_source": "early",
                 "thresholds": {"perimeter_m": {"min": 2 * math.pi * (20 + 2 * group - 0.01),
                                                 "max": 2 * math.pi * (20 + 2 * group + 1.4)}}}
                for group in range(6)
            ],
            "exclusion_seed_ids": {"reserved": [987], "heldout": [988], "blind": [989]},
            "seed_audit": {"protocol_sources": {}, "training_ledgers": {}, "prior_audit_sources": {}},
            "analysis_receipt": {"path": "experiments/analysis.json",
                                 "sha256": hashlib.sha256(analysis.read_bytes()).hexdigest()},
            "structural_reference_seeds": [], "training_corpus_signatures": [],
        }
        self.protocol_path = self.root / "experiments" / "frozen.json"
        self._freeze()

    def _freeze(self):
        self.protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")

    @staticmethod
    def _audit(seeds, *, repo_root, **_sources):
        assert Path(repo_root).is_dir()
        return {"passed": True, "proposed_seeds": list(seeds),
                "proposed_seeds_sha256": hashlib.sha256(
                    json.dumps(seeds, separators=(",", ":")).encode("ascii")
                ).hexdigest(), "matched_collisions": [], "read_paths": ["synthetic"]}

    def _build(self, factory=FakeRoadEnv):
        return generate_catalog(self.protocol_path, repo_root=self.root,
                                env_factory=factory, audit_seeds=self._audit)

    def test_deterministic_two_stage_quotas_disjoint_diagnostic_and_metadata(self):
        scan, catalog = self._build()
        self.assertEqual(catalog["format"], CATALOG_FORMAT)
        self.assertEqual((len(catalog["train"]), len(catalog["train_diagnostic"])), (120, 16))
        self.assertEqual(Counter(row["stage"] for row in catalog["train"]),
                         {"representative": 60, "variant": 60})
        for stage in ("representative", "variant"):
            self.assertEqual(Counter(row["family"] for row in catalog["train"] if row["stage"] == stage),
                             {f"curve-{group}": 10 for group in range(6)})
        self.assertEqual(Counter(row["family"] for row in catalog["train_diagnostic"]),
                         {"curve-0": 3, "curve-1": 3, "curve-2": 3,
                          "curve-3": 3, "curve-4": 2, "curve-5": 2})
        selected = catalog["train"] + catalog["train_diagnostic"]
        self.assertEqual(len({row["geometry_seed"] for row in selected}), 136)
        self.assertEqual(len({row["road_coordinate_sha256"] for row in selected}), 136)
        self.assertEqual(scan["candidate_count"], len(scan["rows"]) + len(scan["unused_candidate_seeds"]))
        self.assertGreater(len(scan["unused_candidate_seeds"]), 0)
        self.assertEqual(scan["rows"][0]["scan_stage"], "representative")
        self.assertTrue(any(row["scan_stage"] == "variant" for row in scan["rows"]))
        self.assertEqual(scan["status"], "success")
        self.assertTrue(all(row["track_id"] == 1 and row["structure"]["finish_static_only"]
                            and row["verification"]["one_valid_raw_step"]
                            and row["verification"]["fresh_tracker_after_reset"]
                            and row["cyclic_signature_sha256"] == row["features"]["cyclic_signature_sha256"]
                            and row["features"]["signature_sha256"] == row["signature_sha256"]
                            for row in selected))
        self.assertTrue(all(not record["symmetry_feasible_among_scanned"] and record["selected_directions"]["right"] == 0
                            for record in catalog["symmetry"].values()))
        first_seeds = [row["geometry_seed"] for row in selected]
        second_scan, second_catalog = self._build()
        self.assertEqual(first_seeds, [row["geometry_seed"] for row in
                                      second_catalog["train"] + second_catalog["train_diagnostic"]])
        self.assertEqual([row["reason"] for row in scan["rows"] if row["status"] == "rejected"],
                         [row["reason"] for row in second_scan["rows"] if row["status"] == "rejected"])
        self.assertTrue(all(env.closed for env in FakeRoadEnv.instances))

    def test_blind_exclusion_and_analysis_hash_stop_before_any_factory_call(self):
        self.protocol["exclusion_seed_ids"]["blind"] = [SEEDS[0]]
        self._freeze()
        with self.assertRaisesRegex(CatalogFailure, "overlaps"):
            self._build()
        self.assertEqual(FakeRoadEnv.instances, [])
        self.protocol["exclusion_seed_ids"]["blind"] = [989]
        self.protocol["analysis_receipt"]["sha256"] = "0" * 64
        self._freeze()
        with self.assertRaisesRegex(CatalogFailure, "hash mismatch"):
            self._build()
        self.assertEqual(FakeRoadEnv.instances, [])

    def test_production_protocol_requires_frozen_official_source_hashes(self):
        from scripts.generate_drq_training_geometry import _protocol

        with self.assertRaisesRegex(CatalogFailure, "source code hashes"):
            _protocol(self.protocol_path, self.root)

    def test_failed_seed_audit_stops_before_reset(self):
        def failed_audit(*_args, **_kwargs):
            return {"passed": False, "proposed_seeds": SEEDS, "matched_collisions": [SEEDS[0]]}

        with self.assertRaisesRegex(CatalogFailure, "audit failed"):
            generate_catalog(self.protocol_path, repo_root=self.root,
                             env_factory=FakeRoadEnv, audit_seeds=failed_audit)
        self.assertEqual(FakeRoadEnv.instances, [])

    def test_default_auditor_and_changed_pinned_receipt_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "SHA-pinned source"):
            generate_catalog(self.protocol_path, repo_root=self.root, env_factory=FakeRoadEnv)
        self.assertEqual(FakeRoadEnv.instances, [])
        pinned = self.root / "runs" / "candidate-seed-audit.json"
        receipt = self._audit(SEEDS, repo_root=self.root)
        receipt["read_paths"] = ["tampered"]
        pinned.write_text(json.dumps(receipt), encoding="utf-8")
        self.protocol["seed_audit_receipt"] = {
            "path": "runs/candidate-seed-audit.json", "sha256": hashlib.sha256(pinned.read_bytes()).hexdigest(),
        }
        self._freeze()
        with self.assertRaisesRegex(CatalogFailure, "differs"):
            self._build()
        self.assertEqual(FakeRoadEnv.instances, [])

    def test_consumed_reference_near_duplicate_rejected_without_blind_reset(self):
        consumed = self.root / "experiments" / "analysis.json"
        descriptor = describe_track(_ring(2_000, radius_offset=0.0005))
        consumed.write_text(json.dumps({
            "format": "haic-drq-consumed-geometry-failures-v1", "status": "complete",
            "road_geometry": {"9000": {"road_coordinate_sha256":
                                    _road_hash(_ring(2_000, radius_offset=0.0005)),
                                    "signature": descriptor["signature"],
                                    "signature_sha256": descriptor["signature_sha256"]}},
        }), encoding="utf-8")
        self.protocol["analysis_receipt"]["sha256"] = hashlib.sha256(consumed.read_bytes()).hexdigest()
        self.protocol["structural_reference_seeds"] = [{
            "geometry_seed": 9_000, "partition": "screen", "consumed": True,
            "road_coordinate_sha256": _road_hash(_ring(2_000, radius_offset=0.0005)),
            "signature": descriptor["signature"],
            "evidence": self.protocol["analysis_receipt"].copy(),
        }]
        self._freeze()
        scan, catalog = self._build()
        self.assertEqual(scan["rows"][0]["reason"], "near_reference_duplicate")
        self.assertEqual(scan["references"][0]["geometry_seed"], 9_000)
        self.assertNotIn(9_000, [seed for env in FakeRoadEnv.instances for _, seed in env.calls])
        self.assertNotIn(989, [seed for env in FakeRoadEnv.instances for _, seed in env.calls])
        self.assertEqual(len(catalog["train"]), 120)

    def test_exact_and_cyclic_duplicates_are_distinguished(self):
        class DuplicateRoadEnv(FakeRoadEnv):
            def reset(self, *, seed, options):
                observation, info = super().reset(seed=2_000 if seed in {2_001, 2_002} else seed,
                                                  options=options)
                self.calls[-1] = ("reset", seed)
                if seed == 2_002:
                    self.track = np.roll(self.track, 8, axis=0)
                return observation, info

        scan, catalog = self._build(DuplicateRoadEnv)
        self.assertEqual(scan["rows"][1]["reason"], "exact_selected_duplicate")
        self.assertEqual(scan["rows"][2]["reason"], "cyclic_selected_duplicate")
        self.assertEqual(len(catalog["train"]), 120)

    def test_mid_course_direction_balances_when_both_sides_are_available(self):
        groups = [{"name": f"curve-{group}", "direction_source": "mid_curve",
                   "thresholds": {"point_count": {"min": 20}}} for group in range(6)]
        rows = []
        for group in range(6):
            for slot in range(30):
                seed = 10_000 + group * 100 + slot
                signature = {"turn_sequence": [0.03 * (group + 1) + slot * 0.0002] * 64,
                             "relative_width_sequence": [1.0] * 64,
                             "perimeter_m": 100.0 + seed / 100,
                             "width_median_m": 40 / 3}
                rows.append({"geometry_seed": seed, "status": "eligible", "family": groups[group]["name"],
                             "direction": "left" if slot % 2 == 0 else "right", "signature": signature,
                             "road_coordinate_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
                             "cyclic_signature_sha256": hashlib.sha256(str(seed + 100_000).encode()).hexdigest()})
        protocol = {"family_rules": groups, "near_duplicate_threshold": 1e-7,
                    "near_duplicate_allow_mirror": False}
        train, diagnostic, symmetry = _select(rows, protocol, lambda _stage: False)
        self.assertEqual((len(train), len(diagnostic)), (120, 16))
        for record in symmetry.values():
            self.assertTrue(record["symmetry_feasible_among_scanned"])
            self.assertLessEqual(abs(record["selected_directions"]["left"]
                                     - record["selected_directions"]["right"]), 1)

    def test_overlapping_structural_rules_fill_later_families_after_earlier_quotas(self):
        from unittest.mock import patch

        groups = [
            {"name": f"family-{index}", "direction_source": "early",
             "thresholds": {"point_count": {"min": 20}}}
            for index in range(6)
        ]
        rows = [
            {
                "geometry_seed": 50_000 + index,
                "status": "eligible",
                "family": "family-0",
                "matching_families": [group["name"] for group in groups],
                "direction": "left",
                "road_coordinate_sha256": hashlib.sha256(f"road-{index}".encode()).hexdigest(),
                "cyclic_signature_sha256": hashlib.sha256(f"shape-{index}".encode()).hexdigest(),
                "signature": {"turn_sequence": [0.01] * 64,
                              "relative_width_sequence": [1.0] * 64,
                              "perimeter_m": 300.0 + index, "width_median_m": 40 / 3},
            }
            for index in range(136)
        ]
        protocol = {"family_rules": groups, "near_duplicate_threshold": 0.06,
                    "near_duplicate_allow_mirror": True}
        with patch("scripts.generate_drq_training_geometry._near", return_value=1.0):
            train, diagnostic, _ = _select(rows, protocol, lambda _stage: False)
        self.assertEqual(len(train), 120)
        self.assertEqual(len(diagnostic), 16)
        self.assertEqual(Counter(row["family"] for row in train),
                         {f"family-{index}": 20 for index in range(6)})

    def test_insufficient_distinct_roads_preserves_failure_scan_no_result(self):
        with self.assertRaisesRegex(CatalogFailure, "insufficient distinct"):
            run_catalog(self.protocol_path, repo_root=self.root,
                        run_root=Path("runs/study"), result_path=Path("experiments/result.json"),
                        env_factory=lambda: FakeRoadEnv(identical=True), audit_seeds=self._audit)
        scan_path = self.root / "runs" / "study" / "scan.json"
        self.assertTrue(scan_path.exists())
        self.assertEqual(json.loads(scan_path.read_text())["status"], "failed")
        self.assertFalse((self.root / "runs" / "study" / "catalog.json").exists())
        self.assertFalse((self.root / "experiments" / "result.json").exists())
        self.assertTrue(FakeRoadEnv.instances[0].closed)

    def test_regeneration_hash_mismatch_stops_success_result(self):
        class NondeterministicRoadEnv(FakeRoadEnv):
            def reset(self, *, seed, options):
                observation, info = super().reset(seed=seed, options=options)
                if seed == 2_000 and sum(event == ("reset", seed) for event in self.calls) == 2:
                    self.track = _ring(seed, radius_offset=0.1)
                return observation, info

        with self.assertRaisesRegex(CatalogFailure, "regenerated road coordinates changed"):
            run_catalog(self.protocol_path, repo_root=self.root,
                        run_root=Path("runs/unstable"), result_path=Path("experiments/unstable.json"),
                        env_factory=NondeterministicRoadEnv, audit_seeds=self._audit)
        self.assertEqual(json.loads((self.root / "runs" / "unstable" / "scan.json").read_text())["status"],
                         "failed")
        self.assertFalse((self.root / "runs" / "unstable" / "catalog.json").exists())
        self.assertFalse((self.root / "experiments" / "unstable.json").exists())

    def test_immutable_success_result_rejects_reuse_before_new_reset(self):
        result = run_catalog(self.protocol_path, repo_root=self.root,
                             run_root=Path("runs/study"), result_path=Path("experiments/result.json"),
                             env_factory=FakeRoadEnv, audit_seeds=self._audit)
        self.assertEqual(result["train_count"], 120)
        self.assertEqual(result["train_diagnostic_count"], 16)
        catalog_bytes = (self.root / result["catalog_path"]).read_bytes()
        self.assertEqual(hashlib.sha256(catalog_bytes).hexdigest(), result["catalog_sha256"])
        calls = len(FakeRoadEnv.instances)
        with self.assertRaisesRegex(CatalogFailure, "exists"):
            run_catalog(self.protocol_path, repo_root=self.root,
                        run_root=Path("runs/study"), result_path=Path("experiments/result.json"),
                        env_factory=FakeRoadEnv, audit_seeds=self._audit)
        self.assertEqual(len(FakeRoadEnv.instances), calls)


if __name__ == "__main__":
    unittest.main()
