"""Synthetic-only TD allocation audit tests. No simulator or protected outcomes."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import audit_tdmpc2_train_diag_seeds as audit


class TdDiagnosticInventoryTests(unittest.TestCase):
    CANDIDATE = "experiments/tdmpc2-future-diagnostic.json"

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cells = [
            {"partition": "TRAIN", "track_id": track, "geometry_seed": 3000 + (track - 1) * 6 + i,
             "obstacles": True}
            for track in range(1, 5) for i in range(6)
        ]
        self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                    "purpose": "TRAIN-DIAGNOSTIC", "cells": self.cells})
        reserved = list(range(3910800001, 3910800513))
        self.write(audit.CATALOG_PROTOCOL, {"format": "haic-drq-training-geometry-protocol-v1",
                                             "candidate_seeds": reserved,
                                             "exclusion_seed_ids": {"blind": [88000],
                                                                    "heldout": [88001], "reserved": [88002]}})
        self.write(audit.CATALOG, {"protocol_sha256": self.sha(audit.CATALOG_PROTOCOL),
                                    "seed_audit": {"proposed_seeds": reserved},
                                    "train": [{"geometry_seed": seed} for seed in reserved[:120]],
                                    "train_diagnostic": [{"geometry_seed": seed} for seed in reserved[120:136]]})
        g0 = [{"partition": "TRAIN", "track_id": 1, "geometry_seed": 4272000001 + i,
               "obstacles": True} for i in range(12)]
        self.write(audit.G0_PROTOCOL, {"cells": g0})
        self.write(audit.G0_AUDIT, {"candidate_seeds": [c["geometry_seed"] for c in g0]})
        self.write(audit.G0_CELLS, self.rows([{"geometry_seed": c["geometry_seed"],
                                               "partition": "TRAIN", "track_id": 1} for c in g0]))
        self.write(f"{audit.G0_CLAIMS}/.lock", b"")
        self.write(f"{audit.G0_CLAIMS}/claim.json",
                   {"geometry_seeds": [c["geometry_seed"] for c in g0]})
        self.write(f"{audit.TRAIN_CLAIMS}/.gitkeep", b"")
        for path in audit.TD_JOURNALS:
            self.write(path, self.rows([{"event": "start", "protocol_sha256": "0" * 64},
                                         {"event": "reset_intent", "geometry_seed": 90001,
                                          "track_id": 1},
                                         {"event": "reset", "geometry_seed": 90001, "track_id": 1},
                                         {"event": "episode", "geometry_seed": 90001,
                                          "track_id": 1}]))
        self.write("runs/other-lane/episodes.jsonl", self.rows([
            {"event": "reset", "seed": 90002, "track_id": 2},
            {"event": "end", "seed": 90002, "track_id": 2},
        ]))
        self.write("experiments/pixel-rlpd-protocol-v1.json",
                   {"format": "haic-pixel-rlpd-study-v1",
                    "reserved_training_seeds": [90003],
                    "partitions": {"confirmation": {"seeds": [88003]},
                                   "blind": {"seeds": [88004]}}})

    def write(self, path: str, value: dict | bytes) -> None:
        dest = self.root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

    def sha(self, path: str) -> str:
        return hashlib.sha256((self.root / path).read_bytes()).hexdigest()

    @staticmethod
    def rows(values: list[dict]) -> bytes:
        return b"".join(json.dumps(row).encode() + b"\n" for row in values)

    def audit(self, **kwargs) -> dict:
        return audit.audit_tdmpc2_train_diag_seeds(self.CANDIDATE, repo_root=self.root, **kwargs)

    def test_clean_synthetic_inventory_still_blocks_and_never_claims(self) -> None:
        report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["collisions"], [])
        self.assertEqual([b["field"] for b in report["blockers"]], ["coverage"])
        self.assertEqual(report["blind_episode_reads"], 0)
        self.assertFalse(report["reservation_or_claim"])
        self.assertEqual(len(report["cells"]), 24)
        self.assertEqual(len({c["geometry_seed"] for c in report["cells"]}), 24)
        self.assertEqual(report["candidate_sha256"], self.sha(self.CANDIDATE))
        self.assertFalse((self.root / audit.TRAIN_CLAIMS / "seed-3000.json").exists())

    def test_cross_track_geometry_alias_is_rejected_before_audit(self) -> None:
        cells = [dict(c) for c in self.cells]
        cells[6]["geometry_seed"] = cells[0]["geometry_seed"]
        self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                    "purpose": "TRAIN-DIAGNOSTIC", "cells": cells})
        with self.assertRaisesRegex(audit.InventoryError, "cross-track geometry alias"):
            self.audit()

    def test_input_must_be_explicit_frozen_shape_not_seed_generation(self) -> None:
        for bad in ({"format": audit.FORMAT, "purpose": "TRAIN-DIAGNOSTIC",
                     "seed_start": 3000, "cells": self.cells},
                    {"format": audit.FORMAT, "purpose": "TRAIN-DIAGNOSTIC",
                     "cells": self.cells[:-1]},
                    {"format": audit.FORMAT, "purpose": "TRAIN-DIAGNOSTIC",
                     "cells": [dict(self.cells[0], obstacles=False)] + self.cells[1:]}):
            with self.subTest(bad=bad):
                self.write(self.CANDIDATE, bad)
                with self.assertRaises(audit.InventoryError):
                    self.audit()

    def test_drq_512_reservation_cross_track_and_selected_diagnostic(self) -> None:
        for seed in (3910800001, 3910800121, 3910800501):
            with self.subTest(seed=seed):
                cells = [dict(c) for c in self.cells]
                cells[18]["geometry_seed"] = seed  # track 4, reserved on other tracks.
                self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                            "purpose": "TRAIN-DIAGNOSTIC", "cells": cells})
                report = self.audit()
                hit = next(c for c in report["collisions"] if c["geometry_seed"] == seed)
                self.assertEqual(hit["track_id"], 4)
                self.assertTrue(any(s["path"] == audit.CATALOG_PROTOCOL for s in hit["sources"]))

    def test_protected_id_metadata_only_and_no_protected_outcome_reads(self) -> None:
        cells = [dict(c) for c in self.cells]
        cells[0]["geometry_seed"] = 88003
        self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                    "purpose": "TRAIN-DIAGNOSTIC", "cells": cells})
        self.write("runs/other-lane/evaluations/blind/episodes.jsonl", b"DO NOT READ\n")
        self.write("runs/other-lane/privateEpisodes/episodes.jsonl", b"DO NOT READ\n")
        self.write("experiments/other-lane-blind-result-v1.json", b"DO NOT READ\n")
        self.write("evaluations/protected/episodes.jsonl", b"DO NOT READ\n")
        original = Path.read_bytes

        def guarded(path: Path) -> bytes:
            self.assertFalse(any(marker in str(path).lower() for marker in
                                 ("privateepisodes", "/evaluations/", "blind-result")))
            return original(path)

        with patch.object(Path, "read_bytes", guarded):
            report = self.audit()
        hit = next(c for c in report["collisions"] if c["geometry_seed"] == 88003)
        self.assertTrue(any("pixel-rlpd-protocol-v1.json" in s["path"] for s in hit["sources"]))
        self.assertEqual(report["blind_episode_reads"], 0)

    def test_g0_claim_g1_registry_and_train_ledger_across_tracks(self) -> None:
        cells = [dict(c) for c in self.cells]
        cells[0]["geometry_seed"] = 4272000001
        cells[6]["geometry_seed"] = 90002
        cells[12]["geometry_seed"] = 90004
        self.write(self.CANDIDATE, {"format": audit.FORMAT,
                                    "purpose": "TRAIN-DIAGNOSTIC", "cells": cells})
        self.write("runs/other-lane/episodes.jsonl", self.rows([
            {"event": "reset", "seed": 90002, "track_id": 4},
            {"event": "end", "seed": 90002, "track_id": 4},
        ]))
        self.write(f"{audit.TRAIN_CLAIMS}/seed-90004.json", {
            "format": "haic-train-seed-claim-v1", "study_id": "rlpd-g1", "protocol_id": None,
            "protocol_path": None, "protocol_sha256": None, "partition": "TRAIN",
            "track_id": 1, "geometry_seed": 90004, "obstacles": True,
            "status": "reserved", "claimed_at_utc": "2026-09-28T00:00:00Z",
            "audit_source": "synthetic", "audit_digest_sha256": "0" * 64,
        })
        report = self.audit()
        sources = {row["geometry_seed"]: {s["path"] for s in row["sources"]}
                   for row in report["collisions"]}
        self.assertIn(audit.G0_PROTOCOL, sources[4272000001])
        self.assertIn("runs/other-lane/episodes.jsonl", sources[90002])
        self.assertIn(f"{audit.TRAIN_CLAIMS}/seed-90004.json", sources[90004])

    def test_td_partial_and_intent_without_reset_are_preserved(self) -> None:
        for suffix in ("v1", "v2"):
            with self.subTest(suffix=suffix):
                path = f"runs/tdmpc2-long-20260928-{suffix}/training.jsonl"
                self.write(path, self.rows([
                    {"event": "start", "protocol_sha256": "0" * 64},
                    {"event": "reset_intent", "geometry_seed": 3000, "track_id": 3},
                    *([{"event": "reset", "geometry_seed": 3000, "track_id": 3}]
                      if suffix == "v1" else []),
                    {"event": "partial", "decisions": 17},
                ]))
                report = self.audit()
                hit = next(c for c in report["collisions"] if c["geometry_seed"] == 3000)
                self.assertIn(path, {s["path"] for s in hit["sources"]})
                self.write(path, self.rows([{"event": "start", "protocol_sha256": "0" * 64},
                                            {"event": "reset_intent", "geometry_seed": 90001,
                                             "track_id": 1},
                                            {"event": "reset", "geometry_seed": 90001,
                                             "track_id": 1},
                                            {"event": "episode", "geometry_seed": 90001,
                                             "track_id": 1}]))

    def test_unknown_range_candidate_blocks_and_disjoint_warns(self) -> None:
        path = "experiments/other-train-allocation.json"
        for value in ({"road_seed_range": {"start": 2990, "end": 3000}},
                      {"road_seed_range": {"start": 2999, "count": 2}},
                      {"road_seed_range": [2990, 3000]},
                      {"road_seed_range": {"start": "unbounded", "end": 5000}}):
            with self.subTest(value=value):
                self.write(path, value)
                report = self.audit()
                self.assertTrue(any(b["path"] == path for b in report["blockers"]))
        self.write(path, {"road_seed_range": {"start": 5000, "end": 6000}})
        report = self.audit()
        self.assertFalse(any(b["path"] == path for b in report["blockers"]))
        self.assertTrue(any(w["path"] == path for w in report["provenance_warnings"]))

    def test_unrelated_metadata_drift_is_a_warning_not_a_collision(self) -> None:
        original = self.audit()
        self.write("experiments/other-lane-protocol.json", {"unrelated": "new metadata"})
        changed = self.audit(expected_source_inventory_sha256=original["source_inventory_sha256"])
        self.assertEqual(changed["collisions"], [])
        self.assertFalse(any(b["path"] == "source_inventory" for b in changed["blockers"]))
        self.assertTrue(any(w["path"] == "source_inventory" and w["field"] == "names+bytes"
                            for w in changed["provenance_warnings"]))
        self.assertEqual([b["field"] for b in changed["blockers"]], ["coverage"])

    def test_candidate_relevant_unknown_protocol_and_torn_td_record_block(self) -> None:
        self.write("experiments/unknown-lane-protocol.json", {"unknown_road_ids": [3000]})
        path = audit.TD_JOURNALS[0]
        self.write(path, self.rows([{"event": "start", "protocol_sha256": "0" * 64},
                                    {"event": "reset_intent", "geometry_seed": 3001,
                                     "track_id": 2}]) + b"{bad json\n")
        report = self.audit()
        self.assertTrue(any(b["path"] == "experiments/unknown-lane-protocol.json"
                            for b in report["blockers"]))
        self.assertTrue(any(b["path"] == path for b in report["blockers"]))
        self.assertIn(3001, {c["geometry_seed"] for c in report["collisions"]})

    def test_cli_has_no_reserve_or_seed_start_option_and_returns_blocked(self) -> None:
        with patch("sys.argv", ["audit", "--repo-root", str(self.root),
                                "--candidate-path", self.CANDIDATE]), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(audit.main(), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "BLOCKED")
        self.assertEqual(list((self.root / audit.TRAIN_CLAIMS).iterdir()),
                         [self.root / audit.TRAIN_CLAIMS / ".gitkeep"])
        with (patch("sys.argv", ["audit", "--repo-root", str(self.root),
                                 "--candidate-path", self.CANDIDATE, "--reserve"]),
              redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected):
            audit.main()
        self.assertEqual(rejected.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
