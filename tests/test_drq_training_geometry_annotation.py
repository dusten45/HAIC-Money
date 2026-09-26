from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import annotate_drq_training_geometry as annotation


def _fixture() -> dict[str, dict]:
    signature = {"turn_sequence": [0.0] * 64, "relative_width_sequence": [1.0] * 64,
                 "perimeter_m": 1000.0, "width_median_m": 40.0 / 3.0}
    catalog_rows, result_rows, scan_rows = [], [], []
    for index in range(136):
        seed = 3910800001 + index
        family = "early-entry" if index < 68 else "mid-reversal"
        stage = "representative" if index < 60 else "variant" if index < 120 else "diagnostic"
        row = {
            "geometry_seed": seed, "track_id": 1, "family": family, "stage": stage,
            "direction": "left", "road_coordinate_sha256": f"{seed:064x}",
            "signature": deepcopy(signature),
            "features": {
                "perimeter_m": 1000.0, "fixed_half_width_m": 40.0 / 6.0,
                "abs_curvature_p90_per_m": 0.06, "early_strongest_bend": None,
                "mid_strongest_bend": None, "late_strongest_bend": None,
                "mid_reversals_65m": 0, "mid_consecutive_same_sign_65m": 0,
            },
            "structure": {"centerline_connected": True},
            "verification": {"one_valid_raw_step": True},
        }
        catalog_rows.append(row)
        scan_rows.append({"geometry_seed": seed, "status": "selected", "family": family,
                          "stage": stage, "road_coordinate_sha256": row["road_coordinate_sha256"]})
        result_rows.append({
            "geometry_seed": seed, "family": family,
            "partition": "train" if index < 120 else "train_diagnostic",
            "stage": stage, "road_coordinate_sha256": row["road_coordinate_sha256"],
            "provisional_difficulty": "useful_boundary",
            "finish_reachability": "observed_one_actor",
            "by_source": {"0": {"finished": False}, "1": {"finished": True}},
        })
    return {
        annotation.CATALOG.name: {
            "train": catalog_rows[:120], "train_diagnostic": catalog_rows[120:],
            "protocol_sha256": annotation.PROTOCOL_SHA,
        },
        annotation.SCAN.name: {"protocol_sha256": annotation.PROTOCOL_SHA, "rows": scan_rows},
        annotation.PROTOCOL.name: {
            "family_rules": [
                {"name": name, "targeted_failure_mode": "frozen structural hypothesis"}
                for name in ("early-entry", "mid-reversal")
            ],
            "generator_version": "official seeded generator",
            "generator_control": {"checkpoint_count": 12},
            "near_duplicate_threshold": 0.06,
            "exclusion_seed_ids": {"reserved": [42], "heldout": [31001], "blind": [33201]},
            "structural_reference_seeds": [{"signature": deepcopy(signature)}],
            "training_corpus_signatures": [{"signature": deepcopy(signature)}],
        },
        annotation.SUMMARY.name: {
            "catalog_sha256": annotation.CATALOG_SHA,
            "protocol_sha256": "5935664e132603de62cd35601fda12b998ea2009548332736720bab905d2c3f3",
            "geometry": result_rows,
        },
    }


class TrainingGeometryAnnotationTests(unittest.TestCase):
    def test_joins_all_roads_without_policy_deletion(self) -> None:
        fixture = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            with patch.object(annotation, "_pinned", side_effect=lambda _root, path, _sha: fixture[path.name]), \
                 patch.object(annotation, "signature_distance", return_value=0.2):
                result = annotation.annotate(root)
            self.assertEqual((result["train"], result["train_diagnostic"]), (120, 16))
            payload = json.loads((root / annotation.OUTPUT).read_text())
            self.assertEqual(payload["removed_for_policy_failure"], 0)
            self.assertEqual(len(payload["roads"]), 136)
            self.assertEqual(payload["roads"][120]["partition"], "TRAIN-DIAGNOSTIC")

    def test_rejects_blind_collision_and_changed_road_identity(self) -> None:
        fixture = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            fixture[annotation.PROTOCOL.name]["exclusion_seed_ids"]["blind"].append(3910800001)
            with patch.object(annotation, "_pinned", side_effect=lambda _root, path, _sha: fixture[path.name]), \
                 patch.object(annotation, "signature_distance", return_value=0.2):
                with self.assertRaisesRegex(ValueError, "held-out/blind"):
                    annotation.annotate(root)
            self.assertFalse((root / annotation.OUTPUT).exists())
            fixture[annotation.PROTOCOL.name]["exclusion_seed_ids"]["blind"].remove(3910800001)
            fixture[annotation.SUMMARY.name]["geometry"][0]["road_coordinate_sha256"] = "0" * 64
            with patch.object(annotation, "_pinned", side_effect=lambda _root, path, _sha: fixture[path.name]), \
                 patch.object(annotation, "signature_distance", return_value=0.2):
                with self.assertRaisesRegex(ValueError, "changed between selection"):
                    annotation.annotate(root)
            self.assertFalse((root / annotation.OUTPUT).exists())


if __name__ == "__main__":
    unittest.main()
