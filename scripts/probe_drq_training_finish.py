"""Verify a valid forward finish crossing exists for frozen training-only roads.

This is a geometric/finish-tracker sanity check, not a physically driven lap or
policy evaluation. It never loads an actor or any held-out or blind road.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from core.finish_line import FinishLineTracker
from scripts.generate_drq_training_geometry import _road_hash


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward_crossing_probe(tracker: FinishLineTracker) -> dict[str, Any]:
    """Test finish logic with a physically *idealized* centerline crossing."""
    test = copy.deepcopy(tracker)
    if test.finish_time_s is not None or test.departed_start_area:
        raise ValueError("finish tracker must start unqualified in its reset state")
    center_x, center_y = test.center
    forward_x, forward_y = test.forward
    if abs((forward_x * forward_x + forward_y * forward_y) - 1.0) > 1e-4:
        raise ValueError("finish forward vector is not unit length")
    if test.half_width <= 0 or test.half_depth <= 0 or test.qualification_ratio >= 1.0:
        raise ValueError("finish gate dimensions/qualification are invalid")
    progress = (0.0, 0.96, 0.96, 0.96, 0.96)
    longitudinals = (-2.0, -1.5, -0.5, 0.5, 1.5)
    outcomes = []
    for index, (fraction, covered) in enumerate(zip(longitudinals, progress)):
        offset = fraction * test.half_depth
        position = (center_x + offset * forward_x, center_y + offset * forward_y)
        outcomes.append(test.update(position, (forward_x, forward_y), covered, index * 0.1))
    if (outcomes != [False, False, False, False, True]
            or test.finish_time_s is None or test.qualified_time_s is None):
        raise ValueError("finish tracker did not accept a qualified forward centerline crossing")
    return {
        "qualified_forward_centerline_crossing_accepted": True,
        "qualified_time_s": test.qualified_time_s,
        "finish_time_s": test.finish_time_s,
        "original_tracker_unmodified": tracker.finish_time_s is None,
        "physical_car_reachability_proven": False,
    }


def probe_catalog(catalog_path: Path, expected_sha: str, output_path: Path,
                  *, repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    catalog_path = catalog_path.resolve()
    output_path = output_path.resolve()
    if not catalog_path.is_relative_to(root / "runs") or not output_path.is_relative_to(root / "runs"):
        raise ValueError("catalog and output must be under runs/")
    if output_path.exists() or not output_path.parent.is_dir() or _sha(catalog_path) != expected_sha:
        raise ValueError("output exists, parent is missing, or catalog hash changed")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("format") != "haic-drq-training-geometry-catalog-v1":
        raise ValueError("unknown training-only road catalog format")
    rows = catalog.get("train", []) + catalog.get("train_diagnostic", [])
    if (len(catalog.get("train", [])) != 120 or len(catalog.get("train_diagnostic", [])) != 16
            or len(rows) != len({row["geometry_seed"] for row in rows})):
        raise ValueError("catalog does not define 136 distinct training-only roads")
    from core.vendor.car_racing import CarRacing

    env = CarRacing(render_mode=None)
    results = []
    try:
        for row in rows:
            seed = row["geometry_seed"]
            _observation, _info = env.reset(seed=seed, options={"track_id": row["track_id"]})
            if _road_hash(env.track) != row["road_coordinate_sha256"]:
                raise ValueError(f"catalog road changed before finish probe: {seed}")
            results.append({
                "geometry_seed": seed,
                "track_id": row["track_id"],
                "partition": "TRAIN" if row in catalog["train"] else "TRAIN-DIAGNOSTIC",
                **forward_crossing_probe(env.finish_line_tracker),
            })
    finally:
        env.close()
    receipt = {
        "format": "haic-drq-training-road-finish-logic-v1",
        "catalog_path": catalog_path.relative_to(root).as_posix(),
        "catalog_sha256": expected_sha,
        "checked_geometry_count": len(results),
        "per_geometry": results,
        "limit": "Synthetic finish tracker trajectory proves logic accepts a valid crossing; it does not prove Box2D vehicle can follow the road or that frozen actors finish.",
    }
    with output_path.open("xb") as destination:
        destination.write((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        receipt = probe_catalog(args.catalog, args.catalog_sha256, args.output,
                                repo_root=args.repo_root)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"checked_geometry_count": receipt["checked_geometry_count"],
                      "catalog_sha256": receipt["catalog_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
