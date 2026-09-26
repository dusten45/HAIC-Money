"""Build a training-only catalog from frozen, audited official CarRacing seeds.

Run from the repository root with ``python -m scripts.generate_drq_training_geometry``.
No actor, outcome, blind geometry, or custom-map generator is used here. A static
finish check is not evidence that an actor can complete a lap.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from haic.algorithms.drq_v2.geometry_features import (
    describe_track,
    signature_distance,
    validate_track_structure,
)


PROTOCOL_FORMAT = "haic-drq-training-geometry-protocol-v1"
CATALOG_FORMAT = "haic-drq-training-geometry-catalog-v1"
SEED_LIMIT = 2**32
FEATURES = {
    "point_count", "perimeter_m", "abs_curvature_p50_per_m",
    "abs_curvature_p90_per_m", "abs_curvature_p99_per_m",
    "sharp_left_points", "sharp_right_points", "sharp_entry_count",
    "longest_straight_before_sharp_m", "rapid_reversals_65m",
    "consecutive_same_sign_65m", "early_bend_count", "late_bend_count",
    "mid_bend_count", "first_bend_abs_turn_rad", "first_bend_preceding_straight_m",
    "first_bend_start_m", "first_bend_start_fraction",
    "mid_strongest_abs_turn_rad", "mid_reversals_65m",
    "late_strongest_abs_turn_rad",
}


class CatalogFailure(ValueError):
    """A closed gate; ``scan`` retains every seed examined before the failure."""

    def __init__(self, message: str, scan: dict[str, Any] | None = None):
        super().__init__(message)
        self.scan = scan


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _cyclic_hash(signature: dict[str, Any]) -> str:
    pairs = list(zip(signature["turn_sequence"], signature["relative_width_sequence"]))
    canonical = min(pairs[index:] + pairs[:index] for index in range(len(pairs)))
    return _digest(json.dumps({
        "signed_turn_and_relative_width": canonical,
        "perimeter_m": signature["perimeter_m"],
        "width_median_m": signature["width_median_m"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _uint32_seeds(values: Any, name: str, *, minimum: int = 0) -> list[int]:
    if (not isinstance(values, list) or len(values) < minimum
            or any(type(value) is not int or not 0 <= value < SEED_LIMIT for value in values)
            or len(values) != len(set(values))):
        raise CatalogFailure(f"{name} must have at least {minimum} distinct uint32 integers")
    return values


def _inside(root: Path, relative: str, directory: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CatalogFailure(f"{directory} path is required")
    path = (root / relative).resolve()
    if not path.is_relative_to((root / directory).resolve()):
        raise CatalogFailure(f"path must be under {directory}/: {relative}")
    return path


def _pinned(root: Path, source: Any, directory: str) -> tuple[Path, bytes]:
    if not isinstance(source, dict) or not _sha256(source.get("sha256")):
        raise CatalogFailure(f"{directory} source needs a pinned sha256")
    path = _inside(root, source.get("path"), directory)
    raw = path.read_bytes()
    if _digest(raw) != source["sha256"]:
        raise CatalogFailure(f"source hash mismatch: {path}")
    return path, raw


def _rules(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 6:
        raise CatalogFailure("exactly six frozen structural family rules are required")
    names: set[str] = set()
    for rule in value:
        if not isinstance(rule, dict) or not isinstance(rule.get("name"), str) or not rule["name"]:
            raise CatalogFailure("each family needs a nonempty name")
        if rule["name"] in names:
            raise CatalogFailure("family names must be unique")
        names.add(rule["name"])
        if rule.get("direction_source") not in {"early", "mid_curve", "mid_s", "late"}:
            raise CatalogFailure("family direction_source must be early, mid_curve, mid_s or late")
        tests = rule.get("thresholds")
        if not isinstance(tests, dict) or not tests or not set(tests) <= FEATURES:
            raise CatalogFailure("each family must constrain recognized measured features")
        for metric, bounds in tests.items():
            if (not isinstance(bounds, dict) or not bounds or not set(bounds) <= {"min", "max"}
                    or any(type(n) not in (int, float) or not math.isfinite(n) for n in bounds.values())
                    or bounds.get("min", -math.inf) > bounds.get("max", math.inf)):
                raise CatalogFailure(f"invalid frozen threshold for {rule['name']}.{metric}")
    return value


def _protocol(path: Path, root: Path, *, verify_sources: bool = True
              ) -> tuple[dict[str, Any], str, str]:
    protocol_path = path.resolve()
    if not protocol_path.is_relative_to((root / "experiments").resolve()):
        raise CatalogFailure("frozen protocol must be under experiments/")
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if not isinstance(protocol, dict) or protocol.get("format") != PROTOCOL_FORMAT:
        raise CatalogFailure(f"protocol format must be {PROTOCOL_FORMAT}")
    if verify_sources:
        sources = protocol.get("source_sha256")
        required = {
            "core/vendor/car_racing.py", "core/track_variables.py", "core/finish_line.py",
            "haic/algorithms/drq_v2/geometry_features.py",
            "scripts/generate_drq_training_geometry.py", "scripts/audit_drq_training_seeds.py",
        }
        if not isinstance(sources, dict) or not required.issubset(sources):
            raise CatalogFailure("source code hashes for the unchanged generator and selector are required")
        for name, expected in sources.items():
            if (not isinstance(name, str) or Path(name).is_absolute()
                    or any(part in (".", "..") for part in Path(name).parts)
                    or not _sha256(expected)
                    or not (root / name).is_file()
                    or _digest((root / name).read_bytes()) != expected):
                raise CatalogFailure(f"frozen catalog source changed or is unsafe: {name}")
    pool = _uint32_seeds(protocol.get("candidate_seeds"), "candidate_seeds", minimum=256)
    if protocol.get("train_target") != 120 or protocol.get("diagnostic_target") != 16:
        raise CatalogFailure("frozen quotas must be 6 x (10 representatives + 10 variants) + 16 diagnostic")
    _rules(protocol.get("family_rules"))
    threshold = protocol.get("near_duplicate_threshold")
    if type(threshold) not in (float, int) or not math.isfinite(threshold) or threshold <= 0:
        raise CatalogFailure("near_duplicate_threshold must be frozen, positive and finite")
    if type(protocol.get("near_duplicate_allow_mirror")) is not bool:
        raise CatalogFailure("near_duplicate_allow_mirror must be a frozen boolean")

    exclusions = protocol.get("exclusion_seed_ids")
    if not isinstance(exclusions, dict) or set(exclusions) != {"reserved", "heldout", "blind"}:
        raise CatalogFailure("reserved, heldout and blind exclusion-only seed lists are required")
    excluded = set().union(*(
        _uint32_seeds(exclusions[key], f"exclusion_seed_ids.{key}") for key in exclusions
    ))
    if excluded.intersection(pool):
        raise CatalogFailure("candidate pool overlaps reserved/heldout/blind exclusions across track IDs")

    audit = protocol.get("seed_audit")
    if (not isinstance(audit, dict)
            or set(audit) != {"protocol_sources", "training_ledgers", "prior_audit_sources"}
            or any(not isinstance(audit[key], dict) for key in audit)):
        raise CatalogFailure("seed_audit needs explicit pinned protocol/ledger/audit source mappings")
    if "seed_audit_receipt" in protocol:
        _pinned(root, protocol["seed_audit_receipt"], "runs")
    _path, analysis = _pinned(root, protocol.get("analysis_receipt"), "experiments")
    receipt = json.loads(analysis)
    if (not isinstance(receipt, dict) or receipt.get("format") != "haic-drq-consumed-geometry-failures-v1"
            or receipt.get("status") != "complete" or not isinstance(receipt.get("road_geometry"), dict)):
        raise CatalogFailure("a completed, pinned failure-analysis receipt is required")
    references = protocol.get("structural_reference_seeds", [])
    if not isinstance(references, list):
        raise CatalogFailure("structural_reference_seeds must be an explicit list")
    seen: set[int] = set()
    for reference in references:
        if (not isinstance(reference, dict) or reference.get("partition") not in {"screen", "confirmation"}
                or reference.get("consumed") is not True):
            raise CatalogFailure("only explicitly consumed screen/confirmation references are permitted")
        seed = _uint32_seeds([reference.get("geometry_seed")], "structural reference", minimum=1)[0]
        if seed in seen or seed in pool or seed in exclusions["blind"]:
            raise CatalogFailure("reference seeds must be unique, consumed and outside the candidate pool")
        seen.add(seed)
        if reference.get("evidence") != protocol["analysis_receipt"]:
            raise CatalogFailure("structural reference must be sourced from the pinned consumed analysis")
        recorded = receipt.get("road_geometry", {}).get(str(seed))
        if (not isinstance(recorded, dict) or not _sha256(reference.get("road_coordinate_sha256"))
                or recorded.get("road_coordinate_sha256") != reference["road_coordinate_sha256"]):
            raise CatalogFailure("consumed reference must have an exact road hash in the analysis receipt")
        signature = recorded.get("signature")
        if (not isinstance(signature, dict) or reference.get("signature") != signature
                or not _sha256(recorded.get("signature_sha256"))
                or _digest(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode())
                != recorded["signature_sha256"]):
            raise CatalogFailure("consumed reference signature must match the pinned analysis")
        signature_distance(signature, signature)

    corpus = protocol.get("training_corpus_signatures", [])
    if not isinstance(corpus, list):
        raise CatalogFailure("training_corpus_signatures must be an explicit list")
    for reference in corpus:
        if (not isinstance(reference, dict) or reference.get("origin") != "training"
                or not _sha256(reference.get("road_coordinate_sha256"))
                or not _sha256(reference.get("signature_sha256"))
                or not isinstance(reference.get("signature"), dict)):
            raise CatalogFailure("training corpus needs a road hash, signature, and training provenance")
        _pinned(root, reference.get("evidence"), "runs")
        if _digest(json.dumps(reference["signature"], sort_keys=True, separators=(",", ":")).encode()) != reference["signature_sha256"]:
            raise CatalogFailure("training corpus signature digest mismatch")
        signature_distance(reference["signature"], reference["signature"])

    return protocol, _digest(raw), _digest(analysis)


def _audit(pool: list[int], protocol: dict[str, Any], root: Path,
           audit_seeds: Callable[..., dict[str, Any]] | None) -> dict[str, Any]:
    if audit_seeds is None:
        try:
            from scripts.audit_drq_training_seeds import audit_training_seeds
        except ImportError as error:
            raise CatalogFailure("blind-safe seed auditor unavailable; no geometry resets permitted") from error
        audit_seeds = audit_training_seeds
    report = audit_seeds(pool, repo_root=root, **protocol["seed_audit"])
    expected_sha = _digest(json.dumps(pool, separators=(",", ":")).encode())
    if (not isinstance(report, dict) or report.get("passed") is not True
            or report.get("proposed_seeds") != pool
            or report.get("proposed_seeds_sha256") != expected_sha
            or report.get("matched_collisions") != []):
        raise CatalogFailure("candidate seed audit failed or did not bind the exact frozen pool")
    if "seed_audit_receipt" in protocol:
        _, sealed = _pinned(root, protocol["seed_audit_receipt"], "runs")
        if json.loads(sealed) != report:
            raise CatalogFailure("pre-existing seed audit receipt differs from current pinned-source audit")
    return report


def _road_hash(track: Any) -> str:
    points = np.asarray(track, dtype="<f8")
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError("invalid raw road coordinates")
    return _digest(np.ascontiguousarray(points[:, 2:4]).tobytes())


def _measured(features: dict[str, Any]) -> dict[str, float]:
    bends = features["bend_events"]
    mid = [bend for bend in bends if 0.20 <= bend["start_fraction"] < 0.80]
    first = bends[0] if bends else None
    strongest = max(mid, key=lambda bend: abs(bend["turn_angle_rad"]), default=None)
    late = features.get("late_strongest_bend")
    reversals = sum(
        before["direction"] != after["direction"] and 0 <= after["start_m"] - before["end_m"] <= 65
        for before, after in zip(mid, mid[1:])
    )
    return {
        **{key: features[key] for key in FEATURES if key in features},
        "mid_bend_count": len(mid),
        "mid_reversals_65m": reversals,
        "first_bend_abs_turn_rad": abs(first["turn_angle_rad"]) if first else 0.0,
        "first_bend_preceding_straight_m": first["entry_straight_since_spawn_m"] if first else 0.0,
        "first_bend_start_m": first["start_m"] if first else 0.0,
        "first_bend_start_fraction": first["start_fraction"] if first else 0.0,
        "mid_strongest_abs_turn_rad": abs(strongest["turn_angle_rad"]) if strongest else 0.0,
        "late_strongest_abs_turn_rad": abs(late["turn_angle_rad"]) if late else 0.0,
    }


def _direction(features: dict[str, Any], source: str) -> str | None:
    bends = features["bend_events"]
    if source == "early":
        bend = features["early_strongest_bend"]
    elif source == "late":
        bend = features.get("late_strongest_bend")
    else:
        mid = [bend for bend in bends if 0.20 <= bend["start_fraction"] < 0.80]
        if source == "mid_s":
            pairs = [(before, after) for before, after in zip(mid, mid[1:])
                     if before["direction"] != after["direction"]
                     and 0 <= after["start_m"] - before["end_m"] <= 65]
            bend = pairs[0][0] if pairs else None
        else:
            bend = max(mid, key=lambda item: abs(item["turn_angle_rad"]), default=None)
    return bend["direction"] if bend is not None else None


def _near(first: dict[str, Any], second: dict[str, Any], *, mirror: bool) -> float:
    return signature_distance(first["signature"], second["signature"], allow_mirror=mirror)


def _scan_candidate(env: Any, seed: int, protocol: dict[str, Any],
                    comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    entry: dict[str, Any] = {"geometry_seed": seed, "track_id": 1, "status": "rejected"}
    try:
        observation, info = env.reset(seed=seed, options={"track_id": 1})
        if not isinstance(info, dict) or np.asarray(observation).shape != (96, 96, 3):
            raise ValueError("raw reset did not return a legal observation/info pair")
        track = env.unwrapped.track
        structure = validate_track_structure(track)
        features = describe_track(track)
        entry.update(road_coordinate_sha256=_road_hash(track), signature_sha256=features["signature_sha256"],
                     cyclic_signature_sha256=features["cyclic_signature_sha256"],
                     signature=features["signature"], features=features, structure=structure)
        if not structure["centerline_connected"]:
            entry["reason"] = "invalid_static_structure"
            return entry
        measures = _measured(features)
        matches = [rule for rule in protocol["family_rules"] if all(
            bounds.get("min", -math.inf) <= measures[metric] <= bounds.get("max", math.inf)
            for metric, bounds in rule["thresholds"].items()
        )]
        entry["measured_attributes"] = measures
        entry["matching_families"] = [rule["name"] for rule in matches]
        if not matches:
            entry["reason"] = "no_structural_family"
            return entry
        # An overlapping road remains eligible for the next family after an
        # earlier family's *predeclared* quota fills; never select by actor score.
        rule = next((rule for rule in matches if _direction(features, rule["direction_source"]) is not None), None)
        if rule is None:
            entry["reason"] = "no_measured_family_direction"
            return entry
        entry["family"] = rule["name"]
        entry["direction"] = _direction(features, rule["direction_source"])
        for reference in comparisons:
            if entry["road_coordinate_sha256"] == reference["road_coordinate_sha256"]:
                entry.update(reason="exact_reference_duplicate", duplicate_of=reference["reference_id"])
                return entry
            if entry["cyclic_signature_sha256"] == reference["cyclic_signature_sha256"]:
                entry.update(reason="cyclic_reference_duplicate", duplicate_of=reference["reference_id"])
                return entry
            distance = _near(entry, reference, mirror=protocol["near_duplicate_allow_mirror"])
            if distance < protocol["near_duplicate_threshold"]:
                entry.update(reason="near_reference_duplicate", duplicate_of=reference["reference_id"],
                             nearest_distance=distance)
                return entry
        entry["status"] = "eligible"
        return entry
    except Exception as error:
        entry.update(reason="reset_or_geometry_error", detail=str(error))
        return entry


def _select(entries: list[dict[str, Any]], protocol: dict[str, Any],
            scan_next: Callable[[str], bool]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    groups = protocol["family_rules"]
    threshold = protocol["near_duplicate_threshold"]
    mirror = protocol["near_duplicate_allow_mirror"]
    distance_cache: dict[tuple[int, int], float] = {}
    for stage, quotas in (
        ("representative", [10] * 6),
        ("variant", [10] * 6),
        ("diagnostic", [3, 3, 3, 3, 2, 2]),
    ):
        for rule, quota in zip(groups, quotas):
            family = rule["name"]
            for _ in range(quota):
                while True:
                    eligible: list[tuple[dict[str, Any], float]] = []
                    for row in entries:
                        if (row["status"] != "eligible"
                                or family not in row.get("matching_families", [row.get("family")])):
                            continue
                        direction = (_direction(row["features"], rule["direction_source"])
                                     if "features" in row else row["direction"])
                        if direction not in {"left", "right"}:
                            continue
                        exact = next((prior for prior in selected if row["road_coordinate_sha256"] == prior["road_coordinate_sha256"]), None)
                        if exact is not None:
                            row.update(status="rejected", reason="exact_selected_duplicate", duplicate_of=exact["geometry_seed"])
                            continue
                        cyclic = next((prior for prior in selected if row["cyclic_signature_sha256"] == prior["cyclic_signature_sha256"]), None)
                        if cyclic is not None:
                            row.update(status="rejected", reason="cyclic_selected_duplicate", duplicate_of=cyclic["geometry_seed"])
                            continue
                        distances = []
                        for prior in selected:
                            pair = (min(row["geometry_seed"], prior["geometry_seed"]),
                                    max(row["geometry_seed"], prior["geometry_seed"]))
                            if pair not in distance_cache:
                                distance_cache[pair] = _near(row, prior, mirror=mirror)
                            distances.append((distance_cache[pair], prior))
                        if distances:
                            closest, prior = min(distances, key=lambda pair: pair[0])
                            if closest < threshold:
                                row.update(status="rejected", reason="near_selected_duplicate",
                                           duplicate_of=prior["geometry_seed"], nearest_distance=closest)
                                continue
                        eligible.append((row, min((pair[0] for pair in distances), default=float("inf"))))
                    if eligible:
                        break
                    if scan_next(stage):
                        continue
                    raise CatalogFailure(f"insufficient distinct {family} roads for {stage}")
                if rule["direction_source"] in {"mid_curve", "mid_s"}:
                    counts = {side: sum(prior["family"] == family and prior["direction"] == side
                                        for prior in selected) for side in ("left", "right")}
                    minority = min(counts, key=counts.get)
                    if counts[minority] < counts["right" if minority == "left" else "left"]:
                        available = [item for item in eligible if (
                            _direction(item[0]["features"], rule["direction_source"])
                            if "features" in item[0] else item[0]["direction"]
                        ) == minority]
                        if available:
                            eligible = available
                    elif counts["left"] == counts["right"]:
                        # Equal counts: let the structural distance choose the next direction.
                        pass
                row, distance = max(eligible, key=lambda item: (item[1], -item[0]["geometry_seed"]))
                assigned_direction = (
                    _direction(row["features"], rule["direction_source"])
                    if "features" in row else row["direction"]
                )
                row.update(status="selected", stage=stage, family=family, direction=assigned_direction,
                           nearest_selected_distance=None if math.isinf(distance) else distance)
                selected.append(row)

    symmetry: dict[str, Any] = {}
    for rule in groups:
        family = rule["name"]
        observed = [row for row in entries
                    if family in row.get("matching_families", [row.get("family")])]
        directions = {side: sum(
            (_direction(row["features"], rule["direction_source"])
             if "features" in row else row.get("direction")) == side
            for row in observed
        ) for side in ("left", "right")}
        symmetry[family] = {"direction_source": rule["direction_source"],
                            "scanned_directions": directions,
                            "symmetry_feasible_among_scanned": bool(all(directions.values())),
                            "note": ("opening-bend symmetry infeasible among scanned roads; unscanned seeds unknown"
                                     if rule["direction_source"] == "early" and not all(directions.values()) else None)}
    for row in entries:
        if row["status"] == "eligible":
            row.update(status="rejected", reason="structural_quota_filled")
    for rule in groups:
        rows = [row for row in selected if row["family"] == rule["name"]]
        symmetry[rule["name"]]["selected_directions"] = {
            side: sum(row["direction"] == side for row in rows) for side in ("left", "right")
        }
    train = [row for row in selected if row["stage"] != "diagnostic"]
    diagnostic = [row for row in selected if row["stage"] == "diagnostic"]
    if len(train) != 120 or len(diagnostic) != 16 or len(train) < 100:
        raise CatalogFailure("fewer than 100 validated distinct training geometries")
    return train, diagnostic, symmetry


def _check_spawn(env: Any, track: Any) -> Any:
    first = np.asarray(track[0], dtype=np.float64)
    tracker = env.unwrapped.finish_line_tracker
    if tracker is None or env.unwrapped.car is None:
        raise CatalogFailure("spawn has no car or finish tracker")
    forward = np.array([-math.sin(first[1]), math.cos(first[1])])
    if (not np.allclose(tracker.center, first[2:4], atol=1e-5)
            or not np.allclose(tracker.forward, forward, atol=1e-5)
            or not np.allclose(env.unwrapped.car.hull.position, first[2:4], atol=1)
            or not math.isclose(tracker.half_width, 40 / 6, abs_tol=1e-4)
            or not math.isclose(tracker.half_depth, 3.5 * 0.25, abs_tol=1e-4)
            or not math.isclose(tracker.qualification_ratio, 0.95, abs_tol=1e-5)
            or tracker.finish_time_s is not None or tracker.qualified_time_s is not None
            or env.unwrapped.finish_time_s is not None or env.unwrapped.new_lap):
        raise CatalogFailure("spawn/finish tracker geometry or reset state is invalid")
    return tracker


def _verify(env: Any, row: dict[str, Any]) -> dict[str, Any]:
    seed = row["geometry_seed"]
    first_obs, first_info = env.reset(seed=seed, options={"track_id": 1})
    if np.asarray(first_obs).shape != (96, 96, 3) or not isinstance(first_info, dict):
        raise CatalogFailure(f"selected seed {seed}: illegal raw reset")
    road = env.unwrapped.track
    if _road_hash(road) != row["road_coordinate_sha256"]:
        raise CatalogFailure(f"selected seed {seed}: regenerated road coordinates changed")
    tracker = _check_spawn(env, road)
    action = np.zeros(3, dtype=np.float32)
    if not env.action_space.contains(action):
        raise CatalogFailure("raw neutral action is outside continuous action space")
    observation, reward, terminated, truncated, info = env.step(action)
    if (np.asarray(observation).shape != (96, 96, 3)
            or not math.isfinite(float(reward)) or type(terminated) not in (bool, np.bool_)
            or type(truncated) not in (bool, np.bool_) or not isinstance(info, dict)
            or type(info.get("finished")) is not bool
            or info["finished"] != (tracker.finish_time_s is not None)
            or bool(truncated) != bool(env.unwrapped.new_lap) or terminated or truncated):
        raise CatalogFailure(f"selected seed {seed}: illegal raw step/finish/truncation state")
    final_obs, final_info = env.reset(seed=seed, options={"track_id": 1})
    if (np.asarray(final_obs).shape != (96, 96, 3) or not isinstance(final_info, dict)
            or _road_hash(env.unwrapped.track) != row["road_coordinate_sha256"]
            or env.unwrapped.finish_line_tracker is tracker):
        raise CatalogFailure(f"selected seed {seed}: raw reset failed to regenerate fresh state")
    _check_spawn(env, env.unwrapped.track)
    return {"regenerated_coordinate_hash_match": True, "raw_reset": True,
            "one_valid_raw_step": True, "fresh_tracker_after_reset": True,
            "finish_plausibility": "static_only_not_a_successful_lap"}


def generate_catalog(protocol_path: Path, *, repo_root: Path,
                     env_factory: Callable[[], Any] | None = None,
                     audit_seeds: Callable[..., dict[str, Any]] | None = None
                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a frozen protocol, scan, select and recheck; never evaluate a policy."""
    root = repo_root.resolve()
    protocol, protocol_sha, analysis_sha = _protocol(
        protocol_path, root, verify_sources=env_factory is None
    )
    audit_report = _audit(protocol["candidate_seeds"], protocol, root, audit_seeds)
    if env_factory is None:
        if root != Path(__file__).resolve().parents[1]:
            raise CatalogFailure("official environment requires the source repository root")
        from core.vendor.car_racing import CarRacing
        env_factory = lambda: CarRacing(render_mode=None)
    project_root = Path(__file__).resolve().parents[1]
    source_sha256 = {
        name: _digest((project_root / name).read_bytes()) for name in (
            "core/vendor/car_racing.py", "core/finish_line.py",
            "haic/algorithms/drq_v2/geometry_features.py", "scripts/generate_drq_training_geometry.py",
        )
    }
    scan: dict[str, Any] = {
        "format": "haic-drq-training-geometry-scan-v1", "protocol_sha256": protocol_sha,
        "analysis_sha256": analysis_sha, "seed_audit": audit_report,
        "source_sha256": source_sha256, "candidate_count": len(protocol["candidate_seeds"]),
        "selection_stage_order": ["representative", "variant", "diagnostic"],
        "rows": [], "references": [],
    }
    env = env_factory()
    try:
        comparisons = []
        for corpus in protocol.get("training_corpus_signatures", []):
            comparisons.append({"reference_id": f"training:{corpus['road_coordinate_sha256']}",
                                "road_coordinate_sha256": corpus["road_coordinate_sha256"],
                                "cyclic_signature_sha256": _cyclic_hash(corpus["signature"]),
                                "signature": corpus["signature"]})
        _, analysis = _pinned(root, protocol["analysis_receipt"], "experiments")
        observed = json.loads(analysis)["road_geometry"]
        for reference in protocol.get("structural_reference_seeds", []):
            seed = reference["geometry_seed"]
            recorded = observed[str(seed)]
            signature = recorded["signature"]
            record = {"reference_id": f"consumed-{reference['partition']}:{seed}",
                      "road_coordinate_sha256": recorded["road_coordinate_sha256"],
                      "signature": signature, "cyclic_signature_sha256": _cyclic_hash(signature)}
            scan["references"].append({**record, "geometry_seed": seed, "track_id": 1,
                                       "signature_sha256": recorded["signature_sha256"],
                                       "source": protocol["analysis_receipt"]["path"]})
            comparisons.append(record)
        seed_stream = iter(protocol["candidate_seeds"])

        def scan_next(stage: str) -> bool:
            seed = next(seed_stream, None)
            if seed is None:
                return False
            row = _scan_candidate(env, seed, protocol, comparisons)
            row["scan_stage"] = stage
            scan["rows"].append(row)
            return True

        try:
            train, diagnostic, symmetry = _select(scan["rows"], protocol, scan_next)
            for row in train + diagnostic:
                row["verification"] = _verify(env, row)
        except Exception as error:
            for row in scan["rows"]:
                if row["status"] == "eligible":
                    row.update(status="rejected", reason="not_selected_before_stop")
            scan["status"] = "failed"
            scan["failure_reason"] = str(error)
            scan["unused_candidate_seeds"] = protocol["candidate_seeds"][len(scan["rows"]):]
            raise CatalogFailure(str(error), scan) from error
    finally:
        env.close()
    scan["status"] = "success"
    scan["symmetry"] = symmetry
    scan["unused_candidate_seeds"] = protocol["candidate_seeds"][len(scan["rows"]):]
    catalog = {
        "format": CATALOG_FORMAT, "protocol_sha256": protocol_sha,
        "analysis_sha256": analysis_sha,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train": [{key: row[key] for key in (
            "geometry_seed", "track_id", "family", "stage", "direction",
            "road_coordinate_sha256", "signature_sha256", "cyclic_signature_sha256", "signature", "features",
            "structure", "verification")}
            for row in train],
        "train_diagnostic": [{key: row[key] for key in (
            "geometry_seed", "track_id", "family", "stage", "direction",
            "road_coordinate_sha256", "signature_sha256", "cyclic_signature_sha256", "signature", "features",
            "structure", "verification")}
            for row in diagnostic],
        "scanned_count": len(scan["rows"]),
        "rejected_count": sum(row["status"] == "rejected" for row in scan["rows"]),
        "unused_candidate_count": len(scan["unused_candidate_seeds"]),
        "symmetry": symmetry,
        "seed_audit": audit_report,
        "seed_audit_receipt": protocol.get("seed_audit_receipt"),
        "source_sha256": source_sha256,
        "finish_evidence_limit": "Static finish geometry and one raw step do not establish lap completion.",
    }
    return scan, catalog


def run_catalog(protocol_path: Path, *, repo_root: Path, run_root: Path,
                result_path: Path, env_factory: Callable[[], Any] | None = None,
                audit_seeds: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write a full immutable scan under runs and a success-only result under experiments."""
    root = repo_root.resolve()
    target = _inside(root, str(run_root), "runs")
    result = _inside(root, str(result_path), "experiments")
    if target.exists() or result.exists() or not target.parent.is_dir() or not result.parent.is_dir():
        raise CatalogFailure("output path exists or output parent is missing")
    try:
        scan, catalog = generate_catalog(protocol_path, repo_root=root, env_factory=env_factory,
                                         audit_seeds=audit_seeds)
    except CatalogFailure as error:
        if error.scan is not None:
            target.mkdir(exist_ok=False)
            (target / "scan.json").write_bytes(_json_bytes(error.scan))
        raise
    target.mkdir(exist_ok=False)
    scan_path = target / "scan.json"
    scan_path.write_bytes(_json_bytes(scan))
    catalog["scan_path"] = scan_path.relative_to(root).as_posix()
    catalog["scan_sha256"] = _digest(scan_path.read_bytes())
    catalog_path = target / "catalog.json"
    catalog_path.write_bytes(_json_bytes(catalog))
    result_record = {"format": "haic-drq-training-geometry-result-v1", "status": "success",
                     "protocol_sha256": catalog["protocol_sha256"], "analysis_sha256": catalog["analysis_sha256"],
                     "catalog_path": catalog_path.relative_to(root).as_posix(),
                     "catalog_sha256": _digest(catalog_path.read_bytes()),
                     "scan_path": catalog["scan_path"], "scan_sha256": catalog["scan_sha256"],
                     "train_count": len(catalog["train"]),
                     "train_diagnostic_count": len(catalog["train_diagnostic"])}
    with result.open("xb") as output:
        output.write(_json_bytes(result_record))
    return result_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="already frozen experiments/ JSON")
    parser.add_argument("--run-root", type=Path, required=True, help="new immutable runs/ directory")
    parser.add_argument("--result", type=Path, required=True, help="new success-only experiments/ JSON")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = run_catalog(args.protocol, repo_root=args.repo_root, run_root=args.run_root,
                             result_path=args.result)
    except (CatalogFailure, OSError, ValueError) as error:
        parser.exit(2, f"training geometry catalog stopped: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
