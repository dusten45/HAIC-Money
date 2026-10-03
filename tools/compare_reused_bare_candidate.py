"""Source-bound three-arm comparison on exactly 23 already consumed development cells.

This is diagnosis only. It never opens fresh confirmation or blind partitions and
never promotes an Agent. Actual Agent routes run in separate cold worker processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_bare_generalization as fresh

ARMS = ("baseline", "margin", "ego_switch")
CLASSES = {"baseline": "_CompoundClearingBrakeCarryController",
           "margin": "_ObservedMarginArbitrationController",
           "ego_switch": "_ObservedEgoSideSwitchController"}
LEGACY_CELLS = ((1, 11), (1, 21), (1, 42), (1, 17), (1, 516237), (2, 644062), (3, 1007))
SCREEN_PROTOCOL = ROOT / "experiments" / "observed-margin-generalization-v1.json"
SCREEN_RUN = ROOT / ".haic-artifacts" / "observed-margin-generalization-v1" / "run"


def expected_development_cells(screen_protocol: dict) -> tuple[tuple[int, int], ...]:
    if screen_protocol.get("name") != "observed-margin-generalization-v1":
        raise ValueError("wrong consumed screen protocol")
    screen = screen_protocol["partitions"]["screen"]
    grid = tuple((track, seed) for track in screen["track_ids"] for seed in screen["seeds"])
    if len(grid) != 16 or len(set(grid)) != 16 or len({seed for _, seed in grid}) != 4:
        raise ValueError("consumed screen must have exactly four track IDs by four geometry seeds")
    cells = LEGACY_CELLS + grid
    if len(set(cells)) != 23:
        raise ValueError("legacy and consumed screen cells overlap")
    return cells


def sealed_holdout_seeds(experiments_dir: Path) -> set[int]:
    """Reserve every documented confirmation/blind geometry across all IDs."""
    protected: set[int] = set()

    def is_holdout_name(name: str) -> bool:
        return "confirmation" in name or "blind" in name

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            partitions = value.get("partitions")
            if isinstance(partitions, dict):
                for name, partition in partitions.items():
                    if is_holdout_name(name) and isinstance(partition, dict):
                        protected.update(fresh._seed_list(partition.get("seeds")))
            for key, child in value.items():
                if is_holdout_name(key):
                    protected.update(fresh._seed_list(child.get("seeds") if isinstance(child, dict) else child))
                if is_holdout_name(key) and "seed" in key:
                    protected.update(fresh._seed_list(child))
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    for path in sorted(experiments_dir.glob("*.json")):
        inspect(fresh._read_json(path))
    return protected


def validate_protocol(protocol: dict, screen_protocol: dict, screen_sha256: str,
                      sealed_seeds: set[int]) -> tuple[tuple[int, int], ...]:
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("development protocol schema_version must be 1")
    if protocol.get("status") != "PREREGISTERED" or not isinstance(protocol.get("name"), str):
        raise ValueError("development protocol must be PREREGISTERED and named")
    if protocol.get("max_steps") != 2000 or protocol.get("frame_skip") != 4:
        raise ValueError("official evaluation requires max_steps 2000 and frame_skip 4")
    fresh._sha256_string(protocol.get("screen_protocol_sha256"), "screen_protocol_sha256")
    if protocol["screen_protocol_sha256"] != screen_sha256:
        raise ValueError("consumed screen protocol hash mismatch")
    if protocol.get("controller_classes") != CLASSES:
        raise ValueError("unexpected three actual Agent routes")
    hashes = protocol.get("agent_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(ARMS):
        raise ValueError("agent_sha256 must pin the three arms")
    for arm in ARMS:
        fresh._sha256_string(hashes[arm], f"agent_sha256.{arm}")
    if (hashes["baseline"], hashes["margin"]) != (
        screen_protocol.get("control_agent_sha256"), screen_protocol.get("candidate_agent_sha256")
    ):
        raise ValueError("baseline and margin must match frozen screen source hashes")
    fresh._sha256_string(protocol.get("model_sha256"), "model_sha256")
    helpers = protocol.get("runtime_helper_sha256")
    if not isinstance(helpers, dict) or set(helpers) != {"action_smoothing.py", "action_representation.py"}:
        raise ValueError("both runtime helper hashes are required")
    for name, value in helpers.items():
        fresh._sha256_string(value, name)
    expected = expected_development_cells(screen_protocol)
    actual = protocol.get("development_cells")
    if not isinstance(actual, list) or actual != [list(cell) for cell in expected]:
        raise ValueError("development_cells must be the exact 7 legacy plus 16 consumed screen cells, in order")
    overlap = {seed for _, seed in expected} & sealed_seeds
    if overlap:
        raise ValueError(f"confirmation/blind geometry seed requested: {sorted(overlap)}")
    gates = protocol.get("gates")
    if not isinstance(gates, dict) or gates.get("fresh_confirmation") != "SEALED" or gates.get("blind") != "SEALED" or gates.get("sota_promotion") is not False:
        raise ValueError("development diagnosis cannot open confirmation/blind or SOTA promotion")
    return expected


def verify_screen_consumed(screen_protocol: dict, screen_run: Path, identity: dict) -> int:
    if identity.get("protocol_sha256") is None:
        raise ValueError("screen receipt identity is incomplete")
    screen = screen_protocol["partitions"]["screen"]
    count = 0
    for track in screen["track_ids"]:
        for seed in screen["seeds"]:
            for arm in ("control", "candidate"):
                row = fresh.load_cell(screen_run, identity, "screen", arm, track, seed, 0)
                if row is None or row.get("error"):
                    raise ValueError(f"unconsumed or failed screen cell: {track}/{seed}/{arm}")
            count += 1
    return count


def build_identity(protocol_path: Path, protocol: dict, paths: dict[str, Path],
                   screen_sha256: str) -> dict:
    synthetic = {"control_agent_sha256": protocol["agent_sha256"]["baseline"],
                 "candidate_agent_sha256": protocol["agent_sha256"]["ego_switch"],
                 "model_sha256": protocol["model_sha256"],
                 "runtime_helper_sha256": protocol["runtime_helper_sha256"]}
    base = fresh.build_identity(protocol_path, synthetic, paths["baseline"], paths["ego_switch"], paths["model"])
    margin_hash = fresh.digest(paths["margin"])
    if margin_hash != protocol["agent_sha256"]["margin"]:
        raise ValueError("margin agent source hash mismatch")
    base["source_sha256"] = {"baseline": base["source_sha256"]["control"],
                             "margin": margin_hash,
                             "ego_switch": base["source_sha256"]["candidate"]}
    base["screen_protocol_sha256"] = screen_sha256
    base["screen_freeze_sha256"] = fresh.digest(SCREEN_RUN / "freeze.json")
    base["screen_summary_sha256"] = fresh.digest(SCREEN_RUN / "screen-summary.json")
    base["comparator_sha256"] = fresh.digest(Path(__file__))
    return base


def check_frozen_inputs(identity: dict, paths: dict[str, Path]) -> None:
    virtual = {**identity, "source_sha256": {
        "control": identity["source_sha256"]["baseline"],
        "candidate": identity["source_sha256"]["ego_switch"]}}
    fresh.check_frozen_inputs(virtual, {"control": paths["baseline"],
        "candidate": paths["ego_switch"], "model": paths["model"], "protocol": paths["protocol"]})
    if fresh.digest(paths["margin"]) != identity["source_sha256"]["margin"]:
        raise ValueError("margin source changed during the run")
    if fresh.digest(paths["screen_protocol"]) != identity["screen_protocol_sha256"]:
        raise ValueError("consumed screen protocol changed during the run")
    if (fresh.digest(SCREEN_RUN / "freeze.json") != identity["screen_freeze_sha256"] or
            fresh.digest(SCREEN_RUN / "screen-summary.json") != identity["screen_summary_sha256"]):
        raise ValueError("consumed screen receipts changed during the run")
    if fresh.digest(Path(__file__)) != identity["comparator_sha256"]:
        raise ValueError("development comparator changed during the run")


def cell_path(root: Path, arm: str, track: int, seed: int) -> Path:
    if arm not in ARMS:
        raise ValueError("unknown development arm")
    return root / "cells" / arm / f"track-{track}-seed-{seed}.json"


def prepare_run(root: Path, identity: dict) -> None:
    fresh.prepare_run(root, identity)


def load_cell(root: Path, identity: dict, arm: str, track: int, seed: int) -> dict | None:
    path = cell_path(root, arm, track, seed)
    if not path.exists():
        return None
    envelope = fresh._read_json(path)
    if not isinstance(envelope, dict) or envelope.get("identity") != identity:
        raise ValueError(f"cell identity mismatch: {path}")
    row = envelope.get("row")
    if not isinstance(row, dict) or (row.get("arm"), row.get("track_id"), row.get("seed")) != (arm, track, seed):
        raise ValueError(f"cell coordinates mismatch: {path}")
    if envelope.get("digest") != hashlib.sha256(fresh._canonical({"identity": identity, "row": row})).hexdigest():
        raise ValueError(f"cell digest mismatch: {path}")
    return row


def record_cell(root: Path, identity: dict, row: dict) -> bool:
    path = cell_path(root, row["arm"], row["track_id"], row["seed"])
    if path.exists():
        if load_cell(root, identity, row["arm"], row["track_id"], row["seed"]) != row:
            raise ValueError(f"existing cell differs: {path}")
        return False
    payload = {"identity": identity, "row": row}
    fresh._atomic_json(path, {**payload, "digest": hashlib.sha256(fresh._canonical(payload)).hexdigest()})
    return True


def compare_arms(rows: list[dict], cells: tuple[tuple[int, int], ...],
                 old: str, new: str) -> dict:
    if old not in ARMS or new not in ARMS or old == new:
        raise ValueError("invalid arm comparison")
    mapped = [{**row, "arm": "control" if row["arm"] == old else "candidate", "repeat": 0}
              for row in rows if row["arm"] in (old, new)]
    comparison = fresh.compare_pairs(mapped, [(track, seed, 0) for track, seed in cells])
    safety_gain = (comparison["candidate_contacts"] < comparison["control_contacts"] or
                   comparison["candidate_damage"] + 1e-9 < comparison["control_damage"])
    by_cell = {(row["track_id"], row["seed"], row["arm"]): row for row in rows}
    no_lap_slowdown = all(
        not (by_cell.get((track, seed, old), {}).get("finished") and
             by_cell.get((track, seed, new), {}).get("finished")) or
        by_cell[(track, seed, new)]["lap_time_ms"] <= by_cell[(track, seed, old)]["lap_time_ms"]
        for track, seed in cells
    )
    if (old, new) == ("margin", "ego_switch") and comparison["decision"] == "INCONCLUSIVE" and (
        comparison["candidate_finishes"] >= comparison["control_finishes"] and
        comparison["candidate_mean_progress"] + 1e-9 >= comparison["control_mean_progress"] and
        safety_gain and no_lap_slowdown
    ):
        # Development-only attribution: a strict safety improvement is useful
        # even when official rank metrics tie. The frozen fresh gate is untouched.
        comparison["decision"] = "RETAIN_SAFETY_REPAIR"
    return {**comparison, "control_arm": old, "candidate_arm": new,
            "evidence_scope": "reused development diagnosis only",
            "fresh_generalization": False, "sota_promotion": False}


def report(root: Path, identity: dict, cells: tuple[tuple[int, int], ...]) -> dict:
    rows = []
    for track, seed in cells:
        for arm in ARMS:
            row = load_cell(root, identity, arm, track, seed)
            if row is not None:
                rows.append(row)
    pairs = {"ego_vs_baseline": compare_arms(rows, cells, "baseline", "ego_switch"),
             "ego_vs_margin": compare_arms(rows, cells, "margin", "ego_switch"),
             "margin_vs_baseline_context": compare_arms(rows, cells, "baseline", "margin")}
    primary = (pairs["ego_vs_baseline"]["decision"], pairs["ego_vs_margin"]["decision"])
    decision = ("REJECT" if any(row.get("error") for row in rows) else
                "RETAIN_DIAGNOSTIC_CANDIDATE" if primary[0] == "RETAIN" and
                primary[1] in ("RETAIN", "RETAIN_SAFETY_REPAIR") else
                "REJECT" if "REJECT" in primary else
                "INCOMPLETE" if "INCOMPLETE" in primary else "INCONCLUSIVE")
    return {"decision": decision, "comparisons": pairs, "rows_recorded": len(rows),
            "episodes_expected": len(cells) * len(ARMS), "cells_expected": len(cells),
            "protocol_sha256": identity["protocol_sha256"],
            "screen_protocol_sha256": identity["screen_protocol_sha256"],
            "evidence_scope": "reused development diagnosis only",
            "fresh_generalization": False, "confirmation_unlocked": False,
            "blind_opened": False, "sota_promotion": False}


def run(root: Path, identity: dict, protocol: dict, paths: dict[str, Path],
        cells: tuple[tuple[int, int], ...]) -> dict:
    with fresh._run_lock(root):
        for track, seed in cells:
            for arm in ARMS:
                prior = load_cell(root, identity, arm, track, seed)
                if prior is not None:
                    if prior.get("error"):
                        raise RuntimeError(f"recorded operational failure: {arm}/{track}/{seed}")
                    continue
                check_frozen_inputs(identity, paths)
                payload = {"source": str(paths[arm].resolve()),
                           "model": str(paths["model"].resolve()),
                           "controller_class": protocol["controller_classes"][arm],
                           "track_id": track, "seed": seed}
                command = [sys.executable, str(Path(fresh.__file__).resolve()), "--worker",
                           json.dumps(payload, separators=(",", ":"))]
                row = {"arm": arm, "track_id": track, "seed": seed}
                try:
                    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                               timeout=300, check=True)
                    measured = json.loads(completed.stdout)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                    stderr = getattr(error, "stderr", "") or ""
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode("utf-8", errors="replace")
                    row.update({"error": str(error), "stderr": stderr[-4000:]})
                    record_cell(root, identity, row)
                    failure_summary = report(root, identity, cells)
                    fresh._atomic_json(root / "summary.json", failure_summary)
                    print(json.dumps(failure_summary, indent=2, allow_nan=False), flush=True)
                    raise RuntimeError(f"worker failed at {arm}/{track}/{seed}") from error
                check_frozen_inputs(identity, paths)
                row.update(measured)
                record_cell(root, identity, row)
                print(json.dumps({"cell": [arm, track, seed], "finished": row["finished"],
                                  "progress": row["progress"], "contacts": row["collision_count"]}), flush=True)
    summary = report(root, identity, cells)
    path = root / "summary.json"
    if path.exists() and fresh._read_json(path) != summary:
        raise ValueError("existing development summary differs")
    if not path.exists():
        fresh._atomic_json(path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--baseline-agent", required=True, type=Path)
    parser.add_argument("--margin-agent", required=True, type=Path)
    parser.add_argument("--ego-switch-agent", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol = fresh._read_json(args.protocol)
    screen = fresh._read_json(SCREEN_PROTOCOL)
    screen_hash = fresh.digest(SCREEN_PROTOCOL)
    cells = validate_protocol(protocol, screen, screen_hash,
                              sealed_holdout_seeds(ROOT / "experiments"))
    if not (SCREEN_RUN / "screen-summary.json").is_file():
        raise ValueError("consumed screen completion receipt is missing")
    screen_identity = fresh._read_json(SCREEN_RUN / "freeze.json")
    if screen_identity.get("protocol_sha256") != screen_hash:
        raise ValueError("consumed screen receipt protocol hash mismatch")
    if screen_identity.get("source_sha256") != {
        "control": screen["control_agent_sha256"],
        "candidate": screen["candidate_agent_sha256"]
    }:
        raise ValueError("consumed screen receipt source hash mismatch")
    consumed_count = verify_screen_consumed(screen, SCREEN_RUN, screen_identity)
    screen_summary = fresh.report_phase(SCREEN_RUN, screen_identity, screen, "screen")
    if fresh._read_json(SCREEN_RUN / "screen-summary.json") != screen_summary or screen_summary["decision"] != "REJECT":
        raise ValueError("consumed screen summary or spot checks changed")
    paths = {"baseline": args.baseline_agent, "margin": args.margin_agent,
             "ego_switch": args.ego_switch_agent, "model": args.model,
             "protocol": args.protocol, "screen_protocol": SCREEN_PROTOCOL}
    identity = build_identity(args.protocol, protocol, paths, screen_hash)
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "identity": identity,
                          "consumed_screen_cells": consumed_count,
                          "legacy_cells": len(LEGACY_CELLS),
                          "development_cells": len(cells),
                          "cold_episodes": len(cells) * len(ARMS),
                          "fresh_generalization": False}, indent=2))
        return 0
    prepare_run(args.output_root, identity)
    summary = run(args.output_root, identity, protocol, paths, cells)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
