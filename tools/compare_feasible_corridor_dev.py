"""Compare a camera corridor candidate on exactly 55 consumed development cells.

This tool never opens confirmation or blind partitions, freezes a finalist, or
claims fresh generalization. Every new episode uses the actual Agent route in
the frozen generalization worker. The completed ego screen's control receipts
are reused only after exact source and environment lineage checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import compare_reused_bare_candidate as prior
from tools import evaluate_bare_generalization as fresh

ARMS = ("baseline", "candidate")
CLASSES = {"baseline": "_CompoundClearingBrakeCarryController",
           "candidate": "_FeasibleCorridorObstacleController"}
MARGIN_SCREEN_PROTOCOL = ROOT / "experiments" / "observed-margin-generalization-v1.json"
MARGIN_SCREEN_RUN = ROOT / ".haic-artifacts" / "observed-margin-generalization-v1" / "run"
EGO_SCREEN_PROTOCOL = ROOT / "experiments" / "observed-ego-side-switch-generalization-v1.json"
EGO_SCREEN_RUN = ROOT / ".haic-artifacts" / "observed-ego-side-switch-generalization-v1" / "run"
MECHANISM_CELLS = (
    (1, 3857792434), (3, 3857792434), (1, 3892761381), (1, 42),
    (2, 644062), (4, 2951861974), (1, 17), (3, 2951861974),
    (1, 1190129265), (2, 4089604952),
)


def _screen_cells(protocol: dict, name: str, geometry_count: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(protocol, dict) or protocol.get("name") != name:
        raise ValueError(f"wrong consumed screen protocol: {name}")
    partitions = protocol.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"screen", "confirmation", "blind"}:
        raise ValueError(f"{name} needs the exact three frozen partitions")
    screen = partitions["screen"]
    tracks, seeds = screen.get("track_ids"), screen.get("seeds")
    if (tracks != [1, 2, 3, 4] or not isinstance(seeds, list)
            or len(seeds) != geometry_count or len(set(seeds)) != geometry_count
            or any(type(seed) is not int or not 0 <= seed <= fresh.MAX_SEED for seed in seeds)):
        raise ValueError(f"{name} screen must be four IDs by {geometry_count} distinct seeds")
    cells = tuple((track, seed) for track in tracks for seed in seeds)
    if len(set(cells)) != 4 * geometry_count:
        raise ValueError(f"{name} screen has duplicate cells")
    return cells


def margin_screen_cells(protocol: dict) -> tuple[tuple[int, int], ...]:
    return _screen_cells(protocol, "observed-margin-generalization-v1", 4)


def ego_screen_cells(protocol: dict) -> tuple[tuple[int, int], ...]:
    return _screen_cells(protocol, "observed-ego-side-switch-generalization-v1", 8)


def expected_development_cells(margin: dict, ego: dict) -> tuple[tuple[int, int], ...]:
    cells = prior.LEGACY_CELLS + margin_screen_cells(margin) + ego_screen_cells(ego)
    seeds = [seed for _, seed in prior.LEGACY_CELLS]
    seeds += margin["partitions"]["screen"]["seeds"]
    seeds += ego["partitions"]["screen"]["seeds"]
    if len(cells) != 55 or len(set(cells)) != 55 or len(seeds) != len(set(seeds)):
        raise ValueError("consumed development geometry seeds overlap")
    return cells


def validate_protocol(protocol: dict, margin: dict, ego: dict, margin_sha256: str,
                      ego_sha256: str, sealed_seeds: set[int]) -> tuple[tuple[int, int], ...]:
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("development protocol schema_version must be 1")
    if protocol.get("name") != "feasible-corridor-dev-v1" or protocol.get("status") != "PREREGISTERED":
        raise ValueError("wrong or unregistered corridor development protocol")
    if protocol.get("evidence_scope") != "CONSUMED_DEVELOPMENT_ONLY":
        raise ValueError("evidence_scope must be CONSUMED_DEVELOPMENT_ONLY")
    if protocol.get("max_steps") != 2000 or protocol.get("frame_skip") != 4:
        raise ValueError("official evaluation requires max_steps 2000 and frame_skip 4")
    for prefix, screen_hash in (("margin", margin_sha256), ("ego", ego_sha256)):
        for suffix in ("protocol", "freeze", "summary", "receipts"):
            fresh._sha256_string(protocol.get(f"{prefix}_screen_{suffix}_sha256"),
                                 f"{prefix}_screen_{suffix}_sha256")
        if protocol[f"{prefix}_screen_protocol_sha256"] != screen_hash:
            raise ValueError(f"{prefix} screen protocol hash mismatch")
    if protocol.get("controller_classes") != CLASSES:
        raise ValueError("unexpected actual Agent routes")
    source_hashes = protocol.get("agent_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(ARMS):
        raise ValueError("both agent source hashes are required")
    for arm in ARMS:
        fresh._sha256_string(source_hashes[arm], f"agent_sha256.{arm}")
    if source_hashes["baseline"] != ego.get("control_agent_sha256"):
        raise ValueError("baseline must be current ego screen's active control source")
    fresh._sha256_string(protocol.get("model_sha256"), "model_sha256")
    if protocol["model_sha256"] != margin.get("model_sha256") or protocol["model_sha256"] != ego.get("model_sha256"):
        raise ValueError("screen and development model hashes differ")
    helpers = protocol.get("runtime_helper_sha256")
    if not isinstance(helpers, dict) or set(helpers) != {"action_smoothing.py", "action_representation.py"}:
        raise ValueError("both runtime helper hashes are required")
    for name, value in helpers.items():
        fresh._sha256_string(value, name)
    if helpers != margin.get("runtime_helper_sha256") or helpers != ego.get("runtime_helper_sha256"):
        raise ValueError("screen and development helper hashes differ")
    cells = expected_development_cells(margin, ego)
    if protocol.get("development_cells") != [list(cell) for cell in cells]:
        raise ValueError("development_cells must be the exact 55 consumed cells in order")
    if protocol.get("mechanism_cells") != [list(cell) for cell in MECHANISM_CELLS]:
        raise ValueError("mechanism_cells must match the fixed ten-cell triage")
    if not set(MECHANISM_CELLS).issubset(cells):
        raise ValueError("mechanism cells are outside the consumed development grid")
    overlap = {seed for _, seed in cells} & sealed_seeds
    if overlap:
        raise ValueError(f"confirmation/blind geometry seed requested: {sorted(overlap)}")
    gates = protocol.get("gates")
    if (not isinstance(gates, dict) or gates.get("fresh_confirmation") != "SEALED"
            or gates.get("blind") != "SEALED" or gates.get("sota_promotion") is not False):
        raise ValueError("development diagnosis cannot open confirmation/blind or SOTA promotion")
    return cells


def validate_output_root(root: Path) -> Path:
    root = root.resolve()
    artifacts = (ROOT / ".haic-artifacts").resolve()
    if root == artifacts or not root.is_relative_to(artifacts):
        raise ValueError("output root must be a child of .haic-artifacts")
    return root


def _receipt_manifest(protocol: dict, root: Path) -> str:
    entries = {}
    for track, seed, repeat in fresh.expected_cells(protocol, "screen"):
        for arm in fresh.ARMS:
            path = fresh.cell_path(root, "screen", arm, track, seed, repeat)
            if not path.is_file():
                raise ValueError(f"missing consumed screen receipt: {path}")
            entries[str(path.relative_to(root)).replace("\\", "/")] = fresh.digest(path)
    return hashlib.sha256(fresh._canonical(entries)).hexdigest()


def verify_rejected_screen(protocol: dict, root: Path, protocol_sha256: str,
                           reference: dict) -> int:
    """Fail closed unless every frozen screen receipt reproduces its REJECT summary."""
    if not (root / "freeze.json").is_file() or not (root / "screen-summary.json").is_file():
        raise ValueError("completed consumed screen freeze or summary is missing")
    identity = fresh._read_json(root / "freeze.json")
    if identity.get("protocol_sha256") != protocol_sha256:
        raise ValueError("consumed screen protocol hash mismatch")
    if identity.get("source_sha256") != {
        "control": protocol.get("control_agent_sha256"),
        "candidate": protocol.get("candidate_agent_sha256"),
    }:
        raise ValueError("consumed screen source hash mismatch")
    for key, value in reference.items():
        if identity.get(key) != value:
            raise ValueError(f"consumed screen {key} lineage mismatch")
    rows = []
    for track, seed, repeat in fresh.expected_cells(protocol, "screen"):
        for arm in fresh.ARMS:
            row = fresh.load_cell(root, identity, "screen", arm, track, seed, repeat)
            if row is None or row.get("error"):
                raise ValueError(f"unconsumed or failed screen cell: {track}/{seed}/{arm}/{repeat}")
            fresh._sha256_string(row.get("action_trace_sha256"), "action_trace_sha256")
            rows.append(row)
    spot_failures = fresh._verify_spot_checks(rows, protocol, "screen")
    if spot_failures:
        raise ValueError(f"consumed screen spot check failed: {spot_failures}")
    count = len(protocol["partitions"]["screen"]["track_ids"]) * len(
        protocol["partitions"]["screen"]["seeds"])
    summary = fresh.report_phase(root, identity, protocol, "screen")
    if fresh._read_json(root / "screen-summary.json") != summary or summary["decision"] != "REJECT":
        raise ValueError("consumed screen summary or spot checks changed")
    _receipt_manifest(protocol, root)
    for phase in ("confirmation", "blind"):
        if (root / "cells" / phase).exists():
            raise ValueError(f"consumed screen {phase} partition was opened")
    return count


def build_identity(protocol_path: Path, protocol: dict, paths: dict[str, Path],
                   margin: dict, ego: dict) -> dict:
    synthetic = {
        "control_agent_sha256": protocol["agent_sha256"]["baseline"],
        "candidate_agent_sha256": protocol["agent_sha256"]["candidate"],
        "model_sha256": protocol["model_sha256"],
        "runtime_helper_sha256": protocol["runtime_helper_sha256"],
    }
    identity = fresh.build_identity(protocol_path, synthetic, paths["baseline"],
                                    paths["candidate"], paths["model"])
    identity["source_sha256"] = {
        "baseline": identity["source_sha256"]["control"],
        "candidate": identity["source_sha256"]["candidate"],
    }
    reference = {key: identity[key] for key in (
        "model_sha256", "helper_sha256", "harness_sha256", "environment_sha256",
        "platform", "python",
    )}
    for prefix, screen, run in (("margin", margin, MARGIN_SCREEN_RUN),
                                ("ego", ego, EGO_SCREEN_RUN)):
        screen_path = paths[f"{prefix}_screen_protocol"]
        hashes = {
            "protocol": fresh.digest(screen_path),
            "freeze": fresh.digest(run / "freeze.json"),
            "summary": fresh.digest(run / "screen-summary.json"),
        }
        for suffix, observed in hashes.items():
            expected = protocol[f"{prefix}_screen_{suffix}_sha256"]
            if observed != expected:
                raise ValueError(f"{prefix} screen {suffix} hash mismatch")
            identity[f"{prefix}_screen_{suffix}_sha256"] = observed
        verify_rejected_screen(screen, run, hashes["protocol"], reference)
        receipts_hash = _receipt_manifest(screen, run)
        if receipts_hash != protocol[f"{prefix}_screen_receipts_sha256"]:
            raise ValueError(f"{prefix} screen receipts hash mismatch")
        identity[f"{prefix}_screen_receipts_sha256"] = receipts_hash
    identity["comparator_sha256"] = fresh.digest(Path(__file__))
    identity["evidence_scope"] = "CONSUMED_DEVELOPMENT_ONLY"
    return identity


def check_frozen_inputs(identity: dict, paths: dict[str, Path],
                        screens: dict[str, dict]) -> None:
    virtual = {**identity, "source_sha256": {
        "control": identity["source_sha256"]["baseline"],
        "candidate": identity["source_sha256"]["candidate"],
    }}
    fresh.check_frozen_inputs(virtual, {
        "control": paths["baseline"], "candidate": paths["candidate"],
        "model": paths["model"], "protocol": paths["protocol"],
    })
    for prefix, run in (("margin", MARGIN_SCREEN_RUN), ("ego", EGO_SCREEN_RUN)):
        for suffix, path in (("protocol", paths[f"{prefix}_screen_protocol"]),
                             ("freeze", run / "freeze.json"),
                             ("summary", run / "screen-summary.json")):
            if fresh.digest(path) != identity[f"{prefix}_screen_{suffix}_sha256"]:
                raise ValueError(f"{prefix} screen {suffix} changed during development")
        if _receipt_manifest(screens[prefix], run) != identity[f"{prefix}_screen_receipts_sha256"]:
            raise ValueError(f"{prefix} consumed screen receipts changed during development")
    if fresh.digest(Path(__file__)) != identity["comparator_sha256"]:
        raise ValueError("development comparator changed during the run")


def work_plan(cells: tuple[tuple[int, int], ...],
              ego_cells: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], str, str], ...]:
    ego_set = set(ego_cells)
    if len(ego_set) != 32 or len(set(cells)) != len(cells):
        raise ValueError("current screen reuse requires exactly 32 consumed cells")
    return tuple((cell, arm, "reuse" if arm == "baseline" and cell in ego_set else "worker")
                 for cell in cells for arm in ARMS)


def cell_path(root: Path, arm: str, track: int, seed: int) -> Path:
    if arm not in ARMS:
        raise ValueError("unknown development arm")
    return root / "cells" / arm / f"track-{track}-seed-{seed}.json"


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


def import_ego_control(screen_root: Path, screen_identity: dict, output_root: Path,
                       dev_identity: dict, track: int, seed: int) -> dict:
    original = fresh.load_cell(screen_root, screen_identity, "screen", "control", track, seed, 0)
    if original is None or original.get("error"):
        raise ValueError(f"missing or failed ego screen control receipt: {track}/{seed}")
    fresh._sha256_string(original.get("action_trace_sha256"), "action_trace_sha256")
    receipt = fresh.cell_path(screen_root, "screen", "control", track, seed, 0)
    imported = {key: value for key, value in original.items()
                if key not in ("partition", "arm", "repeat")}
    imported.update(arm="baseline", receipt_origin="EGO_SCREEN_CONTROL",
                    source_receipt_sha256=fresh.digest(receipt))
    record_cell(output_root, dev_identity, imported)
    return imported


def report(root: Path, identity: dict, cells: tuple[tuple[int, int], ...],
           *, mechanism_only: bool) -> dict:
    rows = []
    for track, seed in cells:
        for arm in ARMS:
            current = load_cell(root, identity, arm, track, seed)
            if current is not None:
                rows.append(current)
    mapped = [{**row, "arm": "control" if row["arm"] == "baseline" else "candidate",
               "repeat": 0} for row in rows]
    comparison = fresh.compare_pairs(mapped, [(track, seed, 0) for track, seed in cells])
    if mechanism_only:
        decision = ("REJECT_OPERATIONAL" if any(row.get("error") for row in rows) else
                    "INCOMPLETE" if comparison["missing_cells"] else
                    "MECHANISM_DIAGNOSTIC")
    else:
        decision = ("REJECT" if comparison["decision"] == "REJECT" else
                    "INCOMPLETE" if comparison["decision"] == "INCOMPLETE" else
                    "RETAIN_DIAGNOSTIC_CANDIDATE" if comparison["decision"] == "RETAIN" else
                    "INCONCLUSIVE")
    return {
        "decision": decision, "comparison": comparison,
        "rows_recorded": len(rows), "cells_expected": len(cells),
        "episodes_expected": len(cells) * len(ARMS),
        "reused_baseline_recorded": sum(row.get("receipt_origin") == "EGO_SCREEN_CONTROL" for row in rows),
        "distinct_geometry_seeds": len({seed for _, seed in cells}),
        "mechanism_only": mechanism_only,
        "protocol_sha256": identity["protocol_sha256"],
        "margin_screen_protocol_sha256": identity.get("margin_screen_protocol_sha256"),
        "ego_screen_protocol_sha256": identity.get("ego_screen_protocol_sha256"),
        "candidate_agent_sha256": identity.get("source_sha256", {}).get("candidate"),
        "evidence_scope": "CONSUMED_DEVELOPMENT_ONLY",
        "fresh_generalization": False, "confirmation_unlocked": False,
        "blind_opened": False, "sota_promotion": False,
    }


def _run_worker(arm: str, track: int, seed: int, protocol: dict,
                paths: dict[str, Path]) -> dict:
    payload = {
        "source": str(paths[arm].resolve()), "model": str(paths["model"].resolve()),
        "controller_class": protocol["controller_classes"][arm],
        "track_id": track, "seed": seed,
    }
    command = [sys.executable, str(Path(fresh.__file__).resolve()), "--worker",
               json.dumps(payload, separators=(",", ":"))]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                               timeout=300, check=True)
    measured = json.loads(completed.stdout)
    if not isinstance(measured, dict) or measured.get("error") is not None:
        raise ValueError("cold worker returned an invalid result")
    fresh._sha256_string(measured.get("action_trace_sha256"), "action_trace_sha256")
    if type(measured.get("steps")) is not int or not 0 <= measured["steps"] <= 2000:
        raise ValueError("cold worker returned an invalid action count")
    return measured


def run(root: Path, identity: dict, protocol: dict, paths: dict[str, Path],
        screens: dict[str, dict], ego_identity: dict,
        cells: tuple[tuple[int, int], ...], ego_cells: tuple[tuple[int, int], ...],
        *, mechanism_only: bool) -> dict:
    root = validate_output_root(root)
    fresh.prepare_run(root, identity)
    with fresh._run_lock(root):
        for (track, seed), arm, origin in work_plan(cells, ego_cells):
            existing = load_cell(root, identity, arm, track, seed)
            if existing is not None:
                if existing.get("error"):
                    raise RuntimeError(f"recorded operational failure: {arm}/{track}/{seed}")
                continue
            check_frozen_inputs(identity, paths, screens)
            if origin == "reuse":
                row = import_ego_control(EGO_SCREEN_RUN, ego_identity, root, identity, track, seed)
                check_frozen_inputs(identity, paths, screens)
            else:
                row = {"arm": arm, "track_id": track, "seed": seed,
                       "receipt_origin": "COLD_WORKER"}
                try:
                    row.update(_run_worker(arm, track, seed, protocol, paths))
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                        json.JSONDecodeError, ValueError) as error:
                    stderr = getattr(error, "stderr", "") or ""
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode("utf-8", errors="replace")
                    row.update(error=str(error), stderr=stderr[-4000:])
                    record_cell(root, identity, row)
                    failure = report(root, identity, cells, mechanism_only=mechanism_only)
                    fresh._atomic_json(root / ("mechanism-summary.json" if mechanism_only else "summary.json"), failure)
                    raise RuntimeError(f"worker failed: {arm}/{track}/{seed}") from error
                check_frozen_inputs(identity, paths, screens)
                record_cell(root, identity, row)
            print(json.dumps({"cell": [arm, track, seed], "receipt_origin": row["receipt_origin"],
                              "finished": row["finished"], "progress": row["progress"],
                              "contacts": row["collision_count"]}), flush=True)
    check_frozen_inputs(identity, paths, screens)
    summary = report(root, identity, cells, mechanism_only=mechanism_only)
    summary_path = root / ("mechanism-summary.json" if mechanism_only else "summary.json")
    if summary_path.exists() and fresh._read_json(summary_path) != summary:
        raise ValueError(f"existing consumed development summary differs: {summary_path}")
    if not summary_path.exists():
        fresh._atomic_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--baseline-agent", required=True, type=Path)
    parser.add_argument("--candidate-agent", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--mechanism-only", action="store_true")
    args = parser.parse_args()
    output_root = validate_output_root(args.output_root)
    margin = fresh._read_json(MARGIN_SCREEN_PROTOCOL)
    ego = fresh._read_json(EGO_SCREEN_PROTOCOL)
    margin_hash, ego_hash = fresh.digest(MARGIN_SCREEN_PROTOCOL), fresh.digest(EGO_SCREEN_PROTOCOL)
    protocol = fresh._read_json(args.protocol)
    cells = validate_protocol(protocol, margin, ego, margin_hash, ego_hash,
                              prior.sealed_holdout_seeds(ROOT / "experiments"))
    paths = {"protocol": args.protocol, "baseline": args.baseline_agent,
             "candidate": args.candidate_agent, "model": args.model,
             "margin_screen_protocol": MARGIN_SCREEN_PROTOCOL,
             "ego_screen_protocol": EGO_SCREEN_PROTOCOL}
    identity = build_identity(args.protocol, protocol, paths, margin, ego)
    ego_identity = fresh._read_json(EGO_SCREEN_RUN / "freeze.json")
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "evidence_scope": "CONSUMED_DEVELOPMENT_ONLY",
                          "identity": identity, "development_cells": len(cells),
                          "mechanism_cells": len(MECHANISM_CELLS),
                          "cold_worker_episodes_full": 78,
                          "reused_ego_control_receipts": 32}, indent=2))
        return 0
    selected = MECHANISM_CELLS if args.mechanism_only else cells
    summary = run(output_root, identity, protocol, paths,
                  {"margin": margin, "ego": ego}, ego_identity, selected,
                  ego_screen_cells(ego), mechanism_only=args.mechanism_only)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
