"""Source-bound v2 corridor comparison on consumed development cells only.

This wrapper keeps the v1 cell boundary, screen receipts, worker, and triage
logic frozen while requiring the fallback-speed route and a separate v2 freeze.
It never opens confirmation or blind partitions or makes a promotion claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import compare_feasible_corridor_dev as v1
from tools import compare_reused_bare_candidate as prior
from tools import evaluate_bare_generalization as fresh

NAME = "feasible-corridor-dev-v2"
CLASSES = {"baseline": "_CompoundClearingBrakeCarryController",
           "candidate": "_FeasibleCorridorFallbackSpeedController"}
PROTOCOL_PATH = ROOT / "experiments" / f"{NAME}.json"
OUTPUT_PARENT = ROOT / ".haic-artifacts" / NAME
OUTPUT_ROOT = OUTPUT_PARENT / "run"
MECHANISM_CELLS = v1.MECHANISM_CELLS
PRIOR_V1_PATHS = {
    "protocol": ROOT / "experiments" / "feasible-corridor-dev-v1.json",
    "result": ROOT / "experiments" / "feasible-corridor-dev-v1-result.json",
    "triage_summary": ROOT / ".haic-artifacts" / "feasible-corridor-dev-v1" / "run" / "mechanism-summary.json",
    "freeze": ROOT / ".haic-artifacts" / "feasible-corridor-dev-v1" / "run" / "freeze.json",
    "full_summary": ROOT / ".haic-artifacts" / "feasible-corridor-dev-v1" / "run" / "summary.json",
}


def _require_v2_route(protocol: dict) -> None:
    if not isinstance(protocol, dict) or protocol.get("name") != NAME:
        raise ValueError("v2 protocol name is required")
    if protocol.get("controller_classes") != CLASSES:
        raise ValueError("v2 fallback-speed actual Agent route is required")


def validate_protocol_path(path: Path) -> Path:
    path = path.resolve()
    if path != PROTOCOL_PATH.resolve():
        raise ValueError("v2 requires experiments/feasible-corridor-dev-v2.json")
    return path


def validate_output_root(root: Path) -> Path:
    root = v1.validate_output_root(root)
    if root != OUTPUT_ROOT.resolve():
        raise ValueError("output root must be exactly .haic-artifacts/feasible-corridor-dev-v2/run")
    return root


def validate_prior_development(protocol: dict) -> dict:
    pinned = protocol.get("prior_development_v1")
    if (not isinstance(pinned, dict)
            or set(pinned) != {"protocol_sha256", "result_sha256", "triage_summary_sha256", "status"}
            or pinned["status"] != "TRIAGE_BLOCK_FULL"):
        raise ValueError("prior v1 development proof must pin the blocked triage")
    for key in ("protocol_sha256", "result_sha256", "triage_summary_sha256"):
        fresh._sha256_string(pinned[key], f"prior_development_v1.{key}")
    for key, path_key in (("protocol_sha256", "protocol"), ("result_sha256", "result"),
                          ("triage_summary_sha256", "triage_summary")):
        path = PRIOR_V1_PATHS[path_key]
        if not path.is_file() or fresh.digest(path) != pinned[key]:
            raise ValueError(f"prior v1 {path_key} hash mismatch")
    if PRIOR_V1_PATHS["full_summary"].exists():
        raise ValueError("prior v1 full comparison was unexpectedly opened")
    freeze_path = PRIOR_V1_PATHS["freeze"]
    if not freeze_path.is_file():
        raise ValueError("prior v1 frozen identity is missing")
    result = fresh._read_json(PRIOR_V1_PATHS["result"])
    summary = fresh._read_json(PRIOR_V1_PATHS["triage_summary"])
    freeze = fresh._read_json(freeze_path)
    freeze_sha256 = fresh.digest(freeze_path)
    comparator_sha256 = fresh.digest(Path(v1.__file__))
    if (result.get("experiment") != "feasible-corridor-dev-v1"
            or result.get("status") != pinned["status"]
            or result.get("protocol_sha256") != pinned["protocol_sha256"]
            or result.get("mechanism_summary_sha256") != pinned["triage_summary_sha256"]
            or result.get("freeze_sha256") != freeze_sha256
            or summary.get("decision") != "TRIAGE_BLOCK_FULL"
            or summary.get("protocol_sha256") != pinned["protocol_sha256"]
            or summary.get("triage_gate", {}).get("status") != "BLOCK_FULL"
            or freeze.get("protocol_sha256") != pinned["protocol_sha256"]
            or freeze.get("comparator_sha256") != comparator_sha256):
        raise ValueError("prior v1 result, triage gate, or frozen comparator lineage mismatch")
    ego_path = v1.EGO_SCREEN_PROTOCOL
    if fresh.digest(ego_path) != freeze.get("ego_screen_protocol_sha256"):
        raise ValueError("prior v1 ego screen protocol changed")
    ego_cells = v1.ego_screen_cells(fresh._read_json(ego_path))
    receipt_root = freeze_path.parent
    for track, seed in MECHANISM_CELLS:
        for arm in v1.ARMS:
            row = v1.load_cell(receipt_root, freeze, arm, track, seed)
            if row is None or row.get("error"):
                raise ValueError(f"prior v1 triage receipt missing or failed: {arm}/{track}/{seed}")
    recomputed = v1.report(receipt_root, freeze, MECHANISM_CELLS,
                           mechanism_only=True, ego_cells=ego_cells)
    if recomputed != summary:
        raise ValueError("prior v1 triage summary differs from frozen receipts")
    return {**pinned, "freeze_sha256": freeze_sha256,
            "comparator_sha256": comparator_sha256}


def validate_protocol(protocol: dict, margin: dict, ego: dict, margin_sha256: str,
                      ego_sha256: str, sealed_seeds: set[int]) -> tuple[tuple[int, int], ...]:
    _require_v2_route(protocol)
    validate_prior_development(protocol)
    delegated = {**protocol, "name": "feasible-corridor-dev-v1",
                 "controller_classes": v1.CLASSES}
    return v1.validate_protocol(delegated, margin, ego, margin_sha256,
                                ego_sha256, sealed_seeds)


def build_identity(protocol_path: Path, protocol: dict, paths: dict[str, Path],
                   margin: dict, ego: dict) -> dict:
    _require_v2_route(protocol)
    prior_proof = validate_prior_development(protocol)
    inherited = v1.build_identity(protocol_path, protocol, paths, margin, ego)
    return {**inherited, "wrapper_sha256": fresh.digest(Path(__file__)),
            "protocol_name": NAME, "candidate_route_class": CLASSES["candidate"],
            "prior_development_v1": prior_proof}


def check_frozen_inputs(identity: dict, paths: dict[str, Path],
                        screens: dict[str, dict]) -> None:
    if (identity.get("wrapper_sha256") != fresh.digest(Path(__file__))
            or identity.get("protocol_name") != NAME
            or identity.get("candidate_route_class") != CLASSES["candidate"]):
        raise ValueError("v2 wrapper or route changed during development")
    v1.check_frozen_inputs(identity, paths, screens)
    if validate_prior_development(fresh._read_json(paths["protocol"])) != identity.get("prior_development_v1"):
        raise ValueError("prior v1 development proof changed during v2 run")


def report(root: Path, identity: dict, cells: tuple[tuple[int, int], ...],
           *, mechanism_only: bool, ego_cells: tuple[tuple[int, int], ...]) -> dict:
    summary = v1.report(root, identity, cells, mechanism_only=mechanism_only,
                        ego_cells=ego_cells)
    return {**summary, "protocol_name": NAME,
            "candidate_route_class": CLASSES["candidate"],
            "wrapper_sha256": identity["wrapper_sha256"]}


def require_triage_gate(root: Path, identity: dict,
                        ego_cells: tuple[tuple[int, int], ...]) -> dict:
    path = root / "mechanism-summary.json"
    if not path.is_file():
        raise ValueError("v2 mechanism triage summary missing; run --mechanism-only first")
    for track, seed in MECHANISM_CELLS:
        for arm in v1.ARMS:
            if v1.load_cell(root, identity, arm, track, seed) is None:
                raise ValueError("v2 mechanism triage incomplete: all 20 rows are required")
    current = report(root, identity, MECHANISM_CELLS,
                     mechanism_only=True, ego_cells=ego_cells)
    if fresh._read_json(path) != current:
        raise ValueError("v2 mechanism triage summary differs from its 20 frozen rows")
    if current["triage_gate"]["status"] != "ALLOW_FULL_CONSUMED_DEVELOPMENT":
        raise ValueError("v2 mechanism triage blocked full development; a new protocol is required")
    return current


def validate_run_scope(protocol: dict, cells: tuple[tuple[int, int], ...],
                       ego_cells: tuple[tuple[int, int], ...], mechanism_only: bool) -> None:
    _require_v2_route(protocol)
    margin = fresh._read_json(v1.MARGIN_SCREEN_PROTOCOL)
    ego = fresh._read_json(v1.EGO_SCREEN_PROTOCOL)
    full = v1.expected_development_cells(margin, ego)
    expected_ego = v1.ego_screen_cells(ego)
    selected = MECHANISM_CELLS if mechanism_only else full
    if (type(mechanism_only) is not bool
            or protocol.get("development_cells") != [list(cell) for cell in full]
            or protocol.get("mechanism_cells") != [list(cell) for cell in MECHANISM_CELLS]
            or cells != selected or ego_cells != expected_ego):
        raise ValueError("v2 run scope must be the exact protocol-selected 10 or 55 cells and 32 ego cells")


def run(root: Path, identity: dict, protocol: dict, paths: dict[str, Path],
        screens: dict[str, dict], ego_identity: dict,
        cells: tuple[tuple[int, int], ...], ego_cells: tuple[tuple[int, int], ...],
        *, mechanism_only: bool) -> dict:
    validate_run_scope(protocol, cells, ego_cells, mechanism_only)
    root = validate_output_root(root)
    fresh.prepare_run(root, identity)
    with fresh._run_lock(root):
        if not mechanism_only:
            require_triage_gate(root, identity, ego_cells)
        for (track, seed), arm, origin in v1.work_plan(cells, ego_cells):
            existing = v1.load_cell(root, identity, arm, track, seed)
            if existing is not None:
                if existing.get("error"):
                    raise RuntimeError(f"recorded operational failure: {arm}/{track}/{seed}")
                continue
            check_frozen_inputs(identity, paths, screens)
            if origin == "reuse":
                row = v1.import_ego_control(v1.EGO_SCREEN_RUN, ego_identity,
                                            root, identity, track, seed)
                check_frozen_inputs(identity, paths, screens)
            else:
                row = {"arm": arm, "track_id": track, "seed": seed,
                       "receipt_origin": "COLD_WORKER"}
                try:
                    row.update(v1._run_worker(arm, track, seed, protocol, paths))
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                        json.JSONDecodeError, ValueError) as error:
                    stderr = getattr(error, "stderr", "") or ""
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode("utf-8", errors="replace")
                    row.update(error=str(error), stderr=stderr[-4000:])
                    v1.record_cell(root, identity, row)
                    failure = report(root, identity, cells, mechanism_only=mechanism_only,
                                     ego_cells=ego_cells)
                    fresh._atomic_json(root / ("mechanism-summary.json" if mechanism_only else "summary.json"),
                                       failure)
                    raise RuntimeError(f"worker failed: {arm}/{track}/{seed}") from error
                check_frozen_inputs(identity, paths, screens)
                v1.record_cell(root, identity, row)
            print(json.dumps({"cell": [arm, track, seed], "receipt_origin": row["receipt_origin"],
                              "finished": row["finished"], "progress": row["progress"],
                              "contacts": row["collision_count"]}), flush=True)
    check_frozen_inputs(identity, paths, screens)
    summary = report(root, identity, cells, mechanism_only=mechanism_only,
                     ego_cells=ego_cells)
    summary_path = root / ("mechanism-summary.json" if mechanism_only else "summary.json")
    if summary_path.exists() and fresh._read_json(summary_path) != summary:
        raise ValueError(f"existing v2 development summary differs: {summary_path}")
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
    protocol_path = validate_protocol_path(args.protocol)
    output_root = validate_output_root(args.output_root)
    margin = fresh._read_json(v1.MARGIN_SCREEN_PROTOCOL)
    ego = fresh._read_json(v1.EGO_SCREEN_PROTOCOL)
    protocol = fresh._read_json(protocol_path)
    cells = validate_protocol(protocol, margin, ego,
                              fresh.digest(v1.MARGIN_SCREEN_PROTOCOL),
                              fresh.digest(v1.EGO_SCREEN_PROTOCOL),
                              prior.sealed_holdout_seeds(ROOT / "experiments"))
    paths = {"protocol": protocol_path, "baseline": args.baseline_agent,
             "candidate": args.candidate_agent, "model": args.model,
             "margin_screen_protocol": v1.MARGIN_SCREEN_PROTOCOL,
             "ego_screen_protocol": v1.EGO_SCREEN_PROTOCOL}
    frozen = build_identity(protocol_path, protocol, paths, margin, ego)
    ego_identity = fresh._read_json(v1.EGO_SCREEN_RUN / "freeze.json")
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "evidence_scope": "CONSUMED_DEVELOPMENT_ONLY",
                          "identity": frozen, "development_cells": len(cells),
                          "mechanism_cells": len(MECHANISM_CELLS),
                          "cold_worker_episodes_full": 78,
                          "reused_ego_control_receipts": 32}, indent=2))
        return 0
    selected = MECHANISM_CELLS if args.mechanism_only else cells
    summary = run(output_root, frozen, protocol, paths,
                  {"margin": margin, "ego": ego}, ego_identity, selected,
                  v1.ego_screen_cells(ego), mechanism_only=args.mechanism_only)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
