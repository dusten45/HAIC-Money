"""Source-bound v4 temporal-side comparison on consumed development cells.

The v1/v2/v3 comparators and their blocked evidence remain frozen. This wrapper
requires an independently pinned v3 failure proof and a separate v4 triage
before any full consumed-cell comparison. No fresh partition is opened here.
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
from tools import compare_feasible_corridor_dev as v1
from tools import compare_feasible_corridor_dev_v2 as v2
from tools import compare_feasible_corridor_dev_v3 as v3
from tools import compare_reused_bare_candidate as prior
from tools import evaluate_bare_generalization as fresh

NAME = "feasible-corridor-dev-v4"
CLASSES = {"baseline": "_CompoundClearingBrakeCarryController",
           "candidate": "_FeasibleCorridorTemporalSideController"}
PROTOCOL_PATH = ROOT / "experiments" / f"{NAME}.json"
OUTPUT_PARENT = ROOT / ".haic-artifacts" / NAME
OUTPUT_ROOT = OUTPUT_PARENT / "run"
MECHANISM_CELLS = v1.MECHANISM_CELLS
PRIOR_V3_PATHS = {
    "protocol": v3.PROTOCOL_PATH,
    "result": ROOT / "experiments" / "feasible-corridor-dev-v3-result.json",
    "triage_summary": v3.OUTPUT_ROOT / "mechanism-summary.json",
    "freeze": v3.OUTPUT_ROOT / "freeze.json",
    "full_summary": v3.OUTPUT_ROOT / "summary.json",
}


def _require_v4_route(protocol: dict) -> None:
    if not isinstance(protocol, dict) or protocol.get("name") != NAME:
        raise ValueError("v4 protocol name is required")
    if protocol.get("controller_classes") != CLASSES:
        raise ValueError("v4 temporal-side actual Agent route is required")


def validate_protocol_path(path: Path) -> Path:
    path = path.resolve()
    if path != PROTOCOL_PATH.resolve():
        raise ValueError("v4 requires experiments/feasible-corridor-dev-v4.json")
    return path


def validate_output_root(root: Path) -> Path:
    root = v1.validate_output_root(root)
    if root != OUTPUT_ROOT.resolve():
        raise ValueError("v4 output root must be exactly .haic-artifacts/feasible-corridor-dev-v4/run")
    return root


def _v3_protocol_view(protocol: dict) -> dict:
    return {**protocol, "name": v3.NAME, "controller_classes": v3.CLASSES}


def _v3_identity_view(identity: dict) -> dict:
    return {**identity, "wrapper_sha256": identity["wrapper_v3_sha256"],
            "protocol_name": v3.NAME, "candidate_route_class": v3.CLASSES["candidate"]}


def _v3_receipt_manifest(root: Path) -> str:
    expected = {v1.cell_path(root, arm, track, seed)
                for track, seed in MECHANISM_CELLS for arm in v1.ARMS}
    observed = set((root / "cells").rglob("*.json"))
    if observed != expected:
        raise ValueError("prior v3 triage receipts include unexpected or missing cell files")
    entries = {}
    for track, seed in MECHANISM_CELLS:
        for arm in v1.ARMS:
            path = v1.cell_path(root, arm, track, seed)
            entries[str(path.relative_to(root)).replace("\\", "/")] = fresh.digest(path)
    return hashlib.sha256(fresh._canonical(entries)).hexdigest()


def validate_prior_development(protocol: dict) -> dict:
    pinned = protocol.get("prior_development_v3")
    required = {"protocol_sha256", "result_sha256", "triage_summary_sha256",
                "triage_receipts_sha256", "status"}
    if (not isinstance(pinned, dict) or set(pinned) != required
            or pinned["status"] != "TRIAGE_BLOCK_FULL"):
        raise ValueError("prior v3 development proof must pin the blocked triage and receipts")
    for key in required - {"status"}:
        fresh._sha256_string(pinned[key], f"prior_development_v3.{key}")
    for key, path_key in (("protocol_sha256", "protocol"), ("result_sha256", "result"),
                          ("triage_summary_sha256", "triage_summary")):
        path = PRIOR_V3_PATHS[path_key]
        if not path.is_file() or fresh.digest(path) != pinned[key]:
            raise ValueError(f"prior v3 {path_key} hash mismatch")
    if PRIOR_V3_PATHS["full_summary"].exists():
        raise ValueError("prior v3 full comparison was unexpectedly opened")
    freeze_path = PRIOR_V3_PATHS["freeze"]
    if not freeze_path.is_file():
        raise ValueError("prior v3 frozen identity is missing")
    v3_protocol = fresh._read_json(PRIOR_V3_PATHS["protocol"])
    result = fresh._read_json(PRIOR_V3_PATHS["result"])
    summary = fresh._read_json(PRIOR_V3_PATHS["triage_summary"])
    freeze = fresh._read_json(freeze_path)
    freeze_sha256 = fresh.digest(freeze_path)
    v1_comparator_sha256 = fresh.digest(Path(v1.__file__))
    v2_wrapper_sha256 = fresh.digest(Path(v2.__file__))
    v3_wrapper_sha256 = fresh.digest(Path(v3.__file__))
    if (v3_protocol.get("name") != v3.NAME
            or result.get("experiment") != v3.NAME
            or result.get("status") != pinned["status"]
            or result.get("protocol_sha256") != pinned["protocol_sha256"]
            or result.get("mechanism_summary_sha256") != pinned["triage_summary_sha256"]
            or result.get("freeze_sha256") != freeze_sha256
            or result.get("baseline_agent_sha256") != freeze.get("source_sha256", {}).get("baseline")
            or result.get("candidate_agent_sha256") != freeze.get("source_sha256", {}).get("candidate")
            or result.get("model_sha256") != freeze.get("model_sha256")
            or result.get("gates", {}).get("mechanism") != "BLOCK_FULL"
            or result.get("gates", {}).get("full_development") != "NOT_OPENED"
            or result.get("gates", {}).get("fresh_screen") != "NOT_OPENED"
            or result.get("gates", {}).get("confirmation") != "SEALED_UNRUN"
            or result.get("gates", {}).get("blind") != "SEALED_UNRUN"
            or result.get("gates", {}).get("sota_promotion") is not False
            or summary.get("decision") != "TRIAGE_BLOCK_FULL"
            or summary.get("protocol_sha256") != pinned["protocol_sha256"]
            or summary.get("triage_gate", {}).get("status") != "BLOCK_FULL"
            or freeze.get("protocol_sha256") != pinned["protocol_sha256"]
            or freeze.get("comparator_sha256") != v1_comparator_sha256
            or freeze.get("wrapper_v2_sha256") != v2_wrapper_sha256
            or freeze.get("wrapper_sha256") != v3_wrapper_sha256
            or freeze.get("protocol_name") != v3.NAME
            or freeze.get("candidate_route_class") != v3.CLASSES["candidate"]):
        raise ValueError("prior v3 result, blocked gate, or frozen source lineage mismatch")
    v2_proof = v3.validate_prior_development(v3_protocol)
    if freeze.get("prior_development_v2") != v2_proof:
        raise ValueError("prior v3 frozen v2 lineage differs from recomputed proof")
    margin = fresh._read_json(v1.MARGIN_SCREEN_PROTOCOL)
    ego = fresh._read_json(v1.EGO_SCREEN_PROTOCOL)
    v3.validate_protocol(v3_protocol, margin, ego,
                         fresh.digest(v1.MARGIN_SCREEN_PROTOCOL),
                         fresh.digest(v1.EGO_SCREEN_PROTOCOL),
                         prior.sealed_holdout_seeds(ROOT / "experiments"))
    source_snapshots = v3_protocol.get("source_snapshots")
    if not isinstance(source_snapshots, dict) or set(source_snapshots) != set(v1.ARMS):
        raise ValueError("prior v3 source snapshots are missing")
    v3_paths = {"protocol": PRIOR_V3_PATHS["protocol"],
                "baseline": ROOT / source_snapshots["baseline"],
                "candidate": ROOT / source_snapshots["candidate"],
                "model": ROOT / "model.pt",
                "margin_screen_protocol": v1.MARGIN_SCREEN_PROTOCOL,
                "ego_screen_protocol": v1.EGO_SCREEN_PROTOCOL}
    v3.check_frozen_inputs(freeze, v3_paths, {"margin": margin, "ego": ego})
    if fresh.digest(v1.EGO_SCREEN_PROTOCOL) != freeze.get("ego_screen_protocol_sha256"):
        raise ValueError("prior v3 ego screen protocol changed")
    receipt_root = freeze_path.parent
    if _v3_receipt_manifest(receipt_root) != pinned["triage_receipts_sha256"]:
        raise ValueError("prior v3 triage receipt manifest hash mismatch")
    for track, seed in MECHANISM_CELLS:
        for arm in v1.ARMS:
            row = v1.load_cell(receipt_root, freeze, arm, track, seed)
            if row is None or row.get("error"):
                raise ValueError(f"prior v3 triage receipt missing or failed: {arm}/{track}/{seed}")
    recomputed = v3.report(receipt_root, freeze, MECHANISM_CELLS,
                           mechanism_only=True, ego_cells=v1.ego_screen_cells(ego))
    if recomputed != summary:
        raise ValueError("prior v3 triage summary differs from frozen receipts")
    return {**pinned, "freeze_sha256": freeze_sha256,
            "v1_comparator_sha256": v1_comparator_sha256,
            "v2_wrapper_sha256": v2_wrapper_sha256,
            "v3_wrapper_sha256": v3_wrapper_sha256,
            "prior_development_v2": v2_proof}


def validate_protocol(protocol: dict, margin: dict, ego: dict, margin_sha256: str,
                      ego_sha256: str, sealed_seeds: set[int]) -> tuple[tuple[int, int], ...]:
    _require_v4_route(protocol)
    validate_prior_development(protocol)
    return v3.validate_protocol(_v3_protocol_view(protocol), margin, ego,
                                margin_sha256, ego_sha256, sealed_seeds)


def build_identity(protocol_path: Path, protocol: dict, paths: dict[str, Path],
                   margin: dict, ego: dict) -> dict:
    _require_v4_route(protocol)
    prior_proof = validate_prior_development(protocol)
    inherited = v3.build_identity(protocol_path, _v3_protocol_view(protocol),
                                  paths, margin, ego)
    return {**inherited, "wrapper_v3_sha256": inherited["wrapper_sha256"],
            "wrapper_sha256": fresh.digest(Path(__file__)),
            "protocol_name": NAME, "candidate_route_class": CLASSES["candidate"],
            "prior_development_v3": prior_proof}


def check_frozen_inputs(identity: dict, paths: dict[str, Path],
                        screens: dict[str, dict]) -> None:
    if (identity.get("wrapper_sha256") != fresh.digest(Path(__file__))
            or identity.get("protocol_name") != NAME
            or identity.get("candidate_route_class") != CLASSES["candidate"]
            or identity.get("wrapper_v3_sha256") != fresh.digest(Path(v3.__file__))):
        raise ValueError("v4 wrapper or route changed during development")
    v3.check_frozen_inputs(_v3_identity_view(identity), paths, screens)
    if validate_prior_development(fresh._read_json(paths["protocol"])) != identity.get("prior_development_v3"):
        raise ValueError("prior v3 development proof changed during v4 run")


def report(root: Path, identity: dict, cells: tuple[tuple[int, int], ...],
           *, mechanism_only: bool, ego_cells: tuple[tuple[int, int], ...]) -> dict:
    summary = v3.report(root, identity, cells, mechanism_only=mechanism_only,
                        ego_cells=ego_cells)
    return {**summary, "protocol_name": NAME,
            "candidate_route_class": CLASSES["candidate"],
            "wrapper_sha256": identity["wrapper_sha256"],
            "wrapper_v3_sha256": identity["wrapper_v3_sha256"]}


def require_triage_gate(root: Path, identity: dict,
                        ego_cells: tuple[tuple[int, int], ...]) -> dict:
    path = root / "mechanism-summary.json"
    if not path.is_file():
        raise ValueError("v4 mechanism triage summary missing; run --mechanism-only first")
    for track, seed in MECHANISM_CELLS:
        for arm in v1.ARMS:
            if v1.load_cell(root, identity, arm, track, seed) is None:
                raise ValueError("v4 mechanism triage incomplete: all 20 rows are required")
    current = report(root, identity, MECHANISM_CELLS,
                     mechanism_only=True, ego_cells=ego_cells)
    if fresh._read_json(path) != current:
        raise ValueError("v4 mechanism triage summary differs from its 20 frozen rows")
    if current["triage_gate"]["status"] != "ALLOW_FULL_CONSUMED_DEVELOPMENT":
        raise ValueError("v4 mechanism triage blocked full development; a new protocol is required")
    return current


def validate_run_scope(protocol: dict, cells: tuple[tuple[int, int], ...],
                       ego_cells: tuple[tuple[int, int], ...], mechanism_only: bool) -> None:
    _require_v4_route(protocol)
    v3.validate_run_scope(_v3_protocol_view(protocol), cells, ego_cells, mechanism_only)


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
        raise ValueError(f"existing v4 development summary differs: {summary_path}")
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
