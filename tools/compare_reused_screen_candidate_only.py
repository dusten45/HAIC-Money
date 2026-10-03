"""Compare a candidate on the consumed v6 screen without rerunning references.

This is development evidence only. The 32 screen cells are already consumed;
confirmation, blind, and older development cells are never eligible here.
Original baseline/v6 receipts remain in their frozen run and are loaded and
validated in place. Only the new candidate uses a cold episode worker.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import compare_consumed_candidate as dev
from tools import compare_temporal_reachability_generalization as comparator
from tools import evaluate_bare_generalization as fresh


ARTIFACTS = dev.ARTIFACTS
V6_RUN = dev.V6_RUN
OUTPUT_PARENT = ARTIFACTS / "reused-screen-candidate-only"
OUTCOME_KEYS = ("finished", "progress", "lap_time_ms", "collision_count",
                "damage", "retire_reason")


def select_cells(raw_cells: list[str], all_screen: bool) -> list[tuple[int, int]]:
    if all_screen and raw_cells:
        raise ValueError("choose --all-screen or explicit screen cells")
    screen = dev.screen_cells()
    if all_screen:
        return sorted(screen)
    if not raw_cells:
        raise ValueError("at least one consumed screen cell is required")
    try:
        cells = dev.select_cells(raw_cells, False)
    except ValueError as error:
        raise ValueError("requested cell is outside the consumed v6 screen") from error
    if not set(cells) <= screen:
        raise ValueError("candidate-only comparator accepts consumed v6 screen cells only")
    return cells


def _require_screen_cells(cells: list[tuple[int, int]]) -> None:
    if not cells or len(cells) != len(set(cells)) or not set(cells) <= dev.screen_cells():
        raise ValueError("requested cells must be unique consumed v6 screen cells")


def _receipt_path(arm: str, track: int, seed: int) -> Path:
    return fresh.cell_path(V6_RUN, "screen", arm, track, seed, 0)


def load_original_pair(track: int, seed: int, freeze: dict,
                       screen_summary: dict) -> dict[str, dict]:
    if (track, seed) not in dev.screen_cells():
        raise ValueError("original pair is outside the consumed v6 screen")
    frozen_pair = next((item for item in screen_summary["paired_cells"]
                        if item["track_id"] == track and item["seed"] == seed), None)
    if frozen_pair is None or frozen_pair.get("status") != "COMPLETE":
        raise ValueError("frozen screen summary lacks the original pair")
    rows = {}
    for old_arm, name in (("control", "baseline"), ("candidate", "v6")):
        row = fresh.load_cell(V6_RUN, freeze, "screen", old_arm, track, seed, 0)
        if row is None or row.get("error"):
            raise ValueError(f"original screen receipt missing or failed: {old_arm}/{track}/{seed}")
        projected = {key: row.get(key) for key in OUTCOME_KEYS}
        if projected != frozen_pair[old_arm]:
            raise ValueError(f"original {old_arm} receipt differs from pinned screen summary")
        rows[name] = row
    return rows


def build_identity(candidate: Path, candidate_class: str,
                   cells: list[tuple[int, int]]) -> dict:
    _require_screen_cells(cells)
    identity = dev.build_identity(candidate, candidate_class, cells)
    _, _, freeze, summary = dev._historical_records()
    receipts = {}
    for track, seed in cells:
        load_original_pair(track, seed, freeze, summary)
        for arm in ("control", "candidate"):
            receipts[f"{track}:{seed}:{arm}"] = fresh.digest(_receipt_path(arm, track, seed))
    return {**identity, "candidate_only_tool_sha256": fresh.digest(Path(__file__)),
            "original_receipt_sha256": receipts}


def check_inputs(identity: dict, paths: dict[str, Path],
                 freeze: dict, screen_summary: dict) -> None:
    dev.check_inputs(identity, paths)
    if fresh.digest(Path(__file__)) != identity.get("candidate_only_tool_sha256"):
        raise ValueError("candidate-only comparator changed during the run")
    cells = [tuple(cell) for cell in identity["cells"]]
    _require_screen_cells(cells)
    expected = identity.get("original_receipt_sha256")
    if not isinstance(expected, dict) or len(expected) != 2 * len(cells):
        raise ValueError("original screen receipt manifest is incomplete")
    for track, seed in cells:
        load_original_pair(track, seed, freeze, screen_summary)
        for arm in ("control", "candidate"):
            key = f"{track}:{seed}:{arm}"
            if fresh.digest(_receipt_path(arm, track, seed)) != expected.get(key):
                raise ValueError(f"original screen receipt hash changed: {key}")


def report_cells(root: Path, identity: dict, cells: list[tuple[int, int]],
                 freeze: dict, screen_summary: dict) -> dict:
    _require_screen_cells(cells)
    comparisons = {}
    for reference in ("baseline", "v6"):
        rows = []
        for track, seed in cells:
            original = load_original_pair(track, seed, freeze, screen_summary)[reference]
            candidate = dev.load_cell(root, identity, "v7", track, seed)
            rows.append({**original, "arm": "control", "repeat": 0})
            if candidate is not None:
                rows.append({**candidate, "arm": "candidate", "repeat": 0})
        comparisons[f"versus_{reference}"] = comparator.compare_pairs(
            rows, [(track, seed, 0) for track, seed in cells])
    return {"evidence": "REUSED_DIAGNOSTIC_ONLY", "cells": [[t, s] for t, s in cells],
            "identity_sha256": hashlib.sha256(fresh._canonical(identity)).hexdigest(),
            **comparisons}


def _run_cold_episode(identity: dict, paths: dict[str, Path], classes: dict[str, str],
                      arm: str, track: int, seed: int) -> dict:
    if arm != "v7":
        raise ValueError("only candidate cold workers are permitted")
    return dev._run_cold_episode(identity, paths, classes, arm, track, seed)


def run_cells(root: Path, identity: dict, paths: dict[str, Path], classes: dict[str, str],
              cells: list[tuple[int, int]], workers: int = 1) -> dict:
    _require_screen_cells(cells)
    dev.validate_worker_count(workers)
    if root.resolve() == OUTPUT_PARENT.resolve() or not root.resolve().is_relative_to(OUTPUT_PARENT.resolve()):
        raise ValueError("output root must be inside candidate-only development artifacts")
    if identity.get("cells") != [[track, seed] for track, seed in cells]:
        raise ValueError("requested screen cells differ from frozen run identity")
    if classes != {**dev.CLASSES, "v7": identity.get("v7_controller_class")}:
        raise ValueError("candidate class differs from frozen run identity")
    _, _, freeze, summary = dev._historical_records()
    check_inputs(identity, paths, freeze, summary)
    fresh.prepare_run(root, identity)
    with fresh._run_lock(root):
        jobs = []
        for track, seed in cells:
            prior = dev.load_cell(root, identity, "v7", track, seed)
            if prior is not None:
                if prior.get("error"):
                    raise RuntimeError(f"recorded candidate worker failure: {track}/{seed}")
                continue
            jobs.append((track, seed))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            next_job = 0

            def fill_window() -> None:
                nonlocal next_job
                while len(pending) < workers and next_job < len(jobs):
                    track, seed = jobs[next_job]
                    pending[next_job] = pool.submit(_run_cold_episode, identity, paths,
                                                    classes, "v7", track, seed)
                    next_job += 1

            fill_window()
            for index, (track, seed) in enumerate(jobs):
                measured = pending.pop(index).result()
                check_inputs(identity, paths, freeze, summary)
                row = {"arm": "v7", "track_id": track, "seed": seed, **measured}
                dev.record_cell(root, identity, row)
                if row.get("error"):
                    raise RuntimeError(f"candidate worker failed at {track}/{seed}: {row['error']}")
                print(json.dumps({"cell": ["v7", track, seed], "finished": row["finished"],
                                  "progress": row["progress"],
                                  "contacts": row["collision_count"]}), flush=True)
                fill_window()
    check_inputs(identity, paths, freeze, summary)
    result = report_cells(root, identity, cells, freeze, summary)
    target = root / "summary.json"
    if target.exists() and fresh._read_json(target) != result:
        raise ValueError("existing candidate-only summary differs from receipts")
    if not target.exists():
        fresh._atomic_json(target, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-agent", required=True, type=Path)
    parser.add_argument("--candidate-class", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cell", action="append", default=[], metavar="TRACK:SEED")
    selection.add_argument("--all-screen", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cells = select_cells(args.cell, args.all_screen)
    dev.validate_worker_count(args.workers)
    candidate = dev.validate_candidate_path(args.candidate_agent)
    paths = {"baseline": dev.V6_CONTROL, "v6": dev.V6_CANDIDATE,
             "v7": candidate, "model": dev.MODEL}
    classes = {**dev.CLASSES, "v7": args.candidate_class}
    identity = build_identity(candidate, args.candidate_class, cells)
    _, _, freeze, summary = dev._historical_records()
    check_inputs(identity, paths, freeze, summary)
    if args.preflight_only:
        fresh._load_agent(candidate, dev.MODEL, args.candidate_class)
        print(json.dumps({"preflight": "PASS", "cells": identity["cells"],
                          "candidate_sha256": identity["v7_source_sha256"],
                          "reused_original_receipts": len(identity["original_receipt_sha256"])}))
        return 0
    cell_key = hashlib.sha256(fresh._canonical(identity["cells"])).hexdigest()[:12]
    root = (args.output_root or OUTPUT_PARENT /
            f"{identity['v7_source_sha256'][:12]}-{cell_key}").resolve()
    result = run_cells(root, identity, paths, classes, cells, workers=args.workers)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
