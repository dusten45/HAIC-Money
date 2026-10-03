"""Compare a snapshot against frozen v6 and baseline on consumed cells only.

This is a development diagnostic, never fresh generalization or promotion
evidence. Only the completed v6 screen grid and the older ten-cell diagnostic
grid are eligible. Every requested arm runs as a separate cold official worker.
The v6 confirmation and blind reservations are inaccessible through this tool.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_bare_generalization as fresh
from tools import compare_temporal_reachability_generalization as comparator


ARTIFACTS = ROOT / ".haic-artifacts"
V6_PROTOCOL = ROOT / "experiments/temporal-reachability-generalization-v1.json"
OLD_RESULT = ROOT / "experiments/temporal-reachability-dev-v1-result.json"
V6_RUN = ARTIFACTS / "temporal-reachability-generalization-v1/run"
V6_CONTROL = ARTIFACTS / "temporal-reachability-generalization-v1/snapshots/control_agent.py"
V6_CANDIDATE = ARTIFACTS / "temporal-reachability-generalization-v1/snapshots/candidate_agent.py"
MODEL = ROOT / "model.pt"
V6_PROTOCOL_SHA = "b4f3bd8d72f507c0530103f95ec220a4c5c8b4284a13af31231d77e1d68d821d"
OLD_RESULT_SHA = "932f65b8924859bd79ae204146c651eb35db782d4cf512f388875e3c0591b2e9"
V6_FREEZE_SHA = "957842c72aadbe8141672af4b6242aadd6507fff1b4a2d9428171854fb320c61"
V6_SUMMARY_SHA = "6e3448ea7d78fdf06c6f15a6d002c8304d04bd427d830491b0023ebc77f96f20"
ARMS = ("baseline", "v6", "v7")
CLASSES = {"baseline": "_CompoundClearingBrakeCarryController",
           "v6": "_TemporalReachabilityController"}
CANDIDATE_CLASSES = frozenset({
    "_TemporalCorridorMarginController", "_TemporalFullSteerController",
    "_TemporalCombinedCorridorController", "_TemporalEncounterController",
})


def _require_sha(path: Path, expected: str) -> None:
    if fresh.digest(path) != expected:
        raise ValueError(f"historical input SHA256 changed: {path}")


def _historical_records() -> tuple[dict, dict, dict, dict]:
    for path, sha in ((V6_PROTOCOL, V6_PROTOCOL_SHA), (OLD_RESULT, OLD_RESULT_SHA),
                      (V6_RUN / "freeze.json", V6_FREEZE_SHA),
                      (V6_RUN / "screen-summary.json", V6_SUMMARY_SHA)):
        _require_sha(path, sha)
    protocol, old, freeze, summary = (
        fresh._read_json(path) for path in
        (V6_PROTOCOL, OLD_RESULT, V6_RUN / "freeze.json", V6_RUN / "screen-summary.json")
    )
    if (summary.get("decision") != "REJECT" or summary.get("canonical_cells") != 32
            or summary.get("missing_cells") != [] or freeze.get("protocol_sha256") != V6_PROTOCOL_SHA
            or summary.get("protocol_sha256") != V6_PROTOCOL_SHA):
        raise ValueError("historical v6 screen is not a complete rejected evaluation")
    if old.get("status") != "DIAGNOSTIC_PACE_GATE_FAIL" or len(old.get("cells", [])) != 10:
        raise ValueError("old consumed diagnostic record is incomplete")
    return protocol, old, freeze, summary


def screen_cells() -> set[tuple[int, int]]:
    protocol, _, _, _ = _historical_records()
    screen = protocol["partitions"]["screen"]
    return {(track, seed) for track in screen["track_ids"] for seed in screen["seeds"]}


def old_cells() -> set[tuple[int, int]]:
    _, old, _, _ = _historical_records()
    return {(row["track_id"], row["seed"]) for row in old["cells"]}


def allowed_cells() -> set[tuple[int, int]]:
    protocol, old, _, _ = _historical_records()
    screen = protocol["partitions"]["screen"]
    current = {(track, seed) for track in screen["track_ids"] for seed in screen["seeds"]}
    historical = {(row["track_id"], row["seed"]) for row in old["cells"]}
    sealed = (set(protocol["partitions"]["confirmation"]["seeds"])
              | set(protocol["partitions"]["blind"]["seeds"]))
    combined = current | historical
    if len(current) != 32 or len(historical) != 10 or len(combined) != 42:
        raise ValueError("consumed-cell allowlist has changed")
    if {seed for _, seed in combined} & sealed:
        raise ValueError("consumed-cell allowlist overlaps sealed geometry")
    return combined


def select_cells(raw_cells: list[str], all_consumed: bool) -> list[tuple[int, int]]:
    allowed = allowed_cells()
    if all_consumed and raw_cells:
        raise ValueError("choose --all-consumed or explicit --cell values")
    if all_consumed:
        return sorted(allowed)
    if not raw_cells:
        raise ValueError("at least one consumed --cell is required")
    cells = []
    for raw in raw_cells:
        if not re.fullmatch(r"[1-4]:[0-9]+", raw):
            raise ValueError(f"invalid cell {raw!r}; use TRACK:SEED")
        track, seed = (int(part) for part in raw.split(":"))
        if (track, seed) not in allowed:
            raise ValueError(f"cell {raw} is outside the consumed allowlist")
        cells.append((track, seed))
    if len(set(cells)) != len(cells):
        raise ValueError("duplicate cell")
    return sorted(cells)


def validate_worker_count(workers: int) -> int:
    if type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError("workers must be an integer from 1 to 3")
    return workers


def validate_candidate_path(path: Path) -> Path:
    path = path.resolve()
    if (path.suffix.lower() != ".py" or not path.is_relative_to(ARTIFACTS.resolve())
            or path in {V6_CONTROL.resolve(), V6_CANDIDATE.resolve()}):
        raise ValueError("candidate must be a distinct .py snapshot under .haic-artifacts")
    return path


def _require_receipts(cells: list[tuple[int, int]], freeze: dict) -> None:
    screen = screen_cells()
    for track, seed in cells:
        if (track, seed) not in screen:
            continue
        for arm in fresh.ARMS:
            row = fresh.load_cell(V6_RUN, freeze, "screen", arm, track, seed, 0)
            if row is None or row.get("error"):
                raise ValueError(f"completed v6 screen receipt missing: {track}/{seed}/{arm}")


def build_identity(candidate: Path, candidate_class: str,
                   cells: list[tuple[int, int]]) -> dict:
    protocol, _, freeze, _ = _historical_records()
    if candidate_class not in CANDIDATE_CLASSES:
        raise ValueError("candidate class must name one of the four v7 controllers")
    candidate = validate_candidate_path(candidate)
    base = fresh.build_identity(V6_PROTOCOL, protocol, V6_CONTROL, V6_CANDIDATE, MODEL)
    for key in ("protocol_sha256", "source_sha256", "model_sha256", "harness_sha256",
                "helper_sha256", "environment_sha256"):
        if base[key] != freeze[key]:
            raise ValueError(f"current {key} differs from frozen v6 evaluation")
    _require_receipts(cells, freeze)
    return {**base, "development_only": True, "v7_source_sha256": fresh.digest(candidate),
            "v7_controller_class": candidate_class, "tool_sha256": fresh.digest(Path(__file__)),
            "comparator_sha256": fresh.digest(Path(comparator.__file__)),
            "old_result_sha256": OLD_RESULT_SHA, "v6_freeze_sha256": V6_FREEZE_SHA,
            "v6_screen_summary_sha256": V6_SUMMARY_SHA,
            "cells": [[track, seed] for track, seed in cells]}


def check_inputs(identity: dict, paths: dict[str, Path]) -> None:
    fresh.check_frozen_inputs(identity, {"control": paths["baseline"],
                                        "candidate": paths["v6"],
                                        "model": paths["model"], "protocol": V6_PROTOCOL})
    for path, sha in ((paths["v7"], identity["v7_source_sha256"]),
                      (OLD_RESULT, identity["old_result_sha256"]),
                      (V6_RUN / "freeze.json", identity["v6_freeze_sha256"]),
                      (V6_RUN / "screen-summary.json", identity["v6_screen_summary_sha256"]),
                      (Path(__file__), identity["tool_sha256"]),
                      (Path(comparator.__file__), identity["comparator_sha256"])):
        _require_sha(path, sha)


def cell_path(root: Path, arm: str, track: int, seed: int) -> Path:
    if arm not in ARMS:
        raise ValueError("invalid development arm")
    return root / "cells" / arm / f"track-{track}-seed-{seed}.json"


def record_cell(root: Path, identity: dict, row: dict) -> bool:
    path = cell_path(root, row["arm"], row["track_id"], row["seed"])
    if path.exists():
        if load_cell(root, identity, row["arm"], row["track_id"], row["seed"]) != row:
            raise ValueError(f"existing development cell differs: {path}")
        return False
    payload = {"identity": identity, "row": row}
    fresh._atomic_json(path, {**payload, "digest": hashlib.sha256(fresh._canonical(payload)).hexdigest()})
    return True


def load_cell(root: Path, identity: dict, arm: str, track: int, seed: int) -> dict | None:
    path = cell_path(root, arm, track, seed)
    if not path.exists():
        return None
    envelope = fresh._read_json(path)
    row = envelope.get("row")
    if envelope.get("identity") != identity or not isinstance(row, dict):
        raise ValueError(f"development cell identity mismatch: {path}")
    if any(row.get(key) != value for key, value in (("arm", arm), ("track_id", track), ("seed", seed))):
        raise ValueError(f"development cell coordinates mismatch: {path}")
    payload = {"identity": identity, "row": row}
    if envelope.get("digest") != hashlib.sha256(fresh._canonical(payload)).hexdigest():
        raise ValueError(f"development cell digest mismatch: {path}")
    return row


def report_cells(root: Path, identity: dict, cells: list[tuple[int, int]]) -> dict:
    comparisons = {}
    for reference in ("baseline", "v6"):
        rows = []
        for track, seed in cells:
            for arm in (reference, "v7"):
                stored = load_cell(root, identity, arm, track, seed)
                if stored is not None:
                    rows.append({**stored, "arm": "control" if arm == reference else "candidate",
                                 "repeat": 0})
        comparisons[f"versus_{reference}"] = comparator.compare_pairs(
            rows, [(track, seed, 0) for track, seed in cells])
    return {"evidence": "REUSED_DIAGNOSTIC_ONLY", "cells": [[t, s] for t, s in cells],
            "identity_sha256": hashlib.sha256(fresh._canonical(identity)).hexdigest(),
            **comparisons}


def _run_cold_episode(identity: dict, paths: dict[str, Path], classes: dict[str, str],
                      arm: str, track: int, seed: int) -> dict:
    check_inputs(identity, paths)
    payload = {"source": str(paths[arm].resolve()), "model": str(paths["model"].resolve()),
               "controller_class": classes[arm], "track_id": track, "seed": seed}
    command = [sys.executable, str(Path(fresh.__file__).resolve()),
               "--worker", json.dumps(payload, separators=(",", ":"))]
    try:
        try:
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                       timeout=300, check=True)
            measured = json.loads(completed.stdout)
            if not isinstance(measured, dict):
                raise ValueError("worker returned a non-object result")
            return measured
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, ValueError) as error:
            stderr = getattr(error, "stderr", "") or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return {"error": str(error), "stderr": stderr[-4000:]}
    finally:
        check_inputs(identity, paths)


def run_cells(root: Path, identity: dict, paths: dict[str, Path], classes: dict[str, str],
              cells: list[tuple[int, int]], workers: int = 1) -> dict:
    # Recheck the complete allowlist before any subprocess, including direct API calls.
    if not cells or len(set(cells)) != len(cells) or not set(cells) <= allowed_cells():
        raise ValueError("requested cell is outside the consumed allowlist")
    validate_worker_count(workers)
    _, _, freeze, _ = _historical_records()
    _require_receipts(cells, freeze)
    if identity.get("cells") != [[t, s] for t, s in cells]:
        raise ValueError("selected cells differ from frozen run identity")
    check_inputs(identity, paths)
    fresh.prepare_run(root, identity)
    with fresh._run_lock(root):
        jobs = []
        for track, seed in cells:
            for arm in ARMS:
                previous = load_cell(root, identity, arm, track, seed)
                if previous is not None:
                    if previous.get("error"):
                        raise RuntimeError(f"recorded worker failure: {arm}/{track}/{seed}")
                    continue
                jobs.append((arm, track, seed))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            next_job = 0

            def fill_window() -> None:
                nonlocal next_job
                while len(pending) < workers and next_job < len(jobs):
                    arm, track, seed = jobs[next_job]
                    pending[next_job] = pool.submit(_run_cold_episode, identity, paths,
                                                    classes, arm, track, seed)
                    next_job += 1

            fill_window()
            for index, (arm, track, seed) in enumerate(jobs):
                measured = pending.pop(index).result()
                row = {"arm": arm, "track_id": track, "seed": seed, **measured}
                record_cell(root, identity, row)
                if row.get("error"):
                    raise RuntimeError(f"worker failed at {arm}/{track}/{seed}: {row['error']}")
                print(json.dumps({"cell": [arm, track, seed], "finished": row["finished"],
                                  "progress": row["progress"],
                                  "contacts": row["collision_count"]}), flush=True)
                fill_window()
    summary = report_cells(root, identity, cells)
    path = root / "summary.json"
    if path.exists() and fresh._read_json(path) != summary:
        raise ValueError("existing summary differs from frozen receipts")
    if not path.exists():
        fresh._atomic_json(path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-agent", required=True, type=Path,
                        help="immutable candidate Agent source snapshot under .haic-artifacts")
    parser.add_argument("--candidate-class", required=True,
                        help="actual controller class selected by candidate Agent")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cell", action="append", default=[], metavar="TRACK:SEED")
    selection.add_argument("--all-consumed", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cells = select_cells(args.cell, args.all_consumed)
    validate_worker_count(args.workers)
    candidate = validate_candidate_path(args.candidate_agent)
    paths = {"baseline": V6_CONTROL, "v6": V6_CANDIDATE, "v7": candidate, "model": MODEL}
    classes = {**CLASSES, "v7": args.candidate_class}
    identity = build_identity(candidate, args.candidate_class, cells)
    check_inputs(identity, paths)
    if args.preflight_only:
        for arm in ARMS:
            fresh._load_agent(paths[arm], MODEL, classes[arm])
        print(json.dumps({"preflight": "PASS", "cells": identity["cells"],
                          "source_sha256": {"baseline": identity["source_sha256"]["control"],
                                            "v6": identity["source_sha256"]["candidate"],
                                            "v7": identity["v7_source_sha256"]}}, indent=2))
        return 0
    default_name = hashlib.sha256(fresh._canonical(identity["cells"])).hexdigest()[:12]
    root = (args.output_root or ARTIFACTS / "consumed-candidate-comparisons" /
            f"{identity['v7_source_sha256'][:12]}-{default_name}").resolve()
    if not root.is_relative_to((ARTIFACTS / "consumed-candidate-comparisons").resolve()):
        raise ValueError("development output root must stay under .haic-artifacts/consumed-candidate-comparisons")
    summary = run_cells(root, identity, paths, classes, cells, workers=args.workers)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
