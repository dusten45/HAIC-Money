"""Frozen, fresh paired comparison for the temporal reachability controller.

The protocol is a draft until the candidate route and committed source hashes
are bound. Reused feasible-corridor diagnostic cells are excluded. Each official
episode runs in a cold process through evaluate_bare_generalization's worker.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_bare_generalization as fresh


NAME = "temporal-reachability-generalization-v1"
PROTOCOL_PATH = ROOT / "experiments" / f"{NAME}.json"
OUTPUT_ROOT = ROOT / ".haic-artifacts" / NAME / "run"
SOURCE_PATHS = {
    "control": f".haic-artifacts/{NAME}/snapshots/control_agent.py",
    "candidate": f".haic-artifacts/{NAME}/snapshots/candidate_agent.py",
}
CONTROL_CLASS = "_CompoundClearingBrakeCarryController"
TRACKS = [1, 2, 3, 4]
SEED_COUNTS = {"screen": 8, "confirmation": 16, "blind": 8}
SPOT_INDEX = {"screen": ((1, 0), (4, 7)),
              "confirmation": ((1, 0), (2, 1), (3, 2), (4, 3)),
              "blind": ((1, 0), (4, 7))}
THRESHOLDS = {
    "maximum_both_dnf_progress_loss": .05,
    "maximum_aggregate_shared_finish_time_ratio": 1.05,
    "maximum_single_shared_finish_time_ratio": 1.10,
    "maximum_aggregate_contact_increase": 0,
    "maximum_aggregate_damage_increase": 0.0,
}
RUNTIME_LIMITS = {"initialization_ms": 10000.0, "reset_ms": 5000.0,
                  "action_latency_max_ms": 5000.0, "peak_worker_rss_mib": 1024.0}


def validate_protocol_path(path: Path) -> Path:
    path = path.resolve()
    if path != PROTOCOL_PATH.resolve():
        raise ValueError("v6 protocol path must be the registered fresh protocol")
    return path


def validate_output_root(path: Path) -> Path:
    path = path.resolve()
    if path != OUTPUT_ROOT.resolve():
        raise ValueError("v6 output root must be the registered fresh run directory")
    return path


def _derived_seeds(phase: str) -> list[int]:
    return [int.from_bytes(hashlib.sha256(f"{NAME}:{phase}:{index}".encode("utf-8")).digest()[:4], "big")
            for index in range(SEED_COUNTS[phase])]


def validate_protocol(protocol: dict, historical_seeds: set[int]) -> None:
    if not isinstance(protocol, dict) or protocol.get("name") != NAME:
        raise ValueError("v6 exact protocol name is required")
    if protocol.get("status") != "PREREGISTERED":
        raise ValueError("v6 protocol must be PREREGISTERED before a fresh episode")
    fresh.validate_protocol(protocol, historical_seeds)
    classes = protocol["controller_classes"]
    candidate_class = classes["candidate"]
    if classes != {"control": CONTROL_CLASS, "candidate": candidate_class} or candidate_class.startswith("PENDING_"):
        raise ValueError("v6 actual Agent route classes must be bound")
    if protocol.get("decision_thresholds") != THRESHOLDS:
        raise ValueError("v6 decision thresholds changed")
    if protocol.get("source_snapshots") != SOURCE_PATHS:
        raise ValueError("v6 source snapshots changed")
    construction = protocol.get("source_construction")
    if not isinstance(construction, dict):
        raise ValueError("v6 source construction is required")
    commit = construction.get("implementation_commit")
    if (not isinstance(commit, str) or not 7 <= len(commit) <= 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError("v6 implementation commit must be a hexadecimal Git commit")
    fresh._sha256_string(construction.get("agent_blob_sha256"), "agent_blob_sha256")
    if (construction.get("agent_blob_sha256") != protocol["control_agent_sha256"]
            or construction.get("selector_from") != f"{CONTROL_CLASS}()"
            or construction.get("selector_to") != f"{candidate_class}()"):
        raise ValueError("v6 source selector construction is inconsistent")
    if protocol.get("training") is not False:
        raise ValueError("v6 candidate must be evaluated as a fixed bare Agent")
    for phase in fresh.PHASES:
        partition = protocol["partitions"][phase]
        seeds = _derived_seeds(phase)
        if partition["track_ids"] != TRACKS:
            raise ValueError(f"v6 {phase} track IDs changed")
        if partition["seeds"] != seeds:
            raise ValueError(f"v6 {phase} seeds differ from derived fresh geometry")
        expected_spots = [[track, seeds[index]] for track, index in SPOT_INDEX[phase]]
        if partition.get("spot_check_cells") != expected_spots:
            raise ValueError(f"v6 {phase} deterministic spot checks changed")


def validate_source_pair(protocol: dict, control: Path, candidate: Path) -> None:
    original = control.read_bytes()
    selected = candidate.read_bytes()
    construction = protocol["source_construction"]
    old = construction["selector_from"].encode("ascii")
    new = construction["selector_to"].encode("ascii")
    if original.count(old) != 1 or selected != original.replace(old, new, 1):
        raise ValueError("candidate source must differ solely in the Agent route selector")
    if (fresh.digest(control) != protocol["control_agent_sha256"]
            or fresh.digest(candidate) != protocol["candidate_agent_sha256"]):
        raise ValueError("source snapshot SHA256 mismatch")


def verify_committed_source(protocol: dict, control: Path) -> None:
    commit = protocol["source_construction"]["implementation_commit"]
    completed = subprocess.run(["git", "show", f"{commit}:agent.py"], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if completed.stdout != control.read_bytes():
        raise ValueError("control snapshot differs from the committed agent.py Git blob")
    if hashlib.sha256(completed.stdout).hexdigest() != protocol["source_construction"]["agent_blob_sha256"]:
        raise ValueError("committed agent.py blob SHA256 mismatch")


def _runtime_errors(row: dict, track: int, seed: int, arm: str) -> list[str]:
    errors = []
    for field, limit in RUNTIME_LIMITS.items():
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > limit:
            errors.append(f"{track}/{seed}/{arm}: runtime {field} invalid or exceeded")
    return errors


def compare_pairs(rows: list[dict], cells: list[tuple[int, int, int]]) -> dict:
    """Apply the registered finish/progress/pace hierarchy and safety gates."""
    lookup = {(row["track_id"], row["seed"], row["repeat"], row["arm"]): row for row in rows}
    canonical = [(track, seed) for track, seed, repeat in cells if repeat == 0]
    reasons: list[str] = []
    missing: list[list[int]] = []
    paired: list[dict] = []
    totals = {arm: {"finishes": 0, "progress": 0.0, "contacts": 0, "damage": 0.0,
                    "offtrack": 0, "partial_offtrack": 0} for arm in fresh.ARMS}
    shared_time = {arm: 0.0 for arm in fresh.ARMS}
    shared_count = 0
    for track, seed in canonical:
        control = lookup.get((track, seed, 0, "control"))
        candidate = lookup.get((track, seed, 0, "candidate"))
        if control is None or candidate is None:
            missing.append([track, seed])
            paired.append({"track_id": track, "seed": seed, "status": "MISSING_PAIR"})
            continue
        if control.get("error") or candidate.get("error"):
            reasons.append(f"{track}/{seed}: operational failure")
            paired.append({"track_id": track, "seed": seed, "status": "OPERATIONAL_FAILURE",
                           "control_error": control.get("error"), "candidate_error": candidate.get("error")})
            continue
        for arm, row in (("control", control), ("candidate", candidate)):
            reasons.extend(_runtime_errors(row, track, seed, arm))
            totals[arm]["finishes"] += int(row["finished"])
            totals[arm]["progress"] += float(row["progress"])
            totals[arm]["contacts"] += int(row["collision_count"])
            totals[arm]["damage"] += float(row["damage"])
            totals[arm]["offtrack"] += int(row.get("offtrack_samples", 0))
            totals[arm]["partial_offtrack"] += int(row.get("partial_offtrack_samples", 0))
        flags = []
        if control["finished"] and not candidate["finished"]:
            flags.append("control finish lost")
        if candidate.get("retire_reason") == "crash" and control.get("retire_reason") != "crash":
            flags.append("new crash")
        if (not control["finished"] and not candidate["finished"]
                and float(control["progress"]) - float(candidate["progress"]) >
                THRESHOLDS["maximum_both_dnf_progress_loss"] + 1e-9):
            flags.append("severe both-DNF progress loss")
        if control["finished"] and candidate["finished"]:
            shared_count += 1
            for arm, row in (("control", control), ("candidate", candidate)):
                shared_time[arm] += float(row["lap_time_ms"])
            if (float(candidate["lap_time_ms"]) >
                    THRESHOLDS["maximum_single_shared_finish_time_ratio"] * float(control["lap_time_ms"]) + 1e-9):
                flags.append("single shared finish time regression")
        reasons.extend(f"{track}/{seed}: {flag}" for flag in flags)
        paired.append({"track_id": track, "seed": seed, "status": "COMPLETE",
                       "control": {key: control.get(key) for key in
                                   ("finished", "progress", "lap_time_ms", "collision_count", "damage", "retire_reason")},
                       "candidate": {key: candidate.get(key) for key in
                                     ("finished", "progress", "lap_time_ms", "collision_count", "damage", "retire_reason")},
                       "flags": flags})
    if not missing and len(paired) == len(canonical):
        if totals["candidate"]["contacts"] > totals["control"]["contacts"] + THRESHOLDS["maximum_aggregate_contact_increase"]:
            reasons.append("aggregate contacts increased")
        if totals["candidate"]["damage"] > totals["control"]["damage"] + THRESHOLDS["maximum_aggregate_damage_increase"] + 1e-9:
            reasons.append("aggregate damage increased")
        if shared_count and shared_time["candidate"] > THRESHOLDS["maximum_aggregate_shared_finish_time_ratio"] * shared_time["control"] + 1e-9:
            reasons.append("aggregate shared finish time regression")
    denominator = len(canonical)
    mean_delta = (totals["candidate"]["progress"] - totals["control"]["progress"]) / denominator
    if reasons:
        decision = "REJECT"
    elif missing:
        decision = "INCOMPLETE"
    elif totals["candidate"]["finishes"] > totals["control"]["finishes"]:
        decision = "RETAIN"
    elif (totals["candidate"]["finishes"] == totals["control"]["finishes"]
          and (mean_delta > 1e-9 or
               (abs(mean_delta) <= 1e-9 and shared_count and shared_time["candidate"] < shared_time["control"] - 1e-9))):
        decision = "RETAIN"
    else:
        decision = "INCONCLUSIVE"
    return {"decision": decision, "reasons": reasons, "missing_cells": missing,
            "canonical_cells": denominator, "paired_cells": paired,
            "control_finishes": totals["control"]["finishes"],
            "candidate_finishes": totals["candidate"]["finishes"],
            "control_mean_progress": totals["control"]["progress"] / denominator,
            "candidate_mean_progress": totals["candidate"]["progress"] / denominator,
            "control_contacts": totals["control"]["contacts"],
            "candidate_contacts": totals["candidate"]["contacts"],
            "control_damage": totals["control"]["damage"],
            "candidate_damage": totals["candidate"]["damage"],
            "control_offtrack_samples": totals["control"]["offtrack"],
            "candidate_offtrack_samples": totals["candidate"]["offtrack"],
            "control_partial_offtrack_samples": totals["control"]["partial_offtrack"],
            "candidate_partial_offtrack_samples": totals["candidate"]["partial_offtrack"],
            "shared_completed_cells": shared_count,
            "control_shared_completed_time_ms": shared_time["control"],
            "candidate_shared_completed_time_ms": shared_time["candidate"]}


def report_phase(root: Path, identity: dict, protocol: dict, phase: str) -> dict:
    cells = fresh.expected_cells(protocol, phase)
    rows = []
    for track, seed, repeat in cells:
        for arm in fresh.ARMS:
            row = fresh.load_cell(root, identity, phase, arm, track, seed, repeat)
            if row is not None:
                rows.append(row)
    summary = compare_pairs(rows, cells)
    spot_failures = fresh._verify_spot_checks(rows, protocol, phase)
    if spot_failures:
        summary["reasons"].extend(spot_failures)
        summary["decision"] = "INCOMPLETE" if any("missing" in item for item in spot_failures) else "REJECT"
    summary.update({"partition": phase, "protocol_sha256": identity["protocol_sha256"],
                    "candidate_agent_sha256": identity["source_sha256"]["candidate"],
                    "wrapper_sha256": identity["wrapper_sha256"]})
    return summary


def _require_seal(root: Path, identity: dict, protocol: dict,
                  name: str, predecessor: str) -> None:
    seal = fresh._read_seal(root, identity, name)
    if seal is None:
        raise ValueError(f"{name} seal is missing")
    summary = report_phase(root, identity, protocol, predecessor)
    if hashlib.sha256(fresh._canonical(summary)).hexdigest() != seal["summary_sha256"]:
        raise ValueError(f"{name} seal no longer matches {predecessor} receipts")


def require_phase(root: Path, identity: dict, protocol: dict, phase: str) -> None:
    if phase == "screen":
        return
    if phase == "confirmation":
        _require_seal(root, identity, protocol, "finalist", "screen")
        return
    if phase == "blind":
        require_phase(root, identity, protocol, "confirmation")
        _require_seal(root, identity, protocol, "confirmation-accepted", "confirmation")
        return
    raise ValueError("unknown partition")


def freeze_finalist(root: Path, identity: dict, protocol: dict) -> None:
    summary = report_phase(root, identity, protocol, "screen")
    if summary["decision"] != "RETAIN":
        raise ValueError("screen must RETAIN before finalist freeze")
    fresh._write_seal(root, identity, "finalist", summary)


def seal_confirmation(root: Path, identity: dict, protocol: dict) -> None:
    require_phase(root, identity, protocol, "confirmation")
    summary = report_phase(root, identity, protocol, "confirmation")
    if summary["decision"] != "RETAIN":
        raise ValueError("confirmation must RETAIN before blind")
    fresh._write_seal(root, identity, "confirmation-accepted", summary)


def build_identity(path: Path, protocol: dict, paths: dict[str, Path]) -> dict:
    identity = fresh.build_identity(path, protocol, paths["control"],
                                    paths["candidate"], paths["model"])
    return {**identity, "wrapper_sha256": fresh.digest(Path(__file__)),
            "protocol_name": NAME, "candidate_route_class": protocol["controller_classes"]["candidate"]}


def check_frozen_inputs(identity: dict, protocol: dict, paths: dict[str, Path]) -> None:
    fresh.check_frozen_inputs(identity, paths)
    if (identity["wrapper_sha256"] != fresh.digest(Path(__file__))
            or identity["protocol_name"] != NAME
            or identity["candidate_route_class"] != protocol["controller_classes"]["candidate"]):
        raise ValueError("v6 wrapper or actual route changed")
    validate_source_pair(protocol, paths["control"], paths["candidate"])


def validate_worker_count(workers: int) -> int:
    if type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError("workers must be an integer from 1 to 3")
    return workers


def _run_cold_episode(identity: dict, protocol: dict, paths: dict[str, Path],
                      arm: str, track: int, seed: int) -> dict:
    """Run one independent process, checking frozen inputs on both sides."""
    check_frozen_inputs(identity, protocol, paths)
    payload = {"source": str(paths[arm].resolve()), "model": str(paths["model"].resolve()),
               "controller_class": protocol["controller_classes"][arm],
               "track_id": track, "seed": seed}
    command = [sys.executable, str(Path(fresh.__file__).resolve()),
               "--worker", json.dumps(payload, separators=(",", ":"))]
    try:
        try:
            completed = subprocess.run(command, cwd=ROOT, text=True,
                                       capture_output=True, timeout=300, check=True)
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
        check_frozen_inputs(identity, protocol, paths)


def run_partition(root: Path, identity: dict, protocol: dict,
                  paths: dict[str, Path], phase: str, workers: int = 1) -> dict:
    workers = validate_worker_count(workers)
    require_phase(root, identity, protocol, phase)
    with fresh._run_lock(root):
        jobs: list[tuple[str, int, int, int]] = []
        for track, seed, repeat in fresh.expected_cells(protocol, phase):
            for arm in fresh.ARMS:
                previous = fresh.load_cell(root, identity, phase, arm, track, seed, repeat)
                if previous is not None:
                    if previous.get("error"):
                        raise RuntimeError(f"recorded operational failure: {phase}/{arm}/{track}/{seed}/{repeat}")
                    continue
                jobs.append((arm, track, seed, repeat))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            next_job = 0

            def fill_window() -> None:
                nonlocal next_job
                while len(pending) < workers and next_job < len(jobs):
                    arm, track, seed, _repeat = jobs[next_job]
                    pending[next_job] = pool.submit(_run_cold_episode, identity, protocol,
                                                    paths, arm, track, seed)
                    next_job += 1

            fill_window()
            for index, (arm, track, seed, repeat) in enumerate(jobs):
                measured = pending.pop(index).result()
                row = {"partition": phase, "arm": arm, "track_id": track,
                       "seed": seed, "repeat": repeat}
                row.update(measured)
                if row.get("error"):
                    fresh.record_cell(root, identity, row)
                    failure = report_phase(root, identity, protocol, phase)
                    fresh._atomic_json(root / f"{phase}-summary.json", failure)
                    raise RuntimeError(f"worker failed at {phase}/{arm}/{track}/{seed}/{repeat}")
                fresh.record_cell(root, identity, row)
                print(json.dumps({"cell": [phase, arm, track, seed, repeat],
                                  "finished": row["finished"], "progress": row["progress"],
                                  "contacts": row["collision_count"]}), flush=True)
                fill_window()
    summary = report_phase(root, identity, protocol, phase)
    path = root / f"{phase}-summary.json"
    if path.exists() and fresh._read_json(path) != summary:
        raise ValueError(f"existing {phase} summary differs from frozen receipts")
    if not path.exists():
        fresh._atomic_json(path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--control-agent", type=Path, default=ROOT / SOURCE_PATHS["control"])
    parser.add_argument("--candidate-agent", type=Path, default=ROOT / SOURCE_PATHS["candidate"])
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent cold worker processes (1-3; default 1)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--partition", choices=fresh.PHASES)
    mode.add_argument("--freeze-finalist", action="store_true")
    mode.add_argument("--seal-confirmation", action="store_true")
    args = parser.parse_args()
    try:
        validate_worker_count(args.workers)
    except ValueError as error:
        parser.error(str(error))
    validate_protocol_path(args.protocol)
    validate_output_root(args.output_root)
    for arm, path in (("control", args.control_agent), ("candidate", args.candidate_agent)):
        if path.resolve() != (ROOT / SOURCE_PATHS[arm]).resolve():
            raise ValueError(f"v6 {arm} source path changed")
    protocol = fresh._read_json(args.protocol)
    historical = fresh.historical_geometry_seeds(ROOT / "experiments", exclude=args.protocol)
    validate_protocol(protocol, historical)
    paths = {"control": args.control_agent, "candidate": args.candidate_agent,
             "model": args.model, "protocol": args.protocol}
    validate_source_pair(protocol, paths["control"], paths["candidate"])
    verify_committed_source(protocol, paths["control"])
    identity = build_identity(args.protocol, protocol, paths)
    check_frozen_inputs(identity, protocol, paths)
    if args.preflight_only:
        for arm in fresh.ARMS:
            fresh._load_agent(paths[arm], paths["model"], protocol["controller_classes"][arm])
        print(json.dumps({"preflight": "PASS", "identity": identity,
                          "canonical_pairs": {phase: len(protocol["partitions"][phase]["seeds"]) * 4
                                              for phase in fresh.PHASES}}, indent=2))
        return 0
    fresh.prepare_run(args.output_root, identity)
    if args.freeze_finalist:
        freeze_finalist(args.output_root, identity, protocol)
        summary = report_phase(args.output_root, identity, protocol, "screen")
    elif args.seal_confirmation:
        seal_confirmation(args.output_root, identity, protocol)
        summary = report_phase(args.output_root, identity, protocol, "confirmation")
    else:
        summary = run_partition(args.output_root, identity, protocol, paths,
                                args.partition, workers=args.workers)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
