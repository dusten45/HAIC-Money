"""Source-bound fresh evaluation for one frozen camera-only bare controller.

The registered seed/decision protocol is intentionally unbound until candidate
development ends. This runner never evaluates a fresh cell with a draft source.
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


NAME = "camera-policy-generalization-v1"
PROTOCOL_PATH = ROOT / "experiments" / f"{NAME}.json"
OUTPUT_ROOT = ROOT / ".haic-artifacts" / NAME / "run"
SOURCE_PATHS = {
    arm: f".haic-artifacts/{NAME}/snapshots/{arm}_agent.py"
    for arm in fresh.ARMS
}
CONTROL_CLASS = "_CompoundClearingBrakeCarryController"
TRACKS = [1, 2, 3, 4]
SEED_COUNTS = {"screen": 8, "confirmation": 16, "blind": 8}
SPOT_INDEX = {"screen": ((1, 0), (4, 7)),
              "confirmation": ((1, 0), (2, 1), (3, 2), (4, 3)),
              "blind": ((1, 0), (4, 7))}
THRESHOLDS = {
    "minimum_net_finish_gain": {"screen": 3, "confirmation": 5, "blind": 3},
    "maximum_lost_control_finishes": {"screen": 1, "confirmation": 2, "blind": 1},
    "maximum_new_candidate_crashes": {"screen": 1, "confirmation": 2, "blind": 1},
    "moderate_both_dnf_progress_loss": .05,
    "maximum_moderate_both_dnf_losses": {"screen": 1, "confirmation": 2, "blind": 1},
    "severe_both_dnf_progress_loss": .15,
    "maximum_severe_both_dnf_losses": 0,
    "maximum_single_cell_contact_increase": 2,
    "maximum_single_cell_damage_increase": .4,
    "maximum_aggregate_shared_finish_time_ratio": 1.10,
    "maximum_single_shared_finish_time_ratio": 1.15,
    "maximum_aggregate_contact_increase": 0,
    "maximum_aggregate_damage_increase": 0.0,
}


def derived_seeds(phase: str) -> list[int]:
    return [int.from_bytes(hashlib.sha256(f"{NAME}:{phase}:{index}".encode()).digest()[:4], "big")
            for index in range(SEED_COUNTS[phase])]


def validate_protocol(protocol: dict, historical_seeds: set[int]) -> None:
    if protocol.get("name") != NAME or protocol.get("status") != "PREREGISTERED":
        raise ValueError("registered camera-policy protocol and frozen source are required")
    fresh.validate_protocol(protocol, historical_seeds)
    if protocol.get("decision_thresholds") != THRESHOLDS:
        raise ValueError("registered decision thresholds changed")
    if protocol.get("source_snapshots") != SOURCE_PATHS:
        raise ValueError("source snapshot paths changed")
    classes = protocol["controller_classes"]
    candidate_class = classes["candidate"]
    if (classes["control"] != CONTROL_CLASS or not candidate_class.startswith("_")
            or candidate_class == CONTROL_CLASS or "PENDING" in candidate_class):
        raise ValueError("actual control and candidate route classes must be bound")
    construction = protocol.get("source_construction", {})
    commit = construction.get("implementation_commit")
    if (not isinstance(commit, str) or not 7 <= len(commit) <= 40
            or any(char not in "0123456789abcdef" for char in commit)):
        raise ValueError("source construction needs a committed Git SHA")
    fresh._sha256_string(construction.get("agent_blob_sha256"), "agent_blob_sha256")
    if (construction["agent_blob_sha256"] != protocol["control_agent_sha256"]
            or construction.get("selector_from") != f"{CONTROL_CLASS}()"
            or construction.get("selector_to") != f"{candidate_class}()"):
        raise ValueError("source selector construction is inconsistent")
    if protocol.get("training") is not False:
        raise ValueError("only a fixed bare Agent may enter this evaluation")
    for phase in fresh.PHASES:
        part = protocol["partitions"][phase]
        seeds = derived_seeds(phase)
        if part["track_ids"] != TRACKS or part["seeds"] != seeds:
            raise ValueError(f"{phase} registered track or seed grid changed")
        if part.get("spot_check_cells") != [[track, seeds[index]] for track, index in SPOT_INDEX[phase]]:
            raise ValueError(f"{phase} deterministic spot checks changed")


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
        raise ValueError("control snapshot differs from committed agent.py")
    if hashlib.sha256(completed.stdout).hexdigest() != protocol["source_construction"]["agent_blob_sha256"]:
        raise ValueError("committed agent.py SHA256 mismatch")


def build_identity(protocol_path: Path, protocol: dict, paths: dict[str, Path]) -> dict:
    identity = fresh.build_identity(protocol_path, protocol, paths["control"],
                                    paths["candidate"], paths["model"])
    runner_hash = fresh.digest(Path(__file__))
    return {**identity, "runner_sha256": runner_hash,
            "protocol_name": NAME,
            "candidate_route_class": protocol["controller_classes"]["candidate"]}


def check_frozen_inputs(identity: dict, protocol: dict, paths: dict[str, Path]) -> None:
    fresh.check_frozen_inputs(identity, paths)
    if (identity["runner_sha256"] != fresh.digest(Path(__file__))
            or identity["protocol_name"] != NAME
            or identity["candidate_route_class"] != protocol["controller_classes"]["candidate"]):
        raise ValueError("camera-policy runner or route changed")
    validate_source_pair(protocol, paths["control"], paths["candidate"])


def _metric_errors(row: dict, track: int, seed: int, arm: str) -> list[str]:
    label = f"{track}/{seed}/{arm}"
    errors = []
    for field, limit in (("initialization_ms", 10000.0), ("reset_ms", 5000.0),
                         ("action_latency_max_ms", 5000.0), ("peak_worker_rss_mib", 1024.0)):
        value = row.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
            errors.append(f"{label}: runtime {field} invalid or exceeded")
    progress = row.get("progress")
    contacts = row.get("collision_count")
    damage = row.get("damage")
    if type(row.get("finished")) is not bool:
        errors.append(f"{label}: finished must be boolean")
    if type(progress) not in (int, float) or not math.isfinite(progress) or not 0 <= progress <= 1.01:
        errors.append(f"{label}: invalid progress")
    if type(contacts) is not int or contacts < 0:
        errors.append(f"{label}: invalid contact count")
    if type(damage) not in (int, float) or not math.isfinite(damage) or not 0 <= damage <= 1.0:
        errors.append(f"{label}: invalid damage")
    lap_time = row.get("lap_time_ms")
    if row.get("finished") and (type(lap_time) not in (int, float)
                                or not math.isfinite(lap_time) or lap_time <= 0):
        errors.append(f"{label}: finished lap needs positive lap time")
    if not row.get("finished") and lap_time is not None:
        errors.append(f"{label}: DNF lap time must be null")
    trace = row.get("action_trace_sha256")
    if not isinstance(trace, str) or len(trace) != 64 or any(ch not in "0123456789abcdef" for ch in trace):
        errors.append(f"{label}: missing action trace SHA256")
    return errors


def compare_pairs(rows: list[dict], cells: list[tuple[int, int, int]], phase: str) -> dict:
    """Apply the finish-first rule and fixed safety/pace budgets to canonical pairs."""
    lookup = {(row["track_id"], row["seed"], row["repeat"], row["arm"]): row for row in rows}
    canonical = [(track, seed) for track, seed, repeat in cells if repeat == 0]
    reasons: list[str] = []
    missing: list[list[int]] = []
    paired: list[dict] = []
    totals = {arm: {"finishes": 0, "progress": 0.0, "contacts": 0, "damage": 0.0}
              for arm in fresh.ARMS}
    shared_time = {arm: 0.0 for arm in fresh.ARMS}
    shared_count = 0
    losses = crashes = moderate_dnf = severe_dnf = 0
    for track, seed in canonical:
        control = lookup.get((track, seed, 0, "control"))
        candidate = lookup.get((track, seed, 0, "candidate"))
        if control is None or candidate is None:
            missing.append([track, seed])
            paired.append({"track_id": track, "seed": seed, "status": "MISSING_PAIR"})
            continue
        if control.get("error") or candidate.get("error"):
            reasons.append(f"{track}/{seed}: operational failure")
            paired.append({"track_id": track, "seed": seed, "status": "OPERATIONAL_FAILURE"})
            continue
        row_errors = [_metric_errors(row, track, seed, arm)
                      for arm, row in (("control", control), ("candidate", candidate))]
        if row_errors[0] or row_errors[1]:
            reasons.extend(row_errors[0] + row_errors[1])
            paired.append({"track_id": track, "seed": seed, "status": "INVALID_METRIC"})
            continue
        for arm, row in (("control", control), ("candidate", candidate)):
            totals[arm]["finishes"] += int(row["finished"])
            totals[arm]["progress"] += float(row["progress"])
            totals[arm]["contacts"] += int(row["collision_count"])
            totals[arm]["damage"] += float(row["damage"])
        flags = []
        if control["finished"] and not candidate["finished"]:
            losses += 1
            flags.append("control finish lost")
        if candidate.get("retire_reason") == "crash" and control.get("retire_reason") != "crash":
            crashes += 1
            flags.append("new candidate crash")
        if not control["finished"] and not candidate["finished"]:
            progress_loss = float(control["progress"]) - float(candidate["progress"])
            if progress_loss > THRESHOLDS["moderate_both_dnf_progress_loss"] + 1e-9:
                moderate_dnf += 1
                flags.append("moderate both-DNF progress loss")
            if progress_loss > THRESHOLDS["severe_both_dnf_progress_loss"] + 1e-9:
                severe_dnf += 1
                flags.append("severe both-DNF progress loss")
        if candidate["collision_count"] - control["collision_count"] > THRESHOLDS["maximum_single_cell_contact_increase"]:
            flags.append("single-cell contact increase")
        if candidate["damage"] - control["damage"] > THRESHOLDS["maximum_single_cell_damage_increase"] + 1e-9:
            flags.append("single-cell damage increase")
        if control["finished"] and candidate["finished"]:
            shared_count += 1
            for arm, row in (("control", control), ("candidate", candidate)):
                shared_time[arm] += float(row["lap_time_ms"])
            if (candidate["lap_time_ms"] >
                    THRESHOLDS["maximum_single_shared_finish_time_ratio"] * control["lap_time_ms"] + 1e-9):
                flags.append("single shared-finish time regression")
        reasons.extend(f"{track}/{seed}: {flag}" for flag in flags if flag in (
            "single-cell contact increase", "single-cell damage increase",
            "single shared-finish time regression"))
        paired.append({"track_id": track, "seed": seed, "status": "COMPLETE",
                       "control": {key: control.get(key) for key in
                                   ("finished", "progress", "lap_time_ms", "collision_count", "damage", "retire_reason")},
                       "candidate": {key: candidate.get(key) for key in
                                     ("finished", "progress", "lap_time_ms", "collision_count", "damage", "retire_reason")},
                       "flags": flags})
    if losses > THRESHOLDS["maximum_lost_control_finishes"][phase]:
        reasons.append("lost control finishes exceed phase budget")
    if crashes > THRESHOLDS["maximum_new_candidate_crashes"][phase]:
        reasons.append("new candidate crashes exceed phase budget")
    if moderate_dnf > THRESHOLDS["maximum_moderate_both_dnf_losses"][phase]:
        reasons.append("moderate both-DNF progress losses exceed phase budget")
    if severe_dnf > THRESHOLDS["maximum_severe_both_dnf_losses"]:
        reasons.append("severe both-DNF progress loss")
    if not missing:
        if totals["candidate"]["contacts"] > totals["control"]["contacts"] + THRESHOLDS["maximum_aggregate_contact_increase"]:
            reasons.append("aggregate contacts increased")
        if totals["candidate"]["damage"] > totals["control"]["damage"] + THRESHOLDS["maximum_aggregate_damage_increase"] + 1e-9:
            reasons.append("aggregate damage increased")
        if shared_count and shared_time["candidate"] > THRESHOLDS["maximum_aggregate_shared_finish_time_ratio"] * shared_time["control"] + 1e-9:
            reasons.append("aggregate shared-finish time regression")
    denominator = len(canonical)
    net_finishes = totals["candidate"]["finishes"] - totals["control"]["finishes"]
    decision = ("REJECT" if reasons else "INCOMPLETE" if missing else
                "RETAIN" if net_finishes >= THRESHOLDS["minimum_net_finish_gain"][phase]
                else "INCONCLUSIVE")
    return {"decision": decision, "reasons": reasons, "missing_cells": missing,
            "canonical_cells": denominator, "paired_cells": paired,
            "control_finishes": totals["control"]["finishes"],
            "candidate_finishes": totals["candidate"]["finishes"],
            "net_finish_gain": net_finishes,
            "lost_control_finishes": losses, "new_candidate_crashes": crashes,
            "moderate_both_dnf_losses": moderate_dnf,
            "severe_both_dnf_losses": severe_dnf,
            "control_mean_progress": totals["control"]["progress"] / denominator,
            "candidate_mean_progress": totals["candidate"]["progress"] / denominator,
            "control_contacts": totals["control"]["contacts"],
            "candidate_contacts": totals["candidate"]["contacts"],
            "control_damage": totals["control"]["damage"],
            "candidate_damage": totals["candidate"]["damage"],
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
    summary = compare_pairs(rows, cells, phase)
    spot_failures = fresh._verify_spot_checks(rows, protocol, phase)
    if spot_failures:
        summary["reasons"].extend(spot_failures)
        summary["decision"] = ("INCOMPLETE" if any("missing" in item for item in spot_failures)
                               and summary["decision"] != "REJECT" else "REJECT")
    summary.update({"partition": phase, "protocol_sha256": identity["protocol_sha256"],
                    "candidate_agent_sha256": identity["source_sha256"]["candidate"],
                    "runner_sha256": identity["runner_sha256"]})
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


def validate_worker_count(workers: int) -> int:
    if type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError("workers must be an integer from 1 to 3")
    return workers


def _run_cold_episode(identity: dict, protocol: dict, paths: dict[str, Path],
                      arm: str, track: int, seed: int) -> dict:
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
                raise ValueError("cold worker returned a non-object result")
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
    validate_worker_count(workers)
    require_phase(root, identity, protocol, phase)
    with fresh._run_lock(root):
        jobs = []
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
                    arm, track, seed, _ = jobs[next_job]
                    pending[next_job] = pool.submit(_run_cold_episode, identity, protocol,
                                                    paths, arm, track, seed)
                    next_job += 1

            fill_window()
            for index, (arm, track, seed, repeat) in enumerate(jobs):
                measured = pending.pop(index).result()
                row = {"partition": phase, "arm": arm, "track_id": track,
                       "seed": seed, "repeat": repeat, **measured}
                fresh.record_cell(root, identity, row)
                if row.get("error"):
                    failure = report_phase(root, identity, protocol, phase)
                    fresh._atomic_json(root / f"{phase}-summary.json", failure)
                    raise RuntimeError(f"worker failed at {phase}/{arm}/{track}/{seed}/{repeat}")
                print(json.dumps({"cell": [phase, arm, track, seed, repeat],
                                  "finished": row["finished"], "progress": row["progress"],
                                  "contacts": row["collision_count"]}), flush=True)
                fill_window()
    summary = report_phase(root, identity, protocol, phase)
    target = root / f"{phase}-summary.json"
    if target.exists() and fresh._read_json(target) != summary:
        raise ValueError(f"existing {phase} summary differs from frozen receipts")
    if not target.exists():
        fresh._atomic_json(target, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--control-agent", type=Path, default=ROOT / SOURCE_PATHS["control"])
    parser.add_argument("--candidate-agent", type=Path, default=ROOT / SOURCE_PATHS["candidate"])
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument("--workers", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--partition", choices=fresh.PHASES)
    mode.add_argument("--freeze-finalist", action="store_true")
    mode.add_argument("--seal-confirmation", action="store_true")
    args = parser.parse_args()
    validate_worker_count(args.workers)
    if args.protocol.resolve() != PROTOCOL_PATH.resolve() or args.output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError("registered protocol and output root are required")
    for arm in fresh.ARMS:
        if getattr(args, f"{arm}_agent").resolve() != (ROOT / SOURCE_PATHS[arm]).resolve():
            raise ValueError(f"registered {arm} source path is required")
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
        print(json.dumps({"preflight": "PASS", "identity": identity}, indent=2))
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
