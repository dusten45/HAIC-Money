"""Frozen, reused-cell paired diagnosis; never a fresh evaluation or promotion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CELLS = ((1, 42), (1, 11), (1, 17), (1, 21), (1, 516237), (2, 644062), (3, 1007))
DOCUMENTS = ("RESTRICTIONs.md", "COMPETITION_INFO.md", "RESULTS.md", "SOTA.md", "report.pdf")
CONTROLLERS = {
    "compound-brake-onset-v1": "_CompoundBrakeOnsetController",
    "visible-compound-base-brake-v1": "_VisibleCompoundBaseBrakeController",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(protocol_path: Path, expected: dict[str, str]) -> dict:
    for filename in DOCUMENTS:
        if not (ROOT / filename).is_file() or not (ROOT / filename).stat().st_size:
            raise ValueError(f"Required document missing/empty: {filename}")
    paths = {"source": ROOT / "agent.py", "model": ROOT / "model.pt", "protocol": protocol_path}
    for name, path in paths.items():
        if digest(path) != expected[name].lower():
            raise ValueError(f"Frozen {name} hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if tuple(map(tuple, protocol["development_cells"])) != CELLS:
        raise ValueError("Diagnostic cells must match the preregistered reused-cell grid")
    if protocol["max_steps"] != 1000 or protocol["frame_skip"] != 4:
        raise ValueError("Diagnostic budget changed")
    experiment = protocol.get("experiment")
    if experiment not in CONTROLLERS or (
        protocol["control_class"], protocol["candidate_class"]
    ) != ("_CompoundClearingBrakeCarryController", CONTROLLERS[experiment]):
        raise ValueError("Unexpected controller pair")
    if any(protocol.get(name) is not False for name in ("training", "submission", "sota_promotion")):
        raise ValueError("This runner permits diagnosis only")
    protected: set[int] = set()
    training_exclusions: set[int] = set()
    for path in sorted((ROOT / "experiments").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, partition in record.get("partitions", {}).items():
            if name not in {"development", "screen"}:
                protected.update(partition.get("seeds", []))
        # Training exclusions include known consumed smoke cells (not holdouts).
        # They do not authorize opening any actual confirmation/blind partition.
        training_exclusions.update(record.get("reserved_training_seeds", []))
        for key, value in record.items():
            if "reserved" in key and "seed" in key and key != "reserved_training_seeds":
                if isinstance(value, list) and all(type(seed) is int for seed in value):
                    protected.update(value)
    requested = {seed for _, seed in CELLS}
    if requested & protected:
        raise ValueError(f"Reserved holdout geometry requested: {sorted(requested & protected)}")
    return {
        "protocol": protocol,
        "hashes": expected,
        "training_exclusion_reuse": sorted(requested & training_exclusions),
        "holdout_overlap": [],
        "documents": {name: digest(ROOT / name) for name in DOCUMENTS},
    }


def compare(rows: list[dict], *, require_complete: bool = True) -> dict:
    reasons = []
    paired_time_delta = 0
    completed_pairs = 0
    changed_calls = sum(
        row.get("brake_envelope_counts", {}).get("changed_calls", 0)
        for row in rows if row["arm"] == "candidate"
    )
    for track, seed in CELLS:
        pair = {row["arm"]: row for row in rows if (row["track_id"], row["seed"]) == (track, seed)}
        if set(pair) != {"control", "candidate"}:
            if require_complete:
                reasons.append(f"{track}/{seed}: incomplete pair")
            continue
        old, new = pair["control"], pair["candidate"]
        if old.get("error") or new.get("error"):
            reasons.append(f"{track}/{seed}: operational failure")
            continue
        if old["finished"] and not new["finished"]:
            reasons.append(f"{track}/{seed}: finish loss")
        for metric in ("damage", "collision_count", "offtrack_samples", "partial_offtrack_samples"):
            if new[metric] > old[metric]:
                reasons.append(f"{track}/{seed}: {metric} increased")
        if not old["finished"] and not new["finished"] and new["progress"] < old["progress"]:
            reasons.append(f"{track}/{seed}: DNF progress decreased")
        if old["finished"] and new["finished"]:
            completed_pairs += 1
            paired_time_delta += new["lap_time_ms"] - old["lap_time_ms"]
    return {
        "decision": "REJECT" if reasons else (
            "RETAIN_DIAGNOSTIC_CANDIDATE"
            if completed_pairs and paired_time_delta < 0 and changed_calls > 0
            else "INCONCLUSIVE"
        ),
        "reasons": reasons,
        "paired_completed_cells": completed_pairs,
        "paired_completed_time_delta_ms": paired_time_delta,
        "candidate_changed_brake_calls": changed_calls,
        "sota_promotion": False,
        "evidence_scope": "reused development diagnosis; no unseen-track claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/compound-brake-onset-v1.json")
    for kind in ("source", "model", "protocol"):
        parser.add_argument(f"--{kind}-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    expected = {kind: getattr(args, f"{kind}_sha256") for kind in ("source", "model", "protocol")}
    receipt = preflight(args.protocol, expected)
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", **receipt}, indent=2))
        return 0

    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import agent as agent_module
    from local_simulator.logging import save_run_log
    from local_simulator.schema import MapSpec
    from local_simulator.session import SimulationSession

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = ROOT / ".haic-artifacts" / receipt["protocol"]["experiment"] / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    receipt["harness_sha256"] = digest(Path(__file__))
    receipt["environment_hashes"] = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((ROOT / "core").rglob("*.py")) + [ROOT / "env_wrapper.py", ROOT / "damage.py"]
    }
    (output / "freeze.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    rows = []
    for track, seed in CELLS:
        for arm in ("control", "candidate"):
            preflight(args.protocol, expected)
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            row = {"track_id": track, "seed": seed, "arm": arm, "error": None}
            session = None
            try:
                started = time.perf_counter()
                policy = agent_module.Agent(model_path=str(ROOT / "model.pt"))
                controller_class = getattr(agent_module, receipt["protocol"][f"{arm}_class"])
                policy._forward_controller = controller_class()
                controller = policy._forward_controller
                brake_counts = {
                    "envelope_calls": 0, "changed_calls": 0,
                    "max_absolute_brake_relief": 0.0,
                }
                original_envelope = controller._curve_brake_envelope
                control_class = getattr(agent_module, receipt["protocol"]["control_class"])

                def counted_envelope(*, excess, base_brake):
                    value = original_envelope(excess=excess, base_brake=base_brake)
                    reference = control_class._curve_brake_envelope(
                        controller, excess=excess, base_brake=base_brake,
                    )
                    brake_counts["envelope_calls"] += 1
                    brake_counts["changed_calls"] += int(value != reference)
                    brake_counts["max_absolute_brake_relief"] = max(
                        brake_counts["max_absolute_brake_relief"], abs(reference - value),
                    )
                    return value

                controller._curve_brake_envelope = counted_envelope
                row["initialization_ms"] = (time.perf_counter() - started) * 1000
                session = SimulationSession.start(MapSpec(track, seed, "official", (), 1000, 4), policy)
                latencies = []
                offtrack_samples = 0
                partial_offtrack_samples = 0
                while not session.done and len(session.steps) < 1000:
                    started = time.perf_counter()
                    action = np.asarray(policy.act(session.observation), dtype=np.float32)
                    latencies.append((time.perf_counter() - started) * 1000)
                    if action.shape != (3,) or not np.isfinite(action).all():
                        raise ValueError("Invalid action")
                    if np.any(action < [-1, 0, 0]) or np.any(action > 1):
                        raise ValueError("Action outside official bounds")
                    # Explicit action bypasses the UI runner's exception-to-no-op fallback.
                    session.step(action)
                    wheels = session.raw_environment.car.wheels
                    contacts = [bool(wheel.tiles) for wheel in wheels]
                    offtrack_samples += int(not any(contacts))
                    partial_offtrack_samples += int(not all(contacts))
                log = session.finish()
                filename = f"track-{track}-seed-{seed}-{arm}.json"
                save_run_log(log, output / filename)
                row.update(log.summary)
                row.update({
                    "run_log": filename, "run_sha256": digest(output / filename),
                    "steps": len(log.steps), "offtrack_samples": offtrack_samples,
                    "partial_offtrack_samples": partial_offtrack_samples,
                    "action_latency_max_ms": max(latencies, default=0),
                    "action_latency_mean_ms": sum(latencies) / max(1, len(latencies)),
                    "brake_envelope_counts": brake_counts,
                })
                if row["initialization_ms"] > 10000 or row["action_latency_max_ms"] > 5000:
                    row["error"] = "Official initialization/action latency limit exceeded"
            except (Exception, KeyboardInterrupt) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                if session is not None:
                    save_run_log(session.finish(reason="diagnostic_error"), output / f"track-{track}-seed-{seed}-{arm}-error.json")
            finally:
                if session is not None:
                    session.close()
            rows.append(row)
            (output / "summary.json").write_text(json.dumps({"rows": rows, "comparison": compare(rows)}, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
            if row["error"]:
                print(f"Stopped on operational failure. Artifacts: {output}", flush=True)
                return 1
            partial = compare(rows, require_complete=False)
            if partial["reasons"]:
                # RULES requires stopping on safety regression. Do not spend
                # remaining cells after a completed pair already disqualifies it.
                (output / "stopped.json").write_text(json.dumps({
                    "reason": "paired_safety_regression", "comparison": partial,
                    "completed_episodes": len(rows),
                }, indent=2), encoding="utf-8")
                print(json.dumps({"output": str(output), **partial}), flush=True)
                return 0
    preflight(args.protocol, expected)
    print(json.dumps({"output": str(output), **compare(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
