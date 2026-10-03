"""Bounded paired diagnosis of a preregistered observation-only corridor repair."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.trace_corridor_failures import DOCUMENTS, diagnostic_json, digest, install_hooks, scalar_state

CELLS = ((1, 11), (1, 21), (1, 42), (1, 17), (1, 516237), (2, 644062), (3, 1007))
CANDIDATES = {
    "observed-road-target-v1": "_ObservedRoadTargetController",
    "observed-road-side-commit-v1": "_ObservedRoadSideCommitController",
    "observed-curve-arbitration-v1": "_ObservedCurveArbitrationController",
    "observed-centerline-arbitration-v1": "_ObservedCenterlineArbitrationController",
}
CONTROL = "_CompoundClearingBrakeCarryController"


def preflight(protocol_path: Path, expected: dict[str, str]) -> dict:
    for name in DOCUMENTS:
        if not (ROOT / name).is_file() or not (ROOT / name).stat().st_size:
            raise ValueError(f"Required document missing/empty: {name}")
    for kind, path in {"source": ROOT / "agent.py", "model": ROOT / "model.pt", "protocol": protocol_path}.items():
        if digest(path) != expected[kind].lower():
            raise ValueError(f"Frozen {kind} hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if tuple(map(tuple, protocol["development_cells"])) != CELLS:
        raise ValueError("Unexpected reused development cell order")
    if protocol.get("max_steps") != 1000 or protocol.get("frame_skip") != 4:
        raise ValueError("Protocol requires max_steps 1000 and official skip 4")
    if protocol.get("experiment") not in CANDIDATES or (
        protocol.get("control_class"), protocol.get("candidate_class")
    ) != (CONTROL, CANDIDATES[protocol["experiment"]]):
        raise ValueError("Unexpected preregistered controller pair")
    if any(protocol.get(key) is not False for key in ("training", "submission", "sota_promotion")):
        raise ValueError("Only development diagnosis is permitted")
    protected: set[int] = set()
    training_exclusions: set[int] = set()
    for path in sorted((ROOT / "experiments").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, partition in record.get("partitions", {}).items():
            if name not in {"development", "screen"}:
                protected.update(partition.get("seeds", []))
        training_exclusions.update(record.get("reserved_training_seeds", []))
        for key, value in record.items():
            if "reserved" in key and "seed" in key and key != "reserved_training_seeds":
                if isinstance(value, list) and all(type(seed) is int for seed in value):
                    protected.update(value)
    requested = {seed for _, seed in CELLS}
    if requested & protected:
        raise ValueError(f"Reserved holdout geometry requested: {sorted(requested & protected)}")
    return {"protocol": protocol, "hashes": expected, "holdout_overlap": [],
            "training_exclusion_reuse": sorted(requested & training_exclusions),
            "documents": {name: digest(ROOT / name) for name in DOCUMENTS}}


def compare(rows: list[dict], *, require_complete: bool = True) -> dict:
    reasons = []
    missing = []
    old_finishes = new_finishes = paired_time_delta = completed_pairs = 0
    for track, seed in CELLS:
        pair = {row["arm"]: row for row in rows if (row["track_id"], row["seed"]) == (track, seed)}
        if any(row.get("error") for row in pair.values()):
            reasons.append(f"{track}/{seed}: operational failure")
            continue
        if set(pair) != {"control", "candidate"}:
            missing.append([track, seed])
            continue
        old, new = pair["control"], pair["candidate"]
        old_finishes += int(old["finished"])
        new_finishes += int(new["finished"])
        if old["finished"] and not new["finished"]:
            reasons.append(f"{track}/{seed}: control finish lost")
        for metric in ("collision_count", "damage"):
            if new[metric] > old[metric]:
                reasons.append(f"{track}/{seed}: {metric} increased")
        if not old["finished"] and not new["finished"] and new["progress"] < old["progress"]:
            reasons.append(f"{track}/{seed}: DNF progress decreased")
        if old["finished"] and new["finished"]:
            completed_pairs += 1
            paired_time_delta += new["lap_time_ms"] - old["lap_time_ms"]
            if new["lap_time_ms"] > old["lap_time_ms"] * 1.05:
                reasons.append(f"{track}/{seed}: completed lap more than 5% slower")
    decision = "REJECT" if reasons else (
        "INCOMPLETE" if missing else (
            "RETAIN_DIAGNOSTIC_CANDIDATE" if (
                new_finishes > old_finishes or (new_finishes == old_finishes and paired_time_delta < 0)
            ) else "INCONCLUSIVE"
        )
    )
    return {"decision": decision, "reasons": reasons, "missing_cells": missing if require_complete else [],
            "control_finishes": old_finishes, "candidate_finishes": new_finishes,
            "paired_completed_cells": completed_pairs, "paired_completed_time_delta_ms": paired_time_delta,
            "sota_promotion": False, "evidence_scope": "reused development diagnosis; no unseen-track claim"}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=diagnostic_json), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    for kind in ("source", "model", "protocol"):
        parser.add_argument(f"--{kind}-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--record-observations", action="store_true")
    args = parser.parse_args()
    expected = {kind: getattr(args, f"{kind}_sha256").lower() for kind in ("source", "model", "protocol")}
    receipt = preflight(args.protocol, expected)
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", **receipt}, indent=2))
        return 0

    import numpy as np
    import torch
    import agent as agent_module
    from local_simulator.logging import run_log_to_dict
    from local_simulator.schema import MapSpec
    from local_simulator.session import SimulationSession

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = ROOT / ".haic-artifacts" / receipt["protocol"]["experiment"] / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    receipt["harness_sha256"] = digest(Path(__file__))
    receipt["record_observations"] = args.record_observations
    receipt["environment_hashes"] = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((ROOT / "core").rglob("*.py")) + [ROOT / "env_wrapper.py", ROOT / "damage.py"]
    }
    write_json(output / "freeze.json", receipt)
    print(str(output), flush=True)
    rows = []
    for track, seed in CELLS:
        for arm in ("control", "candidate"):
            preflight(args.protocol, expected)
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            cell_output = output / f"track-{track}-seed-{seed}-{arm}"
            cell_output.mkdir()
            row = {"track_id": track, "seed": seed, "arm": arm, "error": None}
            session = None
            observations = []
            latencies = []
            offtrack = partial_offtrack = changed_actions = 0
            try:
                started = time.perf_counter()
                policy = agent_module.Agent(model_path=str(ROOT / "model.pt"))
                controller = getattr(agent_module, receipt["protocol"][f"{arm}_class"])()
                policy._forward_controller = controller
                row["initialization_ms"] = (time.perf_counter() - started) * 1000
                if row["initialization_ms"] > 10000:
                    raise RuntimeError("Initialization exceeded official 10 second limit")
                shadow = None
                if arm == "candidate":
                    shadow = agent_module.Agent(model_path=str(ROOT / "model.pt"))
                    shadow._forward_controller = getattr(agent_module, CONTROL)()
                events: list[dict] = []
                install_hooks(controller, events)
                session = SimulationSession.start(MapSpec(track, seed, "official", (), 1000, 4), policy)
                if shadow is not None:
                    shadow.reset(session.observation)
                with (cell_output / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
                    while not session.done and len(session.steps) < 1000:
                        observation = np.asarray(session.observation)
                        if observation.shape != (4, 84, 84) or observation.dtype != np.float32:
                            raise ValueError("Invalid official observation shape/dtype")
                        if args.record_observations:
                            observations.append(observation.copy())
                        events.clear()
                        trace = {"step": len(session.steps), "state_before": scalar_state(controller)}
                        try:
                            started = time.perf_counter()
                            action = np.asarray(policy.act(observation), dtype=np.float32)
                            latencies.append((time.perf_counter() - started) * 1000)
                            if latencies[-1] > 5000:
                                raise RuntimeError("Action exceeded official 5 second limit")
                            if action.shape != (3,) or not np.isfinite(action).all():
                                raise ValueError("Invalid policy action")
                            if np.any(action < [-1, 0, 0]) or np.any(action > 1):
                                raise ValueError("Action outside official bounds")
                            trace["action"] = action.tolist()
                            if shadow is not None:
                                shadow_action = shadow.act(observation)
                                changed_actions += int(not np.array_equal(action, shadow_action))
                                trace["shadow_control_action"] = shadow_action.tolist()
                            # Only pixel observations enter either policy. Diagnostic world state follows act.
                            trace["outcome"] = asdict(session.step(action))
                            trace["environment_info"] = dict(session.last_info)
                            contacts = [bool(wheel.tiles) for wheel in session.raw_environment.car.wheels]
                            offtrack += int(not any(contacts))
                            partial_offtrack += int(not all(contacts))
                        except BaseException:
                            trace["error"] = traceback.format_exc()
                            raise
                        finally:
                            trace["hooks"] = list(events)
                            trace["state_after"] = scalar_state(controller)
                            stream.write(json.dumps(trace, allow_nan=False, default=diagnostic_json) + "\n")
                            stream.flush()
            except (Exception, KeyboardInterrupt) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                (cell_output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                if args.record_observations:
                    np.savez_compressed(cell_output / "observations.npz", observations=(
                        np.stack(observations) if observations else np.empty((0, 4, 84, 84), dtype=np.float32)
                    ))
                if session is not None:
                    try:
                        log = session.finish(reason="diagnostic_error" if row["error"] else None)
                        write_json(cell_output / "run.json", run_log_to_dict(log))
                        row.update(log.summary)
                        row["steps"] = len(log.steps)
                        row["run_log"] = str((cell_output / "run.json").relative_to(output))
                        row["run_sha256"] = digest(cell_output / "run.json")
                    finally:
                        session.close()
            row.update({"offtrack_samples": offtrack, "partial_offtrack_samples": partial_offtrack,
                        "changed_actions_on_candidate_observations": changed_actions,
                        "action_latency_max_ms": max(latencies, default=0),
                        "action_latency_mean_ms": sum(latencies) / max(1, len(latencies))})
            rows.append(row)
            comparison = compare(rows)
            write_json(output / "summary.json", {"rows": rows, "comparison": comparison})
            print(json.dumps(row), flush=True)
            if row["error"] or comparison["reasons"]:
                write_json(output / "stopped.json", {"reason": "operational_failure" if row["error"] else "paired_regression",
                                                       "comparison": comparison, "completed_episodes": len(rows)})
                print(json.dumps({"output": str(output), **comparison}), flush=True)
                return 1 if row["error"] else 0
    preflight(args.protocol, expected)
    print(json.dumps({"output": str(output), **compare(rows)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
