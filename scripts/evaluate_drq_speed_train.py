"""Paired, reused-development speed diagnostic for one immutable DrQ actor.

Run from the repository root with ``python -m scripts.evaluate_drq_speed_train``.
This operator never accesses official-server or protected evaluation cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from haic.algorithms.drq_v2.speed_control import release_light_brake


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-drq-speed-reused-development-v1"
SOURCE_ACTOR_SHA = "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954"
REQUIRED_SOURCES = {
    "scripts/evaluate_drq_speed_train.py", "haic/algorithms/drq_v2/speed_control.py",
    "agent.py", "train.py", "env_wrapper.py", "damage.py",
    "action_smoothing.py", "action_representation.py",
    "core/__init__.py", "core/finish_line.py", "core/obstacle_contacts.py",
    "core/track_variables.py", "core/vendor/__init__.py",
    "core/vendor/car_dynamics.py", "core/vendor/car_racing.py",
}
EXPECTED_CELLS = {(1, seed) for seed in (
    3910800153, 3910800045, 3910800148, 3910800124,
    3910800134, 3910800160, 3910800163, 3910800171,
    3910800187, 3910800172, 3910800175, 3910800182,
    3910800169, 3910800162, 3910800174, 3910800164,
)} | {(track, seed) for track in (2, 3) for seed in range(4000000001, 4000000009)}
ACCEPTANCE = {
    "max_lost_finishes_per_track": 0,
    "min_control_finishes_per_track": 1,
    "min_mutual_finishes_per_track": 1,
    "min_total_mutual_finishes": 3,
    "max_mean_paired_lap_delta_ms": -4000,
    "max_track_mean_paired_lap_delta_ms": 0,
}
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}
EXPECTED_EVIDENCE = {
    "diagnostic_track1": {
        "path": "runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/episodes.jsonl",
        "sha256": "aee2ae9b85172feb0e9a09697bdd92a34b29631cfc094586364f7c0b9f283441",
    },
    "residual_tracks2_3": {
        "path": "evaluations/drqv2-residual-options-pilot-iteration-1.json",
        "sha256": "9192e550e6de0545304c93a5d415b7c8d93aff0db83ab4e5151cb264c952d847",
    },
}
HISTORICAL_PINS = {
    "experiments/drqv2-geometry-mix-v1-r6.json":
        "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab",
    "runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json":
        "fb14fe9eab14f61cdecc45453b69e44d34fdb967368c44c255e827292804b380",
    "experiments/drqv2-residual-options-pilot-v1.json":
        "3fd1222a5c0833633d8c35bd3e765470556b23ae50ec03f33fe6496d907be56d",
    "runs/20260924-drqv2-residual-options-pilot/iteration-1-retry/manifest.completed.json":
        "5bb24edfe356aacef868c40cd26805dbe612e05fcb251d7421192378371ce3eb",
}
AUDIT_PATH = "experiments/drqv2-speed-reused-development-v1-audit.json"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_source(reference: dict, label: str) -> Path:
    relative = reference["path"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"unsafe {label} path")
    path = ROOT / relative
    if not path.is_file() or sha256(path) != reference["sha256"]:
        raise ValueError(f"{label} source hash mismatch: {relative}")
    return path


def check_current_claims() -> None:
    registry = ROOT / "experiments/train-seed-claims"
    if not registry.is_dir():
        raise ValueError("shared TRAIN claim registry is unavailable")
    candidate_seeds = {seed for _, seed in EXPECTED_CELLS}
    for path in registry.glob("*.json"):
        row = json.loads(path.read_text())
        if row.get("format") != "haic-train-seed-claim-v1" or type(row.get("geometry_seed")) is not int:
            raise ValueError(f"ambiguous shared claim record: {path}")
        if row["geometry_seed"] in candidate_seeds:
            raise ValueError(f"active shared claim overlaps speed cohort: {path}")


def check_reused_cells(protocol: dict) -> None:
    evidence = protocol["reuse_evidence"]
    if evidence != EXPECTED_EVIDENCE:
        raise ValueError("reuse evidence is not the frozen historical source set")
    diagnostic = checked_source(evidence["diagnostic_track1"], "track1 diagnostic")
    residual = checked_source(evidence["residual_tracks2_3"], "track2/3 residual pilot")
    for name, digest in HISTORICAL_PINS.items():
        checked_source({"path": name, "sha256": digest}, "historical protocol/manifest")
    r6 = json.loads((ROOT / "experiments/drqv2-geometry-mix-v1-r6.json").read_text())
    r6_manifest = json.loads((ROOT / "runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json").read_text())
    pilot = json.loads((ROOT / "experiments/drqv2-residual-options-pilot-v1.json").read_text())
    pilot_manifest = json.loads((ROOT / "runs/20260924-drqv2-residual-options-pilot/iteration-1-retry/manifest.completed.json").read_text())
    if (r6["diagnostic_pool"]["track_ids"] != [1] or r6["diagnostic_pool"]["frame_skip"] != 4
            or set(r6["diagnostic_pool"]["geometry_seeds"]) != {seed for track, seed in EXPECTED_CELLS if track == 1}
            or r6_manifest["files_sha256"]["episodes.jsonl"] != EXPECTED_EVIDENCE["diagnostic_track1"]["sha256"]
            or pilot["environment"]["obstacles"] is not True or pilot["environment"]["frame_skip"] != 4
            or pilot["environment"]["max_decisions_per_episode"] != 2000
            or pilot["development_screen"]["geometry_seeds"] != list(range(4000000001, 4000000009))
            or pilot_manifest["status"] != "completed"
            or pilot_manifest["protocol_sha256"] != HISTORICAL_PINS["experiments/drqv2-residual-options-pilot-v1.json"]):
        raise ValueError("prior development protocol or completed manifest does not establish reuse")
    track1 = set()
    for line in diagnostic.read_text().splitlines():
        row = json.loads(line)
        if (row.get("role") == "unchanged-source" and row.get("actor_sha256") == SOURCE_ACTOR_SHA
                and row.get("track_id") == 1 and row.get("repeat") == 0):
            track1.add(row["geometry_seed"])
    record = json.loads(residual.read_text())
    if record.get("base_actor_sha256") != SOURCE_ACTOR_SHA:
        raise ValueError("residual-pilot control actor differs from frozen source")
    track23 = set()
    for row in record["cells"]:
        control = row.get("control", {})
        if (row.get("track_id") in (2, 3) and row.get("geometry_seed") == control.get("geometry_seed")
                and row["track_id"] == control.get("track_id") and control.get("done") is True):
            track23.add((row["track_id"], row["geometry_seed"]))
    cells = [(row["track_id"], row["geometry_seed"]) for row in protocol["cells"]]
    if len(cells) != len(EXPECTED_CELLS) or set(cells) != EXPECTED_CELLS:
        raise ValueError("speed study must retain the entire predeclared reused-development cohort")
    for track, seed in cells:
        if type(track) is not int or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("invalid track or geometry seed")
        if (track == 1 and seed not in track1) or (track in (2, 3) and (track, seed) not in track23):
            raise ValueError(f"cell {(track, seed)} has no exact consumed-development outcome")


def preflight(protocol_path: Path) -> tuple[dict, dict]:
    if not protocol_path.is_file() or protocol_path.parent.resolve() != (ROOT / "experiments").resolve():
        raise ValueError("frozen protocol must be an existing experiments/ file")
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get("format") != FORMAT or protocol.get("actor", {}).get("sha256") != SOURCE_ACTOR_SHA
            or protocol.get("roles") != ["control", "light_brake_release"]
            or protocol.get("repeats") != 2 or protocol.get("max_steps") != 2000
            or protocol.get("frame_skip") != 4 or protocol.get("obstacles") is not True
            or protocol.get("reward_shaping") is not False or protocol.get("reuse_only") is not True
            or protocol.get("acceptance") != ACCEPTANCE):
        raise ValueError("speed study contract is not the frozen reused-development comparison")
    checked_source(protocol["actor"], "actor")
    if set(protocol["source_sha256"]) != REQUIRED_SOURCES:
        raise ValueError("speed operator or environment source pin set is incomplete")
    for name, digest in protocol["source_sha256"].items():
        checked_source({"path": name, "sha256": digest}, name)
    if protocol["cell_audit"].get("path") != AUDIT_PATH:
        raise ValueError("speed study requires the fixed candidate-specific audit receipt")
    audit = json.loads(checked_source(protocol["cell_audit"], "candidate-specific cell audit").read_text())
    if (audit.get("format") != "haic-drq-speed-cell-audit-v1"
            or audit.get("status") != "reuse_only" or audit.get("actor_sha256") != SOURCE_ACTOR_SHA
            or {(row["track_id"], row["geometry_seed"]) for row in audit["cells"]} != EXPECTED_CELLS
            or len(audit["cells"]) != len(EXPECTED_CELLS)
            or audit.get("active_claim_overlap") != [] or audit.get("protected_overlap") != []
            or audit.get("partial_reset_overlap") != []):
        raise ValueError("candidate-specific reused-development audit does not clear the exact cohort")
    check_current_claims()
    check_reused_cells(protocol)
    return protocol, {
        "format": FORMAT,
        "protocol_sha256": sha256(protocol_path),
        "actor_sha256": SOURCE_ACTOR_SHA,
        "cell_count": len(protocol["cells"]),
        "planned_episodes": len(protocol["cells"]) * 4,
        "reuse_only": True,
        "cell_audit_sha256": protocol["cell_audit"]["sha256"],
        "provenance": "Previously consumed local development; not fresh, confirmation, blind or official",
    }


def run_episode(model, env, track: int, seed: int, role: str, repeat: int, max_steps: int,
                progress: dict | None = None) -> tuple[dict, np.ndarray]:
    observation, reset_info = env.reset()
    if reset_info.get("track_id") != track or reset_info.get("seed") != seed:
        raise ValueError("reset escaped frozen track/seed cell")
    raw = env.unwrapped
    if (raw.track_id != track or raw.track_seed != seed or raw.track_variables is None
            or len(raw.track_variables.obstacles) != 6 or len(raw.obstacles) != 6):
        raise ValueError("underlying CarRacing road/obstacles do not match the frozen cell")
    points = np.asarray(raw.track, dtype="<f8")
    obstacles = np.asarray([(*spec.position, spec.radius) for spec in raw.track_variables.obstacles],
                           dtype="<f8")
    if (points.ndim != 2 or points.shape[1] != 4 or len(points) < 20
            or obstacles.shape != (6, 3) or not np.isfinite(points).all()
            or not np.isfinite(obstacles).all()):
        raise ValueError("invalid reset road or obstacle fingerprint source")
    road_sha = hashlib.sha256(np.ascontiguousarray(points[:, 2:4]).tobytes()).hexdigest()
    obstacle_sha = hashlib.sha256(np.ascontiguousarray(obstacles).tobytes()).hexdigest()
    observation_sha = hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()
    model.reset(observation)
    start_time = float(raw.t)
    actions = []
    reward_sum = 0.0
    intervention_count = 0
    info = {}
    for step in range(1, max_steps + 1):
        baseline = np.asarray(model.act(observation), dtype=np.float32)
        if (baseline.shape != (3,) or not np.isfinite(baseline).all()
                or not -1.0 <= baseline[0] <= 1.0
                or np.any(baseline[1:] < 0.0) or np.any(baseline[1:] > 1.0)):
            raise ValueError("base DrQ actor returned an invalid official action")
        action = release_light_brake(baseline) if role == "light_brake_release" else baseline
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("invalid applied action")
        intervention_count += int(not np.array_equal(action, baseline))
        observation, reward, terminated, truncated, info = env.step(action)
        info = {**reset_info, **info}
        if info.get("track_id") != track or info.get("seed") != seed:
            raise ValueError("transition escaped frozen track/seed cell")
        actions.append(action.copy())
        if progress is not None:
            progress["decisions"] = step
        reward_sum += float(reward)
        if terminated or truncated:
            break
    else:
        raise RuntimeError("TimeLimit failed to terminate at the declared horizon")
    finished = info.get("finished") is True
    finish_time = info.get("finish_time_s")
    if raw.track_id != track or raw.track_seed != seed or len(raw.obstacles) != 6:
        raise ValueError("underlying CarRacing road changed before episode ended")
    if finished != (finish_time is not None):
        raise ValueError("finish flag and crossing time disagree")
    lap_ms = round((float(finish_time) - start_time) * 1000) if finished and finish_time is not None else None
    if (lap_ms is not None and lap_ms <= 0) or not math.isfinite(reward_sum):
        raise ValueError("invalid lap time or reward")
    for key, lower, upper in (("progress", 0.0, 1.0), ("damage", 0.0, 1.0)):
        value = float(info[key])
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"invalid {key} telemetry")
    trace = np.asarray(actions, dtype=np.float32)
    return {
        "track_id": track, "geometry_seed": seed, "role": role, "repeat": repeat,
        "steps": len(actions), "finished": finished, "lap_time_ms": lap_ms,
        "start_time_s": start_time, "finish_time_s": finish_time,
        "road_sha256": road_sha, "obstacle_sha256": obstacle_sha,
        "reset_observation_sha256": observation_sha,
        "progress": float(info["progress"]), "damage": float(info["damage"]),
        "retire_reason": info.get("retire_reason"), "raw_reward": reward_sum,
        "interventions": intervention_count,
        "action_trace_sha256": hashlib.sha256(trace.tobytes()).hexdigest(),
    }, trace


def summarize(episodes: list[dict], cells: list[dict]) -> dict:
    keys = [(row["track_id"], row["geometry_seed"], row["role"], row["repeat"]) for row in episodes]
    if len(keys) != len(cells) * 4 or len(set(keys)) != len(keys):
        raise ValueError("missing or duplicate episode/reload cell")
    canonical = {(row["track_id"], row["geometry_seed"], row["role"]): row
                 for row in episodes if row["repeat"] == 0}
    if len(canonical) != len(cells) * 2:
        raise ValueError("missing or duplicate canonical episodes")
    for row in episodes:
        other = canonical[row["track_id"], row["geometry_seed"], row["role"]]
        comparable = ("finished", "lap_time_ms", "steps", "action_trace_sha256", "interventions",
                      "start_time_s", "finish_time_s", "progress", "damage", "retire_reason", "raw_reward",
                      "road_sha256", "obstacle_sha256", "reset_observation_sha256")
        if any(row.get(key) != other.get(key) for key in comparable):
            raise ValueError("CPU reload repeat changed the deterministic episode")
    by_track = {}
    for track in (1, 2, 3):
        pairs = [(canonical[track, row["geometry_seed"], "control"],
                  canonical[track, row["geometry_seed"], "light_brake_release"])
                 for row in cells if row["track_id"] == track]
        for base, test in pairs:
            for key in ("road_sha256", "obstacle_sha256", "reset_observation_sha256", "start_time_s"):
                if base.get(key) != test.get(key):
                    raise ValueError("paired policies did not start from the same road/obstacles/observation")
        mutual = [(base, test) for base, test in pairs if base["finished"] and test["finished"]]
        by_track[str(track)] = {
            "cells": len(pairs), "control_finishes": sum(base["finished"] for base, _ in pairs),
            "treatment_finishes": sum(test["finished"] for _, test in pairs),
            "lost_finishes": sum(base["finished"] and not test["finished"] for base, test in pairs),
            "gained_finishes": sum(not base["finished"] and test["finished"] for base, test in pairs),
            "mutual_finishes": len(mutual),
            "paired_lap_delta_ms": [test["lap_time_ms"] - base["lap_time_ms"] for base, test in mutual],
        }
    paired_deltas = [delta for track in by_track.values() for delta in track["paired_lap_delta_ms"]]
    eligible = (all(track["lost_finishes"] == 0 for track in by_track.values())
                and all(track["control_finishes"] > 0 for track in by_track.values())
                and all(track["mutual_finishes"] > 0 for track in by_track.values()))
    return {
        "by_track": by_track, "mutual_finishes": len(paired_deltas),
        "mean_paired_lap_delta_ms": sum(paired_deltas) / len(paired_deltas) if paired_deltas else None,
        "retention_and_speed_measurable": eligible,
        "speed_signal": (eligible and len(paired_deltas) >= 3
                         and all(sum(track["paired_lap_delta_ms"]) / len(track["paired_lap_delta_ms"]) <= 0
                                 for track in by_track.values())
                         and sum(paired_deltas) / len(paired_deltas) <= -4000),
        "ranked": False, "official_score": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    protocol, receipt = preflight(args.protocol)
    import cv2
    import gymnasium
    import torch

    if (sys.platform != "linux" or sys.version_info[:2] != (3, 11)
            or any(os.environ.get(name) != value for name, value in CPU_ENV.items())
            or torch.cuda.is_available() or torch.version.cuda is not None
            or not torch.__version__.startswith("2.1.0")
            or np.__version__ != "1.26.0" or gymnasium.__version__ != "0.29.1"
            or getattr(cv2, "__version__", None) != "4.8.1"):
        raise RuntimeError("paired DrQ driving requires the isolated official-compatible CPU runtime")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    receipt["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                          "numpy": np.__version__, "gymnasium": gymnasium.__version__,
                          "opencv": getattr(cv2, "__version__", None), "cpu_environment": CPU_ENV,
                          "torch_threads": torch.get_num_threads(),
                          "torch_interop_threads": torch.get_num_interop_threads()}
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"never overwrite an existing speed-study run: {run_dir}")
    if not run_dir.parent.is_dir() or run_dir.parent != (ROOT / "runs").resolve():
        raise ValueError("run directory must be a new direct child of runs/")
    if args.preflight:
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return
    run_dir.mkdir()
    (run_dir / "preflight.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    from agent import Agent
    from train import build_env
    episodes = []
    traces = run_dir / "traces"
    traces.mkdir()
    actor_copy = run_dir / "actor.pt"
    shutil.copyfile(ROOT / protocol["actor"]["path"], actor_copy)
    if sha256(actor_copy) != SOURCE_ACTOR_SHA:
        raise ValueError("run-local DrQ actor copy differs from frozen source")
    active_cell = None
    try:
        with ((run_dir / "episodes.jsonl").open("x") as ledger,
              (run_dir / "reset-intents.jsonl").open("x") as intents):
            for role in protocol["roles"]:
                for repeat in range(protocol["repeats"]):
                    if sha256(actor_copy) != SOURCE_ACTOR_SHA:
                        raise ValueError("run-local actor changed before CPU reload")
                    model = Agent(model_path=str(actor_copy))
                    for cell in protocol["cells"]:
                        track, seed = cell["track_id"], cell["geometry_seed"]
                        active_cell = {"role": role, "repeat": repeat, "track_id": track,
                                       "geometry_seed": seed, "decisions": 0}
                        check_current_claims()
                        intents.write(json.dumps({"event": "reset_intent", **active_cell}, sort_keys=True) + "\n")
                        intents.flush()
                        os.fsync(intents.fileno())
                        env = build_env(track, seed, protocol["max_steps"], protocol["frame_skip"],
                                        reward_shaping=False, obstacles=True)
                        try:
                            record, trace = run_episode(model, env, track, seed, role, repeat,
                                                        protocol["max_steps"], progress=active_cell)
                        finally:
                            env.close()
                        trace_path = traces / f"{role}-repeat{repeat}-track{track}-seed{seed}.npz"
                        np.savez_compressed(trace_path, official_action=trace)
                        record["trace_path"] = trace_path.relative_to(ROOT).as_posix()
                        record["trace_sha256"] = sha256(trace_path)
                        episodes.append(record)
                        ledger.write(json.dumps(record, sort_keys=True) + "\n")
                        ledger.flush()
                        os.fsync(ledger.fileno())
                        intents.write(json.dumps({"event": "end", "role": role, "repeat": repeat,
                                                  "track_id": track, "geometry_seed": seed,
                                                  "decisions": record["steps"],
                                                  "action_trace_sha256": record["action_trace_sha256"]},
                                                 sort_keys=True) + "\n")
                        intents.flush()
                        os.fsync(intents.fileno())
                        active_cell = None
                        print(json.dumps({"role": role, "repeat": repeat, "track_id": track,
                                          "seed": seed, "finished": record["finished"]}), flush=True)
        result = {**receipt, "episodes": len(episodes), "summary": summarize(episodes, protocol["cells"]),
                  "actor_copy_sha256": sha256(actor_copy),
                  "episode_ledger_sha256": sha256(run_dir / "episodes.jsonl"),
                  "reset_intents_sha256": sha256(run_dir / "reset-intents.jsonl"),
                  "trace_files_sha256": {row["trace_path"]: row["trace_sha256"] for row in episodes}}
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except Exception as error:
        (run_dir / "failure.json").write_text(json.dumps({**receipt, "episodes_completed": len(episodes),
                                                          "active_cell": active_cell,
                                                          "error": repr(error)}, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
