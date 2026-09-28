"""Compare two random-action seed distributions on already-consumed TRAIN roads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from haic.algorithms.tdmpc2.haic_env import environment_action, make_training_env


ROOT = Path(__file__).resolve().parents[1]
V2_PROTOCOL = "experiments/tdmpc2-reused-train-pilot-v2.json"
ARMS = ("independent_3d", "exclusive_2d")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(protocol: dict, protocol_path: Path) -> str:
    if (protocol.get("format") != "haic-tdmpc2-exploration-v2"
            or protocol.get("partition") != "consumed-TRAIN-development"
            or protocol.get("arms") != list(ARMS)):
        raise ValueError("invalid exploration scope")
    if protocol_path.resolve() != ROOT / "experiments/tdmpc2-exploration-v2.json":
        raise ValueError("unrecognized exploration protocol")
    source = protocol["source_sha256"]
    if set(source) != {"scripts/diagnose_tdmpc2_exploration.py",
                       "haic/algorithms/tdmpc2/haic_env.py", "env_wrapper.py", "damage.py",
                       "core/vendor/car_racing.py", "core/vendor/car_dynamics.py",
                       "core/track_variables.py", "core/obstacle_contacts.py", "core/finish_line.py"}:
        raise ValueError("incomplete exploration source map")
    for name, digest in source.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"exploration source drift: {name}")
    if sha256(ROOT / V2_PROTOCOL) != protocol["v2_protocol_sha256"]:
        raise ValueError("v2 road provenance drift")
    v2 = json.loads((ROOT / V2_PROTOCOL).read_text())
    if (protocol["cells"] != v2["cells"] or protocol["max_steps"] != 2000
            or protocol["repeats"] != 2 or protocol["rng_seed"] != 7332026
            or protocol["max_decisions_per_arm"] != 16000):
        raise ValueError("exploration cells or comparison budget changed")
    if protocol["environment"] != v2["environment"]:
        raise ValueError("environment is not the v2 TRAIN environment")
    if (protocol["resource_floor_bytes"] != {"cgroup": 16 * 1024**3,
                                               "disk": 5 * 1024**3}):
        raise ValueError("resource floor changed")
    for name in ("memory.max", "memory.current"):
        if not (Path("/sys/fs/cgroup") / name).is_file():
            raise ValueError("raw cgroup memory unavailable")
    memory = int(Path("/sys/fs/cgroup/memory.max").read_text()) - int(
        Path("/sys/fs/cgroup/memory.current").read_text())
    disk = os.statvfs(ROOT / "runs")
    if (memory < protocol["resource_floor_bytes"]["cgroup"]
            or disk.f_bavail * disk.f_frsize < protocol["resource_floor_bytes"]["disk"]):
        raise ValueError("insufficient raw resource headroom")
    return sha256(protocol_path)


def action_for(draw: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    if draw.shape != (3,) or not np.isfinite(draw).all() or (np.abs(draw) > 1).any():
        raise ValueError("random draw must be a bounded three-vector")
    if arm == "independent_3d":
        model = draw.copy()
    elif arm == "exclusive_2d":
        model = draw[:2].copy()
    else:
        raise ValueError("unknown exploration arm")
    if arm == "independent_3d":
        return model, environment_action(model)
    steer, longitudinal = model
    return model, np.array([steer, max(0.0, longitudinal), max(0.0, -longitudinal)],
                           dtype=np.float32)


def state_bins(env) -> tuple[int, int, int, int]:
    hull = env.unwrapped.car.hull
    x, y = float(hull.position[0]), float(hull.position[1])
    vx, vy = float(hull.linearVelocity[0]), float(hull.linearVelocity[1])
    return (math.floor(x / 10), math.floor(y / 10),
            math.floor((hull.angle % (2 * math.pi)) * 8 / (2 * math.pi)),
            min(9, math.floor(math.hypot(vx, vy) / 5)))


def collect(protocol: dict, output: Path, digest: str) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run: {output}")
    output.mkdir()
    (output / "protocol.json").write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n")
    rows = []
    started = time.monotonic()
    with (output / "steps.jsonl").open("x") as steps, (output / "episodes.jsonl").open("x") as episodes:
        for repeat in range(protocol["repeats"]):
            for index, cell in enumerate(protocol["cells"]):
                for arm in ARMS:
                    row = {"arm": arm, "repeat": repeat, **cell, "steps": 0,
                           "return": 0.0, "finished": False, "status": "started"}
                    rows.append(row)
                    episodes.write(json.dumps(row, sort_keys=True) + "\n")
                    episodes.flush()
                    validate(protocol, ROOT / "experiments/tdmpc2-exploration-v2.json")
                    rng = np.random.default_rng(protocol["rng_seed"] + 100 * repeat + index)
                    env = make_training_env(protocol["max_steps"])
                    occupied = set()
                    xy = set()
                    try:
                        obs, _ = env.reset(seed=cell["geometry_seed"],
                                              options={"track_id": cell["track_id"]})
                        if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                                or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
                            raise ValueError("reset did not honor consumed TRAIN cell")
                        for step in range(protocol["max_steps"]):
                            draw = rng.uniform(-1, 1, size=3).astype(np.float32)
                            model, applied = action_for(draw, arm)
                            obs, reward, terminated, truncated, info = env.step(applied)
                            if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                                    or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
                                raise ValueError("TRAIN road changed within episode")
                            if not math.isfinite(float(reward)):
                                raise ValueError("nonfinite raw reward")
                            bins = state_bins(env)
                            occupied.add(bins)
                            xy.add(bins[:2])
                            row.update(steps=step + 1, **{"return": row["return"] + float(reward)},
                                       progress=float(info["progress"]), damage=float(info["damage"]),
                                       finished=bool(info["finished"]))
                            if not all(math.isfinite(row[name]) for name in ("return", "progress", "damage")):
                                raise ValueError("nonfinite environment telemetry")
                            record = {"arm": arm, "repeat": repeat, **cell, "step": step,
                                      "model_action": model.tolist(), "applied_action": applied.tolist(),
                                      "reward": float(reward), "progress": row["progress"],
                                      "damage": row["damage"], "state_bins": bins,
                                      "terminated": bool(terminated), "truncated": bool(truncated)}
                            steps.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                            if terminated or truncated:
                                row.update(status="complete", terminal=bool(terminated or info["finished"]),
                                           censored=bool(truncated and not info["finished"]),
                                           retire_reason=info.get("retire_reason"),
                                           unique_spatial_bins=len(xy), unique_state_bins=len(occupied),
                                           spatial_bins_per_100=100 * len(xy) / (step + 1))
                                break
                        if row["status"] != "complete":
                            raise ValueError("episode reached max_steps without boundary")
                        episodes.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        episodes.flush()
                    except BaseException as exc:
                        row.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
                        episodes.write(json.dumps(row, sort_keys=True) + "\n")
                        episodes.flush()
                        raise
                    finally:
                        env.close()
    summary = {"status": "complete", "protocol_sha256": digest,
               "partition": protocol["partition"], "elapsed_seconds": time.monotonic() - started,
               "episodes": len(rows), "steps_sha256": sha256(output / "steps.jsonl"),
               "episodes_sha256": sha256(output / "episodes.jsonl"), "arms": {}}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summary["arms"][arm] = {"episodes": len(arm_rows),
                                 "decisions": sum(row["steps"] for row in arm_rows),
                                 "finishes": sum(row["finished"] for row in arm_rows),
                                 **{f"mean_{name}": float(np.mean([row[name] for row in arm_rows]))
                                    for name in ("progress", "return", "damage", "steps",
                                                 "unique_spatial_bins", "spatial_bins_per_100")}}
        if summary["arms"][arm]["decisions"] > protocol["max_decisions_per_arm"]:
            raise ValueError("arm exceeded decision budget")
    (output / "result.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/tdmpc2-exploration-v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/tdmpc2-exploration-20260928-v2")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    digest = validate(protocol, args.protocol)
    print(json.dumps({"status": "preflight_only", "protocol_sha256": digest,
                      "environment_resets": 0}) if args.preflight else
          json.dumps(collect(protocol, args.output, digest), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
