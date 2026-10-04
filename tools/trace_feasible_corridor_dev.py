"""Source-bound telemetry for the fixed, consumed corridor mechanism cells.

Each invocation replays one actual Agent route. It requires the complete
development triage, both immutable target receipts, and the original frozen
source/environment lineage before starting an official-generator episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import compare_feasible_corridor_dev as compare
from tools import compare_feasible_corridor_dev_v2 as compare_v2
from tools import compare_reused_bare_candidate as prior_compare
from tools import evaluate_bare_generalization as fresh
from tools import trace_consumed_screen as prior_trace
from tools import trace_ego_screen_regressions as ego_trace


PROTOCOL = ROOT / "experiments/feasible-corridor-dev-v1.json"
RUN_ROOT = ROOT / ".haic-artifacts/feasible-corridor-dev-v1/run"
OUTPUT_ROOT = ROOT / ".haic-artifacts/feasible-corridor-dev-telemetry-v1"
MODEL = ROOT / "model.pt"
V2_PROTOCOL = ROOT / "experiments/feasible-corridor-dev-v2.json"
V2_RUN_ROOT = ROOT / ".haic-artifacts/feasible-corridor-dev-v2/run"
V2_OUTPUT_ROOT = ROOT / ".haic-artifacts/feasible-corridor-dev-telemetry-v2"
TARGETS = frozenset(compare.MECHANISM_CELLS)
SUMMARY_KEYS = (
    "finished", "progress", "lap_time_ms", "collision_count", "damage",
    "retire_reason",
)


def require_consumed_cell(protocol: dict, track_id: int, seed: int) -> None:
    if type(track_id) is not int or type(seed) is not int:
        raise ValueError("track ID and seed must be exact integers")
    if protocol.get("mechanism_cells") != [list(cell) for cell in compare.MECHANISM_CELLS]:
        raise ValueError("frozen mechanism grid differs")
    if protocol.get("development_cells") is None or [track_id, seed] not in protocol["development_cells"]:
        raise ValueError("cell is outside consumed development")
    if (track_id, seed) not in TARGETS:
        raise ValueError("cell is outside fixed consumed mechanism cells")


def study_config(version: int) -> tuple[Path, Path, Path, Any]:
    if version == 1:
        return PROTOCOL, RUN_ROOT, OUTPUT_ROOT, compare
    if version == 2:
        return V2_PROTOCOL, V2_RUN_ROOT, V2_OUTPUT_ROOT, compare_v2
    raise ValueError("unknown corridor study version")


def require_frozen_receipts(
    root: Path, identity: dict, track_id: int, seed: int,
    ego_cells: tuple[tuple[int, int], ...] = (),
) -> tuple[dict[str, dict], dict[str, str]]:
    rows: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for arm in compare.ARMS:
        row = compare.load_cell(root, identity, arm, track_id, seed)
        if row is None or row.get("error") is not None:
            raise ValueError("both completed development receipts are required")
        origin = ("EGO_SCREEN_CONTROL" if arm == "baseline" and (track_id, seed) in ego_cells
                  else "COLD_WORKER")
        if row.get("receipt_origin") != origin:
            raise ValueError("receipt does not match its original cold worker or frozen screen")
        fresh._sha256_string(row.get("action_trace_sha256"), "action_trace_sha256")
        rows[arm] = row
        hashes[arm] = fresh.digest(compare.cell_path(root, arm, track_id, seed))
    return rows, hashes


def _paths(protocol: dict, protocol_path: Path = PROTOCOL) -> dict[str, Path]:
    artifact_root = (ROOT / ".haic-artifacts").resolve()
    sources = {}
    for arm in compare.ARMS:
        source = (ROOT / protocol["source_snapshots"][arm]).resolve()
        if not source.is_relative_to(artifact_root):
            raise ValueError("Agent source snapshot must be in .haic-artifacts")
        sources[arm] = source
    return {
        "protocol": protocol_path, "model": MODEL,
        "baseline": sources["baseline"], "candidate": sources["candidate"],
        "margin_screen_protocol": compare.MARGIN_SCREEN_PROTOCOL,
        "ego_screen_protocol": compare.EGO_SCREEN_PROTOCOL,
    }


def preflight(arm: str, track_id: int, seed: int, *, version: int = 1) -> tuple[dict, dict]:
    if arm not in compare.ARMS:
        raise ValueError("unknown frozen Agent arm")
    protocol_path, run_root, _, comparator = study_config(version)
    protocol = fresh._read_json(protocol_path)
    require_consumed_cell(protocol, track_id, seed)
    margin = fresh._read_json(compare.MARGIN_SCREEN_PROTOCOL)
    ego = fresh._read_json(compare.EGO_SCREEN_PROTOCOL)
    comparator.validate_protocol(
        protocol, margin, ego, fresh.digest(compare.MARGIN_SCREEN_PROTOCOL),
        fresh.digest(compare.EGO_SCREEN_PROTOCOL),
        prior_compare.sealed_holdout_seeds(ROOT / "experiments"),
    )
    paths = _paths(protocol, protocol_path)
    freeze_path = run_root / "freeze.json"
    summary_path = run_root / "mechanism-summary.json"
    if not freeze_path.is_file() or not summary_path.is_file():
        raise ValueError("complete frozen mechanism triage is required")
    freeze = fresh._read_json(freeze_path)
    identity = comparator.build_identity(protocol_path, protocol, paths, margin, ego)
    if freeze != identity:
        raise ValueError("development freeze differs from frozen source/environment")
    comparator.check_frozen_inputs(identity, paths, {"margin": margin, "ego": ego})
    ego_cells = compare.ego_screen_cells(ego)
    for cell_track, cell_seed in compare.MECHANISM_CELLS:
        require_frozen_receipts(run_root, identity, cell_track, cell_seed,
                                ego_cells)
    summary = comparator.report(run_root, identity, compare.MECHANISM_CELLS,
                                mechanism_only=True, ego_cells=ego_cells)
    if fresh._read_json(summary_path) != summary:
        raise ValueError("frozen mechanism summary differs from its 20 receipts")
    rows, receipt_hashes = require_frozen_receipts(
        run_root, identity, track_id, seed, ego_cells,
    )
    if (track_id, seed) in ego_cells:
        original_path = fresh.cell_path(compare.EGO_SCREEN_RUN, "screen", "control",
                                        track_id, seed, 0)
        if rows["baseline"].get("source_receipt_sha256") != fresh.digest(original_path):
            raise ValueError("reused control receipt no longer matches the frozen screen")
    diagnostic_identity = {
        **identity,
        "development_freeze_sha256": fresh.digest(freeze_path),
        "mechanism_summary_sha256": fresh.digest(summary_path),
        "target_receipt_sha256": receipt_hashes,
        "prior_trace_sha256": fresh.digest(Path(prior_trace.__file__)),
        "ego_trace_sha256": fresh.digest(Path(ego_trace.__file__)),
        "telemetry_tool_sha256": fresh.digest(Path(__file__)),
    }
    if version == 2:
        diagnostic_identity["study_version"] = version
    return diagnostic_identity, {
        "source": paths[arm], "model": MODEL,
        "class": protocol["controller_classes"][arm],
        "reference": rows[arm],
    }


def _serialize_plan(plan: Any) -> dict | None:
    if plan is None:
        return None
    return {
        "side": float(plan["side"]),
        "target_x": float(plan["target_x"]),
        "clearance_px": float(plan["clearance_px"]),
        "bbox": list(plan["bbox"]),
        "control_waypoints": [[float(y), float(x)] for y, x in plan["control_waypoints"]],
        "edges": {str(y): [float(left), float(right)]
                  for y, (left, right) in plan["edges"].items()},
    }


def act_with_diagnostics(agent: Any, observation: Any) -> tuple[np.ndarray, dict]:
    """Observe the one real Agent.act call; restore instance methods afterward."""
    controller = agent._forward_controller
    names = ["_road_centers", "_nearest_obstacle", "_adjust_road_steering",
             "_adjust_obstacle_steering"]
    candidate = hasattr(controller, "_corridor_candidate")
    if candidate:
        names.append("_corridor_candidate")
    originals = {name: getattr(controller, name) for name in names}
    prior = {name: (name in vars(controller), vars(controller).get(name)) for name in names}
    obstacles: list[Any] = []
    corridors: list[dict] = []
    centers_detail: dict | None = None
    decision_centers_detail: dict | None = None
    road_detail: dict | None = None
    steering_detail: dict | None = None
    speed_detail: dict | None = None
    if hasattr(controller, "_adjust_target_speed_for_steering"):
        names.append("_adjust_target_speed_for_steering")
        originals["_adjust_target_speed_for_steering"] = getattr(
            controller, "_adjust_target_speed_for_steering")
        prior["_adjust_target_speed_for_steering"] = (
            "_adjust_target_speed_for_steering" in vars(controller),
            vars(controller).get("_adjust_target_speed_for_steering"),
        )

    def centers_call(*args, **kwargs):
        nonlocal centers_detail
        result = originals["_road_centers"](*args, **kwargs)
        centers_detail = {str(row): float(x) for row, x in result.items()}
        return result

    def obstacle_call(*args, **kwargs):
        result = originals["_nearest_obstacle"](*args, **kwargs)
        obstacles.append(result)
        return result

    def corridor_call(side):
        result = originals["_corridor_candidate"](side)
        corridors.append({
            "side": float(side),
            "target_x": None if result is None else float(result[0]),
            "clearance_px": None if result is None else float(result[1]),
            "near_y": None if result is None else float(result[2]),
        })
        return result

    def road_call(*args, **kwargs):
        nonlocal road_detail, decision_centers_detail
        result = originals["_adjust_road_steering"](*args, **kwargs)
        # _nearest_obstacle may inspect earlier stack frames through
        # _road_centers. The centers passed to this decision are authoritative.
        decision_centers_detail = {
            str(row): float(x) for row, x in kwargs["centers"].items()
        }
        road_detail = {
            "input": float(kwargs["steering"]),
            "straight": bool(kwargs["straight"]),
            "output": float(result),
        }
        return result

    def steering_call(*args, **kwargs):
        nonlocal steering_detail
        result = originals["_adjust_obstacle_steering"](*args, **kwargs)
        steering_detail = {
            "base": float(kwargs["base_steering"]),
            "bias": float(kwargs["obstacle_bias"]),
            "straight": bool(kwargs["straight"]),
            "output": float(result),
        }
        return result

    def speed_call(*args, **kwargs):
        nonlocal speed_detail
        result = originals["_adjust_target_speed_for_steering"](*args, **kwargs)
        speed_detail = {
            "input": float(kwargs["target_speed"]),
            "output": float(result),
            "corridor_plan_present": getattr(controller, "_corridor_plan", None) is not None,
        }
        return result

    try:
        controller._road_centers = centers_call
        controller._nearest_obstacle = obstacle_call
        controller._adjust_road_steering = road_call
        controller._adjust_obstacle_steering = steering_call
        if "_adjust_target_speed_for_steering" in originals:
            controller._adjust_target_speed_for_steering = speed_call
        if candidate:
            controller._corridor_candidate = corridor_call
        action = fresh.validate_action(agent.act(observation))
    finally:
        for name in names:
            had_instance, value = prior[name]
            prior_trace._restore_instance_attribute(controller, name, had_instance, value)
    detected = obstacles[-1] if obstacles else None
    bbox = getattr(controller, "_corridor_bbox", None) if candidate else None
    if bbox is None and detected is not None:
        bbox_record = ego_trace.matched_obstacle_box(
            np.asarray(observation)[-1], detected, float(controller.OBSTACLE_LOW),
        )
        if bbox_record is not None:
            bbox = bbox_record["bbox_inclusive"]
    detail = {
        "obstacle": None if detected is None else [float(value) for value in detected],
        "obstacle_bbox": None if bbox is None else list(bbox),
        "corridor_candidates": corridors if candidate else None,
        "corridor_plan": _serialize_plan(getattr(controller, "_corridor_plan", None)),
        "corridor_edges": None if not candidate else {
            str(row): [float(left), float(right)]
            for row, (left, right) in controller._corridor_edges.items()
        },
        "corridor_previous": None if getattr(controller, "_corridor_previous", None) is None
        else [float(value) for value in controller._corridor_previous],
        "corridor_misses": None if not candidate else int(controller._corridor_misses),
        "road_centers": (decision_centers_detail if decision_centers_detail is not None
                         else centers_detail),
        "road_sweep": None if getattr(controller, "_pace_sweep", None) is None
        else float(controller._pace_sweep),
        "bend_displacement": None if getattr(controller, "_observed_bend_displacement", None) is None
        else float(controller._observed_bend_displacement),
        "far_center": None if getattr(controller, "_observed_far_center", None) is None
        else float(controller._observed_far_center),
        "road_steering": road_detail,
        "obstacle_steering": steering_detail,
        "steering_speed_target": speed_detail,
    }
    return action, detail


def verify_exact_replay(episode: dict, receipt: dict) -> None:
    summary = episode["summary"]
    mismatches = [key for key in SUMMARY_KEYS if summary.get(key) != receipt.get(key)]
    rows = episode["steps"]
    if (len(rows) != receipt.get("steps")
            or episode["action_trace_sha256"] != receipt.get("action_trace_sha256")):
        mismatches.append("action trace")
    if sum(row["wheels_on_road"] == 0 for row in rows) != receipt.get("offtrack_samples"):
        mismatches.append("offtrack samples")
    if sum(row["wheels_on_road"] < 4 for row in rows) != receipt.get("partial_offtrack_samples"):
        mismatches.append("partial offtrack samples")
    if mismatches:
        raise RuntimeError(f"telemetry replay differs from frozen development receipt: {mismatches}")


def _episode(arm: str, track_id: int, seed: int, inputs: dict) -> dict:
    from local_simulator.schema import MapSpec
    from local_simulator.session import SimulationSession

    random.seed(0)
    np.random.seed(0)
    started = time.perf_counter()
    agent = fresh._load_agent(inputs["source"], inputs["model"], inputs["class"])
    init_ms = (time.perf_counter() - started) * 1000.0
    if init_ms > 10000:
        raise RuntimeError("Agent initialization exceeded 10 seconds")

    class Policy:
        name = "actual-Agent-route-consumed-corridor-telemetry"
        reset_ms = 0.0

        def reset(self, observation):
            began = time.perf_counter()
            agent.reset(observation)
            self.reset_ms = (time.perf_counter() - began) * 1000.0
            if self.reset_ms > 5000:
                raise RuntimeError("Agent.reset exceeded 5 seconds")

        def act(self, observation):
            raise AssertionError("telemetry calls Agent.act explicitly")

    policy = Policy()
    session = SimulationSession.start(MapSpec(track_id, seed, "official", (), 2000, 4), policy)
    rows: list[dict] = []
    actions = hashlib.sha256()
    try:
        raw = session.raw_environment
        total_tiles = len(raw.track)
        if total_tiles <= 0:
            raise ValueError("official-generator track has no tiles")
        while not session.done and len(session.steps) < 2000:
            observation = session.observation
            observation_hash = hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()
            tiles_before = int(raw.tile_visited_count)
            side_before = float(agent._forward_controller._obstacle_side)
            velocity_before = raw.car.hull.linearVelocity
            speed_before = math.hypot(float(velocity_before[0]), float(velocity_before[1]))
            began = time.perf_counter()
            action, detail = act_with_diagnostics(agent, observation)
            act_ms = (time.perf_counter() - began) * 1000.0
            if act_ms > 5000:
                raise RuntimeError("Agent.act exceeded 5 seconds")
            state = prior_trace._controller_state(agent._forward_controller)
            actions.update(action.tobytes())
            step, reward = prior_trace.step_with_reward(session, action)
            wheel_on_road = [bool(wheel.tiles) for wheel in raw.car.wheels]
            if len(wheel_on_road) != 4:
                raise RuntimeError("official car no longer has four wheels")
            row = prior_trace.make_step_row(
                step=step, observation_sha256=observation_hash, act_ms=act_ms,
                obstacle=detail["obstacle"], side_before=side_before,
                controller_state=state, tiles_before=tiles_before,
                tiles_after=int(raw.tile_visited_count), track_tiles=total_tiles,
                reward=reward, wheel_on_road=wheel_on_road,
                offtrack_counter=int(session.environment.off_track_counter),
            )
            row["speed_world_before"] = speed_before
            row["decision_detail"] = detail
            rows.append(row)
        summary = session.finish().summary
    finally:
        session.close()
    episode = {
        "arm": arm, "track_id": track_id, "seed": seed,
        "initialization_ms": init_ms, "reset_ms": policy.reset_ms,
        "summary": summary, "action_trace_sha256": actions.hexdigest(),
        "steps": rows,
    }
    verify_exact_replay(episode, inputs["reference"])
    return episode


def _artifact_path(arm: str, track_id: int, seed: int, *, version: int = 1,
                   artifact_tag: str | None = None) -> Path:
    if artifact_tag is not None and re.fullmatch(r"[A-Za-z0-9_-]{1,32}", artifact_tag) is None:
        raise ValueError("artifact tag must be a short filename token")
    suffix = "" if artifact_tag is None else f".{artifact_tag}"
    return study_config(version)[2] / f"track-{track_id}-seed-{seed}" / f"{arm}{suffix}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=compare.ARMS)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--study-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--artifact-tag", type=str)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    identity, inputs = preflight(args.arm, args.track_id, args.seed,
                                 version=args.study_version)
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "arm": args.arm,
                          "cell": [args.track_id, args.seed],
                          "source_sha256": identity["source_sha256"][args.arm],
                          "mechanism_receipts": "PASS"}, sort_keys=True))
        return 0
    path = _artifact_path(args.arm, args.track_id, args.seed,
                          version=args.study_version, artifact_tag=args.artifact_tag)
    with fresh._run_lock(study_config(args.study_version)[1]):
        if path.exists():
            envelope = fresh._read_json(path)
            payload = {"identity": identity, "episode": envelope.get("episode")}
            if (envelope.get("identity") != identity
                    or envelope.get("digest") != hashlib.sha256(fresh._canonical(payload)).hexdigest()):
                raise ValueError(f"existing telemetry artifact differs: {path}")
            print(json.dumps({"artifact": str(path), "status": "EXISTING",
                              "summary": envelope["episode"]["summary"]}))
            return 0
        episode = _episode(args.arm, args.track_id, args.seed, inputs)
        after, _ = preflight(args.arm, args.track_id, args.seed,
                             version=args.study_version)
        if after != identity:
            raise RuntimeError("frozen inputs changed during telemetry replay")
        payload = {"identity": identity, "episode": episode}
        envelope = {**payload, "digest": hashlib.sha256(fresh._canonical(payload)).hexdigest()}
        fresh._atomic_json(path, envelope)
    print(json.dumps({"artifact": str(path), "status": "RECORDED",
                      "summary": episode["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
