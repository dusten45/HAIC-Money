"""Replay two consumed fresh-screen failures with source-bound step telemetry.

This is a diagnostic only. It requires a completed screen summary and every
original screen receipt before starting an episode. It never evaluates or
unlocks the confirmation or blind partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_bare_generalization as frozen_eval
from tools import trace_consumed_screen as prior_trace


PROTOCOL = ROOT / "experiments/observed-ego-side-switch-generalization-v1.json"
SCREEN_RUN = ROOT / ".haic-artifacts/observed-ego-side-switch-generalization-v1/run"
OUTPUT_ROOT = ROOT / ".haic-artifacts/observed-ego-side-switch-screen-telemetry-v1"
MODEL = ROOT / "model.pt"
PROTOCOL_SHA256 = "6d2651440c8edc4d5311f23314492ad7c270455a877e47074f662a2f65767234"
FREEZE_SHA256 = "b29eb69a8eecdaa6e61ddc0b5d4366858b48f42e8b04ae90d8d59d08d2024a20"
HARNESS_SHA256 = "3c44b5be578f659dc2d48930f58af70d65013ffa3fc45f4905b9319cfbb95273"
PRIOR_TRACE_SHA256 = "992ef3c5c0a02e5e0563ed52feb0822ac284d6841b106c884381e0613944db71"
SCREEN_TRACKS = (1, 2, 3, 4)
SCREEN_SEEDS = (2938666309, 533603584, 4089604952, 1162965374,
                1190129265, 583088201, 3876897788, 1180885023)
TARGET_SEEDS = (1190129265, 583088201)
ARMS = ("control", "candidate")
INSTRUMENTED_METHODS = (
    "_road_centers", "_nearest_obstacle", "_adjust_road_steering",
    "_allow_obstacle_side_switch", "_adjust_obstacle_steering",
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_consumed_cell(protocol: dict, track_id: int, seed: int) -> None:
    """Limit replay to the two already inspected fresh-screen regressions."""
    if type(track_id) is not int or type(seed) is not int:
        raise ValueError("track ID and seed must be exact integers")
    partitions = protocol.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"screen", "confirmation", "blind"}:
        raise ValueError("fresh screen grid differs")
    screen = partitions["screen"]
    if (screen.get("track_ids") != list(SCREEN_TRACKS)
            or screen.get("seeds") != list(SCREEN_SEEDS)
            or screen.get("spot_check_cells") != [[1, 2938666309], [4, 1180885023]]):
        raise ValueError("fresh screen grid differs")
    if track_id != 1 or seed not in TARGET_SEEDS:
        raise ValueError("cell is outside the two consumed screen cells")


def require_completed_screen(
    root: Path, freeze: dict, protocol: dict, summary: dict, track_id: int, seed: int,
) -> dict[str, dict]:
    """Check every original receipt and independently recompute the final screen."""
    require_consumed_cell(protocol, track_id, seed)
    if (summary.get("partition") != "screen"
            or summary.get("protocol_sha256") != freeze.get("protocol_sha256")
            or summary.get("canonical_cells") != 32
            or summary.get("missing_cells") != []):
        raise ValueError("completed fresh screen summary is required")
    for cell_track, cell_seed, repeat in frozen_eval.expected_cells(protocol, "screen"):
        for arm in ARMS:
            row = frozen_eval.load_cell(root, freeze, "screen", arm, cell_track, cell_seed, repeat)
            if row is None or row.get("error") is not None:
                raise ValueError("all completed screen receipts are required")
    if frozen_eval.report_phase(root, freeze, protocol, "screen") != summary:
        raise ValueError("screen summary differs from completed screen receipts")
    return {
        arm: frozen_eval.load_cell(root, freeze, "screen", arm, track_id, seed, 0)
        for arm in ARMS
    }


def _environment_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "core").rglob("*.py"))
    paths += sorted((ROOT / "local_simulator").rglob("*.py"))
    paths += [ROOT / "env_wrapper.py", ROOT / "damage.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): frozen_eval.digest(path)
            for path in paths}


def preflight(arm: str, track_id: int, seed: int) -> tuple[dict, dict]:
    if arm not in ARMS:
        raise ValueError("unknown frozen Agent arm")
    for path, expected in (
        (PROTOCOL, PROTOCOL_SHA256),
        (SCREEN_RUN / "freeze.json", FREEZE_SHA256),
        (ROOT / "tools/evaluate_bare_generalization.py", HARNESS_SHA256),
        (ROOT / "tools/trace_consumed_screen.py", PRIOR_TRACE_SHA256),
    ):
        prior_trace._require_hash(path, expected)
    if not (SCREEN_RUN / "screen-summary.json").is_file():
        raise ValueError("completed fresh screen summary is required")
    protocol = _json(PROTOCOL)
    freeze = _json(SCREEN_RUN / "freeze.json")
    summary_path = SCREEN_RUN / "screen-summary.json"
    summary = _json(summary_path)
    require_consumed_cell(protocol, track_id, seed)
    if (freeze.get("protocol_sha256") != PROTOCOL_SHA256
            or freeze.get("harness_sha256") != HARNESS_SHA256
            or freeze.get("source_sha256") != {
                name: protocol[f"{name}_agent_sha256"] for name in ARMS
            }
            or freeze.get("model_sha256") != protocol.get("model_sha256")
            or freeze.get("helper_sha256") != protocol.get("runtime_helper_sha256")):
        raise ValueError("fresh screen freeze lineage mismatch")
    sources = {}
    artifact_root = (ROOT / ".haic-artifacts").resolve()
    for name in ARMS:
        path = (ROOT / protocol["source_snapshots"][name]).resolve()
        if not path.is_relative_to(artifact_root):
            raise ValueError("frozen source must remain inside .haic-artifacts")
        prior_trace._require_hash(path, freeze["source_sha256"][name])
        sources[name] = path
    prior_trace._require_hash(MODEL, freeze["model_sha256"])
    for path, digest in freeze["helper_sha256"].items():
        prior_trace._require_hash(ROOT / path, digest)
    environment = _environment_hashes()
    if environment != freeze.get("environment_sha256"):
        raise ValueError("official-generator environment differs from fresh screen")
    references = require_completed_screen(SCREEN_RUN, freeze, protocol, summary, track_id, seed)
    identity = {
        "protocol_sha256": PROTOCOL_SHA256,
        "freeze_sha256": FREEZE_SHA256,
        "screen_summary_sha256": frozen_eval.digest(summary_path),
        "harness_sha256": HARNESS_SHA256,
        "prior_trace_sha256": PRIOR_TRACE_SHA256,
        "telemetry_tool_sha256": frozen_eval.digest(Path(__file__)),
        "source_sha256": freeze["source_sha256"],
        "model_sha256": freeze["model_sha256"],
        "helper_sha256": freeze["helper_sha256"],
        "environment_sha256": environment,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    return identity, {"source": sources[arm], "model": MODEL,
                      "class": protocol["controller_classes"][arm],
                      "reference": references[arm]}


def matched_obstacle_box(frame: np.ndarray, obstacle: Any, low: float) -> dict | None:
    """Locate the bright connected component that produced the detector center."""
    if obstacle is None:
        return None
    bright = np.asarray(frame >= low, dtype=np.bool_)
    bright[:22, :] = False
    bright[62:, :] = False
    visited = np.zeros(bright.shape, dtype=np.bool_)
    target_y, target_x = float(obstacle[0]), float(obstacle[1])
    width = bright.shape[1]
    for start_y in range(22, 62):
        for start_x in range(width):
            if not bright[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            sum_x = sum_y = area = 0
            while stack:
                x, y = stack.pop()
                area += 1
                sum_x += x
                sum_y += y
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for neighbor_y in range(max(22, y - 1), min(62, y + 2)):
                    for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                        if bright[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                            visited[neighbor_y, neighbor_x] = True
                            stack.append((neighbor_x, neighbor_y))
            box_width = max_x - min_x + 1
            box_height = max_y - min_y + 1
            if not (4 <= area <= 80 and 2 <= box_width <= 9 and 2 <= box_height <= 10):
                continue
            if abs(sum_x / area - target_x) < 1e-9 and abs(sum_y / area - target_y) < 1e-9:
                return {"bbox_inclusive": [min_x, min_y, max_x, max_y], "area": area}
    raise RuntimeError("detected obstacle has no matching visible component")


def act_with_diagnostics(agent: Any, observation: Any) -> tuple[np.ndarray, dict]:
    """Capture actual detector and steering calls without making an extra policy call."""
    controller = agent._forward_controller
    originals = {name: getattr(controller, name) for name in INSTRUMENTED_METHODS}
    instance_prior = {name: (name in vars(controller), vars(controller).get(name))
                      for name in INSTRUMENTED_METHODS}
    detail: dict[str, Any] = {
        "road_centers": None, "obstacle": None, "road_steering": None,
        "side_switch": None, "obstacle_steering": None,
    }

    def road_centers(*args, **kwargs):
        result = originals["_road_centers"](*args, **kwargs)
        detail["road_centers"] = {str(row): float(center) for row, center in result.items()}
        return result

    def nearest_obstacle(*args, **kwargs):
        result = originals["_nearest_obstacle"](*args, **kwargs)
        detail["obstacle"] = None if result is None else [float(value) for value in result]
        return result

    def road_steering(*args, **kwargs):
        result = originals["_adjust_road_steering"](*args, **kwargs)
        detail["road_steering"] = {
            "input": float(kwargs["steering"]), "straight": bool(kwargs["straight"]),
            "output": float(result),
        }
        return result

    def side_switch(*args, **kwargs):
        result = originals["_allow_obstacle_side_switch"](*args, **kwargs)
        detail["side_switch"] = {
            "obstacle_y": float(kwargs["obstacle_y"]),
            "obstacle_x": float(kwargs["obstacle_x"]),
            "candidate_side": float(kwargs["candidate_side"]),
            "allowed": bool(result),
        }
        return result

    def obstacle_steering(*args, **kwargs):
        result = originals["_adjust_obstacle_steering"](*args, **kwargs)
        detail["obstacle_steering"] = {
            "base": float(kwargs["base_steering"]),
            "bias": float(kwargs["obstacle_bias"]),
            "straight": bool(kwargs["straight"]), "output": float(result),
        }
        return result

    wrappers = {
        "_road_centers": road_centers,
        "_nearest_obstacle": nearest_obstacle,
        "_adjust_road_steering": road_steering,
        "_allow_obstacle_side_switch": side_switch,
        "_adjust_obstacle_steering": obstacle_steering,
    }
    try:
        for name, wrapper in wrappers.items():
            setattr(controller, name, wrapper)
        action = frozen_eval.validate_action(agent.act(observation))
    finally:
        for name in INSTRUMENTED_METHODS:
            had_instance, prior = instance_prior[name]
            prior_trace._restore_instance_attribute(controller, name, had_instance, prior)
    obstacle = detail["obstacle"]
    detail["obstacle_box"] = matched_obstacle_box(
        np.asarray(observation)[-1], obstacle, float(controller.OBSTACLE_LOW),
    )
    return action, detail


def _controller_detail(controller: Any) -> dict:
    names = (
        "_last_steer", "_pace_latched_target", "_pace_command_target",
        "_pace_sweep", "_centerline_override_side", "_observed_far_center",
        "_observed_bend_displacement", "_carry_steer_request",
    )
    return {name: None if getattr(controller, name, None) is None
            else float(getattr(controller, name)) for name in names}


def _episode(arm: str, track_id: int, seed: int, inputs: dict) -> dict:
    from local_simulator.schema import MapSpec
    from local_simulator.session import SimulationSession

    random.seed(0)
    np.random.seed(0)
    started = time.perf_counter()
    agent = frozen_eval._load_agent(inputs["source"], inputs["model"], inputs["class"])
    initialization_ms = (time.perf_counter() - started) * 1000
    if initialization_ms > 10000:
        raise RuntimeError("Agent initialization exceeded 10 seconds")

    class Policy:
        name = "actual-Agent-route-fresh-screen-telemetry"
        reset_ms = 0.0

        def reset(self, observation):
            began = time.perf_counter()
            agent.reset(observation)
            self.reset_ms = (time.perf_counter() - began) * 1000
            if self.reset_ms > 5000:
                raise RuntimeError("Agent.reset exceeded 5 seconds")

        def act(self, observation):
            raise AssertionError("telemetry must explicitly call Agent.act")

    policy = Policy()
    session = SimulationSession.start(MapSpec(track_id, seed, "official", (), 2000, 4), policy)
    rows: list[dict] = []
    action_hash = hashlib.sha256()
    try:
        raw = session.raw_environment
        track_tiles = len(raw.track)
        if track_tiles <= 0:
            raise ValueError("official-generator track has no tiles")
        while not session.done and len(session.steps) < 2000:
            observation = session.observation
            observation_sha256 = hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()
            tiles_before = int(raw.tile_visited_count)
            side_before = float(agent._forward_controller._obstacle_side)
            began = time.perf_counter()
            action, detail = act_with_diagnostics(agent, observation)
            act_ms = (time.perf_counter() - began) * 1000
            if act_ms > 5000:
                raise RuntimeError("Agent.act exceeded 5 seconds")
            state = prior_trace._controller_state(agent._forward_controller)
            extra_state = _controller_detail(agent._forward_controller)
            action_hash.update(action.tobytes())
            step, reward = prior_trace.step_with_reward(session, action)
            tiles_after = int(raw.tile_visited_count)
            wheel_on_road = [bool(wheel.tiles) for wheel in raw.car.wheels]
            if len(wheel_on_road) != 4:
                raise RuntimeError("official car no longer has four wheels")
            row = prior_trace.make_step_row(
                step=step, observation_sha256=observation_sha256, act_ms=act_ms,
                obstacle=detail["obstacle"], side_before=side_before, controller_state=state,
                tiles_before=tiles_before, tiles_after=tiles_after, track_tiles=track_tiles,
                reward=reward, wheel_on_road=wheel_on_road,
                offtrack_counter=int(session.environment.off_track_counter),
            )
            row["decision_detail"] = detail
            row["controller_detail"] = extra_state
            rows.append(row)
        summary = session.finish().summary
    finally:
        session.close()
    episode = {"arm": arm, "track_id": track_id, "seed": seed,
               "initialization_ms": initialization_ms, "reset_ms": policy.reset_ms,
               "summary": summary, "action_trace_sha256": action_hash.hexdigest(),
               "steps": rows}
    reference = inputs["reference"]
    keys = ("finished", "progress", "lap_time_ms", "collision_count", "damage", "retire_reason")
    mismatches = [key for key in keys if summary[key] != reference[key]]
    if len(rows) != reference["steps"] or episode["action_trace_sha256"] != reference["action_trace_sha256"]:
        mismatches.append("action trace")
    if sum(int(row["wheels_on_road"] == 0) for row in rows) != reference["offtrack_samples"]:
        mismatches.append("offtrack samples")
    if sum(int(row["wheels_on_road"] < 4) for row in rows) != reference["partial_offtrack_samples"]:
        mismatches.append("partial offtrack samples")
    if mismatches:
        raise RuntimeError(f"telemetry replay differs from frozen fresh-screen receipt: {mismatches}")
    return episode


def _artifact_path(arm: str, track_id: int, seed: int) -> Path:
    return OUTPUT_ROOT / f"track-{track_id}-seed-{seed}" / f"{arm}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    identity, inputs = preflight(args.arm, args.track_id, args.seed)
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "arm": args.arm, "track_id": args.track_id,
                          "seed": args.seed, "source_sha256": identity["source_sha256"][args.arm],
                          "screen_receipts": "PASS"}, sort_keys=True))
        return 0
    path = _artifact_path(args.arm, args.track_id, args.seed)
    if path.exists():
        envelope = _json(path)
        payload = {"identity": identity, "episode": envelope.get("episode")}
        if (envelope.get("identity") != identity
                or envelope.get("digest") != hashlib.sha256(prior_trace._canonical(payload)).hexdigest()):
            raise ValueError(f"existing telemetry artifact has different identity or content: {path}")
        print(json.dumps({"artifact": str(path), "status": "EXISTING", "summary": envelope["episode"]["summary"]}))
        return 0
    episode = _episode(args.arm, args.track_id, args.seed, inputs)
    after_identity, _ = preflight(args.arm, args.track_id, args.seed)
    if after_identity != identity:
        raise RuntimeError("frozen input identity changed during telemetry")
    payload = {"identity": identity, "episode": episode}
    envelope = {**payload, "digest": hashlib.sha256(prior_trace._canonical(payload)).hexdigest()}
    frozen_eval._atomic_json(path, envelope)
    print(json.dumps({"artifact": str(path), "status": "RECORDED", "summary": episode["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
