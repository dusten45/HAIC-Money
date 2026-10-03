"""Source-bound telemetry for already consumed observed-margin screen cells.

Invoke one cell and one frozen actual-Agent route per process. This diagnostic
does not evaluate a new candidate or unlock confirmation/blind. It refuses any
cell outside the 16 completed screen cells, and requires the original screen
receipts before an episode can start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_bare_generalization as frozen_eval


SCREEN_PROTOCOL = ROOT / "experiments/observed-margin-generalization-v1.json"
DEV_PROTOCOL = ROOT / "experiments/observed-ego-side-switch-dev-v1.json"
SCREEN_RUN = ROOT / ".haic-artifacts/observed-margin-generalization-v1/run"
OUTPUT_ROOT = ROOT / ".haic-artifacts/consumed-screen-telemetry-v1"
MODEL = ROOT / "model.pt"
SCREEN_PROTOCOL_SHA256 = "dfddf7f28686501f9c35298e33cc3e5ec28c53e61e2e3dfaeae2262a840d1632"
DEV_PROTOCOL_SHA256 = "eadf0ebdd5a52f73c8bb6cc75e6c668127299896b6c98a640d4387df07714b3c"
SCREEN_FREEZE_SHA256 = "0d115acf9d331afef28069e2f42cc17b95bcd4b9bdd317bf5ed81fa7af1aabf9"
SCREEN_SUMMARY_SHA256 = "9ae6180d3407044c01e188cf8f09b27b8532a56f13341779dee3be969f2b3acc"
SCREEN_HARNESS_SHA256 = "3c44b5be578f659dc2d48930f58af70d65013ffa3fc45f4905b9319cfbb95273"
SCREEN_TRACKS = (1, 2, 3, 4)
SCREEN_SEEDS = (3857792434, 2951861974, 2786341258, 3892761381)
ARMS = ("baseline", "margin", "ego_switch")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    return frozen_eval.digest(path)


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"frozen input hash mismatch: {path}")


def require_consumed_screen_cell(protocol: dict, track_id: int, seed: int) -> None:
    """Reject new geometry, sealed partitions, and edited screen grids."""
    if type(track_id) is not int or type(seed) is not int:
        raise ValueError("track ID and seed must be exact integers")
    partitions = protocol.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"screen", "confirmation", "blind"}:
        raise ValueError("frozen screen grid is missing")
    screen = partitions["screen"]
    if screen.get("track_ids") != list(SCREEN_TRACKS) or screen.get("seeds") != list(SCREEN_SEEDS):
        raise ValueError("frozen screen grid differs")
    sealed = set(partitions["confirmation"].get("seeds", [])) | set(partitions["blind"].get("seeds", []))
    if sealed.intersection(SCREEN_SEEDS):
        raise ValueError("frozen screen grid overlaps sealed geometry")
    if track_id not in SCREEN_TRACKS or seed not in SCREEN_SEEDS:
        raise ValueError("cell is outside the consumed screen grid")


def require_consumed_receipts(screen_root: Path, freeze: dict, track_id: int, seed: int) -> dict[str, dict]:
    receipts = {
        "baseline": frozen_eval.load_cell(screen_root, freeze, "screen", "control", track_id, seed, 0),
        "margin": frozen_eval.load_cell(screen_root, freeze, "screen", "candidate", track_id, seed, 0),
    }
    if any(row is None or row.get("error") is not None for row in receipts.values()):
        raise ValueError("both completed original screen receipts are required")
    return receipts


def _source_paths(dev_protocol: dict) -> dict[str, Path]:
    paths = {arm: (ROOT / dev_protocol["source_snapshots"][arm]).resolve() for arm in ARMS}
    artifact_root = (ROOT / ".haic-artifacts").resolve()
    for arm, path in paths.items():
        if not path.is_relative_to(artifact_root):
            raise ValueError(f"{arm} source snapshot must stay inside .haic-artifacts")
    return paths


def preflight(arm: str, track_id: int, seed: int) -> tuple[dict, dict]:
    """Bind every executable input and demand both original screen receipts."""
    if arm not in ARMS:
        raise ValueError("unknown frozen Agent arm")
    for path, expected in (
        (SCREEN_PROTOCOL, SCREEN_PROTOCOL_SHA256),
        (DEV_PROTOCOL, DEV_PROTOCOL_SHA256),
        (SCREEN_RUN / "freeze.json", SCREEN_FREEZE_SHA256),
        (SCREEN_RUN / "screen-summary.json", SCREEN_SUMMARY_SHA256),
        (ROOT / "tools/evaluate_bare_generalization.py", SCREEN_HARNESS_SHA256),
    ):
        _require_hash(path, expected)
    screen_protocol = _json(SCREEN_PROTOCOL)
    dev_protocol = _json(DEV_PROTOCOL)
    freeze = _json(SCREEN_RUN / "freeze.json")
    summary = _json(SCREEN_RUN / "screen-summary.json")
    require_consumed_screen_cell(screen_protocol, track_id, seed)
    if (summary.get("decision") != "REJECT" or summary.get("canonical_cells") != 16
            or summary.get("missing_cells") != []):
        raise ValueError("original screen is not a complete rejected evaluation")
    if dev_protocol.get("screen_protocol_sha256") != SCREEN_PROTOCOL_SHA256:
        raise ValueError("development protocol has the wrong screen lineage")
    if freeze.get("protocol_sha256") != SCREEN_PROTOCOL_SHA256 or freeze.get("harness_sha256") != SCREEN_HARNESS_SHA256:
        raise ValueError("screen freeze lineage mismatch")
    if summary.get("protocol_sha256") != SCREEN_PROTOCOL_SHA256:
        raise ValueError("screen summary lineage mismatch")
    for name in ("control", "candidate"):
        expected = screen_protocol[f"{name}_agent_sha256"]
        if freeze["source_sha256"].get(name) != expected:
            raise ValueError("screen source lineage mismatch")
    paths = _source_paths(dev_protocol)
    for arm_name, path in paths.items():
        _require_hash(path, dev_protocol["agent_sha256"][arm_name])
    if (dev_protocol["agent_sha256"]["baseline"] != freeze["source_sha256"]["control"]
            or dev_protocol["agent_sha256"]["margin"] != freeze["source_sha256"]["candidate"]):
        raise ValueError("development sources differ from the screen sources")
    _require_hash(MODEL, dev_protocol["model_sha256"])
    if freeze.get("model_sha256") != dev_protocol["model_sha256"]:
        raise ValueError("model lineage mismatch")
    if freeze.get("helper_sha256") != dev_protocol["runtime_helper_sha256"]:
        raise ValueError("helper lineage mismatch")
    for name, expected in freeze["helper_sha256"].items():
        _require_hash(ROOT / name, expected)
    environment_paths = sorted((ROOT / "core").rglob("*.py")) + sorted((ROOT / "local_simulator").rglob("*.py"))
    environment_paths += [ROOT / "env_wrapper.py", ROOT / "damage.py"]
    actual_environment = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
        for path in environment_paths
    }
    if actual_environment != freeze.get("environment_sha256"):
        raise ValueError("official-generator environment differs from the frozen screen")
    receipts = require_consumed_receipts(SCREEN_RUN, freeze, track_id, seed)
    identity = {
        "screen_protocol_sha256": SCREEN_PROTOCOL_SHA256,
        "development_protocol_sha256": DEV_PROTOCOL_SHA256,
        "screen_freeze_sha256": SCREEN_FREEZE_SHA256,
        "screen_summary_sha256": SCREEN_SUMMARY_SHA256,
        "screen_harness_sha256": SCREEN_HARNESS_SHA256,
        "telemetry_tool_sha256": _sha256(Path(__file__)),
        "source_sha256": dev_protocol["agent_sha256"],
        "model_sha256": dev_protocol["model_sha256"],
        "helper_sha256": freeze["helper_sha256"],
        "environment_sha256": actual_environment,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    return identity, {"source": paths[arm], "model": MODEL, "class": dev_protocol["controller_classes"][arm],
                      "reference": receipts.get(arm)}


def _restore_instance_attribute(obj: Any, name: str, had_instance: bool, prior: Any) -> None:
    if had_instance:
        setattr(obj, name, prior)
    else:
        delattr(obj, name)


def act_with_obstacle(agent: Any, observation: Any):
    """Record the detector's existing return; do not call perception twice."""
    controller = agent._forward_controller
    had_instance = "_nearest_obstacle" in vars(controller)
    prior = vars(controller).get("_nearest_obstacle")
    detector = controller._nearest_obstacle
    observed: list[Any] = []

    def capture(*args, **kwargs):
        result = detector(*args, **kwargs)
        observed.append(result)
        return result

    controller._nearest_obstacle = capture
    try:
        action = frozen_eval.validate_action(agent.act(observation))
    finally:
        _restore_instance_attribute(controller, "_nearest_obstacle", had_instance, prior)
    obstacle = observed[-1] if observed else None
    return action, obstacle


def step_with_reward(session: Any, action: Any):
    """Read CarEnvironment's real four-frame reward while Session.step owns state."""
    environment = session.environment
    had_instance = "step" in vars(environment)
    prior = vars(environment).get("step")
    step_method = environment.step
    rewards: list[float] = []

    def capture(selected_action):
        result = step_method(selected_action)
        rewards.append(float(result[1]))
        return result

    environment.step = capture
    try:
        record = session.step(action)
    finally:
        _restore_instance_attribute(environment, "step", had_instance, prior)
    if len(rewards) != 1 or not math.isfinite(rewards[0]):
        raise RuntimeError("Session.step did not produce one finite official reward")
    return record, rewards[0]


def _controller_state(controller: Any) -> dict:
    return {
        "road_visible": bool(controller.road_visible),
        "obstacle_side": float(controller._obstacle_side),
        "obstacle_missing": int(controller._obstacle_missing),
        "side_offset": None if controller._last_obstacle_side_offset is None else float(controller._last_obstacle_side_offset),
        "target_speed": None if controller._target_speed is None else float(controller._target_speed),
        "observed_side_margins": None if getattr(controller, "_observed_side_margins", None) is None else [float(x) for x in controller._observed_side_margins],
    }


def make_step_row(
    *, step: Any, observation_sha256: str, act_ms: float,
    obstacle: Any, side_before: float, controller_state: dict,
    tiles_before: int, tiles_after: int, track_tiles: int,
    reward: float, wheel_on_road: list[bool], offtrack_counter: int,
) -> dict:
    if track_tiles <= 0:
        raise ValueError("official-generator track has no tiles")
    new_tiles = tiles_after - tiles_before
    return {
        "step": step.step,
        "observation_sha256": observation_sha256,
        "action": list(step.action),
        "act_ms": act_ms,
        "detected_obstacle": None if obstacle is None else [float(value) for value in obstacle],
        "obstacle_side_before": side_before,
        "controller": controller_state,
        "progress_before": tiles_before / track_tiles,
        "progress": step.progress,
        "progress_delta": step.progress - tiles_before / track_tiles,
        "visited_tiles": tiles_after,
        "new_tiles": new_tiles,
        "new_tile_reward": 1000.0 * new_tiles / track_tiles,
        "official_reward": reward,
        "wheel_on_road": wheel_on_road,
        "wheels_on_road": sum(wheel_on_road),
        "offtrack_counter": offtrack_counter,
        "position": list(step.position),
        "velocity": list(step.velocity),
        "speed_world": math.hypot(*step.velocity),
        "angle": step.angle,
        "collision": step.collision,
        "damage": step.damage,
        "terminated": step.terminated,
        "truncated": step.truncated,
    }


def _episode(arm: str, track_id: int, seed: int, inputs: dict) -> dict:
    import numpy as np
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
        name = "actual-Agent-route-telemetry"
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
            action, obstacle = act_with_obstacle(agent, observation)
            act_ms = (time.perf_counter() - began) * 1000
            if act_ms > 5000:
                raise RuntimeError("Agent.act exceeded 5 seconds")
            state = _controller_state(agent._forward_controller)
            action_hash.update(action.tobytes())
            step, reward = step_with_reward(session, action)
            tiles_after = int(raw.tile_visited_count)
            wheel_on_road = [bool(wheel.tiles) for wheel in raw.car.wheels]
            if len(wheel_on_road) != 4:
                raise RuntimeError("official car no longer has four wheels")
            rows.append(make_step_row(
                step=step, observation_sha256=observation_sha256, act_ms=act_ms,
                obstacle=obstacle, side_before=side_before, controller_state=state,
                tiles_before=tiles_before, tiles_after=tiles_after, track_tiles=track_tiles,
                reward=reward, wheel_on_road=wheel_on_road,
                offtrack_counter=int(session.environment.off_track_counter),
            ))
        summary = session.finish().summary
    finally:
        session.close()
    episode = {
        "arm": arm, "track_id": track_id, "seed": seed,
        "initialization_ms": initialization_ms, "reset_ms": policy.reset_ms,
        "summary": summary, "action_trace_sha256": action_hash.hexdigest(),
        "steps": rows,
    }
    reference = inputs["reference"]
    if reference is not None:
        keys = ("finished", "progress", "lap_time_ms", "collision_count", "damage", "retire_reason")
        mismatches = [key for key in keys if summary[key] != reference[key]]
        if len(rows) != reference["steps"] or episode["action_trace_sha256"] != reference["action_trace_sha256"]:
            mismatches.append("action trace")
        if sum(int(row["wheels_on_road"] == 0) for row in rows) != reference["offtrack_samples"]:
            mismatches.append("offtrack samples")
        if sum(int(row["wheels_on_road"] < 4) for row in rows) != reference["partial_offtrack_samples"]:
            mismatches.append("partial offtrack samples")
        if mismatches:
            raise RuntimeError(f"telemetry replay differs from frozen screen receipt: {mismatches}")
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
        if envelope.get("identity") != identity or envelope.get("digest") != hashlib.sha256(_canonical(payload)).hexdigest():
            raise ValueError(f"existing telemetry artifact has different identity or content: {path}")
        print(json.dumps({"artifact": str(path), "status": "EXISTING", "summary": envelope["episode"]["summary"]}))
        return 0
    episode = _episode(args.arm, args.track_id, args.seed, inputs)
    # Refuse an artifact if any pinned source changed while driving.
    after_identity, _ = preflight(args.arm, args.track_id, args.seed)
    if after_identity != identity:
        raise RuntimeError("frozen input identity changed during telemetry")
    payload = {"identity": identity, "episode": episode}
    envelope = {**payload, "digest": hashlib.sha256(_canonical(payload)).hexdigest()}
    frozen_eval._atomic_json(path, envelope)
    print(json.dumps({"artifact": str(path), "status": "RECORDED", "summary": episode["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
