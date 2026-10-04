"""One exact T4 prefix: post-pass phases, camera references and physical edge.

Simulator track/obstacle/pose fields are diagnostic only. They are never
passed into the camera agent, and this prefix is not a completed-lap result.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.slow_phase import ego_support, instrument, physical_state
from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(raw, points):
    hull = raw.car.hull
    offset = np.asarray(points, float)-np.asarray(hull.position, float)
    return np.column_stack((offset@np.asarray(hull.GetWorldVector((1, 0)), float),
                            offset@np.asarray(hull.GetWorldVector((0, 1)), float)))


def center_error(points, track):
    begin, end = track, np.roll(track, -1, axis=0)
    link = end-begin
    relative = np.asarray(points)[:, None]-begin[None]
    phase = np.clip(np.sum(relative*link[None], axis=2)/np.maximum(np.sum(link*link, axis=1), 1e-8), 0., 1.)
    projection = begin[None]+phase[:, :, None]*link[None]
    return np.min(np.linalg.norm(np.asarray(points)[:, None]-projection, axis=2), axis=1)


def ridge_summary(agent, frame, raw_ridge, supported, truth):
    field = agent._distance_field(frame)
    result = {"untrimmed_points": 0 if raw_ridge is None else len(raw_ridge),
              "supported_points": 0 if supported is None else len(supported),
              "eligible_except_phase": False}
    if raw_ridge is not None:
        depth = agent._sample_distance(field, raw_ridge)
        bad = np.flatnonzero(depth < 5.8)
        result.update({"untrimmed_first_unsupported": int(bad[0]) if len(bad) else None,
                       "untrimmed_min_camera_depth_m": float(depth.min()),
                       "untrimmed_center_error_max_m": float(center_error(raw_ridge, truth).max())})
    if supported is not None:
        dense = []
        for a, b in zip(supported[:-1], supported[1:]):
            count = max(2, int(np.ceil(np.linalg.norm(b-a)/.5))+1)
            phase = np.linspace(0., 1., count)
            dense.append(a[None]+phase[:, None]*(b-a)[None])
        depth = agent._sample_distance(field, np.concatenate(dense))
        result.update({"supported_arc_m": float(agent._arc(supported)[-1]),
                       "supported_chord_depth_min_m": float(depth.min()),
                       "supported_center_error_max_m": float(center_error(supported, truth).max()),
                       "supported_start_xy_m": supported[0].tolist(),
                       "supported_path_xy_m": supported.tolist(),
                       "eligible_except_phase": bool(depth.min() >= 1.9)})
    return result


def main():
    root = Path("agents/apex_2026/results/speed-20261005")
    output = root/"postpass-track4-arc-clear-v2.json"
    fixture = root/"postpass-track4-cameras.npz"
    raw_dir = Path(".haic-artifacts/apex-speed-20261005/postpass-track4-arc-clear-v2")
    assert not output.exists() and not fixture.exists() and not raw_dir.exists()
    source = Path("agents/apex_2026/fast_arc_clear_agent.py")
    expected = "15478c179ff1c8fa2c7dfe12bf65eaba91ad215fc250bde893ce65d7eebcd5ba"
    assert digest(source) == expected
    trace_path = Path(".haic-artifacts/apex-speed-20261005/arc-clear-v2-traces/track-4-seed-18800.json")
    trace = json.loads(trace_path.read_text())
    official = ("env_wrapper.py", "damage.py", "core/vendor/car_dynamics.py",
                "core/vendor/car_racing.py", "core/track_variables.py", "core/finish_line.py")
    before_hash = {name: digest(name) for name in official}
    spec = importlib.util.spec_from_file_location("postpass_frozen_candidate", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.Agent()
    map_spec = MapSpec(4, 18800, "official", (), 700, 4)
    environment, raw = create_environment(map_spec, render_mode=None)
    rows, all_observations, fixture_arrays, captured = [], [], {}, {}
    instrument(agent, captured)
    action_hash = hashlib.sha256()
    checkpoints = (86, 89, 93, 94, 96, 99, 100, 101, 103, 104, 105, 106, 111, 130, 139)
    try:
        observation, _ = reset_environment(environment, map_spec)
        agent.reset(observation)
        for step, saved in enumerate(trace[:140]):
            captured.clear()
            frame = agent._frame(observation)
            previous = {name: value for name, value in agent.__dict__.items()
                        if isinstance(value, (int, float, bool, str, type(None)))}
            support = ego_support(frame, agent)
            truth = local(raw, np.asarray(raw.track)[:, 2:4])
            obstacles = local(raw, [list(body.position) for body in raw.obstacles])
            obstacle_rows = [{"index": index, "x_m": float(point[0]), "y_m": float(point[1]),
                              "rear_clearance_m": float(-2.4-point[1]-body.fixtures[0].shape.radius)}
                             for index, (point, body) in enumerate(zip(obstacles, raw.obstacles))]
            raw_ridge = module._ClearRidgeReference._ridge(agent, frame)
            supported = module._ClearSupportedReference._ridge(agent, frame)
            ridge = ridge_summary(agent, frame, raw_ridge, supported, truth)
            state_before = physical_state(raw)
            action = agent.act(observation)
            np.testing.assert_array_equal(action, np.asarray(saved["action"], np.float32))
            action_hash.update(action.tobytes())
            all_observations.append(observation.copy())
            if step in checkpoints:
                fixture_arrays[f"observation_step{step}"] = observation.copy()
                fixture_arrays[f"prestate_step{step}"] = np.asarray(json.dumps(previous))
            road = agent.corridor_road
            road_errors = None
            if road is not None:
                samples = np.asarray([0., 4., 8., 12., 20., 28.])
                center = np.interp(samples, road[0], road[1])
                errors = center_error(np.column_stack((center, samples)), truth)
                road_errors = {str(y): {"x_m": float(x), "center_error_m": float(error)}
                               for y, x, error in zip(samples, center, errors)}
            observation, _, terminated, truncated, info = environment.step(action)
            state_after = physical_state(raw)
            np.testing.assert_allclose(state_after["speed_m_per_s"], saved["speed"], atol=1e-7, rtol=0.)
            policy = {name: getattr(agent, name) for name in
                      ("last_speed", "last_target", "last_yaw", "last_steer", "lost_frames", "spin_frames",
                       "pass_side", "pass_x", "pass_y", "pass_missing", "circle_cooldown", "mode",
                       "corridor_used", "reference_curvature", "arc_guard_used")}
            rows.append({"step": step, "previous": previous, "policy": policy, "action": action.tolist(),
                         "ego_support": support, "ridge": ridge, "calls": dict(captured),
                         "road_center_error": road_errors, "actual_obstacles_before": obstacle_rows,
                         "before": state_before, "after": state_after,
                         "collision": bool(info.get("collision", False)),
                         "damage": float(info.get("damage", 0.)), "progress": float(info.get("progress", 0.))})
            assert not terminated and not truncated
    finally:
        environment.close()
    assert digest(source) == expected
    assert before_hash == {name: digest(name) for name in official}
    raw_dir.mkdir(parents=True)
    np.savez_compressed(raw_dir/"all-cameras.npz", observations=all_observations)
    np.savez_compressed(fixture, **fixture_arrays)
    raw_inspection = raw_dir/"inspection.json"
    raw_inspection.write_text(json.dumps(rows, indent=2)+"\n")
    compact_rows = []
    for row in rows:
        compact_rows.append({"step": row["step"], "action": row["action"], "policy": row["policy"],
            "ego_support": row["ego_support"], "ridge": {k: v for k, v in row["ridge"].items() if k != "supported_path_xy_m"},
            "actual_obstacles_before": row["actual_obstacles_before"],
            "speed_before_m_per_s": row["before"]["speed_m_per_s"],
            "speed_after_m_per_s": row["after"]["speed_m_per_s"],
            "sideslip_after_deg": row["after"]["sideslip_deg"],
            "road_offset_after_m": row["after"]["road_offset_m"],
            "wheel_road_after": row["after"]["wheel_road_count"],
            "circles": row["calls"].get("_circles", []), "collision": row["collision"]})
    result = {"classification": "privileged_exact_prefix_diagnostic_not_lap_validation",
              "source_sha256": expected, "script_sha256": digest(__file__),
              "helper_sha256": digest("agents/apex_2026/research/speed_20261005/slow_phase.py"),
              "trace_sha256": digest(trace_path), "official_sha256": before_hash,
              "track_id": 4, "seed": 18800, "actions_replayed": len(rows),
              "all_actions_and_post_step_speeds_exact": True, "action_prefix_sha256": action_hash.hexdigest(),
              "camera_fixture_sha256": digest(fixture), "raw_inspection_path": str(raw_inspection),
              "raw_inspection_sha256": digest(raw_inspection), "new_holdout_opened": False,
              "rows": compact_rows, "checkpoints": [row for row in rows if row["step"] in checkpoints]}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "checkpoints")}, indent=2))


if __name__ == "__main__":
    main()
