"""Two exact research prefixes inspect remaining camera-control speed losses.

Privileged state measures the failure only. Inference receives unchanged
four-frame grayscale observations; these prefixes are not fresh lap results.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.postpass_probe import center_error, local, ridge_summary
from agents.apex_2026.research.speed_20261005.slow_phase import ego_support, instrument, physical_state
from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_summary(agent, frame, ahead, path, speed, yaw, desired):
    emitted = agent.last_steer+float(np.clip(desired-agent.last_steer, -.24, .24))
    curve = float(np.tan(emitted)/agent.WHEELBASE)
    return {"desired_steer": desired, "prospective_emitted_steer": emitted,
            "actual_emitted_curve_per_m": curve,
            "emitted_force_speed_ceiling_m_per_s": float(np.sqrt(agent.lateral_accel/max(abs(curve), .0005))),
            "corner_depth_min_m": agent._arc_guard_depth(agent._distance_field(frame), emitted),
            "path": {str(y): float(np.interp(y, ahead, path)) for y in (0., 4., 8., 12., 20., 28.)}}


def main():
    result_root = Path("agents/apex_2026/results/speed-20261005")
    output = result_root/"rear-clear-bottleneck.json"
    fixture = result_root/"rear-clear-bottleneck-cameras.npz"
    raw_dir = Path(".haic-artifacts/apex-speed-20261005/rear-clear-bottleneck")
    assert not output.exists() and not fixture.exists() and not raw_dir.exists()
    source = Path("agents/apex_2026/fast_rear_clear_agent.py")
    expected = "093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc"
    assert digest(source) == expected
    spec = importlib.util.spec_from_file_location("rear_clear_frozen_bottleneck", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official = ("env_wrapper.py", "damage.py", "core/vendor/car_racing.py", "core/vendor/car_dynamics.py", "core/track_variables.py", "core/finish_line.py")
    official_before = {name: digest(name) for name in official}
    cases, fixture_arrays = [], {}
    raw_dir.mkdir(parents=True)
    for track, seed, count, focus, checkpoints in (
        (1, 516237, 82, range(48, 82), (53, 56, 58, 59, 60, 61, 62, 65, 66)),
        (4, 18800, 175, range(122, 175), (125, 128, 130, 133, 136, 138, 140, 141, 142, 143, 144, 145, 150, 165, 170)),
    ):
        trace_path = Path(f".haic-artifacts/apex-speed-20261005/rear-clear-v1-traces/track-{track}-seed-{seed}.json")
        trace = json.loads(trace_path.read_text())
        agent = module.Agent()
        map_spec = MapSpec(track, seed, "official", (), 700, 4)
        environment, raw = create_environment(map_spec, render_mode=None)
        rows, all_observations, captured = [], [], {}
        instrument(agent, captured)
        steering = agent._steering

        def capture_steering(ahead, path, speed, yaw):
            desired = steering(ahead, path, speed, yaw)
            captured.setdefault("base_emitted_support", []).append(
                reference_summary(agent, current_frame, ahead, path, speed, yaw, desired))
            return desired

        agent._steering = capture_steering
        action_hash = hashlib.sha256()
        try:
            observation, _ = reset_environment(environment, map_spec)
            agent.reset(observation)
            for step, saved in enumerate(trace[:count]):
                captured.clear()
                current_frame = agent._frame(observation)
                previous = {name: value for name, value in agent.__dict__.items()
                            if isinstance(value, (int, float, bool, str, type(None)))}
                before = physical_state(raw)
                ridge = None
                support = ego_support(current_frame, agent)
                if step in focus:
                    truth = local(raw, np.asarray(raw.track)[:, 2:4])
                    original_ridge = module._ClearRidgeReference._ridge(agent, current_frame)
                    supported_ridge = module._ClearSupportedReference._ridge(agent, current_frame)
                    ridge = ridge_summary(agent, current_frame, original_ridge, supported_ridge, truth)
                action = agent.act(observation)
                np.testing.assert_array_equal(action, np.asarray(saved["action"], np.float32))
                action_hash.update(action.tobytes())
                all_observations.append(observation.copy())
                if step in checkpoints:
                    fixture_arrays[f"track{track}_observation_step{step}"] = observation.copy()
                    fixture_arrays[f"track{track}_prestate_step{step}"] = np.asarray(json.dumps(previous))
                road_error = None
                if step in focus and agent.corridor_road is not None:
                    road = agent.corridor_road
                    y = np.asarray([0., 4., 8., 12., 20., 28.])
                    x = np.interp(y, road[0], road[1])
                    error = center_error(np.column_stack((x, y)), truth)
                    road_error = {str(at): float(value) for at, value in zip(y, error)}
                observation, _, terminated, truncated, info = environment.step(action)
                after = physical_state(raw)
                np.testing.assert_allclose(after["speed_m_per_s"], saved["speed"], atol=1e-7, rtol=0.)
                policy = {name: getattr(agent, name) for name in
                          ("last_speed", "last_target", "last_yaw", "last_steer", "reference_curvature", "lost_frames",
                           "spin_frames", "pass_side", "pass_x", "pass_y", "pass_missing", "circle_cooldown", "mode", "arc_guard_used", "corridor_used")}
                rows.append({"step": step, "previous": previous, "policy": policy, "action": action.tolist(),
                    "before": before, "after": after, "ridge": ridge, "ego_support": support,
                    "road_center_error": road_error, "calls": dict(captured),
                    "hud_rear_m_per_s": agent._rear_speed(current_frame),
                    "force_demand_m_per_s2": max(policy["last_speed"]*abs(policy["last_yaw"]), policy["last_speed"]**2*abs(policy["reference_curvature"])),
                    "collision": bool(info.get("collision", False)), "damage": float(info.get("damage", 0.)),
                    "progress": float(info.get("progress", 0.))})
                assert not terminated and not truncated
        finally:
            environment.close()
        raw_path = raw_dir/f"track{track}-inspection.json"
        raw_path.write_text(json.dumps(rows, indent=2)+"\n")
        np.savez_compressed(raw_dir/f"track{track}-all-cameras.npz", observations=all_observations)
        compact = []
        for row in rows:
            compact.append({"step": row["step"], "action": row["action"], "policy": row["policy"],
                "speed_after_m_per_s": row["after"]["speed_m_per_s"], "sideslip_after_deg": row["after"]["sideslip_deg"],
                "wheel_road_after": row["after"]["wheel_road_count"], "road_offset_after_m": row["after"]["road_offset_m"],
                "ego_support": row["ego_support"], "ridge": None if row["ridge"] is None else {k: v for k, v in row["ridge"].items() if k != "supported_path_xy_m"},
                "force_demand_m_per_s2": row["force_demand_m_per_s2"], "hud_rear_m_per_s": row["hud_rear_m_per_s"],
                "circles": row["calls"].get("_circles", []), "collision": row["collision"]})
        cases.append({"track_id": track, "seed": seed, "actions_replayed": len(rows),
                      "trace_sha256": digest(trace_path), "action_prefix_sha256": action_hash.hexdigest(),
                      "raw_inspection_path": str(raw_path), "raw_inspection_sha256": digest(raw_path),
                      "rows": compact, "checkpoints": [row for row in rows if row["step"] in checkpoints]})
    assert digest(source) == expected
    assert official_before == {name: digest(name) for name in official}
    np.savez_compressed(fixture, **fixture_arrays)
    result = {"classification": "privileged_exact_camera_prefix_bottleneck_diagnostic_not_lap_validation",
              "source_sha256": expected, "script_sha256": digest(__file__), "official_sha256": official_before,
              "helper_sha256": {name: digest(f"agents/apex_2026/research/speed_20261005/{name}.py") for name in ("slow_phase", "postpass_probe")},
              "all_actions_and_post_step_speeds_exact": True, "new_holdout_opened": False,
              "camera_fixture_sha256": digest(fixture), "cases": cases}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
