"""Privileged diagnostic of one exact camera-agent replay prefix.

No privileged fields enter the controller. This never produces a lap result
or a legal candidate: it checks exact actions and observes why speed falls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np

from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def road_summary(road):
    if road is None:
        return None
    ahead, center = road[:2]
    return {"samples": len(ahead), "ahead_min_m": float(ahead[0]),
            "ahead_max_m": float(ahead[-1]),
            "center_at_m": {str(y): float(np.interp(y, ahead, center))
                            for y in (0, 4, 8, 12, 20, 28)}}


def ego_support(frame, agent):
    asphalt = (frame >= .32) & (frame <= .51)
    asphalt[73:] = False
    candidates = np.argwhere(asphalt[53:73])
    if not len(candidates):
        return {"pixels": 0, "row_span": 0, "seed": None, "accepted": False}
    candidates[:, 0] += 53
    distance = ((candidates[:, 1]-agent.CAR_X)/agent.PX_X)**2
    distance += ((candidates[:, 0]-agent.CAR_Y)/agent.PX_Y)**2
    seed = candidates[int(np.argmin(distance))]
    connected = np.zeros((84, 84), dtype=bool)
    connected[tuple(seed)] = True
    for _ in range(12):
        updated = agent._expand(connected) & asphalt
        if np.array_equal(updated, connected):
            break
        connected = updated
    support = np.argwhere(connected)
    return {"pixels": len(support), "row_span": int(np.ptp(support[:, 0])),
            "seed": seed.tolist(), "accepted": bool(len(support) >= 24 and np.ptp(support[:, 0]) >= 8)}


def physical_state(raw):
    hull = raw.car.hull
    velocity = np.asarray(hull.linearVelocity, float)
    forward = np.asarray(hull.GetWorldVector((0, 1)), float)
    right = np.asarray(hull.GetWorldVector((1, 0)), float)
    vf, vs = float(velocity@forward), float(velocity@right)
    wheels = []
    for wheel in raw.car.wheels:
        direction = np.asarray(wheel.GetWorldVector((0, 1)), float)
        sideways = np.asarray(wheel.GetWorldVector((1, 0)), float)
        v = np.asarray(wheel.linearVelocity, float)
        wheel_vf, wheel_vs = float(v@direction), float(v@sideways)
        wheels.append({"road": bool(wheel.tiles), "roll_m_per_s": float(wheel.omega*wheel.wheel_rad),
                       "forward_m_per_s": wheel_vf, "side_m_per_s": wheel_vs,
                       "joint_angle_rad": float(wheel.joint.angle),
                       "gas": float(wheel.gas), "brake": float(wheel.brake)})
    track = np.asarray(raw.track, float)
    position = np.asarray(hull.position, float)
    index = int(np.argmin(np.sum((track[:, 2:4]-position)**2, axis=1)))
    normal = np.asarray([math.cos(track[index, 1]), math.sin(track[index, 1])])
    return {"speed_m_per_s": float(np.linalg.norm(velocity)),
            "forward_m_per_s": vf, "side_m_per_s": vs,
            "sideslip_deg": math.degrees(math.atan2(vs, vf)),
            "yaw_camera_rad_per_s": -float(hull.angularVelocity),
            "road_offset_m": float((position-track[index, 2:4])@normal),
            "wheel_road_count": sum(w["road"] for w in wheels), "wheels": wheels}


def instrument(agent, store):
    for name in ("_road", "_circles", "_route", "_target", "_ridge", "_ridge_target", "_recover"):
        original = getattr(agent, name)

        def captured(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            if _name == "_road":
                value = road_summary(result)
            elif _name == "_circles":
                value = [list(map(float, point)) for point in result]
            elif _name == "_route":
                path, imminent, distance = result
                ahead = args[0]
                value = {"imminent": bool(imminent), "obstacle_distance_m": distance,
                         "path_at_m": {str(y): float(np.interp(y, ahead, path))
                                       for y in (0, 4, 8, 12, 20, 28)}}
            elif _name == "_ridge":
                value = None if result is None else {"points": len(result),
                    "arc_m": float(agent._arc(result)[-1]), "path": result.tolist()}
            elif _name == "_recover":
                value = result.tolist()
            else:
                value = float(result)
            store.setdefault(_name, []).append(value)
            return result

        setattr(agent, name, captured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("diagnostic output already exists")
    expected_sha = "e5ce693a4b9ec00d7ecbab8abf71d4a2c350ab95b4f816ae2b6547be91cfc00b"
    assert digest(args.source) == expected_sha
    official_files = ("env_wrapper.py", "damage.py", "core/vendor/car_dynamics.py",
                      "core/vendor/car_racing.py", "core/finish_line.py", "core/track_variables.py")
    official_before = {name: digest(name) for name in official_files}
    trace = json.loads(args.trace.read_text())
    spec = importlib.util.spec_from_file_location("slow_phase_submission", args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.Agent()
    map_spec = MapSpec(3, 1007, "official", (), 700, 4)
    environment, raw = create_environment(map_spec, render_mode=None)
    rows, frames, observations = [], [], []
    store = {}
    instrument(agent, store)
    actions = hashlib.sha256()
    try:
        observation, _ = reset_environment(environment, map_spec)
        agent.reset(observation)
        for step, saved in enumerate(trace[:120]):
            store.clear()
            frame = agent._frame(observation)
            before = physical_state(raw)
            support = ego_support(frame, agent)
            previous = {name: getattr(agent, name) for name in
                        ("last_target", "spin_frames", "pass_side", "pass_y", "circle_cooldown")}
            rear_estimate = agent._rear_speed(frame)
            action = agent.act(observation)
            np.testing.assert_array_equal(action, np.asarray(saved["action"], np.float32))
            actions.update(action.tobytes())
            observations.append(observation.copy())
            frames.append(frame.copy())
            next_observation, _, terminated, truncated, info = environment.step(action)
            after = physical_state(raw)
            np.testing.assert_allclose(after["speed_m_per_s"], saved["speed"], atol=1e-7, rtol=0.)
            rows.append({"step": step, "action": action.tolist(), "before": before, "after": after,
                         "ego_support": support, "previous": previous, "calls": dict(store),
                         "policy": {name: getattr(agent, name) for name in
                                    ("last_speed", "last_target", "last_yaw", "last_steer", "reference_curvature",
                                     "lost_frames", "spin_frames", "pass_side", "pass_y", "pass_missing",
                                     "corridor_used", "circle_cooldown", "mode")},
                         "hud_rear_m_per_s": rear_estimate,
                         "collision": bool(info.get("collision", False)),
                         "damage": float(info.get("damage", 0.)), "progress": float(info.get("progress", 0.))})
            observation = next_observation
            assert not terminated and not truncated
    finally:
        environment.close()
    assert digest(args.source) == expected_sha
    assert official_before == {name: digest(name) for name in official_files}
    args.raw_dir.mkdir(parents=True, exist_ok=False)
    camera_path = args.raw_dir/"cameras.npz"
    np.savez_compressed(camera_path, frames=frames, observations=observations)
    for step in (67, 72, 81, 82, 88, 94, 97):
        cv2.imwrite(str(args.raw_dir/f"camera-{step:03}.png"), np.rint(255*frames[step]).astype(np.uint8))
    payload = {"classification": "privileged_exact_replay_diagnostic_not_legal_candidate",
               "track_id": 3, "seed": 1007, "actions_replayed": len(rows),
               "all_actions_and_post_step_speeds_exact": True, "parameters": {},
               "source_sha256": expected_sha, "script_sha256": digest(__file__),
               "input_trace_sha256": digest(args.trace), "action_prefix_sha256": actions.hexdigest(),
               "official_files_sha256": official_before, "camera_npz_path": str(camera_path),
               "camera_npz_sha256": digest(camera_path), "new_holdout_opened": False,
               "rows": rows}
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
