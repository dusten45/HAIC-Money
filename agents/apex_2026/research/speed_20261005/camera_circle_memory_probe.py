"""Fixed offline camera/control circle-association assessment, never an Agent."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.camera_circle_memory_study import (
    interval_pose, motion_samples, path_projections, transform_points,
)
from agents.apex_2026.research.speed_20261005.physics_camera_observer import (
    CameraObserver, calibration_from_receipts,
)
from agents.apex_2026.research.speed_20261005.physics_four_tire_model import predict_step, rotate


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = ROOT / ".haic-artifacts/apex-speed-20261005"
RESULT = ROOT / "agents/apex_2026/results/speed-20261005/camera-circle-memory-v1.json"
MODES = ("previous_se2", "legacy_current_euler", "current_se2", "endpoint_mean",
         "past_slope_forecast", "camera_four_tire_forecast")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result.Agent


def stats(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def four_tire_pose(state, action):
    center = state["local_center"].copy()
    displacement = np.zeros(2)
    predicted = copy.deepcopy(state)
    for _ in range(4):
        predicted, _ = predict_step(predicted, [-action[0], action[1], action[2]])
        displacement += .02*predicted["velocity"]
    displacement += center-rotate(center, predicted["angle"])
    return -float(predicted["angle"]), displacement


def main():
    if RESULT.exists():
        raise ValueError("result exists; preserve the fixed study")
    parent = ROOT / "agents/apex_2026/fast_rear_clear_agent.py"
    perception = ROOT / "agents/apex_2026/fast_early_far_agent.py"
    assert digest(parent) == "093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc"
    assert digest(perception) == "1835002767a09f6f3839d6e9695c0a0d3b9b7ee47722ee06e7b7bad7cb4580fb"
    parent_type, perception_type = load(parent, "memory_parent"), load(perception, "memory_perception")
    records, cases, arc_records = [], [], []
    for track, seed in ((1, 516237), (2, 644062), (4, 18800)):
        if track == 2:
            camera_path = ARTIFACT / "rear-geometry-track2/frames.npz"
            frames = np.load(camera_path)["frames"]
        else:
            camera_path = ARTIFACT / f"rear-clear-bottleneck/track{track}-all-cameras.npz"
            frames = np.load(camera_path)["observations"][:, -1]
        trace_path = ARTIFACT / f"rear-clear-v1-traces/track-{track}-seed-{seed}.json"
        actions = [r["action"] for r in json.loads(trace_path.read_text())]
        reference, detector = parent_type(), perception_type()
        observer = CameraObserver(calibration_from_receipts())
        observed, samples, predicted_poses, sensors = [], [], [], []
        for step, frame in enumerate(frames):
            np.testing.assert_array_equal(reference.act(np.repeat(frame[None], 4, axis=0)), actions[step])
            road = detector._road(frame)
            circles = detector._circles(frame, road) if road is not None else []
            observed.append(np.array([[x, y] for y, x in circles]).reshape(-1, 2))
            samples.append([reference._speed(frame), reference._yaw(frame)])
            state, sensor = observer.observe(frame)
            predicted_poses.append(four_tire_pose(state, actions[step]))
            sensors.append(sensor)
            observer.advance(actions[step])
            if circles:
                path = detector._ridge(frame)
                if path is not None:
                    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
                    ego_gap = float(np.linalg.norm(path[0]))
                    for y, x in circles:
                        projections = path_projections(path, [x, y], .25)
                        nearest = min(projections, key=lambda p: p["distance"])
                        if nearest["distance"] <= 3.55:
                            arc_records.append({"track_id": track, "step": step, "circle_xy": [x, y],
                                                "forward_distance": y, "ridge_arc_projection": nearest["arc"],
                                                "ego_to_first_ridge_gap": ego_gap,
                                                "geometric_ego_arc_projection": ego_gap+nearest["arc"],
                                                "normal_distance": nearest["distance"],
                                                "visible_ridge_length": float(arc[-1]),
                                                "projection_branch_count": len(projections)})
        poses = {mode: [] for mode in MODES}
        for step in range(len(frames)-1):
            previous, current = samples[step], samples[step+1]
            poses["previous_se2"].append(interval_pose(*previous))
            poses["current_se2"].append(interval_pose(*current))
            poses["endpoint_mean"].append(interval_pose(*motion_samples(previous, current, "endpoint_mean")))
            poses["legacy_current_euler"].append((float(np.clip(.08*current[1], -.64, .64)),
                                                   np.array([0., .08*current[0]])))
            before = samples[max(0, step-1)]
            poses["past_slope_forecast"].append(interval_pose(*motion_samples(before, previous, "past_slope_forecast")))
            poses["camera_four_tire_forecast"].append(predicted_poses[step])
        ambiguous = unmatched = 0
        for step, circles in enumerate(observed[:-1]):
            for circle_index, circle in enumerate(circles):
                forecast = {mode: circle.copy() for mode in MODES}
                selected = None
                for gap in range(1, min(4, len(frames)-step)):
                    for mode in MODES:
                        turn, translation = poses[mode][step+gap-1]
                        forecast[mode] = transform_points(forecast[mode][None], turn, translation)[0]
                    future = observed[step+gap]
                    if not len(future):
                        continue
                    costs = np.array([np.linalg.norm(future-forecast[mode], axis=1) for mode in MODES])
                    eligible = np.flatnonzero(np.min(costs, axis=0) <= 6.)
                    if len(eligible) != 1:
                        if len(eligible) > 1:
                            ambiguous += 1
                            selected = False
                            break
                        continue
                    target = int(eligible[0])
                    selected = {"track_id": track, "step": step, "next_step": step+gap,
                                "gap_actions": gap, "circle_index": circle_index,
                                "old_circle_xy": circle.tolist(), "next_circle_xy": future[target].tolist(),
                                "hud_yaw_change": samples[step+gap][1]-samples[step][1],
                                "observer_innovation_suspect": sensors[step]["dynamics_innovation_suspect"],
                                "errors_m": {mode: float(costs[index, target]) for index, mode in enumerate(MODES)}}
                    break
                if isinstance(selected, dict):
                    records.append(selected)
                elif selected is None:
                    unmatched += 1
        cases.append({"track_id": track, "seed": seed, "frames": len(frames),
                      "camera_sha256": digest(camera_path), "trace_sha256": digest(trace_path),
                      "parent_actions_exact": True, "multi_circle_frames": sum(len(o) > 1 for o in observed),
                      "maximum_current_circles": max(map(len, observed)),
                      "ambiguous_associations_rejected": ambiguous, "unmatched_old_circles": unmatched})
    subsets = {"all_qualified": records,
               "one_action": [r for r in records if r["gap_actions"] == 1],
               "rapid_hud_change": [r for r in records if abs(r["hud_yaw_change"]) >= .5],
               "observer_innovation_not_suspect": [r for r in records if not r["observer_innovation_suspect"]]}
    dependencies = [Path(__file__), parent, perception,
                    ROOT / "agents/apex_2026/research/speed_20261005/camera_circle_memory_study.py",
                    ROOT / "agents/apex_2026/research/speed_20261005/physics_camera_observer.py",
                    ROOT / "agents/apex_2026/research/speed_20261005/physics_four_tire_model.py"]
    result = {"classification": "offline_camera_control_memory_geometry_assessment_not_agent_or_lap",
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in dependencies}, "cases": cases,
              "timing_contract": {"memory_update_modes": ["legacy_current_euler", "current_se2", "endpoint_mean"],
                                  "upcoming_forecast_modes": ["previous_se2", "past_slope_forecast", "camera_four_tire_forecast"],
                                  "future_camera_association_used_only_offline": True},
              "association_gate_m": 6., "max_gap_actions": 3,
              "stats": {subset: {mode: stats([r["errors_m"][mode] for r in group]) for mode in MODES}
                        for subset, group in subsets.items()},
              "associations": records, "ridge_arc_examples": arc_records,
              "new_episodes": 0, "new_holdout_opened": False,
              "limits": ["Temporal camera associations are not simulator identity or semantic ground-truth labels.",
                         "A unique six-meter multi-model gate may still associate a different object.",
                         "Samples are correlated replays of development prefixes, not prospective coverage.",
                         "The four-tire pipeline also changes HUD calibration and includes unbounded latent sideslip.",
                         "Arc distance includes a geometric ego-to-ridge chord, not a dynamically verified join.",
                         "No residual quantile is a certified worst-case observer or memory uncertainty bound."]}
    RESULT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"cases": cases, "stats": result["stats"]}, indent=2))


if __name__ == "__main__":
    main()
