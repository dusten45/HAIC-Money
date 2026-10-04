"""Source-bound saved-camera perception audit, with no new simulator episode."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RESEARCH = ROOT / ".haic-artifacts" / "apex-speed-20261005"
RESULT = ROOT / "agents/apex_2026/results/speed-20261005/early-far-v1-perception.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def transport(xy, frame, agent, next_frame=None):
    speed, yaw = agent._speed(frame), agent._yaw(frame)
    if next_frame is not None:
        speed = .5*(speed+agent._speed(next_frame))
        yaw = .5*(yaw+agent._yaw(next_frame))
    theta = .08*yaw
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    translation = (np.array([speed/yaw*(1.-np.cos(theta)), speed/yaw*np.sin(theta)])
                   if abs(yaw) > 1e-9 else np.array([0., .08*speed]))
    return rotation @ (xy-translation)


def main():
    parent = ROOT / "agents/apex_2026/fast_rear_clear_agent.py"
    candidate = ROOT / "agents/apex_2026/fast_early_far_agent.py"
    assert digest(parent) == "093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc"
    reference_type, candidate_type = load(parent, "far_parent"), load(candidate, "far_candidate")
    cases = []
    for track, seed in ((1, 516237), (2, 644062), (4, 18800)):
        if track == 2:
            camera_path = RESEARCH / "rear-geometry-track2/frames.npz"
            observations = np.repeat(np.load(camera_path)["frames"][:, None], 4, axis=1)
        else:
            camera_path = RESEARCH / f"rear-clear-bottleneck/track{track}-all-cameras.npz"
            observations = np.load(camera_path)["observations"]
        trace_path = RESEARCH / f"rear-clear-v1-traces/track-{track}-seed-{seed}.json"
        # Only saved public actions are read; world-state trace fields are unused.
        expected_actions = [row["action"] for row in json.loads(trace_path.read_text())]
        reference, candidate_agent = reference_type(), candidate_type()
        old_circles, new_circles = [], []
        for step, observation in enumerate(observations):
            np.testing.assert_array_equal(reference.act(observation), expected_actions[step])
            frame = observation[-1]
            road = reference._road(frame)
            old_circles.append(reference._circles(frame, road) if road is not None else [])
            new_circles.append(candidate_agent._circles(frame, road) if road is not None else [])
        changes = []
        for step, (old, new) in enumerate(zip(old_circles, new_circles)):
            if old == new:
                continue
            additions = [p for p in new if p not in old]
            matches = []
            for distance, lateral in additions:
                xy = np.array([lateral, distance])
                trapezoid_xy = xy.copy()
                best = trapezoid_best = None
                for offset in range(1, min(4, len(observations)-step)):
                    xy = transport(xy, observations[step+offset-1, -1], reference)
                    trapezoid_xy = transport(trapezoid_xy, observations[step+offset-1, -1], reference,
                                             observations[step+offset, -1])
                    for fy, fx in old_circles[step+offset]:
                        error = float(np.linalg.norm(xy-np.array([fx, fy])))
                        if best is None or error < best["error_m"]:
                            best = {"next_step": step+offset, "error_m": error}
                        trapezoid_error = float(np.linalg.norm(trapezoid_xy-np.array([fx, fy])))
                        if trapezoid_best is None or trapezoid_error < trapezoid_best["error_m"]:
                            trapezoid_best = {"next_step": step+offset, "error_m": trapezoid_error}
                matches.append({"circle_yx": [distance, lateral], "next_existing_camera_match": best,
                                "trapezoid_hud_camera_match": trapezoid_best})
            changes.append({"step": step, "hud_speed_mps": reference._speed(observations[step, -1]),
                            "old_circles": old, "new_circles": new, "matches": matches})
        cases.append({"track_id": track, "seed": seed, "frames": len(observations),
                      "all_saved_parent_actions_equal": True, "camera_sha256": digest(camera_path),
                      "trace_sha256": digest(trace_path), "changed_frames": changes})
    result = {"classification": "offline_saved_camera_perception_not_lap_validation",
              "parent_source_sha256": digest(parent), "candidate_source_sha256": digest(candidate),
              "script_sha256": digest(Path(__file__)), "cases": cases,
              "raw_camera_center_horizon_m": 63./1.701,
              "fully_supported_circle_horizon_at_road_row5_m": (63.-5.-1.2*1.701-.5)/1.701,
              "no_new_episode": True, "no_holdout": True,
              "limits": ["Future camera association confirms continuity, not a semantic ground-truth label.",
                         "The raster centroid uncertainty is assumed half a pixel per axis, not certified.",
                         "Trapezoid camera association uses later HUD pixels offline, never candidate inference.",
                         "Additional circles also inhibit the inherited ridge phase; pace gain is unproven.",
                         "Track1/4 camera prefixes do not cover their complete laps."]}
    RESULT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"candidate_sha256": result["candidate_source_sha256"],
                      "changes": [{"track": c["track_id"], "steps": [x["step"] for x in c["changed_frames"]]}
                                  for c in cases]}))


if __name__ == "__main__":
    main()
