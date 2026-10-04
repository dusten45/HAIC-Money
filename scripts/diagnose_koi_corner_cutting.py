"""Geometry-only corner-cutting screen on archived consumed champion traces."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from scripts.diagnose_koi_corner_target import BUNDLE, digest, geometry


def road_polygons(track):
    track = np.asarray(track, dtype=float)
    normals = np.column_stack((np.cos(track[:, 1]), np.sin(track[:, 1])))
    left = track[:, 2:4] - (40 / 6) * normals
    right = track[:, 2:4] + (40 / 6) * normals
    return np.stack((left, right, np.roll(right, -1, axis=0), np.roll(left, -1, axis=0)), axis=1)


def road_membership(points, polygons):
    points = np.asarray(points)
    positive = np.ones((len(points), len(polygons)), dtype=bool)
    negative = positive.copy()
    for j in range(4):
        edge = polygons[:, (j + 1) % 4] - polygons[:, j]
        delta = points[:, None] - polygons[None, :, j]
        cross = edge[None, :, 0] * delta[:, :, 1] - edge[None, :, 1] * delta[:, :, 0]
        positive &= cross >= -1e-8
        negative &= cross <= 1e-8
    return positive | negative


def on_road(points, polygons):
    return np.any(road_membership(points, polygons), axis=1)


def projection(points, track):
    track = np.asarray(track, dtype=float)
    starts = track[:, 2:4]
    edges = np.roll(starts, -1, axis=0) - starts
    lengths = np.linalg.norm(edges, axis=1)
    amounts = np.clip(np.sum((points[:, None] - starts[None]) * edges[None], axis=2)
                      / lengths[None] ** 2, 0, 1)
    closest = starts[None] + amounts[:, :, None] * edges[None]
    indices = np.argmin(np.sum((points[:, None] - closest) ** 2, axis=2), axis=1)
    tangent = edges[indices] / lengths[indices, None]
    heading = np.arctan2(-tangent[:, 0], tangent[:, 1])
    normal = np.column_stack((np.cos(heading), np.sin(heading)))
    lateral = np.sum((points - closest[np.arange(len(points)), indices]) * normal, axis=1)
    stations = np.r_[0., np.cumsum(lengths)]
    return stations[indices] + amounts[np.arange(len(points)), indices] * lengths[indices], heading, lateral


def longest_run(mask, durations):
    best = current = 0.
    for active, dt in zip(mask, durations):
        current = current + float(dt) if active else 0.
        best = max(best, current)
    return best


def measure(points, speeds, track, polygons, obstacles, reference_steering=None, actual_steering=None) -> dict:
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.r_[0., np.cumsum(ds)]
    velocity = np.gradient(points, axis=0)
    yaw = np.unwrap(np.arctan2(-velocity[:, 0], velocity[:, 1]))
    curvature = np.gradient(yaw, np.maximum.accumulate(distance + np.arange(len(distance)) * 1e-9))
    # Piecewise-linear resampling creates vertex spikes; assess demand over an
    # actual .08s controller hold, not artificial sub-physics interpolants.
    radius = max(1, round(float(np.median(speeds)) * .08 / max(float(np.median(ds)), .01) / 2))
    radius = min(radius, (len(points) - 5) // 2)
    curvature = np.convolve(np.pad(curvature, radius, mode="edge"),
                            np.ones(2 * radius + 1) / (2 * radius + 1), mode="valid")
    kinematic_steering = np.arctan(3.24 * curvature)
    steering = kinematic_steering if reference_steering is None else (
        actual_steering + kinematic_steering - np.asarray(reference_steering))
    durations = ds / np.maximum((speeds[:-1] + speeds[1:]) / 2, 1.)
    normals = np.column_stack((np.cos(yaw), np.sin(yaw)))
    tangents = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    wheels = np.stack([points + x * normals + y * tangents
                       for x, y in ((-1.1, 1.6), (1.1, 1.6), (-1.1, -1.64), (1.1, -1.64))], axis=1)
    membership = road_membership(wheels.reshape(-1, 2), polygons)
    support = np.any(membership, axis=1).reshape(-1, 4)
    any_off = ~np.all(support, axis=1)
    all_off = ~np.any(support, axis=1)
    stations, road_heading, lateral = projection(points, track)
    error = np.abs((yaw - road_heading + math.pi) % (2 * math.pi) - math.pi)
    reentries = np.flatnonzero(any_off[:-1] & ~any_off[1:]) + 1
    minimum_clearance = math.inf
    for obj in obstacles:
        delta = np.array([obj["x"], obj["y"]]) - points
        x = np.abs(np.sum(delta * normals, axis=1)) - 1.6
        y = np.abs(np.sum(delta * tangents, axis=1)) - 2.61
        clearance = np.hypot(np.maximum(x, 0), np.maximum(y, 0)) - obj["radius"]
        minimum_clearance = min(minimum_clearance, float(np.min(clearance)))
    return dict(path_length=float(distance[-1]), estimated_duration=float(sum(durations)),
                estimated_any_wheel_offroad_seconds=float(sum(durations[any_off[1:]])),
                estimated_longest_any_wheel_offroad_seconds=longest_run(any_off[1:], durations),
                estimated_longest_all_wheels_offroad_seconds=longest_run(all_off[1:], durations),
                reentry_heading_max_deg=float(np.degrees(np.max(error[reentries]))) if len(reentries) else None,
                exit_heading_error_deg=float(np.degrees(error[-1])),
                reentered=bool(not any_off[-1]),
                obstacle_rectangle_clearance=None if not obstacles else minimum_clearance,
                max_required_wheel_angle=float(np.max(np.abs(steering[2:-2]))),
                p95_required_lateral_acceleration=float(np.percentile(np.abs(curvature[2:-2]) * speeds[2:-2] ** 2, 95)),
                max_wheel_angle_rate=float(np.max(np.abs(np.diff(steering)) / np.maximum(durations, .001))),
                curvature_filter_distance=float((2 * radius + 1) * np.median(ds)),
                kinematic_steering=kinematic_steering.tolist(),
                wheel_center_tile_proxy=np.flatnonzero(np.any(membership, axis=0)).tolist(),
                station_backward=float(max(0, -np.min(np.diff(stations)))),
                lateral_min=float(np.min(lateral)), lateral_max=float(np.max(lateral)))


def diagnose(path) -> dict:
    stem = path.name.replace(".decisions.jsonl", "")
    primary_path, raw_path = path.with_name(stem + ".json"), path.with_name(stem + ".raw.jsonl")
    primary = json.loads(primary_path.read_text())
    decisions = [json.loads(line) for line in path.read_text().splitlines()]
    raw = [decisions[0]["controller"]["evaluation_only"]["pre"]]
    raw.extend(json.loads(line) for line in raw_path.read_text().splitlines())
    track = primary["catalog"]["track"]
    polygons = road_polygons(track)
    _, _, stations, corners = geometry(track)
    events = []
    for corner in corners:
        if corner["turn_deg"] < 45:
            continue
        event: dict = dict(id=f"{primary['track_id']}/{primary['seed']}/{corner['road_indices'][0]}",
                     track_id=primary["track_id"], seed=primary["seed"], **corner)
        start, end = max(1, corner["start"] - 10), min(stations[-1] - 1, corner["end"] + 10)
        entry_i = next((i for i in range(1, len(raw)) if raw[i - 1]["station"] < start <= raw[i]["station"]
                        and raw[i]["station"] - raw[i - 1]["station"] < 10), None)
        if entry_i is None:
            event["status"] = "NOT_REACHED"
            events.append(event)
            continue
        exit_i = next((i for i in range(entry_i, len(raw)) if raw[i]["station"] >= end), None)
        window = raw[entry_i:exit_i + 1 if exit_i is not None else len(raw)]
        if exit_i is None or any(not -1 <= b["station"] - a["station"] < 10 for a, b in zip(window, window[1:])):
            event["status"] = "CENSORED_OR_NONFORWARD"
            events.append(event)
            continue
        xy = np.array([[r["x"], r["y"]] for r in window])
        cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        if cumulative[-1] < 5:
            raise ValueError("Degenerate corner passage")
        fractions = np.linspace(0, 1, max(81, math.ceil(cumulative[-1] / .4) + 1))
        samples = fractions * cumulative[-1]
        actual = np.column_stack([np.interp(samples, cumulative, xy[:, j]) for j in range(2)])
        speeds = np.interp(samples, cumulative, [r["speed"] for r in window])
        chord = actual[0] + fractions[:, None] * (actual[-1] - actual[0])
        delta = np.sin(math.pi * fractions)[:, None] ** 4 * (chord - actual)
        capacity = float(np.max(np.linalg.norm(delta, axis=1)))
        control = [r for r in decisions if r["controller"]["evaluation_only"]["post"]["t"] > window[0]["t"]
                   and r["controller"]["evaluation_only"]["pre"]["t"] <= window[-1]["t"]]
        blocked = any(r["controller"].get("near_object") or r["controller"].get("far_active")
                      or r["controller"].get("contact_active") or r["controller"]["shield"]["active"]
                      or r["controller"].get("impact_steps_remaining", 0)
                      or r["controller"].get("recovery_changed") for r in control)
        baseline = measure(actual, speeds, track, polygons, primary["catalog"]["obstacles"])
        recorded_angles = np.array([np.mean([w["joint_angle"] for w in r["wheel_state"][:2]]) for r in window])
        wheel_angles = np.interp(samples, cumulative, recorded_angles)
        raw_dt = np.diff([r["t"] for r in window])
        baseline["observed_max_front_wheel_angle"] = float(np.max(np.abs(recorded_angles)))
        baseline["observed_max_front_wheel_angle_rate"] = float(np.max(np.abs(np.diff(recorded_angles)) / raw_dt))
        baseline["observed_fraction_near_steering_stop"] = float(np.mean(np.abs(recorded_angles) >= .39))
        baseline.update(actual_path_length=float(cumulative[-1]),
                        centerline_arc_length=float(window[-1]["station"] - window[0]["station"]),
                        actual_duration=window[-1]["t"] - window[0]["t"],
                        actual_longest_any_wheel_offroad_seconds=longest_run(
                            [not all(r["wheel_road_contacts"]) for r in window[1:]], raw_dt),
                        actual_longest_all_wheels_offroad_seconds=longest_run(
                            [not any(r["wheel_road_contacts"]) for r in window[1:]], raw_dt),
                        actual_collisions=sum(bool(r.get("collision") or r["contacts"]) for r in window),
                        mean_abs_lateral=float(np.mean([abs(r["lateral"]) for r in window])))
        alternatives = []
        for shift in (.75, 1.5, 2.25, 3.):
            if capacity < shift:
                continue
            proposed = actual + (shift / capacity) * delta
            metrics = measure(proposed, speeds, track, polygons, primary["catalog"]["obstacles"],
                              baseline["kinematic_steering"], wheel_angles)
            metrics.pop("kinematic_steering")
            saving = baseline["actual_path_length"] - metrics["path_length"]
            reasons = []
            if saving < 2 or saving / baseline["actual_path_length"] < .03:
                reasons.append("small_path_gain")
            if metrics["obstacle_rectangle_clearance"] is not None and metrics["obstacle_rectangle_clearance"] < .5:
                reasons.append("obstacle_interference")
            if metrics["estimated_longest_any_wheel_offroad_seconds"] > .5:
                reasons.append("long_offroad_prediction")
            if not metrics["reentered"] or metrics["exit_heading_error_deg"] > 15:
                reasons.append("exit_not_aligned_onroad")
            if metrics["reentry_heading_max_deg"] is not None and metrics["reentry_heading_max_deg"] > 15:
                reasons.append("reentry_heading")
            # Box2D measured joint motion can briefly exceed motor targets;
            # do not reject a path solely because its unchanged baseline did.
            angle_limit = max(.4, baseline["observed_max_front_wheel_angle"]) + .005
            rate_limit = 1.1 * max(3., baseline["observed_max_front_wheel_angle_rate"])
            if metrics["max_required_wheel_angle"] > angle_limit or metrics["max_wheel_angle_rate"] > rate_limit:
                reasons.append("steering_demand")
            if metrics["p95_required_lateral_acceleration"] > 1.1 * baseline["p95_required_lateral_acceleration"]:
                reasons.append("higher_lateral_demand")
            if metrics["station_backward"] > 1:
                reasons.append("nonforward_projection")
            if blocked:
                reasons.append("baseline_obstacle_or_recovery_confounded")
            metrics.update(apex_shift=shift, path_saving=saving,
                           newly_skipped_tile_proxy=sorted(set(baseline["wheel_center_tile_proxy"])
                                                          - set(metrics["wheel_center_tile_proxy"])),
                           steering_angle_screen_limit=angle_limit, steering_rate_screen_limit=rate_limit,
                           path_saving_fraction=saving / baseline["actual_path_length"],
                           break_even_fractional_mean_speed_loss=saving / baseline["actual_path_length"],
                           optimistic_time_saving=baseline["estimated_duration"] - metrics["estimated_duration"],
                           half_speed_offroad_sensitivity=2 * metrics["estimated_longest_any_wheel_offroad_seconds"],
                           rejected_by=reasons, screen_pass=not reasons)
            alternatives.append(metrics)
        baseline.pop("kinematic_steering")
        chord_metrics = measure(chord, speeds, track, polygons, primary["catalog"]["obstacles"])
        chord_metrics.pop("kinematic_steering")
        event.update(status="COMPLETE", baseline=baseline, alternatives=alternatives,
                     chord_lower_bound=chord_metrics,
                     entry_t=window[0]["t"], exit_t=window[-1]["t"],
                     baseline_confounded=blocked, entry_speed=window[0]["speed"],
                     qualifying=any(p["screen_pass"] for p in alternatives))
        events.append(event)
    return dict(track_id=primary["track_id"], seed=primary["seed"], completed=primary["completed"],
                inputs={str(p): digest(p) for p in (path, primary_path, raw_path)}, events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/koi-nominal-trajectory-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    preserved = {n: digest(BUNDLE / n) for n in manifest["files_sha256"]
                 if n == "submission.zip" or n.startswith("source/")}
    if any(h != manifest["files_sha256"][n] for n, h in preserved.items()):
        raise ValueError("Frozen champion changed")
    episodes = [diagnose(p) for p in sorted(args.run.glob("*-frozen_shield.decisions.jsonl"))]
    if len(episodes) != 8:
        raise ValueError("Expected the eight consumed champion episodes")
    events = [e for ep in episodes for e in ep["events"]]
    qualified = [e for e in events if e.get("qualifying")]
    roads = sorted({e["seed"] for e in qualified})
    summary: dict = dict(episodes=len(episodes), geometries=len({e["seed"] for e in episodes}),
                   sharp_corner_observations=len(events), complete=sum(e["status"] == "COMPLETE" for e in events),
                   qualifying_corners=[e["id"] for e in qualified], qualifying_geometries=roads,
                   implementation_gate=len(roads) >= 2 and len(qualified) >= 3)
    complete = [e for e in events if e["status"] == "COMPLETE"]
    summary["baseline_already_shorter_than_centerline"] = sum(
        e["baseline"]["actual_path_length"] < e["baseline"]["centerline_arc_length"] for e in complete)
    summary["unconfounded_complete"] = sum(not e["baseline_confounded"] for e in complete)
    summary["geometric_gain_only_corners"] = [e["id"] for e in complete if not e["baseline_confounded"]
        and any(p["path_saving"] >= 2 and p["path_saving_fraction"] >= .03 for p in e["alternatives"])]
    summary["rejections"] = dict(Counter(reason for e in complete for p in e["alternatives"] for reason in p["rejected_by"]))
    summary["unconfounded_best"] = [dict(id=e["id"], actual=e["baseline"]["actual_path_length"],
        arc=e["baseline"]["centerline_arc_length"], chord=e["chord_lower_bound"]["path_length"],
        best=min(e["alternatives"], key=lambda p: p["path_length"], default=None))
        for e in complete if not e["baseline_confounded"]]
    result = dict(scope="Passive consumed TRAIN geometry, not simulated trajectories or measured counterfactual gains",
                  status="GEOMETRIC_OPPORTUNITY" if summary["implementation_gate"] else "NO_REPEATED_FEASIBLE_GEOMETRIC_GAIN",
                  direction="OPEN_FOR_CANDIDATE" if summary["implementation_gate"] else "CLOSED_AT_DIAGNOSIS",
                  new_simulator_resets=0, candidate_created=False, ab_evaluation="NOT_RUN",
                  script_sha256=digest(Path(__file__)), geometry_helper_sha256=digest(Path("scripts/diagnose_koi_corner_target.py")),
                  frozen_champion=preserved, summary=summary, episodes=episodes,
                  definitions=dict(
                      family="Actual path blended toward endpoint chord with sin(pi*u)^4 envelope; max inward displacement .75/1.5/2.25/3 distance units; end positions/first two derivatives unchanged analytically",
                      chord="Unconstrained Euclidean lower bound, not steering/heading-feasible minimum",
                      offroad="Predicted wheel-center membership in recorded road quadrilateral union; point contacts are proxies, not exact wheel fixtures; <=.5s longest run is allowed, not zero; half-speed sensitivity is not a physical bound",
                      time="Use archived physical speed along normalized baseline path; ignore grass friction and closed-loop feedback, so time savings are optimistic, not lap predictions",
                      kinematics="Differentiate path with curvature averaged over approximately one .08s hold. Candidate wheel demand = observed baseline front-wheel angle + atan(3.24*k_candidate)-atan(3.24*k_baseline); local slip-calibrated optimistic proxy, not dynamic feasibility proof. Baseline/chord show uncalibrated bicycle demand for reference.",
                      progress="Frozen finish requires >=95% unique-tile coverage plus forward start crossing. Newly skipped wheel-center tile sets are geometric proxies, not fixture-contact or full-lap coverage proofs; repeated cuts can consume the remaining coverage budget.",
                      gate="At least three qualifying corners on >=2 geometries; gain>=2 distance units AND3%; brief offroad allowed; reentry/exit<=15deg; >=.5 obstacle bounding-rectangle clearance; wheel angle<=max(.4,observed baseline max)+.005; rate<=110% max(3,observed baseline max); p95 lateral demand<=110% baseline; no recorded obstacle/recovery interference",
                      geometry="Only existing archived geometry for passive diagnosis; any future runtime must be pixel-only, not seeded/world-coordinate lookup"))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "unconfounded_best"}, indent=2))


if __name__ == "__main__":
    main()
