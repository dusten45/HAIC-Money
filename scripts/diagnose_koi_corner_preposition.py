"""Passive outside-entry spline screen; never constructs an Agent or environment."""

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path

import numpy as np

from scripts.diagnose_koi_corner_cutting import measure, projection, road_polygons
from scripts.diagnose_koi_corner_target import BUNDLE, digest


PRIOR = Path("experiments/koi-corner-cutting-diagnosis-v1.json")
RUN = Path("runs/koi-nominal-trajectory-v1")


def clamped_spline(knots, values, start_derivative, end_derivative, samples,
                   start_second=None, end_second=None):
    """C2 cubic, with endpoint quintic corrections for observed curvature jets."""
    knots, values = np.asarray(knots, float), np.asarray(values, float)
    h = np.diff(knots)
    if np.any(h <= 0):
        raise ValueError("Spline knots must be strictly ordered")
    n = len(knots)
    matrix, rhs = np.zeros((n, n)), np.zeros_like(values)
    matrix[0, :2] = 2 * h[0], h[0]
    matrix[-1, -2:] = h[-1], 2 * h[-1]
    rhs[0] = 6 * ((values[1] - values[0]) / h[0] - start_derivative)
    rhs[-1] = 6 * (end_derivative - (values[-1] - values[-2]) / h[-1])
    for i in range(1, n - 1):
        matrix[i, i - 1:i + 2] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 6 * ((values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1])
    second = np.linalg.solve(matrix, rhs)
    first = np.vstack(((values[1:] - values[:-1]) / h[:, None]
                       - h[:, None] * (2 * second[:-1] + second[1:]) / 6,
                       np.asarray(end_derivative)))
    if start_second is not None:
        second[0] = start_second
    if end_second is not None:
        second[-1] = end_second
    idx = np.clip(np.searchsorted(knots, samples, side="right") - 1, 0, n - 2)
    width = h[idx, None]
    u = (samples - knots[idx])[:, None] / width
    a0, a1, a2 = values[idx], width * first[idx], width ** 2 * second[idx] / 2
    r0 = values[idx + 1] - a0 - a1 - a2
    r1 = width * first[idx + 1] - a1 - 2 * a2
    r2 = width ** 2 * second[idx + 1] - 2 * a2
    a3, a4, a5 = 10 * r0 - 4 * r1 + r2 / 2, -15 * r0 + 7 * r1 - r2, 6 * r0 - 3 * r1 + r2 / 2
    position = a0 + a1 * u + a2 * u ** 2 + a3 * u ** 3 + a4 * u ** 4 + a5 * u ** 5
    d1 = (a1 + 2 * a2 * u + 3 * a3 * u ** 2 + 4 * a4 * u ** 3 + 5 * a5 * u ** 4) / width
    d2 = (2 * a2 + 6 * a3 * u + 12 * a4 * u ** 2 + 20 * a5 * u ** 3) / width ** 2
    d3 = (6 * a3 + 24 * a4 * u + 60 * a5 * u ** 2) / width ** 3
    return position, d1, d2, d3


def analytic_demand(d1, d2, d3, speed):
    norm = np.linalg.norm(d1, axis=1)
    norm = np.maximum(norm, 1e-9)
    cross12 = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    cross13 = d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0]
    curvature = cross12 / norm ** 3
    derivative = cross13 / norm ** 4 - 3 * cross12 * np.sum(d1 * d2, axis=1) / norm ** 6
    steer = np.arctan(3.24 * curvature)
    rate = 3.24 * derivative * speed / (1 + (3.24 * curvature) ** 2)
    return curvature, steer, rate


def center_at(track, station):
    xy = np.asarray(track, float)[:, 2:4]
    points = np.vstack((xy, xy[0]))
    stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    i = int(np.clip(np.searchsorted(stations, station, side="right") - 1, 0, len(xy) - 1))
    f = (station - stations[i]) / (stations[i + 1] - stations[i])
    position = points[i] + f * (points[i + 1] - points[i])
    direction = points[i + 1] - points[i]
    direction /= np.linalg.norm(direction)
    return position, np.array([direction[1], -direction[0]])


def observed_jet(raw, index):
    nearby = raw[max(0, index - 4):min(len(raw), index + 5)]
    ss = np.array([r["station"] - raw[index]["station"] for r in nearby])
    if ss[-1] - ss[0] <= .1 or np.any(np.diff(ss) <= 0):
        raise ValueError("Endpoint not forward enough to preserve tangent")
    coefficients = np.polyfit(ss, [[r["x"], r["y"]] for r in nearby], 3)
    return coefficients[-2], 2 * coefficients[-3]


def phase_lengths(points, stations, start, end):
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    middle = (stations[:-1] + stations[1:]) / 2
    return dict(setup=float(np.sum(ds[middle < start])),
                corner=float(np.sum(ds[(middle >= start) & (middle <= end)])),
                return_path=float(np.sum(ds[middle > end])))


def one_case(event) -> dict:
    stem = f"{event['track_id']}-{event['seed']}-r0-frozen_shield"
    paths = [RUN / (stem + ext) for ext in (".json", ".raw.jsonl", ".decisions.jsonl")]
    primary = json.loads(paths[0].read_text())
    decisions = [json.loads(line) for line in paths[2].read_text().splitlines()]
    raw = [decisions[0]["controller"]["evaluation_only"]["pre"]]
    raw.extend(json.loads(line) for line in paths[1].read_text().splitlines())
    track = primary["catalog"]["track"]
    polygons = road_polygons(track)
    times = np.array([r["t"] for r in raw])
    turn_index = next(i for i, r in enumerate(raw) if r["station"] >= event["start"])
    entry_time = raw[turn_index]["t"]
    entry_decision = max(i for i, row in enumerate(decisions)
                         if row["controller"]["evaluation_only"]["pre"]["t"] <= entry_time)
    return_index = next((i for i in range(turn_index, len(raw)) if raw[i]["station"] >= event["end"] + 20), None)
    out: dict = dict(id=event["id"], seed=event["seed"], track_id=event["track_id"],
               inputs={str(p): digest(p) for p in paths}, leads=[])
    if return_index is None:
        out["status"] = "RETURN_WINDOW_CENSORED"
        return out
    out["status"] = "ANALYZED"
    for lead in (10, 20, 30):
        if entry_decision < lead:
            out["leads"].append(dict(decisions_before=lead, status="INSUFFICIENT_HISTORY"))
            continue
        start_control = decisions[entry_decision - lead]
        start_time = start_control["controller"]["evaluation_only"]["pre"]["t"]
        start_index = int(np.argmin(np.abs(times - start_time)))
        window = raw[start_index:return_index + 1]
        ss = np.array([r["station"] for r in window])
        if np.any(np.diff(ss) < -.1):
            out["leads"].append(dict(decisions_before=lead, status="NONFORWARD_BASELINE"))
            continue
        s0, s1 = ss[0], ss[-1]
        samples = np.linspace(s0, s1, math.ceil((s1 - s0) / .4) + 1)
        xy = np.array([[r["x"], r["y"]] for r in window])
        actual = np.column_stack([np.interp(samples, ss, xy[:, i]) for i in range(2)])
        speed = np.interp(samples, ss, [r["speed"] for r in window])
        baseline = measure(actual, speed, track, polygons, primary["catalog"]["obstacles"])
        baseline.pop("kinematic_steering")
        baseline["raw_path_length"] = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        baseline["actual_duration"] = window[-1]["t"] - window[0]["t"]
        baseline["min_speed"] = float(np.min(speed))
        baseline["phase_path_lengths"] = phase_lengths(actual, samples, event["start"], event["end"])
        d0, dd0 = observed_jet(raw, start_index)
        d1, dd1 = observed_jet(raw, return_index)
        _, endpoint_steer, _ = analytic_demand(np.array([d0, d1]), np.array([dd0, dd1]),
                                               np.zeros((2, 2)), np.ones(2))
        overlap = [row for row in decisions if start_time <= row["controller"]["evaluation_only"]["pre"]["t"] <= window[-1]["t"]]
        interference = sum(bool(row["controller"].get("near_object") or row["controller"].get("far_active")
                               or row["controller"].get("contact_active") or row["controller"]["shield"]["active"]
                               or row["controller"].get("impact_steps_remaining", 0)
                               or row["controller"].get("recovery_changed")) for row in overlap)
        centers = start_control["controller"]["road_centers"]
        visual_bend = (centers["30"] - 2 * centers["42"] + centers["54"]
                       if all(k in centers for k in ("30", "42", "54")) else None)
        # Same extended window and speed-vs-station schedule for the old direct pull.
        direct = actual.copy()
        core = (samples >= event["start"] - 10) & (samples <= event["end"] + 10)
        indices = np.flatnonzero(core)
        u = np.linspace(0, 1, len(indices))
        chord = actual[indices[0]] + u[:, None] * (actual[indices[-1]] - actual[indices[0]])
        delta = np.sin(np.pi * u)[:, None] ** 4 * (chord - actual[indices])
        capacity = np.max(np.linalg.norm(delta, axis=1))
        direct[indices] += min(1., 3 / max(capacity, 1e-9)) * delta
        direct_station, _, _ = projection(direct, track)
        direct_speed = np.interp(direct_station, ss, [r["speed"] for r in window])
        direct_metrics = measure(direct, direct_speed, track, polygons, primary["catalog"]["obstacles"])
        direct_metrics.pop("kinematic_steering")
        lead_out: dict = dict(decisions_before=lead, seconds_before=entry_time - start_time,
                        start_station=s0, return_station=s1, baseline=baseline, direct_pull=direct_metrics,
                        earlier_interference_decisions=interference, visual_second_difference=visual_bend,
                        status="ANALYZED", alternatives=[])
        for outside, inside, apex_fraction, turnin_before in itertools.product((2., 4.), (2., 4., 6.), (.5, .65), (5., 10.)):
            turn_s = event["start"] - turnin_before
            apex_s = event["start"] + apex_fraction * (event["end"] - event["start"])
            exit_s = event["end"] + 5
            if turn_s <= s0 + 2:
                continue
            knots = [s0]
            points = [actual[0]]
            # Preserve intervening approach-road shape; a long endpoint spline
            # must not get credit for cutting an unrelated earlier corner.
            for station in np.arange(s0 + 8, turn_s - 1, 8):
                fraction = (station - s0) / (turn_s - s0)
                weight = 10 * fraction ** 3 - 15 * fraction ** 4 + 6 * fraction ** 5
                original = np.array([np.interp(station, ss, xy[:, j]) for j in range(2)])
                center, normal = center_at(track, station)
                knots.append(station)
                points.append((1 - weight) * original + weight * (center + event["direction"] * outside * normal))
            for station, offset in ((turn_s, outside), (event["start"], outside),
                                    (apex_s, -inside), (exit_s, outside)):
                center, normal = center_at(track, station)
                knots.append(station)
                points.append(center + event["direction"] * offset * normal)
            knots.append(s1)
            points.append(actual[-1])
            knots = np.asarray(knots)
            proposed, deriv1, deriv2, deriv3 = clamped_spline(knots, points, d0, d1, samples, dd0, dd1)
            projected_station, road_heading, lateral = projection(proposed, track)
            proposed_speed = np.interp(projected_station, ss, [r["speed"] for r in window])
            metrics = measure(proposed, proposed_speed, track, polygons, primary["catalog"]["obstacles"])
            metrics.pop("kinematic_steering")
            curvature, steer, rate = analytic_demand(deriv1, deriv2, deriv3, proposed_speed)
            yaw = np.unwrap(np.arctan2(-deriv1[:, 0], deriv1[:, 1]))
            crossings = np.flatnonzero((projected_station[:-1] < event["start"])
                                       & (projected_station[1:] >= event["start"])) + 1
            entry_i = int(crossings[0]) if len(crossings) else int(np.argmin(np.abs(projected_station - event["start"])))
            entry_outside = event["direction"] * lateral[entry_i]
            entry_heading = (yaw[entry_i] - road_heading[entry_i] + math.pi) % (2 * math.pi) - math.pi
            phases = {}
            for phase, mask in (("setup", projected_station < event["start"]),
                                ("corner", (projected_station >= event["start"]) & (projected_station <= event["end"])),
                                ("return", projected_station > event["end"])):
                if not np.any(mask):
                    phases[phase] = dict(unobserved=True)
                    continue
                phases[phase] = dict(max_curvature=float(np.max(np.abs(curvature[mask]))),
                                    max_wheel_angle=float(np.max(np.abs(steer[mask]))),
                                    max_wheel_rate=float(np.max(np.abs(rate[mask]))),
                                    max_lateral_acceleration=float(np.max(np.abs(curvature[mask]) * proposed_speed[mask] ** 2)))
            saving = baseline["path_length"] - metrics["path_length"]
            reasons = []
            if saving < 2:
                reasons.append("net_saving_below_2")
            if entry_outside < 1:
                reasons.append("not_actually_outside_at_entry")
            if event["direction"] * entry_heading < 0:
                reasons.append("entry_heading_not_toward_corner")
            if metrics["station_backward"] > 1 or np.max(np.diff(projected_station)) > 10 or len(crossings) != 1:
                reasons.append("nonforward_or_loop")
            if not metrics["reentered"] or metrics["exit_heading_error_deg"] > 15:
                reasons.append("return_not_settled")
            if metrics["reentry_heading_max_deg"] is not None and metrics["reentry_heading_max_deg"] > 15:
                reasons.append("reentry_heading")
            if metrics["estimated_longest_any_wheel_offroad_seconds"] > .5:
                reasons.append("long_offroad")
            if metrics["obstacle_rectangle_clearance"] is not None and metrics["obstacle_rectangle_clearance"] < .5:
                reasons.append("obstacle_interference")
            if np.max(np.abs(steer)) > .405 or np.max(np.abs(rate)) > 3.1:
                reasons.append("analytic_bicycle_demand")
            if metrics["max_required_wheel_angle"] >= .95 * direct_metrics["max_required_wheel_angle"]:
                reasons.append("no_same_model_peak_steering_improvement")
            join_steer_difference = np.array([steer[0], steer[-1]]) - endpoint_steer
            if np.max(np.abs(join_steer_difference)) > .03:
                reasons.append("endpoint_curvature_debt")
            if interference:
                reasons.append("setup_overlaps_existing_avoidance_or_recovery")
            metrics.update(outside=outside, inside=inside, apex_fraction=apex_fraction,
                           turnin_before=turnin_before, net_path_saving=saving,
                           optimistic_time_saving=baseline["estimated_duration"] - metrics["estimated_duration"],
                           entry_outside=entry_outside, entry_heading_error_deg=math.degrees(entry_heading),
                           entry_heading_toward_corner_deg=event["direction"] * math.degrees(entry_heading),
                           analytic_phases=phases, max_offset=float(np.max(np.abs(lateral))),
                           phase_path_lengths=phase_lengths(proposed, projected_station, event["start"], event["end"]),
                           peak_curvature_parameter_station=float(samples[np.argmax(np.abs(curvature))]),
                           entry_station_projection_error=float(projected_station[entry_i] - event["start"]),
                           endpoint_analytic_steer=[float(steer[0]), float(steer[-1])],
                           endpoint_steer_difference=join_steer_difference.tolist(),
                           skipped_tile_proxy=sorted(set(baseline["wheel_center_tile_proxy"]) - set(metrics["wheel_center_tile_proxy"])),
                           rejected_by=reasons, screen_pass=not reasons)
            lead_out["alternatives"].append(metrics)
        out["leads"].append(lead_out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(PRIOR.read_text())
    ids = set(prior["summary"]["geometric_gain_only_corners"])
    source = [e for ep in prior["episodes"] for e in ep["events"] if e["id"] in ids]
    cases = [one_case(e) for e in source]
    alternatives = [p for c in cases for lead in c["leads"] for p in lead.get("alternatives", [])]
    passed = [c for c in cases if any(p["screen_pass"] for lead in c["leads"] for p in lead.get("alternatives", []))]
    summary: dict = dict(cases=len(cases), geometries=len({c["seed"] for c in cases}),
                   paths=len(alternatives), passed_cases=[c["id"] for c in passed],
                   passed_geometries=sorted({c["seed"] for c in passed}),
                   geometric_screen_pass=len(passed) >= 3 and len({c["seed"] for c in passed}) >= 2,
                   rejection_counts=dict(Counter(reason for p in alternatives for reason in p["rejected_by"])))
    summary["positive_net_paths"] = sum(p["net_path_saving"] > 0 for p in alternatives)
    summary["net_gain_at_least_2_paths"] = sum(p["net_path_saving"] >= 2 for p in alternatives)
    summary["pass_without_analytic_demand_veto"] = sum(
        not (set(p["rejected_by"]) - {"analytic_bicycle_demand"}) for p in alternatives)
    summary["pass_geometry_only"] = sum(not (set(p["rejected_by"]) - {
        "analytic_bicycle_demand", "no_same_model_peak_steering_improvement",
        "setup_overlaps_existing_avoidance_or_recovery", "endpoint_curvature_debt"}) for p in alternatives)
    summary["positive_gain_pass_without_demand_or_overlap_veto"] = sum(
        p["net_path_saving"] > 0 and not (set(p["rejected_by"]) - {
            "net_saving_below_2", "analytic_bicycle_demand", "no_same_model_peak_steering_improvement",
            "setup_overlaps_existing_avoidance_or_recovery", "endpoint_curvature_debt"}) for p in alternatives)
    summary["relaxed_positive_details"] = [dict(id=c["id"], lead=lead["decisions_before"],
        saving=p["net_path_saving"], offroad=p["estimated_longest_any_wheel_offroad_seconds"],
        reentry_heading=p["reentry_heading_max_deg"], remaining_reasons=p["rejected_by"])
        for c in cases for lead in c["leads"] for p in lead.get("alternatives", [])
        if p["net_path_saving"] > 0 and not (set(p["rejected_by"]) - {
            "net_saving_below_2", "analytic_bicycle_demand", "no_same_model_peak_steering_improvement",
            "setup_overlaps_existing_avoidance_or_recovery", "endpoint_curvature_debt"})]
    summary["best_net_by_lead"] = [dict(id=c["id"], lead=lead["decisions_before"],
        baseline_length=lead["baseline"]["path_length"],
        best=max(lead["alternatives"], key=lambda p: p["net_path_saving"]))
        for c in cases for lead in c["leads"] if lead.get("alternatives")]
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    pins = {n: digest(BUNDLE / n) for n in manifest["files_sha256"] if n == "submission.zip" or n.startswith("source/")}
    if any(h != manifest["files_sha256"][n] for n, h in pins.items()):
        raise ValueError("Frozen champion changed")
    result = dict(scope="Passive consumed TRAIN; no Agent, simulator or oracle runtime",
                  status="GEOMETRIC_GATE_MET_REQUIRES_NATURAL_AND_RUNTIME_CHECKS" if summary["geometric_screen_pass"]
                  else "NO_REPEATED_FEASIBILITY_IMPROVEMENT_DIRECTION_CLOSED",
                  prior_sha256=digest(PRIOR), script_sha256=digest(Path(__file__)),
                  helper_sha256=digest(Path("scripts/diagnose_koi_corner_cutting.py")),
                  frozen_champion=pins, candidate_created=False, new_simulator_resets=0, ab_evaluation="NOT_RUN",
                  summary=summary, cases=cases, definitions=dict(
                      family="C2 Cartesian waypoint spline: gradual setup with every8-unit approach waypoint blending original path toward outside via quintic smoothstep; outside turn-in AND original entry, inside apex, outside exit, original return at end+20. Endpoint quintics preserve locally fitted tangent/curvature jets. Do not replace intervening approach corners by a chord.",
                      grid="Lead10/20/30 actual decisions, outside2/4, inside2/4/6, apex50/65% of geometric corner, turn-in5/10 before geometric start",
                      speed="Same logged PHYSICAL speed-versus-projected-road-station schedule for every path. No claimed preservation of speed in real driving; grass changes force, not a speed multiplier.",
                      endpoints="Match original position, locally fitted dXY/dstation and d2XY/dstation2; velocity tangent is not necessarily hull heading. Geometric jet continuity does not guarantee equal dynamic steering/slip/terminal speed.",
                      demand="Report analytic unfiltered bicycle peak and same legacy .08s-filtered kinematic measure separately; neither proves dynamic feasibility. Do not transport observed steering residual from different entry states.",
                      gate="Net saving>=2, genuinely outside/inward-heading entry at actual projected station crossing, short<=.5s offroad, return/reentry<=15deg, obstacle clearance>=.5, no loops, analytic bicycle angle<=.405/rate<=3.1, >=5% same-model peak-steering improvement vs direct pull, endpoint steering-proxy debt<=.03, no existing avoidance/recovery in full setup window; >=3 cases across>=2 geometries. Natural corroboration/runtime visibility still required before implementation.",
                      limits="Only a bounded fixed-waypoint family; not the optimal racing line, a physical impossibility proof, a measured lap gain or fresh-road generalization. Wheel-center contacts/velocity-tangent hull pose and skipped tiles are proxies."))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "best_net_by_lead"}, indent=2))
    for row in summary["best_net_by_lead"]:
        p = row["best"]
        print(json.dumps(dict(id=row["id"], lead=row["lead"], baseline=row["baseline_length"],
              saving=p["net_path_saving"], outside_entry=p["entry_outside"],
              offroad=p["estimated_longest_any_wheel_offroad_seconds"], reasons=p["rejected_by"])))


if __name__ == "__main__":
    main()
