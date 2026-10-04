"""Passive natural-entry associations in eight archived champion episodes only."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean, median


OFFSETS = (0, 10, 20, 30)
METRICS = (
    "path_length", "centerline_arc", "path_to_arc", "duration_seconds",
    "max_abs_issued_steering", "max_abs_front_wheel_joint_angle_rad",
    "min_physical_speed", "max_any_wheel_offroad_seconds",
    "max_all_wheels_offroad_seconds", "damage_increase", "collision_raw_states",
    "max_abs_reentry_body_heading_deg",
)
RUN = Path("runs/koi-nominal-trajectory-v1")
PRIOR = Path("experiments/koi-corner-cutting-diagnosis-v1.json")
OUTPUT = Path("experiments/koi-corner-entry-natural-v1.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = [v for v in values if v is not None]
    return (dict(n=len(values), min=min(values), median=median(values),
                 mean=mean(values), max=max(values)) if values else dict(n=0))


def lateral_class(outside_positive):
    return "outside" if outside_positive > 1 else "inside" if outside_positive < -1 else "center"


def flags(row):
    c = row["controller"]
    return dict(
        obstacle=bool(c.get("near_object") or c.get("far_active") or c.get("contact_active")
                      or c.get("shield", {}).get("active") or c.get("arrival_cap") is not None),
        impact=bool(c.get("impact_steps_remaining") or c.get("impact_proxy_trigger")
                    or c.get("braking_proxy_veto")),
        recovery=bool(c.get("recovery_changed") or c.get("recovery_steps_remaining")),
    )


def interference(rows):
    counts = {name: sum(flags(r)[name] for r in rows) for name in ("obstacle", "impact", "recovery")}
    return dict(decisions=len(rows), **counts, any_recorded=any(counts.values()))


def snapshot(state, direction, entry_t, row) -> dict:
    c = row["controller"]
    centers = c.get("road_centers", {})
    missing = [key for key in ("30", "42", "54") if key not in centers]
    outside = direction * state["lateral"]
    return dict(
        status="OBSERVED", t=state["t"], decision_step=row["step"],
        seconds_before_geometric_entry=entry_t - state["t"], station=state["station"],
        raw_lateral=state["lateral"], outside_positive_lateral=outside, lateral_class=lateral_class(outside),
        body_heading_error_rad=state["heading_error"], body_heading_error_deg=math.degrees(state["heading_error"]),
        corner_signed_body_heading_error_deg=direction * math.degrees(state["heading_error"]),
        physical_speed=state["speed"], wheel_road_contacts=state["wheel_road_contacts"],
        damage=state["environment_state"]["damage"], front_wheel_joint_angles_rad=[
            w["joint_angle"] for w in state["wheel_state"][:2]],
        issued_action=row["action"], recorded_interference=flags(row), road_centers=centers,
        road_centers_decision_pre_t=c["evaluation_only"]["pre"]["t"],
        pixel_second_difference=None if missing else centers["30"] - 2 * centers["42"] + centers["54"],
        pixel_second_difference_missing_rows=missing,
    )


def entry_history(raw, decisions, corner) -> tuple[dict, int | None]:
    i = next((i for i in range(1, len(raw)) if raw[i - 1]["station"] < corner["start"] <= raw[i]["station"]
              and raw[i]["station"] - raw[i - 1]["station"] < 10), None)
    if i is None:
        return {str(k): dict(status="GEOMETRIC_ENTRY_NOT_REACHED") for k in OFFSETS}, None
    t = raw[i]["t"]
    times = [r["controller"]["evaluation_only"]["pre"]["t"] for r in decisions]
    anchor = bisect_right(times, t) - 1
    history = {"0": snapshot(raw[i], corner["direction"], t, decisions[anchor])}
    history["0"].update(
        raw_index=i, station_overshoot=raw[i]["station"] - corner["start"],
        crossing_bracket_before_t=raw[i - 1]["t"], crossing_bracket_before_station=raw[i - 1]["station"],
        crossing_bracket_before_lateral=raw[i - 1]["lateral"],
        crossing_bracket_before_body_heading_rad=raw[i - 1]["heading_error"],
        anchor_decision_pre_t=times[anchor], anchor_decision_step=decisions[anchor]["step"],
    )
    for offset in OFFSETS[1:]:
        if anchor < offset:
            history[str(offset)] = dict(status="MISSING_PRE_EPISODE_HISTORY", available_prior_decisions=anchor,
                                        requested_prior_decisions=offset)
        else:
            row = decisions[anchor - offset]
            history[str(offset)] = snapshot(row["controller"]["evaluation_only"]["pre"], corner["direction"], t, row)
    return history, anchor


def matching_decisions(decisions, start, end):
    return [r for r in decisions if r["controller"]["evaluation_only"]["post"]["t"] > start
            and r["controller"]["evaluation_only"]["pre"]["t"] < end]


def longest_run(mask, durations):
    best = current = 0.0
    for active, dt in zip(mask, durations):
        current = current + dt if active else 0.0
        best = max(best, current)
    return best


def actual_metrics(window, control, decision_collisions) -> dict:
    dt = [b["t"] - a["t"] for a, b in zip(window, window[1:])]
    any_off = [not all(r["wheel_road_contacts"]) for r in window]
    all_off = [not any(r["wheel_road_contacts"]) for r in window]
    reentries = [dict(t=b["t"], body_heading_error_rad=b["heading_error"],
                     body_heading_error_deg=math.degrees(b["heading_error"]))
                 for a, b in zip(window, window[1:])
                 if not all(a["wheel_road_contacts"]) and all(b["wheel_road_contacts"])]
    path = sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(window, window[1:]))
    arc = window[-1]["station"] - window[0]["station"]
    damages = [r["environment_state"]["damage"] for r in window]
    return dict(
        start_t=window[0]["t"], end_t=window[-1]["t"], raw_states=len(window), raw_intervals=len(dt),
        decisions=len(control), path_length=path, centerline_arc=arc, path_to_arc=path / arc if arc > 0 else None,
        duration_seconds=sum(dt), max_abs_issued_steering=max((abs(r["action"][0]) for r in control), default=None),
        max_abs_front_wheel_joint_angle_rad=max(abs(w["joint_angle"]) for r in window for w in r["wheel_state"][:2]),
        min_physical_speed=min(r["speed"] for r in window), any_wheel_offroad_raw_states=sum(any_off),
        all_wheels_offroad_raw_states=sum(all_off),
        any_wheel_offroad_seconds=sum(d for m, d in zip(any_off[1:], dt) if m),
        all_wheels_offroad_seconds=sum(d for m, d in zip(all_off[1:], dt) if m),
        max_any_wheel_offroad_seconds=longest_run(any_off[1:], dt),
        max_all_wheels_offroad_seconds=longest_run(all_off[1:], dt),
        offroad_left_censored=any_off[0], offroad_right_censored=any_off[-1],
        damage_at_start=damages[0], damage_at_end=damages[-1], damage_increase=max(damages) - damages[0],
        collision_raw_states=sum(bool(r.get("collision") or r["contacts"]) for r in window),
        collision_positive_decisions=sum(decision_collisions[r["step"]] for r in control),
        reentries=reentries, max_abs_reentry_body_heading_deg=max(
            (abs(r["body_heading_error_deg"]) for r in reentries), default=None),
        reentry_status="RETURN_OBSERVED" if reentries else "NO_RETURN_IN_WINDOW" if any(any_off) else "NO_DEPARTURE",
    )


def analyze_event(prior, raw, decisions, decision_collisions, track_length) -> dict:
    event = {k: prior[k] for k in ("id", "track_id", "seed", "start", "end", "apex", "turn_deg",
                                 "max_curvature", "direction", "road_indices", "status")}
    event["prior_baseline_confounded"] = prior.get("baseline_confounded")
    history, anchor = entry_history(raw, decisions, prior)
    event["entry_history"] = history
    event["approach_interference"] = (interference(decisions[max(0, anchor - 30):anchor + 1])
                                      if anchor is not None else None)
    event["approach_history_complete"] = anchor is not None and anchor >= 30
    if anchor is not None:
        start_t = decisions[max(0, anchor - 30)]["controller"]["evaluation_only"]["pre"]["t"]
        approach = [r for r in raw if start_t <= r["t"] <= history["0"]["t"]]
        event["approach_state"] = dict(
            start_t=start_t, end_t=history["0"]["t"], raw_states=len(approach),
            any_wheel_offroad_raw_states=sum(not all(r["wheel_road_contacts"]) for r in approach),
            collision_raw_states=sum(bool(r.get("collision") or r["contacts"]) for r in approach),
            damage_increase=max(r["environment_state"]["damage"] for r in approach)
                            - approach[0]["environment_state"]["damage"],
        )
    if prior["status"] == "COMPLETE":
        times = {r["t"]: i for i, r in enumerate(raw)}
        a, b = times[prior["entry_t"]], times[prior["exit_t"]]
    else:
        start, end = max(1, prior["start"] - 10), min(track_length - 1, prior["end"] + 10)
        a = next((i for i in range(1, len(raw)) if raw[i - 1]["station"] < start <= raw[i]["station"]
                  and raw[i]["station"] - raw[i - 1]["station"] < 10), None)
        b = None if a is None else next((i for i in range(a, len(raw)) if raw[i]["station"] >= end), len(raw) - 1)
        if (a is None) != (prior["status"] == "NOT_REACHED"):
            raise ValueError("Prior censor does not match archived window")
    event["actual"] = None
    event["window_interference"] = None
    if a is not None:
        assert b is not None
        window = raw[a:b + 1]
        control = matching_decisions(decisions, window[0]["t"], window[-1]["t"])
        event["actual"] = actual_metrics(window, control, decision_collisions)
        event["window_interference"] = interference(control)
        event["actual"]["complete_window"] = prior["status"] == "COMPLETE"
        if prior["status"] == "COMPLETE":
            for current, previous in (("path_length", "actual_path_length"), ("centerline_arc", "centerline_arc_length"),
                                      ("max_any_wheel_offroad_seconds", "actual_longest_any_wheel_offroad_seconds"),
                                      ("max_all_wheels_offroad_seconds", "actual_longest_all_wheels_offroad_seconds"),
                                      ("collision_raw_states", "actual_collisions")):
                if not math.isclose(event["actual"][current], prior["baseline"][previous], rel_tol=1e-10, abs_tol=1e-9):
                    raise ValueError(f"Window metric mismatch: {prior['id']} {current}")
    return event


def group_table(events, offset):
    table = {}
    for stratum in ("all", "prior_window_unconfounded", "prior_window_confounded", "no_recorded_window_or_approach_interference"):
        selected = [e for e in events if e["status"] == "COMPLETE"]
        if stratum.startswith("prior_window_"):
            selected = [e for e in selected if e["prior_baseline_confounded"] == (stratum == "prior_window_confounded")]
        elif stratum == "no_recorded_window_or_approach_interference":
            selected = [e for e in selected if e["approach_history_complete"]
                        and not e["window_interference"]["any_recorded"] and not e["approach_interference"]["any_recorded"]
                        and not e["approach_state"]["collision_raw_states"] and not e["approach_state"]["damage_increase"]]
        groups = {}
        for label in ("outside", "center", "inside", "center_or_inside", "missing"):
            group = [e for e in selected if (e["entry_history"][str(offset)].get("lateral_class", "missing")
                     in ("center", "inside") if label == "center_or_inside" else
                     e["entry_history"][str(offset)].get("lateral_class", "missing") == label)]
            groups[label] = dict(n=len(group), geometries=sorted({e["seed"] for e in group}),
                                 ids=[e["id"] for e in group], metrics={
                                     k: stats([e["actual"][k] for e in group]) for k in METRICS})
        table[stratum] = dict(complete_denominator=len(selected), groups=groups)
    return table


def natural_pairs(events, catalogs):
    grouped = defaultdict(list)
    for e in events:
        grouped[(e["seed"], tuple(e["road_indices"]))].append(e)
    pairs = []
    for group in grouped.values():
        for a, b in combinations(group, 2):
            ca, cb = catalogs[(a["track_id"], a["seed"])], catalogs[(b["track_id"], b["seed"])]
            if ca["track"] != cb["track"]:
                raise ValueError("Same-seed corner does not have identical archived geometry")
            comparisons = {}
            for offset in OFFSETS:
                x, y = a["entry_history"][str(offset)], b["entry_history"][str(offset)]
                if x["status"] != "OBSERVED" or y["status"] != "OBSERVED":
                    comparisons[str(offset)] = dict(status="MISSING_HISTORY")
                    continue
                outside_pair = (x["lateral_class"] == "outside") != (y["lateral_class"] == "outside")
                comparisons[str(offset)] = dict(
                    status="OBSERVED", classes=[x["lateral_class"], y["lateral_class"]],
                    outside_positive_lateral=[x["outside_positive_lateral"], y["outside_positive_lateral"]],
                    body_heading_error_deg=[x["body_heading_error_deg"], y["body_heading_error_deg"]],
                    physical_speed=[x["physical_speed"], y["physical_speed"]],
                    entries_differ=abs(x["outside_positive_lateral"] - y["outside_positive_lateral"]) > 1e-9
                                   or abs(x["body_heading_error_deg"] - y["body_heading_error_deg"]) > 1e-9,
                    outside_vs_center_or_inside=outside_pair,
                    outside_id=(a if x["lateral_class"] == "outside" else b)["id"] if outside_pair else None,
                )
                if outside_pair and a["status"] == b["status"] == "COMPLETE":
                    sign = 1 if x["lateral_class"] == "outside" else -1
                    comparisons[str(offset)]["outside_minus_other"] = {
                        k: sign * (a["actual"][k] - b["actual"][k])
                        if a["actual"][k] is not None and b["actual"][k] is not None else None for k in METRICS}
            pairs.append(dict(
                ids=[a["id"], b["id"]], seed=a["seed"], road_indices=a["road_indices"],
                same_geometry=True, obstacle_layout_differs=ca["obstacles"] != cb["obstacles"],
                statuses=[a["status"], b["status"]], both_complete=a["status"] == b["status"] == "COMPLETE",
                both_prior_window_unconfounded=a["prior_baseline_confounded"] is False and b["prior_baseline_confounded"] is False,
                both_full_approach_and_window_without_recorded_interference=all(
                    e["approach_history_complete"] and not e["approach_interference"]["any_recorded"]
                    and e["window_interference"] is not None and not e["window_interference"]["any_recorded"]
                    for e in (a, b)),
                approach_interference=[a["approach_interference"], b["approach_interference"]],
                matched_controller_ab=False, independent=False, comparisons=comparisons,
            ))
    return pairs


def summarize(events, episodes, pairs) -> dict:
    complete = [e for e in events if e["status"] == "COMPLETE"]
    result: dict = dict(episodes=len(episodes), geometries=len({e["seed"] for e in events}),
                  decisions=sum(e["decisions"] for e in episodes), raw_ticks=sum(e["raw_ticks"] for e in episodes),
                  sharp_corner_observations=len(events), event_status=dict(Counter(e["status"] for e in events)),
                  complete=len(complete), prior_window_unconfounded=sum(not e["prior_baseline_confounded"] for e in complete),
                  actual_shorter_than_centerline=sum(e["actual"]["path_to_arc"] < 1 for e in complete),
                  same_geometry_layout_pairs=len(pairs), entry_counts={}, pair_counts={}, descriptive_groups={})
    result["complete_interference_counts"] = {
        period: dict(denominator=len(complete), **{
            flag: sum(bool(e[period][flag]) for e in complete) for flag in ("obstacle", "impact", "recovery", "any_recorded")})
        for period in ("window_interference", "approach_interference")}
    result["complete_outcome_counts"] = dict(
        denominator=len(complete), any_wheel_offroad=sum(e["actual"]["any_wheel_offroad_raw_states"] > 0 for e in complete),
        all_wheels_offroad=sum(e["actual"]["all_wheels_offroad_raw_states"] > 0 for e in complete),
        positive_damage=sum(e["actual"]["damage_increase"] > 0 for e in complete),
        positive_collision=sum(e["actual"]["collision_raw_states"] > 0 or e["actual"]["collision_positive_decisions"] > 0 for e in complete),
        observed_reentry=sum(bool(e["actual"]["reentries"]) for e in complete),
        offroad_right_censored=sum(e["actual"]["offroad_right_censored"] for e in complete),
    )
    for offset in OFFSETS:
        key = str(offset)
        counts = Counter(e["entry_history"][key].get("lateral_class", e["entry_history"][key]["status"]) for e in events)
        result["entry_counts"][key] = dict(denominator=len(events), counts=dict(counts))
        eligible = [p for p in pairs if p["both_complete"] and p["comparisons"][key].get("outside_vs_center_or_inside")]
        both_clean = [p for p in eligible if p["both_prior_window_unconfounded"]]
        shorter_gentler = [p for p in eligible if p["comparisons"][key]["outside_minus_other"]["path_length"] < 0
                          and p["comparisons"][key]["outside_minus_other"]["max_abs_front_wheel_joint_angle_rad"] < 0]
        result["pair_counts"][key] = dict(
            complete_outside_vs_other=len(eligible), geometries=sorted({p["seed"] for p in eligible}),
            both_prior_window_unconfounded=len(both_clean),
            both_prior_window_unconfounded_ids=[p["ids"] for p in both_clean],
            both_full_approach_and_window_without_recorded_interference=sum(
                p["both_full_approach_and_window_without_recorded_interference"] for p in eligible),
            outside_shorter=sum(p["comparisons"][key]["outside_minus_other"]["path_length"] < 0 for p in eligible),
            outside_lower_front_wheel_max=sum(p["comparisons"][key]["outside_minus_other"]["max_abs_front_wheel_joint_angle_rad"] < 0 for p in eligible),
            outside_shorter_and_lower_front_wheel_max=len(shorter_gentler),
            outside_shorter_and_lower_front_wheel_max_geometries=sorted({p["seed"] for p in shorter_gentler}),
            outside_minus_other={k: stats([p["comparisons"][key]["outside_minus_other"][k] for p in eligible]) for k in METRICS},
        )
        result["descriptive_groups"][key] = group_table(events, offset)
    return result


def analyze() -> dict:
    prior = json.loads(PRIOR.read_text())
    paths = sorted(RUN.glob("*-frozen_shield.decisions.jsonl"))
    if len(paths) != 8:
        raise ValueError("Expected exactly eight frozen champion episodes")
    previous = {(e["track_id"], e["seed"]): e for e in prior["episodes"]}
    episodes, catalogs = [], {}
    for path in paths:
        stem = path.name.removesuffix(".decisions.jsonl")
        primary_path, raw_path = path.with_name(stem + ".json"), path.with_name(stem + ".raw.jsonl")
        primary = json.loads(primary_path.read_text())
        old = previous[(primary["track_id"], primary["seed"])]
        inputs = {str(p): digest(p) for p in (path, raw_path, primary_path)}
        if inputs != old["inputs"]:
            raise ValueError("Input triplet differs from prior cutting diagnosis")
        decisions = [json.loads(line) for line in path.read_text().splitlines()]
        raw = [decisions[0]["controller"]["evaluation_only"]["pre"]]
        raw.extend(json.loads(line) for line in raw_path.read_text().splitlines())
        if any(a["t"] >= b["t"] for a, b in zip(raw, raw[1:])):
            raise ValueError("Raw times are not strictly increasing")
        if [r["step"] for r in decisions] != list(range(1, len(decisions) + 1)):
            raise ValueError("Decision history is not contiguous")
        by_time = {r["t"]: r for r in raw}
        for r in decisions:
            for state in r["controller"]["evaluation_only"].values():
                if isinstance(state, dict) and "t" in state:
                    if any(state[k] != by_time[state["t"]][k] for k in ("station", "lateral", "heading_error", "speed")):
                        raise ValueError("Decision pre/post state does not match raw log")
        collisions = {r["step"]: bool(r["collision"]) for r in primary["decision_trace"]}
        track = primary["catalog"]["track"]
        track_length = sum(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(track, track[1:] + track[:1]))
        catalogs[(primary["track_id"], primary["seed"])] = primary["catalog"]
        episodes.append(dict(track_id=primary["track_id"], seed=primary["seed"], completed=primary["completed"],
                             inputs=inputs, decisions=len(decisions), raw_ticks=len(raw) - 1,
                             events=[analyze_event(e, raw, decisions, collisions, track_length) for e in old["events"]]))
    events = [e for episode in episodes for e in episode["events"]]
    if len(events) != 62 or sum(e["status"] == "COMPLETE" for e in events) != 55:
        raise ValueError("Expected preserved 62 events / 55 complete windows")
    pairs = natural_pairs(events, catalogs)
    summary = summarize(events, episodes, pairs)
    four = [e for e in events if e["id"] in prior["summary"]["geometric_gain_only_corners"]]
    if len(four) != 4:
        raise ValueError("Expected four prior clean geometric-gain rejected cases")
    return dict(
        scope="Passive natural-entry association only; existing consumed TRAIN, no counterfactual geometry",
        status="DESCRIPTIVE_ONLY_NO_CAUSAL_OR_IMPLEMENTATION_GATE", new_simulator_resets=0,
        policy_imports_or_execution=False, candidate_created=False, ab_evaluation="NOT_RUN",
        source_sha256=digest(Path(__file__)), prior_artifact=dict(path=str(PRIOR), sha256=digest(PRIOR)),
        definitions=dict(
            windows="Retain all prior statuses. COMPLETE outcomes use exact saved entry_t/exit_t (start-10 to end+10 station crossings). Censored metrics are partial, never pooled with complete outcomes.",
            geometric_entry="First raw tick with previous station < corner.start <= station and forward jump <10; no interpolation. Crossing bracket and overshoot retained; body heading, not velocity/path-tangent heading.",
            history="Anchor is latest decision pre.t <= geometric entry tick. -10/-20/-30 are exactly that many controller decisions before anchor, using logged pre-state. Missing pre-episode history is not padded or wrapped. Offset0 is raw crossing, not anchor pre-state.",
            sign="Positive road-heading curvature is left and raw positive lateral points right. Outside-positive lateral = corner.direction * raw lateral. outside>1, center[-1,1], inside<-1 world units. Direction*body heading error is positive toward the turn.",
            metrics="XY raw-increment path; centerline endpoint station difference; physical speed in world units/s. Issued steering is action[0], front angle is maximum absolute INDIVIDUAL front joint, radians, not target or two-wheel average.",
            timing="Only decision holds with post.t>window.start and pre.t<window.end overlap. Offroad durations sum raw intervals whose RIGHT endpoint lacks any/all wheel support, matching prior convention; boundary-spanning runs are window-limited, not full excursion durations.",
            reentry="Actual transition from any wheel unsupported to all four supported; use archived BODY heading_error at that raw tick. No return is null, not zero. Do not search past fixed window.",
            confounding="Preserve prior baseline_confounded (15/55 false). Additional obstacle flags include arrival_cap; impact includes proxy trigger/veto; recovery flags retained. 30-decision approach flags, offroad/collision/damage and missing-history status are separate. No recorded flag does not establish a randomized/unconfounded passage.",
            pixel_context="Recorded decision road_centers and c30-2*c42+c54 only, missing rows explicit. No runtime detection or policy replay; these do not show that a future sharp corner was identifiable.",
            comparison="Complete-window descriptive means/medians only; all62 status/history counts retained. Same-seed/corner pairs verify identical full track geometry, but differ in obstacle layout/prior state and often entry speed/heading. Repeated layouts, overlapping windows and pair contrasts are NOT independent observations or causal A/B.",
            interpretation="Cross-corner pooled means mix shape/speed/history. Same-geometry endpoints can overshoot stations differently. Brief offroad is measured, not an automatic failure. No fresh-road, official-score, safety guarantee, or measured counterfactual lap-gain claim.",
        ), summary=summary,
        assessment="Natural comparisons are insufficient to isolate an entry-position effect. Some earlier-outside histories have shorter/gentler later windows, but no outside contrast has two full approaches and windows without recorded interference; no causal A/B or candidate gate follows.",
        prior_four_geometric_gain_rejected=four, same_geometry_layout_pairs=pairs, episodes=episodes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, choices=[OUTPUT])
    args = parser.parse_args()
    result = analyze()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result["summary"].items() if k != "descriptive_groups"}, indent=2))


if __name__ == "__main__":
    main()
