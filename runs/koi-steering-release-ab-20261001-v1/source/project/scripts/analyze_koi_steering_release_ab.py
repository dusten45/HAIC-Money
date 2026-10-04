"""Fail-closed passive steering-release comparison on the24 consumed TRAIN cells."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

from scripts import analyze_koi_adaptive_ab as physical
from scripts import analyze_koi_minimum_clearance_ab as frozen

ROOT = physical.ROOT
ARMS = ("crossing_projection", "steering_release_v1")
CELLS, SEEDS = physical.CELLS, physical.SEEDS
BASELINE_SHA = "a4b35c56659555f5e60a372261df722628060e4afcd73c88ed9e9688cdb82bb8"
EXTRA_MEMBERS = {"haic_agent/steering_release_runtime.py", "haic_agent/steering_terms.py"}
ANALYSIS_SPEC = {
    "version": "koi-steering-release-measurement-v1",
    "slots": 48, "cells": 24, "geometry_seeds": 8, "objects_per_arm": 144,
    "window": frozen.ANALYSIS_SPEC["window"],
    "physical_passage": physical.ANALYSIS_SPEC["physical_passage"],
    "station_continuity": physical.ANALYSIS_SPEC["station_continuity"],
    "baseline_windows": 119,
    "eligibility": "all119 baseline complete continuous station+-25 windows with passed valid physical passage and complete steering/component coverage retained by candidate; no detector conditioning or survivor credit",
    "components": "passive frozen-source decomposition before/after every clip and residual; actual avoidance_component held over each executed decision interval; active means abs(component)>1e-12; duration seconds and abs component integral command-s",
    "lateral": "time-mean abs lateral by trapezoids and max abs lateral over interpolated fixed window endpoints and raw50Hz samples",
    "relative": "for each baseline-eligible object (candidate-baseline)/baseline only if baseline>0; zero baseline stays an explicit None, candidate positive is regression, both zero is preserved not a reduction; mean valid relative objects per cell then equally weight eligible cells; reduction metrics require all eligible cells to have a nonzero denominator; geometry seed mean equally weights its eligible cells; path/actual-steer nonincrease separately compares complete matched119-object arithmetic means including zeros",
    "return_legacy": frozen.ANALYSIS_SPEC["return"],
    "return_prospective": "separate after first valid physical rear-clear; cutoff=min(clear+2s,next later obstacle station-25 entry after clear,episode end); next means smallest greater station, never seam wrap; next-entry search begins at rear-clear, independent of legacy initial-seam exclusions; already beyond next-entry at clear means zero followup censor; abs lateral<=1unit sustained>=.24s with confirmation by cutoff; return_time_s is confirmed run onset delay, confirmation_delay_s separately records confirmation delay; never returned-only unlabeled mean",
    "return_censor_bound": "available confirmation_horizon_s retains full cutoff-minus-rear-clear duration; censored onset return_lower_bound_s=max(0,confirmation_horizon_s-.24), confirmation_lower_bound_s=confirmation_horizon_s; pending centerline run start/delay/last-observed time/length only from samples at or before cutoff, never future confirmation or extrapolated run length",
    "return_comparable": "common matched valid rear-clear cohort; truncate both arms to min(available followup seconds); candidate censor count<=baseline; restricted_response_s descriptively mixes observed confirmed onset delays and censored conservative onset lower bounds max(0,common horizon-.24), not a Kaplan-Meier cohort estimate or properly estimated restricted mean; invalid/unpassed/lost comparable objects fail coverage",
    "excess": "observer-only full skin-inclusive hull/wheel lateral separation>0 AND finite projected x abs(x-42)>=6 AND actual avoidance nonzero; require consecutive unique near associations to SAME object and same reacquisition event; dropout/ambiguity/object switch break event; branch off or projection unavailable are separate, not excess proof",
    "timing": "retain first transverse separation in local station+-25 approach (already separated at first entry observation is left censored), first separated sample during actual front/rear longitudinal overlap, rear-clear separately; measured active-to-finite-off epochs persist only across consecutive uniquely associated same-object decisions, subsequent projection unavailable separately labeled not recast as finite off",
    "gates": "all48 valid uncensored,24 geometry/initial/actual10 prefix pairs; no lost finish, per-cell damage/collision decisions nonincrease, no new baseline whole-episode-no-hit object hit across144 objects, all119 baseline windows retained; >=5% equal-cell max lateral reduction across>=2 geometry seeds AND >=5% avoidance duration OR integral reduction across>=2 geometry seeds; path mean and actual steering integral nonincrease, all21 kept mean lap nonincrease, no worsened return censor count on comparable followup; intervention coverage>=2 uniquely stable physical-separated valid-projection-off geometry seeds",
    "threshold": "5% is predeclared local meaningful threshold, not a p-value or fresh generalization claim",
    "invalid": "coverage absent/zero denominators/not valid => gate false, exploratory_scope false, insufficient_intervention where relevant; no promotion",
    "safety": "strict finite JSON (including1e400 overflow); complete raw/decision/event identity; protocol/operator/analyzer/transitive helper/environment142/source-copy/model inventories/ledger/process hashes; partial evidence retained never passes",
}
sha256, read_json = frozen.sha256, frozen.read_json


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON numeric overflow")


def read_jsonl(path, partial=False):
    lines = Path(path).read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            raise ValueError("blank JSONL event")
        try:
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON number: " + x)))
        except json.JSONDecodeError:
            if partial and index == len(lines) - 1:
                return rows, True
            raise
        finite_tree(row)
        if not isinstance(row, dict):
            raise ValueError("JSONL event must be object")
        rows.append(row)
    return rows, False


def statistics(values):
    values = list(values)
    finite_tree(values)
    return dict(n=len(values), min=min(values) if values else None,
                mean=math.fsum(values) / len(values) if values else None,
                max=max(values) if values else None)


def component_window(decisions, start, end):
    integral = duration = covered = 0.0
    last = start
    for decision in decisions:
        controller = decision["controller"]
        interval = controller["evaluation_only"]
        a, b = max(start, interval["pre"]["t"]), min(end, interval["post"]["t"])
        if b <= a:
            continue
        terms = controller.get("steering_terms", {})
        value = terms.get("avoidance_component")
        if terms.get("reconstruction_valid") is not True or terms.get("residual_valid") is not True or terms.get("unidentifiable"):
            return dict(status="unidentified_component_or_residual")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return dict(status="missing_or_invalid_component")
        if abs(a - last) > 1e-7:
            return dict(status="incomplete_or_overlapping_coverage")
        integral += abs(value) * (b - a)
        duration += (b - a) * (abs(value) > 1e-12)
        covered += b - a
        last = b
    if end <= start or abs(covered - (end - start)) > 1e-7:
        return dict(status="incomplete_or_overlapping_coverage")
    return dict(status="complete", duration_s=duration, integral_abs_s=integral, covered_s=covered)


def prospective_return(states, obstacle, physical_passage, obstacles, episode, horizon=None):
    result: dict[str, Any] = dict(status="unpassed", return_time_s=None, observed_after_clear_s=None,
                  episode_completed=bool(episode["completed"]), dnf=not episode["completed"],
                   planned_censor=episode.get("retire_reason") == "max_steps", restricted_response_s=None,
                   restricted_response_kind=None, confirmation_horizon_s=None, confirmation_delay_s=None,
                   confirmation_lower_bound_s=None, return_lower_bound_s=None, pending_centerline_run=None)
    passed, clear = physical_passage["pass_index"], physical_passage["pass_t"]
    if passed is None:
        return result
    if not physical_passage["geometry_valid"]:
        result["status"] = "invalid_physical_passage"
        return result
    later = sorted((o for o in obstacles if o["station"] > obstacle["station"]), key=lambda o: o["station"])
    deadlines = [(clear + 2., "fixed_time"), (states[-1]["t"], "episode_end")]
    next_entry = None
    if later:
        next_entry = physical.common_entry(states[passed:], later[0])
        if next_entry["t"] is not None:
            deadlines.append((next_entry["t"], "next_obstacle_entry"))
        elif next_entry["reason"] == "initially_beyond_entry":
            deadlines.append((clear, "next_obstacle_already_entered"))
        elif next_entry["reason"] not in {"entry_not_reached"}:
            result["status"] = "invalid_next_entry:" + str(next_entry["reason"])
            return result
    if horizon is not None:
        if not math.isfinite(horizon) or horizon < 0:
            raise ValueError("invalid common followup horizon")
        deadlines.append((clear + horizon, "common_followup"))
    cutoff, reason = min(deadlines, key=lambda item: item[0])
    followup = max(0., cutoff - clear)
    # The onset may already exist but still need .24s of future confirmation.
    onset_lower_bound = max(0., followup - .24)
    result.update(status="censored_" + reason, clear_t=clear, cutoff_t=cutoff, cutoff_reason=reason,
                  next_obstacle_id=later[0]["id"] if later else None, next_entry=next_entry,
                  observed_after_clear_s=followup, confirmation_horizon_s=followup,
                  return_lower_bound_s=onset_lower_bound, confirmation_lower_bound_s=followup,
                  restricted_response_s=onset_lower_bound, restricted_response_kind="censored_onset_lower_bound",
                  restriction_is_censored_lower_bound=True)
    onset = None
    previous = None
    for state in states[passed:]:
        if state["t"] > cutoff + 1e-9:
            break
        if previous is not None and not physical.continuous(previous, state):
            result.update(status="invalid_followup_continuity", restricted_response_s=None, restricted_response_kind=None,
                          return_lower_bound_s=None, confirmation_lower_bound_s=None)
            return result
        if abs(state["lateral"]) <= 1.:
            onset = state["t"] if onset is None else onset
            if state["t"] - onset >= .24 - 1e-9:
                delay = onset - clear
                result.update(status="returned", return_time_s=delay, onset_t=onset, confirmed_t=state["t"],
                              restricted_response_s=delay, restriction_is_censored_lower_bound=False,
                              restricted_response_kind="observed_onset_delay", return_lower_bound_s=None,
                              confirmation_delay_s=state["t"] - clear, confirmation_lower_bound_s=None)
                break
        else:
            onset = None
        previous = state
    if result["status"].startswith("censored_") and onset is not None and previous is not None:
        result["pending_centerline_run"] = dict(start_t=onset, onset_delay_s=onset - clear,
                                               last_observed_t=previous["t"], observed_length_s=previous["t"] - onset,
                                               required_length_s=.24, confirmation_fully_observed=False)
    return result


def chronological_diagnostics(episode):
    rows = []
    last_object = None
    event = 0
    event_length = 0
    previous_component = None
    last_projection = None
    off_epoch = None
    active_to_off = False
    for decision in episode["decision_trace"]:
        c = decision["controller"]
        pre = c["evaluation_only"]["pre"]
        association: dict[str, Any] = physical.associate_detection(c["near_object"], pre, episode["catalog"]["obstacles"]) if c.get("near_object") is not None else dict(status="undetected", obstacle_id=None)
        oid = association["obstacle_id"]
        if oid is None or oid != last_object:
            event += 1
            event_length = 0
            previous_component = None
            last_projection = None
            off_epoch = None
            active_to_off = False
        event_length += 1
        stable = oid is not None and oid == last_object and event_length >= 2
        projected = c.get("projected_obstacle_x")
        valid = isinstance(projected, (int, float)) and not isinstance(projected, bool) and math.isfinite(projected)
        projection_off = valid and abs(projected - 42) >= 6
        if valid:
            if projection_off:
                if last_projection != "off":
                    off_epoch = pre["t"]
                    active_to_off = stable and last_projection == "crossing"
                last_projection = "off"
            else:
                last_projection = "crossing"
                off_epoch = None
                active_to_off = False
        separation = pre.get("lateral_separation", [])[oid] if oid is not None and len(pre.get("lateral_separation", [])) > oid else None
        component = c.get("steering_terms", {}).get("avoidance_component")
        active = isinstance(component, (int, float)) and not isinstance(component, bool) and math.isfinite(component) and abs(component) > 1e-12
        separated = isinstance(separation, (int, float)) and math.isfinite(separation) and separation > 0
        baseline_steer = c.get("baseline_steer")
        changed = (c.get("steering_release_changed") is True and isinstance(baseline_steer, (int, float))
                   and math.isfinite(baseline_steer) and abs(decision["steer"] - baseline_steer) > 1e-7)
        rows.append(dict(step=decision["step"], t=pre["t"], event_id=event,
                         detection_association=association, stable_same_object=stable,
                         reacquisition_or_object_switch=oid is not None and not stable,
                         projection_available=valid, valid_projection_off=projection_off,
                         last_valid_projection_state=last_projection, valid_off_epoch_t=off_epoch,
                         observed_active_to_off_epoch=active_to_off,
                         projection_unavailable_after_same_object_off_epoch=stable and not valid and last_projection == "off",
                         full_footprint_lateral_separation_m=separation, full_footprint_laterally_separated=separated,
                         physical_rear_clear=oid is not None and pre["rear"][oid] > episode["catalog"]["obstacles"][oid]["radius"],
                         avoidance_component=component, avoidance_active=active,
                         excess_observed=stable and separated and projection_off and active,
                         continued_avoidance_after_measured_off_epoch=stable and separated and active_to_off and active,
                         release_changed=changed, release_reason=c.get("steering_release_reason"),
                         qualified_release=decision["step"] > 10 and stable and separated and projection_off and changed,
                         avoidance_reversal_same_event=stable and previous_component is not None and active and component * previous_component < 0,
                         steering_terms=c.get("steering_terms"), runtime_gate=c.get("steering_release_gate"),
                         runtime_safe=c.get("steering_release_safe"), actual_steer=decision["steer"]))
        last_object = oid
        previous_component = component if active and oid is not None else None
    return rows


def lateral_timing(states, obstacle):
    oid, radius = obstacle["id"], obstacle["radius"]
    local = [s for s in states if obstacle["station"] - 25 <= s["station"] <= obstacle["station"] + 25
             and abs(s["road_index"] - obstacle["anchor_index"]) <= 15]
    safe = [s for s in local if len(s.get("lateral_separation", [])) > oid and s["lateral_separation"][oid] > 0]
    overlap = [s for s in local if s["front"][oid] >= -radius and s["rear"][oid] <= radius]
    overlap_safe = [s for s in overlap if len(s.get("lateral_separation", [])) > oid and s["lateral_separation"][oid] > 0]
    left_censored = bool(safe and local and safe[0] is local[0])
    return dict(local_approach_first_t=local[0]["t"] if local else None,
                approach_first_separated_t=safe[0]["t"] if safe else None,
                approach_separation_status="left_censored_already_separated" if left_censored else "observed_transition" if safe else "not_observed" if local else "unreached_or_seam",
                longitudinal_overlap_first_t=overlap[0]["t"] if overlap else None,
                overlap_first_separated_t=overlap_safe[0]["t"] if overlap_safe else None,
                overlap_separation_status="left_censored_already_separated" if overlap_safe and overlap_safe[0] is overlap[0] else "observed_transition" if overlap_safe else "not_observed" if overlap else "no_observed_overlap",
                rear_clear_first_t=next((s["t"] for s in local if s["rear"][oid] > radius), None))


def analyze_episode(episode, raw):
    finite_tree(episode)
    finite_tree(raw)
    result = frozen.analyze_episode(episode, raw)
    states = [dict(episode["initial_state"], collision=False), *raw]
    for report in result["obstacles"]:
        window = frozen.fixed_window(states, report["obstacle"])
        if window["status"] == "complete":
            points = window["points"]
            integral = math.fsum(.5 * (abs(a["lateral"]) + abs(b["lateral"])) * (b["t"] - a["t"]) for a, b in zip(points, points[1:]))
            report["fixed_window"]["mean_abs_lateral_m"] = integral / window["duration_s"]
            report["fixed_window"]["avoidance"] = component_window(episode["decision_trace"], window["entry"]["t"], window["exit"]["t"])
        report["prospective_return"] = prospective_return(states, report["obstacle"], report["physical"], episode["catalog"]["obstacles"], episode)
        report["lateral_separation_timing"] = lateral_timing(states, report["obstacle"])
        report["_states"] = states
    result["steering_timeline"] = chronological_diagnostics(episode)
    for report in result["obstacles"]:
        oid = report["obstacle"]["id"]
        durations, integrals, counts = Counter(), Counter(), Counter()
        first_safe = report["lateral_separation_timing"]["approach_first_separated_t"]
        for decision, row in zip(episode["decision_trace"], result["steering_timeline"]):
            if not row["stable_same_object"] or row["detection_association"]["obstacle_id"] != oid:
                continue
            categories = []
            if row["excess_observed"]:
                categories.append("measured_off_and_laterally_separated")
            if row["continued_avoidance_after_measured_off_epoch"]:
                categories.append("same_object_measured_active_to_off_epoch")
            if row["projection_unavailable_after_same_object_off_epoch"] and row["avoidance_active"]:
                categories.append("projection_unavailable_after_off_epoch")
            if first_safe is not None and row["t"] >= first_safe and row["avoidance_active"]:
                categories.append("after_first_local_lateral_separation")
            interval = decision["controller"]["evaluation_only"]
            dt = interval["post"]["t"] - interval["pre"]["t"]
            for category in categories:
                counts[category] += 1
                durations[category] += dt
                integrals[category] += abs(row["avoidance_component"]) * dt
        report["same_object_avoidance_periods"] = {category: dict(decisions=counts[category], duration_s=durations[category], integral_abs_s=integrals[category])
                                                 for category in ("measured_off_and_laterally_separated", "same_object_measured_active_to_off_epoch", "projection_unavailable_after_off_epoch", "after_first_local_lateral_separation")}
    result["release_changed_decisions"] = sum(r["release_changed"] for r in result["steering_timeline"])
    return result


def eligible(report):
    window, passage = report["fixed_window"], report["physical"]
    return (window["status"] == "complete" and window["steering"]["status"] == "complete"
            and window.get("avoidance", {}).get("status") == "complete"
            and passage["status"] == "passed" and passage["geometry_valid"])


def relative_change(baseline, candidate):
    if baseline < 0 or candidate < 0 or not math.isfinite(baseline) or not math.isfinite(candidate):
        raise ValueError("invalid nonnegative metric")
    return (candidate - baseline) / baseline if baseline > 0 else None


def summarize(slots, episodes, errors=()):
    cells, hits, returns, outcomes = [], [], [], Counter()
    metrics = {name: [] for name in ("max_lateral", "avoidance_duration", "avoidance_integral", "path", "actual_steer")}
    seeds = {name: {} for name in metrics}
    windows = lost = 0
    absolute_deltas = {"path": [], "actual_steer": []}
    all_denominators = {name: True for name in metrics}
    qualified_seeds = set()
    laps = []
    for track, seed in CELLS:
        pair = [episodes.get((track, seed, arm)) for arm in ARMS]
        cell: dict[str, Any] = dict(track_id=track, seed=seed, pair_available=all(pair), windows=[])
        cells.append(cell)
        if not all(pair):
            cell["status"] = "incomplete_pair"
            continue
        (base, b), (candidate, c) = pair
        geometry = base["catalog"] == candidate["catalog"] and base["geometry_sha256"] == candidate["geometry_sha256"]
        initial = base["initial_state"] == candidate["initial_state"] and base["initial_observation_sha256"] == candidate["initial_observation_sha256"]
        keys = ("steer", "gas", "brake", "car_x", "car_y", "car_yaw", "progress")
        parity = min(len(base["decision_trace"]), len(candidate["decision_trace"])) >= 10 and [[d.get(k) for k in keys] for d in base["decision_trace"][:10]] == [[d.get(k) for k in keys] for d in candidate["decision_trace"][:10]]
        outcome = "kept" if b["completed"] and c["completed"] else "lost" if b["completed"] else "gained" if c["completed"] else "neither"
        outcomes[outcome] += 1
        cell.update(completion=outcome, geometry_equal=geometry, initial_equal=initial, first10_parity=parity,
                    damage_nonincrease=b["damage"] is not None and c["damage"] is not None and c["damage"] <= b["damage"],
                    collisions_nonincrease=b["collisions"] is not None and c["collisions"] is not None and c["collisions"] <= b["collisions"])
        if outcome == "kept" and b["lapTimeMs"] is not None and c["lapTimeMs"] is not None:
            laps.append(c["lapTimeMs"] - b["lapTimeMs"])
        if any(r["qualified_release"] for r in c["steering_timeline"]):
            qualified_seeds.add(seed)
        bobjects, cobjects = b["obstacles"], c["obstacles"]
        if [r["obstacle"]["id"] for r in bobjects] != [r["obstacle"]["id"] for r in cobjects]:
            raise ValueError("paired object inventory differs")
        changes = {name: [] for name in metrics}
        for br, cr in zip(bobjects, cobjects):
            bw, cw = br["fixed_window"], cr["fixed_window"]
            # Baseline path eligibility cannot be reduced by missing new diagnostics.
            be = (bw["status"] == "complete" and bw["steering"]["status"] == "complete"
                  and br["physical"]["status"] == "passed" and br["physical"]["geometry_valid"])
            ce = eligible(cr)
            windows += be
            lost += be and (not ce or not eligible(br))
            matched = geometry and initial and parity and be and ce and eligible(br)
            fractional = {}
            if matched:
                values = dict(max_lateral=(bw["max_abs_lateral_m"], cw["max_abs_lateral_m"]),
                              avoidance_duration=(bw["avoidance"]["duration_s"], cw["avoidance"]["duration_s"]),
                              avoidance_integral=(bw["avoidance"]["integral_abs_s"], cw["avoidance"]["integral_abs_s"]),
                              path=(bw["path_length_m"], cw["path_length_m"]),
                              actual_steer=(bw["steering"]["integral_abs_s"], cw["steering"]["integral_abs_s"]))
                for name, (bv, cv) in values.items():
                    change = relative_change(bv, cv)
                    fractional[name] = dict(baseline=bv, candidate=cv, fractional_change=change,
                                            zero_denominator=bv == 0, zero_preserved=bv == 0 and cv == 0)
                    if change is not None:
                        changes[name].append(change)
                    elif cv > 0:
                        all_denominators[name] = False
                    if name in absolute_deltas:
                        absolute_deltas[name].append(cv - bv)
            if geometry and not br["whole_episode_safety"]["hit"] and cr["whole_episode_safety"]["hit"]:
                hits.append(dict(track_id=track, seed=seed, obstacle_id=br["obstacle"]["id"]))
            bp, cp = br["prospective_return"], cr["prospective_return"]
            comparable = geometry and initial and parity and bp["observed_after_clear_s"] is not None and cp["observed_after_clear_s"] is not None and not bp["status"].startswith("invalid") and not cp["status"].startswith("invalid")
            common = None
            if comparable:
                horizon = min(bp["observed_after_clear_s"], cp["observed_after_clear_s"])
                paired = [prospective_return(r["_states"], r["obstacle"], r["physical"], e["catalog"]["obstacles"], e, horizon)
                          for r, e in ((br, base), (cr, candidate))]
                comparable = horizon >= .24 - 1e-9 and all(not p["status"].startswith("invalid") for p in paired)
                common = dict(horizon_s=horizon, baseline=paired[0], candidate=paired[1])
            rr = dict(track_id=track, seed=seed, obstacle_id=br["obstacle"]["id"], comparable=comparable, common=common,
                      baseline=bp, candidate=cp, baseline_legacy=br["centerline_return"], candidate_legacy=cr["centerline_return"])
            returns.append(rr)
            cell["windows"].append(dict(obstacle_id=br["obstacle"]["id"], baseline_eligible=be, candidate_eligible=ce,
                                        eligible=matched, metrics=fractional, baseline_window=bw, candidate_window=cw,
                                        baseline_physical=br["physical"], candidate_physical=cr["physical"]))
        cell["relative_metrics"] = {name: statistics(v) for name, v in changes.items()}
        for name, values in changes.items():
            if values:
                mean = math.fsum(values) / len(values)
                metrics[name].append(mean)
                seeds[name].setdefault(seed, []).append(mean)
            elif any(w["baseline_eligible"] for w in cell["windows"]):
                all_denominators[name] = False
    qualifying = {name: sorted(s for s, v in seeds[name].items() if math.fsum(v) / len(v) <= -.05 + 1e-12) for name in metrics}
    matched_cells = sum(any(w["baseline_eligible"] for w in c["windows"]) for c in cells)
    reductions = {name: bool(v) and len(v) == matched_cells and all_denominators[name] and math.fsum(v) / len(v) <= -.05 + 1e-12 and len(qualifying[name]) >= 2 for name, v in metrics.items()}
    comparable = [r for r in returns if r["comparable"]]
    censor_counts = {arm: sum(r["common"][label]["status"] != "returned" for r in comparable) for arm, label in zip(ARMS, ("baseline", "candidate"))}
    baseline_comparable = [r for r in returns if r["baseline"]["observed_after_clear_s"] is not None and not r["baseline"]["status"].startswith("invalid") and r["baseline"]["observed_after_clear_s"] >= .24 - 1e-9]
    gates = dict(all48_valid=len(episodes) == 48 and len(slots) == 48 and all(s["status"] == "completed" for s in slots) and not errors,
                 model_only_pair_safety=all(c.get("geometry_equal") and c.get("initial_equal") and c.get("first10_parity") for c in cells),
                 lost_finishes_zero=outcomes["lost"] == 0 and all(c["pair_available"] for c in cells),
                 per_cell_damage_nonincrease=all(c.get("damage_nonincrease", False) for c in cells),
                 per_cell_collision_nonincrease=all(c.get("collisions_nonincrease", False) for c in cells),
                 no_episode_censor=len(episodes) == 48 and all(not r[1]["planned_censor"] for r in episodes.values()),
                 all144_objects_each=len(episodes) == 48 and all(sum(len(r[1]["obstacles"]) for k, r in episodes.items() if k[2] == arm) == 144 for arm in ARMS),
                 no_new_clean_obstacle_hit=not hits and all(c.get("geometry_equal") for c in cells) and not any(r[1]["unassociated_collision_events"] for r in episodes.values()),
                 baseline119_windows_retained=windows == 119 and lost == 0 and all(c["pair_available"] for c in cells),
                 max_lateral_reduction_5pct_two_geometries=reductions["max_lateral"],
                 avoidance_reduction_5pct_two_geometries=reductions["avoidance_duration"] or reductions["avoidance_integral"],
                 path_mean_nonincrease=len(absolute_deltas["path"]) == 119 and lost == 0 and math.fsum(absolute_deltas["path"]) <= 0,
                 actual_steer_integral_nonincrease=len(absolute_deltas["actual_steer"]) == 119 and lost == 0 and math.fsum(absolute_deltas["actual_steer"]) <= 0,
                 kept21_mean_lap_nonincrease=outcomes["kept"] == 21 and len(laps) == 21 and math.fsum(laps) <= 0,
                 comparable_return_followup_preserved=bool(baseline_comparable) and all(r["comparable"] for r in baseline_comparable),
                 comparable_return_censor_nonincrease=bool(comparable) and censor_counts[ARMS[1]] <= censor_counts[ARMS[0]],
                 sufficient_intervention_coverage=len(qualified_seeds) >= 2)
    if not episodes:
        gates = {name: False for name in gates}
    summaries = {}
    for arm in ARMS:
        reports = [r[1] for k, r in episodes.items() if k[2] == arm]
        objects = [o for r in reports for o in r["obstacles"]]
        prospectives = [o["prospective_return"] for o in objects]
        label = "baseline" if arm == ARMS[0] else "candidate"
        matched = [w[label + "_window"] for c in cells for w in c["windows"] if w["eligible"]]
        summaries[arm] = dict(episodes_available=len(reports), finishes=sum(r["completed"] for r in reports),
                              object_denominator=len(objects), whole_episode_hit_objects=sum(o["whole_episode_safety"]["hit"] for o in objects),
                              damage_total=math.fsum(r["damage"] for r in reports if r["damage"] is not None),
                              collision_decisions_total=sum(r["collisions"] for r in reports if r["collisions"] is not None),
                              raw_collision_ticks=sum(r["raw_collision_ticks"] for r in reports),
                              max_abs_centerline_lateral_m=max((r["max_abs_centerline_lateral_m"] for r in reports), default=None),
                              maximum_matched_window_abs_lateral_m=max((w["max_abs_lateral_m"] for w in matched), default=None),
                              matched_window_path_m=statistics(w["path_length_m"] for w in matched),
                              matched_window_mean_abs_lateral_m=statistics(w["mean_abs_lateral_m"] for w in matched),
                              matched_window_max_abs_lateral_m=statistics(w["max_abs_lateral_m"] for w in matched),
                              matched_window_actual_steer_integral=statistics(w["steering"]["integral_abs_s"] for w in matched),
                              matched_window_actual_steer_time_mean=statistics(w["steering"]["mean_abs"] for w in matched),
                              matched_window_actual_steer_max=statistics(w["steering"]["max_abs"] for w in matched),
                              matched_window_avoidance_duration=statistics(w["avoidance"]["duration_s"] for w in matched),
                              matched_window_avoidance_integral=statistics(w["avoidance"]["integral_abs_s"] for w in matched),
                              legacy_return_status_counts=dict(Counter(o["centerline_return"]["status"] for o in objects)),
                              prospective_return_status_counts=dict(Counter(r["status"] for r in prospectives)),
                              prospective_return_observations=prospectives, prospective_return_denominator=len(objects),
                              return_time_summary=statistics(r["return_time_s"] for r in prospectives) if prospectives and all(r["status"] == "returned" for r in prospectives) else None,
                              comparable_return_censor_count=censor_counts[arm],
                              common_restricted_response=statistics(r["common"][label]["restricted_response_s"] for r in comparable),
                              restricted_note="Descriptive mixture of confirmed onset delays and censored onset lower bounds max(0,confirmation_horizon-.24); not a Kaplan-Meier cohort estimate, properly estimated restricted mean, or returned-only population mean. Pending runs are not confirmed returns.",
                              release_changed_decisions=sum(r["release_changed_decisions"] for r in reports),
                              physical_separated_valid_projection_off_residual_decisions=sum(row["excess_observed"] for r in reports for row in r["steering_timeline"]))
        summaries[arm]["same_object_avoidance_periods"] = {category: dict(decisions=sum(o["same_object_avoidance_periods"][category]["decisions"] for o in objects),
                                                                        duration_s=math.fsum(o["same_object_avoidance_periods"][category]["duration_s"] for o in objects),
                                                                        integral_abs_s=math.fsum(o["same_object_avoidance_periods"][category]["integral_abs_s"] for o in objects))
                                                          for category in ("measured_off_and_laterally_separated", "same_object_measured_active_to_off_epoch", "projection_unavailable_after_off_epoch", "after_first_local_lateral_separation")}
    # Private raw-state references support common-horizon recalculation but are not duplicated in JSON.
    public_episodes = []
    for _, report in episodes.values():
        public_episodes.append({**report, "obstacles": [{k: v for k, v in o.items() if k != "_states"} for o in report["obstacles"]]})
    result = dict(scope="explicit consumed TRAIN internal proxies; not fresh, protected or official", exploratory_scope=False,
                  slots=slots, expected_slots=48, episodes_available=len(episodes), cells=cells,
                  completion={**{k: outcomes[k] for k in ("kept", "lost", "gained", "neither")}, "expected_cells": 24},
                  baseline_eligible_windows=windows, candidate_lost_baseline_eligible_windows=lost,
                  cell_weighted_relative_metrics={name: statistics(v) for name, v in metrics.items()},
                  matched_absolute_metric_deltas={name: statistics(v) for name, v in absolute_deltas.items()},
                  metric_denominators_valid=all_denominators, qualifying_geometry_seeds=qualifying,
                  per_geometry_relative_metrics={name: {str(s): statistics(v) for s, v in mapping.items()} for name, mapping in seeds.items()},
                  qualified_release_geometry_seeds=sorted(qualified_seeds), insufficient_intervention=len(qualified_seeds) < 2,
                  common_return_cohort=returns, common_return_cohort_n=len(comparable), baseline_return_comparable_n=len(baseline_comparable),
                  mutually_finished_lap_delta_ms=statistics(laps), new_clean_obstacle_hits=hits, arm_summaries=summaries,
                  gates=gates, gate_passed=all(gates.values()), verification_errors=list(errors), episodes=public_episodes,
                  interpretation="Absent intervention/measurement coverage is insufficient evidence, not a steering-release improvement; five percent is local meaningful threshold, not significance. Do not promote on survivors or returned-only means.")
    finite_tree(result)
    return result


def verify_models(protocol, run, candidate, manifest):
    receipt = read_json(manifest)
    if not receipt.get("candidate", "").startswith("koi-steering-release-"):
        raise ValueError("not a steering-release manifest")
    if sha256(manifest) != protocol["candidate_manifest_sha256"] or receipt["candidate_zip_sha256"] != protocol["model_hashes"][ARMS[1]]:
        raise ValueError("candidate manifest hash mismatch")
    if protocol["model_hashes"][ARMS[0]] != BASELINE_SHA:
        raise ValueError("immutable baseline differs")
    for arm, path in ((ARMS[0], run / "crossing-projection-source-reconstruction.zip"), (ARMS[1], candidate)):
        if sha256(path) != protocol["model_hashes"][arm]:
            raise ValueError("model hash mismatch")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            expected = protocol["model_source_sha256"][arm]
            if len(names) != len(set(names)) or set(names) != set(expected) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
                raise ValueError("model member inventory differs")
            if any(hashlib.sha256(archive.read(n)).hexdigest() != expected[n] for n in names):
                raise ValueError("model member hash mismatch")
    baseline, treatment = (protocol["model_source_sha256"][a] for a in ARMS)
    manifest_files = {r["path"]: r["sha256"] for r in receipt["files"]}
    if len(manifest_files) != len(receipt["files"]) or manifest_files != treatment or receipt["baseline_source_sha256"] != baseline:
        raise ValueError("model manifest inventory differs")
    if set(treatment) != set(baseline) | EXTRA_MEMBERS or any(treatment[n] != h for n, h in baseline.items() if n != "agent.py"):
        raise ValueError("candidate changed frozen dependencies")
    helpers = protocol["helper_source_sha256"]
    matches = [h for p, h in helpers.items() if p.endswith("/haic/algorithms/koi/steering_terms.py")]
    if matches != [treatment["haic_agent/steering_terms.py"]]:
        raise ValueError("steering terms package/source closure differs")


def validate_episode(episode, raw, key, protocol):
    if (episode["track_id"], episode["seed"], episode["mode"]) != key:
        raise ValueError("episode identity differs")
    validate_raw_rows(raw)
    if len(raw) != episode["raw_ticks"] or len(episode["decision_trace"]) != episode["steps"] or not raw:
        raise ValueError("raw/decision denominator differs")
    catalog = episode["catalog"]
    if hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest() != episode["geometry_sha256"]:
        raise ValueError("geometry hash mismatch")
    states = [episode["initial_state"], *raw]
    count = len(catalog["obstacles"])
    if count != 6 or [o["id"] for o in catalog["obstacles"]] != list(range(count)):
        raise ValueError("object inventory differs")
    if any(len(s[k]) != count for s in states for k in ("front", "rear", "clearance", "lateral_separation")):
        raise ValueError("raw object arrays differ")
    if any(type(i) is not int or i not in range(count) for s in states for i in s["contacts"]):
        raise ValueError("invalid contact event object ID")
    if any(b["t"] <= a["t"] or b["t"] - a["t"] > .020001 for a, b in zip(states, states[1:])):
        raise ValueError("raw timing gap/order differs")
    by_time = {s["t"]: s for s in states}
    previous = states[0]["t"]
    for index, decision in enumerate(episode["decision_trace"], 1):
        interval = decision["controller"]["evaluation_only"]
        if decision["step"] != index or interval["pre"]["t"] != previous:
            raise ValueError("decision event order/coverage differs")
        for endpoint in ("pre", "post"):
            value = interval[endpoint]
            reference = by_time.get(value["t"])
            if reference is None or any(value[k] != reference[k] for k in ("x", "y", "yaw", "speed", "station", "lateral", "heading_error", "road_index", "front", "rear", "clearance", "lateral_separation", "contacts", "wheel_road_contacts")):
                raise ValueError("decision/world telemetry differs from raw")
        ticks = [r for r in raw if interval["pre"]["t"] < r["t"] <= interval["post"]["t"]]
        if not ticks or len(ticks) > 4 or any(r["step"] != index for r in ticks) or decision.get("collision") is not any(r["collision"] for r in ticks):
            raise ValueError("raw eventID/decision collision association differs")
        terms = decision["controller"].get("steering_terms")
        if not isinstance(terms, dict) or terms.get("schema") != protocol["steering_terms_schema"]:
            raise ValueError("steering component schema differs")
        validate_terms(terms, decision["steer"])
        baseline_terms = decision["controller"].get("baseline_steering_terms")
        if not isinstance(baseline_terms, dict) or baseline_terms.get("schema") != protocol["steering_terms_schema"]:
            raise ValueError("baseline steering component schema differs")
        validate_terms(baseline_terms, decision["controller"]["baseline_steer"])
        previous = interval["post"]["t"]
    if previous != raw[-1]["t"] or episode["collisions"] != sum(d["collision"] for d in episode["decision_trace"]):
        raise ValueError("uncovered raw/collision denominator differs")
    for module in episode.get("runtime_module_paths", {}):
        member = "agent.py" if module == "agent" else module.replace(".", "/") + ".py"
        inventory = protocol["model_source_sha256"][key[2]]
        if member not in inventory and member.replace(".py", "/__init__.py") not in inventory:
            raise ValueError("runtime module outside model")


def validate_terms(terms, issued_steer):
    required = {"raw_terms", "stages", "contributions", "residual", "final_sum_residual", "reconstructed_steer", "actual_final_steer", "avoidance_component", "reconstruction_valid", "residual_valid"}
    if not required.issubset(terms) or not isinstance(terms["raw_terms"], dict) or not isinstance(terms["stages"], list) or not isinstance(terms["contributions"], dict):
        raise ValueError("incomplete steering decomposition trace")
    finite_tree(terms)
    if type(issued_steer) not in (int, float) or not math.isfinite(issued_steer) or not -1 <= issued_steer <= 1:
        raise ValueError("invalid issued steering")
    if terms["actual_final_steer"] != issued_steer:
        raise ValueError("steering decomposition issued action differs")
    if terms["reconstruction_valid"] is True:
        if type(terms["reconstructed_steer"]) not in (int, float) or type(terms["avoidance_component"]) not in (int, float):
            raise ValueError("identified steering term is not numeric")
        residual = issued_steer - terms["reconstructed_steer"]
        if abs(residual) > 2e-7 or type(terms["residual"]) not in (int, float) or abs(residual - terms["residual"]) > 1e-12:
            raise ValueError("steering reconstruction residual differs")
    if terms["residual_valid"] is True:
        if any(type(v) not in (int, float) for v in terms["contributions"].values()):
            raise ValueError("identified contribution is not numeric")
        residual = issued_steer - math.fsum(terms["contributions"].values())
        if abs(residual) > 2e-7 or type(terms["final_sum_residual"]) not in (int, float) or abs(residual - terms["final_sum_residual"]) > 1e-12:
            raise ValueError("steering contribution residual differs")
    for stage in terms["stages"]:
        if abs(stage["after"] - stage["before"] - stage["delta"]) > 1e-12:
            raise ValueError("steering clip stage arithmetic differs")


def validate_raw_rows(rows):
    scalar = ("t", "x", "y", "yaw", "speed", "station", "lateral", "heading_error")
    for row in rows:
        if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in scalar):
            raise ValueError("invalid raw scalar telemetry")
        if type(row.get("step")) is not int or row["step"] < 1 or type(row.get("road_index")) is not int or type(row.get("collision")) is not bool:
            raise ValueError("invalid raw eventID/type")
        if row["speed"] < 0 or row["t"] < 0:
            raise ValueError("invalid raw nonnegative telemetry")
        for key in ("front", "rear", "clearance", "lateral_separation"):
            vector = row.get(key)
            if not isinstance(vector, list) or len(vector) != 6 or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                raise ValueError("invalid raw object arrays")
        wheels = row.get("wheel_road_contacts")
        contacts = row.get("contacts")
        if not isinstance(wheels, list) or len(wheels) != 4 or any(type(v) is not int or v < 0 for v in wheels):
            raise ValueError("invalid raw wheel contacts")
        if not isinstance(contacts, list) or len(set(contacts)) != len(contacts) or any(type(v) is not int or v not in range(6) for v in contacts):
            raise ValueError("invalid contact event object ID")
    if any(b["t"] <= a["t"] or b["t"] - a["t"] > .020001 or b["step"] not in (a["step"], a["step"] + 1) for a, b in zip(rows, rows[1:])):
        raise ValueError("raw timing/eventID gap/order differs")


def load_run(run, candidate, manifest):
    from scripts import evaluate_koi_steering_release_ab as operator
    run = Path(run).resolve()
    protocol, report = read_json(run / "protocol.json"), read_json(run / "episode-report.json")
    errors = []
    if report["protocol_sha256"] != sha256(run / "protocol.json"):
        raise ValueError("protocol hash mismatch")
    if protocol["analysis_spec"] != ANALYSIS_SPEC or protocol["analyzer_sha256"] != sha256(__file__) or protocol["operator_sha256"] != sha256(operator.__file__):
        raise ValueError("analysis/operator frozen source differs")
    if [(r["track_id"], r["geometry_seed"]) for r in protocol["cells"]] != list(CELLS) or protocol["episodes"] != 48:
        raise ValueError("fixed cohort differs")
    if any(r.get("partition") != "TRAIN" or r.get("obstacles") is not True for r in protocol["cells"]):
        raise ValueError("cell conditions differ")
    if protocol["schedule"] != operator.scheduled_slots() or any(protocol.get(k) != v for k, v in (("frame_skip", 4), ("warmup_ticks", 50), ("raw_fps", 50), ("max_decisions", 1200))):
        raise ValueError("schedule/runtime conditions differ")
    if len(protocol["environment_source_sha256"]) != 142:
        raise ValueError("environment142 source closure differs")
    if set(protocol["helper_source_sha256"]) != {str(p) for p in operator.helper_paths()}:
        raise ValueError("helper source closure differs")
    verify_models(protocol, run, candidate, manifest)
    for group in ("environment_source_sha256", "helper_source_sha256"):
        for path, expected in protocol[group].items():
            if sha256(path) != expected:
                raise ValueError("source changed: " + path)
    for suffix, expected in physical.FROZEN_SOURCE_SHA256.items():
        if [h for p, h in protocol["environment_source_sha256"].items() if p.endswith("/" + suffix)] != [expected]:
            raise ValueError("frozen environment differs: " + suffix)
    for path, expected in protocol["consumed_reuse_evidence"]["source_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("consumed evidence changed")
    copies = {}
    for row in protocol["source_copies"]:
        physical.verify_file(run, row["file"], row["sha256"])
        if row["source"] in copies:
            raise ValueError("duplicate source copy")
        copies[row["source"]] = row["sha256"]
    if any(copies.get(p) != h for group in ("environment_source_sha256", "helper_source_sha256") for p, h in protocol[group].items()):
        raise ValueError("transitive source copies differ")
    slots: dict[Any, dict[str, Any]] = {(t, s, a): dict(track_id=t, seed=s, mode=a, status="unrun") for t, s in CELLS for a in ARMS}
    seen, episodes = set(), {}
    for row in report["rows"]:
        key = row["track_id"], row["seed"], row["mode"]
        if key not in slots or key in seen:
            raise ValueError("duplicate/unknown slot")
        seen.add(key)
        slots[key].update(row)
        if "file" not in row:
            partial = run / f"{key[0]}-{key[1]}-{key[2]}.raw.jsonl"
            if partial.exists():
                ticks, truncated = read_jsonl(partial, partial=True)
                validate_raw_rows(ticks)
                slots[key]["partial_raw"] = dict(file=partial.name, sha256=sha256(partial), complete_ticks=len(ticks), truncated_last_line=truncated, endpoint=ticks[-1] if ticks else None)
            continue
        path = physical.verify_file(run, row["file"], row["sha256"])
        episode = read_json(path)
        process = read_json(physical.verify_file(run, row["process_file"], row["process_sha256"]))
        if process["status"] != "completed" or process["error"] or process["slot_id"] != operator.slot_id(*key) or process["operator_sha256"] != protocol["operator_sha256"]:
            raise ValueError("process event/source receipt differs")
        raw, _ = read_jsonl(physical.verify_file(run, episode["raw_trace_file"], episode["raw_trace_sha256"]))
        original_process = physical.verify_file(run, process["original_process_file"], process["original_process_sha256"])
        original_receipt = read_json(original_process)
        if any(process[k] != original_receipt[k] for k in ("status", "error", "wall_time_s")):
            raise ValueError("bound process differs from actual process")
        validate_episode(episode, raw, key, protocol)
        if episode.get("error") or episode.get("invalid_actions") or episode.get("retire_reason") in physical.OPERATIONAL_RETIREMENTS or not episode.get("damage_telemetry_valid") or not episode.get("collision_telemetry_valid") or episode.get("versions") != protocol["runtime_versions"] or not episode.get("runtime_module_paths"):
            errors.append("invalid_episode:" + str(key))
        episodes[key] = episode, analyze_episode(episode, raw)
    if len(seen) != 48:
        errors.append("not_all48_slots")
    if report.get("operator_error"):
        errors.append("operator_error:" + report["operator_error"])
    ledger, _ = read_jsonl(physical.verify_file(run, "reset-ledger.jsonl", report["reset_ledger_sha256"]))
    expected = [(r["track_id"], r["seed"], r["mode"]) for r in protocol["schedule"]]
    completed = set()
    for index, event in enumerate(ledger):
        position = index // 2
        if position >= 48 or event.get("status") != ("reset_intent" if index % 2 == 0 else "completed"):
            raise ValueError("reset ledger duplicate/unknown/event order")
        key = (event["track"], event["seed"], event["arm"]) if index % 2 == 0 else (event["track_id"], event["seed"], event["mode"])
        if key != expected[position] or event.get("slot_id") != operator.slot_id(*key):
            raise ValueError("reset ledger eventID/schedule association differs")
        if index % 2:
            if event != {k: v for k, v in slots[key].items() if k != "partial_raw"}:
                raise ValueError("ledger completion differs from report")
            completed.add(key)
    if completed != set(episodes):
        raise ValueError("ledger episode association differs")
    if len(ledger) != 96:
        errors.append("incomplete_reset_ledger")
    return protocol, list(slots.values()), episodes, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    protocol, slots, episodes, errors = load_run(args.run, args.candidate, args.manifest or args.candidate.with_suffix(".manifest.json"))
    result = summarize(slots, episodes, errors)
    result.update(analysis_spec=ANALYSIS_SPEC, analyzer_sha256=sha256(__file__), protocol_sha256=sha256(args.run / "protocol.json"),
                  episode_report_sha256=sha256(args.run / "episode-report.json"), model_hashes=protocol["model_hashes"],
                  passive_analysis=True, environment_resets=0, model_updates=0, official_action=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("episodes_available", "completion", "gates", "gate_passed", "insufficient_intervention")}, indent=2))


if __name__ == "__main__":
    main()
