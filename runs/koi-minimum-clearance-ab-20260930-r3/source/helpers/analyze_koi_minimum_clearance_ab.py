"""Passive footprint/path analysis on the fixed, explicitly consumed TRAIN cohort."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

from scripts import analyze_koi_adaptive_ab as original


ROOT = original.ROOT
ARMS = ("crossing_projection", "minimum_clearance_v1")
CELLS = original.CELLS
SEEDS = original.SEEDS
ANALYSIS_SPEC = {
    "version": "koi-minimum-clearance-ab-measurement-v1",
    "slots": 48, "cells": 24, "geometry_seeds": 8,
    "physical_passage": original.ANALYSIS_SPEC["physical_passage"],
    "station_continuity": original.ANALYSIS_SPEC["station_continuity"],
    "window": "fixed obstacle station-25 to station+25; interpolated world endpoints; no seam wrap; continuous forward station only",
    "path": "sum world XY chord lengths, clipped at common road window interpolated endpoints; cell mean fractional paired change, then equal-weight cells",
    "eligibility": "same catalog/initial state/10-decision prefix parity, both complete continuous fixed windows and valid physical passage; no detector conditioning; every baseline-eligible window must preserve candidate eligibility or gate fails",
    "return": "after first physical all-fixture rear>radius sample, abs centerline lateral<=1m continuously >=0.24s; confirmation must fall by fixed window exit; onset time minus rear-clear time",
    "return_status": "returned, censored_window_exit, censored_episode_end, unpassed, invalid_window; all retained; no returned-only aggregate",
    "steering": "executed abs steer time-weighted mean/max/integral on exactly clipped window, held over each decision pre/post interval; full coverage required",
    "clearance": "minimum circle-to-all-hull/wheel polygons including Box2D skin; sampled50Hz, not continuous-time safety; whole-episode and physical passage",
    "offroad": "raw ticks any wheel without road contact; side uses signed centerline lateral positive=right, negative=left; road-departure events are transitions into wheel-contact loss",
    "gates": "all48 valid uncensored episodes; no lost finishes; every cell damage/collision decisions nonincrease; no new hit of any whole-episode baseline-no-hit object (including unreached/invalid-passage); every baseline-eligible window retained; >=2% reduction equal-cell-weight matched-window fractional path across >=2 distinct improving geometry seeds; matched finished mean lap nonincrease",
    "descriptive": "lateral extrema, steering, return statuses/lower bounds, departures and baseline crossing-branch-off residual avoidance; projection unavailable distinguished from measured projection off; not speed tuning or official score",
    "safety": "protocol/ledger/episode/raw/ZIP/member/environment/transitive helper source hashes verified; passive only; partial slots retained fail closed",
}
sha256 = original.sha256
stats = original.stats


def read_json(path):
    data = original.read_json(path)

    def finite(value):
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON numeric overflow")

    finite(data)
    return data


def fixed_window(states, obstacle):
    entry = original.common_entry(states, obstacle)
    result = {"entry": entry, "exit": None, "status": entry["reason"], "points": []}
    if entry["t"] is None:
        return result
    target = obstacle["station"] + 25.0
    start = entry["index"]
    end = next((i for i in range(start + 1, len(states))
                if states[i - 1]["station"] < target <= states[i]["station"]), None)
    segment = states[start:(end + 1 if end is not None else len(states))]
    if any(not original.continuous(a, b) for a, b in zip(segment, segment[1:])):
        result["status"] = "nonlocal_reverse_or_seam"
        return result
    if end is None:
        result["status"] = "exit_not_reached"
        return result

    def interpolate(a, b, station):
        fraction = (station - a["station"]) / (b["station"] - a["station"])
        return {key: a[key] + fraction * (b[key] - a[key])
                for key in ("t", "x", "y", "station", "lateral")}

    first = interpolate(states[start], states[start + 1], obstacle["station"] - 25.0)
    last = interpolate(states[end - 1], states[end], target)
    points = [first, *[r for r in states[start + 1:end] if first["t"] < r["t"] < last["t"]], last]
    result.update(status="complete", exit={"t": last["t"], "station": target, "bracket": [states[end - 1]["t"], states[end]["t"]]},
                  points=points, path_length_m=sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(points, points[1:])),
                  duration_s=last["t"] - first["t"], max_abs_lateral_m=max(abs(r["lateral"]) for r in points))
    return result


def steering_window(decisions, start, end):
    integral = covered = maximum = 0.0
    previous_end = start
    for decision in decisions:
        telemetry = decision["controller"]["evaluation_only"]
        a = max(start, telemetry["pre"]["t"])
        b = min(end, telemetry["post"]["t"])
        if b <= a:
            continue
        if abs(a - previous_end) > 1e-7:
            return {"status": "incomplete_or_overlapping_coverage"}
        steer = abs(decision["steer"])
        integral += steer * (b - a)
        covered += b - a
        maximum = max(maximum, steer)
        previous_end = b
    if abs(covered - (end - start)) > 1e-7:
        return {"status": "incomplete_or_overlapping_coverage"}
    return {"status": "complete", "mean_abs": integral / covered, "max_abs": maximum, "integral_abs_s": integral}


def centerline_return(states, physical, window, episode):
    passed = physical["pass_index"]
    result = {"status": "unpassed", "return_time_s": None, "observed_after_clear_s": None,
              "episode_completed": bool(episode["completed"]), "dnf": not episode["completed"] and episode.get("retire_reason") != "max_steps",
              "planned_censor": episode.get("retire_reason") == "max_steps"}
    if passed is None:
        return result
    if window["status"] not in ("complete", "exit_not_reached") or not physical["geometry_valid"]:
        result["status"] = "invalid_window"
        return result
    cutoff = window["exit"]["t"] if window["exit"] else states[-1]["t"]
    if cutoff < physical["pass_t"]:
        result["status"] = "invalid_window"
        return result
    result.update(status="censored_window_exit" if window["exit"] else "censored_episode_end",
                  observed_after_clear_s=max(0.0, cutoff - physical["pass_t"]))
    after_clear = [s for s in states[passed:] if s["t"] <= cutoff + 1e-9]
    result["departures_after_clear"] = departures(after_clear)
    result["endpoint_side"] = "right" if after_clear[-1]["lateral"] > 0 else "left" if after_clear[-1]["lateral"] < 0 else "center"
    onset = None
    previous = None
    for state in states[passed:]:
        if state["t"] > cutoff + 1e-9:
            break
        if previous is not None and not original.continuous(previous, state):
            result["status"] = "invalid_window"
            return result
        if abs(state["lateral"]) <= 1.0:
            if onset is None:
                onset = state["t"]
            if state["t"] - onset >= .24 - 1e-9:
                result.update(status="returned", return_time_s=onset - physical["pass_t"],
                              onset_t=onset, confirmed_t=state["t"])
                break
        else:
            onset = None
        previous = state
    return result


def departures(states):
    ticks = Counter()
    events = Counter()
    active = False
    for state in states:
        off = any(n == 0 for n in state["wheel_road_contacts"])
        side = "right" if state["lateral"] > 0 else "left" if state["lateral"] < 0 else "center"
        if off:
            ticks[side] += 1
            if not active:
                events[side] += 1
        active = off
    return {"offroad_ticks": sum(ticks.values()), "offroad_ticks_by_side": dict(ticks),
            "departure_events_by_side": dict(events), "offroad_at_endpoint": active}


def analyze_episode(episode, raw):
    # The original analyzer is passive and supplies conservative whole-episode hit
    # association and physical fixture passage; its adaptive-speed fields are unused.
    result = original.analyze_episode(episode, raw)
    states = [dict(episode["initial_state"], collision=False), *raw]
    for report in result["obstacles"]:
        window = fixed_window(states, report["obstacle"])
        report["centerline_return"] = centerline_return(states, report["physical"], window, episode)
        if window["status"] == "complete":
            window["steering"] = steering_window(episode["decision_trace"], window["entry"]["t"], window["exit"]["t"])
            window["departures"] = departures([r for r in raw if window["entry"]["t"] <= r["t"] <= window["exit"]["t"]])
        window.pop("points")
        report["fixed_window"] = window
    result.update(**departures(raw), max_abs_centerline_lateral_m=max(abs(r["lateral"]) for r in states),
                  minimum_hull_wheel_clearance_m=min((c for r in states for c in r["clearance"]), default=None))
    diagnostics, branch_off, unavailable = [], [], []
    for decision in episode["decision_trace"]:
        controller = decision["controller"]
        projection = controller.get("clearance_projection_active")
        avoidance = controller.get("clearance_avoidance_steering_active")
        if projection is False and avoidance is True:
            association = original.associate_detection(controller["near_object"], controller["evaluation_only"]["pre"], episode["catalog"]["obstacles"]) if controller.get("near_object") is not None else {"status": "undetected", "obstacle_id": None}
            row = {"step": decision["step"], "steer": decision["steer"],
                                "t": controller["evaluation_only"]["pre"]["t"],
                                "lateral": controller["evaluation_only"]["pre"]["lateral"],
                                "reason": controller.get("minimum_clearance_reason"),
                                "baseline_steer": controller.get("baseline_steer"),
                                "projection_available": controller.get("clearance_projection_available"),
                                "projected_obstacle_x": controller.get("projected_obstacle_x"),
                                "detection_association": association,
                                "recovery_active": controller.get("clearance_recovery_active")}
            branch_off.append(row)
            projected = controller.get("projected_obstacle_x")
            if projected is not None and math.isfinite(projected) and abs(projected - 42) >= 6:
                diagnostics.append(row)
            elif projected is None:
                unavailable.append(row)
    result["projection_off_avoidance_steering"] = diagnostics
    result["crossing_branch_off_avoidance_steering"] = branch_off
    result["projection_unavailable_avoidance_steering"] = unavailable
    result["clearance_changed_decisions"] = sum(bool(d["controller"].get("minimum_clearance_changed")) for d in episode["decision_trace"])
    result["projection_diagnostic_measured_decisions"] = sum("clearance_projection_active" in d["controller"] for d in episode["decision_trace"])
    return result


def verify_models(protocol, run, candidate, manifest):
    receipt = read_json(manifest)
    if sha256(manifest) != protocol["candidate_manifest_sha256"]:
        raise ValueError("candidate manifest hash mismatch")
    if receipt["candidate_zip_sha256"] != protocol["model_hashes"][ARMS[1]]:
        raise ValueError("candidate manifest ZIP mismatch")
    if not receipt.get("candidate", "").startswith("koi-minimum-clearance"):
        raise ValueError("not a minimum-clearance candidate")
    if protocol["model_hashes"][ARMS[0]] != "a4b35c56659555f5e60a372261df722628060e4afcd73c88ed9e9688cdb82bb8":
        raise ValueError("not the immutable baseline source reconstruction")
    for arm, path in ((ARMS[0], run / "crossing-projection-source-reconstruction.zip"), (ARMS[1], candidate)):
        if sha256(path) != protocol["model_hashes"][arm]:
            raise ValueError("model hash mismatch: " + arm)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            expected = protocol["model_source_sha256"][arm]
            if len(set(names)) != len(names) or set(names) != set(expected):
                raise ValueError("model inventory mismatch")
            if any(hashlib.sha256(archive.read(n)).hexdigest() != expected[n] for n in names):
                raise ValueError("model member hash mismatch")
    if receipt["baseline_source_sha256"] != protocol["model_source_sha256"][ARMS[0]]:
        raise ValueError("baseline source reconstruction mismatch")
    if {r["path"]: r["sha256"] for r in receipt["files"]} != protocol["model_source_sha256"][ARMS[1]]:
        raise ValueError("candidate source manifest mismatch")
    baseline = protocol["model_source_sha256"][ARMS[0]]
    candidate_sources = protocol["model_source_sha256"][ARMS[1]]
    if set(candidate_sources) != set(baseline) | {"haic_agent/minimum_clearance_runtime.py"} or any(candidate_sources[n] != h for n, h in baseline.items() if n != "agent.py"):
        raise ValueError("candidate altered a frozen baseline dependency")


def load_run(run, candidate, manifest):
    run = Path(run).resolve()
    protocol = read_json(run / "protocol.json")
    report = read_json(run / "episode-report.json")
    errors = []
    if report["protocol_sha256"] != sha256(run / "protocol.json"):
        raise ValueError("protocol hash mismatch")
    if protocol.get("analysis_spec") != ANALYSIS_SPEC:
        errors.append("analysis_spec_differs")
    for key, path in (("analyzer_sha256", Path(__file__)), ("operator_sha256", ROOT / "scripts/evaluate_koi_minimum_clearance_ab.py")):
        if protocol.get(key) != sha256(path):
            errors.append(key + "_differs")
    if [(r["track_id"], r["geometry_seed"]) for r in protocol["cells"]] != list(CELLS) or protocol["episodes"] != 48:
        raise ValueError("fixed cohort differs")
    if any(r.get("partition") != "TRAIN" or r.get("obstacles") is not True for r in protocol["cells"]):
        raise ValueError("cell conditions differ")
    if any(protocol.get(k) != v for k, v in (("frame_skip", 4), ("warmup_ticks", 50), ("raw_fps", 50), ("max_decisions", 1200))):
        raise ValueError("runtime conditions differ")
    verify_models(protocol, run, candidate, manifest)
    for group in ("environment_source_sha256", "helper_source_sha256"):
        for path, expected in protocol[group].items():
            if sha256(path) != expected:
                raise ValueError("source changed: " + path)
    for suffix, expected in original.FROZEN_SOURCE_SHA256.items():
        matching = [h for p, h in protocol["environment_source_sha256"].items() if p.endswith("/" + suffix)]
        if matching != [expected]:
            raise ValueError("frozen environment source mismatch: " + suffix)
    for path, expected in protocol["consumed_reuse_evidence"]["source_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("consumed evidence changed: " + path)
    for copy in protocol["source_copies"]:
        original.verify_file(run, copy["file"], copy["sha256"])
    copies = {copy["source"]: copy["sha256"] for copy in protocol["source_copies"]}
    if any(copies.get(p) != h for group in ("environment_source_sha256", "helper_source_sha256") for p, h in protocol[group].items()):
        raise ValueError("transitive source copy inventory differs")
    slots: dict[Any, dict[str, Any]] = {(t, s, a): dict(track_id=t, seed=s, mode=a, status="unrun") for t, s in CELLS for a in ARMS}
    seen = set()
    episodes = {}
    for row in report["rows"]:
        key = (row["track_id"], row["seed"], row["mode"])
        if key not in slots or key in seen:
            raise ValueError("unexpected or duplicate slot")
        seen.add(key)
        slots[key].update(row)
        if "file" not in row:
            partial = run / f"{key[0]}-{key[1]}-{key[2]}.raw.jsonl"
            if partial.exists():
                ticks = original.read_raw(partial, partial=True)
                slots[key]["partial_raw"] = dict(file=partial.name, sha256=sha256(partial), complete_ticks=len(ticks), endpoint=ticks[-1] if ticks else None)
            continue
        path = original.verify_file(run, row["file"], row["sha256"])
        episode = read_json(path)
        if (episode["track_id"], episode["seed"], episode["mode"]) != key:
            raise ValueError("episode identity mismatch")
        raw = original.read_raw(original.verify_file(run, episode["raw_trace_file"], episode["raw_trace_sha256"]))
        if len(raw) != episode["raw_ticks"]:
            raise ValueError("raw denominator mismatch")
        if episode.get("catalog") is None or episode.get("initial_state") is None:
            errors.append("missing_initial_geometry:" + str(key))
            continue
        if hashlib.sha256(json.dumps(episode["catalog"], sort_keys=True).encode()).hexdigest() != episode["geometry_sha256"]:
            raise ValueError("geometry hash mismatch")
        states_by_time = {r["t"]: r for r in [episode["initial_state"], *raw]}
        for decision in episode.get("decision_trace") or []:
            for endpoint in ("pre", "post"):
                state = decision["controller"]["evaluation_only"][endpoint]
                reference = states_by_time.get(state["t"])
                if reference is None or any(state[k] != reference[k] for k in ("x", "y", "yaw", "speed", "station", "lateral", "heading_error", "road_index", "front", "rear", "clearance", "contacts", "wheel_road_contacts")):
                    raise ValueError("decision/world telemetry differs from raw samples")
        if (episode.get("error") or episode.get("invalid_actions") or episode.get("retire_reason") in original.OPERATIONAL_RETIREMENTS
                or not episode.get("damage_telemetry_valid") or not episode.get("collision_telemetry_valid")
                or len(episode.get("decision_trace") or []) != episode.get("steps") or episode.get("versions") != protocol["runtime_versions"]):
            errors.append("invalid_episode:" + str(key))
        if not episode.get("runtime_module_paths"):
            errors.append("missing_model_only_receipt:" + str(key))
        for module, module_path in episode.get("runtime_module_paths", {}).items():
            member = "agent.py" if module == "agent" else module.replace(".", "/") + ".py"
            expected = protocol["model_source_sha256"][key[2]]
            if member not in expected and member.replace(".py", "/__init__.py") not in expected:
                raise ValueError("runtime module outside frozen model")
        episodes[key] = (episode, analyze_episode(episode, raw))
    if len(seen) != 48:
        errors.append("not_all48_slots")
    if report.get("operator_error"):
        errors.append("operator_error:" + report["operator_error"])
    ledger_path = original.verify_file(run, "reset-ledger.jsonl", report.get("reset_ledger_sha256"))
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    intents = [(r["track"], r["seed"], r["arm"]) for r in ledger if r["status"] == "reset_intent"]
    expected_intents = [(r["track_id"], r["seed"], r["mode"]) for r in protocol["schedule"]]
    if intents != expected_intents[:len(intents)] or len(set(intents)) != len(intents):
        raise ValueError("reset ledger schedule mismatch")
    if len(intents) != 48:
        errors.append("incomplete_reset_ledger")
    completed_rows = [r for r in ledger if r["status"] == "completed"]
    completed = {(r["track_id"], r["seed"], r["mode"]) for r in completed_rows}
    if len(completed) != len(completed_rows) or any(r["status"] not in {"completed", "reset_intent"} for r in ledger):
        raise ValueError("reset ledger duplicate/unknown event")
    for i, row in enumerate(ledger):
        if row["status"] != ("reset_intent" if i % 2 == 0 else "completed"):
            raise ValueError("reset ledger event order mismatch")
        if i % 2:
            prior = ledger[i - 1]
            key = (row["track_id"], row["seed"], row["mode"])
            if key != (prior["track"], prior["seed"], prior["arm"]) or row.get("sha256") != slots[key].get("sha256") or row.get("file") != slots[key].get("file"):
                raise ValueError("reset ledger completion differs from report")
    if completed != set(episodes):
        errors.append("ledger_episode_mismatch")
    return protocol, list(slots.values()), episodes, errors


def summarize(slots, episodes, errors=()):
    cells = []
    outcomes = Counter()
    new_hits = []
    cell_changes = []
    seed_changes = {}
    baseline_windows = lost_windows = 0
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
                    damage_nonincrease=c["damage"] is not None and b["damage"] is not None and c["damage"] <= b["damage"],
                    collisions_nonincrease=c["collisions"] is not None and b["collisions"] is not None and c["collisions"] <= b["collisions"],
                    lap_delta_ms=c["lapTimeMs"] - b["lapTimeMs"] if outcome == "kept" and c["lapTimeMs"] is not None and b["lapTimeMs"] is not None else None,
                    endpoints={arm: {k: r.get(k) for k in ("completed", "lapTimeMs", "damage", "collisions", "progress", "offroad_ticks", "max_abs_centerline_lateral_m", "minimum_hull_wheel_clearance_m")} for arm, r in zip(ARMS, (b, c))})
        changes = []
        for br, cr in zip(b["obstacles"], c["obstacles"]):
            bw, cw = br["fixed_window"], cr["fixed_window"]
            baseline_eligible = bw["status"] == "complete" and bw["steering"]["status"] == "complete" and br["physical"]["status"] == "passed" and br["physical"]["geometry_valid"]
            candidate_eligible = cw["status"] == "complete" and cw["steering"]["status"] == "complete" and cr["physical"]["status"] == "passed" and cr["physical"]["geometry_valid"]
            baseline_windows += baseline_eligible
            lost_windows += baseline_eligible and not candidate_eligible
            eligible = geometry and initial and parity and baseline_eligible and candidate_eligible
            change = (cw["path_length_m"] - bw["path_length_m"]) / bw["path_length_m"] if eligible and bw["path_length_m"] > 0 else None
            cell["windows"].append(dict(obstacle_id=br["obstacle"]["id"], eligible=eligible, baseline_eligible=baseline_eligible, candidate_eligible=candidate_eligible, fractional_path_change=change,
                                        baseline_window=bw, candidate_window=cw,
                                        baseline_return=br["centerline_return"], candidate_return=cr["centerline_return"],
                                        baseline_physical=br["physical"], candidate_physical=cr["physical"]))
            if change is not None:
                changes.append(change)
            if geometry and not br["whole_episode_safety"]["hit"] and not b["unassociated_collision_events"] and cr["whole_episode_safety"]["hit"]:
                new_hits.append(dict(track_id=track, seed=seed, obstacle_id=br["obstacle"]["id"]))
        cell["fractional_path_change"] = stats(changes)
        if changes:
            mean = sum(changes) / len(changes)
            cell_changes.append(mean)
            seed_changes.setdefault(seed, []).append(mean)
    improving_seeds = sorted(seed for seed, changes in seed_changes.items() if sum(changes) / len(changes) <= -.02 + 1e-12)
    laps = [c["lap_delta_ms"] for c in cells if c.get("lap_delta_ms") is not None]
    gates = dict(all48_valid=len(episodes) == 48 and len(slots) == 48 and all(s["status"] == "completed" for s in slots) and not errors,
                 model_only_pair_safety=all(c.get("geometry_equal") and c.get("initial_equal") and c.get("first10_parity") for c in cells),
                 lost_finishes_zero=outcomes["lost"] == 0 and all(c["pair_available"] for c in cells),
                 per_cell_damage_nonincrease=all(c.get("damage_nonincrease", False) for c in cells),
                 per_cell_collision_nonincrease=all(c.get("collisions_nonincrease", False) for c in cells),
                 baseline_eligible_windows_preserved=baseline_windows > 0 and lost_windows == 0 and all(c["pair_available"] for c in cells),
                 no_episode_censor=all(not value[1]["planned_censor"] for value in episodes.values()) and len(episodes) == 48,
                 no_new_clean_obstacle_hit=not new_hits and all(c.get("geometry_equal") for c in cells) and not any(r[1]["unassociated_collision_events"] for r in episodes.values()),
                 cell_weighted_path_reduction_2pct=bool(cell_changes) and sum(cell_changes) / len(cell_changes) <= -.02 + 1e-12,
                 path_reduction_two_geometry_seeds=len(improving_seeds) >= 2,
                 matched_finished_mean_lap_nonincrease=bool(laps) and len(laps) == outcomes["kept"] and sum(laps) / len(laps) <= 0)
    arms = {}
    for arm in ARMS:
        reports = [value[1] for key, value in episodes.items() if key[2] == arm]
        objects = [o for r in reports for o in r["obstacles"]]
        returns = [o["centerline_return"] for r in reports for o in r["obstacles"]]
        matched = [w["baseline_window" if arm == ARMS[0] else "candidate_window"] for c in cells for w in c["windows"] if w["eligible"]]
        arms[arm] = dict(episodes_available=len(reports), expected_episodes=24, finishes=sum(r["completed"] for r in reports),
                         obstacle_denominator=len(objects), whole_episode_no_hit_objects=sum(not o["whole_episode_safety"]["hit"] for o in objects),
                         clean_completed_physical_passages=sum(o["physical"]["clean"] for o in objects),
                         damage_total=sum(r["damage"] for r in reports if r["damage"] is not None),
                         collision_decisions_total=sum(r["collisions"] for r in reports if r["collisions"] is not None),
                         offroad_ticks=sum(r["offroad_ticks"] for r in reports),
                         departure_events_by_side=dict(sum((Counter(r["departure_events_by_side"]) for r in reports), Counter())),
                         matched_window_path_m=stats([w["path_length_m"] for w in matched]),
                         matched_window_max_abs_lateral_m=stats([w["max_abs_lateral_m"] for w in matched]),
                         matched_window_steering={k: stats([w["steering"][k] for w in matched]) for k in ("mean_abs", "max_abs", "integral_abs_s")},
                         return_status_counts=dict(Counter(r["status"] for r in returns)), return_denominator=len(returns),
                         no_return_after_pass=sum(r["status"] in ("censored_window_exit", "censored_episode_end") for r in returns),
                         no_return_after_pass_by_side=dict(Counter(r["endpoint_side"] for r in returns if r["status"] in ("censored_window_exit", "censored_episode_end"))),
                         return_observations=returns,
                         return_time_summary=stats([r["return_time_s"] for r in returns]) if returns and all(r["status"] == "returned" for r in returns) else None,
                         return_summary_note="No returned-only mean: all censor/unpassed/invalid statuses and observation lower bounds retained.",
                         projection_off_avoidance_steering_decisions=sum(len(r["projection_off_avoidance_steering"]) for r in reports),
                         crossing_branch_off_avoidance_steering_decisions=sum(len(r["crossing_branch_off_avoidance_steering"]) for r in reports),
                         projection_unavailable_avoidance_steering_decisions=sum(len(r["projection_unavailable_avoidance_steering"]) for r in reports),
                         clearance_changed_decisions=sum(r["clearance_changed_decisions"] for r in reports))
    return dict(scope="explicit consumed TRAIN internal proxies, not fresh or official", slots=slots, expected_slots=48, episodes_available=len(episodes),
                completion={**{k: outcomes[k] for k in ("kept", "lost", "gained", "neither")}, "paired_cells": sum(outcomes.values()), "expected_cells": 24},
                cells=cells, arm_summaries=arms, per_cell_mean_fractional_path_change=stats(cell_changes),
                baseline_eligible_windows=baseline_windows, candidate_lost_baseline_eligible_windows=lost_windows,
                improving_geometry_seeds=improving_seeds, per_geometry_fractional_path_change={str(s): stats(v) for s, v in seed_changes.items()},
                mutually_finished_lap_delta_ms=stats(laps), new_clean_obstacle_hits=new_hits,
                gates=gates, gate_passed=all(gates.values()), verification_errors=list(errors), episodes=[value[1] for value in episodes.values()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest or args.candidate.with_suffix(".manifest.json")
    protocol, slots, episodes, errors = load_run(args.run, args.candidate, manifest)
    result = summarize(slots, episodes, errors)
    result.update(analysis_spec=ANALYSIS_SPEC, analyzer_sha256=sha256(__file__), protocol_sha256=sha256(args.run / "protocol.json"),
                  episode_report_sha256=sha256(args.run / "episode-report.json"), model_hashes=protocol["model_hashes"],
                  passive_analysis=True, environment_resets=0, model_updates=0, official_action=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("episodes_available", "completion", "gates", "gate_passed")}, indent=2))


if __name__ == "__main__":
    main()
