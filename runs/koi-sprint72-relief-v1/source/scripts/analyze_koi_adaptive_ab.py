"""Passive, fail-closed analysis of the frozen KOI consumed-TRAIN A/B receipts.

No Agent, environment, renderer, or evaluator is imported or executed. All world
state is evaluation-only evidence, never a runtime input or a tuning signal.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("crossing_projection", "adaptive_v1")
SEEDS = (38300, 38301, 38302, 38303, 50300, 50301, 50302, 50303)
CELLS = tuple((track, seed) for seed in SEEDS for track in (1, 2, 3))
CANDIDATE_SHA256 = "5f7057a432074d2e7215f5d2aeb863aafc4d557d9ba93c9fb3d34ee2b46e679c"
MANIFEST_SHA256 = "01e72476b10fe7fe75189a35783cac3bc686226b9af0fe1d27c8848a0fb90b3e"
OPERATIONAL_RETIREMENTS = {"act_timeout", "reset_timeout", "agent_reset_error", "agent_setup_error", "evaluation_error", "invalid_action", "unknown"}
FROZEN_SOURCE_SHA256 = {
    "core/vendor/car_racing.py": "bf7d7a16690568c96d454322f5bba83c79bd18fb5b44eaf66d6293493058593d",
    "haic_agent/pixel_features.py": "f68895b9c0297ac5075ea47ab7f2a868e739b0ad24bee980082ce818b661d22a",
    "env_wrapper.py": "d76a03bd9b051df502f332eef5a2a3b77a607bda458890a032785469cf21c39d",
    "haic_agent/far_hazard_runtime.py": "12fc519edb1e3ac33e1f339ebf5dcf95913a529d51705cbd827e9eff2557f6d4",
}
ANALYSIS_SPEC = {
    "version": "koi-adaptive-ab-measurement-v1",
    "slots": 48, "cells": 24, "geometry_seeds": 8,
    "detector_state": "controller.evaluation_only.pre; near_object and far_object or far_objects[0]; no tracked predictions",
    "camera": "rotate displacement by -yaw; zoom=.6*max(1-t,0)+16.2*min(t,1); window=(500+zoom*dx,200+zoom*dy); flip y=799-y; pixel=(84*x/1000-.0625,84*y/800-.0625)",
    "camera_note": "continuous camera center, not exact rasterized bright centroid; fixed 4px absorbs rasterization/resampling",
    "match_radius_px": 4.0,
    "matching": "exactly one center within inclusive4px; unmatched and ambiguous retained; ambiguous candidates exclude their obstacle pairs",
    "physical_passage": "first local forward front crossing -radius to first rear>radius; all hull/wheel polygon skins in raw front/rear; retain incomplete encounters",
    "physical_locality": "abs(station-obstacle.station)<=40 and abs(road_index-anchor_index)<=15 at approach crossing; no seam wrapping",
    "sampling": "50Hz; passage includes crossing bracket and rear-clear tick; no sub-tick cleanliness claim; pass time is first strictly clear sample",
    "clean": "complete passage, every bracketed raw clearance>0, no associated Box2D contact, no raw collision, no overlapping decision collision",
    "common_entry": "first forward interpolated station crossing obstacle.station-25; reject negative/seam or initially beyond threshold; reject reverse/nonlocal/discontinuous entry-to-pass",
    "station_continuity": "dt in(0,.020001]; road_index change<=4; station increment>=-1e-6 and <=hypot(dx,dy)+7; no wrap",
    "segment_eligibility": "same geometry; both unique detected by pass, no ambiguity, valid common entry and passed; detection may precede common entry",
    "time_gate": "strictly negative mean of per-cell mean matched common-entry-to-pass deltas; nonempty; also report segment-weighted mean",
    "speed_gate": ">=2 distinct geometry seed values with candidate clean physical MIN actual raw speed>44m/s and uniquely associated adaptive+changed decision between common entry and pass",
    "other_gates": "all48 valid slots, no operational retirements; lost finishes=0; per-cell damage AND collision decisions nonincrease; no whole-episode baseline-clean obstacle newly hit, including later reentry; fail closed on unassociated collision events in either arm",
    "hud": "44 is controller/HUD intended native speed target, not proof of actual44; estimate scale from pixel_speed vs mean(previous,current pre.speed); initial pre duplicated; clipped80 samples excluded",
    "failure": "last10 decision endpoints, contact IDs, lateral/heading extrema, wheel road contacts, wrapper reasons; observational association not causal attribution",
    "safety": "verify protocol/episode/raw/model/member/source hashes and first10 action/world parity; passive model-only evaluation, no tuning/reset/import of models or env",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path) -> Any:
    def reject(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def read_raw(path, partial=False) -> list[dict[str, Any]]:
    rows = []
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if partial and i == len(lines) - 1:
                break
            raise
        for key in ("t", "x", "y", "yaw", "speed", "station", "lateral", "heading_error", "front", "rear", "clearance"):
            if not np.all(np.isfinite(row[key])):
                raise ValueError(f"nonfinite raw telemetry: {key}")
        rows.append(row)
    return rows


def project_center(obstacle: dict[str, Any], state: dict[str, Any]):
    """Frozen affine camera, vertical reflection, and final resize coordinates."""
    dx, dy = obstacle["x"] - state["x"], obstacle["y"] - state["y"]
    c, s = math.cos(state["yaw"]), math.sin(state["yaw"])
    zoom = .6 * max(1 - state["t"], 0) + 16.2 * min(state["t"], 1)
    x = 500 + zoom * (dx * c + dy * s)
    y = 799 - (200 + zoom * (-dx * s + dy * c))
    return [84 * x / 1000 - .0625, 84 * y / 800 - .0625]


def associate_detection(obj, state, obstacles) -> dict[str, Any]:
    center = [float(obj[1]), float(obj[0])]
    candidates = []
    for obstacle in obstacles:
        projected = project_center(obstacle, state)
        distance = math.dist(center, projected)
        if distance <= ANALYSIS_SPEC["match_radius_px"]:
            candidates.append({"id": obstacle["id"], "distance_px": distance, "projected": projected})
    return {"status": "matched" if len(candidates) == 1 else "ambiguous" if candidates else "unmatched",
            "center": center, "candidates": candidates,
            "obstacle_id": candidates[0]["id"] if len(candidates) == 1 else None}


def stats(values):
    return {"n": len(values), "min": min(values) if values else None,
            "mean": float(np.mean(values)) if values else None, "max": max(values) if values else None}


def continuous(a, b):
    dt = b["t"] - a["t"]
    ds = b["station"] - a["station"]
    return (0 < dt <= .020001 and abs(b["road_index"] - a["road_index"]) <= 4
            and -1e-6 <= ds <= math.hypot(b["x"] - a["x"], b["y"] - a["y"]) + 7)


def common_entry(states, obstacle):
    threshold = obstacle["station"] - 25.0
    if threshold < 0:
        return {"t": None, "index": None, "reason": "seam_entry"}
    if not states or states[0]["station"] >= threshold:
        return {"t": None, "index": None, "reason": "initially_beyond_entry"}
    for i in range(1, len(states)):
        a, b = states[i - 1], states[i]
        if a["station"] < threshold <= b["station"]:
            if not continuous(a, b):
                return {"t": None, "index": i, "reason": "discontinuous_crossing"}
            amount = (threshold - a["station"]) / (b["station"] - a["station"])
            return {"t": a["t"] + amount * (b["t"] - a["t"]), "index": i - 1,
                    "reason": None, "station": threshold, "bracket": [a["t"], b["t"]]}
    return {"t": None, "index": None, "reason": "entry_not_reached"}


def passage(states, obstacle, index, decisions) -> dict[str, Any]:
    """Keep the first local encounter even when the episode never clears it."""
    radius = obstacle["radius"]
    start = end = None
    for i in range(1, len(states)):
        a, b = states[i - 1], states[i]
        local = (abs(b["station"] - obstacle["station"]) <= 40
                 and abs(b["road_index"] - obstacle["anchor_index"]) <= 15)
        if (local and a["front"][index] < -radius <= b["front"][index]
                and b["front"][index] > a["front"][index]):
            start = i
            break
    if start is not None:
        end = next((i for i in range(start, len(states)) if states[i]["rear"][index] > radius), None)
    window = states[max(0, start - 1):(end + 1 if end is not None else len(states))] if start is not None else []
    overlapping = [d for d in decisions if window and d["controller"]["evaluation_only"]["post"]["t"] >= window[0]["t"]
                   and d["controller"]["evaluation_only"]["pre"]["t"] <= window[-1]["t"]]
    minimum_clearance = min((r["clearance"][index] for r in window), default=None)
    contacts = [r["t"] for r in window if obstacle["id"] in r["contacts"]]
    collision = any(r["collision"] for r in window) or any(d.get("collision") is not False for d in overlapping)
    hit = minimum_clearance is not None and (minimum_clearance <= 0 or bool(contacts) or collision)
    geometry_valid = bool(window) and all(continuous(a, b) for a, b in zip(window, window[1:]))
    if end is not None:
        assert start is not None
        geometry_valid = geometry_valid and abs(states[end]["station"] - obstacle["station"]) <= 40
    return {"start_t": states[start]["t"] if start is not None else None,
            "start_bracket": [states[start - 1]["t"], states[start]["t"]] if start is not None else None,
            "pass_t": states[end]["t"] if end is not None else None,
            "pass_bracket": [states[max(start or 0, end - 1)]["t"], states[end]["t"]] if end is not None else None,
            "start_index": start, "pass_index": end,
            "status": "passed" if end is not None else "incomplete" if start is not None else "not_reached",
            "endpoint_t": states[-1]["t"] if states else None,
            "elapsed_to_endpoint_s": states[-1]["t"] - states[start]["t"] if start is not None else None,
            "physical_speed_mps": stats([r["speed"] for r in window]),
            "minimum_clearance": minimum_clearance, "contact_times": contacts, "collision": collision,
            "geometry_valid": geometry_valid,
            "hit": bool(hit), "clean": geometry_valid and end is not None and minimum_clearance is not None and minimum_clearance > 0
            and not contacts and not collision}


def failure_context(episode, raw):
    decisions = episode.get("decision_trace") or []
    last = decisions[-10:]
    cutoff = last[0]["controller"]["evaluation_only"]["pre"]["t"] if last else -math.inf
    window = [r for r in raw if r["t"] >= cutoff]
    endpoints = [{"step": d["step"], "collision": d.get("collision"), "damage": d.get("damage"),
                  "adaptive_reason": d["controller"].get("adaptive_speed_reason"),
                  "evaluation_only": d["controller"]["evaluation_only"]} for d in last]
    flags = {k: any(d["controller"]["evaluation_only"].get(k, False) for d in last)
             for k in ("wrapper_off_track", "wrapper_crashed", "simulator_terminated")}
    contact_ids = sorted({oid for r in window for oid in r["contacts"]})
    labels = []
    if contact_ids:
        labels.append("obstacle_contact_associated")
    if flags["wrapper_off_track"]:
        labels.append("wrapper_off_track")
    if flags["wrapper_crashed"]:
        labels.append("wrapper_crashed")
    if flags["simulator_terminated"]:
        labels.append("simulator_terminated")
    if any(any(n == 0 for n in r["wheel_road_contacts"]) for r in window):
        labels.append("wheel_road_contact_loss_observed")
    return {"retire_reason": episode.get("retire_reason"), "error": episode.get("error"),
            "completed": episode.get("completed", False), "labels": labels, "contact_obstacle_ids": contact_ids,
            "lateral": stats([r["lateral"] for r in window]),
            "absolute_heading_rad": stats([abs(r["heading_error"]) for r in window]),
            "wheel_road_contacts_endpoint": window[-1]["wheel_road_contacts"] if window else None,
            "wrapper_flags": flags, "last_decisions": endpoints,
            "causal_conclusion": None, "note": "terminal temporal association only; no inferred cause"}


def analyze_episode(episode: dict[str, Any], raw: list[dict[str, Any]]) -> dict[str, Any]:
    obstacles = episode["catalog"]["obstacles"]
    decisions = episode["decision_trace"]
    states = [dict(episode["initial_state"], collision=False), *raw]
    if [o["id"] for o in obstacles] != list(range(len(obstacles))):
        raise ValueError("obstacle IDs must preserve frozen raw array order")
    if not all(len(r[k]) == len(obstacles) for r in states for k in ("front", "rear", "clearance")):
        raise ValueError("raw obstacle arrays do not match catalog")
    if any(b["t"] <= a["t"] for a, b in zip(states, states[1:])):
        raise ValueError("raw timestamps are not strictly ordered")
    if any(d["step"] != i for i, d in enumerate(decisions, 1)):
        raise ValueError("decision sequence is not contiguous")
    by_id = {o["id"]: {"near": [], "far": [], "ambiguous": []} for o in obstacles}
    unassociated_collisions = [{"t": r["t"], "step": r["step"], "source": "raw"} for r in raw
                              if r["collision"] and not r["contacts"] and not any(c <= 0 for c in r["clearance"])]
    for d in decisions:
        if d.get("collision") and not any(r["step"] == d["step"] and r["collision"] for r in raw):
            unassociated_collisions.append({"step": d["step"], "source": "decision_without_raw_collision"})
    events = []
    reasons = Counter()
    changed = []
    speed_pairs = []
    previous_speed = states[0]["speed"]
    for d in decisions:
        controller = d["controller"]
        state = controller["evaluation_only"]["pre"]
        reason = controller.get("adaptive_speed_reason", "baseline")
        reasons[reason] += 1
        if controller.get("adaptive_speed_changed", False):
            changed.append({"step": d["step"], "t": state["t"], "reason": reason,
                            "baseline_pedals": controller.get("baseline_pedals"),
                            "gas": d["gas"], "brake": d["brake"],
                            "target_hud": controller.get("adaptive_speed_target"),
                            "pass_target_hud": controller.get("adaptive_pass_speed")})
        pixel_speed = controller.get("pixel_speed")
        if pixel_speed is not None and 0 < pixel_speed < 80:
            speed_pairs.append((.5 * (previous_speed + state["speed"]), pixel_speed))
        previous_speed = state["speed"]
        far = controller.get("far_object") if "far_object" in controller else next(iter(controller.get("far_objects") or []), None)
        for kind, obj in (("near", controller.get("near_object")), ("far", far)):
            if obj is None:
                continue
            event = dict(associate_detection(obj, state, obstacles), kind=kind, step=d["step"], t=state["t"],
                         adaptive_active=reason == "adaptive", adaptive_changed=bool(controller.get("adaptive_speed_changed", False)))
            events.append(event)
            if event["status"] == "matched":
                by_id[event["obstacle_id"]][kind].append(event)
            elif event["status"] == "ambiguous":
                for candidate in event["candidates"]:
                    by_id[candidate["id"]]["ambiguous"].append(event)
    reports = []
    for i, o in enumerate(obstacles):
        detections = by_id[o["id"]]
        physical = passage(states, o, i, decisions)
        entry = common_entry(states, o)
        pass_t = physical["pass_t"]
        exclusions = []
        if entry["t"] is None:
            exclusions.append(entry["reason"])
        elif physical["pass_index"] is not None:
            segment = states[entry["index"]:physical["pass_index"] + 1]
            if entry["t"] > pass_t or any(not continuous(a, b) for a, b in zip(segment, segment[1:])):
                exclusions.append("nonlocal_or_reverse_segment")
        first = {kind: detections[kind][0] if detections[kind] else None for kind in ("near", "far")}
        valid_detections = [e for kind in ("near", "far") for e in detections[kind] if pass_t is not None and e["t"] <= pass_t]
        if not valid_detections:
            exclusions.append("no_unique_detection_by_pass")
        if detections["ambiguous"]:
            exclusions.append("ambiguous_detection")
        if pass_t is None:
            exclusions.append("not_passed")
        elif not physical["geometry_valid"]:
            exclusions.append("nonlocal_or_reverse_physical_passage")
        activation = [e for e in detections["near"] if e["adaptive_active"] and e["adaptive_changed"]
                      and entry["t"] is not None and pass_t is not None and entry["t"] <= e["t"] <= pass_t]
        global_contacts = [r["t"] for r in states if o["id"] in r["contacts"]]
        global_geometric_hits = [r["t"] for r in states if r["clearance"][i] <= 0]
        whole_episode_hit = bool(global_contacts or global_geometric_hits)
        whole_episode = {"hit": whole_episode_hit, "contact_times": global_contacts,
                         "geometric_hit_times": global_geometric_hits,
                         "minimum_clearance": min(r["clearance"][i] for r in states),
                         "baseline_clean_reference": physical["clean"] and not whole_episode_hit and not unassociated_collisions}
        timings = {}
        for kind, event in first.items():
            timings[kind] = {"detection_t": event["t"] if event else None,
                             "detection_to_pass_s": pass_t - event["t"] if event and pass_t is not None and pass_t >= event["t"] else None,
                             "elapsed_to_endpoint_s": states[-1]["t"] - event["t"] if event else None,
                             "status": "undetected" if event is None else "passed" if pass_t is not None and pass_t >= event["t"] else "unpassed_or_postpass_detection"}
        reports.append({"obstacle": o, "first_detection": first, "detection_times": timings,
                        "ambiguous_events": detections["ambiguous"], "common_entry": entry, "physical": physical,
                        "whole_episode_safety": whole_episode,
                        "segment_eligible": not exclusions, "segment_exclusions": exclusions,
                        "common_entry_to_pass_s": pass_t - entry["t"] if not exclusions and pass_t is not None and entry["t"] is not None else None,
                        "associated_adaptive_changed_steps": [e["step"] for e in activation],
                        "clean_actual_above44": physical["clean"] and physical["physical_speed_mps"]["min"] > 44,
                        "qualifying_adaptive_above44": not exclusions and physical["clean"]
                        and physical["physical_speed_mps"]["min"] > 44 and bool(activation)})
    actual = np.asarray([p[0] for p in speed_pairs])
    hud = np.asarray([p[1] for p in speed_pairs])
    scale = float(actual @ hud / (actual @ actual)) if len(actual) and actual @ actual > 0 else None
    return {"mode": episode["mode"], "track_id": episode["track_id"], "seed": episode["seed"],
            "completed": episode["completed"], "lapTimeMs": episode.get("lapTimeMs"),
            "simulation_start": states[0]["t"], "simulation_end": states[-1]["t"],
            "damage": episode.get("damage"), "collisions": episode.get("collisions"),
            "progress": episode.get("progress"), "obstacles": reports, "detection_events": events,
            "detection_counts": dict(Counter(e["status"] for e in events)),
            "adaptive_reason_counts": dict(reasons), "changed_decisions": changed,
            "decision_count": len(decisions), "adaptive_active_decisions": reasons.get("adaptive", 0),
            "unassociated_collision_events": unassociated_collisions,
            "raw_collision_ticks": sum(bool(r["collision"]) for r in raw),
            "raw_contact_ticks": sum(bool(r["contacts"]) for r in raw),
            "planned_censor": episode.get("retire_reason") == "max_steps",
            "hud_calibration": {"n": len(speed_pairs), "hud_per_actual_mps": scale,
                                "differs_from_unity_by_over5pct": abs(scale - 1) > .05 if scale is not None else None,
                                "absolute_error": stats([abs(a - h) for a, h in speed_pairs]),
                                "note": "descriptive through-origin scale, last2 pre speeds; quantization/clipping/temporal averaging, not retuned"},
            "failure_context": failure_context(episode, raw)}


def verify_file(root, name, expected):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact outside run: {name}")
    if not expected or sha256(path) != expected:
        raise ValueError(f"hash mismatch: {name}")
    return path


def verify_models(protocol, run, candidate, manifest):
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("frozen model manifest hash mismatch")
    receipt = read_json(manifest)
    if receipt["candidate_zip_sha256"] != CANDIDATE_SHA256 or sha256(candidate) != CANDIDATE_SHA256:
        raise ValueError("candidate ZIP is not the frozen unchanged model")
    paths = {ARMS[0]: run / "crossing-projection-source-reconstruction.zip", ARMS[1]: candidate}
    for arm, path in paths.items():
        if sha256(path) != protocol["model_hashes"][arm]:
            raise ValueError(f"model hash mismatch: {arm}")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            expected = protocol["model_source_sha256"][arm]
            if len(set(names)) != len(names) or set(names) != set(expected):
                raise ValueError(f"model member inventory mismatch: {arm}")
            for name in names:
                if hashlib.sha256(archive.read(name)).hexdigest() != expected[name]:
                    raise ValueError(f"model member hash mismatch: {arm}/{name}")
    if protocol["model_source_sha256"][ARMS[0]] != receipt["baseline_source_sha256"]:
        raise ValueError("baseline is not the frozen source reconstruction")
    expected_candidate = {r["path"]: r["sha256"] for r in receipt["files"]}
    if protocol["model_source_sha256"][ARMS[1]] != expected_candidate:
        raise ValueError("candidate source manifest differs")


def load_run(run, candidate, manifest):
    """Verify primary receipts, retaining unrun/error slots rather than dropping them."""
    run = Path(run).resolve()
    protocol = read_json(run / "protocol.json")
    report = read_json(run / "episode-report.json")
    errors = []
    if report["protocol_sha256"] != sha256(run / "protocol.json"):
        raise ValueError("protocol hash mismatch")
    if protocol.get("analysis_spec") != ANALYSIS_SPEC:
        errors.append("analysis_spec_not_frozen_or_differs")
    if protocol.get("analyzer_sha256") != sha256(__file__):
        errors.append("analyzer_source_not_frozen_or_differs")
    if protocol.get("operator_sha256") != sha256(ROOT / "scripts/evaluate_koi_adaptive_ab.py"):
        errors.append("operator_source_not_frozen_or_differs")
    declared_cells = [(r["track_id"], r["geometry_seed"]) for r in protocol.get("cells", [])]
    if declared_cells != list(CELLS) or protocol.get("episodes") != 48:
        raise ValueError("protocol is not the fixed complete TRAIN cohort")
    if any(r.get("partition") != "TRAIN" or r.get("obstacles") is not True for r in protocol["cells"]):
        raise ValueError("protocol cell conditions differ")
    if any(protocol.get(k) != value for k, value in (("frame_skip", 4), ("warmup_ticks", 50), ("raw_fps", 50), ("max_decisions", 1200))):
        raise ValueError("frozen evaluator conditions differ")
    verify_models(protocol, run, candidate, manifest)
    for suffix, expected in FROZEN_SOURCE_SHA256.items():
        matching = [(p, h) for p, h in protocol["environment_source_sha256"].items() if p.endswith("/" + suffix)]
        if len(matching) != 1 or matching[0][1] != expected:
            raise ValueError(f"frozen measurement source mismatch: {suffix}")
    for path, expected in protocol["environment_source_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"environment source changed: {path}")
    slots = {(t, s, a): {"track_id": t, "seed": s, "mode": a, "status": "unrun"}
             for t, s in CELLS for a in ARMS}
    seen = set()
    episodes = {}
    for row in report["rows"]:
        key = (row["track_id"], row["seed"], row["mode"])
        if key not in slots or key in seen:
            raise ValueError(f"unexpected or duplicate slot: {key}")
        seen.add(key)
        slots[key].update(row)
        if "file" not in row:
            partial = run / f"{key[0]}-{key[1]}-{key[2]}.raw.jsonl"
            if partial.exists():
                partial_rows = read_raw(partial, partial=True)
                slots[key]["partial_raw"] = {"file": partial.name, "sha256": sha256(partial),
                                             "complete_ticks": len(partial_rows),
                                             "endpoint": partial_rows[-1] if partial_rows else None,
                                             "note": "retained without episode catalog; not eligible evidence"}
            continue
        path = verify_file(run, row["file"], row.get("sha256"))
        episode = read_json(path)
        if (episode["track_id"], episode["seed"], episode["mode"]) != key:
            raise ValueError("episode identity differs from manifest slot")
        raw_path = verify_file(run, episode["raw_trace_file"], episode["raw_trace_sha256"])
        raw = read_raw(raw_path)
        if len(raw) != episode["raw_ticks"]:
            raise ValueError("raw tick denominator differs")
        if episode.get("catalog") is None or episode.get("initial_state") is None:
            slots[key]["analysis_error"] = "missing_initial_geometry"
            errors.append(f"incomplete_episode:{key}")
            continue
        geometry = hashlib.sha256(json.dumps(episode["catalog"], sort_keys=True).encode()).hexdigest()
        if geometry != episode["geometry_sha256"]:
            raise ValueError("geometry catalog hash mismatch")
        for module, path in episode.get("runtime_module_paths", {}).items():
            member = "agent.py" if module == "agent" else module.replace(".", "/") + ".py"
            if member not in protocol["model_source_sha256"][key[2]] and member.replace(".py", "/__init__.py") not in protocol["model_source_sha256"][key[2]]:
                raise ValueError(f"runtime model module outside frozen package: {module}")
        if (episode.get("error") or episode.get("invalid_actions") or episode.get("retire_reason") in OPERATIONAL_RETIREMENTS or not episode.get("damage_telemetry_valid")
                or not episode.get("collision_telemetry_valid")):
            errors.append(f"invalid_episode:{key}")
        if not episode.get("runtime_module_paths"):
            errors.append(f"missing_model_only_runtime_receipt:{key}")
        if len(episode.get("decision_trace") or []) != episode.get("steps"):
            errors.append(f"decision_denominator_differs:{key}")
        episodes[key] = (episode, analyze_episode(episode, raw))
    if len(seen) != 48:
        errors.append("manifest_does_not_enumerate_all48_slots")
    if report.get("operator_error"):
        errors.append("operator_error:" + report["operator_error"])
    return protocol, list(slots.values()), episodes, errors


def summarize(slots, episodes: dict[Any, Any], errors=()):
    cells = []
    qualified_seeds = set()
    new_hits = []
    all_deltas = []
    cell_deltas = []
    kept = lost = gained = neither = 0
    for track, seed in CELLS:
        pair = [episodes.get((track, seed, arm)) for arm in ARMS]
        cell = {"track_id": track, "seed": seed, "pair_available": all(pair), "segments": []}
        if pair[0] is None or pair[1] is None:
            cell["status"] = "incomplete_pair"
            cells.append(cell)
            continue
        base, b = pair[0]
        adaptive, a = pair[1]
        parity_keys = ("steer", "gas", "brake", "car_x", "car_y", "car_yaw", "progress")
        parity = [[d.get(k) for k in parity_keys] for d in base["decision_trace"][:10]] == [[d.get(k) for k in parity_keys] for d in adaptive["decision_trace"][:10]]
        geometry_equal = base["geometry_sha256"] == adaptive["geometry_sha256"] and base["catalog"] == adaptive["catalog"]
        initial_equal = base["initial_observation_sha256"] == adaptive["initial_observation_sha256"] and base["initial_state"] == adaptive["initial_state"]
        outcome = "kept" if b["completed"] and a["completed"] else "lost" if b["completed"] else "gained" if a["completed"] else "neither"
        kept += outcome == "kept"
        lost += outcome == "lost"
        gained += outcome == "gained"
        neither += outcome == "neither"
        cell.update(completion=outcome, geometry_equal=geometry_equal, initial_equal=initial_equal, first10_parity=parity,
                    endpoints={ARMS[0]: {k: b.get(k) for k in ("lapTimeMs", "simulation_end", "progress", "damage", "collisions")},
                               ARMS[1]: {k: a.get(k) for k in ("lapTimeMs", "simulation_end", "progress", "damage", "collisions")}},
                    lap_delta_ms=a["lapTimeMs"] - b["lapTimeMs"] if outcome == "kept" else None)
        cell["damage_nonincrease"] = a["damage"] is not None and b["damage"] is not None and a["damage"] <= b["damage"]
        cell["collisions_nonincrease"] = a["collisions"] is not None and b["collisions"] is not None and a["collisions"] <= b["collisions"]
        local = []
        by_id: dict[int, Any] = {r["obstacle"]["id"]: r for r in a["obstacles"]}
        for br in b["obstacles"]:
            oid = br["obstacle"]["id"]
            ar = by_id.get(oid)
            eligible = bool(ar and geometry_equal and initial_equal and parity and br["segment_eligible"] and ar["segment_eligible"])
            delta = ar["common_entry_to_pass_s"] - br["common_entry_to_pass_s"] if eligible and ar is not None else None
            entry_delta = ar["common_entry"]["t"] - br["common_entry"]["t"] if eligible and ar is not None else None
            cell["segments"].append({"obstacle_id": oid, "eligible": eligible, "delta_s": delta,
                                     "baseline_duration_s": br["common_entry_to_pass_s"] if eligible else None,
                                     "candidate_duration_s": ar["common_entry_to_pass_s"] if eligible and ar is not None else None,
                                     "pass_delta_bracket_s": [ar["physical"]["pass_bracket"][0] - br["physical"]["pass_bracket"][1],
                                                              ar["physical"]["pass_bracket"][1] - br["physical"]["pass_bracket"][0]] if eligible and ar is not None else None,
                                     "duration_delta_bracket_s": [ar["physical"]["pass_bracket"][0] - br["physical"]["pass_bracket"][1] - entry_delta,
                                                                  ar["physical"]["pass_bracket"][1] - br["physical"]["pass_bracket"][0] - entry_delta] if eligible and ar is not None and entry_delta is not None else None,
                                     "baseline_exclusions": br["segment_exclusions"],
                                     "candidate_exclusions": ar["segment_exclusions"] if ar else ["missing_obstacle"]})
            if eligible:
                local.append(delta)
                all_deltas.append(delta)
            if ar and geometry_equal and br["whole_episode_safety"]["baseline_clean_reference"] and ar["whole_episode_safety"]["hit"]:
                new_hits.append({"track_id": track, "seed": seed, "obstacle_id": oid})
            if ar and geometry_equal and initial_equal and parity and ar["qualifying_adaptive_above44"]:
                qualified_seeds.add(seed)
        cell["matched_segment_delta_s"] = stats(local)
        if local:
            cell_deltas.append(float(np.mean(local)))
        cells.append(cell)
    valid_slots = len(episodes) == 48 and all(r.get("status") == "completed" for r in slots) and not errors
    gates = {"all48_valid": valid_slots,
             "model_only_pair_safety": len(cells) == 24 and all(c.get("geometry_equal") and c.get("initial_equal") and c.get("first10_parity") for c in cells),
             "lost_finishes_zero": lost == 0 and all(c["pair_available"] for c in cells),
             "per_cell_damage_nonincrease": all(c.get("damage_nonincrease", False) for c in cells),
             "per_cell_collision_nonincrease": all(c.get("collisions_nonincrease", False) for c in cells),
             "no_new_clean_obstacle_hit": not new_hits and all(c["pair_available"] and c.get("geometry_equal") for c in cells)
             and not any(value[1]["unassociated_collision_events"] for value in episodes.values()),
             "matched_common_entry_time_reduction": bool(cell_deltas) and float(np.mean(cell_deltas)) < 0,
             "adaptive_clean_actual_above44_two_geometry_seeds": len(qualified_seeds) >= 2}
    arm_summaries = {}
    for arm in ARMS:
        selected = [value[1] for key, value in episodes.items() if key[2] == arm]
        objects = [o for r in selected for o in r["obstacles"]]
        reasons = Counter()
        for r in selected:
            reasons.update(r["adaptive_reason_counts"])
        mutual_laps = [c["endpoints"][arm]["lapTimeMs"] for c in cells if c.get("completion") == "kept"]
        arm_summaries[arm] = {
            "episodes_available": len(selected), "expected_episodes": 24,
            "finishes": sum(r["completed"] for r in selected),
            "damage_total": sum(r["damage"] for r in selected if r["damage"] is not None),
            "damage_measured_episodes": sum(r["damage"] is not None for r in selected),
            "collision_decisions_total": sum(r["collisions"] for r in selected if r["collisions"] is not None),
            "collision_measured_episodes": sum(r["collisions"] is not None for r in selected),
            "simulation_elapsed_s": sum(r["simulation_end"] - r["simulation_start"] for r in selected),
            "mutually_finished_lap_ms": dict(stats(mutual_laps), median=float(np.median(mutual_laps)) if mutual_laps else None),
            "matched_common_entry_to_pass_s": stats([s["baseline_duration_s" if arm == ARMS[0] else "candidate_duration_s"] for c in cells for s in c["segments"] if s["eligible"]]),
            "all_finished_lap_ms": stats([r["lapTimeMs"] for r in selected if r["completed"] and r["lapTimeMs"] is not None]),
            "planned_censored_episodes": sum(r["planned_censor"] for r in selected),
            "near_detection_to_pass_s": stats([o["detection_times"]["near"]["detection_to_pass_s"] for o in objects if o["detection_times"]["near"]["detection_to_pass_s"] is not None]),
            "far_detection_to_pass_s": stats([o["detection_times"]["far"]["detection_to_pass_s"] for o in objects if o["detection_times"]["far"]["detection_to_pass_s"] is not None]),
            "obstacles": len(objects), "passed_obstacles": sum(o["physical"]["status"] == "passed" for o in objects),
            "clean_physical_passages": sum(o["physical"]["clean"] for o in objects),
            "clean_actual_min_above44_passages": sum(o["clean_actual_above44"] for o in objects),
            "physical_passage_min_speed_mps": stats([o["physical"]["physical_speed_mps"]["min"] for o in objects if o["physical"]["physical_speed_mps"]["n"]]),
            "physical_passage_mean_speed_mps": stats([o["physical"]["physical_speed_mps"]["mean"] for o in objects if o["physical"]["physical_speed_mps"]["n"]]),
            "physical_passage_max_speed_mps": stats([o["physical"]["physical_speed_mps"]["max"] for o in objects if o["physical"]["physical_speed_mps"]["n"]]),
            "decision_count": sum(r["decision_count"] for r in selected),
            "adaptive_active_decisions": sum(r["adaptive_active_decisions"] for r in selected),
            "adaptive_changed_decisions": sum(len(r["changed_decisions"]) for r in selected),
            "adaptive_reason_counts": dict(reasons),
            "unassociated_collision_events": sum(len(r["unassociated_collision_events"]) for r in selected),
        }
    return {"scope": "internal consumed-TRAIN proxies, not official ranking or fresh generalization",
            "slots": slots, "episodes_available": len(episodes), "expected_slots": 48,
            "completion": {"kept": kept, "lost": lost, "gained": gained, "neither": neither,
                           "paired_cells": kept + lost + gained + neither, "expected_cells": 24},
            "cells": cells, "matched_segment_delta_s": stats(all_deltas),
            "arm_summaries": arm_summaries,
            "mutually_finished_lap_delta_ms": stats([c["lap_delta_ms"] for c in cells if c.get("completion") == "kept"]),
            "per_cell_mean_segment_delta_s": stats(cell_deltas), "new_clean_obstacle_hits": new_hits,
            "qualifying_geometry_seeds": sorted(qualified_seeds), "gates": gates,
            "timing_caveat": "pass samples have20ms brackets; entry interpolation approximates crossing; negative mean is descriptive, not statistical significance or sub-tick proof",
            "gate_passed": all(gates.values()), "verification_errors": list(errors),
            "episodes": [value[1] for value in episodes.values()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=ROOT / "submissions/koi-adaptive-avoidance-v1.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "submissions/koi-adaptive-avoidance-v1.manifest.json")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()) and args.output.name in {"protocol.json", "episode-report.json"}:
        parser.error("analysis output must not overwrite primary evidence")
    protocol, slots, episodes, errors = load_run(args.run, args.candidate, args.manifest)
    result = summarize(slots, episodes, errors)
    result.update(analysis_spec=ANALYSIS_SPEC, analyzer_sha256=sha256(__file__),
                  protocol_sha256=sha256(args.run / "protocol.json"),
                  episode_report_sha256=sha256(args.run / "episode-report.json"),
                  model_hashes=protocol["model_hashes"], passive_analysis=True,
                  model_manifest_sha256=sha256(args.manifest),
                  environment_resets=0, model_updates=0, official_action=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("episodes_available", "completion", "gates", "gate_passed")}, indent=2))


if __name__ == "__main__":
    main()
