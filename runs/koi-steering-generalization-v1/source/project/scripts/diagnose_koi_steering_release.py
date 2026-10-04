"""Hashed zero-reset steering/physical timeline diagnosis of archived baseline24.

Run from repository root with python -m scripts.diagnose_koi_steering_release.
No environment, original runtime module, or controller is imported/constructed.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

from haic.algorithms.koi.steering_terms import reconstruct_steering, RESIDUAL_TOLERANCE
from scripts.package_koi_adaptive_avoidance import frozen_baseline_files


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/koi-minimum-clearance-ab-20260930-r3"
DEFAULT_OUTPUT = ROOT / "experiments/koi-steering-release-baseline-diagnosis-v1.json"
DEFAULT_ROWS = ROOT / "runs/koi-steering-release-diagnosis-20261001-v1"
SNAPSHOT = Path("/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace")
CELLS = {(track, seed) for track in (1, 2, 3) for seed in (*range(38300, 38304), *range(50300, 50304))}
HALF_WIDTH = 1.1 + .28 * math.cos(.4) + .54 * math.sin(.4) + .01
HALF_LENGTH = 2.61
SPEC = {
    "version": "koi-steering-release-diagnosis-v1",
    "scope": "archived consumed TRAIN observational file analysis; current trajectory is not goal reference",
    "decomposition": "source-formula nonlinear staged reconstruction, float32 at original assignments; .55 REPLACES .34; ordered marginal afterclip attribution",
    "original_controller_prestate": "each full same-cell decision sequence starts at original constructor/reset steps0/impact0; next pre-impact is preceding archived actual_impact_left; no World input or new controller/reset; missing predecessor counter remains unknown",
    "residual_tolerance": RESIDUAL_TOLERANCE,
    "association": "unique <=4px rendered-centroid match among ALL physical objects; no nearest fallback; projection requires previous/current same identity",
    "separation": "exact full-fixture signed circle separation>0 plus longitudinal involvement is observed noncontact, not lateral certificate; separately require global yaw-enclosing skin-inclusive hull/wheel lateral interval gap>0 in fixed station+/-25 local approach/prepass window or actual longitudinal involvement; first window-safe and first overlap-safe separately retained",
    "lateral_bound": "max wheel steer .4rad halfwidth1.578182983m and half-length2.61m enclosing union; actual fixture lateral extents not archived; conservative nominal bound NOT measured lateral extents",
    "rear_clear": "all-fixture observed rear>circle radius; local involvement->clear transition; bracket50Hz, not continuous safety",
    "projection": "finite abs(projected_x-42)>=6 measured off distinguished from unavailable; active->off transitions consecutive same unique object; all epochs retained",
    "release": "actual afterclip avoidance component meaningful weakening <=half epoch peak (peak>=.05), disappearance <=.01, sign reversal; same-object contiguous decisions only",
    "integrals": "actual hold pre/post overlap with physical separated epochs, per-object unique associated decision only; command*time, not lateral-distance guarantee",
    "representatives": "top4 whole-episode maxabs-lateral descending plus fixed3:38301,1:38302,2:50302; chronological full decision/physical rows; failures/seams retained",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    def validate(item):
        if isinstance(item, dict):
            for v in item.values():
                validate(v)
        elif isinstance(item, list):
            for v in item:
                validate(v)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON")
    validate(value)
    return value


def project_center(obstacle, state):
    dx, dy = obstacle["x"] - state["x"], obstacle["y"] - state["y"]
    c, s = math.cos(state["yaw"]), math.sin(state["yaw"])
    zoom = .6 * max(1 - state["t"], 0) + 16.2 * min(state["t"], 1)
    return [84 * (500 + zoom * (dx * c + dy * s)) / 1000 - .0625,
            84 * (799 - (200 + zoom * (-dx * s + dy * c))) / 800 - .0625]


def associate_detection(obj, state, obstacles):
    if obj is None:
        return {"status": "undetected", "obstacle_id": None, "candidates": []}
    candidates = []
    for o in obstacles:
        projected = project_center(o, state)
        distance = math.dist([obj[1], obj[0]], projected)
        if distance <= 4:
            candidates.append(dict(id=o["id"], distance_px=distance, projected=projected))
    return dict(status="matched" if len(candidates) == 1 else "ambiguous" if candidates else "unmatched",
                obstacle_id=candidates[0]["id"] if len(candidates) == 1 else None, candidates=candidates)


def physical_row(state, obstacle, index):
    tx, ty = obstacle["tangent"]
    nx, ny = ty, -tx
    dx, dy = state["x"] - obstacle["x"], state["y"] - obstacle["y"]
    longitudinal = dx * tx + dy * ty
    lateral = dx * nx + dy * ny
    beta = math.atan2(-tx, ty)
    relative = state["yaw"] - beta
    yaw_halfwidth = HALF_WIDTH * abs(math.cos(relative)) + HALF_LENGTH * abs(math.sin(relative))
    nominal_gap = abs(lateral) - yaw_halfwidth - obstacle["radius"]
    front, rear, clearance = (state[k][index] for k in ("front", "rear", "clearance"))
    local = abs(state["station"] - obstacle["station"]) <= 40 and abs(state["road_index"] - obstacle["anchor_index"]) <= 15
    involved = local and front >= -obstacle["radius"] and rear <= obstacle["radius"]
    in_window = local and abs(state["station"] - obstacle["station"]) <= 25
    approach = in_window and longitudinal <= 0 and rear <= obstacle["radius"]
    exact_positive = clearance > 0 and obstacle["id"] not in state["contacts"]
    separated = (involved or approach) and exact_positive and nominal_gap > 0
    return dict(t=state["t"], station=state["station"], lane_lateral=state["lateral"], speed=state["speed"],
                obstacle_lateral=lateral, longitudinal_center=longitudinal, front=front, rear=rear,
                full_fixture_clearance=clearance, exact_positive_clearance=exact_positive,
                local=local, longitudinal_involved=involved, local_approach=approach,
                in_fixed_station_window=in_window,
                nominal_yaw_lateral_gap=nominal_gap, nominal_yaw_halfwidth=yaw_halfwidth,
                conservative_lateral_separated=bool(separated),
                conservative_lateral_safe_in_window=bool(in_window and rear <= obstacle["radius"] and exact_positive and nominal_gap > 0),
                conservative_lateral_safe_while_involved=bool(involved and exact_positive and nominal_gap > 0),
                observed_clear_while_involved=bool(involved and exact_positive),
                full_rear_clear=bool(local and rear > obstacle["radius"]),
                physical_flank=1 if lateral > 0 else -1 if lateral < 0 else 0,
                contact=obstacle["id"] in state["contacts"], wheel_road_contacts=state["wheel_road_contacts"])


def epochs(rows, key):
    """Every true interval; timing bounds retain entry/exit sample uncertainty."""
    output, start = [], None
    for i, row in enumerate(rows):
        if row[key] and start is None:
            start = i
        if start is not None and (not row[key] or i == len(rows) - 1):
            end = i - 1 if not row[key] else i
            output.append(dict(start_t=rows[start]["t"], end_t=rows[end]["t"],
                               start_bracket=[rows[max(0, start - 1)]["t"], rows[start]["t"]],
                               end_bracket=[rows[end]["t"], row["t"]] if not row[key] else None,
                               right_censored=bool(row[key] and i == len(rows) - 1), samples=end - start + 1))
            start = None
    return output


def decision_rows(episode):
    rows = []
    previous = None
    impact_before = 0
    for decision in episode["decision_trace"]:
        if decision["step"] != len(rows) + 1:
            raise ValueError("original controller decision sequence is not contiguous from constructor")
        d = decision["controller"]
        pre, post = d["evaluation_only"]["pre"], d["evaluation_only"]["post"]
        association = associate_detection(d.get("near_object"), pre, episode["catalog"]["obstacles"])
        controller_pre = dict(steps=decision["step"] - 1, impact_left=impact_before,
                              source="original_constructor_reset_zero" if not rows else "previous_same_cell_original_post_counter")
        terms = reconstruct_steering(d, decision["steer"], state={
            "steps": decision["step"], "impact_left_before_act": impact_before,
            "impact_left_before_act_source": controller_pre["source"],
        })
        oid = association["obstacle_id"]
        same_previous = previous is not None and oid is not None and oid == previous["obstacle_id"]
        projection_identity_valid = same_previous and terms["projection_state"] in ("off", "crossing")
        rows.append(dict(track_id=episode["track_id"], seed=episode["seed"], step=decision["step"], t=pre["t"], end_t=post["t"],
                         obstacle_id=oid, association=association, previous_same_object=same_previous,
                         original_controller_pre_act_state=controller_pre,
                         projection_identity_valid=projection_identity_valid,
                         projection_state=terms["projection_state"], contact_active=d.get("contact_active"),
                         steer=decision["steer"], gas=decision["gas"], brake=decision["brake"],
                         lane_lateral=pre["lateral"], heading_error=pre["heading_error"], speed=pre["speed"],
                         post_lane_lateral=post["lateral"], near_object=d.get("near_object"),
                         projected_obstacle_x=d.get("projected_obstacle_x"), original_diagnostics={k: v for k, v in d.items() if k != "evaluation_only"},
                         decomposition=terms, original_evaluation_only=d["evaluation_only"]))
        previous = rows[-1]
        impact_before = d.get("actual_impact_left")
    return rows


def object_timeline(episode, states, decisions, obstacle, index):
    physical = [physical_row(s, obstacle, index) for s in states]
    separation = epochs(physical, "conservative_lateral_separated")
    window_safe = epochs(physical, "conservative_lateral_safe_in_window")
    overlap_safe = epochs(physical, "conservative_lateral_safe_while_involved")
    for epoch_group, eligibility in ((separation, None), (window_safe, "in_fixed_station_window"), (overlap_safe, "longitudinal_involved")):
        for epoch in epoch_group:
            i = next(i for i, p in enumerate(physical) if p["t"] == epoch["start_t"])
            prior_eligible = i > 0 and (physical[i - 1][eligibility] if eligibility is not None else
                                      physical[i - 1]["local_approach"] or physical[i - 1]["longitudinal_involved"])
            epoch["onset_status"] = "observed_lateral_gap_transition" if prior_eligible else "already_separated_at_eligibility_entry"
            epoch["first_sample_gap"] = physical[i]["nominal_yaw_lateral_gap"]
            epoch["first_sample_clearance"] = physical[i]["full_fixture_clearance"]
    involved = epochs(physical, "longitudinal_involved")
    rear = next((i for i in range(1, len(physical)) if physical[i]["full_rear_clear"] and physical[i - 1]["local"]
                 and physical[i - 1]["rear"] <= obstacle["radius"]
                 and any(p["longitudinal_involved"] for p in physical[:i])), None)
    associated = [d for d in decisions if d["obstacle_id"] == obstacle["id"]]
    projection_epochs, transitions, weakening, reversals, dropouts = [], [], [], [], []
    previous, peak, current_epoch = None, 0.0, None
    for row in associated:
        terms = row["decomposition"]
        component = terms["avoidance_component"]
        contiguous = previous is not None and row["step"] == previous["step"] + 1 and row["previous_same_object"]
        if not contiguous:
            peak = 0.0
        if row["projection_identity_valid"]:
            if not contiguous or current_epoch is None or current_epoch["state"] != row["projection_state"]:
                current_epoch = dict(state=row["projection_state"], start_step=row["step"], end_step=row["step"], start_t=row["t"], end_t=row["end_t"])
                projection_epochs.append(current_epoch)
            else:
                current_epoch.update(end_step=row["step"], end_t=row["end_t"])
        else:
            current_epoch = None
        if contiguous:
            assert previous is not None
        if contiguous and previous is not None and previous["contact_active"] and row["projection_identity_valid"] and row["projection_state"] == "off":
            transitions.append(dict(active_step=previous["step"], off_step=row["step"], bracket=[previous["t"], row["t"]],
                                    previous_projected=previous["projected_obstacle_x"], projected=row["projected_obstacle_x"]))
        if component is not None:
            if contiguous and peak >= .05 and abs(component) <= .5 * peak:
                assert previous is not None
                weakening.append(dict(step=row["step"], t=row["t"], bracket=[previous["t"], row["t"]], component=component, epoch_peak=peak,
                                       path=terms["active_path"]))
            if contiguous and previous is not None and component * (previous["decomposition"]["avoidance_component"] or 0) < 0:
                reversals.append(dict(step=row["step"], t=row["t"], previous_side=previous["decomposition"]["selected_flank"], side=terms["selected_flank"]))
            peak = max(peak, abs(component))
        previous = row
    all_by_step = {d["step"]: d for d in decisions}
    for row in associated:
        nxt = all_by_step.get(row["step"] + 1)
        if nxt is not None and nxt["obstacle_id"] != obstacle["id"]:
            dropouts.append(dict(last_step=row["step"], next_step=nxt["step"], bracket=[row["t"], nxt["t"]], next_status=nxt["association"]["status"],
                                 next_object_id=nxt["obstacle_id"], physical_rear_at_dropout=nxt["original_evaluation_only"]["pre"]["rear"][index],
                                 next_avoidance_component=nxt["decomposition"]["avoidance_component"],
                                 next_steer=nxt["steer"], next_projection_state=nxt["projection_state"],
                                 global_avoidance_weakening_observed=nxt["decomposition"]["avoidance_component"] is not None
                                 and abs(nxt["decomposition"]["avoidance_component"]) <= .5 * abs(row["decomposition"]["avoidance_component"] or 0),
                                 same_object_release_identifiable=False,
                                 disappearance_is_clearance=False))
    persistence = []
    for row in associated:
        comp = row["decomposition"]["avoidance_component"]
        if comp is None or not row["decomposition"]["reconstruction_valid"]:
            continue
        for epoch in separation:
            a, b = max(row["t"], epoch["start_t"]), min(row["end_t"], epoch["end_t"])
            if b <= a or abs(comp) <= .01:
                continue
            physical_index = max(0, min(len(physical) - 1, round((a - states[0]["t"]) / .02)))
            p = physical[physical_index]
            lane_sign = 1 if p["lane_lateral"] > 0 else -1 if p["lane_lateral"] < 0 else 0
            side = row["decomposition"]["selected_flank"] or 0
            persistence.append(dict(step=row["step"], start_t=a, end_t=b, duration_s=b - a, avoidance_component=comp,
                                    steer=row["steer"], lane_lateral=p["lane_lateral"], physical_flank=p["physical_flank"], selected_flank=side,
                                    selected_outward=comp * side > 0, physical_outward=comp * p["physical_flank"] > 0,
                                    lane_outward=comp * lane_sign > 0, lane_restoring=comp * lane_sign < 0,
                                    final_lane_outward=row["steer"] * lane_sign > 0,
                                    projection_state=row["projection_state"], projection_identity_valid=row["projection_identity_valid"],
                                    nominal_yaw_lateral_gap=p["nominal_yaw_lateral_gap"], full_fixture_clearance=p["full_fixture_clearance"],
                                    front=p["front"], rear=p["rear"], longitudinal_involved=p["longitudinal_involved"],
                                    term_integral_abs_s=abs(comp) * (b - a)))
    overlaps = []
    for p in projection_epochs:
        if p["state"] != "off":
            continue
        for s in separation:
            a, b = max(p["start_t"], s["start_t"]), min(p["end_t"], s["end_t"])
            if b > a:
                transitions_in_epoch = [t for t in transitions if t["off_step"] == p["start_step"]]
                overlaps.append(dict(start_t=a, end_t=b, duration_s=b - a, projection_epoch=p,
                                     separation_epoch=s, matched_active_to_off=transitions_in_epoch,
                                     active_to_off_observed=bool(transitions_in_epoch)))
    return dict(track_id=episode["track_id"], seed=episode["seed"], obstacle_id=obstacle["id"], obstacle=obstacle,
                completed=episode["completed"], retire_reason=episode.get("retire_reason"), seam_flag=episode["seed"] == 38300,
                first_safe_lateral=separation[0] if separation else None, separated_epochs=separation,
                first_safe_lateral_window=window_safe[0] if window_safe else None,
                first_safe_lateral_while_involved=overlap_safe[0] if overlap_safe else None,
                safe_lateral_window_epochs=window_safe, safe_lateral_overlap_epochs=overlap_safe,
                observed_positive_involvement_epochs=epochs(physical, "observed_clear_while_involved"), involvement_epochs=involved,
                full_rear_clear=None if rear is None else dict(t=physical[rear]["t"], bracket=[physical[rear - 1]["t"], physical[rear]["t"]]),
                projection_epochs=projection_epochs, active_to_measured_off=transitions,
                projection_off_separation_overlaps=overlaps, meaningful_weakening=weakening,
                avoidance_reversals=reversals, detector_dropouts=dropouts, persistence_after_separation=persistence,
                persistence_duration_s=sum(r["duration_s"] for r in persistence),
                persistence_integral_abs_s=sum(r["term_integral_abs_s"] for r in persistence),
                outward_duration_s=sum(r["duration_s"] for r in persistence if r["physical_outward"]),
                restoring_duration_s=sum(r["duration_s"] for r in persistence if r["lane_restoring"]),
                hit=any(p["contact"] or p["full_fixture_clearance"] <= 0 for p in physical),
                physical_rows=[p for p in physical if p["local"]], associated_steps=[d["step"] for d in associated])


def compact_decision(row):
    terms = row["decomposition"]
    return {**{k: row[k] for k in ("step", "t", "end_t", "obstacle_id", "near_object", "projected_obstacle_x", "projection_identity_valid",
                                    "contact_active", "steer", "lane_lateral", "post_lane_lateral", "speed", "heading_error")},
            **{k: terms[k] for k in ("raw_terms", "contributions", "avoidance_component", "active_path", "saturation", "residual")}}


def load_inputs(run):
    run = Path(run)
    protocol, report = read_json(run / "protocol.json"), read_json(run / "episode-report.json")
    if report["protocol_sha256"] != sha(run / "protocol.json"):
        raise ValueError("protocol hash mismatch")
    inventory = {str(run / n): sha(run / n) for n in ("protocol.json", "episode-report.json")}
    for copy in protocol["source_copies"]:
        path = run / copy["file"]
        if sha(path) != copy["sha256"]:
            raise ValueError("source copy hash mismatch: " + str(path))
        inventory[str(path)] = sha(path)
    baseline = frozen_baseline_files()
    with zipfile.ZipFile(run / "crossing-projection-source-reconstruction.zip") as z:
        if set(z.namelist()) != set(baseline) or len(z.namelist()) != len(baseline) or any(z.read(k) != v for k, v in baseline.items()):
            raise ValueError("frozen baseline source mismatch")
    if sha(run / "crossing-projection-source-reconstruction.zip") != protocol["model_hashes"]["crossing_projection"]:
        raise ValueError("baseline ZIP hash mismatch")
    inventory[str(run / "crossing-projection-source-reconstruction.zip")] = sha(run / "crossing-projection-source-reconstruction.zip")
    for name, data in baseline.items():
        if name.startswith("haic_agent/") and name != "haic_agent/__init__.py":
            path = SNAPSHOT / name
            if path.read_bytes() != data:
                raise ValueError("frozen snapshot mismatch: " + str(path))
            inventory[str(path)] = sha(path)
    episodes, seen = [], set()
    for row in report["rows"]:
        if row["mode"] != "crossing_projection":
            continue
        key = row["track_id"], row["seed"]
        if key not in CELLS or key in seen:
            raise ValueError("duplicate/unexpected baseline cell")
        seen.add(key)
        path = run / row["file"]
        if sha(path) != row["sha256"]:
            raise ValueError("episode hash mismatch")
        episode = read_json(path)
        raw_path = run / episode["raw_trace_file"]
        if sha(raw_path) != episode["raw_trace_sha256"]:
            raise ValueError("raw hash mismatch")
        raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
        if len(raw) != episode["raw_ticks"] or len(episode["decision_trace"]) != episode["steps"]:
            raise ValueError("denominator mismatch")
        if hashlib.sha256(json.dumps(episode["catalog"], sort_keys=True).encode()).hexdigest() != episode["geometry_sha256"]:
            raise ValueError("catalog hash mismatch")
        states = [episode["initial_state"], *raw]
        by_time = {r["t"]: r for r in states}
        for d in episode["decision_trace"]:
            for endpoint in ("pre", "post"):
                state = d["controller"]["evaluation_only"][endpoint]
                ref = by_time.get(state["t"])
                if ref is None or any(state[k] != ref[k] for k in ("x", "y", "yaw", "speed", "station", "lateral", "front", "rear", "clearance", "contacts")):
                    raise ValueError("decision/raw state mismatch")
        inventory.update({str(path): sha(path), str(raw_path): sha(raw_path)})
        episodes.append((episode, raw))
    if seen != CELLS:
        raise ValueError("baseline24 incomplete")
    for path in (Path(__file__), ROOT / "haic/algorithms/koi/steering_terms.py", ROOT / "scripts/package_koi_adaptive_avoidance.py",
                 ROOT / "tests/test_koi_steering_terms.py", ROOT / "tests/test_diagnose_koi_steering_release.py"):
        inventory[str(path)] = sha(path)
    return episodes, inventory, {n: hashlib.sha256(v).hexdigest() for n, v in baseline.items()}


def prior_sideflip_context():
    """Bound old failed treatment risk context, NOT another baseline cohort."""
    path = ROOT / "experiments/koi-minimum-clearance-r2-lost-finish-diagnosis.json"
    risk = read_json(path)
    run = ROOT / risk["run"]
    inventory = {str(path): sha(path)}
    episodes = {}
    raw_by_arm = {}
    for arm, identity in (("crossing_projection", "baseline"), ("minimum_clearance_v1", "candidate")):
        episode_path = run / f"3-38301-{arm}.json"
        raw_path = run / f"3-38301-{arm}.raw.jsonl"
        if sha(episode_path) != risk["identities"][identity]["episode_sha256"] or sha(raw_path) != risk["identities"][identity]["raw_sha256"]:
            raise ValueError("prior sideflip episode/raw hash mismatch")
        inventory.update({str(episode_path): sha(episode_path), str(raw_path): sha(raw_path)})
        episodes[identity] = read_json(episode_path)
        raw_by_arm[identity] = [json.loads(line) for line in raw_path.read_text().splitlines()]
    zip_path = run / "candidate.zip"
    if sha(zip_path) != risk["identities"]["candidate_zip_sha256"]:
        raise ValueError("prior sideflip ZIP hash mismatch")
    inventory[str(zip_path)] = sha(zip_path)
    primary = episodes["candidate"]
    rows = []
    impact_before = 0
    for d in primary["decision_trace"]:
        original = d["controller"]
        controller_pre = impact_before
        impact_before = original.get("actual_impact_left")
        if d["step"] not in (31, 121, 122, 123, 124, 125, 126):
            continue
        terms = reconstruct_steering(original, original["baseline_steer"], state={
            "steps": d["step"], "impact_left_before_act": controller_pre,
            "impact_left_before_act_source": "previous_same_cell_original_post_counter",
        })
        if not terms["reconstruction_valid"]:
            raise ValueError("prior same-observation crossing reconstruction mismatch")
        obj = original["near_object"]
        rows.append(dict(step=d["step"], t=original["evaluation_only"]["pre"]["t"], near_object=obj,
                         association=associate_detection(obj, original["evaluation_only"]["pre"], primary["catalog"]["obstacles"]),
                         side=original["actual_obstacle_side"], offset=None if obj is None else obj[1] - obj[2],
                         candidate_steer=d["steer"], original_same_observation_steer=original["baseline_steer"],
                         candidate_changed=original.get("minimum_clearance_changed"),
                         reason=original.get("minimum_clearance_reason"), crossing_terms=terms,
                         collision=d["collision"], pre=original["evaluation_only"]["pre"], post=original["evaluation_only"]["post"]))
    contacts = [r for r in raw_by_arm["candidate"] if 3 in r["contacts"]]
    return dict(scope="prior failed r2 treatment3:38301 only; not included in baseline24 denominators or new mechanism performance",
                baseline_completed=episodes["baseline"]["completed"], failed_candidate_completed=primary["completed"],
                sole_changed_decisions=[d["step"] for d in primary["decision_trace"] if d["controller"].get("minimum_clearance_changed")],
                original_same_observation_rows=rows,
                object3_contact_ticks=len(contacts), first_object3_contact_t=contacts[0]["t"] if contacts else None,
                all_object3_contact_ticks_onroad=all(all(n > 0 for n in r["wheel_road_contacts"]) for r in contacts),
                risk="Initial changed object0 passed clean; later inherited source side flip at object3 after trajectory divergence, not direct initial hit/early return. Preserve crossing-active commands. Timing perturbation can hit a different downstream obstacle even with a locally safe held command.",
                episode_sha256={key: risk["identities"][key]["episode_sha256"] for key in episodes}), inventory


def diagnose(run=DEFAULT_RUN, output=DEFAULT_OUTPUT, rows_directory=DEFAULT_ROWS, *, write_artifacts=True):
    if Path(output).exists() or (Path(rows_directory).exists() and any(Path(rows_directory).iterdir())):
        raise FileExistsError("diagnosis is immutable; output must be new")
    episodes, inventory, baseline_hashes = load_inputs(run)
    prior_context, prior_inventory = prior_sideflip_context()
    inventory.update(prior_inventory)
    decisions, events, summaries = [], [], []
    for episode, raw in episodes:
        rows = decision_rows(episode)
        decisions.extend(rows)
        states = [episode["initial_state"], *raw]
        events.extend(object_timeline(episode, states, rows, o, i) for i, o in enumerate(episode["catalog"]["obstacles"]))
        last = states[-1]
        summaries.append(dict(track_id=episode["track_id"], seed=episode["seed"], completed=episode["completed"],
                              retire_reason=episode.get("retire_reason"), max_abs_lateral=max(abs(s["lateral"]) for s in states),
                              seam_flag=episode["seed"] == 38300, decisions=len(rows), raw_ticks=len(raw),
                              endpoint={k: last[k] for k in ("t", "speed", "station", "lateral", "contacts", "wheel_road_contacts")},
                              terminal_wrapper_flags={k: rows[-1]["original_evaluation_only"].get(k) for k in ("wrapper_off_track", "wrapper_crashed", "simulator_terminated")}))
    ranked = sorted(summaries, key=lambda s: (-s["max_abs_lateral"], s["track_id"], s["seed"]))
    selected = {(s["track_id"], s["seed"]) for s in ranked[:4]} | {(3, 38301), (1, 38302), (2, 50302)}
    evidence = sorted([dict(track_id=e["track_id"], seed=e["seed"], obstacle_id=e["obstacle_id"], **r)
                       for e in events for r in e["persistence_after_separation"]
                       if r["physical_outward"] and r["projection_identity_valid"] and r["projection_state"] == "off"],
                      key=lambda r: (-r["term_integral_abs_s"], r["track_id"], r["seed"], r["step"]))
    core_cases = []
    for e in events:
        matched_epochs = [p for p in e["projection_off_separation_overlaps"] if p["active_to_off_observed"]]
        if not matched_epochs:
            continue
        rows = [r for r in e["persistence_after_separation"] if r["physical_outward"] and any(
            p["start_t"] <= r["start_t"] < p["end_t"] for p in matched_epochs)]
        if rows:
            steps = {r["step"] for r in rows}
            steps |= {t["active_step"] for t in e["active_to_measured_off"]} | {t["off_step"] for t in e["active_to_measured_off"]}
            core_cases.append(dict(track_id=e["track_id"], seed=e["seed"], obstacle_id=e["obstacle_id"],
                                   seam_flag=e["seam_flag"], completed=e["completed"], first_safe_lateral=e["first_safe_lateral"],
                                   full_rear_clear=e["full_rear_clear"], matched_epoch_overlaps=matched_epochs,
                                   persistence=rows, integral_abs_s=sum(r["term_integral_abs_s"] for r in rows),
                                   duration_s=sum(r["duration_s"] for r in rows),
                                   chronology=[compact_decision(d) for d in decisions if d["track_id"] == e["track_id"] and d["seed"] == e["seed"]
                                               and any(abs(d["step"] - step) <= 1 for step in steps)]))
    core_cases.sort(key=lambda c: (c["track_id"], c["seed"], c["obstacle_id"]))
    representative_chronologies = []
    for summary in ranked:
        cell = summary["track_id"], summary["seed"]
        if cell not in selected:
            continue
        cell_rows = [d for d in decisions if (d["track_id"], d["seed"]) == cell]
        peak = max(cell_rows, key=lambda d: max(abs(d["lane_lateral"]), abs(d["post_lane_lateral"])))
        representative_chronologies.append(dict(track_id=cell[0], seed=cell[1], max_lateral_step=peak["step"],
                                                max_lateral_context=[compact_decision(d) for d in cell_rows if abs(d["step"] - peak["step"]) <= 3],
                                                terminal_context=[compact_decision(d) for d in cell_rows[-4:]],
                                                objects=[dict(obstacle_id=e["obstacle_id"], first_safe_lateral=e["first_safe_lateral"],
                                                              full_rear_clear=e["full_rear_clear"], active_to_off=e["active_to_measured_off"],
                                                              weakening=e["meaningful_weakening"], reversals=e["avoidance_reversals"],
                                                              persistence_duration_s=e["persistence_duration_s"])
                                                         for e in events if (e["track_id"], e["seed"]) == cell]))
    terms = [d["decomposition"] for d in decisions]
    result = dict(study="koi-steering-release-baseline-diagnosis-v1", measurement_spec=SPEC,
                  environment_imports=0, environment_constructions=0, environment_resets=0, model_constructions=0, official_action=False,
                  input_and_source_sha256=inventory, frozen_baseline_source_sha256=baseline_hashes,
                  baseline="c4e224d source reconstruction; missing original crossing ZIP not restored",
                  episodes=len(episodes), decisions=len(decisions), raw_ticks=sum(len(raw) for _, raw in episodes), obstacles=len(events),
                  coverage=dict(valid=sum(t["reconstruction_valid"] for t in terms), invalid=sum(not t["reconstruction_valid"] for t in terms),
                                sum_residual_valid=sum(t["residual_valid"] for t in terms), max_abs_residual=max(abs(t["residual"] or 0) for t in terms),
                                max_abs_sum_residual=max(abs(t["final_sum_residual"] or 0) for t in terms),
                                unidentifiable_counts=dict(Counter(k for t in terms for k in t["unidentifiable"])),
                                impact_prestate_source_counts=dict(Counter(t["impact_prestate_source"] for t in terms)),
                                impact_assignment_ambiguous=sum(t["impact_assignment_ambiguous"] for t in terms),
                                path_counts=dict(Counter(t.get("active_path", "unidentified") for t in terms)),
                                saturated=sum(bool(t["saturation"]) for t in terms),
                                max_abs_float32_rounding=max((abs(s["delta"]) for t in terms for s in t.get("float32_rounding", [])), default=0)),
                  projection_counts=dict(Counter(d["projection_state"] for d in decisions)),
                  association_counts=dict(Counter(d["association"]["status"] for d in decisions)),
                  active_to_off_transitions=sum(len(e["active_to_measured_off"]) for e in events),
                  separated_objects=sum(bool(e["first_safe_lateral"]) for e in events),
                  full_rear_clear_objects=sum(e["full_rear_clear"] is not None for e in events),
                  outward_projection_off_evidence_rows=len(evidence), evidence_compact=evidence[:24],
                  same_object_active_off_separated_core_cases=core_cases,
                  core_case_objects=len(core_cases), core_case_duration_s=sum(c["duration_s"] for c in core_cases),
                  core_case_integral_abs_s=sum(c["integral_abs_s"] for c in core_cases),
                  physical_separated_projection_off_duration_s=sum(r["duration_s"] for r in evidence),
                  physical_separated_projection_off_integral_abs_s=sum(r["term_integral_abs_s"] for r in evidence),
                  representative_chronologies=representative_chronologies,
                  prior_sideflip_context=prior_context,
                  episode_summaries=summaries, representative_rule=SPEC["representatives"],
                  representative_episodes=[dict(s, selection="top4_maxlat" if s in ranked[:4] else "fixed_failure_or_sideflip_context") for s in ranked if (s["track_id"], s["seed"]) in selected],
                  object_summaries=[{k: v for k, v in e.items() if k not in ("physical_rows", "persistence_after_separation", "associated_steps")} for e in events],
                  proposed_mechanism=dict(status="hypothesis only; no candidate or A/B implemented here",
                                          mechanism="Release only the inherited near-urgency term after consecutive uniquely tracked same-object active->valid-off trajectory and pixel full-body unchanged-margin opposite-flank gap; preserve crossing-active and impact paths and all original speed/pedals; road+damping restoring action must not project full body back into circle/bounds over actual .08s hold. Re-evaluate each call; do not use a desired target as current-motion evidence.",
                                          exclusion="No global steering scale, speed-target change, clearance/margin reduction, missing-projection shortcut or privileged runtime physical state.",
                                          fixed_full_body_halfwidth_m=HALF_WIDTH, fixed_full_body_half_length_m=HALF_LENGTH,
                                          fixed_pixel_margin_px=1.5, original_projection_band_px=6,
                                          risk="2:50302 is an on-road obstacle-contact/stationary blocker despite off_track retirement label. Detector identity/road support/crop and steering dynamics remain uncertainty; disappearance is not rear clear."),
                  caveats=["Observed fixture contours were evaluated by frozen observer, not archived as vertices; exact lateral projection interval unavailable. Yaw-envelope lateral gap is conservative nominal model plus skin, separately labeled from observed exact clearance.",
                           "Snapshot initializer differs from arrival release initializer; all eight steering/runtime dependency modules match frozen_baseline_files bytewise. Only the immutable ZIP initializer is runtime evidence; snapshot initializer is not imported.",
                           "A first local safe sample already separated at approach entry is left-censored separation, not proof steering just escaped; onset_status distinguishes this from an observed gap transition.",
                           "50Hz sample brackets do not certify between-tick safety; original projection follows current trajectory, not desired goal lane.",
                           "Afterclip avoidance attribution is a same-state counterfactual bookkeeping delta, not causal excess or safe counterfactual trajectory.",
                           "All24 layouts/eight geometry seeds already consumed; no fresh/confirmation/blind/generalization/official claim.",
                           "No stored rendered images beyond archived numeric diagnostics; no reconstruction of missing pixels."])
    if not write_artifacts:
        return result
    Path(rows_directory).mkdir(exist_ok=True)
    outputs = {}
    for name, rows in (("decisions.jsonl", decisions), ("events.jsonl", events)):
        path = Path(rows_directory) / name
        with path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        outputs[str(path)] = sha(path)
    # Refuse result publication if an input changed while the analysis ran.
    if any(sha(path) != digest for path, digest in inventory.items()):
        raise ValueError("input/source changed during diagnosis")
    result["output_sha256"] = outputs
    with Path(output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-directory", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--check-only", action="store_true", help="compute and print evidence without writing artifacts")
    args = parser.parse_args()
    result = diagnose(args.run, args.output, args.rows_directory, write_artifacts=not args.check_only)
    keys = ["episodes", "decisions", "raw_ticks", "obstacles", "coverage", "active_to_off_transitions", "outward_projection_off_evidence_rows"]
    if args.check_only:
        keys += ["representative_episodes", "core_case_objects", "core_case_duration_s", "core_case_integral_abs_s"]
    print(json.dumps({k: result[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
