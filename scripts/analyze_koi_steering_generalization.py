"""Passive, source-bound frozen-v2 generalization on newly audited TRAIN roads."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import analyze_koi_steering_release_ab as prior

ROOT, ARMS, BASELINE_SHA = prior.ROOT, prior.ARMS, prior.BASELINE_SHA
EXTRA_MEMBERS = prior.EXTRA_MEMBERS
SEEDS = tuple(range(3184000001, 3184000025))
CELLS = [(track, seed) for seed in SEEDS for track in (1, 2, 3)]
SLOTS = len(CELLS) * len(ARMS)
CANDIDATE_SHA = "b1911d7dddf05d7c3f405b900f40c8c5696607130d2ef5e1cf473a82166147ce"
ANALYSIS_SPEC = {
    **prior.ANALYSIS_SPEC,
    "version": "koi-steering-frozen-v2-train-generalization-v1",
    "slots": SLOTS, "cells": len(CELLS), "geometry_seeds": len(SEEDS),
    "objects_per_arm": len(CELLS) * 6,
    "baseline_windows": "all prospectively baseline-eligible windows, no fixed historical119 count",
    "eligibility": "same frozen measurement; all baseline complete continuous station+-25 windows with valid passed physical passage and complete steering/component coverage must be retained; no survivor credit",
    "relative": "per eligible baseline object (candidate-baseline)/baseline only if baseline>0; zero staysNone, candidate-positive from zero invalidates reduction; both-zero gets no reduction credit; mean valid relative objects per cell then equal-cell and equal-geometry means; all eligible cells require nonzero denominators; matched absolute means cover all dynamically eligible baseline windows including zeros",
    "gates": "all144 valid uncensored;72 geometry/initial/actual10 pairs; no lost finish; per-cell damage/collision nonincrease; all432-object no new hit; all prospectively baseline-eligible windows retained; same5% cell-weighted lateral and avoidance duration OR integral across>=2geometries; additionally5% equal-geometry lateral and avoidance duration OR integral and joint lateral/integral decrease on>=half eligible geometries; matched path/actual-steer/kept-lap nonincrease; baseline comparable returns retained and no more common-horizon censors; qualified release>=2geometries;24 distinct road geometry hashes; unchangedv2/baseline/root and fresh audit",
    "threshold": "predeclared internal practical thresholds, not significance, protected confirmation or official score",
    "frozen_policy": "v2 and baseline remain byte-identical regardless of outcome; any followup is a distinct candidate and separate authorization, no road replacement or outcome-driven extension",
    "generalization_coverage": "baseline-eligible windows on>=18 of24 road seeds and>=12 seeds per track; distinct road-only signatures across24 seeds, disjoint from all eight r2 road signatures; no survivor-only generalization",
    "failure_trace": "retain full raw50Hz and all decisions; extract first new contact, terminal, or peak lateral with +/-12 decision chronology; observations separated from causal hypotheses",
}
sha256, read_json, read_jsonl = prior.sha256, prior.read_json, prior.read_jsonl
statistics, analyze_episode = prior.statistics, prior.analyze_episode


def summarize(slots, episodes, errors=()):
    cells, hits, returns, failures = [], [], [], []
    outcomes = Counter()
    names = ("max_lateral", "avoidance_duration", "avoidance_integral", "path", "actual_steer")
    metric_cells = {name: [] for name in names}
    geometry_metrics = {name: {} for name in names}
    absolute = {name: [] for name in names}
    valid_denominators = {name: True for name in names}
    baseline_windows = lost_windows = baseline_finishes = 0
    laps, qualified = [], set()
    road_hashes = {}
    for track, seed in CELLS:
        pair = [episodes.get((track, seed, arm)) for arm in ARMS]
        cell: dict[str, Any] = dict(track_id=track, seed=seed, pair_available=all(pair), windows=[])
        cells.append(cell)
        if not all(pair):
            cell["status"] = "incomplete_pair"
            continue
        (be, b), (ce, c) = pair
        geometry = be["catalog"] == ce["catalog"] and be["geometry_sha256"] == ce["geometry_sha256"]
        initial = be["initial_state"] == ce["initial_state"] and be["initial_observation_sha256"] == ce["initial_observation_sha256"]
        keys = ("steer", "gas", "brake", "car_x", "car_y", "car_yaw", "progress")
        parity = min(len(be["decision_trace"]), len(ce["decision_trace"])) >= 10 and [[d.get(k) for k in keys] for d in be["decision_trace"][:10]] == [[d.get(k) for k in keys] for d in ce["decision_trace"][:10]]
        road = be["catalog"].get("track", be["catalog"].get("centerline"))
        if road is None:
            raise ValueError("catalog lacks road geometry")
        road_hash = hashlib.sha256(json.dumps(road, sort_keys=True).encode()).hexdigest()
        road_hashes.setdefault(seed, set()).add(road_hash)
        outcome = "kept" if b["completed"] and c["completed"] else "lost" if b["completed"] else "gained" if c["completed"] else "neither"
        outcomes[outcome] += 1
        baseline_finishes += b["completed"]
        cell.update(completion=outcome, geometry_equal=geometry, initial_equal=initial, first10_parity=parity,
                    road_geometry_sha256=road_hash,
                    damage_nonincrease=b["damage"] is not None and c["damage"] is not None and c["damage"] <= b["damage"],
                    collisions_nonincrease=b["collisions"] is not None and c["collisions"] is not None and c["collisions"] <= b["collisions"],
                    baseline={k: b[k] for k in ("completed", "damage", "collisions", "lapTimeMs", "max_abs_centerline_lateral_m")},
                    candidate={k: c[k] for k in ("completed", "damage", "collisions", "lapTimeMs", "max_abs_centerline_lateral_m")})
        if outcome == "kept" and b["lapTimeMs"] is not None and c["lapTimeMs"] is not None:
            laps.append(dict(track_id=track, seed=seed, baseline_ms=b["lapTimeMs"], candidate_ms=c["lapTimeMs"], delta_ms=c["lapTimeMs"] - b["lapTimeMs"]))
        if any(r["qualified_release"] for r in c["steering_timeline"]):
            qualified.add(seed)
        if [o["obstacle"]["id"] for o in b["obstacles"]] != [o["obstacle"]["id"] for o in c["obstacles"]]:
            raise ValueError("paired obstacle inventory differs")
        changes = {name: [] for name in names}
        cell_hits = []
        for bo, co in zip(b["obstacles"], c["obstacles"]):
            bw, cw = bo["fixed_window"], co["fixed_window"]
            baseline_eligible = bw["status"] == "complete" and bw["steering"]["status"] == "complete" and bo["physical"]["status"] == "passed" and bo["physical"]["geometry_valid"]
            candidate_eligible = prior.eligible(co)
            baseline_windows += baseline_eligible
            lost_windows += baseline_eligible and (not prior.eligible(bo) or not candidate_eligible)
            matched = geometry and initial and parity and baseline_eligible and prior.eligible(bo) and candidate_eligible
            metrics = {}
            if matched:
                values = dict(max_lateral=(bw["max_abs_lateral_m"], cw["max_abs_lateral_m"]),
                              avoidance_duration=(bw["avoidance"]["duration_s"], cw["avoidance"]["duration_s"]),
                              avoidance_integral=(bw["avoidance"]["integral_abs_s"], cw["avoidance"]["integral_abs_s"]),
                              path=(bw["path_length_m"], cw["path_length_m"]),
                              actual_steer=(bw["steering"]["integral_abs_s"], cw["steering"]["integral_abs_s"]))
                for name, (bv, cv) in values.items():
                    change = prior.relative_change(bv, cv)
                    metrics[name] = dict(baseline=bv, candidate=cv, fractional_change=change, zero_denominator=bv == 0)
                    absolute[name].append(cv - bv)
                    if change is not None:
                        changes[name].append(change)
                    elif cv > 0:
                        valid_denominators[name] = False
            if geometry and not bo["whole_episode_safety"]["hit"] and co["whole_episode_safety"]["hit"]:
                hit = dict(track_id=track, seed=seed, obstacle_id=bo["obstacle"]["id"])
                hits.append(hit)
                cell_hits.append(hit["obstacle_id"])
            bp, cp = bo["prospective_return"], co["prospective_return"]
            comparable = geometry and initial and parity and all(p["observed_after_clear_s"] is not None and not p["status"].startswith("invalid") for p in (bp, cp))
            common = None
            if comparable:
                horizon = min(bp["observed_after_clear_s"], cp["observed_after_clear_s"])
                paired = [prior.prospective_return(o["_states"], o["obstacle"], o["physical"], e["catalog"]["obstacles"], e, horizon) for o, e in ((bo, be), (co, ce))]
                comparable = horizon >= .24 - 1e-9 and all(not p["status"].startswith("invalid") for p in paired)
                common = dict(horizon_s=horizon, baseline=paired[0], candidate=paired[1])
            returns.append(dict(track_id=track, seed=seed, obstacle_id=bo["obstacle"]["id"], comparable=comparable, common=common, baseline=bp, candidate=cp))
            cell["windows"].append(dict(obstacle_id=bo["obstacle"]["id"], baseline_eligible=baseline_eligible, candidate_eligible=candidate_eligible, eligible=matched, metrics=metrics, baseline_window=bw, candidate_window=cw, baseline_physical=bo["physical"], candidate_physical=co["physical"]))
        cell["relative_metrics"] = {n: statistics(v) for n, v in changes.items()}
        for name, values in changes.items():
            if values:
                mean = math.fsum(values) / len(values)
                metric_cells[name].append(mean)
                geometry_metrics[name].setdefault(seed, []).append(mean)
            elif any(w["baseline_eligible"] for w in cell["windows"]):
                valid_denominators[name] = False
        if not b["completed"] or not c["completed"] or b["planned_censor"] or c["planned_censor"] or cell_hits or not cell["damage_nonincrease"] or not cell["collisions_nonincrease"]:
            failures.append(failure_trace(be, ce, b, c, outcome, cell_hits))
    eligible_cells = sum(any(w["baseline_eligible"] for w in c["windows"]) for c in cells)
    geometry_means = {n: {s: math.fsum(v) / len(v) for s, v in m.items()} for n, m in geometry_metrics.items()}
    qualifying = {n: sorted(s for s, v in m.items() if v <= -.05 + 1e-12) for n, m in geometry_means.items()}
    reductions = {n: bool(v) and len(v) == eligible_cells and valid_denominators[n] and math.fsum(v) / len(v) <= -.05 + 1e-12 and len(qualifying[n]) >= 2 for n, v in metric_cells.items()}
    equal_geometry = {n: statistics(m.values()) for n, m in geometry_means.items()}
    geometry_reductions = {}
    for name in names:
        mean = equal_geometry[name]["mean"]
        geometry_reductions[name] = reductions[name] and mean is not None and mean <= -.05 + 1e-12
    eligible_seeds = {c["seed"] for c in cells if any(w["baseline_eligible"] for w in c["windows"])}
    joint = sorted(s for s in eligible_seeds if geometry_means["max_lateral"].get(s, 0) < 0 and geometry_means["avoidance_integral"].get(s, 0) < 0)
    comparable = [r for r in returns if r["comparable"]]
    baseline_comparable = [r for r in returns if r["baseline"]["observed_after_clear_s"] is not None and not r["baseline"]["status"].startswith("invalid") and r["baseline"]["observed_after_clear_s"] >= .24 - 1e-9]
    censors = {a: sum(r["common"][label]["status"] != "returned" for r in comparable) for a, label in zip(ARMS, ("baseline", "candidate"))}
    gates = dict(all_slots_valid=len(episodes) == SLOTS and len(slots) == SLOTS and all(s["status"] == "completed" for s in slots) and not errors,
                 model_only_pair_safety=all(c.get("geometry_equal") and c.get("initial_equal") and c.get("first10_parity") for c in cells),
                 lost_finishes_zero=outcomes["lost"] == 0 and all(c["pair_available"] for c in cells),
                 per_cell_damage_nonincrease=all(c.get("damage_nonincrease", False) for c in cells),
                 per_cell_collision_nonincrease=all(c.get("collisions_nonincrease", False) for c in cells),
                 no_episode_censor=len(episodes) == SLOTS and all(not r[1]["planned_censor"] for r in episodes.values()),
                 all_objects_each=len(episodes) == SLOTS and all(sum(len(r[1]["obstacles"]) for k, r in episodes.items() if k[2] == a) == len(CELLS) * 6 for a in ARMS),
                 no_new_clean_obstacle_hit=not hits and all(c.get("geometry_equal") for c in cells) and not any(r[1]["unassociated_collision_events"] for r in episodes.values()),
                 all_baseline_windows_retained=baseline_windows > 0 and lost_windows == 0 and all(c["pair_available"] for c in cells),
                 max_lateral_reduction_5pct_two_geometries=reductions["max_lateral"],
                 avoidance_reduction_5pct_two_geometries=reductions["avoidance_duration"] or reductions["avoidance_integral"],
                 equal_geometry_lateral_reduction_5pct=geometry_reductions["max_lateral"],
                 equal_geometry_avoidance_reduction_5pct=geometry_reductions["avoidance_duration"] or geometry_reductions["avoidance_integral"],
                 joint_mechanism_on_half_eligible_geometries=bool(eligible_seeds) and len(joint) * 2 >= len(eligible_seeds),
                 sufficient_eligible_geometry_coverage=len(eligible_seeds) >= 18 and all(len({c["seed"] for c in cells if c["track_id"] == t and any(w["baseline_eligible"] for w in c["windows"])}) >= 12 for t in (1, 2, 3)),
                 path_mean_nonincrease=len(absolute["path"]) == baseline_windows > 0 and lost_windows == 0 and math.fsum(absolute["path"]) <= 0,
                 actual_steer_integral_nonincrease=len(absolute["actual_steer"]) == baseline_windows > 0 and lost_windows == 0 and math.fsum(absolute["actual_steer"]) <= 0,
                 all_kept_mean_lap_nonincrease=baseline_finishes > 0 and outcomes["kept"] == len(laps) == baseline_finishes and math.fsum(r["delta_ms"] for r in laps) <= 0,
                 comparable_return_followup_preserved=bool(baseline_comparable) and all(r["comparable"] for r in baseline_comparable),
                 comparable_return_censor_nonincrease=bool(comparable) and censors[ARMS[1]] <= censors[ARMS[0]],
                 sufficient_intervention_coverage=len(qualified) >= 2,
                 distinct_new_road_geometries=len(road_hashes) == len(SEEDS) and all(len(v) == 1 for v in road_hashes.values()) and len(set().union(*road_hashes.values())) == len(SEEDS))
    if not episodes:
        gates = {k: False for k in gates}
    summaries = {}
    for arm, label in zip(ARMS, ("baseline", "candidate")):
        reports = [r[1] for k, r in episodes.items() if k[2] == arm]
        objects = [o for r in reports for o in r["obstacles"]]
        matched = [w[label + "_window"] for c in cells for w in c["windows"] if w["eligible"]]
        summaries[arm] = dict(episodes_available=len(reports), finishes=sum(r["completed"] for r in reports),
                             damage_total=math.fsum(r["damage"] for r in reports if r["damage"] is not None),
                             collision_decisions_total=sum(r["collisions"] for r in reports if r["collisions"] is not None),
                             whole_episode_hit_objects=sum(o["whole_episode_safety"]["hit"] for o in objects), object_denominator=len(objects),
                             max_abs_centerline_lateral_m=max((r["max_abs_centerline_lateral_m"] for r in reports), default=None),
                             matched_window_path_m=statistics(w["path_length_m"] for w in matched),
                             matched_window_max_abs_lateral_m=statistics(w["max_abs_lateral_m"] for w in matched),
                             matched_window_avoidance_duration=statistics(w["avoidance"]["duration_s"] for w in matched),
                             matched_window_avoidance_integral=statistics(w["avoidance"]["integral_abs_s"] for w in matched),
                             matched_window_actual_steer_integral=statistics(w["steering"]["integral_abs_s"] for w in matched),
                             prospective_return_status_counts=dict(Counter(o["prospective_return"]["status"] for o in objects)),
                             prospective_return_denominator=len(objects),
                             return_time_summary=statistics(o["prospective_return"]["return_time_s"] for o in objects) if objects and all(o["prospective_return"]["status"] == "returned" for o in objects) else None,
                             comparable_return_censor_count=censors[arm],
                             common_restricted_response=statistics(r["common"][label]["restricted_response_s"] for r in comparable),
                             matched_kept_lap_ms=statistics(r[label + "_ms"] for r in laps),
                             release_changed_decisions=sum(r["release_changed_decisions"] for r in reports),
                             generation_suppressed_decisions=sum(d["controller"].get("steering_generation_ambiguous_motion_ignored") is True for k, (e, _) in episodes.items() if k[2] == arm for d in e["decision_trace"]))
    geometry_rows = []
    for seed in SEEDS:
        group = [c for c in cells if c["seed"] == seed]
        geometry_rows.append(dict(seed=seed, cells=len(group), completion=dict(Counter(c.get("completion", "unavailable") for c in group)),
                                  relative_metrics={n: geometry_means[n].get(seed) for n in names},
                                  arms={a: dict(finishes=sum(c[l]["completed"] for c in group if l in c), damage_total=math.fsum(c[l]["damage"] for c in group if l in c and c[l]["damage"] is not None), collision_decisions=sum(c[l]["collisions"] for c in group if l in c and c[l]["collisions"] is not None), max_lateral=max((c[l]["max_abs_centerline_lateral_m"] for c in group if l in c), default=None)) for a, l in zip(ARMS, ("baseline", "candidate"))}))
    return dict(scope="fresh-to-frozen-candidate nonprotected TRAIN generalization, not protected/blind/official", exploratory_scope=False,
                expected_slots=SLOTS, episodes_available=len(episodes), slots=slots, cells=cells,
                completion={**{k: outcomes[k] for k in ("kept", "lost", "gained", "neither")}, "expected_cells": len(CELLS)},
                baseline_eligible_windows=baseline_windows, candidate_lost_baseline_eligible_windows=lost_windows,
                cell_weighted_relative_metrics={n: statistics(v) for n, v in metric_cells.items()},
                equal_geometry_relative_metrics=equal_geometry, metric_denominators_valid=valid_denominators,
                matched_absolute_metric_deltas={n: statistics(v) for n, v in absolute.items()},
                qualifying_geometry_seeds=qualifying, joint_mechanism_geometry_seeds=joint, eligible_geometry_seeds=sorted(eligible_seeds),
                qualified_release_geometry_seeds=sorted(qualified), per_geometry_results=geometry_rows,
                per_track_results=[dict(track_id=t, completion=dict(Counter(c.get("completion", "unavailable") for c in cells if c["track_id"] == t)), baseline_eligible_windows=sum(w["baseline_eligible"] for c in cells if c["track_id"] == t for w in c["windows"]), relative_metrics={n: statistics(c["relative_metrics"][n]["mean"] for c in cells if c["track_id"] == t and c.get("relative_metrics", {}).get(n, {}).get("mean") is not None) for n in names}) for t in (1, 2, 3)],
                baseline_window_status_counts=dict(Counter(o["fixed_window"]["status"] for k, (_, r) in episodes.items() if k[2] == ARMS[0] for o in r["obstacles"])),
                common_return_cohort=returns, common_return_cohort_n=len(comparable), baseline_return_comparable_n=len(baseline_comparable),
                mutually_finished_laps=laps, mutually_finished_lap_delta_ms=statistics(r["delta_ms"] for r in laps),
                new_clean_obstacle_hits=hits, arm_summaries=summaries, failure_traces=failures,
                gates=gates, gate_passed=all(gates.values()), verification_errors=list(errors),
                decision="GENERALIZED_INTERNAL_ADOPTION_CANDIDATE" if all(gates.values()) else "NOT_PROMOTED_FROZEN_V2",
                interpretation="Association of less avoidance/lateral and safety is not causal mediation proof. Failed gates cannot be offset by survivors. Return restricted response is a descriptive onset/censor-bound mixture, not KM/RMST or population return time. v2 remains frozen; followup must be separate.")


def failure_trace(baseline, candidate, baseline_report, candidate_report, outcome, new_hits):
    traces = {}
    for arm, episode, report in zip(ARMS, (baseline, candidate), (baseline_report, candidate_report)):
        decisions = episode["decision_trace"]
        states = report["obstacles"][0]["_states"]
        peak = max(states, key=lambda s: abs(s["lateral"]))
        contact_times = [t for o in report["obstacles"] if o["obstacle"]["id"] in new_hits for kind in ("contact_times", "geometric_hit_times") for t in o["whole_episode_safety"][kind]]
        contact_t = min(contact_times) if contact_times else None

        def step_at(t):
            return next((d["step"] for d in decisions if d["controller"]["evaluation_only"]["pre"]["t"] - 1e-9 <= t <= d["controller"]["evaluation_only"]["post"]["t"] + 1e-9), None)

        anchors = {"terminal": dict(step=len(decisions), t=states[-1]["t"]), "peak_lateral": dict(step=step_at(peak["t"]), t=peak["t"], raw_lateral=peak["lateral"])}
        intervention = next((d["step"] for d in reversed(decisions) if d["controller"].get("steering_release_changed") or d["controller"].get("steering_generation_ambiguous_motion_ignored")), None)
        if intervention is not None:
            anchors["last_intervention"] = dict(step=intervention, t=decisions[intervention - 1]["controller"]["evaluation_only"]["pre"]["t"])
        if contact_t is not None:
            anchors["first_new_contact"] = dict(step=step_at(contact_t), t=contact_t, raw50hz_or_initial=True)
        excerpts = {}
        for name, anchor in anchors.items():
            step = anchor["step"]
            if step is None:
                excerpts[name] = dict(anchor=anchor, decisions=[], status="outside_executed_decision_intervals")
                continue
            rows = []
            for d in decisions[max(0, step - 13):min(len(decisions), step + 12)]:
                ctrl = d["controller"]
                state = ctrl["evaluation_only"]["post"]
                rows.append(dict(step=d["step"], steer=d["steer"], gas=d["gas"], brake=d["brake"],
                                 state={k: state[k] for k in ("t", "speed", "station", "lateral", "heading_error", "contacts", "clearance")},
                                 near_object=ctrl.get("near_object"), projected_obstacle_x=ctrl.get("projected_obstacle_x"),
                                 release_changed=ctrl.get("steering_release_changed"), release_reason=ctrl.get("steering_release_reason"),
                                 release_gate=ctrl.get("steering_release_gate"), avoidance_component=ctrl["steering_terms"].get("avoidance_component"),
                                 actual_obstacle_side=ctrl.get("actual_obstacle_side"), generation={k: v for k, v in ctrl.items() if k.startswith("steering_generation_")}))
            excerpts[name] = dict(anchor=anchor, anchor_step=step, decisions=rows)
        traces[arm] = dict(completed=episode["completed"], retire_reason=episode["retire_reason"], raw_trace_file=episode["raw_trace_file"], raw_trace_sha256=episode["raw_trace_sha256"], excerpts=excerpts)
    return dict(track_id=baseline["track_id"], seed=baseline["seed"], completion=outcome, new_hit_objects=new_hits, arms=traces,
                causal_status="chronological observations only; no v2 correction or counterfactual rerun")


def load_run(run, candidate, manifest):
    from scripts import evaluate_koi_steering_generalization as operator
    run = Path(run).resolve()
    protocol, report = read_json(run / "protocol.json"), read_json(run / "episode-report.json")
    errors = []
    if report["protocol_sha256"] != sha256(run / "protocol.json"):
        raise ValueError("protocol hash mismatch")
    validated, exposure = operator.validate_frozen(sha256(run / "protocol.json"))
    if validated != protocol:
        raise ValueError("canonical protocol differs")
    review_pin = report["review_receipt"]
    operator.validate_review(protocol, report["protocol_sha256"], review_pin["path"], review_pin["sha256"])
    if read_json(run / "execution-review.json") != review_pin:
        raise ValueError("execution review/report differs")
    if protocol["analysis_spec"] != ANALYSIS_SPEC or protocol["analyzer_sha256"] != sha256(__file__) or protocol["operator_sha256"] != sha256(operator.__file__):
        raise ValueError("frozen analysis/operator differs")
    if [(r["track_id"], r["geometry_seed"]) for r in protocol["cells"]] != CELLS or protocol["episodes"] != SLOTS:
        raise ValueError("fixed new cohort differs")
    if any(r.get("partition") != "TRAIN" or r.get("obstacles") is not True for r in protocol["cells"]):
        raise ValueError("TRAIN obstacle conditions differ")
    if protocol["schedule"] != operator.scheduled_slots() or any(protocol.get(k) != v for k, v in (("frame_skip", 4), ("warmup_ticks", 50), ("raw_fps", 50), ("max_decisions", 1200))):
        raise ValueError("schedule/runtime conditions differ")
    if len(protocol["environment_source_sha256"]) != 142 or set(protocol["helper_source_sha256"]) != {str(p) for p in operator.helper_paths()}:
        raise ValueError("transitive source closure differs")
    prior.verify_models(protocol, run, candidate, manifest)
    if protocol["model_hashes"][ARMS[1]] != CANDIDATE_SHA:
        raise ValueError("frozen v2 differs")
    runtime = [h for p, h in protocol["helper_source_sha256"].items() if p.endswith("/haic/algorithms/koi/steering_release.py")]
    if runtime != [protocol["model_source_sha256"][ARMS[1]]["haic_agent/steering_release_runtime.py"]]:
        raise ValueError("candidate runtime/source differs")
    for group in ("environment_source_sha256", "helper_source_sha256"):
        for path, expected in protocol[group].items():
            if sha256(path) != expected:
                raise ValueError("source changed: " + path)
    for suffix, expected in prior.physical.FROZEN_SOURCE_SHA256.items():
        if [h for p, h in protocol["environment_source_sha256"].items() if p.endswith("/" + suffix)] != [expected]:
            raise ValueError("frozen environment differs: " + suffix)
    for path, expected in protocol["reference_evidence"]["source_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("frozen prior evidence changed")
    for path, expected in protocol["root_agent_source_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("root Agent changed")
    copies = {}
    for row in protocol["source_copies"]:
        prior.physical.verify_file(run, row["file"], row["sha256"])
        if row["source"] in copies:
            raise ValueError("duplicate source copy")
        copies[row["source"]] = row["sha256"]
    if any(copies.get(p) != h for group in ("environment_source_sha256", "helper_source_sha256") for p, h in protocol[group].items()):
        raise ValueError("transitive source copies differ")
    slots: dict[Any, dict[str, Any]] = {(t, s, a): dict(track_id=t, seed=s, mode=a, status="unrun") for t, s in CELLS for a in ARMS}
    episodes, seen, saved = {}, set(), set()
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
                prior.validate_raw_rows(ticks)
                slots[key]["partial_raw"] = dict(file=partial.name, sha256=sha256(partial), complete_ticks=len(ticks), truncated_last_line=truncated)
            decisions_path = run / f"{key[0]}-{key[1]}-{key[2]}.decisions.jsonl"
            if decisions_path.exists():
                decisions, truncated = read_jsonl(decisions_path, partial=True)
                validate_action_event_order(decisions)
                slots[key]["partial_decisions"] = dict(file=decisions_path.name, sha256=sha256(decisions_path), records=len(decisions), truncated_last_line=truncated)
            bound_path = run / f"{key[0]}-{key[1]}-{key[2]}.bound-process.json"
            if bound_path.exists():
                bound = read_json(bound_path)
                if bound["slot_id"] != operator.slot_id(*key) or bound["operator_sha256"] != protocol["operator_sha256"]:
                    raise ValueError("partial process source/event differs")
                original = prior.physical.verify_file(run, bound["original_process_file"], bound["original_process_sha256"])
                receipt = read_json(original)
                if any(bound[k] != receipt[k] for k in ("status", "error", "wall_time_s")):
                    raise ValueError("partial bound process differs")
                slots[key]["partial_process"] = dict(file=bound_path.name, sha256=sha256(bound_path), original_file=original.name, original_sha256=sha256(original), receipt=bound)
            continue
        episode = read_json(prior.physical.verify_file(run, row["file"], row["sha256"]))
        if (episode.get("track_id"), episode.get("seed"), episode.get("mode")) != key:
            raise ValueError("saved episode identity differs")
        saved.add(key)
        process = read_json(prior.physical.verify_file(run, row["process_file"], row["process_sha256"]))
        if process["status"] != "completed" or process["error"] or process["slot_id"] != operator.slot_id(*key) or process["operator_sha256"] != protocol["operator_sha256"]:
            raise ValueError("process event/source differs")
        raw, _ = read_jsonl(prior.physical.verify_file(run, episode["raw_trace_file"], episode["raw_trace_sha256"]))
        receipt = read_json(prior.physical.verify_file(run, process["original_process_file"], process["original_process_sha256"]))
        if any(process[k] != receipt[k] for k in ("status", "error", "wall_time_s")):
            raise ValueError("bound process differs")
        decisions, _ = read_jsonl(prior.physical.verify_file(run, episode["partial_decisions_file"], episode["partial_decisions_sha256"]))
        if type(episode.get("raw_ticks")) is not int or episode["raw_ticks"] != len(raw):
            raise ValueError("saved raw denominator differs")
        invalid = episode.get("error") or episode.get("invalid_actions") or episode.get("retire_reason") in prior.physical.OPERATIONAL_RETIREMENTS or not episode.get("damage_telemetry_valid") or not episode.get("collision_telemetry_valid") or episode.get("versions") != protocol["runtime_versions"] or not episode.get("runtime_module_paths")
        if invalid:
            errors.append("invalid_episode:" + str(key))
            prior.validate_raw_rows(raw)
            validate_streamed_decisions(episode, decisions, partial=True)
            slots[key]["invalid_saved_episode"] = dict(error=episode.get("error"), retire_reason=episode.get("retire_reason"), raw_trace_file=episode["raw_trace_file"], raw_trace_sha256=episode["raw_trace_sha256"], complete_raw_ticks=len(raw), partial_decisions_file=episode["partial_decisions_file"], partial_decisions_sha256=episode["partial_decisions_sha256"], decision_records=len(decisions))
            continue
        prior.validate_episode(episode, raw, key, protocol)
        validate_streamed_decisions(episode, decisions)
        road_digest = hashlib.sha256(json.dumps(episode["catalog"]["track"], sort_keys=True).encode()).hexdigest()
        if episode["track_geometry_sha256"] != road_digest:
            raise ValueError("road-only digest differs")
        if road_digest in protocol["reference_evidence"]["prior_road_sha256"]:
            errors.append("old_consumed_road_geometry:" + str(key))
        episodes[key] = episode, analyze_episode(episode, raw)
    if len(seen) != SLOTS:
        errors.append("incomplete_schedule")
    if report.get("operator_error"):
        errors.append("operator_error:" + report["operator_error"])
    ledger, _ = read_jsonl(prior.physical.verify_file(run, "reset-ledger.jsonl", report["reset_ledger_sha256"]))
    expected = [(r["track_id"], r["seed"], r["mode"]) for r in protocol["schedule"]]
    completed = set()
    for i, event in enumerate(ledger):
        position = i // 2
        if position >= SLOTS or event.get("status") != ("reset_intent" if i % 2 == 0 else "completed"):
            raise ValueError("reset ledger event order differs")
        key = (event["track"], event["seed"], event["arm"]) if i % 2 == 0 else (event["track_id"], event["seed"], event["mode"])
        if key != expected[position] or event.get("slot_id") != operator.slot_id(*key):
            raise ValueError("reset ledger identity differs")
        if i % 2:
            if event != {k: v for k, v in slots[key].items() if k not in ("partial_raw", "partial_decisions", "invalid_saved_episode")}:
                raise ValueError("ledger report differs")
            completed.add(key)
    if completed != saved:
        raise ValueError("ledger episode association differs")
    if len(ledger) != SLOTS * 2:
        errors.append("incomplete_reset_ledger")
    operator.check_claims(exposure, protocol_sha256=sha256(run / "protocol.json"))
    return protocol, list(slots.values()), episodes, errors


def validate_streamed_decisions(episode, rows, partial=False):
    validate_action_event_order(rows)
    expected = episode["steps"] * 2
    if len(episode["decision_trace"]) != episode["steps"] or len(rows) < expected or len(rows) - expected not in ((0, 1, 2) if partial else (0,)):
        raise ValueError("streamed decision denominator differs")
    for index, decision in enumerate(episode["decision_trace"]):
        intent, returned = rows[index * 2:index * 2 + 2]
        controller = decision["controller"]
        if intent.get("status") != "act_intent" or returned.get("status") != "act_returned":
            raise ValueError("streamed action event order differs")
        if returned["action"] != [decision[k] for k in ("steer", "gas", "brake")]:
            raise ValueError("streamed action differs")
        if returned["steering_terms"] != controller["steering_terms"]:
            raise ValueError("streamed components differ")
        if any(row["baseline_pre_act_state"] != controller["baseline_pre_act_state"] for row in (intent, returned)) or returned["baseline_post_act_state"] != controller["baseline_post_act_state"]:
            raise ValueError("streamed controller state differs")
    for offset, row in enumerate(rows[expected:]):
        if row.get("status") != ("act_intent" if offset == 0 else "act_returned"):
            raise ValueError("unapplied streamed action event order differs")


def validate_action_event_order(rows):
    for index, row in enumerate(rows):
        if row.get("status") != ("act_intent" if index % 2 == 0 else "act_returned"):
            raise ValueError("streamed action event order differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, slots, episodes, errors = load_run(args.run, args.candidate, args.manifest or args.candidate.with_suffix(".manifest.json"))
    result = summarize(slots, episodes, errors)
    result.update(study=protocol["study"], analysis_spec=ANALYSIS_SPEC, analyzer_sha256=sha256(__file__), protocol_sha256=sha256(args.run / "protocol.json"), episode_report_sha256=sha256(args.run / "episode-report.json"), model_hashes=protocol["model_hashes"], passive_analysis=True, environment_resets=0, model_updates=0, official_action=False)
    prior.finite_tree(result)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("episodes_available", "completion", "gates", "gate_passed", "decision")}, indent=2))


if __name__ == "__main__":
    main()
