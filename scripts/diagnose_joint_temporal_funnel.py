"""Passive, source-bound reduction of the interval pilot's saved decisions.

No policy, predictor, scene extraction, simulator or new forecasts are run. The
fixed first-failure order is a presentation convention, NOT causal attribution.
Literal reasons, intersections and cost/risk cross-tables preserve overlaps.
Run with ``python -B -m scripts.diagnose_joint_temporal_funnel``. Output is exclusive.
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
from typing import Any, TypeGuard


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "runs/joint-temporal-interval-v1/pilot"
PROTOCOL_SHA256 = "11fac3c4c5d337487675b8a69fc650a5420c0653f48548d0b13f2599620224c3"
FLOOR = .05
PRE_ORDER = (
    "observer_or_forward", "mapping", "pilot_envelope",
    "shield_recovery_feedback", "calibration_or_budget",
    "unclassified_source_reason", "comparison_missing",
)
POST_ORDER = (
    "common_17_tick_cost_unsupported", "absolute_support_missing",
    "road_margin_veto", "obstacle_margin_veto", "unclassified_risk_veto",
    "material_baseline_advantage", "central_midpoint_no_nominal_gain",
    "shared_stress_removes_nominal_gain", "residual_allowance_removes_robust_gain",
    "supported_no_risk_gain",
)
PRE_REASONS = {
    "observer_invalid": "observer_or_forward",
    "not_supported_forward_motion": "observer_or_forward",
    "mapping_invalid": "mapping",
    "mapping_history_missing": "mapping",
    "hud_speed_outside_pilot_envelope": "pilot_envelope",
    "action_outside_pilot_envelope": "pilot_envelope",
    "shield_active_threat_or_blocked": "shield_recovery_feedback",
    "shield_projection_unsupported": "shield_recovery_feedback",
    "nominal_recovery_or_contact": "shield_recovery_feedback",
    "feedback_capture_unavailable": "shield_recovery_feedback",
    "calibration_missing_or_incomplete": "calibration_or_budget",
    "calibration_invalid": "calibration_or_budget",
    "intervention_budget": "calibration_or_budget",
    "never_intervene": "calibration_or_budget",
    "comparison_error": "comparison_missing",
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def finite(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flat(value):
    if isinstance(value, list):
        return [item for child in value for item in flat(child)]
    return [value]


def shape(value, dimensions, name):
    if not dimensions:
        return
    if not isinstance(value, list) or len(value) != dimensions[0]:
        raise ValueError(f"{name} must have shape {dimensions}")
    for child in value:
        shape(child, dimensions[1:], name)


def same(actual, expected, name):
    a, b = flat(actual), flat(expected)
    if len(a) != len(b):
        raise ValueError(f"{name}: incompatible lengths")
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if not finite(x) or not finite(y) or not math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"{name}: saved/reduced values differ ({x!r}, {y!r})")


def minimum(values):
    known = [v for v in flat(values) if finite(v)]
    return min(known) if known else None


def reason_statistics(reason_sets):
    """Counts have decision denominators; an overlap is not a unique exclusion."""
    sets = [sorted(set(reasons)) for reasons in reason_sets]
    counts = Counter(reason for reasons in sets for reason in reasons)
    combinations = Counter(tuple(reasons) for reasons in sets)
    intersections = Counter(pair for reasons in sets for pair in itertools.combinations(reasons, 2))
    return dict(denominator=len(sets), nonexclusive_counts=dict(sorted(counts.items())),
                exact_combinations=[dict(reasons=list(key), count=n)
                                    for key, n in sorted(combinations.items())],
                pairwise_overlap=[dict(reasons=list(key), count=n)
                                  for key, n in sorted(intersections.items())])


def cost_category(center, shared, final, supported):
    """Baseline advantage is final lower>.05, not an uncertainty rejection.

    This material-baseline subset precedes central>=-.05 so classes are disjoint.
    The broader central-no-gain predicate is ALSO counted independently.
    """
    if not supported:
        return "common_17_tick_cost_unsupported"
    if final[0] > FLOOR:
        return "material_baseline_advantage"
    if center >= -FLOOR:
        return "central_midpoint_no_nominal_gain"
    if shared[1] >= -FLOOR:
        return "shared_stress_removes_nominal_gain"
    if final[1] >= -FLOOR:
        return "residual_allowance_removes_robust_gain"
    return "robust_gain"


def reduce_comparison(comparison, policy, residual) -> dict[str, Any]:
    costs = comparison["reference_costs"]
    shape(costs, (2, 19, 5), "reference_costs")
    shape(comparison["reference_delta"], (2, 19, 5), "reference_delta")
    shape(comparison["delta_interval"], (2, 2), "delta_interval")
    for key in ("supported_ticks", "footprint_known_ticks", "absolute_supported", "veto"):
        shape(comparison[key], (2, 19), key)
    shape(comparison["cost_supported"], (2,), "cost_supported")
    shape(comparison["abstain_reasons"], (2,), "abstain_reasons")
    if comparison["comparison_ticks"] != 17:
        raise ValueError("comparison does not use the complete 17 endpoints")
    # Subtract each shared (state, reference) BEFORE taking extrema. Never use
    # independently optimized baseline/alternative or marginal min/max costs.
    deltas = [[[a - b if finite(a) and finite(b) else None
                for a, b in zip(scenario, baseline)]
               for scenario, baseline in zip(arm, costs[0])] for arm in costs]
    same(comparison["reference_delta"], deltas, "aligned reference_delta")
    all_finite = all(finite(v) for v in flat(costs))
    complete_ticks = all(v == 17 for v in flat(comparison["supported_ticks"]))
    full = bool(comparison["common_support"] and all(comparison["cost_supported"]))
    if full != (bool(comparison["common_support"]) and all_finite and complete_ticks):
        raise ValueError("inconsistent common full-horizon support")
    center = deltas[1][0][0]
    values = flat(deltas[1])
    shared = [min(values), max(values)] if all(finite(v) for v in values) else None
    if full and shared is None:
        raise ValueError("supported comparison has missing shared costs")
    final = [shared[0] - residual, shared[1] + residual] if full and shared is not None else None
    same(comparison["delta_interval"][1], final or [None, None], "final interval")
    if full:
        same(comparison["delta_interval"][0], [0., 0.], "baseline interval")
    reasons = list(comparison["abstain_reasons"][1])
    absolute = all(comparison["absolute_supported"][1])
    road = "road_margin" in reasons
    obstacle = "obstacle_margin" in reasons
    veto = any(comparison["veto"][1])
    if veto != (not absolute or road or obstacle):
        raise ValueError("veto mask disagrees with literal support/road/obstacle reasons")
    risk = (["absolute_support_missing"] if not absolute else [])
    risk += [name for name, yes in (("road_margin_veto", road), ("obstacle_margin_veto", obstacle)) if yes]
    category = cost_category(center, shared, final, full)
    performance = category
    if category == "robust_gain":
        performance = "gain_but_risk_veto" if veto else "supported_no_risk_gain"
    post_flags = ([POST_ORDER[0]] if not full else []) + risk
    post_flags += [performance] if category != "robust_gain" else ([] if veto else [performance])
    first = next(key for key in POST_ORDER if key in post_flags)
    # Source creates the candidate in a float32 proposal copy. Reconstruct only
    # that recorded comparison's action, never a forecast for a pre-excluded row.
    proposal = policy["proposal_action"]
    shape(proposal, (3,), "proposal_action")
    f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    steer, gas, brake = proposal
    candidate = [f32(math.copysign(max(0., abs(steer) - .04), steer) if steer else .04),
                 f32(min(1., gas + .05)), f32(max(0., brake - .05))]
    pose_delta = None
    if "poses" in comparison:
        shape(comparison["poses"], (2, 19, 17, 3), "poses")
        pose_delta = [a - b for a, b in zip(comparison["poses"][1][0][-1],
                                           comparison["poses"][0][0][-1])]
    return dict(full_cost_support=full, all_38_paths_have_17_cost_ticks=complete_ticks,
                all_reference_costs_finite=all_finite,
                support_ticks_by_arm=comparison["supported_ticks"],
                footprint_known_ticks_by_arm=comparison["footprint_known_ticks"],
                absolute_support_by_arm=[all(arm) for arm in comparison["absolute_supported"]],
                absolute_supported_scenarios_by_arm=[sum(arm) for arm in comparison["absolute_supported"]],
                alternative_absolute_support=absolute, road_veto=road, obstacle_veto=obstacle,
                any_risk_veto=veto, risk_reasons=risk, comparison_reasons=reasons,
                central_midpoint_delta=center, shared_19x5_interval=shared,
                final_interval=final, residual_allowance=residual,
                shared_extreme_variants=({side: [list(index) for index in itertools.product(range(19), range(5))
                                               if deltas[1][index[0]][index[1]] == value]
                                         for side, value in zip(("lower", "upper"), shared)} if shared else None),
                cost_category=category, performance_category=performance, post_first_failure=first,
                nominal_no_gain=center >= -FLOOR if finite(center) else None,
                central_material_baseline_advantage=center > FLOOR if finite(center) else None,
                shared_material_baseline_advantage=shared[0] > FLOOR if full and shared is not None else None,
                final_material_baseline_advantage=final[0] > FLOOR if final is not None else None,
                nominal_beneficial_margin=-FLOOR - center if finite(center) else None,
                shared_beneficial_margin=-FLOOR - shared[1] if full and shared is not None else None,
                final_beneficial_margin=-FLOOR - final[1] if final is not None else None,
                proposal_action=proposal, candidate_action=candidate,
                candidate_action_delta=[a - b for a, b in zip(candidate, proposal)],
                observer_gas_state=policy.get("observer_state", {}).get("gas_state"),
                observer_forward_speed=policy.get("observer_state", {}).get("forward_speed"),
                pixel_speed=policy.get("champion_proposal_diagnostics", {}).get("pixel_speed"),
                central_H4_pose_delta=pose_delta,
                known_absolute_road_min=minimum(comparison.get("absolute_road_clearance", [[], []])[1]),
                known_absolute_obstacle_min=minimum(comparison.get("absolute_obstacle_clearance", [[], []])[1]))


def summarize(records) -> dict[str, Any]:
    comparisons = [r for r in records if r["comparison_computed"]]
    pre = [r for r in records if not r["comparison_computed"]]
    supported = [r for r in comparisons if r["full_cost_support"]]
    safe = [r for r in supported if not r["any_risk_veto"]]
    predicates = (
        "full_cost_support", "all_38_paths_have_17_cost_ticks", "all_reference_costs_finite",
        "alternative_absolute_support", "road_veto", "obstacle_veto", "any_risk_veto",
        "nominal_no_gain", "central_material_baseline_advantage",
        "shared_material_baseline_advantage", "final_material_baseline_advantage",
    )
    cross = Counter((r["cost_category"], tuple(r["risk_reasons"])) for r in comparisons)
    categories = ("cost_category", "performance_category", "post_first_failure")
    best = {}
    for scope, subset in (("full_cost_supported", supported), ("full_cost_supported_no_risk", safe)):
        best[scope] = dict(denominator=len(subset))
        for margin in ("nominal_beneficial_margin", "shared_beneficial_margin", "final_beneficial_margin"):
            winner = max(subset, key=lambda r: r[margin]) if subset else None
            best[scope][margin] = dict(maximum=winner[margin], decision=winner["id"]) if winner else None
    group_keys = Counter((r["comparison_computed"], r["first_failure"], tuple(r["source_reasons"]))
                        for r in records)
    return dict(denominators=dict(decision_ends=len(records), pre_comparison_no_result=len(pre),
                                 computed_comparisons=len(comparisons), full_cost_supported=len(supported),
                                 full_cost_supported_no_risk=len(safe)),
                interventions=sum(r["intervention"] for r in records),
                exclusive_funnel=dict(sorted(Counter(r["first_failure"] for r in records).items())),
                pre_exclusive={key: sum(r["first_failure"] == "pre:" + key for r in pre) for key in PRE_ORDER},
                literal_reasons_all=reason_statistics([r["source_reasons"] for r in records]),
                literal_reasons_pre=reason_statistics([r["source_reasons"] for r in pre]),
                literal_reasons_post=reason_statistics([r["source_reasons"] for r in comparisons]),
                pre_gate_overlap=reason_statistics([r["pre_groups"] for r in pre]),
                observer_invalid_details=reason_statistics([r["observer_invalid_reasons"] for r in records]),
                comparison_predicates={key: dict(count=sum(r.get(key) is True for r in comparisons),
                                                 known=sum(r.get(key) is not None for r in comparisons),
                                                 denominator=len(comparisons)) for key in predicates},
                comparison_categories={key: dict(sorted(Counter(r[key] for r in comparisons).items())) for key in categories},
                risk_cost_cross_table=[dict(cost_category=key[0], risk_reasons=list(key[1]), count=n)
                                       for key, n in sorted(cross.items())],
                best_beneficial_margins=best,
                decision_group_counts=[dict(comparison_computed=key[0], first_failure=key[1],
                                            source_reasons=list(key[2]), count=n)
                                       for key, n in sorted(group_keys.items())])


def reduce_decisions(rows, calibration) -> dict[str, Any]:
    residual = calibration["paired_cost_residual"]
    if not finite(residual) or residual < 0 or calibration.get("cost_floor", FLOOR) != FLOOR:
        raise ValueError("invalid or different calibrated cost contract")
    records: list[dict[str, Any]] = []
    identities = set()
    for row in rows:
        if row.get("event") != "decision":
            continue
        policy = row["policy"]
        identity = f"{row['cell']}/{row['step']}"
        if identity in identities:
            raise ValueError("duplicate decision end: " + identity)
        identities.add(identity)
        reasons = sorted(set(policy.get("reasons", [])))
        computed = policy.get("comparison") is not None
        record: dict[str, Any] = dict(id=identity, cell=row["cell"], step=row["step"],
                      decision_index=policy.get("decision_index"), source_reasons=reasons,
                      observer_invalid_reasons=policy.get("observer_invalid_reasons", []),
                      comparison_computed=computed, logged_eligible=policy.get("eligible"),
                      intervention=bool(policy.get("intervention")), pre_groups=[])
        if not computed:
            groups = {("mapping" if reason.startswith("mapping_calibration_unsupported:") else
                       PRE_REASONS.get(reason, "unclassified_source_reason")) for reason in reasons}
            if not groups:
                groups.add("comparison_missing")
            record.update(pre_groups=[key for key in PRE_ORDER if key in groups],
                          first_failure="pre:" + next(key for key in PRE_ORDER if key in groups),
                          cost_status="NOT_COMPUTED", counterfactual_forecast="NOT_COMPUTED")
        else:
            record.update(reduce_comparison(policy["comparison"], policy, residual))
            record["first_failure"] = "post:" + record["post_first_failure"]
            should_intervene = (record["full_cost_support"] and not record["any_risk_veto"]
                                and record["cost_category"] == "robust_gain")
            if should_intervene != record["intervention"]:
                raise ValueError("reduced selection disagrees with logged intervention: " + identity)
        records.append(record)
    report: dict[str, Any] = dict(schema="haic-joint-temporal-funnel-v1", method=dict(
        pre_first_failure_order=list(PRE_ORDER), post_first_failure_order=list(POST_ORDER),
        ordering="First matching gate only; fixed diagnostic order, not runtime order or causal attribution.",
        cost_categories="Full support first; final lower>.05 material baseline advantage; otherwise central>=-.05 no nominal gain; otherwise shared upper>=-.05 stress alone removes gain; otherwise final upper>=-.05 residual alone removes robust gain; otherwise robust gain. Risk independently splits robust gain into gain_but_risk_veto or supported_no_risk_gain.",
        cost_delta="alternative minus baseline at the SAME state and SAME reference; negative is beneficial.",
        center="shared state index0 and reference index0 (midpoint), not midpoint of extrema",
        shared_variants="19 ordered states x 5 ordered references; no independent extrema subtraction",
        beneficial_margin="-.05 minus central delta/shared upper/final upper; >0 clears material-gain floor",
        missing_comparison="NOT_COMPUTED: no image frames available; no counterfactual forecast, invented cost, support or risk",
        risk="Alternative's absolute support and source literal road_margin/obstacle_margin flags, independent of cost. Masked clearance minima are descriptive only, not a replacement for source flags.",
        interpretation="Saved consumed-TRAIN proxy predictions only. Not measured candidate effects, causal attribution, lap gains or safety guarantees."),
        summary=summarize(records),
        by_cell={cell: summarize([r for r in records if r["cell"] == cell])
                 for cell in sorted({r["cell"] for r in records})},
        comparisons=[r for r in records if r["comparison_computed"]])
    return report


def load_pilot(pilot, expected_protocol):
    """Verify only the pilot inputs and five source files used to interpret them."""
    pins = {}

    def bound(path, expected=None, parse=True) -> Any:
        path = Path(path)
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("source/input hash differs: " + str(path))
        pins[str(path.resolve())] = digest
        return json.loads(path.read_text()) if parse else None

    protocol = bound(pilot / "protocol.json", expected_protocol)
    if protocol["schema"] != "haic-joint-temporal-pilot-v1":
        raise ValueError("unrecognized pilot protocol")
    calibration = bound(pilot / "calibration.json", protocol["calibration_sha256"])
    for relative in ("haic/algorithms/joint_control/successor.py", "haic/algorithms/joint_control/observer.py",
                     "haic/algorithms/joint_control/comparison.py", "haic/algorithms/joint_control/interval_comparison.py",
                     "scripts/run_joint_temporal_pilot.py"):
        digest = protocol["source_sha256"][relative]
        bound(pilot / "source" / relative, digest, parse=False)
        if "joint_control" in relative and not relative.endswith("successor.py"):
            if calibration["source_pins"][str(Path(protocol["repository_root"]) / relative)] != digest:
                raise ValueError("calibration/pilot source mismatch: " + relative)
    receipt = bound(pilot / "episode-report.json")
    if receipt["protocol_sha256"] != expected_protocol:
        raise ValueError("episode receipt protocol differs")
    rows = []
    streams = []
    for slot in protocol["schedule"]:
        if slot["mode"] != "successor":
            continue
        entries = [r for r in receipt["rows"] if r["file"] == slot["file"]]
        if len(entries) != 1 or entries[0]["status"] != "completed":
            raise ValueError("requires completed source-bound successor episode")
        episode = bound(pilot / slot["file"], entries[0]["sha256"])
        path = pilot / episode["decision_stream_file"]
        bound(path, episode["decision_stream_sha256"], parse=False)
        cell = f"{slot['track_id']}/{slot['seed']}"
        count = 0
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["event"] == "decision":
                    count += 1
                    if row["step"] != count:
                        raise ValueError("nonconsecutive decision ends")
                    rows.append(dict(row, cell=cell))
        if count != episode["steps"]:
            raise ValueError("decision ends differ from episode receipt")
        streams.append(dict(cell=cell, file=str(path.resolve()), decision_ends=count))
    return rows, calibration, pins, streams


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--protocol-sha256", default=PROTOCOL_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/joint-temporal-diagnosis-v1/funnel.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, calibration, pins, streams = load_pilot(args.pilot, args.protocol_sha256)
    report = reduce_decisions(rows, calibration)
    pins[str(Path(__file__).resolve())] = sha256(__file__)
    test = ROOT / "tests/test_joint_temporal_funnel.py"
    pins[str(test)] = sha256(test)
    report["provenance"] = dict(protocol_sha256=args.protocol_sha256, input_and_source_sha256=pins,
                                streams=streams, predictor_calls=0, simulator_resets=0)
    # Recheck just these consumed inputs before an exclusive output, not old audits.
    for path, digest in pins.items():
        if sha256(path) != digest:
            raise ValueError("input changed during reduction: " + path)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(encoded)
    print(json.dumps(dict(output=str(args.output), sha256=sha256(args.output),
                          denominators=report["summary"]["denominators"],
                          exclusive_funnel=report["summary"]["exclusive_funnel"],
                          comparison_categories=report["summary"]["comparison_categories"],
                          best_beneficial_margins=report["summary"]["best_beneficial_margins"]), indent=2))
    return report


if __name__ == "__main__":
    main()
