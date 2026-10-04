"""Source-bound passive diagnostics for the completed new-road matched prefix."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from scripts import analyze_koi_steering_generalization as analysis
from scripts import evaluate_koi_steering_generalization as operator


def decision_view(d):
    c = d["controller"]
    state = c["evaluation_only"]["post"]
    return dict(step=d["step"], t=state["t"], steer=d["steer"], gas=d["gas"], brake=d["brake"],
                state={k: state[k] for k in ("station", "lateral", "heading_error", "speed", "contacts", "clearance")},
                near_object=c.get("near_object"), projected_obstacle_x=c.get("projected_obstacle_x"),
                release_changed=c.get("steering_release_changed"), release_reason=c.get("steering_release_reason"),
                release_gate=c.get("steering_release_gate"), actual_obstacle_side=c.get("actual_obstacle_side"),
                generation={k: v for k, v in c.items() if k.startswith("steering_generation_")},
                steering_terms=c["steering_terms"])


def analyze(result, episodes) -> dict[str, Any]:
    pairs = [c for c in result["cells"] if c["pair_available"]]
    complete = {arm: [episodes[(c["track_id"], c["seed"], arm)] for c in pairs] for arm in analysis.ARMS}
    summaries = {}
    for arm in analysis.ARMS:
        rows = complete[arm]
        returns = [r["baseline" if arm == analysis.ARMS[0] else "candidate"] for r in result["common_return_cohort"]]
        hits = 0
        for e, raw in rows:
            states = [e["initial_state"], *raw]
            for index, obj in enumerate(e["catalog"]["obstacles"]):
                hits += any(obj["id"] in s["contacts"] or s["clearance"][index] <= 0 for s in states)
        summaries[arm] = dict(episode_denominator=len(rows), completed=sum(e["completed"] for e, _ in rows),
                             damage_total=math.fsum(e["damage"] for e, _ in rows),
                             collision_positive_decisions=sum(e["collisions"] for e, _ in rows),
                             object_denominator=len(rows) * 6, whole_episode_hit_objects=hits,
                             whole_episode_max_lateral_m=max(abs(s["lateral"]) for e, raw in rows for s in [e["initial_state"], *raw]),
                             return_object_denominator=len(returns), return_status_counts=dict(Counter(r["status"] for r in returns)),
                             common_return_followups=result["common_return_cohort_n"],
                             common_return_censors=result["arm_summaries"][arm]["comparable_return_censor_count"],
                             qualified_kept_laps=result["arm_summaries"][arm]["matched_kept_lap_ms"],
                             matched_window_metrics={k: v for k, v in result["arm_summaries"][arm].items() if k.startswith("matched_window_")},
                             release_changed_decisions=sum(bool(d["controller"].get("steering_release_changed")) for e, _ in rows for d in e["decision_trace"]),
                             generation_suppressed_decisions=sum(bool(d["controller"].get("steering_generation_ambiguous_motion_ignored")) for e, _ in rows for d in e["decision_trace"]))
    failures = []
    for cell in pairs:
        if cell["completion"] != "lost":
            continue
        track, seed = cell["track_id"], cell["seed"]
        be, _ = episodes[(track, seed, analysis.ARMS[0])]
        ce, raw = episodes[(track, seed, analysis.ARMS[1])]
        trace = ce["decision_trace"]
        release = [d for d in trace if d["controller"].get("steering_release_changed")]
        generation = [d for d in trace if d["controller"].get("steering_generation_ambiguous_motion_ignored")]
        first_diff = next((i for i, (b, c) in enumerate(zip(be["decision_trace"], trace)) if any(abs(b[k] - c[k]) > 1e-6 for k in ("steer", "gas", "brake"))), None)
        states = [ce["initial_state"], *raw]
        peak = max(states, key=lambda s: abs(s["lateral"]))
        events = []
        for index, obj in enumerate(ce["catalog"]["obstacles"]):
            touched = [s for s in states if obj["id"] in s["contacts"] or s["clearance"][index] <= 0]
            if touched:
                t = touched[0]["t"]
                previous = [d for d in release if d["controller"]["evaluation_only"]["pre"]["t"] <= t]
                events.append(dict(obstacle_id=obj["id"], first_hit_t=t, contact_listener=obj["id"] in touched[0]["contacts"],
                                   lateral=touched[0]["lateral"], heading_error=touched[0]["heading_error"],
                                   last_prior_release=decision_view(previous[-1]) if previous else None))
        indices = {0, len(trace) - 1}
        if first_diff is not None:
            indices.update(range(max(0, first_diff - 3), min(len(trace), first_diff + 7)))
        for d in generation:
            indices.update(range(max(0, d["step"] - 3), min(len(trace), d["step"] + 3)))
        failures.append(dict(track_id=track, seed=seed, geometry_equal=cell["geometry_equal"], initial_equal=cell["initial_equal"], first10_parity=cell["first10_parity"],
                             baseline=cell["baseline"], candidate=cell["candidate"], candidate_retire_reason=ce["retire_reason"],
                             baseline_retire_reason=be["retire_reason"], candidate_release_changes=len(release), candidate_generation_suppressions=len(generation),
                             first_action_divergence_step=None if first_diff is None else first_diff + 1,
                             divergence_baseline=None if first_diff is None else decision_view(be["decision_trace"][first_diff]),
                             divergence_candidate=None if first_diff is None else decision_view(trace[first_diff]),
                             first_release=None if not release else decision_view(release[0]), last_release=None if not release else decision_view(release[-1]),
                             peak_raw_state=peak, obstacle_hit_events=events,
                             sampled_candidate_decisions=[decision_view(trace[i]) for i in sorted(indices)],
                             raw_file=ce["raw_trace_file"], raw_sha256=ce["raw_trace_sha256"],
                             causal_limit="chronological associations only; post-divergence B/C observations and candidate stabilized shadow are not counterfactual actions"))
    per_geometry = []
    for seed in sorted({c["seed"] for c in pairs}):
        group = [c for c in pairs if c["seed"] == seed]
        per_geometry.append(dict(seed=seed, matched_cells=len(group), completion=dict(Counter(c["completion"] for c in group)),
                                 arms={arm: dict(completed=sum(episodes[(c["track_id"], seed, arm)][0]["completed"] for c in group),
                                                  damage_total=math.fsum(episodes[(c["track_id"], seed, arm)][0]["damage"] for c in group),
                                                  collision_positive_decisions=sum(episodes[(c["track_id"], seed, arm)][0]["collisions"] for c in group),
                                                  max_lateral_m=max(abs(s["lateral"]) for c in group for e, raw in [episodes[(c["track_id"], seed, arm)]] for s in [e["initial_state"], *raw])) for arm in analysis.ARMS},
                                 relative_metrics=next(r["relative_metrics"] for r in result["per_geometry_results"] if r["seed"] == seed)))
    return dict(scope="completed fresh-to-candidate TRAIN matched prefix; original planned matrix incomplete",
                matched_pairs=len(pairs), matched_geometry_seeds=len(per_geometry), planned_pairs=len(analysis.CELLS), planned_geometry_seeds=len(analysis.SEEDS),
                completed_episodes=result["episodes_available"], planned_episodes=analysis.SLOTS,
                excluded_unmatched_episodes=result["episodes_available"] - len(pairs) * 2,
                paired_arm_summaries=summaries, paired_completion={k: result["completion"][k] for k in ("kept", "lost", "gained", "neither")},
                baseline_eligible_windows=result["baseline_eligible_windows"], lost_baseline_windows=result["candidate_lost_baseline_eligible_windows"],
                conditional_cell_weighted_relative_metrics=result["cell_weighted_relative_metrics"], conditional_equal_geometry_relative_metrics=result["equal_geometry_relative_metrics"],
                conditional_mechanism_limit=f"{result['candidate_lost_baseline_eligible_windows']} missing baseline windows and incomplete roads prevent population reduction/causal mediation claims; lost finishes and new hits cannot be offset by surviving-window or kept-lap gains",
                per_geometry_results=per_geometry, new_clean_obstacle_hits=result["new_clean_obstacle_hits"], lost_finish_failure_diagnostics=failures,
                irreversible_predeclared_failure=bool(failures or result["new_clean_obstacle_hits"]),
                decision="NOT_GENERALIZED_ADOPTION_CANDIDATE" if failures or result["new_clean_obstacle_hits"] else "INCOMPLETE_NO_GENERALIZATION_VERDICT",
                mathematical_limit="Appending unexecuted cells cannot remove an already observed matched lost finish or new clean-obstacle hit. This is rejection evidence, not completion of the planned24-road study.",
                environment_resets=0, model_updates=0, protected_observations=0, official_action=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if analysis.sha256(args.result) != args.result_sha256:
        raise ValueError("guarded primary result changed")
    result = analysis.read_json(args.result)
    protocol, _ = operator.validate_frozen(result["protocol_sha256"])
    if result["episode_report_sha256"] != analysis.sha256(args.run / "episode-report.json") or result["model_hashes"] != protocol["model_hashes"]:
        raise ValueError("primary evidence/model binding differs")
    report = analysis.read_json(args.run / "episode-report.json")
    episodes = {}
    for row in report["rows"]:
        if row["status"] != "completed":
            continue
        e = analysis.read_json(analysis.prior.physical.verify_file(args.run, row["file"], row["sha256"]))
        raw, _ = analysis.read_jsonl(analysis.prior.physical.verify_file(args.run, e["raw_trace_file"], e["raw_trace_sha256"]))
        k = row["track_id"], row["seed"], row["mode"]
        analysis.prior.validate_episode(e, raw, k, protocol)
        if k in episodes:
            raise ValueError("duplicate primary episode")
        episodes[k] = e, raw
    output = analyze(result, episodes)
    output.update(primary_result_sha256=args.result_sha256, protocol_sha256=result["protocol_sha256"], episode_report_sha256=result["episode_report_sha256"],
                  model_hashes=result["model_hashes"], diagnostic_source_sha256=analysis.sha256(__file__))
    analysis.prior.finite_tree(output)
    operator.save(args.output, output)
    print(json.dumps({k: output[k] for k in ("matched_pairs", "matched_geometry_seeds", "paired_completion", "paired_arm_summaries", "decision")}, indent=2))


if __name__ == "__main__":
    main()
