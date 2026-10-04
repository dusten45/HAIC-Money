"""New-cohort gates use synthetic traces only; zero real resets."""

from copy import deepcopy
import hashlib
import json

import pytest

from scripts import analyze_koi_steering_generalization as analyzer
from tests.test_analyze_koi_steering_release_ab import fixture


def full_pairs(exclusions=7, finishes=72):
    slots, episodes = [], {}
    for index, (track, seed) in enumerate(analyzer.CELLS):
        for arm in analyzer.ARMS:
            episode, raw = fixture(arm, track, seed)
            report = analyzer.analyze_episode(episode, raw)
            episode["catalog"]["track"] = [[seed, 0, 0, 0]]
            episode["geometry_sha256"] = hashlib.sha256(json.dumps(episode["catalog"], sort_keys=True).encode()).hexdigest()
            episode["catalog"]["obstacles"] = [dict(episode["catalog"]["obstacles"][0], id=i) for i in range(6)]
            report["obstacles"] = [deepcopy(report["obstacles"][0]) for _ in range(6)]
            for oid, obj in enumerate(report["obstacles"]):
                obj["obstacle"]["id"] = oid
                if index * 6 + oid < exclusions:
                    obj["fixed_window"]["status"] = "initially_beyond_entry"
                if arm == analyzer.ARMS[1]:
                    obj["fixed_window"]["max_abs_lateral_m"] *= .9
                    obj["fixed_window"]["path_length_m"] *= .99
                    obj["fixed_window"]["steering"]["integral_abs_s"] *= .99
                    obj["fixed_window"]["avoidance"]["duration_s"] *= .9
                    obj["fixed_window"]["avoidance"]["integral_abs_s"] *= .9
            if index >= finishes:
                report.update(completed=False, lapTimeMs=None)
                episode.update(completed=False, lapTimeMs=None)
            if arm == analyzer.ARMS[1]:
                report["steering_timeline"][10]["qualified_release"] = True
            episode.update(raw_trace_file=f"{track}-{seed}-{arm}.raw.jsonl", raw_trace_sha256="synthetic")
            slots.append(dict(track_id=track, seed=seed, mode=arm, status="completed"))
            episodes[(track, seed, arm)] = episode, report
    return slots, episodes


@pytest.mark.parametrize("exclusions,finishes", [(0, 72), (7, 69), (431, 1), (432, 0)])
def test_denominators_derive_from_new_baseline_not_old119_or21(exclusions, finishes):
    slots, episodes = full_pairs(exclusions, finishes)
    result = analyzer.summarize(slots, episodes)
    assert result["expected_slots"] == 144
    assert result["baseline_eligible_windows"] == 432 - exclusions
    assert result["completion"]["kept"] == finishes
    assert result["gates"]["all_kept_mean_lap_nonincrease"] == (finishes > 0)
    assert result["gates"]["all_baseline_windows_retained"] == (exclusions < 432)
    assert result["arm_summaries"][analyzer.ARMS[0]]["object_denominator"] == 432
    assert len(result["per_geometry_results"]) == 24
    assert len(result["per_track_results"]) == 3
    if exclusions <= 7:
        assert result["gate_passed"]
    else:
        assert not result["gate_passed"]


@pytest.mark.parametrize("defect", ["lost_finish", "damage", "collision", "new_hit", "lost_window", "lost_return", "lap", "partial", "censor", "duplicate_road", "source_error", "zero_integral"])
def test_generalization_fails_closed_without_candidate_change(defect):
    slots, episodes = full_pairs()
    key = (2, analyzer.SEEDS[3], analyzer.ARMS[1])
    episode, report = episodes[key]
    obj = report["obstacles"][0]
    errors = []
    if defect == "lost_finish":
        report["completed"] = False
        episode["completed"] = False
    elif defect == "damage":
        report["damage"] = .2
    elif defect == "collision":
        report["collisions"] = 1
    elif defect == "new_hit":
        obj["whole_episode_safety"]["hit"] = True
        obj["physical"]["status"] = "not_reached"
    elif defect == "lost_window":
        obj["fixed_window"]["status"] = "nonlocal_reverse_or_seam"
    elif defect == "lost_return":
        obj["prospective_return"]["observed_after_clear_s"] = None
    elif defect == "lap":
        report["lapTimeMs"] += 20
    elif defect == "partial":
        del episodes[key]
        next(s for s in slots if (s["track_id"], s["seed"], s["mode"]) == key)["status"] = "unrun"
    elif defect == "censor":
        report["planned_censor"] = True
    elif defect == "duplicate_road":
        for (track, seed, arm), (e, _) in episodes.items():
            if seed == analyzer.SEEDS[3]:
                e["catalog"]["track"] = [[analyzer.SEEDS[2], 0, 0, 0]]
    elif defect == "source_error":
        errors.append("source_tamper")
    else:
        for (_, _, arm), (_, r) in episodes.items():
            if arm == analyzer.ARMS[0]:
                for o in r["obstacles"]:
                    o["fixed_window"]["avoidance"].update(integral_abs_s=0, duration_s=0)
    result = analyzer.summarize(slots, episodes, errors)
    assert not result["gate_passed"]
    assert result["decision"] == "NOT_PROMOTED_FROZEN_V2"
    if defect == "new_hit":
        assert result["new_clean_obstacle_hits"]
        assert result["failure_traces"][0]["new_hit_objects"] == [0]


def test_empty_summary_is_not_promotion():
    result = analyzer.summarize([], {})
    assert not any(result["gates"].values())


def test_two_surviving_geometries_cannot_claim24_geometry_generalization():
    slots, episodes = full_pairs()
    for (_, seed, _), (_, report) in episodes.items():
        if seed not in analyzer.SEEDS[:2]:
            for obj in report["obstacles"]:
                obj["fixed_window"]["status"] = "initially_beyond_entry"
    result = analyzer.summarize(slots, episodes)
    assert len(result["eligible_geometry_seeds"]) == 2
    assert result["gates"]["joint_mechanism_on_half_eligible_geometries"]
    assert not result["gates"]["sufficient_eligible_geometry_coverage"]
    assert not result["gate_passed"]


def test_spec_and_cells_exclude_old_and_protected_seeds():
    assert len(set(analyzer.SEEDS)) == 24
    assert not set(analyzer.SEEDS) & {38300, 38301, 38302, 38303, 50300, 50301, 50302, 50303, 49300, 49301, 49302, 49303, 51300, 51301, 51302, 51303}
    assert analyzer.ANALYSIS_SPEC["frozen_policy"].startswith("v2 and baseline remain byte-identical")


@pytest.mark.parametrize("defect", [None, "missing", "action", "components", "state", "order"])
def test_streamed_decisions_bind_applied_actions_and_components(defect):
    episode, _ = fixture(seed=analyzer.SEEDS[0])
    rows = []
    for d in episode["decision_trace"]:
        c = d["controller"]
        c.update(baseline_pre_act_state={"steps": d["step"] - 1}, baseline_post_act_state={"steps": d["step"]})
        rows.extend([dict(status="act_intent", baseline_pre_act_state=c["baseline_pre_act_state"]),
                     dict(status="act_returned", baseline_pre_act_state=c["baseline_pre_act_state"], baseline_post_act_state=c["baseline_post_act_state"], action=[d[k] for k in ("steer", "gas", "brake")], steering_terms=deepcopy(c["steering_terms"]))])
    if defect == "missing":
        rows.pop()
    elif defect == "action":
        rows[1]["action"][0] += .01
    elif defect == "components":
        rows[1]["steering_terms"]["avoidance_component"] += .01
    elif defect == "state":
        rows[1]["baseline_post_act_state"] = {"steps": -1}
    elif defect == "order":
        rows[0], rows[1] = rows[1], rows[0]
    if defect:
        with pytest.raises(ValueError, match="streamed"):
            analyzer.validate_streamed_decisions(episode, rows)
    else:
        analyzer.validate_streamed_decisions(episode, rows)
