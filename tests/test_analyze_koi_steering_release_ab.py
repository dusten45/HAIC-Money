"""Synthetic passive receipts only: no Agent or environment construction/reset."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import analyze_koi_steering_release_ab as analyzer
from scripts import evaluate_koi_steering_release_ab as operator
from tests.test_analyze_koi_adaptive_ab import fixture as old_fixture


SCHEMA = "koi-steering-terms-v1"


def fixture(arm=analyzer.ARMS[0], track=1, seed=38300, until=80):
    episode, raw = old_fixture(arm=arm, track=track, seed=seed, until=until)
    for state in [episode["initial_state"], *raw]:
        state["lateral_separation"] = [1.]
    for decision in episode["decision_trace"]:
        c = decision["controller"]
        for endpoint in ("pre", "post"):
            c["evaluation_only"][endpoint]["lateral_separation"] = [1.]
        c.update(steering_terms=dict(schema=SCHEMA, avoidance_component=.2, reconstruction_valid=True, residual_valid=True,
                                    raw_terms=dict(near_urgency=.2), stages=[dict(before=.1, after=.1, delta=0.)],
                                    contributions=dict(road=-.1, avoidance=.2), residual=0., final_sum_residual=0.,
                                    reconstructed_steer=.1, actual_final_steer=.1),
                 baseline_steer=.1, projected_obstacle_x=50., steering_release_changed=False)
        if arm == analyzer.ARMS[1] and decision["step"] == 11:
            c.update(steering_release_changed=True, steering_release_reason="synthetic_release")
            decision["steer"] = .09
            c["steering_terms"].update(contributions=dict(road=-.11, avoidance=.2), reconstructed_steer=.09, actual_final_steer=.09)
        c["baseline_steering_terms"] = deepcopy(c["steering_terms"])
        c["baseline_steering_terms"].update(contributions=dict(road=-.1, avoidance=.2), reconstructed_steer=.1, actual_final_steer=.1)
    episode["steps"] = len(episode["decision_trace"])
    return episode, raw


def test_fixed_window_component_integrals_and_lateral_time_mean():
    episode, raw = fixture()
    result = analyzer.analyze_episode(episode, raw)
    window = result["obstacles"][0]["fixed_window"]
    assert window["status"] == "complete"
    assert window["path_length_m"] == 50
    assert window["mean_abs_lateral_m"] == pytest.approx(3)
    assert window["avoidance"] == pytest.approx(dict(status="complete", duration_s=1., integral_abs_s=.2, covered_s=1.))
    assert result["obstacles"][0]["centerline_return"]["status"] == "censored_window_exit"
    assert result["obstacles"][0]["prospective_return"]["status"] == "censored_episode_end"


@pytest.mark.parametrize("defect", ["missing", "unidentified", "residual", "ambiguous", "gap", "infinity"])
def test_component_coverage_never_fills_unknown_with_zero(defect):
    episode, _ = fixture()
    decision = episode["decision_trace"][5]
    if defect == "missing":
        del decision["controller"]["steering_terms"]
    elif defect in ("unidentified", "residual"):
        decision["controller"]["steering_terms"]["reconstruction_valid" if defect == "unidentified" else "residual_valid"] = False
    elif defect == "ambiguous":
        decision["controller"]["steering_terms"]["unidentifiable"] = ["expired impact clipping ambiguity"]
    elif defect == "gap":
        decision["controller"]["evaluation_only"]["pre"]["t"] += .001
    else:
        decision["controller"]["steering_terms"]["avoidance_component"] = float("inf")
    window = analyzer.component_window(episode["decision_trace"], 1.32, 2.32)
    assert window["status"] != "complete"
    assert "duration_s" not in window


def test_prospective_late_return_separate_from_prior25m_censor():
    episode, raw = fixture(until=180)
    for state in raw:
        if state["station"] >= 70:
            state["lateral"] = .1
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert report["centerline_return"]["status"] == "censored_window_exit"
    assert report["prospective_return"]["status"] == "returned"
    assert report["prospective_return"]["return_time_s"] == pytest.approx(.52)
    assert report["prospective_return"]["confirmed_t"] == pytest.approx(2.66)


@pytest.mark.parametrize("next_station,status", [(90, "censored_next_obstacle_entry"), (240, "censored_fixed_time")])
def test_followup_stops_at_earliest_next_entry_or2s(next_station, status):
    episode, raw = fixture(until=180)
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    obstacles = [report["obstacle"], dict(report["obstacle"], id=1, station=next_station)]
    result = analyzer.prospective_return([episode["initial_state"], *raw], report["obstacle"], report["physical"], obstacles, episode)
    assert result["status"] == status
    assert result["observed_after_clear_s"] == pytest.approx(.42 if next_station == 90 else 2.)
    assert result["return_time_s"] is None
    assert result["restriction_is_censored_lower_bound"]


def test_common_followup_can_remove_late_return_without_returned_only_bias():
    episode, raw = fixture(until=180)
    for state in raw:
        if state["station"] >= 70:
            state["lateral"] = .1
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    result = analyzer.prospective_return(report["_states"], report["obstacle"], report["physical"], episode["catalog"]["obstacles"], episode, horizon=.5)
    assert result["status"] == "censored_common_followup"
    assert result["confirmation_horizon_s"] == pytest.approx(.5)
    assert result["confirmation_lower_bound_s"] == pytest.approx(.5)
    assert result["return_lower_bound_s"] == pytest.approx(.26)
    assert result["restricted_response_s"] == pytest.approx(.26)
    assert result["pending_centerline_run"] is None  # Onset .52 has not been observed.


@pytest.mark.parametrize("onset_station,horizon,onset_delay", [(64, .5, .4), (49, .2, .1), (44, .1, 0.), (44, 0., 0.)])
def test_censored_onset_bound_accounts_for_future_confirmation(onset_station, horizon, onset_delay):
    episode, raw = fixture(until=180)
    for state in raw:
        if state["station"] >= onset_station:
            state["lateral"] = .1
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    full = report["prospective_return"]
    short = analyzer.prospective_return(report["_states"], report["obstacle"], report["physical"], episode["catalog"]["obstacles"], episode, horizon=horizon)
    assert report["physical"]["pass_t"] == pytest.approx(1.9)
    assert full["status"] == "returned"
    assert full["return_time_s"] == pytest.approx(onset_delay)
    assert full["confirmation_delay_s"] == pytest.approx(onset_delay + .24)
    assert full["pending_centerline_run"] is None
    assert short["status"] == "censored_common_followup"
    assert short["return_time_s"] is None and short["confirmation_delay_s"] is None
    assert short["confirmation_horizon_s"] == pytest.approx(horizon)
    assert short["observed_after_clear_s"] == pytest.approx(horizon)
    assert short["return_lower_bound_s"] == pytest.approx(max(0., horizon - .24))
    assert short["return_lower_bound_s"] <= full["return_time_s"] + 1e-9
    assert short["restricted_response_s"] == short["return_lower_bound_s"]
    assert short["restricted_response_kind"] == "censored_onset_lower_bound"
    pending = short["pending_centerline_run"]
    assert pending["start_t"] == pytest.approx(1.9 + onset_delay)
    assert pending["onset_delay_s"] == pytest.approx(onset_delay)
    assert pending["last_observed_t"] <= short["cutoff_t"] + 1e-9
    assert pending["observed_length_s"] == pytest.approx(horizon - onset_delay)
    assert pending["observed_length_s"] < pending["required_length_s"] == .24
    assert pending["confirmation_fully_observed"] is False
    assert "confirmed_t" not in short


def test_confirmation_at_cutoff_retains_onset_delay_and_no_pending_run():
    episode, raw = fixture(until=180)
    for state in raw:
        if state["station"] >= 64:
            state["lateral"] = .1
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    result = analyzer.prospective_return(report["_states"], report["obstacle"], report["physical"], episode["catalog"]["obstacles"], episode, horizon=.64)
    assert result["status"] == "returned"
    assert result["onset_t"] == pytest.approx(2.3)
    assert result["confirmed_t"] == pytest.approx(2.54)
    assert result["return_time_s"] == pytest.approx(.4)
    assert result["confirmation_delay_s"] == pytest.approx(.64)
    assert result["restricted_response_s"] == pytest.approx(.4)
    assert result["restricted_response_kind"] == "observed_onset_delay"
    assert result["return_lower_bound_s"] is None and result["confirmation_lower_bound_s"] is None
    assert result["pending_centerline_run"] is None


def test_censored_pending_run_uses_only_raw_samples_before_cutoff():
    episode, raw = fixture(until=180)
    for state in raw:
        if state["station"] >= 64:
            state["lateral"] = .1
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    args = (report["obstacle"], report["physical"], episode["catalog"]["obstacles"], episode)
    before = analyzer.prospective_return(report["_states"], *args, horizon=.51)
    assert before["pending_centerline_run"]["last_observed_t"] == pytest.approx(2.4)
    assert before["pending_centerline_run"]["observed_length_s"] == pytest.approx(.1)
    assert before["confirmation_horizon_s"] == pytest.approx(.51)
    changed_future = deepcopy(report["_states"])
    for state in changed_future:
        if state["t"] > before["cutoff_t"]:
            state.update(lateral=100., road_index=1000)
    after = analyzer.prospective_return(changed_future, *args, horizon=.51)
    assert after == before


def test_pending_run_that_ended_before_censor_is_not_carried_as_current():
    episode, raw = fixture(until=180)
    for state in raw:
        if 64 <= state["station"] <= 66:
            state["lateral"] = .1
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    result = analyzer.prospective_return(report["_states"], report["obstacle"], report["physical"], episode["catalog"]["obstacles"], episode, horizon=.5)
    assert result["status"] == "censored_common_followup"
    assert result["pending_centerline_run"] is None
    assert result["return_time_s"] is None


@pytest.mark.parametrize("defect", ["unpassed", "discontinuous", "hold11samples", "dnf"])
def test_followup_invalid_unpassed_hold_and_episode_censor_explicit(defect):
    episode, raw = fixture(until=40 if defect == "unpassed" else 50 if defect == "dnf" else 180)
    if defect == "discontinuous":
        raw[50]["road_index"] += 100
    elif defect == "hold11samples":
        for state in raw:
            state["lateral"] = .5 if 70 <= state["station"] <= 80 else 3
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]["prospective_return"]
    assert report["status"] == {"unpassed": "unpassed", "discontinuous": "invalid_followup_continuity",
                                "hold11samples": "censored_fixed_time", "dnf": "censored_episode_end"}[defect]
    assert report["return_time_s"] is None
    if defect == "dnf":
        assert report["dnf"] and report["observed_after_clear_s"] == pytest.approx(.12)


def test_excess_requires_stable_object_full_lateral_separation_and_finite_projection():
    episode, _ = fixture()
    for d in episode["decision_trace"]:
        pre = d["controller"]["evaluation_only"]["pre"]
        x, y = analyzer.physical.project_center(episode["catalog"]["obstacles"][0], pre)
        d["controller"]["near_object"] = [y, x, 42]
    rows = analyzer.chronological_diagnostics(episode)
    assert not rows[0]["excess_observed"]
    assert rows[1]["excess_observed"]
    episode["decision_trace"][2]["controller"]["near_object"] = None
    episode["decision_trace"][4]["controller"]["projected_obstacle_x"] = None
    episode["decision_trace"][5]["controller"]["projected_obstacle_x"] = float("inf")
    episode["decision_trace"][6]["controller"]["projected_obstacle_x"] = 47.99
    episode["decision_trace"][7]["controller"]["evaluation_only"]["pre"]["lateral_separation"] = [0]
    rows = analyzer.chronological_diagnostics(episode)
    assert not any(r["excess_observed"] for r in rows[2:8])
    assert rows[1]["event_id"] != rows[3]["event_id"]
    assert rows[3]["reacquisition_or_object_switch"]


def test_same_object_off_epoch_retains_unavailable_projection_without_relabeling_clear():
    episode, _ = fixture()
    for d in episode["decision_trace"]:
        pre = d["controller"]["evaluation_only"]["pre"]
        x, y = analyzer.physical.project_center(episode["catalog"]["obstacles"][0], pre)
        d["controller"]["near_object"] = [y, x, 42]
    episode["decision_trace"][0]["controller"]["projected_obstacle_x"] = 42
    episode["decision_trace"][2]["controller"]["projected_obstacle_x"] = None
    episode["decision_trace"][3]["controller"]["near_object"] = None
    episode["decision_trace"][4]["controller"]["projected_obstacle_x"] = None
    rows = analyzer.chronological_diagnostics(episode)
    assert rows[1]["observed_active_to_off_epoch"]
    assert rows[2]["continued_avoidance_after_measured_off_epoch"]
    assert rows[2]["projection_unavailable_after_same_object_off_epoch"]
    assert not rows[2]["valid_projection_off"] and not rows[2]["excess_observed"]
    assert not rows[4]["continued_avoidance_after_measured_off_epoch"]


def test_lateral_approach_overlap_and_rear_timing_distinct_left_censors():
    episode, raw = fixture()
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    timing = report["lateral_separation_timing"]
    assert timing["approach_separation_status"] == "left_censored_already_separated"
    assert timing["overlap_separation_status"] == "left_censored_already_separated"
    assert timing["approach_first_separated_t"] < timing["overlap_first_separated_t"] < timing["rear_clear_first_t"]
    assert timing["rear_clear_first_t"] == pytest.approx(report["physical"]["pass_t"])


def full_pairs():
    slots, episodes = [], {}
    exclusions = 25
    for index, (track, seed) in enumerate(analyzer.CELLS):
        for arm in analyzer.ARMS:
            episode, raw = fixture(arm, track, seed)
            report = analyzer.analyze_episode(episode, raw)
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
            if index < 3:
                report.update(completed=False, lapTimeMs=None)
                episode.update(completed=False, lapTimeMs=None)
            # This is fabricated qualified observer evidence for gate tests only.
            if arm == analyzer.ARMS[1]:
                report["steering_timeline"][10]["qualified_release"] = True
            slots.append(dict(track_id=track, seed=seed, mode=arm, status="completed"))
            episodes[(track, seed, arm)] = episode, report
    return slots, episodes


def test_prospective5percent_gate_all119_144_21_denominators():
    slots, episodes = full_pairs()
    result = analyzer.summarize(slots, episodes)
    assert result["gate_passed"]
    assert result["baseline_eligible_windows"] == 119
    assert result["completion"]["kept"] == 21
    assert result["cell_weighted_relative_metrics"]["max_lateral"]["mean"] == pytest.approx(-.1)
    assert len(result["qualifying_geometry_seeds"]["max_lateral"]) >= 2
    assert result["common_return_cohort_n"] == 144
    assert all(r["prospective_return_denominator"] == 144 and r["return_time_summary"] is None for r in result["arm_summaries"].values())
    assert not result["exploratory_scope"]


@pytest.mark.parametrize("case", ["worsened_censor", "both_pending", "short_followup"])
def test_corrected_onset_bounds_cannot_forge_comparable_censor_improvement(case):
    slots, episodes = full_pairs()
    for arm, onset_station in zip(analyzer.ARMS, (49 if case == "worsened_censor" else 64,
                                                 49 if case == "short_followup" else 59 if case == "both_pending" else 64)):
        episode, report = episodes[(2, 38301, arm)]
        obj = report["obstacles"][0]
        states = deepcopy(obj["_states"])
        for state in states:
            if state["station"] >= onset_station:
                state["lateral"] = .1
        if arm == analyzer.ARMS[1]:
            cutoff = 2.1 if case == "short_followup" else 2.4
            states = [state for state in states if state["t"] <= cutoff + 1e-9]
        obj["_states"] = states
        obj["prospective_return"] = analyzer.prospective_return(states, obj["obstacle"], obj["physical"], episode["catalog"]["obstacles"], episode)
    result = analyzer.summarize(slots, episodes)
    row = next(r for r in result["common_return_cohort"] if (r["track_id"], r["seed"], r["obstacle_id"]) == (2, 38301, 0))
    base, candidate = row["common"]["baseline"], row["common"]["candidate"]
    summaries = result["arm_summaries"]
    assert "Kaplan-Meier" in summaries[analyzer.ARMS[1]]["restricted_note"]
    assert summaries[analyzer.ARMS[1]]["return_time_summary"] is None
    if case == "worsened_censor":
        assert row["comparable"] and row["common"]["horizon_s"] == pytest.approx(.5)
        assert base["status"] == "returned" and candidate["status"].startswith("censored_")
        assert summaries[analyzer.ARMS[0]]["comparable_return_censor_count"] == 143
        assert summaries[analyzer.ARMS[1]]["comparable_return_censor_count"] == 144
        assert not result["gates"]["comparable_return_censor_nonincrease"] and not result["gate_passed"]
    elif case == "both_pending":
        assert row["comparable"] and base["status"].startswith("censored_") and candidate["status"].startswith("censored_")
        assert base["return_lower_bound_s"] == pytest.approx(.26)
        assert candidate["return_lower_bound_s"] == pytest.approx(.26)
        assert result["gates"]["comparable_return_censor_nonincrease"]
        assert summaries[analyzer.ARMS[0]]["comparable_return_censor_count"] == summaries[analyzer.ARMS[1]]["comparable_return_censor_count"] == 144
        assert summaries[analyzer.ARMS[0]]["common_restricted_response"] == summaries[analyzer.ARMS[1]]["common_restricted_response"]
    else:
        assert not row["comparable"] and row["common"]["horizon_s"] == pytest.approx(.2)
        assert candidate["return_lower_bound_s"] == 0 and candidate["return_time_s"] is None
        assert result["common_return_cohort_n"] == 143 and result["baseline_return_comparable_n"] == 144
        assert not result["gates"]["comparable_return_followup_preserved"] and not result["gate_passed"]


@pytest.mark.parametrize("defect", ["lost", "damage", "collision", "new_hit", "lost_window", "lateral", "avoidance", "one_seed", "path", "steer", "lap", "censor", "prefix0", "prefix9", "coverage", "return", "missing", "source"])
def test_each_gate_fails_closed(defect):
    slots, episodes = full_pairs()
    episode, report = episodes[(2, 38301, analyzer.ARMS[1])]
    errors = []
    obj = report["obstacles"][0]
    if defect == "lost":
        report["completed"] = False
    elif defect == "damage":
        report["damage"] = .1
    elif defect == "collision":
        report["collisions"] = 1
    elif defect == "new_hit":
        obj["whole_episode_safety"]["hit"] = True
        obj["physical"]["status"] = "not_reached"
    elif defect == "lost_window":
        report["obstacles"][1]["fixed_window"]["status"] = "nonlocal_reverse_or_seam"
    elif defect in ("lateral", "avoidance", "one_seed"):
        for (_, seed, arm), (_, r) in episodes.items():
            if arm == analyzer.ARMS[1] and (defect != "one_seed" or seed != 38301):
                for o in r["obstacles"]:
                    if defect != "avoidance":
                        o["fixed_window"]["max_abs_lateral_m"] = 3
                    if defect != "lateral":
                        o["fixed_window"]["avoidance"].update(duration_s=1., integral_abs_s=.2)
    elif defect in ("path", "steer"):
        for (_, _, arm), (_, r) in episodes.items():
            if arm == analyzer.ARMS[1]:
                for o in r["obstacles"]:
                    if defect == "path":
                        o["fixed_window"]["path_length_m"] = 51
                    else:
                        o["fixed_window"]["steering"]["integral_abs_s"] = .11
    elif defect == "lap":
        report["lapTimeMs"] += 20
    elif defect == "censor":
        report["planned_censor"] = True
    elif defect.startswith("prefix"):
        for arm in analyzer.ARMS:
            e = episodes[(2, 38301, arm)][0]
            e["decision_trace"] = e["decision_trace"][:0 if defect == "prefix0" else 9]
    elif defect == "coverage":
        for (_, _, arm), (_, r) in episodes.items():
            if arm == analyzer.ARMS[1]:
                for row in r["steering_timeline"]:
                    row["qualified_release"] = False
    elif defect == "return":
        obj["prospective_return"].update(status="unpassed", observed_after_clear_s=None)
    elif defect == "missing":
        del episodes[(2, 38301, analyzer.ARMS[1])]
    else:
        errors = ["source mismatch"]
    result = analyzer.summarize(slots, episodes, errors)
    assert not result["gate_passed"]
    assert len(result["slots"]) == 48 and len(result["cells"]) == 24


def test_zero_avoidance_denominators_preserved_not_credited_as_reduction():
    slots, episodes = full_pairs()
    for (_, _, _), (_, r) in episodes.items():
        for o in r["obstacles"]:
            o["fixed_window"]["avoidance"].update(duration_s=0., integral_abs_s=0.)
    result = analyzer.summarize(slots, episodes)
    assert not result["gates"]["avoidance_reduction_5pct_two_geometries"]
    assert result["cell_weighted_relative_metrics"]["avoidance_integral"]["n"] == 0
    metric = next(w["metrics"]["avoidance_integral"] for c in result["cells"] for w in c["windows"] if w["eligible"])
    assert metric["fractional_change"] is None and metric["zero_preserved"]


def test_empty_not_valid_has_no_zero_means_and_all_false_gates():
    result = analyzer.summarize(operator.scheduled_slots(), {})
    assert not any(result["gates"].values())
    assert result["insufficient_intervention"]
    assert all(r["n"] == 0 and r["mean"] is None for r in result["cell_weighted_relative_metrics"].values())


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    prior = operator.ROOT / "runs/koi-minimum-clearance-ab-20260930-r3"
    baseline_path = prior / "crossing-projection-source-reconstruction.zip"
    (tmp_path / baseline_path.name).write_bytes(baseline_path.read_bytes())
    with zipfile.ZipFile(baseline_path) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    baseline = {n: hashlib.sha256(d).hexdigest() for n, d in files.items()}
    terms_path = operator.ROOT / "haic/algorithms/koi/steering_terms.py"
    files.update({"haic_agent/steering_terms.py": terms_path.read_bytes(), "haic_agent/steering_release_runtime.py": b"# never executed synthetic candidate\n"})
    candidate = tmp_path / "candidate.zip"
    with zipfile.ZipFile(candidate, "w") as archive:
        for n, d in files.items():
            archive.writestr(n, d)
    treatment = {n: hashlib.sha256(d).hexdigest() for n, d in files.items()}
    manifest = candidate.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(dict(candidate="koi-steering-release-synthetic", candidate_zip_sha256=operator.sha(candidate),
                                        baseline_source_sha256=baseline, files=[dict(path=n, sha256=h) for n, h in treatment.items()])))
    old_protocol = analyzer.read_json(prior / "protocol.json")
    sources = old_protocol["environment_source_sha256"]
    # Future wrapper/packager are not authored by this child; synthetic source closure
    # lives only in the test temp directory and is never imported or executed.
    future_wrapper, future_package = tmp_path / "future-wrapper.py", tmp_path / "future-package.py"
    future_wrapper.write_text("# synthetic source closure only\n")
    future_package.write_text("# synthetic source closure only\n")
    helpers = [Path(operator.__file__), Path(analyzer.__file__), terms_path, future_wrapper, future_package]
    monkeypatch.setattr(operator, "helper_paths", lambda: helpers)
    helper_hashes = {str(p): operator.sha(p) for p in helpers}
    copies = []
    for i, (path, expected) in enumerate([*sources.items(), *helper_hashes.items()]):
        copy = tmp_path / f"source-{i}.py"
        copy.write_bytes(Path(path).read_bytes())
        copies.append(dict(file=copy.name, source=path, sha256=expected))
    protocol = dict(analysis_spec=analyzer.ANALYSIS_SPEC, analyzer_sha256=operator.sha(analyzer.__file__),
                    operator_sha256=operator.sha(operator.__file__), cells=[dict(track_id=t, geometry_seed=s, partition="TRAIN", obstacles=True) for t, s in analyzer.CELLS],
                    episodes=48, frame_skip=4, warmup_ticks=50, raw_fps=50, max_decisions=1200, steering_terms_schema=SCHEMA,
                    model_source_sha256={analyzer.ARMS[0]: baseline, analyzer.ARMS[1]: treatment},
                    model_hashes={analyzer.ARMS[0]: operator.sha(tmp_path / baseline_path.name), analyzer.ARMS[1]: operator.sha(candidate)},
                    candidate_manifest_sha256=operator.sha(manifest), environment_source_sha256=sources, helper_source_sha256=helper_hashes,
                    source_copies=copies, consumed_reuse_evidence=dict(source_sha256={}), schedule=operator.scheduled_slots(), runtime_versions=dict(synthetic=True))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    episode, raw = fixture()
    objects = [dict(episode["catalog"]["obstacles"][0], id=i, x=i * 50.) for i in range(6)]
    episode["catalog"]["obstacles"] = objects
    episode["geometry_sha256"] = hashlib.sha256(json.dumps(episode["catalog"], sort_keys=True).encode()).hexdigest()
    for state in [episode["initial_state"], *raw, *[d["controller"]["evaluation_only"][endpoint] for d in episode["decision_trace"] for endpoint in ("pre", "post")]]:
        for key in ("front", "rear", "clearance", "lateral_separation"):
            state[key] = state[key][:1] * 6
    path = tmp_path / "1-38300-crossing_projection.json"
    raw_path = path.with_suffix(".raw.jsonl")
    raw_path.write_text("".join(json.dumps(r) + "\n" for r in raw))
    episode.update(raw_trace_file=raw_path.name, raw_trace_sha256=operator.sha(raw_path), versions=protocol["runtime_versions"], runtime_module_paths={"agent": "/tmp/synthetic/agent.py"})
    path.write_text(json.dumps(episode))
    identity = operator.slot_id(1, 38300, analyzer.ARMS[0])
    process = path.with_suffix(".process.json")
    process.write_text(json.dumps(dict(status="completed", error=None, wall_time_s=.1)))
    bound = path.with_suffix(".bound-process.json")
    bound.write_text(json.dumps(dict(status="completed", error=None, wall_time_s=.1, slot_id=identity, operator_sha256=protocol["operator_sha256"], original_process_file=process.name, original_process_sha256=operator.sha(process))))
    rows = operator.scheduled_slots()
    rows[0].update(file=path.name, status="completed", sha256=operator.sha(path), slot_id=identity, process_file=bound.name, process_sha256=operator.sha(bound))
    ledger = tmp_path / "reset-ledger.jsonl"
    ledger.write_text(json.dumps(dict(status="reset_intent", track=1, seed=38300, arm=analyzer.ARMS[0], slot_id=identity)) + "\n" + json.dumps(rows[0]) + "\n")
    (tmp_path / "episode-report.json").write_text(json.dumps(dict(protocol_sha256=operator.sha(tmp_path / "protocol.json"), rows=rows, operator_error=None, reset_ledger_sha256=operator.sha(ledger))))
    return candidate, manifest


def test_loader_full_closure_process_receipts_and_all_partial_slots(tmp_path, receipts):
    protocol, slots, episodes, errors = analyzer.load_run(tmp_path, *receipts)
    assert len(protocol["environment_source_sha256"]) == 142
    assert len(slots) == 48 and len(episodes) == 1
    assert errors == ["incomplete_reset_ledger"]
    assert not analyzer.summarize(slots, episodes, errors)["gate_passed"]


@pytest.mark.parametrize("artifact", ["protocol", "episode", "raw", "manifest", "ledger", "copy", "process", "bound"])
def test_loader_rejects_corrupted_primary_and_actual_process_hashes(tmp_path, receipts, artifact):
    paths = dict(protocol=tmp_path / "protocol.json", episode=tmp_path / "1-38300-crossing_projection.json",
                 raw=tmp_path / "1-38300-crossing_projection.raw.jsonl", manifest=receipts[1], ledger=tmp_path / "reset-ledger.jsonl",
                 copy=tmp_path / "source-0.py", process=tmp_path / "1-38300-crossing_projection.process.json",
                 bound=tmp_path / "1-38300-crossing_projection.bound-process.json")
    paths[artifact].write_text(paths[artifact].read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        analyzer.load_run(tmp_path, *receipts)


def rewrite_report_hash(tmp_path, key, path):
    report_path = tmp_path / "episode-report.json"
    report = analyzer.read_json(report_path)
    if key == "ledger":
        report["reset_ledger_sha256"] = operator.sha(path)
    else:
        report["rows"][0][key] = operator.sha(path)
    report_path.write_text(json.dumps(report))


@pytest.mark.parametrize("defect", ["duplicate", "eventID", "wrong_arm", "order", "completion_hash"])
def test_ledger_association_corruption_rejected_even_when_rehashed(tmp_path, receipts, defect):
    ledger = tmp_path / "reset-ledger.jsonl"
    events, _ = analyzer.read_jsonl(ledger)
    if defect == "duplicate":
        events.append(events[-1])
    elif defect == "eventID":
        events[0]["slot_id"] = "different"
    elif defect == "wrong_arm":
        events[0]["arm"] = analyzer.ARMS[1]
    elif defect == "order":
        events.reverse()
    else:
        events[1]["sha256"] = "wrong"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in events))
    rewrite_report_hash(tmp_path, "ledger", ledger)
    with pytest.raises(ValueError, match="ledger"):
        analyzer.load_run(tmp_path, *receipts)


@pytest.mark.parametrize("defect", ["pose", "raw_eventID", "collision", "objectID", "gap", "component_schema"])
def test_primary_world_event_association_fails_closed(tmp_path, receipts, defect):
    path = tmp_path / "1-38300-crossing_projection.json"
    episode = analyzer.read_json(path)
    raw_path = tmp_path / "1-38300-crossing_projection.raw.jsonl"
    raw, _ = analyzer.read_jsonl(raw_path)
    if defect == "pose":
        episode["decision_trace"][0]["controller"]["evaluation_only"]["post"]["x"] += .5
    elif defect == "raw_eventID":
        raw[0]["step"] = 2
    elif defect == "collision":
        episode["decision_trace"][0]["collision"] = True
    elif defect == "objectID":
        raw[0]["contacts"] = [99]
    elif defect == "gap":
        raw.pop(2)
        episode["raw_ticks"] -= 1
    else:
        episode["decision_trace"][0]["controller"]["steering_terms"]["schema"] = "bad"
    raw_path.write_text("".join(json.dumps(r) + "\n" for r in raw))
    episode["raw_trace_sha256"] = operator.sha(raw_path)
    path.write_text(json.dumps(episode))
    rewrite_report_hash(tmp_path, "sha256", path)
    with pytest.raises(ValueError):
        analyzer.load_run(tmp_path, *receipts)


def test_partial_raw_truncation_retained_but_never_passes(tmp_path, receipts):
    _, raw = fixture(until=4)
    for state in raw:
        for key in ("front", "rear", "clearance", "lateral_separation"):
            state[key] *= 6
    path = tmp_path / "1-38300-steering_release_v1.raw.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in raw) + '{"truncated":')
    _, slots, _, errors = analyzer.load_run(tmp_path, *receipts)
    partial = next(s["partial_raw"] for s in slots if "partial_raw" in s)
    assert partial["complete_ticks"] == 4 and partial["truncated_last_line"]
    assert errors
    path.write_text('{"bad":\n' + json.dumps(raw[0]) + "\n")
    with pytest.raises(json.JSONDecodeError):
        analyzer.load_run(tmp_path, *receipts)


def test_valid_json_partial_corrupt_world_row_is_not_trusted(tmp_path, receipts):
    path = tmp_path / "1-38300-steering_release_v1.raw.jsonl"
    path.write_text('{"t":1.02,"x":0}\n')
    with pytest.raises(ValueError, match="invalid raw"):
        analyzer.load_run(tmp_path, *receipts)


def test_prospective_next_entry_is_not_inherited_initial_seam_exclusion():
    episode, raw = fixture(until=180)
    episode["initial_state"]["station"] = 1141
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert report["fixed_window"]["status"] == "initially_beyond_entry"
    obstacles = [report["obstacle"], dict(report["obstacle"], id=1, station=90)]
    result = analyzer.prospective_return(report["_states"], report["obstacle"], report["physical"], obstacles, episode)
    assert result["status"] == "censored_next_obstacle_entry"
    assert result["observed_after_clear_s"] == pytest.approx(.42)


def test_next_obstacle_already_entered_zero_followup_is_explicit_censor():
    episode, raw = fixture()
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    obstacles = [report["obstacle"], dict(report["obstacle"], id=1, station=60)]
    result = analyzer.prospective_return(report["_states"], report["obstacle"], report["physical"], obstacles, episode)
    assert result["status"] == "censored_next_obstacle_already_entered"
    assert result["observed_after_clear_s"] == 0
    assert result["return_time_s"] is None


@pytest.mark.parametrize("text", ['{"value":NaN}', '{"value":[1e400]}', '{"nested":{"v":-1e400}}'])
def test_json_and_jsonl_nonfinite_overflow_rejected(tmp_path, text):
    path = tmp_path / "overflow.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="nonfinite"):
        analyzer.read_json(path)
    with pytest.raises(ValueError, match="nonfinite"):
        analyzer.read_jsonl(path, partial=True)


@pytest.mark.parametrize("defect", ["action", "residual", "contributions", "clip", "missing"])
def test_component_accounting_corruption_fails_even_with_claimed_valid_flags(defect):
    episode, _ = fixture()
    terms = episode["decision_trace"][0]["controller"]["steering_terms"]
    if defect == "action":
        terms["actual_final_steer"] = .2
    elif defect == "residual":
        terms["residual"] = .2
    elif defect == "contributions":
        terms["contributions"]["road"] = .2
    elif defect == "clip":
        terms["stages"][0]["delta"] = .2
    else:
        del terms["raw_terms"]
    with pytest.raises(ValueError, match="steering"):
        analyzer.validate_terms(terms, .1)


def test_component_trace_validator_on_all5703_recorded_baseline_actions():
    from haic.algorithms.koi.steering_terms import reconstruct_steering
    run = operator.ROOT / "runs/koi-minimum-clearance-ab-20260930-r3"
    count = 0
    for track, seed in analyzer.CELLS:
        episode = analyzer.read_json(run / f"{track}-{seed}-crossing_projection.json")
        previous_impact = 0
        for decision in episode["decision_trace"]:
            diagnostics = dict(decision["controller"], impact_left_before_act=previous_impact)
            terms = reconstruct_steering(diagnostics, decision["steer"], state=dict(steps=decision["step"]))
            analyzer.validate_terms(terms, decision["steer"])
            assert terms["reconstruction_valid"] and terms["residual_valid"]
            previous_impact = decision["controller"]["actual_impact_left"]
            count += 1
    assert count == 5703
