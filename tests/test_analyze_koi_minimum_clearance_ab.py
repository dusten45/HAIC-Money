"""Synthetic measurement/regression receipts only; no model or environment execution."""

from copy import deepcopy
import json
import math
import hashlib
from pathlib import Path
import zipfile

import pytest

from scripts import analyze_koi_minimum_clearance_ab as analyzer
from tests.test_analyze_koi_adaptive_ab import fixture as original_fixture


def fixture(arm="minimum_clearance_v1", seed=38300, track=1, until=80):
    episode, raw = original_fixture(arm=arm, seed=seed, track=track, until=until)
    for decision in episode["decision_trace"]:
        decision["controller"].update(clearance_projection_active=False, clearance_projection_available=True,
                                      projected_obstacle_x=50, clearance_avoidance_steering_active=True, minimum_clearance_changed=arm == analyzer.ARMS[1])
    return episode, raw


def test_fixed_window_full50m_not_detector_conditioned():
    episode, raw = fixture()
    for d in episode["decision_trace"]:
        d["controller"].update(near_object=None, far_objects=[])
    result = analyzer.analyze_episode(episode, raw)
    window = result["obstacles"][0]["fixed_window"]
    assert window["status"] == "complete"
    assert window["entry"]["station"] == 15
    assert window["exit"]["station"] == 65
    assert window["path_length_m"] == pytest.approx(50)
    assert window["duration_s"] == pytest.approx(1)
    assert window["max_abs_lateral_m"] == 3
    assert window["steering"] == pytest.approx(dict(status="complete", mean_abs=.1, max_abs=.1, integral_abs_s=.1))


def test_interpolated_endpoints_clip_world_path():
    episode, raw = fixture()
    obstacle = dict(episode["catalog"]["obstacles"][0], station=40.5)
    states = [episode["initial_state"], *raw]
    for state in states:
        state["x"] = .5 * state["station"]
    window = analyzer.fixed_window(states, obstacle)
    assert window["path_length_m"] == pytest.approx(math.hypot(25, 50))
    assert window["entry"]["t"] == pytest.approx(1.02 + 15.5 * .02)
    assert window["exit"]["t"] == pytest.approx(1.02 + 65.5 * .02)


@pytest.mark.parametrize("defect", ["seam", "jump", "reverse", "dnf"])
def test_invalid_or_incomplete_windows_retained(defect):
    episode, raw = fixture(until=60 if defect == "dnf" else 80)
    if defect == "seam":
        episode["initial_state"]["station"] = 1141
    elif defect == "jump":
        raw[49]["road_index"] += 100
    elif defect == "reverse":
        raw[49]["station"] = 48
    result = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert result["fixed_window"]["status"] != "complete"
    assert "path_length_m" not in result["fixed_window"]


def test_sustained_return_starts_only_after_physical_rear_clear():
    episode, raw = fixture()
    for r in raw:
        r["lateral"] = .5 if r["station"] < 43 or r["station"] >= 48 else 2
    result = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert result["physical"]["pass_t"] == pytest.approx(1.90)
    assert result["centerline_return"]["status"] == "returned"
    assert result["centerline_return"]["return_time_s"] == pytest.approx(.08)
    assert result["centerline_return"]["confirmed_t"] == pytest.approx(2.22)


@pytest.mark.parametrize("until,status", [(80, "censored_window_exit"), (50, "censored_episode_end"), (40, "unpassed")])
def test_unreturned_unpassed_and_dnf_status_never_imputed(until, status):
    episode, raw = fixture(until=until)
    result = analyzer.analyze_episode(episode, raw)["obstacles"][0]["centerline_return"]
    assert result["status"] == status
    assert result["return_time_s"] is None
    if until == 50:
        assert result["dnf"]
        assert result["observed_after_clear_s"] == pytest.approx(.12)


def test_late_return_after_window_exit_not_counted():
    episode, raw = fixture()
    for r in raw:
        if r["station"] >= 60:
            r["lateral"] = .1
    result = analyzer.analyze_episode(episode, raw)["obstacles"][0]["centerline_return"]
    assert result["status"] == "censored_window_exit"


def test_24seconds_hold_is_elapsed_time_not_sample_count():
    episode, raw = fixture()
    for r in raw:
        if 44 <= r["station"] <= 55:
            r["lateral"] = 0
    result = analyzer.analyze_episode(episode, raw)["obstacles"][0]["centerline_return"]
    assert result["status"] == "censored_window_exit"  #12 samples span0.22s


def test_steering_time_weighted_actual_commands_and_coverage():
    episode, raw = fixture()
    decisions = episode["decision_trace"]
    decisions[4]["steer"] = -.5
    window = analyzer.analyze_episode(episode, raw)["obstacles"][0]["fixed_window"]
    assert window["steering"]["max_abs"] == .5
    assert window["steering"]["integral_abs_s"] == pytest.approx(.1 + .4 * .08)
    decisions[4]["controller"]["evaluation_only"]["pre"]["t"] += .01
    assert analyzer.analyze_episode(episode, raw)["obstacles"][0]["fixed_window"]["steering"]["status"] != "complete"


def test_whole_episode_offroad_right_left_deviation_and_projection_diagnostic():
    episode, raw = fixture()
    raw[69].update(lateral=-9, wheel_road_contacts=[0, 1, 1, 1])
    raw[70].update(lateral=-8, wheel_road_contacts=[0, 1, 1, 1])
    raw[74].update(lateral=9, wheel_road_contacts=[0, 1, 1, 1])
    report = analyzer.analyze_episode(episode, raw)
    assert report["offroad_ticks"] == 3
    assert report["departure_events_by_side"] == {"left": 1, "right": 1}
    assert report["max_abs_centerline_lateral_m"] == 9
    assert len(report["projection_off_avoidance_steering"]) == len(episode["decision_trace"])
    assert report["obstacles"][0]["fixed_window"]["departures"]["offroad_ticks"] == 0


def full_pairs():
    slots, episodes = [], {}
    for track, seed in analyzer.CELLS:
        for arm in analyzer.ARMS:
            episode, raw = fixture(arm, seed, track)
            report = analyzer.analyze_episode(episode, raw)
            if arm == analyzer.ARMS[1]:
                report["obstacles"][0]["fixed_window"]["path_length_m"] *= .97
            episodes[(track, seed, arm)] = (episode, report)
            slots.append(dict(track_id=track, seed=seed, mode=arm, status="completed"))
    return slots, episodes


def test_predeclared_path_gate_and_censored_return_denominator():
    slots, episodes = full_pairs()
    result = analyzer.summarize(slots, episodes)
    assert result["gate_passed"]
    assert result["completion"]["kept"] == 24
    assert result["per_cell_mean_fractional_path_change"]["mean"] == pytest.approx(-.03)
    assert result["improving_geometry_seeds"] == list(analyzer.SEEDS)
    for report in result["arm_summaries"].values():
        assert report["return_status_counts"] == {"censored_window_exit": 24}
        assert report["return_denominator"] == 24
        assert report["return_time_summary"] is None
        assert report["no_return_after_pass"] == 24


@pytest.mark.parametrize("defect", ["lost", "damage", "collision", "new_hit", "path", "one_seed", "lap", "missing", "source", "prefix", "unassociated"])
def test_each_acceptance_gate_fails_closed(defect):
    slots, episodes = full_pairs()
    episode, report = episodes[(1, 38300, analyzer.ARMS[1])]
    errors = []
    if defect == "lost":
        report["completed"] = False
    elif defect == "damage":
        report["damage"] = .2
    elif defect == "collision":
        report["collisions"] = 1
    elif defect == "new_hit":
        report["obstacles"][0]["whole_episode_safety"]["hit"] = True
    elif defect in ("path", "one_seed"):
        for (_, seed, arm), (_, r) in episodes.items():
            if arm == analyzer.ARMS[1] and (defect == "path" or seed != 38300):
                r["obstacles"][0]["fixed_window"]["path_length_m"] = 50
    elif defect == "lap":
        report["lapTimeMs"] += 20
    elif defect == "missing":
        del episodes[(1, 38300, analyzer.ARMS[1])]
    elif defect == "source":
        errors = ["source_mismatch"]
    elif defect == "prefix":
        episode["decision_trace"][0]["gas"] = 1
    else:
        report["unassociated_collision_events"] = [{"source": "raw"}]
    result = analyzer.summarize(slots, episodes, errors)
    assert not result["gate_passed"]
    assert len(result["cells"]) == 24
    assert len(result["slots"]) == 48


def test_equal_cell_weight_not_pooled_object_count():
    slots, episodes = full_pairs()
    for arm in analyzer.ARMS:
        report = episodes[(1, 38300, arm)][1]
        original = report["obstacles"][0]
        report["obstacles"] *= 10
        if arm == analyzer.ARMS[1]:
            original["fixed_window"]["path_length_m"] = 55
    result = analyzer.summarize(slots, episodes)
    assert result["per_cell_mean_fractional_path_change"]["mean"] == pytest.approx((23 * -.03 + .1) / 24)


@pytest.mark.parametrize("decisions", [0, 9])
def test_parity_requires_ten_actual_decisions_not_equal_short_slices(decisions):
    slots, episodes = full_pairs()
    for arm in analyzer.ARMS:
        episode = episodes[(1, 38300, arm)][0]
        episode["decision_trace"] = episode["decision_trace"][:decisions]
    result = analyzer.summarize(slots, episodes)
    assert not result["cells"][0]["first10_parity"]
    assert not result["gates"]["model_only_pair_safety"]


def test_postpass_reentry_hits_whole_episode_safety():
    slots, episodes = full_pairs()
    episode, raw = fixture()
    raw[-1].update(contacts=[0], clearance=[-.2])
    report = analyzer.analyze_episode(episode, raw)
    assert report["obstacles"][0]["physical"]["clean"]
    episodes[(1, 38300, analyzer.ARMS[1])] = (episode, report)
    result = analyzer.summarize(slots, episodes)
    assert result["new_clean_obstacle_hits"] == [dict(track_id=1, seed=38300, obstacle_id=0)]


def test_candidate_ineligible_windows_cannot_disappear_from_gate_denominator():
    slots, episodes = full_pairs()
    for (_, _, arm), (_, report) in episodes.items():
        second = deepcopy(report["obstacles"][0])
        second["obstacle"]["id"] = 1
        if arm == analyzer.ARMS[1]:
            second["fixed_window"]["status"] = "nonlocal_reverse_or_seam"
        report["obstacles"].append(second)
    result = analyzer.summarize(slots, episodes)
    assert result["baseline_eligible_windows"] == 48
    assert result["candidate_lost_baseline_eligible_windows"] == 24
    assert not result["gates"]["baseline_eligible_windows_preserved"]
    assert not result["gate_passed"]


def test_new_hit_gate_covers_unreached_baseline_no_hit_objects():
    slots, episodes = full_pairs()
    for (_, _, arm), (_, report) in episodes.items():
        second = deepcopy(report["obstacles"][0])
        second["obstacle"]["id"] = 1
        second["fixed_window"]["status"] = "entry_not_reached"
        second["physical"].update(status="not_reached", clean=False)
        second["whole_episode_safety"].update(baseline_clean_reference=False, hit=arm == analyzer.ARMS[1])
        report["obstacles"].append(second)
    result = analyzer.summarize(slots, episodes)
    assert len(result["new_clean_obstacle_hits"]) == 24
    assert not result["gates"]["no_new_clean_obstacle_hit"]


@pytest.mark.parametrize("projected,valid,unavailable", [(None, 0, 1), (42, 0, 0), (47.99, 0, 0), (48, 1, 0), (36, 1, 0), (math.inf, 0, 0), (-math.inf, 0, 0)])
def test_valid_projection_off_is_not_missing_or_centered_projection(projected, valid, unavailable):
    episode, raw = fixture()
    for d in episode["decision_trace"]:
        d["controller"]["projected_obstacle_x"] = projected
    result = analyzer.analyze_episode(episode, raw)
    n = len(episode["decision_trace"])
    assert len(result["crossing_branch_off_avoidance_steering"]) == n
    assert len(result["projection_off_avoidance_steering"]) == n * valid
    assert len(result["projection_unavailable_avoidance_steering"]) == n * unavailable


def test_nonfinite_and_primary_path_hash_guards(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        analyzer.read_json(path)
    path.write_text('{"value":[1e400]}')
    with pytest.raises(ValueError, match="nonfinite"):
        analyzer.read_json(path)
    with pytest.raises(ValueError, match="outside run"):
        analyzer.original.verify_file(tmp_path, "../outside.json", "not-a-hash")


@pytest.fixture
def receipts(tmp_path):
    from scripts import evaluate_koi_minimum_clearance_ab as operator
    baseline_files = operator.frozen_baseline_files()
    baseline = tmp_path / "crossing-projection-source-reconstruction.zip"
    with zipfile.ZipFile(baseline, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(baseline_files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    candidate = tmp_path / "candidate.zip"
    files = {**baseline_files, "haic_agent/minimum_clearance_runtime.py": b"# synthetic, not executed\n"}
    with zipfile.ZipFile(candidate, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    source_hashes = {a: {n: hashlib.sha256(d).hexdigest() for n, d in f.items()} for a, f in zip(analyzer.ARMS, (baseline_files, files))}
    manifest = candidate.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(dict(candidate="koi-minimum-clearance-v1", candidate_zip_sha256=analyzer.sha256(candidate),
                                       baseline_source_sha256=source_hashes[analyzer.ARMS[0]],
                                       files=[dict(path=n, sha256=h) for n, h in source_hashes[analyzer.ARMS[1]].items()])))
    sources = {str(operator.SNAPSHOT / n): h for n, h in analyzer.original.FROZEN_SOURCE_SHA256.items()}
    helpers = {operator.__file__: analyzer.sha256(operator.__file__), analyzer.__file__: analyzer.sha256(analyzer.__file__)}
    copies = []
    for i, (path, expected) in enumerate([*sources.items(), *helpers.items()]):
        copied = tmp_path / f"copy-{i}.py"
        copied.write_bytes(Path(path).read_bytes())
        copies.append(dict(source=path, file=copied.name, sha256=expected))
    protocol = dict(analysis_spec=analyzer.ANALYSIS_SPEC, analyzer_sha256=analyzer.sha256(analyzer.__file__),
                    operator_sha256=analyzer.sha256(operator.__file__), model_source_sha256=source_hashes,
                    model_hashes={analyzer.ARMS[0]: analyzer.sha256(baseline), analyzer.ARMS[1]: analyzer.sha256(candidate)},
                    candidate_manifest_sha256=analyzer.sha256(manifest),
                    cells=[dict(track_id=t, geometry_seed=s, partition="TRAIN", obstacles=True) for t, s in analyzer.CELLS],
                    episodes=48, frame_skip=4, warmup_ticks=50, raw_fps=50, max_decisions=1200,
                    environment_source_sha256=sources, helper_source_sha256=helpers, source_copies=copies,
                    consumed_reuse_evidence=dict(source_sha256={}), runtime_versions=dict(synthetic=True), schedule=operator.scheduled_slots())
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    episode, raw = fixture(analyzer.ARMS[0])
    raw_path = tmp_path / "1-38300-crossing_projection.raw.jsonl"
    raw_path.write_text("".join(json.dumps(r) + "\n" for r in raw))
    episode.update(steps=len(episode["decision_trace"]), raw_trace_file=raw_path.name, raw_trace_sha256=analyzer.sha256(raw_path),
                   versions=dict(synthetic=True), runtime_module_paths={"agent": "/tmp/synthetic-model/agent.py"})
    episode_path = tmp_path / "1-38300-crossing_projection.json"
    episode_path.write_text(json.dumps(episode))
    rows = operator.scheduled_slots()
    rows[0].update(status="completed", file=episode_path.name, sha256=analyzer.sha256(episode_path))
    ledger = tmp_path / "reset-ledger.jsonl"
    ledger.write_text(json.dumps(dict(status="reset_intent", track=1, seed=38300, arm=analyzer.ARMS[0])) + "\n" + json.dumps(rows[0]) + "\n")
    report = dict(rows=rows, protocol_sha256=analyzer.sha256(tmp_path / "protocol.json"), operator_error=None, reset_ledger_sha256=analyzer.sha256(ledger))
    (tmp_path / "episode-report.json").write_text(json.dumps(report))
    return candidate, manifest


def test_primary_loader_verifies_models_sources_and_retains_all_partial_slots(tmp_path, receipts):
    candidate, manifest = receipts
    _, slots, episodes, errors = analyzer.load_run(tmp_path, candidate, manifest)
    assert len(slots) == 48
    assert len(episodes) == 1
    assert errors == ["incomplete_reset_ledger"]
    assert not analyzer.summarize(slots, episodes, errors)["gate_passed"]


@pytest.mark.parametrize("artifact", ["protocol", "episode", "raw", "manifest", "ledger", "copy"])
def test_loader_rejects_primary_hash_corruption(tmp_path, receipts, artifact):
    candidate, manifest = receipts
    paths = dict(protocol=tmp_path / "protocol.json", episode=tmp_path / "1-38300-crossing_projection.json",
                 raw=tmp_path / "1-38300-crossing_projection.raw.jsonl", manifest=manifest,
                 ledger=tmp_path / "reset-ledger.jsonl", copy=tmp_path / "copy-0.py")
    paths[artifact].write_text(paths[artifact].read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        analyzer.load_run(tmp_path, candidate, manifest)


def test_loader_rejects_decision_geometry_not_equal_to_hashed_raw(tmp_path, receipts):
    candidate, manifest = receipts
    path = tmp_path / "1-38300-crossing_projection.json"
    episode = analyzer.read_json(path)
    episode["decision_trace"][0]["controller"]["evaluation_only"]["post"]["x"] += .5
    path.write_text(json.dumps(episode))
    report_path = tmp_path / "episode-report.json"
    report = analyzer.read_json(report_path)
    report["rows"][0]["sha256"] = analyzer.sha256(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="decision/world telemetry"):
        analyzer.load_run(tmp_path, candidate, manifest)


def test_loader_retains_truncated_raw_for_failed_slot(tmp_path, receipts):
    candidate, manifest = receipts
    _, raw = fixture(until=4)
    partial = tmp_path / "1-38300-minimum_clearance_v1.raw.jsonl"
    partial.write_text("".join(json.dumps(r) + "\n" for r in raw) + '{"truncated":')
    _, slots, _, _ = analyzer.load_run(tmp_path, candidate, manifest)
    slot = next(r for r in slots if (r["track_id"], r["seed"], r["mode"]) == (1, 38300, analyzer.ARMS[1]))
    assert slot["partial_raw"]["complete_ticks"] == 4
    assert slot["partial_raw"]["endpoint"]["t"] == raw[-1]["t"]


def test_ledger_duplicate_completion_rejected_even_with_updated_ledger_hash(tmp_path, receipts):
    candidate, manifest = receipts
    ledger = tmp_path / "reset-ledger.jsonl"
    rows = ledger.read_text().splitlines()
    ledger.write_text("\n".join([*rows, rows[1]]) + "\n")
    report_path = tmp_path / "episode-report.json"
    report = analyzer.read_json(report_path)
    report["reset_ledger_sha256"] = analyzer.sha256(ledger)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="duplicate/unknown"):
        analyzer.load_run(tmp_path, candidate, manifest)
