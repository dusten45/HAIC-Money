"""Synthetic receipts only: these tests never construct/reset the environment."""

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import pytest

from scripts import analyze_koi_adaptive_ab as analyzer


def fixture(arm="adaptive_v1", seed=38300, track=1, speed=50.0, until=60.0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obstacle = dict(id=0, x=0.0, y=40.0, radius=1.0, station=40.0, anchor_index=12, tangent=[0.0, 1.0])
    catalog = dict(track=[[0.0, 0.0, 0.0, float(i * 3.5)] for i in range(30)], obstacles=[obstacle])

    def state(station):
        return dict(t=1.02 + station * .02, x=3.0, y=station, yaw=0.0, speed=speed,
                    station=station, road_index=int(station / 3.5), lateral=3.0, heading_error=0.0,
                    front=[station - 40.0 + 2.0], rear=[station - 40.0 - 2.0], clearance=[1.0],
                    contacts=[], wheel_road_contacts=[1, 1, 1, 1])

    initial = state(0.0)
    raw = [dict(state(float(i)), step=(i + 3) // 4, collision=False) for i in range(1, int(until) + 1)]
    decisions = []
    for step, station in enumerate(range(0, int(until), 4), 1):
        pre = state(float(station))
        post = state(float(min(station + 4, until)))
        x, y = analyzer.project_center(obstacle, pre)
        near = [y, x, 42] if 22 <= y < 62 else None
        far = [[y, x, 42]] if 8 <= y < 22 else []
        adaptive = arm == "adaptive_v1" and near is not None
        controller = dict(near_object=near, far_objects=far, pixel_speed=speed,
                          adaptive_speed_reason="adaptive" if adaptive else "no_near_obstacle" if arm == "adaptive_v1" else "baseline",
                          adaptive_speed_changed=adaptive, baseline_pedals=[0.0, 0.2],
                          adaptive_speed_target=50.0 if adaptive else None, adaptive_pass_speed=50.0 if adaptive else None,
                          evaluation_only=dict(pre=pre, post=post, wrapper_off_track=False, wrapper_crashed=False, simulator_terminated=False))
        decisions.append(dict(step=step, controller=controller, collision=False, damage=0.0, speed=speed,
                              steer=0.1, gas=0.0, brake=0.0, car_x=post["x"], car_y=post["y"], car_yaw=0.0, progress=station / 60.0))
    episode = dict(mode=arm, track_id=track, seed=seed, catalog=catalog, initial_state=initial,
                   geometry_sha256=hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
                   initial_observation_sha256="same", decision_trace=decisions, completed=until >= 60,
                   lapTimeMs=1200 if until >= 60 else None, damage=0.0, collisions=0, progress=until / 60.0,
                   retire_reason=None if until >= 60 else "off_track", error=None, invalid_actions=0,
                   damage_telemetry_valid=True, collision_telemetry_valid=True, raw_ticks=len(raw), runtime_module_paths={})
    return episode, raw


def test_camera_rotation_zoom_flip_and_resize():
    origin = dict(t=1.0, x=10.0, y=20.0, yaw=0.0)
    assert analyzer.project_center(dict(x=10, y=20), origin) == pytest.approx([41.9375, 62.8325])
    assert analyzer.project_center(dict(x=11, y=21), origin) == pytest.approx([43.2983, 61.1315])
    origin["yaw"] = math.pi / 2
    assert analyzer.project_center(dict(x=11, y=21), origin) == pytest.approx([43.2983, 64.5335])
    origin.update(t=0, yaw=0)
    assert analyzer.project_center(dict(x=11, y=21), origin) == pytest.approx([41.9879, 62.7695])


@pytest.mark.parametrize("offset,status", [(0, "matched"), (4, "matched"), (4.001, "unmatched")])
def test_detection_inclusive_four_pixel_boundary(offset, status):
    episode, _ = fixture()
    obstacle = episode["catalog"]["obstacles"][0]
    state = episode["initial_state"]
    x, y = analyzer.project_center(obstacle, state)
    assert analyzer.associate_detection([y, x + offset, 42], state, [obstacle])["status"] == status


def test_ambiguity_keeps_all_ids_and_excludes_pair_evidence():
    episode, raw = fixture()
    second = dict(episode["catalog"]["obstacles"][0], id=1, x=.5)
    episode["catalog"]["obstacles"].append(second)
    for state in [episode["initial_state"], *raw]:
        for key in ("front", "rear", "clearance"):
            state[key].append(state[key][0])
    result = analyzer.analyze_episode(episode, raw)
    assert any(e["status"] == "ambiguous" and len(e["candidates"]) == 2 for e in result["detection_events"])
    assert all("ambiguous_detection" in r["segment_exclusions"] for r in result["obstacles"])


def test_detection_uses_pre_camera_not_post_and_keeps_far_near_separately():
    episode, raw = fixture()
    for decision in episode["decision_trace"]:
        decision["car_x"] = 10000
        decision["car_y"] = -10000
    result = analyzer.analyze_episode(episode, raw)
    report = result["obstacles"][0]
    assert report["first_detection"]["near"] is not None
    assert report["first_detection"]["far"] is not None
    assert report["first_detection"]["far"]["t"] < report["first_detection"]["near"]["t"]
    assert report["segment_eligible"]


def test_full_rear_radius_strict_and_conservative_physical_speed():
    episode, raw = fixture()
    result = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert result["physical"]["start_t"] == pytest.approx(1.02 + 37 * .02)
    assert result["physical"]["pass_t"] == pytest.approx(1.02 + 44 * .02)
    assert result["physical"]["pass_bracket"] == pytest.approx([1.88, 1.90])
    assert result["common_entry"]["station"] == 15
    assert result["common_entry_to_pass_s"] == pytest.approx(.58)
    assert result["physical"]["physical_speed_mps"] == dict(n=9, min=50, mean=50, max=50)
    assert result["qualifying_adaptive_above44"]


@pytest.mark.parametrize("defect", ["zero_clearance", "negative_clearance", "contact", "raw_collision", "decision_collision"])
def test_clean_requires_positive_clearance_no_contact_and_no_collision(defect):
    episode, raw = fixture()
    tick = raw[39]
    if defect == "zero_clearance":
        tick["clearance"] = [0]
    elif defect == "negative_clearance":
        tick["clearance"] = [-.1]
    elif defect == "contact":
        tick["contacts"] = [0]
    elif defect == "raw_collision":
        tick["collision"] = True
    else:
        episode["decision_trace"][9]["collision"] = True
    physical = analyzer.analyze_episode(episode, raw)["obstacles"][0]["physical"]
    assert physical["status"] == "passed"
    assert physical["hit"]
    assert not physical["clean"]


def test_incomplete_detected_encounter_retains_endpoint_and_elapsed_times():
    episode, raw = fixture(until=40)
    result = analyzer.analyze_episode(episode, raw)
    obstacle = result["obstacles"][0]
    assert obstacle["physical"]["status"] == "incomplete"
    assert obstacle["physical"]["pass_t"] is None
    assert obstacle["detection_times"]["near"]["detection_to_pass_s"] is None
    assert obstacle["detection_times"]["near"]["elapsed_to_endpoint_s"] > 0
    assert obstacle["physical"]["endpoint_t"] == raw[-1]["t"]
    assert not obstacle["segment_eligible"]
    assert result["failure_context"]["retire_reason"] == "off_track"


def test_every_obstacle_includes_undetected_unreached():
    episode, raw = fixture(until=4)
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert report["physical"]["status"] == "not_reached"
    assert report["first_detection"] == {"near": None, "far": None}
    assert report["detection_times"]["near"]["status"] == "undetected"


def test_unmatched_detector_retained_without_inventing_an_obstacle_id():
    episode, raw = fixture()
    episode["decision_trace"][2]["controller"]["near_object"] = [30, -300, 42]
    events = analyzer.analyze_episode(episode, raw)["detection_events"]
    unmatched = [e for e in events if e["status"] == "unmatched"]
    assert len(unmatched) == 1
    assert unmatched[0]["obstacle_id"] is None


@pytest.mark.parametrize("defect", ["seam", "index_jump", "station_jump", "reverse"])
def test_station_seams_nonlocal_jumps_and_reversal_exclude_segments(defect):
    episode, raw = fixture()
    if defect == "seam":
        episode["catalog"]["obstacles"][0]["station"] = 20
    elif defect == "index_jump":
        raw[19]["road_index"] = 100
    elif defect == "station_jump":
        raw[19]["station"] = 200
    else:
        raw[19]["station"] = 18
    result = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert not result["segment_eligible"]


def test_remote_tangent_plane_crossing_does_not_pass_obstacle():
    episode, raw = fixture()
    states = [dict(episode["initial_state"], collision=False), *raw]
    for state in states[:10]:
        state["front"] = [-2 if state["station"] < 3 else 5]
        state["rear"] = [-3 if state["station"] < 3 else 2]
        state["road_index"] = 100
    physical = analyzer.passage(states, episode["catalog"]["obstacles"][0], 0, episode["decision_trace"])
    assert physical["start_t"] == pytest.approx(1.76)
    assert physical["pass_t"] == pytest.approx(1.90)


def test_predictions_are_not_far_detections():
    episode, raw = fixture()
    for decision in episode["decision_trace"]:
        decision["controller"]["far_objects"] = []
        decision["controller"]["tracked_object"] = [15, 42]
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert report["first_detection"]["far"] is None


def test_baseline_can_already_exceed44_and_peak_speed_is_not_sustained_passage():
    episode, raw = fixture(arm="crossing_projection")
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert report["clean_actual_above44"]
    assert not report["qualifying_adaptive_above44"]
    episode, raw = fixture()
    raw[39]["speed"] = 44
    report = analyzer.analyze_episode(episode, raw)["obstacles"][0]
    assert report["physical"]["physical_speed_mps"]["max"] == 50
    assert not report["qualifying_adaptive_above44"]


def test_hud_scale_aligns_last_two_pre_speeds_and_excludes_clipped80():
    episode, raw = fixture()
    previous = episode["initial_state"]["speed"]
    for i, d in enumerate(episode["decision_trace"]):
        pre = d["controller"]["evaluation_only"]["pre"]
        pre["speed"] = 30 + i
        d["controller"]["pixel_speed"] = 1.1 * (previous + pre["speed"]) / 2
        d["speed"] = 9999
        previous = pre["speed"]
    episode["decision_trace"][-1]["controller"]["pixel_speed"] = 80
    calibration = analyzer.analyze_episode(episode, raw)["hud_calibration"]
    assert calibration["hud_per_actual_mps"] == pytest.approx(1.1)
    assert calibration["differs_from_unity_by_over5pct"]
    assert calibration["n"] == len(episode["decision_trace"]) - 1


def test_failure_classification_is_observational_with_endpoint_evidence():
    episode, raw = fixture(until=40)
    raw[-1].update(contacts=[0], wheel_road_contacts=[0, 0, 1, 1], lateral=8, heading_error=1.2)
    episode["decision_trace"][-1]["controller"]["evaluation_only"]["wrapper_off_track"] = True
    context = analyzer.analyze_episode(episode, raw)["failure_context"]
    assert "obstacle_contact_associated" in context["labels"]
    assert "wheel_road_contact_loss_observed" in context["labels"]
    assert context["contact_obstacle_ids"] == [0]
    assert context["causal_conclusion"] is None
    assert context["lateral"]["max"] == 8
    assert len(context["last_decisions"]) == 10


def full_pairs():
    slots, episodes = [], {}
    for track, seed in analyzer.CELLS:
        for arm in analyzer.ARMS:
            episode, raw = fixture(arm=arm, seed=seed, track=track)
            report = analyzer.analyze_episode(episode, raw)
            if arm == "adaptive_v1":
                report["obstacles"][0]["common_entry_to_pass_s"] -= .1
            episodes[(track, seed, arm)] = (episode, report)
            slots.append(dict(track_id=track, seed=seed, mode=arm, status="completed"))
    return slots, episodes


def test_full48_gate_and_denominators_geometry_seeds_not_track_cells():
    slots, episodes = full_pairs()
    result = analyzer.summarize(slots, episodes)
    assert result["gate_passed"]
    assert result["completion"] == dict(kept=24, lost=0, gained=0, neither=0, paired_cells=24, expected_cells=24)
    assert result["matched_segment_delta_s"]["mean"] == pytest.approx(-.1)
    assert result["qualifying_geometry_seeds"] == list(analyzer.SEEDS)


def test_replication_requires_two_seed_values_not_three_tracks_of_one_seed():
    slots, episodes = full_pairs()
    for (_, seed, arm), (_, report) in episodes.items():
        if arm == "adaptive_v1" and seed != 38300:
            report["obstacles"][0]["qualifying_adaptive_above44"] = False
    result = analyzer.summarize(slots, episodes)
    assert result["qualifying_geometry_seeds"] == [38300]
    assert not result["gate_passed"]


@pytest.mark.parametrize("defect", ["lost", "damage", "collisions", "new_hit", "time", "missing", "provenance", "geometry", "prefix"])
def test_each_adoption_condition_fails_closed(defect):
    slots, episodes = full_pairs()
    key = (1, 38300, "adaptive_v1")
    episode, report = episodes[key]
    errors = []
    if defect == "lost":
        report["completed"] = False
    elif defect == "damage":
        report["damage"] = .2
    elif defect == "collisions":
        report["collisions"] = 1
    elif defect == "new_hit":
        report["obstacles"][0]["whole_episode_safety"]["hit"] = True
    elif defect == "time":
        for (_, _, arm), (_, r) in episodes.items():
            if arm == "adaptive_v1":
                r["obstacles"][0]["common_entry_to_pass_s"] += .1
    elif defect == "missing":
        del episodes[key]
    elif defect == "provenance":
        errors = ["invalid_episode"]
    elif defect == "geometry":
        episode["geometry_sha256"] = "different"
    else:
        episode["decision_trace"][0]["gas"] = 1
    result = analyzer.summarize(slots, episodes, errors)
    assert not result["gate_passed"]
    assert len(result["slots"]) == 48
    assert len(result["cells"]) == 24


def test_failed_endpoints_are_kept_with_null_lap_delta():
    slots, episodes = full_pairs()
    for arm in analyzer.ARMS:
        episode, raw = fixture(arm=arm, until=40)
        episodes[(1, 38300, arm)] = (episode, analyzer.analyze_episode(episode, raw))
    cell = analyzer.summarize(slots, episodes)["cells"][0]
    assert cell["completion"] == "neither"
    assert cell["lap_delta_ms"] is None
    assert cell["endpoints"]["adaptive_v1"]["simulation_end"] == pytest.approx(1.82)


def test_per_cell_mean_not_pooled_segment_selection():
    slots, episodes = full_pairs()
    first = episodes[(1, 38300, "adaptive_v1")][1]["obstacles"][0]
    first["common_entry_to_pass_s"] += 1.0
    result = analyzer.summarize(slots, episodes)
    assert result["per_cell_mean_segment_delta_s"]["n"] == 24
    assert result["matched_segment_delta_s"]["n"] == 24
    assert result["cells"][0]["segments"][0]["pass_delta_bracket_s"] == pytest.approx([-.02, .02])


def test_artifact_hash_and_path_checks(tmp_path):
    artifact = tmp_path / "episode.json"
    artifact.write_text("{}")
    assert analyzer.verify_file(tmp_path, artifact.name, analyzer.sha256(artifact)) == artifact
    with pytest.raises(ValueError, match="hash mismatch"):
        analyzer.verify_file(tmp_path, artifact.name, "bad")
    with pytest.raises(ValueError, match="outside run"):
        analyzer.verify_file(tmp_path, "../outside", "bad")


def test_model_hash_verification_is_passive_and_pinned(tmp_path):
    manifest = analyzer.ROOT / "submissions/koi-adaptive-avoidance-v1.manifest.json"
    candidate = analyzer.ROOT / "submissions/koi-adaptive-avoidance-v1.zip"
    receipt = analyzer.read_json(manifest)
    # Member verification is independent of execution, and rejects a foreign baseline.
    baseline = tmp_path / "crossing-projection-source-reconstruction.zip"
    with zipfile.ZipFile(baseline, "w") as archive:
        archive.writestr("agent.py", "do not import me")
    protocol = dict(model_hashes={"crossing_projection": analyzer.sha256(baseline), "adaptive_v1": analyzer.sha256(candidate)},
                    model_source_sha256={"crossing_projection": {"agent.py": hashlib.sha256(b"do not import me").hexdigest()},
                                         "adaptive_v1": {r["path"]: r["sha256"] for r in receipt["files"]}})
    with pytest.raises(ValueError, match="baseline is not"):
        analyzer.verify_models(protocol, tmp_path, candidate, manifest)


def test_no_environment_or_model_execution_imports():
    source = Path(analyzer.__file__).read_text()
    tree = ast.parse(source)
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
    assert not any(m and (m.startswith(("haic", "gym", "Box2D", "training")) or m == "agent") for m in modules)
    assert "subprocess" not in modules


def test_nonfinite_and_partial_raw_receipts(tmp_path):
    _, raw = fixture(until=1)
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(raw[0]) + '\n{"truncated":')
    assert len(analyzer.read_raw(path, partial=True)) == 1
    with pytest.raises(json.JSONDecodeError):
        analyzer.read_raw(path)
    raw[0]["speed"] = float("nan")
    path.write_text(json.dumps(raw[0]) + "\n")
    with pytest.raises(ValueError, match="nonfinite raw"):
        analyzer.read_raw(path)


def make_receipts(tmp_path):
    from scripts.package_koi_adaptive_avoidance import frozen_baseline_files
    baseline = tmp_path / "crossing-projection-source-reconstruction.zip"
    files = frozen_baseline_files()
    with zipfile.ZipFile(baseline, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    candidate = analyzer.ROOT / "submissions/koi-adaptive-avoidance-v1.zip"
    manifest = analyzer.ROOT / "submissions/koi-adaptive-avoidance-v1.manifest.json"
    receipt = analyzer.read_json(manifest)
    snapshot = Path("/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace")
    protocol = dict(model_hashes={"crossing_projection": analyzer.sha256(baseline), "adaptive_v1": analyzer.sha256(candidate)},
                    model_source_sha256={"crossing_projection": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
                                         "adaptive_v1": {r["path"]: r["sha256"] for r in receipt["files"]}},
                    analysis_spec=analyzer.ANALYSIS_SPEC, analyzer_sha256=analyzer.sha256(analyzer.__file__),
                    operator_sha256=analyzer.sha256(analyzer.ROOT / "scripts/evaluate_koi_adaptive_ab.py"),
                    environment_source_sha256={str(snapshot / name): value for name, value in analyzer.FROZEN_SOURCE_SHA256.items()},
                    cells=[dict(track_id=t, geometry_seed=s, partition="TRAIN", obstacles=True) for t, s in analyzer.CELLS],
                    episodes=48, frame_skip=4, warmup_ticks=50, raw_fps=50, max_decisions=1200)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    episode, raw = fixture()
    episode["steps"] = len(episode["decision_trace"])
    episode["runtime_module_paths"] = {"agent": "/tmp/model-only/agent.py"}
    raw_path = tmp_path / "1-38300-adaptive_v1.raw.jsonl"
    raw_path.write_text("".join(json.dumps(r) + "\n" for r in raw))
    episode.update(raw_trace_file=raw_path.name, raw_trace_sha256=analyzer.sha256(raw_path))
    episode_path = tmp_path / "1-38300-adaptive_v1.json"
    episode_path.write_text(json.dumps(episode))
    rows = [dict(track_id=t, seed=s, mode=a, status="unrun") for t, s in analyzer.CELLS for a in analyzer.ARMS]
    row = next(r for r in rows if (r["track_id"], r["seed"], r["mode"]) == (1, 38300, "adaptive_v1"))
    row.update(status="completed", file=episode_path.name, sha256=analyzer.sha256(episode_path))
    report = dict(protocol_sha256=analyzer.sha256(tmp_path / "protocol.json"), rows=rows, operator_error=None)
    (tmp_path / "episode-report.json").write_text(json.dumps(report))
    return candidate, manifest


def test_loader_verifies_real_frozen_members_and_retains48_partial_slots(tmp_path):
    candidate, manifest = make_receipts(tmp_path)
    _, slots, episodes, errors = analyzer.load_run(tmp_path, candidate, manifest)
    assert len(slots) == 48
    assert len(episodes) == 1
    assert errors == []
    assert not analyzer.summarize(slots, episodes, errors)["gate_passed"]


@pytest.mark.parametrize("artifact", ["protocol", "episode", "raw", "manifest"])
def test_loader_rejects_primary_hash_corruption(tmp_path, artifact):
    candidate, manifest = make_receipts(tmp_path)
    path = {"protocol": tmp_path / "protocol.json", "episode": tmp_path / "1-38300-adaptive_v1.json",
            "raw": tmp_path / "1-38300-adaptive_v1.raw.jsonl", "manifest": manifest}[artifact]
    if artifact == "manifest":
        copied = tmp_path / "foreign-manifest.json"
        copied.write_text(path.read_text() + " ")
        manifest = copied
    else:
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        analyzer.load_run(tmp_path, candidate, manifest)


def test_loader_rejects_duplicate_or_out_of_cohort_slots(tmp_path):
    candidate, manifest = make_receipts(tmp_path)
    report_path = tmp_path / "episode-report.json"
    report = analyzer.read_json(report_path)
    report["rows"].append(deepcopy(report["rows"][0]))
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="duplicate slot"):
        analyzer.load_run(tmp_path, candidate, manifest)


def test_loader_retains_failed_process_raw_endpoint_without_catalog(tmp_path):
    candidate, manifest = make_receipts(tmp_path)
    _, raw = fixture(until=4)
    (tmp_path / "1-38300-crossing_projection.raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw) + '{"partial":')
    _, slots, _, _ = analyzer.load_run(tmp_path, candidate, manifest)
    failed = next(r for r in slots if r["mode"] == "crossing_projection" and r["track_id"] == 1 and r["seed"] == 38300)
    assert failed["status"] == "unrun"
    assert failed["partial_raw"]["complete_ticks"] == 4
    assert failed["partial_raw"]["endpoint"]["t"] == raw[-1]["t"]


def test_loader_marks_unfrozen_analyzer_as_ineligible_not_new_measurement_rule(tmp_path):
    candidate, manifest = make_receipts(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = analyzer.read_json(protocol_path)
    protocol["analysis_spec"] = {"version": "other"}
    protocol_path.write_text(json.dumps(protocol))
    report_path = tmp_path / "episode-report.json"
    report = analyzer.read_json(report_path)
    report["protocol_sha256"] = analyzer.sha256(protocol_path)
    report_path.write_text(json.dumps(report))
    _, _, _, errors = analyzer.load_run(tmp_path, candidate, manifest)
    assert "analysis_spec_not_frozen_or_differs" in errors


def test_later_reentry_contact_is_new_clean_obstacle_hit_even_after_clean_first_pass():
    slots, episodes = full_pairs()
    episode, raw = fixture()
    raw[-1]["contacts"] = [0]
    raw[-1]["clearance"] = [-1]
    report = analyzer.analyze_episode(episode, raw)
    assert report["obstacles"][0]["physical"]["clean"]
    assert report["obstacles"][0]["whole_episode_safety"]["hit"]
    episodes[(1, 38300, "adaptive_v1")] = (episode, report)
    result = analyzer.summarize(slots, episodes)
    assert result["new_clean_obstacle_hits"] == [dict(track_id=1, seed=38300, obstacle_id=0)]
    assert not result["gates"]["no_new_clean_obstacle_hit"]


def test_unassociated_transient_collision_fails_closed_without_guessing_nearest_id():
    slots, episodes = full_pairs()
    episode, raw = fixture()
    raw[-1]["collision"] = True
    report = analyzer.analyze_episode(episode, raw)
    assert report["unassociated_collision_events"] == [dict(t=raw[-1]["t"], step=raw[-1]["step"], source="raw")]
    assert not report["obstacles"][0]["whole_episode_safety"]["hit"]
    episodes[(1, 38300, "adaptive_v1")] = (episode, report)
    assert not analyzer.summarize(slots, episodes)["gates"]["no_new_clean_obstacle_hit"]


@pytest.mark.parametrize("reason", ["act_timeout", "reset_timeout", "agent_reset_error", "agent_setup_error", "evaluation_error", "invalid_action"])
def test_operational_retirement_cannot_count_as_valid_even_with_prior_valid_telemetry(tmp_path, reason):
    candidate, manifest = make_receipts(tmp_path)
    path = tmp_path / "1-38300-adaptive_v1.json"
    episode = analyzer.read_json(path)
    episode["retire_reason"] = reason
    path.write_text(json.dumps(episode))
    report_path = tmp_path / "episode-report.json"
    report = analyzer.read_json(report_path)
    row = next(r for r in report["rows"] if r.get("file") == path.name)
    row["sha256"] = analyzer.sha256(path)
    report_path.write_text(json.dumps(report))
    _, _, _, errors = analyzer.load_run(tmp_path, candidate, manifest)
    assert any(e.startswith("invalid_episode:") for e in errors)


def test_duration_bracket_adjusts_for_different_common_entry_clocks():
    slots, episodes = full_pairs()
    report = episodes[(1, 38300, "adaptive_v1")][1]
    report["obstacles"][0]["common_entry"]["t"] += .3
    segment = analyzer.summarize(slots, episodes)["cells"][0]["segments"][0]
    assert segment["pass_delta_bracket_s"] == pytest.approx([-.02, .02])
    assert segment["duration_delta_bracket_s"] == pytest.approx([-.32, -.28])


def test_arm_reporting_exposes_baseline_actual44_and_matched_denominators():
    slots, episodes = full_pairs()
    result = analyzer.summarize(slots, episodes)
    base = result["arm_summaries"]["crossing_projection"]
    adaptive = result["arm_summaries"]["adaptive_v1"]
    assert base["episodes_available"] == 24
    assert base["clean_actual_min_above44_passages"] == 24
    assert base["adaptive_changed_decisions"] == 0
    assert adaptive["adaptive_changed_decisions"] > 0
    assert base["matched_common_entry_to_pass_s"]["n"] == 24
    assert adaptive["simulation_elapsed_s"] == pytest.approx(28.8)
    assert result["mutually_finished_lap_delta_ms"] == dict(n=24, min=0, mean=0, max=0)
