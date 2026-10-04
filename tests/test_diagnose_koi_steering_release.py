import ast
from pathlib import Path
import sys

import pytest

from scripts.diagnose_koi_steering_release import (
    DEFAULT_RUN, HALF_WIDTH, associate_detection, decision_rows, epochs,
    load_inputs, object_timeline, physical_row, prior_sideflip_context, project_center, read_json,
)
from haic.algorithms.koi.steering_terms import near_scaled_steer


def obstacle(x=0., y=0., oid=0):
    return dict(id=oid, x=x, y=y, radius=1., tangent=[0., 1.], station=50., anchor_index=10)


def state(t=1., x=5., y=-5., front=-2., rear=-7., clearance=3.):
    return dict(t=t, x=x, y=y, yaw=0., station=45. + y + 5., road_index=10, lateral=x, speed=40.,
                front=[front], rear=[rear], clearance=[clearance], contacts=[], wheel_road_contacts=[1, 1, 1, 1])


def test_positive_clearance_far_is_not_lateral_escape():
    s = state(y=-80., front=-77., rear=-83., clearance=75.)
    p = physical_row(s, obstacle(), 0)
    assert p["exact_positive_clearance"] and not p["conservative_lateral_separated"]


def test_exact_positive_involvement_not_automatically_lateral_certificate():
    s = state(x=2.5, y=-1., front=2., rear=-4., clearance=.1)
    p = physical_row(s, obstacle(), 0)
    assert p["observed_clear_while_involved"]
    assert p["nominal_yaw_lateral_gap"] < 0 and not p["conservative_lateral_separated"]


def test_separated_before_rear_clear_involvement_retained():
    p = physical_row(state(y=-1., front=2., rear=-4.), obstacle(), 0)
    assert p["conservative_lateral_separated"] and p["longitudinal_involved"]
    assert not p["full_rear_clear"] and HALF_WIDTH > 1.57
    assert p["conservative_lateral_safe_while_involved"] and p["conservative_lateral_safe_in_window"]


def test_contact_or_nonpositive_observed_clearance_vetoes_nominal_safe():
    s = state(clearance=-.01)
    assert not physical_row(s, obstacle(), 0)["conservative_lateral_separated"]
    s["clearance"] = [1.]
    s["contacts"] = [0]
    assert not physical_row(s, obstacle(), 0)["conservative_lateral_separated"]


def test_passed_is_not_already_separated():
    p = physical_row(state(y=10., front=13., rear=7.), obstacle(), 0)
    assert p["full_rear_clear"] and not p["conservative_lateral_separated"]


def test_unique_association_no_nearest_shortcut():
    s = state()
    o = obstacle()
    x, y = project_center(o, s)
    assert associate_detection((y, x, 42.), s, [o])["obstacle_id"] == 0
    assert associate_detection((y, x, 42.), s, [o, obstacle(.1, 0., 1)])["status"] == "ambiguous"
    assert associate_detection(None, s, [o])["status"] == "undetected"


def test_epochs_keep_reentry_and_censor_brackets():
    rows = [dict(t=i * .02, safe=v) for i, v in enumerate([False, True, True, False, True])]
    result = epochs(rows, "safe")
    assert len(result) == 2 and result[0]["start_bracket"] == [0., .02]
    assert result[0]["end_bracket"] == [.04, .06] and result[1]["right_censored"]


@pytest.fixture(scope="module")
def archived():
    return load_inputs(DEFAULT_RUN)


def test_all_archived_source_reconstructions_zero_environment(archived):
    modules_before = set(sys.modules)
    episodes, inventory, baseline = archived
    rows = [r for e, _ in episodes for r in decision_rows(e)]
    assert len(episodes) == 24 and len(rows) == 5703
    assert sum(len(raw) for _, raw in episodes) == 22779
    invalid = [(r["track_id"], r["seed"], r["step"], r["decomposition"].get("checkpoint_residuals"), r["decomposition"].get("unidentifiable")) for r in rows if not r["decomposition"]["reconstruction_valid"]]
    assert not invalid, invalid[:20]
    assert all(r["decomposition"]["residual"] == 0 for r in rows)
    assert all(near_scaled_steer(r["decomposition"], 1.) == r["steer"] for r in rows)
    assert all(near_scaled_steer(r["decomposition"], 0.) == r["steer"] - r["decomposition"]["avoidance_component"]
               for r in rows if r["step"] > 10 and r["decomposition"]["active_path"] in ("inherited_damping", "geometry_repair"))
    assert len(baseline) == 10 and len(inventory) >= 210
    assert not any(n.startswith(("gymnasium", "Box2D", "haic_agent")) for n in set(sys.modules) - modules_before)


def test_projection_identity_disallows_first_ever_far_off(archived):
    e, _ = next((e, raw) for e, raw in archived[0] if e["track_id"] == 1 and e["seed"] == 38301)
    rows = decision_rows(e)
    for r in rows:
        if not r["previous_same_object"]:
            assert not r["projection_identity_valid"]


def test_archive_pre_counter_is_source_local_previous_same_cell(archived):
    expired = []
    for episode, _ in archived[0]:
        rows = decision_rows(episode)
        assert rows[0]["original_controller_pre_act_state"] == {
            "steps": 0, "impact_left": 0, "source": "original_constructor_reset_zero",
        }
        for i, row in enumerate(rows):
            terms = row["decomposition"]
            expected = episode["decision_trace"][i - 1]["controller"]["actual_impact_left"] if i else 0
            assert terms["impact_left_before_act"] == expected
            assert not terms["impact_assignment_ambiguous"] and not terms["unidentifiable"]
            assert terms["residual"] == 0
            if expected == 1 and row["original_diagnostics"]["actual_impact_left"] == 0:
                expired.append((row["track_id"], row["seed"], row["step"], terms["active_path"]))
    assert expired == [(2, 50302, 152, "impact_replacement")]


def test_onroad_stuck_failure_not_hidden_by_retire_label(archived):
    e, raw = next((e, raw) for e, raw in archived[0] if e["track_id"] == 2 and e["seed"] == 50302)
    assert not e["completed"]
    assert raw[-1]["speed"] < .001 and raw[-1]["contacts"]
    assert all(n > 0 for n in raw[-1]["wheel_road_contacts"])


def test_all144_timeline_objects_and_seams_retained(archived):
    events = []
    for e, raw in archived[0]:
        decisions = decision_rows(e)
        events.extend(object_timeline(e, [e["initial_state"], *raw], decisions, o, i) for i, o in enumerate(e["catalog"]["obstacles"]))
    assert len(events) == 144 and sum(e["seam_flag"] for e in events) == 18
    assert all(e["physical_rows"] is not None and "full_rear_clear" in e for e in events)
    assert all(not r["disappearance_is_clearance"] for e in events for r in e["detector_dropouts"])
    assert sum(len(e["active_to_measured_off"]) for e in events) == 42
    assert all(t["off_step"] == t["active_step"] + 1 for e in events for t in e["active_to_measured_off"])
    assert all("first_safe_lateral_window" in e and "first_safe_lateral_while_involved" in e for e in events)


def test_prior_sideflip_risk_is_bound_not_new_baseline_cell():
    context, hashes = prior_sideflip_context()
    assert context["sole_changed_decisions"] == [31]
    assert context["baseline_completed"] and not context["failed_candidate_completed"]
    assert context["object3_contact_ticks"] == 404 and context["all_object3_contact_ticks_onroad"]
    assert context["original_same_observation_rows"] is not None
    by_step = {r["step"]: r for r in context["original_same_observation_rows"]}
    assert by_step[121]["side"] == 1 and by_step[122]["side"] == -1 and by_step[124]["side"] == 1
    assert all(r["crossing_terms"]["reconstruction_valid"] for r in by_step.values())
    assert len(hashes) == 6


@pytest.mark.parametrize("kind", ["source", "episode", "raw"])
def test_loader_hash_corruption_failclosed(monkeypatch, kind):
    from scripts import diagnose_koi_steering_release as module
    protocol = read_json(DEFAULT_RUN / "protocol.json")
    if kind == "source":
        target = DEFAULT_RUN / protocol["source_copies"][0]["file"]
    elif kind == "episode":
        target = DEFAULT_RUN / "1-38301-crossing_projection.json"
    else:
        target = DEFAULT_RUN / "1-38301-crossing_projection.raw.jsonl"
    original = module.sha
    monkeypatch.setattr(module, "sha", lambda path: "0" * 64 if Path(path) == target else original(path))
    with pytest.raises(ValueError, match="hash mismatch"):
        module.load_inputs(DEFAULT_RUN)


def test_no_environment_or_model_construction_in_diagnosis_sources():
    for path in (Path("scripts/diagnose_koi_steering_release.py"), Path("haic/algorithms/koi/steering_terms.py")):
        tree = ast.parse(path.read_text())
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(n.startswith(("gym", "haic_agent", "Box2D")) for n in imports)
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        assert "reset" not in calls and "act" not in calls and "make" not in calls


def test_nonfinite_json_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"bad": [1e400]}')
    with pytest.raises(ValueError, match="nonfinite"):
        read_json(p)
