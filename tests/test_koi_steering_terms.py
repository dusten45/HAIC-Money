"""Pure accounting tests. Never instantiate an original controller or environment."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from haic.algorithms.koi.steering_terms import (
    SCHEMA, decompose, near_scaled_steer, near_scaled_terms, observed_center, reconstruct_steering,
)


def diagnostic(side=1., obj=(40., 38., 42.), correction=.04, centers=None, crossing=False) -> dict[str, Any]:
    centers = {30: 42., 42: 44., 54: 42.} if centers is None else centers
    road = .022 * (centers.get(42, 42.) - 42) + .018 * (centers.get(42, 42.) - centers.get(54, 42.))
    near = side * .34 * float(np.clip((obj[0] - 22) / 18, 0, 1)) if obj is not None else 0.
    inherited = float(np.float32(np.clip(road + near, -.7, .7)))
    damping = float(np.float32(np.clip(inherited + correction, -.7, .7)))
    return dict(road_centers=centers, near_object=obj, correction=correction, actual_obstacle_side=side,
                steps=20, inherited_steer=inherited, damping_steer=damping, parent_action=[damping, .1, 0.],
                geometry_repaired=False, contact_mode="crossing_projection", contact_active=crossing,
                far_mode="arrival_speed", braking_proxy_veto=False, impact_steps_remaining=0,
                impact_proxy_trigger=False, actual_impact_left=0, projected_obstacle_x=42. if crossing else 51.)


def test_inherited_exact_stages_and_sum():
    d = diagnostic()
    r = reconstruct_steering(d, d["parent_action"][0])
    assert r["schema"] == SCHEMA and r["reconstruction_valid"]
    assert r["effective_avoidance_raw"] == .34
    assert sum(r["contributions"].values()) == r["actual_final_steer"]
    assert r["residual"] == 0
    assert [s["name"] for s in r["stages"]][:2] == ["base_clip", "base_float32"]


def test_crossing_replaces_not_double_counts():
    d = diagnostic(crossing=True)
    final = float(np.float32(.08 + .55 + .04))
    r = reconstruct_steering(d, final)
    assert r["reconstruction_valid"] and r["effective_avoidance_raw"] == .55
    assert r["raw_terms"]["near_urgency"] == .34
    assert r["raw_terms"]["crossing_projection"] == .55
    assert r["avoidance_component"] == pytest.approx(.55, abs=1e-7)
    assert r["stages"][-1]["operation"] == "replace"


def test_nested_clip_not_clip_flat_sum():
    d = diagnostic(centers={30: 65., 42: 65., 54: 42.}, correction=-.2, crossing=True)
    final = float(np.float32(.7 - .2))
    r = reconstruct_steering(d, final)
    assert r["reconstruction_valid"] and r["saturation"]
    assert r["reconstructed_steer"] != float(np.float32(np.clip(.022 * 23 + .018 * 23 + .55 - .2, -.7, .7)))


def test_geometry_repair_single_clip():
    d = diagnostic(centers={30: 60., 38: 54., 54: 40.}, correction=-.1, crossing=True)
    d["geometry_repaired"] = True
    m, n = observed_center(d["road_centers"], 42), observed_center(d["road_centers"], 54)
    assert m is not None and n is not None
    d["parent_action"][0] = float(np.float32(np.clip(.022 * (m - 42) + .018 * (m - n) + .34 - .1, -.7, .7)))
    final = float(np.float32(np.clip(.022 * (m - 42) + .018 * (m - n) + .55 - .1, -.7, .7)))
    r = reconstruct_steering(d, final)
    assert r["reconstruction_valid"]
    assert not any(s["name"] == "crossing_inner_clip" for s in r["stages"])


@pytest.mark.parametrize("remaining,trigger", [(11, True), (3, False), (0, False)])
def test_impact_replacement_including_expired_last_call(remaining, trigger):
    d = diagnostic()
    d.update(impact_steps_remaining=remaining, impact_proxy_trigger=trigger)
    d["parent_action"][0] = float(np.float32(.08 + .04))
    r = reconstruct_steering(d, d["parent_action"][0])
    assert r["reconstruction_valid"] and r["avoidance_component"] == 0
    assert r["active_path"] == "impact_replacement"
    if remaining == 0:
        assert r["inference"]


def test_impact_braking_veto_restores_damping():
    d = diagnostic()
    d.update(impact_proxy_trigger=True, braking_proxy_veto=True)
    r = reconstruct_steering(d, d["parent_action"][0])
    assert r["reconstruction_valid"] and r["effective_avoidance_raw"] == .34


@pytest.mark.parametrize("before,expected_path,expected_raw", [
    (1, "impact_replacement", 0.0),
    (0, "inherited_damping", .34),
    (None, "ambiguous_impact_or_inherited", None),
])
def test_expired_saturated_impact_truthful_raw_stages(before, expected_path, expected_raw):
    d = diagnostic(centers={30: 42., 42: 64.5, 54: 42.}, correction=0.)
    if before is not None:
        d["impact_left_before_act"] = before
    r = reconstruct_steering(d, d["parent_action"][0])
    assert r["reconstruction_valid"] and r["residual_valid"] and r["residual"] == 0
    assert r["active_path"] == expected_path and r["effective_avoidance_raw"] == expected_raw
    assert r["avoidance_component"] == r["contributions"]["avoidance"] == 0
    if before is None:
        assert r["impact_assignment"] is None and r["impact_assignment_ambiguous"]
        assert r["raw_terms"]["impact_replacement_raw"] is None
        assert "impact_assignment" in r["unidentifiable"]
        assert any(s["operation"] == "ambiguous_replace" and s["applied"] is None for s in r["stages"])
        assert near_scaled_steer(r, 0.) is None and near_scaled_steer(r, 1.) is None
    elif before:
        assert r["impact_assignment"] and not r["impact_assignment_ambiguous"]
        assert any(s["name"] == "impact_replace" for s in r["stages"])
        assert near_scaled_steer(r, 0.) is None
    else:
        assert r["impact_assignment"] is False and not r["unidentifiable"]
        assert not any(s["name"] == "impact_replace" for s in r["stages"])
        assert near_scaled_steer(r, 0.) == r["actual_final_steer"]


@pytest.mark.parametrize("state_before,diagnostic_before", [(0, 1), (1, 0)])
def test_explicit_pre_state_takes_precedence_over_diagnostic(state_before, diagnostic_before):
    d = diagnostic(centers={30: 42., 42: 64.5, 54: 42.}, correction=0.)
    d["impact_left_before_act"] = diagnostic_before
    r = reconstruct_steering(d, d["parent_action"][0], state={"impact_left_before_act": state_before})
    assert r["reconstruction_valid"] and r["impact_assignment"] is bool(state_before)
    assert r["impact_left_before_act"] == state_before
    assert r["impact_prestate_source"] == "explicit_state_pre_act"


def test_pre_act_capture_passes_through_decompose_without_mutation():
    d = diagnostic(centers={30: 42., 42: 64.5, 54: 42.}, correction=0.)
    d["impact_left_before_act"] = 1
    original = copy.deepcopy(d)
    driver = SimpleNamespace(base=SimpleNamespace(_obstacle_side=1.), steps=20, impact_left=0)
    r = decompose(driver, None, d["parent_action"], d)
    assert r["active_path"] == "impact_replacement" and r["effective_avoidance_raw"] == 0
    assert d == original and driver.impact_left == 0


def test_equal_latent_paths_remain_ambiguous_within_numeric_tolerance():
    d = diagnostic(centers={30: 42., 42: 64.5, 54: 42.}, correction=0.)
    d["parent_action"][0] = .7
    r = reconstruct_steering(d, .7)
    assert r["reconstruction_valid"] and r["impact_assignment_ambiguous"]
    assert r["effective_avoidance_raw"] is None and near_scaled_steer(r, 0.) is None


def test_pre_capture_preserves_recovery_then_braking_veto_priority():
    d = diagnostic()
    d.update(impact_left_before_act=0, impact_proxy_trigger=True, impact_steps_remaining=11, braking_proxy_veto=True)
    r = reconstruct_steering(d, d["parent_action"][0])
    names = [s["name"] for s in r["stages"]]
    assert r["reconstruction_valid"] and r["impact_assignment"]
    assert names.index("impact_replace") < names.index("impact_veto_replace")
    assert r["active_path"] == "inherited_damping" and r["effective_avoidance_raw"] == .34


def test_missing_middle_row_keeps_repair_priority_and_explicit_float32():
    d = diagnostic(centers={30: 60., 38: 54., 54: 40.}, correction=-.1)
    d.update(impact_left_before_act=1, geometry_repaired=True)
    m = observed_center(d["road_centers"], 42)
    assert m is not None
    raw = .022 * (m - 42) + .018 * (m - 40) + .34 - .1
    final = float(np.float32(np.clip(raw, -.7, .7)))
    d["parent_action"][0] = final
    r = reconstruct_steering(d, final)
    assert r["reconstruction_valid"] and not r["impact_assignment"]
    assert r["active_path"] == "geometry_repair" and r["effective_avoidance_raw"] == .34
    rounding = next(s for s in r["float32_rounding"] if s["name"] == "geometry_repair_float32")
    assert rounding["delta"] == final - raw and rounding["delta"] != 0


@pytest.mark.parametrize("bad", [-1, 1.5, float("nan"), float("inf")])
def test_invalid_pre_act_counter_is_not_inferred(bad):
    d = diagnostic()
    d["impact_left_before_act"] = bad
    r = reconstruct_steering(d, d["parent_action"][0])
    assert not r["reconstruction_valid"] and "valid_impact_left_before_act" in r["unidentifiable"]
    assert near_scaled_steer(r, 0.) is None


def test_frozen_recovery_expired_saturation_has_two_equal_observations():
    path = Path("/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace/haic_agent/fixed_high_speed_recovery_v2.py")
    tree = ast.parse(path.read_text())
    act = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "act")
    recovery = next(n for n in act.body if isinstance(n, ast.If) and "memory_reacquire" in ast.unparse(n.test))
    decrement = next(n for n in act.body if isinstance(n, ast.If) and ast.unparse(n.test) == "self.impact_left")
    outputs = []
    for before in (0, 1):
        holder = SimpleNamespace(steps=20, recovery_mode="impact_clear", impact_left=before, last={"correction": 0.})
        namespace = {"np": np, "self": holder, "complete": True, "centers": {42: 64.5, 54: 42.},
                     "action": np.asarray([.7, .1, 0.], dtype=np.float32)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[recovery, decrement], type_ignores=[])), str(path), "exec"), namespace)
        outputs.append((float(namespace["action"][0]), holder.impact_left))
    assert outputs[0] == outputs[1] == (float(np.float32(.7)), 0)


def test_archived_constructor_prestate_zero_matches_frozen_reset_source():
    root = Path("/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace/haic_agent")
    for name, attribute in (("fixed_high_speed_runtime.py", "steps"), ("fixed_high_speed_recovery_v2.py", "impact_left")):
        tree = ast.parse((root / name).read_text())
        reset = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "reset")
        assignments = [n for n in reset.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self" and t.attr == attribute
            for t in n.targets)]
        assert len(assignments) == 1 and ast.literal_eval(assignments[0].value) == 0
    tree = ast.parse((root / "fixed_high_speed_runtime.py").read_text())
    constructor = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "reset"
               and isinstance(n.func.value, ast.Name) and n.func.value.id == "self" for n in ast.walk(constructor))


def test_prefix_can_have_base_avoidance():
    d = diagnostic(correction=0)
    d["steps"] = 3
    r = reconstruct_steering(d, d["parent_action"][0])
    assert r["reconstruction_valid"] and r["avoidance_component"] > .33


@pytest.mark.parametrize("missing", ["actual_obstacle_side", "correction", "steps", "road_centers", "contact_active", "geometry_repaired", "near_object", "far_mode"])
def test_missing_is_unidentifiable_not_zero(missing):
    d = diagnostic()
    del d[missing]
    r = reconstruct_steering(d, .1)
    assert not r["reconstruction_valid"] and r["residual"] is None
    assert r["unidentifiable"] and r["avoidance_component"] is None


@pytest.mark.parametrize("projected,status", [(None, "unavailable"), (48., "off"), (36., "off"), (47.99, "crossing"), (float("inf"), "invalid")])
def test_projection_available_is_not_branch_active(projected, status):
    d = diagnostic()
    d["projected_obstacle_x"] = projected
    assert reconstruct_steering(d, d["parent_action"][0])["projection_state"] == status


def test_residual_failure_retains_unclassified_value():
    d = diagnostic()
    r = reconstruct_steering(d, d["parent_action"][0] + .1)
    assert not r["reconstruction_valid"] and not r["residual_valid"]
    assert r["unclassified_residual"] == pytest.approx(.1)


def test_decompose_post_act_api_no_mutation():
    d = diagnostic()
    driver = SimpleNamespace(base=SimpleNamespace(_obstacle_side=1.), steps=20, impact_left=0)
    before = copy.deepcopy(d)
    r = decompose(driver, None, [d["parent_action"][0], .1, 0], d)
    assert r["reconstruction_valid"] and d == before


@pytest.mark.parametrize("alpha", [0., .25, .5, 1.])
def test_near_only_scaled_pipeline_and_actual_accounting(alpha):
    d = diagnostic(centers={30: 65., 42: 65., 54: 42.}, correction=-.2)
    baseline = reconstruct_steering(d, d["parent_action"][0])
    before = copy.deepcopy(baseline)
    expected = float(np.float32(np.clip(float(np.float32(np.clip(.022 * 23 + .018 * 23 + .34 * alpha, -.7, .7))) - .2, -.7, .7)))
    assert near_scaled_steer(baseline, alpha) == expected
    actual = near_scaled_terms(baseline, alpha, final_steer=expected)
    assert actual is not None
    assert actual["reconstruction_valid"] and actual["residual_valid"] and actual["residual"] == 0
    assert actual["reconstruction_context"] == "actual_postrelease"
    assert baseline == before


@pytest.mark.parametrize("protected", ["crossing", "impact", "prefix", "invalid"])
def test_near_scaling_protected_and_unidentified_failclosed(protected):
    d = diagnostic(crossing=protected == "crossing")
    if protected == "impact":
        d["impact_steps_remaining"] = 4
        d["parent_action"][0] = float(np.float32(.12))
    if protected == "prefix":
        d["steps"] = 4
    final = float(np.float32(.67)) if protected == "crossing" else d["parent_action"][0]
    r = reconstruct_steering(d, final + (.1 if protected == "invalid" else 0))
    assert near_scaled_steer(r, 0.) is None


@pytest.mark.parametrize("alpha", [-.1, 1.1, float("nan"), float("inf")])
def test_near_scale_invalid_numeric_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        near_scaled_steer({}, alpha)


def test_observed_center_stable_tie_order():
    assert observed_center({38: 40., 46: 48.}, 42) == 44.
    assert observed_center({30: 40.}, 42) is None
    assert observed_center({50: 80., 54: 70.}, 42) == 83.


def test_frozen_contact_function_executable_without_controller():
    path = Path("/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace/haic_agent/contact_continuity_runtime.py")
    tree = ast.parse(path.read_text())
    act = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "act")
    function = next(n for n in act.body if isinstance(n, ast.FunctionDef) and n.name == "avoid_steer")
    for repaired in (False, True):
        for road, correction, term in ((1.1, -.3, .55), (-1.2, .4, -.55), (.08, .04, .55)):
            namespace = {"np": np, "self": SimpleNamespace(last={"geometry_repaired": repaired, "correction": correction}), "road": road + correction}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
            expected = namespace["avoid_steer"](term)
            d = diagnostic(centers={30: 42., 42: 42. + road / .04, 54: 42.}, correction=correction, crossing=True)
            if repaired:
                d["geometry_repaired"] = True
                d["parent_action"][0] = float(np.float32(np.clip(road + .34 + correction, -.7, .7)))
            # This fixture uses .55's sign from the original observed side.
            if term < 0:
                d = diagnostic(side=-1., centers=d["road_centers"], correction=correction, crossing=True)
                d["geometry_repaired"] = repaired
                if repaired:
                    d["parent_action"][0] = float(np.float32(np.clip(road - .34 + correction, -.7, .7)))
            result = reconstruct_steering(d, float(np.float32(expected)))
            assert result["reconstruction_valid"]
