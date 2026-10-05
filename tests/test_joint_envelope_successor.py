"""Synthetic performance comparisons and frozen Agent pixels only; no simulator."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import FunctionType, MethodType
from unittest.mock import Mock

import numpy as np
import pytest

from haic.algorithms.joint_control import comparison, interval_comparison, paired_residual, successor
from haic.algorithms.joint_control import envelope_successor as new
from tests.test_joint_temporal_successor import (
    Champion, Nominal, ValidObserver, calibration as absolute_calibration,
    equal, frozen, image,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_PINS = {
    "comparison.py": "8918ddf09d7853134c75eba656c1900341f8c40d4daeca1ae4f70cdca24640e1",
    "hud.py": "07013302813d626af0780bd6b11d25ec4d0a3d693ab4ff28f86cf6cb98112ed8",
    "interval_comparison.py": "d360147a96fb990e9e48af6bfe48fe93a99182b82591ff49e77787cbb8be6b7c",
    "motion.py": "cd5dec6a13e2b5e7fadcb9c16dbf07128a978b789be32ab16271eb1ade729f0e",
    "observer.py": "168d806fa11e6502b1ddde46fa46f9ba91e47c371e0a570b5b25c090732d8346",
    "physics.py": "de7d9d83983476ef3909e76e8dde79456853e6c59fadaa44c4cf3705a30c9590",
    "successor.py": "5bfb7ba054d8204c5c1fc9f8dd930d9f00bf1217c57539698c2f72db45676378",
}


@pytest.fixture(scope="module", autouse=True)
def pinned_sources_unchanged():
    def hashes():
        return {name: hashlib.sha256((ROOT / "haic/algorithms/joint_control" / name).read_bytes()).hexdigest()
                for name in OLD_PINS}
    assert hashes() == OLD_PINS
    yield
    assert hashes() == OLD_PINS


def calibration():
    absolute = absolute_calibration()
    absolute["paired_cost_residual"] = .44961874671412616
    return dict(schema=paired_residual.SCHEMA, calibration_complete=True,
                absolute_calibration=absolute, performance=dict(lower=.05, upper=.05, floor=.05,
                    target=new.PERFORMANCE_TARGET))


def template(upper=-.2, *, supported=True, veto=False):
    delta = np.zeros((2, 19, 5))
    delta[1] = np.linspace(upper - .1, upper, 95).reshape(19, 5)
    if not supported:
        delta[1, 7, 2] = np.nan
    return dict(reference_delta=delta, common_support=supported,
                cost_supported=np.full(2, supported), absolute_supported=np.ones((2, 19), bool),
                veto=np.full((2, 19), veto), poses=np.zeros((2, 19, 17, 3)),
                costs=np.ones((2, 19)), components=dict(progress=np.ones((2, 19))),
                absolute_road_clearance=np.full((2, 19, 17), .1 if veto else 1.),
                absolute_obstacle_clearance=np.full((2, 19, 17), np.inf))


def fake_interval(result):
    """Mock physics/geometry, but exercise the REAL new residual expansion."""
    def compare(scene, hypotheses, actions, *, observer_valid, position_residual,
                yaw_residual, paired_cost_residual):
        value = deepcopy(result)
        delta = value["reference_delta"]
        supported = value["cost_supported"] & value["common_support"]
        interval = np.full((2, 2), np.nan)
        interval[supported, 0] = delta[supported].min(axis=(1, 2))
        interval[supported, 1] = delta[supported].max(axis=(1, 2))
        interval[1:] += [-paired_cost_residual, paired_cost_residual]
        sign = np.where(interval[:, 1] < -.05, -1, np.where(interval[:, 0] > .05, 1, 0))
        value.update(delta_interval=interval, robust_sign=sign,
            abstain=~supported | value["veto"].any(axis=1) | (sign == 0),
            abstain_reasons=tuple(tuple(
                (["common_full_horizon_unsupported"] if not supported[i] else [])
                + (["road_margin"] if value["veto"][i].any() else [])
                + (["empirical_order_uncertain"] if sign[i] == 0 else [])) for i in range(2)))
        return value
    return Mock(side_effect=compare)


def make(frozen, *, config=None, real=False, never=False):
    champion = frozen.Agent() if real else Champion(frozen.shield.CollisionShieldAgent(Nominal()))
    wrapper = new.EnvelopeSuccessor(champion, calibration() if config is None else config,
                                    never_intervene=never)
    wrapper.observer = ValidObserver()
    wrapper._envelope_act.__func__.__globals__["extract_scene"] = Mock(return_value=object())
    return wrapper


def old_reference(champion, config, comparator):
    wrapper = successor.TemporalSuccessor(champion, config["absolute_calibration"])
    wrapper.observer = ValidObserver()
    original = successor.TemporalSuccessor.act
    namespace = dict(original.__globals__, compare_candidates=comparator,
                     extract_scene=Mock(return_value=object()))
    wrapper.act = MethodType(FunctionType(original.__code__, namespace), wrapper)
    return wrapper


def test_inherits_exact_lifecycle_and_binds_only_private_act_namespace(frozen):
    original = successor.TemporalSuccessor.act
    globals_before = dict(original.__globals__)
    wrapper, other = make(frozen), make(frozen)
    for name in ("reset", "_clear", "_record", "_calibration", "_motions", "_query", "last_step_diagnostics"):
        assert getattr(new.EnvelopeSuccessor, name) is getattr(successor.TemporalSuccessor, name)
    assert "act" not in vars(wrapper)
    assert wrapper.act.__func__ is new.EnvelopeSuccessor.act
    assert wrapper._envelope_act.__func__.__code__ is original.__code__
    assert wrapper._envelope_act.__self__ is wrapper
    assert wrapper._envelope_act.__func__.__globals__ is not original.__globals__
    assert wrapper._envelope_act.__func__.__globals__ is not other._envelope_act.__func__.__globals__
    assert original.__globals__.keys() == globals_before.keys()
    assert all(original.__globals__[key] is value for key, value in globals_before.items())


@pytest.mark.parametrize("never,support,upper", [(True, True, -.2), (False, False, -.2), (False, True, .1)])
def test_full_frozen_state_noop_parity_for_never_unknown_and_nonwinning(frozen, monkeypatch, never, support, upper):
    config = calibration()
    base_compare = fake_interval(template(upper, supported=support))
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", base_compare)
    wrapper = make(frozen, config=config, real=True, never=never)
    reference = old_reference(frozen.Agent(), config, base_compare)
    for step in range(12):
        obs = image(speed=76., center=44, box=step % 3 == 0)
        np.testing.assert_array_equal(wrapper.act(obs), reference.act(obs))
        equal(vars(wrapper.champion), vars(reference.champion))
        assert not wrapper.last_diagnostics["intervention"]
        assert wrapper.last_diagnostics["performance_method"] == new.PERFORMANCE_METHOD
        assert wrapper.last_diagnostics["performance_target"] == new.PERFORMANCE_TARGET
        json.dumps(wrapper.last_step_diagnostics(), allow_nan=False)
    assert wrapper.observer.commits == 12 and wrapper.observer.observations == 11
    assert wrapper.intervention_count == 0


def test_seventeen_synthetic_margins_cross_old_allowance_only(frozen, monkeypatch):
    """Seventeen constructed margins, NOT a replay of the historical 17 rows."""
    values = template()
    base_compare = fake_interval(values)
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", base_compare)
    wrapper = make(frozen)
    wrapper.act(image())
    for upper in np.linspace(-.12, -.3, 17):
        values["reference_delta"][1] = np.linspace(upper - .1, upper, 95).reshape(19, 5)
        before = wrapper.observer.snapshot()
        issued = wrapper.act(image())
        comparison = wrapper.last_diagnostics["comparison"]
        assert wrapper.last_diagnostics["intervention"]
        assert comparison["delta_interval"][1][1] < -.05
        assert comparison["old_delta_interval"][1][1] > -.05
        np.testing.assert_allclose(comparison["old_delta_interval"][1],
            [upper - .1 - .44961874671412616, upper + .44961874671412616])
        np.testing.assert_allclose(comparison["delta_interval"][1], [upper - .15, upper + .05])
        assert base_compare.call_args.kwargs["paired_cost_residual"] == 0.
        assert base_compare.call_args.args[2].shape == (2, 3)
        equal(before["state"], wrapper.observer.snapshot()["state"])
        np.testing.assert_array_equal(wrapper.observer._pending, issued)
        assert wrapper.nominal.brake_history[-1] == float(issued[2])
    assert base_compare.call_count == 17
    assert wrapper.intervention_count == 17
    assert wrapper.champion.queries == wrapper.observer.commits == 18


def test_new_gain_keeps_exact_frozen_feedback_and_followup_state(frozen, monkeypatch):
    values = template()
    base_compare = fake_interval(values)
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", base_compare)
    wrapper = make(frozen, real=True)
    reference = frozen.Agent()
    obs = image(speed=76., box=True)
    np.testing.assert_array_equal(wrapper.act(obs), reference.act(obs))
    prewheel, previous_boxes = wrapper.shield.wheel_angle, wrapper.shield.boxes.copy()
    issued, nominal = wrapper.act(obs), reference.act(obs)
    assert wrapper.last_diagnostics["intervention"]
    assert not np.array_equal(issued, nominal)
    expected = deepcopy(reference.driver)
    expected.driver.brake_history[-1] = float(issued[2])
    paths, wheels, hold = frozen.shield.project_paths([issued[0]], issued[0],
        expected.last_shield["projection_speed"], prewheel, actions=1)
    merged = np.concatenate((frozen.shield.detect_boxes(obs[-1]), previous_boxes))
    expected.boxes = frozen.shield.advance_boxes(merged, paths[0, hold])
    expected.wheel_angle = float(wheels[0])
    equal(vars(wrapper.shield), vars(expected))
    assert wrapper.observer.snapshot()["state"].wheel_angle == -.12
    assert wrapper.shield.wheel_angle > 0
    reference.driver = expected
    values["veto"][:] = True
    for has_box in (False, False, True):
        obs = image(speed=76., box=has_box)
        np.testing.assert_array_equal(wrapper.act(obs), reference.act(obs))
        equal(vars(wrapper.champion), vars(reference))
        assert not wrapper.last_diagnostics["intervention"]
        assert wrapper.last_diagnostics["action_history_check"]["nominal_brake_history_matches"]
    assert wrapper.observer.commits == 5 and wrapper.observer.observations == 4


@pytest.mark.parametrize("guard", ["veto", "absolute_supported", "common_support", "cost_supported"])
def test_performance_gain_cannot_bypass_unchanged_physical_or_support_guard(frozen, monkeypatch, guard):
    values = template()
    if guard == "veto":
        values[guard][1, 12] = True
    elif guard == "absolute_supported":
        values[guard][1, 12] = False
    elif guard == "common_support":
        values[guard] = False
        values["cost_supported"][:] = False
    else:
        values[guard][1] = False
    base_compare = fake_interval(values)
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", base_compare)
    wrapper = make(frozen)
    wrapper.act(image())
    np.testing.assert_array_equal(wrapper.act(image()), wrapper.nominal.action)
    assert wrapper.last_diagnostics["eligible"]
    assert not wrapper.last_diagnostics["intervention"]


def test_real_single_prediction_preserves_physical_arrays_and_components(frozen, monkeypatch):
    from tests.test_joint_interval_comparison import fixture

    wrapper = make(frozen)
    frames, motions = fixture()
    scene = interval_comparison.extract_scene(frames, motions=motions)
    hypotheses = ValidObserver().shared_hypotheses()
    actions = np.array([[.04, 0., .2], [0., .05, .15]], np.float32)
    old = interval_comparison.compare_candidates(scene, hypotheses, actions, observer_valid=True,
        **wrapper._calibration()[0])
    predictor = Mock(wraps=comparison.predict)
    monkeypatch.setattr(comparison, "predict", predictor)
    adapted = wrapper._envelope_act.__func__.__globals__["compare_candidates"]
    result = adapted(scene, hypotheses, actions, observer_valid=True, **wrapper._calibration()[0])
    assert predictor.call_count == 1
    for key in ("poses", "costs", "components", "reference_delta", "reference_costs", "veto",
                "absolute_supported", "absolute_road_clearance", "absolute_obstacle_clearance",
                "road_clearance", "obstacle_clearance", "supported_ticks"):
        equal(result[key], old[key])
    np.testing.assert_array_equal(result["old_delta_interval"], old["delta_interval"])
    controls = predictor.call_args.args[1]
    assert controls.shape == (38, 16, 3)
    np.testing.assert_array_equal(controls[:, 4:], np.broadcast_to(actions[0], (38, 12, 3)))
    assert tuple(vars(predictor.call_args.kwargs["parameters"]).values()) == (1., 1., 1.)


@pytest.mark.parametrize("bad", ["schema", "incomplete", "no_absolute", "absolute_incomplete",
    "absolute_schema", "absolute_floor", "absolute_cost", "mapping", "performance", "lower", "upper",
    "nan", "none", "tuple", "bool", "floor", "target"])
def test_calibration_validation_is_closed_before_query(frozen, bad):
    config = calibration()
    if bad == "schema": config["schema"] = "unknown"
    elif bad == "incomplete": config["calibration_complete"] = False
    elif bad == "no_absolute": del config["absolute_calibration"]
    elif bad == "absolute_incomplete": config["absolute_calibration"]["calibration_complete"] = False
    elif bad == "absolute_schema": config["absolute_calibration"]["schema"] = "unknown"
    elif bad == "absolute_floor": config["absolute_calibration"]["position_residual"][9] = .4
    elif bad == "absolute_cost": config["absolute_calibration"]["paired_cost_residual"] = None
    elif bad == "mapping": config["absolute_calibration"]["mapping"] = None
    elif bad == "performance": config["performance"] = None
    elif bad == "lower": config["performance"]["lower"] = .049
    elif bad == "upper": config["performance"]["upper"] = -.1
    elif bad == "nan": config["performance"]["upper"] = np.nan
    elif bad == "none": config["performance"]["lower"] = None
    elif bad == "tuple": config["performance"]["lower"] = (.05,)
    elif bad == "bool": config["performance"]["lower"] = True
    elif bad == "floor": config["performance"]["floor"] = 0.
    elif bad == "target": config["performance"]["target"] = "every prediction"
    champion = Champion(frozen.shield.CollisionShieldAgent(Nominal()))
    before = deepcopy(vars(champion.driver))
    with pytest.raises(ValueError):
        new.EnvelopeSuccessor(champion, config)
    assert champion.queries == 0
    equal(vars(champion.driver), before)


def test_tuple_arrays_copied_and_no_cross_instance_or_source_global_change(frozen, monkeypatch):
    config = calibration()
    for key in ("position_residual", "yaw_residual"):
        config["absolute_calibration"][key] = tuple(config["absolute_calibration"][key])
    original = deepcopy(config)
    base_compare = fake_interval(template())
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", base_compare)
    old_binding = successor.TemporalSuccessor.act.__globals__["compare_candidates"]
    first, other = make(frozen, config=config), make(frozen, config=config)
    config["performance"]["upper"] = 12.
    config["absolute_calibration"]["position_residual"] = (100.,) * 17
    for wrapper in (first, other):
        wrapper.act(image())
        wrapper.act(image())
        equal(wrapper.envelope_calibration, original)
        equal(wrapper.calibration, original["absolute_calibration"])
    assert first.calibration is first.envelope_calibration["absolute_calibration"]
    assert first.calibration is not other.calibration
    assert first.envelope_calibration["performance"] is not other.envelope_calibration["performance"]
    assert successor.TemporalSuccessor.act.__globals__["compare_candidates"] is old_binding


def test_budget_does_not_reduce_episode_or_change_candidate_direction(frozen, monkeypatch):
    base_compare = fake_interval(template())
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", base_compare)
    wrapper = make(frozen)
    for index in range(44):
        wrapper.nominal.action[0] = (.2, -.2, 0., .02)[index % 4]
        action = wrapper.act(image())
        if wrapper.last_diagnostics["intervention"]:
            steer = float(wrapper.nominal.action[0])
            expected = np.sign(steer) * max(0., abs(steer) - .04) if steer else .04
            assert action[0] == np.float32(expected)
            np.testing.assert_allclose(action[1:], [.1, .15], atol=1e-8)
    assert wrapper.intervention_count == base_compare.call_count == 40
    assert wrapper.champion.queries == wrapper.observer.commits == 44
    assert wrapper.observer.observations == 43
    assert wrapper.last_diagnostics["reasons"] == ["intervention_budget"]


def test_inherited_reset_and_exception_hook_restoration(frozen):
    wrapper = make(frozen)
    cloned = wrapper._envelope_act
    wrapper.act(image())
    wrapper.intervention_count = 40
    wrapper.reset(image())
    assert wrapper._envelope_act is cloned
    assert "act" not in vars(wrapper)
    assert wrapper.intervention_count == 0 and len(wrapper.frames) == 1
    wrapper.act(image())
    assert wrapper.observer.snapshot()["decision_index"] == 0
    hook = wrapper._shield_module.advance_boxes
    original_act = wrapper.shield.act.__func__
    wrapper.nominal.act = Mock(side_effect=RuntimeError("synthetic proposal failure"))
    with pytest.raises(RuntimeError, match="synthetic proposal failure"):
        wrapper.act(image())
    assert wrapper._shield_module.advance_boxes is hook
    assert wrapper.shield.act.__func__ is original_act
    assert "act" not in vars(wrapper.shield)
    assert wrapper.last_diagnostics["performance_method"] == new.PERFORMANCE_METHOD
    assert wrapper.last_diagnostics["act_cpu_seconds"] >= 0


def test_outer_timing_encloses_provenance_and_commit_not_inner_segment(frozen, monkeypatch):
    base_compare = fake_interval(template())
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", base_compare)
    wrapper = make(frozen)
    wrapper.act(image())
    calls = []

    def cpu():
        calls.append("cpu")
        if len(calls) > 2:
            assert wrapper.observer.commits == 2
            assert wrapper.last_diagnostics["performance_method"] == new.PERFORMANCE_METHOD
            return 7.
        return 2.

    def wall():
        calls.append("wall")
        return 11. if len(calls) > 2 else 3.

    monkeypatch.setattr(new, "process_time", cpu)
    monkeypatch.setattr(new, "perf_counter", wall)
    wrapper.act(image())
    assert calls == ["cpu", "wall", "cpu", "wall"]
    assert wrapper.last_diagnostics["act_cpu_seconds"] == 5.
    assert wrapper.last_diagnostics["act_wall_seconds"] == 8.
