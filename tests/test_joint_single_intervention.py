"""Synthetic B-branch causality and frozen champion replay; no simulator or data."""

from copy import deepcopy
import hashlib
import json
from unittest.mock import Mock

import numpy as np
import pytest

from haic.algorithms.joint_control import envelope_successor, paired_residual, successor
from haic.algorithms.joint_control import single_intervention as single
from tests.test_joint_envelope_successor import (
    ROOT, calibration, fake_interval, pinned_sources_unchanged, template,
)
from tests.test_joint_temporal_successor import Champion, Nominal, ValidObserver, equal, frozen, image


@pytest.fixture(scope="module", autouse=True)
def envelope_sources_unchanged():
    pins = {
        "envelope_successor.py": "4bb6bdf088f35dfb1f90f4abc77b8eb96ae4f7c418f013b9247df743277331d1",
        "paired_residual.py": "98c262128681634b02f2f942d9d104cbd1849a63e1a359ff8510fe69c336cd7c",
    }
    def hashes():
        return {name: hashlib.sha256((ROOT / "haic/algorithms/joint_control" / name).read_bytes()).hexdigest()
                for name in pins}
    assert hashes() == pins
    yield
    assert hashes() == pins


class FreshNominal(Nominal):
    def __init__(self):
        super().__init__()
        self.steps = 0

    def reset(self, observation=None):
        super().reset(observation)
        self.steps = 0


def candidate(action):
    result = np.array(action, dtype=np.float32, copy=True)
    steer, gas, brake = map(float, result)
    result[0] = np.sign(steer) * max(0., abs(steer) - .04) if steer else .04
    result[1:] = min(1., gas + .05), max(0., brake - .05)
    return result


def observe_with_mock_support(wrapper):
    wrapper.observer = ValidObserver()
    wrapper._envelope_act.__func__.__globals__["extract_scene"] = Mock(return_value=object())
    return wrapper


@pytest.fixture
def setup(frozen, monkeypatch):
    nominal = FreshNominal()
    champion = Champion(frozen.shield.CollisionShieldAgent(nominal))
    wrapper = observe_with_mock_support(single.SingleInterventionSuccessor(
        champion, calibration(), expected_action=candidate(nominal.action)))
    values = template()
    compare = fake_interval(values)
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", compare)
    return wrapper, compare, values


def prefix(wrapper, count=33):
    for _ in range(count):
        wrapper.act(image())


def test_exact_inherited_feedback_lifecycle_and_unchanged_c_budget(setup):
    wrapper, _, _ = setup
    for name in ("reset", "_query", "_record", "_motions", "last_step_diagnostics"):
        assert getattr(single.SingleInterventionSuccessor, name) is getattr(envelope_successor.EnvelopeSuccessor, name)
    assert wrapper._envelope_act.__func__.__code__ is successor.TemporalSuccessor.act.__code__
    assert "act" not in vars(wrapper)
    assert successor.MAX_INTERVENTIONS == 40
    assert wrapper.never_intervene and wrapper.steps == 0


def test_only_step34_queries_comparator_and_all_calls_commit_once(setup):
    wrapper, compare, _ = setup
    expected = wrapper.expected_action.copy()
    for step in range(1, 61):
        issued = wrapper.act(image())
        diag = wrapper.last_diagnostics
        assert wrapper.steps == wrapper.nominal.steps == step
        assert diag["decision_step"] == step and diag["decision_index"] == step - 1
        assert diag["one_shot_scheduled"] is (step == 34)
        assert diag["intervention"] is (step == 34)
        assert diag["one_shot_executed"] is (step >= 34)
        assert diag["action_returned"] and not diag["one_shot_failed"]
        assert all(diag["one_shot_checks"].values())
        assert wrapper.never_intervene
        target = expected if step == 34 else wrapper.nominal.action
        assert issued.tobytes() == target.tobytes()
        assert wrapper.observer._pending.astype(np.float32).tobytes() == issued.tobytes()
        assert wrapper.nominal.brake_history[-1] == float(issued[2])
        if step != 34:
            assert diag["comparison"] is None and diag["forecast"] is None
        else:
            assert np.asarray(diag["forecast"]["poses"]).shape == (19, 17, 3)
        json.dumps(wrapper.last_step_diagnostics(), allow_nan=False)
    assert compare.call_count == 1
    assert wrapper.champion.queries == wrapper.observer.commits == 60
    assert wrapper.observer.resets == 1 and wrapper.observer.observations == 59
    assert wrapper.intervention_count == 1


def test_full_frozen_prefix_equals_c_despite_skipped_pure_comparisons(frozen, monkeypatch):
    values = template(upper=.1)
    compare = fake_interval(values)
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", compare)
    wrapper = observe_with_mock_support(single.SingleInterventionSuccessor(
        frozen.Agent(), calibration(), expected_action=np.array([0., .05, .15], np.float32)))
    c = observe_with_mock_support(envelope_successor.EnvelopeSuccessor(frozen.Agent(), calibration()))
    before = deepcopy(wrapper.envelope_calibration)
    for step in range(1, 34):
        obs = image(speed=76., center=44, box=step % 3 == 0)
        assert wrapper.act(obs).tobytes() == c.act(obs).tobytes()
        equal(vars(wrapper.champion), vars(c.champion))
        equal(wrapper.snapshots, c.snapshots)
        equal(wrapper.frames, c.frames)
        equal(wrapper.issued_actions, c.issued_actions)
        equal(wrapper.observer.snapshot(), c.observer.snapshot())
        equal(wrapper.observer._pending, c.observer._pending)
        assert wrapper.last_diagnostics["comparison"] is None
    assert compare.call_count > 0  # C actually evaluated pure, nonwinning proposals.
    assert wrapper.observer.hypotheses == 0 and c.observer.hypotheses > 0
    equal(wrapper.envelope_calibration, before)
    assert c.never_intervene is False and successor.MAX_INTERVENTIONS == 40


def test_frozen_c34_feedback_and_fresh_35_36_observations(frozen, monkeypatch):
    """All images/actions are synthetic, not A/C records or saved future actions."""
    obs = image(speed=76., center=44, box=True)
    # Independent in-memory champion supplies a synthetic expected C34 action.
    nominal_reference = frozen.Agent()
    for _ in range(33):
        nominal_reference.act(obs)
    expected = candidate(nominal_reference.act(obs))
    wrapper = observe_with_mock_support(single.SingleInterventionSuccessor(
        frozen.Agent(), calibration(), expected_action=expected))
    c = observe_with_mock_support(envelope_successor.EnvelopeSuccessor(frozen.Agent(), calibration()))
    values = template(upper=.1)
    compare = fake_interval(values)
    monkeypatch.setattr(paired_residual.interval_comparison, "compare_candidates", compare)
    for _ in range(33):
        assert wrapper.act(obs).tobytes() == c.act(obs).tobytes()
    prewheel, old_boxes = wrapper.shield.wheel_angle, wrapper.shield.boxes.copy()
    values["reference_delta"][1] = -.2
    issued = wrapper.act(obs)
    assert issued.tobytes() == c.act(obs).tobytes() == expected.tobytes()
    equal(vars(wrapper.champion), vars(c.champion))
    assert wrapper.nominal.brake_history[-1] == float(issued[2])
    paths, wheels, hold = frozen.shield.project_paths([issued[0]], issued[0],
        wrapper.shield.last_shield["projection_speed"], prewheel, actions=1)
    # Detect+retain the exact unadvanced boxes: current synthetic pixels do not
    # move with predicted ego translation, so older off-road boxes remain hidden.
    nominal_reference.driver = deepcopy(c.shield)
    assert wrapper.shield.wheel_angle == float(wheels[0])
    assert wrapper.shield.boxes.shape[0] > 1
    assert not np.array_equal(wrapper.shield.boxes[:len(old_boxes)], old_boxes)
    assert wrapper.observer.snapshot()["state"].wheel_angle == -.12
    after34 = deepcopy(nominal_reference.driver)
    compare.reset_mock()
    compare.side_effect = AssertionError("B must not compare again after C34")
    continuation = []
    for following in (image(speed=55., center=48, box=False), image(speed=80., center=38, box=True)):
        actual = wrapper.act(following)
        expected_next = nominal_reference.act(following)
        assert actual.tobytes() == expected_next.tobytes()
        equal(vars(wrapper.champion), vars(nominal_reference))
        assert not wrapper.last_diagnostics["intervention"]
        continuation.append(actual)
    # A stale frame really would produce a different next action in this fixture.
    stale = frozen.Agent()
    stale.driver = after34
    assert stale.act(obs).tobytes() != continuation[0].tobytes()
    compare.assert_not_called()
    assert wrapper.steps == 36 and wrapper.intervention_count == 1
    assert wrapper.observer.commits == 36 and wrapper.observer.observations == 35


def test_retained_occluded_boxes_use_actual_steering_not_nominal(setup, frozen):
    wrapper, _, _ = setup
    prefix(wrapper)
    wrapper.shield.boxes = np.array([[-18., 8., -16., 10.]])
    wrapper.shield.box_ages = np.array([0])
    wrapper.shield.box_threatened = np.array([False])
    before, prewheel = wrapper.shield.boxes.copy(), wrapper.shield.wheel_angle
    inactive = deepcopy(vars(wrapper.nominal.base))
    action = wrapper.act(image(box=True))
    merged = np.concatenate((frozen.shield.detect_boxes(image(box=True)[-1]), before))
    poses, wheels, hold = frozen.shield.project_paths([action[0]], action[0],
        wrapper.shield.last_shield["projection_speed"], prewheel, actions=1)
    equal(wrapper.shield.boxes, frozen.shield.advance_boxes(merged, poses[0, hold]))
    assert wrapper.shield.wheel_angle == float(wheels[0])
    equal(vars(wrapper.nominal.base), inactive)
    assert wrapper.shield.box_ages.tolist() == [0, 1]
    assert wrapper.observer._pending.astype(np.float32).tobytes() == action.tobytes()


@pytest.mark.parametrize("failure", ["abstain", "veto", "mapping", "observer", "wrong_action", "signed_zero", "scorer"])
def test_c34_failure_raises_before_action_return_and_cannot_retry(setup, failure):
    wrapper, compare, values = setup
    prefix(wrapper)
    if failure == "abstain": values["reference_delta"][1] = .1
    elif failure == "veto": values["veto"][1, 0] = True
    elif failure == "mapping": wrapper.observer.provenance = "unseen"
    elif failure == "observer": wrapper.observer.valid = False
    elif failure == "wrong_action": wrapper.nominal.action[2] = np.float32(.25)
    elif failure == "signed_zero":
        wrapper.nominal.action[0] = -.02
        wrapper.expected_action = candidate(wrapper.nominal.action)
        wrapper.expected_action[0] = 0.  # Must not accept +0 for the issued -0.
    elif failure == "scorer": compare.side_effect = ValueError("synthetic unsupported comparison")
    with pytest.raises(RuntimeError, match="C34 intervention not reproduced"):
        wrapper.act(image())
    assert wrapper.steps == 33 and not wrapper.one_shot_executed
    assert wrapper.nominal.steps == 34 and wrapper.champion.queries == 34
    assert wrapper.last_diagnostics["one_shot_scheduled"]
    assert wrapper.last_diagnostics["one_shot_failed"]
    assert not wrapper.last_diagnostics["action_returned"]
    assert not all(wrapper.last_diagnostics["one_shot_checks"].values())
    assert wrapper.never_intervene
    commits, calls = wrapper.observer.commits, compare.call_count
    with pytest.raises(RuntimeError, match="failure is latched"):
        wrapper.act(image())
    assert wrapper.champion.queries == 34 and wrapper.observer.commits == commits
    assert compare.call_count == calls
    json.dumps(wrapper.last_step_diagnostics(), allow_nan=False)


def test_no_extra_query_or_commit_on_preindex_mismatch(setup):
    wrapper, compare, _ = setup
    wrapper.nominal.steps = 1
    with pytest.raises(RuntimeError, match="pre-action index/state mismatch"):
        wrapper.act(image())
    assert wrapper.champion.queries == wrapper.observer.commits == 0
    compare.assert_not_called()


def test_scheduled_query_exception_restores_hook_and_latches(setup):
    wrapper, _, _ = setup
    prefix(wrapper)
    advance = wrapper._shield_module.advance_boxes
    act = wrapper.shield.act.__func__
    wrapper.nominal.act = Mock(side_effect=RuntimeError("synthetic champion failure"))
    with pytest.raises(RuntimeError, match="synthetic champion failure"):
        wrapper.act(image())
    assert wrapper._shield_module.advance_boxes is advance
    assert wrapper.shield.act.__func__ is act and "act" not in vars(wrapper.shield)
    assert wrapper.never_intervene and wrapper.last_diagnostics["one_shot_failed"]
    assert wrapper.observer.commits == 33


@pytest.mark.parametrize("failed", [False, True])
def test_inherited_reset_clears_used_counter_and_failure_latch(setup, failed):
    wrapper, _, values = setup
    prefix(wrapper)
    if failed:
        values["veto"][1, 0] = True
        with pytest.raises(RuntimeError): wrapper.act(image())
    else:
        wrapper.act(image())
        assert wrapper.one_shot_executed
    binding = wrapper._envelope_act
    wrapper.reset(image())  # Agent/observer only, never a simulator reset.
    assert wrapper.steps == wrapper.nominal.steps == wrapper.intervention_count == 0
    assert not wrapper.one_shot_executed and not wrapper._one_shot_failed
    assert wrapper.never_intervene and wrapper._envelope_act is binding
    wrapper.act(image())
    assert wrapper.steps == 1 and wrapper.last_diagnostics["decision_index"] == 0
    assert not wrapper.last_diagnostics["one_shot_scheduled"]


@pytest.mark.parametrize("step", [-1, 0, 33, 35, 34., True, "34"])
def test_only_integer34_is_authorized(frozen, step):
    with pytest.raises(ValueError, match="integer 34"):
        single.SingleInterventionSuccessor(frozen.Agent(), calibration(),
            expected_action=np.array([0., .05, .15], np.float32), intervention_step=step)


@pytest.mark.parametrize("action", [[0., .05], [np.nan, .05, .15], [0., np.inf, .15],
    [1.000001, .05, .15], [0., -.01, .15], [0., .05, 1.1], [[0., .05, .15]], None])
def test_expected_action_has_strict_shape_finiteness_and_bounds(frozen, action):
    with pytest.raises(ValueError):
        single.SingleInterventionSuccessor(frozen.Agent(), calibration(), expected_action=action)


def test_expected_action_is_copied_float32_and_readonly(frozen):
    expected = np.array([-.0, .05, .15], np.float32)
    wrapper = single.SingleInterventionSuccessor(frozen.Agent(), calibration(), expected_action=expected.tolist())
    assert wrapper.expected_action.tobytes() == expected.tobytes()
    expected[:] = 1
    assert wrapper.expected_action.tolist() != expected.tolist()
    assert not wrapper.expected_action.flags.writeable


def test_outer_timers_include_schedule_validation_and_diagnostics(setup, monkeypatch):
    wrapper, _, _ = setup
    prefix(wrapper)
    calls = []
    def cpu():
        calls.append("cpu")
        if len(calls) > 2:
            assert wrapper.last_diagnostics["action_returned"]
            assert all(wrapper.last_diagnostics["one_shot_checks"].values())
            assert wrapper.observer.commits == 34
            return 9.
        return 2.
    def wall():
        calls.append("wall")
        return 15. if len(calls) > 2 else 3.
    monkeypatch.setattr(single, "process_time", cpu)
    monkeypatch.setattr(single, "perf_counter", wall)
    wrapper.act(image())
    assert calls == ["cpu", "wall", "cpu", "wall"]
    assert wrapper.last_diagnostics["act_cpu_seconds"] == 7.
    assert wrapper.last_diagnostics["act_wall_seconds"] == 12.
