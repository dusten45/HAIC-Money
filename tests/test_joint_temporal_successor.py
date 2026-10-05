"""Synthetic pixels, frozen Agent state replay and mock ranking; no simulator."""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from haic.algorithms.joint_control import successor
from haic.algorithms.joint_control.physics import PhysicsState


FROZEN = (Path(__file__).resolve().parents[1] / "submissions" /
          "20261002-crossing-projection-collision-shield-v1-baseline/source")


@pytest.fixture(scope="module")
def frozen():
    """Execute exact frozen source bytes; isolate imports and never write pyc."""
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in FROZEN.rglob("*.py")}
    names = [name for name in sys.modules if name == "haic_agent" or name.startswith("haic_agent.")]
    saved = {name: sys.modules.pop(name) for name in names}
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    sys.path.insert(0, str(FROZEN))
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("_successor_frozen_agent", FROZEN / "agent.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        shield = sys.modules["haic_agent.collision_shield_runtime"]
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        for name in list(sys.modules):
            if name == "haic_agent" or name.startswith("haic_agent."):
                del sys.modules[name]
        sys.modules.update(saved)
    yield SimpleNamespace(Agent=module.Agent, shield=shield)
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in FROZEN.rglob("*.py")}


def calibration():
    return dict(schema=successor.CALIBRATION_SCHEMA, calibration_complete=True,
                position_residual=[0.] + [.5] * 16, yaw_residual=[0.] + [.05] * 16,
                paired_cost_residual=.125,
                mapping=dict(by_provenance={name: dict(supported=True, n_cal=3,
                    position_residual=.25, yaw_residual=.025) for name in (
                        "measured_image", "predicted_corrected_issued_action")}))


def image(speed=60., center=44, box=False, road=True):
    frame = np.full((84, 84), .1, dtype=np.float32)
    if road:
        frame[8:73, center - 12:center + 13] = .4
    frame[74:] = 0.
    frame[77:83, 10:13] = (speed * .085 + .27) / 18
    if box:
        frame[42:46, 64:68] = .9
    return np.repeat(frame[None], 4, axis=0)


def equal(left, right):
    """Compare complete recursively owned state, including dtype and array bits."""
    assert type(left) is type(right)
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif is_dataclass(left):
        equal(asdict(left), asdict(right))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple, deque)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    elif hasattr(left, "__dict__"):
        equal(vars(left), vars(right))
    elif isinstance(left, float) and np.isnan(left):
        assert np.isnan(right)
    else:
        assert left == right


class ValidObserver:
    """Explicit valid snapshots, without implying synthetic pixels prove validity."""

    def __init__(self):
        self.resets = self.observations = self.commits = self.hypotheses = 0
        self._pending = None
        self.valid = True
        self.provenance = "measured_image"
        self.forward = 60.
        self.invalid_mapping = False
        self.events = []

    def reset(self, observation):
        self.resets += 1
        self.events.append("reset")
        self._pending = None
        self.index = 0
        return self.snapshot()

    def observe(self, observation):
        assert self._pending is not None
        self.observations += 1
        self.events.append("observe")
        self._pending = None
        self.index += 1
        return self.snapshot()

    def snapshot(self):
        return dict(valid=self.valid, decision_index=self.index, invalid_reasons=(),
                    state=PhysicsState(self.forward, wheel_angle=-.12, wheel_omega=np.ones(4)),
                    mapping_motion=dict(valid=not self.invalid_mapping,
                        provenance=self.provenance, right=0., forward=4.8, yaw_delta=.01,
                        residual_p90=.02, position_uncertainty=.1, yaw_uncertainty=.01))

    def shared_hypotheses(self):
        assert self._pending is None
        self.hypotheses += 1
        self.events.append("hypotheses")
        return PhysicsState(np.full(19, self.forward), wheel_angle=-.12,
                            wheel_omega=np.full((19, 4), 100.))

    def commit_action(self, action):
        assert self._pending is None
        assert action.dtype == np.float32
        self.commits += 1
        self.events.append("commit")
        self._pending = action.astype(float).copy()


class Nominal:
    contact_mode = "crossing_projection"

    def __init__(self):
        self.steps = 11
        self.brake_history = []
        self.action = np.array([.2, .05, .2], np.float32)
        self.speed = 60.
        self.base = SimpleNamespace(_frame=lambda obs: obs[-1],
            _estimate_speed=lambda frame: self.speed,
            _last_gas=.81, _last_brake=.02, _gas_frames=18, _brake_frames=9)
        self.last_command, self.visible_steer = .72, -.63
        self.recovering = False
        self.extra_info = {}

    def act(self, observation):
        self.steps += 1
        self.brake_history = (self.brake_history + [float(self.action[2])])[-4:]
        return self.action.copy()

    def reset(self, observation=None):
        self.brake_history = []

    def last_step_diagnostics(self):
        return dict(pixel_speed=self.speed, road_centers={30: 42., 42: 42., 54: 42.}, **self.extra_info)


class Champion:
    def __init__(self, shield):
        self.driver = shield
        self.queries = 0

    def act(self, observation):
        self.queries += 1
        return self.driver.act(observation)

    def reset(self, observation=None):
        self.driver.reset(observation)


def ranking(**changes):
    result = dict(common_support=True, cost_supported=np.ones(2, bool),
                  delta_interval=np.array([[0., 0.], [-.4, -.2]]),
                  absolute_supported=np.ones((2, 19), bool), veto=np.zeros((2, 19), bool),
                  abstain_reasons=((), ()), poses=np.zeros((2, 19, 17, 3)),
                  components=dict(progress=np.ones((2, 19))),
                  absolute_road_clearance=np.full((2, 19, 17), 1.))
    result.update(changes)
    return result


@pytest.fixture
def setup(frozen, monkeypatch):
    observer = ValidObserver()
    monkeypatch.setattr(successor, "TemporalObserver", lambda **kwargs: observer)
    champion = Champion(frozen.shield.CollisionShieldAgent(Nominal()))
    wrapper = successor.TemporalSuccessor(champion, calibration())
    compare = Mock(return_value=ranking())
    scene = Mock(return_value=object())
    monkeypatch.setattr(successor, "compare_candidates", compare)
    monkeypatch.setattr(successor, "extract_scene", scene)
    wrapper.act(image())
    assert compare.call_count == 0  # No invented preceding mapping on reset.
    return wrapper, champion, observer, compare, scene


@pytest.mark.parametrize("never", [True, False])
def test_complete_frozen_champion_noop_parity_with_occlusion_and_invalid_projection(frozen, never):
    wrapped, reference = frozen.Agent(), frozen.Agent()
    before = deepcopy(vars(wrapped))
    wrapper = successor.TemporalSuccessor(wrapped, None, never_intervene=never)
    equal(vars(wrapped), before)
    seen = set()
    for step in range(30):
        obs = image(speed=(0., 60., 76., 80.)[(step // 6) % 4],
                    center=42 + step % 3, box=step % 5 < 2, road=step % 7 != 0)
        expected = reference.act(obs)
        np.testing.assert_array_equal(wrapper.act(obs), expected)
        equal(vars(wrapped), vars(reference))
        assert "act" not in vars(wrapped.driver)
        assert wrapper.last_diagnostics["action_history_check"]["nominal_brake_history_matches"]
        json.dumps(wrapper.last_diagnostics, allow_nan=False)
        seen.add(wrapped.driver.last_shield["reason"])
    assert "invalid_projection" in seen
    assert seen & {"baseline_clear", "no_visible_obstacle"}
    assert wrapper.observer.snapshot()["decision_index"] == 29
    assert wrapper.intervention_count == 0


def test_forced_joint_feedback_is_actual_action_and_unadvanced_box_propagation(setup, frozen):
    wrapper, champion, observer, compare, scene = setup
    nominal, shield = wrapper.nominal, wrapper.shield
    nominal.base._estimate_speed = lambda frame: 75.
    prewheel = shield.wheel_angle
    inactive = (nominal.last_command, nominal.visible_steer, deepcopy(vars(nominal.base)))
    # Include a retained box which is not detected in the current frame.
    shield.boxes = np.array([[-18., 8., -16., 10.]])
    shield.box_ages = np.array([0])
    shield.box_threatened = np.array([False])
    obs = image(box=True)
    merged = np.concatenate((frozen.shield.detect_boxes(obs[-1]), shield.boxes.copy()))
    before = observer.snapshot()
    issued = wrapper.act(obs)
    np.testing.assert_allclose(issued, [.16, .1, .15], atol=1e-8)
    assert issued.dtype == np.float32
    assert observer.resets == 1 and observer.observations == 1 and observer.commits == 2
    assert champion.queries == 2 and compare.call_count == 1 and scene.call_count == 1
    np.testing.assert_array_equal(observer._pending, issued)
    assert nominal.brake_history[-1] == float(issued[2])
    assert len(nominal.brake_history) == 2
    paths, wheels, hold = frozen.shield.project_paths([issued[0]], issued[0],
        shield.last_shield["projection_speed"], prewheel, actions=1)
    np.testing.assert_array_equal(shield.boxes, frozen.shield.advance_boxes(merged, paths[0, hold]))
    assert shield.wheel_angle == wheels[0]
    assert shield.last_shield["projection_speed"] == 75.
    assert shield.wheel_angle > 0 and observer.snapshot()["state"].wheel_angle == -.12
    equal(observer.snapshot()["state"], before["state"])
    assert (nominal.last_command, nominal.visible_steer) == inactive[:2]
    equal(vars(nominal.base), inactive[2])
    np.testing.assert_array_equal(shield.box_ages, [0, 1])
    assert shield.encounter_actions == 0 and not shield.encounter_open
    diagnostics = wrapper.last_diagnostics
    assert diagnostics["eligible"] and diagnostics["intervention"]
    assert diagnostics["reasons"] == []
    assert diagnostics["action_history_check"]["shield_boxes_match"]
    assert diagnostics["action_history_check"]["shield_wheel_matches"]
    assert diagnostics["candidate_count"] == 2
    assert np.asarray(diagnostics["forecast"]["poses"]).shape == (19, 17, 3)
    np.testing.assert_array_equal(diagnostics["forecast"]["actions"][0], issued)
    np.testing.assert_array_equal(diagnostics["forecast"]["actions"][1:], [nominal.action] * 3)
    equal(compare.call_args.kwargs["position_residual"], np.array(calibration()["position_residual"]))
    assert compare.call_args.kwargs["paired_cost_residual"] == .125
    json.dumps(diagnostics, allow_nan=False)


def test_forced_intervention_on_actual_frozen_champion_changes_only_active_feedback(frozen, monkeypatch):
    observer = ValidObserver()
    monkeypatch.setattr(successor, "TemporalObserver", lambda **kwargs: observer)
    champion, reference = frozen.Agent(), frozen.Agent()
    wrapper = successor.TemporalSuccessor(champion, calibration())
    monkeypatch.setattr(successor, "extract_scene", lambda *args, **kwargs: object())
    compare = Mock(return_value=ranking())
    monkeypatch.setattr(successor, "compare_candidates", compare)
    obs = image(speed=76., center=44, box=True)
    np.testing.assert_array_equal(wrapper.act(obs), reference.act(obs))
    prewheel = champion.driver.wheel_angle
    previous_boxes = champion.driver.boxes.copy()
    issued = wrapper.act(obs)
    nominal_action = reference.act(obs)
    assert wrapper.last_diagnostics["intervention"], wrapper.last_diagnostics["reasons"]
    assert not np.array_equal(issued, nominal_action)
    expected = deepcopy(vars(reference))
    expected_shield = expected["driver"]
    expected_shield.driver.brake_history[-1] = float(issued[2])
    paths, wheels, hold = frozen.shield.project_paths([issued[0]], issued[0],
        reference.driver.last_shield["projection_speed"], prewheel, actions=1)
    # The static synthetic image has not moved with the predicted ego pose, so
    # its old box is too far away to associate and remains in occlusion memory.
    merged = np.concatenate((frozen.shield.detect_boxes(obs[-1]), previous_boxes))
    expected_shield.boxes = frozen.shield.advance_boxes(merged, paths[0, hold])
    expected_shield.wheel_angle = float(wheels[0])
    equal(vars(champion), expected)
    assert compare.call_count == 1
    reference.driver = expected_shield
    compare.return_value = ranking(delta_interval=np.array([[0., 0.], [-.1, .1]]))
    for box in (False, False, True):
        followup = image(speed=76., center=44, box=box)
        np.testing.assert_array_equal(wrapper.act(followup), reference.act(followup))
        equal(vars(champion), vars(reference))
        assert not wrapper.last_diagnostics["intervention"]
        assert wrapper.last_diagnostics["action_history_check"]["nominal_brake_history_matches"]
    assert wrapper.intervention_count == 1
    assert observer.commits == 5 and observer.observations == 4


def test_computed_noop_preserves_complete_frozen_state(frozen, monkeypatch):
    observer = ValidObserver()
    monkeypatch.setattr(successor, "TemporalObserver", lambda **kwargs: observer)
    champion, reference = frozen.Agent(), frozen.Agent()
    wrapper = successor.TemporalSuccessor(champion, calibration())
    monkeypatch.setattr(successor, "extract_scene", lambda *args, **kwargs: object())
    compare = Mock(return_value=ranking(delta_interval=np.array([[0., 0.], [-.1, .1]])))
    monkeypatch.setattr(successor, "compare_candidates", compare)
    for _ in range(6):
        obs = image(speed=76., box=True)
        np.testing.assert_array_equal(wrapper.act(obs), reference.act(obs))
        equal(vars(champion), vars(reference))
    assert compare.call_count == 5
    assert wrapper.intervention_count == 0
    assert observer.commits == 6 and observer.observations == 5


@pytest.mark.parametrize("steer,expected", [(.2, .16), (-.2, -.16), (0., .04), (.02, 0.), (-.02, 0.)])
def test_one_fixed_candidate_direction(setup, steer, expected):
    wrapper, _, _, compare, _ = setup
    wrapper.nominal.action[0] = steer
    result = wrapper.act(image())
    assert result[0] == pytest.approx(expected)
    actions = compare.call_args.args[2]
    assert actions.shape == (2, 3)
    np.testing.assert_array_equal(actions[0], wrapper.nominal.action)
    np.testing.assert_array_equal(actions[1], result)


@pytest.mark.parametrize("field,value", [
    ("common_support", False),
    ("cost_supported", np.array([True, False])),
    ("delta_interval", np.array([[0., 0.], [-.3, -.05]])),
    ("delta_interval", np.array([[0., 0.], [-.3, np.nan]])),
    ("absolute_supported", np.array([[True] * 19, [False] + [True] * 18])),
    ("veto", np.array([[False] * 19, [True] + [False] * 18])),
])
def test_robust_full_cost_absolute_and_margin_veto_fallback(setup, field, value):
    wrapper, _, _, compare, _ = setup
    compare.return_value[field] = value
    issued = wrapper.act(image())
    np.testing.assert_array_equal(issued, wrapper.nominal.action)
    assert wrapper.last_diagnostics["eligible"]
    assert not wrapper.last_diagnostics["intervention"]
    assert wrapper.last_diagnostics["reasons"]
    json.dumps(wrapper.last_diagnostics, allow_nan=False)


@pytest.mark.parametrize("change", ["missing", "incomplete", "null_cost", "nan_cost", "floor",
                                   "unseen", "unsupported", "null_mapping", "invalid_mapping"])
def test_missing_calibration_and_mapping_support_never_zero_default(setup, change):
    wrapper, _, observer, compare, _ = setup
    if change == "missing":
        wrapper.calibration = {}
    elif change == "incomplete":
        wrapper.calibration["calibration_complete"] = False
    elif change == "null_cost":
        wrapper.calibration["paired_cost_residual"] = None
    elif change == "nan_cost":
        wrapper.calibration["paired_cost_residual"] = np.nan
    elif change == "floor":
        wrapper.calibration["position_residual"][4] = 0.
    elif change == "unseen":
        observer.provenance = "unseen"
    elif change == "unsupported":
        wrapper.calibration["mapping"]["by_provenance"]["measured_image"]["supported"] = False
    elif change == "null_mapping":
        wrapper.calibration["mapping"]["by_provenance"]["measured_image"]["yaw_residual"] = None
    elif change == "invalid_mapping":
        observer.invalid_mapping = True
    np.testing.assert_array_equal(wrapper.act(image()), wrapper.nominal.action)
    assert not wrapper.last_diagnostics["eligible"]
    compare.assert_not_called()
    assert wrapper.last_diagnostics["reasons"]


def test_mapping_widening_causal_last_four_and_no_observer_mutation(setup):
    wrapper, _, observer, compare, scene = setup
    wrapper.calibration["mapping"]["by_provenance"]["measured_image"]["position_residual"] = .75
    for index in range(1, 7):
        obs = image()
        obs[-1, 0, 0] = index / 10
        wrapper.act(obs)
    used_frames = scene.call_args.args[0]
    np.testing.assert_array_equal(used_frames[:, 0, 0], np.array([.3, .4, .5, .6], np.float32))
    assert len(wrapper.frames) == len(wrapper.snapshots) == 4
    assert all(s["mapping_motion"]["position_uncertainty"] == .1 for s in wrapper.snapshots)
    assert observer.snapshot()["mapping_motion"]["position_uncertainty"] == .1
    assert all(m["position_uncertainty"] == .75 for m in scene.call_args.kwargs["motions"])
    assert all(m["yaw_uncertainty"] == .025 for m in scene.call_args.kwargs["motions"])
    assert observer.resets == 1 and observer.observations == 6 and observer.commits == 7
    assert compare.call_count == 6


def test_older_unsupported_link_keeps_newer_calibrated_suffix_without_invented_history(setup, monkeypatch):
    from haic.algorithms.joint_control.interval_comparison import extract_scene

    wrapper, _, observer, compare, _ = setup
    scene = Mock(wraps=extract_scene)
    monkeypatch.setattr(successor, "extract_scene", scene)
    observer.provenance = "unseen_old_link"
    wrapper.act(image())
    compare.assert_not_called()
    observer.provenance = "measured_image"
    wrapper.act(image())
    assert wrapper.last_diagnostics["eligible"]
    assert compare.call_args.args[0].history_used == 1
    assert compare.call_args.args[0].motion_provenance == ("measured_image",)
    motions = scene.call_args.kwargs["motions"]
    assert not motions[0]["valid"] and motions[1]["valid"]
    assert wrapper.snapshots[1]["mapping_motion"]["valid"]  # Only external copy invalidated.
    wrapper.act(image())
    assert compare.call_args.args[0].history_used == 2
    assert compare.call_args.args[0].motion_provenance == ("measured_image",) * 2


def test_missing_unrelated_mapping_class_does_not_globally_block_supported_class(setup):
    wrapper, _, _, compare, _ = setup
    del wrapper.calibration["mapping"]["by_provenance"]["predicted_corrected_issued_action"]
    wrapper.act(image())
    assert wrapper.last_diagnostics["intervention"]
    assert compare.call_count == 1


@pytest.mark.parametrize("cause", ["speed_low", "speed_high", "reverse", "invalid_observer",
                                   "steer", "gas", "brake_low", "brake_high",
                                   "blocked", "encounter", "threat", "recovery", "impact", "contact"])
def test_runtime_envelope_and_champion_guard_abstentions(setup, cause):
    wrapper, _, observer, compare, _ = setup
    if cause == "speed_low":
        wrapper.nominal.speed = 39.
    elif cause == "speed_high":
        wrapper.nominal.speed = 81.
    elif cause == "reverse":
        observer.forward = -60.
    elif cause == "invalid_observer":
        observer.valid = False
    elif cause == "steer":
        wrapper.nominal.action[0] = .36
    elif cause == "gas":
        wrapper.nominal.action[1] = .11
    elif cause == "brake_low":
        wrapper.nominal.action[2] = .019
    elif cause == "brake_high":
        wrapper.nominal.action[2] = .51
    elif cause == "blocked":
        wrapper.shield.rearm_blocked = True
    elif cause == "encounter":
        wrapper.shield.encounter_open = True
    elif cause == "threat":
        wrapper._shield_module.__dict__["footprint_gaps"] = lambda paths, boxes: np.full(len(paths), -1.)
    elif cause == "recovery":
        wrapper.nominal.extra_info["recovery_changed"] = True
    elif cause == "impact":
        wrapper.nominal.impact_left = 1
    elif cause == "contact":
        wrapper.nominal.extra_info["contact_active"] = True
    wrapper.act(image(box=cause == "threat"))
    assert not wrapper.last_diagnostics["eligible"]
    assert not wrapper.last_diagnostics["intervention"]
    compare.assert_not_called()


def test_float32_boundary_envelope_and_pedal_clipping(setup):
    wrapper, _, _, compare, _ = setup
    wrapper.nominal.action[:] = [.35, .1, .02]
    issued = wrapper.act(image())
    assert wrapper.last_diagnostics["intervention"]
    np.testing.assert_array_equal(issued, np.array([.31, .15, 0.], np.float32))
    assert compare.call_count == 1


def test_active_shield_intervention_and_encounter_budget_are_not_replaced(setup):
    wrapper, _, observer, compare, _ = setup

    def one_safe(paths, boxes):
        gaps = np.full(len(paths), -1.)
        if len(paths) > 1:
            gaps[-1] = 1.
        return gaps

    wrapper._shield_module.__dict__["footprint_gaps"] = one_safe
    action = wrapper.act(image(box=True))
    assert wrapper.shield.last_shield["active"]
    assert wrapper.shield.last_shield["baseline_threat"]
    assert wrapper.shield.encounter_actions == 1
    assert wrapper.intervention_count == 0
    assert action[0] == np.float32(.4)
    np.testing.assert_array_equal(observer._pending, action)
    np.testing.assert_array_equal(action[1:], wrapper.nominal.action[1:])
    compare.assert_not_called()


def test_real_geometry_and_comparator_use_fixed_full_horizon_and_calibration(setup, monkeypatch):
    from haic.algorithms.joint_control import comparison, interval_comparison
    from tests.test_joint_temporal_comparison import road_frames

    wrapper, _, _, _, _ = setup
    monkeypatch.setattr(successor, "extract_scene", interval_comparison.extract_scene)
    monkeypatch.setattr(successor, "compare_candidates", interval_comparison.compare_candidates)
    frames, _ = road_frames(advance=4.8)
    wrapper.nominal.action[0] = .04
    with patch.object(comparison, "predict", wraps=comparison.predict) as model:
        for frame in frames[1:]:
            wrapper.act(np.repeat(frame[None], 4, axis=0))
    diagnostics = wrapper.last_diagnostics
    result = diagnostics["comparison"]
    assert result is not None, diagnostics["reasons"]
    assert np.asarray(result["poses"]).shape == (2, 19, 17, 3)
    state, actions, dt = model.call_args.args
    assert dt == .02 and actions.shape == (38, 16, 3)
    assert asdict(model.call_args.kwargs["parameters"]) == dict(
        mass_scale=1., yaw_inertia_scale=1., tire_stiffness_scale=1.)
    np.testing.assert_array_equal(actions[:, 4:], np.broadcast_to(wrapper.nominal.action, (38, 12, 3)))
    for key in asdict(state):
        value = np.asarray(getattr(state, key))
        np.testing.assert_array_equal(value[:19], value[19:])
    assert diagnostics["forecast"]["position_residual"] == calibration()["position_residual"]
    delta = np.asarray(result["reference_delta"], dtype=float)
    if np.isfinite(delta).all():
        np.testing.assert_allclose(result["delta_interval"][1],
            [delta[1].min() - .125, delta[1].max() + .125])
    json.dumps(diagnostics, allow_nan=False)


def test_forty_interventions_do_not_truncate_episode_or_observer(setup):
    wrapper, champion, observer, compare, _ = setup
    for _ in range(45):
        wrapper.act(image())
    assert wrapper.intervention_count == 40
    assert compare.call_count == 40
    assert champion.queries == observer.commits == 46
    assert observer.observations == 45
    assert wrapper.last_diagnostics["reasons"] == ["intervention_budget"]
    np.testing.assert_array_equal(wrapper.last_diagnostics["issued_action"], wrapper.nominal.action)


def test_hook_restored_on_champion_exception_without_commit_or_shared_module_mutation(setup, frozen):
    wrapper, champion, observer, compare, _ = setup
    original = frozen.shield.advance_boxes
    private = wrapper._shield_module.advance_boxes
    original_act = wrapper.shield.act.__func__
    wrapper.nominal.act = Mock(side_effect=RuntimeError("synthetic failure"))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        wrapper.act(image())
    assert frozen.shield.advance_boxes is original
    assert wrapper._shield_module.advance_boxes is private
    assert wrapper.shield.act.__func__ is original_act
    assert "act" not in vars(wrapper.shield)
    assert observer.commits == 1 and observer._pending is None
    assert champion.queries == 2
    compare.assert_not_called()
    assert wrapper.last_diagnostics["act_cpu_seconds"] >= 0


def test_hook_cannot_capture_a_different_champion_in_same_module(setup, frozen):
    wrapper, _, _, _, _ = setup
    other = Champion(frozen.shield.CollisionShieldAgent(Nominal()))
    original = wrapper.nominal.act
    module_advance = frozen.shield.advance_boxes

    def interleaved(obs):
        assert frozen.shield.advance_boxes is module_advance
        other.act(obs)
        return original(obs)

    wrapper.nominal.act = interleaved
    wrapper.act(image(box=True))
    assert wrapper.last_diagnostics["intervention"]
    assert other.queries == 1
    assert "act" not in vars(other.driver)


def test_comparator_exception_is_baseline_with_one_actual_commit(setup):
    wrapper, champion, observer, compare, _ = setup
    compare.side_effect = ValueError("unsupported synthetic geometry")
    np.testing.assert_array_equal(wrapper.act(image()), wrapper.nominal.action)
    assert wrapper.last_diagnostics["reasons"] == ["comparison_error"]
    assert observer.commits == champion.queries == 2
    assert observer._pending is not None


def test_reset_consumes_initial_endpoint_once_and_clears_budget(setup):
    wrapper, _, observer, _, _ = setup
    wrapper.intervention_count = 40
    wrapper.reset(image())
    assert wrapper.intervention_count == 0 and len(wrapper.frames) == 1
    assert not wrapper.issued_actions
    resets, observations = observer.resets, observer.observations
    wrapper.act(image())
    assert observer.resets == resets and observer.observations == observations
    assert observer.commits == 2
    wrapper.reset()
    assert not wrapper.frames
    wrapper.act(image())
    assert observer.resets == resets + 1
    assert observer.observations == observations


def test_complete_act_timers_enclose_observe_proposal_scoring_and_commit(setup):
    wrapper, _, observer, compare, _ = setup
    events = observer.events
    events.clear()
    original = wrapper.nominal.act

    def nominal(obs):
        events.append("proposal")
        return original(obs)

    def compare_now(*args, **kwargs):
        events.append("compare")
        return ranking()

    def cpu():
        events.append("cpu")
        return 1. if len(events) == 1 else 3.

    def wall():
        events.append("wall")
        return 10. if len(events) == 2 else 14.

    wrapper.nominal.act = nominal
    compare.side_effect = compare_now
    with patch.object(successor, "process_time", side_effect=cpu), patch.object(successor, "perf_counter", side_effect=wall):
        wrapper.act(image())
    assert events == ["cpu", "wall", "observe", "proposal", "hypotheses", "compare", "commit", "cpu", "wall"]
    assert wrapper.last_diagnostics["act_cpu_seconds"] == 2.
    assert wrapper.last_diagnostics["act_wall_seconds"] == 4.
