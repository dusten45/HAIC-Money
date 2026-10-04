"""Current-frame magnitude and frozen-shield integration; zero simulator resets."""

import copy
import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from haic.algorithms.koi import avoidance_magnitude as magnitude
from haic.algorithms.koi import collision_shield as shield
from haic.algorithms.koi.steering_terms import observed_center, reconstruct_steering


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'submissions/20261002-crossing-projection-collision-shield-v1-baseline'
SHIELD_SHA = 'ad772bde9a9c3f4596fdfc742e7cac33f1b605ff52a2b3f76fbd4b75d6361d96'


def image(left=37, top=40, width=4, height=4):
    frame = np.full((84, 84), .1, np.float32)
    frame[8:72, 22:64] = .4
    frame[top:top + height, left:left + width] = .7
    return np.repeat(frame[None], 4, axis=0)


class Crossing:
    contact_mode = 'crossing_projection'

    def __init__(self, *, crossing=False, repaired=False, centers=None, correction=.04):
        self.steps = 11
        self.impact_left = 0
        self.calls = 0
        self.speed = 30.
        self.crossing, self.repaired = crossing, repaired
        self.centers = {30: 42., 42: 44., 54: 42.} if centers is None else centers
        self.correction = correction
        self.projected: float | None = None
        self.info = {}
        self.base = SimpleNamespace(
            _frame=lambda obs: obs[-1] if obs.ndim == 3 else obs,
            _estimate_speed=lambda frame: self.speed,
            _last_obstacle=None, _obstacle_side=1.)
        self.last_action: Any = None

    def act(self, observation):
        self.calls += 1
        side = self.base._obstacle_side
        obj = self.base._last_obstacle
        urgency = float(np.clip((obj[0] - 22) / 18, 0., 1.)) if obj is not None else 0.
        p = .022 * (self.centers.get(42, 42.) - 42.)
        l = .018 * (self.centers.get(42, 42.) - self.centers.get(54, 42.))
        road = p + l
        inherited = float(np.float32(np.clip(p + l + side * .34 * urgency, -.7, .7)))
        damping = float(np.float32(np.clip(inherited + self.correction, -.7, .7)))
        parent = damping
        if self.impact_left:
            parent = float(np.float32(np.clip(p + l + self.correction, -.7, .7)))
            self.impact_left -= 1
        if self.repaired or self.crossing:
            middle, near = observed_center(self.centers, 42), observed_center(self.centers, 54)
            assert middle is not None and near is not None
            road = .022 * (middle - 42.) + .018 * (middle - near)
            parent = float(np.float32(np.clip(
                road + side * .34 * urgency + self.correction, -.7, .7))) if self.repaired else parent
        final = parent
        if self.crossing:
            final = float(np.float32(np.clip(
                (road + side * .55 * urgency if self.repaired
                 else np.clip(road + side * .55 * urgency, -.7, .7))
                + self.correction, -.7, .7)))
        self.info = dict(road_centers=dict(self.centers), near_object=obj,
                         correction=self.correction, inherited_steer=inherited,
                         damping_steer=damping, parent_action=[parent, .6, .07],
                         geometry_repaired=self.repaired, contact_mode=self.contact_mode,
                         contact_active=self.crossing, far_mode='arrival_speed',
                         braking_proxy_veto=False, impact_proxy_trigger=False,
                         impact_steps_remaining=self.impact_left,
                         projected_obstacle_x=self.projected, pixel_speed=self.speed)
        self.last_action = np.array([final, .6, .07], np.float32)
        return self.last_action.copy()

    def reset(self, observation=None):
        self.calls = 0
        self.info = {}

    def last_step_diagnostics(self):
        return copy.deepcopy(self.info)


def driver_for(observation, **kwargs):
    driver = Crossing(**kwargs)
    frame = observation[-1] if observation.ndim == 3 else observation
    ys, xs = np.nonzero(frame >= .54)
    driver.base._last_obstacle = float(ys.mean()), float(xs.mean()), 42.
    return driver


@pytest.mark.parametrize('cap', [.34, .55])
def test_small_clearance_means_smaller_continuous_magnitude(cap):
    width = shield.HALF_WIDTH + .5 / shield.X_SCALE
    outputs = []
    for shift in (0., .01, .02, .1, .5, 1.):
        proposal = magnitude.bounded_magnitude((-5., 18., -width + shift, 20.), 1., 40., cap)
        assert proposal is not None
        outputs.append(proposal)
    assert outputs[0]['magnitude'] == 0.
    assert all(a['magnitude'] < b['magnitude'] for a, b in zip(outputs, outputs[1:]))
    assert outputs[-1]['magnitude'] < cap
    assert outputs[2]['magnitude'] - outputs[1]['magnitude'] < .001


@pytest.mark.parametrize('side', [-1., 1.])
@pytest.mark.parametrize('cap', [.34, .55])
def test_imminent_deep_overlap_retains_original_strong_magnitude(side, cap):
    proposal = magnitude.bounded_magnitude((-1., 3., 1., 5.), side, 40., cap)
    assert proposal is not None
    assert proposal['collision_risk'] == 1.
    assert proposal['magnitude'] == cap


def test_forward_distance_and_speed_control_collision_urgency():
    far = magnitude.bounded_magnitude((-1., 20., 1., 22.), 1., 20., .55)
    near = magnitude.bounded_magnitude((-1., 10., 1., 12.), 1., 20., .55)
    fast = magnitude.bounded_magnitude((-1., 10., 1., 12.), 1., 70., .55)
    assert far is not None and near is not None and fast is not None
    assert far['geometric_demand'] < near['geometric_demand']
    assert far['ttc_s'] > near['ttc_s'] > fast['ttc_s']
    assert fast['collision_risk'] > near['collision_risk']


def test_projected_collision_can_restore_strong_term_when_current_flank_is_clear():
    width = shield.HALF_WIDTH + .5 / shield.X_SCALE
    box = (-width - 2., 3., -width - .1, 5.)
    clear = magnitude.bounded_magnitude(box, 1., 40., .34)
    closing = magnitude.bounded_magnitude(box, 1., 40., .34, projected_shift_m=3.)
    assert clear is not None and closing is not None
    assert clear['required_clearance_m'] == closing['required_clearance_m'] == 0.
    assert clear['magnitude'] == 0.
    assert closing['collision_risk'] == 1. and closing['magnitude'] == .34


def test_two_clear_endpoints_do_not_hide_imminent_crossing_overlap():
    width = shield.HALF_WIDTH + .5 / shield.X_SCALE
    box = (-width - 2., 3., -width - .1, 5.)
    shift = 2. * width + 2.2
    assert box[2] < -width and box[0] + shift > width
    proposal = magnitude.bounded_magnitude(box, 1., 70., .55, shift)
    assert proposal is not None and proposal['required_clearance_m'] == 0.
    assert proposal['swept_gap_m'] < 0.
    assert proposal['collision_risk'] == 1. and proposal['magnitude'] == .55


def test_full_wheels_and_existing_half_pixel_allowance_are_not_shrunk():
    proposal = magnitude.bounded_magnitude((-1., 20., 1., 22.), 1., 40., .34)
    assert proposal is not None
    assert proposal['required_clearance_m'] == shield.HALF_WIDTH + .5 / shield.X_SCALE + 1.
    assert proposal['approach_m'] == 20. - shield.FRONT - .5 / shield.Y_SCALE
    wider = magnitude.bounded_magnitude((-1., 20., 2., 22.), 1., 40., .34)
    assert wider is not None
    assert wider['magnitude'] > proposal['magnitude']


def test_mirrored_side_is_exact_and_never_changed():
    right = magnitude.bounded_magnitude((-4., 12., -.7, 14.), 1., 40., .55, 1.)
    left = magnitude.bounded_magnitude((.7, 12., 4., 14.), -1., 40., .55, -1.)
    assert right is not None and left is not None
    for key in ('required_clearance_m', 'geometric_demand', 'swept_gap_m',
                'collision_risk', 'magnitude'):
        assert left[key] == right[key]
    assert right['side'] == 1. and left['side'] == -1.


@pytest.mark.parametrize('box,side,speed,cap,shift', [
    ((-1., 10., 1., 12.), 0., 40., .34, 0.),
    ((-1., 10., 1., 12.), 1., 80., .34, 0.),
    ((-1., 10., 1., 12.), 1., 0., .34, 0.),
    ((-1., 10., 1., 12.), 1., 40., .56, 0.),
    ((-1., 10., 1., 12.), 1., 40., .34, float('nan')),
    ((1., 10., -1., 12.), 1., 40., .34, 0.),
    ((-1., float('inf'), 1., 12.), 1., 40., .34, 0.),
])
def test_invalid_geometry_never_proposes_a_reduction(box, side, speed, cap, shift):
    assert magnitude.bounded_magnitude(box, side, speed, cap, shift) is None


@pytest.mark.parametrize('crossing,repaired', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('side', [-1., 1.])
def test_reduction_replaces_one_term_preserving_pipeline_and_pedals(crossing, repaired, side):
    observation = image(left=37 if side > 0 else 44)
    driver = driver_for(observation, crossing=crossing, repaired=repaired)
    driver.base._obstacle_side = side
    adapter = magnitude.AvoidanceMagnitudeAgent(driver)
    action = adapter.act(observation)
    assert driver.calls == 1 and adapter.last['magnitude_changed']
    assert adapter.last['magnitude_reason'] == 'bounded_magnitude'
    proposal = adapter.last['magnitude_proposal']
    assert proposal['legacy_magnitude'] == (.55 if crossing else .34)
    assert 0. < proposal['magnitude'] < proposal['legacy_magnitude']
    np.testing.assert_array_equal(action[1:], driver.last_action[1:])
    assert driver.base._obstacle_side == side
    raw = reconstruct_steering(driver.info, float(driver.last_action[0]), state=dict(
        obstacle_side=side, steps=11, impact_left_before_act=0))
    raw['active_geometry_repaired'] = repaired
    assert action[0] == magnitude.replacement_steer(raw, proposal['magnitude'])
    assert magnitude.replacement_steer(raw, proposal['legacy_magnitude']) == driver.last_action[0]
    box = magnitude.component_bounds(observation[-1], driver.base._last_obstacle)
    assert box is not None
    left, top, width, height = box
    expected_box = ((left - 1 - 42) / shield.X_SCALE, (63 - top - height) / shield.Y_SCALE,
                    (left + width - 42) / shield.X_SCALE, (64 - top) / shield.Y_SCALE)
    assert proposal['current_bbox_m'] == list(expected_box)
    np.testing.assert_array_equal(np.asarray([expected_box]), shield.detect_boxes(observation[-1]))


def test_nested_clip_and_float32_are_not_flat_road_plus_delta():
    observation = image()
    driver = driver_for(observation, crossing=True, centers={30: 65., 42: 65., 54: 42.},
                        correction=-.2)
    adapter = magnitude.AvoidanceMagnitudeAgent(driver)
    action = adapter.act(observation)
    # Road alone is already clipped. Subtracting the raw .55 from the final
    # saturated command would invent a different (and wrong) steering pipeline.
    proposal = adapter.last['magnitude_proposal']
    assert action[0] == np.float32(.5)
    assert action[0] != np.float32(driver.last_action[0] - .55 + proposal['magnitude'])


def test_cap_is_exact_original_action_even_with_nontrivial_rounding():
    observation = image(left=40, top=55)
    driver = driver_for(observation, crossing=True, correction=.039671582759)
    adapter = magnitude.AvoidanceMagnitudeAgent(driver)
    action = adapter.act(observation)
    assert adapter.last['magnitude_proposal']['magnitude'] == .55
    np.testing.assert_array_equal(action, driver.last_action)
    assert not adapter.last['magnitude_changed']


@pytest.mark.parametrize('guard', ['prefix', 'dropout', 'impact_expiry', 'bad_hud',
                                  'nonfinite_prediction', 'nonfinite_frame', 'unmatched_component'])
def test_unknown_or_protected_paths_return_exact_original(guard):
    observation = image()
    driver = driver_for(observation)
    if guard == 'prefix':
        driver.steps = 10
    elif guard == 'dropout':
        driver.base._last_obstacle = None
    elif guard == 'impact_expiry':
        driver.impact_left = 1
    elif guard == 'bad_hud':
        driver.base._estimate_speed = lambda frame: 80.
    elif guard == 'nonfinite_prediction':
        driver.projected = float('nan')
    elif guard == 'nonfinite_frame':
        observation[-1, 0, 0] = float('nan')
    elif guard == 'unmatched_component':
        driver.base._last_obstacle = (41., 39., 42.)
    adapter = magnitude.AvoidanceMagnitudeAgent(driver)
    action = adapter.act(observation)
    np.testing.assert_array_equal(action, driver.last_action)
    assert not adapter.last['magnitude_changed']
    assert adapter.last['magnitude_proposal'] is None


def test_only_diagnostics_survive_and_observation_drives_every_new_decision():
    observation = image()
    driver = driver_for(observation)
    adapter = magnitude.AvoidanceMagnitudeAgent(driver)
    initial = adapter.act(observation)
    assert set(vars(adapter)) == {'nominal', 'last'}
    driver.base._last_obstacle = None
    unchanged = adapter.act(observation)
    np.testing.assert_array_equal(unchanged, driver.last_action)
    assert adapter.last['magnitude_proposal'] is None
    driver.base._last_obstacle = (41.5, 38.5, 42.)
    np.testing.assert_array_equal(adapter.act(observation), initial)
    assert set(vars(adapter)) == {'nominal', 'last'}
    adapter.reset()
    assert adapter.last == {} and driver.calls == 0
    assert not hasattr(adapter, 'driver') and not hasattr(adapter, 'observe_executed_action')


def test_braking_veto_is_not_mistaken_for_genuine_impact_suppression():
    observation = image()
    driver = driver_for(observation)
    original_act = driver.act

    def vetoed_act(observation):
        action = original_act(observation)
        driver.info.update(impact_proxy_trigger=True, braking_proxy_veto=True)
        return action

    driver.act = vetoed_act
    adapter = magnitude.AvoidanceMagnitudeAgent(driver)
    adapter.act(observation)
    assert adapter.last['magnitude_changed']
    assert adapter.last['magnitude_reason'] == 'bounded_magnitude'
    assert adapter.last['magnitude_proposal']['legacy_magnitude'] == .34


def test_requires_crossing_not_recovery_or_release():
    with pytest.raises(ValueError):
        magnitude.AvoidanceMagnitudeAgent(SimpleNamespace(contact_mode='contact_coast'))


def test_unchanged_v1_is_final_layer_and_receives_reduced_nominal(monkeypatch):
    observation = image()
    driver = driver_for(observation)
    adapter = magnitude.AvoidanceMagnitudeAgent(driver)
    final_layer = shield.CollisionShieldAgent(adapter)
    assert final_layer.driver is adapter

    def collision_on_nominal(paths, boxes):
        gaps = np.ones(paths.shape[0])
        gaps[0] = -1.
        return gaps

    monkeypatch.setattr(shield, 'footprint_gaps', collision_on_nominal)
    action = final_layer.act(observation)
    assert adapter.last['magnitude_changed'] and final_layer.last_shield['active']
    assert final_layer.last_shield['baseline_threat']
    assert final_layer.last_shield['baseline_action'][0] != driver.last_action[0]
    assert final_layer.last_shield['chosen_clearance'] > 0
    np.testing.assert_array_equal(action[1:], driver.last_action[1:])


def test_v1_clear_prediction_hands_through_candidate_without_release_state(monkeypatch):
    observation = image()
    driver = driver_for(observation)
    adapter = magnitude.AvoidanceMagnitudeAgent(driver)
    final_layer = shield.CollisionShieldAgent(adapter)
    monkeypatch.setattr(shield, 'footprint_gaps', lambda paths, boxes: np.ones(paths.shape[0]))
    action = final_layer.act(observation)
    assert not final_layer.last_shield['active']
    np.testing.assert_array_equal(action, np.asarray(adapter.last['magnitude_baseline_action'])
                                  + [adapter.last['final_steer'] - driver.last_action[0], 0., 0.])
    assert set(vars(adapter)) == {'nominal', 'last'}


def test_real_geometry_shield_can_intervene_when_nominal_reduction_predicts_collision():
    observation = image(left=33, top=45)
    driver = driver_for(observation, centers={30: 31., 42: 31., 54: 31.}, correction=0.)
    driver.speed = 40.
    adapter = magnitude.AvoidanceMagnitudeAgent(driver)
    final_layer = shield.CollisionShieldAgent(adapter)
    action = final_layer.act(observation)
    assert adapter.last['magnitude_changed']
    assert final_layer.last_shield['baseline_threat'] and final_layer.last_shield['active']
    assert final_layer.last_shield['baseline_clearance'] <= 0
    assert final_layer.last_shield['chosen_clearance'] > 0
    assert action[0] != adapter.last['final_steer']
    np.testing.assert_array_equal(action[1:], driver.last_action[1:])


def test_frozen_bundle_and_live_geometry_helper_shield_bytes_are_unchanged():
    assert hashlib.sha256(Path(shield.__file__).read_bytes()).hexdigest() == SHIELD_SHA
    assert hashlib.sha256((BUNDLE / 'source/haic_agent/collision_shield_runtime.py').read_bytes()).hexdigest() == SHIELD_SHA
    assert hashlib.sha256((BUNDLE / 'submission.zip').read_bytes()).hexdigest() == (
        'c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801')


def test_actual_frozen_source_branch_and_adapter_without_environment(monkeypatch):
    monkeypatch.syspath_prepend(str(BUNDLE / 'source'))
    # Clear only module cache entries in this test and let monkeypatch restore them.
    for name in list(sys.modules):
        if name == 'haic_agent' or name.startswith('haic_agent.'):
            monkeypatch.delitem(sys.modules, name)
    cls = importlib.import_module('haic_agent.contact_continuity_runtime').ContactContinuityAgent
    original = cls('crossing_projection')
    observation = image()
    original.steps = 10
    adapter = magnitude.AvoidanceMagnitudeAgent(original)
    action = adapter.act(observation)
    reference = cls('crossing_projection')
    reference.steps = 10
    baseline = reference.act(observation)
    assert action.dtype == np.float32 and adapter.last['magnitude_changed']
    assert adapter.last['magnitude_reason'] == 'bounded_magnitude'
    np.testing.assert_array_equal(adapter.last['magnitude_baseline_action'], baseline)
    np.testing.assert_array_equal(action[1:], baseline[1:])
    assert adapter.last['magnitude_proposal']['legacy_magnitude'] == .34
    assert action[0] < baseline[0]
    assert original.base._obstacle_side == 1.
    assert set(vars(adapter)) == {'nominal', 'last'}


def test_frozen_crossing_source_retains_strong_term_for_two_clear_endpoints(monkeypatch):
    monkeypatch.syspath_prepend(str(BUNDLE / 'source'))
    for name in list(sys.modules):
        if name == 'haic_agent' or name.startswith('haic_agent.'):
            monkeypatch.delitem(sys.modules, name)
    cls = importlib.import_module('haic_agent.contact_continuity_runtime').ContactContinuityAgent
    original = cls('crossing_projection')
    observation = image(left=36, top=43, width=2, height=6)
    observation[:, 77:83, 10:13] = (.27 + 70. * .085) / 18.
    original.steps = 10
    original.previous_object = (43.5, 36.5 - 10.5 * 2. / 14.5, 42.)
    adapter = magnitude.AvoidanceMagnitudeAgent(original)
    action = adapter.act(observation)
    assert adapter.last['contact_active']
    assert adapter.last['projected_obstacle_x'] == pytest.approx(47.)
    proposal = adapter.last['magnitude_proposal']
    assert proposal['ttc_s'] < .08 and proposal['required_clearance_m'] == 0.
    assert proposal['collision_risk'] == 1. and proposal['magnitude'] == .55
    assert not adapter.last['magnitude_changed']
    np.testing.assert_array_equal(action, adapter.last['magnitude_baseline_action'])
