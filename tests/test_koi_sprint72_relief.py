"""Exact single-branch tests, with no simulator construction or reset."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from haic.algorithms.koi.sprint72_relief import apply_sprint72_relief


def state(speed=73., **changes):
    action = np.asarray([.125, 0., .15], dtype=np.float32)
    info = dict(pixel_speed=speed, target_speed=60.,
                road_centers={42: 42., 54: 42.}, parent_action=action.tolist(),
                free_samples=4, free_distance=max(speed, 72.) * .4,
                impact_proxy_trigger=False, braking_proxy_veto=False,
                geometry_repaired=False)
    info.update(changes)
    return SimpleNamespace(last=info, steps=11, track=None, recovery_left=0,
                           brake_history=[.1, float(action[2])]), action


@pytest.mark.parametrize('speed,brake', [(72., 0.), (72.5, .01), (73., .02),
                                      (75., .06), (79.5, .15), (85., .15)])
def test_only_fixed_pedals_are_substituted(speed, brake):
    agent, action = state(speed)
    old = action.copy()
    before = copy.deepcopy(agent.__dict__)
    apply_sprint72_relief(agent, action, None, None, 0)
    np.testing.assert_array_equal(action, np.asarray([old[0], 0., brake], np.float32))
    assert agent.last['sprint72_eligible']
    assert agent.last['sprint72_applied'] == (speed < 79.5)
    for key, value in before['last'].items():
        assert agent.last[key] == value
    assert agent.brake_history == before['brake_history']
    assert agent.last['sprint72_prearrival_action'] == action.tolist()
    assert agent.last['sprint72_pre_action'] == old.tolist()


@pytest.mark.parametrize('field,value', [
    ('pixel_speed', np.nextafter(72., -np.inf)),
    ('pixel_speed', float('nan')), ('pixel_speed', float('inf')),
    ('target_speed', np.nextafter(60., -np.inf)),
    ('target_speed', np.nextafter(60., np.inf)),
    ('road_centers', {42: 42.}), ('road_centers', {54: 42.}),
    ('road_centers', {42: 42., 54: 45.}),
    ('free_samples', 3), ('free_distance', 18.),
    ('geometry_repaired', True), ('impact_proxy_trigger', True),
])
def test_each_noneligible_branch_is_exact_noop(field, value):
    agent, action = state(**{field: value})
    old = action.copy()
    before = copy.deepcopy(agent.brake_history)
    apply_sprint72_relief(agent, action, None, None, 0)
    np.testing.assert_array_equal(action, old)
    assert not agent.last['sprint72_eligible']
    assert not agent.last['sprint72_applied']
    assert agent.brake_history == before


@pytest.mark.parametrize('reason', ['prefix', 'near', 'far', 'track', 'recovery',
                                  'impact_final_tick', 'gas', 'brake', 'steer',
                                  'parent_steer_limit'])
def test_protected_origins_are_unchanged(reason):
    agent, action = state()
    near = far = None
    impact = 0
    if reason == 'prefix':
        agent.steps = 10
    elif reason == 'near':
        near = (30., 40., 42.)
    elif reason == 'far':
        far = (18., 40., 42.)
    elif reason == 'track':
        agent.track = np.asarray([23., 40.])
    elif reason == 'recovery':
        agent.recovery_left = 1
    elif reason == 'impact_final_tick':
        impact = 1
    elif reason == 'gas':
        action[1] = .001
    elif reason == 'brake':
        action[2] = np.nextafter(np.float32(.15), np.float32(0.))
    elif reason == 'steer':
        action[0] += .01
    elif reason == 'parent_steer_limit':
        action[0] = .18
        agent.last['parent_action'][0] = float(action[0])
    old = action.copy()
    apply_sprint72_relief(agent, action, near, far, impact)
    np.testing.assert_array_equal(action, old)
    assert not agent.last['sprint72_eligible']


def test_unfixed_floating_point_free_distance_boundary():
    agent, action = state(72., free_distance=28.799999999999994)
    apply_sprint72_relief(agent, action, None, None, 0)
    assert not agent.last['sprint72_space_ok']
    assert action[2] == np.float32(.15)
    agent.last['free_distance'] = 28.8
    apply_sprint72_relief(agent, action, None, None, 0)
    assert agent.last['sprint72_eligible'] and action[2] == 0.


def test_vetoed_proxy_is_not_a_blanket_exclusion():
    agent, action = state(73., impact_proxy_trigger=True, braking_proxy_veto=True)
    apply_sprint72_relief(agent, action, None, None, 1)
    assert agent.last['sprint72_eligible'] and action[2] == np.float32(.02)


def test_no_private_direction_gate_or_added_controller_state():
    forward, a = state()
    backward, b = state()
    backward.evaluator_heading = np.pi
    expected_fields = set(backward.__dict__)
    apply_sprint72_relief(forward, a, None, None, 0)
    apply_sprint72_relief(backward, b, None, None, 0)
    np.testing.assert_array_equal(a, b)
    assert set(backward.__dict__) == expected_fields
    assert backward.last['target_speed'] == 60.


def test_below72_sprint_pedals_remain_unchanged():
    agent, action = state(71.99)
    action[1:] = 1., 0.
    before = action.copy()
    apply_sprint72_relief(agent, action, None, None, 0)
    np.testing.assert_array_equal(action, before)
    assert not agent.last['sprint72_eligible']


@pytest.fixture
def packaged_runtime(tmp_path, monkeypatch):
    from scripts.evaluate_koi_sprint72_relief import candidate_members

    for name, content in candidate_members().items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    saved = {k: v for k, v in sys.modules.items()
             if k == 'haic_agent' or k.startswith('haic_agent.')}
    for name in saved:
        sys.modules.pop(name)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        yield SimpleNamespace(
            contact=importlib.import_module('haic_agent.contact_continuity_runtime'),
            far=importlib.import_module('haic_agent.far_hazard_runtime'),
            shield=importlib.import_module('haic_agent.collision_shield_runtime'))
    finally:
        for name in list(sys.modules):
            if name == 'haic_agent' or name.startswith('haic_agent.'):
                sys.modules.pop(name)
        sys.modules.update(saved)


def observation(speed=73., object_row=None):
    frame = np.full((84, 84), .1, np.float32)
    frame[6:72, 21:64] = .4
    frame[77:83, 10:13] = (.27 + speed * .085) / 18.
    if object_row is not None:
        frame[object_row:object_row + 3, 41:44] = .7
    return np.repeat(frame[None], 4, axis=0)


def test_packaged_full_chain_records_the_actual_brake(packaged_runtime):
    nominal = packaged_runtime.contact.ContactContinuityAgent('crossing_projection')
    nominal.steps = 10
    shield = packaged_runtime.shield.CollisionShieldAgent(nominal)
    action = shield.act(observation())
    assert nominal.last['sprint72_eligible']
    assert action.dtype == np.float32
    assert action[2] == np.float32(.02 * (nominal.last['pixel_speed'] - 72.))
    assert nominal.brake_history[-1] == float(action[2])
    assert nominal.last['final_brake'] == float(action[2])
    assert shield.last_shield['baseline_action'] == action.tolist()
    assert nominal.last['target_speed'] == 60.
    assert not nominal.last['mechanism_active']
    # Original FarHazard diagnostics include all changes since the sprint copy;
    # far_changed alone must NOT be interpreted as arrival-cap activity.
    assert nominal.last['far_changed'] and not nominal.last['far_active']


@pytest.mark.parametrize('object_row', [18, 46])
def test_packaged_arrival_precedence_is_untouched(packaged_runtime, monkeypatch, object_row):
    nominal = packaged_runtime.contact.ContactContinuityAgent('crossing_projection')
    nominal.steps = 10
    baseline = copy.deepcopy(nominal)
    obs = observation(object_row=object_row)
    action = nominal.act(obs)
    with monkeypatch.context() as m:
        m.setattr(packaged_runtime.far, 'apply_sprint72_relief', lambda *args: None)
        expected = baseline.act(obs)
    assert not nominal.last['sprint72_eligible']
    assert nominal.last['far_active'] and nominal.last['arrival_cap'] is not None
    np.testing.assert_array_equal(action, expected)
    assert nominal.brake_history == baseline.brake_history
    assert nominal.brake_history[-1] == float(action[2])


def test_real_brake_history_can_change_later_impact_veto(packaged_runtime, monkeypatch):
    nominal = packaged_runtime.contact.ContactContinuityAgent('crossing_projection')
    nominal.steps = 10
    baseline = copy.deepcopy(nominal)
    first = nominal.act(observation(72.25))
    assert nominal.last['sprint72_eligible'] and 0. < first[2] < .01
    assert nominal.brake_history[-1] == float(first[2])
    with monkeypatch.context() as m:
        m.setattr(packaged_runtime.far, 'apply_sprint72_relief', lambda *args: None)
        baseline.act(observation(72.25))
        baseline.act(observation(40.))
    nominal.act(observation(40.))
    assert nominal.last['impact_proxy_trigger'] and baseline.last['impact_proxy_trigger']
    assert not nominal.last['braking_proxy_veto']
    assert baseline.last['braking_proxy_veto']
    assert nominal.impact_left == 11 and baseline.impact_left == 0


@pytest.mark.parametrize('matches', [0, 1])
def test_far_track_gate_uses_current_original_tracker_update(packaged_runtime, matches):
    nominal = packaged_runtime.contact.ContactContinuityAgent('crossing_projection')
    nominal.steps = 10
    nominal.track = np.asarray([15., 42.])
    nominal.last_detection = nominal.track.copy()
    nominal.velocity = np.asarray([1., 0.])
    nominal.matches = matches
    action = nominal.act(observation())
    assert nominal.last['sprint72_eligible'] == (matches == 0)
    assert (nominal.track is None) == (matches == 0)
    assert nominal.brake_history[-1] == float(action[2])
    if matches:
        assert action[2] == np.float32(.15)


@pytest.mark.parametrize('speed', [25., 60., 71., 71.99])
def test_packaged_below_cap_policy_and_history_are_identical(packaged_runtime, monkeypatch, speed):
    nominal = packaged_runtime.contact.ContactContinuityAgent('crossing_projection')
    nominal.steps = 10
    baseline = copy.deepcopy(nominal)
    action = nominal.act(observation(speed))
    with monkeypatch.context() as m:
        m.setattr(packaged_runtime.far, 'apply_sprint72_relief', lambda *args: None)
        expected = baseline.act(observation(speed))
    np.testing.assert_array_equal(action, expected)
    assert nominal.brake_history == baseline.brake_history
    assert nominal.last['free_distance'] == baseline.last['free_distance']
    assert nominal.last['free_samples'] == baseline.last['free_samples']


def test_archived_applicability_keeps_all_four_reverse_decisions():
    root = Path(__file__).resolve().parents[1] / 'runs/koi-nominal-trajectory-v1'
    expected = {
        (1, 3184000002): [30, 56],
        (1, 3184000013): [50, 51, 56, 57, 215],
        (1, 3184000015): [38, 39, 44, 45, 114, 139, 140, 145],
        (2, 3184000001): [],
        (2, 3184000006): [40, 41, 46, 194, 195, 447, 448],
        (2, 3184000015): [221, 229],
        (3, 3184000002): [],
        (3, 3184000015): [38, 39, 44, 45],
    }
    total = 0
    reverse = []
    for (track, seed), steps in expected.items():
        path = root / f'{track}-{seed}-r0-frozen_shield.decisions.jsonl'
        if not path.is_file():
            pytest.skip('consumed TRAIN evidence is not present on this checkout')
        eligible = []
        impact_before = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            info = copy.deepcopy(row['controller'])
            info['road_centers'] = {int(k): v for k, v in info['road_centers'].items()}
            # Reconstruct ONLY the original pre-arrival action, never its shielded
            # result. The source-pinned acceleration layer has one active mode.
            action = np.asarray(info['parent_action'], np.float32)
            if info['mechanism_active']:
                assert info['acceleration_target'] == 72.
                action[1:] = 1., 0.
            agent = SimpleNamespace(last=info, steps=row['step'],
                                    track=info['tracked_object'],
                                    recovery_left=info['recovery_steps_remaining'])
            far = info['far_objects'][0] if info['far_objects'] else None
            apply_sprint72_relief(agent, action, info['near_object'], far, impact_before)
            impact_before = info['impact_steps_remaining']
            total += 1
            if info['sprint72_eligible']:
                eligible.append(row['step'])
                if abs(info['evaluation_only']['pre']['heading_error']) > np.pi / 2:
                    reverse.append((track, seed, row['step']))
        assert eligible == steps
    assert total == 2130
    assert reverse == [(2, 3184000006, step) for step in (194, 195, 447, 448)]
