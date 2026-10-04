from types import SimpleNamespace

import numpy as np
import pytest

from haic.algorithms.koi.collision_recovery import CollisionRecoveryAgent, observed_road


def image(center=42):
    frame = np.full((84, 84), .1, np.float32)
    frame[20:70, center-12:center+13] = .4
    return np.tile(frame[None], (4, 1, 1))


class Crossing:
    contact_mode = 'crossing_projection'

    def __init__(self):
        self.steps = 11
        self.base = SimpleNamespace(_frame=lambda obs: obs[-1], _obstacle_side=1.)
        self.brake_history = [0.]
        self.centers = {30: 42., 42: 42., 54: 42.}
        self.obj = None
        self.action = np.array([.12, .6, 0.], np.float32)

    def act(self, observation):
        return self.action.copy()

    def reset(self, observation):
        pass

    def last_step_diagnostics(self):
        return dict(road_centers=self.centers, near_object=self.obj,
                    correction=0., pixel_speed=55., contact_active=False)


def test_requires_crossing_not_v2():
    with pytest.raises(ValueError):
        CollisionRecoveryAgent(SimpleNamespace())


def test_unarmed_normal_driving_is_exact():
    source = Crossing()
    candidate = CollisionRecoveryAgent(source)
    for _ in range(12):
        np.testing.assert_array_equal(candidate.act(image()), source.action)
        assert candidate.last['priority_mode'] == 'baseline'


def test_heading_alignment_precedes_lateral_reentry_and_brakes():
    source = Crossing()
    source.centers = {30: 67., 42: 60., 54: 50.}
    candidate = CollisionRecoveryAgent(source)
    candidate.mode = 'avoid'
    action = candidate.act(image(50))
    assert candidate.last['priority_mode'] == 'align'
    assert action[0] == pytest.approx(.4)
    assert action[1] == 0 and action[2] > 0
    assert source.brake_history[-1] == float(action[2])


def test_reentry_targets_edge_not_center_and_needs_stable_handoff():
    source = Crossing()
    source.centers = {30: 49., 42: 49., 54: 49.}
    road = observed_road(image(49)[-1], source.centers)
    assert road is not None and road['supported'] and road['edge_error'] == 0
    candidate = CollisionRecoveryAgent(source)
    candidate.mode = 'align'
    for _ in range(2):
        action = candidate.act(image(49))
        assert action[0] == 0.
        assert candidate.last['priority_mode'] == 'reenter'
    np.testing.assert_array_equal(candidate.act(image(49)), source.action)
    assert candidate.last['priority_mode'] == 'handoff'
    assert candidate.mode == 'baseline'


def test_missing_road_brakes_and_cannot_report_handoff():
    source = Crossing()
    source.centers = {}
    candidate = CollisionRecoveryAgent(source)
    candidate.mode, candidate.last_heading = 'align', .5
    dark = np.full((4, 84, 84), .1, np.float32)
    action = source.action
    for _ in range(6):
        action = candidate.act(dark)
        assert action[1] == 0 and action[2] > 0
        assert candidate.last['priority_mode'] == 'road_search'
    assert action[0] == 0.
    candidate.reset(dark)
    assert candidate.mode == 'baseline' and candidate.age == 0
