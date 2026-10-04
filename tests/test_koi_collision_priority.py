from types import SimpleNamespace

import numpy as np

from haic.algorithms.koi.collision_priority import (
    CollisionPriorityAgent, NEGATIVE_REWARD_LIMIT, arc_clearance, obstacle_box,
)


def pixels(x: int | None = 39, y=47):
    frame = np.full((84, 84), .1, np.float32)
    frame[20:70, 27:58] = .4
    if x is not None:
        frame[y-1:y+2, x-1:x+2] = .7
    return np.tile(frame[None], (4, 1, 1))


class V2:
    stabilize_ambiguous_flank = True

    def __init__(self):
        self.driver = SimpleNamespace(steps=11, base=SimpleNamespace(
            _obstacle_side=1., _frame=lambda obs: obs[-1]))
        self.obj: tuple[float, float, float] | None = (47., 39., 35.)
        self._last_steer = None
        self.centers = {30: 28., 42: 35., 54: 43.}
        self.action = np.array([.042, .2, .1], np.float32)

    def reset(self, observation):
        pass

    def act(self, observation):
        return self.action.copy()

    def last_step_diagnostics(self):
        return dict(near_object=self.obj, pixel_speed=44., road_centers=self.centers,
                    steering_terms=dict(reconstruction_valid=True, unidentifiable=[],
                    raw_terms=dict(selected_road_position=.022 * (self.centers[42]-42),
                                   selected_lookahead=.018 * (self.centers[42]-self.centers[54]), damping=0.)))


def test_exact_default_without_combined_conflict():
    for condition in ('no_object', 'straight', 'same_direction', 'prefix'):
        v2 = V2()
        if condition == 'no_object':
            v2.obj = None
        elif condition == 'straight':
            v2.centers = {30: 42., 42: 42., 54: 42.}
        elif condition == 'same_direction':
            v2.driver.base._obstacle_side = -1.
        else:
            v2.driver.steps = 10
        candidate = CollisionPriorityAgent(v2)
        np.testing.assert_array_equal(candidate.act(pixels()), v2.action)
        assert candidate.last['priority_mode'] == 'v2'


def test_collision_priority_changes_only_steer_and_keeps_actual_history():
    v2 = V2()
    candidate = CollisionPriorityAgent(v2)
    action = candidate.act(pixels())
    assert candidate.last['priority_mode'] == 'avoid'
    assert action[0] > v2.action[0]
    np.testing.assert_array_equal(action[1:], v2.action[1:])
    assert v2._last_steer == float(action[0])
    assert v2.driver.last_command == tuple(action)
    assert candidate.last['priority_clearance'] > candidate.last['priority_base_clearance']
    assert 'steering_terms' not in candidate.last


def test_boundary_is_not_a_hard_veto():
    v2 = V2()
    narrow = pixels()
    narrow[:, 20:70, 45:] = .1
    candidate = CollisionPriorityAgent(v2)
    assert candidate.act(narrow)[0] > v2.action[0]


def test_budget_forces_recovery_using_verified_limit():
    assert NEGATIVE_REWARD_LIMIT == 100
    candidate = CollisionPriorityAgent(V2())
    candidate.age, candidate.side = NEGATIVE_REWARD_LIMIT // 4 - 1, 1.
    action = candidate.act(pixels())
    assert candidate.last['priority_mode'] == 'budget_recovery'
    assert action[0] == np.float32(candidate.last['priority_road_steer'])
    assert not candidate.last['priority_reward_counter_observable']


def test_recovery_ends_after_road_alignment_and_reset_clears_state():
    v2 = V2()
    candidate = CollisionPriorityAgent(v2)
    candidate.act(pixels())
    v2.obj = None
    v2.centers = {30: 42., 42: 42., 54: 42.}
    candidate.box = None
    assert candidate.act(pixels(None))[0] == 0.
    assert candidate.last['priority_mode'] == 'recover'
    candidate.act(pixels(None))
    assert candidate.age == 0
    candidate.reset(pixels(None))
    assert candidate.box is None and candidate.last == {}


def test_signed_sweep_detects_overlap_and_lateral_separation():
    box = obstacle_box(pixels(42, 50)[-1], (50., 42., 42.))
    assert box is not None
    assert arc_clearance(box, 0., 10.) < 0
    box[:, 0] += 20
    assert arc_clearance(box, 0., 10.) > 0


def test_optical_approach_gate_requires_time_and_nominal_overlap():
    v2 = V2()
    candidate = CollisionPriorityAgent(v2)
    candidate.previous_object = (43., 39., 35.)
    candidate.act(pixels())
    assert candidate.last['priority_motion_risk']
    candidate.reset(pixels())
    candidate.previous_object = (46., 39., 35.)
    candidate.act(pixels())
    assert not candidate.last['priority_motion_risk']


def test_missing_detection_is_not_immediate_clearance():
    v2 = V2()
    candidate = CollisionPriorityAgent(v2)
    candidate.act(pixels())
    assert candidate.box is not None
    v2.obj = None
    candidate.act(pixels(None))
    assert candidate.box is not None
    assert candidate.missing == 1
