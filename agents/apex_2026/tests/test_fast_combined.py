"""Selection and actual steering synchronization between camera controllers."""
from types import SimpleNamespace
import numpy as np
try:
    from agents.apex_2026.fast_combined_agent import Agent
except ImportError:
    from agents.apex_2026.fast_early_circle_agent import Agent


class Stub:
    def __init__(self, action, target, plan=None):
        self.action = np.asarray(action, np.float32)
        self.last_target = target
        self.last_pass_plan = plan
        self.last_steer = 0.
        self.calls = 0
        self._metric = SimpleNamespace(last_steer=0., last_target=target)
        self._robust = SimpleNamespace(_last_steer=0.)

    def act(self, observation):
        self.calls += 1
        return self.action.copy()

    def reset(self, observation=None):
        self.last_pass_plan = None
        self.calls = 0


def controls(plan):
    controller = Agent()
    controller.fast = Stub([.2, .6, 0], 90.)
    controller.hazard = Stub([-.1, 0, .2], 55., plan)
    return controller


def test_rejected_reference_selects_fast_and_both_observe():
    controller = controls(None)
    action = controller.act(np.zeros((4, 84, 84), np.float32))
    np.testing.assert_array_equal(action, controller.fast.action)
    assert controller.fast.calls == controller.hazard.calls == 1
    assert controller.mode == 'fast'


def test_current_accepted_pass_selects_hazard_and_syncs_actual_steering():
    controller = controls({'valid': True})
    action = controller.act(np.zeros((4, 84, 84), np.float32))
    np.testing.assert_array_equal(action, controller.hazard.action)
    assert controller.mode == 'hazard'
    steer = float(action[0])
    assert controller.last_steer == controller.fast.last_steer == steer
    assert controller.hazard.last_steer == controller.hazard._metric.last_steer == steer
    assert controller.hazard._robust._last_steer == steer
    assert controller.last_target == 55.
    assert controller.fast.last_target == 90.
    assert controller.hazard._metric.last_target == 55.


def test_reset_clears_both_references_and_public_memory():
    controller = controls({'valid': True})
    controller.act(np.zeros((4, 84, 84), np.float32))
    controller.reset()
    assert controller.hazard.last_pass_plan is None
    assert controller.last_steer == 0 and controller.mode == 'fast'
    assert controller.fast.calls == controller.hazard.calls == 0


def test_real_embedded_controllers_preserve_camera_and_reset_contract():
    from agents.apex_2026.tests.test_fast_envelope import camera
    controller = Agent()
    observation = camera(speed=60, curvature=.003)
    original = observation.copy()
    action = controller.act(observation)
    assert action.dtype == np.float32 and action.shape == (3,)
    assert np.isfinite(action).all()
    np.testing.assert_array_equal(observation, original)
    assert np.isfinite(controller.act(np.full((4, 84, 84), np.nan, np.float32))).all()
    controller.reset()
    np.testing.assert_array_equal(controller.act(observation), Agent().act(observation))
