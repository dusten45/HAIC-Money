"""Both independent camera fixes must coexist in one control state."""
import numpy as np
try:
    from agents.apex_2026.fast_arc_memory_agent import Agent
except ImportError:
    from agents.apex_2026.fast_arc_clear_agent import Agent
from agents.apex_2026.tests.test_fast_memory_corridor import image, prepare, route
from agents.apex_2026.tests.test_fast_arc_guard import saved_controller, emitted_envelope_depth
from agents.apex_2026.research.speed_20261005.graph_geometry import distance_to_path
from agents.apex_2026.tests.test_fast_envelope import camera


def test_saved_missed_circle_keeps_memory_constraint_and_single_transport():
    for step in (95, 97):
        controller = prepare(Agent, step)
        frame = image(step)
        x, y = controller.pass_x, controller.pass_y - .08 * controller._speed(frame)
        turn = .08 * controller._yaw(frame)
        expected = [x*np.cos(turn)-y*np.sin(turn), x*np.sin(turn)+y*np.cos(turn)]
        road, path, circles = route(controller, frame)
        assert circles == []
        np.testing.assert_allclose([controller.pass_x, controller.pass_y], expected, atol=1e-12)
        gap = distance_to_path(np.asarray(expected)[None], np.column_stack((path, road[0])))[0]
        assert gap >= 3.3


def test_saved_clear_bend_retains_arc_guard_and_physical_target():
    for step in (77, 78):
        controller, observation = saved_controller(Agent, step)
        action = controller.act(observation)
        assert emitted_envelope_depth(controller, observation, action[0]) >= 1.9
        assert action[0] < -.3 and action[1] == 0
        curve = abs(np.tan(float(action[0])) / controller.WHEELBASE)
        assert controller.last_target <= np.sqrt(controller.lateral_accel/curve) + 1e-6


def test_combined_state_preserves_pixels_invalid_recovery_and_reset():
    controller = Agent()
    observation = camera(speed=60, curvature=.003)
    original = observation.copy()
    action = controller.act(observation)
    assert action.dtype == np.float32 and np.isfinite(action).all()
    np.testing.assert_array_equal(observation, original)
    assert np.isfinite(controller.act(np.full((4, 84, 84), np.nan, np.float32))).all()
    controller.reset()
    np.testing.assert_array_equal(controller.act(observation), Agent().act(observation))
