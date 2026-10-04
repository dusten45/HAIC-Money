"""Camera-local route smoothing must retain explicit geometric constraints."""
import importlib.util
from pathlib import Path

import numpy as np

from agents.apex_2026.tests.test_fast_envelope import camera


def agent_type():
    root = Path(__file__).resolve().parents[1]
    source = root / "fast_corridor_agent.py"
    if not source.exists():
        source = root / "fast_path_agent.py"
    spec = importlib.util.spec_from_file_location("fast_corridor_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def route(controller, observation):
    frame = observation[-1]
    road = controller._road(frame)
    path, imminent, distance = controller._route(
        road[0], road[1], controller._circles(frame, road), 70., 0.)
    return road, path, imminent, distance


def test_clear_bend_uses_available_corridor_to_reduce_curvature():
    controller = agent_type()()
    road, path, _, _ = route(controller, camera(curvature=.012, speed=70))
    spacing = road[0][1] - road[0][0]
    old_energy = np.sum(np.diff(road[1], n=2) ** 2) / spacing ** 4
    new_energy = np.sum(np.diff(path, n=2) ** 2) / spacing ** 4
    assert new_energy < .8 * old_energy
    assert abs(float(np.interp(0., road[0], path))) < .15


def test_obstacle_relative_pass_keeps_sampled_center_clearance():
    controller = agent_type()()
    observation = camera(speed=70, obstacle=(44, 27))
    road, path, imminent, distance = route(controller, observation)
    measured = controller._circles(observation[-1], road)[0]
    sample = np.linspace(max(0., road[0][0]), road[0][-1], 500)
    x = np.interp(sample, road[0], path)
    separation = np.sqrt((sample - measured[0]) ** 2 + (x - measured[1]) ** 2)
    assert separation.min() > 3.55
    assert not imminent and distance is not None


def test_infeasible_circle_corridor_uses_existing_recovery_route():
    controller = agent_type()()
    observation = camera(speed=70)
    road = controller._road(observation[-1])
    path, _, _ = controller._route(road[0], road[1], [(15., -3.5), (15., 3.5)], 70., 0.)
    assert np.isfinite(path).all()
    assert not getattr(controller, "corridor_used", False)


def test_finite_action_reset_and_pixel_preservation():
    controller = agent_type()()
    observation = camera(curvature=.01, speed=45)
    original = observation.copy()
    action = controller.act(observation)
    np.testing.assert_array_equal(observation, original)
    assert action.dtype == np.float32 and np.isfinite(action).all()
    invalid = controller.act(np.full((4, 84, 84), np.nan, dtype=np.float32))
    assert np.isfinite(invalid).all()
    assert np.all(invalid >= [-1, 0, 0]) and np.all(invalid <= 1)
    controller.reset()
    np.testing.assert_array_equal(controller.act(observation), agent_type()().act(observation))
