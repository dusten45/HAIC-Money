"""Contract checks for the standalone camera-only robust racing candidate."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "agents" / "apex_2026" / "robust_agent.py"


def load_module():
    assert SOURCE.is_file(), "standalone robust candidate must exist"
    spec = importlib.util.spec_from_file_location("apex_robust_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def straight_observation(speed=0.0):
    observation = np.full((4, 84, 84), 0.20, dtype=np.float32)
    observation[:, :74, 30:54] = 0.40
    observation[:, 74:] = 0.0
    # Rendered white HUD pixels have mass 0.27+0.085*v. This synthetic
    # antialiased bar preserves that independently known calibration.
    mass = 0.27 + 0.085 * speed
    whole = int(mass)
    flat = observation[:, 74:83, 10:13].reshape(4, -1).copy()
    flat[:, :whole] = 1.0
    if whole < flat.shape[1]:
        flat[:, whole] = mass - whole
    observation[:, 74:83, 10:13] = flat.reshape(4, 9, 3)
    return observation


def test_candidate_loads_in_isolated_folder_without_model(tmp_path):
    load_module()
    target = tmp_path / "robust_agent.py"
    target.write_bytes(SOURCE.read_bytes())
    code = "import robust_agent; a=robust_agent.Agent(model_path='absent.pt'); print(type(a).__name__)"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Agent"
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert not imports & {"agent", "torch", "gymnasium", "core", "training", "stable_baselines3", "cv2"}


def test_reset_reproduces_initial_action_and_invalid_frames_are_bounded():
    module = load_module()
    agent = module.Agent()
    observation = straight_observation(35)
    first = agent.act(observation)
    for _ in range(8):
        agent.act(observation)
    agent.reset(observation)
    np.testing.assert_array_equal(agent.act(observation), first)
    for value in (np.zeros((4, 84, 84), dtype=np.float32), None, np.full((4, 84, 84), np.nan)):
        action = agent.act(value)
        assert action.shape == (3,) and action.dtype == np.float32
        assert np.isfinite(action).all()
        assert np.all(action >= [-1, 0, 0]) and np.all(action <= [1, 1, 1])


def test_hud_speed_above80_reaches_feedback_and_brakes_for_target():
    module = load_module()
    agent = module.Agent(cruise_speed=85.0, propulsion=0.8, corner_boost=0.5)
    slow = straight_observation(35)
    fast = straight_observation(110)
    assert abs(agent._estimate_speed(fast[-1]) - 110) < 0.1
    assert agent.act(slow)[1] >= 0.6
    agent.reset(fast)
    action = agent.act(fast)
    assert action[1] == 0.0 and action[2] > 0.0


def test_clear_corner_underspeed_gets_more_propulsion_without_changing_steer():
    module = load_module()
    observation = straight_observation(30)
    frame = observation[-1]
    for row in range(20, 65):
        frame[row] = 0.20
        center = round(41.5 + (54-row) * 0.17)
        frame[row, center-12:center+12] = 0.40
    observation[:] = frame
    reference = module._ClearRoadRow42DropoutController()
    reference.reset(observation)
    expected = reference.act(observation)
    candidate = module.Agent(propulsion=0.8, corner_boost=0.5)
    actual = candidate.act(observation)
    assert actual[1] >= 0.40
    assert abs(float(actual[0])-float(expected[0])) < 1e-7


def test_optional_heading_preview_anticipates_visible_clear_corner():
    module = load_module()
    observation = straight_observation(35)
    frame = observation[-1]
    for row in range(20, 65):
        frame[row] = 0.20
        center = round(41.5 + (54-row) * 0.17)
        frame[row, center-12:center+12] = 0.40
    observation[:] = frame
    reference = module.Agent(cruise_speed=68, propulsion=.65, corner_boost=.22)
    preview = module.Agent(cruise_speed=68, propulsion=.65, corner_boost=.22,
        corner_target=65, heading_preview=.5)
    for _ in range(4):
        reference_action = reference.act(observation)
        preview_action = preview.act(observation)
    assert preview_action[0] > reference_action[0] + .01
    assert preview._pace_effective_target == 65


def test_optional_fast_corner_keeps_inherited_hazard_command():
    module = load_module()
    observation = straight_observation(35)
    observation[:, 39:44, 39:44] = .70
    reference = module.Agent(cruise_speed=68, propulsion=.65, corner_boost=.22)
    preview = module.Agent(cruise_speed=68, propulsion=.65, corner_boost=.22,
        corner_target=65, heading_preview=.5)
    for _ in range(4):
        expected = reference.act(observation)
        actual = preview.act(observation)
        np.testing.assert_array_equal(actual, expected)
