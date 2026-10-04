"""Compute-only standalone batching preserves the frozen camera/controller."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from agents.apex_2026.research.speed_20261005.predictive_batch_model import pack_states
from agents.apex_2026.research.speed_20261005.predictive_batch_model import predict_batch as reference_batch
from agents.apex_2026.tests.test_fast_envelope import camera
from agents.apex_2026.tests.test_predictive_control import legal_state, straight_geometry
from agents.apex_2026.tests.test_predictive_batch_control import compare

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE/'fast_predictive_batch_agent.py'
PARENT = BASE/'fast_predictive_v2_agent.py'
BUILDER = BASE/'research/speed_20261005/build_predictive_batch.py'
HISTORY = Path(__file__).parent/'fixtures'/'predictive-batch-history.npz'
PARENT_SHA = 'b70af66e0ba39ea975d53909ec6d7a19df2a73005a036465d71dd7ac8caeff8a'


def load(path):
    spec = importlib.util.spec_from_file_location('predictive_batch_standalone_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def candidate():
    return load(SOURCE if SOURCE.exists() else PARENT)


def warm(module, step, history):
    agent = module.Agent()
    for index in range(step):
        observation = history['observations'][index]
        module._BaseRearClearAgent.act(agent, observation)
        agent._observer.observe(agent._frame(observation))
        action = history['actions'][index].copy()
        agent._observer.advance(action)
        agent._predictive_previous_action = action.copy()
        agent.last_steer = float(action[0])
    return agent


def assert_state_equal(left, right):
    for key, expected in right.items():
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(left[key], expected)
        else:
            assert left[key] == expected


def test_namespace_captures_scalar_planner_without_replacing_observer_predict_step():
    module, parent = candidate(), load(PARENT)
    assert hasattr(module, 'FixedBatchControlPlanner'), 'batch control export missing'
    assert module.FixedControlPlanner is module._BudgetedBatchControlPlanner
    assert module.FixedBatchControlPlanner.__bases__ == (module._ScalarPlannerReference,)
    assert module.Agent.__bases__ == (module._PredictiveV2Reference,)
    assert module._PredictiveV2Reference.act is module.Agent.act
    assert module._PredictiveV2Reference.reset is module.Agent.reset
    left, right = legal_state(70.), legal_state(70.)
    for command in ([.06, .3, 0.], [-.03, 0., 1.]):
        for _ in range(4):
            left, diagnostics = module.predict_step(left, command)
            right, expected = parent.predict_step(right, command)
            assert_state_equal(left, right)
            assert diagnostics == expected
    exported = module.CameraObserver(module._fixed_camera_calibration())
    scalar = parent.CameraObserver(parent._fixed_camera_calibration())
    for speed, action in ((0., [0., 1., 0.]), (5., [.06, .3, 0.]), (10., [-.03, 0., 1.])):
        exported.observe(camera(speed=speed)[-1])
        scalar.observe(camera(speed=speed)[-1])
        left, _ = exported.advance(action)
        right, _ = scalar.advance(action)
        assert_state_equal(left, right)


def test_exported_batch_model_recurrence_matches_frozen_kernel_and_no_aliases():
    module = candidate()
    assert hasattr(module, 'predict_batch'), 'batch physics export missing'
    original = pack_states([legal_state(0.), legal_state(70.), legal_state(100.)])
    left, right = copy.deepcopy(original), copy.deepcopy(original)
    commands = np.asarray([[.24, 1., 0.], [-.06, 0., .55], [.03, 0., 1.]], np.float32)
    for _ in range(16):
        left, diagnostics = module.predict_batch(left, commands)
        right, expected = reference_batch(right, commands)
        assert_state_equal(left, right)
        for key in expected:
            np.testing.assert_array_equal(diagnostics[key], expected[key])
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            assert not np.shares_memory(left[key], value)


def test_offline_builder_is_exact_append_only_and_static_inference_imports_allowed():
    assert BUILDER.exists(), 'offline compute-only builder missing'
    assert SOURCE.exists()
    assert load(BUILDER).build_text(BASE.parents[1]) == SOURCE.read_text()
    assert SOURCE.read_bytes().startswith(PARENT.read_bytes())
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    tree = ast.parse(SOURCE.read_text())
    imports = {name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names}
    assert imports <= {'numpy', 'math', 'copy', 'time'}
    assert not any(isinstance(node, ast.ImportFrom) for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                   node.func.id in ('open', '__import__', 'eval', 'exec') for node in ast.walk(tree))


def test_parent_called_once_and_only_final_emitted_action_advances_observer(monkeypatch):
    module = candidate()
    assert hasattr(module, 'FixedBatchControlPlanner'), 'batch control missing'
    chosen = np.asarray([.03, .6, 0.], np.float32)
    calls, advances = [], []
    original = module._BaseRearClearAgent.act
    def parent_once(self, observation):
        calls.append('parent')
        return original(self, observation)
    monkeypatch.setattr(module._BaseRearClearAgent, 'act', parent_once)
    monkeypatch.setattr(module.FixedControlPlanner, 'plan', lambda self, states, previous:
                        {'feasible': True, 'action': chosen.copy(), 'reason': 'feasible'})
    agent = module.Agent()
    advance = agent._observer.advance
    def emitted_once(action):
        advances.append(np.asarray(action).copy())
        return advance(action)
    monkeypatch.setattr(agent._observer, 'advance', emitted_once)
    observation = camera(speed=40.)
    before = observation.copy()
    np.testing.assert_array_equal(agent.act(observation), chosen)
    assert calls == ['parent'] and len(advances) == 1
    np.testing.assert_array_equal(advances[0], chosen)
    assert agent._observer.actions_seen == 1
    np.testing.assert_array_equal(observation, before)


def test_geometry_deadline_is_guarded_before_after_and_final_selection(monkeypatch):
    module = candidate()
    assert hasattr(module.Agent, 'check_planning_budget'), 'owner budget callback missing'
    agent = module.Agent()
    agent._predictive_deadline = 10.
    tick = [9.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: tick[0])
    geometry = straight_geometry()
    original = geometry.evaluate
    def crosses_deadline(*args, **kwargs):
        result = original(*args, **kwargs)
        tick[0] = 11.
        return result
    geometry.evaluate = crosses_deadline
    planner = module.FixedControlPlanner(agent._predictive_step, geometry)
    with pytest.raises(module._PlanningBudgetExceeded):
        planner._geometry(np.zeros((1, 2)), np.zeros(1), np.zeros((1, 2)))
    tick[0] = 9.
    def last_score_crosses(self, states, previous):
        tick[0] = 11.
        return {'feasible': True, 'action': np.zeros(3, np.float32), 'reason': 'feasible'}
    monkeypatch.setattr(module.FixedBatchControlPlanner, 'plan', last_score_crosses)
    result = planner.plan([legal_state(70.)], np.zeros(3, np.float32))
    assert result['action'] is None and result['reason'] == 'planning_error'
    assert '_PlanningBudgetExceeded' in result['error']


def test_deadline_fallback_advances_scalar_history_once_and_reset_is_identical():
    module, parent = candidate(), load(PARENT)
    agent, scalar = module.Agent(planning_budget_s=0.), parent.Agent(planning_budget_s=0.)
    observation = camera(speed=40.)
    np.testing.assert_array_equal(agent.act(observation), scalar.act(observation))
    assert agent.predictive_status == scalar.predictive_status == 'fallback_timeout'
    assert agent._observer.actions_seen == scalar._observer.actions_seen == 1
    assert_state_equal(agent._observer.state, scalar._observer.state)
    agent.reset()
    scalar.reset()
    assert_state_equal(agent._observer.state, scalar._observer.state)
    np.testing.assert_array_equal(agent._predictive_previous_action, scalar._predictive_previous_action)


def test_grass_island_unknown_input_and_scenario_copies_remain_exact():
    module, parent = candidate(), load(PARENT)
    agent, scalar = module.Agent(), parent.Agent()
    frame = np.full((84, 84), .4, np.float32)
    frame[73:] = 0.
    frame[60:66, 41:44] = .1
    frame[40:45, 42:47] = .63
    field, grass = agent._predictive_field(frame, [])
    expected_field, expected_grass = scalar._predictive_field(frame, [])
    np.testing.assert_array_equal(field, expected_field)
    np.testing.assert_array_equal(grass, expected_grass)
    assert field[42, 44] == 0. and grass[42, 44]
    before = legal_state(70.)
    states, expected = agent._predictive_scenarios(before), scalar._predictive_scenarios(before)
    for left, right in zip(states, expected):
        assert_state_equal(left, right)
    states[0]['velocity'][0] = 99.
    assert before['velocity'][0] == 0. and states[1]['velocity'][0] != 99.
    for observation in (camera(speed=10.), np.full((4, 84, 84), np.nan, np.float32)):
        result = agent.act(observation)
        reference = scalar.act(observation)
        np.testing.assert_array_equal(result, reference)
        assert result.dtype == np.float32 and np.isfinite(result).all()
        assert np.all(result >= [-1., 0., 0.]) and np.all(result <= 1.)


@pytest.mark.parametrize('step', (20, 40, 60))
def test_actual_saved_history_command_and_result_match_v2_scalar(step):
    assert SOURCE.exists() and HISTORY.exists(), 'candidate or camera-only history fixture missing'
    history = np.load(HISTORY)
    module, parent = candidate(), load(PARENT)
    agent, scalar = warm(module, step, history), warm(parent, step, history)
    observation = history['observations'][step]
    before = observation.copy()
    result = agent.act(observation)
    reference = scalar.act(observation)
    np.testing.assert_array_equal(result, reference)
    assert agent.predictive_status == scalar.predictive_status
    if scalar.predictive_result:
        compare(agent.predictive_result, scalar.predictive_result)
        assert agent.predictive_model_calls == agent.predictive_result['batched_model_rows']
    assert_state_equal(agent._observer.state, scalar._observer.state)
    assert agent._observer.actions_seen == scalar._observer.actions_seen == step+1
    np.testing.assert_array_equal(observation, before)
