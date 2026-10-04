"""Harness contract tests use synthetic environments and consume no track cells."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def harness():
    path = Path(__file__).with_name('evaluate.py')
    assert path.exists(), 'evaluation harness has not been implemented'
    spec = importlib.util.spec_from_file_location('apex_evaluate_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Environment:
    def __init__(self, output, finish=True, reset_error=False):
        self.output = output
        self.finish = finish
        self.reset_error = reset_error
        self.unwrapped = SimpleNamespace(t=1.02, finish_time_s=None,
                                        finish_qualified_time_s=None)
        self.observation = np.ones((4, 84, 84), dtype=np.float32)
        self.closed = False
        self.steps = 0

    def reset(self, **kwargs):
        assert json.loads(self.output.read_text())['status'] == 'started'
        assert kwargs == {'seed': 42, 'options': {'track_id': 1}}
        if self.reset_error:
            raise RuntimeError('reset failed')
        return self.observation, {}

    def step(self, action):
        assert np.all(self.observation == 1), 'policy mutated environment observation'
        self.steps += 1
        self.unwrapped.t = 1.10
        if self.finish:
            self.unwrapped.finish_time_s = 1.075
            self.unwrapped.finish_qualified_time_s = 1.06
        return self.observation, 2., True, False, {'damage': .1, 'collision': True}

    def _calculate_progress(self):
        return .98

    def close(self):
        self.closed = True


def policy_file(tmp_path, body='observation[:] = 0\n        return [0, 0.5, 0]'):
    path = tmp_path / 'agent.py'
    path.write_text('class Agent:\n    def act(self, observation):\n        ' + body + '\n')
    return path


def test_finish_uses_actual_crossing_timestamp_and_copies_observation(tmp_path):
    module = harness()
    output = tmp_path / 'result.json'
    env = Environment(output)
    result = module.run_episode(policy_file(tmp_path), 1, 42, output,
                                env_factory=lambda: env, trace=tmp_path / 'trace.jsonl')
    assert result['status'] == 'completed'
    assert result['finished'] is True
    assert result['lapTimeMs'] == 55
    assert result['steps'] == 1
    assert result['collision_steps'] == 1
    assert result['progress'] == .98
    assert result['provenance']['agent_sha256']
    assert env.closed


def test_reset_failure_is_error_not_dnf_and_closes_environment(tmp_path):
    module = harness()
    output = tmp_path / 'result.json'
    env = Environment(output, reset_error=True)
    with pytest.raises(RuntimeError, match='reset failed'):
        module.run_episode(policy_file(tmp_path), 1, 42, output, env_factory=lambda: env)
    result = json.loads(output.read_text())
    assert result['status'] == 'error'
    assert result.get('finished') is None
    assert env.closed


def test_tenth_consecutive_invalid_action_retires_without_step(tmp_path):
    module = harness()
    output = tmp_path / 'result.json'
    env = Environment(output, finish=False)
    def step(action):
        env.steps += 1
        return env.observation, 0., False, False, {}
    env.step = step
    result = module.run_episode(policy_file(tmp_path, 'return [float("nan"), 0, 0]'),
                                1, 42, output, env_factory=lambda: env)
    assert result['retire_reason'] == 'invalid_action'
    assert result['invalid_actions'] == 10
    assert result['steps'] == env.steps == 9
    assert result['lapTimeMs'] is None


def test_policy_sibling_import_and_config(tmp_path):
    module = harness()
    (tmp_path / 'unique_apex_helper.py').write_text('VALUE = 17\n')
    path = tmp_path / 'agent.py'
    path.write_text('from unique_apex_helper import VALUE\nclass Agent:\n'
                    '    def __init__(self, gain): self.value = gain + VALUE\n')
    assert module.load_agent(path, {'gain': 3}).value == 20


def test_action_clipping_shape_and_exception():
    module = harness()
    action, valid = module.safe_act(SimpleNamespace(act=lambda obs: [-2, 2, -.1]), None)
    assert valid
    np.testing.assert_array_equal(action, [-1, 1, 0])
    _, valid = module.safe_act(SimpleNamespace(act=lambda obs: [1, 2]), None)
    assert not valid


def test_existing_receipt_is_never_overwritten(tmp_path):
    module = harness()
    output = tmp_path / 'result.json'
    output.write_text('{"status":"started"}')
    with pytest.raises(FileExistsError):
        module.run_episode(policy_file(tmp_path), 1, 42, output)
    assert output.read_text() == '{"status":"started"}'


def test_episode_horizon_is_2000_actions(tmp_path):
    module = harness()
    output = tmp_path / 'result.json'
    env = Environment(output, finish=False)
    def step(action):
        env.steps += 1
        return env.observation, 0., False, False, {}
    env.step = step
    result = module.run_episode(policy_file(tmp_path, 'return [0, 0, 0]'), 1, 42,
                                output, env_factory=lambda: env)
    assert result['steps'] == 2000
    assert result['retire_reason'] == 'max_steps'
    assert result['finished'] is False


def test_timeout_returns_noop_and_reset_timeout_raises():
    from threading import Event
    module = harness()
    release = Event()
    agent = SimpleNamespace(act=lambda obs: release.wait(), reset=lambda obs: release.wait())
    try:
        action, valid = module.safe_act(agent, None, timeout_sec=.001)
        assert not valid
        np.testing.assert_array_equal(action, [0, 0, 0])
        with pytest.raises(TimeoutError):
            module.safe_reset(agent, None, timeout_sec=.001)
    finally:
        release.set()
