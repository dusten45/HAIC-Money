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


def test_resources_eligible_and_peak_memory_recorded(tmp_path, monkeypatch):
    m = harness()
    assert hasattr(m, 'peak_memory_mib'), 'peak memory monitor missing'
    monkeypatch.setattr(m, 'peak_memory_mib', lambda: 1024.)
    out = tmp_path/'result.json'
    r = m.run_episode(policy_file(tmp_path), 1, 42, out, env_factory=lambda: Environment(out))
    assert r['resource_eligible'] is True
    assert 0 <= r['import_constructor_time_s'] <= 10
    assert r['reset_time_s'] == 0
    assert r['peak_memory_mib'] == 1024
    assert r['act_timeout_count'] == r['reset_timeout_count'] == r['import_constructor_timeout_count'] == 0


def test_excess_memory_retains_driving_result(tmp_path, monkeypatch):
    m = harness()
    assert hasattr(m, 'peak_memory_mib'), 'peak memory monitor missing'
    monkeypatch.setattr(m, 'peak_memory_mib', lambda: 1024.1)
    out = tmp_path/'result.json'
    r = m.run_episode(policy_file(tmp_path), 1, 42, out, env_factory=lambda: Environment(out))
    assert r['finished'] and r['lapTimeMs'] == 55
    assert r['resource_eligible'] is False


def test_constructor_timeout_is_incomplete_and_does_not_reset(tmp_path, monkeypatch):
    from threading import Event
    m = harness()
    assert hasattr(m, 'IMPORT_TIMEOUT_SECONDS'), 'constructor bound missing'
    monkeypatch.setattr(m, 'IMPORT_TIMEOUT_SECONDS', .001)
    release = Event()
    monkeypatch.setattr(m, 'load_agent', lambda *args: release.wait())
    out = tmp_path/'result.json'
    try:
        with pytest.raises(TimeoutError):
            m.run_episode(policy_file(tmp_path), 1, 42, out,
                          env_factory=lambda: pytest.fail('reset after constructor timeout'))
    finally:
        release.set()
    r = json.loads(out.read_text())
    assert r['status'] == 'error' and r.get('finished') is None
    assert r['resource_eligible'] is False
    assert r['import_constructor_timeout_count'] == 1


def test_timeout_details_distinguish_invalid_action():
    from threading import Event
    m = harness()
    details = {}
    _, valid = m.safe_act(SimpleNamespace(act=lambda obs: [1,2]), None, details=details)
    assert not valid and details['timed_out'] is False
    release = Event()
    try:
        _, valid = m.safe_act(SimpleNamespace(act=lambda obs: release.wait()), None,
                              timeout_sec=.001, details=details)
        assert not valid and details['timed_out'] is True
    finally:
        release.set()


def test_proc_peak_memory_is_current_process_hwm(tmp_path):
    m = harness()
    assert hasattr(m, 'peak_memory_mib'), 'peak memory monitor missing'
    path = tmp_path/'status'
    path.write_text('Name:\ttest\nVmRSS:\t1024 kB\nVmHWM:\t2048 kB\n')
    assert m.peak_memory_mib(path) == 2.
    path.write_text('Name:\tno_hwm\n')
    assert m.peak_memory_mib(path) is None


def test_policy_diagnostics_sanitized_and_errors_do_not_change_driving(tmp_path):
    m = harness()
    p = policy_file(tmp_path, 'self.diagnostics = {"array": __import__("numpy").array([1.,float("nan")])}\n        return [0,0,0]')
    out=tmp_path/'result.json';trace=tmp_path/'trace.jsonl'
    r=m.run_episode(p,1,42,out,env_factory=lambda: Environment(out),trace=trace)
    row=json.loads(trace.read_text())
    assert row['policy_diagnostics'] == {'array':[1.,None]}
    assert r['finished']
    p.write_text('class Agent:\n    def act(self, obs): return [0,0,0]\n    def last_step_diagnostics(self): raise ValueError("diagnostic broke")\n')
    out=tmp_path/'error-diag.json';trace=tmp_path/'error-diag.jsonl'
    r=m.run_episode(p,1,42,out,env_factory=lambda: Environment(out),trace=trace)
    assert r['finished']
    assert 'diagnostic_error' in json.loads(trace.read_text())['policy_diagnostics']


def test_reset_timeout_receipt_and_close(tmp_path, monkeypatch):
    from threading import Event
    m=harness();monkeypatch.setattr(m,'POLICY_TIMEOUT_SECONDS',.001)
    release=Event();agent=SimpleNamespace(reset=lambda obs: release.wait())
    monkeypatch.setattr(m,'load_agent',lambda *args: agent)
    out=tmp_path/'result.json';env=Environment(out)
    try:
        with pytest.raises(TimeoutError):
            m.run_episode(policy_file(tmp_path),1,42,out,env_factory=lambda:env)
    finally:
        release.set()
    r=json.loads(out.read_text())
    assert r['status']=='error' and r['reset_timeout_count']==1
    assert r['reset_time_s']>0 and not r['resource_eligible'] and env.closed


def test_act_timeout_preserves_raw_finish_and_marks_ineligible(tmp_path, monkeypatch):
    from threading import Event
    m=harness();monkeypatch.setattr(m,'POLICY_TIMEOUT_SECONDS',.001)
    release=Event();agent=SimpleNamespace(act=lambda obs: release.wait())
    monkeypatch.setattr(m,'load_agent',lambda *args:agent)
    out=tmp_path/'result.json'
    try:
        r=m.run_episode(policy_file(tmp_path),1,42,out,env_factory=lambda:Environment(out))
    finally:
        release.set()
    assert r['status']=='completed' and r['finished'] and r['lapTimeMs']==55
    assert r['act_timeout_count']==1 and not r['resource_eligible']
