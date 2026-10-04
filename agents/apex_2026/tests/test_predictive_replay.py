"""Replay integrity must reject changed inputs and near-equal actions."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents.apex_2026.evaluate import ENV_FILES, MANDATORY_CELLS


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'agents/apex_2026/research/speed_20261005/predictive_replay.py'


def helper():
    assert SOURCE.exists(), 'Source-bound predictive replay helper is not implemented'
    spec = importlib.util.spec_from_file_location('predictive_replay_tests', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_files(tmp_path, text=None):
    source = tmp_path / 'agent.py'
    source.write_text(text or 'class Agent:\n    def __init__(self, **kwargs):\n        self.parameters = kwargs\n')
    actions = np.array([[.1, .5, 0.], [.2, 0., .4]], np.float32)
    rows = []
    for track, seed in MANDATORY_CELLS:
        rows.append(dict(track_id=track, seed=seed, finished=True, lap_time_ms=160,
                         progress=1., collision_count=0, damage=0., retire_reason=None,
                         steps=2, offtrack_samples=0, partial_offtrack_samples=0,
                         source_sha256=digest(source), parameters={'yaw_gain': .15},
                         action_trace_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                         error=None))
    freeze = dict(source_sha256=digest(source), parameters={'yaw_gain': .15},
                  suite='mandatory', cells=[list(c) for c in MANDATORY_CELLS],
                  max_steps=700,
                  environment_sha256={name: digest(ROOT/name) for name in ENV_FILES})
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'freeze': freeze, 'rows': rows}))
    return source, receipt, actions


def test_binding_reads_actual_source_bytes_and_uses_frozen_parameters(tmp_path):
    h = helper()
    source, receipt, _ = frozen_files(tmp_path)
    binding = h.validate_binding(source, receipt, 1, 516237)
    assert binding['source_sha256'] == digest(source)
    assert binding['receipt_sha256'] == digest(receipt)
    agent, latency = h.load_bound_agent(binding)
    assert agent.parameters == {'yaw_gain': .15}
    assert latency >= 0.


def test_changed_source_is_rejected_before_top_level_code_can_run(tmp_path):
    h = helper()
    marker = tmp_path / 'executed'
    text = f'from pathlib import Path\nPath({str(marker)!r}).write_text("executed")\nclass Agent:\n    pass\n'
    source, receipt, _ = frozen_files(tmp_path, text)
    binding = h.validate_binding(source, receipt, 1, 516237)
    source.write_text(text + '\n# changed after validation\n')
    with pytest.raises(ValueError, match='source.*changed'):
        h.load_bound_agent(binding)
    assert not marker.exists()


def test_wrong_source_receipt_is_rejected_before_import(tmp_path):
    h = helper()
    source, receipt, _ = frozen_files(tmp_path)
    source.write_text(source.read_text() + '\n# different candidate\n')
    with pytest.raises(ValueError, match='source.*hash'):
        h.validate_binding(source, receipt, 1, 516237)


def test_receipt_change_after_binding_is_rejected(tmp_path):
    h = helper()
    source, receipt, _ = frozen_files(tmp_path)
    binding = h.validate_binding(source, receipt, 1, 516237)
    receipt.write_text(receipt.read_text() + '\n')
    with pytest.raises(ValueError, match='receipt.*changed'):
        h.require_current_hashes(binding)


@pytest.mark.parametrize('mutate,reason', [
    (lambda j: j['rows'].append(j['rows'][0].copy()), 'cells'),
    (lambda j: j['rows'][0].update(parameters={}), 'parameters'),
    (lambda j: j['rows'][0].update(source_sha256='0'*64), 'source'),
    (lambda j: j['rows'][0].update(error='worker timed out'), 'error'),
    (lambda j: j['freeze']['environment_sha256'].update({'env_wrapper.py': '0'*64}), 'environment'),
])
def test_inconsistent_original_receipt_cannot_authorize_a_replay(tmp_path, mutate, reason):
    h = helper()
    source, receipt, _ = frozen_files(tmp_path)
    content = json.loads(receipt.read_text())
    mutate(content)
    receipt.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=reason):
        h.validate_binding(source, receipt, 1, 516237)


def test_unmeasured_geometry_is_rejected_even_if_caller_requests_it(tmp_path):
    h = helper()
    source, receipt, _ = frozen_files(tmp_path)
    with pytest.raises(ValueError, match='mandatory'):
        h.validate_binding(source, receipt, 1, 3249018572)


def test_missing_outcome_fields_are_rejected_before_replay(tmp_path):
    h = helper()
    source, receipt, _ = frozen_files(tmp_path)
    content = json.loads(receipt.read_text())
    content['rows'][0].pop('finished')
    receipt.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='semantic fields'):
        h.validate_binding(source, receipt, 1, 516237)


def test_action_trace_matches_evaluator_float32_byte_order(tmp_path):
    h = helper()
    _, _, actions = frozen_files(tmp_path)
    trace = h.ActionTrace()
    for action in actions.astype(np.float64):
        value = trace.add(action)
        assert value.dtype == np.float32
    assert trace.verify(hashlib.sha256(actions.tobytes()).hexdigest(), 2)


def test_near_equal_single_ulp_action_does_not_count_as_exact_repeat(tmp_path):
    h = helper()
    _, _, actions = frozen_files(tmp_path)
    changed = actions.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(1.))
    assert np.allclose(actions, changed)
    trace = h.ActionTrace()
    for action in changed:
        trace.add(action)
    with pytest.raises(ValueError, match='action.*hash'):
        trace.verify(hashlib.sha256(actions.tobytes()).hexdigest(), 2)


def test_reordered_actions_and_short_trace_are_rejected(tmp_path):
    h = helper()
    _, _, actions = frozen_files(tmp_path)
    trace = h.ActionTrace()
    for action in actions[::-1]:
        trace.add(action)
    with pytest.raises(ValueError, match='action.*hash'):
        trace.verify(hashlib.sha256(actions.tobytes()).hexdigest(), 2)
    with pytest.raises(ValueError, match='steps'):
        trace.verify(trace.sha256(), 3)


def test_invalid_action_is_rejected_without_advancing_hash_or_count():
    h = helper()
    trace = h.ActionTrace()
    original = trace.sha256()
    for action in ([0., np.nan, 0.], [0., 1.01, 0.], [0., .3]):
        with pytest.raises(ValueError):
            trace.add(action)
    assert trace.count == 0 and trace.sha256() == original


def test_diagnostic_snapshot_copies_arrays_without_advancing_observer():
    h = helper()
    action = np.array([.03, .16, 0.], np.float32)
    sensor = {'innovation': np.float32(.2), 'unavailable': np.nan}
    agent = SimpleNamespace(predictive_status='predictive', predictive_result={'action': action},
                            predictive_sensor=sensor, predictive_latency_s=.002,
                            predictive_model_calls=12, _observer=SimpleNamespace(actions_seen=3),
                            _predictive_previous_action=action, last_speed=70.)
    captured = h.capture_diagnostics(agent)
    action[:] = 0.
    assert captured['predictive_result']['action'][1] == pytest.approx(.16)
    assert captured['observer_actions_seen'] == 3 and agent._observer.actions_seen == 3
    assert captured['predictive_sensor']['unavailable'] == {'nonfinite': 'nan'}
    json.dumps(captured, allow_nan=False)


def test_changed_outcome_is_rejected_even_when_action_hash_matches(tmp_path):
    h = helper()
    source, receipt, _ = frozen_files(tmp_path)
    original = h.validate_binding(source, receipt, 1, 516237)['row']
    actual = original.copy()
    actual['collision_count'] = 1
    with pytest.raises(ValueError, match='collision_count'):
        h.verify_outcome(actual, original)


def test_summary_counts_actual_statuses_and_never_guesses_missing_status():
    h = helper()
    rows = [dict(action_latency_ms=2., diagnostics={'predictive_status': 'predictive', 'predictive_model_calls': 12}),
            dict(action_latency_ms=1., diagnostics={'predictive_status': 'fallback_low_speed', 'predictive_model_calls': 0}),
            dict(action_latency_ms=1.5, diagnostics={})]
    result = h.summarize(rows)
    assert result['predictive_active_steps'] == 1
    assert result['predictive_active_fraction'] == pytest.approx(1/3)
    assert result['status_counts'] == {'predictive': 1, 'fallback_low_speed': 1, '<missing>': 1}
    assert result['model_calls_max'] == 12 and result['model_calls_sum'] == 12
