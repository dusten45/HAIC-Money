"""Read-only geometry localization from exact action-only camera caches.

The inherited camera parent is called once; the predictive Agent.act and
control search are never called. Observer and steering memory advance only
with recorded actual actions. Privileged cache truth is not opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
V1_SHA = 'd2bd8a0855876f5cc4d281bd1183646ad29bd27e7f9231a4af6d917e901b844e'
V2_SHA = 'b70af66e0ba39ea975d53909ec6d7a19df2a73005a036465d71dd7ac8caeff8a'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same(left, right):
    if isinstance(left, (list, tuple, np.ndarray)):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return bool(left == right)


def validate_parent_reconstruction(replay_path):
    replay = json.loads(replay_path.read_text())
    source = ROOT/'agents/apex_2026/fast_predictive_agent.py'
    assert digest(source) == replay['source_sha256'] == V1_SHA
    assert replay['exact_repeat']
    cameras = Path(replay['camera_path'])
    assert digest(cameras) == replay['camera_file_sha256']
    module = load_module(source, 'support_validation_v1')
    agent = module.Agent()
    observer = module.CameraObserver(module._fixed_camera_calibration())
    observations = np.load(cameras)['observations']
    for row, observation in zip(replay['rows'], observations):
        module._BaseRearClearAgent.act(agent, observation)
        _, sensor = observer.observe(observation[-1])
        agent.last_steer = float(np.float32(row['action'][0]))
        for name, expected in row['diagnostics']['debug'].items():
            actual = getattr(agent, name)
            if name == 'corridor_road' and expected is not None:
                assert all(same(a, b) for a, b in zip(actual, expected)), (row['step'], name)
            else:
                assert same(actual, expected), (row['step'], name)
        assert sensor == row['diagnostics']['predictive_sensor'], row['step']
        observer.advance(row['action'])
        assert all(same(value, row['diagnostics']['observer_state_after_advance'][name])
                   for name, value in observer.state.items()), row['step']
    assert len(replay['rows']) == len(observations)
    return {'replay_path': str(replay_path), 'replay_sha256': digest(replay_path),
            'source_sha256': V1_SHA, 'rows': len(observations),
            'all_parent_passive_state_exact': True, 'all_observer_sensor_and_state_exact': True}


def analyze_geometry(module, support, agent, frame, state):
    if agent.corridor_road is None:
        return {'branch': 'parent_road_missing'}
    if agent.lost_frames:
        return {'branch': 'parent_road_lost'}
    circles = [(float(x), float(y)) for y, x in agent._circles(frame, agent.corridor_road)]
    observed_count = len(circles)
    if (agent.pass_side and agent.pass_y is not None and agent.pass_missing <= 4 and
            np.isfinite([agent.pass_x, agent.pass_y]).all()):
        remembered = np.array([agent.pass_x, agent.pass_y])
        if all(np.linalg.norm(remembered-np.asarray(circle)) > .5 for circle in circles):
            circles.append(tuple(remembered))
    field, unknown = agent._predictive_field(frame, circles)
    raw = module._ClearRidgeReference._ridge(agent, frame)
    common = {'observed_circles_xy_m': circles[:observed_count],
              'constraint_circles_xy_m': circles, 'raw_reference_nodes': 0 if raw is None else len(raw)}
    if raw is None:
        return {**common, 'branch': 'raw_ridge_absent'}
    first_bad = None
    for left, right in zip(raw[:-1], raw[1:]):
        dense = np.linspace(left, right, max(2, int(np.ceil(np.linalg.norm(right-left)/.25))+1))
        depths = agent._sample_distance(field, dense)
        bad = np.flatnonzero(depths < 1.9)
        if len(bad):
            index = int(bad[0])
            first_bad = {'first_unsupported_depth_m': float(depths[index]),
                         'nearest_unknown': support.nearest_unknown(frame, unknown, dense[index], circles)}
            break
    path = agent._predictive_supported_prefix(raw, field)
    common.update(raw_reference_start_xy_m=raw[0].tolist(), raw_reference_endpoint_xy_m=raw[-1].tolist(),
                  supported_reference_nodes=0 if path is None else len(path), prefix_first_bad=first_bad)
    if path is None or len(path) < 5:
        return {**common, 'branch': 'strict_prefix_too_short'}
    dense = np.concatenate([np.linspace(left, right, 10) for left, right in zip(path[:-1], path[1:])])
    depths = agent._sample_distance(field, dense)
    common.update(reference_endpoint_xy_m=path[-1].tolist(), strict_dense_min_depth_m=float(depths.min()))
    if float(depths.min()) < 1.9:
        first = int(np.flatnonzero(depths < 1.9)[0])
        return {**common, 'branch': 'strict_dense_reference_unsupported',
                'nearest_unknown': support.nearest_unknown(frame, unknown, dense[first], circles)}
    geometry = module._HullCameraGeometry(path, field, circles, unknown)
    initial = geometry.evaluate(np.zeros((1, 2)), np.zeros(1), np.asarray(state['velocity'])[None], .02)
    common.update(initial_road_slack_m=float(initial['road_slack_m'][0]),
                  initial_obstacle_slack_m=None if np.isinf(initial['obstacle_slack_m'][0]) else float(initial['obstacle_slack_m'][0]),
                  initial_front_slack_m=float(initial['reference_front_slack_m'][0]))
    if initial['road_slack_m'][0] < 0.:
        points = module.body_points(np.zeros((1, 2)), np.zeros(1))[0]
        point = points[int(np.argmin(module.sample_field(field, points)))]
        return {**common, 'branch': 'initial_body_road_unsupported',
                'nearest_unknown': support.nearest_unknown(frame, unknown, point, circles)}
    if initial['obstacle_slack_m'][0] < 0.:
        return {**common, 'branch': 'initial_circle_guard_unsupported'}
    if initial['reference_front_slack_m'][0] < 0.:
        return {**common, 'branch': 'initial_reference_front_unsupported'}
    return {**common, 'branch': 'geometry_supported'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-replay', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve existing diagnostic receipts')
    source = ROOT/'agents/apex_2026/fast_predictive_v2_agent.py'
    support_source = Path(__file__).with_name('physics_predictive_support.py')
    assert digest(source) == V2_SHA
    source_hash = digest(__file__)
    audit_path = args.cache/'audit.json'
    frames_path = args.cache/'frames.npz'
    audit = json.loads(audit_path.read_text())
    assert audit['source_sha256'] == V2_SHA and audit['exact_action_bytes'] and audit['exact_post_trace']
    assert audit['exact_semantic_row'] and not audit['agent_act_called']
    assert digest(frames_path) == audit['frames_sha256'] and digest(args.trace) == audit['trace_sha256']
    module = load_module(source, 'action_support_v2')
    support = load_module(support_source, 'action_support_pixel_localizer')
    validations = [validate_parent_reconstruction(path) for path in args.validate_replay]
    agent = module.Agent()
    observer = module.CameraObserver(module._fixed_camera_calibration())
    frames = np.load(frames_path)['observations']
    # Extract actual actions and post-contact flags only. Position, speed,
    # angle and every separate truth-cache label remain outside inference.
    raw_trace = json.loads(args.trace.read_text())
    trace = [{'step': row['step'], 'action': row['action'], 'collision': row['collision'],
              'debug': row['debug']} for row in raw_trace]
    del raw_trace
    assert len(trace) == len(frames) == audit['steps']
    action_bytes = np.asarray([row['action'] for row in trace], np.float32).tobytes()
    assert hashlib.sha256(action_bytes).hexdigest() == audit['action_trace_sha256']
    records = []
    trace_parent_exact = True
    previous = np.zeros(3, np.float32)
    grid_provider = module.FixedControlPlanner(module.predict_step, None)
    for row, observation in zip(trace, frames):
        parent_action = np.asarray(module._BaseRearClearAgent.act(agent, observation), np.float32)
        state, sensor = observer.observe(observation[-1])
        actual = np.asarray(row['action'], np.float32)
        if sensor['speed_mps'] < 20.:
            gate = 'fallback_low_speed'
        elif sensor['dynamics_innovation_suspect'] or sensor['rear_bar_top_nonzero']:
            gate = 'fallback_innovation'
        elif sensor['yaw_beyond_calibration']:
            gate = 'fallback_yaw'
        else:
            gate = 'geometry_eligible'
        result = analyze_geometry(module, support, agent, observation[-1], state) if gate == 'geometry_eligible' else {'branch': 'not_geometry_eligible'}
        # Grid membership diagnoses impossible predictive actions, never
        # establishes original planner success or failure by searching it.
        candidates = grid_provider.candidate_actions(previous)
        grid_member = any(np.array_equal(actual, candidate) for candidate in candidates)
        if gate != 'geometry_eligible':
            implied = gate
        elif result['branch'] != 'geometry_supported':
            implied = 'fallback_unsupported'
        elif not grid_member:
            implied = 'fallback_control_search_or_budget'
        else:
            implied = 'original_predictive_or_matching_fallback_not_identified'
        agent.last_steer = float(actual[0])
        for name, expected in row['debug'].items():
            trace_parent_exact &= same(getattr(agent, name), expected)
        records.append({**result, 'step': row['step'], 'camera_sensor': sensor,
                        'deterministic_gate': gate, 'original_status_implied': implied,
                        'actual_action': actual.tolist(), 'parent_action': parent_action.tolist(),
                        'actual_matches_parent': np.array_equal(actual, parent_action),
                        'actual_in_predictive_control_grid': grid_member,
                        'last_target_mps': agent.last_target, 'lost_frames': agent.lost_frames,
                        'pass_side': agent.pass_side, 'pass_missing': agent.pass_missing,
                        'pass_xy_m': [agent.pass_x, agent.pass_y], 'post_action_collision': row['collision']})
        observer.advance(actual)
        previous = actual.copy()
    assert trace_parent_exact
    assert digest(source) == V2_SHA and digest(__file__) == source_hash
    report = {'classification': 'read_only_action_cache_causal_camera_geometry_localization',
              'source_sha256': V2_SHA, 'analysis_source_sha256': source_hash,
              'pixel_localizer_source_sha256': digest(support_source),
              'cache_audit_path': str(audit_path), 'cache_audit_sha256': digest(audit_path),
              'camera_npz_path': str(frames_path), 'camera_npz_sha256': audit['frames_sha256'],
              'trace_path': str(args.trace), 'trace_sha256': audit['trace_sha256'],
              'action_trace_sha256': audit['action_trace_sha256'],
              'all_recorded_parent_debug_fields_exact': trace_parent_exact,
              'prior_exact_replay_reconstruction_validations': validations,
              'predictive_act_calls': 0, 'inherited_parent_act_calls': len(records),
              'control_search_calls': 0, 'new_world_steps': 0, 'new_driving_episodes': 0,
              'new_holdout_opened': False, 'privileged_truth_cache_opened': False,
              'branch_counts': dict(Counter(row['branch'] for row in records)),
              'implied_status_counts': dict(Counter(row['original_status_implied'] for row in records)),
              'contact_steps': [row['step'] for row in records if row['post_action_collision']],
              'rows': records,
              'limits': ['A control-grid member is not proof of an original planner success.',
                         'No search is rerun; feasible-control and timeout counts are not identified.',
                         'Nearest unknown pixels are spectral/context labels, not semantic grass truth.',
                         'Three recorded parent debug fields and optional older full-state replays verify reconstruction; V2 full hidden debug was not logged.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('branch_counts', 'implied_status_counts', 'contact_steps', 'all_recorded_parent_debug_fields_exact')}, indent=2))


if __name__ == '__main__':
    main()
