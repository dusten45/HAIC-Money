"""Read-only unsupported-geometry localization on an exact existing replay.

No act, world, planning search, laps or labels initialize this analysis. The
camera observer is rebuilt causally from saved images and logged legal actions;
passive parent fields are restored solely for deterministic camera geometry.
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
SOURCE_SHA = 'd2bd8a0855876f5cc4d281bd1183646ad29bd27e7f9231a4af6d917e901b844e'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nearest_unknown(frame, unknown, point, circles):
    pixels = np.argwhere(unknown)
    if not len(pixels):
        return {'kind': 'no_unknown_pixel'}
    metric = np.column_stack(((pixels[:, 1]-42.)/1.3608, (63.-pixels[:, 0])/1.701))
    distances = np.linalg.norm(metric-np.asarray(point), axis=1)
    chosen = int(np.argmin(distances))
    row, column = map(int, pixels[chosen])
    x, y = metric[chosen]
    gray = float(frame[row, column])
    if row in (0, 72) or column in (0, 83):
        kind = 'camera_visibility_boundary'
    elif (abs(x) <= 1.6+1./1.3608 and -2.4-1./1.701 <= y <= 2.6+1./1.701 and gray < .32):
        kind = 'dark_car_raster_outside_exact_fill'
    elif circles and np.min(np.linalg.norm(np.asarray(circles)-[x, y], axis=1)) < 2.5 and .51 < gray <= .705:
        kind = 'detected_circle_paint_or_halo'
    elif gray > .51:
        kind = 'bright_grass_or_unknown_paint'
    else:
        kind = 'dark_or_antialiased_unknown'
    return {'kind': kind, 'pixel_row_column': [row, column], 'gray': gray,
            'nearest_pixel_distance_m': float(distances[chosen]), 'point_xy_m': np.asarray(point).tolist()}


def analyze_frame(module, agent, frame, state):
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
    path = agent._ridge(frame)
    common = {'observed_circle_count': observed_count, 'constraint_circle_count': len(circles),
              'pass_side': agent.pass_side, 'pass_missing': agent.pass_missing}
    if path is None or len(path) < 5:
        raw = module._ClearRidgeReference._ridge(agent, frame)
        return {**common, 'branch': 'raw_ridge_absent' if raw is None else 'legacy_depth5p8_prefix_too_short',
                'raw_ridge_nodes': 0 if raw is None else len(raw)}
    dense = np.concatenate([np.linspace(left, right, 10) for left, right in zip(path[:-1], path[1:])])
    depths = agent._sample_distance(field, dense)
    common.update(reference_nodes=len(path), reference_endpoint_xy_m=path[-1].tolist(),
                  strict_dense_min_depth_m=float(depths.min()))
    if float(depths.min()) < 1.9:
        first = int(np.flatnonzero(depths < 1.9)[0])
        return {**common, 'branch': 'strict_dense_reference_unsupported',
                'first_unsupported_depth_m': float(depths[first]),
                'nearest_unknown': nearest_unknown(frame, unknown, dense[first], circles)}
    geometry = module._HullCameraGeometry(path, field, circles, unknown)
    initial = geometry.evaluate(np.zeros((1, 2)), np.zeros(1), np.asarray(state['velocity'])[None], .02)
    common.update(initial_road_slack_m=float(initial['road_slack_m'][0]),
                  initial_obstacle_slack_m=None if np.isinf(initial['obstacle_slack_m'][0]) else float(initial['obstacle_slack_m'][0]),
                  initial_front_slack_m=float(initial['reference_front_slack_m'][0]))
    if initial['road_slack_m'][0] < 0.:
        points = module.body_points(np.zeros((1, 2)), np.zeros(1))[0]
        point = points[int(np.argmin(module.sample_field(field, points)))]
        return {**common, 'branch': 'initial_body_road_unsupported',
                'nearest_unknown': nearest_unknown(frame, unknown, point, circles)}
    if initial['obstacle_slack_m'][0] < 0.:
        return {**common, 'branch': 'initial_circle_guard_unsupported'}
    if initial['reference_front_slack_m'][0] < 0.:
        return {**common, 'branch': 'initial_reference_front_unsupported'}
    return {**common, 'branch': 'geometry_supported'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve receipts')
    source = ROOT/'agents/apex_2026/fast_predictive_agent.py'
    assert digest(source) == SOURCE_SHA
    source_hash = digest(__file__)
    replay_hash = digest(args.replay)
    replay = json.loads(args.replay.read_text())
    assert replay['exact_repeat'] and replay['source_sha256'] == SOURCE_SHA
    cameras = Path(replay['camera_path'])
    assert digest(cameras) == replay['camera_file_sha256']
    spec = importlib.util.spec_from_file_location('predictive_support_source', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observer = module.CameraObserver(module._fixed_camera_calibration())
    agent = module.Agent()
    observations = np.load(cameras)['observations']
    records = []
    sensor_match, state_match, branch_match = True, True, True
    for row, observation in zip(replay['rows'], observations):
        frame = observation[-1]
        state, sensor = observer.observe(frame)
        diagnostic = row['diagnostics']
        sensor_match &= sensor == diagnostic['predictive_sensor']
        for name, value in diagnostic['debug'].items():
            if name == 'corridor_road' and value is not None:
                value = tuple(np.asarray(item) for item in value)
            setattr(agent, name, value)
        status = diagnostic['predictive_status']
        analysis = analyze_frame(module, agent, frame, state) if status in ('predictive', 'fallback_unsupported') else {'branch': 'not_geometry_eligible'}
        if status in ('predictive', 'fallback_unsupported'):
            branch_match &= (analysis['branch'] == 'geometry_supported') == (status == 'predictive')
        analysis.update(step=row['step'], actual_status=status, camera_speed_mps=sensor['speed_mps'])
        records.append(analysis)
        observer.advance(row['action'])
        for name, value in observer.state.items():
            stored = diagnostic['observer_state_after_advance'][name]
            state_match &= np.array_equal(value, stored) if isinstance(value, np.ndarray) else value == stored
    assert len(records) == len(replay['rows']) == len(observations)
    assert sensor_match and state_match and branch_match
    counts = Counter(r['branch'] for r in records)
    reasons = Counter(r.get('nearest_unknown', {}).get('kind', 'no_pixel_localization') for r in records if r['actual_status'] == 'fallback_unsupported')
    examples = {}
    for row in records:
        if row['actual_status'] == 'fallback_unsupported':
            key = row['branch']+'/'+row.get('nearest_unknown', {}).get('kind', '')
            if key not in examples or row['camera_speed_mps'] > examples[key]['camera_speed_mps']:
                examples[key] = row
    assert source_hash == digest(__file__) and digest(source) == SOURCE_SHA and replay_hash == digest(args.replay)
    report = {'classification': 'read_only_exact_replay_camera_geometry_branch_localization',
              'source_sha256': SOURCE_SHA, 'analysis_source_sha256': source_hash,
              'replay_path': str(args.replay), 'replay_sha256': replay_hash,
              'camera_npz_path': str(cameras), 'camera_npz_sha256': replay['camera_file_sha256'],
              'new_act_calls': 0, 'new_planning_searches': 0, 'new_world_steps': 0, 'new_driving_episodes': 0,
              'new_holdout_opened': False, 'truth_used_for_observer_initialization_or_update': False,
              'all_rebuilt_sensors_exact': sensor_match, 'all_rebuilt_observer_states_exact': state_match,
              'all_geometry_branch_decisions_match_replay': branch_match,
              'branch_counts': dict(counts), 'unsupported_nearest_unknown_counts': dict(reasons),
              'highest_speed_example_per_branch_and_pixel_kind': examples, 'rows': records,
              'limits': ['Nearest unknown pixel is a spectral/context diagnosis, not a semantic ground-truth grass label.',
                         'Parent passive camera geometry state is restored from its existing exact diagnostic snapshot.',
                         'No new control outcome or reliability gain is measured.',
                         'The strict field and all frozen candidate sources remain unchanged.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps({k: report[k] for k in ('branch_counts', 'unsupported_nearest_unknown_counts',
                                           'highest_speed_example_per_branch_and_pixel_kind')}, indent=2))


if __name__ == '__main__':
    main()
