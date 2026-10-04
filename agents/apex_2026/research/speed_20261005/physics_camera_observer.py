"""Observe-only causal camera dynamics diagnostic on existing saved frames.

The observer receives an84x84 frame and past legal actions only. Simulator
labels are read by the separate evaluator after each camera update/forecast.
No simulator state initializes or corrects the observer. No world steps/laps.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_dynamics import ROOT, digest
from agents.apex_2026.research.speed_20261005.physics_four_tire_model import (
    predict_step, rotate, summarize,
)
from agents.apex_2026.research.speed_20261005.physics_hud_joint_body import decode

RESULTS = ROOT/'agents/apex_2026/results/speed-20261005'
BODY_CALIBRATION = RESULTS/'physics-hud-joint-body-v1.json'
REAR_CALIBRATION = RESULTS/'physics-hud-rear-isolated-v1.json'
YAW_CALIBRATION = RESULTS/'audit-yaw.json'
DEFAULT_FIXTURE = ROOT/'.haic-artifacts/apex-speed-20261005/slow-phase-track3-v2/inspection.json'

# Mechanical constants are from the official public hull/wheel shapes, with
# the vehicle at rest. They are not estimated from this episode's truth.
PUBLIC_GEOMETRY = {'mass': 7.3019199669361115, 'inertia': 19.170645977936854,
                   'local_center': [0., -.0804591735470409],
                   'wheel_anchors': [[-1.1, 1.6], [1.1, 1.6], [-1.1, -1.64], [1.1, -1.64]]}
SENSOR_BOUNDS = {'body_speed_mps': 1.6, 'joint_rad': .006,
                 'rear_individual_rolling_mps': 1.94,
                 'yaw_radps_provisional': .15}
PREDECLARED_SCREEN = {'body_sideslip_abs_error_p90_deg': 1.,
                      'body_sideslip_abs_error_p99_deg': 3.,
                      'forecast_speed_abs_error_p90_mps': 2.5,
                      'forecast_yaw_abs_error_p90_radps': .2,
                      'forecast_body_side_abs_error_p90_mps': 1.,
                      'forecast_rear_longitudinal_slip_max_abs_error_p90_mps': 2.5}


class CameraObserver:
    """Camera-only interface: observe(frame), advance(legal_action), reset()."""

    def __init__(self, calibration):
        self.calibration = copy.deepcopy(calibration)
        self.reset()

    def reset(self):
        center = np.array(PUBLIC_GEOMETRY['local_center'], float)
        self.state = {'angle': 0., 'velocity': np.zeros(2), 'yaw': 0.,
                      'mass': PUBLIC_GEOMETRY['mass'], 'inertia': PUBLIC_GEOMETRY['inertia'],
                      'local_center': center,
                      'wheel_offsets': np.array(PUBLIC_GEOMETRY['wheel_anchors'], float)-center,
                      'joint': np.zeros(4), 'omega': np.zeros(4), 'gas': np.zeros(4),
                      'wheel_velocity_override': None}
        self.actions_seen = 0
        self.last_sensor = None

    def observe(self, frame):
        frame = np.asarray(frame, float)
        if frame.shape != (84, 84) or not np.isfinite(frame).all():
            raise ValueError('expected finite84x84 camera')
        speed, joint = decode(frame, self.calibration['body_joint'])
        speed = float(np.clip(speed, 0., 100.))
        rear_mass = frame[74:83, [20, 22]].sum(axis=0)
        rear_rolling = np.maximum(0., (rear_mass-self.calibration['rear_intercept']) @
                                  self.calibration['rear_inverse'])*.54
        red = frame[75:81, 51:83]
        red = np.where((red > 0.) & (red < .37), red, 0.)
        mass = float(red.sum())
        centroid = float(red.sum(axis=0) @ np.arange(51, 83)/mass) if mass else 62.5
        magnitude = max(0., (mass-self.calibration['yaw_intercept'])/self.calibration['yaw_slope'])
        camera_yaw = magnitude if centroid >= 62.5 else -magnitude
        body_yaw = float(np.clip(-camera_yaw, -8., 8.))
        old = self.state
        local_velocity = rotate(old['velocity'], -old['angle'])
        prior_speed = float(np.linalg.norm(local_velocity))
        prior_rear = .54*old['omega'][2:]
        prior_yaw = old['yaw']
        # Retain the latent lateral velocity predicted from past controls.
        # HUD magnitude corrects forward speed; it does not supply sideslip.
        side = float(np.clip(local_velocity[0], -.8*speed, .8*speed))
        old['velocity'] = np.array([side, math.sqrt(max(0., speed*speed-side*side))])
        old['angle'] = 0.
        old['yaw'] = body_yaw
        # A known motor/omega forecast inside the camera raster interval is
        # preferable to repeatedly replacing it with a quantized point.
        old['joint'][:2] = np.clip(old['joint'][:2], joint-SENSOR_BOUNDS['joint_rad'],
                                  joint+SENSOR_BOUNDS['joint_rad'])
        radius = SENSOR_BOUNDS['rear_individual_rolling_mps']/.54
        old['omega'][2:] = np.maximum(0., np.clip(old['omega'][2:],
                                                rear_rolling/.54-radius,
                                                rear_rolling/.54+radius))
        old['wheel_velocity_override'] = None
        self.last_sensor = {'speed_mps': speed, 'body_yaw_radps': body_yaw,
                            'joint_rad': joint, 'rear_rolling_mps': rear_rolling.tolist(),
                            'speed_innovation_mps': speed-prior_speed,
                            'yaw_innovation_radps': body_yaw-prior_yaw,
                            'rear_innovation_max_mps': float(np.max(abs(rear_rolling-prior_rear))),
                            'rear_bar_top_nonzero': bool(np.any(frame[74, [20, 22]] > 0.)),
                            'yaw_beyond_calibration': abs(body_yaw) > 4.5,
                            'dynamics_innovation_suspect': self.actions_seen > 0 and
                            (abs(speed-prior_speed) > 3.2 or abs(body_yaw-prior_yaw) > .3 or
                             float(np.max(abs(rear_rolling-prior_rear))) > 6.),
                            'lateral_uncertainty_certified': False}
        return copy.deepcopy(old), copy.deepcopy(self.last_sensor)

    def advance(self, legal_action):
        action = np.asarray(legal_action, float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('expected finite3-vector legal action')
        action = np.clip(action, [-1., 0., 0.], [1., 1., 1.])
        diagnostics = []
        for _ in range(4):
            self.state, row = predict_step(self.state, [-action[0], action[1], action[2]])
            diagnostics.append(row)
        self.actions_seen += 1
        return copy.deepcopy(self.state), diagnostics


def calibration_from_receipts():
    body = json.loads(BODY_CALIBRATION.read_text())
    rear = json.loads(REAR_CALIBRATION.read_text())
    yaw = json.loads(YAW_CALIBRATION.read_text())['raster_calibration']
    return {'body_joint': body['calibration'], 'rear_intercept': np.array(rear['intercept']),
            'rear_inverse': np.array(rear['inverse_response_matrix']),
            'yaw_slope': yaw['synthetic_nonzero_affine_mass_per_abs_yaw'],
            'yaw_intercept': yaw['synthetic_nonzero_affine_mass_intercept']}


def label_summary(label):
    """Only the evaluator calls this; the observer never receives a label."""
    return {'speed_mps': label['speed_m_per_s'], 'body_side_mps': label['side_m_per_s'],
            'body_sideslip_deg': label['sideslip_deg'],
            'body_yaw_radps': -label['yaw_camera_rad_per_s'],
            'front_joint_rad': [r['joint_angle_rad'] for r in label['wheels'][:2]],
            'rear_longitudinal_slip_mps': [r['roll_m_per_s']-r['forward_m_per_s'] for r in label['wheels'][2:]]}


def estimate_summary(state):
    result = summarize(state)
    result['body_side_mps'] = float(rotate(state['velocity'], -state['angle'])[0])
    return result


def errors(estimate, truth):
    result = {name: abs(estimate[name]-truth[name]) for name in
              ('speed_mps', 'body_yaw_radps', 'body_sideslip_deg', 'body_side_mps')}
    for name in ('front_joint_rad', 'rear_longitudinal_slip_mps'):
        result[name] = float(np.max(abs(np.array(estimate[name])-truth[name])))
    return result


def stats(rows, key):
    values = np.array([r[key] for r in rows], float)
    return {'count': len(values), 'mean': float(values.mean()), 'p50': float(np.percentile(values, 50)),
            'p90': float(np.percentile(values, 90)), 'p99': float(np.percentile(values, 99)),
            'max': float(values.max())} if len(values) else {'count': 0}


def behavioral_tests(calibration, frames, actions):
    checks = []
    observer = CameraObserver(calibration)
    state, _ = observer.observe(np.zeros((84, 84)))
    assert summarize(state)['speed_mps'] == 0. and state['yaw'] == 0.
    state, _ = observer.advance([-.4, 1., 0.])
    np.testing.assert_allclose(state['gas'][2:], .4, atol=1e-12)
    assert np.max(abs(state['joint'][:2])) <= .24+1e-12
    checks.append('Cold zero camera and known first action preserve4-step gas and motor bounds.')
    observer.reset()
    assert observer.actions_seen == 0 and np.max(abs(observer.state['omega'])) == 0.
    checks.append('Reset discards all prior body/wheel/control history.')
    outputs = []
    for _ in range(2):
        observer.reset()
        values = []
        for frame, action in zip(frames, actions):
            observer.observe(frame)
            state, _ = observer.advance(action)
            values.append(estimate_summary(state))
        outputs.append(json.dumps(values, sort_keys=True))
    assert outputs[0] == outputs[1]
    checks.append('Two cold runs are byte-identical from camera/actions alone; no labels are passed.')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve the prior receipt')
    dependencies = [Path(__file__), ROOT/'agents/apex_2026/research/speed_20261005/physics_four_tire_model.py',
                    ROOT/'agents/apex_2026/research/speed_20261005/physics_hud_joint_body.py',
                    BODY_CALIBRATION, REAR_CALIBRATION, YAW_CALIBRATION, args.fixture]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in dependencies}
    fixture = json.loads(args.fixture.read_text())
    camera_path = ROOT/fixture['camera_npz_path']
    assert digest(camera_path) == fixture['camera_npz_sha256']
    frames = np.load(camera_path)['frames']
    rows = fixture['rows']
    assert len(rows) == len(frames) == 120
    calibration = calibration_from_receipts()
    checks = behavioral_tests(calibration, frames, [r['action'] for r in rows])
    observer = CameraObserver(calibration)
    records = []
    for row, frame in zip(rows, frames):
        state, sensor = observer.observe(frame)
        estimate = estimate_summary(state)
        predicted, _ = observer.advance(row['action'])
        forecast = estimate_summary(predicted)
        before = label_summary(row['before'])
        after = label_summary(row['after'])
        records.append({'step': row['step'], 'sensor': sensor,
                        'observer': {k: estimate[k] for k in before},
                        'forecast_after_action': {k: forecast[k] for k in after},
                        'before_label': before, 'after_label': after,
                        'observer_absolute_error': errors(estimate, before),
                        'forecast_absolute_error': errors(forecast, after),
                        'all_wheels_on_road_before_label_only': row['before']['wheel_road_count'] == 4,
                        'all_wheels_on_road_after_label_only': row['after']['wheel_road_count'] == 4,
                        'collision_label_only': row['collision']})
    masks = {'all120': records,
             'labeled_all_road_before_speed20plus': [r for r in records if r['all_wheels_on_road_before_label_only'] and
                                                    r['before_label']['speed_mps'] >= 20.],
             'labeled_all_road_before_and_after_speed20plus': [r for r in records if
                        r['all_wheels_on_road_before_label_only'] and r['all_wheels_on_road_after_label_only'] and
                        r['before_label']['speed_mps'] >= 20.],
             'camera_innovation_not_suspect_speed20plus': [r for r in records if r['sensor']['speed_mps'] >= 20. and
                        not r['sensor']['dynamics_innovation_suspect'] and not r['sensor']['rear_bar_top_nonzero'] and
                        not r['sensor']['yaw_beyond_calibration']]}
    aggregate = {}
    for name, group in masks.items():
        aggregate[name] = {kind: {metric: stats([r[kind] for r in group], metric) for metric in
                                 ('speed_mps', 'body_yaw_radps', 'body_sideslip_deg', 'body_side_mps',
                                  'front_joint_rad', 'rear_longitudinal_slip_mps')}
                           for kind in ('observer_absolute_error', 'forecast_absolute_error')}
    road = aggregate['labeled_all_road_before_and_after_speed20plus']
    measured = {'body_sideslip_abs_error_p90_deg': road['observer_absolute_error']['body_sideslip_deg']['p90'],
                'body_sideslip_abs_error_p99_deg': road['observer_absolute_error']['body_sideslip_deg']['p99'],
                'forecast_speed_abs_error_p90_mps': road['forecast_absolute_error']['speed_mps']['p90'],
                'forecast_yaw_abs_error_p90_radps': road['forecast_absolute_error']['body_yaw_radps']['p90'],
                'forecast_body_side_abs_error_p90_mps': road['forecast_absolute_error']['body_side_mps']['p90'],
                'forecast_rear_longitudinal_slip_max_abs_error_p90_mps': road['forecast_absolute_error']['rear_longitudinal_slip_mps']['p90']}
    passed = {metric: measured[metric] <= bound for metric, bound in PREDECLARED_SCREEN.items()}
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in dependencies}
    assert digest(camera_path) == fixture['camera_npz_sha256']
    report = {'classification': 'causal_camera_only_observer_privileged_labels_offline_diagnostic',
              'is_legal_driving_candidate': False, 'new_world_steps': 0, 'new_driving_episodes': 0,
              'new_holdout_opened': False, 'observer_inputs': ['camera84x84', 'past_legal_actions', 'reset_zero_state'],
              'truth_used_for_observer_initialization_or_update': False,
              'source_and_input_sha256': hashes, 'camera_npz_path': fixture['camera_npz_path'],
              'camera_npz_sha256': fixture['camera_npz_sha256'], 'source_fixture_old_agent_sha256': fixture['source_sha256'],
              'public_mechanical_geometry': PUBLIC_GEOMETRY, 'sensor_bounds': SENSOR_BOUNDS,
              'behavioral_checks_passed': checks, 'predeclared_uniform_road_screen': PREDECLARED_SCREEN,
              'screen_measured': measured, 'screen_passed_by_metric': passed,
              'all_uniform_road_screen_metrics_passed': all(passed.values()),
              'aggregates': aggregate, 'rows': records,
              'limits': ['Only one consumed mandatory-track prefix; this is not fresh coverage or lap validation.',
                         'Latent lateral velocity/front omega use uniform-asphalt model history, without direct camera measurement.',
                         'Labels define road-only reporting subsets; those labels are unavailable to a legal controller.',
                         'Innovation gate is heuristic and cannot certify asphalt, contacts or a sideslip interval.',
                         'Rear raster uncertainty is up to1.94m/s before body/latent-state errors; force uncertainty can be large.',
                         'Yaw affine calibration was audited at only nine stationary rates, not a full sensor-error certificate.',
                         'Statistical screen success would not establish safe MPC controls or speed improvements.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps({'screen_measured': measured, 'screen_passed': passed,
                     'counts': {k: len(v) for k, v in masks.items()},
                     'all120_observer': aggregate['all120']['observer_absolute_error'],
                     'road_forecast': road['forecast_absolute_error']}, indent=2))


if __name__ == '__main__':
    main()
