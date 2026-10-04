"""PRIVILEGED stationary two-column rear-HUD calibration; no episodes.

Fits individual purple bar columns20/22 to synthetic omega states. This tests
sensor resolution only, never a torque controller or a driving candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_dynamics import OFFICIAL_FILES, ROOT, digest, make_car
from agents.apex_2026.research.speed_20261005.physics_hud_rear import render


def columns(frame):
    return np.asarray(frame[74:83, [20, 22]].sum(axis=0), float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    official = {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    hashes = {'source_sha256': digest(__file__),
              'hud_helper_sha256': digest(ROOT/'agents/apex_2026/research/speed_20261005/physics_hud_rear.py')}
    world, car = make_car(0.)
    training = []
    for left in range(80, 251, 17):
        for right in range(80, 251, 17):
            _, frame = render(car, (left, right))
            training.append({'omega': [left, right], 'mass': columns(frame).tolist()})
    omega = np.asarray([r['omega'] for r in training], float)
    masses = np.asarray([r['mass'] for r in training], float)
    design = np.column_stack((omega, np.ones(len(omega))))
    coefficients = np.linalg.lstsq(design, masses, rcond=None)[0]
    inverse = np.linalg.inv(coefficients[:2])

    def decode(frame):
        return np.maximum(0., (columns(frame)-coefficients[2])@inverse)*.54

    dense = []
    plateau_begin, previous_raster, plateaus = 120, None, []
    for equal in range(120, 201):
        _, frame = render(car, (equal, equal), front=(160., 160.))
        speeds = decode(frame)
        dense.append({'omega': equal, 'decoded_per_rear_speed_mps': speeds.tolist(),
                      'mean_error_mps': float(speeds.mean()-.54*equal)})
        raster = np.rint(frame[74:83, [20, 22]]*255).astype(np.uint8).tobytes()
        if previous_raster is not None and raster != previous_raster:
            if equal-plateau_begin > 1:
                plateaus.append({'omega_interval_radps': [plateau_begin, equal-1],
                                 'true_speed_span_mps': .54*(equal-1-plateau_begin)})
            plateau_begin = equal
        previous_raster = raster
    if 201-plateau_begin > 1:
        plateaus.append({'omega_interval_radps': [plateau_begin, 200],
                         'true_speed_span_mps': .54*(200-plateau_begin)})
    cases = []
    max_bleed = 0.
    for rear in ((0., 0.), (120., 200.), (200., 120.), (80., 240.), (240., 80.),
                 (130., 180.), (180., 130.), (160., 160.)):
        _, reference = render(car, rear)
        for front in ((0., 0.), (200., 200.), (400., 400.), (0., 400.), (400., 0.)):
            _, frame = render(car, rear, front)
            decoded = decode(frame)
            max_bleed = max(max_bleed, float(np.max(abs(decoded-decode(reference)))))
            cases.append({'rear_omega_radps': list(rear), 'front_omega_radps': list(front),
                          'decoded_per_rear_speed_mps': decoded.tolist(),
                          'true_per_rear_speed_mps': [.54*v for v in rear],
                          'mean_error_mps': float(decoded.mean()-.54*np.mean(rear))})
    errors = np.asarray([r['mean_error_mps'] for r in dense])
    per_rear_errors = np.asarray([np.asarray(r['decoded_per_rear_speed_mps'])-.54*r['omega'] for r in dense])
    # Tests require independent bar separation and error bounds on new states.
    assert max_bleed < 1e-9
    assert np.max(abs(errors)) < 2.
    assert np.max(abs(per_rear_errors)) < 2.
    swapped = [r for r in cases if r['front_omega_radps'] == [0., 0.] and
               r['rear_omega_radps'] in ([120., 200.], [200., 120.])]
    assert abs(np.mean(swapped[0]['decoded_per_rear_speed_mps'])-
               np.mean(swapped[1]['decoded_per_rear_speed_mps'])) < .5
    assert np.max(decode(np.zeros((84, 84), np.float32))) < 1e-9
    assert np.linalg.cond(coefficients[:2]) < 10.
    assert official == {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    assert hashes['source_sha256'] == digest(__file__)
    assert hashes['hud_helper_sha256'] == digest(ROOT/'agents/apex_2026/research/speed_20261005/physics_hud_rear.py')
    report = {'classification': 'privileged_stationary_individual_rear_bar_column_calibration',
              'is_legal_camera_candidate': False, 'is_driving_episode': False, 'world_steps': 0,
              **hashes, 'official_file_sha256': official, 'render_count': len(training)+len(dense)+48,
              'formula': '(mass_columns20_22-intercept) @ inverse_response_matrix * .54',
              'mass_roi': 'frame[74:83,[20,22]].sum(axis=0)',
              'fit_omega_grid': {'start': 80, 'stop_inclusive': 250, 'step': 17},
              'response_matrix': coefficients[:2].tolist(), 'intercept': coefficients[2].tolist(),
              'inverse_response_matrix': inverse.tolist(),
              'response_matrix_condition_number': float(np.linalg.cond(coefficients[:2])),
              'dense_equal_omega_validation_range': [120, 200],
              'dense_equal_omega_mean_error_mps': {'min': float(errors.min()), 'max': float(errors.max()),
                                                'max_absolute': float(np.max(abs(errors)))},
              'dense_equal_omega_max_individual_error_mps': float(np.max(abs(per_rear_errors))),
              'max_front_change_effect_mps': max_bleed, 'validation_cases': cases,
              'equal_raster_plateaus': plateaus,
              'max_sampled_equal_raster_speed_span_mps': max(p['true_speed_span_mps'] for p in plateaus),
              'behavioral_checks_passed': ['New-state mean/individual error<2m/s over omega120..200.',
                                           'Front0/200/400 changes do not affect isolated bar columns.',
                                           'Swapped rear120/200 mean difference<.5m/s.',
                                           'Zero frame decodes zero and response matrix is well-conditioned.'],
              'limits': ['Only stationary synthetic HUD raster, no torque or tire-control test.',
                         'Discrete omega grid understates continuous raster plateau width.',
                         'HUD body-speed error remains when subtracting wheel and body gauges.',
                         'Fit limits apply to sampled rear omega/raster resolution, not clipped bars.',
                         'A failed dynamics torque formula is not validated by this calibration.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps({k: report[k] for k in ('render_count', 'response_matrix', 'intercept',
                                           'dense_equal_omega_mean_error_mps',
                                           'dense_equal_omega_max_individual_error_mps',
                                           'max_front_change_effect_mps',
                                           'max_sampled_equal_raster_speed_span_mps')}, indent=2))


if __name__ == '__main__':
    main()
