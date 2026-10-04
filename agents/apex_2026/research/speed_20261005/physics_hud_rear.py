"""PRIVILEGED stationary official rear-wheel HUD calibration, no episodes.

Only wheel omega and body velocity are assigned for rendering. No world step,
driving action, road geometry or legal candidate source is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pygame

from agents.apex_2026.research.speed_20261005.physics_dynamics import OFFICIAL_FILES, ROOT, digest, make_car
from core.vendor.car_racing import CarRacing, STATE_W, STATE_H, WINDOW_W, WINDOW_H
from env_wrapper import image_preprocessing


def rear_speed(frame):
    crop = frame[74:83, 19:24]
    mass = float(np.where(crop <= .20, crop, 0.).sum())
    return mass/.01485*.54, mass


def body_speed(frame):
    return float(np.clip((float(frame[74:83, 10:13].sum())-.20)/.0882, 0., 180.))


def render(car, rear, front=(0., 0.), speed=70.):
    for wheel, omega in zip(car.wheels, (*front, *rear)):
        wheel.omega = omega
    car.hull.linearVelocity = (0., speed)
    renderer = CarRacing.__new__(CarRacing)
    renderer.car = car
    renderer.surf = pygame.Surface((WINDOW_W, WINDOW_H))
    renderer.surf.fill((0, 0, 0))
    CarRacing._render_indicators(renderer, WINDOW_W, WINDOW_H)
    rgb = CarRacing._create_image_array(renderer, renderer.surf, (STATE_W, STATE_H))
    frame = image_preprocessing(rgb)
    decoded, mass = rear_speed(frame)
    true = .54*float(np.mean(rear))
    return {'rear_omega_radps': list(rear), 'front_omega_radps': list(front),
            'body_speed_mps': speed, 'true_mean_rear_rolling_mps': true,
            'decoded_rear_speed_mps': decoded, 'rear_mass': mass,
            'error_mps': decoded-true, 'decoded_body_speed_mps': body_speed(frame),
            'decoded_slip_excess_mps': decoded-body_speed(frame),
            'true_mean_rear_excess_mps': true-speed,
            'column_masses': frame[74:83, 19:24].sum(axis=0).tolist()}, frame


def metrics(rows):
    errors = np.asarray([row['error_mps'] for row in rows])
    return {'case_count': len(rows), 'bias_mps': float(errors.mean()),
            'min_error_mps': float(errors.min()), 'max_error_mps': float(errors.max()),
            'max_absolute_error_mps': float(np.max(abs(errors)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    source_hash = digest(__file__)
    official = {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    world, car = make_car(0.)
    checkpoints = []
    for omega in (0., 40., 80., 120., 160., 200., 240., 300.):
        row, _ = render(car, (omega, omega))
        checkpoints.append(row)
    bleed = []
    for rear in ((0., 0.), (120., 120.), (160., 160.), (200., 200.),
                 (120., 200.), (200., 120.), (80., 240.), (240., 80.)):
        for front in ((0., 0.), (200., 200.), (400., 400.), (0., 400.), (400., 0.)):
            row, _ = render(car, rear, front)
            bleed.append(row)
    dense, raster_keys = [], []
    for omega in range(0, 401):
        row, frame = render(car, (float(omega), float(omega)), front=(160., 160.))
        dense.append(row)
        raster_keys.append(np.rint(frame[74:83, 19:24]*255).astype(np.uint8).tobytes())
    plateaus = []
    begin = 0
    for index in range(1, len(raster_keys)+1):
        if index == len(raster_keys) or raster_keys[index] != raster_keys[begin]:
            if index-begin > 1:
                plateaus.append({'omega_interval_radps': [begin, index-1],
                                 'true_speed_span_mps': .54*(index-1-begin),
                                 'decoded_rear_speed_mps': dense[begin]['decoded_rear_speed_mps']})
            begin = index
    ranges = {}
    for low, high in ((0, 300), (120, 200), (130, 210), (200, 300), (300, 400)):
        rows = dense[low:high+1]
        summary = metrics(rows)
        summary['true_rolling_speed_range_mps'] = [.54*low, .54*high]
        ranges[f'omega{low}_to_{high}'] = summary
    bleed_effects = []
    for rear in sorted(set(tuple(row['rear_omega_radps']) for row in bleed)):
        rows = [r for r in bleed if tuple(r['rear_omega_radps']) == rear]
        baseline = next(r for r in rows if r['front_omega_radps'] == [0., 0.])
        bleed_effects.append({'rear_omega_radps': list(rear),
                              'max_front_omega_change_effect_mps': max(abs(r['decoded_rear_speed_mps']-
                                                                          baseline['decoded_rear_speed_mps']) for r in rows)})
    operating_plateaus = [p for p in plateaus if 120 <= p['omega_interval_radps'][0] <= 200]
    slip_errors = np.asarray([r['decoded_slip_excess_mps']-r['true_mean_rear_excess_mps']
                             for r in dense[120:201]])
    pure_gray = int(cv2.cvtColor(np.asarray([[[51, 0, 255]]], np.uint8), cv2.COLOR_RGB2GRAY)[0, 0])
    # Behavioral checks of the renderer/measurement, not candidate tests.
    assert all(checkpoints[i]['decoded_rear_speed_mps'] <= checkpoints[i+1]['decoded_rear_speed_mps']
               for i in range(len(checkpoints)-1))
    assert all(np.isfinite(row['decoded_rear_speed_mps']) for row in dense+bleed)
    assert rear_speed(np.zeros((84, 84), np.float32)) == (0., 0.)
    for speed in (0., 70., 100.):
        _, frame = render(car, (160., 160.), speed=speed)
        assert abs(rear_speed(frame)[0]-checkpoints[4]['decoded_rear_speed_mps']) < 1e-7
    assert official == {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    assert digest(__file__) == source_hash
    report = {'classification': 'privileged_stationary_official_rear_hud_multi_render',
              'is_legal_camera_candidate': False, 'is_driving_episode': False, 'world_steps': 0,
              'source_sha256': source_hash, 'official_file_sha256': official,
              'render_count': len(checkpoints)+len(bleed)+len(dense)+3,
              'wheel_radius_m': .54, 'rear_rgb': [51, 0, 255], 'pure_gray_uint8': pure_gray,
              'formula': 'sum(frame[74:83,19:24] where pixel<=.20)/.01485*.54',
              'checkpoints': checkpoints, 'front_bleed_and_asymmetry_cases': bleed,
              'front_bleed_effects': bleed_effects, 'dense_equal_rear_range_summary': ranges,
              'dense_omega_grid': {'minimum': 0, 'maximum': 400, 'step': 1},
              'operating_range_equal_raster_plateaus': operating_plateaus,
              'operating_range_max_equal_raster_true_speed_span_mps': max(p['true_speed_span_mps'] for p in operating_plateaus),
              'operating_range_slip_error_at_body70_mps': {'min': float(slip_errors.min()),
                                                         'max': float(slip_errors.max()),
                                                         'max_abs': float(np.max(abs(slip_errors)))},
              'tire_force_constants': {'slip_stiffness_N_per_mps': 82., 'friction_limit_N_per_tire': 400.,
                                      'body_mass_kg': float(car.hull.mass+sum(w.mass for w in car.wheels)),
                                      'maximum_zero_lateral_slip_budget_mps': 400./82.},
              'behavioral_checks': ['Rear measurement monotonic at requested eight checkpoints.',
                                    'All rendered measurements finite; zero frame decodes zero.',
                                    'Body speed changes0/70/100 do not alter fixed rear160 meter.'],
              'limits': ['Synthetic stationary HUD only, not dynamic tire control validation.',
                         'Integer omega sampling understates continuous raster plateau span.',
                         'Pooled rear bars do not resolve each rear wheel separately.',
                         'Dense error bounds apply only to sampled omega range and front160 configuration.',
                         'Sensor calibration does not validate a torque formula that failed dynamics.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps({'checkpoints': [[r['rear_omega_radps'][0], r['decoded_rear_speed_mps'], r['error_mps']]
                                      for r in checkpoints], 'sampled_ranges': ranges,
                      'max_front_bleed_mps': max(r['max_front_omega_change_effect_mps'] for r in bleed_effects),
                      'operating_raster_span_mps': report['operating_range_max_equal_raster_true_speed_span_mps'],
                      'slip_error_mps': report['operating_range_slip_error_at_body70_mps']}, indent=2))


if __name__ == '__main__':
    main()
