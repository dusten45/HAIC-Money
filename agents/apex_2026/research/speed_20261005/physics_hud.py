"""PRIVILEGED one-frame official HUD raster calibration; no driving episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pygame

from agents.apex_2026.research.speed_20261005.physics_dynamics import (
    OFFICIAL_FILES, ROOT, digest, make_car,
)
from core.vendor.car_racing import CarRacing, STATE_W, STATE_H, WINDOW_W, WINDOW_H
from env_wrapper import image_preprocessing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    before = {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    world, car = make_car(70.)
    for wheel, omega in zip(car.wheels, (0., 0., 140., 200.)):
        wheel.omega = omega
    renderer = CarRacing.__new__(CarRacing)
    renderer.car = car
    renderer.surf = pygame.Surface((WINDOW_W, WINDOW_H))
    renderer.surf.fill((0, 0, 0))
    CarRacing._render_indicators(renderer, WINDOW_W, WINDOW_H)
    rgb = CarRacing._create_image_array(renderer, renderer.surf, (STATE_W, STATE_H))
    frame = image_preprocessing(rgb)
    roi = frame[73:83, 18:24]
    gray = int(cv2.cvtColor(np.asarray([[[51, 0, 255]]], np.uint8), cv2.COLOR_RGB2GRAY)[0, 0])
    report = {'classification': 'privileged_one_render_official_hud_calibration',
              'is_legal_camera_candidate': False, 'is_driving_episode': False,
              'source_sha256': digest(__file__), 'official_file_sha256': before,
              'render_count': 1, 'wheel_radius_m': car.wheels[0].wheel_rad,
              'rear_omega_radps': [140., 200.], 'front_omega_radps': [0., 0.],
              'rear_rgb': [51, 0, 255], 'pure_rear_gray_uint8': gray,
              'nominal_pixel_geometry': {'left_rear_x': [18.9, 21.0],
                                         'right_rear_x': [21.0, 23.1],
                                         'base_y': 81.9, 'height_per_omega': .021},
              'rear_roi': '[73:83,18:24]', 'rear_roi_mass': float(roi.sum()),
              'rear_roi_column_mass': roi.sum(axis=0).tolist(),
              'rear_roi_pixels': roi.tolist(),
              'nominal_mass_per_mean_omega': 2*2.1*.021*(gray/255.),
              'nominal_mean_omega_from_mass': float(roi.sum())/(2*2.1*.021*(gray/255.)),
              'physical_mean_omega': 170.,
              'limits': ['One rendered synthetic HUD state; no fitted empirical intercept/slope.',
                         'Real front-wheel bars may bleed into pooled col18 at the resize boundary.',
                         'Quantization and smoothing prevent treating nominal pixel areas as exact.']}
    assert before == {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps({k: report[k] for k in ('wheel_radius_m', 'rear_roi_mass',
                                           'rear_roi_column_mass',
                                           'nominal_mass_per_mean_omega',
                                           'nominal_mean_omega_from_mass')}, indent=2))


if __name__ == '__main__':
    main()
