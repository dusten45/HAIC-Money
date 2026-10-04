"""PRIVILEGED official yaw-HUD raster calibration; no driving episode.

Nine synthetic body yaw rates are rendered through the official indicator and
camera resize path. Direct Box2D state is used only to construct known HUD
inputs, never as an Agent observation or a competition result.
"""
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
from core.vendor.car_racing import CarRacing, STATE_H, STATE_W, WINDOW_H, WINDOW_W
from env_wrapper import image_preprocessing


NAV_SOURCE = ROOT / "agents/apex_2026/fast_path_agent.py"
YAW_RATES = (0., -.5, .5, -1.5, 1.5, -3.5, 3.5, -4.5, 4.5)
RED_GRAY = float(cv2.cvtColor(np.array([[[255, 0, 0]]], dtype=np.uint8),
                                  cv2.COLOR_RGB2GRAY)[0, 0]) / 255.
NOMINAL_MASS_PER_RADPS = 4.2 * .299 * 1.68


def legacy_v4_decode(frame):
    """Exact V4 formula captured before the navigation lane edited its file."""
    crop = frame[76:80, 54:75]
    weights = np.where((crop >= .22) & (crop <= .37), crop, 0.)
    mass = float(weights.sum())
    if mass < .03:
        return 0.
    middle = float((weights * np.arange(54, 75)).sum()) / mass
    magnitude = mass / (4. * .299 * 1.68)
    return float(np.clip(magnitude if middle >= 62.5 else -magnitude, -8., 8.))


def wide_decode(frame):
    """Compare the proposed full strip while retaining its mass and centroid."""
    crop = frame[75:81, 51:83]
    weights = np.where((crop > 0.) & (crop < .37), crop, 0.)
    mass = float(weights.sum())
    columns = weights.sum(axis=0)
    if mass < .03:
        return 0., mass, None, columns
    middle = float(np.dot(columns, np.arange(51, 83))) / mass
    magnitude = mass / NOMINAL_MASS_PER_RADPS
    estimate = float(np.clip(magnitude if middle >= 62.5 else -magnitude, -8., 8.))
    return estimate, mass, middle, columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve prior measurements")

    source_hash = digest(__file__)
    nav_hash = digest(NAV_SOURCE)
    before = {path: digest(ROOT / path) for path in OFFICIAL_FILES}
    world, car = make_car(70.)
    renderer = CarRacing.__new__(CarRacing)
    renderer.car = car
    rows = []
    for yaw in YAW_RATES:
        car.hull.angularVelocity = yaw
        renderer.surf = pygame.Surface((WINDOW_W, WINDOW_H))
        renderer.surf.fill((0, 0, 0))
        CarRacing._render_indicators(renderer, WINDOW_W, WINDOW_H)
        rgb = CarRacing._create_image_array(renderer, renderer.surf,
                                            (STATE_W, STATE_H))
        frame = image_preprocessing(rgb)
        old = legacy_v4_decode(frame)
        wide, mass, middle, columns = wide_decode(frame)
        old_crop = frame[76:80, 54:75]
        old_mass = float(np.where((old_crop >= .22) & (old_crop <= .37),
                                  old_crop, 0.).sum())
        old_full_mass = float(np.where((old_crop > 0.) & (old_crop < .37),
                                       old_crop, 0.).sum())
        expected = -yaw  # Official gauge draws -.8 times body angular velocity.
        rows.append({
            "physical_body_yaw_radps": yaw,
            "expected_camera_hud_sign_radps": expected,
            "navigation_v4_decoded_radps": old,
            "wide_nominal_decoded_radps": wide,
            "navigation_v4_error_radps": old - expected,
            "wide_nominal_error_radps": wide - expected,
            "navigation_v4_selected_mass": old_mass,
            "old_crop_all_dim_red_mass": old_full_mass,
            "wide_selected_mass": mass,
            "wide_mass_centroid_x": middle,
            "wide_mass_by_column": [float(v) for v in columns],
            "wide_effective_height_px": (mass / (abs(yaw) * RED_GRAY * 1.68)
                                         if yaw else None),
        })

    assert digest(__file__) == source_hash
    assert digest(NAV_SOURCE) == nav_hash
    assert {path: digest(ROOT / path) for path in OFFICIAL_FILES} == before
    assert rows[0]["wide_selected_mass"] == 0.
    nonzero_rates = np.asarray([abs(row["physical_body_yaw_radps"])
                                for row in rows[1:]])
    nonzero_masses = np.asarray([row["wide_selected_mass"]
                                 for row in rows[1:]])
    fitted_slope, fitted_intercept = np.polyfit(nonzero_rates, nonzero_masses, 1)
    groups = {}
    for rate in sorted(set(nonzero_rates)):
        paired = [row for row in rows if abs(row["physical_body_yaw_radps"]) == rate]
        groups[str(rate)] = {
            "navigation_v4_mean_absolute_error_radps": float(np.mean([
                abs(row["navigation_v4_error_radps"]) for row in paired])),
            "wide_mean_absolute_error_radps": float(np.mean([
                abs(row["wide_nominal_error_radps"]) for row in paired])),
        }
    report = {
        "classification": "privileged_official_yaw_hud_raster_diagnostic",
        "is_legal_camera_candidate": False,
        "is_driving_episode": False,
        "is_lap_benchmark": False,
        "render_count": len(rows),
        "source_sha256": source_hash,
        "navigation_live_source_sha256": nav_hash,
        "navigation_v4_formula_source_sha256_prechange": "71c9ad3fa505615c6316abd99e6bf2d4989cf7e987087eec055148f67ca06bfc",
        "official_file_sha256": before,
        "hud_inputs": {"hull_forward_speed_mps": 70.,
                       "front_joint_steering_rad": 0.,
                       "yaw_rates_radps": list(YAW_RATES)},
        "renderer": "official _render_indicators, _create_image_array, image_preprocessing",
        "nominal_red_gray": RED_GRAY,
        "source_hud_layout_scaled_to_84px": {
            "yaw_bar_pivot_x": 63.,
            "yaw_bar_extent_at_abs_4p5": [55.44, 70.56],
            "green_steering_bar_max_right_x_at_joint_limit": 50.4,
            "wheel_gauges_right_x": 23.1,
            "score_text_center_x": 5.04,
        },
        "wide_decoder": {"crop": "[75:81,51:83]", "gray_band": "0 < gray < .37",
                         "mass_per_radps": NOMINAL_MASS_PER_RADPS,
                         "sign_pivot_x": 62.5},
        "navigation_v4_decoder": {"crop": "[76:80,54:75]",
                                  "gray_band": ".22 <= gray <= .37",
                                  "mass_per_radps": 4. * .299 * 1.68},
        "raster_calibration": {
            "zero_yaw_wide_mass": rows[0]["wide_selected_mass"],
            "synthetic_nonzero_affine_mass_per_abs_yaw": float(fitted_slope),
            "synthetic_nonzero_affine_mass_intercept": float(fitted_intercept),
            "synthetic_nonzero_zero_intercept_mass_per_abs_yaw": float(
                np.dot(nonzero_rates, nonzero_masses) /
                np.dot(nonzero_rates, nonzero_rates)),
            "paired_absolute_errors": groups,
        },
        "rows": rows,
        "limits": [
            "Known body yaw is privileged input to a synthetic HUD render; no world step or episode.",
            "No steering gauge is drawn: front wheel joints are zero. The green bar can approach the left edge of a wider grayscale yaw crop under extreme steering.",
            "The speed and wheel gauges occupy far-left columns; score text is not rendered by this indicators-only probe.",
            "Pixel quantization and image resizing may vary across supported graphics library versions.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps([{
        "physical_yaw": row["physical_body_yaw_radps"],
        "old": row["navigation_v4_decoded_radps"],
        "wide": row["wide_nominal_decoded_radps"],
        "wide_height": row["wide_effective_height_px"],
    } for row in rows], indent=2))


if __name__ == "__main__":
    main()
