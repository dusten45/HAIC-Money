"""Fixed synthetic diagnostic of road and circle guard compatibility.

This does not simulate a car or change a policy. The public straight-road
width is compared with the frozen predictive guard at one centered circle.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'agents/apex_2026/fast_predictive_v2_agent.py'
SOURCE_SHA = 'b70af66e0ba39ea975d53909ec6d7a19df2a73005a036465d71dd7ac8caeff8a'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert digest(SOURCE) == SOURCE_SHA
    spec = importlib.util.spec_from_file_location('frozen_guard_compatibility', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    half_width = 40./6.  # Public TRACK_WIDTH=40/SCALE; SCALE=6.
    body_half_width, road_margin, center_guard = 1.6, 1.9, 3.7
    largest_road_center_offset = half_width-body_half_width-road_margin
    x_pixels = (np.arange(84)-42.)/1.3608
    field = np.tile(np.maximum(0., half_width-abs(x_pixels)), (73, 1))
    offsets = [0., 2.8, largest_road_center_offset, 3.55, center_guard]
    poses = np.asarray([[offset, 15.] for offset in offsets])
    angles = np.zeros(len(poses))
    road = module.road_slack(field, poses, angles, road_margin)
    obstacle = module.obstacle_slack(poses, angles, [(0., 15.)])
    assert np.allclose(road, half_width-np.asarray(offsets)-body_half_width-road_margin)
    assert obstacle[2] < 0. and road[4] < 0.
    output = ROOT/'agents/apex_2026/results/speed-20261005/physics-predictive-guard-compatibility.json'
    if output.exists():
        raise FileExistsError('preserve diagnostic receipt')
    report = {
        'classification': 'fixed_synthetic_guard_compatibility_diagnostic_not_physical_impossibility',
        'source_sha256': SOURCE_SHA, 'analysis_source_sha256': digest(Path(__file__)),
        'public_width_source': 'core/vendor/car_racing.py: SCALE=6, TRACK_WIDTH=40/SCALE',
        'public_width_source_sha256': digest(ROOT/'core/vendor/car_racing.py'),
        'road_half_width_m': half_width, 'body_half_width_m': body_half_width,
        'full_body_road_margin_m': road_margin, 'center_circle_guard_m': center_guard,
        'rectangle_circle_radius_plus_margin_m': 1.2+.75,
        'largest_aligned_road_supported_center_offset_m': largest_road_center_offset,
        'minimum_center_guard_passing_offset_m': center_guard,
        'gap_between_necessary_passing_and_supported_offsets_m': center_guard-largest_road_center_offset,
        'maximum_road_margin_at_center_guard_pass_m': half_width-body_half_width-center_guard,
        'rows': [{'center_offset_m': x, 'frozen_road_slack_m': float(r),
                  'frozen_obstacle_slack_m': float(o), 'actual_envelope_circle_clearance_m': x-1.6-1.2}
                 for x, r, o in zip(offsets, road, obstacle)],
        'new_world_steps': 0, 'new_driving_episodes': 0, 'new_control_searches': 0,
        'new_holdout_opened': False,
        'limits': ['Fixed straight aligned synthetic crossing, not any mandatory track map or a lap.',
                   'The 1.9m guard outside the whole body conflicts with the circle guard at a centered crossing.',
                   'Physical vehicle dynamics, swept trajectory, camera errors and adoption are not tested.',
                   'No candidate or frozen component is changed; changing margins requires separate justified validation.']}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
