"""Compact a frozen full-brake receipt without rerunning any dynamics.

Keep all144 pose/error/speed/yaw rows, four initial summaries and fixed gates.
Full tire-force vectors remain byte-for-byte in the ignored original artifact.
Integrated relative heading is explicitly derived from recorded endpoint yaw.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.apex_2026.research.speed_20261005.physics_dynamics import ROOT, digest

RAW_SHA = '89501b4d73794db63dcacb654f7259641f155871bfa831696dbfe74a7bc0c026'
MODEL_TEST_SHA = '5eabe53c68e77e9f6f3efafc52ab5e2d9da943c711bcaf5cc2c8e5e2a29cd915'


def compact(raw, raw_path):
    cases = []
    count = 0
    for case in raw['rows']:
        rows = []
        predicted_heading, actual_heading = 0., 0.
        for row in case['full_raw_step_pose_and_state_rows']:
            predicted_heading += .02*row['predicted']['body_yaw_radps']
            actual_heading += .02*row['actual']['body_yaw_radps']
            assert abs(abs(predicted_heading-actual_heading)-row['absolute_error']['hull_angle_error_rad']) < 1e-5
            rows.append({'time_s': row['time_s'],
                         'predicted_hull_xy_m': row['predicted_hull_xy_relative_initial_body_m'],
                         'actual_hull_xy_m': row['actual_hull_xy_relative_initial_body_m'],
                         'predicted_speed_mps': row['predicted']['speed_mps'],
                         'actual_speed_mps': row['actual']['speed_mps'],
                         'predicted_body_yaw_radps': row['predicted']['body_yaw_radps'],
                         'actual_body_yaw_radps': row['actual']['body_yaw_radps'],
                         'predicted_integrated_relative_heading_rad': predicted_heading,
                         'actual_integrated_relative_heading_rad': actual_heading,
                         'absolute_error': row['absolute_error']})
            assert rows[-1]['absolute_error'] == row['absolute_error']
            count += 1
        cases.append({key: case[key] for key in
                      ('configuration', 'actual_initial_summary', 'tail_command_world_joint_gas_brake',
                       'predicted_stop_speed_half_yaw_tenth_s', 'actual_stop_speed_half_yaw_tenth_s',
                       'maximum_error_over_all36_raw_steps', 'acceptance_passed')})
        cases[-1]['raw_step_pose_error_speed_yaw_rows'] = rows
    assert count == 144
    report = {key: raw[key] for key in
              ('is_legal_camera_candidate', 'is_lap_benchmark', 'new_driving_episodes', 'new_holdout_opened',
               'uniform_fixed_cases', 'horizon_s', 'model_sha256', 'helper_sha256', 'official_sha256',
               'acceptance_thresholds', 'all_case_acceptance_passed', 'limits')}
    report.update({'classification': 'compact_privileged_fixed_uniform_asphalt_full_brake_tail_validation',
                   'compactor_source_sha256': digest(__file__), 'full_raw_receipt_sha256': RAW_SHA,
                   'fixed_validation_source_sha256': MODEL_TEST_SHA,
                   'full_raw_receipt_ignored_path': str(raw_path.relative_to(ROOT)),
                   'preserved_raw_step_count': count, 'rows': cases,
                   'faithful_copy_checks': ['All144 exact pose/error/speed/yaw scalar rows copied from the frozen raw receipt.',
                                            'All4 fixed configurations, recorded initial summaries and prospective gate results copied.',
                                            'Raw forces/vectors remain in original SHA-bound artifact; no dynamics reruns.'],
                   'derived_heading_note': 'Relative headings are semiimplicit .02s sums of recorded endpoint yaw, not newly sampled hull-angle labels; angle-error consistency checked<1e-5rad.',
                   'initial_state_note': 'The raw receipt stores complete initial summaries, not every Box2D state field; no missing private state is inferred.'})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve the compact receipt')
    assert digest(args.input) == RAW_SHA
    raw = json.loads(args.input.read_text())
    assert raw['source_sha256'] == MODEL_TEST_SHA
    report = compact(raw, args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    assert digest(args.input) == RAW_SHA
    print(json.dumps({'raw_steps': report['preserved_raw_step_count'], 'bytes': args.output.stat().st_size,
                      'compact_sha256': digest(args.output), 'all_accepted': report['all_case_acceptance_passed']}))


if __name__ == '__main__':
    main()
