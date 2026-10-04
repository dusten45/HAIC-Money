"""PRIVILEGED DIAGNOSTIC: rear-engine budget and high-speed spin threshold.

Uses the official car on synthetic ideal asphalt; never a camera candidate.
Fixed throttle and the requested state-adaptive budget are measured separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_dynamics import (
    DT, OFFICIAL_FILES, ROOT, digest, make_car, response, tick,
)


def adaptive_response(speed, steer):
    world, car = make_car(speed)
    samples = []
    for i in range(150):
        current = float(np.linalg.norm(car.hull.linearVelocity))
        lateral = current * current * math.tan(steer) / 3.24
        gas = float(np.clip(.008 * current * math.sqrt(max(0., 1. - (lateral / 210.) ** 2)), 0., 1.))
        row = tick(world, car, steer, gas, 0.)
        row.update(time_s=(i + 1) * DT, gas=gas)
        samples.append(row)
    means = {key: float(np.mean([s[key] for s in samples[-30:]]))
             for key in samples[0] if key != 'time_s'}
    return {"initial_speed_mps": speed, "front_target_rad": steer,
            "equivalent_action_steer": -steer,
            "throttle_mode": "adaptive_0.008v_sqrt_210_budget",
            "last_06s_means": means,
            "transient_per_action": samples[3::4]}


def compact(row):
    samples = row.pop('transient_per_action')
    row['max_abs_body_sideslip_deg'] = max(abs(s['body_sideslip_deg']) for s in samples)
    row['spin_observed_over_10deg'] = row['max_abs_body_sideslip_deg'] > 10.
    row['sustained_over_95pct_initial_speed'] = row['last_06s_means']['speed_mps'] >= .95 * row['initial_speed_mps']
    row['sampled_checkpoints'] = [samples[i] for i in (0, 2, 5, 11, 24, 36)]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    source_hash = digest(__file__)
    helper_hash = digest(ROOT/'agents/apex_2026/research/speed_20261005/physics_dynamics.py')
    hashes = {p: digest(ROOT / p) for p in OFFICIAL_FILES}
    rows = [compact(response(speed, steer, gas)) for speed in (70., 100.)
            for steer in (.03, .06) for gas in (.3, .5, .7, .8)]
    rows += [compact(adaptive_response(speed, steer)) for speed in (70., 100.)
             for steer in (.03, .06)]
    # Additional requested grid distinguishes .10rad at70/90/100 and the
    # fixed .25/.4/.65 gas bins from the new adaptive formula.
    rows += [compact(response(speed, steer, gas)) for speed in (70., 90., 100.)
             for steer in (.03, .06, .10) for gas in (.25, .4, .65)]
    assert source_hash == digest(__file__)
    assert helper_hash == digest(ROOT/'agents/apex_2026/research/speed_20261005/physics_dynamics.py')
    assert hashes == {p: digest(ROOT / p) for p in OFFICIAL_FILES}
    report = {'classification': 'privileged_uniform_asphalt_throttle_diagnostic',
              'is_legal_camera_candidate': False, 'is_lap_benchmark': False,
              'source_sha256': source_hash, 'helper_sha256': helper_hash,
              'official_file_sha256': hashes,
              'budget_formula': '0.008*v*sqrt(max(0,1-(v*v*tan(delta)/(3.24*210))**2))',
              'rows': rows,
              'limitations': ['Ideal uniform asphalt contacts with no edge/obstacle.',
                              'Three seconds per case, not long-run stability certification.',
                              'Slip>10deg is a diagnostic label, not an official DNF rule.',
                              'Adaptive budget uses raw speed and target steering, unavailable as raw legal inputs.']}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps({'output': str(args.output), 'source_sha256': source_hash,
                      'cases': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
