"""PRIVILEGED three-second straight launch; no road/lap or legal agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_dynamics import (
    DT, OFFICIAL_FILES, ROOT, digest, make_car, tick,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    before = {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    world, car = make_car(0.)
    thresholds, checkpoints = {}, []
    distance = 0.
    prior = np.asarray(car.hull.position, float)
    for i in range(150):
        row = tick(world, car, 0., 1., 0.)
        xy = np.asarray(car.hull.position, float)
        distance += float(np.linalg.norm(xy-prior))
        prior = xy
        row.update(time_s=(i+1)*DT, distance_m=distance,
                   rear_omega_radps=[float(w.omega) for w in car.wheels[2:]])
        for target in (25., 50., 70., 90., 99.):
            if row['speed_mps'] >= target and str(target) not in thresholds:
                thresholds[str(target)] = dict(row)
        if (i+1) % 20 == 0 or i == 149:
            checkpoints.append(dict(row))
    assert before == {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    report = {'classification': 'privileged_uniform_asphalt_straight_launch',
              'is_legal_camera_candidate': False, 'is_lap_benchmark': False,
              'source_sha256': digest(__file__),
              'helper_sha256': digest(ROOT/'agents/apex_2026/research/speed_20261005/physics_dynamics.py'),
              'official_file_sha256': before, 'initial_speed_mps': 0.,
              'gas': 1., 'steer': 0., 'duration_s': 3.,
              'thresholds_mps': thresholds, 'checkpoints': checkpoints,
              'three_second_distance_m': distance,
              'three_second_time_penalty_vs_instant_100mps_s': 3.-distance/100.,
              'limits': ['Uniform synthetic road contact and no curvature.',
                         'The measured start penalty is not a shortest-lap lower bound.',
                         'No road geometry, finish criterion, or obstacle is evaluated.']}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps({key: report[key] for key in ('thresholds_mps',
                                                 'three_second_distance_m',
                                                 'three_second_time_penalty_vs_instant_100mps_s')}, indent=2))


if __name__ == '__main__':
    main()
