"""PRIVILEGED offline alternative obstacle passing-side study; no episodes.

Pass sides are prefiltered by a road-normal point-car cross-section. This is
not an exhaustive search of all world-space trajectories. Each side family
gets convex curvature refinement followed by local point-model time search.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import scipy

from agents.apex_2026.research.speed_20261005.physics_dynamics import ROOT, digest
from agents.apex_2026.research.speed_20261005.physics_racingline import speed_profile
from agents.apex_2026.research.speed_20261005.physics_racingline_refine import refine
from agents.apex_2026.research.speed_20261005.physics_racingline_timepoint import search


def families(row):
    road, normals = np.asarray(row['road']), np.asarray(row['road_normals'])
    obstacles = np.asarray(row['obstacle_positions'])
    ds = np.linalg.norm(np.roll(road, -1, axis=0)-road, axis=1)
    s = np.r_[0., np.cumsum(ds[:-1])]
    length = float(ds.sum())
    obstacle_specs, choices = [], []
    for obstacle in obstacles:
        index = int(np.argmin(np.sum((road-obstacle)**2, axis=1)))
        lateral = float((obstacle-road[index])@normals[index])
        sides = [side for side in (-1, 1) if abs(lateral+side*3.7) <= 4.8]
        if not sides:
            raise ValueError('No point-car road-normal cross-section pass')
        choices.append(sides)
        obstacle_specs.append({'road_index': index, 'lateral_m': lateral, 'eligible_sides': sides})
    for sides in itertools.product(*choices):
        offsets = np.zeros(len(road))
        for spec, side in zip(obstacle_specs, sides):
            distance = (s-s[spec['road_index']]+length/2.) % length-length/2.
            bump = np.zeros(len(road))
            entry = (distance >= -40.) & (distance < 0.)
            exit = (distance >= 0.) & (distance <= 55.)
            bump[entry] = .5*(1+np.cos(np.pi*distance[entry]/40.))
            bump[exit] = .5*(1+np.cos(np.pi*distance[exit]/55.))
            amplitude = np.clip(spec['lateral_m']+side*3.9, -4.8, 4.8)
            offsets += amplitude*bump
        offsets = np.clip(offsets, -4.8, 4.8)
        route = road+offsets[:, None]*normals
        initial = dict(row, route=route.tolist(), offsets_m=offsets.tolist(),
                       model_time_s=speed_profile(route)[0])
        yield list(sides), obstacle_specs, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-iterations', type=int, default=300)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.time_iterations <= 300:
        parser.error('output must be new and iterations bounded at300')
    geometry_data = json.loads(args.geometry.read_text())
    helper_paths = [ROOT/'agents/apex_2026/research/speed_20261005'/name for name in
                    ('physics_racingline.py', 'physics_racingline_refine.py', 'physics_racingline_timepoint.py')]
    hashes = {'source_sha256': digest(__file__), 'geometry_sha256': digest(args.geometry),
              'helper_sha256': {str(path): digest(path) for path in helper_paths}}
    rows = []
    began = time.perf_counter()
    for geometry_row in geometry_data['rows']:
        trials = []
        for sides, obstacle_specs, initial in families(geometry_row):
            try:
                curvature = refine(initial, 6000)
                optimized = search(geometry_row, curvature, args.time_iterations)
                optimized.update(sides=sides, curvature_model_time_s=curvature['model_time_s'],
                                 curvature_solver_success=curvature['solver_success'])
                assert optimized['maximum_abs_offset_m'] <= 4.8+1e-10
                assert optimized['minimum_obstacle_center_clearance_m'] >= 3.7-1e-9
                trials.append(optimized)
                print(json.dumps({'track_id': geometry_row['track_id'], 'sides': sides,
                                  'model_time_s': optimized['model_time_s'],
                                  'solver_success': optimized['solver_success']}), flush=True)
            except ValueError as error:
                trials.append({'sides': sides, 'feasible': False, 'reason': str(error)})
                print(json.dumps({'track_id': geometry_row['track_id'], 'sides': sides,
                                  'feasible': False, 'reason': str(error)}), flush=True)
        feasible = [trial for trial in trials if 'model_time_s' in trial]
        if not feasible:
            raise ValueError('All chosen-side supporting-plane families failed')
        best = min(feasible, key=lambda trial: trial['model_time_s'])
        row = {'track_id': geometry_row['track_id'], 'seed': geometry_row['seed'],
               'obstacle_specs': obstacle_specs, 'families_considered': len(trials),
               'best': best, 'trials': trials}
        rows.append(row)
        cell_path = args.output.with_name(args.output.stem+f"-track{geometry_row['track_id']}.json")
        with cell_path.open('x') as file:
            json.dump({'hashes': hashes, 'row': row}, file, indent=2)
    assert hashes['source_sha256'] == digest(__file__)
    assert hashes['geometry_sha256'] == digest(args.geometry)
    assert hashes['helper_sha256'] == {str(path): digest(path) for path in helper_paths}
    report = {'classification': 'privileged_offline_local_alternative_pass_side_point_model',
              'is_legal_camera_candidate': False, 'is_driving_episode': False,
              'is_lap_time_lower_bound': False, 'new_driving_episodes': 0,
              **hashes, 'scipy_version': scipy.__version__,
              'maximum_time_iterations_per_family': args.time_iterations,
              'model': geometry_data['model'], 'rows': rows,
              'wall_time_s': time.perf_counter()-began,
              'limits': geometry_data['limits']+[
                  'Road-normal point cross-section side prefilter is not exhaustive world-space feasibility.',
                  'Circle supporting-plane families restrict each local search.',
                  'Distinct initial routes may converge to different local optima.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)


if __name__ == '__main__':
    main()
