"""PRIVILEGED offline full-point minimum-curvature refinement, no episodes.

The convex quadratic is an approximation to curvature on fixed road distances.
Circle supporting halfplanes preserve the chosen passing side. The resulting
time remains an optimistic point-model estimate, not a global time optimum.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

from agents.apex_2026.research.speed_20261005.physics_dynamics import ROOT, digest
from agents.apex_2026.research.speed_20261005.physics_racingline import (
    obstacle_clearance, speed_profile,
)


def refine(row, max_iterations=6000):
    road = np.asarray(row['road'], float)
    normals = np.asarray(row['road_normals'], float)
    initial = np.asarray(row['route'], float)
    offsets = np.asarray(row['offsets_m'], float)
    obstacles = np.asarray(row['obstacle_positions'], float)
    count = len(road)
    road_ds = np.linalg.norm(np.roll(road, -1, axis=0)-road, axis=1)
    previous_ds = np.roll(road_ds, 1)
    weights = 1./(.5*(road_ds+previous_ds))
    difference = np.zeros((count, count))
    for i in range(count):
        difference[i, (i-1) % count] = 1./previous_ds[i]
        difference[i, i] = -1./previous_ds[i]-1./road_ds[i]
        difference[i, (i+1) % count] = 1./road_ds[i]
    a_x, a_y = difference*normals[:, 0], difference*normals[:, 1]
    b = difference@road
    q = a_x.T@(weights[:, None]*a_x)+a_y.T@(weights[:, None]*a_y)
    c = a_x.T@(weights*b[:, 0])+a_y.T@(weights*b[:, 1])
    q += np.eye(count)*1e-7
    lower, upper = np.full(count, -4.8), np.full(count, 4.8)
    links = np.roll(initial, -1, axis=0)-initial
    length_sq = np.sum(links*links, axis=1)
    supporting_planes = 0
    for obstacle in obstacles:
        fractions = np.clip(np.sum((obstacle-initial)*links, axis=1)/length_sq, 0., 1.)
        closest = initial+fractions[:, None]*links
        delta = closest-obstacle
        distances = np.linalg.norm(delta, axis=1)
        for segment in np.flatnonzero(distances < 12.):
            normal = delta[segment]/distances[segment]
            for point in (segment, (segment+1) % count):
                coefficient = float(normals[point]@normal)
                requirement = 3.7001-float((road[point]-obstacle)@normal)
                if abs(coefficient) < 1e-9:
                    if requirement > 0:
                        raise ValueError('Supporting circle plane impossible within fixed road normals')
                elif coefficient > 0:
                    lower[point] = max(lower[point], requirement/coefficient)
                else:
                    upper[point] = min(upper[point], requirement/coefficient)
                supporting_planes += 1
    if np.any(lower > upper):
        raise ValueError('Chosen circle supporting planes conflict with road offset bounds')
    initial_offsets = np.clip(offsets, lower, upper)
    began = time.perf_counter()
    fit = minimize(lambda x: float(x@q@x+2*c@x), initial_offsets,
                   jac=lambda x: 2*(q@x+c), method='L-BFGS-B',
                   bounds=list(zip(lower, upper)),
                   options={'maxiter': max_iterations, 'ftol': 1e-13, 'gtol': 1e-7, 'maxls': 30})
    route = road+fit.x[:, None]*normals
    model_time, speeds, ds, signed = speed_profile(route)
    clearance = obstacle_clearance(route, obstacles)
    road_time = speed_profile(road)[0]
    return {'track_id': row['track_id'], 'seed': row['seed'],
            'model_time_s': model_time, 'prior_spline_model_time_s': row['model_time_s'],
            'road_center_no_obstacle_model_time_s': road_time,
            'route_length_m': float(ds.sum()), 'maximum_abs_offset_m': float(np.max(abs(fit.x))),
            'minimum_obstacle_center_clearance_m': float(clearance.min()),
            'obstacle_center_clearances_m': clearance.tolist(),
            'max_abs_curvature_per_m': float(np.max(abs(signed))),
            'solver_success': bool(fit.success), 'solver_message': str(fit.message),
            'solver_iterations': int(fit.nit), 'solver_evaluations': int(fit.nfev),
            'solver_wall_time_s': time.perf_counter()-began,
            'supporting_endpoint_halfplanes': supporting_planes,
            'route': route.tolist(), 'offsets_m': fit.x.tolist(),
            'speed_profile_mps': speeds.tolist(), 'signed_curvature_per_m': signed.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=6000)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.iterations <= 20000:
        parser.error('output must be new and iterations bounded at20000')
    data = json.loads(args.input.read_text())
    hashes = {'source_sha256': digest(__file__), 'input_sha256': digest(args.input),
              'racingline_helper_sha256': digest(ROOT/'agents/apex_2026/research/speed_20261005/physics_racingline.py')}
    rows = []
    for row in data['rows']:
        output = refine(row, args.iterations)
        rows.append(output)
        print(json.dumps({k: output[k] for k in ('track_id', 'model_time_s',
                         'prior_spline_model_time_s', 'solver_success',
                         'minimum_obstacle_center_clearance_m', 'solver_wall_time_s')}), flush=True)
    assert hashes['source_sha256'] == digest(__file__)
    assert hashes['input_sha256'] == digest(args.input)
    report = dict(data)
    report.update(hashes)
    report.update(classification='privileged_offline_full_point_convex_curvature_approximation',
                  is_lap_time_lower_bound=False, is_driving_episode=False,
                  is_legal_camera_candidate=False, scipy_version=scipy.__version__,
                  maximum_solver_iterations=args.iterations, rows=rows)
    report['limits'] = data['limits']+[
        'Minimizes convex curvature approximation, not exact time objective.',
        'Passing-side supporting halfplanes restrict available route families.',
        'No driving episodes are run; prior teacher routes remain immutable.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)


if __name__ == '__main__':
    main()
