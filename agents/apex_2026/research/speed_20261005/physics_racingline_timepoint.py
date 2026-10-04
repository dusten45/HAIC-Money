"""PRIVILEGED offline full-point local time search, no driving episodes.

Independent acceleration limits and point-circle separation are optimistic.
Passing-side supporting planes restrict the search; no global bound is claimed.
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
from agents.apex_2026.research.speed_20261005.physics_racingline import obstacle_clearance, speed_profile


def batch_objective(offsets, road, normals):
    routes = road[None, :, :]+offsets[:, :, None]*normals[None, :, :]
    links = np.roll(routes, -1, axis=1)-routes
    ds = np.linalg.norm(links, axis=2)
    previous = np.roll(links, 1, axis=1)
    turn = np.arctan2(previous[:, :, 0]*links[:, :, 1]-previous[:, :, 1]*links[:, :, 0],
                      np.sum(previous*links, axis=2))
    curvature = abs(turn)/np.maximum(.5*(ds+np.roll(ds, 1, axis=1)), .1)
    cap = np.minimum(100., np.sqrt(219./np.maximum(curvature, 1e-7)))
    speed = np.column_stack((cap, cap[:, 0]))
    speed[:, 0] = 0.
    for i in range(len(road)):
        speed[:, i+1] = np.minimum(speed[:, i+1], np.sqrt(speed[:, i]**2+80.*ds[:, i]))
    for i in range(len(road)-1, -1, -1):
        speed[:, i] = np.minimum(speed[:, i], np.sqrt(speed[:, i+1]**2+300.*ds[:, i]))
    duration = np.sum(2*ds/np.maximum(speed[:, :-1]+speed[:, 1:], .1), axis=1)
    second = np.roll(offsets, 1, axis=1)-2*offsets+np.roll(offsets, -1, axis=1)
    return duration+.001*np.sum(second*second, axis=1)


def search(geometry_row, initial_row, iterations):
    road, normals = np.asarray(geometry_row['road']), np.asarray(geometry_row['road_normals'])
    initial_route = np.asarray(initial_row['route'])
    initial = np.asarray(initial_row['offsets_m'])
    obstacles = np.asarray(geometry_row['obstacle_positions'])
    count = len(road)
    lower, upper = np.full(count, -4.8), np.full(count, 4.8)
    links = np.roll(initial_route, -1, axis=0)-initial_route
    length_sq = np.sum(links*links, axis=1)
    for obstacle in obstacles:
        fraction = np.clip(np.sum((obstacle-initial_route)*links, axis=1)/length_sq, 0., 1.)
        delta = initial_route+fraction[:, None]*links-obstacle
        distances = np.linalg.norm(delta, axis=1)
        for segment in np.flatnonzero(distances < 12.):
            normal = delta[segment]/distances[segment]
            for point in (segment, (segment+1) % count):
                coefficient = float(normals[point]@normal)
                requirement = 3.7-float((road[point]-obstacle)@normal)
                if coefficient > 1e-9:
                    lower[point] = max(lower[point], requirement/coefficient)
                elif coefficient < -1e-9:
                    upper[point] = min(upper[point], requirement/coefficient)
                elif requirement > 0:
                    raise ValueError('Infeasible fixed supporting plane')
    if np.any(lower > upper):
        raise ValueError('Supporting plane conflicts with road bounds')
    initial = np.clip(initial, lower, upper)
    # Simultaneous finite differences use NumPy batches, not driving episodes.
    def value_gradient(values):
        epsilon = 1e-5
        candidates = np.tile(values, (count+1, 1))
        candidates[np.arange(1, count+1), np.arange(count)] += epsilon
        evaluated = batch_objective(candidates, road, normals)
        return float(evaluated[0]), (evaluated[1:]-evaluated[0])/epsilon

    scalar_check = speed_profile(road+initial[:, None]*normals)[0]
    second = np.roll(initial, 1)-2*initial+np.roll(initial, -1)
    assert abs(value_gradient(initial)[0]-scalar_check-.001*float(second@second)) < 1e-9
    began = time.perf_counter()
    fit = minimize(value_gradient, initial, jac=True, method='L-BFGS-B',
                   bounds=list(zip(lower, upper)),
                   options={'maxiter': iterations, 'ftol': 2e-10, 'gtol': 1e-5, 'maxls': 30})
    route = road+fit.x[:, None]*normals
    duration, speeds, ds, signed = speed_profile(route)
    clearance = obstacle_clearance(route, obstacles)
    return {'track_id': geometry_row['track_id'], 'seed': geometry_row['seed'],
            'model_time_s': duration, 'prior_curvature_model_time_s': initial_row['model_time_s'],
            'route_length_m': float(ds.sum()), 'maximum_abs_offset_m': float(np.max(abs(fit.x))),
            'minimum_obstacle_center_clearance_m': float(clearance.min()),
            'obstacle_center_clearances_m': clearance.tolist(),
            'max_abs_curvature_per_m': float(np.max(abs(signed))),
            'solver_success': bool(fit.success), 'solver_message': str(fit.message),
            'solver_iterations': int(fit.nit), 'solver_evaluations': int(fit.nfev),
            'solver_wall_time_s': time.perf_counter()-began,
            'route': route.tolist(), 'offsets_m': fit.x.tolist(), 'speed_profile_mps': speeds.tolist(),
            'signed_curvature_per_m': signed.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=150)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.iterations <= 300:
        parser.error('output must be new and iterations bounded at300')
    geometry_data, initial_data = json.loads(args.geometry.read_text()), json.loads(args.initial.read_text())
    hashes = {'source_sha256': digest(__file__), 'geometry_sha256': digest(args.geometry),
              'initial_sha256': digest(args.initial),
              'racingline_helper_sha256': digest(ROOT/'agents/apex_2026/research/speed_20261005/physics_racingline.py')}
    rows = []
    for geometry_row, initial_row in zip(geometry_data['rows'], initial_data['rows']):
        assert (geometry_row['track_id'], geometry_row['seed']) == (initial_row['track_id'], initial_row['seed'])
        row = search(geometry_row, initial_row, args.iterations)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('track_id', 'model_time_s', 'prior_curvature_model_time_s',
                         'solver_success', 'minimum_obstacle_center_clearance_m', 'solver_wall_time_s')}), flush=True)
    assert hashes['source_sha256'] == digest(__file__)
    assert hashes['geometry_sha256'] == digest(args.geometry)
    assert hashes['initial_sha256'] == digest(args.initial)
    report = dict(geometry_data)
    report.update(hashes)
    report.update(classification='privileged_offline_full_point_local_time_search',
                  scipy_version=scipy.__version__, maximum_solver_iterations=args.iterations,
                  is_lap_time_lower_bound=False, is_driving_episode=False, is_legal_camera_candidate=False, rows=rows)
    report['limits'] = geometry_data['limits']+['Fixed supporting-plane passing sides restrict search.',
                                              'Finite-difference local optimization is not global certification.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)


if __name__ == '__main__':
    main()
