"""PRIVILEGED offline point-car racing-line model; no driving episodes.

Whole-track center points, obstacle positions and both road bounds are used.
The independent acceleration limits are optimistic: this model has no tire
force coupling, hull sweep, wheel spin, steering latency, damage or finish gate.
Its local numerical optimum is neither a realizable lap nor a lower bound.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from agents.apex_2026.research.speed_20261005.physics_dynamics import ROOT, digest
from agents.apex_2026.research.speed_20261005.physics_teacher import (
    CELLS, OFFICIAL_FILES, route_from_offsets,
)
from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


def geometry(route):
    links = np.roll(route, -1, axis=0)-route
    ds = np.linalg.norm(links, axis=1)
    previous = np.roll(links, 1, axis=0)
    turn = np.arctan2(previous[:, 0]*links[:, 1]-previous[:, 1]*links[:, 0],
                      np.sum(previous*links, axis=1))
    signed = turn/np.maximum(.5*(ds+np.roll(ds, 1)), .1)
    return ds, signed


def speed_profile(route, lateral=219., acceleration=40., braking=150.):
    ds, signed = geometry(route)
    cap = np.minimum(100., np.sqrt(lateral/np.maximum(abs(signed), 1e-7)))
    # One full circuit from rest; terminal speed is free, not a repeated stop.
    speeds = np.r_[cap, cap[0]]
    speeds[0] = 0.
    for i in range(len(ds)):
        speeds[i+1] = min(speeds[i+1], np.sqrt(speeds[i]**2+2*acceleration*ds[i]))
    for i in range(len(ds)-1, -1, -1):
        speeds[i] = min(speeds[i], np.sqrt(speeds[i+1]**2+2*braking*ds[i]))
    duration = float(np.sum(2*ds/np.maximum(speeds[:-1]+speeds[1:], .1)))
    return duration, speeds, ds, signed


def obstacle_clearance(route, obstacles):
    links = np.roll(route, -1, axis=0)-route
    length_sq = np.maximum(np.sum(links*links, axis=1), .01)
    result = []
    for point in obstacles:
        relative = point-route
        fraction = np.clip(np.sum(relative*links, axis=1)/length_sq, 0., 1.)
        closest = route+fraction[:, None]*links
        result.append(float(np.min(np.linalg.norm(closest-point, axis=1))))
    return np.asarray(result)


def optimize(raw, max_iterations):
    track = np.asarray(raw.track, float)
    road = track[:, 2:4]
    normals = np.column_stack((np.cos(track[:, 1]), np.sin(track[:, 1])))
    obstacles = np.asarray([list(body.position) for body in raw.obstacles], float)
    count = len(road)
    knots = max(24, min(60, count//5))
    sample = np.arange(count)/count
    knot_x = np.linspace(0., 1., knots+1)
    basis = np.eye(knots)
    matrix = CubicSpline(knot_x, np.vstack((basis, basis[0])),
                         bc_type='periodic', axis=0)(sample)
    _, initial_route, *_ = route_from_offsets(raw, dict(shift=4.1, before=40., after=55.))
    initial_offsets = np.sum((initial_route-road)*normals, axis=1)
    initial = np.clip(np.linalg.lstsq(matrix, initial_offsets, rcond=None)[0], -4.8, 4.8)

    def decode(values):
        offsets = matrix@values
        return road+offsets[:, None]*normals, offsets

    def constraints(values):
        route, offsets = decode(values)
        return np.r_[4.8-offsets, 4.8+offsets, obstacle_clearance(route, obstacles)-3.7]

    def objective(values):
        route, offsets = decode(values)
        duration, _, _, _ = speed_profile(route)
        second = np.roll(offsets, 1)-2*offsets+np.roll(offsets, -1)
        return duration+.001*float(np.sum(second*second))

    start = time.perf_counter()
    fitted = minimize(objective, initial, method='SLSQP', bounds=[(-4.8, 4.8)]*knots,
                      constraints={'type': 'ineq', 'fun': constraints},
                      options={'maxiter': max_iterations, 'ftol': 2e-7, 'disp': False})
    route, offsets = decode(fitted.x)
    duration, speeds, ds, signed = speed_profile(route)
    clearance = obstacle_clearance(route, obstacles)
    baseline_time = speed_profile(initial_route)[0]
    route_s = np.r_[0., np.cumsum(ds[:-1])]
    return {'road': road.tolist(), 'road_normals': normals.tolist(),
            'route': route.tolist(), 'route_s': route_s.tolist(),
            'signed_curvature_per_m': signed.tolist(), 'speed_profile_mps': speeds.tolist(),
            'offsets_m': offsets.tolist(), 'obstacle_positions': obstacles.tolist(),
            'model_time_s': duration, 'route_length_m': float(ds.sum()),
            'initial_broad_route_model_time_s': baseline_time,
            'maximum_abs_offset_m': float(np.max(abs(offsets))),
            'minimum_obstacle_center_clearance_m': float(clearance.min()),
            'obstacle_center_clearances_m': clearance.tolist(),
            'max_abs_curvature_per_m': float(np.max(abs(signed))),
            'solver_success': bool(fitted.success), 'solver_message': str(fitted.message),
            'solver_iterations': int(fitted.nit), 'solver_evaluations': int(fitted.nfev),
            'minimum_constraint_slack_m': float(constraints(fitted.x).min()),
            'solver_wall_time_s': time.perf_counter()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=90)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.iterations <= 150:
        parser.error('output must be new and iterations bounded at150')
    official = {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    source_hash = digest(__file__)
    rows = []
    for track, seed in CELLS:
        spec = MapSpec(track, seed, 'official', (), 1, 4)
        environment, raw = create_environment(spec, render_mode=None)
        try:
            reset_environment(environment, spec)
            row = optimize(raw, args.iterations)
            row.update(track_id=track, seed=seed)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('track_id', 'model_time_s', 'route_length_m',
                              'solver_success', 'minimum_constraint_slack_m',
                              'solver_iterations', 'solver_wall_time_s')}), flush=True)
        finally:
            environment.close()
    assert official == {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    assert digest(__file__) == source_hash
    report = {'classification': 'privileged_offline_optimistic_point_car_local_optimization',
              'is_legal_camera_candidate': False, 'is_driving_episode': False,
              'is_lap_time_lower_bound': False, 'source_sha256': source_hash,
              'official_file_sha256': official, 'scipy_version': scipy.__version__,
              'model': {'speed_cap_mps': 100., 'lateral_mps2': 219.,
                        'acceleration_mps2': 40., 'braking_mps2': 150.,
                        'offset_bounds_m': [-4.8, 4.8], 'obstacle_center_separation_m': 3.7,
                        'initial_speed_mps': 0., 'smoothness_objective_weight': .001},
              'limits': ['Local numerical search, not certified global optimum.',
                         'Independent longitudinal/lateral limits omit tire-force coupling.',
                         'Center-point road bounds and circle distance omit rotated hull sweep.',
                         'No steering latency, wheel spin, damage, tracking error or finish gate.',
                         'Entire closed route modeled; official qualification differs.'],
              'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)


if __name__ == '__main__':
    main()
