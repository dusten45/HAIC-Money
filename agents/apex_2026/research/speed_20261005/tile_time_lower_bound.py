"""Offline tile-contact geometry relaxation; never a driving Agent or lap.

Only already-consumed mandatory seeds are reconstructed by official track math.
The sink below creates polygon shapes but no Box2D world, Car, reset or Step.
The conditional time conversion is separate from the geometric calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def qualification_count(n: int) -> int:
    if not isinstance(n, int) or n < 1:
        raise ValueError('positive integer tile count required')
    return (95*n + 99)//100


def interval_coverage_width(a, b, k: int, *, start=None) -> float:
    """Exact minimum span intersecting k closed intervals, optionally a point.

    For each sorted-a prefix j >= k, its kth-largest b is the largest possible
    minimum right endpoint of k intervals. Exhausting prefixes is exhaustive
    over all subsets' maximum left endpoint. No route order is imposed.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.ndim != 1 or a.shape != b.shape or not 1 <= k <= len(a)
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or np.any(a > b) or (start is not None and not math.isfinite(start))):
        raise ValueError('finite ordered intervals and valid coverage required')
    order = np.argsort(a, kind='stable')
    a, b = a[order], b[order]
    best = math.inf
    for j in range(k, len(a)+1):
        left = float(a[j-1])
        right = float(np.partition(b[:j], j-k)[j-k])
        width = (max(0., left-right) if start is None
                 else max(left, float(start))-min(right, float(start)))
        best = min(best, width)
    return best


def projection_integral_certificate(vertices, k: int, contact_radius_m: float,
                                    angle_bins: int, *, start=None) -> dict:
    """Bound integral of exact relaxed widths using a Lipschitz certificate.

    The result is an integral bound, not an upper bound on a realizable lap.
    W(theta) is 2D-Lipschitz when every polygon vertex and the optional starting
    point lie within radius D of the chosen origin. Each midpoint bin's error
    is at most 2D * bin_width**2 / 4.
    """
    points = np.asarray(vertices, float)
    if (points.ndim != 3 or points.shape[-1] != 2 or not len(points)
            or not np.all(np.isfinite(points)) or not 1 <= k <= len(points)
            or not math.isfinite(contact_radius_m) or contact_radius_m < 0
            or not isinstance(angle_bins, int) or angle_bins < 4):
        raise ValueError('finite polygons, nonnegative radius and valid bins required')
    required_start = None if start is None else np.asarray(start, float)
    if required_start is not None and (required_start.shape != (2,)
                                      or not np.all(np.isfinite(required_start))):
        raise ValueError('finite two-dimensional start required')
    origin = (points.reshape(-1, 2).min(axis=0)+points.reshape(-1, 2).max(axis=0))/2
    centered = points-origin
    radius = float(np.max(np.linalg.norm(centered, axis=-1)))
    if required_start is not None:
        required_start = required_start-origin
        radius = max(radius, float(np.linalg.norm(required_start)))
    h = math.pi/angle_bins
    widths = []
    for index in range(angle_bins):
        theta = (index+.5)*h
        normal = np.array([math.cos(theta), math.sin(theta)])
        projected = centered@normal
        a = projected.min(axis=1)-contact_radius_m
        b = projected.max(axis=1)+contact_radius_m
        point = None if required_start is None else float(required_start@normal)
        widths.append(interval_coverage_width(a, b, k, start=point))
    widths = np.asarray(widths)
    bin_error = 2*radius*h*h/4
    # This is a generous floating-point guard, not interval-arithmetic proof of
    # libm/IEEE rounding. The quadrature-error inequality itself is analytic.
    roundoff_guard = 1e-6
    return {'lower_m': max(0., float(np.maximum(0., h*widths-bin_error).sum())-roundoff_guard),
            'upper_m': float((h*widths+bin_error).sum())+roundoff_guard,
            'midpoint_estimate_m': float(h*widths.sum()),
            'total_quadrature_error_bound_m': float(angle_bins*bin_error),
            'floating_roundoff_guard_m_not_machine_verified': roundoff_guard,
            'lipschitz_per_radian_m': 2*radius,
            'angle_bins': angle_bins,
            'contact_radius_m': contact_radius_m,
            'start_included': start is not None}


def consumed_layout_without_world(seed: int):
    """Run unmodified official _create_track with a nonphysics geometry sink."""
    from Box2D.b2 import polygonShape
    from gymnasium.utils import seeding
    from core.vendor.car_racing import CarRacing

    class GeometrySink:
        def __init__(self):
            self.polygons = []

        def CreateStaticBody(self, *, fixtures):
            self.polygons.append(np.array(fixtures.shape.vertices, dtype=float))
            return SimpleNamespace(fixtures=[SimpleNamespace(sensor=False)])

    sink = GeometrySink()
    state = SimpleNamespace(np_random=seeding.np_random(seed)[0], verbose=False,
                            road_color=np.full(3, 102.), road_poly=[], world=sink,
                            fd_tile=SimpleNamespace(shape=polygonShape()))
    attempts = 0
    while True:
        attempts += 1
        if attempts > 100:
            raise RuntimeError('unexpected track-generation retries')
        if CarRacing._create_track(state):
            break
    polygons = np.stack(sink.polygons)
    centers = np.asarray(state.track, float)[:, 2:4]
    return polygons, centers, np.asarray(state.track, float), attempts


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--angle-bins', type=int, default=1024)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    consumed = [(1, 516237), (2, 644062), (3, 1007), (4, 18800)]
    receipt_path = root/'agents/apex_2026/results/speed-20261005/rear-clear-v1.json'
    receipt = json.loads(receipt_path.read_text())
    checkpoint = {(r['track_id'], r['seed']): r for r in receipt['rows']}
    rows = []
    for track_id, seed in consumed:
        polygons, centers, track, attempts = consumed_layout_without_world(seed)
        n = len(polygons)
        k = qualification_count(n)
        prior = checkpoint[track_id, seed]
        trace_path = root/f'.haic-artifacts/apex-speed-20261005/rear-clear-v1-traces/track-{track_id}-seed-{seed}.json'
        trace = json.loads(trace_path.read_text())
        action_hash = hashlib.sha256(np.array([r['action'] for r in trace],
                                             dtype=np.float32).tobytes()).hexdigest()
        if action_hash != prior['action_trace_sha256'] or len(trace) != prior['steps']:
            raise ValueError('saved actions do not match source-bound RearClear receipt')
        if any(abs(r['progress']*n-round(r['progress']*n)) > 1e-8 for r in trace):
            raise ValueError('reconstructed tile count disagrees with recorded progress')
        sampled_positions = np.array([r['position'] for r in trace], dtype=float)
        sampled_length = float(np.linalg.norm(np.diff(sampled_positions, axis=0), axis=1).sum())
        previous_time = prior['lap_time_ms']/1000
        # 3m generously encloses nominal four-wheel contacts relative to car COM.
        # 10m closing segment assumes rigid collision-free ordinary integration,
        # start near the finish center, and normal candidate finish semantics.
        # These assumptions are NOT universal Box2D collision/solver bounds.
        beta = float(track[0, 1])
        # Nominal compound COM relative to the initial hull origin.
        c = np.array([0., -.0804589843])
        rotation = np.array([[math.cos(beta), -math.sin(beta)],
                             [math.sin(beta), math.cos(beta)]])
        start = centers[0]+rotation@c
        integral = projection_integral_certificate(polygons, k, 3., args.angle_bins,
                                                    start=start)
        geometric = max(0., integral['lower_m']-10.)
        wheel_integral = projection_integral_certificate(polygons, k,
                                                        math.hypot(.28, .54)+.02,
                                                        args.angle_bins)
        network_lower = max(0., wheel_integral['lower_m']/2-7.64)
        centerline_length = float(np.linalg.norm(centers-np.roll(centers, 1, axis=0),
                                                axis=1).sum())
        projections = projection_integral_certificate(polygons, n, 0., args.angle_bins)
        rows.append({'track_id': track_id, 'seed': seed, 'tile_count': n,
                     'required_unique_tiles': k, 'allowed_omitted_tiles': n-k,
                     'track_generation_attempts': attempts,
                     'geometry_sha256': hashlib.sha256(polygons.tobytes()).hexdigest(),
                     'track_points_sha256': hashlib.sha256(track.tobytes()).hexdigest(),
                     'official_centerline_polygon_length_m': centerline_length,
                     'sampled_rear_clear_path_length_m_approx': sampled_length,
                     'rear_clear_finish_time_s': previous_time,
                     'rear_clear_trace_sha256': file_sha(trace_path),
                     'rear_clear_actions_sha256_verified_against_receipt': action_hash,
                     'rear_clear_trace_last_physics_time_s': trace[-1]['sim_time_s'],
                     'rear_clear_trace_first_physics_time_s': trace[0]['sim_time_s'],
                     'reconstructed_tile_count_matches_all_trace_progress': True,
                     'tile_projection_integral_certificate': integral,
                     'conditional_COM_path_lower_bound_m': geometric,
                     'conditional_100m_s_time_lower_bound_s': geometric/100,
                     'wheel_network_integral_certificate': wheel_integral,
                     'four_wheel_total_path_geometry_lower_bound_m': network_lower,
                     'conditional_4x100m_s_network_time_lower_bound_s': network_lower/400,
                     'all_tile_raw_projection_integral_certificate': projections,
                     'distance_budgets_without_startup_or_turn_penalty': [
                         {'target_s': t, '100m_s_budget_m': 100*t,
                          'sampled_path_required_cut_fraction': max(0., 1-100*t/sampled_length)}
                         for t in (10., 10.5, 11., 13.)]})
    output = {'purpose': 'offline consumed-layout geometric feasibility relaxation; no laps or Agent',
              'worlds_created': 0, 'physics_steps': 0, 'holdout_opened': False,
              'official_source_sha256': {name: file_sha(root/name) for name in
                  ['core/vendor/car_racing.py', 'core/vendor/car_dynamics.py', 'core/finish_line.py']},
              'helper_sha256': file_sha(__file__),
              'rear_clear_receipt_sha256': file_sha(receipt_path),
              'rear_clear_source_sha256': receipt['freeze']['source_sha256'],
              'conditional_contact_radius_m': 3., 'conditional_closing_segment_cap_m': 10.,
              'conditional_speed_cap_m_s': 100.,
              'assumptions': ['nominal rigid vehicle joints for the 3m COM contact-radius model',
                  'collision-free ordinary Box2D integration and COM-preserving internal corrections',
                  'post-warmup starting COM near nominal finish center for the 10m closing segment',
                  '2m integration-stage COM displacement per .02s is NOT a global TOI/contact theorem'],
              'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'worlds_created': 0, 'physics_steps': 0,
                      'conditional_bounds_s': [r['conditional_100m_s_time_lower_bound_s'] for r in rows]}))


if __name__ == '__main__':
    main()
