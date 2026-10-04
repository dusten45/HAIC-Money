"""Offline fixed-observed-path speed envelopes. NO simulator imports or resets.

Run from repository root: .venv/bin/python agents/apex_2026/v2/diagnostics/ideal_speed.py
--output agents/apex_2026/v2/results/ideal-speed.json
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MASS = 7.3019199669361115
INERTIA = 4000*.02**2
RADIUS = 27*.02
FORCE = 1000000*.02**2
EFFECTIVE_MASS = MASS+4*INERTIA/RADIUS**2
ENGINE_COEFFICIENT = 2*(100000000*.02**2)/EFFECTIVE_MASS
FRICTION_ACCEL = 4*FORCE/MASS
REAR_ACCEL = 2*FORCE/MASS
BRAKE_COEFFICIENT = 4*INERTIA*15/(.02*RADIUS*EFFECTIVE_MASS)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curvature(points, radius_m=None):
    """Unsigned curvature of GPS chords; optional spatial quadratic fit."""
    p = np.asarray(points, float)
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0., np.cumsum(ds)]
    k = np.zeros(len(p))
    for i in range(1, len(p)-1):
        u, v = p[i]-p[i-1], p[i+1]-p[i]
        k[i] = 2*abs(u[0]*v[1]-u[1]*v[0])/max(np.linalg.norm(u)*np.linalg.norm(v)*np.linalg.norm(u+v), 1e-15)
    k[0], k[-1] = k[1], k[-2]
    if radius_m is not None:
        for i in range(len(p)):
            indices = np.flatnonzero(abs(s-s[i]) <= radius_m)
            if len(indices) < 3:
                indices = np.argsort(abs(s-s[i]))[:3]
            q = np.polyfit(s[indices]-s[i], p[indices], 2)
            d, dd = q[1], 2*q[0]
            k[i] = abs(d[0]*dd[1]-d[1]*dd[0])/max(float(np.linalg.norm(d)**3), 1e-15)
    return k


def envelope(s, caps, initial_speed, acceleration, brake):
    """Independent lateral/longitudinal constraints; final speed unrestricted.

    Backward constant-brake envelope followed by forward engine envelope.
    Squared-speed midpoint integration, then constant-acceleration segment time.
    """
    q = np.asarray(caps, float)**2
    for i in range(len(q)-2, -1, -1):
        q[i] = min(q[i], q[i+1]+2*brake*(s[i+1]-s[i]))
    q[0] = min(q[0], initial_speed**2)
    for i in range(len(q)-1):
        ds = s[i+1]-s[i]
        lo, hi = q[i], q[i]+2*acceleration(float(np.sqrt(q[i])))*ds
        for _ in range(36):
            mid = (lo+hi)/2
            residual = mid-q[i]-2*acceleration(float(np.sqrt((q[i]+mid)/2)))*ds
            if residual > 0:
                hi = mid
            else:
                lo = mid
        q[i+1] = min(q[i+1], (lo+hi)/2)
    v = np.sqrt(q)
    return v, float(np.sum(2*np.diff(s)/np.maximum(v[:-1]+v[1:], 1e-12)))


def episode(path, resolution):
    receipt = json.loads(path.read_text())
    trace = path.with_suffix('.jsonl')
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    points = [rows[0]['before']['position']]
    for row in rows:
        b, a = row['before'], row['after']
        dt = min(a['sim_time_s'], receipt['finish_time_s'])-b['sim_time_s']
        if dt <= 0:
            break
        fraction = dt/(a['sim_time_s']-b['sim_time_s'])
        points.append((np.asarray(b['position'])+fraction*(np.asarray(a['position'])-b['position'])).tolist())
    p = np.asarray(points)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1)>1e-8]]
    original_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.linspace(0, original_s[-1], int(np.ceil(original_s[-1]/resolution))+1)
    cases = {}
    for smoothing in [None, 8., 16.]:
        k_original = curvature(p, smoothing)
        k = np.interp(s, original_s, k_original)
        table = {}
        for lat, brake, top in itertools.product([100., FRICTION_ACCEL], [65., FRICTION_ACCEL], [100., None]):
            name = f'lat{lat:.3f}_brake{brake:.3f}_max{top}'
            caps = np.sqrt(lat/np.maximum(k, 1e-12))
            if top is not None:
                caps = np.minimum(caps, top)
            v, time = envelope(s, caps, rows[0]['before']['speed_m_s'],
                lambda speed: min(REAR_ACCEL, ENGINE_COEFFICIENT/(speed+5*RADIUS)), brake)
            table[name] = {'lateral_accel': lat, 'brake_accel': brake, 'speed_limit': top,
                'time_s': time, 'speed_min_max_m_s': [float(v.min()), float(v.max())],
                'fraction_distance_speed_gt100': float(np.sum(np.diff(s)[(v[:-1]+v[1:])/2>100])/s[-1])}
        baseline = table['lat100.000_brake65.000_max100.0']['time_s']
        best_capped = table[f'lat{FRICTION_ACCEL:.3f}_brake{FRICTION_ACCEL:.3f}_max100.0']['time_s']
        best_unbounded = table[f'lat{FRICTION_ACCEL:.3f}_brake{FRICTION_ACCEL:.3f}_maxNone']['time_s']
        marginal = {}
        for axis, column, low, high in [('lateral','lateral_accel',100.,FRICTION_ACCEL), ('braking','brake_accel',65.,FRICTION_ACCEL), ('speed_limiter','speed_limit',100.,None)]:
            savings = []
            other = [x for x in ['lateral_accel','brake_accel','speed_limit'] if x != column]
            for a in table.values():
                if a[column] != low:
                    continue
                b = next(x for x in table.values() if x[column] == high and all(x[c] == a[c] for c in other))
                savings.append(a['time_s']-b['time_s'])
            marginal[axis] = {'minimum_s': min(savings), 'maximum_s': max(savings)}
        cases[str(smoothing)] = {'curvature_abs_per_m_p50_p90_max': np.percentile(k,[50,90,100]).tolist(),
            'scenarios': table, 'baseline_proxy_s': baseline, 'mechanics_capped_proxy_s': best_capped,
            'mechanics_unbounded_counterfactual_s': best_unbounded,
            'observed_minus_baseline_proxy_s': receipt['lapTimeMs']/1000-baseline,
            'axis_saving_range_across_other_factor_settings_s': marginal}
    return {'track_id':receipt['track_id'], 'seed':receipt['seed'], 'observed_lap_s':receipt['lapTimeMs']/1000,
        'source_receipt':str(path), 'receipt_sha256':sha(path), 'trace_sha256':sha(trace),
        'observed_chord_distance_m':float(s[-1]), 'initial_speed_m_s':rows[0]['before']['speed_m_s'],
        'curvature_sensitivity':cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('/tmp/apex-final-frozen/required'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution-m', type=float, default=.5)
    args = parser.parse_args()
    if args.resolution_m <= 0:
        parser.error('resolution must be positive')
    records = [episode(p,args.resolution_m) for p in sorted(args.input_dir.glob('t*.json'))]
    assert len(records)==4
    source = ROOT/'core/vendor/car_dynamics.py'
    result = {'schema_version':1, 'scope':'P1 required4 observed GPS trajectories only; zero resets; no new holdout access',
        'analysis_sha256':sha(Path(__file__)), 'public_dynamics_sha256':sha(source),
        'mass_measurement_source': 'agents/apex_2026/results/feasibility.json',
        'mass_measurement_source_sha256': sha(ROOT/'agents/apex_2026/results/feasibility.json'),
        'resolution_m':args.resolution_m,
        'constants':{'total_mass':MASS, 'wheel_rotational_inertia':INERTIA, 'wheel_radius':RADIUS,
            'rolling_effective_mass':EFFECTIVE_MASS, 'engine_acceleration_numerator':ENGINE_COEFFICIENT,
            'engine_acceleration_denominator_offset':5*RADIUS,
            'four_wheel_total_friction_COM_accel':FRICTION_ACCEL, 'rear_pair_friction_COM_accel':REAR_ACCEL,
            'rolling_brake_acceleration_per_unit_command':BRAKE_COEFFICIENT},
        'assumptions':[
            'Fixed GPS chord path, including actual deviations, truncated at true finish timestamp; not centerline or a globally shortest legal path.',
            'GPS curvature estimated from three-point circumcircles; quadratic spatial fits with8m/16m radius are sensitivity checks that may erase real curvature. Endpoints extrapolate nearest curvature.',
            'Initial measured speed, unrestricted terminal speed. Global future curvature known; ideal instantaneous tracking, no steering limits, no obstacles, no damage, no grass, no throttle ramp.',
            'Independent lateral and longitudinal bounds deliberately omit their shared friction circle and axle torque/traction allocation: optimistic relaxation, not dynamically feasible.',
            'Engine acceleration2735/(v+2.7) uses rolling effective mass M+4I/r². It is an approximate steady rolling model, not a universal transient bound. Driven rear-pair COM friction clips acceleration at2F/M.',
            'Braking219m/s² is optimistic4F/M COM bound. Partial-brake rolling relation≈303.9*b is not applicable at b>=.9, which locks wheel angular velocity. Simultaneous cornering reduces available braking.',
            'max100 retains approximate Box2D2m/raw20ms translation restriction. maxNone explicitly removes that official numerical restriction: engine-only counterfactual, NOT an official permitted speed upgrade.',
            'Neither baseline nor mechanics result is a guaranteed lower bound or executable lap. Finite GPS sampling, curvature fits, rolling approximation and ideal tracking can bias in either direction.',
            'Marginal savings ranges compare factorial settings; they overlap and must not be summed. Actual minus baseline proxy is not an identified removable controller loss.'
        ], 'episodes':records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in records:
        print(r['track_id'],[(k,round(v['baseline_proxy_s'],3),round(v['mechanics_capped_proxy_s'],3),round(v['mechanics_unbounded_counterfactual_s'],3)) for k,v in r['curvature_sensitivity'].items()])


if __name__=='__main__':
    main()
