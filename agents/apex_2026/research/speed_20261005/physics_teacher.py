"""PRIVILEGED track/pose/wheel-state racing-line diagnostic, never an Agent.

Four already-open mandatory cells only. Complete road/obstacle state and a
100m planning horizon are unavailable to legal camera inference. Official
physics, damage, reward, watchdog and finish conditions remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_dynamics import ROOT, digest
from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec

ARCHIVE = ROOT/'agents/apex_2026/results/cloud-20261004/physics-teacher/apex_privileged_teacher_v1.py.txt'
OFFICIAL_FILES = ('env_wrapper.py', 'damage.py', 'core/vendor/car_racing.py',
                  'core/vendor/car_dynamics.py', 'core/finish_line.py',
                  'core/track_variables.py', 'core/obstacle_contacts.py')
CELLS = ((1, 516237), (2, 644062), (3, 1007), (4, 18800))
SETTINGS = {
    'legacy_reference': dict(control='archived', cruise=110., lateral=180., braking=85.,
                             preview_time=.17, shift=3.7, before=32., after=22.),
    'legacy_broad_return': dict(control='legacy', cruise=110., lateral=180., braking=85.,
                                preview_time=.17, shift=3.7, before=32., after=55.),
    'legacy_minimal_hold': dict(control='legacy', cruise=110., lateral=180., braking=85.,
                                preview_time=.17, shift=1.3, before=40., after=55.),
    'feedforward_minimal': dict(control='feedforward', cruise=100., lateral=190., braking=105.,
                               preview_time=.20, shift=1.3, before=40., after=55.,
                               ff_blend=.65, heading_gain=.25, yaw_gain=.04),
    'legacy_brake_ff': dict(control='legacy_brake_ff', cruise=100., lateral=190., braking=85.,
                           preview_time=.17, shift=3.7, before=32., after=55.),
    'continuous_brake_ff': dict(control='continuous_brake_ff', cruise=100., lateral=190., braking=85.,
                               preview_time=.20, shift=3.7, before=32., after=55.),
    'optimized_190': dict(control='continuous_brake_ff', cruise=100., lateral=190., braking=85.,
                          preview_time=.20, shift=3.7, before=32., after=55., optimized_route=True),
    'optimized_210': dict(control='continuous_brake_ff', cruise=100., lateral=210., braking=85.,
                          preview_time=.20, shift=3.7, before=32., after=55., optimized_route=True),
}
OPTIMIZED_ROUTE = ROOT/'agents/apex_2026/results/speed-20261005/physics-racingline-v1.json'


def archived_module():
    loader = importlib.machinery.SourceFileLoader('apex_archived_privileged_teacher', str(ARCHIVE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


def route_from_offsets(raw, params):
    track = np.asarray(raw.track, float)
    road = track[:, 2:4]
    normals = np.column_stack((np.cos(track[:, 1]), np.sin(track[:, 1])))
    ds = np.linalg.norm(np.roll(road, -1, axis=0)-road, axis=1)
    road_s = np.r_[0., np.cumsum(ds[:-1])]
    road_length = float(ds.sum())
    offsets = np.zeros(len(road))
    for obstacle in raw.obstacles:
        point = np.asarray(obstacle.position, float)
        index = int(np.argmin(np.sum((road-point)**2, axis=1)))
        lateral = float((point-road[index]) @ normals[index])
        distance = (road_s-road_s[index]+road_length/2.) % road_length-road_length/2.
        entry = (distance >= -params['before']) & (distance < 0)
        exit = (distance >= 0) & (distance <= params['after'])
        bump = np.zeros(len(road))
        bump[entry] = .5*(1+np.cos(np.pi*distance[entry]/params['before']))
        bump[exit] = .5*(1+np.cos(np.pi*distance[exit]/params['after']))
        offsets += (-1. if lateral >= 0 else 1.)*params['shift']*bump
    offsets = np.clip(offsets, -4., 4.)
    route = road+offsets[:, None]*normals
    route = .2*np.roll(route, 1, axis=0)+.6*route+.2*np.roll(route, -1, axis=0)
    links = np.roll(route, -1, axis=0)-route
    lengths = np.linalg.norm(links, axis=1)
    s = np.r_[0., np.cumsum(lengths[:-1])]
    previous = np.roll(links, 1, axis=0)
    turn = np.arctan2(previous[:, 0]*links[:, 1]-previous[:, 1]*links[:, 0],
                      np.sum(previous*links, axis=1))
    signed = turn / np.maximum(.5*(lengths+np.roll(lengths, 1)), .25)
    smooth_abs = (np.roll(abs(signed), -2)+2*np.roll(abs(signed), -1)+3*abs(signed)
                  +2*np.roll(abs(signed), 1)+np.roll(abs(signed), 2))/9.
    return road, route, s, float(lengths.sum()), signed, smooth_abs


def wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


class Teacher:
    def __init__(self, raw, params, old, track_id=None):
        self.raw, self.params = raw, params
        self.base = old.PrivilegedTeacher(raw, params)
        self.debug = {}
        self.signed = None
        if params['control'] != 'archived':
            road, route, s, length, signed, smooth = route_from_offsets(raw, params)
            self.base.road, self.base.route, self.base.s, self.base.length = road, route, s, length
            self.signed = signed
            self.base.kappa = smooth if params['control'] == 'legacy' else np.abs(signed)
        if params.get('optimized_route'):
            row = next(r for r in json.loads(OPTIMIZED_ROUTE.read_text())['rows'] if r['track_id'] == track_id)
            self.base.road = np.asarray(row['road'], float)
            self.base.route = np.asarray(row['route'], float)
            self.base.s = np.asarray(row['route_s'], float)
            self.base.length = row['route_length_m']
            self.signed = np.asarray(row['signed_curvature_per_m'], float)
            self.base.kappa = abs(self.signed)

    def point_at(self, distance):
        value = float(distance % self.base.length)
        index = int(np.searchsorted(self.base.s, value, side='right')-1) % len(self.base.route)
        next_index = (index+1) % len(self.base.route)
        segment = self.base.route[next_index]-self.base.route[index]
        length = float(np.linalg.norm(segment))
        fraction = (value-float(self.base.s[index]))/max(length, .01)
        return self.base.route[index]+fraction*segment

    def action(self):
        if self.params['control'] in ('archived', 'legacy'):
            action = self.base.action()
            self.debug = {'target_speed_mps': self.base.target_samples[-1],
                          'route_error_m': self.base.route_error_samples[-1]}
            return action
        hull = self.raw.car.hull
        xy = np.asarray(hull.position, float)
        velocity = np.asarray(hull.linearVelocity, float)
        speed = float(np.linalg.norm(velocity))
        ids = np.arange(self.base.last_index-5, self.base.last_index+34) % len(self.base.route)
        first = self.base.route[ids]
        segment = self.base.route[(ids+1) % len(self.base.route)]-first
        length_sq = np.sum(segment*segment, axis=1)
        fraction = np.clip(np.sum((xy-first)*segment, axis=1)/np.maximum(length_sq, .01), 0, 1)
        projected = first+fraction[:, None]*segment
        selected = int(np.argmin(np.sum((projected-xy)**2, axis=1)))
        if self.params['control'] == 'legacy_brake_ff':
            selected = int(np.argmin(np.sum((first-xy)**2, axis=1)))
            fraction[selected] = 0.
            projected[selected] = first[selected]
        index = int(ids[selected])
        self.base.last_index = index
        tangent = segment[selected]/math.sqrt(max(length_sq[selected], .01))
        anchor_s = float(self.base.s[index])+float(fraction[selected])*math.sqrt(length_sq[selected])
        preview = float(np.clip(7+self.params['preview_time']*speed, 12, 34))
        target = self.point_at(anchor_s+preview)
        if self.params['control'] == 'legacy_brake_ff':
            target_index = int(np.searchsorted(self.base.s, (anchor_s+preview) % self.base.length,
                                               side='left')) % len(self.base.route)
            target = self.base.route[target_index]
        angle = float(hull.angle)
        forward = np.array([-math.sin(angle), math.cos(angle)])
        right = np.array([math.cos(angle), math.sin(angle)])
        relative = target-xy
        pursuit = 2*float(relative @ right)/max(float(relative @ relative), 1.)
        desired = math.atan(3.24*pursuit/1.033)
        current_k = float(self.signed[index])
        heading_error = wrap(math.atan2(-tangent[0], tangent[1])-angle)
        cross_track = float((xy-projected[selected]) @ np.array([tangent[1], -tangent[0]]))
        if self.params['control'] == 'feedforward':
            joint_ff = math.atan(3.24*current_k/1.033)
            joint_pp = -desired
            desired_joint = (self.params['ff_blend']*joint_ff
                             +(1-self.params['ff_blend'])*joint_pp
                             +self.params['heading_gain']*heading_error
                             +math.atan2(.7*cross_track, max(speed, 20))
                             +self.params['yaw_gain']*(speed*current_k-float(hull.angularVelocity)))
            desired = -desired_joint
        grip = float(self.raw.car.grip_multiplier)
        lateral = self.params['lateral']*grip
        ahead = (self.base.s-anchor_s) % self.base.length
        mask = ahead <= 100.
        curve_speed = np.sqrt(lateral/np.maximum(self.base.kappa[mask], .0005))
        permitted = np.sqrt(curve_speed**2+2*self.params['braking']*np.maximum(ahead[mask]-5., 0.))
        limiting = int(np.argmin(permitted))
        descending_envelope = bool(ahead[mask][limiting] > 5. and permitted[limiting] < self.params['cruise'])
        target_speed = min(self.params['cruise'], float(np.min(permitted)),
                           math.sqrt(lateral/max(abs(pursuit), .0005)),
                           math.sqrt(lateral/max(abs(math.tan(desired)/3.24*1.033), .0005)))
        physical_limit = min(.4, math.atan(3.24*210.*grip/max(speed*speed*1.033, 100.)))
        desired = float(np.clip(desired, -physical_limit, physical_limit))
        self.base.last_steer += float(np.clip(desired-self.base.last_steer, -.24, .24))
        demanded_lateral = speed*speed*abs(math.tan(self.base.last_steer)/3.24*1.033)
        error = target_speed-speed
        if self.params['control'] in ('legacy_brake_ff', 'continuous_brake_ff'):
            feedforward = self.params['braking'] if descending_envelope and error < 3. else 0.
            requested_decel = max(0., feedforward-8.*error)
            available_decel = math.sqrt(max(0., (219.*grip)**2-demanded_lateral**2))
            if requested_decel > 3.:
                gas, brake = 0., min(.8, requested_decel/309., available_decel/309.)
            else:
                cap = (1. if demanded_lateral < 45. and speed < 90. else
                       .65 if demanded_lateral < 80. else .3 if demanded_lateral < 150. else .16)
                gas, brake = float(np.clip(.35+.09*error, 0., cap)), 0.
                if demanded_lateral > 60.:
                    excess = max(w.omega*w.wheel_rad-float(np.asarray(w.linearVelocity) @
                                 np.asarray(w.GetWorldVector((0, 1)))) for w in self.raw.car.wheels[2:])
                    if excess > 2.:
                        gas = min(gas, .05)
        elif error < -1.5:
            gas, brake = 0., float(np.clip(-error*.03, .04, .8))
        else:
            cap = 1. if speed < 35 and demanded_lateral < 40 else (
                .6 if demanded_lateral < 60 else .3 if demanded_lateral < 140 else .16)
            gas, brake = float(np.clip(.35+.06*error, 0., cap)), 0.
            if demanded_lateral > 60:
                excess = max(w.omega*w.wheel_rad-float(np.asarray(w.linearVelocity) @
                             np.asarray(w.GetWorldVector((0, 1)))) for w in self.raw.car.wheels[2:])
                if excess > 2.:
                    gas = min(gas, .05)
        self.debug = {'target_speed_mps': target_speed, 'route_error_m': float(np.linalg.norm(xy-projected[selected])),
                      'cross_track_m': cross_track, 'heading_error_rad': heading_error,
                      'path_curvature_per_m': current_k, 'pursuit_curvature_per_m': pursuit,
                      'demanded_lateral_mps2': demanded_lateral, 'preview_m': preview,
                      'descending_speed_envelope': descending_envelope}
        return np.asarray([self.base.last_steer, gas, brake], np.float32)


def quantiles(values):
    return {'min': float(np.min(values)), 'p50': float(np.percentile(values, 50)),
            'p90': float(np.percentile(values, 90)), 'max': float(np.max(values))}


def run_cell(name, params, track, seed, old, raw_dir, max_steps):
    spec = MapSpec(track, seed, 'official', (), max_steps, 4)
    environment, raw = create_environment(spec, render_mode=None)
    samples, contacts, offroad, partial = [], 0, 0, 0
    action_hash = hashlib.sha256()
    try:
        reset_environment(environment, spec)
        start = float(raw.t)
        teacher = Teacher(raw, params, old, track)
        info = {}
        for step in range(max_steps):
            action = teacher.action()
            action_hash.update(action.tobytes())
            _, reward, terminated, truncated, info = environment.step(action)
            contacts += int(bool(info.get('collision', False)))
            wheel_road = [bool(w.tiles) for w in raw.car.wheels]
            offroad += int(not any(wheel_road)); partial += int(not all(wheel_road))
            v = np.asarray(raw.car.hull.linearVelocity, float)
            body_f = np.asarray(raw.car.hull.GetWorldVector((0, 1)), float)
            body_r = np.asarray(raw.car.hull.GetWorldVector((1, 0)), float)
            samples.append({'step': step, 'sim_time_s': float(raw.t)-start,
                            'action': action.tolist(), 'speed_mps': float(np.linalg.norm(v)),
                            'yaw_radps': float(raw.car.hull.angularVelocity),
                            'sideslip_deg': math.degrees(math.atan2(float(v @ body_r), float(v @ body_f))),
                            'position': list(raw.car.hull.position), 'progress': float(info.get('progress', 0.)),
                            'collision': bool(info.get('collision', False)), 'wheels_on_road': sum(wheel_road),
                            'rear_roll_mps': [float(w.omega*w.wheel_rad) for w in raw.car.wheels[2:]],
                            'reward': float(reward), 'debug': dict(teacher.debug)})
            if terminated or truncated:
                break
        trace_path = raw_dir/f'physics-teacher-v2-{name}-track{track}-seed{seed}.json'
        with trace_path.open('x') as file:
            json.dump(samples, file, separators=(',', ':'))
        finish = raw.finish_time_s
        row = {'setting': name, 'parameters': params, 'track_id': track, 'seed': seed,
               'finished': finish is not None, 'lap_time_ms': round(1000*(finish-start)) if finish is not None else None,
               'progress': float(info.get('progress', 0.)), 'collision_count': contacts,
               'retire_reason': info.get('retire_reason'), 'steps': len(samples),
               'elapsed_sim_ms': round(1000*(float(raw.t)-start)), 'offroad_samples': offroad,
               'partial_offroad_samples': partial, 'damage': float(info.get('damage', 0.)),
               'route_length_m': teacher.base.length, 'route_max_curvature_per_m': float(np.max(teacher.base.kappa)),
               'speed_mps': quantiles([s['speed_mps'] for s in samples]),
               'abs_sideslip_deg': quantiles([abs(s['sideslip_deg']) for s in samples]),
               'route_error_m': quantiles([s['debug']['route_error_m'] for s in samples]),
               'action_trace_sha256': action_hash.hexdigest(), 'raw_trace_path': str(trace_path),
               'raw_trace_sha256': digest(trace_path), 'diagnostic_checkpoints': samples[::max(1, len(samples)//6)]}
        return row
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=SETTINGS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=450)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.freeze.json').exists():
        parser.error('output or freeze exists')
    if not 1 <= args.max_steps <= 600:
        parser.error('max-steps must be bounded at600')
    source_hash, archive_hash = digest(__file__), digest(ARCHIVE)
    before = {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    helper_hash = digest(ROOT/'agents/apex_2026/research/speed_20261005/physics_dynamics.py')
    route_hash = digest(OPTIMIZED_ROUTE)
    freeze = {'classification': 'privileged_official_mandatory_diagnostic_not_legal_agent',
              'is_legal_camera_candidate': False, 'is_prospective_development_trial': False,
              'source_sha256': source_hash, 'archived_teacher_sha256': archive_hash,
              'helper_sha256': helper_hash, 'official_file_sha256': before,
              'optimized_route_sha256': route_hash,
              'setting': args.setting, 'parameters': SETTINGS[args.setting],
              'cells': CELLS, 'max_actions': args.max_steps, 'python_version': sys.version,
              'limitations': ['Complete road/obstacle, pose, velocity and wheel state are privileged.',
                             '100m speed planning horizon exceeds camera visibility.',
                             'No tested route/control certifies global optimum or a lap-time lower bound.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.freeze.json').open('x') as file:
        json.dump(freeze, file, indent=2)
    raw_dir = ROOT/'.haic-artifacts/apex-speed-20261005'
    raw_dir.mkdir(parents=True, exist_ok=True)
    old = archived_module()
    rows = []
    began = time.perf_counter()
    for track, seed in CELLS:
        row = run_cell(args.setting, SETTINGS[args.setting], track, seed, old, raw_dir, args.max_steps)
        assert digest(__file__) == source_hash and digest(ARCHIVE) == archive_hash
        assert before == {p: digest(ROOT/p) for p in OFFICIAL_FILES}
        assert digest(ROOT/'agents/apex_2026/research/speed_20261005/physics_dynamics.py') == helper_hash
        assert digest(OPTIMIZED_ROUTE) == route_hash
        rows.append(row)
        cell_path = args.output.with_name(args.output.stem+f'-track{track}.json')
        with cell_path.open('x') as file:
            json.dump({'freeze': freeze, 'row': row}, file, indent=2)
        print(json.dumps({k: row[k] for k in ('setting', 'track_id', 'finished', 'lap_time_ms',
                                             'progress', 'collision_count', 'retire_reason')}), flush=True)
    with args.output.open('x') as file:
        json.dump({'freeze': freeze, 'rows': rows, 'wall_time_s': time.perf_counter()-began}, file, indent=2)


if __name__ == '__main__':
    main()
