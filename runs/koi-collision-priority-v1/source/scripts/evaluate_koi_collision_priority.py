"""Small, isolated CPU21 A/B on completed, consumed TRAIN development cells.

Only an explicit CLI run resets environments. Imports and metric helpers do not.
The original v2 ZIP is the comparator, never its submission derivative.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace')
PYTHON = Path('/tmp/kilo/haic-cpu21/bin/python')
V2_ZIP = ROOT / 'submissions/koi-steering-release-v2.zip'
V2_SHA = 'b1911d7dddf05d7c3f405b900f40c8c5696607130d2ef5e1cf473a82166147ce'
PRIOR = ROOT / 'runs/koi-steering-generalization-v1'
WRAPPER = ROOT / 'haic/algorithms/koi/collision_priority.py'
ARMS = ('steering_release_v2', 'collision_priority')
PRIOR_ARM = 'steering_release_v1'  # Historical name; protocol binds the v2 ZIP.
SCOPE = 'reused-development TRAIN; already consumed; NOT fresh, confirmation, blind, or official'
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'SDL_VIDEODRIVER': 'dummy',
       'SDL_AUDIODRIVER': 'dummy', 'PYGAME_HIDE_SUPPORT_PROMPT': '1',
       'PYTHONDONTWRITEBYTECODE': '1'}
METRIC_DEFINITIONS = {
    'collision_positive_decisions': 'Wrapper collision=True decisions, not physical contact events.',
    'off_track_count_max': 'Exact CarEnvironment.off_track_counter AFTER each decision; summed raw reward <0 increments, otherwise resets. Retirement iff counter>max_off_track_steps (default100). Not wheel contact.',
    'physical_contacts': 'Touching Box2D contacts between any hull/wheel fixture and each obstacle, sampled after every raw tick. Events count false-to-true transitions per object.',
    'clearance': 'Minimum signed circle-to-union-of-hull/wheel-fixtures separation, including fixture skins, over initial state and ALL post-warmup raw ticks, ALL objects, including failed episodes.',
    'no_wheel_road_contact': 'All four wheel.tiles sets empty. Consecutive post-warmup 50Hz samples; ticks/50 is sampled occupancy, not the reward-streak rule. Any-wheel-offroad is separately reported.',
    'road_reacquisition': 'For every no-wheel-road-contact onset, associate the nearest object by fixture clearance at onset (observational, not causal/controller identity). Passage is the first observed forward crossing of whole-fixture rear>object radius from rear<=radius, not an object already behind at reset. Measure onset to first sample with ANY wheel road contact at/after this passage, allowing passage before onset. Also retain first physical contact return separately, which may precede passage. If passage or subsequent contact is unobserved, right-censor at the last available tick; duration is then a lower bound, not a return time. Episode-end horizon; no returned-only population mean. Onset/return quantized to20ms; onset already offroad at reset is left-censored.',
    'lap_time': 'Only mutually finished valid pairs contribute to matched lap time. Repeats are deterministic development repetitions, not independent geometries.',
    'prefix': 'Exact initial geometry/state/pixel equality and action/observation/pre-state equality before first different action; first different action PRE-state equality reported separately.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write('\n')


def append_jsonl(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def parse_cells(value):
    try:
        cells = [tuple(map(int, cell.split(':'))) for cell in value.split(',')]
    except ValueError as error:
        raise ValueError('cells must be track:seed,track:seed,...') from error
    if not cells or any(len(cell) != 2 for cell in cells) or len(set(cells)) != len(cells):
        raise ValueError('cells must be distinct track:seed pairs')
    return cells


def completed_cells(report):
    """Completed execution includes natural DNFs; it does not mean a finished lap."""
    rows = {}
    for row in report['rows']:
        track, seed = row['track_id'], row['seed']
        if (row['mode'] == PRIOR_ARM and row['status'] == 'completed'
                and track in (1, 2, 3) and 3184000001 <= seed <= 3184000016
                and (track, seed) != (3, 3184000016)
                and row.get('error') is None and row.get('invalid_actions') == 0
                and (row.get('completed') is True or row.get('retire_reason') in ('off_track', 'crash'))):
            if (track, seed) in rows:
                raise ValueError('duplicate prior completed cell')
            rows[track, seed] = row
    return rows


def validate_cells(cells, prior=PRIOR):
    """Selected receipts only, not a broad provenance/freshness audit."""
    protocol, report = read_json(prior / 'protocol.json'), read_json(prior / 'episode-report.json')
    if protocol['model_hashes'][PRIOR_ARM] != V2_SHA:
        raise ValueError('parent comparator is not the original frozen v2 ZIP')
    if report['protocol_sha256'] != sha(prior / 'protocol.json'):
        raise ValueError('parent report/protocol mismatch')
    allowed = completed_cells(report)
    evidence = []
    for cell in cells:
        if cell not in allowed:
            raise ValueError(f'not a completed consumed v2 TRAIN cell: {cell}')
        row = allowed[cell]
        path = (prior / row['file']).resolve()
        if not path.is_relative_to(prior.resolve()) or sha(path) != row['sha256']:
            raise ValueError(f'prior selected episode mismatch: {cell}')
        episode = read_json(path)
        if ((episode['track_id'], episode['seed']) != cell or episode['mode'] != PRIOR_ARM
                or episode.get('error') or episode.get('invalid_actions')
                or not (episode.get('completed') or episode.get('retire_reason') in ('off_track', 'crash'))):
            raise ValueError(f'invalid prior selected episode: {cell}')
        evidence.append(dict(track_id=cell[0], seed=cell[1], file=str(path), sha256=row['sha256']))
    return evidence


def physical_metrics(states, obstacles) -> dict:
    """Pure observer analysis, including unpassed objects and censored departures."""
    if not states:
        return dict(observed=False)
    radii = [obj['radius'] for obj in obstacles]
    passages = [None] * len(obstacles)
    minima = [float('inf')] * len(obstacles)
    contacts, events = [0] * len(obstacles), [0] * len(obstacles)
    previous_contacts = set()
    streak = any_streak = max_streak = max_any = no_road_ticks = any_off_ticks = 0
    departures = []
    previous_off = False
    for index, state in enumerate(states):
        for obj, radius in enumerate(radii):
            minima[obj] = min(minima[obj], state['clearance'][obj])
            if (index and passages[obj] is None and states[index - 1]['rear'][obj] <= radius
                    and state['rear'][obj] > radius):
                passages[obj] = state['t']
        touched = set(state['contacts'])
        for obj in touched:
            contacts[obj] += int(index > 0)
            events[obj] += int(obj not in previous_contacts)
        previous_contacts = touched
        off = not any(state['wheel_road_contacts'])
        any_off = not all(state['wheel_road_contacts'])
        # Index0 is reset-end, not an additional elapsed physics tick.
        if index:
            streak = streak + 1 if off else 0
            any_streak = any_streak + 1 if any_off else 0
            no_road_ticks += int(off)
            any_off_ticks += int(any_off)
            max_streak, max_any = max(max_streak, streak), max(max_any, any_streak)
        if off and not previous_off:
            obj = min(range(len(obstacles)), key=lambda i: state['clearance'][i]) if obstacles else None
            departures.append(dict(index=index, object_id=obj, onset_t=state['t'], left_censored=index == 0))
        previous_off = off
    for event in departures:
        after = states[event.pop('index') + 1:]
        passage = passages[event['object_id']] if event['object_id'] is not None else None
        first_return = next((s['t'] for s in after if any(s['wheel_road_contacts'])), None)
        full_return = next((s['t'] for s in after if all(s['wheel_road_contacts'])), None)
        postpass_return = next((s['t'] for s in after if passage is not None and s['t'] >= passage
                                and any(s['wheel_road_contacts'])), None)
        event.update(passage_t=passage, first_contact_return_t=first_return,
                     first_all_wheels_contact_return_t=full_return,
                     first_contact_return_duration_s=None if first_return is None else first_return - event['onset_t'],
                     postpass_contact_return_t=postpass_return, censored=postpass_return is None,
                     censor_reason=('passage_unobserved' if passage is None else 'contact_return_unobserved') if postpass_return is None else None,
                     duration_or_censored_lower_bound_s=(postpass_return if postpass_return is not None else states[-1]['t']) - event['onset_t'])
    return dict(observed=True, raw_ticks=len(states) - 1,
                min_fixture_clearance=min(minima) if minima else None,
                objects=[dict(object_id=i, min_fixture_clearance=value, contact_ticks=contacts[i],
                              contact_events=events[i], passage_t=passages[i]) for i, value in enumerate(minima)],
                hit_objects=[i for i, count in enumerate(events) if count],
                physical_contact_ticks=sum(bool(s['contacts']) for s in states[1:]),
                physical_contact_events=sum(events), no_wheel_road_contact_ticks=no_road_ticks,
                no_wheel_road_contact_max_ticks=max_streak, no_wheel_road_contact_max_s=max_streak / 50.,
                any_wheel_offroad_ticks=any_off_ticks, any_wheel_offroad_max_ticks=max_any,
                any_wheel_offroad_max_s=max_any / 50., road_reacquisition=departures,
                road_reacquisition_censored=sum(event['censored'] for event in departures))


def pair_metrics(baseline, candidate) -> dict:
    left, right = baseline['decision_trace'], candidate['decision_trace']
    action = lambda row: [row[key] for key in ('steer', 'gas', 'brake')]
    shared = min(len(left), len(right))
    first = next((i for i in range(shared) if action(left[i]) != action(right[i])), None)
    prefix = shared if first is None else first
    same_state = lambda i: all(left[i]['controller']['evaluation_only'][key] == right[i]['controller']['evaluation_only'][key]
                               for key in ('observation_sha256', 'pre'))
    both = baseline['completed'] and candidate['completed']
    return dict(track_id=baseline['track_id'], seed=baseline['seed'], repeat=baseline['repeat'],
                geometry_equal=baseline['geometry_sha256'] == candidate['geometry_sha256'],
                initial_state_equal=baseline['initial_state'] == candidate['initial_state'],
                initial_pixels_equal=baseline['initial_observation_sha256'] == candidate['initial_observation_sha256'],
                equal_action_prefix_decisions=prefix, shared_decisions=shared,
                prefix_observation_and_pre_state_equal=all(same_state(i) for i in range(prefix)),
                first_changed_action_step=None if first is None else left[first]['step'],
                first_changed_action_pre_state_equal=None if first is None else same_state(first),
                all_actions_equal=len(left) == len(right) and first is None,
                both_finished=both, baseline_finished=baseline['completed'], candidate_finished=candidate['completed'],
                baseline_lap_ms=baseline['lapTimeMs'] if both else None,
                candidate_lap_ms=candidate['lapTimeMs'] if both else None,
                matched_lap_delta_ms=candidate['lapTimeMs'] - baseline['lapTimeMs'] if both else None)


def make_observer(environment, raw_stream, ledger, identity, protocol_sha256, footprint_measurements):
    """Geometry recipe copied from steering_release_ab; only observer sees world state."""
    import numpy as np

    class Observer:
        def __init__(self):
            self.inner, self.environment = environment, environment.environment
            self.active = False
            self.step_number = self.raw_count = self.reset_count = 0
            self.max_counter = 0
            self.last = {}
            raw_step, raw_reset = self.unwrapped.step, self.unwrapped.reset

            def observed_reset(*args, **kwargs):
                self.reset_count += 1
                append_jsonl(ledger, dict(identity, event='reset_intent', reset_index=self.reset_count,
                                         protocol_sha256=protocol_sha256, time_ns=time.time_ns()))
                return raw_reset(*args, **kwargs)

            def observed_step(action):
                result = raw_step(action)
                if self.active:
                    state = self.state()
                    state.update(step=self.step_number, collision=bool(result[4].get('collision', False)))
                    raw_stream.write(json.dumps(state, allow_nan=False) + '\n')
                    self.raw_count += 1
                return result

            self.unwrapped.reset, self.unwrapped.step = observed_reset, observed_step

        @property
        def unwrapped(self):
            return self.inner.unwrapped

        def reset(self):
            result = self.inner.reset()
            raw = self.unwrapped
            self.track = np.asarray(raw.track, dtype=np.float64)
            xy = self.track[:, 2:4]
            self.stations = np.r_[0., np.cumsum(np.linalg.norm(xy[1:] - xy[:-1], axis=1))]
            self.road_edges = np.roll(xy, -1, axis=0) - xy
            self.road_lengths = np.linalg.norm(self.road_edges, axis=1)
            self.obstacles = []
            for index, body in enumerate(raw.obstacles):
                position = np.asarray(tuple(body.position))
                anchor = int(np.argmin(np.sum((xy - position)**2, axis=1)))
                beta = float(self.track[anchor, 1])
                self.obstacles.append(dict(id=index, x=float(position[0]), y=float(position[1]),
                                           radius=float(body.fixtures[0].shape.radius), anchor_index=anchor,
                                           station=float(self.stations[anchor]), tangent=[-float(np.sin(beta)), float(np.cos(beta))]))
            self.centers = np.asarray([[o['x'], o['y']] for o in self.obstacles])
            self.normals = np.asarray([o['tangent'] for o in self.obstacles])
            self.radii = np.asarray([o['radius'] for o in self.obstacles])
            self.catalog = dict(track=raw.track, obstacles=self.obstacles)
            self.geometry_sha256 = hashlib.sha256(json.dumps(self.catalog, sort_keys=True).encode()).hexdigest()
            self.initial = self.state()
            self.initial_observation_sha256 = hashlib.sha256(result[0].tobytes()).hexdigest()
            self.active = True
            return result

        def state(self):
            raw = self.unwrapped
            hull = raw.car.hull
            bodies = [hull, *raw.car.wheels]
            polygons, skins = [], []
            for body in bodies:
                for fixture in body.fixtures:
                    polygons.append(np.asarray([tuple(body.GetWorldPoint(v)) for v in fixture.shape.vertices]))
                    skins.append(float(fixture.shape.radius))
            clearance, rear, front = footprint_measurements(polygons, skins, self.centers, self.normals, self.radii)
            touched = set()
            for contact in raw.world.contacts:
                if contact.touching:
                    a, b = contact.fixtureA.body, contact.fixtureB.body
                    for index, obstacle in enumerate(raw.obstacles):
                        if (a == obstacle and any(b == body for body in bodies)) or (b == obstacle and any(a == body for body in bodies)):
                            touched.add(index)
            position = np.asarray(tuple(hull.position))
            amounts = np.clip(np.sum((position - self.track[:, 2:4]) * self.road_edges, axis=1) / np.maximum(self.road_lengths**2, 1e-12), 0, 1)
            points = self.track[:, 2:4] + amounts[:, None] * self.road_edges
            road_index = int(np.argmin(np.sum((points - position)**2, axis=1)))
            tangent = self.road_edges[road_index] / self.road_lengths[road_index]
            beta = float(np.arctan2(-tangent[0], tangent[1]))
            return dict(t=float(raw.t), x=float(position[0]), y=float(position[1]), yaw=float(hull.angle),
                        speed=float(np.hypot(*hull.linearVelocity)), road_index=road_index,
                        station=float(self.stations[road_index] + amounts[road_index] * self.road_lengths[road_index]),
                        lateral=float(np.dot(position - points[road_index], [np.cos(beta), np.sin(beta)])),
                        heading_error=float((hull.angle - beta + np.pi) % (2 * np.pi) - np.pi),
                        rear=rear.tolist(), front=front.tolist(), clearance=clearance.tolist(),
                        contacts=sorted(touched), wheel_road_contacts=[len(w.tiles) for w in raw.car.wheels])

        def step(self, action):
            self.step_number += 1
            before, counter = self.state(), int(self.environment.off_track_counter)
            result = self.inner.step(action)
            after_counter = int(self.environment.off_track_counter)
            self.max_counter = max(self.max_counter, after_counter)
            self.last = dict(pre=before, post=self.state(), off_track_count_before=counter,
                             off_track_count_after=after_counter, decision_reward=float(result[1]),
                             **{key: bool(result[4].get(key, False)) for key in ('simulator_terminated', 'wrapper_off_track', 'wrapper_crashed')})
            return result

        def close(self):
            self.inner.close()

    return Observer()


def worker(output, slot_index, *, import_only=False):
    """Invoked only in a fresh -I process with no repository import path."""
    from importlib import import_module
    output = Path(output).resolve()
    protocol = read_json(output / 'protocol.json')
    slot = protocol['schedule'][slot_index]
    for group in ('source_sha256', 'environment_sha256', 'model_source_sha256'):
        for path, expected in protocol[group].items():
            if sha(path) != expected:
                raise ValueError(f'frozen source changed: {path}')
    model_dir = output / 'model'
    sys.path.insert(0, str(model_dir))
    model = import_module('agent').Agent()
    if not Path(str(sys.modules['agent'].__file__)).resolve().is_relative_to(model_dir):
        raise ValueError('agent imported outside immutable ZIP extraction')
    if slot['mode'] == ARMS[1]:
        spec = importlib.util.spec_from_file_location('collision_priority_frozen', output / 'source/collision_priority.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model = module.CollisionPriorityAgent(driver=model.driver)
    import haic_agent
    haic_agent.__path__.append(str(SNAPSHOT / 'haic_agent'))
    sys.path.insert(0, str(SNAPSHOT))
    sys.path.insert(0, str(output / 'source'))
    from scripts.evaluate_koi_minimum_clearance_ab import footprint_measurements
    import numpy as np
    import cv2
    import torch
    from training.env_factory import create_training_environment
    run_episode = import_module('training.evaluate_closed_loop').run_episode
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(slot['seed'])
    if not torch.__version__.startswith('2.1.') or '+cpu' not in torch.__version__:
        raise ValueError('CPU21 runtime required')
    if import_only:
        print(json.dumps(dict(import_only=True, environment_resets=0, model_type=type(model).__name__,
                              agent_file=sys.modules['agent'].__file__,
                              evaluator_file=sys.modules['training.evaluate_closed_loop'].__file__,
                              footprint_file=sys.modules['scripts.evaluate_koi_minimum_clearance_ab'].__file__)))
        return
    holder = {}
    episode_path = output / slot['file']
    raw_path = episode_path.with_suffix('.raw.jsonl')
    decision_path = episode_path.with_suffix('.decisions.jsonl')

    class MeasuredAgent:
        def reset(self, observation):
            return model.reset(observation)

        def act(self, observation):
            self.observation_sha256 = hashlib.sha256(observation.tobytes()).hexdigest()
            self.action = np.asarray(model.act(observation)).tolist()
            return self.action

        def last_step_diagnostics(self):
            diagnostics = {key: value for key, value in model.last_step_diagnostics().items()
                           if key.startswith(('priority_', 'steering_release_')) or key == 'baseline_steer'}
            diagnostics['evaluation_only'] = dict(holder['environment'].last,
                                                  observation_sha256=self.observation_sha256)
            decision_stream.write(json.dumps(dict(step=holder['environment'].step_number,
                                                  action=self.action, controller=diagnostics), allow_nan=False) + '\n')
            return diagnostics

    with raw_path.open('x', buffering=1) as raw_stream, decision_path.open('x', buffering=1) as decision_stream:
        def factory(**kwargs):
            observer = make_observer(create_training_environment(**kwargs), raw_stream,
                                     output / 'reset-ledger.jsonl', slot, sha(output / 'protocol.json'), footprint_measurements)
            holder['environment'] = observer
            return observer
        result = run_episode(mode=slot['mode'], track_id=slot['track_id'], seed=slot['seed'], agent=MeasuredAgent(),
                             max_decisions=1200, plan_budget_seconds=4.5, capture_trace=True,
                             fail_on_invalid_action=True, environment_factory=factory)
    observer = holder.get('environment')
    initial = getattr(observer, 'initial', None)
    states = ([initial] if initial else []) + [json.loads(line) for line in raw_path.read_text().splitlines()]
    trace = result['decision_trace']
    result.update(repeat=slot['repeat'], scope=SCOPE, catalog=getattr(observer, 'catalog', None),
                  geometry_sha256=getattr(observer, 'geometry_sha256', None), initial_state=initial,
                  initial_observation_sha256=getattr(observer, 'initial_observation_sha256', None),
                  raw_trace_file=raw_path.name, raw_trace_sha256=sha(raw_path),
                  decision_stream_file=decision_path.name, decision_stream_sha256=sha(decision_path),
                  off_track_count_max=getattr(observer, 'max_counter', None),
                  off_track_threshold=getattr(getattr(observer, 'environment', None), 'max_off_track_steps', None),
                  off_track_predicate='off_track_counter > max_off_track_steps',
                  physical=physical_metrics(states, getattr(observer, 'obstacles', [])),
                  priority_changed_decisions=sum(bool(row['controller'].get('priority_changed')) for row in trace),
                  priority_modes={mode: sum(row['controller'].get('priority_mode') == mode for row in trace)
                                  for mode in sorted({row['controller']['priority_mode'] for row in trace if 'priority_mode' in row['controller']})},
                  versions=dict(python=sys.version, numpy=np.__version__, cv2=getattr(cv2, '__version__'), torch=torch.__version__))
    save(episode_path, result)
    append_jsonl(output / 'reset-ledger.jsonl', dict(slot, event='episode_end',
                                                  retire_reason=result['retire_reason'], error=result['error']))


def summarize(output, schedule):
    episodes, pairs = {}, []
    for row in schedule:
        if row['status'] == 'completed':
            episodes[row['repeat'], row['track_id'], row['seed'], row['mode']] = read_json(output / row['file'])
    for repeat, track, seed in sorted({key[:3] for key in episodes}):
        if all((repeat, track, seed, arm) in episodes for arm in ARMS):
            pairs.append(pair_metrics(*(episodes[repeat, track, seed, arm] for arm in ARMS)))
    arms = {}
    for arm in ARMS:
        selected = [episode for key, episode in episodes.items() if key[-1] == arm]
        arms[arm] = dict(valid_episodes=len(selected), finished=sum(e['completed'] for e in selected),
                         damage=sum(e['damage'] for e in selected), collision_positive_decisions=sum(e['collisions'] for e in selected),
                         physical_contact_events=sum(e['physical']['physical_contact_events'] for e in selected),
                         hit_objects=sum(len(e['physical']['hit_objects']) for e in selected),
                         off_track_count_max=max((e['off_track_count_max'] for e in selected), default=None),
                         no_wheel_road_contact_max_ticks=max((e['physical']['no_wheel_road_contact_max_ticks'] for e in selected), default=None),
                         road_reacquisition_censored=sum(e['physical']['road_reacquisition_censored'] for e in selected),
                         priority_changed_decisions=sum(e['priority_changed_decisions'] for e in selected))
    matched = [pair for pair in pairs if pair['both_finished']]
    return dict(scope=SCOPE, arms=arms, pairs=pairs, valid_pairs=len(pairs), mutually_finished_pairs=len(matched),
                mutually_finished_mean_baseline_ms=sum(p['baseline_lap_ms'] for p in matched) / len(matched) if matched else None,
                mutually_finished_mean_candidate_ms=sum(p['candidate_lap_ms'] for p in matched) / len(matched) if matched else None,
                mutually_finished_mean_delta_ms=sum(p['matched_lap_delta_ms'] for p in matched) / len(matched) if matched else None)


def run(cells, output, repeat=1):
    if repeat < 1 or not cells or len(set(cells)) != len(cells):
        raise ValueError('positive repeat and distinct nonempty cells required')
    evidence = validate_cells(cells)
    if sha(V2_ZIP) != V2_SHA:
        raise ValueError('original frozen v2 ZIP mismatch')
    if not WRAPPER.is_file():
        raise ValueError('collision_priority.py is not ready')
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    schedule, error = [], None
    for repetition in range(repeat):
        for track, seed in cells:
            arms = ARMS if (track + seed + repetition) % 2 else tuple(reversed(ARMS))
            for arm in arms:
                schedule.append(dict(track_id=track, seed=seed, mode=arm, repeat=repetition,
                                     file=f'{track}-{seed}-r{repetition}-{arm}.json', status='unrun'))
    try:
        source_hashes = {}
        sources = {Path(__file__): 'scripts/evaluate_koi_collision_priority.py', WRAPPER: 'collision_priority.py'}
        for name in ('__init__.py', 'evaluate_koi_minimum_clearance_ab.py', 'evaluate_koi_adaptive_ab.py', 'package_koi_adaptive_avoidance.py'):
            sources[ROOT / 'scripts' / name] = 'scripts/' + name
        for source, name in sources.items():
            copied = output / 'source' / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            source_hashes[str(copied)] = sha(copied)
        shutil.copyfile(V2_ZIP, output / 'steering-release-v2.zip')
        if sha(output / 'steering-release-v2.zip') != V2_SHA:
            raise ValueError('copied original ZIP mismatch')
        with zipfile.ZipFile(output / 'steering-release-v2.zip') as archive:
            if any(Path(name).is_absolute() or '..' in Path(name).parts for name in archive.namelist()):
                raise ValueError('unsafe ZIP member')
            archive.extractall(output / 'model')
            model_hashes = {str(output / 'model' / name): sha(output / 'model' / name) for name in archive.namelist()}
        env_sources = [SNAPSHOT / 'env_wrapper.py', SNAPSHOT / 'damage.py']
        for name in ('training', 'core', 'haic_agent'):
            env_sources.extend(sorted((SNAPSHOT / name).rglob('*.py')))
        protocol = dict(scope=SCOPE, cells=cells, repeat=repeat, schedule=schedule,
                        comparator_zip_sha256=V2_SHA, wrapper_sha256=sha(output / 'source/collision_priority.py'),
                        source_sha256=source_hashes, environment_sha256={str(p): sha(p) for p in env_sources},
                        model_source_sha256=model_hashes, prior_completed_evidence=evidence,
                        prior_report_sha256=sha(PRIOR / 'episode-report.json'),
                        python=str(PYTHON), snapshot=str(SNAPSHOT), max_decisions=1200, child_timeout_s=180,
                        frame_skip=4, warmup_ticks=50, raw_fps=50, metric_definitions=METRIC_DEFINITIONS,
                        telemetry_observer_only=True, fresh=False, official_action=False)
        save(output / 'protocol.json', protocol)
        for index, slot in enumerate(schedule):
            slot['status'] = 'started'
            started = time.monotonic()
            command = [str(PYTHON), '-I', '-B', str(output / 'source/scripts/evaluate_koi_collision_priority.py'),
                       '--worker', str(index), '--output', str(output)]
            child_error = None
            stdout = stderr = ''
            try:
                child = subprocess.run(command, cwd=output / 'model', env=ENV, capture_output=True,
                                       text=True, timeout=180)
                stdout, stderr = child.stdout, child.stderr
                if child.returncode:
                    child_error = f'child_exit_{child.returncode}'
            except subprocess.TimeoutExpired as failure:
                stdout, stderr = failure.stdout or '', failure.stderr or ''
                child_error = 'child_timeout'
            except BaseException as failure:
                child_error = f'{type(failure).__name__}: {failure}'
                raise
            finally:
                path = output / slot['file']
                for suffix, text in (('.stdout.txt', stdout), ('.stderr.txt', stderr)):
                    path.with_suffix(suffix).write_text(text.decode(errors='replace') if isinstance(text, bytes) else text)
                save(path.with_suffix('.process.json'), dict(command=command, wall_time_s=time.monotonic() - started, error=child_error))
            if child_error:
                raise RuntimeError(child_error)
            episode = read_json(output / slot['file'])
            if (episode['error'] or episode['invalid_actions'] or episode['damage'] is None or episode['collisions'] is None
                    or not (episode['completed'] or episode['retire_reason'] in ('off_track', 'crash'))):
                raise RuntimeError(f"invalid or capped episode: {episode['retire_reason']}: {episode['error']}")
            slot.update(status='completed', sha256=sha(output / slot['file']))
            print(json.dumps(dict(slot, finished=episode['completed'], damage=episode['damage'], collisions=episode['collisions'])), flush=True)
    except BaseException as failure:
        error = f'{type(failure).__name__}: {failure}'
        for row in schedule:
            if row['status'] == 'started':
                row.update(status='operator_error', error=error)
        raise
    finally:
        save(output / 'episode-report.json', dict(scope=SCOPE, rows=schedule, operator_error=error))
        save(output / 'summary.json', summarize(output, schedule))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', help='Explicit completed consumed TRAIN cells, track:seed,...')
    parser.add_argument('--output', required=True, type=Path, help='New output directory; never overwrite/resume')
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--import-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.import_only and args.worker is None:
        parser.error('--import-only requires an existing frozen --worker slot')
    if args.worker is not None:
        worker(args.output, args.worker, import_only=args.import_only)
    elif not args.cells:
        parser.error('--cells is required')
    else:
        run(parse_cells(args.cells), args.output, args.repeat)


if __name__ == '__main__':
    main()
