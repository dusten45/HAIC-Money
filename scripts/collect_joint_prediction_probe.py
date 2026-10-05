"""Bounded consumed-TRAIN data intervention, never a controller or lap A/B.

Import, ``template`` and ``validate`` are stdlib-only and never import an Agent or
simulator. Only explicitly authorized ``collect`` creates fresh isolated workers.
The parent owns the finalized protocol, exposure audit and resource forecast.
"""

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
import zipfile


ROOT = Path('/workspace/HAIC-Money')
PYTHON = Path('/tmp/kilo/haic-cpu21/bin/python')
SNAPSHOT = Path('/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace')
BUNDLE = ROOT / 'submissions/20261002-crossing-projection-collision-shield-v1-baseline'
CHAMPION = BUNDLE / 'source'
PRIOR = ROOT / 'runs/koi-sprint72-relief-v1'
PRIOR_SHA = '7fca6b24509812aadde8f6dd2b2dd3a0a57bcb217a796ed9f35e5c6cb73d771d'
ZIP_SHA = 'c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801'
SEEDS = (3184000002, 3184000013, 3184000015)
ARMS = ('baseline', 'repeat', 'alternative')
ANCHOR, HORIZON, SKIP, WARMUP = 10, 4, 4, 51
MAX_RESETS, MAX_RAW = 9, 963
SCHEMA = 'haic-joint-prediction-probe-v1'
SCOPE = 'Prescribed data intervention; CONSUMED TRAIN; NOT controller/A-B lap performance.'
PARITY_LIMIT = 'Exact accessible physical-state and image parity, NOT full hidden Box2D solver proof.'
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    """Atomic durable progress: a killed process leaves its previous receipt intact."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(encoded(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_json(path, value):
    with Path(path).open('ab') as stream:
        stream.write(encoded(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def safe_path(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('a nonempty contained relative path is required')
    path = root / relative
    if path.resolve() != path or not path.resolve().is_relative_to(root):
        raise ValueError('symlink/escaped path refused')
    return path


def output_path(path):
    path = Path(path).absolute()
    if (path != path.resolve() or not path.is_relative_to(ROOT / 'runs')
            or path == ROOT / 'runs' or path.exists() or not path.parent.is_dir()):
        raise ValueError('new nonsymlink output beneath runs/ with an existing parent required')
    return path


def action_values(action):
    if len(action) != 3:
        raise ValueError('exactly three action components required')
    values = [float(v) for v in action]
    if any(not math.isfinite(v) or not lo <= v <= hi
           for v, lo, hi in zip(values, (-1, 0, 0), (1, 1, 1))):
        raise ValueError('finite in-range official action required; no implicit clipping')
    return values


def joint_alternative(nominal):
    steer, gas, brake = action_values(nominal)
    steer = max(0.0, steer - .05) if steer > 0 else min(0.0, steer + .05) if steer < 0 else .05
    return [steer, max(0.0, gas - .05), min(1.0, brake + .05)]


def tail_actions(nominal, arm):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    actions = [action_values(nominal) for _ in range(HORIZON)]
    if arm == 'alternative':
        actions[0] = joint_alternative(nominal)
    return actions


def assert_parity(expected, actual, label):
    """No tolerance, omission, approximate float equality, or NaN equivalence."""
    if encoded(expected) != encoded(actual):
        raise ValueError('strict accessible-state/image parity mismatch: ' + label)


def raw_parity_required(arm, raw_index):
    if arm not in ARMS or type(raw_index) is not int or not 1 <= raw_index <= 107:
        raise ValueError('valid arm and one-based raw index required')
    return arm == 'repeat' or (arm == 'alternative' and raw_index <= WARMUP + ANCHOR * SKIP)


def schedule() -> list[dict[str, Any]]:
    return [dict(seed=seed, track_id=1, arm=arm, slot=f'1-{seed}-{arm}')
            for seed in SEEDS for arm in ARMS]


def fixed_contract() -> dict[str, Any]:
    return dict(schema=SCHEMA, scope=SCOPE, parity_limitation=PARITY_LIMIT,
        fresh=False, official_action=False, prior_absolute_gate_preserved=True,
        interpretation='separate relative-effect falsification probe, no predictor fitting/scoring',
        schedule=schedule(), anchor_prefix_decisions=ANCHOR, tail_decisions=HORIZON,
        continuation='hold-current-nominal a0 for four holds, NOT future champion feedback',
        intervention='first hold only: steer toward zero by .05 (exact zero -> +.05), gas=max(0,gas-.05), brake=min(1,brake+.05); then three a0 holds',
        agent_calls='baseline only: ten natural prefix calls plus one anchor call; zero during tail; repeat/alternative replay with no Agent instance',
        profile=dict(factory='training.env_factory.create_training_environment', track_id=1,
            site_map=None, max_decisions=14, render_mode=None, continuous=True,
            warmup_wrapper_ticks=50, raw_reset_internal_ticks=1, skip_frames=4,
            stack_frames=4, raw_fps=50, time_limit_raw_ticks=306,
            obstacles=6, obstacle_radius=1.2, grass_friction_multiplier=.6,
            obstacle_rng='SeedSequence([track_id, seed, 0x4F425354])'),
        limits=dict(resets=MAX_RESETS, resets_per_arm=1, decisions_per_arm=14,
            warmup_raw_per_arm=WARMUP, driving_raw_per_arm=56, raw_per_arm=107, raw_total=MAX_RAW),
        python=str(PYTHON), snapshot=str(SNAPSHOT), champion_source=str(CHAMPION),
        champion_zip_sha256=ZIP_SHA,
        versions={'torch': '2.1.0+cpu', 'numpy': '1.26.0', 'cv2': '4.8.1'})


def pinned_inputs():
    """Source/receipt inspection only, including the exact historical Track1 slots."""
    if sha(PRIOR / 'protocol.json') != PRIOR_SHA:
        raise ValueError('historical frozen protocol differs')
    prior = read_json(PRIOR / 'protocol.json')
    if (prior['python'] != str(PYTHON) or prior['snapshot'] != str(SNAPSHOT)
            or prior['frame_skip'] != SKIP or prior['warmup_ticks'] != 50
            or prior['conditions']['environment_factory'] != 'training.env_factory.create_training_environment'):
        raise ValueError('historical environment profile differs')
    manifest = read_json(BUNDLE / 'manifest.json')
    if manifest['zip_sha256'] != ZIP_SHA or len(manifest['submission_members']) != 11:
        raise ValueError('exact champion manifest required')
    champion = {str(safe_path(CHAMPION, n)): manifest['files_sha256']['source/' + n]
                for n in manifest['submission_members']}
    if sha(BUNDLE / 'submission.zip') != ZIP_SHA:
        raise ValueError('exact champion ZIP required')
    with zipfile.ZipFile(BUNDLE / 'submission.zip') as archive:
        if sorted(archive.namelist()) != sorted(manifest['submission_members']):
            raise ValueError('champion member inventory differs')
        for name in archive.namelist():
            if hashlib.sha256(archive.read(name)).hexdigest() != champion[str(CHAMPION / name)]:
                raise ValueError('champion source/ZIP mismatch')
    references = {str(seed): dict(path=str(PRIOR / f'1-{seed}-r0-frozen_shield.json'),
                                 sha256=sha(PRIOR / f'1-{seed}-r0-frozen_shield.json')) for seed in SEEDS}
    report = read_json(PRIOR / 'episode-report.json')
    if report['protocol_sha256'] != PRIOR_SHA or report.get('operator_error') is not None:
        raise ValueError('historical completed report differs')
    for seed, pin in references.items():
        rows = [r for r in report['rows'] if r['track_id'] == 1 and r['seed'] == int(seed)
                and r['mode'] == 'frozen_shield' and r['repeat'] == 0]
        if len(rows) != 1 or rows[0]['status'] != 'completed' or rows[0]['sha256'] != pin['sha256']:
            raise ValueError('consumed slot does not match historical receipt')
    pins = dict(champion_source_sha256=champion, environment_sha256=prior['environment_sha256'],
        reference_protocol=dict(path=str(PRIOR / 'protocol.json'), sha256=PRIOR_SHA),
        reference_report=dict(path=str(PRIOR / 'episode-report.json'), sha256=sha(PRIOR / 'episode-report.json')),
        champion_manifest=dict(path=str(BUNDLE / 'manifest.json'), sha256=sha(BUNDLE / 'manifest.json')),
        references=references, runtime_executable_sha256=prior['runtime_executable_sha256'])
    for group in (champion, pins['environment_sha256']):
        for path, digest in group.items():
            if sha(path) != digest:
                raise ValueError('pinned input differs: ' + path)
    if sha(PYTHON.resolve()) != pins['runtime_executable_sha256']:
        raise ValueError('CPU21 executable differs')
    return pins


def protocol_template():
    return dict(contract=fixed_contract(), inputs=pinned_inputs(), status='PREPARATION_ONLY',
        collector=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        output_directory=None, exposure_receipt=None, reuse_claim=None, resource_receipt=None,
        resource_limits=dict(child_wall_s=None, total_wall_s=None, address_space_bytes=None,
            max_resident_bytes=None, max_output_bytes=None,
            min_memory_headroom_bytes=None, min_disk_free_bytes=None))


def validate_protocol(path, digest):
    if sha(path) != digest:
        raise ValueError('finalized protocol SHA256 required')
    protocol = read_json(path)
    assert_parity(fixed_contract(), protocol['contract'], 'fixed protocol contract')
    if protocol['status'] != 'FROZEN_FOR_AUTHORIZED_COLLECTION':
        raise ValueError('preparation is not collection authorization')
    bound_output = Path(protocol['output_directory'])
    if (not bound_output.is_absolute() or bound_output != bound_output.resolve()
            or not bound_output.is_relative_to(ROOT / 'runs') or bound_output == ROOT / 'runs'):
        raise ValueError('one canonical output directory beneath runs/ must be frozen')
    assert_parity(pinned_inputs(), protocol['inputs'], 'input pins')
    if (sha(protocol['collector']['path']) != protocol['collector']['sha256']
            or sha(__file__) != protocol['collector']['sha256']):
        raise ValueError('executed collector differs from frozen source')
    for field in ('exposure_receipt', 'reuse_claim', 'resource_receipt'):
        pin = protocol[field]
        if not isinstance(pin, dict) or sha(pin['path']) != pin['sha256']:
            raise ValueError('parent-owned pinned ' + field + ' required')
    limits = protocol['resource_limits']
    required = {'child_wall_s', 'total_wall_s', 'address_space_bytes', 'max_resident_bytes', 'max_output_bytes',
                'min_memory_headroom_bytes', 'min_disk_free_bytes'}
    if set(limits) != required or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                                      for v in limits.values()):
        raise ValueError('positive finite parent-declared resource limits required')
    return protocol


class Budget:
    """Serialized children share a write-ahead ledger; in-flight calls stay consumed."""

    def __init__(self, directory, slot, deadline):
        self.directory, self.slot, self.deadline = Path(directory), slot, deadline
        self.path = self.directory / 'counters.json'
        self.value = read_json(self.path)
        if slot not in self.value['arms']:
            self.value['arms'][slot] = {k: 0 for k in (
                'resets_started', 'resets_completed', 'raw_started', 'raw_completed',
                'warmup_started', 'warmup_completed', 'decisions_started', 'decisions_completed')}
            save_json(self.path, self.value)

    @property
    def arm(self):
        return self.value['arms'][self.slot]

    def event(self, kind, **extra):
        append_json(self.directory / 'events.jsonl', dict(event=kind, slot=self.slot,
            monotonic=time.monotonic(), time_ns=time.time_ns(), counters=self.value, **extra))

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('total/child wall watchdog')

    def begin(self, kind, *, warmup=False):
        self.check()
        arm, total = self.arm, self.value
        key = kind + '_started'
        if kind == 'resets':
            allowed = arm[key] < 1 and total[key] < MAX_RESETS
        elif kind == 'raw':
            driving = arm[key] - arm['warmup_started']
            allowed = (arm['resets_started'] == 1 and arm[key] < 107 and total[key] < MAX_RAW
                and (arm['warmup_started'] < WARMUP if warmup else
                     arm['warmup_completed'] == WARMUP and driving < 56
                     and driving < SKIP * arm['decisions_started']))
        elif kind == 'decisions':
            allowed = (arm['resets_completed'] == 1 and arm[key] < 14
                and arm['raw_started'] == arm['raw_completed'] == WARMUP + SKIP * arm[key])
        else:
            raise ValueError('unknown budget operation')
        if not allowed or arm[key] != arm[kind + '_completed']:
            self.event('budget_denied', operation=kind, warmup=warmup)
            raise RuntimeError('hard operation budget/in-flight guard: ' + kind)
        arm[key] += 1
        total[key] += 1
        if warmup:
            arm['warmup_started'] += 1
            total['warmup_started'] += 1
        save_json(self.path, total)
        self.event(kind + '_intent', warmup=warmup)

    def complete(self, kind, *, warmup=False):
        if self.arm[kind + '_started'] != self.arm[kind + '_completed'] + 1:
            raise RuntimeError('completion without exactly one outstanding operation')
        self.arm[kind + '_completed'] += 1
        self.value[kind + '_completed'] += 1
        if warmup:
            self.arm['warmup_completed'] += 1
            self.value['warmup_completed'] += 1
        save_json(self.path, self.value)


def empty_counters():
    result = {k: 0 for k in ('resets_started', 'resets_completed', 'raw_started', 'raw_completed',
        'warmup_started', 'warmup_completed', 'decisions_started', 'decisions_completed')}
    return dict(result, arms={})


def resource_admission(directory, protocol):
    """Host and every visible finite cgroup-v2 ancestor, not just the leaf."""
    limits = protocol['resource_limits']
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) * 1024
    membership = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                      if line.startswith('0::'))
    mounts = [line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines()
              if ' - cgroup2 ' in line]
    if len(mounts) != 1:
        raise RuntimeError('unambiguous cgroup-v2 ancestry required')
    mount_root, mount = Path(mounts[0][3]), Path(mounts[0][4])
    # Cgroup namespaces may expose their own root as '/', independently of host paths.
    member = Path(membership)
    leaf = mount / member.relative_to(mount_root) if member.is_relative_to(mount_root) else mount / member.relative_to('/')
    if not leaf.is_dir():
        raise RuntimeError('cannot resolve current cgroup ancestry')
    ancestors = []
    for node in (leaf, *leaf.parents):
        if not node.is_relative_to(mount):
            break
        maximum = (node / 'memory.max').read_text().strip()
        current = int((node / 'memory.current').read_text())
        headroom = None if maximum == 'max' else int(maximum) - current
        if headroom is not None:
            available = min(available, headroom)
        ancestors.append(dict(path=str(node), maximum=maximum, current=current, headroom=headroom,
            events=(node / 'memory.events').read_text(), pressure=(node / 'memory.pressure').read_text()))
    used = sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())
    free = shutil.disk_usage(directory).free
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    resident = int(status['VmRSS'].split()[0]) * 1024
    report = dict(memory_headroom_bytes=available, resident_bytes=resident,
        disk_free_bytes=free, output_bytes=used, ancestors=ancestors)
    append_json(Path(directory) / 'resources.jsonl', dict(report, time_ns=time.time_ns()))
    if (available < limits['min_memory_headroom_bytes'] or free < limits['min_disk_free_bytes']
            or used >= limits['max_output_bytes'] or resident > limits['max_resident_bytes']):
        raise RuntimeError('parent-declared resource admission failed')
    return report


def json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    convert = getattr(value, 'tolist', None)
    if callable(convert):
        return json_value(convert())
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError('unsupported accessible state type: ' + type(value).__name__)


def body_state(body, *, fixtures=False) -> dict[str, Any]:
    result: dict[str, Any] = {key: float(getattr(body, key)) for key in
              ('angle', 'angularVelocity', 'mass', 'inertia', 'linearDamping', 'angularDamping', 'gravityScale')}
    result.update({key: list(map(float, getattr(body, key))) for key in
                   ('position', 'linearVelocity', 'localCenter', 'worldCenter')})
    result.update({key: bool(getattr(body, key)) for key in
                   ('awake', 'active', 'bullet', 'fixedRotation', 'sleepingAllowed')})
    result['type'] = int(body.type)
    if fixtures:
        result['fixtures'] = []
        for f in body.fixtures:
            row: dict[str, Any] = dict(radius=float(f.shape.radius), sensor=bool(f.sensor), density=float(f.density),
                friction=float(f.friction), restitution=float(f.restitution),
                filter={k: int(getattr(f.filterData, k)) for k in ('categoryBits', 'maskBits', 'groupIndex')})
            if hasattr(f.shape, 'vertices'):
                row['vertices'] = [list(map(float, v)) for v in f.shape.vertices]
            else:
                row['position'] = list(map(float, f.shape.pos))
            result['fixtures'].append(row)
    return result


def physical_state(environment, np):
    """Read-only accessible state. No setters, render call, solver call or model input."""
    wrapper, raw = environment.environment, environment.unwrapped
    car = raw.car
    roads, obstacles = raw.road, raw.obstacles
    if len(car.wheels) != 4 or any(tile.idx != i for i, tile in enumerate(roads)):
        raise ValueError('wheel or stable tile identity differs')

    def particle(p):
        return None if p is None else json_value(vars(p))

    wheels = []
    for wheel in car.wheels:
        ids = sorted(int(tile.idx) for tile in wheel.tiles)
        if any(i < 0 or i >= len(roads) or roads[i] not in wheel.tiles for i in ids):
            raise ValueError('wheel touches unrecognized tile')
        friction = max([float(car.grass_friction_multiplier)] + [float(roads[i].road_friction) for i in ids])
        row = body_state(wheel, fixtures=True)
        row.update({key: float(getattr(wheel, key)) for key in ('omega', 'gas', 'brake', 'steer', 'phase', 'wheel_rad')})
        row.update(tiles=ids, tile_frictions=[float(roads[i].road_friction) for i in ids],
            terrain_friction_multiplier=friction, effective_friction_limit=400 * friction * float(car.grip_multiplier),
            skid_start=None if wheel.skid_start is None else list(map(float, wheel.skid_start)),
            skid_particle=particle(wheel.skid_particle),
            joint={key: float(getattr(wheel.joint, key)) for key in
                   ('angle', 'speed', 'motorSpeed', 'lowerLimit', 'upperLimit')})
        row['joint'].update(motorEnabled=bool(wheel.joint.motorEnabled), limitEnabled=bool(wheel.joint.limitEnabled))
        wheels.append(row)

    def fixture_id(f):
        body = f.body
        for name, bodies in (('hull', [car.hull]), ('wheel', car.wheels), ('road', roads), ('obstacle', obstacles)):
            for i, candidate in enumerate(bodies):
                if body == candidate:
                    return [name, i, next(j for j, fixture in enumerate(body.fixtures) if fixture == f)]
        raise ValueError('unrecognized contact body')

    contacts = []
    for contact in raw.world.contacts:
        manifold = contact.manifold
        # Sensor/non-touching manifolds have zero points. Their unused C++ vector
        # fields are not initialized physical state and must not be interpreted.
        count = int(manifold.pointCount)
        contacts.append(dict(a=fixture_id(contact.fixtureA), b=fixture_id(contact.fixtureB),
            touching=bool(contact.touching), enabled=bool(contact.enabled),
            friction=float(contact.friction), restitution=float(contact.restitution),
            tangent_speed=float(contact.tangentSpeed), point_count=count,
            type=int(manifold.type_) if count else None,
            local_normal=list(map(float, manifold.localNormal)) if count else None,
            local_point=list(map(float, manifold.localPoint)) if count else None,
            points=[dict(local_point=list(map(float, p.localPoint)), normal_impulse=float(p.normalImpulse),
                         tangent_impulse=float(p.tangentImpulse), id=int(p.id.key))
                    for p in manifold.points[:count]]))
    contacts.sort(key=lambda row: encoded(row))
    result = dict(track_id=int(raw.track_id), seed=int(raw.track_seed), t=float(raw.t),
        hull=body_state(car.hull, fixtures=True), wheels=wheels, contacts=contacts,
        road_visited=[bool(tile.road_visited) for tile in roads],
        road_friction=[float(tile.road_friction) for tile in roads],
        obstacles=[dict(body=body_state(b, fixtures=True), hit=bool(b.userData.hit)) for b in obstacles],
        car={key: float(getattr(car, key)) for key in ('fuel_spent', 'grass_friction_multiplier',
             'grip_multiplier', 'engine_multiplier', 'steering_multiplier')},
        particles=[particle(p) for p in car.particles], damage=float(wrapper.damage.damage),
        damage_effects=json_value(wrapper.damage.effects), off_track_counter=int(wrapper.off_track_counter),
        damage_telemetry_valid=bool(wrapper._damage_telemetry_valid),
        reward=float(raw.reward), prev_reward=float(raw.prev_reward), tile_visited_count=int(raw.tile_visited_count),
        collision=bool(raw._collision_this_step), new_lap=bool(raw.new_lap),
        finish_tracker=json_value(raw.finish_line_tracker),
        rng=json_value(raw.np_random.bit_generator.state),
        time_limit=dict(elapsed=wrapper.env._elapsed_steps, maximum=int(wrapper.env._max_episode_steps)),
        world=dict(gravity=list(map(float, raw.world.gravity)), autoClearForces=bool(raw.world.autoClearForces),
                   warmStarting=bool(raw.world.warmStarting), continuousPhysics=bool(raw.world.continuousPhysics),
                   subStepping=bool(raw.world.subStepping)),
        raw_image_sha256=None if not hasattr(raw, 'state') else
            hashlib.sha256(np.ascontiguousarray(raw.state).tobytes()).hexdigest())
    encoded(result)
    return result


def verify_geometry(environment, reference, np):
    raw = environment.unwrapped
    expected = reference['catalog']
    assert_parity(expected['track'], json_value(raw.track), 'historical geometry')
    actual = [dict(id=i, x=float(b.position[0]), y=float(b.position[1]), radius=float(b.fixtures[0].shape.radius))
              for i, b in enumerate(raw.obstacles)]
    old = [{k: row[k] for k in ('id', 'x', 'y', 'radius')} for row in expected['obstacles']]
    assert_parity(old, actual, 'historical obstacles')
    if (raw.track_id != 1 or raw.track_seed != reference['seed'] or len(actual) != 6
            or raw.car.grass_friction_multiplier != .6):
        raise ValueError('historical Track1 profile mismatch')
    return dict(track=json_value(raw.track), obstacles=actual,
        road_bodies=[body_state(b, fixtures=True) for b in raw.road],
        geometry_sha256=reference['geometry_sha256'])


def load_runtime(protocol, baseline, seed):
    """Called only by an authorized fresh worker; never falls back to root Agent."""
    forbidden = ('agent', 'haic_agent', 'training', 'core', 'env_wrapper', 'damage')
    if any(name.split('.')[0] in forbidden for name in sys.modules):
        raise ValueError('fresh isolated process required')
    if (Path(sys.executable).absolute() != PYTHON or sys.version_info[:2] != (3, 11)
            or any(os.environ.get(k) != '1' for k in THREAD_ENV)):
        raise ValueError('pinned single-thread CPU21 required')
    sys.path[:] = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]
    sys.path.insert(0, str(CHAMPION))
    package = importlib.import_module('haic_agent')
    if Path(str(package.__file__)).resolve() != CHAMPION / 'haic_agent/__init__.py':
        raise ValueError('champion package origin differs')
    entry = importlib.import_module('agent') if baseline else None
    if entry is not None and Path(str(entry.__file__)).resolve() != CHAMPION / 'agent.py':
        raise ValueError('Agent origin differs; no fallback permitted')
    package.__path__.append(str(SNAPSHOT / 'haic_agent'))
    sys.path.insert(0, str(SNAPSHOT))
    np, cv2, torch = [importlib.import_module(n) for n in ('numpy', 'cv2', 'torch')]
    factory_module = importlib.import_module('training.env_factory')
    for module in (np, cv2, torch):
        if module.__version__ != protocol['contract']['versions'][module.__name__]:
            raise ValueError('pinned dependency version differs: ' + module.__name__)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    verify_imports(protocol)
    return np, factory_module.create_training_environment, entry.Agent() if entry else None


def verify_imports(protocol):
    pins = dict(protocol['inputs']['environment_sha256'])
    pins.update(protocol['inputs']['champion_source_sha256'])
    origins = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(CHAMPION) or path.is_relative_to(SNAPSHOT):
                if pins.get(str(path)) != sha(path):
                    raise ValueError('import outside pinned source closure: ' + str(path))
                origins[name] = str(path)
            elif name.split('.')[0] in ('agent', 'haic_agent', 'training', 'core', 'env_wrapper', 'damage'):
                raise ValueError('project module imported outside frozen source: ' + name)
    return origins


def worker(directory, index, protocol_path, digest, authorization):
    directory = Path(directory)
    run = read_json(directory / 'run.json')
    if (authorization != digest or run['protocol_sha256'] != digest
            or run['status'] != 'RUNNING' or run['parent_pid'] != os.getppid()):
        raise ValueError('authorized live parent required')
    protocol = validate_protocol(protocol_path, digest)
    if str(directory.resolve()) != protocol['output_directory']:
        raise ValueError('worker output differs from the frozen one-use directory')
    slot = schedule()[index]
    arm_dir = safe_path(directory, slot['slot'])
    arm_dir.mkdir(exist_ok=False)
    deadline = min(run['deadline_monotonic'], time.monotonic() + protocol['resource_limits']['child_wall_s'])
    budget = Budget(directory, slot['slot'], deadline)
    limits = protocol['resource_limits']
    resource.setrlimit(resource.RLIMIT_AS, (int(limits['address_space_bytes']), int(limits['address_space_bytes'])))
    resource.setrlimit(resource.RLIMIT_FSIZE, (int(limits['max_output_bytes']), int(limits['max_output_bytes'])))

    def timeout(signum, frame):
        raise TimeoutError('child wall watchdog')

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    data: dict[str, Any] = dict(schema=SCHEMA, scope=SCOPE, parity_limitation=PARITY_LIMIT, slot=slot,
                protocol_sha256=digest, status='STARTED', agent_act_calls=0, nominal=None,
                endpoints=[], decisions=[], geometry=None, failure=None)
    images, times, actions, raw_frames, raw_times, raw_actions = [], [], [], [], [], []
    np: Any = None
    environment: Any = None

    def persist():
        data['counters'] = dict(budget.arm)
        if np is not None:
            temp = arm_dir / 'data.npz.tmp'
            with temp.open('wb') as stream:
                np.savez_compressed(stream,
                    observations=np.asarray(images, dtype=np.float32).reshape((-1, 4, 84, 84)),
                    endpoint_time=np.asarray(times, dtype=np.float64),
                    actions=np.asarray(actions, dtype=np.float64).reshape((-1, 3)),
                    raw_frames=np.asarray(raw_frames, dtype=np.uint8).reshape((-1, 96, 96, 3)),
                    raw_time=np.asarray(raw_times, dtype=np.float64),
                    raw_actions=np.asarray(raw_actions, dtype=np.float64).reshape((-1, 3)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, arm_dir / 'data.npz')
        save_json(arm_dir / 'state.json', data)
        artifacts = {p.name: sha(p) for p in arm_dir.iterdir() if p.is_file() and p.name not in ('receipt.json', 'receipt.json.tmp')}
        save_json(arm_dir / 'receipt.json', dict(status=data['status'], protocol_sha256=digest,
            slot=slot, counters=budget.arm, artifacts_sha256=artifacts))

    persist()
    try:
        resource_admission(directory, protocol)
        np, factory, model = load_runtime(protocol, slot['arm'] == 'baseline', slot['seed'])
        reference = read_json(protocol['inputs']['references'][str(slot['seed'])]['path'])
        baseline: dict[str, Any] = {}
        baseline_images: Any = None
        baseline_raw = []
        if slot['arm'] != 'baseline':
            base_dir = safe_path(directory, f"1-{slot['seed']}-baseline")
            receipt = read_json(base_dir / 'receipt.json')
            if receipt['status'] != 'COMPLETE' or receipt['protocol_sha256'] != digest:
                raise ValueError('complete contemporary baseline required')
            for name, expected in receipt['artifacts_sha256'].items():
                if sha(safe_path(base_dir, name)) != expected:
                    raise ValueError('baseline artifact changed')
            baseline = read_json(base_dir / 'state.json')
            baseline_raw = [json.loads(line) for line in (base_dir / 'raw.jsonl').read_text().splitlines()]
            if len(baseline_raw) != 214:
                raise ValueError('complete baseline raw PRE/POST stream required')
            with np.load(base_dir / 'data.npz', allow_pickle=False) as arrays:
                baseline_images = arrays['observations'].copy()
            data['baseline_receipt_sha256'] = sha(base_dir / 'receipt.json')
        # Construction is deliberately below all authorization, pin and admission gates.
        environment = factory(track_id=1, seed=slot['seed'], max_decisions=14, render_mode=None)
        raw = environment.unwrapped
        original_reset, original_step = raw.reset, raw.step
        warmup, decision = True, -1

        def observed_reset(*args, **kwargs):
            validate_protocol(protocol_path, digest)
            resource_admission(directory, protocol)
            budget.begin('resets')
            result = original_reset(*args, **kwargs)
            budget.complete('resets')
            return result

        def observed_step(action):
            issued = None if action is None else action_values(action)
            index = budget.arm['raw_started'] + 1
            # The first internal reset tick has null raw-image/TimeLimit fields.
            before = physical_state(environment, np)
            pre = dict(event='PRE', raw_index=index, warmup=warmup, decision=decision, action=issued, state=before)
            append_json(arm_dir / 'raw.jsonl', pre)
            if raw_parity_required(slot['arm'], index):
                assert_parity(baseline_raw[2 * (index - 1)], pre, f'raw PRE {index}')
            budget.begin('raw', warmup=warmup)
            result = original_step(action)  # Delegate exactly once, unchanged arguments/result.
            budget.complete('raw', warmup=warmup)
            raw_frames.append(np.array(result[0], copy=True))
            raw_times.append(float(raw.t))
            raw_actions.append([0., 0., 0.] if issued is None else issued)
            post = dict(event='POST', raw_index=budget.arm['raw_completed'],
                warmup=warmup, decision=decision, action=issued, internal_reset_no_action=issued is None,
                t=float(raw.t), reward=float(result[1]), terminated=bool(result[2]), truncated=bool(result[3]),
                info=json_value(result[4]), state=physical_state(environment, np))
            append_json(arm_dir / 'raw.jsonl', post)
            if raw_parity_required(slot['arm'], index):
                assert_parity(baseline_raw[2 * (index - 1) + 1], post, f'raw POST {index}')
            return result

        raw.reset, raw.step = observed_reset, observed_step
        observation, info = environment.reset()
        warmup = False
        if budget.arm['resets_completed'] != 1 or budget.arm['warmup_completed'] != WARMUP:
            raise ValueError('actual reset/warmup counts differ; no retry')
        data['geometry'] = verify_geometry(environment, reference, np)
        if baseline:
            assert_parity(baseline['geometry'], data['geometry'], 'static geometry/fixtures')

        def endpoint(observation, info, reward=None, terminated=False, truncated=False) -> dict[str, Any]:
            arr = np.asarray(observation)
            if (arr.dtype != np.float32 or arr.shape != (4, 84, 84)
                    or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1
                    or not np.array_equal(arr, environment.environment.stack_state)):
                raise ValueError('exact normalized observation stack required')
            row: dict[str, Any] = dict(state=physical_state(environment, np), info=json_value(info), reward=reward,
                terminated=bool(terminated), truncated=bool(truncated),
                image_sha256=hashlib.sha256(arr.tobytes()).hexdigest())
            images.append(arr.copy())
            times.append(float(raw.t))
            data['endpoints'].append(row)
            persist()  # Preserve even the mismatch before raising.
            i = len(images) - 1
            if baseline and (slot['arm'] == 'repeat' or i <= ANCHOR):
                assert_parity(baseline['endpoints'][i], row, f'endpoint {i}')
                if (baseline_images[i].dtype != arr.dtype or baseline_images[i].tobytes() != arr.tobytes()):
                    raise ValueError('strict observation byte mismatch')
            return row

        initial = endpoint(observation, info)
        if initial['image_sha256'] != reference['initial_observation_sha256']:
            raise ValueError('initial image differs from consumed historical profile')
        if model is not None:
            model.reset(observation.copy())
        for decision in range(ANCHOR + HORIZON):
            budget.check()
            if decision < ANCHOR:
                if model is not None:
                    action = action_values(model.act(observation.copy()))
                    data['agent_act_calls'] += 1
                else:
                    action = baseline['decisions'][decision]['action']
            else:
                if decision == ANCHOR:
                    if model is not None:
                        data['nominal'] = action_values(model.act(observation.copy()))
                        data['agent_act_calls'] += 1
                    else:
                        data['nominal'] = baseline['nominal']
                    data['tail_actions'] = tail_actions(data['nominal'], slot['arm'])
                    persist()
                action = data['tail_actions'][decision - ANCHOR]
            action = action_values(action)
            budget.begin('decisions')
            pre = data['endpoints'][-1]
            if baseline and (slot['arm'] == 'repeat' or decision < ANCHOR):
                assert_parity(baseline['decisions'][decision]['action'], action, 'replayed action')
            actions.append(action)
            data['decisions'].append(dict(index=decision, action=action, pre_endpoint=decision,
                post_endpoint=None, raw_before=budget.arm['raw_completed'], status='INTENT'))
            persist()
            observation, reward, terminated, truncated, info = environment.step(np.asarray(action, dtype=np.float64))
            budget.complete('decisions')
            row = data['decisions'][-1]
            row.update(post_endpoint=decision + 1, raw_after=budget.arm['raw_completed'], status='COMPLETE')
            post = endpoint(observation, info, float(reward), terminated, truncated)
            if (row['raw_after'] - row['raw_before'] != SKIP
                    or not math.isclose(post['state']['t'] - pre['state']['t'], .08, rel_tol=0, abs_tol=1e-12)):
                raise ValueError('partial/changed action hold; retain censor and stop')
            if (terminated or truncated) and decision != ANCHOR + HORIZON - 1:
                raise ValueError('early terminal retained; no replacement/reset')
            resource_admission(directory, protocol)
        if data['agent_act_calls'] != (11 if slot['arm'] == 'baseline' else 0):
            raise ValueError('unexpected future Agent call')
        if budget.arm['raw_completed'] != 107 or budget.arm['decisions_completed'] != 14:
            raise ValueError('complete-arm actual counters differ')
        data['import_origins'] = verify_imports(protocol)
        validate_protocol(protocol_path, digest)
        data['status'] = 'COMPLETE'
    except BaseException as error:
        data['status'] = 'PARTIAL_FAILED'
        data['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        persist()
        if environment is not None:
            environment.close()


def collect(protocol_path, digest, output, authorization):
    if authorization != digest:
        raise ValueError('explicit --authorize-collection must equal the finalized protocol hash')
    protocol = validate_protocol(protocol_path, digest)
    output = output_path(output)
    if str(output) != protocol['output_directory']:
        raise ValueError('one protocol permits only its frozen output; no retry under a new path')
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    run: dict[str, Any] = dict(schema=SCHEMA, status='RUNNING', scope=SCOPE, protocol_sha256=digest,
        parent_pid=os.getpid(), started_monotonic=start,
        deadline_monotonic=start + protocol['resource_limits']['total_wall_s'], rows=[], failure=None)
    save_json(output / 'run.json', run)
    save_json(output / 'counters.json', empty_counters())
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
        SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy', CUDA_VISIBLE_DEVICES='')
    environment.update({k: '1' for k in THREAD_ENV})
    try:
        for index, slot in enumerate(schedule()):
            validate_protocol(protocol_path, digest)
            resource_admission(output, protocol)
            remaining = run['deadline_monotonic'] - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('total wall budget exhausted before child')
            command = [str(PYTHON), '-I', '-B', protocol['collector']['path'], '_worker',
                       '--protocol', str(Path(protocol_path).resolve()), '--sha256', digest,
                       '--output', str(output), '--index', str(index), '--authorize-collection', authorization]
            row: dict[str, Any] = dict(slot=slot, command=command, status='STARTED', returncode=None)
            run['rows'].append(row)
            save_json(output / 'run.json', run)
            with (output / (slot['slot'] + '.log')).open('xb') as log:
                try:
                    result = subprocess.run(command, env=environment, cwd=output, stdout=log,
                        stderr=subprocess.STDOUT, timeout=min(remaining, protocol['resource_limits']['child_wall_s'] + 5), check=False)
                    row['returncode'] = result.returncode
                    row['status'] = 'EXITED'
                except subprocess.TimeoutExpired:
                    row['status'] = 'KILLED_TIMEOUT'
                    raise
                finally:
                    row['counters'] = read_json(output / 'counters.json')
                    row['log_sha256'] = sha(output / (slot['slot'] + '.log'))
                    receipt = output / slot['slot'] / 'receipt.json'
                    row['receipt_sha256'] = sha(receipt) if receipt.is_file() else None
                    arm_dir = output / slot['slot']
                    row['final_artifacts_sha256'] = ({p.name: sha(p) for p in arm_dir.iterdir() if p.is_file()}
                                                     if arm_dir.is_dir() else {})
                    save_json(output / 'run.json', run)
            if result.returncode or read_json(receipt)['status'] != 'COMPLETE':
                raise RuntimeError('partial child failure: no later arm, retry or replacement')
        counters = read_json(output / 'counters.json')
        if (counters['resets_started'] != MAX_RESETS
                or counters['resets_completed'] != MAX_RESETS
                or counters['raw_started'] != MAX_RAW or counters['raw_completed'] != MAX_RAW):
            raise RuntimeError('final actual counters differ')
        validate_protocol(protocol_path, digest)
        run['status'] = 'COMPLETE'
    except BaseException as error:
        run['status'], run['failure'] = 'PARTIAL_FAILED', f'{type(error).__name__}: {error}'
        raise
    finally:
        run['counters'] = read_json(output / 'counters.json')
        run['wall_s'] = time.monotonic() - start
        run['counter_meaning'] = 'started is durable upper bound; completed is returned calls. In-flight difference is ambiguous consumed cost, never fresh.'
        save_json(output / 'run.json', run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('template', 'validate', 'collect', '_worker'))
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--authorize-collection')
    parser.add_argument('--index', type=int)
    args = parser.parse_args(argv)
    if args.command == 'template':
        print(json.dumps(protocol_template(), indent=2, allow_nan=False))
        return
    if args.protocol is None or args.sha256 is None:
        parser.error('--protocol and --sha256 required')
    if args.command == 'validate':
        validate_protocol(args.protocol, args.sha256)
        print(json.dumps(dict(valid=True, simulator_imports=0, agent_instances=0, resets=0, steps=0)))
    elif args.command == 'collect':
        collect(args.protocol, args.sha256, args.output, args.authorize_collection)
    else:
        if args.index is None or not 0 <= args.index < len(schedule()):
            parser.error('valid --index required')
        worker(args.output, args.index, args.protocol, args.sha256, args.authorize_collection)


if __name__ == '__main__':
    main()
