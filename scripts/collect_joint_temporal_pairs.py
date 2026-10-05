"""Consumed TRAIN temporal pairs, with no policy calls during the H4 suffix.

Import/template/validate are stdlib-only. Collection requires a parent-frozen
protocol and an explicit matching authorization hash. This is not a controller,
an official evaluation, or an exactly resumable/hidden-solver clone facility.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import struct
import subprocess
import sys
import time
from types import ModuleType
from typing import Any
import zlib


ROOT = Path('/workspace/HAIC-Money')
HELPER = ROOT / 'scripts/collect_joint_prediction_probe.py'
HELPER_SHA = '047c263aeae617785c550868d6bf721e0e661a0cd1922450d0ec017ffc3a4acb'
_source = HELPER.read_bytes()
if hashlib.sha256(_source).hexdigest() != HELPER_SHA:
    raise ValueError('immutable legacy helper source differs')
legacy = ModuleType('_joint_temporal_pinned_helper')
legacy.__file__ = str(HELPER)
exec(compile(_source, str(HELPER), 'exec'), legacy.__dict__)
del _source

PYTHON, SNAPSHOT, CHAMPION = legacy.PYTHON, legacy.SNAPSHOT, legacy.CHAMPION
sha, encoded, read_json = legacy.sha, legacy.encoded, legacy.read_json
save_json, safe_path = legacy.save_json, legacy.safe_path
action_values, assert_parity = legacy.action_values, legacy.assert_parity
json_value, body_state = legacy.json_value, legacy.body_state
ARMS = ('baseline', 'repeat', 'alternative')
REGIMES = ('high_speed_straight', 'left_entry', 'right_entry', 'braking_turn')
HORIZON, SKIP, WARMUP, MAX_ANCHOR = 4, 4, 51, 180
SCHEMA = 'haic-joint-temporal-pairs-v1'
SCOPE = 'CONSUMED TRAIN data intervention only; NOT controller, lap A/B or official performance.'
PARITY_LIMIT = legacy.PARITY_LIMIT
RESOURCE_FIELDS = {
    'child_wall_s', 'total_wall_s', 'address_space_bytes', 'max_resident_bytes',
    'max_output_bytes', 'min_memory_headroom_bytes', 'min_disk_free_bytes',
    'forecast_output_bytes', 'serialization_reserve_bytes',
    'first_baseline_max_s_per_raw', 'throughput_safety_factor',
}


def canonical_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def catalog_sha(catalog):
    """Historical catalogs deliberately used json.dumps' default separators."""
    return hashlib.sha256(json.dumps(catalog, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_anchors(anchors):
    if not isinstance(anchors, list) or len(anchors) != 8:
        raise ValueError('exactly eight predeclared anchors required')
    keys = {'anchor_id', 'regime', 'track_id', 'seed', 'anchor_prefix_decisions',
            'steering_delta', 'pedal_direction', 'reference_id'}
    for anchor in anchors:
        if not isinstance(anchor, dict) or set(anchor) != keys:
            raise ValueError('anchor fields differ from the declared schema')
        for key in ('anchor_id', 'reference_id'):
            if not isinstance(anchor[key], str) or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,79}', anchor[key]):
                raise ValueError('contained stable anchor/reference identifier required')
        if (type(anchor['track_id']) is not int or anchor['track_id'] not in (1, 2, 3)
                or type(anchor['seed']) is not int or not 0 <= anchor['seed'] <= 0xFFFFFFFF
                or type(anchor['anchor_prefix_decisions']) is not int
                or not 1 <= anchor['anchor_prefix_decisions'] <= MAX_ANCHOR
                or type(anchor['steering_delta']) not in (int, float)
                or anchor['steering_delta'] not in (-.04, .04)
                or anchor['pedal_direction'] not in ('up', 'down')):
            raise ValueError('invalid fixed TRAIN anchor/action direction')
    if Counter(a['regime'] for a in anchors) != Counter({r: 2 for r in REGIMES}):
        raise ValueError('two anchors in each of the four declared regimes required')
    if len({a['anchor_id'] for a in anchors}) != 8:
        raise ValueError('duplicate anchor identifier')
    if len({(a['reference_id'], a['anchor_prefix_decisions']) for a in anchors}) != 8:
        raise ValueError('duplicate reference/decision anchor')
    for regime in REGIMES:
        pair = [a for a in anchors if a['regime'] == regime]
        if {a['steering_delta'] for a in pair} != {-.04, .04} or {a['pedal_direction'] for a in pair} != {'up', 'down'}:
            raise ValueError('opposed steering and pedal directions required in each regime')


def schedule(anchors) -> list[dict[str, Any]]:
    validate_anchors(anchors)
    return [dict(anchor, arm=arm, slot=anchor['anchor_id'] + '-' + arm,
                 decisions=anchor['anchor_prefix_decisions'] + HORIZON,
                 raw_ticks=WARMUP + SKIP * (anchor['anchor_prefix_decisions'] + HORIZON))
            for anchor in anchors for arm in ARMS]


def budgets(slots) -> dict[str, Any]:
    return dict(resets=len(slots), decisions=sum(s['decisions'] for s in slots),
                warmup_raw=len(slots) * WARMUP, raw_total=sum(s['raw_ticks'] for s in slots),
                per_arm={s['slot']: dict(resets=1, decisions=s['decisions'],
                                        warmup_raw=WARMUP, raw_ticks=s['raw_ticks']) for s in slots})


def tail_actions(nominal, arm, steering_delta, pedal_direction):
    if arm not in ARMS or steering_delta not in (-.04, .04) or pedal_direction not in ('up', 'down'):
        raise ValueError('fixed arm and signed intervention required')
    actions = [action_values(nominal) for _ in range(HORIZON)]
    if arm == 'alternative':
        steer, gas, brake = actions[0]
        delta = .05 if pedal_direction == 'up' else -.05
        actions[0] = [max(-1., min(1., steer + steering_delta)),
                      max(0., min(1., gas + delta)), max(0., min(1., brake - delta))]
    return actions


def raw_parity_required(slot, index):
    if slot['arm'] not in ARMS or type(index) is not int or not 1 <= index <= slot['raw_ticks']:
        raise ValueError('valid arm and one-based raw index required')
    return slot['arm'] == 'repeat' or (slot['arm'] == 'alternative'
            and index <= WARMUP + SKIP * slot['anchor_prefix_decisions'])


def fixed_contract(anchors) -> dict[str, Any]:
    slots = schedule(anchors) if anchors else []
    return dict(schema=SCHEMA, scope=SCOPE, parity_limitation=PARITY_LIMIT,
        fresh=False, official_action=False, prior_results_unchanged=True,
        anchors=copy.deepcopy(anchors), schedule=slots, limits=budgets(slots),
        horizon_decisions=HORIZON, frame_skip=SKIP, warmup_raw=WARMUP,
        maximum_anchor_prefix_decisions=max((a['anchor_prefix_decisions'] for a in anchors), default=0),
        implementation_anchor_ceiling=MAX_ANCHOR,
        continuation='current nominal a0 held four times; alternative changes first hold only, then three a0; no future Agent calls',
        agent_calls='baseline: prefix plus one anchor call; repeat/alternative: no Agent instance',
        ledger='51-tick warmup and 4-tick decision writeahead reservations; midblock upper bound is consumed/ambiguous, never fresh or exactly resumable',
        capture='decision84 stacks and full accessible boundary state; raw dynamic PRE/POST and RGB digests; one static fixture catalog, no raw RGB arrays',
        python=str(PYTHON), snapshot=str(SNAPSHOT), champion_source=str(CHAMPION),
        champion_zip_sha256=legacy.ZIP_SHA,
        versions={'torch': '2.1.0+cpu', 'numpy': '1.26.0', 'cv2': '4.8.1'})


def historical_profile(track_id):
    if type(track_id) is not int or track_id not in (1, 2, 3):
        raise ValueError('only historical TRAIN Track1/2/3 profiles allowed')
    return dict(factory='training.env_factory.create_training_environment', track_id=track_id,
        site_map=None, render_mode=None, continuous=True, obstacle_mode='official',
        obstacle_count=6, obstacle_radius_source=1.2, grass_friction_multiplier=.6,
        warmup_wrapper_ticks=50, raw_reset_internal_ticks=1, skip_frames=4, stack_frames=4,
        raw_fps=50, time_limit_raw_ticks='50 + 4*(anchor_prefix_decisions+4) + 200',
        obstacle_rng='SeedSequence([track_id, seed, 0x4F425354])')


def file_pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def check_pin(pin):
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('exact path/sha256 file pin required')
    path = Path(pin['path'])
    if not path.is_absolute() or path != path.resolve() or sha(path) != pin['sha256']:
        raise ValueError('pinned file differs: ' + str(path))
    return path


def bind_reference(episode_path, protocol_path, report_path):
    """File-only reference binding; never imports historical evaluator modules."""
    episode = read_json(episode_path)
    result = dict(episode=file_pin(episode_path), protocol=file_pin(protocol_path), report=file_pin(report_path),
        track_id=episode['track_id'], seed=episode['seed'], mode=episode['mode'], repeat=episode['repeat'],
        catalog_sha256=catalog_sha(episode['catalog']), profile=historical_profile(episode['track_id']))
    validate_reference(result)
    return result


def validate_reference(reference):
    paths = {key: check_pin(reference[key]) for key in ('episode', 'protocol', 'report')}
    episode, prior, report = [read_json(paths[key]) for key in ('episode', 'protocol', 'report')]
    if (reference['mode'] != 'frozen_shield' or reference['repeat'] != 0
            or prior.get('baseline_zip_sha256') != legacy.ZIP_SHA
            or report.get('protocol_sha256') != reference['protocol']['sha256']
            or report.get('operator_error') is not None):
        raise ValueError('completed exact-champion historical reference required')
    if (prior.get('python') != str(PYTHON) or prior.get('snapshot') != str(SNAPSHOT)
            or prior.get('frame_skip') != 4 or prior.get('warmup_ticks') != 50
            or prior.get('raw_fps') != 50):
        raise ValueError('historical reference runtime/cadence differs')
    # The canonical nominal-trajectory protocol predates the `conditions` field.
    # Bind its actual environment closure, not a fabricated metadata default.
    frozen = read_json(legacy.PRIOR / 'protocol.json')
    assert_parity(frozen['environment_sha256'], prior['environment_sha256'], 'historical environment closure')
    assert_parity(frozen['runtime_executable_sha256'], prior['runtime_executable_sha256'], 'historical interpreter')
    for field in ('track_id', 'seed', 'mode', 'repeat'):
        assert_parity(reference[field], episode[field], 'historical ' + field)
    assert_parity(historical_profile(reference['track_id']), reference['profile'], 'historical profile')
    rows = [row for row in report['rows'] if all(row.get(k) == reference[k]
            for k in ('track_id', 'seed', 'mode', 'repeat'))]
    if (len(rows) != 1 or rows[0]['status'] != 'completed'
            or rows[0]['sha256'] != reference['episode']['sha256']):
        raise ValueError('historical consumption is not receipt-backed')
    if (catalog_sha(episode['catalog']) != reference['catalog_sha256']
            or episode['geometry_sha256'] != reference['catalog_sha256']
            or episode.get('map_id') is not None or episode.get('obstacle_mode') != 'official'
            or episode.get('site_obstacle_count') != 0 or episode.get('obstacle_count') != 6
            or len(episode['catalog']['obstacles']) != 6):
        raise ValueError('exact historical catalog/profile differs')
    return episode


def protocol_template(anchors=None, references=None) -> dict[str, Any]:
    return dict(status='PREPARATION_ONLY', contract=fixed_contract(anchors or []),
        inputs=legacy.pinned_inputs(), references=references or {},
        collector=file_pin(__file__), legacy_helper=dict(path=str(HELPER), sha256=HELPER_SHA),
        output_directory=None, exposure_receipt=None, reuse_claim=None, resource_receipt=None,
        analysis_sources={'observer': None, 'comparison': None, 'analyzer': None},
        resource_limits={field: None for field in sorted(RESOURCE_FIELDS)})


def validate_parent_receipts(protocol):
    claim = read_json(protocol['reuse_claim']['path'])
    exposure = read_json(protocol['exposure_receipt']['path'])
    if (claim.get('format') != 'haic-consumed-train-reuse-claim-v1'
            or claim.get('study_id') != 'joint-temporal-v1' or claim.get('status') != 'reserved-consumed-reuse'
            or claim.get('partition') != 'TRAIN' or claim.get('fresh') is not False
            or claim.get('exposure_sha256') != protocol['exposure_receipt']['sha256']):
        raise ValueError('own consumed reuse claim with bound exposure required')
    claimed_anchors = [dict({key: value for key, value in row.items() if key != 'collection_regime'},
                            regime=row['collection_regime']) for row in claim['anchors']]
    assert_parity(protocol['contract']['anchors'], claimed_anchors, 'own claim anchors/directions')
    assert_parity(list(ARMS), claim['arms'], 'own claim arms')
    limits = protocol['contract']['limits']
    slots = protocol['contract']['schedule']
    for field, expected in [('maximum_resets', limits['resets']),
                            ('maximum_total_decisions', limits['decisions']),
                            ('maximum_raw_ticks_including_reset', limits['raw_total']), ('tail_decisions', HORIZON),
                            ('maximum_prefix_decisions', protocol['contract']['maximum_anchor_prefix_decisions']),
                            ('maximum_decisions_per_arm', max(s['decisions'] for s in slots)),
                            ('maximum_raw_ticks_per_arm', max(s['raw_ticks'] for s in slots)),
                            ('maximum_champion_act_calls', sum(a['anchor_prefix_decisions'] + 1 for a in claimed_anchors))]:
        assert_parity(expected, claim[field], 'own claim ' + field)
    expected_cells = {(a['track_id'], a['seed']) for a in protocol['contract']['anchors']}
    actual_cells = {(c['track_id'], c['geometry_seed']) for c in claim['cells']}
    if (expected_cells != actual_cells or len(actual_cells) != len(claim['cells'])
            or any(c['partition'] != 'TRAIN' or c['regime_id'] != 0 or c['obstacles'] is not True for c in claim['cells'])):
        raise ValueError('claim cells/profile differ')
    if (exposure.get('status') != 'COMPATIBLE_CONSUMED_REUSE_NO_ACTIVE_CONFLICT_FOUND_IN_SCOPED_METADATA'
            or exposure.get('fresh') is not False or exposure.get('blockers') != []
            or exposure.get('unmatched_candidate_intents') != []):
        raise ValueError('scoped exposure is not clear for consumed reuse')
    profiles = {row['reference_id']: row for row in exposure['profiles']}
    if set(profiles) != set(protocol['references']):
        raise ValueError('scoped exposure profiles differ')
    for key, reference in protocol['references'].items():
        profile = profiles[key]
        for field in ('track_id', 'seed', 'catalog_sha256'):
            assert_parity(reference[field], profile[field], 'exposure reference ' + field)
        assert_parity(reference['episode']['sha256'], profile['episode_sha256'], 'exposure episode digest')
    audit_pin = dict(claim['anchor_audit'])
    audit_pin['path'] = str((ROOT / audit_pin['path']).resolve())
    check_pin(audit_pin)


def validate_protocol(path, digest):
    if sha(path) != digest:
        raise ValueError('final protocol hash required')
    protocol = read_json(path)
    if protocol['status'] != 'FROZEN_FOR_AUTHORIZED_COLLECTION':
        raise ValueError('preparation is not collection authorization')
    anchors = protocol['contract']['anchors']
    validate_anchors(anchors)
    assert_parity(fixed_contract(anchors), protocol['contract'], 'fixed collection contract')
    assert_parity(legacy.pinned_inputs(), protocol['inputs'], 'frozen CPU21/champion/environment pins')
    if len(protocol['inputs']['environment_sha256']) != 142:
        raise ValueError('exact 142-file environment closure required')
    check_pin(protocol['collector'])
    if sha(__file__) != protocol['collector']['sha256']:
        raise ValueError('executing collector differs')
    assert_parity(dict(path=str(HELPER), sha256=HELPER_SHA), protocol['legacy_helper'], 'legacy source pin')
    check_pin(protocol['legacy_helper'])
    bound = Path(protocol['output_directory'])
    if not bound.is_absolute() or bound != bound.resolve() or not bound.is_relative_to(ROOT / 'runs') or bound == ROOT / 'runs':
        raise ValueError('one canonical output directory beneath runs required')
    for field in ('exposure_receipt', 'reuse_claim', 'resource_receipt'):
        check_pin(protocol[field])
    validate_parent_receipts(protocol)
    if not {'observer', 'comparison', 'analyzer'} <= set(protocol['analysis_sources']):
        raise ValueError('predata observer/comparison/analyzer pins required')
    for pin in protocol['analysis_sources'].values():
        check_pin(pin)
    if set(protocol['references']) != {a['reference_id'] for a in anchors}:
        raise ValueError('exact scoped consumed reference inventory required')
    for key, reference in protocol['references'].items():
        episode = validate_reference(reference)
        for anchor in (a for a in anchors if a['reference_id'] == key):
            if (anchor['track_id'] != reference['track_id'] or anchor['seed'] != reference['seed']
                    or anchor['anchor_prefix_decisions'] >= len(episode['decision_trace'])):
                raise ValueError('anchor differs from the consumed historical cell or is unreached')
    limits = protocol['resource_limits']
    if set(limits) != RESOURCE_FIELDS or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in limits.values()):
        raise ValueError('positive finite parent resource limits required')
    if (limits['forecast_output_bytes'] > limits['max_output_bytes']
            or limits['max_resident_bytes'] >= limits['address_space_bytes']
            or limits['throughput_safety_factor'] < 1):
        raise ValueError('inconsistent resource forecast/caps')
    return protocol


COUNT_KEYS = ('resets_started', 'resets_actual_started', 'resets_completed',
              'raw_started', 'raw_actual_started', 'raw_completed',
              'warmup_started', 'warmup_actual_started', 'warmup_completed',
              'decisions_started', 'decisions_actual_started', 'decisions_completed')


def empty_counters(slots):
    return dict(schema=SCHEMA, limits=budgets(slots), totals={k: 0 for k in COUNT_KEYS}, arms={},
        meaning='started: durable reservation upper bound; actual_started/completed: last persisted actual counts; open block may have additional calls; never resume/retry')


class BlockBudget:
    """Reserve BEFORE reset/hold; actual calls are counted in memory until sealing."""

    def __init__(self, directory, slot, deadline):
        self.path, self.slot, self.deadline = Path(directory) / 'counters.json', slot, deadline
        self.value = read_json(self.path)
        if slot['slot'] not in self.value['limits']['per_arm'] or slot['slot'] in self.value['arms']:
            raise RuntimeError('unknown or previously attempted arm; no retry/resume')
        self.arm: dict[str, Any] = dict({k: 0 for k in COUNT_KEYS}, block=None, status='ADMITTED', actual_counts_exact=False)
        self.value['arms'][slot['slot']] = self.arm
        self.persist()

    def persist(self):
        self.value['totals'] = {key: sum(arm[key] for arm in self.value['arms'].values()) for key in COUNT_KEYS}
        save_json(self.path, self.value)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('parent/arm wall limit')

    def reserve(self, warmup=False):
        self.check()
        arm = self.arm
        if arm['block'] is not None or arm['status'] == 'PARTIAL_FAILED':
            raise RuntimeError('open/failed block cannot be resumed')
        count = WARMUP if warmup else SKIP
        if warmup:
            allowed = arm['resets_started'] == 0 and arm['raw_started'] == 0
        else:
            allowed = (arm['resets_completed'] == 1 and arm['warmup_completed'] == WARMUP
                and arm['decisions_started'] == arm['decisions_completed'] < self.slot['decisions']
                and arm['raw_started'] == arm['raw_completed'] == WARMUP + SKIP * arm['decisions_completed'])
        if not allowed or arm['raw_started'] + count > self.slot['raw_ticks']:
            raise RuntimeError('hard per-arm reset/raw/decision budget')
        if self.value['totals']['raw_started'] + count > self.value['limits']['raw_total']:
            raise RuntimeError('hard total raw budget')
        arm['raw_started'] += count
        if warmup:
            arm['resets_started'] += 1
            arm['warmup_started'] += count
        else:
            arm['decisions_started'] += 1
        arm['block'] = dict(warmup=warmup, reserved=count, before=arm['raw_completed'],
                            actual_started=0, completed=0)
        arm['status'], arm['actual_counts_exact'] = 'BLOCK_RESERVED', False
        self.persist()

    def begin(self, kind):
        self.check()
        arm, block = self.arm, self.arm['block']
        if block is None or arm['status'] == 'PARTIAL_FAILED':
            raise RuntimeError('operation without durable block reservation')
        if kind == 'raw':
            if (block['actual_started'] != block['completed'] or block['actual_started'] >= block['reserved']
                    or (block['warmup'] and arm['resets_actual_started'] != 1)
                    or (not block['warmup'] and arm['decisions_actual_started'] != arm['decisions_completed'] + 1)):
                raise RuntimeError('raw block ceiling/in-flight guard')
            block['actual_started'] += 1
            if block['warmup']:
                arm['warmup_actual_started'] += 1
        elif kind == 'resets':
            if not block['warmup'] or arm['resets_actual_started'] != 0:
                raise RuntimeError('hidden warmup reset/retry refused before delegation')
        elif kind == 'decisions':
            if block['warmup'] or arm['decisions_actual_started'] != arm['decisions_completed']:
                raise RuntimeError('decision in-flight guard')
        else:
            raise ValueError('unknown operation')
        arm[kind + '_actual_started'] += 1

    def complete(self, kind):
        arm, block = self.arm, self.arm['block']
        if block is None or arm[kind + '_actual_started'] != arm[kind + '_completed'] + 1:
            raise RuntimeError('completion without one outstanding actual call')
        arm[kind + '_completed'] += 1
        if kind == 'raw':
            block['completed'] += 1
            if block['warmup']:
                arm['warmup_completed'] += 1

    def seal(self):
        arm, block = self.arm, self.arm['block']
        if (block is None or arm['status'] == 'PARTIAL_FAILED' or block['completed'] != block['reserved']
                or block['actual_started'] != block['reserved']
                or any(arm[k + '_started'] != arm[k + '_actual_started'] or arm[k + '_actual_started'] != arm[k + '_completed']
                       for k in ('resets', 'raw', 'warmup', 'decisions'))):
            raise RuntimeError('partial block retained; no seal/retry')
        arm['block'], arm['status'], arm['actual_counts_exact'] = None, 'SEALED', True
        self.persist()

    def fail(self):
        # Keep the reservation even when a caught exception reveals smaller cost.
        self.arm['status'], self.arm['actual_counts_exact'] = 'PARTIAL_FAILED', True
        self.persist()


class Journal:
    """Buffered raw/decision/image evidence, flushed only at block boundaries."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.streams = {name: (self.directory / name).open('xb') for name in
                        ('raw.jsonl', 'boundaries.jsonl', 'decisions.jsonl', 'observations.zlib-stream')}
        self.pending = {name: [] for name in self.streams}

    def record(self, name, value):
        self.pending[name].append(encoded(value) + b'\n')

    def observation(self, blob):
        payload = zlib.compress(blob, level=1)
        self.pending['observations.zlib-stream'].append(struct.pack('>Q', len(payload)) + payload)

    def flush(self):
        for name, rows in self.pending.items():
            if not rows:
                continue
            stream = self.streams[name]
            stream.write(b''.join(rows))
            stream.flush()
            os.fsync(stream.fileno())
            rows.clear()

    def close(self):
        try:
            self.flush()
        finally:
            for stream in self.streams.values():
                stream.close()


BODY_DYNAMIC = ('angle', 'angularVelocity', 'position', 'linearVelocity', 'worldCenter', 'awake', 'active')


class PhysicalCapture:
    """Pure getters with cached SWIG fixture-pointer -> canonical fixture IDs.

    Addresses are process-local lookup keys ONLY, never artifact state or parity
    identifiers. Static definitions are emitted once, dynamic values every raw
    PRE/POST, and the expanded accessible state at every decision boundary.
    """

    def __init__(self, environment, np):
        self.environment, self.np = environment, np
        raw = environment.unwrapped
        self.roads = list(raw.road)
        self.groups = {'hull': [raw.car.hull], 'wheel': list(raw.car.wheels),
                       'road': self.roads, 'obstacle': list(raw.obstacles)}
        if len(self.groups['wheel']) != 4 or any(tile.idx != i for i, tile in enumerate(self.roads)):
            raise ValueError('stable four-wheel/road identities required')
        self.fixture_ids, self.definitions = {}, {}
        for name, bodies in self.groups.items():
            self.definitions[name] = []
            for index, body in enumerate(bodies):
                self.definitions[name].append(body_state(body, fixtures=True))
                for fixture_index, fixture in enumerate(body.fixtures):
                    key = int(fixture.this)
                    if key in self.fixture_ids:
                        raise ValueError('duplicate fixture pointer')
                    self.fixture_ids[key] = [name, index, fixture_index]
        self.static = dict(schema=SCHEMA, bodies=self.definitions,
            road_friction=[float(tile.road_friction) for tile in self.roads],
            track=json_value(raw.track), identity='group/body-index/fixture-index; no pointer in artifacts')
        self.static_sha256 = canonical_sha(self.static)

    def fixture_id(self, fixture):
        try:
            return self.fixture_ids[int(fixture.this)]
        except KeyError as error:
            raise ValueError('unrecognized contact fixture; cached catalog is not extended silently') from error

    def dynamic_body(self, body) -> dict[str, Any]:
        return dict(angle=float(body.angle), angularVelocity=float(body.angularVelocity),
            position=list(map(float, body.position)), linearVelocity=list(map(float, body.linearVelocity)),
            worldCenter=list(map(float, body.worldCenter)), awake=bool(body.awake), active=bool(body.active))

    def dynamic(self):
        raw, wrapper = self.environment.unwrapped, self.environment.environment
        car = raw.car
        particle = lambda p: None if p is None else json_value(vars(p))
        wheels = []
        for wheel in car.wheels:
            ids = sorted(int(tile.idx) for tile in wheel.tiles)
            if any(i < 0 or i >= len(self.roads) or self.roads[i] not in wheel.tiles for i in ids):
                raise ValueError('unknown wheel tile')
            friction = max([float(car.grass_friction_multiplier)] + [float(self.roads[i].road_friction) for i in ids])
            row = self.dynamic_body(wheel)
            row.update({key: float(getattr(wheel, key)) for key in ('omega', 'gas', 'brake', 'steer', 'phase', 'wheel_rad')})
            row.update(tiles=ids, tile_frictions=[float(self.roads[i].road_friction) for i in ids],
                terrain_friction_multiplier=friction, effective_friction_limit=400 * friction * float(car.grip_multiplier),
                skid_start=None if wheel.skid_start is None else list(map(float, wheel.skid_start)),
                skid_particle=particle(wheel.skid_particle),
                joint={key: float(getattr(wheel.joint, key)) for key in ('angle', 'speed', 'motorSpeed', 'lowerLimit', 'upperLimit')})
            row['joint'].update(motorEnabled=bool(wheel.joint.motorEnabled), limitEnabled=bool(wheel.joint.limitEnabled))
            wheels.append(row)
        contacts = []
        for contact in raw.world.contacts:
            manifold, count = contact.manifold, int(contact.manifold.pointCount)
            contacts.append(dict(a=self.fixture_id(contact.fixtureA), b=self.fixture_id(contact.fixtureB),
                touching=bool(contact.touching), enabled=bool(contact.enabled), friction=float(contact.friction),
                restitution=float(contact.restitution), tangent_speed=float(contact.tangentSpeed), point_count=count,
                type=int(manifold.type_) if count else None,
                local_normal=list(map(float, manifold.localNormal)) if count else None,
                local_point=list(map(float, manifold.localPoint)) if count else None,
                points=[dict(local_point=list(map(float, p.localPoint)), normal_impulse=float(p.normalImpulse),
                             tangent_impulse=float(p.tangentImpulse), id=int(p.id.key)) for p in manifold.points[:count]]))
        contacts.sort(key=encoded)
        image = getattr(raw, 'state', None)
        return dict(static_sha256=self.static_sha256, track_id=int(raw.track_id), seed=int(raw.track_seed), t=float(raw.t),
            hull=self.dynamic_body(car.hull), wheels=wheels, contacts=contacts,
            road_visited_indices=[i for i, tile in enumerate(self.roads) if tile.road_visited],
            obstacles=[dict(body=self.dynamic_body(body), hit=bool(body.userData.hit)) for body in raw.obstacles],
            car={key: float(getattr(car, key)) for key in ('fuel_spent', 'grass_friction_multiplier',
                 'grip_multiplier', 'engine_multiplier', 'steering_multiplier')},
            particles=[particle(p) for p in car.particles], damage=float(wrapper.damage.damage),
            damage_effects=json_value(wrapper.damage.effects), off_track_counter=int(wrapper.off_track_counter),
            damage_telemetry_valid=bool(wrapper._damage_telemetry_valid), reward=float(raw.reward), prev_reward=float(raw.prev_reward),
            tile_visited_count=int(raw.tile_visited_count), collision=bool(raw._collision_this_step), new_lap=bool(raw.new_lap),
            finish_tracker=json_value(raw.finish_line_tracker), rng=json_value(raw.np_random.bit_generator.state),
            time_limit=dict(elapsed=wrapper.env._elapsed_steps, maximum=int(wrapper.env._max_episode_steps)),
            world=dict(gravity=list(map(float, raw.world.gravity)), autoClearForces=bool(raw.world.autoClearForces),
                       warmStarting=bool(raw.world.warmStarting), continuousPhysics=bool(raw.world.continuousPhysics),
                       subStepping=bool(raw.world.subStepping)),
            raw_image_sha256=None if image is None else hashlib.sha256(self.np.ascontiguousarray(image).tobytes()).hexdigest())

    def full(self):
        state = self.dynamic()
        state['hull'] = dict(self.definitions['hull'][0], **state['hull'])
        state['wheels'] = [dict(definition, **dynamic) for definition, dynamic in zip(self.definitions['wheel'], state['wheels'])]
        state['obstacles'] = [dict(row, body=dict(definition, **row['body']))
                              for definition, row in zip(self.definitions['obstacle'], state['obstacles'])]
        visited = set(state.pop('road_visited_indices'))
        state['road_visited'] = [i in visited for i in range(len(self.roads))]
        state['road_friction'] = [float(tile.road_friction) for tile in self.roads]
        return state


def verify_geometry(environment, reference, np):
    raw, expected = environment.unwrapped, reference['catalog']
    assert_parity(expected['track'], json_value(raw.track), 'historical geometry')
    xy = np.asarray(raw.track, dtype=np.float64)[:, 2:4]
    stations = np.r_[0., np.cumsum(np.linalg.norm(xy[1:] - xy[:-1], axis=1))]
    obstacles = []
    for index, body in enumerate(raw.obstacles):
        position = np.asarray(tuple(body.position))
        anchor = int(np.argmin(np.sum((xy - position) ** 2, axis=1)))
        beta = float(raw.track[anchor][1])
        obstacles.append(dict(id=index, x=float(position[0]), y=float(position[1]),
            radius=float(body.fixtures[0].shape.radius), anchor_index=anchor,
            station=float(stations[anchor]), tangent=[-float(np.sin(beta)), float(np.cos(beta))]))
    catalog = dict(track=json_value(raw.track), obstacles=obstacles)
    assert_parity(expected, catalog, 'exact historical obstacle/road catalog')
    if (catalog_sha(catalog) != reference['geometry_sha256'] or raw.track_id != reference['track_id']
            or raw.track_seed != reference['seed'] or len(obstacles) != 6
            or raw.car.grass_friction_multiplier != .6):
        raise ValueError('consumed profile/catalog mismatch')
    return dict(catalog, geometry_sha256=reference['geometry_sha256'])


def resource_admission(directory, protocol):
    limits = protocol['resource_limits']
    used = sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())
    remaining = max(0, limits['forecast_output_bytes'] - used)
    adapted = dict(protocol, resource_limits=dict(limits,
        min_disk_free_bytes=limits['min_disk_free_bytes'] + remaining + limits['serialization_reserve_bytes']))
    report = legacy.resource_admission(directory, adapted)
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    high_water = int(status['VmHWM'].split()[0]) * 1024
    if high_water > limits['max_resident_bytes']:
        raise RuntimeError('process VmHWM exceeds parent RSS cap')
    return dict(report, vm_hwm_bytes=high_water, remaining_forecast_bytes=remaining)


def load_baseline(directory, slot, digest, np):
    base_dir = safe_path(directory, slot['anchor_id'] + '-baseline')
    receipt = read_json(base_dir / 'receipt.json')
    if receipt['status'] != 'COMPLETE' or receipt['protocol_sha256'] != digest:
        raise ValueError('complete contemporary baseline required')
    for name, expected in receipt['artifacts_sha256'].items():
        if sha(safe_path(base_dir, name)) != expected:
            raise ValueError('baseline artifact differs')
    state = read_json(base_dir / 'state.json')
    raw = [json.loads(line) for line in (base_dir / 'raw.jsonl').read_text().splitlines()]
    if len(raw) != 2 * slot['raw_ticks']:
        raise ValueError('complete baseline raw PRE/POST stream required')
    with np.load(base_dir / 'data.npz', allow_pickle=False) as arrays:
        images = arrays['observations'].copy()
    return state, images, raw, read_json(base_dir / 'static.json'), sha(base_dir / 'receipt.json')


def champion_action(model, observation, data, decision):
    """Time only .act; observation copying and action validation stay outside."""
    slot = data['slot']
    if (slot['arm'] != 'baseline' or model is None or type(decision) is not int
            or not 0 <= decision <= slot['anchor_prefix_decisions']
            or decision != data['agent_act_calls']
            or data['agent_act_calls'] != data['agent_act_completed']):
        raise ValueError('only sequential baseline prefix/anchor Agent calls allowed')
    copied, act = observation.copy(), model.act
    data['agent_act_calls'] += 1
    started = time.perf_counter_ns()
    try:
        action = act(copied)
    finally:
        data['agent_act_latency_ms'].append((time.perf_counter_ns() - started) / 1_000_000.)
    action = action_values(action)
    data['agent_act_completed'] += 1
    return action


def validate_act_timings(data):
    """Complete arms have one finite timing per call, never replay/future calls."""
    slot = data['slot']
    expected = slot['anchor_prefix_decisions'] + 1 if slot['arm'] == 'baseline' else 0
    latencies = data['agent_act_latency_ms']
    if (data['agent_act_calls'] != expected or data['agent_act_completed'] != expected
            or len(latencies) != expected
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in latencies)):
        raise ValueError('Agent call counts or per-call latencies differ')


def worker(directory, index, protocol_path, digest, authorization):
    directory = Path(directory)
    run = read_json(directory / 'run.json')
    if (authorization != digest or run['protocol_sha256'] != digest or run['status'] != 'RUNNING'
            or run['parent_pid'] != os.getppid() or not sys.flags.isolated or not sys.flags.dont_write_bytecode):
        raise ValueError('authorized live isolated parent required')
    protocol = validate_protocol(protocol_path, digest)
    if str(directory.resolve()) != protocol['output_directory']:
        raise ValueError('worker output differs')
    slots = protocol['contract']['schedule']
    if type(index) is not int or not 0 <= index < len(slots):
        raise ValueError('invalid worker slot')
    slot, limits = slots[index], protocol['resource_limits']
    arm_dir = safe_path(directory, slot['slot'])
    arm_dir.mkdir(exist_ok=False)
    start = time.monotonic()
    deadline = min(run['deadline_monotonic'], start + limits['child_wall_s'])
    budget = BlockBudget(directory, slot, deadline)
    journal = Journal(arm_dir)
    resource.setrlimit(resource.RLIMIT_AS, (int(limits['address_space_bytes']), int(limits['address_space_bytes'])))
    resource.setrlimit(resource.RLIMIT_FSIZE, (int(limits['max_output_bytes']), int(limits['max_output_bytes'])))

    def timeout(signum, frame):
        raise TimeoutError('child wall watchdog')

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    data: dict[str, Any] = dict(schema=SCHEMA, scope=SCOPE, parity_limitation=PARITY_LIMIT, slot=slot,
        protocol_sha256=digest, status='STARTED', agent_act_calls=0, agent_act_completed=0,
        agent_act_latency_ms=[],
        agent_act_timing_scope='perf_counter_ns around champion.act only; excludes observation copy, action validation, capture, raw steps and flush; baseline list index is decision0..anchor inclusive, replay lists empty',
        agent_reset_started=0, agent_reset_completed=0, nominal=None,
        endpoints=[], decisions=[], geometry=None, failure=None, resource_samples=[],
        phase_wall_s=dict(runtime_load=0., construction=0., reset=0., prefix=0., suffix=0.,
                          raw_delegate=0., raw_capture=0., block_flush=0., serialization=0.),
        phase_timing_meaning='wall timings; raw_delegate/raw_capture/block_flush are nested in reset/prefix/suffix, not additive')
    images, times, actions, raw_times, raw_actions = [], [], [], [], []
    raw_returns = []
    np: Any = None
    environment: Any = None
    capture: Any = None
    baseline, baseline_raw, baseline_static = {}, [], None
    baseline_images: Any = None
    warmup, decision = True, -1

    def persist():
        serial_start = time.monotonic()
        journal.flush()
        data['counters'] = copy.deepcopy(budget.arm)
        data['wall_s'] = time.monotonic() - start
        if data['status'] != 'COMPLETE':
            data['returned_raw_evidence'] = raw_returns
        if np is not None:
            temporary = arm_dir / 'data.npz.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,
                    observations=np.asarray(images, dtype=np.float32).reshape((-1, 4, 84, 84)),
                    endpoint_time=np.asarray(times, dtype=np.float64),
                    actions=np.asarray(actions, dtype=np.float64).reshape((-1, 3)),
                    raw_time=np.asarray(raw_times, dtype=np.float64),
                    raw_actions=np.asarray(raw_actions, dtype=np.float64).reshape((-1, 3)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, arm_dir / 'data.npz')
        data['phase_wall_s']['serialization'] += time.monotonic() - serial_start
        save_json(arm_dir / 'state.json', data)
        artifacts = {p.name: sha(p) for p in arm_dir.iterdir() if p.is_file() and p.name not in ('receipt.json', 'receipt.json.tmp')}
        save_json(arm_dir / 'receipt.json', dict(status=data['status'], protocol_sha256=digest,
            slot=slot, counters=budget.arm, artifacts_sha256=artifacts))

    try:
        data['resource_samples'].append(resource_admission(directory, protocol))
        phase_start = time.monotonic()
        np, factory, model = legacy.load_runtime(protocol, slot['arm'] == 'baseline', slot['seed'])
        data['phase_wall_s']['runtime_load'] = time.monotonic() - phase_start
        reference = read_json(protocol['references'][slot['reference_id']]['episode']['path'])
        if slot['arm'] != 'baseline':
            baseline, baseline_images, baseline_raw, baseline_static, digest_base = load_baseline(directory, slot, digest, np)
            data['baseline_receipt_sha256'] = digest_base
        # All imports, references, pins, admission and authorization precede construction.
        phase_start = time.monotonic()
        environment = factory(track_id=slot['track_id'], seed=slot['seed'], max_decisions=slot['decisions'], render_mode=None)
        data['phase_wall_s']['construction'] = time.monotonic() - phase_start
        raw = environment.unwrapped
        original_reset, original_step = raw.reset, raw.step

        def observed_reset(*args, **kwargs):
            budget.begin('resets')  # Veto wrapper's hidden warmup retry before delegation.
            result = original_reset(*args, **kwargs)
            budget.complete('resets')
            return result

        def observed_step(action):
            nonlocal capture
            capture_start = time.monotonic()
            if capture is None:
                capture = PhysicalCapture(environment, np)
                save_json(arm_dir / 'static.json', capture.static)
                if baseline_static is not None:
                    assert_parity(baseline_static, capture.static, 'static fixture catalog')
            issued = None if action is None else action_values(action)
            raw_index = budget.arm['raw_actual_started'] + 1
            if warmup and ((raw_index == 1 and issued is not None)
                           or (raw_index != 1 and issued != [0., 0., 0.])):
                raise ValueError('unexpected warmup action/order')
            before = capture.dynamic()
            pre = dict(event='PRE', raw_index=raw_index, warmup=warmup, decision=decision,
                       action=issued, state=before, state_sha256=canonical_sha(before))
            journal.record('raw.jsonl', pre)
            if raw_parity_required(slot, raw_index):
                assert_parity(baseline_raw[2 * (raw_index - 1)], pre, 'raw PRE ' + str(raw_index))
            data['phase_wall_s']['raw_capture'] += time.monotonic() - capture_start
            budget.begin('raw')
            delegate_start = time.monotonic()
            result = original_step(action)
            data['phase_wall_s']['raw_delegate'] += time.monotonic() - delegate_start
            budget.complete('raw')
            # Retain returned raw data BEFORE optional capture/parity can raise.
            capture_start = time.monotonic()
            raw_times.append(float(raw.t))
            raw_actions.append([0., 0., 0.] if issued is None else issued)
            returned = dict(raw_index=raw_index, action=issued, t=float(raw.t), reward=float(result[1]),
                terminated=bool(result[2]), truncated=bool(result[3]), info=json_value(result[4]),
                raw_image_sha256=hashlib.sha256(np.ascontiguousarray(result[0]).tobytes()).hexdigest())
            raw_returns.append(returned)
            after = capture.dynamic()
            post = dict(event='POST', raw_index=raw_index, warmup=warmup, decision=decision,
                action=issued, internal_reset_no_action=issued is None, t=float(raw.t), reward=float(result[1]),
                terminated=bool(result[2]), truncated=bool(result[3]), info=json_value(result[4]),
                state=after, state_sha256=canonical_sha(after))
            journal.record('raw.jsonl', post)
            if raw_parity_required(slot, raw_index):
                assert_parity(baseline_raw[2 * (raw_index - 1) + 1], post, 'raw POST ' + str(raw_index))
            data['phase_wall_s']['raw_capture'] += time.monotonic() - capture_start
            return result

        raw.reset, raw.step = observed_reset, observed_step

        def endpoint(observation, info, reward=None, terminated=False, truncated=False):
            arr = np.asarray(observation)
            if (arr.dtype != np.float32 or arr.shape != (4, 84, 84) or not np.isfinite(arr).all()
                    or arr.min() < 0 or arr.max() > 1 or not np.array_equal(arr, environment.environment.stack_state)):
                raise ValueError('exact normalized decision observation required')
            state = capture.full()
            row = dict(state=state, state_sha256=canonical_sha(state), info=json_value(info), reward=reward,
                terminated=bool(terminated), truncated=bool(truncated), image_sha256=hashlib.sha256(arr.tobytes()).hexdigest())
            images.append(arr.copy())
            times.append(float(raw.t))
            data['endpoints'].append(row)
            journal.record('boundaries.jsonl', dict(index=len(images) - 1, **row))
            journal.observation(arr.tobytes())
            i = len(images) - 1
            if baseline and (slot['arm'] == 'repeat' or i <= slot['anchor_prefix_decisions']):
                assert_parity(baseline['endpoints'][i], row, 'full boundary ' + str(i))
                if baseline_images[i].dtype != arr.dtype or baseline_images[i].tobytes() != arr.tobytes():
                    raise ValueError('decision observation bytes differ')
            return row

        validate_protocol(protocol_path, digest)
        data['resource_samples'].append(resource_admission(directory, protocol))
        budget.reserve(warmup=True)
        phase_start = time.monotonic()
        observation, info = environment.reset()
        warmup = False
        initial = endpoint(observation, info)
        data['geometry'] = verify_geometry(environment, reference, np)
        if baseline:
            assert_parity(baseline['geometry'], data['geometry'], 'static geometry')
        if initial['image_sha256'] != reference['initial_observation_sha256']:
            raise ValueError('initial observation differs from consumed profile')
        flush_start = time.monotonic()
        journal.flush()
        budget.seal()
        data['phase_wall_s']['block_flush'] += time.monotonic() - flush_start
        data['phase_wall_s']['reset'] = time.monotonic() - phase_start
        if model is not None:
            data['agent_reset_started'] += 1
            model.reset(observation.copy())
            data['agent_reset_completed'] += 1
        for decision in range(slot['decisions']):
            phase_start = time.monotonic()
            budget.check()
            if decision < slot['anchor_prefix_decisions']:
                if model is not None:
                    action = champion_action(model, observation, data, decision)
                else:
                    action = baseline['decisions'][decision]['action']
            else:
                if decision == slot['anchor_prefix_decisions']:
                    if model is not None:
                        data['nominal'] = champion_action(model, observation, data, decision)
                    else:
                        data['nominal'] = baseline['nominal']
                    data['tail_actions'] = tail_actions(data['nominal'], slot['arm'], slot['steering_delta'], slot['pedal_direction'])
                action = data['tail_actions'][decision - slot['anchor_prefix_decisions']]
            action = action_values(action)
            if model is not None and decision <= slot['anchor_prefix_decisions']:
                old = reference['decision_trace'][decision]
                assert_parity([old[key] for key in ('steer', 'gas', 'brake')], action,
                              'natural champion historical action ' + str(decision))
                assert_parity(old['controller']['evaluation_only']['observation_sha256'],
                              data['endpoints'][-1]['image_sha256'], 'natural champion historical pixels')
            if baseline and (slot['arm'] == 'repeat' or decision < slot['anchor_prefix_decisions']):
                assert_parity(baseline['decisions'][decision]['action'], action, 'prefix/repeat action')
            pre = data['endpoints'][-1]
            budget.reserve()
            actions.append(action)
            row: dict[str, Any] = dict(index=decision, action=action, pre_endpoint=decision, post_endpoint=None,
                       raw_before=budget.arm['raw_completed'], status='INTENT')
            data['decisions'].append(row)
            journal.record('decisions.jsonl', dict(row))
            budget.begin('decisions')
            observation, reward, terminated, truncated, info = environment.step(np.asarray(action, dtype=np.float64))
            budget.complete('decisions')
            row.update(post_endpoint=decision + 1, raw_after=budget.arm['raw_completed'], status='COMPLETE')
            journal.record('decisions.jsonl', dict(row))
            post = endpoint(observation, info, float(reward), terminated, truncated)
            if (row['raw_after'] - row['raw_before'] != SKIP
                    or not math.isclose(post['state']['t'] - pre['state']['t'], .08, rel_tol=0, abs_tol=1e-12)):
                raise ValueError('partial/changed hold retained; no replacement')
            if (terminated or truncated) and decision != slot['decisions'] - 1:
                raise ValueError('early terminal retained; no replacement')
            flush_start = time.monotonic()
            journal.flush()
            budget.seal()
            data['phase_wall_s']['block_flush'] += time.monotonic() - flush_start
            data['resource_samples'].append(resource_admission(directory, protocol))
            phase = 'prefix' if decision < slot['anchor_prefix_decisions'] else 'suffix'
            data['phase_wall_s'][phase] += time.monotonic() - phase_start
        validate_act_timings(data)
        if budget.arm['raw_completed'] != slot['raw_ticks']:
            raise ValueError('actual arm budget differs')
        data['import_origins'] = legacy.verify_imports(protocol)
        validate_protocol(protocol_path, digest)
        data['status'] = 'COMPLETE'
    except BaseException as error:
        data['status'], data['failure'] = 'PARTIAL_FAILED', f'{type(error).__name__}: {error}'
        # Flush evidence before committing exact caught-exception counters.
        journal.flush()
        budget.fail()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            persist()
        finally:
            journal.close()
            if environment is not None:
                environment.close()


def output_path(path):
    path = Path(path).absolute()
    if (path != path.resolve() or not path.is_relative_to(ROOT / 'runs') or path == ROOT / 'runs'
            or path.exists() or not path.parent.is_dir()):
        raise ValueError('new contained runs output with existing parent required; no retry')
    return path


def collect(protocol_path, digest, output, authorization):
    if authorization != digest or not digest:
        raise ValueError('explicit --authorize-collection must match final protocol SHA')
    protocol = validate_protocol(protocol_path, digest)
    output = output_path(output)
    if str(output) != protocol['output_directory']:
        raise ValueError('one-use frozen output mismatch')
    slots, limits = protocol['contract']['schedule'], protocol['resource_limits']
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    run: dict[str, Any] = dict(schema=SCHEMA, scope=SCOPE, protocol_sha256=digest, status='RUNNING',
        parent_pid=os.getpid(), started_monotonic=start, deadline_monotonic=start + limits['total_wall_s'], rows=[], failure=None)
    save_json(output / 'run.json', run)
    save_json(output / 'counters.json', empty_counters(slots))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
        SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy', CUDA_VISIBLE_DEVICES='')
    environment.update({key: '1' for key in legacy.THREAD_ENV})
    try:
        for index, slot in enumerate(slots):
            validate_protocol(protocol_path, digest)
            resource_admission(output, protocol)
            remaining = run['deadline_monotonic'] - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('total wall limit exhausted before next arm')
            command = [str(PYTHON), '-I', '-B', protocol['collector']['path'], '_worker',
                '--protocol', str(Path(protocol_path).resolve()), '--sha256', digest,
                '--output', str(output), '--index', str(index), '--authorize-collection', authorization]
            row: dict[str, Any] = dict(slot=slot, command=command, status='STARTED', returncode=None)
            run['rows'].append(row)
            save_json(output / 'run.json', run)
            print(json.dumps(dict(event='ARM_START', index=index, slot=slot, remaining_wall_s=remaining)), flush=True)
            arm_start = time.monotonic()
            with (output / (slot['slot'] + '.log')).open('xb') as log:
                try:
                    result = subprocess.run(command, cwd=output, env=environment, stdout=log,
                        stderr=subprocess.STDOUT, timeout=min(remaining, limits['child_wall_s'] + 10), check=False)
                    row.update(returncode=result.returncode, status='EXITED')
                except subprocess.TimeoutExpired:
                    row['status'] = 'KILLED_TIMEOUT'
                    raise
                finally:
                    row['wall_s'] = time.monotonic() - arm_start
                    row['counters'] = read_json(output / 'counters.json')
                    row['log_sha256'] = sha(output / (slot['slot'] + '.log'))
                    arm_dir = output / slot['slot']
                    receipt_path = arm_dir / 'receipt.json'
                    row['receipt_sha256'] = sha(receipt_path) if receipt_path.is_file() else None
                    row['final_artifacts_sha256'] = ({p.name: sha(p) for p in arm_dir.iterdir() if p.is_file()}
                                                     if arm_dir.is_dir() else {})
                    if (arm_dir / 'state.json').is_file():
                        saved = read_json(arm_dir / 'state.json')
                        row['phase_wall_s'] = saved['phase_wall_s']
                        row['peak_sampled_vm_hwm_bytes'] = max((r['vm_hwm_bytes'] for r in saved['resource_samples']), default=None)
                    save_json(output / 'run.json', run)
                    print(json.dumps(dict(event='ARM_END', slot=slot['slot'], status=row['status'],
                        wall_s=row['wall_s'], returncode=row['returncode'], phase_wall_s=row.get('phase_wall_s'),
                        peak_sampled_vm_hwm_bytes=row.get('peak_sampled_vm_hwm_bytes'))), flush=True)
            if result.returncode or read_json(receipt_path)['status'] != 'COMPLETE':
                raise RuntimeError('partial child failure; no later arm/retry/replacement')
            if index == 0:
                measured = row['wall_s'] / slot['raw_ticks']
                forecast = measured * sum(s['raw_ticks'] for s in slots[1:]) * limits['throughput_safety_factor']
                run['first_baseline_admission'] = dict(wall_s=row['wall_s'], raw_ticks=slot['raw_ticks'],
                    seconds_per_raw=measured, remaining_wall_forecast_s=forecast,
                    remaining_wall_s=run['deadline_monotonic'] - time.monotonic())
                save_json(output / 'run.json', run)
                print(json.dumps(dict(event='FIRST_BASELINE_RESOURCE', **run['first_baseline_admission'])), flush=True)
                if measured > limits['first_baseline_max_s_per_raw'] or forecast > run['deadline_monotonic'] - time.monotonic():
                    raise RuntimeError('first baseline throughput cannot fit frozen remaining budget; no expansion')
        counts = read_json(output / 'counters.json')['totals']
        for kind, expected in [('resets', len(slots)), ('decisions', sum(s['decisions'] for s in slots)),
                               ('raw', sum(s['raw_ticks'] for s in slots)), ('warmup', WARMUP * len(slots))]:
            if any(counts[kind + suffix] != expected for suffix in ('_started', '_actual_started', '_completed')):
                raise RuntimeError('final reserved/actual counters differ')
        validate_protocol(protocol_path, digest)
        run['status'] = 'COMPLETE'
    except BaseException as error:
        run['status'], run['failure'] = 'PARTIAL_FAILED', f'{type(error).__name__}: {error}'
        raise
    finally:
        run['counters'] = read_json(output / 'counters.json')
        run['wall_s'] = time.monotonic() - start
        run['evidence_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name not in ('run.json', 'run.json.tmp')}
        save_json(output / 'run.json', run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('template', 'validate', 'collect', '_worker'))
    parser.add_argument('--anchors', type=Path)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--authorize-collection')
    args = parser.parse_args(argv)
    if args.command == 'template':
        print(json.dumps(protocol_template(read_json(args.anchors) if args.anchors else None,
                                          read_json(args.references) if args.references else None), indent=2))
        return
    if args.protocol is None or args.sha256 is None:
        parser.error('--protocol and --sha256 required')
    if args.command == 'validate':
        validate_protocol(args.protocol, args.sha256)
        print(json.dumps(dict(valid=True, simulator_imports=0, agent_instances=0, resets=0, steps=0)))
    else:
        if args.output is None:
            parser.error('--output required')
        if args.command == 'collect':
            collect(args.protocol, args.sha256, args.output, args.authorize_collection)
        else:
            worker(args.output, args.index, args.protocol, args.sha256, args.authorize_collection)


if __name__ == '__main__':
    main()
