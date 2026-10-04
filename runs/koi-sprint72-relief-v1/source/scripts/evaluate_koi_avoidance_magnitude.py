"""One frozen four-pair consumed-TRAIN study; only `run` may create/reset an env.

The magnitude adapter changes the nominal action before the exact submitted v1
shield. Taps are passive: no shadow controller, action feedback, or plan replay.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile

from scripts import evaluate_koi_nominal_trajectory as nominal
from scripts import analyze_koi_collision_shield as shield_analysis
from scripts import evaluate_koi_minimum_clearance_ab as observer_helpers


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'submissions/20261002-crossing-projection-collision-shield-v1-baseline'
BASELINE_SHA, SHIELD_SHA = nominal.BASELINE_SHA, nominal.SHIELD_SHA
SCHEMA = 'koi-avoidance-magnitude-v1'
ARMS = ('frozen_shield', 'avoidance_magnitude')
CELLS = ((1, 3184000013), (1, 3184000015), (2, 3184000015), (3, 3184000015))
SUBGROUPS = {f'{t}:{s}': 'ordinary_controls' for t, s in CELLS}
PROTECTED = (49300, 49301, 49302, 49303, 51300, 51301, 51302, 51303)
GATE = dict(nominal.GATE)
METRICS = dict(nominal.METRICS,
    scope='Four outcome-selected consumed ordinary TRAIN layouts/two roads; not fresh/confirmation/blind/official',
    wrapper_state='Raw ticks precede outer damage/counter updates. Raw damage/counter equals decision pre-state; post damage/counter reproduces source collision/reward update; every decision damage and final headline bind to post-wrapper telemetry. Playfield terminal reward substitutes -100 at last raw tick.',
    treatment='One crossing call -> stateless magnitude action -> unchanged submitted v1 shield. Independent taps bind all three actions; pedals exact; no final-action feedback, retained plan, release or recovery state. Existing v1 encounter/occlusion/actuator bookkeeping is unchanged administrative state, not a new controller.')
HELPERS = (*nominal.HELPERS, 'scripts/evaluate_koi_avoidance_magnitude.py',
           'scripts/analyze_koi_avoidance_magnitude.py',
           'haic/algorithms/koi/avoidance_magnitude.py', 'haic/algorithms/koi/steering_terms.py')
STATE_KEYS = ('last', 'nominal')
PROPOSAL_FIELDS = ('required_clearance_m', 'approach_m', 'ttc_s', 'geometric_demand',
                   'collision_risk', 'legacy_magnitude', 'magnitude', 'side',
                   'current_bbox_m', 'projected_shift_m')
sha, read_json, save, local_path = nominal.sha, nominal.read_json, nominal.save, nominal.local_path
ActionProbe = nominal.ActionProbe
shield = nominal.shield


def scheduled_slots():
    rows = []
    for track, seed in CELLS:
        arms = ARMS if (track + seed) % 2 else ARMS[::-1]
        rows.extend(dict(track_id=track, seed=seed, mode=arm, repeat=0,
                         file=f'{track}-{seed}-r0-{arm}.json', status='unrun') for arm in arms)
    return rows


def valid_episode(episode):
    return (nominal.valid_episode(episode) and math.isfinite(episode['damage'])
            and type(episode.get('steps')) is int
            and (not episode['completed'] or (type(episode.get('lapTimeMs')) in (int, float)
                 and math.isfinite(episode['lapTimeMs']) and episode['lapTimeMs'] >= 0)))


def exclusion_audit(root) -> dict[str, Any]:
    """Read allocation metadata only, never protected observations or trajectories."""
    root = Path(root)
    directory = root / 'runs/koi-steering-generalization-v1'
    protocol = read_json(directory / 'protocol.json')
    if protocol.get('excludes') != list(PROTECTED) or any(t == 4 or s in PROTECTED or s < 3184000001 for t, s in CELLS):
        raise ValueError('protected/old24/Track4 exclusion differs')
    pins = {str(directory / 'protocol.json'): sha(directory / 'protocol.json')}
    registry = root / 'experiments/train-seed-claims'
    for seed in sorted(set(PROTECTED) | {s for _, s in CELLS}):
        path = registry / f'seed-{seed}.json'
        if not path.exists():
            if seed not in PROTECTED:
                raise ValueError('consumed TRAIN allocation receipt missing')
            continue
        claim = read_json(path)
        if claim.get('geometry_seed') != seed:
            raise ValueError('allocation identity differs')
        if seed not in PROTECTED and (claim.get('partition') != 'TRAIN'
                or claim.get('obstacles') is not True or claim.get('study_id') != 'koi-steering-generalization-v1'):
            raise ValueError('candidate-relevant allocation differs')
        pins[str(path)] = sha(path)
    return dict(excludes=list(PROTECTED), source_sha256=pins, observations_read=0,
                selection='Explicit consumed ordinary TRAIN reuse, not freshness or a new seed claim.')


def consumed_evidence():
    """Rehash completed selected receipts, including the original partial study."""
    selected = shield.legacy.validate_cells(list(CELLS), baseline='crossing_projection')
    prior = shield.legacy.PRIOR
    pins = {str(prior / n): sha(prior / n) for n in ('protocol.json', 'episode-report.json')}
    prior_report = read_json(prior / 'episode-report.json')
    prior_protocol = read_json(prior / 'protocol.json')
    peaks = []
    for item in selected:
        path = Path(item['file'])
        episode = read_json(path)
        if not valid_episode(episode):
            raise ValueError('invalid consumed TRAIN episode')
        pins[str(path)] = item['sha256']
        for field in ('raw_trace',):
            stream = local_path(prior, episode[field + '_file'])
            if sha(stream) != episode[field + '_sha256']:
                raise ValueError('consumed raw stream differs')
            pins[str(stream)] = sha(stream)
        row = next(r for r in prior_report['rows'] if r['file'] == path.name)
        process = local_path(prior, row['process_file'])
        if sha(process) != row['process_sha256']:
            raise ValueError('consumed process receipt differs')
        receipt = read_json(process)
        if (receipt.get('error') is not None or receipt.get('status') != 'completed'
                or receipt.get('operator_sha256') != prior_protocol['operator_sha256']
                or receipt.get('slot_id') != row['slot_id']):
            raise ValueError('invalid consumed process receipt')
        pins[str(process)] = sha(process)
        original = local_path(prior, receipt['original_process_file'])
        if sha(original) != receipt['original_process_sha256']:
            raise ValueError('consumed original process receipt differs')
        if read_json(original).get('status') != 'completed' or read_json(original).get('error') is not None:
            raise ValueError('invalid consumed original process receipt')
        pins[str(original)] = sha(original)
        peak = episode.get('peak_rss_bytes')
        if type(peak) is not int or peak <= 0:
            raise ValueError('comparable serialization peak missing')
        peaks.append(peak)
    if sha(prior / 'reset-ledger.jsonl') != prior_report['reset_ledger_sha256']:
        raise ValueError('partial prior reset ledger differs')
    pins[str(prior / 'reset-ledger.jsonl')] = prior_report['reset_ledger_sha256']
    directory = ROOT / 'runs/koi-nominal-trajectory-v1'
    protocol_sha = sha(directory / 'protocol.json')
    parent = nominal.validate_frozen(directory, protocol_sha)
    if any(parent['subgroups'].get(f'{t}:{s}') != 'ordinary_controls' for t, s in CELLS):
        raise ValueError('ordinary consumed subgroup differs')
    report = read_json(directory / 'episode-report.json')
    if report.get('operator_error') is not None or report.get('protocol_sha256') != protocol_sha:
        raise ValueError('completed nominal study required')
    shield_analysis.validate_ledger(directory, report['rows'], protocol_sha)
    seen, walls, sizes, decisions, outcomes = set(), [], [], [], []
    for row in report['rows']:
        key = row['track_id'], row['seed'], row['mode']
        if key[:2] not in CELLS:
            continue
        if key in seen or key[2] not in nominal.ARMS or row['status'] != 'completed' or row['repeat'] != 0:
            raise ValueError('consumed nominal coverage differs')
        seen.add(key)
        episode, states = shield_analysis.load_episode(directory, row, protocol_sha)
        if not valid_episode(episode):
            raise ValueError('invalid consumed nominal episode')
        paths = [local_path(directory, row['file']),
                 local_path(directory, episode['raw_trace_file']),
                 local_path(directory, episode['decision_stream_file']),
                 local_path(directory, row['file']).with_suffix('.process.json')]
        for path in paths:
            pins[str(path)] = sha(path)
        wall = read_json(paths[-1])['wall_time_s']
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError('invalid comparable throughput receipt')
        walls.append(wall)
        sizes.append(sum(p.stat().st_size for p in paths))
        decisions.append(episode['steps'])
        if key[2] == ARMS[0]:
            outcomes.append(dict(track_id=key[0], seed=key[1], finished=episode['completed'],
                                 damage=episode['damage'], collisions=episode['collisions'],
                                 geometry_sha256=episode['geometry_sha256'], raw_ticks=len(states) - 1))
    if seen != {(t, s, arm) for t, s in CELLS for arm in nominal.ARMS}:
        raise ValueError('all eight consumed ordinary nominal episodes required')
    for name in ('protocol.json', 'episode-report.json', 'reset-ledger.jsonl'):
        pins[str(directory / name)] = sha(directory / name)
    exclusion = exclusion_audit(ROOT)
    pins.update(exclusion['source_sha256'])
    return dict(source_sha256=pins, exclusion_audit=exclusion, baseline_outcomes=outcomes,
                cells=[list(c) for c in CELLS], fresh=False,
                prior_partial_operator_error=prior_report.get('operator_error'),
                historical=dict(episodes=len(walls), peak_serialization_rss_bytes=max(peaks),
                    max_episode_bytes=max(sizes), total_child_wall_s=math.fsum(walls),
                    max_child_wall_s=max(walls), total_decisions=sum(decisions),
                    slowest_seconds_per_decision=max(w / n for w, n in zip(walls, decisions)),
                    decisions_per_child_wall_second=sum(decisions) / math.fsum(walls)),
                selection='Four outcome-selected ordinary cells already consumed by crossing/v2 and nominal A/B; two road geometries, not fresh development or generalization. Partial generalization/reset evidence retained, not retried or relabeled.')


def resource_measurements(destination):
    value: dict[str, Any] = observer_helpers.resource_measurements(destination)
    groups = value['visible_cgroup_ancestors']
    if not groups or groups[-1]['path'] != '/sys/fs/cgroup':
        raise ValueError('cgroup ancestry telemetry incomplete')
    finite = [g['raw_headroom_bytes'] for g in groups if g['memory_max'] != 'max']
    value.update(measured_at_utc=datetime.now(timezone.utc).isoformat(), measured_at_ns=time.time_ns(),
                 minimum_raw_cgroup_headroom_bytes=min(finite) if finite else None,
                 cpu_load_average=list(os.getloadavg()), cpu_threads=1, gpu_used=False,
                 current_process_rss_bytes=resident_rss_bytes())
    for group in groups:
        stats = dict(line.split() for line in group['memory_stat'].splitlines())
        group['possibly_reclaimable_clean_inactive_file_bytes'] = max(0, int(stats['inactive_file'])
            - int(stats['file_dirty']) - int(stats['file_writeback']))
    return value


def resident_rss_bytes():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def resource_forecast(evidence):
    old = evidence['historical']
    return dict(peak_child_bytes=math.ceil(1.5 * old['peak_serialization_rss_bytes']),
        memory_reserve_bytes=256 * 1024**2, competing_ram_growth_bytes=256 * 1024**2,
        per_episode_disk_bytes=2 * old['max_episode_bytes'], disk_reserve_bytes=128 * 1024**2,
        competing_disk_growth_bytes=128 * 1024**2, temp_bytes=16 * 1024**2,
        child_timeout_s=math.ceil(max(4 * old['max_child_wall_s'],
                                    1.6 * 1200 * old['slowest_seconds_per_decision'])),
        estimated_eight_episode_wall_s=1.6 * old['total_child_wall_s'],
        serial_children=1, gpu=False, historical=old,
        rationale='Current measurements are taken at freeze and each admission/reset. Comparable selected CPU21 episodes include serialization RSS, raw/decision/report bytes and child-wall throughput. 1.5x peak plus256MiB observer growth reserve and256MiB competing-job allowance;2x maximum slot bytes plus128MiB receipt reserve and128MiB competing writers. No checkpoints/replay/GPU. Timeout covers1.6x slowest decision cost at1200 decisions or4x longest episode; latency forecast is not a driving benchmark or official quota. CPU contention and shared-cgroup OOM increases are recorded warnings when the measured next-operation capacity still fits; missing telemetry or a forecast shortfall blocks.')


def check_resources(measurement, forecast, remaining, *, allocated_child_bytes=0, previous=None) -> dict[str, Any]:
    warnings = []
    ram = max(0, forecast['peak_child_bytes'] - allocated_child_bytes) + forecast['memory_reserve_bytes'] + forecast['competing_ram_growth_bytes']
    if measurement['host_mem_available_bytes'] < ram or not measurement['visible_cgroup_ancestors']:
        raise ValueError('host/cgroup RAM forecast does not fit or telemetry missing')
    for group in measurement['visible_cgroup_ancestors']:
        if group['raw_headroom_bytes'] is not None and group['raw_headroom_bytes'] < ram:
            raise ValueError('finite ancestor RAM forecast does not fit; cache not credited')
        events = dict(line.split() for line in group['memory_events'].splitlines())
        if not {'oom', 'oom_kill'} <= set(events) or not group['memory_pressure']:
            raise ValueError('OOM/pressure telemetry missing')
        if previous is not None:
            old = next((g for g in previous['visible_cgroup_ancestors'] if g['path'] == group['path']), None)
            if old is None:
                raise ValueError('cgroup ancestry changed')
            old_events = dict(line.split() for line in old['memory_events'].splitlines())
            if any(int(events[k]) < int(old_events[k]) for k in ('oom', 'oom_kill')):
                raise ValueError('cgroup OOM counters reset; telemetry identity uncertain')
            if any(int(events[k]) > int(old_events[k]) for k in ('oom', 'oom_kill')):
                warnings.append('shared-cgroup OOM counters increased: ' + group['path'])
    out, temporary = measurement['disk']
    need = remaining * forecast['per_episode_disk_bytes'] + forecast['disk_reserve_bytes'] + forecast['competing_disk_growth_bytes']
    temp_need = forecast['temp_bytes'] + forecast['disk_reserve_bytes']
    if out['device'] == temporary['device']:
        if out['free_bytes'] < need + forecast['temp_bytes']:
            raise ValueError('shared filesystem forecast does not fit')
    elif out['free_bytes'] < need or temporary['free_bytes'] < temp_need:
        raise ValueError('separate output/temporary filesystem forecast does not fit')
    return dict(required_incremental_ram_bytes=ram, remaining_output_bytes=remaining * forecast['per_episode_disk_bytes'],
                required_output_filesystem_bytes=need, temporary_filesystem_bytes=temp_need,
                cache_credited=False, allowed=True, warnings=warnings)


def freeze_protocol(output):
    output = Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('new output directory under an existing parent required')
    evidence = consumed_evidence()
    manifest = read_json(BUNDLE / 'manifest.json')
    if manifest['zip_sha256'] != BASELINE_SHA or sha(BUNDLE / 'submission.zip') != BASELINE_SHA:
        raise ValueError('exact frozen champion required')
    if sha(ROOT / 'haic/algorithms/koi/collision_shield.py') != SHIELD_SHA:
        raise ValueError('unchanged v1 shield helper required')
    environment = read_json(ROOT / 'runs/koi-nominal-trajectory-v1/protocol.json')['environment_sha256']
    if len(environment) != 142 or any(sha(p) != h for p, h in environment.items()):
        raise ValueError('frozen environment142 differs')
    forecast = resource_forecast(evidence)
    measurement = resource_measurements(output.parent)
    check_resources(measurement, forecast, len(scheduled_slots()))
    output.mkdir(exist_ok=False)
    sources = {}
    for relative in HELPERS:
        source, copied = ROOT / relative, output / 'source' / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        sources[relative] = dict(original=str(source), sha256=sha(copied))
    shutil.copyfile(BUNDLE / 'submission.zip', output / 'baseline.zip')
    if sha(output / 'baseline.zip') != BASELINE_SHA:
        raise ValueError('copied champion ZIP differs')
    model = {}
    with zipfile.ZipFile(output / 'baseline.zip') as archive:
        names = archive.namelist()
        if (len(names) != 11 or len(set(names)) != 11 or set(names) != set(manifest['submission_members'])
                or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names)):
            raise ValueError('unsafe or changed champion members')
        for name in names:
            expected = manifest['files_sha256']['source/' + name]
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('champion member differs')
            model[name] = expected
        archive.extractall(output / 'model')
    if model['haic_agent/collision_shield_runtime.py'] != SHIELD_SHA:
        raise ValueError('submitted shield source differs')
    preserved = {str(ROOT / 'agent.py'): sha(ROOT / 'agent.py'), str(BUNDLE / 'manifest.json'): sha(BUNDLE / 'manifest.json')}
    for name, expected in manifest['files_sha256'].items():
        path = local_path(BUNDLE, name)
        if sha(path) != expected:
            raise ValueError('frozen champion bundle differs: ' + name)
        preserved[str(path)] = expected
    review_path = ROOT / 'experiments/koi-avoidance-magnitude-v1-runtime-review.json'
    review_pins = {}
    if review_path.exists():
        review = read_json(review_path)
        if (review.get('status') != 'RUNTIME_REVIEW_PASS_NO_DRIVING_CLAIM'
                or review.get('simulator_constructions') != 0 or review.get('simulator_resets') != 0
                or review.get('official_action') is not False):
            raise ValueError('zero-reset runtime review differs')
        review_pins[str(review_path)] = sha(review_path)
        for group in ('source_sha256', 'preservation_sha256'):
            for name, expected in review[group].items():
                source = local_path(ROOT, name)
                if sha(source) != expected:
                    raise ValueError('reviewed runtime source differs: ' + name)
                review_pins[str(source)] = expected
    protocol = dict(schema=SCHEMA, scope=shield.legacy.SCOPE, repository_root=str(ROOT),
        fresh=False, official_action=False, cells=[list(c) for c in CELLS], arms=list(ARMS),
        subgroups=SUBGROUPS, gate=GATE, metric_definitions=METRICS, schedule=scheduled_slots(),
        baseline_zip_sha256=BASELINE_SHA, shield_sha256=SHIELD_SHA,
        source_inventory=sources, model_source_sha256=model, environment_sha256=environment,
        consumed_evidence=evidence, preservation_sha256=preserved, runtime_review_sha256=review_pins,
        python=str(shield.legacy.PYTHON), snapshot=str(shield.legacy.SNAPSHOT),
        runtime_executable_sha256=sha(shield.legacy.PYTHON.resolve()),
        max_decisions=1200, frame_skip=4, warmup_ticks=50, raw_fps=50,
        child_timeout_s=forecast['child_timeout_s'], resource_plan=dict(forecast=forecast, freeze_measurement=measurement),
        candidate_state_keys=list(STATE_KEYS), candidate_action_feedback=False,
        execution='serial isolated CPU21; one child; no GPU; full natural episodes; no automatic retry/tuning/repeat')
    save(output / 'protocol.json', protocol)
    return sha(output / 'protocol.json')


def validate_frozen(output, protocol_sha256):
    output = Path(output).resolve()
    if sha(output / 'protocol.json') != protocol_sha256:
        raise ValueError('protocol hash mismatch')
    protocol = read_json(output / 'protocol.json')
    contract = dict(schema=SCHEMA, cells=[list(c) for c in CELLS], arms=list(ARMS),
        subgroups=SUBGROUPS, gate=GATE, metric_definitions=METRICS, schedule=scheduled_slots(),
        baseline_zip_sha256=BASELINE_SHA, shield_sha256=SHIELD_SHA, fresh=False, official_action=False,
        max_decisions=1200, frame_skip=4, warmup_ticks=50, raw_fps=50,
        python=str(shield.legacy.PYTHON), snapshot=str(shield.legacy.SNAPSHOT),
        candidate_state_keys=list(STATE_KEYS), candidate_action_feedback=False)
    if any(protocol.get(k) != v for k, v in contract.items()):
        raise ValueError('fixed magnitude protocol contract differs')
    if set(protocol['source_inventory']) != set(HELPERS) or len(protocol['environment_sha256']) != 142:
        raise ValueError('source closure differs')
    for relative, pin in protocol['source_inventory'].items():
        if sha(pin['original']) != pin['sha256'] or sha(local_path(output / 'source', relative)) != pin['sha256']:
            raise ValueError('imported helper source differs: ' + relative)
    if protocol['source_inventory']['haic/algorithms/koi/collision_shield.py']['sha256'] != SHIELD_SHA:
        raise ValueError('shield helper source differs')
    for pins in (protocol['environment_sha256'], protocol['preservation_sha256'], protocol['consumed_evidence']['source_sha256']):
        if not pins or any(sha(p) != h for p, h in pins.items()):
            raise ValueError('frozen inventory differs')
    if any(sha(p) != h for p, h in protocol['runtime_review_sha256'].items()):
        raise ValueError('runtime review inventory differs')
    if exclusion_audit(protocol['repository_root']) != protocol['consumed_evidence']['exclusion_audit']:
        raise ValueError('protected allocation inventory differs')
    forecast = resource_forecast(protocol['consumed_evidence'])
    if protocol['resource_plan']['forecast'] != forecast or protocol['child_timeout_s'] != forecast['child_timeout_s']:
        raise ValueError('resource forecast differs')
    if sha(output / 'baseline.zip') != BASELINE_SHA or sha(Path(protocol['python']).resolve()) != protocol['runtime_executable_sha256']:
        raise ValueError('frozen champion/runtime differs')
    with zipfile.ZipFile(output / 'baseline.zip') as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != 11 or set(names) != set(protocol['model_source_sha256']):
            raise ValueError('model inventory differs')
        for name, expected in protocol['model_source_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected or sha(local_path(output / 'model', name)) != expected:
                raise ValueError('model member differs')
    return protocol


def candidate_state(candidate):
    if candidate is None:
        return dict(candidate_state_keys=[], candidate_driver_hidden=True, candidate_feedback_method=False)
    if (tuple(sorted(vars(candidate))) != STATE_KEYS or not isinstance(candidate.last, dict)
            or hasattr(candidate, 'driver') or hasattr(candidate, 'observe_executed_action')):
        raise ValueError('candidate must hide driver and have only nominal/last diagnostics, no feedback or retained state')
    return dict(candidate_state_keys=list(STATE_KEYS), candidate_driver_hidden=True, candidate_feedback_method=False)


def compose(model, mode, candidate_class=None):
    if mode not in ARMS or (mode == ARMS[1] and candidate_class is None):
        raise ValueError('known magnitude arm and candidate constructor required')
    original = ActionProbe(model.driver.driver)
    candidate = candidate_class(original) if mode == ARMS[1] and candidate_class is not None else None
    tap = ActionProbe(candidate) if candidate is not None else original
    model.driver.driver = tap
    if candidate is not None and candidate.nominal is not original:
        raise ValueError('candidate must use the once-called original tap')
    candidate_state(candidate)
    return model, tap, original, candidate


def measured_act(model, tap, original, candidate, observation):
    before = tap.calls, original.calls
    candidate_state(candidate)
    action = model.act(observation)
    if (tap.calls, original.calls) != (before[0] + 1, before[1] + 1):
        raise ValueError('each nominal must be called exactly once')
    candidate_state(candidate)
    issued = [float(v) for v in action]
    if issued[1:] != original.action[1:] or tap.action[1:] != original.action[1:]:
        raise ValueError('nominal/shield pedals differ from original crossing')
    if model.driver.last_shield['baseline_action'] != tap.action:
        raise ValueError('changed nominal was not the unchanged shield input')
    return action


def worker(output, index, protocol_sha256, *, import_only=False):
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if not 0 <= index < len(protocol['schedule']):
        raise ValueError('worker outside frozen schedule')
    slot = protocol['schedule'][index]
    sys.path.insert(0, str(output / 'model'))
    entry = importlib.import_module('agent')
    if Path(str(entry.__file__)).resolve() != output / 'model/agent.py':
        raise ValueError('Agent imported outside champion bundle')
    from haic.algorithms.koi.avoidance_magnitude import AvoidanceMagnitudeAgent
    model, tap, original, candidate = compose(entry.Agent(), slot['mode'], AvoidanceMagnitudeAgent)
    import haic_agent
    haic_agent.__path__.append(str(Path(protocol['snapshot']) / 'haic_agent'))
    sys.path.insert(0, protocol['snapshot'])
    import cv2
    import numpy as np
    import torch
    from training.env_factory import create_training_environment
    run_episode = importlib.import_module('training.evaluate_closed_loop').run_episode
    pins = dict(protocol['environment_sha256'])
    pins.update({str(output / 'source' / p): v['sha256'] for p, v in protocol['source_inventory'].items()})
    pins.update({str(output / 'model' / p): h for p, h in protocol['model_source_sha256'].items()})
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(output) or path.is_relative_to(protocol['snapshot']):
                if str(path) not in pins or sha(path) != pins[str(path)]:
                    raise ValueError('imported module outside frozen closure: ' + str(path))
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(slot['seed'])
    if torch.__version__ != '2.1.0+cpu' or np.__version__ != '1.26.0' or getattr(cv2, '__version__') != '4.8.1':
        raise ValueError('pinned CPU21 dependencies required')
    if import_only:
        print(json.dumps(dict(import_only=True, environment_resets=0, mode=slot['mode'],
                              peak_import_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                              current_resident_rss_bytes=resident_rss_bytes(),
                              rss_note='ru_maxrss high-water retained separately; it may include launcher history and is not current resident memory.',
                              **candidate_state(candidate))))
        return
    start = read_json(output / 'run-start.json')
    if start['protocol_sha256'] != protocol_sha256 or start['preflight_sha256'] != sha(output / 'preflight.json'):
        raise ValueError('worker requires bound run-start/preflight')
    validate_preflight(read_json(output / 'preflight.json'), protocol_sha256, protocol['resource_plan']['forecast'])
    ledger_path = output / 'reset-ledger.jsonl'
    if ledger_path.exists() and any(json.loads(line)['file'] == slot['file'] for line in ledger_path.read_text().splitlines()):
        raise ValueError('existing slot reset evidence; never retry')
    path = output / slot['file']
    raw_path, decisions_path = path.with_suffix('.raw.jsonl'), path.with_suffix('.decisions.jsonl')
    holder = {}

    class MeasuredAgent:
        def reset(self, observation):
            return model.reset(observation)

        def act(self, observation):
            self.pixels = hashlib.sha256(observation.tobytes()).hexdigest()
            self.action = np.asarray(measured_act(model, tap, original, candidate, observation)).tolist()
            return self.action

        def last_step_diagnostics(self):
            info = dict(tap.nominal.last_step_diagnostics())
            info.update(shield=copy.deepcopy(model.driver.last_shield),
                        baseline_probe_action=copy.deepcopy(tap.action), crossing_probe_action=copy.deepcopy(original.action),
                        nominal_probe_calls=tap.calls, crossing_probe_calls=original.calls,
                        executed_action_observations=0, **candidate_state(candidate),
                        evaluation_only=dict(holder['environment'].last, observation_sha256=self.pixels))
            decisions.write(json.dumps(dict(step=holder['environment'].step_number,
                                           action=self.action, controller=info), allow_nan=False) + '\n')
            return info

    with raw_path.open('x', buffering=1) as raw, decisions_path.open('x', buffering=1) as decisions:
        def factory(**kwargs):
            validate_frozen(output, protocol_sha256)
            measurement = resource_measurements(output)
            try:
                assessment = check_resources(measurement, protocol['resource_plan']['forecast'], len(protocol['schedule']) - index,
                    allocated_child_bytes=measurement['current_process_rss_bytes'], previous=protocol['resource_plan']['freeze_measurement'])
            except BaseException as failure:
                shield.legacy.append_jsonl(path.with_suffix('.resources.jsonl'), dict(phase='before_environment', measurement=measurement, error=f'{type(failure).__name__}: {failure}'))
                raise
            shield.legacy.append_jsonl(path.with_suffix('.resources.jsonl'), dict(phase='before_environment', measurement=measurement, assessment=assessment, error=None))
            observer = shield.make_observer(create_training_environment(**kwargs), raw,
                output / 'reset-ledger.jsonl', slot, protocol_sha256, observer_helpers.footprint_measurements)
            raw_reset = observer.unwrapped.reset

            def guarded_reset(*args, **reset_kwargs):
                if observer.reset_count:
                    raise ValueError('one reset per child; never retry')
                validate_frozen(output, protocol_sha256)
                measured = resource_measurements(output)
                try:
                    allowed = check_resources(measured, protocol['resource_plan']['forecast'], len(protocol['schedule']) - index,
                        allocated_child_bytes=measured['current_process_rss_bytes'], previous=protocol['resource_plan']['freeze_measurement'])
                except BaseException as failure:
                    shield.legacy.append_jsonl(path.with_suffix('.resources.jsonl'), dict(phase='before_reset', measurement=measured, error=f'{type(failure).__name__}: {failure}'))
                    raise
                shield.legacy.append_jsonl(path.with_suffix('.resources.jsonl'), dict(phase='before_reset', measurement=measured, assessment=allowed, error=None))
                return raw_reset(*args, **reset_kwargs)

            observer.unwrapped.reset = guarded_reset
            holder['environment'] = observer
            return observer
        result = run_episode(mode=slot['mode'], track_id=slot['track_id'], seed=slot['seed'],
            agent=MeasuredAgent(), max_decisions=1200, plan_budget_seconds=4.5,
            capture_trace=True, fail_on_invalid_action=True, environment_factory=factory)
    observer = holder.get('environment')
    initial = getattr(observer, 'initial', None)
    states = ([initial] if initial is not None else []) + [json.loads(line) for line in raw_path.read_text().splitlines()]
    result.update(repeat=0, scope=protocol['scope'], protocol_sha256=protocol_sha256,
        catalog=getattr(observer, 'catalog', None), initial_state=initial,
        geometry_sha256=getattr(observer, 'geometry_sha256', None),
        initial_observation_sha256=getattr(observer, 'initial_observation_sha256', None),
        raw_trace_file=raw_path.name, raw_trace_sha256=sha(raw_path),
        decision_stream_file=decisions_path.name, decision_stream_sha256=sha(decisions_path),
        off_track_count_max=getattr(observer, 'max_counter', None),
        physical=shield.legacy.physical_metrics(states, getattr(observer, 'obstacles', [])),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        current_resident_rss_bytes=resident_rss_bytes(),
        versions=dict(python=sys.version, numpy=np.__version__, cv2=getattr(cv2, '__version__'), torch=torch.__version__))
    save(path, result)
    shield.legacy.append_jsonl(output / 'reset-ledger.jsonl', dict(slot, event='episode_end',
        protocol_sha256=protocol_sha256, retire_reason=result['retire_reason'], error=result['error']))


def worker_command(output, index, protocol_sha256, python, *, import_only=False):
    return [python, '-I', '-B', '-c',
        'import sys; sys.path.insert(0,sys.argv[1]); from scripts.evaluate_koi_avoidance_magnitude import worker; worker(sys.argv[2],int(sys.argv[3]),sys.argv[4],import_only=sys.argv[5]=="True")',
        str(Path(output) / 'source'), str(output), str(index), protocol_sha256, str(import_only)]


def validate_preflight(receipt, protocol_sha256, forecast):
    arms = receipt.get('arms', [])
    if (receipt.get('protocol_sha256') != protocol_sha256 or receipt.get('environment_resets') != 0
            or len(arms) != 2 or {r.get('mode') for r in arms} != set(ARMS)):
        raise ValueError('bound zero-reset preflight required')
    for row in arms:
        rss = row.get('current_resident_rss_bytes')
        expected_keys = list(STATE_KEYS) if row['mode'] == ARMS[1] else []
        if (row.get('import_only') is not True or row.get('environment_resets') != 0
                or row.get('candidate_driver_hidden') is not True or row.get('candidate_feedback_method') is not False
                or row.get('candidate_state_keys') != expected_keys
                or type(rss) is not int or not 0 < rss <= forecast['peak_child_bytes']):
            raise ValueError('stateless arm preflight/RSS differs')


def preflight(output, protocol_sha256):
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if (output / 'preflight.json').exists():
        raise ValueError('existing preflight receipt; never overwrite')
    receipts = []
    for index in (0, 1):
        child = subprocess.run(worker_command(output, index, protocol_sha256, protocol['python'], import_only=True),
            cwd=output / 'model', env=shield.legacy.ENV, capture_output=True, text=True, timeout=30)
        if child.returncode:
            raise RuntimeError(child.stderr)
        receipt = json.loads(child.stdout)
        if (receipt.get('import_only') is not True or receipt.get('environment_resets') != 0
                or receipt.get('mode') != protocol['schedule'][index]['mode']):
            raise ValueError('zero-reset arm preflight differs')
        rss = receipt.get('current_resident_rss_bytes')
        if type(rss) is not int or not 0 < rss <= protocol['resource_plan']['forecast']['peak_child_bytes']:
            raise ValueError('preflight resident RSS exceeds/misses frozen forecast')
        receipts.append(receipt)
    result = dict(protocol_sha256=protocol_sha256, environment_resets=0, arms=receipts)
    validate_preflight(result, protocol_sha256, protocol['resource_plan']['forecast'])
    save(output / 'preflight.json', result)
    return result


def run(output, protocol_sha256):
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if sha(__file__) != protocol['source_inventory']['scripts/evaluate_koi_avoidance_magnitude.py']['sha256']:
        raise ValueError('operator differs from frozen source')
    if any((output / name).exists() for name in ('run-start.json', 'episode-report.json', 'reset-ledger.jsonl')):
        raise ValueError('existing study evidence; never restart')
    if any(list(output.glob(Path(row['file']).stem + '.*')) for row in protocol['schedule']):
        raise ValueError('existing slot evidence; never overwrite')
    receipt = read_json(output / 'preflight.json')
    validate_preflight(receipt, protocol_sha256, protocol['resource_plan']['forecast'])
    save(output / 'run-start.json', dict(protocol_sha256=protocol_sha256, time_ns=time.time_ns(), preflight_sha256=sha(output / 'preflight.json')))
    rows, error = copy.deepcopy(protocol['schedule']), None
    try:
        for index, row in enumerate(rows):
            measurement = resource_measurements(output)
            try:
                assessment = check_resources(measurement, protocol['resource_plan']['forecast'], len(rows) - index,
                                             previous=protocol['resource_plan']['freeze_measurement'])
            except BaseException as failure:
                shield.legacy.append_jsonl(output / 'resource-receipts.jsonl', dict(slot=index, measurement=measurement, error=f'{type(failure).__name__}: {failure}'))
                raise
            shield.legacy.append_jsonl(output / 'resource-receipts.jsonl', dict(slot=index, measurement=measurement, assessment=assessment, error=None))
            row['status'] = 'started'
            path = output / row['file']
            command = worker_command(output, index, protocol_sha256, protocol['python'])
            started, child_error, stdout, stderr = time.monotonic(), None, '', ''
            try:
                child = subprocess.run(command, cwd=output / 'model', env=shield.legacy.ENV,
                    capture_output=True, text=True, timeout=protocol['child_timeout_s'])
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
                for suffix, text in (('.stdout.txt', stdout), ('.stderr.txt', stderr)):
                    with path.with_suffix(suffix).open('xb') as stream:
                        stream.write(text if isinstance(text, bytes) else text.encode())
                process = path.with_suffix('.process.json')
                save(process, dict(command=command, protocol_sha256=protocol_sha256,
                                   wall_time_s=time.monotonic() - started, error=child_error))
                row['process_sha256'] = sha(process)
            if child_error:
                raise RuntimeError(child_error)
            if not valid_episode(read_json(path)):
                raise ValueError('invalid/capped episode; retain evidence and stop')
            row.update(status='completed', sha256=sha(path),
                artifacts=[dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size)
                           for p in sorted(output.glob(path.stem + '.*'))])
            print(json.dumps(row), flush=True)
    except BaseException as failure:
        error = f'{type(failure).__name__}: {failure}'
        for row in rows:
            if row['status'] == 'started':
                row.update(status='operator_error', error=error,
                    partial_artifacts=[dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size)
                                       for p in sorted(output.glob(Path(row['file']).stem + '.*'))])
        raise
    finally:
        save(output / 'episode-report.json', dict(protocol_sha256=protocol_sha256, rows=rows, operator_error=error,
            operator_artifacts=[dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size)
                for name in ('run-start.json', 'preflight.json', 'reset-ledger.jsonl', 'resource-receipts.jsonl')
                if (p := output / name).exists()]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'validate', 'preflight', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if args.command == 'freeze':
        print(freeze_protocol(args.output))
    elif not args.protocol_sha256:
        parser.error('--protocol-sha256 required')
    elif args.command == 'validate':
        validate_frozen(args.output, args.protocol_sha256)
        print(json.dumps(dict(valid=True, environment_resets=0)))
    elif args.command == 'preflight':
        print(json.dumps(preflight(args.output, args.protocol_sha256)))
    else:
        run(args.output, args.protocol_sha256)


if __name__ == '__main__':
    main()
