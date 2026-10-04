"""Source-bound eight-pair consumed-TRAIN operator; only explicit run may reset.

The candidate is the exact champion ZIP with one pre-arrival FarHazard call and
one added helper. No shadow policy or steering-only magnitude adapter is used.
"""

import argparse
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile

from scripts import evaluate_koi_avoidance_magnitude as magnitude


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'submissions/20261002-crossing-projection-collision-shield-v1-baseline'
SCHEMA = 'koi-sprint72-relief-v1'
OPERATOR = 'scripts/evaluate_koi_sprint72_relief.py'
ANALYZER = 'scripts/analyze_koi_sprint72_relief.py'
HELPER = 'haic/algorithms/koi/sprint72_relief.py'
HELPER_MEMBER = 'haic_agent/sprint72_relief_runtime.py'
FAR_MEMBER = 'haic_agent/far_hazard_runtime.py'
FAR_SHA = '12fc519edb1e3ac33e1f339ebf5dcf95913a529d51705cbd827e9eff2557f6d4'
PARENT_SHA = '77bda836df249b08d0a0863a82ffbf7f0016123e76fa97046a2393515ad38078'
BASELINE_SHA, SHIELD_SHA = magnitude.BASELINE_SHA, magnitude.SHIELD_SHA
ARMS = ('frozen_shield', 'sprint72_relief')
CELLS = ((1, 3184000002), (2, 3184000006), (2, 3184000001), (3, 3184000002),
         (1, 3184000013), (1, 3184000015), (2, 3184000015), (3, 3184000015))
SUBGROUPS = dict(magnitude.nominal.SUBGROUPS)
PROTECTED = magnitude.PROTECTED
REVERSE_CASE = dict(track_id=2, seed=3184000006, historical_decisions=[194, 195, 447, 448],
                    exclusion_allowed=False)
GATE = dict(max_kept_lap_increase_ms=20, min_improving_geometry_seeds=2,
            complete_matched_evidence=True, no_lost_finish=True,
            no_per_cell_damage_collision_increase=True, no_new_hit=True,
            offroad_nonincrease=True, baseline_window_coverage=True,
            cap_undershoot_reduced=True, repeated_brake_gas_reduced=True,
            common_lap_mean_negative=True, source_law_and_shield_contract=True,
            prefix_identity=True, tuning=False)
HELPERS = (*magnitude.HELPERS, OPERATOR, ANALYZER, HELPER)
PATCH_POINT = b'        side = clearance = cap = None\n        if self.steps > 10:\n'
IMPORT_POINT = b'from haic_agent.pixel_features import current_frame, road_centers, center_at\n'
HELPER_IMPORT = b'from haic_agent.sprint72_relief_runtime import apply_sprint72_relief\n'
HELPER_CALL = b'        apply_sprint72_relief(self, action, near, far, impact_before)\n'
DIAGNOSTIC_FIELDS = ('sprint72_eligible', 'sprint72_applied', 'sprint72_space_ok',
                     'sprint72_pre_action', 'sprint72_prearrival_action',
                     'sprint72_brake_formula', 'sprint72_impact_before')
sha, read_json, save, local_path = magnitude.sha, magnitude.read_json, magnitude.save, magnitude.local_path
shield, observer_helpers = magnitude.shield, magnitude.observer_helpers
ActionProbe, valid_episode = magnitude.ActionProbe, magnitude.valid_episode
resource_measurements = magnitude.resource_measurements
resident_rss_bytes, check_resources = magnitude.resident_rss_bytes, magnitude.check_resources


def scheduled_slots():
    rows = []
    for track, seed in CELLS:
        arms = ARMS if (track + seed) % 2 else ARMS[::-1]
        rows.extend(dict(track_id=track, seed=seed, mode=arm, repeat=0,
                         file=f'{track}-{seed}-r0-{arm}.json', status='unrun') for arm in arms)
    return rows


def champion_members(path=None):
    path = Path(path) if path is not None else BUNDLE / 'submission.zip'
    if sha(path) != BASELINE_SHA:
        raise ValueError('exact frozen champion ZIP required')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != 11 or len(set(names)) != 11 or any(
                Path(n).is_absolute() or '..' in Path(n).parts or n.endswith('/') for n in names)):
            raise ValueError('unsafe or changed champion members')
        members = {name: archive.read(name) for name in names}
    if (hashlib.sha256(members[FAR_MEMBER]).hexdigest() != FAR_SHA or
            hashlib.sha256(members['haic_agent/collision_shield_runtime.py']).hexdigest() != SHIELD_SHA):
        raise ValueError('frozen FarHazard/shield source differs')
    return members


def patch_far_source(source):
    if hashlib.sha256(source).hexdigest() != FAR_SHA:
        raise ValueError('frozen FarHazard source hash differs')
    if source.count(PATCH_POINT) != 1 or source.count(IMPORT_POINT) != 1:
        raise ValueError('unique pre-arrival patch point required')
    result = source.replace(IMPORT_POINT, IMPORT_POINT + HELPER_IMPORT, 1)
    result = result.replace(PATCH_POINT, PATCH_POINT.replace(b'        if self.steps',
                            HELPER_CALL + b'        if self.steps'), 1)
    compile(result, FAR_MEMBER, 'exec')
    return result


def candidate_members(*, baseline_zip=None, helper=None):
    """Return bytes only; never patch originals, import policy, or construct an env."""
    members = champion_members(baseline_zip)
    source = (Path(helper) if helper is not None else ROOT / HELPER).read_bytes()
    compile(source, HELPER_MEMBER, 'exec')
    members[FAR_MEMBER] = patch_far_source(members[FAR_MEMBER])
    members[HELPER_MEMBER] = source
    return members


def exclusion_audit(root) -> dict[str, Any]:
    """Allocation metadata only; this is explicit reuse, never a freshness audit."""
    root = Path(root)
    path = root / 'runs/koi-steering-generalization-v1/protocol.json'
    if read_json(path).get('excludes') != list(PROTECTED):
        raise ValueError('protected exclusions differ')
    pins = {str(path): sha(path)}
    for seed in sorted(set(PROTECTED) | {s for _, s in CELLS}):
        path = root / f'experiments/train-seed-claims/seed-{seed}.json'
        if not path.exists() and seed in PROTECTED:
            continue
        claim = read_json(path)
        if claim.get('geometry_seed') != seed or (seed not in PROTECTED and (
                claim.get('partition') != 'TRAIN' or claim.get('obstacles') is not True or
                claim.get('study_id') != 'koi-steering-generalization-v1')):
            raise ValueError('consumed TRAIN allocation differs')
        pins[str(path)] = sha(path)
    return dict(excludes=list(PROTECTED), source_sha256=pins, observations_read=0,
                selection='All eight consumed nominal-trajectory cells; no fresh/protected/Track4 cells.')


def consumed_evidence() -> dict[str, Any]:
    directory = ROOT / 'runs/koi-nominal-trajectory-v1'
    parent = magnitude.nominal.validate_frozen(directory, PARENT_SHA)
    if parent['cells'] != [list(c) for c in CELLS]:
        raise ValueError('all eight canonical consumed cells required')
    report = read_json(directory / 'episode-report.json')
    if report.get('operator_error') is not None or report.get('protocol_sha256') != PARENT_SHA:
        raise ValueError('completed canonical study required')
    ledger = magnitude.shield_analysis.validate_ledger(directory, report['rows'], PARENT_SHA)
    if ledger['reset_intents'] != 16 or ledger['episode_ends'] != 16:
        raise ValueError('canonical reset coverage differs')
    pins = {str(directory / n): sha(directory / n)
            for n in ('protocol.json', 'episode-report.json', 'reset-ledger.jsonl')}
    # The earlier partial run includes serialization RSS and irreversible exposure.
    prior = shield.legacy.PRIOR
    prior_report = read_json(prior / 'episode-report.json')
    prior_protocol = read_json(prior / 'protocol.json')
    for name in ('protocol.json', 'episode-report.json', 'reset-ledger.jsonl'):
        pins[str(prior / name)] = sha(prior / name)
    if pins[str(prior / 'reset-ledger.jsonl')] != prior_report['reset_ledger_sha256']:
        raise ValueError('partial prior reset ledger differs')
    peaks = []
    for item in shield.legacy.validate_cells(list(CELLS), baseline='crossing_projection'):
        path = Path(item['file'])
        episode = read_json(path)
        if not valid_episode(episode):
            raise ValueError('invalid consumed TRAIN episode')
        pins[str(path)] = item['sha256']
        raw = local_path(prior, episode['raw_trace_file'])
        if sha(raw) != episode['raw_trace_sha256']:
            raise ValueError('consumed raw stream differs')
        pins[str(raw)] = sha(raw)
        row = next(r for r in prior_report['rows'] if r['file'] == path.name)
        process = local_path(prior, row['process_file'])
        receipt = read_json(process)
        if (sha(process) != row['process_sha256'] or receipt.get('error') is not None
                or receipt.get('status') != 'completed' or receipt.get('slot_id') != row['slot_id']
                or receipt.get('operator_sha256') != prior_protocol['operator_sha256']):
            raise ValueError('consumed process receipt differs')
        original = local_path(prior, receipt['original_process_file'])
        if (sha(original) != receipt['original_process_sha256'] or
                read_json(original).get('status') != 'completed' or read_json(original).get('error') is not None):
            raise ValueError('consumed original process differs')
        pins.update({str(process): sha(process), str(original): sha(original)})
        peak = episode.get('peak_rss_bytes')
        if type(peak) is not int or peak <= 0:
            raise ValueError('comparable serialization RSS missing')
        peaks.append(peak)
    seen, walls, sizes, decisions, outcomes = set(), [], [], [], []
    for row in report['rows']:
        key = row['track_id'], row['seed'], row['mode']
        if (key in seen or key[:2] not in CELLS or key[2] not in magnitude.nominal.ARMS
                or row['status'] != 'completed' or row['repeat'] != 0):
            raise ValueError('canonical consumed coverage differs')
        seen.add(key)
        episode, states = magnitude.shield_analysis.load_episode(directory, row, PARENT_SHA)
        if not valid_episode(episode):
            raise ValueError('invalid canonical episode')
        paths = [local_path(directory, row['file']), local_path(directory, episode['raw_trace_file']),
                 local_path(directory, episode['decision_stream_file']),
                 local_path(directory, row['file']).with_suffix('.process.json')]
        pins.update({str(p): sha(p) for p in paths})
        wall = read_json(paths[-1])['wall_time_s']
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError('comparable child throughput missing')
        walls.append(wall)
        sizes.append(sum(p.stat().st_size for p in paths))
        decisions.append(episode['steps'])
        if key[2] == ARMS[0]:
            outcomes.append(dict(track_id=key[0], seed=key[1], finished=episode['completed'],
                damage=episode['damage'], collisions=episode['collisions'],
                geometry_sha256=episode['geometry_sha256'], raw_ticks=len(states) - 1))
    if seen != {(t, s, a) for t, s in CELLS for a in magnitude.nominal.ARMS}:
        raise ValueError('all sixteen canonical episodes required')
    exclusions = exclusion_audit(ROOT)
    pins.update(exclusions['source_sha256'])
    return dict(source_sha256=pins, cells=[list(c) for c in CELLS], fresh=False,
        exclusion_audit=exclusions, baseline_outcomes=outcomes,
        prior_partial_operator_error=prior_report.get('operator_error'), reverse_case=REVERSE_CASE,
        historical=dict(episodes=len(walls), peak_serialization_rss_bytes=max(peaks),
            max_episode_bytes=max(sizes), total_child_wall_s=math.fsum(walls), max_child_wall_s=max(walls),
            total_decisions=sum(decisions),
            slowest_seconds_per_decision=max(w / n for w, n in zip(walls, decisions)),
            decisions_per_child_wall_second=sum(decisions) / math.fsum(walls)),
        selection='All eight diagnosed consumed TRAIN cells/five roads, including natural DNF and reverse windows; no subset optimization or freshness claim.')


def resource_forecast(evidence):
    result = magnitude.resource_forecast(evidence)
    result['estimated_sixteen_episode_wall_s'] = result.pop('estimated_eight_episode_wall_s')
    return result


def fixed_contract():
    analyzer = importlib.import_module('scripts.analyze_koi_sprint72_relief')
    if not callable(getattr(analyzer, 'analyze', None)) or not isinstance(analyzer.METRIC_DEFINITIONS, dict):
        raise ValueError('full callable analyzer and metric definitions required before freeze')
    return dict(schema=SCHEMA, scope=shield.legacy.SCOPE, cells=[list(c) for c in CELLS],
        arms=list(ARMS), subgroups=SUBGROUPS, gate=GATE, metric_definitions=analyzer.METRIC_DEFINITIONS,
        schedule=scheduled_slots(), baseline_zip_sha256=BASELINE_SHA, shield_sha256=SHIELD_SHA,
        far_source_sha256=FAR_SHA, parent_protocol_sha256=PARENT_SHA, reverse_case=REVERSE_CASE,
        fresh=False, official_action=False, max_decisions=1200, frame_skip=4, warmup_ticks=50, raw_fps=50,
        python=str(shield.legacy.PYTHON), snapshot=str(shield.legacy.SNAPSHOT),
        candidate_action_feedback=False, source_check_stage='prearrival before unchanged shield',
        conditions=dict(base_seed='slot seed, unchanged', track_id='slot track, unchanged',
            environment_factory='training.env_factory.create_training_environment',
            obstacles='unchanged track-defined obstacles', frame_stack=4, render_mode=None))


def freeze_protocol(output):
    output = Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('new output directory under an existing parent required')
    # Resolve the complete analyzer/helper closure BEFORE creating any output.
    contract = fixed_contract()
    sources = {name: dict(original=str(ROOT / name), sha256=sha(ROOT / name)) for name in HELPERS}
    baseline, candidate = champion_members(), candidate_members()
    manifest = read_json(BUNDLE / 'manifest.json')
    if manifest['zip_sha256'] != BASELINE_SHA or set(manifest['submission_members']) != set(baseline):
        raise ValueError('champion manifest differs')
    preserved = {str(BUNDLE / 'manifest.json'): sha(BUNDLE / 'manifest.json')}
    for root, inventory in ((BUNDLE, manifest['files_sha256']),
                            (ROOT, manifest['preserved_external_artifacts_sha256'])):
        for relative, expected in inventory.items():
            path = local_path(root, relative)
            if sha(path) != expected:
                raise ValueError('preservation inventory differs: ' + str(path))
            preserved[str(path)] = expected
    evidence = consumed_evidence()
    environment = read_json(ROOT / 'runs/koi-nominal-trajectory-v1/protocol.json')['environment_sha256']
    if len(environment) != 142 or any(sha(p) != h for p, h in environment.items()):
        raise ValueError('frozen environment142 differs')
    if sources['haic/algorithms/koi/collision_shield.py']['sha256'] != SHIELD_SHA:
        raise ValueError('unchanged v1 shield helper required')
    forecast, measurement = resource_forecast(evidence), resource_measurements(output.parent)
    check_resources(measurement, forecast, len(scheduled_slots()))
    output.mkdir(exist_ok=False)
    for name, pin in sources.items():
        copied = output / 'source' / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pin['original'], copied)
        if sha(copied) != pin['sha256']:
            raise ValueError('source changed while freezing')
    shutil.copyfile(BUNDLE / 'submission.zip', output / 'baseline.zip')
    with zipfile.ZipFile(output / 'candidate.zip', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(candidate.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=9)
    models = {}
    for arm, filename, members in ((ARMS[0], 'baseline.zip', baseline), (ARMS[1], 'candidate.zip', candidate)):
        directory = 'models/' + arm
        for name, data in members.items():
            path = output / directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
        models[arm] = dict(directory=directory, zip_file=filename, zip_sha256=sha(output / filename),
            source_sha256={n: hashlib.sha256(b).hexdigest() for n, b in members.items()})
    protocol = dict(contract, repository_root=str(ROOT), source_inventory=sources, models=models,
        environment_sha256=environment, consumed_evidence=evidence, preservation_sha256=preserved,
        runtime_executable_sha256=sha(shield.legacy.PYTHON.resolve()), child_timeout_s=forecast['child_timeout_s'],
        resource_plan=dict(forecast=forecast, freeze_measurement=measurement),
        execution='serial isolated CPU21; one child per full natural episode; no GPU/retry/tuning/repeat')
    save(output / 'protocol.json', protocol)
    identity = sha(output / 'protocol.json')
    validate_frozen(output, identity)
    return identity


def validate_frozen(output, protocol_sha256):
    output = Path(output).resolve()
    if sha(output / 'protocol.json') != protocol_sha256:
        raise ValueError('protocol hash mismatch')
    protocol = read_json(output / 'protocol.json')
    if any(protocol.get(k) != v for k, v in fixed_contract().items()):
        raise ValueError('fixed sprint72 protocol contract differs')
    if set(protocol['source_inventory']) != set(HELPERS) or len(protocol['environment_sha256']) != 142:
        raise ValueError('source closure differs')
    for name, pin in protocol['source_inventory'].items():
        if sha(pin['original']) != pin['sha256'] or sha(local_path(output / 'source', name)) != pin['sha256']:
            raise ValueError('imported helper source differs: ' + name)
    if protocol['source_inventory']['haic/algorithms/koi/collision_shield.py']['sha256'] != SHIELD_SHA:
        raise ValueError('shield helper source differs')
    for pins in (protocol['environment_sha256'], protocol['preservation_sha256'],
                 protocol['consumed_evidence']['source_sha256']):
        if not pins or any(sha(p) != h for p, h in pins.items()):
            raise ValueError('frozen inventory differs')
    if exclusion_audit(protocol['repository_root']) != protocol['consumed_evidence']['exclusion_audit']:
        raise ValueError('consumed allocation inventory differs')
    if sha(Path(protocol['python']).resolve()) != protocol['runtime_executable_sha256']:
        raise ValueError('frozen CPU21 executable differs')
    forecast = resource_forecast(protocol['consumed_evidence'])
    if protocol['resource_plan']['forecast'] != forecast or protocol['child_timeout_s'] != forecast['child_timeout_s']:
        raise ValueError('resource forecast differs')
    baseline = champion_members(output / 'baseline.zip')
    candidate = candidate_members(baseline_zip=output / 'baseline.zip', helper=output / 'source' / HELPER)
    if set(protocol['models']) != set(ARMS):
        raise ValueError('independent arm model inventories required')
    for arm, filename, members in ((ARMS[0], 'baseline.zip', baseline), (ARMS[1], 'candidate.zip', candidate)):
        model = protocol['models'][arm]
        expected = {n: hashlib.sha256(b).hexdigest() for n, b in members.items()}
        if (model['directory'] != 'models/' + arm or model['zip_file'] != filename
                or model['source_sha256'] != expected or sha(output / filename) != model['zip_sha256']):
            raise ValueError('arm model inventory differs')
        directory = local_path(output, model['directory'])
        if {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} != set(members):
            raise ValueError('unexpected or missing extracted model files')
        with zipfile.ZipFile(output / filename) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(members):
                raise ValueError('arm ZIP inventory differs')
            for name, data in members.items():
                if archive.read(name) != data or sha(local_path(directory, name)) != expected[name]:
                    raise ValueError('arm model member differs')
    return protocol


def controller_state(driver):
    """Passive existing controller state, without last diagnostics or world data."""
    state = {k: v for k, v in vars(driver).items() if k not in ('last', 'base')}
    state['base'] = vars(driver.base)
    return json.loads(json.dumps(state, default=lambda value: value.tolist(), allow_nan=False))


def compose(model, mode):
    if mode not in ARMS:
        raise ValueError('known sprint72 arm required')
    tap = ActionProbe(model.driver.driver)
    model.driver.driver = tap
    return model, tap


def measured_act(model, tap, observation):
    before, keys = tap.calls, set(vars(tap.nominal))
    action = model.act(observation)
    if tap.calls != before + 1:
        raise ValueError('final nominal must be called exactly once')
    if set(vars(tap.nominal)) != keys:
        raise ValueError('candidate added persistent controller state')
    issued = [float(v) for v in action]
    if issued[1:] != tap.action[1:]:
        raise ValueError('unchanged shield must preserve nominal pedals')
    if model.driver.last_shield['baseline_action'] != tap.action:
        raise ValueError('final nominal differs from unchanged shield input')
    if not tap.nominal.brake_history or tap.nominal.brake_history[-1] != issued[2]:
        raise ValueError('brake_history must record actual issued brake')
    return action


def load_model(output, mode, protocol):
    """Reject contamination, rather than unloading another arm's cached modules."""
    if mode not in ARMS or any(name == 'agent' or name == 'haic_agent' or
            name.startswith('haic_agent.') for name in sys.modules):
        raise ValueError('arm requires a fresh isolated package import')
    directory = local_path(output, protocol['models'][mode]['directory'])
    sys.path.insert(0, str(directory))
    entry = importlib.import_module('agent')
    if Path(str(entry.__file__)).resolve() != directory / 'agent.py':
        raise ValueError('Agent imported outside arm package')
    return compose(entry.Agent(), mode)


def worker(output, index, protocol_sha256, *, import_only=False):
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if not 0 <= index < len(protocol['schedule']):
        raise ValueError('worker outside frozen schedule')
    slot = protocol['schedule'][index]
    model, tap = load_model(output, slot['mode'], protocol)
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
    inventory = protocol['models'][slot['mode']]
    pins.update({str(output / inventory['directory'] / p): h for p, h in inventory['source_sha256'].items()})
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
            model_directory=inventory['directory'], model_zip_sha256=inventory['zip_sha256'],
            driver_state_keys=sorted(vars(tap.nominal)), executed_action_observations=0,
            peak_import_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            current_resident_rss_bytes=resident_rss_bytes())))
        return
    start = read_json(output / 'run-start.json')
    if start['protocol_sha256'] != protocol_sha256 or start['preflight_sha256'] != sha(output / 'preflight.json'):
        raise ValueError('worker requires bound run-start/preflight')
    validate_preflight(read_json(output / 'preflight.json'), protocol_sha256, protocol)
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
            self.before = controller_state(tap.nominal)
            self.action = np.asarray(measured_act(model, tap, observation)).tolist()
            self.after = controller_state(tap.nominal)
            return self.action

        def last_step_diagnostics(self):
            info = copy.deepcopy(tap.nominal.last_step_diagnostics())
            if slot['mode'] == ARMS[1] and not set(DIAGNOSTIC_FIELDS) <= set(info):
                raise ValueError('prearrival helper diagnostics missing')
            info.update(shield=copy.deepcopy(model.driver.last_shield),
                baseline_probe_action=copy.deepcopy(tap.action), nominal_probe_action=copy.deepcopy(tap.action),
                nominal_probe_calls=tap.calls, executed_action_observations=0,
                evaluation_only=dict(holder['environment'].last, observation_sha256=self.pixels,
                    controller_state_before=self.before, controller_state_after=self.after))
            decisions.write(json.dumps(dict(step=holder['environment'].step_number,
                action=self.action, controller=info), allow_nan=False) + '\n')
            return info

    def admit(phase):
        validate_frozen(output, protocol_sha256)
        measurement = resource_measurements(output)
        try:
            assessment = check_resources(measurement, protocol['resource_plan']['forecast'], len(protocol['schedule']) - index,
                allocated_child_bytes=measurement['current_process_rss_bytes'], previous=protocol['resource_plan']['freeze_measurement'])
        except BaseException as failure:
            shield.legacy.append_jsonl(path.with_suffix('.resources.jsonl'), dict(phase=phase,
                measurement=measurement, error=f'{type(failure).__name__}: {failure}'))
            raise
        shield.legacy.append_jsonl(path.with_suffix('.resources.jsonl'), dict(phase=phase,
            measurement=measurement, assessment=assessment, error=None))

    with raw_path.open('x', buffering=1) as raw, decisions_path.open('x', buffering=1) as decisions:
        def factory(**kwargs):
            if kwargs != dict(track_id=slot['track_id'], seed=slot['seed'], max_decisions=1200) or holder:
                raise ValueError('one exact frozen environment per child required')
            admit('before_environment')
            observer = shield.make_observer(create_training_environment(**kwargs), raw, ledger_path,
                slot, protocol_sha256, observer_helpers.footprint_measurements)
            raw_reset = observer.unwrapped.reset

            def guarded_reset(*args, **reset_kwargs):
                if observer.reset_count:
                    raise ValueError('one reset per child; never retry')
                admit('before_reset')
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
    shield.legacy.append_jsonl(ledger_path, dict(slot, event='episode_end',
        protocol_sha256=protocol_sha256, retire_reason=result['retire_reason'], error=result['error']))


def worker_command(output, index, protocol_sha256, python, *, import_only=False):
    return [python, '-I', '-B', '-c',
        'import sys; sys.path.insert(0,sys.argv[1]); from scripts.evaluate_koi_sprint72_relief import worker; worker(sys.argv[2],int(sys.argv[3]),sys.argv[4],import_only=sys.argv[5]=="True")',
        str(Path(output) / 'source'), str(output), str(index), protocol_sha256, str(import_only)]


def validate_preflight(receipt, protocol_sha256, protocol):
    arms = receipt.get('arms', [])
    if (receipt.get('protocol_sha256') != protocol_sha256 or receipt.get('environment_resets') != 0
            or len(arms) != 2 or {r.get('mode') for r in arms} != set(ARMS)):
        raise ValueError('bound zero-reset preflight required')
    state_keys = arms[0].get('driver_state_keys')
    for row in arms:
        rss, model = row.get('current_resident_rss_bytes'), protocol['models'][row['mode']]
        if (row.get('import_only') is not True or row.get('environment_resets') != 0
                or row.get('executed_action_observations') != 0 or not state_keys
                or row.get('driver_state_keys') != state_keys
                or row.get('model_directory') != model['directory']
                or row.get('model_zip_sha256') != model['zip_sha256']
                or type(rss) is not int or not 0 < rss <= protocol['resource_plan']['forecast']['peak_child_bytes']):
            raise ValueError('independent arm preflight/model/state/RSS differs')


def preflight(output, protocol_sha256):
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if (output / 'preflight.json').exists():
        raise ValueError('existing preflight receipt; never overwrite')
    receipts = []
    for index in (0, 1):
        mode = protocol['schedule'][index]['mode']
        child = subprocess.run(worker_command(output, index, protocol_sha256, protocol['python'], import_only=True),
            cwd=output / protocol['models'][mode]['directory'], env=shield.legacy.ENV,
            capture_output=True, text=True, timeout=30)
        if child.returncode:
            raise RuntimeError(child.stderr)
        receipt = json.loads(child.stdout)
        if receipt.get('mode') != mode:
            raise ValueError('preflight arm identity differs')
        receipts.append(receipt)
    result = dict(protocol_sha256=protocol_sha256, environment_resets=0, arms=receipts)
    validate_preflight(result, protocol_sha256, protocol)
    save(output / 'preflight.json', result)
    return result


def run(output, protocol_sha256):
    """Fail-safe serial launcher; preserves partial evidence and never auto-analyzes."""
    output = Path(output).resolve()
    protocol = validate_frozen(output, protocol_sha256)
    if sha(__file__) != protocol['source_inventory'][OPERATOR]['sha256']:
        raise ValueError('operator differs from frozen source')
    if any((output / name).exists() for name in ('run-start.json', 'episode-report.json', 'reset-ledger.jsonl')):
        raise ValueError('existing study evidence; never restart')
    if any(list(output.glob(Path(row['file']).stem + '.*')) for row in protocol['schedule']):
        raise ValueError('existing slot evidence; never overwrite')
    validate_preflight(read_json(output / 'preflight.json'), protocol_sha256, protocol)
    save(output / 'run-start.json', dict(protocol_sha256=protocol_sha256, time_ns=time.time_ns(),
        preflight_sha256=sha(output / 'preflight.json')))
    rows, error = copy.deepcopy(protocol['schedule']), None
    try:
        for index, row in enumerate(rows):
            validate_frozen(output, protocol_sha256)
            measurement = resource_measurements(output)
            try:
                assessment = check_resources(measurement, protocol['resource_plan']['forecast'], len(rows) - index,
                    previous=protocol['resource_plan']['freeze_measurement'])
            except BaseException as failure:
                shield.legacy.append_jsonl(output / 'resource-receipts.jsonl', dict(slot=index,
                    measurement=measurement, error=f'{type(failure).__name__}: {failure}'))
                raise
            shield.legacy.append_jsonl(output / 'resource-receipts.jsonl', dict(slot=index,
                measurement=measurement, assessment=assessment, error=None))
            row['status'] = 'started'
            path = output / row['file']
            command = worker_command(output, index, protocol_sha256, protocol['python'])
            started, child_error, stdout, stderr = time.monotonic(), None, '', ''
            try:
                child = subprocess.run(command, cwd=output / protocol['models'][row['mode']]['directory'],
                    env=shield.legacy.ENV, capture_output=True, text=True, timeout=protocol['child_timeout_s'])
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
            row.update(status='completed', sha256=sha(path), artifacts=[
                dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size) for p in sorted(output.glob(path.stem + '.*'))])
            print(json.dumps(row), flush=True)
    except BaseException as failure:
        error = f'{type(failure).__name__}: {failure}'
        for row in rows:
            if row['status'] == 'started':
                row.update(status='operator_error', error=error, partial_artifacts=[
                    dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size)
                    for p in sorted(output.glob(Path(row['file']).stem + '.*'))])
        raise
    finally:
        save(output / 'episode-report.json', dict(protocol_sha256=protocol_sha256, rows=rows,
            operator_error=error, operator_artifacts=[dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size)
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
