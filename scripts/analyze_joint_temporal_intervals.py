"""Cache-only interval revalidation on consumed TRAIN, never simulator execution.

Run from the repository root with ``python -B -m
scripts.analyze_joint_temporal_intervals``. Outputs are exclusive new files.
CAL maxima are descriptive empirical envelopes, not confidence or safety bounds.
The geometry-held-out rows were previously consumed; only this fitting excludes
them. No weights, physics constants, observer source or old evidence are changed.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SHA256 = 'f24fa32f4b85ab90ef677653bd52870f06a82a0415d551806448f43b5957cab9'
CAL = (3184000002, 3184000013, 3184000015)
HELD_OUT_FROM_FIT = (3184000001, 3184000006)
POSITION_FLOOR = np.r_[0., np.full(16, .5)]
YAW_FLOOR = np.r_[0., np.full(16, .05)]
COST_FLOOR = .05
PROVENANCES = ('measured_image', 'predicted_corrected_issued_action')
OUTPUTS = ('paired-analysis.json', 'calibration.json', 'summary.txt')
LIMITATION = ('All cells are consumed TRAIN. Held out means excluded from this fitting only; '
              'no held-out right-entry anchor. Three CAL roads cannot establish statistical '
              'coverage. Five shared road realizations and nineteen shared initial states '
              'are finite empirical stresses, not exhaustive nonlinear bounds or safety guarantees. '
              'Cost/progress/clearance proxies are not official scores or lap improvements.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def jsonable(value) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def split(seed):
    if seed in CAL:
        return 'CAL'
    require(seed in HELD_OUT_FROM_FIT, 'undeclared geometry seed')
    return 'HELD_OUT_FROM_FIT'


def pose_errors(actual, predicted):
    """Signed actual minus predicted, with wrapped yaw and Euclidean position."""
    difference = np.asarray(actual, float) - np.asarray(predicted, float)
    difference[..., 2] = (difference[..., 2] + np.pi) % (2 * np.pi) - np.pi
    return dict(signed=difference, position=np.linalg.norm(difference[..., :2], axis=-1),
                yaw=np.abs(difference[..., 2]))


def trajectory_arrays(row):
    predicted = np.asarray(row['comparison']['poses'], float)
    actual = np.asarray([r['poses'] for r in row['actual_trajectories']], float)
    oracle = np.stack([row['oracle_prediction'][k] for k in
                       ('relative_x', 'relative_y', 'yaw_delta')], axis=-1)[:, 0]
    oracle = np.concatenate((np.zeros((2, 1, 3)), oracle), axis=1)
    require(predicted.shape == (2, 19, 17, 3) and actual.shape == oracle.shape == (2, 17, 3),
            'cached trajectory shape differs from fixed H4/two arms/nineteen hypotheses')
    return actual, predicted, oracle


def calibrate_trajectories(rows):
    """Fit per-tick CENTRAL runtime errors, including dynamics, on CAL only.

    The same conservative tube will surround EACH shared hypothesis. The CAL
    central path is contained by construction, not evidence of generalization.
    NaN/Infinity is propagated into invalid calibration, never nanmax/zero-filled.
    """
    diagnostics, position, yaw = [], [], []
    for row in rows:
        actual, predicted, oracle = trajectory_arrays(row)
        errors = dict(actual_minus_runtime=pose_errors(actual, predicted[:, 0]),
                      actual_minus_oracle=pose_errors(actual, oracle),
                      oracle_minus_runtime=pose_errors(oracle, predicted[:, 0]))
        role = split(row['anchor']['seed'])
        diagnostics.append(dict(anchor_id=row['plan_anchor_id'], seed=row['anchor']['seed'],
                                split=role, observer_valid=row['observer_valid'], **errors))
        if role == 'CAL':
            position.extend(errors['actual_minus_runtime']['position'])
            yaw.extend(errors['actual_minus_runtime']['yaw'])
    require(position, 'no CAL trajectory paths')
    position, yaw = np.asarray(position), np.asarray(yaw)
    finite = np.isfinite(position).all() and np.isfinite(yaw).all()
    p = np.maximum(POSITION_FLOOR, np.max(position, axis=0))
    y = np.maximum(YAW_FLOOR, np.max(yaw, axis=0))
    initial_zero = bool(np.all(position[:, 0] == 0) and np.all(yaw[:, 0] == 0))
    return dict(supported=bool(finite and initial_zero), position_residual=p, yaw_residual=y,
                n_cal_paths=len(position), n_cal_roads=len({r['anchor']['seed'] for r in rows
                                                          if split(r['anchor']['seed']) == 'CAL'}),
                nonfinite_position_indices=np.argwhere(~np.isfinite(position)),
                nonfinite_yaw_indices=np.argwhere(~np.isfinite(yaw)), initial_zero=initial_zero,
                diagnostic_rows=diagnostics,
                method='max_CAL_arms_central_runtime_error_per_raw_tick_with_original_floors',
                oracle_caveat='Actual-oracle includes dynamics/reduced-state representation; '
                              'oracle-runtime isolates initialization sensitivity, not an additive RMSE decomposition.')


def whole_path_containment(actual, predicted, position, yaw):
    """One scenario must meet BOTH position and yaw at EVERY tick of a path."""
    predicted = np.asarray(predicted, float)
    require(predicted.ndim == 3 and predicted.shape[1:] == (17, 3), 'path shape')
    p, y = np.asarray(position, float), np.asarray(yaw, float)
    require(p.shape == y.shape == (17,), 'seventeen time-index residuals required')
    errors = pose_errors(np.asarray(actual, float)[None], predicted)
    valid = bool(np.isfinite(p).all() and np.isfinite(y).all() and (p >= 0).all() and (y >= 0).all())
    joint = ((errors['position'] <= p) & (errors['yaw'] <= y)
             & np.isfinite(errors['position']) & np.isfinite(errors['yaw']) & valid)
    whole = joint.all(axis=1)
    return dict(calibration_valid=valid, contained=bool(whole.any()), central_contained=bool(whole[0]),
                matching_whole_path_scenarios=np.flatnonzero(whole),
                per_tick_union_contained=bool(joint.any(axis=0).all()),
                joint_tick_mask=joint, missed_ticks_by_scenario=(~joint).sum(axis=1),
                position_errors=errors['position'], yaw_errors=errors['yaw'],
                semantics='one fixed scenario per entire path, coupled position AND yaw')


def unique_windows(windows):
    seen = {}
    for window in windows:
        identity = (window['track_id'], window['seed'], window['input_identity'])
        fingerprint = canonical_hash(dict(snapshot=window['snapshot'],
                                          errors=window.get('mapping_interval_error')))
        if identity in seen:
            require(seen[identity] == fingerprint, 'same causal input has conflicting mapping evidence')
            continue
        require(not window['duplicate_of_prior_causal_input'], 'duplicate has no prior causal input')
        seen[identity] = fingerprint
        yield window


def calibrate_mapping(windows) -> dict:
    rows = []
    for window in unique_windows(windows):
        if window['index'] == 0:
            continue
        motion = window['snapshot']['mapping_motion']
        error = window.get('mapping_interval_error', {})
        values = np.asarray([error.get(k, np.nan) for k in ('right', 'forward', 'yaw_delta')], float)
        rows.append(dict(anchor_id=window['anchor_id'], index=window['index'],
                         input_identity=window['input_identity'], seed=window['seed'],
                         split=split(window['seed']), provenance=motion.get('provenance', 'unknown'),
                         valid=bool(motion.get('valid', False) and motion.get('accepted_for_correction', True)),
                         position_error=float(np.hypot(*values[:2])),
                         yaw_error=float(abs((values[2] + np.pi) % (2 * np.pi) - np.pi)),
                         old_position=motion.get('position_uncertainty'), old_yaw=motion.get('yaw_uncertainty')))
    by_provenance = {}
    for provenance in sorted(set(PROVENANCES) | {r['provenance'] for r in rows}):
        selected = [r for r in rows if r['split'] == 'CAL' and r['valid'] and r['provenance'] == provenance]
        errors = np.asarray([[r['position_error'], r['yaw_error']] for r in selected], float).reshape(-1, 2)
        supported = bool(len(selected) and np.isfinite(errors).all())
        maxima = errors.max(axis=0) if len(errors) else np.full(2, np.nan)
        by_provenance[provenance] = dict(supported=supported, n_cal=len(selected),
            n_cal_roads=len({r['seed'] for r in selected}), position_residual=maxima[0], yaw_residual=maxima[1],
            nonfinite_indices=np.argwhere(~np.isfinite(errors)))
    for row in rows:
        fit = by_provenance[row['provenance']]
        old = np.asarray([row['old_position'], row['old_yaw']], float)
        error = np.asarray([row['position_error'], row['yaw_error']], float)
        new = np.maximum(old, [fit['position_residual'], fit['yaw_residual']])
        usable = row['valid'] and np.isfinite(old).all() and (old >= 0).all() and np.isfinite(error).all()
        row.update(old_joint_contained=bool(usable and (error <= old).all()),
                   calibrated_joint_contained=bool(usable and fit['supported'] and (error <= new).all()),
                   calibrated_position=new[0], calibrated_yaw=new[1])
    summary = {}
    for role in ('CAL', 'HELD_OUT_FROM_FIT'):
        summary[role] = {}
        for provenance in by_provenance:
            selected = [r for r in rows if r['split'] == role and r['provenance'] == provenance]
            valid = [r for r in selected if r['valid']]
            summary[role][provenance] = dict(n_intervals=len(selected), n_valid=len(valid),
                n_roads=len({r['seed'] for r in selected}),
                old_joint_misses=sum(not r['old_joint_contained'] for r in valid),
                calibrated_joint_misses=sum(not r['calibrated_joint_contained'] for r in valid))
    return dict(by_provenance=by_provenance, intervals=rows, summary=summary,
                method='CAL per-interval Euclidean position/wrapped-yaw maxima by provenance; '
                       'externally max with each original bound, never change means or upgrade validity')


def inflate_mapping_motion(motion, calibration):
    result = deepcopy(motion)
    if not result.get('valid', False):
        return result
    fit = calibration['by_provenance'].get(result.get('provenance'))
    fields = ('position_uncertainty', 'yaw_uncertainty')
    old = np.asarray([result.get(k, np.nan) for k in fields], float)
    residual = np.asarray([fit.get(k) for k in ('position_residual', 'yaw_residual')], float) if fit else np.full(2, np.nan)
    if (not fit or not fit['supported'] or not np.isfinite(residual).all()
            or not np.isfinite(old).all() or (old < 0).any() or (residual < 0).any()):
        result.update(valid=False, calibration_reason='unsupported_mapping_calibration')
        return result
    result.update(zip(fields, np.maximum(old, residual)))
    return result


def interval_order(interval, supported=True):
    values = np.asarray(interval, float)
    if not supported or values.shape != (2,) or not np.isfinite(values).all():
        return 'ambiguous'
    require(values[0] <= values[1], 'reversed cost interval')
    return 'baseline' if values[0] > COST_FLOOR else 'alternative' if values[1] < -COST_FLOOR else 'ambiguous'


def paired_cost_errors(predicted, actual, supported) -> dict:
    """Preserve the shared reference axis when subtracting action costs."""
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    require(predicted.shape == (2, 19, 5) and actual.shape == (2, 5), 'paired reference costs shape')
    predicted_delta = predicted[1] - predicted[0]
    actual_delta = actual[1] - actual[0]
    error = actual_delta[None] - predicted_delta
    finite = bool(np.isfinite(error).all())
    return dict(supported=bool(supported), finite=finite, predicted_delta=predicted_delta,
                actual_delta=actual_delta, signed_error=error,
                central_midpoint_error=error[0, 0],
                maximum_absolute_error=float(np.max(np.abs(error))) if finite else None)


def calibrate_paired_cost(rows) -> dict:
    diagnostics, values = [], []
    bad_supported = False
    for row in rows:
        error = row['paired_error']
        reason = ('outside_fitting' if row['split'] != 'CAL' else
                  'unsupported_common_scene' if not error['supported'] else
                  'nonfinite_supported_error' if not error['finite'] else 'used')
        if row['split'] == 'CAL' and error['supported']:
            if error['finite']:
                values.append(error['maximum_absolute_error'])
            else:
                bad_supported = True
        diagnostics.append(dict(anchor_id=row['anchor_id'], seed=row['seed'], split=row['split'],
                                status=reason, **error))
    supported = bool(values and not bad_supported)
    return dict(supported=supported, paired_cost_residual=max(values) if supported else None,
                n_cal_used=len(values), n_cal_total=sum(r['split'] == 'CAL' for r in rows),
                cal_used_seeds=sorted({r['seed'] for r in diagnostics if r['status'] == 'used'}),
                diagnostic_rows=diagnostics,
                method='max_CAL absolute(actual shared-reference delta minus predicted '
                       'same-reference delta), across all nineteen states and five references; '
                       'unsupported rows excluded explicitly, supported nonfinite errors invalidate fitting')


def load_runtime(cache, project_root=ROOT):
    """Pin only the executed modules and cached predictor dependencies, not a tree inventory."""
    sources, pins = {}, {}
    for name in ('hud', 'motion', 'physics', 'observer', 'comparison', 'interval_comparison'):
        path = project_root / 'haic/algorithms/joint_control' / (name + '.py')
        blob = path.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        if name != 'interval_comparison':
            require(sha == cache['import_origins'][name]['sha256'], 'cached runtime source differs: ' + name)
        sources[name] = (path, blob)
        pins[str(path)] = sha
    pins[str(Path(__file__).resolve())] = digest(__file__)
    package_name = '_joint_interval_' + canonical_hash(pins)[:16]
    require(package_name not in sys.modules, 'fresh runtime namespace required')
    package = ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    modules = {}
    for name, (path, blob) in sources.items():
        module = ModuleType(package_name + '.' + name)
        module.__package__, module.__file__ = package_name, str(path)
        sys.modules[module.__name__] = module
        exec(compile(blob, str(path), 'exec'), module.__dict__)
        modules[name] = module
    return modules, pins


def load_prefixes(cache, run_dir):
    """Read just baseline observation arrays and road-only static identities."""
    prefixes, pins, seed_roads, road_roles = {}, {}, {}, {}
    for row in cache['anchors']:
        anchor = row['anchor']
        name = anchor['anchor_id']
        require(Path(name).name == name and name not in ('.', '..'), 'invalid anchor path')
        paths = {key: run_dir / (name + '-baseline') / key for key in ('data.npz', 'static.json')}
        for path in paths.values():
            sha = digest(path)
            require(sha == cache['input_sha256'].get(str(path)), 'cached prefix pin mismatch: ' + str(path))
            pins[str(path)] = sha
        with np.load(paths['data.npz'], allow_pickle=False) as archive:
            count = anchor['anchor_prefix_decisions'] + 1
            observations = archive['observations'][:count].copy()
            actions = archive['actions'][:count].copy()
        require(observations.shape == (count, 4, 84, 84) and observations.dtype == np.float32,
                'prefix observation shape/dtype')
        static = json.loads(paths['static.json'].read_text())
        road_sha = canonical_hash(static['track'])
        seed = anchor['seed']
        require(seed_roads.get(seed, road_sha) == road_sha, 'one seed has different road geometry')
        require(road_roles.get(road_sha, split(seed)) == split(seed), 'road hash crosses fitting split')
        seed_roads[seed], road_roles[road_sha] = road_sha, split(seed)
        prefixes[name] = dict(observations=observations, actions=actions, road_sha256=road_sha)
    return prefixes, pins, seed_roads


def scene_at(window, prefix, indexed, module, mapping=None):
    index = window['index']
    first = max(0, index - 3)
    motions = [indexed[i]['snapshot']['mapping_motion'] for i in range(first + 1, index + 1)]
    if mapping is not None:
        motions = [inflate_mapping_motion(m, mapping) for m in motions]
    return module.extract_scene(prefix['observations'][first:index + 1, -1], motions=motions)


def component_report(predicted, actual, weights, reference='midpoint'):
    output = {}
    for key, weight in weights.items():
        prediction = np.asarray(predicted['components'][key], float)
        observed = np.asarray(actual['components'][key], float)
        pdelta, adelta = prediction[1] - prediction[0], observed[1] - observed[0]
        output[key] = dict(weight=weight, reference=reference, predicted_delta=pdelta, actual_delta=adelta,
                           weighted_predicted_delta=weight * pdelta, weighted_actual_delta=weight * adelta)
    return output


def comparison_report(comparison, residual) -> dict:
    interval = np.asarray(comparison['delta_interval'], float).copy()
    calibration_valid = residual is not None and np.isfinite(residual) and residual >= 0
    if residual is not None and calibration_valid:
        allowance = float(residual)
        interval[1:] += [-allowance, allowance]
    else:
        interval[1:] = np.nan
    orders = [interval_order(v, bool(s) and calibration_valid)
              for v, s in zip(interval, comparison['cost_supported'])]
    eligible = [bool(order == 'alternative' and np.all(comparison['absolute_supported'][i])
                     and not np.any(comparison['veto'][i])) for i, order in enumerate(orders)]
    reasons = []
    for i, order in enumerate(orders):
        flags = set(comparison['abstain_reasons'][i])
        if order == 'ambiguous':
            flags.add('empirical_order_uncertain')
        if not calibration_valid:
            flags.add('paired_cost_calibration_missing')
        reasons.append(sorted(flags))
    return dict(cost_supported=comparison['cost_supported'], common_support=comparison['common_support'],
                supported_ticks=comparison['supported_ticks'], footprint_known_ticks=comparison['footprint_known_ticks'],
                reference_costs=comparison['reference_costs'],
                reference_delta=comparison['reference_delta'], unexpanded_delta_interval=comparison['delta_interval'],
                delta_interval=interval, order=orders[1], orders=orders,
                absolute_supported=comparison['absolute_supported'], veto=comparison['veto'],
                eligible_alternative=eligible[1], eligible_alternatives=eligible[1:], eligible_gain=any(eligible[1:]),
                calibration_valid=bool(calibration_valid), abstain_reasons=reasons)


def actual_report(scored):
    delta = np.asarray(scored['reference_costs'], float)[1] - scored['reference_costs'][0]
    supported = bool(scored['common_support'] and np.isfinite(delta).all())
    interval = [float(delta.min()), float(delta.max())] if supported else [None, None]
    return dict(common_support=supported, reference_costs=scored['reference_costs'],
                delta_by_reference=delta, delta_interval=interval, order=interval_order(interval, supported),
                supported_ticks=scored['supported_ticks'], footprint_known_ticks=scored['footprint_known_ticks'],
                components=scored['components'],
                road_clearance=scored['road_clearance'], obstacle_clearance=scored['obstacle_clearance'])


def outcome_report(row):
    diagnostics = row['privileged_catalog_diagnostics']
    progress = np.asarray([r['signed_centerline_progress'] for r in diagnostics], float)
    distance = np.asarray([r['centerline_distance'] for r in diagnostics], float)
    gaps = [np.asarray(r['hull_origin_circle_gaps'], float) for r in diagnostics]
    minimum = [float(g.min()) if g.size else math.inf for g in gaps]
    return dict(signed_centerline_progress=progress, progress_delta=progress[1, -1] - progress[0, -1],
                maximum_centerline_distance=distance.max(axis=1),
                minimum_hull_origin_circle_gap=minimum,
                circle_gap_delta=0. if minimum[0] == minimum[1] else minimum[1] - minimum[0],
                physical_labels=deepcopy(row['absolute']['labels']),
                caveat='Privileged outcomes only, never runtime references; circle gaps are not fixture clearance.')


def cached_support_ticks(row, y_scale):
    """Recover frozen reference tick masks, not a new old-cost reevaluation."""
    poses = np.asarray(row['comparison']['poses'], float)
    valid = np.isfinite(np.asarray(row['scene']['center_x'], float)[::-1])
    segment = np.cumsum(valid & ~np.r_[False, valid[:-1]])
    lo = np.clip(np.floor(np.clip(poses[..., 1] * y_scale + 20, 0, 83)).astype(int), 0, 82)
    reference = ((poses[..., 1] >= -20 / y_scale) & (poses[..., 1] <= 63 / y_scale)
                 & valid[lo] & valid[lo + 1] & (segment[lo] == segment[lo[..., :1]]))
    known = ~np.isnan(np.asarray(row['comparison']['road_clearance'], float))
    return dict(supported_ticks=(reference & known).sum(axis=-1), footprint_known_ticks=known.sum(axis=-1))


def analyze(cache, prefixes, modules):
    rows = cache['anchors']
    require(len(rows) == 8 and len({r['plan_anchor_id'] for r in rows}) == 8, 'exactly eight anchors required')
    require(not cache['global_integrity_errors'] and all(r['integrity']['complete'] for r in rows),
            'requires completed pinned cache; no broad integrity rerun')
    require({r['anchor']['seed'] for r in rows} == set(CAL + HELD_OUT_FROM_FIT), 'five fixed geometry seeds required')
    trajectory = calibrate_trajectories(rows)
    mapping = calibrate_mapping(cache['natural_windows'])
    indexed = {}
    for window in cache['natural_windows']:
        indexed.setdefault(window['anchor_id'], {})[window['index']] = window
    new, physics = modules['interval_comparison'], modules['physics']
    weights = dict(modules['comparison'].COST_WEIGHTS)
    if not trajectory['supported']:
        # Retain a diagnostic artifact rather than attempting a scorer that
        # rejects nonfinite poses, or silently replacing a failed fit with zero.
        calibration = dict(schema='haic-joint-temporal-interval-calibration-v1', calibration_complete=False,
            CAL=list(CAL), HELD_OUT_FROM_FIT=list(HELD_OUT_FROM_FIT),
            position_residual=trajectory['position_residual'], yaw_residual=trajectory['yaw_residual'],
            paired_cost_residual=None, mapping=mapping, trajectory=trajectory,
            paired_cost=dict(supported=False, n_cal_used=0, n_cal_total=sum(split(r['anchor']['seed']) == 'CAL' for r in rows),
                             reason='trajectory_calibration_invalid'),
            cost_floor=COST_FLOOR, cost_weights=weights, limitation=LIMITATION)
        report = dict(schema='haic-joint-temporal-interval-paired-analysis-v1',
            status='BLOCKED_TRAJECTORY_CALIBRATION_INVALID', calibration_complete=False,
            anchors=[dict(anchor_id=r['plan_anchor_id'], seed=r['anchor']['seed'], split=split(r['anchor']['seed']),
                          old_cost_supported=r['cost_supported'], new_cost_status='not_scored_invalid_calibration')
                     for r in rows], trajectory=trajectory, simulator_resets=0, pilot_executed=False,
            limitation=LIMITATION)
        return report, calibration
    stages, fit_rows = {}, []
    for row in rows:
        anchor = row['anchor']
        name, a = anchor['anchor_id'], anchor['anchor_prefix_decisions']
        window = indexed[name][a]
        hypotheses = physics.PhysicsState(**row['shared_hypotheses'])
        actual, predicted, oracle = trajectory_arrays(row)
        controls = np.asarray([v['controls'] for v in row['actual_trajectories']], float)
        actions = np.asarray(row['actions'], float)
        require(np.array_equal(actions[0], prefixes[name]['actions'][a]), 'nominal/prefix action mismatch')
        require(canonical_hash(window['hypotheses']) == row['shared_hypotheses_sha256'], 'cached hypothesis join')
        require(canonical_hash(window['snapshot']) == canonical_hash(row['observer_snapshot']), 'cached snapshot join')
        stage = {}
        for label, map_fit, pos, yaw in (
                ('scorer_only', None, POSITION_FLOOR, YAW_FLOOR),
                ('calibrated', mapping, trajectory['position_residual'] if trajectory['supported'] else None,
                 trajectory['yaw_residual'] if trajectory['supported'] else None)):
            scene = scene_at(window, prefixes[name], indexed[name], new, map_fit)
            comparison = new.compare_candidates(scene, hypotheses, actions, observer_valid=row['observer_valid'],
                position_residual=pos, yaw_residual=yaw, paired_cost_residual=0.)
            require(np.array_equal(comparison['poses'], predicted), 'runtime poses differ from pinned shared forecasts')
            observed = new.score_trajectories(scene, actual, controls)
            oracle_score = new.score_trajectories(scene, oracle, controls)
            error = paired_cost_errors(comparison['reference_costs'], observed['reference_costs'],
                                       comparison['common_support'] and observed['common_support'])
            stage[label] = dict(comparison=comparison, actual=actual_report(observed),
                oracle=actual_report(oracle_score), paired_error=error,
                components=component_report(comparison, observed, weights),
                scene=dict(history_used=scene.history_used, motion_provenance=scene.motion_provenance,
                           observed_interior_pixels=int(scene.road.sum()), known_pixels=int(scene.known.sum()),
                           uncertain_boundary_pixels=int(scene.uncertain_boundary.sum()),
                           unobserved_pixels=int(scene.unobserved.sum()), center_bounds=scene.center_bounds))
        stages[name] = stage
        fit_rows.append(dict(anchor_id=row['plan_anchor_id'], seed=anchor['seed'], split=split(anchor['seed']),
                             paired_error=stage['calibrated']['paired_error']))
    cost = calibrate_paired_cost(fit_rows)
    calibration = dict(schema='haic-joint-temporal-interval-calibration-v1',
        calibration_complete=bool(trajectory['supported'] and cost['supported']),
        CAL=list(CAL), HELD_OUT_FROM_FIT=list(HELD_OUT_FROM_FIT),
        position_residual=trajectory['position_residual'], yaw_residual=trajectory['yaw_residual'],
        paired_cost_residual=cost['paired_cost_residual'], mapping=mapping,
        trajectory=trajectory, paired_cost=cost, cost_floor=COST_FLOOR, cost_weights=weights,
        limitation=LIMITATION)
    output_rows = []
    for row in rows:
        anchor = row['anchor']
        actual, predicted, _ = trajectory_arrays(row)
        output: dict = dict(anchor_id=row['plan_anchor_id'], anchor=anchor, seed=anchor['seed'],
            split=split(anchor['seed']), observer_valid=row['observer_valid'],
            observer_invalid_reasons=row['observer_snapshot']['invalid_reasons'],
            road_sha256=prefixes[anchor['anchor_id']]['road_sha256'],
            actions=row['actions'], old=dict(cost_supported=row['cost_supported'],
                predicted_cost_supported=bool(np.all(row['comparison']['cost_supported'])),
                actual_cost_supported=row['actual_cost_supported'],
                predicted_delta_interval=row['comparison']['delta_interval'][1],
                predicted_order=interval_order(row['comparison']['delta_interval'][1],
                                               bool(np.all(row['comparison']['cost_supported']))),
                actual_delta=row['actual_delta_cost'], actual_order=interval_order([row['actual_delta_cost']] * 2,
                                                                               row['actual_cost_supported']),
                components=component_report(row['comparison'], row['actual_score'], weights, 'original_single'),
                predicted_delta=np.asarray(row['comparison']['delta'], float)[1],
                predicted_delta_error=np.asarray(row['comparison']['delta'], float)[1] -
                    np.asarray(row['actual_delta_cost'], float),
                **cached_support_ticks(row, modules['comparison'].Y_SCALE),
                cached_per_tick_union=row['absolute']['containment'],
                current_speed_tracking_form=row['old_tracking_form']),
            outcome=outcome_report(row), trajectory_containment={})
        for phase in ('scorer_only', 'calibrated'):
            phase_data = stages[anchor['anchor_id']][phase]
            comparison = comparison_report(phase_data['comparison'], 0. if phase == 'scorer_only' else cost['paired_cost_residual'])
            output[phase] = dict(phase_data, comparison=comparison)
            output[phase]['paired_supported'] = bool(comparison['common_support'] and phase_data['actual']['common_support'])
            output[phase]['sign_agreement'] = (comparison['order'] == phase_data['actual']['order']
                if comparison['order'] != 'ambiguous' and phase_data['actual']['order'] != 'ambiguous' else None)
            output[phase]['paired_error_contained'] = (bool(np.max(np.abs(phase_data['paired_error']['signed_error']))
                <= cost['paired_cost_residual']) if phase == 'calibrated' and cost['supported']
                and phase_data['paired_error']['supported'] and phase_data['paired_error']['finite'] else None)
        for label, p, y in (('original_floor', POSITION_FLOOR, YAW_FLOOR),
                            ('calibrated', trajectory['position_residual'], trajectory['yaw_residual'])):
            output['trajectory_containment'][label] = [whole_path_containment(a, pth, p, y)
                                                       for a, pth in zip(actual, predicted)]
        clear = np.all(output['calibrated']['comparison']['absolute_supported'], axis=1) & ~np.any(
            output['calibrated']['comparison']['veto'], axis=1)
        output['predicted_clear'] = clear
        output['predicted_clear_but_range_miss'] = [bool(ok and not c['contained']) for ok, c in
            zip(clear, output['trajectory_containment']['calibrated'])]
        output['actual_physical_failure'] = [not label['safe'] for label in row['absolute']['labels']]
        output_rows.append(output)
    # Coverage keeps the original causal identities and regime denominator, not
    # just anchors with a complete suffix or a supported post-outcome cost label.
    natural, coverage = [], {}
    for window in unique_windows(cache['natural_windows']):
        name = window['anchor_id']
        for domain, old in window['comparisons'].items():
            entry: dict = dict(anchor_id=name, index=window['index'], input_identity=window['input_identity'],
                         seed=window['seed'], split=split(window['seed']), domain=domain,
                         old_valid_supported=old['valid_supported'])
            for phase, map_fit, p, y in (
                    ('scorer_only', None, POSITION_FLOOR, YAW_FLOOR),
                    ('calibrated', mapping, trajectory['position_residual'] if trajectory['supported'] else None,
                     trajectory['yaw_residual'] if trajectory['supported'] else None)):
                scene = scene_at(window, prefixes[name], indexed[name], new, map_fit)
                compared = new.compare_candidates(scene, physics.PhysicsState(**window['hypotheses']), old['actions'],
                    observer_valid=window['snapshot']['valid'], position_residual=p, yaw_residual=y, paired_cost_residual=0.)
                result = comparison_report(compared, 0. if phase == 'scorer_only' else cost['paired_cost_residual'])
                entry[phase] = dict(valid_supported=bool(window['snapshot']['valid'] and old['effective']
                    and np.all(compared['cost_supported'])), comparison=result,
                    eligible_gain=bool(old['effective'] and result['eligible_gain']))
            natural.append(entry)
    for domain, old in cache['coverage'].items():
        selected = [r for r in natural if r['domain'] == domain]
        require(len(selected) == old['observed_applicable'], 'natural denominator changed')
        require(sum(r['old_valid_supported'] for r in selected) == old['valid_supported'], 'old coverage changed')
        coverage[domain] = dict(applicable=old['applicable'], observed_applicable=len(selected),
            unknown_possible_applicable=old['unknown_possible_applicable'], old_valid_supported=old['valid_supported'],
            **{phase:dict(valid_supported=sum(r[phase]['valid_supported'] for r in selected),
                         eligible_gain=sum(r[phase]['eligible_gain'] for r in selected))
               for phase in ('scorer_only', 'calibrated')})
    summary = {}
    for role in ('ALL', 'CAL', 'HELD_OUT_FROM_FIT'):
        selected = [r for r in output_rows if role == 'ALL' or r['split'] == role]
        summary[role] = dict(n_anchors=len(selected), n_paths=2 * len(selected),
            n_roads=len({r['seed'] for r in selected}),
            whole_path_misses=sum(not c['contained'] for r in selected for c in r['trajectory_containment']['calibrated']),
            central_path_misses=sum(not c['central_contained'] for r in selected for c in r['trajectory_containment']['calibrated']),
            **{phase:dict(support_overlap=dict(Counter(
                    'both' if r['old']['cost_supported'] and r[phase]['paired_supported'] else
                    'old_only' if r['old']['cost_supported'] else
                    'new_only' if r[phase]['paired_supported'] else 'neither' for r in selected)),
                predicted_orders=dict(Counter(r[phase]['comparison']['order'] for r in selected)),
                actual_orders=dict(Counter(r[phase]['actual']['order'] for r in selected)))
                for phase in ('scorer_only', 'calibrated')})
    report = dict(schema='haic-joint-temporal-interval-paired-analysis-v1', anchors=output_rows,
                  summary=summary, natural_windows=natural, coverage=coverage,
                  simulator_resets=0, pilot_executed=False, cost_weights=weights, cost_floor=COST_FLOOR,
                  limitation=LIMITATION, calibration_complete=calibration['calibration_complete'])
    report['by_geometry'] = {str(seed): dict(split=split(seed),
        n_paths=2 * sum(r['seed'] == seed for r in output_rows),
        whole_path_misses=sum(not c['contained'] for r in output_rows if r['seed'] == seed
                              for c in r['trajectory_containment']['calibrated']),
        anchor_ids=[r['anchor_id'] for r in output_rows if r['seed'] == seed]) for seed in CAL + HELD_OUT_FROM_FIT}
    return report, calibration


def console_summary(report, calibration):
    if report.get('status') == 'BLOCKED_TRAJECTORY_CALIBRATION_INVALID':
        return ('BLOCKED: nonfinite or nonzero-origin CAL trajectory evidence; diagnostics retained, '
                'no zero residual substituted, no paired cost or simulator execution.\n' + LIMITATION + '\n')
    lines = ['Consumed TRAIN interval replay; no simulator execution.',
             'anchor split old/scorer/cal support  actual->predicted  GT-progress-delta']
    for row in report['anchors']:
        support = '/'.join(str(int(v)) for v in (row['old']['cost_supported'],
            row['scorer_only']['paired_supported'], row['calibrated']['paired_supported']))
        lines.append(f"{row['anchor_id']:9s} {row['split']:17s} {support}  "
                     f"{row['calibrated']['actual']['order']}->{row['calibrated']['comparison']['order']}  "
                     f"{row['outcome']['progress_delta']:+.6f}")
    held = report['summary']['HELD_OUT_FROM_FIT']
    lines.append(f"Held-out-from-fitting whole-path misses: {held['whole_path_misses']}/{held['n_paths']}; "
                 f"central misses: {held['central_path_misses']}/{held['n_paths']}.")
    lines.append(f"CAL paired-cost residual: {calibration['paired_cost_residual']}; "
                 f"used {calibration['paired_cost']['n_cal_used']}/{calibration['paired_cost']['n_cal_total']} anchors; "
                 f"calibration_complete={calibration['calibration_complete']}.")
    lines.append(LIMITATION)
    return '\n'.join(lines) + '\n'


def write_outputs(directory, report, calibration):
    directory = Path(directory)
    require(directory.parent.is_dir(), 'output parent must already exist')
    if not directory.exists():
        directory.mkdir()
    require(directory.is_dir() and not any((directory / n).exists() for n in OUTPUTS),
            'exclusive new output files required; existing evidence is never overwritten')
    summary = console_summary(report, calibration)
    # Exclusive opens also protect against another process racing the precheck.
    # A partial write is retained; no cleanup erases evidence of a failed run.
    for name, value in zip(OUTPUTS, (report, calibration, summary)):
        with (directory / name).open('x') as stream:
            if isinstance(value, str):
                stream.write(value)
            else:
                json.dump(jsonable(value), stream, indent=2, allow_nan=False)
                stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'runs/joint-temporal-v1/analysis.json')
    parser.add_argument('--analysis-sha256', default=PRIMARY_SHA256)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/joint-temporal-interval-v1')
    args = parser.parse_args(argv)
    source = args.analysis.resolve()
    require(not any((args.output_dir / n).exists() for n in OUTPUTS), 'output evidence already exists')
    require(digest(source) == args.analysis_sha256, 'primary analysis pin mismatch')
    cache = json.loads(source.read_text())
    require(cache['schema'] == 'haic-joint-temporal-analysis-v1', 'wrong primary schema')
    modules, source_pins = load_runtime(cache)
    import cv2
    cv2.setNumThreads(1)
    prefixes, input_pins, seed_roads = load_prefixes(cache, source.parent)
    input_pins[str(source)] = args.analysis_sha256
    report, calibration = analyze(cache, prefixes, modules)
    metadata = dict(source_pins=source_pins, input_pins=input_pins, road_only_sha256=seed_roads,
        runtime=dict(executable=str(Path(sys.executable).resolve()),
                     executable_sha256=digest(Path(sys.executable).resolve()), python=sys.version,
                     numpy=np.__version__, cv2=getattr(cv2, '__version__'), cv2_threads=cv2.getNumThreads()),
        reproduction=['python', '-B', '-m', 'scripts.analyze_joint_temporal_intervals',
                      '--analysis', str(source), '--analysis-sha256', args.analysis_sha256,
                      '--output-dir', str(args.output_dir.resolve())])
    for path, sha in {**source_pins, **input_pins}.items():
        require(digest(path) == sha, 'targeted input/source changed during replay: ' + path)
    report.update(metadata)
    calibration.update(metadata)
    print(write_outputs(args.output_dir, report, calibration), end='')


if __name__ == '__main__':
    main()
