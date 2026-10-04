"""Passive, fail-closed four-cell magnitude comparison; never executes an Agent."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import evaluate_koi_avoidance_magnitude as evaluate
from scripts import analyze_koi_nominal_trajectory as nominal


shield = nominal.shield
METRIC_NAMES = nominal.METRIC_NAMES
steering_window, eligible = nominal.steering_window, nominal.eligible


def validate_wrapper_phases(episode, states):
    """Raw ticks precede CarEnvironment's once-per-decision damage/counter update."""
    by_step = {}
    for state in states[1:]:
        by_step.setdefault(state['step'], []).append(state)
    fields = {'off_track_counter', 'damage', 'reward', 'prev_reward', 'tile_visited_count', 'fuel_spent'}
    for state in states:
        value = state.get('environment_state')
        if not isinstance(value, dict) or not fields <= set(value):
            raise ValueError('wrapper environment telemetry missing')
        if (any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) for k in fields)
                or type(value['off_track_counter']) is not int or value['off_track_counter'] < 0
                or type(value['tile_visited_count']) is not int or value['tile_visited_count'] < 0
                or not 0 <= value['damage'] <= 1):
            raise ValueError('invalid wrapper environment telemetry')
    initial = states[0]['environment_state']
    if initial['damage'] != 0 or initial['off_track_counter'] != 0:
        raise ValueError('wrapper reset-end damage/counter differs')
    counters = []
    for index, row in enumerate(episode['decision_trace'], 1):
        observed = row['controller']['evaluation_only']
        pre, post = observed['pre']['environment_state'], observed['post']['environment_state']
        if not fields <= set(pre) or not fields <= set(post):
            raise ValueError('wrapper decision telemetry missing')
        ticks = by_step.get(index, [])
        if not ticks:
            raise ValueError('wrapper raw decision coverage missing')
        if any(s['environment_state']['damage'] != pre['damage'] or
               s['environment_state']['off_track_counter'] != pre['off_track_counter'] for s in ticks):
            raise ValueError('raw wrapper pre-update damage/counter differs')
        endpoint = ticks[-1]['environment_state']
        if any(endpoint[k] != post[k] for k in fields - {'damage', 'off_track_counter'}):
            raise ValueError('raw/decision environment state differs')
        # CarRacing substitutes -100 on a playfield terminal tick, instead of
        # changing its cumulative reward. Sum in the wrapper's original order.
        previous, rewards = pre['prev_reward'], []
        for state in ticks:
            value = state['environment_state']
            rewards.append(value['reward'] - previous)
            previous = value['prev_reward']
        if type(observed.get('simulator_terminated')) is not bool:
            raise ValueError('simulator terminal telemetry missing')
        if observed['simulator_terminated']:
            rewards[-1] = -100.
        reward = sum(rewards)
        counter = pre['off_track_counter'] + 1 if reward < 0 else 0
        damage = min(1., pre['damage'] + .2) if row['collision'] else pre['damage']
        if (observed.get('decision_reward') != reward or observed.get('off_track_count_before') != pre['off_track_counter']
                or observed.get('off_track_count_after') != counter or post['off_track_counter'] != counter):
            raise ValueError('wrapper reward/counter phase transition differs')
        if (post['damage'] != damage or row.get('damage') != damage
                or row.get('damage_telemetry_valid') is not True or row.get('collision_telemetry_valid') is not True):
            raise ValueError('wrapper damage transition/decision telemetry differs')
        if observed.get('wrapper_crashed') is not (damage >= 1.) or observed.get('wrapper_off_track') is not (counter > 100):
            raise ValueError('wrapper retirement predicates differ')
        counters.append(counter)
    final = episode['decision_trace'][-1]['controller']['evaluation_only']['post']['environment_state']
    if episode['damage'] != final['damage']:
        raise ValueError('headline damage differs from final wrapper damage')
    if episode.get('off_track_count_max', max(counters)) != max(counters):
        raise ValueError('headline wrapper counter maximum differs')


def treatment_metrics(trace, mode):
    """Decompose actual taps, not a self-reported activity flag or shadow action."""
    if mode not in evaluate.ARMS:
        raise ValueError('unknown magnitude arm')
    errors, changes, reasons = [], [], Counter()
    nominal_changes = final_changes = overlaps = cancellations = shield_changes = 0
    changed_duration = 0.
    for index, row in enumerate(trace, 1):
        c = row['controller']
        original, changed, issued = c['crossing_probe_action'], c['baseline_probe_action'], shield.action(row)
        if any(not isinstance(a, list) or len(a) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in a)
               for a in (original, changed, issued)):
            raise ValueError('invalid magnitude action tap')
        if any(not -1 <= a[0] <= 1 or not 0 <= a[1] <= 1 or not 0 <= a[2] <= 1 for a in (original, changed, issued)):
            raise ValueError('out-of-range magnitude action tap')
        active, override, final = changed != original, issued != changed, issued != original
        nominal_changes += active
        shield_changes += override
        final_changes += final
        overlaps += active and override
        cancellations += active and not final
        if (any(type(c.get(k)) is not int for k in ('nominal_probe_calls', 'crossing_probe_calls'))
                or c.get('nominal_probe_calls') != index or c.get('crossing_probe_calls') != index):
            errors.append(dict(step=index, error='once-per-decision call coverage'))
        if type(c.get('executed_action_observations')) is not int or c.get('executed_action_observations') != 0 or c.get('candidate_feedback_method') is not False:
            errors.append(dict(step=index, error='stateless candidate received executed-action feedback'))
        if c.get('candidate_driver_hidden') is not True:
            errors.append(dict(step=index, error='candidate driver could be unwrapped'))
        if original[1:] != changed[1:] or issued[1:] != original[1:]:
            errors.append(dict(step=index, error='magnitude/shield pedals differ'))
        if c['shield']['baseline_action'] != changed:
            errors.append(dict(step=index, error='magnitude action not shield input'))
        if mode == evaluate.ARMS[0]:
            if active or c.get('candidate_state_keys') != [] or any(k.startswith('magnitude_') for k in c):
                errors.append(dict(step=index, error='baseline has magnitude treatment'))
        else:
            if c.get('candidate_state_keys') != list(evaluate.STATE_KEYS):
                errors.append(dict(step=index, error='new plan/recovery/actuator state'))
            if (c.get('magnitude_baseline_action') != original or type(c.get('magnitude_changed')) is not bool
                    or c['magnitude_changed'] != active or not isinstance(c.get('magnitude_reason'), str)
                    or type(c.get('final_steer')) not in (int, float) or c.get('final_steer') != changed[0]):
                errors.append(dict(step=index, error='magnitude diagnostics differ from taps'))
            proposal = c.get('magnitude_proposal')
            valid = proposal is None and not active
            if isinstance(proposal, dict) and set(evaluate.PROPOSAL_FIELDS) <= set(proposal):
                box = proposal['current_bbox_m']
                numeric = [proposal[k] for k in evaluate.PROPOSAL_FIELDS if k != 'current_bbox_m']
                valid = (all(type(v) in (int, float) and math.isfinite(v) for v in numeric)
                    and isinstance(box, list) and len(box) == 4
                    and all(type(v) in (int, float) and math.isfinite(v) for v in box))
                if valid:
                    valid = (box[0] < box[2] and box[1] < box[3] and proposal['side'] in (-1, 1)
                        and 0 <= proposal['collision_risk'] <= 1
                        and 0 < proposal['legacy_magnitude'] <= .55 and 0 <= proposal['magnitude'] <= proposal['legacy_magnitude']
                        and all(proposal[k] >= 0 for k in ('required_clearance_m', 'approach_m', 'ttc_s', 'geometric_demand')))
            if not valid:
                errors.append(dict(step=index, error='missing/invalid magnitude geometry-risk proposal'))
            reason = c.get('magnitude_reason')
            if isinstance(reason, str):
                reasons[reason] += 1
        if active:
            duration = c['evaluation_only']['post']['t'] - c['evaluation_only']['pre']['t']
            changed_duration += duration
            changes.append(dict(step=index, crossing_action=original, magnitude_action=changed, issued_action=issued,
                magnitude_delta=changed[0] - original[0], shield_delta=issued[0] - changed[0],
                final_delta=issued[0] - original[0], shield_override=override,
                reason=c.get('magnitude_reason'), proposal=c.get('magnitude_proposal')))
    return dict(decisions=len(trace), magnitude_changed_decisions=nominal_changes,
        magnitude_changed_duration_s=changed_duration, shield_override_decisions=shield_changes,
        final_vs_crossing_changed_decisions=final_changes, magnitude_shield_overlap_decisions=overlaps,
        magnitude_cancelled_by_shield_decisions=cancellations, reasons=dict(reasons), changes=changes,
        contract_errors=errors, contract_valid=not errors)


def episode_metrics(episode, states):
    if not evaluate.valid_episode(episode) or (episode['track_id'], episode['seed']) not in evaluate.CELLS:
        raise ValueError('invalid magnitude episode/cell')
    # The old pure validator checks all raw ticks, all six footprint objects,
    # exact held commands and returns. New modes deliberately have zero feedback.
    result = nominal.episode_metrics(episode, states)
    validate_wrapper_phases(episode, states)
    result['treatment'] = treatment_metrics(episode['decision_trace'], episode['mode'])
    return result


def pair_metrics(b, c, bs, cs):
    if ((b['track_id'], b['seed']) != (c['track_id'], c['seed'])
            or (b['track_id'], b['seed']) not in evaluate.CELLS
            or b['mode'] != evaluate.ARMS[0] or c['mode'] != evaluate.ARMS[1]
            or not evaluate.valid_episode(b) or not evaluate.valid_episode(c)):
        raise ValueError('magnitude pair identity differs')
    result = nominal.pair_metrics(b, c, bs, cs)
    result['subgroup'] = 'ordinary_controls'
    result['road_geometry_sha256'] = hashlib.sha256(json.dumps(b['catalog']['track'], sort_keys=True).encode()).hexdigest()
    for arm, episode in (('baseline', b), ('candidate', c)):
        states = bs if arm == 'baseline' else cs
        validate_wrapper_phases(episode, states)
        result[arm]['treatment'] = treatment_metrics(episode['decision_trace'], episode['mode'])
    return result


def summarize(cells, complete):
    gate = evaluate.GATE
    ids = [(c['track_id'], c['seed']) for c in cells]
    geometry_hashes = {}
    for cell in cells:
        geometry_hashes.setdefault(str(cell['seed']), set()).add(cell['road_geometry_sha256'])
    geometry_consistent = (set(geometry_hashes) == {'3184000013', '3184000015'}
        and all(len(v) == 1 for v in geometry_hashes.values())
        and len(set().union(*geometry_hashes.values())) == 2)
    objects_complete = bool(cells) and all([o['obstacle_id'] for o in c[a]['objects']] == list(range(6))
        for c in cells for a in ('baseline', 'candidate'))
    kept = [c for c in cells if c['baseline']['finished'] and c['candidate']['finished']]
    required_windows = [w for c in cells for w in c['windows'] if w['baseline_eligible']]
    required_returns = [r for c in cells for r in c['returns'] if r['baseline_required']]
    comparable = [r for r in required_returns if r['comparable']]
    ordinary = [c for c in cells if c['subgroup'] == 'ordinary_controls']
    fractional, geometry = {n: [] for n in METRIC_NAMES}, {n: {} for n in METRIC_NAMES}
    denominators, cell_means = [], []
    denominator_valid = {n: True for n in METRIC_NAMES}
    for cell in ordinary:
        selected = [w for w in cell['windows'] if w['baseline_eligible']]
        coverage, means = {}, {}
        for name in METRIC_NAMES:
            values = [w['metrics'][name]['relative'] for w in selected
                      if name in w['metrics'] and w['metrics'][name]['relative'] is not None]
            valid = bool(selected) and len(values) == len(selected)
            denominator_valid[name] &= valid
            coverage[name] = dict(required=len(selected), usable=len(values), valid=valid)
            means[name] = math.fsum(values) / len(values) if valid else None
            if valid:
                fractional[name].append(means[name])
                geometry[name].setdefault(cell['seed'], []).append(means[name])
        denominators.append(dict(track_id=cell['track_id'], seed=cell['seed'], metrics=coverage))
        cell_means.append(dict(track_id=cell['track_id'], seed=cell['seed'], relative=means))
    equal_cell = {n: math.fsum(v) / len(v) if v else None for n, v in fractional.items()}
    roads = {n: {str(s): math.fsum(v) / len(v) for s, v in g.items()} for n, g in geometry.items()}
    laps = [c['candidate']['lap_ms'] - c['baseline']['lap_ms'] for c in kept]
    new_hits = [dict(track_id=c['track_id'], seed=c['seed'], obstacle_id=bo['obstacle_id'])
        for c in cells for bo, co in zip(c['baseline']['objects'], c['candidate']['objects']) if not bo['hit'] and co['hit']]
    return_censors = {a: sum(r['common'][a]['status'] != 'returned' for r in comparable) for a in ('baseline', 'candidate')}
    return_bounds = {a: math.fsum(r['common'][a]['restricted_response_s'] for r in comparable) for a in ('baseline', 'candidate')}
    variation_complete = bool(required_windows) and all('steering_variation' in w['metrics'] for w in required_windows)
    variation_delta = math.fsum(w['metrics'].get('steering_variation', {}).get('delta', 0) for w in required_windows)
    checks = dict(
        complete_matched_evidence=complete and len(ids) == len(set(ids)) == 4 and set(ids) == set(evaluate.CELLS)
            and objects_complete and all(c['parity']['matched'] for c in cells),
        ordinary_geometry_consistency=geometry_consistent,
        no_lost_finish=bool(cells) and all(not c['baseline']['finished'] or c['candidate']['finished'] for c in cells),
        per_cell_damage_collision_nonincrease=bool(cells) and all(c['candidate']['damage'] <= c['baseline']['damage']
            and c['candidate']['collision_decisions'] <= c['baseline']['collision_decisions']
            and c['candidate']['physical']['physical_contact_events'] <= c['baseline']['physical']['physical_contact_events'] for c in cells),
        no_new_hit=objects_complete and not new_hits and not any(c[a]['unassociated_collision'] for c in cells for a in ('baseline', 'candidate')),
        unchanged_shield_contract=bool(cells) and all(c[a]['shield']['contract_valid'] for c in cells for a in ('baseline', 'candidate')),
        stateless_magnitude_action_decomposition=bool(cells) and all(c[a]['treatment']['contract_valid']
            and not c[a]['nominal_contract_errors'] for c in cells for a in ('baseline', 'candidate')),
        baseline_windows_retained=bool(required_windows) and all(w['candidate_eligible'] for w in required_windows),
        return_coverage_retained=bool(required_returns) and len(comparable) == len(required_returns),
        return_censors_nonincrease=bool(comparable) and return_censors['candidate'] <= return_censors['baseline'],
        common_return_bound_nonincrease=bool(comparable) and return_bounds['candidate'] <= return_bounds['baseline'],
        steering_variation_nonincrease=variation_complete and variation_delta <= 0,
        every_kept_lap_within_20ms=bool(laps) and all(v <= gate['max_kept_lap_increase_ms'] for v in laps),
        matched_lap_mean_reduction_20ms=bool(laps) and math.fsum(laps) / len(laps) <= -gate['mean_lap_reduction_ms'])
    for name, threshold in (('max_lateral', gate['ordinary_max_lateral_fraction']),
                            ('path', gate['ordinary_path_fraction']),
                            ('steering_integral', gate['ordinary_steering_integral_fraction'])):
        mean = equal_cell[name]
        checks['ordinary_' + name + '_reduction'] = (len(ordinary) == 4 and denominator_valid[name]
            and len(fractional[name]) == 4 and mean is not None and mean <= threshold
            and len(roads[name]) == gate['improving_ordinary_geometry_seeds'] and all(v < 0 for v in roads[name].values()))
    arms = {}
    for arm in ('baseline', 'candidate'):
        rows = [c[arm] for c in cells]
        arms[arm] = dict(episodes=len(rows), safety_objects=sum(len(r['objects']) for r in rows),
            finishes=sum(r['finished'] for r in rows), damage=math.fsum(r['damage'] for r in rows),
            collision_decisions=sum(r['collision_decisions'] for r in rows),
            physical_contact_events=sum(r['physical']['physical_contact_events'] for r in rows),
            hit_objects=sum(o['hit'] for r in rows for o in r['objects']),
            decisions=sum(r['treatment']['decisions'] for r in rows),
            shield_intervention_decisions=sum(r['shield']['intervention_decisions'] for r in rows),
            magnitude_changed_decisions=sum(r['treatment']['magnitude_changed_decisions'] for r in rows),
            final_vs_crossing_changed_decisions=sum(r['treatment']['final_vs_crossing_changed_decisions'] for r in rows),
            magnitude_shield_overlap_decisions=sum(r['treatment']['magnitude_shield_overlap_decisions'] for r in rows),
            magnitude_cancelled_by_shield_decisions=sum(r['treatment']['magnitude_cancelled_by_shield_decisions'] for r in rows),
            prospective_return_statuses=dict(Counter(o['prospective_return']['status'] for r in rows for o in r['objects'])))
    return dict(checks=checks, gate_passed=all(checks.values()), new_hits=new_hits, aggregate_arms=arms,
        ordinary_equal_cell_relative=equal_cell, ordinary_equal_geometry_relative=roads, ordinary_cell_relative=cell_means,
        ordinary_road_geometry_sha256={s: sorted(v) for s, v in geometry_hashes.items()},
        metric_denominators_valid=denominator_valid, metric_denominator_coverage=denominators,
        baseline_eligible_windows=len(required_windows), baseline_required_returns=len(required_returns),
        common_return_cohort_n=len(comparable), common_return_censor_counts=return_censors,
        common_return_descriptive_bound_sums_s=return_bounds, fixed_window_steering_variation_delta=variation_delta,
        mutually_finished_lap_pairs=len(laps), mutually_finished_lap_deltas_ms=laps,
        mean_lap_delta_ms=math.fsum(laps) / len(laps) if laps else None,
        finishes={a: sum(c[a]['finished'] for c in cells) for a in ('baseline', 'candidate')},
        lost_finishes=sum(c['baseline']['finished'] and not c['candidate']['finished'] for c in cells),
        gained_finishes=sum(c['candidate']['finished'] and not c['baseline']['finished'] for c in cells))


def validate_artifacts(directory, pins, actual):
    if len(pins) != len({p['file'] for p in pins}) or sorted(p['file'] for p in pins) != sorted(actual):
        raise ValueError('evidence inventory differs')
    for pin in pins:
        path = evaluate.local_path(directory, pin['file'])
        if evaluate.sha(path) != pin['sha256'] or path.stat().st_size != pin['bytes']:
            raise ValueError('evidence hash/size differs')


def analyze(directory, protocol_sha256):
    directory = Path(directory).resolve()
    protocol = evaluate.validate_frozen(directory, protocol_sha256)
    if evaluate.sha(__file__) != protocol['source_inventory']['scripts/analyze_koi_avoidance_magnitude.py']['sha256']:
        raise ValueError('analyzer differs from frozen source')
    report = evaluate.read_json(directory / 'episode-report.json')
    if report['protocol_sha256'] != protocol_sha256 or len(report['rows']) != 8:
        raise ValueError('report/protocol differs')
    operator_names = ('run-start.json', 'preflight.json', 'reset-ledger.jsonl', 'resource-receipts.jsonl')
    validate_artifacts(directory, report['operator_artifacts'], [n for n in operator_names if (directory / n).exists()])
    start = evaluate.read_json(directory / 'run-start.json')
    preflight = evaluate.read_json(directory / 'preflight.json')
    if start['protocol_sha256'] != protocol_sha256 or start['preflight_sha256'] != evaluate.sha(directory / 'preflight.json'):
        raise ValueError('run-start/import-only binding differs')
    evaluate.validate_preflight(preflight, protocol_sha256, protocol['resource_plan']['forecast'])
    ledger = shield.validate_ledger(directory, report['rows'], protocol_sha256)
    ledger_path = directory / 'reset-ledger.jsonl'
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
    counts = Counter(e['file'] for e in events if e['event'] == 'reset_intent')
    if any(n != 1 for n in counts.values()):
        raise ValueError('automatic reset/retry evidence')
    episodes, metrics = {}, {}
    for index, (row, expected) in enumerate(zip(report['rows'], protocol['schedule'])):
        if any(row[k] != expected[k] for k in ('track_id', 'seed', 'mode', 'repeat', 'file')):
            raise ValueError('scheduled slot differs')
        actual = sorted(p.name for p in directory.glob(Path(row['file']).stem + '.*'))
        if row['status'] in ('completed', 'operator_error'):
            validate_artifacts(directory, row['artifacts'] if row['status'] == 'completed' else row['partial_artifacts'], actual)
            process_path = (directory / row['file']).with_suffix('.process.json')
            if process_path.exists():
                process = evaluate.read_json(process_path)
                if (process['protocol_sha256'] != protocol_sha256 or process['command'] != evaluate.worker_command(
                        directory, index, protocol_sha256, protocol['python'])):
                    raise ValueError('process command/protocol differs')
            if row['status'] == 'completed':
                required = {Path(row['file']).with_suffix(s).name for s in
                    ('.json', '.raw.jsonl', '.decisions.jsonl', '.process.json', '.stdout.txt', '.stderr.txt', '.resources.jsonl')}
                if not required <= set(actual):
                    raise ValueError('completed primary/log/resource evidence missing')
                key = row['track_id'], row['seed'], row['mode']
                episodes[key] = shield.load_episode(directory, row, protocol_sha256)
                episode = episodes[key][0]
                if episode['versions']['numpy'] != '1.26.0' or episode['versions']['cv2'] != '4.8.1' or episode['versions']['torch'] != '2.1.0+cpu':
                    raise ValueError('episode CPU21 dependencies differ')
                metrics[key] = episode_metrics(*episodes[key])
        elif row['status'] != 'unrun' or actual:
            raise ValueError('ambiguous running/unreported partial slot')
    cells, coverage, unmatched = [], [], []
    for track, seed in evaluate.CELLS:
        keys = [(track, seed, a) for a in evaluate.ARMS]
        matched = all(k in episodes for k in keys)
        coverage.append(dict(track_id=track, seed=seed, subgroup='ordinary_controls', matched=matched,
            arms={a: next(r['status'] for r in report['rows'] if (r['track_id'], r['seed'], r['mode']) == (track, seed, a)) for a in evaluate.ARMS}))
        if matched:
            (b, bs), (c, cs) = (episodes[k] for k in keys)
            cells.append(pair_metrics(b, c, bs, cs))
        else:
            unmatched.extend(dict(track_id=track, seed=seed, mode=k[2], metrics=metrics[k]) for k in keys if k in metrics)
    complete = report['operator_error'] is None and len(episodes) == 8 and len(cells) == 4
    summary = summarize(cells, complete)
    result: dict[str, Any] = dict(schema=evaluate.SCHEMA + '-result', protocol_sha256=protocol_sha256,
        report_sha256=evaluate.sha(directory / 'episode-report.json'), complete=complete,
        planned_episodes=8, valid_episodes=len(episodes), matched_pairs=len(cells),
        unmatched_valid_episodes=len(unmatched), unmatched_episodes=unmatched, cell_coverage=coverage,
        operator_error=report['operator_error'], baseline_sha256=evaluate.BASELINE_SHA, shield_sha256=evaluate.SHIELD_SHA,
        scope=protocol['scope'], passive_analysis=True, environment_resets=0, official_action=False,
        safety_objects_per_arm=6 * len(cells), planned_safety_objects_per_arm=24,
        observed_safety_objects_per_arm={a: sum(len(m['objects']) for k, m in metrics.items() if k[2] == a) for a in evaluate.ARMS},
        geometry_seeds=len({c['seed'] for c in cells}), planned_geometry_seeds=2,
        verdict='PASSED_CONSUMED_DEVELOPMENT_GATE' if summary['gate_passed'] else 'REJECTED',
        adopted=False, repeat_authorized=False, tuning_authorized=False,
        source_integrity=dict(verified=True, helper_files=len(protocol['source_inventory']),
                             environment_files=len(protocol['environment_sha256']), model_members=len(protocol['model_source_sha256'])),
        ledger=ledger, metric_definitions=protocol['metric_definitions'], gate=protocol['gate'], cells=cells,
        limitations=['Outcome-selected consumed TRAIN on four layouts/two roads, not fresh validation or official ranking.',
            'All natural DNFs, all six objects/cell and missing/partial evidence remain in scope; short failed paths are not efficiency credit.',
            'Prospective return uses sampled rear-clear and .24s confirmation. Common-horizon bound mixes onset delays and censored lower bounds; not an unbiased mean, KM or RMST.',
            'Station-window steering variation counts issued-command jumps strictly inside each window, with no artificial endpoint jumps. Full-episode steering is also retained per cell.',
            'Sampled fixture clearance is not continuous-time safety. Prefix parity covers logged states/pixels, not hidden Box2D solver serialization.',
            'Existing v1 administrative state remains unchanged; no new controller/plan/feedback. Passing does not authorize tuning, repeats, adoption or official action.'], **summary)
    shield.finite_tree(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.protocol_sha256)
    evaluate.save(args.output, result)
    print(json.dumps({k: result[k] for k in ('verdict', 'complete', 'checks', 'mean_lap_delta_ms')}))


if __name__ == '__main__':
    main()
