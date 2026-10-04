"""Passive strict comparison of the eight frozen consumed-TRAIN nominal pairs."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import evaluate_koi_nominal_trajectory as evaluate
from scripts import analyze_koi_collision_shield as shield
from scripts import analyze_koi_minimum_clearance_ab as windows
from scripts import analyze_koi_adaptive_ab as physical
from scripts import analyze_koi_steering_release_ab as returns


METRIC_NAMES = ('max_lateral', 'path', 'steering_integral', 'steering_variation')


def steering_window(trace, start, end):
    result: dict[str, Any] = windows.steering_window(trace, start, end)
    if result['status'] == 'complete':
        result['variation'] = math.fsum(abs(right['steer'] - left['steer'])
            for left, right in zip(trace, trace[1:])
            if start < right['controller']['evaluation_only']['pre']['t'] < end)
    return result


def episode_metrics(episode, states):
    shield.finite_tree(episode)
    shield.finite_tree(states)
    if not evaluate.valid_episode(episode):
        raise ValueError('invalid episode')
    catalog, trace = episode['catalog'], episode['decision_trace']
    if (len(catalog['obstacles']) != 6 or [o['id'] for o in catalog['obstacles']] != list(range(6))
            or hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest() != episode['geometry_sha256']):
        raise ValueError('geometry/object inventory differs')
    raw = states[1:]
    for state in states:
        if any(len(state[key]) != 6 for key in ('front', 'rear', 'clearance')):
            raise ValueError('raw object arrays differ')
    dt = [b['t'] - a['t'] for a, b in zip(states, states[1:])]
    if not dt or any(abs(delta - .02) > 1e-8 for delta in dt):
        raise ValueError('raw tick continuity differs')
    measured = evaluate.shield.legacy.physical_metrics(states, catalog['obstacles'])
    if measured != episode['physical']:
        raise ValueError('stored physical metrics differ')
    if episode['collisions'] != sum(d['collision'] for d in trace):
        raise ValueError('collision denominator differs')
    contract = shield.shield_metrics(trace)
    nominal_errors = []
    by_step = {}
    for state in raw:
        if type(state['collision']) is not bool:
            raise ValueError('raw collision flag differs')
        by_step.setdefault(state['step'], []).append(state)
    for index, decision in enumerate(trace, 1):
        controller = decision['controller']
        if (type(decision['collision']) is not bool or not -1 <= decision['steer'] <= 1
                or not 0 <= decision['gas'] <= 1 or not 0 <= decision['brake'] <= 1):
            raise ValueError('invalid issued action/collision')
        ticks = by_step.get(index, [])
        if not 1 <= len(ticks) <= 4 or decision['collision'] is not any(s['collision'] for s in ticks):
            raise ValueError('raw/decision collision coverage differs')
        endpoint = ticks[-1]
        if any(controller['evaluation_only']['post'][k] != endpoint[k] for k in (
                't', 'x', 'y', 'yaw', 'speed', 'station', 'lateral', 'heading_error', 'road_index',
                'front', 'rear', 'clearance', 'contacts', 'wheel_road_contacts', 'dynamic_state', 'wheel_state')):
            raise ValueError('raw/decision endpoint differs')
        for field in ('baseline_probe_action', 'crossing_probe_action'):
            value = controller[field]
            if not isinstance(value, list) or len(value) != 3 or any(type(v) not in (int, float) for v in value):
                raise ValueError('invalid independent nominal tap')
        if controller['nominal_probe_calls'] != index or controller['crossing_probe_calls'] != index:
            nominal_errors.append(dict(step=index, error='nominal call coverage'))
        expected = index if episode['mode'] == evaluate.ARMS[1] else 0
        if controller['executed_action_observations'] != expected:
            nominal_errors.append(dict(step=index, error='executed action feedback coverage'))
        if controller['baseline_probe_action'][1:] != controller['crossing_probe_action'][1:]:
            nominal_errors.append(dict(step=index, error='nominal pedals differ'))
        if episode['mode'] == evaluate.ARMS[1]:
            original, nominal = controller['crossing_probe_action'], controller['baseline_probe_action']
            if (controller.get('nominal_baseline_action') != original
                    or type(controller.get('nominal_changed')) is not bool
                    or controller['nominal_changed'] != (nominal != original)
                    or not isinstance(controller.get('nominal_reason'), str)):
                nominal_errors.append(dict(step=index, error='nominal diagnostics/tap differ'))
    objects = []
    for index, obstacle in enumerate(catalog['obstacles']):
        passage = physical.passage(states, obstacle, index, trace)
        window = windows.fixed_window(states, obstacle)
        if window['status'] == 'complete':
            window['steering'] = steering_window(trace, window['entry']['t'], window['exit']['t'])
        window.pop('points')
        hit = any(index in state['contacts'] or state['clearance'][index] <= 0 for state in states)
        objects.append(dict(obstacle_id=index, physical=passage, fixed_window=window, hit=hit,
                            prospective_return=returns.prospective_return(states, obstacle, passage, catalog['obstacles'], episode)))
    unassociated = any(state['collision'] and not state['contacts'] and not any(g <= 0 for g in state['clearance']) for state in raw)
    full_steering = steering_window(trace, states[0]['t'], states[-1]['t'])
    return dict(finished=episode['completed'], damage=episode['damage'], collision_decisions=episode['collisions'],
                lap_ms=episode['lapTimeMs'] if episode['completed'] else None, raw_ticks=len(raw), decisions=len(trace),
                max_abs_lateral_m=max(abs(state['lateral']) for state in states),
                lateral_integral_m_s=math.fsum(abs(state['lateral']) * delta for state, delta in zip(raw, dt)),
                path_length_m=math.fsum(math.hypot(b['x'] - a['x'], b['y'] - a['y']) for a, b in zip(states, raw)),
                steering=full_steering, physical=measured, objects=objects, unassociated_collision=unassociated,
                shield=contract, nominal_contract_errors=nominal_errors)


def eligible(obj):
    window, passage = obj['fixed_window'], obj['physical']
    return (window['status'] == 'complete' and window['steering']['status'] == 'complete'
            and passage['status'] == 'passed' and passage['geometry_valid'])


def pair_metrics(b, c, bs, cs):
    result: dict[str, Any] = dict(track_id=b['track_id'], seed=b['seed'],
                  subgroup=evaluate.SUBGROUPS[f"{b['track_id']}:{b['seed']}"],
                  parity=shield.pair_metrics(b, c, bs, cs),
                  baseline=episode_metrics(b, bs), candidate=episode_metrics(c, cs), windows=[], returns=[])
    for bo, co in zip(result['baseline']['objects'], result['candidate']['objects']):
        be, ce = eligible(bo), eligible(co)
        values = {}
        if be and ce:
            bw, cw = bo['fixed_window'], co['fixed_window']
            for name, bv, cv in (
                    ('max_lateral', bw['max_abs_lateral_m'], cw['max_abs_lateral_m']),
                    ('path', bw['path_length_m'], cw['path_length_m']),
                    ('steering_integral', bw['steering']['integral_abs_s'], cw['steering']['integral_abs_s']),
                    ('steering_variation', bw['steering']['variation'], cw['steering']['variation'])):
                values[name] = dict(baseline=bv, candidate=cv, delta=cv - bv,
                                    relative=(cv - bv) / bv if bv > 0 else None)
        result['windows'].append(dict(obstacle_id=bo['obstacle_id'], baseline_eligible=be, candidate_eligible=ce, metrics=values))
        bp, cp = bo['prospective_return'], co['prospective_return']
        required = bp['observed_after_clear_s'] is not None and not bp['status'].startswith('invalid') and bp['observed_after_clear_s'] >= .24 - 1e-9
        comparable = required and cp['observed_after_clear_s'] is not None and not cp['status'].startswith('invalid')
        common = None
        if comparable:
            horizon = min(bp['observed_after_clear_s'], cp['observed_after_clear_s'])
            paired = [returns.prospective_return(states, episode['catalog']['obstacles'][bo['obstacle_id']],
                       obj['physical'], episode['catalog']['obstacles'], episode, horizon)
                      for episode, states, obj in ((b, bs, bo), (c, cs, co))]
            comparable = horizon >= .24 - 1e-9 and all(not p['status'].startswith('invalid') for p in paired)
            common = dict(horizon_s=horizon, baseline=paired[0], candidate=paired[1])
        result['returns'].append(dict(obstacle_id=bo['obstacle_id'], baseline_required=required,
                                     comparable=comparable, baseline=bp, candidate=cp, common=common))
    return result


def summarize(cells, complete):
    gate = evaluate.GATE
    kept = [cell for cell in cells if cell['baseline']['finished'] and cell['candidate']['finished']]
    required_windows = [window for cell in cells for window in cell['windows'] if window['baseline_eligible']]
    required_returns = [row for cell in cells for row in cell['returns'] if row['baseline_required']]
    comparable = [row for row in required_returns if row['comparable']]
    ordinary = [cell for cell in cells if cell['subgroup'] == 'ordinary_controls']
    fractional, geometry = {name: [] for name in METRIC_NAMES}, {name: {} for name in METRIC_NAMES}
    denominator_valid = {name: True for name in METRIC_NAMES}
    for cell in ordinary:
        selected = [w for w in cell['windows'] if w['baseline_eligible']]
        for name in METRIC_NAMES:
            values = [w['metrics'][name]['relative'] for w in selected if name in w['metrics'] and w['metrics'][name]['relative'] is not None]
            valid = bool(selected) and len(values) == len(selected)
            denominator_valid[name] &= valid
            if valid:
                mean = math.fsum(values) / len(values)
                fractional[name].append(mean)
                geometry[name].setdefault(cell['seed'], []).append(mean)
    cell_means = {n: math.fsum(v) / len(v) if v else None for n, v in fractional.items()}
    road_means = {n: {str(s): math.fsum(v) / len(v) for s, v in group.items()} for n, group in geometry.items()}
    laps = [cell['candidate']['lap_ms'] - cell['baseline']['lap_ms'] for cell in kept]
    new_hits = [dict(track_id=cell['track_id'], seed=cell['seed'], obstacle_id=bo['obstacle_id'])
                for cell in cells for bo, co in zip(cell['baseline']['objects'], cell['candidate']['objects'])
                if not bo['hit'] and co['hit']]
    checks = dict(complete_matched_evidence=complete and len(cells) == 8 and all(c['parity']['matched'] for c in cells),
                  no_lost_finish=bool(cells) and all(not c['baseline']['finished'] or c['candidate']['finished'] for c in cells),
                  per_cell_damage_collision_nonincrease=bool(cells) and all(c['candidate']['damage'] <= c['baseline']['damage']
                      and c['candidate']['collision_decisions'] <= c['baseline']['collision_decisions']
                      and c['candidate']['physical']['physical_contact_events'] <= c['baseline']['physical']['physical_contact_events'] for c in cells),
                  no_new_hit=bool(cells) and not new_hits and not any(c[a]['unassociated_collision'] for c in cells for a in ('baseline', 'candidate')),
                  unchanged_shield_contract=bool(cells) and all(c[a]['shield']['contract_valid'] and not c[a]['nominal_contract_errors'] for c in cells for a in ('baseline', 'candidate')),
                  baseline_windows_retained=bool(required_windows) and all(w['candidate_eligible'] for w in required_windows),
                  return_coverage_retained=bool(required_returns) and len(comparable) == len(required_returns),
                  return_censors_nonincrease=bool(comparable) and sum(r['common']['candidate']['status'] != 'returned' for r in comparable) <= sum(r['common']['baseline']['status'] != 'returned' for r in comparable),
                  common_return_bound_nonincrease=bool(comparable) and math.fsum(r['common']['candidate']['restricted_response_s'] - r['common']['baseline']['restricted_response_s'] for r in comparable) <= 0,
                  steering_variation_nonincrease=bool(required_windows) and all('steering_variation' in w['metrics'] for w in required_windows)
                      and math.fsum(w['metrics'].get('steering_variation', {}).get('delta', 0) for w in required_windows) <= 0,
                  every_kept_lap_within_20ms=bool(laps) and all(v <= gate['max_kept_lap_increase_ms'] for v in laps),
                  matched_lap_mean_reduction_20ms=bool(laps) and math.fsum(laps) / len(laps) <= -gate['mean_lap_reduction_ms'])
    for name, threshold in (('max_lateral', gate['ordinary_max_lateral_fraction']),
                            ('path', gate['ordinary_path_fraction']),
                            ('steering_integral', gate['ordinary_steering_integral_fraction'])):
        mean = cell_means[name]
        checks['ordinary_' + name + '_reduction'] = (len(ordinary) == 4 and denominator_valid[name]
            and len(fractional[name]) == 4 and mean is not None and mean <= threshold
            and len(road_means[name]) == gate['improving_ordinary_geometry_seeds']
            and all(v < 0 for v in road_means[name].values()))
    return dict(checks=checks, gate_passed=all(checks.values()), new_hits=new_hits,
                ordinary_equal_cell_relative=cell_means, ordinary_equal_geometry_relative=road_means,
                metric_denominators_valid=denominator_valid, baseline_eligible_windows=len(required_windows),
                baseline_required_returns=len(required_returns), common_return_cohort_n=len(comparable),
                mutually_finished_lap_pairs=len(laps), mutually_finished_lap_deltas_ms=laps,
                mean_lap_delta_ms=math.fsum(laps) / len(laps) if laps else None,
                finishes={arm: sum(c[arm]['finished'] for c in cells) for arm in ('baseline', 'candidate')})


def analyze(directory, protocol_sha256):
    directory = Path(directory).resolve()
    protocol = evaluate.validate_frozen(directory, protocol_sha256)
    if evaluate.sha(__file__) != protocol['source_inventory']['scripts/analyze_koi_nominal_trajectory.py']['sha256']:
        raise ValueError('analyzer differs from frozen source')
    report = evaluate.read_json(directory / 'episode-report.json')
    if report['protocol_sha256'] != protocol_sha256 or len(report['rows']) != 16:
        raise ValueError('report/protocol differs')
    ledger = shield.validate_ledger(directory, report['rows'], protocol_sha256)
    episodes = {}
    for row, expected in zip(report['rows'], protocol['schedule']):
        if any(row[k] != expected[k] for k in ('track_id', 'seed', 'mode', 'repeat', 'file')):
            raise ValueError('scheduled slot differs')
        if row['status'] == 'completed':
            episodes[row['track_id'], row['seed'], row['mode']] = shield.load_episode(directory, row, protocol_sha256)
        elif row['status'] == 'operator_error':
            pins = row.get('partial_artifacts', [])
            actual = sorted(p.name for p in directory.glob(Path(row['file']).stem + '.*'))
            if sorted(pin['file'] for pin in pins) != actual:
                raise ValueError('partial evidence inventory differs')
            for pin in pins:
                path = evaluate.local_path(directory, pin['file'])
                if evaluate.sha(path) != pin['sha256'] or path.stat().st_size != pin['bytes']:
                    raise ValueError('partial evidence differs')
        elif row['status'] != 'unrun':
            raise ValueError('ambiguous running slot')
    cells = []
    for track, seed in evaluate.CELLS:
        keys = [(track, seed, arm) for arm in evaluate.ARMS]
        if all(key in episodes for key in keys):
            (b, bs), (c, cs) = (episodes[key] for key in keys)
            cells.append(pair_metrics(b, c, bs, cs))
    complete = report['operator_error'] is None and len(episodes) == 16 and len(cells) == 8
    summary = summarize(cells, complete)
    result = dict(schema=evaluate.SCHEMA + '-result', protocol_sha256=protocol_sha256,
                  report_sha256=evaluate.sha(directory / 'episode-report.json'), complete=complete,
                  planned_episodes=16, valid_episodes=len(episodes), matched_pairs=len(cells),
                  unmatched_valid_episodes=len(episodes) - 2 * len(cells), operator_error=report['operator_error'],
                  baseline_sha256=evaluate.BASELINE_SHA, shield_sha256=evaluate.SHIELD_SHA,
                  scope=protocol['scope'], passive_analysis=True, environment_resets=0, official_action=False,
                  safety_objects_per_arm=6 * len(cells), geometry_seeds=len({c['seed'] for c in cells}),
                  verdict='REPEAT_CANDIDATE' if summary['gate_passed'] else 'REJECTED',
                  ledger=ledger, metric_definitions=protocol['metric_definitions'], gate=protocol['gate'],
                  cells=cells, **summary)
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
