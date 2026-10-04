"""Recompute a fail-closed, source-bound shield comparison without model execution."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scripts import analyze_koi_collision_priority as legacy_analysis
from scripts import evaluate_koi_collision_shield as evaluate


def action(row):
    return [row[key] for key in ('steer', 'gas', 'brake')]


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite evidence')
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def runs(flags, durations):
    """Sampled consecutive occupancy, including terminal right-censored runs."""
    result = []
    for index, (flag, duration) in enumerate(zip(flags, durations)):
        if flag:
            if index == 0 or not flags[index - 1]:
                result.append(dict(start_index=index, decisions_or_ticks=0, duration_s=0.))
            result[-1]['decisions_or_ticks'] += 1
            result[-1]['duration_s'] += duration
            result[-1]['end_index'] = index
            result[-1]['terminal_censored'] = index == len(flags) - 1
    return result


def shield_metrics(trace) -> dict:
    errors, active, durations, reasons = [], [], [], Counter()
    encounter_counts, encounter_count, clear_streak = [], 0, 0
    noop_count = safe_count = 0
    projections = []
    for row in trace:
        controller = row['controller']
        info = controller['shield']
        observed = controller['evaluation_only']
        durations.append(observed['post']['t'] - observed['pre']['t'])
        if not isinstance(info, dict):
            raise ValueError('candidate shield diagnostics missing')
        for field in ('active', 'baseline_threat', 'baseline_clear_observed'):
            if type(info.get(field)) is not bool:
                raise ValueError(f'missing boolean shield diagnostic: {field}')
        for field in ('reason', 'baseline_action', 'baseline_clearance', 'chosen_clearance',
                      'candidate_count', 'encounter_id', 'encounter_actions', 'threat_free_decisions'):
            if field not in info:
                raise ValueError(f'missing shield diagnostic: {field}')
        baseline = controller['baseline_probe_action']
        issued = action(row)
        changed = issued != baseline
        if info['baseline_action'] != baseline:
            errors.append(dict(step=row['step'], error='self-reported baseline differs from independent tap'))
        if changed != info['active']:
            errors.append(dict(step=row['step'], error='active/action mismatch'))
        if issued[1:] != baseline[1:]:
            errors.append(dict(step=row['step'], error='pedals changed'))
        if changed and not info['baseline_threat']:
            errors.append(dict(step=row['step'], error='changed without baseline threat'))
        if info['baseline_clear_observed'] and info['baseline_threat']:
            errors.append(dict(step=row['step'], error='clear and threat simultaneously'))
        if not info['active']:
            noop_count += 1
            if changed:
                errors.append(dict(step=row['step'], error='inactive no-op parity failed'))
        if info['baseline_clear_observed']:
            safe_count += 1
            if changed:
                errors.append(dict(step=row['step'], error='safe baseline changed'))
        clear_streak = clear_streak + 1 if info['baseline_clear_observed'] else 0
        if clear_streak >= evaluate.GATE['rearm_threat_free_decisions'] and encounter_count:
            encounter_counts.append(encounter_count)
            encounter_count = 0
        if changed:
            encounter_count += 1
            if encounter_count > evaluate.GATE['max_actions_per_encounter']:
                errors.append(dict(step=row['step'], error='encounter action budget exceeded'))
            if (not isinstance(info['baseline_clearance'], (int, float))
                    or not isinstance(info['chosen_clearance'], (int, float))
                    or info['chosen_clearance'] <= 0 or info['baseline_clearance'] > 0
                    or type(info['candidate_count']) is not int or info['candidate_count'] < 1):
                errors.append(dict(step=row['step'], error='active projection evidence invalid'))
        active.append(changed)
        reasons[info['reason']] += 1
        projections.append(dict(step=row['step'], active=changed, baseline_clearance=info['baseline_clearance'],
                                chosen_clearance=info['chosen_clearance'], candidate_count=info['candidate_count']))
    if encounter_count:
        encounter_counts.append(encounter_count)
    bursts = runs(active, durations)
    return dict(decisions=len(trace), intervention_decisions=sum(active), intervention_bursts=len(bursts),
                intervention_duration_s=sum(d for d, flag in zip(durations, active) if flag),
                max_consecutive_decisions=max((b['decisions_or_ticks'] for b in bursts), default=0),
                max_consecutive_duration_s=max((b['duration_s'] for b in bursts), default=0.),
                bursts=bursts, encounter_action_counts=encounter_counts,
                max_encounter_actions=max(encounter_counts, default=0), noop_decisions=noop_count,
                safe_baseline_decisions=safe_count, contract_errors=errors, contract_valid=not errors,
                reasons=dict(reasons), projection_clearance_proxy=projections)


def episode_metrics(episode, states) -> dict:
    """All observed ticks/objects remain present, even for natural nonfinishes."""
    physical = evaluate.legacy.physical_metrics(states, episode['catalog']['obstacles'])
    if physical != episode['physical']:
        raise ValueError('stored physical metrics do not reproduce from raw stream')
    interventions = shield_metrics(episode['decision_trace']) if episode['mode'] == evaluate.ARMS[1] else None
    metrics = legacy_analysis.episode_metrics(dict(episode, priority_changed_decisions=(
        interventions['intervention_decisions'] if interventions else 0)))
    dt = [right['t'] - left['t'] for left, right in zip(states, states[1:])]
    if not dt or any(abs(value - .02) > 1e-8 for value in dt):
        raise ValueError('raw tick coverage is not contiguous 50Hz')
    lateral = [abs(state['lateral']) for state in states]
    heading = [abs(state['heading_error']) for state in states]
    heading_changes = [abs((r['yaw'] - l['yaw'] + math.pi) % (2 * math.pi) - math.pi)
                       for l, r in zip(states, states[1:])]
    hit_objects = sorted(set(physical['hit_objects']) | {
        obj['object_id'] for obj in physical['objects'] if obj['min_fixture_clearance'] <= 0})
    offroad = runs([not any(s['wheel_road_contacts']) for s in states[1:]], dt)
    any_offroad = runs([not all(s['wheel_road_contacts']) for s in states[1:]], dt)
    metrics.update(hit_objects=hit_objects, object_count=len(physical['objects']), objects=physical['objects'],
                   physical_contact_events=physical['physical_contact_events'],
                   physical_contact_ticks=physical['physical_contact_ticks'],
                   duration_s=sum(dt), raw_ticks=len(dt), max_abs_lateral_m=max(lateral),
                   lateral_integral_m_s=sum(value * delta for value, delta in zip(lateral[1:], dt)),
                   max_abs_heading_error_rad=max(heading),
                   heading_error_integral_rad_s=sum(value * delta for value, delta in zip(heading[1:], dt)),
                   total_abs_heading_change_rad=sum(heading_changes),
                   max_tick_heading_change_rad=max(heading_changes),
                   path_length_m=sum(math.hypot(r['x'] - l['x'], r['y'] - l['y']) for l, r in zip(states, states[1:])),
                   all_wheels_offroad_ticks=physical['no_wheel_road_contact_ticks'],
                   any_wheel_offroad_ticks=physical['any_wheel_offroad_ticks'],
                   all_wheels_offroad_total_s=physical['no_wheel_road_contact_ticks'] / 50.,
                   any_wheel_offroad_total_s=physical['any_wheel_offroad_ticks'] / 50.,
                   all_wheels_offroad_runs=offroad, any_wheel_offroad_runs=any_offroad,
                   intervention=interventions)
    return metrics


def pair_metrics(baseline, candidate, baseline_states, candidate_states):
    pair = evaluate.legacy.pair_metrics(baseline, candidate)
    first = pair['first_changed_action_step']
    cutoff = first if first is not None else len(baseline['decision_trace']) + 1
    left = [s for s in baseline_states[1:] if s['step'] < cutoff]
    right = [s for s in candidate_states[1:] if s['step'] < cutoff]
    pair['raw_prefix_equal'] = left == right
    prefix = pair['equal_action_prefix_decisions']
    pair['full_decision_prefix_equal'] = all(
        baseline['decision_trace'][i]['controller']['evaluation_only'] ==
        candidate['decision_trace'][i]['controller']['evaluation_only'] for i in range(prefix))
    # Equal actions but unequal trace lengths are not a passing full no-op episode.
    pair['no_divergence_full_episode_equal'] = (None if first is not None else
        pair['all_actions_equal'] and baseline_states == candidate_states
        and baseline['completed'] == candidate['completed']
        and baseline['retire_reason'] == candidate['retire_reason'])
    pair['matched'] = all(pair[key] for key in (
        'geometry_equal', 'initial_state_equal', 'initial_pixels_equal',
        'prefix_observation_and_pre_state_equal', 'raw_prefix_equal', 'full_decision_prefix_equal'))
    pair['matched'] &= pair['first_changed_action_pre_state_equal'] is not False
    pair['matched'] &= pair['no_divergence_full_episode_equal'] is not False
    return pair


def aggregate(cells):
    kept = [c for c in cells if c['baseline']['finished'] and c['candidate']['finished']]
    result: dict = dict(cells=len(cells), geometry_seeds=len({c['seed'] for c in cells}),
                  baseline_finished=sum(c['baseline']['finished'] for c in cells),
                  candidate_finished=sum(c['candidate']['finished'] for c in cells),
                  preserved=len(kept), lost=sum(c['lost_finish'] for c in cells),
                  gained=sum(c['gained_finish'] for c in cells),
                  neither=sum(not c['baseline']['finished'] and not c['candidate']['finished'] for c in cells),
                  new_clean_object_hits=sum(len(c['new_hits']) for c in cells),
                  matched_lap_pairs=len(kept),
                  mutually_finished_mean_lap_delta_ms=(sum(c['candidate']['lap_ms'] - c['baseline']['lap_ms']
                       for c in kept) / len(kept) if kept else None),
                  mutually_finished_mean_path_delta_m=(sum(c['candidate']['path_length_m'] - c['baseline']['path_length_m']
                       for c in kept) / len(kept) if kept else None))
    result['arms'] = {}
    for arm in ('baseline', 'candidate'):
        selected = [c[arm] for c in cells]
        result['arms'][arm] = {key: sum(m[key] for m in selected) for key in (
            'damage', 'collision_decisions', 'physical_contact_events', 'object_count', 'path_length_m',
            'all_wheels_offroad_ticks', 'any_wheel_offroad_ticks', 'lateral_integral_m_s',
            'heading_error_integral_rad_s', 'total_abs_heading_change_rad', 'changed_decisions')}
        result['arms'][arm].update(hit_objects=sum(len(m['hit_objects']) for m in selected),
            max_abs_lateral_m=max((m['max_abs_lateral_m'] for m in selected), default=None),
            max_abs_heading_error_rad=max((m['max_abs_heading_error_rad'] for m in selected), default=None),
            min_clearance_m=min((m['min_clearance_m'] for m in selected), default=None))
    return result


def gate_result(cells, complete):
    gate = evaluate.GATE
    checks = dict(complete_matched_evidence=complete and all(c['parity']['matched'] for c in cells),
                  noop_pedals_projection_and_encounter_budget=bool(cells) and all(
                      c['candidate']['intervention']['contract_valid'] for c in cells),
                  baseline_finishes_preserved=not any(c['lost_finish'] for c in cells),
                  no_new_clean_object_hit=not any(c['new_hits'] for c in cells),
                  per_cell_damage_collision_nonincrease=all(
                      c['candidate']['damage'] <= c['baseline']['damage'] + 1e-9
                      and c['candidate']['collision_decisions'] <= c['baseline']['collision_decisions']
                      and c['candidate']['physical_contact_events'] <= c['baseline']['physical_contact_events'] for c in cells),
                  bounded_offroad_heading_lateral=all(
                      c['candidate']['all_wheels_offroad_ticks'] <= c['baseline']['all_wheels_offroad_ticks'] + gate['max_extra_offroad_ticks']
                      and c['candidate']['all_wheels_offroad_max_s'] <= c['baseline']['all_wheels_offroad_max_s'] + gate['max_extra_offroad_ticks'] / 50. + 1e-9
                      and c['candidate']['max_abs_lateral_m'] <= c['baseline']['max_abs_lateral_m'] + gate['max_lateral_increase_m']
                      and c['candidate']['max_abs_heading_error_rad'] <= c['baseline']['max_abs_heading_error_rad'] + gate['max_heading_error_increase_rad'] for c in cells))
    kept = [c for c in cells if c['baseline']['finished'] and c['candidate']['finished']]
    checks['mutually_finished_efficiency_nonregression'] = bool(kept) and all(
        c['candidate']['path_length_m'] <= c['baseline']['path_length_m'] * gate['max_kept_path_ratio']
        and c['candidate']['lap_ms'] <= c['baseline']['lap_ms'] + gate['max_kept_lap_increase_ms'] for c in kept)
    checks['clear_finish_or_collision_benefit'] = bool(cells) and (
        any(c['gained_finish'] for c in cells) or (
            sum(c['baseline']['damage'] - c['candidate']['damage'] for c in cells) >= gate['minimum_damage_benefit'] - 1e-9
            and sum(c['baseline']['collision_decisions'] - c['candidate']['collision_decisions'] for c in cells) >= gate['minimum_collision_benefit']))
    checks['nonzero_intervention'] = any(c['candidate']['changed_decisions'] for c in cells)
    return checks


def load_episode(directory, row, protocol_sha256):
    path = evaluate.local_path(directory, row['file'])
    if evaluate.sha(path) != row['sha256']:
        raise ValueError('episode hash mismatch')
    episode = evaluate.read_json(path)
    finite_tree(episode)
    if (not evaluate.valid_episode(episode) or episode['protocol_sha256'] != protocol_sha256
            or any(episode[key] != row[key] for key in ('track_id', 'seed', 'repeat', 'mode'))):
        raise ValueError('episode identity or completion mismatch')
    streams = {}
    for field in ('raw_trace', 'decision_stream'):
        stream = evaluate.local_path(directory, episode[field + '_file'])
        if evaluate.sha(stream) != episode[field + '_sha256']:
            raise ValueError('episode stream hash mismatch')
        streams[field] = [json.loads(line) for line in stream.read_text().splitlines()]
        finite_tree(streams[field])
    process_path = path.with_suffix('.process.json')
    if evaluate.sha(process_path) != row['process_sha256']:
        raise ValueError('process receipt hash mismatch')
    process = evaluate.read_json(process_path)
    if process['error'] is not None or process['protocol_sha256'] != protocol_sha256:
        raise ValueError('process failed or unbound')
    trace = episode['decision_trace']
    if len(trace) != len(streams['decision_stream']):
        raise ValueError('decision stream coverage mismatch')
    for index, (decision, streamed) in enumerate(zip(trace, streams['decision_stream']), 1):
        if (decision['step'] != index or streamed != dict(step=index, action=action(decision), controller=decision['controller'])):
            raise ValueError('decision stream differs from episode')
        observed = decision['controller']['evaluation_only']
        for state in (observed['pre'], observed['post']):
            if any(key not in state for key in ('dynamic_state', 'wheel_state', 'environment_state')):
                raise ValueError('full observer state missing')
        if not observed.get('observation_sha256') or not observed.get('post_observation_sha256'):
            raise ValueError('pixel hash missing')
        if index == 1:
            if observed['pre'] != episode['initial_state'] or observed['observation_sha256'] != episode['initial_observation_sha256']:
                raise ValueError('initial state/pixels do not bind first decision')
        else:
            previous = trace[index - 2]['controller']['evaluation_only']
            if previous['post'] != observed['pre'] or previous['post_observation_sha256'] != observed['observation_sha256']:
                raise ValueError('state/pixel decision continuity failed')
        raw = [s for s in streams['raw_trace'] if s['step'] == index]
        if not 1 <= len(raw) <= 4 or abs(raw[-1]['t'] - observed['post']['t']) > 1e-8:
            raise ValueError('raw decision coverage mismatch')
        if abs(len(raw) / 50. - (observed['post']['t'] - observed['pre']['t'])) > 1e-8:
            raise ValueError('raw/decision duration mismatch')
    if any(type(s['step']) is not int or not 1 <= s['step'] <= len(trace) for s in streams['raw_trace']):
        raise ValueError('raw samples outside decision coverage')
    states = [episode['initial_state'], *streams['raw_trace']]
    return episode, states


def validate_ledger(directory, rows, protocol_sha256):
    path = directory / 'reset-ledger.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    allowed = {row['file']: row for row in rows}
    counts, ended = Counter(), set()
    for event in events:
        row = allowed.get(event.get('file'))
        if (row is None or row['status'] == 'unrun' or event.get('protocol_sha256') != protocol_sha256
                or any(event.get(key) != row[key] for key in ('track_id', 'seed', 'repeat', 'mode'))
                or event['file'] in ended):
            raise ValueError('reset ledger identity/order mismatch')
        if event['event'] == 'reset_intent':
            counts[event['file']] += 1
            if event['reset_index'] != counts[event['file']]:
                raise ValueError('reset ledger index mismatch')
        elif event['event'] == 'episode_end' and counts[event['file']]:
            ended.add(event['file'])
        else:
            raise ValueError('reset ledger event mismatch')
    if any(row['status'] == 'completed' and row['file'] not in ended for row in rows):
        raise ValueError('completed episode lacks reset/end ledger')
    return dict(reset_intents=sum(counts.values()), episode_ends=len(ended),
                sha256=evaluate.sha(path) if path.exists() else None)


def analyze(directory, protocol_sha256):
    directory = Path(directory).resolve()
    protocol = evaluate.validate_frozen(directory, protocol_sha256)
    if evaluate.sha(Path(__file__)) != evaluate.sha(directory / 'source/scripts/analyze_koi_collision_shield.py'):
        raise ValueError('analyzer differs from predeclared frozen source')
    report = evaluate.read_json(directory / 'episode-report.json')
    if report['protocol_sha256'] != protocol_sha256 or len(report['rows']) != len(protocol['schedule']):
        raise ValueError('report/protocol coverage mismatch')
    ledger = validate_ledger(directory, report['rows'], protocol_sha256)
    episodes, metrics = {}, {}
    for row, expected in zip(report['rows'], protocol['schedule']):
        if any(row[key] != expected[key] for key in ('track_id', 'seed', 'repeat', 'mode', 'file')):
            raise ValueError('report slot identity mismatch')
        if row['status'] == 'completed':
            key = row['track_id'], row['seed'], row['mode']
            episodes[key] = load_episode(directory, row, protocol_sha256)
            metrics[key] = episode_metrics(*episodes[key])
        elif row['status'] not in ('unrun', 'operator_error'):
            raise ValueError('ambiguous running slot; no final result')
    cells = []
    for track, seed in evaluate.CELLS:
        keys = [(track, seed, arm) for arm in evaluate.ARMS]
        if not all(key in episodes for key in keys):
            continue
        baseline, candidate = (metrics[key] for key in keys)
        b, c = (episodes[key] for key in keys)
        cells.append(dict(track_id=track, seed=seed, subgroup=protocol['subgroups'][f'{track}:{seed}'],
                          baseline=baseline, candidate=candidate, parity=pair_metrics(b[0], c[0], b[1], c[1]),
                          lost_finish=baseline['finished'] and not candidate['finished'],
                          gained_finish=candidate['finished'] and not baseline['finished'],
                          new_hits=sorted(set(candidate['hit_objects']) - set(baseline['hit_objects']))))
    complete = report['operator_error'] is None and len(episodes) == 12 and len(cells) == 6
    checks = gate_result(cells, complete)
    return dict(schema=evaluate.SCHEMA + '-result', run=str(directory), protocol_sha256=protocol_sha256,
                report_sha256=evaluate.sha(directory / 'episode-report.json'), scope=protocol['scope'],
                comparator_sha256=protocol['comparator_zip_sha256'], wrapper_sha256=protocol['wrapper_sha256'],
                verdict='REPEAT_CANDIDATE' if all(checks.values()) else 'NOT_ADOPTED',
                complete=complete, planned_episodes=12, valid_episodes=len(episodes), matched_pairs=len(cells),
                unmatched_valid_episodes=len(episodes) - 2 * len(cells), operator_error=report['operator_error'],
                ledger=ledger,
                gate=protocol['gate'], checks=checks, cells=cells, aggregate=aggregate(cells),
                subgroups={name: aggregate([c for c in cells if c['subgroup'] == name]) for name in sorted(set(evaluate.SUBGROUPS.values()))},
                limitations=[protocol['consumed_evidence']['selection'], protocol['consumed_evidence']['subgroup_definition'],
                    'Six layouts on five geometry seeds; outcome-selected consumed TRAIN, not independent generalization or official ranking.',
                    'All natural failures remain in finish/safety/motion metrics. Short failed paths are not efficiency gains.',
                    'Mutually finished lap/path deltas are conditional and cannot compensate lost finishes or missing pairs.',
                    'Projected runtime clearance is a pixel/kinematic proxy; sampled fixture clearance is exact at logged ticks, not a continuous-time safety proof.',
                    'No-op and pre-divergence comparisons check full logged observer state/pixels/raw prefix, not serialized hidden Box2D solver internals.',
                    'Safe baseline overavoidance is unchanged by contract; no ordinary-obstacle improvement is presumed.',
                    'A passing development gate permits a separately authorized unchanged repeat, not adoption or model promotion.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.protocol_sha256)
    evaluate.save(args.output, result)
    print(json.dumps({key: result[key] for key in ('verdict', 'complete', 'checks', 'aggregate')}))


if __name__ == '__main__':
    main()
