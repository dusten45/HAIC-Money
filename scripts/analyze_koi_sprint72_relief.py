"""Passive, deterministic analysis of the fixed eight-pair sprint72 relief study.

No Agent is imported or executed. Missing, corrupt, unmatched and censored evidence
cannot earn adoption credit. Run with ``python -m scripts.analyze_koi_sprint72_relief``.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

from scripts import analyze_koi_avoidance_magnitude as previous


shield = previous.shield
CELLS = ((1, 3184000002), (2, 3184000006), (2, 3184000001), (3, 3184000002),
         (1, 3184000013), (1, 3184000015), (2, 3184000015), (3, 3184000015))
ARMS = ('frozen_shield', 'sprint72_relief')
SCHEMA = 'koi-sprint72-relief-v1-result'
PRIOR_ELIGIBLE = {'1:3184000002': [30, 56], '2:3184000006': [40, 41, 46, 194, 195, 447, 448],
                  '2:3184000001': [], '3:3184000002': [], '1:3184000013': [50, 51, 56, 57, 215],
                  '1:3184000015': [38, 39, 44, 45, 114, 139, 140, 145],
                  '2:3184000015': [221, 229], '3:3184000015': [38, 39, 44, 45]}
METRIC_DEFINITIONS = {
    'scope': 'All eight consumed TRAIN pairs/five roads and all six objects/arm/cell, including natural DNFs and reverse travel. Not fresh or official.',
    'cap_window': 'First eligible cap hold through NEXT real restriction or terminal; further eligible holds join the same window. First source/eligibility boundary is recorded separately. T<60 while original sprint still issues full gas does not end velocity followup.',
    'restriction': 'Hazard/track/arrival, effective impact-clear, recovery, geometry repair, contact or shield intervention; otherwise T<60 or exact S failure with sprint inactive and an issued restrictive command (brake>0 or gas<1). Terminal is right-censored.',
    'space': 'Original near/centers/parent steering/free samples/free distance inequalities without tolerance; retain distance margin and floating-point-edge failures without relabeling them physical limits.',
    'station_match': 'Baseline-defined raw station samples, shortest-wrap unwrapped station, earliest same-direction chronological candidate crossing after the prior matched window. Linear interpolation only within monotone sampled segments. Reverse monotone passages are supported; nonmonotone, missing and early-restriction spans remain explicit, invalidate coverage, and are not discarded.',
    'undershoot': 'max(0,72-true raw physical speed); own-window min/max deficit/time integral and paired station-trapezoid mean deficit. Equal windows within cell, then equal cells with baseline cap windows. HUD deficits reported separately, not physical speed.',
    'oscillation': 'Actual issued B=brake>0, G=gas>0 and brake==0, otherwise C; consecutive B->G and G->B changes, coasting breaks adjacency. Repeated B->G cycles=max(0,count-1) per cap window. Paired span counts and all own-window counts retained; both must strictly reduce for credit.',
    'arrival': 'Every arrival-cap presence onset and active-braking onset; nearest projected physical object is an OFFLINE association proxy, with all projection errors retained. All six objects and unmatched onsets remain present; no detection-to-object identity proof.',
    'reverse': 'Offline abs(physical heading error)>pi/2 OR negative shortest-wrap station progress. Evaluation tag only, never eligibility veto or policy input.',
    'safety': 'No lost finish, per-cell damage/collision decisions/contact events nonincrease, no new contact-or-clearance<=0 hit, any-wheel AND all-wheel offroad tick counts and longest spells nonincrease.',
    'lap': 'Common-finish deltas only, with all lost/gained/neither cells retained. Every delta<=20ms, strictly negative equal-cell mean and strictly negative road means on at least two distinct geometry seeds. Ties/mixed directions explicit.',
    'adoption': 'Full source/raw/prefix integrity and eight matched pairs; all baseline station windows covered; strict undershoot and repeated-cycle reduction plus all safety/lap gates. Passing is only a consumed-development gate, never automatic adoption/repeat/tuning.',
}
STATE_FIELDS = ('t', 'x', 'y', 'yaw', 'speed', 'station', 'lateral', 'heading_error', 'road_index',
                'front', 'rear', 'clearance', 'contacts', 'wheel_road_contacts', 'dynamic_state', 'wheel_state')


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def context(row) -> dict[str, Any]:
    """Reconstruct the exact logged prearrival guards, not an offline clean mask."""
    c = row['controller']
    before = c['evaluation_only']['controller_state_before']
    centers, parent, speed = c['road_centers'], c['parent_action'], c['pixel_speed']
    margin = c['free_distance'] - max(18., max(speed, 72.) * .4)
    space = (c['near_object'] is None and '42' in centers and '54' in centers
             and abs(centers['54'] - 42.) < 3. and c['free_samples'] >= 4
             and abs(parent[0]) < .18 and margin >= 0.)
    clearing = (before['impact_left'] > 0 or c['impact_proxy_trigger']) and not c['braking_proxy_veto']
    pre = [parent[0], 1., 0.] if c['mechanism_active'] else list(parent)
    eligible = (row['step'] > 10 and speed >= 72. and c['target_speed'] == 60. and space
                and not c['far_objects'] and c['tracked_object'] is None and not clearing
                and c['recovery_steps_remaining'] == 0 and not c['geometry_repaired']
                and pre[1:] == [0., f32(.15)])
    reasons = []
    if c['target_speed'] < 60.:
        reasons.append('target_below_60')
    if not space:
        reasons.append('space_gate_failed')
    for label, flag in (('hazard_or_track', c['near_object'] is not None or bool(c['far_objects'])
                        or c['tracked_object'] is not None or c['arrival_cap'] is not None),
                       ('impact_clear', clearing), ('recovery', c['recovery_steps_remaining'] > 0 or c['recovery_changed']),
                       ('geometry_repair', c['geometry_repaired']), ('contact', c['contact_active']),
                       ('shield', c['shield']['active'])):
        if flag:
            reasons.append(label)
    real = bool(set(reasons) - {'target_below_60', 'space_gate_failed'}) or (
        bool(reasons) and not c['mechanism_active'] and (row['brake'] > 0 or row['gas'] < 1))
    return dict(eligible=eligible, space_ok=space, space_margin=margin,
                space_fp_edge=not space and -1e-12 <= margin < 0., impact_clear=clearing,
                pre_action=pre, reasons=reasons, real_restriction=real)


def issued_metrics(trace, start, end):
    selected = [r for r in trace if r['controller']['evaluation_only']['pre']['t'] < end
                and r['controller']['evaluation_only']['post']['t'] > start]
    labels = ['B' if r['brake'] > 0 else 'G' if r['gas'] > 0 else 'C' for r in selected]
    bg = [b['step'] for a, b, x, y in zip(selected, selected[1:], labels, labels[1:]) if (x, y) == ('B', 'G')]
    gb = [b['step'] for a, b, x, y in zip(selected, selected[1:], labels, labels[1:]) if (x, y) == ('G', 'B')]
    steering = previous.steering_window(trace, start, end) if end > start else {'status': 'empty'}
    return dict(decisions=len(selected), brake_to_gas=len(bg), gas_to_brake=len(gb),
                brake_to_gas_steps=bg, gas_to_brake_steps=gb, repeated_brake_gas=max(0, len(bg) - 1),
                steering=steering)


def activity(trace, start, end):
    result = {k: [] for k in ('impact_trigger', 'impact_veto', 'impact_clear', 'recovery',
                             'geometry_repair', 'arrival_braking', 'contact', 'shield')}
    for row in trace:
        c = row['controller']
        if not start <= c['evaluation_only']['pre']['t'] < end:
            continue
        flags = (c['impact_proxy_trigger'], c['braking_proxy_veto'], context(row)['impact_clear'],
                 c['recovery_changed'] or c['recovery_steps_remaining'] > 0, c['geometry_repaired'],
                 c['far_active'], c['contact_active'], c['shield']['active'])
        for key, flag in zip(result, flags):
            if flag:
                result[key].append(row['step'])
    return {key: dict(count=len(steps), steps=steps) for key, steps in result.items()}


def velocity_points(states, catalog):
    track = catalog['track']
    circumference = math.fsum(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(track, track[1:] + track[:1]))
    if circumference <= 0:
        raise ValueError('invalid road circumference')
    points, station = [], states[0]['station']
    for i, s in enumerate(states):
        delta = 0. if not i else (s['station'] - states[i - 1]['station'] + circumference / 2) % circumference - circumference / 2
        if i:
            station += delta
        points.append(dict(t=s['t'], station=s['station'], unwrapped_station=station, speed=s['speed'],
                           heading_error=s['heading_error'], step=s.get('step', 0),
                           reverse=abs(s['heading_error']) > math.pi / 2 or delta < -1e-9,
                           seam=bool(i and abs(s['station'] - states[i - 1]['station']) > circumference / 2)))
    return points


def undershoot(points):
    if not points:
        return None
    deficits = [max(0., 72. - p['speed']) for p in points]
    minimum = min(p['speed'] for p in points)
    return dict(entry_physical_speed=points[0]['speed'], min_physical_speed=minimum,
                entry_to_min_drop=points[0]['speed'] - minimum, reference_72_minus_min=72. - minimum,
                max_deficit=max(deficits),
                deficit_integral_speed_s=math.fsum((a + b) / 2 * (r['t'] - l['t'])
                    for a, b, l, r in zip(deficits, deficits[1:], points, points[1:])),
                observed_duration_s=points[-1]['t'] - points[0]['t'], samples=len(points))


def cap_windows(episode, points):
    trace, result, i = episode['decision_trace'], [], 0
    while i < len(trace):
        if not context(trace[i])['eligible']:
            i += 1
            continue
        start = i
        boundary = start if context(trace[start])['real_restriction'] else None
        i += int(boundary is None)
        while i < len(trace):
            ctx = context(trace[i])
            if ctx['reasons'] and boundary is None:
                boundary = i
            if ctx['real_restriction']:
                break
            i += 1
        rows = trace[start:max(start + 1, i)]
        t0 = rows[0]['controller']['evaluation_only']['pre']['t']
        t1 = trace[i]['controller']['evaluation_only']['pre']['t'] if i < len(trace) else points[-1]['t']
        sampled = [p for p in points if t0 - 1e-9 <= p['t'] <= t1 + 1e-9]
        deltas = [b['unwrapped_station'] - a['unwrapped_station'] for a, b in zip(sampled, sampled[1:])]
        positive, negative = any(d > 1e-9 for d in deltas), any(d < -1e-9 for d in deltas)

        def onset(index):
            if index is None or index == len(trace):
                return None
            r = trace[index]
            c = r['controller']
            return dict(step=r['step'], pre=c['evaluation_only']['pre'], post=c['evaluation_only']['post'],
                        action=shield.action(r), hud_speed=c['pixel_speed'], target=c['target_speed'], **context(r))

        result.append(dict(id=len(result), start_step=rows[0]['step'], last_step=rows[-1]['step'],
            start_t=t0, end_t=t1, status='next_restriction' if i < len(trace) else 'censored_terminal',
            terminal_reason=episode['retire_reason'] if i == len(trace) else None,
            eligible_steps=[r['step'] for r in rows if context(r)['eligible']],
            left_truncated=bool(start and trace[start - 1]['brake'] > 0),
            reverse_tagged=any(p['reverse'] for p in sampled),
            station_direction=0 if positive == negative else 1 if positive else -1,
            eligibility_boundary=onset(boundary), next_restriction=onset(i),
            velocity=sampled, physical_undershoot=undershoot(sampled),
            hud=dict(samples=[dict(step=r['step'], speed=r['controller']['pixel_speed']) for r in rows],
                     min_speed=min(r['controller']['pixel_speed'] for r in rows)),
            issued=issued_metrics(trace, t0, t1),
            downstream=activity(trace, t1, points[-1]['t'] + 1e-9),
            downstream_issued=issued_metrics(trace, t1, points[-1]['t'])))
        i = max(i, start + 1)
    return result


def arrival_events(episode):
    events, previous_object, previous_active = [], None, False
    for row in episode['decision_trace']:
        c = row['controller']
        if c['arrival_cap'] is None:
            previous_object, previous_active = None, False
            continue
        detected = c['near_object'] if c['near_object'] is not None else c['far_objects'][0]
        state = c['evaluation_only']['pre']
        errors = []
        for obj in episode['catalog']['obstacles']:
            dx, dy = obj['x'] - state['x'], obj['y'] - state['y']
            right = dx * math.cos(state['yaw']) + dy * math.sin(state['yaw'])
            forward = -dx * math.sin(state['yaw']) + dy * math.cos(state['yaw'])
            errors.append(math.hypot(42. + 1.3608 * right - detected[1], 63. - 1.701 * forward - detected[0]))
        obj = min(range(len(errors)), key=errors.__getitem__)
        entry = previous_object != obj
        braking = c['far_active'] and (entry or not previous_active)
        if entry or braking:
            events.append(dict(step=row['step'], t=state['t'], station=state['station'], object_id_proxy=obj,
                cap_entry=entry, braking_onset=braking, cap=c['arrival_cap'], hud_speed=c['pixel_speed'],
                physical_speed=state['speed'], action=shield.action(row), projection_errors_px=errors,
                reverse_tagged=abs(state['heading_error']) > math.pi / 2))
        previous_object, previous_active = obj, c['far_active']
    return events


def validate_treatment(episode):
    errors, eligible, applied, reverse = [], [], [], []
    last_state = None
    for row in episode['decision_trace']:
        c, index = row['controller'], row['step']
        observed = c['evaluation_only']
        before, after = observed['controller_state_before'], observed['controller_state_after']
        if last_state is not None and before != last_state:
            raise ValueError('controller state continuity differs')
        last_state = after
        if any(type(s[k]) is not int or s[k] < 0 for s in (before, after) for k in ('impact_left', 'recovery_left')):
            raise ValueError('invalid controller timer state')
        if not after['brake_history'] or after['brake_history'][-1] != row['brake']:
            errors.append(dict(step=index, error='issued brake history differs'))
        expected_impact = 0 if c['braking_proxy_veto'] else c['impact_steps_remaining']
        if after['recovery_left'] != c['recovery_steps_remaining'] or after['impact_left'] != expected_impact:
            errors.append(dict(step=index, error='timer diagnostics differ'))
        if (c['nominal_probe_calls'] != index or c['executed_action_observations'] != 0
                or c['nominal_probe_action'] != c['baseline_probe_action']):
            errors.append(dict(step=index, error='nominal tap/call coverage differs'))
        ctx = context(row)
        pre = ctx['pre_action']
        proposal = list(pre)
        if ctx['eligible']:
            eligible.append(index)
            if abs(observed['pre']['heading_error']) > math.pi / 2 or observed['post']['station'] < observed['pre']['station']:
                reverse.append(index)
        if episode['mode'] == ARMS[1]:
            formula = min(.15, max(0., .02 * (c['pixel_speed'] - 72.))) if ctx['eligible'] else None
            if formula is not None:
                proposal[1:] = [0., f32(formula)]
            changed = proposal != pre
            if changed:
                applied.append(index)
            expected = dict(sprint72_eligible=ctx['eligible'], sprint72_applied=changed,
                sprint72_space_ok=ctx['space_ok'], sprint72_pre_action=pre, sprint72_prearrival_action=proposal,
                sprint72_brake_formula=formula, sprint72_impact_before=before['impact_left'])
            if any(k not in c or c[k] != v or (isinstance(v, bool) and type(c[k]) is not bool) for k, v in expected.items()):
                errors.append(dict(step=index, error='prearrival relief diagnostics/formula differ'))
        elif any(k.startswith('sprint72_') for k in c):
            errors.append(dict(step=index, error='baseline has relief helper fields'))
        # Crossing and shield change steering only. Arrival retains priority over
        # both original sprint and the new helper's already-cast pedal values.
        if c['arrival_cap'] is not None and c['pixel_speed'] > c['arrival_cap']:
            proposal[1:] = [0., f32(max(proposal[2], min(.6, max(0., .04 * (c['pixel_speed'] - c['arrival_cap'])))))]
        if shield.action(row)[1:] != proposal[1:]:
            errors.append(dict(step=index, error='issued pedals do not bind prearrival/arrival law'))
    return dict(contract_valid=not errors, contract_errors=errors, eligible_steps=eligible,
                applied_steps=applied, reverse_eligible_steps=reverse, eligible_decisions=len(eligible),
                applied_decisions=len(applied))


def episode_metrics(episode, states) -> dict[str, Any]:
    shield.finite_tree(episode)
    shield.finite_tree(states)
    if not previous.evaluate.valid_episode(episode) or episode['mode'] not in ARMS:
        raise ValueError('invalid episode')
    catalog, trace = episode['catalog'], episode['decision_trace']
    if (episode['steps'] != len(trace) or [o['id'] for o in catalog['obstacles']] != list(range(6))
            or hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest() != episode['geometry_sha256']):
        raise ValueError('decision/geometry/object inventory differs')
    if states[0] != episode['initial_state'] or not len(states) > 1:
        raise ValueError('raw initial state/coverage differs')
    by_step = {}
    for i, state in enumerate(states):
        if (any(len(state[k]) != 6 for k in ('front', 'rear', 'clearance')) or len(state['wheel_road_contacts']) != 4
                or state['speed'] < 0 or any(type(k) is not int or k not in range(6) for k in state['contacts'])
                or state['contacts'] != sorted(set(state['contacts']))
                or any(type(k) is not int or k < 0 for k in state['wheel_road_contacts'])):
            raise ValueError('invalid raw physical arrays')
        if (len(state['dynamic_state']) != 5 or len(state['wheel_state']) != 4
                or abs(math.hypot(*state['dynamic_state'][0]['velocity']) - state['speed']) > 1e-8):
            raise ValueError('physical speed/dynamic state differs')
        if i:
            if (abs(state['t'] - states[i - 1]['t'] - .02) > 1e-8 or type(state['collision']) is not bool
                    or type(state['step']) is not int or not 1 <= state['step'] <= len(trace)):
                raise ValueError('raw tick continuity/coverage differs')
            by_step.setdefault(state['step'], []).append(state)
    if [s['step'] for s in states[1:]] != [i for i in range(1, len(trace) + 1) for _ in by_step.get(i, [])]:
        raise ValueError('raw decision ordering differs')
    for index, row in enumerate(trace, 1):
        c = row['controller']
        observed = c['evaluation_only']
        ticks = by_step.get(index, [])
        if (row['step'] != index or not 1 <= len(ticks) <= 4 or (index < len(trace) and len(ticks) != 4)
                or type(row['collision']) is not bool or row['collision'] != any(s['collision'] for s in ticks)):
            raise ValueError('raw/decision coverage or collision differs')
        if any(observed['post'][k] != ticks[-1][k] for k in STATE_FIELDS):
            raise ValueError('raw/decision endpoint differs')
        if (abs(observed['post']['t'] - observed['pre']['t'] - len(ticks) * .02) > 1e-8
                or (index == 1 and observed['pre'] != states[0])
                or (index > 1 and trace[index - 2]['controller']['evaluation_only']['post'] != observed['pre'])):
            raise ValueError('decision pre/post continuity differs')
        for action in (shield.action(row), c['baseline_probe_action'], c['parent_action']):
            if (not isinstance(action, list) or len(action) != 3 or any(type(v) not in (int, float) for v in action)
                    or not -1 <= action[0] <= 1 or not 0 <= action[1] <= 1 or not 0 <= action[2] <= 1):
                raise ValueError('invalid action tap')
        if not 38 <= c['target_speed'] <= 60:
            raise ValueError('preview target outside frozen mapping')
    if episode['collisions'] != sum(r['collision'] for r in trace):
        raise ValueError('headline collisions differ')
    physical = shield.evaluate.legacy.physical_metrics(states, catalog['obstacles'])
    if physical != episode['physical']:
        raise ValueError('stored physical metrics differ')
    previous.validate_wrapper_phases(episode, states)
    points = velocity_points(states, catalog)
    treatment = validate_treatment(episode)
    treatment['reverse_eligible_steps'] = sorted({p['step'] for p in points if p['reverse']
                                                and p['step'] in treatment['eligible_steps']})
    arrivals = arrival_events(episode)
    objects = [dict(obj, hit=any(i in s['contacts'] or s['clearance'][i] <= 0 for s in states),
                    arrival_status='observed' if any(e['object_id_proxy'] == i for e in arrivals) else 'not_observed',
                    arrival_onsets=[e for e in arrivals if e['object_id_proxy'] == i]) for i, obj in enumerate(physical['objects'])]
    windows = cap_windows(episode, points)
    for window in windows:
        window['arrival_onsets'] = [e for e in arrivals if window['start_t'] <= e['t'] <= window['end_t']]
        window['next_arrival_onset'] = next((e for e in arrivals if e['t'] >= window['start_t']), None)
    return dict(finished=episode['completed'], retire_reason=episode['retire_reason'], damage=episode['damage'],
        collision_decisions=episode['collisions'], lap_ms=episode['lapTimeMs'] if episode['completed'] else None,
        decisions=len(trace), raw_ticks=len(states) - 1, duration_s=states[-1]['t'] - states[0]['t'],
        physical=physical, objects=objects, treatment=treatment, shield=shield.shield_metrics(trace),
        unassociated_collision=any(s['collision'] and not s['contacts'] and all(g > 0 for g in s['clearance']) for s in states[1:]),
        offroad_spells={name: shield.runs(flags, [.02] * (len(states) - 1)) for name, flags in (
            ('any_wheel', [not all(s['wheel_road_contacts']) for s in states[1:]]),
            ('all_wheels', [not any(s['wheel_road_contacts']) for s in states[1:]]))},
        issued=issued_metrics(trace, points[0]['t'], points[-1]['t']),
        activity=activity(trace, points[0]['t'], points[-1]['t'] + 1e-9),
        cap_windows=windows, arrival_onsets=arrivals,
        reverse_raw_ticks=sum(p['reverse'] for p in points[1:]))


def station_windows(baseline, candidate, candidate_points):
    """Retain every baseline window, even when no comparable trajectory exists."""
    matches, cursor = [], 0
    trace = candidate['decision_trace']
    for window in baseline['cap_windows']:
        points, direction = window['velocity'], window['station_direction']
        result: dict[str, Any] = dict(window_id=window['id'], baseline_status=window['status'], reverse_tagged=window['reverse_tagged'],
                      status='nonmonotone_baseline', covered=False, trajectory=[], baseline_issued=window['issued'],
                      candidate_issued=None, baseline_mean_deficit=None, candidate_mean_deficit=None,
                      required_samples=len(points), matched_samples=0)
        matches.append(result)
        if not direction or len(points) < 2:
            continue
        start, end = (direction * p['unwrapped_station'] for p in (points[0], points[-1]))
        crossing = next((i for i in range(cursor, len(candidate_points) - 1)
            if direction * candidate_points[i]['unwrapped_station'] <= start
            < direction * candidate_points[i + 1]['unwrapped_station']), None)
        if crossing is None:
            result['status'] = 'start_unreached'
            continue
        i = crossing
        paired = []
        status = 'covered'
        for point in points:
            station = direction * point['unwrapped_station']
            while i + 1 < len(candidate_points) and direction * candidate_points[i + 1]['unwrapped_station'] < station:
                if direction * (candidate_points[i + 1]['unwrapped_station'] - candidate_points[i]['unwrapped_station']) < -1e-9:
                    status = 'candidate_nonmonotone'
                    break
                i += 1
            if status != 'covered' or i + 1 == len(candidate_points):
                status = status if status != 'covered' else 'end_unreached'
                break
            a, b = candidate_points[i:i + 2]
            qa, qb = direction * a['unwrapped_station'], direction * b['unwrapped_station']
            if qb <= qa or not qa - 1e-9 <= station <= qb + 1e-9:
                status = 'candidate_nonmonotone'
                break
            fraction = (station - qa) / (qb - qa)
            paired.append(dict(station=point['station'], unwrapped_station=point['unwrapped_station'],
                baseline_t=point['t'], baseline_speed=point['speed'],
                candidate_t=a['t'] + fraction * (b['t'] - a['t']), candidate_speed=a['speed'] + fraction * (b['speed'] - a['speed'])))
        if paired:
            t0, t1 = paired[0]['candidate_t'], paired[-1]['candidate_t']
            earlier = next((r for r in trace if t0 - 1e-9 <= r['controller']['evaluation_only']['pre']['t'] < t1 - 1e-9
                            and context(r)['real_restriction']), None)
            if earlier is not None:
                status = 'candidate_earlier_restriction'
                result['candidate_restriction_step'] = earlier['step']
                result['candidate_restriction_t'] = earlier['controller']['evaluation_only']['pre']['t']
                for sample in paired:
                    sample['after_candidate_restriction'] = sample['candidate_t'] > result['candidate_restriction_t']
            result['candidate_issued'] = issued_metrics(trace, t0, t1)
            result['candidate_reverse_tagged'] = any(p['reverse'] for p in candidate_points[crossing:i + 2])
            cursor = i
        result.update(status=status, covered=status == 'covered' and len(paired) == len(points), trajectory=paired,
                      required_samples=len(points), matched_samples=len(paired))
        if result['covered'] and end > start:
            for arm in ('baseline', 'candidate'):
                result[arm + '_mean_deficit'] = math.fsum(
                    (max(0., 72. - a[arm + '_speed']) + max(0., 72. - b[arm + '_speed'])) / 2
                    * abs(b['unwrapped_station'] - a['unwrapped_station']) for a, b in zip(paired, paired[1:])) / (end - start)
    return matches


def pair_metrics(b, c, bs, cs, baseline=None, candidate=None) -> dict[str, Any]:
    if ((b['track_id'], b['seed'], b['repeat']) != (c['track_id'], c['seed'], c['repeat'])
            or (b['track_id'], b['seed']) not in CELLS or b['mode'] != ARMS[0] or c['mode'] != ARMS[1]):
        raise ValueError('pair identity differs')
    baseline = baseline if baseline is not None else episode_metrics(b, bs)
    candidate = candidate if candidate is not None else episode_metrics(c, cs)
    arrivals = []
    for obj in range(6):
        left, right = ([e for e in a['arrival_onsets'] if e['object_id_proxy'] == obj] for a in (baseline, candidate))
        for i in range(max(len(left), len(right))):
            be, ce = left[i] if i < len(left) else None, right[i] if i < len(right) else None
            arrivals.append(dict(object_id_proxy=obj, ordinal=i, baseline=be, candidate=ce,
                same_onset_type=bool(be and ce and (be['cap_entry'], be['braking_onset']) == (ce['cap_entry'], ce['braking_onset'])),
                delta_t_s=ce['t'] - be['t'] if be and ce else None,
                delta_station=ce['station'] - be['station'] if be and ce else None))
    parity = shield.pair_metrics(b, c, bs, cs)
    first_steer = next((dict(baseline_step=br['step'], candidate_step=cr['step'],
        baseline_t=br['controller']['evaluation_only']['pre']['t'], candidate_t=cr['controller']['evaluation_only']['pre']['t'],
        baseline_action=shield.action(br), candidate_action=shield.action(cr))
        for br, cr in zip(b['decision_trace'], c['decision_trace']) if br['steer'] != cr['steer']), None)
    return dict(track_id=b['track_id'], seed=b['seed'], baseline=baseline, candidate=candidate,
        road_geometry_sha256=hashlib.sha256(json.dumps(b['catalog']['track'], sort_keys=True).encode()).hexdigest(),
        parity=parity, station_windows=station_windows(baseline, c, velocity_points(cs, c['catalog'])),
        first_steering_difference=first_steer,
        steering_comparison_note='Same decision ordinal, not a same-state causal comparison after pedal divergence.',
        arrival_comparison=arrivals, lost_finish=baseline['finished'] and not candidate['finished'],
        gained_finish=candidate['finished'] and not baseline['finished'],
        new_hits=[i for i, (bo, co) in enumerate(zip(baseline['objects'], candidate['objects'])) if co['hit'] and not bo['hit']])


def summarize(cells, complete) -> dict[str, Any]:
    ids = [(c['track_id'], c['seed']) for c in cells]
    common = [c for c in cells if c['baseline']['finished'] and c['candidate']['finished']]
    laps = [dict(track_id=c['track_id'], seed=c['seed'], delta_ms=c['candidate']['lap_ms'] - c['baseline']['lap_ms']) for c in common]
    roads, hashes = {}, {}
    for c in cells:
        hashes.setdefault(c['seed'], set()).add(c['road_geometry_sha256'])
    for lap in laps:
        roads.setdefault(str(lap['seed']), []).append(lap['delta_ms'])
    road_means = {k: math.fsum(v) / len(v) for k, v in roads.items()}
    all_windows = [w for c in cells for w in c['station_windows']]
    coverage = bool(all_windows) and all(w['covered'] for w in all_windows)
    cap_cells = [c for c in cells if c['station_windows']]
    deficits: dict[str, Any] = {a: (math.fsum(math.fsum(w[a + '_mean_deficit'] for w in c['station_windows']) / len(c['station_windows'])
                for c in cap_cells) / len(cap_cells) if coverage else None) for a in ('baseline', 'candidate')}
    cycles: dict[str, Any] = {a: sum(w[a + '_issued']['repeated_brake_gas'] for w in all_windows)
              if coverage else None for a in ('baseline', 'candidate')}
    own_cycles = {a: sum(w['issued']['repeated_brake_gas'] for c in cells for w in c[a]['cap_windows']) for a in ('baseline', 'candidate')}
    safety_keys = ('no_wheel_road_contact_ticks', 'no_wheel_road_contact_max_ticks',
                   'any_wheel_offroad_ticks', 'any_wheel_offroad_max_ticks')
    objects_complete = bool(cells) and all([o['object_id'] for o in c[a]['objects']] == list(range(6)) for c in cells for a in ('baseline', 'candidate'))
    checks = dict(complete_matched_evidence=complete and len(ids) == len(set(ids)) == 8 and set(ids) == set(CELLS)
                  and all(c['parity']['matched'] for c in cells),
        five_distinct_road_geometries=len(hashes) == 5 and all(len(h) == 1 for h in hashes.values())
                  and len(set().union(*hashes.values())) == 5,
        all_objects_retained=objects_complete,
        no_lost_finish=bool(cells) and not any(c['lost_finish'] for c in cells),
        per_cell_damage_collision_nonincrease=bool(cells) and all(c['candidate']['damage'] <= c['baseline']['damage']
            and c['candidate']['collision_decisions'] <= c['baseline']['collision_decisions']
            and c['candidate']['physical']['physical_contact_events'] <= c['baseline']['physical']['physical_contact_events'] for c in cells),
        no_new_hit=objects_complete and not any(c['new_hits'] for c in cells)
            and not any(c[a]['unassociated_collision'] for c in cells for a in ('baseline', 'candidate')),
        per_cell_offroad_nonincrease=bool(cells) and all(c['candidate']['physical'][k] <= c['baseline']['physical'][k] for c in cells for k in safety_keys),
        source_law_and_shield_contract=bool(cells) and all(c[a]['treatment']['contract_valid'] and c[a]['shield']['contract_valid'] for c in cells for a in ('baseline', 'candidate')),
        baseline_station_window_coverage=coverage,
        cap_undershoot_reduced=coverage and deficits['candidate'] < deficits['baseline'],
        repeated_brake_gas_reduced=coverage and cycles['candidate'] < cycles['baseline'] and own_cycles['candidate'] < own_cycles['baseline'],
        every_common_lap_within_20ms=bool(laps) and all(l['delta_ms'] <= 20 for l in laps),
        common_lap_mean_negative=bool(laps) and math.fsum(l['delta_ms'] for l in laps) < 0,
        at_least_two_improving_geometries=sum(v < 0 for v in road_means.values()) >= 2)
    arms = {}
    for arm in ('baseline', 'candidate'):
        rows = [c[arm] for c in cells]
        arms[arm] = dict(episodes=len(rows), finishes=sum(r['finished'] for r in rows), safety_objects=sum(len(r['objects']) for r in rows),
            damage=math.fsum(r['damage'] for r in rows), collision_decisions=sum(r['collision_decisions'] for r in rows),
            contact_events=sum(r['physical']['physical_contact_events'] for r in rows), hit_objects=sum(o['hit'] for r in rows for o in r['objects']),
            offroad={k: sum(r['physical'][k] for r in rows) if not 'max' in k else max((r['physical'][k] for r in rows), default=0) for k in safety_keys},
            cap_windows=sum(len(r['cap_windows']) for r in rows), cap_statuses=dict(Counter(w['status'] for r in rows for w in r['cap_windows'])),
            reverse_cap_windows=sum(w['reverse_tagged'] for r in rows for w in r['cap_windows']),
            eligible_decisions=sum(r['treatment']['eligible_decisions'] for r in rows), applied_decisions=sum(r['treatment']['applied_decisions'] for r in rows),
            all_episode_brake_to_gas=sum(r['issued']['brake_to_gas'] for r in rows),
            shield_decisions=sum(r['shield']['intervention_decisions'] for r in rows))
    return dict(checks=checks, gate_passed=all(checks.values()), aggregate_arms=arms,
        lost_finishes=sum(c['lost_finish'] for c in cells), gained_finishes=sum(c['gained_finish'] for c in cells),
        common_finishes=len(common), neither_finished=sum(not c['baseline']['finished'] and not c['candidate']['finished'] for c in cells),
        common_lap_deltas=laps, mean_lap_delta_ms=math.fsum(l['delta_ms'] for l in laps) / len(laps) if laps else None,
        geometry_mean_lap_delta_ms=road_means, geometry_lap_directions={k: 'improved' if v < 0 else 'regressed' if v > 0 else 'tie' for k, v in road_means.items()},
        new_hits=[dict(track_id=c['track_id'], seed=c['seed'], object_id=i) for c in cells for i in c['new_hits']],
        baseline_station_windows=len(all_windows), covered_station_windows=sum(w['covered'] for w in all_windows),
        cap_metric_cells=len(cap_cells), cap_metric_geometry_seeds=len({c['seed'] for c in cap_cells}),
        equal_cell_station_mean_deficit=deficits, paired_repeated_brake_gas=cycles, own_window_repeated_brake_gas=own_cycles)


def analyze(directory, protocol_sha256=None) -> dict[str, Any]:
    """Return a report even for incomplete/corrupt evidence; never execute policy."""
    from scripts import evaluate_koi_sprint72_relief as evaluate

    directory = Path(directory).resolve()
    errors, episodes, metrics, cells, coverage, unmatched = [], {}, {}, [], [], []
    protocol, report, ledger = {}, {}, None
    digest = None
    try:
        digest = evaluate.sha(directory / 'protocol.json')
        if protocol_sha256 is not None and digest != protocol_sha256:
            raise ValueError('requested protocol hash differs')
        protocol = evaluate.validate_frozen(directory, digest)
        if (evaluate.sha(__file__) != protocol['source_inventory']['scripts/analyze_koi_sprint72_relief.py']['sha256']
                or protocol['metric_definitions'] != METRIC_DEFINITIONS
                or protocol['cells'] != [list(c) for c in CELLS] or protocol['arms'] != list(ARMS)):
            raise ValueError('analyzer/definitions/cohort differs from frozen source')
        report = evaluate.read_json(directory / 'episode-report.json')
        if report['protocol_sha256'] != digest or len(report['rows']) != 16 or len(protocol['schedule']) != 16:
            raise ValueError('report/protocol coverage differs')
        operator_names = ('run-start.json', 'preflight.json', 'reset-ledger.jsonl', 'resource-receipts.jsonl')
        previous.validate_artifacts(directory, report['operator_artifacts'], [n for n in operator_names if (directory / n).exists()])
        start, preflight = (evaluate.read_json(directory / n) for n in ('run-start.json', 'preflight.json'))
        if start['protocol_sha256'] != digest or start['preflight_sha256'] != evaluate.sha(directory / 'preflight.json'):
            raise ValueError('run-start/preflight binding differs')
        evaluate.validate_preflight(preflight, digest, protocol)
        ledger = shield.validate_ledger(directory, report['rows'], digest)
        events = [json.loads(line) for line in (directory / 'reset-ledger.jsonl').read_text().splitlines()]
        if any(n != 1 for n in Counter(e['file'] for e in events if e['event'] == 'reset_intent').values()):
            raise ValueError('duplicate reset/retry evidence')
        scheduled_ids = [(r['track_id'], r['seed'], r['mode'], r['repeat']) for r in protocol['schedule']]
        if len(set(scheduled_ids)) != 16 or set(scheduled_ids) != {(t, s, a, 0) for t, s in CELLS for a in ARMS}:
            raise ValueError('schedule repeats or omits a required pair')
        for index, (row, expected) in enumerate(zip(report['rows'], protocol['schedule'])):
            try:
                if any(row[k] != expected[k] for k in ('track_id', 'seed', 'mode', 'repeat', 'file')):
                    raise ValueError('scheduled slot differs')
                actual = sorted(p.name for p in directory.glob(Path(row['file']).stem + '.*'))
                if row['status'] in ('completed', 'operator_error'):
                    previous.validate_artifacts(directory, row['artifacts'] if row['status'] == 'completed' else row['partial_artifacts'], actual)
                    process_path = evaluate.local_path(directory, row['file']).with_suffix('.process.json')
                    if process_path.exists():
                        process = evaluate.read_json(process_path)
                        if process['protocol_sha256'] != digest or process['command'] != evaluate.worker_command(directory, index, digest, protocol['python']):
                            raise ValueError('process command/protocol differs')
                    if row['status'] == 'completed':
                        required = {Path(row['file']).with_suffix(s).name for s in ('.json', '.raw.jsonl', '.decisions.jsonl', '.process.json', '.stdout.txt', '.stderr.txt', '.resources.jsonl')}
                        if not required <= set(actual):
                            raise ValueError('completed evidence missing')
                        key = row['track_id'], row['seed'], row['mode']
                        episode, states = shield.load_episode(directory, row, digest)
                        if any(episode['versions'][k] != v for k, v in dict(numpy='1.26.0', cv2='4.8.1', torch='2.1.0+cpu').items()):
                            raise ValueError('CPU21 dependency versions differ')
                        measured = episode_metrics(episode, states)
                        episodes[key], metrics[key] = (episode, states), measured
                elif row['status'] != 'unrun' or actual:
                    raise ValueError('unreported partial/running evidence')
            except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
                errors.append(dict(file=row.get('file'), error=f'{type(error).__name__}: {error}'))
        for track, seed in CELLS:
            keys = [(track, seed, a) for a in ARMS]
            present = all(k in episodes for k in keys)
            coverage.append(dict(track_id=track, seed=seed, valid_arms=[a for a in ARMS if (track, seed, a) in episodes], paired=present))
            if present:
                (b, bs), (c, cs) = (episodes[k] for k in keys)
                cells.append(pair_metrics(b, c, bs, cs, *(metrics[k] for k in keys)))
            else:
                unmatched.extend(dict(track_id=track, seed=seed, mode=k[2], metrics=metrics[k]) for k in keys if k in metrics)
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        errors.append(dict(file='run_integrity', error=f'{type(error).__name__}: {error}'))
    complete = not errors and report.get('operator_error') is None and len(episodes) == 16 and len(cells) == 8 and all(c['parity']['matched'] for c in cells)
    summary = summarize(cells, complete)
    result = dict(schema=SCHEMA, protocol_sha256=digest, complete=complete, integrity_errors=errors,
        report_sha256=evaluate.sha(directory / 'episode-report.json') if (directory / 'episode-report.json').is_file() else None,
        verdict='INCOMPLETE' if not complete else 'PASSED_CONSUMED_DEVELOPMENT_GATE' if summary['gate_passed'] else 'REJECTED',
        adopted=False, tuning_authorized=False, repeat_authorized=False, passive_analysis=True, environment_resets=0, official_action=False,
        planned_episodes=16, valid_episodes=len(episodes), matched_pairs=len(cells), planned_geometry_seeds=5,
        planned_safety_objects_per_arm=48, unmatched_valid_episodes=len(unmatched), unmatched_episodes=unmatched,
        observed_safety_objects_per_arm={a: sum(len(m['objects']) for k, m in metrics.items() if k[2] == a) for a in ARMS},
        cell_coverage=coverage, operator_error=report.get('operator_error'), ledger=ledger,
        source_integrity_verified=bool(protocol) and not any(e['file'] == 'run_integrity' for e in errors),
        metric_definitions=METRIC_DEFINITIONS, gate=protocol.get('gate'), cells=cells,
        prior_diagnosis=dict(source='Historical recall, not new-run applicability or performance evidence', eligible_steps=PRIOR_ELIGIBLE,
                            eligible_decisions=28, reverse_cell=[2, 3184000006], reverse_steps=[194, 195, 447, 448]),
        limitations=['Outcome-selected consumed TRAIN; no independent-road or official-performance claim.',
            'Common-finish lap means cannot offset losses. Every missing pair/window remains a coverage failure, not a zero or a survivor gain.',
            'Physical speed magnitude and HUD estimate are distinct; 72 is the diagnostic reference, not a calibrated physical speed limiter.',
            'Station interpolation, shortest-wrap progress and pixel-to-physical object association are offline proxies, not policy inputs or counterfactual replay.',
            'Reverse and nonmonotone events remain visible. Ambiguous station matching fails coverage rather than excluding difficult events.',
            'Terminal windows remain censored; observed undershoot/cycles describe their observed horizon, not an uncensored recovery time.',
            'Fixture/contact/offroad metrics are sampled at 50Hz, not continuous-time safety guarantees; prefix equality covers logged state, not hidden solver serialization.',
            'Unchanged shield bytes do not establish unchanged downstream safety. No adoption, repeat, retune or external action is authorized by this report.'], **summary)
    shield.finite_tree(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'complete', 'valid_episodes', 'matched_pairs', 'checks')}))


if __name__ == '__main__':
    main()
