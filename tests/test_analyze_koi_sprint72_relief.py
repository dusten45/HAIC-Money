"""Synthetic trace, integrity and gate checks only; no policy or simulator calls."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from scripts import analyze_koi_sprint72_relief as analyze
from scripts import evaluate_koi_sprint72_relief as evaluate


def sampled_episode(mode, track=1, seed=3184000002, reverse=False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obstacles: list[dict[str, Any]] = [dict(id=i, radius=1., station=200. + 100 * i, anchor_index=i,
                      x=200. + 100 * i, y=0., tangent=[1., 0.]) for i in range(6)]
    catalog = dict(track=[[0., 0., 0., 0.], [0., 0., float(1000 + seed % 100), 0.]], obstacles=obstacles)

    def state(tick):
        x = 500. - tick if reverse else float(tick)
        speed = 70. if tick % 8 >= 4 else 74.
        return dict(t=tick / 50., x=x, y=0., yaw=0., speed=speed, station=x, road_index=0,
            lateral=.5, heading_error=math.pi if reverse else 0., rear=[x - o['station'] - 2. for o in obstacles],
            front=[x - o['station'] + 2. for o in obstacles], clearance=[10.] * 6,
            contacts=[], wheel_road_contacts=[1] * 4,
            dynamic_state=[dict(velocity=[speed, 0.], position=[x, 0.], angle=0., angular_velocity=0., awake=True) for _ in range(5)],
            wheel_state=[dict(gas=0., brake=0., steer=0., omega=0., phase=0., joint_angle=0., joint_speed=0., motor_speed=0.) for _ in range(4)],
            environment_state=dict(off_track_counter=0, damage=0., reward=0., prev_reward=0., tile_visited_count=1, fuel_spent=0.))

    states, trace, history = [state(0)], [], []
    for step in range(1, 21):
        cap = step in (11, 13, 18)
        speed = 74. if cap or step == 16 else 70.
        mechanism = step > 10 and step != 16 and not cap
        parent = [0., 0., analyze.f32(.15)]
        pre_action = [0., 1., 0.] if mechanism else parent.copy()
        action = pre_action.copy()
        eligible = cap
        formula = .04 if eligible else None
        if mode == analyze.ARMS[1] and eligible:
            action[2] = analyze.f32(formula)
        before = dict(impact_left=0, recovery_left=0, brake_history=history.copy())
        history = (history + [action[2]])[-4:]
        after = dict(impact_left=0, recovery_left=0, brake_history=history.copy())
        pre, post = state((step - 1) * 4), state(step * 4)
        info = dict(active=False, baseline_threat=False, baseline_clear_observed=True,
            reason='baseline_clear', baseline_action=action.copy(), baseline_clearance=10., chosen_clearance=10.,
            candidate_count=0, encounter_id=0, encounter_actions=0, threat_free_decisions=step)
        controller = dict(target_speed=59.6 if step == 15 else 60., pixel_speed=speed,
            road_centers={'42': 42., '54': 42.}, parent_action=parent, mechanism_active=mechanism,
            free_samples=10, free_distance=10. if step == 16 else 40., near_object=None, far_objects=[],
            tracked_object=None, arrival_cap=None, far_active=False, contact_active=False,
            impact_proxy_trigger=False, braking_proxy_veto=False, recovery_changed=False,
            recovery_steps_remaining=0, impact_steps_remaining=0, geometry_repaired=False,
            shield=info, nominal_probe_action=action.copy(), baseline_probe_action=action.copy(),
            nominal_probe_calls=step, executed_action_observations=0,
            evaluation_only=dict(pre=pre, post=post, observation_sha256=f'p{step-1}', post_observation_sha256=f'p{step}',
                controller_state_before=before, controller_state_after=after,
                decision_reward=0., off_track_count_before=0, off_track_count_after=0,
                simulator_terminated=False, wrapper_crashed=False, wrapper_off_track=False))
        if mode == analyze.ARMS[1]:
            controller.update(sprint72_eligible=eligible, sprint72_applied=eligible,
                sprint72_space_ok=step != 16, sprint72_pre_action=pre_action,
                sprint72_prearrival_action=action.copy(), sprint72_brake_formula=formula, sprint72_impact_before=0)
        trace.append(dict(step=step, steer=action[0], gas=action[1], brake=action[2], collision=False, damage=0.,
            damage_telemetry_valid=True, collision_telemetry_valid=True, controller=controller))
        states.extend(dict(state(tick), step=step, collision=False) for tick in range((step - 1) * 4 + 1, step * 4 + 1))
    episode = dict(mode=mode, track_id=track, seed=seed, repeat=0, completed=True, error=None, invalid_actions=0,
        retire_reason=None, damage=0., collisions=0, lapTimeMs=1600., damage_telemetry_valid=True,
        collision_telemetry_valid=True, steps=len(trace), catalog=catalog,
        geometry_sha256=hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
        initial_state=states[0], initial_observation_sha256='p0', decision_trace=trace,
        versions=dict(numpy='1.26.0', cv2='4.8.1', torch='2.1.0+cpu'))
    episode['physical'] = analyze.shield.evaluate.legacy.physical_metrics(states, obstacles)
    return episode, states


def passing_cells() -> list[dict[str, Any]]:
    cells = []
    for track, seed in analyze.CELLS:
        b, bs = sampled_episode(analyze.ARMS[0], track, seed)
        c, cs = sampled_episode(analyze.ARMS[1], track, seed)
        cell = analyze.pair_metrics(b, c, bs, cs)
        cell['candidate']['lap_ms'] -= 40
        for w in cell['station_windows']:
            w.update(covered=True, baseline_mean_deficit=2., candidate_mean_deficit=1.)
            w['baseline_issued']['repeated_brake_gas'] = 2
            w['candidate_issued']['repeated_brake_gas'] = 1
        for w in cell['baseline']['cap_windows']:
            w['issued']['repeated_brake_gas'] = 2
        for w in cell['candidate']['cap_windows']:
            w['issued']['repeated_brake_gas'] = 1
        cells.append(cell)
    return cells


def pin(path):
    return dict(file=path.name, sha256=evaluate.sha(path), bytes=path.stat().st_size)


def synthetic_run(directory, partial=False) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol: dict[str, Any] = dict(cells=[list(c) for c in analyze.CELLS], arms=list(analyze.ARMS),
        schedule=evaluate.scheduled_slots(), python='unused', scope='consumed TRAIN', gate=evaluate.GATE,
        metric_definitions=analyze.METRIC_DEFINITIONS, resource_plan=dict(forecast={}),
        source_inventory={'scripts/analyze_koi_sprint72_relief.py': dict(sha256=evaluate.sha(analyze.__file__))})
    evaluate.save(directory / 'protocol.json', protocol)
    digest = evaluate.sha(directory / 'protocol.json')
    evaluate.save(directory / 'preflight.json', dict(protocol_sha256=digest, environment_resets=0))
    evaluate.save(directory / 'run-start.json', dict(protocol_sha256=digest, preflight_sha256=evaluate.sha(directory / 'preflight.json')))
    events, rows = [], copy.deepcopy(protocol['schedule'])
    for index, row in enumerate(rows):
        if partial and index > 1:
            continue
        path = directory / row['file']
        events.append(dict(row, event='reset_intent', reset_index=1, protocol_sha256=digest))
        evaluate.save(path.with_suffix('.process.json'), dict(command=evaluate.worker_command(directory, index, digest, 'unused'),
            protocol_sha256=digest, error='child_timeout' if partial and index == 1 else None))
        row['process_sha256'] = evaluate.sha(path.with_suffix('.process.json'))
        if partial and index == 1:
            path.with_suffix('.raw.jsonl').write_text('{"partial":true}\n')
            row.update(status='operator_error', error='child_timeout', partial_artifacts=[pin(p) for p in sorted(directory.glob(path.stem + '.*'))])
            continue
        episode, states = sampled_episode(row['mode'], row['track_id'], row['seed'])
        raw, decisions = path.with_suffix('.raw.jsonl'), path.with_suffix('.decisions.jsonl')
        raw.write_text(''.join(json.dumps(s) + '\n' for s in states[1:]))
        decisions.write_text(''.join(json.dumps(dict(step=r['step'], action=analyze.shield.action(r), controller=r['controller'])) + '\n'
                                     for r in episode['decision_trace']))
        episode.update(protocol_sha256=digest, raw_trace_file=raw.name, raw_trace_sha256=evaluate.sha(raw),
            decision_stream_file=decisions.name, decision_stream_sha256=evaluate.sha(decisions))
        evaluate.save(path, episode)
        for suffix in ('.stdout.txt', '.stderr.txt', '.resources.jsonl'):
            path.with_suffix(suffix).write_text('')
        row.update(status='completed', sha256=evaluate.sha(path), artifacts=[pin(p) for p in sorted(directory.glob(path.stem + '.*'))])
        events.append(dict(row, event='episode_end', protocol_sha256=digest))
    (directory / 'reset-ledger.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    report: dict[str, Any] = dict(protocol_sha256=digest, rows=rows, operator_error='child_timeout' if partial else None,
        operator_artifacts=[pin(directory / n) for n in ('run-start.json', 'preflight.json', 'reset-ledger.jsonl')])
    evaluate.save(directory / 'episode-report.json', report)
    return protocol, report


class MetricsTest(unittest.TestCase):
    def test_exact_treatment_and_reverse_are_not_clean_masked(self):
        for reverse in (False, True):
            episode, states = sampled_episode(analyze.ARMS[1], reverse=reverse)
            measured = analyze.episode_metrics(episode, states)
            self.assertTrue(measured['treatment']['contract_valid'], measured['treatment'])
            self.assertTrue(measured['shield']['contract_valid'])
            self.assertEqual(measured['treatment']['eligible_steps'], [11, 13, 18])
            self.assertEqual(measured['treatment']['reverse_eligible_steps'], [11, 13, 18] if reverse else [])
            self.assertEqual(len(measured['objects']), 6)

    def test_source_boundary_is_not_next_real_restriction(self):
        episode, states = sampled_episode(analyze.ARMS[0])
        measured = analyze.episode_metrics(episode, states)
        first, second = measured['cap_windows']
        self.assertEqual(first['eligibility_boundary']['step'], 15)
        self.assertEqual(first['next_restriction']['step'], 16)
        self.assertEqual(first['eligible_steps'], [11, 13])
        self.assertEqual(first['issued']['brake_to_gas'], 2)
        self.assertEqual(first['issued']['repeated_brake_gas'], 1)
        self.assertEqual(second['status'], 'censored_terminal')
        self.assertEqual(second['velocity'][-1]['t'], states[-1]['t'])
        self.assertEqual(first['physical_undershoot']['entry_to_min_drop'], 4.)
        self.assertEqual(first['physical_undershoot']['max_deficit'], 2.)
        self.assertEqual(first['hud']['min_speed'], 70.)

    def test_same_hold_shield_restriction_is_retained_without_opportunity_credit(self):
        episode, states = sampled_episode(analyze.ARMS[1])
        episode['decision_trace'][10]['controller']['shield']['active'] = True
        windows = analyze.cap_windows(episode, analyze.velocity_points(states, episode['catalog']))
        first = windows[0]
        self.assertEqual(first['eligible_steps'], [11])
        self.assertEqual(first['start_t'], first['end_t'])
        self.assertEqual(first['next_restriction']['step'], 11)
        self.assertEqual(first['issued']['decisions'], 0)
        self.assertEqual(first['station_direction'], 0)

    def test_all_objects_and_exact_offroad_ticks_including_terminal_spell(self):
        b, bs = sampled_episode(analyze.ARMS[0])
        c, cs = sampled_episode(analyze.ARMS[1])
        cs[42]['contacts'] = [5]
        cs[42]['clearance'][5] = -.01
        cs[42]['wheel_road_contacts'] = [0, 1, 1, 1]
        for s in cs[-2:]:
            s['wheel_road_contacts'] = [0] * 4
        c['decision_trace'][-1]['controller']['evaluation_only']['post']['wheel_road_contacts'] = [0] * 4
        c['physical'] = analyze.shield.evaluate.legacy.physical_metrics(cs, c['catalog']['obstacles'])
        pair = analyze.pair_metrics(b, c, bs, cs)
        self.assertEqual(pair['new_hits'], [5])
        self.assertIsNone(pair['candidate']['objects'][5]['passage_t'])
        self.assertEqual(pair['candidate']['physical']['any_wheel_offroad_ticks'], 3)
        self.assertEqual(pair['candidate']['physical']['no_wheel_road_contact_max_ticks'], 2)
        self.assertTrue(pair['candidate']['offroad_spells']['all_wheels'][0]['terminal_censored'])

    def test_natural_nonfinish_is_not_dropped_or_given_lap_credit(self):
        episode, states = sampled_episode(analyze.ARMS[0], reverse=True)
        episode.update(completed=False, retire_reason='off_track', lapTimeMs=None)
        measured = analyze.episode_metrics(episode, states)
        self.assertFalse(measured['finished'])
        self.assertIsNone(measured['lap_ms'])
        self.assertEqual(len(measured['objects']), 6)
        self.assertEqual(measured['cap_windows'][-1]['terminal_reason'], 'off_track')
        self.assertTrue(measured['cap_windows'][-1]['reverse_tagged'])

    def test_station_interpolation_forward_and_reverse(self):
        for reverse in (False, True):
            b, bs = sampled_episode(analyze.ARMS[0], reverse=reverse)
            c, cs = sampled_episode(analyze.ARMS[1], reverse=reverse)
            pair = analyze.pair_metrics(b, c, bs, cs)
            self.assertTrue(pair['parity']['matched'])
            self.assertEqual(pair['parity']['first_changed_action_step'], 11)
            self.assertEqual(len(pair['station_windows']), 2)
            for window in pair['station_windows']:
                self.assertTrue(window['covered'], window)
                self.assertEqual(window['reverse_tagged'], reverse)
                self.assertEqual(window['baseline_mean_deficit'], window['candidate_mean_deficit'])
                self.assertEqual(window['required_samples'], window['matched_samples'])

    def test_no_survivor_window_credit_on_earlier_restriction_or_reverse_loss(self):
        b, bs = sampled_episode(analyze.ARMS[0], reverse=True)
        c, cs = sampled_episode(analyze.ARMS[1], reverse=True)
        baseline = analyze.episode_metrics(b, bs)
        c['decision_trace'][12]['controller']['tracked_object'] = [20., 42.]
        windows = analyze.station_windows(baseline, c, analyze.velocity_points(cs, c['catalog']))
        self.assertFalse(windows[0]['covered'])
        self.assertEqual(windows[0]['status'], 'candidate_earlier_restriction')
        self.assertTrue(windows[0]['reverse_tagged'])
        self.assertEqual(len(windows), 2)
        points = analyze.velocity_points(cs, c['catalog'])[:46]
        windows = analyze.station_windows(baseline, c, points)
        self.assertFalse(windows[-1]['covered'])
        self.assertEqual(len(windows), 2)

    def test_nonmonotone_baseline_retained_and_unmatchable(self):
        b, bs = sampled_episode(analyze.ARMS[0])
        points = analyze.velocity_points(bs, b['catalog'])
        points[43]['unwrapped_station'] -= 10
        baseline = dict(cap_windows=analyze.cap_windows(b, points))
        windows = analyze.station_windows(baseline, b, points)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]['status'], 'nonmonotone_baseline')
        self.assertFalse(windows[0]['covered'])

    def test_coast_breaks_actual_brake_to_gas_adjacency(self):
        episode, _ = sampled_episode(analyze.ARMS[0])
        rows = episode['decision_trace'][10:14]
        self.assertEqual(analyze.issued_metrics(rows, .8, 1.12)['brake_to_gas'], 2)
        rows[1].update(gas=0., brake=0.)
        self.assertEqual(analyze.issued_metrics(rows, .8, 1.12)['brake_to_gas'], 1)
        self.assertEqual(analyze.issued_metrics(rows, .8, 1.12)['repeated_brake_gas'], 0)

    def test_exact_space_edges_impact_veto_and_recovery_semantics(self):
        episode, _ = sampled_episode(analyze.ARMS[0])
        row = episode['decision_trace'][10]
        c = row['controller']
        c['free_distance'] = max(c['pixel_speed'], 72.) * .4
        self.assertTrue(analyze.context(row)['eligible'])
        c['free_distance'] = math.nextafter(c['free_distance'], -math.inf)
        self.assertFalse(analyze.context(row)['eligible'])
        self.assertTrue(analyze.context(row)['space_fp_edge'])
        c['free_distance'] = 40.
        c['evaluation_only']['controller_state_before']['impact_left'] = 1
        self.assertFalse(analyze.context(row)['eligible'])
        c['braking_proxy_veto'] = True
        c['recovery_changed'] = True
        self.assertTrue(analyze.context(row)['eligible'])
        c['tracked_object'] = [20., 42.]
        self.assertFalse(analyze.context(row)['eligible'])

    def test_treatment_reports_law_history_and_field_tampering(self):
        for mutation in ('brake', 'missing', 'history', 'steer'):
            episode, _ = sampled_episode(analyze.ARMS[1])
            c = episode['decision_trace'][10]['controller']
            if mutation == 'brake':
                c['sprint72_prearrival_action'][2] = analyze.f32(.15)
            elif mutation == 'missing':
                del episode['decision_trace'][0]['controller']['sprint72_brake_formula']
            elif mutation == 'history':
                c['evaluation_only']['controller_state_after']['brake_history'][-1] = analyze.f32(.15)
                episode['decision_trace'][11]['controller']['evaluation_only']['controller_state_before'] = copy.deepcopy(c['evaluation_only']['controller_state_after'])
            else:
                c['sprint72_prearrival_action'][0] = .01
            with self.subTest(mutation=mutation):
                self.assertFalse(analyze.validate_treatment(episode)['contract_valid'])

    def test_raw_speed_geometry_and_tick_corruption_reject(self):
        for mutation in ('speed', 'geometry', 'tick', 'endpoint', 'wrapper'):
            episode, states = sampled_episode(analyze.ARMS[0])
            if mutation == 'speed':
                states[1]['speed'] += 1
            elif mutation == 'geometry':
                episode['geometry_sha256'] = 'bad'
            elif mutation == 'tick':
                states[1]['t'] += .001
            elif mutation == 'endpoint':
                episode['decision_trace'][0]['controller']['evaluation_only']['post']['x'] += 1
            else:
                states[1]['environment_state']['damage'] = .2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze.episode_metrics(episode, states)

    def test_prefix_detects_equal_actions_with_changed_physics_and_lengths(self):
        b, bs = sampled_episode(analyze.ARMS[0])
        c, cs = sampled_episode(analyze.ARMS[1])
        cs[1]['x'] += .1
        self.assertFalse(analyze.shield.pair_metrics(b, c, bs, cs)['matched'])
        c, cs = copy.deepcopy(b), copy.deepcopy(bs)
        c['decision_trace'].pop()
        self.assertFalse(analyze.shield.pair_metrics(b, c, bs, cs)['matched'])

    def test_arrival_records_presence_and_braking_onsets_separately(self):
        episode, _ = sampled_episode(analyze.ARMS[0])
        for step in (16, 17):
            c = episode['decision_trace'][step - 1]['controller']
            c.update(arrival_cap=68., far_objects=[[20., 42., 42.]], far_active=step == 17)
        events = analyze.arrival_events(episode)
        self.assertEqual(len(events), 2)
        self.assertTrue(events[0]['cap_entry'])
        self.assertFalse(events[0]['braking_onset'])
        self.assertTrue(events[1]['braking_onset'])
        self.assertEqual(len(events[0]['projection_errors_px']), 6)


class GateTest(unittest.TestCase):
    def test_fixed_gate_passes_only_full_synthetic_improvement(self):
        result = analyze.summarize(passing_cells(), True)
        self.assertTrue(result['gate_passed'], result['checks'])
        self.assertEqual(result['aggregate_arms']['baseline']['safety_objects'], 48)
        self.assertEqual(len(result['geometry_mean_lap_delta_ms']), 5)
        self.assertEqual(result['common_finishes'], 8)

    def test_all_safety_coverage_lap_and_contract_gates_fail_independently(self):
        changes = (
            ('no_lost_finish', lambda c: c[0].update(lost_finish=True)),
            ('per_cell_damage_collision_nonincrease', lambda c: c[0]['candidate'].update(damage=.2)),
            ('per_cell_damage_collision_nonincrease', lambda c: c[0]['candidate'].update(collision_decisions=1)),
            ('per_cell_damage_collision_nonincrease', lambda c: c[0]['candidate']['physical'].update(physical_contact_events=1)),
            ('no_new_hit', lambda c: c[0].update(new_hits=[5])),
            ('no_new_hit', lambda c: c[0]['candidate'].update(unassociated_collision=True)),
            ('per_cell_offroad_nonincrease', lambda c: c[0]['candidate']['physical'].update(any_wheel_offroad_ticks=1)),
            ('per_cell_offroad_nonincrease', lambda c: c[0]['candidate']['physical'].update(no_wheel_road_contact_max_ticks=1)),
            ('source_law_and_shield_contract', lambda c: c[0]['candidate']['treatment'].update(contract_valid=False)),
            ('source_law_and_shield_contract', lambda c: c[0]['baseline']['shield'].update(contract_valid=False)),
            ('baseline_station_window_coverage', lambda c: c[0]['station_windows'][0].update(covered=False)),
            ('every_common_lap_within_20ms', lambda c: c[0]['candidate'].update(lap_ms=1621.)),
            ('complete_matched_evidence', lambda c: c[0]['parity'].update(matched=False)),
        )
        for name, mutate in changes:
            cells = passing_cells()
            mutate(cells)
            result = analyze.summarize(cells, True)
            with self.subTest(gate=name):
                self.assertFalse(result['checks'][name])
                self.assertFalse(result['gate_passed'])
        self.assertFalse(analyze.summarize([], False)['gate_passed'])
        self.assertFalse(analyze.summarize(passing_cells()[:-1], True)['gate_passed'])

    def test_ties_mixed_geometry_and_twenty_ms_boundary(self):
        cells = passing_cells()
        for c in cells:
            c['candidate']['lap_ms'] = c['baseline']['lap_ms'] + (20 if c['seed'] != 3184000015 else -40)
        result = analyze.summarize(cells, True)
        self.assertTrue(result['checks']['every_common_lap_within_20ms'])
        self.assertFalse(result['checks']['at_least_two_improving_geometries'])
        self.assertIn('regressed', result['geometry_lap_directions'].values())
        for c in cells:
            c['candidate']['lap_ms'] = c['baseline']['lap_ms']
        result = analyze.summarize(cells, True)
        self.assertFalse(result['checks']['common_lap_mean_negative'])
        self.assertEqual(set(result['geometry_lap_directions'].values()), {'tie'})

    def test_no_survivor_credit_for_lost_finish_or_missing_reverse_window(self):
        cells = passing_cells()
        cells[1].update(lost_finish=True)
        cells[1]['candidate'].update(finished=False, lap_ms=None)
        cells[1]['station_windows'][0].update(covered=False, reverse_tagged=True)
        result = analyze.summarize(cells, True)
        self.assertEqual(result['lost_finishes'], 1)
        self.assertEqual(result['common_finishes'], 7)
        self.assertEqual(result['baseline_station_windows'], 16)
        self.assertIsNone(result['equal_cell_station_mean_deficit']['candidate'])
        self.assertFalse(result['gate_passed'])


class IntegrityTest(unittest.TestCase):
    def run_analysis(self, directory, protocol):
        with patch.object(evaluate, 'validate_frozen', return_value=protocol), patch.object(evaluate, 'validate_preflight'):
            return analyze.analyze(directory)

    def test_full_source_bound_io_is_deterministic_and_not_automatic_adoption(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            directory = Path(temporary)
            protocol, _ = synthetic_run(directory)
            first = self.run_analysis(directory, protocol)
            self.assertTrue(first['complete'], first['integrity_errors'])
            self.assertEqual(first['valid_episodes'], 16)
            self.assertEqual(first['matched_pairs'], 8)
            self.assertEqual(first, self.run_analysis(directory, protocol))
            self.assertFalse(first['adopted'])
            self.assertEqual(first['verdict'], 'REJECTED')

    def test_partial_run_preserves_unmatched_episode_and_all_coverage_rows(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            directory = Path(temporary)
            protocol, _ = synthetic_run(directory, partial=True)
            result = self.run_analysis(directory, protocol)
            self.assertEqual(result['verdict'], 'INCOMPLETE')
            self.assertEqual(result['valid_episodes'], 1)
            self.assertEqual(result['unmatched_valid_episodes'], 1)
            self.assertEqual(len(result['cell_coverage']), 8)
            self.assertEqual(len(result['unmatched_episodes'][0]['metrics']['objects']), 6)

    def test_corrupt_stream_or_source_pin_is_incomplete_not_adoption(self):
        for mutation in ('raw', 'source'):
            with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
                directory = Path(temporary)
                protocol, report = synthetic_run(directory)
                if mutation == 'raw':
                    path = (directory / report['rows'][0]['file']).with_suffix('.raw.jsonl')
                    path.write_text(path.read_text() + '{}\n')
                else:
                    protocol['source_inventory']['scripts/analyze_koi_sprint72_relief.py']['sha256'] = 'wrong'
                result = self.run_analysis(directory, protocol)
                with self.subTest(mutation=mutation):
                    self.assertEqual(result['verdict'], 'INCOMPLETE')
                    self.assertTrue(result['integrity_errors'])
                    self.assertFalse(result['adopted'])

    def test_duplicate_reset_and_forged_process_command_fail_closed(self):
        for mutation in ('reset', 'command'):
            with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
                directory = Path(temporary)
                protocol, report = synthetic_run(directory)
                if mutation == 'reset':
                    path = directory / 'reset-ledger.jsonl'
                    lines = path.read_text().splitlines()
                    repeated = dict(json.loads(lines[0]), reset_index=2)
                    lines.insert(1, json.dumps(repeated))
                    path.write_text('\n'.join(lines) + '\n')
                    report['operator_artifacts'] = [pin(directory / p['file']) for p in report['operator_artifacts']]
                else:
                    row = report['rows'][0]
                    path = (directory / row['file']).with_suffix('.process.json')
                    process = evaluate.read_json(path)
                    process['command'] = ['wrong']
                    path.write_text(json.dumps(process))
                    row['process_sha256'] = evaluate.sha(path)
                    row['artifacts'] = [pin(directory / p['file']) for p in row['artifacts']]
                (directory / 'episode-report.json').write_text(json.dumps(report))
                result = self.run_analysis(directory, protocol)
                with self.subTest(mutation=mutation):
                    self.assertEqual(result['verdict'], 'INCOMPLETE')
                    self.assertTrue(result['integrity_errors'])

    def test_missing_run_and_expected_protocol_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            directory = Path(temporary)
            self.assertEqual(analyze.analyze(directory)['verdict'], 'INCOMPLETE')
            synthetic_run(directory)
            self.assertEqual(analyze.analyze(directory, 'wrong')['verdict'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
