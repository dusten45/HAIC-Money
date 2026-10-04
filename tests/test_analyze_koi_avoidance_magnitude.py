"""Synthetic raw/stream/source integrity and gates; zero simulator interactions."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from scripts import analyze_koi_avoidance_magnitude as analyze
from scripts import evaluate_koi_avoidance_magnitude as evaluate


def passing_cells():
    cells = []
    for track, seed in evaluate.CELLS:
        arms = {a: dict(finished=True, damage=0., collision_decisions=0,
            lap_ms=20000 if a == 'baseline' else 19960,
            physical={'physical_contact_events': 0}, shield={'contract_valid': True, 'intervention_decisions': 0},
            nominal_contract_errors=[], unassociated_collision=False,
            treatment=dict(contract_valid=True, decisions=100, magnitude_changed_decisions=0,
                final_vs_crossing_changed_decisions=0, magnitude_shield_overlap_decisions=0,
                magnitude_cancelled_by_shield_decisions=0),
            objects=[dict(obstacle_id=i, hit=False, prospective_return={'status': 'returned'}) for i in range(6)])
            for a in ('baseline', 'candidate')}
        windows = [dict(obstacle_id=i, baseline_eligible=True, candidate_eligible=True,
            metrics={n: dict(baseline=1., candidate=.95, delta=-.05, relative=-.05) for n in analyze.METRIC_NAMES}) for i in range(6)]
        returns = [dict(obstacle_id=i, baseline_required=True, comparable=True, common={
            'baseline': {'status': 'returned', 'restricted_response_s': .4},
            'candidate': {'status': 'returned', 'restricted_response_s': .3}}) for i in range(6)]
        cells.append(dict(track_id=track, seed=seed, subgroup='ordinary_controls',
            road_geometry_sha256=f'road-{seed}', parity={'matched': True}, windows=windows, returns=returns, **arms))
    return cells


def sampled_episode(mode, track=1, seed=3184000013):
    obstacles: list[dict[str, Any]] = [dict(id=i, radius=1., station=40. + 100 * i,
        anchor_index=40 + 100 * i, x=40. + 100 * i, y=0., tangent=[1., 0.]) for i in range(6)]
    catalog = dict(track=[[0., 0., 0., 0.], [0., 0., float(600 + seed % 100), 0.]], obstacles=obstacles)

    def state(tick):
        x = float(tick)
        return dict(t=tick / 50., x=x, y=0., yaw=0., speed=50., station=x, road_index=tick,
            lateral=.5, heading_error=0., rear=[x - o['station'] - 2. for o in obstacles],
            front=[x - o['station'] + 2. for o in obstacles], clearance=[10.] * 6,
            contacts=[], wheel_road_contacts=[1] * 4, dynamic_state=[], wheel_state=[], environment_state=dict(
                off_track_counter=0, damage=0., reward=0., prev_reward=0., tile_visited_count=1, fuel_spent=0.))

    states: list[dict[str, Any]] = [state(0)]
    trace: list[dict[str, Any]] = []
    for step in range(1, 51):
        pre, post = state((step - 1) * 4), state(step * 4)
        action = [.1, 1., 0.]
        info = dict(active=False, baseline_threat=False, baseline_clear_observed=True,
            reason='baseline_clear', baseline_action=action.copy(), baseline_clearance=10., chosen_clearance=10.,
            candidate_count=0, encounter_id=0, encounter_actions=0, threat_free_decisions=step)
        controller: dict[str, Any] = dict(shield=info, baseline_probe_action=action.copy(), crossing_probe_action=action.copy(),
            nominal_probe_calls=step, crossing_probe_calls=step, executed_action_observations=0,
            candidate_state_keys=list(evaluate.STATE_KEYS) if mode == evaluate.ARMS[1] else [],
            candidate_driver_hidden=True, candidate_feedback_method=False,
            evaluation_only=dict(pre=pre, post=post, observation_sha256=f'p{step-1}', post_observation_sha256=f'p{step}',
                decision_reward=0., off_track_count_before=0, off_track_count_after=0,
                simulator_terminated=False, wrapper_crashed=False, wrapper_off_track=False))
        if mode == evaluate.ARMS[1]:
            controller.update(magnitude_baseline_action=action.copy(), magnitude_changed=False,
                              magnitude_reason='no_obstacle', magnitude_proposal=None, final_steer=.1)
        trace.append(dict(step=step, steer=.1, gas=1., brake=0., collision=False, damage=0.,
            damage_telemetry_valid=True, collision_telemetry_valid=True, controller=controller))
        states.extend(dict(state(tick), step=step, collision=False) for tick in range((step - 1) * 4 + 1, step * 4 + 1))
    episode: dict[str, Any] = dict(mode=mode, track_id=track, seed=seed, repeat=0, completed=True,
        error=None, invalid_actions=0, retire_reason=None, damage=0., collisions=0, lapTimeMs=4000,
        damage_telemetry_valid=True, collision_telemetry_valid=True, steps=50,
        catalog=catalog, geometry_sha256=hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
        initial_state=states[0], initial_observation_sha256='p0', decision_trace=trace,
        versions=dict(numpy='1.26.0', cv2='4.8.1', torch='2.1.0+cpu'))
    episode['physical'] = evaluate.shield.legacy.physical_metrics(states, obstacles)
    return episode, states


def pin(path):
    return dict(file=path.name, sha256=evaluate.sha(path), bytes=path.stat().st_size)


def synthetic_run(directory, *, partial=False):
    protocol: dict[str, Any] = dict(schedule=evaluate.scheduled_slots(), python='unused', scope='consumed TRAIN',
        gate=evaluate.GATE, metric_definitions=evaluate.METRICS, model_source_sha256={}, environment_sha256={},
        resource_plan=dict(forecast=dict(peak_child_bytes=1000)),
        source_inventory={'scripts/analyze_koi_avoidance_magnitude.py': {'sha256': evaluate.sha(analyze.__file__)}})
    preflight = dict(protocol_sha256='pin', environment_resets=0,
        arms=[dict(mode=a, import_only=True, environment_resets=0, current_resident_rss_bytes=100,
            candidate_state_keys=list(evaluate.STATE_KEYS) if a == evaluate.ARMS[1] else [],
            candidate_driver_hidden=True, candidate_feedback_method=False) for a in evaluate.ARMS])
    evaluate.save(directory / 'preflight.json', preflight)
    evaluate.save(directory / 'run-start.json', dict(protocol_sha256='pin', preflight_sha256=evaluate.sha(directory / 'preflight.json')))
    events, rows = [], copy.deepcopy(protocol['schedule'])
    for index, row in enumerate(rows):
        if partial and index > 1:
            continue
        path = directory / row['file']
        events.append(dict(row, event='reset_intent', reset_index=1, protocol_sha256='pin'))
        process = dict(command=evaluate.worker_command(directory, index, 'pin', 'unused'),
                       protocol_sha256='pin', wall_time_s=1., error='child_timeout' if partial and index == 1 else None)
        evaluate.save(path.with_suffix('.process.json'), process)
        row['process_sha256'] = evaluate.sha(path.with_suffix('.process.json'))
        if partial and index == 1:
            path.with_suffix('.raw.jsonl').write_text('{"partial":true}\n')
            row.update(status='operator_error', error='child_timeout',
                partial_artifacts=[pin(p) for p in sorted(directory.glob(path.stem + '.*'))])
            continue
        episode, states = sampled_episode(row['mode'], row['track_id'], row['seed'])
        raw, decisions = path.with_suffix('.raw.jsonl'), path.with_suffix('.decisions.jsonl')
        raw.write_text(''.join(json.dumps(s) + '\n' for s in states[1:]))
        decisions.write_text(''.join(json.dumps(dict(step=d['step'], action=analyze.shield.action(d), controller=d['controller'])) + '\n'
                                     for d in episode['decision_trace']))
        episode.update(protocol_sha256='pin', raw_trace_file=raw.name, raw_trace_sha256=evaluate.sha(raw),
                       decision_stream_file=decisions.name, decision_stream_sha256=evaluate.sha(decisions))
        evaluate.save(path, episode)
        path.with_suffix('.stdout.txt').write_text('')
        path.with_suffix('.stderr.txt').write_text('')
        path.with_suffix('.resources.jsonl').write_text('{"synthetic":true}\n')
        row.update(status='completed', sha256=evaluate.sha(path), artifacts=[pin(p) for p in sorted(directory.glob(path.stem + '.*'))])
        events.append(dict(row, event='episode_end', protocol_sha256='pin'))
    ledger = directory / 'reset-ledger.jsonl'
    ledger.write_text(''.join(json.dumps(e) + '\n' for e in events))
    report: dict[str, Any] = dict(protocol_sha256='pin', rows=rows, operator_error='child_timeout' if partial else None,
                  operator_artifacts=[pin(directory / n) for n in ('run-start.json', 'preflight.json', 'reset-ledger.jsonl')])
    evaluate.save(directory / 'episode-report.json', report)
    return protocol, report


class AnalyzerTest(unittest.TestCase):
    def test_all_fixed_gates_and_twenty_four_object_denominators(self):
        result = analyze.summarize(passing_cells(), True)
        self.assertTrue(result['gate_passed'], result['checks'])
        self.assertEqual(result['baseline_eligible_windows'], 24)
        self.assertEqual(result['common_return_cohort_n'], 24)
        self.assertEqual(result['mutually_finished_lap_pairs'], 4)
        self.assertEqual(result['aggregate_arms']['candidate']['safety_objects'], 24)
        self.assertEqual(len(result['metric_denominator_coverage']), 4)
        self.assertEqual(set(result['ordinary_equal_geometry_relative']['path']), {'3184000013', '3184000015'})

    def test_each_safety_coverage_state_and_efficiency_failure_rejects(self):
        changes = (
            ('no_lost_finish', lambda c: c[0]['candidate'].update(finished=False, lap_ms=None)),
            ('per_cell_damage_collision_nonincrease', lambda c: c[0]['candidate'].update(damage=.2)),
            ('per_cell_damage_collision_nonincrease', lambda c: c[0]['candidate'].update(collision_decisions=1)),
            ('per_cell_damage_collision_nonincrease', lambda c: c[0]['candidate']['physical'].update(physical_contact_events=1)),
            ('no_new_hit', lambda c: c[0]['candidate']['objects'][5].update(hit=True)),
            ('no_new_hit', lambda c: c[0]['candidate'].update(unassociated_collision=True)),
            ('unchanged_shield_contract', lambda c: c[0]['baseline']['shield'].update(contract_valid=False)),
            ('stateless_magnitude_action_decomposition', lambda c: c[0]['candidate']['treatment'].update(contract_valid=False)),
            ('baseline_windows_retained', lambda c: c[0]['windows'][0].update(candidate_eligible=False, metrics={})),
            ('return_coverage_retained', lambda c: c[0]['returns'][0].update(comparable=False)),
            ('return_censors_nonincrease', lambda c: c[0]['returns'][0]['common']['candidate'].update(status='censored_episode_end')),
            ('common_return_bound_nonincrease', lambda c: c[0]['returns'][0]['common']['candidate'].update(restricted_response_s=20.)),
            ('steering_variation_nonincrease', lambda c: c[0]['windows'][0]['metrics']['steering_variation'].update(delta=2.)),
            ('every_kept_lap_within_20ms', lambda c: c[0]['candidate'].update(lap_ms=20021)),
            ('complete_matched_evidence', lambda c: c[0]['parity'].update(matched=False)),
            ('ordinary_geometry_consistency', lambda c: c[1].update(road_geometry_sha256=c[0]['road_geometry_sha256'])),
        )
        for name, mutate in changes:
            cells = passing_cells()
            mutate(cells)
            with self.subTest(gate=name):
                result = analyze.summarize(cells, True)
                self.assertFalse(result['checks'][name])
                self.assertFalse(result['gate_passed'])
        self.assertFalse(analyze.summarize(passing_cells(), False)['gate_passed'])
        self.assertFalse(analyze.summarize([], False)['gate_passed'])

    def test_equal_cells_not_objects_and_both_geometry_directions_required(self):
        cells = passing_cells()
        cells[0]['windows'] = cells[0]['windows'][:1]
        cells[0]['windows'][0]['metrics']['path']['relative'] = .001
        result = analyze.summarize(cells, True)
        self.assertAlmostEqual(result['ordinary_equal_cell_relative']['path'], (.001 - .05 * 3) / 4)
        self.assertFalse(result['checks']['ordinary_path_reduction'])
        self.assertLess(result['ordinary_equal_cell_relative']['path'], -.001)

    def test_thresholds_zero_denominators_no_survivor_credit_and_kept_laps(self):
        for name, value in (('path', -.0009), ('max_lateral', -.0299), ('steering_integral', -.0299), ('path', None)):
            cells = passing_cells()
            for c in cells:
                for w in c['windows']:
                    w['metrics'][name]['relative'] = value
            with self.subTest(name=name, value=value):
                self.assertFalse(analyze.summarize(cells, True)['checks']['ordinary_' + name + '_reduction'])
        cells = passing_cells()
        cells[0]['windows'][0].update(candidate_eligible=False, metrics={})
        result = analyze.summarize(cells, True)
        self.assertFalse(result['checks']['ordinary_max_lateral_reduction'])
        self.assertEqual(result['metric_denominator_coverage'][0]['metrics']['path']['usable'], 5)
        cells = passing_cells()
        for c in cells:
            c['candidate']['lap_ms'] = 19981
        self.assertFalse(analyze.summarize(cells, True)['checks']['matched_lap_mean_reduction_20ms'])
        cells[0]['candidate'].update(finished=False, lap_ms=None)
        self.assertEqual(analyze.summarize(cells, True)['mutually_finished_lap_pairs'], 3)

    def test_unreached_object_still_counts_for_clean_hit_safety(self):
        cells = passing_cells()
        cells[0]['baseline']['objects'][5]['physical'] = {'status': 'not_reached'}
        cells[0]['candidate']['objects'][5]['hit'] = True
        result = analyze.summarize(cells, True)
        self.assertEqual(result['new_hits'][0]['obstacle_id'], 5)
        self.assertEqual(result['aggregate_arms']['baseline']['safety_objects'], 24)

    def test_exact_held_action_integral_jump_variation_and_partial_last_hold(self):
        trace: list[dict[str, Any]] = [dict(steer=v, controller={'evaluation_only': {'pre': {'t': a}, 'post': {'t': b}}})
                 for a, b, v in ((0., .08, .1), (.08, .16, -.2), (.16, .18, .3))]
        result = analyze.steering_window(trace, .04, .17)
        self.assertAlmostEqual(result['integral_abs_s'], .004 + .016 + .003)
        self.assertAlmostEqual(result['variation'], .8)
        self.assertEqual(analyze.steering_window(trace, .08, .16)['variation'], 0.)
        trace[1]['controller']['evaluation_only']['pre']['t'] = .09
        self.assertNotEqual(analyze.steering_window(trace, .04, .17)['status'], 'complete')

    def test_censored_sustained_return_and_common_horizon_bound(self):
        episode, states = sampled_episode(evaluate.ARMS[1])
        obstacle = episode['catalog']['obstacles'][0]
        passage = analyze.nominal.physical.passage(states, obstacle, 0, episode['decision_trace'])
        clear = passage['pass_t']
        for s in states:
            s['lateral'] = 2. if s['t'] < clear + .08 else .5
        row = analyze.nominal.returns.prospective_return(states, obstacle, passage, episode['catalog']['obstacles'], episode, .2)
        self.assertTrue(row['status'].startswith('censored_'))
        self.assertIsNotNone(row['pending_centerline_run'])
        self.assertEqual(row['restricted_response_s'], 0.)
        longer = analyze.nominal.returns.prospective_return(states, obstacle, passage, episode['catalog']['obstacles'], episode, .4)
        self.assertEqual(longer['status'], 'returned')
        self.assertAlmostEqual(longer['return_time_s'], .08)

    def test_raw_metrics_and_parity_use_all_six_objects(self):
        b, bs = sampled_episode(evaluate.ARMS[0])
        c, cs = sampled_episode(evaluate.ARMS[1])
        result = analyze.pair_metrics(b, c, bs, cs)
        self.assertTrue(result['parity']['matched'])
        self.assertEqual(len(result['candidate']['objects']), 6)
        self.assertEqual(len(result['windows']), 6)
        self.assertEqual(len(result['returns']), 6)
        self.assertEqual(result['candidate']['treatment']['magnitude_changed_decisions'], 0)
        self.assertTrue(result['candidate']['treatment']['contract_valid'])
        self.assertEqual(result['candidate']['raw_ticks'], 200)
        self.assertAlmostEqual(result['candidate']['steering']['integral_abs_s'], .4)

    def test_primary_all_eight_consumed_wrapper_phase_transitions_are_passive(self):
        directory = evaluate.ROOT / 'runs/koi-nominal-trajectory-v1'
        report = evaluate.read_json(directory / 'episode-report.json')
        checked = 0
        for row in report['rows']:
            if (row['track_id'], row['seed']) not in evaluate.CELLS:
                continue
            episode, states = analyze.shield.load_episode(directory, row, report['protocol_sha256'])
            self.assertEqual(episode['decision_trace'][0]['controller']['evaluation_only']['post']['environment_state']['off_track_counter'], 1)
            self.assertEqual(states[4]['environment_state']['off_track_counter'], 0)
            analyze.validate_wrapper_phases(episode, states)
            checked += 1
        self.assertEqual(checked, 8)

    def test_collision_damage_post_update_binds_raw_pre_state_and_headline(self):
        episode, states = sampled_episode(evaluate.ARMS[1])
        episode['damage'] = .2
        episode['decision_trace'][0]['collision'] = True
        for state in states[5:]:
            state['environment_state']['damage'] = .2
        for index, decision in enumerate(episode['decision_trace']):
            observed = decision['controller']['evaluation_only']
            if index:
                observed['pre']['environment_state']['damage'] = .2
            observed['post']['environment_state']['damage'] = .2
            decision['damage'] = .2
        analyze.validate_wrapper_phases(episode, states)
        episode['damage'] = 0.
        with self.assertRaisesRegex(ValueError, 'headline damage'):
            analyze.validate_wrapper_phases(episode, states)
        episode['damage'] = .2
        states[1]['environment_state']['damage'] = .2
        with self.assertRaisesRegex(ValueError, 'pre-update'):
            analyze.validate_wrapper_phases(episode, states)

    def test_counter_and_playfield_reward_phase_transitions(self):
        episode, states = sampled_episode(evaluate.ARMS[1])
        decision = episode['decision_trace'][0]
        episode['decision_trace'] = [decision]
        states = states[:5]
        for tick, state in enumerate(states[1:], 1):
            state['environment_state'].update(reward=-tick * .1, prev_reward=-tick * .1)
        observed = decision['controller']['evaluation_only']
        observed['post']['environment_state'].update(reward=-.4, prev_reward=-.4, off_track_counter=1)
        observed.update(decision_reward=sum([-0.1, -0.1, -.3 + .2, -.4 + .3]), off_track_count_after=1)
        analyze.validate_wrapper_phases(episode, states)
        observed['simulator_terminated'] = True
        observed['decision_reward'] = sum([-0.1, -0.1, -.3 + .2, -100.])
        analyze.validate_wrapper_phases(episode, states)
        observed['off_track_count_after'] = 0
        with self.assertRaisesRegex(ValueError, 'counter phase'):
            analyze.validate_wrapper_phases(episode, states)

    def test_tap_decomposition_counts_nominal_shield_overlap_and_cancellation(self):
        episode, _ = sampled_episode(evaluate.ARMS[1])
        d = episode['decision_trace'][10]
        c = d['controller']
        c['baseline_probe_action'] = [.05, 1., 0.]
        c['shield']['baseline_action'] = [.05, 1., 0.]
        c.update(magnitude_changed=True, final_steer=.05, magnitude_reason='bounded_magnitude',
            magnitude_proposal=dict(required_clearance_m=.5, approach_m=3., ttc_s=.1, geometric_demand=.1,
                collision_risk=.2, legacy_magnitude=.34, magnitude=.1, side=1., current_bbox_m=[-1., 3., 1., 4.], projected_shift_m=0.))
        result = analyze.treatment_metrics(episode['decision_trace'], evaluate.ARMS[1])
        self.assertTrue(result['contract_valid'], result['contract_errors'])
        self.assertEqual(result['magnitude_changed_decisions'], 1)
        self.assertEqual(result['shield_override_decisions'], 1)
        self.assertEqual(result['magnitude_shield_overlap_decisions'], 1)
        self.assertEqual(result['magnitude_cancelled_by_shield_decisions'], 1)
        self.assertEqual(result['final_vs_crossing_changed_decisions'], 0)

    def test_corrupt_nominal_pedals_diagnostics_state_feedback_and_binding_reject(self):
        changes = ({'executed_action_observations': 1}, {'candidate_feedback_method': True},
            {'candidate_state_keys': ['nominal', 'last', 'plan']}, {'candidate_driver_hidden': False},
            {'magnitude_baseline_action': [.9, 1., 0.]}, {'magnitude_changed': True}, {'final_steer': .9},
            {'baseline_probe_action': [.1, .5, 0.]}, {'nominal_probe_calls': 2},
            {'magnitude_proposal': {'collision_risk': .2}})
        for change in changes:
            episode, _ = sampled_episode(evaluate.ARMS[1])
            episode['decision_trace'][0]['controller'].update(change)
            with self.subTest(change=change):
                self.assertFalse(analyze.treatment_metrics(episode['decision_trace'], evaluate.ARMS[1])['contract_valid'])

    def test_raw_corruption_and_different_initial_or_prefix_reject(self):
        for mutation in ('geometry', 'raw_collision', 'endpoint', 'environment'):
            episode, states = sampled_episode(evaluate.ARMS[1])
            if mutation == 'geometry':
                episode['geometry_sha256'] = 'different'
            elif mutation == 'raw_collision':
                states[1]['collision'] = True
            else:
                field = 'x' if mutation == 'endpoint' else 'environment_state'
                episode['decision_trace'][0]['controller']['evaluation_only']['post'][field] = 999. if field == 'x' else {'damage': .2}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze.episode_metrics(episode, states)
        b, bs = sampled_episode(evaluate.ARMS[0])
        c, cs = sampled_episode(evaluate.ARMS[1])
        c['initial_observation_sha256'] = 'different'
        self.assertFalse(analyze.pair_metrics(b, c, bs, cs)['parity']['matched'])

    def test_complete_synthetic_primary_bindings_report_four_cells_all_objects(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            directory = Path(temporary)
            protocol, _ = synthetic_run(directory)
            with patch.object(evaluate, 'validate_frozen', return_value=protocol):
                result = analyze.analyze(directory, 'pin')
        self.assertTrue(result['complete'])
        self.assertEqual((result['valid_episodes'], result['matched_pairs'], result['safety_objects_per_arm']), (8, 4, 24))
        self.assertEqual(len(result['cell_coverage']), 4)
        self.assertEqual(result['ledger']['reset_intents'], 8)
        self.assertTrue(result['checks']['ordinary_geometry_consistency'])
        self.assertEqual(result['verdict'], 'REJECTED')
        self.assertFalse(result['repeat_authorized'])
        self.assertEqual(result['environment_resets'], 0)

    def test_partial_and_unmatched_evidence_remain_pinned_and_cannot_pass(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            directory = Path(temporary)
            protocol, report = synthetic_run(directory, partial=True)
            with patch.object(evaluate, 'validate_frozen', return_value=protocol):
                result = analyze.analyze(directory, 'pin')
                self.assertFalse(result['complete'])
                self.assertEqual(result['unmatched_valid_episodes'], 1)
                self.assertEqual(len(result['unmatched_episodes'][0]['metrics']['objects']), 6)
                partial = directory / report['rows'][1]['partial_artifacts'][0]['file']
                partial.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'hash/size'):
                    analyze.analyze(directory, 'pin')

    def test_extra_unreported_artifact_and_changed_analyzer_source_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            directory = Path(temporary)
            protocol, report = synthetic_run(directory)
            (directory / report['rows'][0]['file']).with_suffix('.unreported.txt').write_text('retained')
            with patch.object(evaluate, 'validate_frozen', return_value=protocol):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    analyze.analyze(directory, 'pin')
            protocol['source_inventory']['scripts/analyze_koi_avoidance_magnitude.py']['sha256'] = 'changed'
            with patch.object(evaluate, 'validate_frozen', return_value=protocol):
                with self.assertRaisesRegex(ValueError, 'analyzer'):
                    analyze.analyze(directory, 'pin')

    def test_wrong_child_command_rejected_even_with_rehashed_receipt(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            directory = Path(temporary)
            protocol, report = synthetic_run(directory)
            row = report['rows'][0]
            process = (directory / row['file']).with_suffix('.process.json')
            data = evaluate.read_json(process)
            data['command'] = ['unexpected', 'child']
            process.write_text(json.dumps(data))
            row['process_sha256'] = evaluate.sha(process)
            row['artifacts'] = [pin(process) if p['file'] == process.name else p for p in row['artifacts']]
            (directory / 'episode-report.json').write_text(json.dumps(report))
            with patch.object(evaluate, 'validate_frozen', return_value=protocol):
                with self.assertRaisesRegex(ValueError, 'process command'):
                    analyze.analyze(directory, 'pin')

    def test_duplicate_reset_rejected_even_with_rehashed_ledger(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            directory = Path(temporary)
            protocol, report = synthetic_run(directory)
            ledger = directory / 'reset-ledger.jsonl'
            events = [json.loads(line) for line in ledger.read_text().splitlines()]
            events.insert(1, dict(events[0], reset_index=2))
            ledger.write_text(''.join(json.dumps(e) + '\n' for e in events))
            report['operator_artifacts'] = [pin(ledger) if p['file'] == ledger.name else p for p in report['operator_artifacts']]
            (directory / 'episode-report.json').write_text(json.dumps(report))
            with patch.object(evaluate, 'validate_frozen', return_value=protocol):
                with self.assertRaisesRegex(ValueError, 'reset/retry'):
                    analyze.analyze(directory, 'pin')

    def test_natural_dnf_metrics_not_dropped_or_treated_as_efficiency_credit(self):
        b, bs = sampled_episode(evaluate.ARMS[0])
        c, cs = sampled_episode(evaluate.ARMS[1])
        b.update(completed=False, retire_reason='off_track', lapTimeMs=None)
        c.update(completed=False, retire_reason='off_track', lapTimeMs=None)
        result = analyze.pair_metrics(b, c, bs, cs)
        self.assertTrue(result['parity']['matched'])
        self.assertFalse(result['candidate']['finished'])
        self.assertIsNone(result['candidate']['lap_ms'])
        self.assertEqual(len(result['candidate']['objects']), 6)
        self.assertEqual(result['candidate']['objects'][5]['prospective_return']['status'], 'unpassed')
        self.assertTrue(result['candidate']['objects'][5]['prospective_return']['dnf'])


if __name__ == '__main__':
    unittest.main()
