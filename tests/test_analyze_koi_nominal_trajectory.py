"""Synthetic strict gates/window tests; no policy or environment execution."""

import copy
import hashlib
import json
import unittest
from typing import Any

from scripts import analyze_koi_nominal_trajectory as analyze
from scripts import evaluate_koi_nominal_trajectory as evaluate


def passing_cells():
    cells = []
    for track, seed in evaluate.CELLS:
        arms = {arm: dict(finished=True, damage=0., collision_decisions=0, lap_ms=20000 if arm == 'baseline' else 19960,
                    physical={'physical_contact_events': 0}, shield={'contract_valid': True},
                    nominal_contract_errors=[], unassociated_collision=False,
                    objects=[dict(obstacle_id=i, hit=False) for i in range(6)]) for arm in ('baseline', 'candidate')}
        metrics = {name: dict(baseline=1., candidate=.95, relative=-.05, delta=-.05) for name in analyze.METRIC_NAMES}
        cells.append(dict(track_id=track, seed=seed, subgroup=evaluate.SUBGROUPS[f'{track}:{seed}'],
                          parity={'matched': True}, **arms,
                          windows=[dict(obstacle_id=0, baseline_eligible=True, candidate_eligible=True, metrics=metrics)],
                          returns=[dict(baseline_required=True, comparable=True, common={
                              'baseline': {'status': 'returned', 'restricted_response_s': .4},
                              'candidate': {'status': 'returned', 'restricted_response_s': .3}})]))
    return cells


def sampled_episode(mode):
    obstacles: list[dict[str, Any]] = [dict(id=i, radius=1., station=40. + 100 * i, anchor_index=40 + 100 * i,
                      x=40. + 100 * i, y=0., tangent=[1., 0.]) for i in range(6)]
    catalog = dict(track=[[0., 0., 0., 0.], [0., 0., 600., 0.]], obstacles=obstacles)

    def state(tick):
        x = float(tick)
        return dict(t=tick / 50., x=x, y=0., yaw=0., speed=50., station=x, road_index=tick,
                    lateral=.5, heading_error=0., rear=[x - o['station'] - 2. for o in obstacles],
                    front=[x - o['station'] + 2. for o in obstacles], clearance=[10.] * 6,
                    contacts=[], wheel_road_contacts=[1] * 4, dynamic_state=[], wheel_state=[], environment_state={})

    states, trace = [state(0)], []
    for step in range(1, 51):
        pre, post = state((step - 1) * 4), state(step * 4)
        action = [.1, 1., 0.]
        info = dict(active=False, baseline_threat=False, baseline_clear_observed=True,
                    reason='baseline_clear', baseline_action=action, baseline_clearance=10., chosen_clearance=10.,
                    candidate_count=0, encounter_id=0, encounter_actions=0, threat_free_decisions=step)
        controller = dict(shield=info, baseline_probe_action=action, crossing_probe_action=action,
                          nominal_probe_calls=step, crossing_probe_calls=step,
                          executed_action_observations=step if mode == evaluate.ARMS[1] else 0,
                          nominal_changed=False, nominal_reason='no_obstacle', nominal_baseline_action=action,
                          evaluation_only=dict(pre=pre, post=post, observation_sha256=f'p{step-1}',
                                               post_observation_sha256=f'p{step}'))
        trace.append(dict(step=step, steer=.1, gas=1., brake=0., collision=False, controller=controller))
        states.extend(dict(state(tick), step=step, collision=False) for tick in range((step - 1) * 4 + 1, step * 4 + 1))
    episode: dict[str, Any] = dict(mode=mode, completed=True, error=None, invalid_actions=0, retire_reason=None,
                   track_id=1, seed=3184000002, repeat=0, damage=0., collisions=0, lapTimeMs=4000,
                   damage_telemetry_valid=True, collision_telemetry_valid=True, steps=50,
                   catalog=catalog, geometry_sha256=hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
                   initial_state=states[0], initial_observation_sha256='p0', decision_trace=trace)
    episode['physical'] = evaluate.shield.legacy.physical_metrics(states, obstacles)
    return episode, states


class AnalyzerTest(unittest.TestCase):
    def test_all_predeclared_gates_pass_synthetic_reductions(self):
        result = analyze.summarize(passing_cells(), True)
        self.assertTrue(result['gate_passed'], result['checks'])
        self.assertEqual(result['baseline_eligible_windows'], 8)
        self.assertEqual(result['common_return_cohort_n'], 8)
        self.assertEqual(result['mutually_finished_lap_pairs'], 8)
        self.assertEqual(set(result['ordinary_equal_geometry_relative']['path']), {'3184000013', '3184000015'})

    def test_every_strict_safety_coverage_gate_rejects_independently(self):
        changes = (
            ('no_lost_finish', lambda c: c[0]['candidate'].update(finished=False, lap_ms=None)),
            ('per_cell_damage_collision_nonincrease', lambda c: c[0]['candidate'].update(damage=.2)),
            ('per_cell_damage_collision_nonincrease', lambda c: c[0]['candidate'].update(collision_decisions=1)),
            ('no_new_hit', lambda c: c[0]['candidate']['objects'][5].update(hit=True)),
            ('no_new_hit', lambda c: c[0]['candidate'].update(unassociated_collision=True)),
            ('unchanged_shield_contract', lambda c: c[0]['baseline']['shield'].update(contract_valid=False)),
            ('unchanged_shield_contract', lambda c: c[0]['candidate'].update(nominal_contract_errors=['feedback'])),
            ('baseline_windows_retained', lambda c: c[0]['windows'][0].update(candidate_eligible=False)),
            ('return_coverage_retained', lambda c: c[0]['returns'][0].update(comparable=False)),
            ('return_censors_nonincrease', lambda c: c[0]['returns'][0]['common']['candidate'].update(status='censored_episode_end')),
            ('common_return_bound_nonincrease', lambda c: c[0]['returns'][0]['common']['candidate'].update(restricted_response_s=2.)),
            ('steering_variation_nonincrease', lambda c: c[0]['windows'][0]['metrics']['steering_variation'].update(delta=2.)),
            ('every_kept_lap_within_20ms', lambda c: c[0]['candidate'].update(lap_ms=20021)),
            ('complete_matched_evidence', lambda c: c[0]['parity'].update(matched=False)),
        )
        for name, mutate in changes:
            cells = passing_cells()
            mutate(cells)
            result = analyze.summarize(cells, True)
            with self.subTest(gate=name):
                self.assertFalse(result['checks'][name])
                self.assertFalse(result['gate_passed'])
        self.assertFalse(analyze.summarize(passing_cells(), False)['gate_passed'])
        self.assertFalse(analyze.summarize([], False)['gate_passed'])

    def test_clean_unreached_object_still_in_safety_gate(self):
        cells = passing_cells()
        cells[0]['baseline']['objects'][5]['physical'] = {'status': 'not_reached'}
        cells[0]['candidate']['objects'][5]['hit'] = True
        result = analyze.summarize(cells, True)
        self.assertEqual(result['new_hits'][0]['obstacle_id'], 5)
        self.assertFalse(result['checks']['no_new_hit'])

    def test_no_survivor_credit_for_lost_window(self):
        cells = passing_cells()
        cells[-1]['windows'][0].update(candidate_eligible=False, metrics={})
        result = analyze.summarize(cells, True)
        self.assertFalse(result['checks']['baseline_windows_retained'])
        self.assertFalse(result['checks']['ordinary_path_reduction'])
        self.assertEqual(result['baseline_eligible_windows'], 8)

    def test_both_roads_required_despite_three_good_layouts_on_one(self):
        cells = passing_cells()
        for metric in cells[4]['windows'][0]['metrics'].values():
            metric.update(relative=.001, delta=.001)
        result = analyze.summarize(cells, True)
        self.assertLess(result['ordinary_equal_cell_relative']['max_lateral'], -.03)
        self.assertFalse(result['checks']['ordinary_max_lateral_reduction'])

    def test_relative_thresholds_zero_denominators_and_lap_mean(self):
        for name, change in (('path', -.0009), ('max_lateral', -.0299), ('steering_integral', -.0299), ('path', None)):
            cells = passing_cells()
            for cell in cells[4:]:
                cell['windows'][0]['metrics'][name]['relative'] = change
            with self.subTest(name=name, change=change):
                self.assertFalse(analyze.summarize(cells, True)['checks']['ordinary_' + name + '_reduction'])
        cells = passing_cells()
        for cell in cells:
            cell['candidate']['lap_ms'] = 19981
        self.assertFalse(analyze.summarize(cells, True)['checks']['matched_lap_mean_reduction_20ms'])

    def test_actual_steer_integral_and_variation_use_clipped_holds(self):
        trace: list[dict[str, Any]] = [dict(steer=steer, controller={'evaluation_only': {'pre': {'t': a}, 'post': {'t': b}}})
                 for a, b, steer in ((0., .08, .1), (.08, .16, -.2), (.16, .18, .3))]
        result = analyze.steering_window(trace, .04, .17)
        self.assertEqual(result['status'], 'complete')
        self.assertAlmostEqual(result['integral_abs_s'], .004 + .016 + .003)
        self.assertAlmostEqual(result['variation'], .3 + .5)
        self.assertAlmostEqual(analyze.steering_window(trace, .08, .16)['variation'], 0.)
        trace[1]['controller']['evaluation_only']['pre']['t'] = .09
        self.assertNotEqual(analyze.steering_window(trace, .04, .17)['status'], 'complete')

    def test_censored_return_bound_is_not_returned_only_mean(self):
        cells = passing_cells()
        for cell in cells:
            cell['returns'][0]['common']['baseline'].update(status='censored_fixed_time', restricted_response_s=1.76)
            cell['returns'][0]['common']['candidate'].update(status='censored_fixed_time', restricted_response_s=1.76)
        result = analyze.summarize(cells, True)
        self.assertTrue(result['checks']['return_censors_nonincrease'])
        self.assertTrue(result['checks']['common_return_bound_nonincrease'])

    def test_eligibility_requires_passage_and_steering_coverage(self):
        obj = dict(fixed_window={'status': 'complete', 'steering': {'status': 'complete'}},
                   physical={'status': 'passed', 'geometry_valid': True})
        self.assertTrue(analyze.eligible(obj))
        for section, field, value in (('fixed_window', 'status', 'exit_not_reached'), ('physical', 'geometry_valid', False)):
            changed = copy.deepcopy(obj)
            changed[section][field] = value
            self.assertFalse(analyze.eligible(changed))

    def test_raw_episode_metrics_and_common_return_pair(self):
        b, bs = sampled_episode(evaluate.ARMS[0])
        c, cs = sampled_episode(evaluate.ARMS[1])
        paired = analyze.pair_metrics(b, c, bs, cs)
        self.assertTrue(paired['parity']['matched'])
        self.assertTrue(paired['candidate']['shield']['contract_valid'])
        self.assertEqual(paired['candidate']['nominal_contract_errors'], [])
        self.assertEqual(paired['candidate']['raw_ticks'], 200)
        self.assertEqual(paired['candidate']['max_abs_lateral_m'], .5)
        self.assertAlmostEqual(paired['candidate']['path_length_m'], 200.)
        self.assertAlmostEqual(paired['candidate']['steering']['integral_abs_s'], .4)
        self.assertEqual(sum(w['baseline_eligible'] for w in paired['windows']), 2)
        self.assertEqual(sum(r['baseline_required'] for r in paired['returns']), 2)
        self.assertEqual(sum(r['comparable'] for r in paired['returns']), 2)

    def test_raw_corruption_and_nominal_feedback_coverage_rejected(self):
        for mutation in ('geometry', 'raw_collision', 'endpoint', 'feedback', 'pedals'):
            episode, states = sampled_episode(evaluate.ARMS[1])
            if mutation == 'geometry':
                episode['geometry_sha256'] = 'different'
            elif mutation == 'raw_collision':
                states[1]['collision'] = True
            elif mutation == 'endpoint':
                episode['decision_trace'][0]['controller']['evaluation_only']['post']['x'] = 999.
            elif mutation == 'feedback':
                episode['decision_trace'][0]['controller']['executed_action_observations'] = 0
            else:
                episode['decision_trace'][0]['controller']['baseline_probe_action'] = [.1, .5, 0.]
            with self.subTest(mutation=mutation):
                if mutation in ('feedback', 'pedals'):
                    self.assertTrue(analyze.episode_metrics(episode, states)['nominal_contract_errors'])
                else:
                    with self.assertRaises(ValueError):
                        analyze.episode_metrics(episode, states)


if __name__ == '__main__':
    unittest.main()
