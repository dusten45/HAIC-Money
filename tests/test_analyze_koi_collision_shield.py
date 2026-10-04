"""Synthetic sampled-state evidence only; no policy or simulator construction."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_koi_collision_shield as analyze
from scripts import evaluate_koi_collision_shield as evaluate


def state(index) -> dict:
    return dict(t=index / 50., x=float(index), y=0., yaw=0., heading_error=0., lateral=0.,
                wheel_road_contacts=[1] * 4, rear=[-1.], clearance=[1.], contacts=[],
                dynamic_state=[dict(velocity=[1., 0.])], wheel_state=[dict(omega=1.)], environment_state={})


def episode(mode=evaluate.ARMS[1], count=3, active=()) -> tuple[dict, list[dict]]:
    states = [state(0)]
    trace = []
    for index in range(1, count + 1):
        pre = state((index - 1) * 4)
        post = state(index * 4)
        changed = index in active
        shield = dict(active=changed, reason='collision_shield' if changed else 'baseline_clear',
                      baseline_action=[0., 1., 0.], baseline_threat=changed,
                      baseline_clear_observed=not changed, baseline_clearance=-.1 if changed else 1.,
                      chosen_clearance=.2 if changed else 1., candidate_count=1,
                      encounter_id=1, encounter_actions=1 if changed else 0, threat_free_decisions=0)
        controller = dict(shield=shield if mode == evaluate.ARMS[1] else None,
                          baseline_probe_action=[0., 1., 0.], evaluation_only=dict(
                          pre=pre, post=post, observation_sha256=f'pixel{index-1}',
                          post_observation_sha256=f'pixel{index}'))
        trace.append(dict(step=index, steer=.1 if changed else 0., gas=1., brake=0., controller=controller))
        states.extend(dict(state(tick), step=index, collision=False) for tick in range((index - 1) * 4 + 1, index * 4 + 1))
    result: dict = dict(mode=mode, track_id=1, seed=3184000002, repeat=0, completed=True,
                  lapTimeMs=20000, retire_reason=None, error=None, invalid_actions=0,
                  damage=0., collisions=0, decision_trace=trace, catalog={'obstacles': [{'radius': 1.}]},
                  off_track_count_max=0, geometry_sha256='geometry', initial_state=states[0],
                  initial_observation_sha256='pixel0', protocol_sha256='protocol')
    result['physical'] = evaluate.legacy.physical_metrics(states, result['catalog']['obstacles'])
    return result, states


def cell() -> dict:
    baseline, bstates = episode(evaluate.ARMS[0])
    candidate, cstates = episode(active=(2,))
    b, c = analyze.episode_metrics(baseline, bstates), analyze.episode_metrics(candidate, cstates)
    b['damage'], b['collision_decisions'], b['hit_objects'] = .2, 1, [0]
    return dict(seed=3184000002, baseline=b, candidate=c, lost_finish=False, gained_finish=False,
                new_hits=[], parity=analyze.pair_metrics(baseline, candidate, bstates, cstates))


class ShieldAnalyzerTest(unittest.TestCase):
    def test_safe_noop_is_exact_through_terminal_state_and_pixels(self):
        baseline, states = episode(evaluate.ARMS[0])
        candidate, cstates = episode()
        parity = analyze.pair_metrics(baseline, candidate, states, cstates)
        self.assertTrue(parity['matched'])
        self.assertTrue(parity['no_divergence_full_episode_equal'])
        candidate['decision_trace'][-1]['controller']['evaluation_only']['post_observation_sha256'] = 'changed'
        self.assertFalse(analyze.pair_metrics(baseline, candidate, states, cstates)['matched'])

    def test_prefix_includes_first_changed_action_pre_state(self):
        baseline, states = episode(evaluate.ARMS[0])
        candidate, cstates = episode(active=(2,))
        parity = analyze.pair_metrics(baseline, candidate, states, cstates)
        self.assertTrue(parity['matched'])
        self.assertEqual(parity['equal_action_prefix_decisions'], 1)
        candidate['decision_trace'][1]['controller']['evaluation_only']['pre']['wheel_state'][0]['omega'] = 9.
        self.assertFalse(analyze.pair_metrics(baseline, candidate, states, cstates)['matched'])

    def test_hidden_raw_prefix_difference_is_not_masked_by_decision_equality(self):
        baseline, states = episode(evaluate.ARMS[0])
        candidate, cstates = episode(active=(2,))
        cstates[1]['dynamic_state'][0]['velocity'][0] = 99.
        self.assertFalse(analyze.pair_metrics(baseline, candidate, states, cstates)['matched'])

    def test_equal_prefix_with_unequal_lengths_cannot_pass_as_noop(self):
        baseline, states = episode(evaluate.ARMS[0], count=3)
        candidate, cstates = episode(count=2)
        self.assertFalse(analyze.pair_metrics(baseline, candidate, states, cstates)['matched'])

    def test_intervention_counts_actual_hold_durations_and_censors(self):
        candidate, _ = episode(count=5, active=(2, 3, 5))
        trace = candidate['decision_trace']
        trace[-1]['controller']['evaluation_only']['post']['t'] -= .06
        result = analyze.shield_metrics(trace)
        self.assertTrue(result['contract_valid'])
        self.assertEqual(result['intervention_decisions'], 3)
        self.assertEqual(result['intervention_bursts'], 2)
        self.assertEqual(result['max_consecutive_decisions'], 2)
        self.assertAlmostEqual(result['intervention_duration_s'], .18)
        self.assertAlmostEqual(result['max_consecutive_duration_s'], .16)
        self.assertTrue(result['bursts'][-1]['terminal_censored'])
        self.assertEqual(result['encounter_action_counts'], [3])

    def test_budget_does_not_reset_on_one_or_two_clear_decisions(self):
        for gap in (1, 2):
            candidate, _ = episode(count=7 + gap, active=tuple(range(1, 7)) + (7 + gap,))
            result = analyze.shield_metrics(candidate['decision_trace'])
            self.assertFalse(result['contract_valid'])
            self.assertEqual(result['max_encounter_actions'], 7)

    def test_three_observed_clear_rearms_but_dropout_does_not(self):
        candidate, _ = episode(count=10, active=tuple(range(1, 7)) + (10,))
        result = analyze.shield_metrics(candidate['decision_trace'])
        self.assertTrue(result['contract_valid'])
        self.assertEqual(result['encounter_action_counts'], [6, 1])
        for decision in candidate['decision_trace'][6:9]:
            decision['controller']['shield']['baseline_clear_observed'] = False
        self.assertFalse(analyze.shield_metrics(candidate['decision_trace'])['contract_valid'])

    def test_noop_pedal_and_diagnostic_lies_are_rejected(self):
        for mutation in ('active', 'pedals', 'reported_baseline', 'safe_change', 'negative_chosen_gap'):
            candidate, _ = episode(active=(2,))
            row = candidate['decision_trace'][1]
            info = row['controller']['shield']
            if mutation == 'active':
                info['active'] = False
            elif mutation == 'pedals':
                row['gas'] = .5
            elif mutation == 'reported_baseline':
                info['baseline_action'][0] = .7
            elif mutation == 'safe_change':
                info['baseline_threat'] = False
                info['baseline_clear_observed'] = True
            else:
                info['chosen_clearance'] = -.2
            with self.subTest(mutation=mutation):
                self.assertFalse(analyze.shield_metrics(candidate['decision_trace'])['contract_valid'])

    def test_missing_clear_observed_field_is_not_silent_false(self):
        candidate, _ = episode()
        del candidate['decision_trace'][0]['controller']['shield']['baseline_clear_observed']
        with self.assertRaisesRegex(ValueError, 'boolean'):
            analyze.shield_metrics(candidate['decision_trace'])

    def test_sampled_metrics_include_dnf_unpassed_object_and_angular_wrap(self):
        candidate, states = episode()
        candidate.update(completed=False, retire_reason='off_track', lapTimeMs=None)
        states[0]['yaw'], states[1]['yaw'] = math.pi - .01, -math.pi + .01
        states[2]['clearance'] = [-.1]
        states[2]['lateral'] = -3.
        states[-1]['wheel_road_contacts'] = [0] * 4
        candidate['physical'] = evaluate.legacy.physical_metrics(states, candidate['catalog']['obstacles'])
        result = analyze.episode_metrics(candidate, states)
        self.assertFalse(result['finished'])
        self.assertEqual(result['hit_objects'], [0])
        self.assertIsNone(result['objects'][0]['passage_t'])
        self.assertEqual(result['min_clearance_m'], -.1)
        self.assertEqual(result['max_abs_lateral_m'], 3.)
        self.assertEqual(result['all_wheels_offroad_ticks'], 1)
        self.assertTrue(result['all_wheels_offroad_runs'][-1]['terminal_censored'])
        self.assertEqual(result['path_length_m'], 12.)
        self.assertLess(result['total_abs_heading_change_rad'], 3.2)

    def test_gate_requires_clear_benefit_and_all_safety_checks(self):
        base = cell()
        self.assertTrue(all(analyze.gate_result([base], True).values()))
        changes = (
            lambda c: c.update(lost_finish=True),
            lambda c: c.update(new_hits=[1]),
            lambda c: c['candidate'].update(damage=.4),
            lambda c: c['candidate'].update(collision_decisions=2),
            lambda c: c['candidate'].update(all_wheels_offroad_ticks=100),
            lambda c: c['candidate'].update(max_abs_heading_error_rad=1.),
            lambda c: c['candidate'].update(max_abs_lateral_m=4.),
            lambda c: c['candidate'].update(path_length_m=100.),
            lambda c: c['candidate'].update(lap_ms=21000),
            lambda c: c['candidate'].update(damage=.2, collision_decisions=1),
        )
        for change in changes:
            changed = copy.deepcopy(base)
            change(changed)
            with self.subTest(change=change):
                self.assertFalse(all(analyze.gate_result([changed], True).values()))
        self.assertFalse(all(analyze.gate_result([base], False).values()))
        self.assertFalse(all(analyze.gate_result([], True).values()))

    def test_finish_losses_not_hidden_by_mutually_finished_lap_or_damage(self):
        kept, lost = cell(), cell()
        lost.update(lost_finish=True)
        lost['candidate'].update(finished=False, lap_ms=None, path_length_m=1.)
        aggregate = analyze.aggregate([kept, lost])
        self.assertEqual(aggregate['matched_lap_pairs'], 1)
        self.assertEqual(aggregate['lost'], 1)
        self.assertEqual(aggregate['preserved'], 1)
        self.assertFalse(all(analyze.gate_result([kept, lost], True).values()))

    def test_tampered_physical_metrics_and_nonfinite_evidence_fail(self):
        candidate, states = episode()
        candidate['physical']['min_fixture_clearance'] = 9.
        with self.assertRaisesRegex(ValueError, 'physical metrics'):
            analyze.episode_metrics(candidate, states)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            analyze.finite_tree({'nested': [float('nan')]})

    def test_missing_reset_receipt_cannot_be_called_complete(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            row = dict(evaluate.scheduled_slots()[0], status='completed')
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'lacks reset'):
                analyze.validate_ledger(directory, [row], 'protocol')
            events = [dict(row, event='reset_intent', reset_index=1, protocol_sha256='protocol'),
                      dict(row, event='episode_end', protocol_sha256='protocol')]
            (directory / 'reset-ledger.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
            self.assertEqual(analyze.validate_ledger(directory, [row], 'protocol')['reset_intents'], 1)
            with self.assertRaisesRegex(ValueError, 'identity/order'):
                analyze.validate_ledger(directory, [dict(row, seed=38300)], 'protocol')

    def test_full_synthetic_report_recomputes_and_stream_tampering_fails(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as temporary:
            directory = Path(temporary)
            analyzer_copy = directory / 'source/scripts/analyze_koi_collision_shield.py'
            analyzer_copy.parent.mkdir(parents=True)
            analyzer_copy.write_bytes(Path(analyze.__file__).read_bytes())
            rows, events = evaluate.scheduled_slots(), []
            for row in rows:
                record, states = episode(row['mode'])
                record.update(track_id=row['track_id'], seed=row['seed'])
                path = directory / row['file']
                raw, decisions = path.with_suffix('.raw.jsonl'), path.with_suffix('.decisions.jsonl')
                raw.write_text(''.join(json.dumps(s) + '\n' for s in states[1:]))
                decisions.write_text(''.join(json.dumps(dict(step=d['step'], action=analyze.action(d),
                    controller=d['controller'])) + '\n' for d in record['decision_trace']))
                record.update(raw_trace_file=raw.name, raw_trace_sha256=evaluate.sha(raw),
                              decision_stream_file=decisions.name, decision_stream_sha256=evaluate.sha(decisions))
                process = path.with_suffix('.process.json')
                evaluate.save(process, dict(error=None, protocol_sha256='protocol'))
                evaluate.save(path, record)
                row.update(status='completed', sha256=evaluate.sha(path), process_sha256=evaluate.sha(process))
                events.extend([dict(row, event='reset_intent', reset_index=1, protocol_sha256='protocol'),
                               dict(row, event='episode_end', protocol_sha256='protocol')])
            (directory / 'reset-ledger.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
            evaluate.save(directory / 'episode-report.json', dict(rows=rows, operator_error=None, protocol_sha256='protocol'))
            protocol = dict(schedule=evaluate.scheduled_slots(), subgroups=evaluate.SUBGROUPS,
                            gate=evaluate.GATE, scope=evaluate.legacy.SCOPE,
                            comparator_zip_sha256=evaluate.legacy.CROSSING_SHA, wrapper_sha256='wrapper',
                            consumed_evidence=dict(selection='Synthetic fixture', subgroup_definition='Synthetic roles'))
            with patch.object(evaluate, 'validate_frozen', return_value=protocol):
                result = analyze.analyze(directory, 'protocol')
                self.assertTrue(result['complete'])
                self.assertEqual(result['matched_pairs'], 6)
                self.assertEqual(result['aggregate']['preserved'], 6)
                self.assertEqual(result['subgroups']['ordinary_controls']['cells'], 2)
                self.assertEqual(result['subgroups']['corner_conflict_challenges']['cells'], 4)
                self.assertEqual(result['verdict'], 'NOT_ADOPTED')  # Exact no-op is not improvement.
                self.assertTrue(result['checks']['complete_matched_evidence'])
                self.assertTrue(result['checks']['noop_pedals_projection_and_encounter_budget'])
                self.assertEqual(result['ledger']['reset_intents'], 12)
                tampered = (directory / rows[0]['file']).with_suffix('.raw.jsonl')
                tampered.write_text(tampered.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'stream hash'):
                    analyze.analyze(directory, 'protocol')


if __name__ == '__main__':
    unittest.main()
