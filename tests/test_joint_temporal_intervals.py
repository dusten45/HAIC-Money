"""Synthetic interval-analysis tests only; no cached run reads or simulator imports."""

from copy import deepcopy
from dataclasses import asdict
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from scripts import analyze_joint_temporal_intervals as analysis
from haic.algorithms.joint_control import comparison, interval_comparison, physics


def trajectory_row(seed=analysis.CAL[0], error=.6) -> dict:
    actual = np.zeros((2, 17, 3))
    actual[:, 1:, 0] = error
    actual[:, 1:, 2] = .07
    oracle = actual[:, None, 1:].copy()
    oracle[..., 0] -= .1
    return dict(plan_anchor_id=str(seed), anchor=dict(seed=seed), observer_valid=True,
                comparison=dict(poses=np.zeros((2, 19, 17, 3))),
                actual_trajectories=[dict(poses=p) for p in actual],
                oracle_prediction={k: oracle[..., i] for i, k in enumerate(
                    ('relative_x', 'relative_y', 'yaw_delta'))})


def mapping_window(seed=analysis.CAL[0], index=1, error=(.3, .4, .1), provenance='predicted_corrected_issued_action') -> dict:
    return dict(anchor_id=str(seed), index=index, track_id=1, seed=seed,
                input_identity=f'{seed}-{index}', duplicate_of_prior_causal_input=False,
                snapshot=dict(mapping_motion=dict(valid=True, provenance=provenance,
                    right=0., forward=6., yaw_delta=0., position_uncertainty=.08,
                    yaw_uncertainty=.02, residual_p90=0.)),
                mapping_interval_error=dict(zip(('right', 'forward', 'yaw_delta'), error)))


def synthetic_cache() -> tuple[dict, dict, dict]:
    col = np.indices((84, 84))[1]
    frame = np.where((col >= 25) & (col <= 59), .4, .65).astype(np.float32)
    frames = np.stack([np.stack([frame] * 4)] * 4)
    actions = np.array([[0., .1, .1], [.04, .15, .05]])
    state = physics.PhysicsState(np.full(19, 50.), wheel_omega=np.full((19, 4), 50 / .54), gas_state=.1)
    hypotheses = analysis.jsonable(asdict(state))
    windows = [mapping_window(index=i, provenance='measured_image' if i % 2 else 'predicted_corrected_issued_action',
                              error=(0., 0., 0.)) for i in range(4)]
    scene = comparison.extract_scene(frames[:, -1], motions=[w['snapshot']['mapping_motion'] for w in windows[1:]])
    old = comparison.compare_candidates(scene, state, actions, observer_valid=True,
                                        absolute_position_uncertainty=.5, absolute_yaw_uncertainty=.05)
    poses = old['poses'][:, 0].copy()
    controls = np.tile(actions[0], (2, 16, 1))
    controls[:, :4] = actions[:, None]
    scored = comparison.score_trajectories(scene, poses, controls)
    rows, natural, prefixes = [], [], {}
    seeds = [*analysis.CAL, *analysis.HELD_OUT_FROM_FIT, *analysis.CAL]
    for i, seed in enumerate(seeds):
        name = f'anchor{i}'
        snapshot = dict(valid=True, invalid_reasons=[], mapping_motion=windows[-1]['snapshot']['mapping_motion'])
        anchor = dict(anchor_id=name, seed=seed, track_id=1, anchor_prefix_decisions=3)
        row = dict(plan_anchor_id=name, anchor=anchor, observer_valid=True, observer_snapshot=deepcopy(snapshot),
            integrity=dict(complete=True), shared_hypotheses=hypotheses,
            shared_hypotheses_sha256=analysis.canonical_hash(hypotheses), actions=actions,
            comparison=old, scene=dict(center_x=scene.center_x),
            actual_trajectories=[dict(poses=p, controls=c) for p, c in zip(poses, controls)],
            oracle_prediction={k: poses[:, None, 1:, j] for j, k in enumerate(('relative_x', 'relative_y', 'yaw_delta'))},
            cost_supported=True, actual_cost_supported=True, actual_score=scored,
            actual_delta_cost=scored['costs'][1] - scored['costs'][0],
            absolute=dict(containment=[], labels=[dict(safe=True)] * 2), old_tracking_form=dict(supported=False),
            privileged_catalog_diagnostics=[dict(signed_centerline_progress=p[:, 1],
                centerline_distance=np.abs(p[:, 0]), hull_origin_circle_gaps=[]) for p in poses])
        rows.append(row)
        prefixes[name] = dict(observations=frames.copy(), actions=np.tile(actions[0], (4, 1)), road_sha256=str(seed))
        for j in range(4):
            window = deepcopy(windows[j])
            window.update(anchor_id=name, seed=seed, input_identity=f'{name}-{j}', hypotheses=hypotheses, comparisons={})
            window['snapshot'].update(valid=True, invalid_reasons=[])
            if j == 3:
                window['snapshot'] = deepcopy(snapshot)
                window['comparisons'] = dict(synthetic=dict(actions=actions, effective=True, valid_supported=True))
            natural.append(window)
    cache = dict(anchors=rows, natural_windows=natural, global_integrity_errors=[],
                 coverage=dict(synthetic=dict(applicable=8, observed_applicable=8, valid_supported=8,
                                              unknown_possible_applicable=0)))
    return cache, prefixes, dict(comparison=comparison, interval_comparison=interval_comparison, physics=physics)


class TrajectoryTests(unittest.TestCase):
    def test_cal_maxima_include_dynamics_but_never_held_out(self):
        cal = trajectory_row()
        held = trajectory_row(analysis.HELD_OUT_FROM_FIT[0], error=100.)
        fit = analysis.calibrate_trajectories([cal, held])
        self.assertTrue(fit['supported'])
        np.testing.assert_allclose(fit['position_residual'], np.r_[0., np.full(16, .6)])
        np.testing.assert_allclose(fit['yaw_residual'], np.r_[0., np.full(16, .07)])
        diagnostics = fit['diagnostic_rows'][0]
        np.testing.assert_allclose(diagnostics['actual_minus_oracle']['position'][:, 1:], .1)
        np.testing.assert_allclose(diagnostics['oracle_minus_runtime']['position'][:, 1:], .5)
        actual, predicted, _ = analysis.trajectory_arrays(cal)
        self.assertTrue(analysis.whole_path_containment(actual[0], predicted[0],
            fit['position_residual'], fit['yaw_residual'])['central_contained'])
        actual, predicted, _ = analysis.trajectory_arrays(held)
        self.assertFalse(analysis.whole_path_containment(actual[0], predicted[0],
            fit['position_residual'], fit['yaw_residual'])['contained'])

    def test_per_tick_floor_not_global_peak(self):
        row = trajectory_row(error=0.)
        for arm in row['actual_trajectories']:
            arm['poses'][:, 2] = 0.
        row['actual_trajectories'][0]['poses'][8, 0] = 2.
        fit = analysis.calibrate_trajectories([row])
        self.assertEqual(fit['position_residual'][8], 2.)
        self.assertEqual(fit['position_residual'][9], .5)
        self.assertEqual(fit['yaw_residual'][9], .05)
        self.assertEqual(fit['position_residual'][0], 0.)

    def test_nonfinite_calibration_is_not_silently_zeroed(self):
        row = trajectory_row()
        row['actual_trajectories'][1]['poses'][5, 0] = np.nan
        fit = analysis.calibrate_trajectories([row])
        self.assertFalse(fit['supported'])
        self.assertTrue(np.isnan(fit['position_residual'][5]))
        self.assertEqual(fit['nonfinite_position_indices'].tolist(), [[1, 5]])
        self.assertIsNone(analysis.jsonable(fit)['position_residual'][5])
        json.dumps(analysis.jsonable(fit), allow_nan=False)

    def test_one_scenario_for_whole_path_not_cherry_picked_each_tick(self):
        truth, predicted = np.zeros((17, 3)), np.zeros((2, 17, 3))
        predicted[0, 1:9, 0] = 1.
        predicted[1, 9:, 0] = 1.
        result = analysis.whole_path_containment(truth, predicted, analysis.POSITION_FLOOR, analysis.YAW_FLOOR)
        self.assertTrue(result['per_tick_union_contained'])
        self.assertFalse(result['contained'])
        predicted[1] = 0.
        self.assertTrue(analysis.whole_path_containment(truth, predicted,
            analysis.POSITION_FLOOR, analysis.YAW_FLOOR)['contained'])

    def test_joint_xy_yaw_and_wrapped_angle(self):
        truth, predicted = np.zeros((17, 3)), np.zeros((2, 17, 3))
        predicted[0, 1:, 2] = .1
        predicted[1, 1:, 0] = 1.
        self.assertFalse(analysis.whole_path_containment(truth, predicted,
            analysis.POSITION_FLOOR, analysis.YAW_FLOOR)['contained'])
        predicted[0, 1:, 2] = 2 * np.pi + .01
        self.assertTrue(analysis.whole_path_containment(truth, predicted,
            analysis.POSITION_FLOOR, analysis.YAW_FLOOR)['contained'])


class MappingTests(unittest.TestCase):
    def test_euclidean_maximum_dedup_and_held_out_miss(self):
        cal = mapping_window()
        duplicate = deepcopy(cal)
        duplicate['duplicate_of_prior_causal_input'] = True
        held = mapping_window(analysis.HELD_OUT_FROM_FIT[0], error=(10., 0., .01))
        fit = analysis.calibrate_mapping([cal, duplicate, held])
        inferred = fit['by_provenance']['predicted_corrected_issued_action']
        self.assertEqual(inferred['n_cal'], 1)
        self.assertEqual(inferred['position_residual'], .5)
        self.assertFalse(fit['intervals'][1]['calibrated_joint_contained'])
        before = deepcopy(cal['snapshot']['mapping_motion'])
        inflated = analysis.inflate_mapping_motion(before, fit)
        self.assertEqual(inflated['position_uncertainty'], .5)
        self.assertEqual(before['position_uncertainty'], .08)
        self.assertEqual(inflated['forward'], before['forward'])

    def test_invalid_unknown_and_nonfinite_never_upgrade(self):
        cal = mapping_window(error=(np.nan, 0., 0.))
        fit = analysis.calibrate_mapping([cal])
        self.assertFalse(fit['by_provenance']['predicted_corrected_issued_action']['supported'])
        motion = cal['snapshot']['mapping_motion']
        self.assertFalse(analysis.inflate_mapping_motion(motion, fit)['valid'])
        motion = dict(motion, valid=False)
        self.assertEqual(analysis.inflate_mapping_motion(motion, fit), motion)
        unknown = dict(motion, valid=True, provenance='unknown')
        self.assertFalse(analysis.inflate_mapping_motion(unknown, fit)['valid'])

    def test_conflicting_duplicate_fails(self):
        a = mapping_window()
        b = deepcopy(a)
        b['mapping_interval_error']['right'] = 99.
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            analysis.calibrate_mapping([a, b])


class PairedCostTests(unittest.TestCase):
    def test_shared_reference_subtraction_not_marginal_extrema(self):
        costs = np.tile(np.arange(5.) * 100, (2, 19, 1))
        costs[1] += 2.
        observed = costs[:, 0].copy()
        observed[1] += .3
        error = analysis.paired_cost_errors(costs, observed, True)
        np.testing.assert_allclose(error['signed_error'], .3)
        self.assertAlmostEqual(error['maximum_absolute_error'], .3)
        cal = dict(anchor_id='cal', seed=analysis.CAL[0], split='CAL', paired_error=error)
        held = dict(anchor_id='held', seed=analysis.HELD_OUT_FROM_FIT[0], split='HELD_OUT_FROM_FIT',
                    paired_error=dict(error, maximum_absolute_error=100.))
        fit = analysis.calibrate_paired_cost([cal, held])
        self.assertAlmostEqual(fit['paired_cost_residual'], .3)
        self.assertEqual(fit['n_cal_used'], 1)

    def test_unsupported_and_nonfinite_have_distinct_diagnostics(self):
        error = analysis.paired_cost_errors(np.full((2, 19, 5), np.nan), np.zeros((2, 5)), False)
        row: dict = dict(anchor_id='cal', seed=analysis.CAL[0], split='CAL', paired_error=error)
        fit = analysis.calibrate_paired_cost([row])
        self.assertIsNone(fit['paired_cost_residual'])
        self.assertEqual(fit['diagnostic_rows'][0]['status'], 'unsupported_common_scene')
        row['paired_error']['supported'] = True
        fit = analysis.calibrate_paired_cost([row])
        self.assertFalse(fit['supported'])
        self.assertEqual(fit['diagnostic_rows'][0]['status'], 'nonfinite_supported_error')

    def test_cost_floor_is_unchanged_and_overlap_is_ambiguous(self):
        for interval in ((-.05, -.05), (.05, .05), (-1., 1.), (None, None)):
            self.assertEqual(analysis.interval_order(interval), 'ambiguous')
        self.assertEqual(analysis.interval_order([.051, .2]), 'baseline')
        self.assertEqual(analysis.interval_order([-.2, -.051]), 'alternative')


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache, cls.prefixes, cls.modules = synthetic_cache()
        cls.report, cls.calibration = analysis.analyze(cls.cache, cls.prefixes, cls.modules)

    def test_eight_rows_full_schema_and_denominators(self):
        self.assertEqual(len(self.report['anchors']), 8)
        self.assertTrue(self.calibration['calibration_complete'])
        self.assertEqual(self.report['summary']['HELD_OUT_FROM_FIT']['n_paths'], 4)
        self.assertEqual(self.report['coverage']['synthetic']['applicable'], 8)
        for row in self.report['anchors']:
            self.assertEqual(np.shape(row['calibrated']['comparison']['reference_costs']), (2, 19, 5))
            self.assertIn('weighted_actual_delta', row['calibrated']['components']['progress'])
            self.assertIn('progress_delta', row['outcome'])
        json.dumps(analysis.jsonable(self.report), allow_nan=False)
        json.dumps(analysis.jsonable(self.calibration), allow_nan=False)

    def test_exclusive_output_and_identical_console_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'new'
            summary = analysis.write_outputs(root, self.report, self.calibration)
            self.assertEqual((root / 'summary.txt').read_text(), summary)
            before = analysis.digest(root / 'calibration.json')
            with self.assertRaisesRegex(ValueError, 'exclusive'):
                analysis.write_outputs(root, self.report, self.calibration)
            self.assertEqual(analysis.digest(root / 'calibration.json'), before)

    def test_invalid_observer_remains_counted(self):
        cache = deepcopy(self.cache)
        row = cache['anchors'][0]
        row['observer_valid'] = False
        row['observer_snapshot']['valid'] = False
        row['observer_snapshot']['invalid_reasons'] = ['synthetic_invalid']
        cache['natural_windows'][3]['snapshot'] = deepcopy(row['observer_snapshot'])
        report, _ = analysis.analyze(cache, self.prefixes, self.modules)
        self.assertEqual(len(report['anchors']), 8)
        self.assertFalse(report['anchors'][0]['calibrated']['comparison']['eligible_alternative'])
        self.assertEqual(report['anchors'][0]['observer_invalid_reasons'], ['synthetic_invalid'])

    def test_scene_replay_never_reads_future_frame(self):
        windows = {w['index']: w for w in self.cache['natural_windows'][:4]}
        prefix = deepcopy(self.prefixes['anchor0'])
        first = analysis.scene_at(windows[2], prefix, windows, interval_comparison)
        prefix['observations'][3] = np.nan
        second = analysis.scene_at(windows[2], prefix, windows, interval_comparison)
        np.testing.assert_array_equal(first.center_bounds, second.center_bounds)

    def test_nonfinite_fit_keeps_serializable_blocked_artifacts(self):
        cache = deepcopy(self.cache)
        cache['anchors'][0]['actual_trajectories'][0]['poses'][8, 0] = np.nan
        report, calibration = analysis.analyze(cache, self.prefixes, self.modules)
        self.assertEqual(report['status'], 'BLOCKED_TRAJECTORY_CALIBRATION_INVALID')
        self.assertEqual(len(report['anchors']), 8)
        self.assertFalse(calibration['calibration_complete'])
        self.assertTrue(np.isnan(calibration['position_residual'][8]))
        with tempfile.TemporaryDirectory() as directory:
            analysis.write_outputs(Path(directory) / 'blocked', report, calibration)
            retained = json.loads((Path(directory) / 'blocked/calibration.json').read_text())
            self.assertIsNone(retained['position_residual'][8])

    def test_old_tick_counts_include_reference_holes(self):
        row = deepcopy(self.cache['anchors'][0])
        row['scene']['center_x'][62] = np.nan
        ticks = analysis.cached_support_ticks(row, comparison.Y_SCALE)
        self.assertTrue((ticks['supported_ticks'] < ticks['footprint_known_ticks']).all())

    def test_only_second_natural_candidate_can_be_eligible(self):
        compare = dict(delta_interval=np.array([[0., 0.], [.2, .3], [-.3, -.2]]),
            cost_supported=np.ones(3, bool), common_support=True, supported_ticks=np.full((3, 19), 17),
            footprint_known_ticks=np.full((3, 19), 17), reference_costs=np.zeros((3, 19, 5)),
            reference_delta=np.zeros((3, 19, 5)), absolute_supported=np.ones((3, 19), bool),
            veto=np.zeros((3, 19), bool), abstain_reasons=[[], [], []])
        result = analysis.comparison_report(compare, 0.)
        self.assertFalse(result['eligible_alternative'])
        self.assertTrue(result['eligible_gain'])
        self.assertEqual(result['eligible_alternatives'], [False, True])
        expanded = analysis.comparison_report(compare, .5)
        self.assertFalse(expanded['eligible_gain'])
        self.assertIn('empirical_order_uncertain', expanded['abstain_reasons'][2])

    def test_targeted_source_pin_mismatch_before_module_execution(self):
        cache = dict(import_origins={name: dict(sha256='0' * 64) for name in ('hud', 'motion', 'physics', 'observer', 'comparison')})
        with self.assertRaisesRegex(ValueError, 'cached runtime source differs'):
            analysis.load_runtime(cache)

    def test_targeted_private_runtime_has_no_simulator_imports(self):
        class NoSimulator(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('agent', 'env_wrapper', 'Box2D', 'gymnasium', 'environment', 'core'):
                    raise AssertionError('simulator import: ' + fullname)
        guard = NoSimulator()
        sys.meta_path.insert(0, guard)
        try:
            directory = analysis.ROOT / 'haic/algorithms/joint_control'
            cache = dict(import_origins={name: dict(sha256=analysis.digest(directory / (name + '.py')))
                for name in ('hud', 'motion', 'physics', 'observer', 'comparison')})
            modules, pins = analysis.load_runtime(cache)
            self.assertEqual(len(modules), 6)
            self.assertEqual(len(pins), 7)
            self.assertTrue(modules['physics'].__name__.startswith('_joint_interval_'))
        finally:
            sys.meta_path.remove(guard)

    def test_same_road_hash_cannot_cross_fitting_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = dict(anchors=[], input_sha256={})
            for i, seed in enumerate((analysis.CAL[0], analysis.HELD_OUT_FROM_FIT[0])):
                name = f'split{i}'
                path = root / (name + '-baseline')
                path.mkdir()
                np.savez(path / 'data.npz', observations=np.zeros((1, 4, 84, 84), np.float32),
                         actions=np.zeros((1, 3)))
                (path / 'static.json').write_text(json.dumps(dict(track=[[0., 0., 0., 0.], [0., 0., 0., 1.]])))
                cache['anchors'].append(dict(anchor=dict(anchor_id=name, seed=seed, anchor_prefix_decisions=0)))
                for item in path.iterdir():
                    cache['input_sha256'][str(item)] = analysis.digest(item)
            with self.assertRaisesRegex(ValueError, 'crosses fitting split'):
                analysis.load_prefixes(cache, root)

    def test_unseen_mapping_provenance_only_blocks_its_links(self):
        cache = deepcopy(self.cache)
        for window in cache['natural_windows']:
            if window['seed'] in analysis.CAL:
                window['snapshot']['mapping_motion']['provenance'] = 'measured_image'
        for row in cache['anchors']:
            if row['anchor']['seed'] in analysis.CAL:
                row['observer_snapshot']['mapping_motion']['provenance'] = 'measured_image'
        report, calibration = analysis.analyze(cache, self.prefixes, self.modules)
        self.assertTrue(calibration['calibration_complete'])
        self.assertFalse(calibration['mapping']['by_provenance']['predicted_corrected_issued_action']['supported'])
        self.assertTrue(all(r['calibrated']['paired_supported'] for r in report['anchors'] if r['split'] == 'CAL'))


if __name__ == '__main__':
    unittest.main()
