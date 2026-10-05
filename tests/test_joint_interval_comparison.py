"""Synthetic interval support regressions; no simulator or outcome tuning."""

import unittest

import numpy as np

from haic.algorithms.joint_control import comparison as old
from haic.algorithms.joint_control import interval_comparison as new
from haic.algorithms.joint_control.physics import PhysicsState


def fixture():
    row, col = np.indices((84, 84))
    frame = np.where((col >= 31) & (col <= 53), .4, .65)
    motions = [dict(valid=True, right=0., forward=6., yaw_delta=0., residual_p90=0.)] * 3
    return np.stack([frame] * 4), motions


def comparison(scene, actions=None, **kwargs):
    state = PhysicsState([50.], wheel_omega=[[50 / .54] * 4], gas_state=.5)
    options = dict(observer_valid=True, position_residual=np.r_[0., [.5] * 16],
                   yaw_residual=np.r_[0., [.05] * 16], paired_cost_residual=0.)
    options.update(kwargs)
    return new.compare_candidates(scene, state,
        [[0., .5, 0.], [.04, .55, 0.]] if actions is None else actions, **options)


class IntervalComparisonTests(unittest.TestCase):
    def test_one_ambiguous_edge_preserves_interior_and_reference(self):
        frames, motions = fixture()
        frames[-1, 38, 30] = .53
        before = old.extract_scene(frames, motions=motions)
        after = new.extract_scene(frames, motions=motions)
        self.assertTrue(np.isnan(before.center_x[38]))
        self.assertTrue(np.isfinite(after.center_bounds[38]).all())
        self.assertGreater(np.ptp(after.center_bounds[38]), 0)
        self.assertTrue(after.uncertain_boundary[38, 30])
        self.assertFalse(after.road[38, 30])
        self.assertFalse(after.known[38, 30])
        np.testing.assert_array_equal(before.road, after.road)
        np.testing.assert_array_equal(before.known, after.known)
        result = comparison(after)
        self.assertTrue(result["cost_supported"].all())
        self.assertEqual(result["comparison_ticks"], 17)

    def test_unbracketed_unknown_is_not_a_road_boundary(self):
        frames, motions = fixture()
        frames[-1, 38, :31] = .1
        scene = new.extract_scene(frames, motions=motions)
        self.assertTrue(np.isnan(scene.center_x[38]))
        self.assertTrue(scene.unobserved[38, 30])
        self.assertFalse(scene.uncertain_boundary[38, 30])
        result = comparison(scene)
        self.assertFalse(result["cost_supported"].any())
        self.assertTrue((result["supported_ticks"] < 17).all())
        self.assertTrue((result["footprint_known_ticks"] == 17).all())

    def test_unknown_interior_hole_is_not_filled_from_history(self):
        frames, motions = fixture()
        frames[-1, 38:41, 41:44] = .1
        scene = new.extract_scene(frames, motions=motions)
        self.assertFalse(scene.known[39, 42])
        self.assertFalse(scene.road[39, 42])
        self.assertTrue(comparison(scene)["abstain"].all())

    def test_fully_known_reference_matches_original(self):
        frames, motions = fixture()
        a, b = old.extract_scene(frames, motions=motions), new.extract_scene(frames, motions=motions)
        np.testing.assert_array_equal(a.center_x, b.center_x)
        np.testing.assert_array_equal(b.center_bounds[:, 0], b.center_bounds[:, 1])

    def test_shared_reference_stresses_preserve_identical_action_equality(self):
        frames, motions = fixture()
        frames[-1, 35:50, 30] = .53
        scene = new.extract_scene(frames, motions=motions)
        result = comparison(scene, [[0., .5, 0.], [0., .5, 0.]])
        np.testing.assert_array_equal(result["reference_delta"], 0.)
        np.testing.assert_array_equal(result["robust_sign"], 0)
        self.assertEqual(result["reference_costs"].shape, (2, 1, 5))

    def test_candidate_cannot_benefit_from_its_own_truncated_support(self):
        frames, motions = fixture()
        scene = new.extract_scene(frames, motions=motions)
        poses = np.zeros((2, 17, 3))
        poses[:, :, 1] = np.linspace(0, 15, 17)
        poses[1, -1, 1] = 50.
        controls = np.zeros((2, 16, 3))
        result = new.score_trajectories(scene, poses, controls)
        self.assertFalse(result["common_support"])
        self.assertTrue(np.isfinite(result["costs"][0]))
        self.assertFalse(np.isfinite(result["costs"][1]))
        self.assertEqual(result["comparison_ticks"], 17)

    def test_calibration_error_widens_paired_order_without_changing_costs(self):
        frames, motions = fixture()
        scene = new.extract_scene(frames, motions=motions)
        a = comparison(scene)
        b = comparison(scene, paired_cost_residual=100.)
        np.testing.assert_array_equal(a["costs"], b["costs"])
        np.testing.assert_allclose(b["delta_interval"][1], a["delta_interval"][1] + [-100., 100.])
        self.assertEqual(b["robust_sign"][1], 0)
        self.assertTrue(b["abstain"].all())

    def test_missing_or_large_trajectory_range_abstains(self):
        frames, motions = fixture()
        scene = new.extract_scene(frames, motions=motions)
        missing = comparison(scene, position_residual=None)
        large = comparison(scene, position_residual=np.full(17, 20.))
        self.assertTrue(missing["abstain"].all())
        self.assertTrue(large["abstain"].all())
        self.assertIn("trajectory_calibration_missing", missing["abstain_reasons"][1])
        self.assertTrue(np.isnan(missing["absolute_road_clearance"]).all())
        self.assertTrue(np.isnan(missing["absolute_obstacle_clearance"]).all())
        np.testing.assert_array_equal(missing["confidence"], 0.)

    def test_structural_support_indicator_uses_new_absolute_support(self):
        frames, motions = fixture()
        result = comparison(new.extract_scene(frames, motions=motions))
        np.testing.assert_array_equal(result["confidence"], result["supported"].astype(float))
        self.assertTrue(result["supported"].all())

    def test_support_and_geometry_inputs_are_not_mutated(self):
        frames, motions = fixture()
        scene = new.extract_scene(frames, motions=motions)
        road, centers = scene.road.copy(), scene.center_bounds.copy()
        comparison(scene)
        np.testing.assert_array_equal(scene.road, road)
        np.testing.assert_array_equal(scene.center_bounds, centers)

    def test_boundary_and_unobserved_are_disjoint_from_known(self):
        frames, motions = fixture()
        frames[-1, 38, 30] = .53
        scene = new.extract_scene(frames, motions=motions)
        self.assertFalse((scene.uncertain_boundary & scene.unobserved).any())
        self.assertFalse((scene.uncertain_boundary & scene.known).any())
        np.testing.assert_array_equal(scene.known | scene.uncertain_boundary | scene.unobserved, True)

    def test_invalid_residuals_fail_closed(self):
        frames, motions = fixture()
        scene = new.extract_scene(frames, motions=motions)
        for value in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                comparison(scene, paired_cost_residual=value)
        with self.assertRaises(ValueError):
            comparison(scene, position_residual=np.zeros(16))

    def test_json_list_paths_match_array_paths(self):
        frames, motions = fixture()
        scene = new.extract_scene(frames, motions=motions)
        poses = np.zeros((2, 17, 3))
        poses[:, :, 1] = np.linspace(0, 15, 17)
        controls = np.zeros((2, 16, 3))
        array = new.score_trajectories(scene, poses, controls)
        lists = new.score_trajectories(scene, poses.tolist(), controls.tolist())
        np.testing.assert_array_equal(array["reference_costs"], lists["reference_costs"])
        np.testing.assert_array_equal(array["supported_ticks"], lists["supported_ticks"])

    def test_state_permutation_keeps_reference_pairing_and_order(self):
        frames, motions = fixture()
        frames[-1, 35:50, 30] = .53
        scene = new.extract_scene(frames, motions=motions)
        actions = [[0., .5, 0.], [.04, .55, 0.]]
        outputs = []
        for speeds in ([49., 51.], [51., 49.]):
            state = PhysicsState(speeds, wheel_omega=np.asarray(speeds)[:, None] / .54 * np.ones((2, 4)))
            outputs.append(new.compare_candidates(scene, state, actions, observer_valid=True,
                position_residual=np.r_[0., [.5] * 16], yaw_residual=np.r_[0., [.05] * 16]))
        np.testing.assert_allclose(outputs[0]["reference_delta"][:, ::-1], outputs[1]["reference_delta"])
        np.testing.assert_allclose(outputs[0]["delta_interval"], outputs[1]["delta_interval"])


if __name__ == "__main__":
    unittest.main()
