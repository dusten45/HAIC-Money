from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from haic.algorithms.joint_control import paired_residual as new
from haic.algorithms.joint_control import interval_comparison as old
from haic.algorithms.joint_control.physics import PhysicsState
from tests.test_joint_interval_comparison import fixture
from scripts.calibrate_joint_paired_envelope import run


def absolute():
    return dict(calibration_complete=True, CAL=[1, 2], HELD_OUT_FROM_FIT=[3],
                position_residual=[0.] + [.5] * 16, yaw_residual=[0.] + [.05] * 16,
                paired_cost_residual=.44961874671412616, mapping={"preserved": [1., 2.]})


def row(seed=1, predicted=None, actual=None):
    return dict(anchor_id=str(seed), seed=seed, role="CAL" if seed < 3 else "HELD_OUT_FROM_FIT", supported=True,
                predicted_delta=[[-.8, -.6], [-.2, -.1]] if predicted is None else predicted,
                actual_delta=[-.4, -.3] if actual is None else actual)


class PairedResidualTests(unittest.TestCase):
    def test_point_spread_is_not_interval_prediction_error(self):
        r = row()
        excess = new.envelope_excess(r["predicted_delta"], r["actual_delta"])
        self.assertEqual((excess["lower"], excess["upper"]), (0., 0.))
        self.assertGreater(np.abs(np.asarray(r["predicted_delta"]) - r["actual_delta"]).max(), .3)

    def test_each_reference_keeps_its_own_pair(self):
        excess = new.envelope_excess([[-1., 1.], [-.5, 2.]], [0., 1.5])
        self.assertEqual(excess["upper"], .5)
        self.assertEqual(excess["lower"], 0.)

    def test_signed_excursions_do_not_penalize_opposite_tail(self):
        e = new.envelope_excess([[.3], [.5]], [.1])
        self.assertAlmostEqual(e["lower"], .2)
        self.assertEqual(e["upper"], 0.)

    def test_new_conservative_floor_and_safety_is_unmodified(self):
        a = absolute()
        b = deepcopy(a)
        fit = new.fit_calibration([row()], a)
        self.assertEqual((fit["performance"]["lower"], fit["performance"]["upper"]), (.05, .05))
        self.assertEqual(fit["absolute_calibration"], b)
        self.assertEqual(a, b)
        self.assertIsNot(fit["absolute_calibration"], a)

    def test_held_out_labels_cannot_change_fit(self):
        one = new.fit_calibration([row(), row(3)], absolute())
        two = new.fit_calibration([row(), row(3, actual=[1e6, -1e6])], absolute())
        self.assertEqual(one, two)

    def test_no_supported_calibration_fails_closed(self):
        r = row()
        r["supported"] = False
        with self.assertRaises(ValueError):
            new.fit_calibration([r, row(3)], absolute())
        with self.assertRaises(ValueError):
            new.fit_calibration([row(predicted=[[np.nan, 1.]])], absolute())

    def test_no_geometry_role_relabeling(self):
        r = row(3)
        r["role"] = "CAL"
        with self.assertRaises(ValueError):
            new.fit_calibration([r], absolute())

    def test_expansion_happens_once(self):
        values = np.array([[[0., 0.], [0., 0.]], [[-.8, -.7], [-.3, -.2]]])
        result = new.expanded_interval(values, lower=.1, upper=.2)
        np.testing.assert_allclose(result, [[0., 0.], [-.9, 0.]])
        np.testing.assert_allclose(new.expanded_interval(values[:, ::-1], lower=.1, upper=.2), result)

    def test_unknown_candidate_does_not_become_cheap(self):
        values = np.zeros((2, 2, 2))
        values[1, 1, 1] = np.nan
        self.assertTrue(np.isnan(new.expanded_interval(values, lower=.05, upper=.05)[1]).all())

    def test_shared_state_subtraction_and_physical_veto_unchanged(self):
        frames, motions = fixture()
        scene = old.extract_scene(frames, motions=motions)
        state = PhysicsState([50.], wheel_omega=[[50 / .54] * 4], gas_state=.5)
        actions = [[0., .5, 0.], [.04, .55, 0.]]
        c = new.fit_calibration([row()], absolute())
        before = old.compare_candidates(scene, state, actions, observer_valid=True,
            position_residual=c["absolute_calibration"]["position_residual"],
            yaw_residual=c["absolute_calibration"]["yaw_residual"], paired_cost_residual=.44961874671412616)
        after = new.compare_candidates(scene, state, actions, observer_valid=True, calibration=c)
        for key in ("poses", "costs", "reference_delta", "veto", "absolute_supported",
                    "absolute_road_clearance", "absolute_obstacle_clearance", "supported_ticks"):
            np.testing.assert_array_equal(after[key], before[key])
        np.testing.assert_allclose(after["delta_interval"][1],
            before["delta_interval"][1] + [.39961874671412616, -.39961874671412616])

    def test_operator_consumes_real_artifact_schema_and_refuses_overwrite(self):
        baseline = np.zeros((2, 2)).tolist()
        predicted = row()["predicted_delta"]
        comparison = dict(cost_supported=[True, True], reference_delta=[baseline, predicted],
                          absolute_supported=[[True, True]] * 2, veto=[[False, False]] * 2,
                          delta_interval=[[0., 0.], [-1.25, .35]])
        anchor = dict(anchor_id="cal", seed=1, split="CAL", calibrated=dict(
            comparison=comparison, actual=dict(common_support=True, delta_by_reference=row()["actual_delta"])))
        a = absolute()
        a["source_pins"] = {}
        with tempfile.TemporaryDirectory(dir="/tmp/kilo") as directory:
            root = Path(directory)
            primary, calibration, output = root / "primary.json", root / "absolute.json", root / "output"
            primary.write_text(json.dumps(dict(anchors=[anchor])))
            calibration.write_text(json.dumps(a))
            report = run(primary, calibration, output)
            self.assertTrue(report["absolute_calibration_unchanged"])
            fitted = json.loads((output / "envelope-calibration.json").read_text())
            self.assertEqual(fitted["absolute_calibration"], a)
            self.assertEqual(fitted["performance"]["upper"], .05)
            with self.assertRaises(FileExistsError):
                run(primary, calibration, output)


if __name__ == "__main__":
    unittest.main()
