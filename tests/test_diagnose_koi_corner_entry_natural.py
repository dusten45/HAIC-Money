import math
import unittest

from scripts.diagnose_koi_corner_entry_natural import (
    METRICS, actual_metrics, analyze_event, entry_history, interference, lateral_class, matching_decisions, natural_pairs,
)


def state(t, station=None, lateral=0.0, heading=0.0, contacts=(1, 1, 1, 1)) -> dict:
    return dict(t=t, station=t if station is None else station, x=0, y=t, lateral=lateral,
                heading_error=heading, speed=10 - t, wheel_road_contacts=list(contacts), contacts=[],
                wheel_state=[dict(joint_angle=.3), dict(joint_angle=-.1)],
                environment_state=dict(damage=0))


def decision(step, pre, post, steer=0.0) -> dict:
    return dict(step=step, action=[steer, .5, 0], controller=dict(
        evaluation_only=dict(pre=pre, post=post), road_centers={"30": 7, "42": 3, "54": 2}))


class NaturalEntryTests(unittest.TestCase):
    def test_sign_thresholds_and_history_are_actual_pre_states(self):
        raw = [state(i / 10, lateral=2, heading=.2) for i in range(42)]
        decisions = [decision(i + 1, raw[i], raw[i + 1]) for i in range(41)]
        history, anchor = entry_history(raw, decisions, dict(start=3.15, direction=-1))
        self.assertEqual(anchor, 32)
        self.assertEqual(history["0"]["lateral_class"], "inside")
        self.assertEqual(history["0"]["outside_positive_lateral"], -2)
        self.assertEqual(history["10"]["t"], 2.2)
        self.assertEqual(history["20"]["t"], 1.2)
        self.assertEqual(history["30"]["t"], .2)
        self.assertAlmostEqual(history["0"]["station_overshoot"], .05)
        self.assertAlmostEqual(history["10"]["body_heading_error_rad"], .2)
        self.assertEqual(history["10"]["pixel_second_difference"], 3)
        self.assertEqual([lateral_class(x) for x in (-1.01, -1, 0, 1, 1.01)],
                         ["inside", "center", "center", "center", "outside"])

    def test_missing_history_and_unreached_are_not_zero_or_wrapped(self):
        raw = [state(i) for i in range(4)]
        rows = [decision(i + 1, raw[i], raw[i + 1]) for i in range(3)]
        history, _ = entry_history(raw, rows, dict(start=1.5, direction=1))
        self.assertEqual(history["10"]["status"], "MISSING_PRE_EPISODE_HISTORY")
        self.assertNotIn("raw_lateral", history["10"])
        history, anchor = entry_history(raw, rows, dict(start=4, direction=1))
        self.assertIsNone(anchor)
        self.assertTrue(all(v["status"] == "GEOMETRIC_ENTRY_NOT_REACHED" for v in history.values()))

    def test_metrics_use_body_heading_individual_wheel_and_separate_offroad_runs(self):
        raw = [state(i / 10, contacts=(0, 0, 0, 0) if i in (1, 2, 4) else (1, 1, 1, 1),
                     heading=.7 if i == 3 else 0) for i in range(6)]
        raw[3]["contacts"] = [2]
        raw[3]["environment_state"]["damage"] = .2
        rows = [decision(1, raw[0], raw[3], -.8), decision(2, raw[3], raw[5], .2)]
        measured = actual_metrics(raw, rows, {1: False, 2: True})
        self.assertAlmostEqual(measured["path_length"], .5)
        self.assertAlmostEqual(measured["max_any_wheel_offroad_seconds"], .2)
        self.assertAlmostEqual(measured["any_wheel_offroad_seconds"], .3)
        self.assertAlmostEqual(measured["max_all_wheels_offroad_seconds"], .2)
        self.assertAlmostEqual(measured["max_abs_reentry_body_heading_deg"], math.degrees(.7))
        self.assertEqual(measured["max_abs_front_wheel_joint_angle_rad"], .3)
        self.assertEqual(measured["max_abs_issued_steering"], .8)
        self.assertEqual(measured["collision_raw_states"], 1)
        self.assertEqual(measured["collision_positive_decisions"], 1)
        self.assertEqual(measured["damage_increase"], .2)
        self.assertEqual(measured["min_physical_speed"], 9.5)
        self.assertEqual(matching_decisions(rows, .1, .3), rows[:1])

    def test_censor_and_confounds_are_preserved(self):
        raw = [state(i) for i in range(4)]
        rows = [decision(i + 1, raw[i], raw[i + 1]) for i in range(3)]
        rows[0]["controller"]["impact_proxy_trigger"] = True
        rows[1]["controller"]["recovery_changed"] = True
        rows[2]["controller"]["arrival_cap"] = 40
        counts = interference(rows)
        self.assertEqual([counts[k] for k in ("obstacle", "impact", "recovery")], [1, 1, 1])
        prior = dict(id="1/1/1", track_id=1, seed=1, start=2, end=8, apex=3, turn_deg=90,
                     max_curvature=.1, direction=1, road_indices=[1, 3], status="CENSORED_OR_NONFORWARD")
        event = analyze_event(prior, raw, rows, {1: False, 2: False, 3: False}, 100)
        self.assertEqual(event["status"], "CENSORED_OR_NONFORWARD")
        self.assertFalse(event["actual"]["complete_window"])
        self.assertEqual(event["actual"]["end_t"], 3)
        prior.update(start=20, end=25, status="NOT_REACHED")
        event = analyze_event(prior, raw, rows, {1: False, 2: False, 3: False}, 100)
        self.assertIsNone(event["actual"])

    def test_no_reentry_is_null_and_geometry_pairs_fail_closed(self):
        raw = [state(i) for i in range(2)]
        measured = actual_metrics(raw, [decision(1, raw[0], raw[1])], {1: False})
        self.assertIsNone(measured["max_abs_reentry_body_heading_deg"])
        self.assertEqual(measured["reentry_status"], "NO_DEPARTURE")
        events = [dict(seed=1, track_id=i, road_indices=[1, 2]) for i in (1, 2)]
        catalogs = {(i, 1): dict(track=[i], obstacles=[]) for i in (1, 2)}
        with self.assertRaisesRegex(ValueError, "identical archived geometry"):
            natural_pairs(events, catalogs)

    def test_same_geometry_pair_orients_outside_minus_other_and_keeps_confounding(self):
        events = []
        for track, lateral, path in ((1, 0, 10), (2, 2, 8)):
            h = dict(status="OBSERVED", lateral_class=lateral_class(lateral), outside_positive_lateral=lateral,
                     body_heading_error_deg=0, physical_speed=10)
            events.append(dict(id=str(track), seed=1, track_id=track, road_indices=[1, 2], status="COMPLETE",
                               prior_baseline_confounded=False, approach_history_complete=True,
                               approach_interference=dict(any_recorded=True), window_interference=dict(any_recorded=False),
                               entry_history={str(k): h for k in (0, 10, 20, 30)},
                               actual={k: path if k == "path_length" else 0 for k in METRICS}))
        catalogs = {(i, 1): dict(track=[1], obstacles=[i]) for i in (1, 2)}
        pair = natural_pairs(events, catalogs)[0]
        self.assertTrue(pair["obstacle_layout_differs"])
        self.assertTrue(pair["both_prior_window_unconfounded"])
        self.assertFalse(pair["both_full_approach_and_window_without_recorded_interference"])
        self.assertFalse(pair["matched_controller_ab"])
        self.assertFalse(pair["independent"])
        self.assertEqual(pair["comparisons"]["0"]["outside_id"], "2")
        self.assertEqual(pair["comparisons"]["0"]["outside_minus_other"]["path_length"], -2)


if __name__ == "__main__":
    unittest.main()
