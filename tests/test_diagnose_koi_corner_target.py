import math
import unittest

from scripts.diagnose_koi_corner_target import decision, geometry, margin, summarize


class CornerTargetDiagnosisTests(unittest.TestCase):
    def test_straight_track_has_no_interior_corner(self):
        track = [[0, 0, 0, y] for y in range(0, 101, 5)]
        self.assertEqual(geometry(track)[3], [])

    def test_margin_is_only_local_footprint_proxy(self):
        self.assertAlmostEqual(margin(dict(lateral=0, heading_error=0)), 40 / 6 - 1.6)
        self.assertAlmostEqual(margin(dict(lateral=2, heading_error=math.pi / 2)), 40 / 6 - 2 - 2.61)

    def test_hud_drives_pedals_and_sprint_is_excluded(self):
        pre = dict(t=1, station=0, speed=20, road_index=0, lateral=0, heading_error=0,
                   wheel_road_contacts=[1] * 4, environment_state=dict(damage=0))
        c = dict(road_centers={"54": 42, "30": 57}, target_speed=48, pixel_speed=53,
                 evaluation_only=dict(pre=pre, post={**pre, "t": 1.08}), shield=dict(active=False))
        row = dict(step=11, action=[0, 0, .06], controller=c)
        measured = decision(row, [0, 0], [0, 3.5, 7])
        self.assertTrue(measured["eligible"])
        self.assertLess(measured["formula_error"], 1e-9)
        c["mechanism_active"] = True
        self.assertFalse(decision(row, [0, 0], [0, 3.5, 7])["eligible"])

    def test_repeated_road_is_not_independent_headroom(self):
        events = []
        for i in range(3):
            events.append(dict(id=str(i), seed=1, status="COMPLETE", unconfounded=True,
                               entry_ahead_deg=65, entry_sweep=17, headroom_observation=True,
                               near_limit=False, entry_target=46, entry_speed=50,
                               min_road_margin_proxy_m=2, clean_passage=True,
                               entry_above_target_2=False, any_wheel_offroad_ticks=0,
                               damage_delta=0, max_curvature=.02, turn_deg=45,
                               brake_onset=None))
        self.assertFalse(summarize([], events)["candidate_gate"])
        events[-1]["seed"] = 2
        self.assertTrue(summarize([], events)["candidate_gate"])
        events[-1]["near_limit"] = True
        self.assertFalse(summarize([], events)["candidate_gate"])


if __name__ == "__main__":
    unittest.main()
