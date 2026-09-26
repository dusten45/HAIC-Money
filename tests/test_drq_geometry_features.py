from __future__ import annotations

import math
import unittest

import numpy as np

from haic.algorithms.drq_v2.geometry_features import (
    describe_track,
    signature_distance,
    validate_track_structure,
)


def _ring(radius: float = 50.0, n: int = 120, direction: int = 1) -> np.ndarray:
    theta = direction * np.linspace(0, 2 * math.pi, n, endpoint=False)
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    beta = theta + math.pi / 2.0
    return np.column_stack((theta, beta, x, y))


class RoadGeometryFeaturesTests(unittest.TestCase):
    def test_reproducible_signature_and_ring_rotation(self) -> None:
        ring = _ring()
        first = describe_track(ring)
        repeat = describe_track(ring.copy())
        shifted = describe_track(np.roll(ring, 12, axis=0))
        self.assertEqual(first["signature_sha256"], repeat["signature_sha256"])
        self.assertEqual(first["cyclic_signature_sha256"], shifted["cyclic_signature_sha256"])
        self.assertLess(signature_distance(first["signature"], shifted["signature"]), 1e-6)
        self.assertGreater(first["perimeter_m"], 300)

    def test_reversed_turns_only_match_when_mirror_is_explicit(self) -> None:
        positive = describe_track(_ring(direction=1))["signature"]
        negative = describe_track(_ring(direction=-1))["signature"]
        self.assertGreater(signature_distance(positive, negative), 0.02)
        self.assertLess(signature_distance(positive, negative, allow_mirror=True), 1e-6)

    def test_nan_or_broken_road_is_rejected(self) -> None:
        road = _ring()
        road[2, 2] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            describe_track(road)
        road = _ring()
        road[2, 2:4] = road[1, 2:4]
        with self.assertRaisesRegex(ValueError, "collapsed"):
            describe_track(road)

    def test_different_radius_is_not_an_exact_duplicate(self) -> None:
        left = describe_track(_ring(radius=50))
        right = describe_track(_ring(radius=90))
        self.assertGreater(signature_distance(left["signature"], right["signature"]), 0.01)

    def test_static_structure_accepts_closed_ring_and_flags_figure_eight(self) -> None:
        ring = validate_track_structure(_ring(radius=50))
        self.assertTrue(ring["centerline_connected"])
        self.assertEqual(ring["centerline_self_intersections"], 0)
        theta = np.linspace(0, 2 * math.pi, 120, endpoint=False)
        crossing = np.column_stack((theta, theta, 50 * np.sin(theta), 25 * np.sin(2 * theta)))
        report = validate_track_structure(crossing)
        self.assertGreater(report["centerline_self_intersections"], 0)
        self.assertFalse(report["centerline_connected"])

    def test_opening_event_caps_straight_at_spawn(self) -> None:
        from core.vendor.car_racing import CarRacing

        env = CarRacing(render_mode=None)
        try:
            env.reset(seed=4260100006, options={"track_id": 1})
            early = describe_track(env.track)["early_strongest_bend"]
            self.assertIsNotNone(early)
            self.assertLessEqual(early["entry_straight_since_spawn_m"], early["start_m"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
