"""Synthetic image transforms only; no simulator interaction."""

import math
import unittest

import cv2
import numpy as np

from haic.algorithms.joint_control.motion import (
    ORIGIN, PIXEL_SCALE, estimate_body_motion, estimate_motion_sequence,
)


class MotionTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        rng = np.random.default_rng(48)
        self.image = cv2.GaussianBlur(rng.integers(0, 256, (84, 84), dtype=np.uint8), (3, 3), 0)

    def warped(self, right, forward, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, s], [-s, c]])
        shift = -rotation @ np.array([right, forward])
        matrix = np.diag(PIXEL_SCALE) @ rotation @ np.diag(1 / PIXEL_SCALE)
        translation = ORIGIN + PIXEL_SCALE * shift - matrix @ ORIGIN
        return cv2.warpAffine(self.image, np.column_stack((matrix, translation)), (84, 84))

    def test_translation_and_rotation_signs(self):
        for right, forward, yaw in ((0, 2, 0), (0.4, 2, 0.08), (-0.3, 1.5, -0.06)):
            with self.subTest(right=right, forward=forward, yaw=yaw):
                result = estimate_body_motion(self.image, self.warped(right, forward, yaw))
                self.assertTrue(result['valid'])
                self.assertAlmostEqual(result['right'], right, delta=0.08)
                self.assertAlmostEqual(result['forward'], forward, delta=0.08)
                self.assertAlmostEqual(result['yaw_delta'], yaw, delta=0.01)

    def test_no_features_is_unknown_not_rest(self):
        result = estimate_body_motion(np.zeros((84, 84), np.uint8), self.image)
        self.assertFalse(result['valid'])

    def test_normalized_frames_and_no_future_dependency(self):
        second = self.warped(0.1, 2, 0.03)
        pair = np.stack((self.image, second))
        result = estimate_motion_sequence(pair)
        longer = estimate_motion_sequence(np.concatenate((pair, np.zeros((1, 84, 84), np.uint8))))
        for key in result:
            np.testing.assert_array_equal(result[key], longer[key][:2])
        self.assertFalse(result['valid'][0])
        normalized = estimate_body_motion(np.asarray(self.image, dtype=float) / 255.,
                                          np.asarray(second, dtype=float) / 255.)
        self.assertAlmostEqual(normalized['forward'], result['forward'][1], places=8)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            estimate_body_motion(self.image, self.image, 0)
        with self.assertRaises(ValueError):
            estimate_body_motion(np.zeros((83, 84)), self.image)
        with self.assertRaises(ValueError):
            estimate_body_motion(self.image.astype(float), self.image)


if __name__ == '__main__':
    unittest.main()
