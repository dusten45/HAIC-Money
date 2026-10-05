import unittest
from unittest.mock import patch

import numpy as np

from haic.algorithms.joint_control.hud import (
    HUD_CHANNELS, HudCalibration, decode_hud, extract_hud_features,
    fit_hud_calibration,
)


def hud_frame(speed=30.0, wheel_angle=.2, yaw_rate=.5, wheel_omega=(0, 0, 0, 0)):
    """Draw only source HUD rectangles, never construct or import a simulator."""
    import cv2
    import pygame

    surface = pygame.Surface((1000, 800))
    surface.fill((0, 0, 0))
    # Stand-in for reward text witnesses a real HUD even when all bars are zero.
    pygame.draw.rect(surface, (255, 255, 255), (25, 735, 12, 25))
    for place, value, scale, color in (
        (5, speed, .02, (255, 255, 255)),
        *((7 + i, v, .01, (0, 0, 255) if i < 2 else (51, 0, 255))
          for i, v in enumerate(wheel_omega)),
    ):
        if abs(value) > 1e-4:
            top = 800 - (20 + 20 * scale * value)
            pygame.draw.polygon(surface, color, [(place * 25, top),
                ((place + 1) * 25, top), ((place + 1) * 25, 780), (place * 25, 780)])
    for anchor, value, scale, color in (
        (500, wheel_angle, 250, (0, 255, 0)), (750, yaw_rate, 20, (255, 0, 0)),
    ):
        if abs(value) > 1e-4:
            end = anchor - scale * value
            pygame.draw.polygon(surface, color, [(anchor, 720), (end, 720),
                                                 (end, 760), (anchor, 760)])
    rgb = np.transpose(np.array(pygame.surfarray.pixels3d(
        pygame.transform.smoothscale(surface, (96, 96)))), (1, 0, 2))
    return cv2.cvtColor(cv2.resize(rgb, (84, 84)), cv2.COLOR_RGB2GRAY)


class TestJointControlHud(unittest.TestCase):
    def test_physical_joint_and_yaw_signs_after_real_grayscale_resizes(self):
        for angle, rate in [(-.4, -3), (-.15, -.5), (.15, .5), (.4, 3)]:
            with self.subTest(angle=angle, rate=rate):
                result = decode_hud(hud_frame(wheel_angle=angle, yaw_rate=rate))
                self.assertTrue(result["valid"][:3].all())
                self.assertAlmostEqual(float(result["wheel_angle"]), angle, delta=.012)
                self.assertAlmostEqual(float(result["yaw_rate"]), rate, delta=.09)
                self.assertAlmostEqual(float(result["speed"]), 30, delta=3)

    def test_tiny_yaw_is_quantized_not_recovered_as_exact_truth(self):
        positive = decode_hud(hud_frame(yaw_rate=.01))
        negative = decode_hud(hud_frame(yaw_rate=-.01))
        self.assertGreater(float(positive["yaw_rate"]), 0)
        self.assertLess(float(negative["yaw_rate"]), 0)
        self.assertLess(float(positive["confidence"][2]), .6)
        self.assertLess(float(negative["confidence"][2]), .6)
        self.assertEqual(float(positive["yaw_rate"]),
                         float(decode_hud(hud_frame(yaw_rate=.03))["yaw_rate"]))

    def test_stationary_hud_and_missing_hud_are_distinguished(self):
        result = decode_hud(hud_frame(0, 0, 0))
        self.assertTrue(result["valid"].all())
        np.testing.assert_array_equal([result[k] for k in HUD_CHANNELS[:3]], 0)
        np.testing.assert_array_equal(result["wheel_omega"], 0)
        blank = decode_hud(np.zeros((84, 84), dtype=np.uint8))
        self.assertFalse(blank["valid"].any())
        self.assertTrue(np.isnan(blank["speed"]))

    def test_uint8_normalized_single_batch_and_empty(self):
        frames = np.stack([hud_frame(), hud_frame(60, -.1, -1)])
        original = frames.copy()
        batch = decode_hud(frames)
        normalized = decode_hud(frames.astype(np.float32) / 255)
        for key in batch:
            np.testing.assert_allclose(batch[key], normalized[key], atol=1e-5, equal_nan=True)
            np.testing.assert_allclose(batch[key][0], decode_hud(frames[0])[key], equal_nan=True)
        self.assertEqual(batch["wheel_omega"].shape, (2, 4))
        self.assertEqual(batch["valid"].shape, (2, 7))
        self.assertEqual(decode_hud(frames[:0])["speed"].shape, (0,))
        np.testing.assert_array_equal(frames, original)

    def test_bad_inputs_are_not_silently_renormalized(self):
        for value in [np.zeros((84, 84, 3), dtype=np.uint8), np.zeros((96, 96)),
                      np.zeros((4, 4, 84, 84)), np.zeros((84, 84), dtype=int),
                      np.full((84, 84), 255.), np.full((84, 84), np.nan),
                      np.full((84, 84), np.inf), np.full((84, 84), -.01)]:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    extract_hud_features(value)

    def test_crop_clipping_overlap_and_wrong_luminance_abstain(self):
        for kwargs, channels in [({"speed": 200}, [0]),
                                 ({"yaw_rate": -15}, [2]),
                                 ({"yaw_rate": 9}, [1, 2]),
                                 ({"wheel_angle": .7}, [1])]:
            with self.subTest(kwargs=kwargs):
                result = decode_hud(hud_frame(**kwargs))
                self.assertFalse(result["valid"][channels].any())
                self.assertTrue((result["confidence"][channels] == 0).all())
        corrupted = hud_frame()
        corrupted[77:79, 60] = 255
        self.assertFalse(decode_hud(corrupted)["valid"][2])
        self.assertFalse(decode_hud(np.full((84, 84), 255, dtype=np.uint8))["valid"].any())

    def test_disconnected_component_is_not_a_valid_bar(self):
        frame = hud_frame(wheel_angle=.1)
        frame[75:80, 31] = 149
        self.assertFalse(decode_hud(frame)["valid"][1])

    def test_wheel_order_sign_and_clipping_are_explicit(self):
        values = (40, 70, 90, 120)
        result = decode_hud(hud_frame(wheel_omega=values))
        self.assertTrue(result["valid"][3:].all())
        np.testing.assert_allclose(result["wheel_omega"], values, atol=12)
        self.assertTrue((result["confidence"][3:] <= .6).all())
        clipped = decode_hud(hud_frame(wheel_omega=(-200, 0, 0, 0)))
        self.assertFalse(clipped["valid"][3])
        self.assertTrue(np.isnan(clipped["wheel_omega"][0]))

    def test_small_wheel_bars_abstain_instead_of_guessing_sign(self):
        for omega in [(20, 20, 20, 20), (-5, -10, -5, -10)]:
            result = decode_hud(hud_frame(wheel_omega=omega))
            self.assertFalse(result["valid"][3:].any())
            self.assertTrue(np.isnan(result["wheel_omega"]).all())
        mixed = decode_hud(hud_frame(wheel_omega=(300, 20, 0, 100)))
        self.assertFalse(mixed["valid"][4])
        negative = decode_hud(hud_frame(wheel_omega=(-20,) * 4))
        self.assertTrue(negative["valid"][3:].all())
        np.testing.assert_allclose(negative["wheel_omega"], -20, atol=6)

    def test_calibration_is_per_channel_and_only_uses_supplied_rows(self):
        frames = np.stack([hud_frame(v, .2, .5) for v in range(10, 90, 5)])
        features = extract_hud_features(frames)
        target = 1.1 * features["raw_values"][:, 0] + 2
        target[-1] = np.nan
        calibration = fit_hud_calibration(features, {"speed": target})
        self.assertEqual(calibration.fit_samples, (15, 0, 0, 0, 0, 0, 0))
        self.assertAlmostEqual(calibration.gain[0], 1.1)
        self.assertAlmostEqual(calibration.bias[0], 2)
        result = decode_hud(frames, calibration)
        np.testing.assert_allclose(result["speed"][:-1], target[:-1])
        np.testing.assert_array_equal(result["wheel_angle"], decode_hud(frames)["wheel_angle"])
        self.assertEqual(float(decode_hud(hud_frame(0, 0, 0), calibration)["speed"]), 0)

    def test_optional_channel_calibration_and_invalid_rows(self):
        frames = np.stack([hud_frame(30, .02 * i, .1 * i,
                                    tuple(50 + 5 * i + j for j in range(4)))
                           for i in range(1, 13)])
        features = extract_hud_features(frames)
        features["valid"][0] = False
        labels = {"wheel_angle": features["raw_values"][:, 1] * .9,
                  "yaw_rate": features["raw_values"][:, 2] * 1.2,
                  "wheel_omega": features["raw_values"][:, 3:] * 1.05}
        calibration = fit_hud_calibration(features, labels)
        self.assertEqual(calibration.fit_samples, (0,) + (11,) * 6)
        np.testing.assert_allclose(calibration.gain, [1, .9, 1.2] + [1.05] * 4)

    def test_bad_calibration_and_label_contract_fail_closed(self):
        for kwargs in [{"gain": (1,)}, {"gain": (-1,) * 7}, {"bias": (np.nan,) * 7},
                       {"fit_samples": (-1,) * 7}, {"fit_samples": (.5,) * 7}]:
            with self.assertRaises(ValueError):
                HudCalibration(**kwargs)
        features = extract_hud_features(np.stack([hud_frame()] * 8))
        for labels in [{}, {"privileged_heading": np.zeros(8)}, {"speed": np.zeros(7)},
                       {"speed": -np.ones(8)}, {"speed": np.zeros(8)}]:
            with self.assertRaises(ValueError):
                fit_hud_calibration(features, labels)
        features["raw_values"][:, 1] = np.arange(8)
        with self.assertRaisesRegex(ValueError, "sign"):
            fit_hud_calibration(features, {"wheel_angle": -np.arange(8)})

    def test_inference_does_not_open_files_or_need_labels(self):
        frame = hud_frame()
        with patch("builtins.open", side_effect=AssertionError("no runtime file access")):
            result = decode_hud(frame, HudCalibration())
        self.assertTrue(np.isfinite(result["speed"]))

    def test_calibration_copies_mutable_serialized_coefficients(self):
        gains = [1.] * 7
        calibration = HudCalibration(gain=gains)
        gains[0] = -1
        self.assertEqual(calibration.gain, (1.,) * 7)


if __name__ == "__main__":
    unittest.main()
