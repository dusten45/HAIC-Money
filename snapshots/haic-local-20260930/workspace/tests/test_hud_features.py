import unittest

import numpy as np

from haic_agent.observation import extract_hud_features


class TestHudFeatures(unittest.TestCase):
    def test_crops_fixed_processed_pixel_coordinates_and_keeps_full_frame(self):
        # Break caught: moving a crop to raw 96x96 coordinates silently reads
        # scenery instead of the rendered HUD after preprocessing.
        frame = np.arange(84 * 84, dtype=np.float32).reshape(84, 84) / 7055.0

        features = extract_hud_features(frame, channels=["speed", "steering"])

        np.testing.assert_array_equal(features.full_frame, frame)
        np.testing.assert_array_equal(features.regions["speed"], frame[72:84, 9:14])
        np.testing.assert_array_equal(features.regions["steering"], frame[74:82, 40:58])
        self.assertEqual(set(features.regions), {"speed", "steering"})

    def test_rejects_non_pixel_observation_shapes(self):
        with self.assertRaisesRegex(ValueError, "84, 84"):
            extract_hud_features(np.zeros((4, 96, 96), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
