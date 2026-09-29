import unittest

import numpy as np
import torch


class TestPixelVisualFeatures(unittest.TestCase):
    def _frame(self, *, obstacle: bool) -> np.ndarray:
        frames = np.zeros((4, 84, 84), dtype=np.float32)
        for row in (30, 34, 38, 42, 46, 50, 54):
            frames[:, row, 30:55] = 0.4
        speed_mass = 0.27 + 0.085 * 40.0
        frames[:, 77:83, 10:13] = speed_mass / 18.0
        if obstacle:
            frames[-1, 39:42, 48:52] = 0.9
        return frames

    def test_pixel_extractor_reads_hud_road_and_obstacle_from_image_only(self):
        from haic_agent.pixel_features import FEATURE_NAMES, extract_visual_features

        observation = self._frame(obstacle=True)
        values = extract_visual_features(observation)
        features = dict(zip(FEATURE_NAMES, values.tolist()))

        self.assertAlmostEqual(features["speed_fraction"], 0.5, delta=0.01)
        self.assertAlmostEqual(features["near_road_center_offset"], 0.0, delta=0.01)
        self.assertAlmostEqual(features["far_road_center_offset"], 0.0, delta=0.01)
        self.assertAlmostEqual(features["road_curve_magnitude"], 0.0, delta=0.01)
        self.assertEqual(features["obstacle_present"], 1.0)
        self.assertAlmostEqual(features["obstacle_lateral_offset"], 0.625, delta=0.05)
        self.assertAlmostEqual(features["obstacle_urgency"], 1.0, delta=0.01)

    def test_absent_obstacle_has_zero_hazard_features(self):
        from haic_agent.pixel_features import FEATURE_NAMES, extract_visual_features

        features = dict(zip(FEATURE_NAMES, extract_visual_features(self._frame(obstacle=False))))

        self.assertEqual(features["obstacle_present"], 0.0)
        self.assertEqual(features["obstacle_lateral_offset"], 0.0)
        self.assertEqual(features["obstacle_urgency"], 0.0)

    def test_ppo_actor_accepts_the_pixel_feature_branch(self):
        from haic_agent.networks import VisualActorCritic
        from haic_agent.pixel_features import extract_visual_features

        observation = self._frame(obstacle=True)
        model = VisualActorCritic(use_hud=False, use_visual_features=True).eval()
        with torch.inference_mode():
            output = model(torch.from_numpy(observation).unsqueeze(0))

        self.assertEqual(output.latent.shape, (1, 128))
        self.assertTrue(torch.isfinite(output.action_mean).all())
        self.assertTrue(model.use_visual_features)
        torch.testing.assert_close(
            model.last_visual_features,
            torch.from_numpy(extract_visual_features(observation)).unsqueeze(0),
        )

    def test_temporal_feature_keeps_previous_obstacle_side_when_newest_frame_is_centered(self):
        from haic_agent.pixel_features import extract_temporal_features

        observation = self._frame(obstacle=False)
        observation[-2, 39:42, 31:35] = 0.9
        observation[-1, 39:42, 42:46] = 0.9

        temporal = extract_temporal_features(observation)

        self.assertLess(float(temporal[0]), -0.4)

    def test_temporal_actor_branch_starts_as_a_zero_residual(self):
        from haic_agent.networks import VisualActorCritic

        model = VisualActorCritic(
            use_hud=False, use_visual_features=True, use_temporal_features=True
        )

        self.assertTrue(torch.equal(model.temporal_feature_encoder.weight, torch.zeros_like(model.temporal_feature_encoder.weight)))
        self.assertTrue(torch.equal(model.temporal_feature_encoder.bias, torch.zeros_like(model.temporal_feature_encoder.bias)))


if __name__ == "__main__":
    unittest.main()
