import json
import tempfile
import unittest
import zipfile

import numpy as np

from action_smoothing import (
    DEFAULT_ACTION_SMOOTHING,
    ActionSmoother,
    StatefulActionSmoother,
    action_smoothing_fingerprint,
    append_action_control_plane,
    build_action_smoother,
    canonical_action_smoothing,
    canonical_action_control,
    normalize_action_smoothing,
    validate_action_control,
    read_embedded_action_smoothing,
    write_embedded_action_smoothing,
)
from action_representation import canonical_action_representation, map_policy_action


class TestActionSmoothing(unittest.TestCase):
    def test_alpha_clips_before_smoothing_and_respects_bounds(self):
        smoother = StatefulActionSmoother(
            low=[-1.0, 0.0], high=[1.0, 1.0], alpha=0.5
        )

        self.assertEqual(smoother.apply([2.0, -1.0]), [1.0, 0.0])
        self.assertEqual(smoother.apply([-1.0, 1.0]), [0.0, 0.5])
        self.assertEqual(smoother.mode, "alpha")

    def test_max_delta_is_applied_per_component_after_clipping(self):
        smoother = ActionSmoother(
            low=[-1.0, 0.0], high=[1.0, 1.0], max_delta=[0.25, 0.5]
        )

        self.assertEqual(smoother([2.0, -1.0]), [1.0, 0.0])
        self.assertEqual(smoother([-2.0, 2.0]), [0.75, 0.5])

    def test_reset_and_initial_action_control_first_state(self):
        smoother = StatefulActionSmoother(
            low=-1.0, high=1.0, alpha=0.5, initial_action=[0.0]
        )

        self.assertEqual(smoother.last_action, [0.0])
        self.assertEqual(smoother.apply([1.0]), [0.5])

        smoother.reset()

        self.assertIsNone(smoother.last_action)
        self.assertEqual(smoother.apply([-1.0]), [-1.0])

    def test_batch_state_and_selective_reset_are_independent(self):
        smoother = StatefulActionSmoother(low=-1.0, high=1.0, max_delta=0.25)

        self.assertEqual(smoother.apply([[0.0], [0.5]]), [[0.0], [0.5]])
        self.assertEqual(smoother.apply([[1.0], [-1.0]]), [[0.25], [0.25]])

        smoother.reset(mask=[True, False])

        self.assertEqual(smoother.apply([[-1.0], [1.0]]), [[-1.0], [0.5]])
        self.assertEqual(smoother.last_action, [[-1.0], [0.5]])

    def test_json_round_trip_preserves_state_and_next_output(self):
        smoother = StatefulActionSmoother(
            low=[-1.0, 0.0], high=[1.0, 1.0], alpha=[0.25, 0.75]
        )
        smoother.apply([[0.8, 0.2], [-0.4, 0.9]])
        encoded = smoother.to_json()

        self.assertEqual(json.loads(encoded), smoother.state_dict())
        restored = StatefulActionSmoother.from_json(encoded)
        self.assertEqual(restored.state, smoother.state)
        self.assertEqual(restored.config_fingerprint, smoother.config_fingerprint)
        self.assertEqual(
            restored.apply([[1.0, 0.0], [0.4, 1.0]]),
            smoother.apply([[1.0, 0.0], [0.4, 1.0]]),
        )

    def test_configuration_helpers_are_canonical_and_fingerprintable(self):
        self.assertEqual(DEFAULT_ACTION_SMOOTHING["method"], "none")
        exponential = canonical_action_smoothing(
            method="ema", alpha=0.25, initial_action=(0, 0, 0)
        )
        normalized = normalize_action_smoothing(exponential)

        self.assertEqual(normalized, exponential)
        self.assertEqual(
            action_smoothing_fingerprint(exponential),
            action_smoothing_fingerprint(normalized),
        )
        self.assertNotEqual(
            action_smoothing_fingerprint(exponential),
            action_smoothing_fingerprint({**exponential, "alpha": 0.5}),
        )

        max_delta = normalize_action_smoothing(
            {"method": "max_delta", "max_delta": 0.2}
        )
        self.assertIsNone(max_delta["alpha"])

    def test_official_builder_uses_explicit_initial_state(self):
        smoother = build_action_smoother({"method": "alpha", "alpha": 0.5})

        self.assertEqual(smoother.apply([1.0, 1.0, 1.0]), [0.5, 0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "length 3"):
            build_action_smoother(
                {"method": "alpha", "alpha": 0.5, "initial_action": [0.0]}
            )

    def test_embedded_checkpoint_config_round_trips(self):
        config = canonical_action_smoothing("ema", 0.35)
        with tempfile.NamedTemporaryFile(suffix=".zip") as temporary:
            with zipfile.ZipFile(temporary.name, "w") as archive:
                archive.writestr("data", "{}")
            write_embedded_action_smoothing(temporary.name, config)
            self.assertEqual(read_embedded_action_smoothing(temporary.name), config)

    def test_invalid_modes_shapes_and_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "low"):
            normalize_action_smoothing({"low": [-1.0, -1.0, 0.0]})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            StatefulActionSmoother(alpha=0.5, max_delta=0.1)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            StatefulActionSmoother(alpha=1.1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            StatefulActionSmoother(max_delta=-0.1)
        with self.assertRaisesRegex(ValueError, "shared length"):
            StatefulActionSmoother(low=[-1.0, -1.0], high=1.0, alpha=[0.5, 0.5, 0.5])

        smoother = StatefulActionSmoother(alpha=0.5)
        with self.assertRaisesRegex(ValueError, "same length"):
            smoother.apply([[0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            smoother.apply([float("nan")])

    def test_previous_steering_plane_encodes_reset_and_applied_state(self):
        control = canonical_action_control("previous-steering-plane")
        observation = np.zeros((4, 84, 84), dtype=np.float32)

        reset = append_action_control_plane(observation, control, [0.0, 0.0, 0.0])
        updated = append_action_control_plane(observation, control, [0.35, 1.0, 1.0])

        self.assertEqual(reset.shape, (5, 84, 84))
        self.assertTrue(np.all(reset[4] == 0.5))
        self.assertTrue(np.all(updated[4] == 0.675))

    def test_previous_steering_plane_rejects_incompatible_smoothing(self):
        control = canonical_action_control("previous-steering-plane")
        with self.assertRaisesRegex(ValueError, "requires alpha"):
            validate_action_control(normalize_action_smoothing(), control)
        validate_action_control(
            canonical_action_smoothing("alpha", [0.35, 1.0, 1.0]), control
        )

    def test_v2_gentle_turn_mapping_extends_v1_without_changing_longitudinal(self):
        representation = canonical_action_representation(
            "multidiscrete-steer-longitudinal-v2"
        )
        np.testing.assert_allclose(
            map_policy_action(np.asarray([5, 0], dtype=np.int64), representation),
            [-0.25, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            map_policy_action(np.asarray([6, 2], dtype=np.int64), representation),
            [0.25, 0.0, 0.8],
        )
        with self.assertRaises(ValueError):
            map_policy_action(np.asarray([7, 1], dtype=np.int64), representation)


if __name__ == "__main__":
    unittest.main()
