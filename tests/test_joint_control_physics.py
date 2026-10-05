"""Synthetic invariants only: no simulator reset, step, Agent, or Box2D import."""

from dataclasses import replace
import subprocess
import sys
import unittest

import numpy as np

from haic.algorithms.joint_control.physics import (
    BODY_MASS,
    BODY_YAW_INERTIA,
    ENGINE_POWER,
    FRICTION_LIMIT,
    PhysicsParameters,
    PhysicsState,
    RAW_DT,
    TIRE_STIFFNESS,
    WHEEL_INERTIA,
    WHEEL_RADIUS,
    WHEEL_X,
    WHEEL_Y,
    predict,
)


class JointControlPhysicsTests(unittest.TestCase):
    def test_import_does_not_load_simulator_or_agent(self):
        code = """
import sys
class BlockSimulator:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'Box2D', 'gymnasium', 'core', 'agent', 'env_wrapper'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockSimulator())
from haic.algorithms.joint_control.physics import PhysicsState, predict
predict(PhysicsState(0), [[[0, 1, 0]]])
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_constants(self):
        self.assertAlmostEqual(WHEEL_RADIUS, 0.54)
        self.assertAlmostEqual(WHEEL_INERTIA, 1.6)
        self.assertAlmostEqual(ENGINE_POWER, 40000)
        self.assertAlmostEqual(TIRE_STIFFNESS, 82)
        self.assertAlmostEqual(FRICTION_LIMIT, 400)
        self.assertAlmostEqual(BODY_MASS, 7.30192)
        self.assertAlmostEqual(BODY_YAW_INERTIA, 19.217918282666666)

    def test_gas_ramps_every_raw_tick_and_decreases_immediately(self):
        actions = np.zeros((1, 4, 3))
        actions[0, :, 1] = [1.0, 1.0, 0.15, 0.0]
        result = predict(PhysicsState(0.0), actions)
        np.testing.assert_allclose(result["gas_state"], [[0.4, 0.8, 0.15, 0.0]], atol=1e-15)
        actions[0, :, 1] = [1.0, 0.9, 0.2, 0.0]
        result = predict(PhysicsState(0.0, gas_state=0.7), actions, decision_dt=RAW_DT)
        np.testing.assert_allclose(result["gas_state"], [[0.8, 0.9, 0.2, 0.0]])

    def test_only_rear_wheels_receive_engine_torque(self):
        result = predict(
            PhysicsState(0.0, wheel_omega=np.zeros(4), friction=0),
            [[[0, 1, 0]]], decision_dt=RAW_DT,
        )
        expected = RAW_DT * ENGINE_POWER * 0.1 / WHEEL_INERTIA / 5.0
        np.testing.assert_allclose(result["wheel_omega"][0, 0], [0, 0, expected, expected])
        self.assertEqual(result["speed"][0, 0], 0.0)

    def test_steering_gain_rate_limit_and_mechanical_bounds(self):
        actions = np.zeros((3, 10, 3))
        actions[:, :, 0] = np.array([1, -1, 0.01])[:, None]
        result = predict(PhysicsState(0.0), actions, decision_dt=RAW_DT)
        np.testing.assert_allclose(result["wheel_angle"][:, 0], [-0.06, 0.06, -0.01])
        np.testing.assert_allclose(result["wheel_angle"][:, -1], [-0.4, 0.4, -0.01])
        damaged = predict(PhysicsState(0.0, steering_multiplier=0.5), actions, decision_dt=RAW_DT)
        np.testing.assert_allclose(damaged["wheel_angle"][:, 0], [-0.03, 0.03, -0.005])

    def test_first_tick_tire_force_precedes_steering_joint_motion(self):
        result = predict(PhysicsState(20.0), [[[1, 0, 0], [1, 0, 0]]], decision_dt=RAW_DT)
        self.assertEqual(result["wheel_angle"][0, 0], -0.06)
        self.assertAlmostEqual(result["yaw_rate"][0, 0], 0.0)
        self.assertAlmostEqual(result["relative_x"][0, 0], 0.0)
        self.assertLess(result["yaw_rate"][0, 1], 0.0)
        self.assertGreater(result["relative_x"][0, 1], 0.0)

    def test_initial_joint_overshoot_is_not_silently_clipped_as_measurement(self):
        angle, speed = 0.44, 10.0
        state = PhysicsState(speed, wheel_angle=angle)
        result = predict(state, [[[-1, 0, 0]]], decision_dt=RAW_DT)
        side_force = -TIRE_STIFFNESS * speed * np.sin(angle)
        torque = -2 * WHEEL_Y[0] * side_force * np.cos(angle)
        self.assertAlmostEqual(result["yaw_rate"][0, 0], RAW_DT * torque / BODY_YAW_INERTIA)
        self.assertEqual(result["wheel_angle"][0, 0], 0.4)
        self.assertEqual(state.wheel_angle, angle)

    def test_right_turn_sign_and_left_right_mirror(self):
        actions = np.zeros((2, 4, 3))
        actions[:, :, 0] = np.array([0.3, -0.3])[:, None]
        result = predict(PhysicsState(30.0), actions)
        self.assertLess(result["yaw_delta"][0, -1], 0.0)
        self.assertGreater(result["relative_x"][0, -1], 0.0)
        for name in ("yaw_delta", "relative_x", "yaw_rate", "lateral_speed", "wheel_angle"):
            np.testing.assert_allclose(result[name][0], -result[name][1], atol=1e-12)
        for name in ("speed", "relative_y", "forward_speed"):
            np.testing.assert_allclose(result[name][0], result[name][1], atol=1e-12)

    def test_braking_is_per_tick_and_hard_lock_is_before_tire_reaction(self):
        omega = np.array([20.0, -20.0, 2.0, -2.0])
        state = PhysicsState(0, wheel_omega=omega, friction=0)
        soft = predict(state, [[[0, 0, 0.2]]], decision_dt=RAW_DT)
        np.testing.assert_allclose(soft["wheel_omega"][0, 0], [17, -17, 0, 0])
        hard = predict(state, [[[0, 1, 0.9]]], decision_dt=RAW_DT)
        np.testing.assert_array_equal(hard["wheel_omega"][0, 0], np.zeros(4))
        moving = predict(PhysicsState(30), [[[0, 0, 1]]], decision_dt=RAW_DT)
        expected_reaction = RAW_DT * FRICTION_LIMIT * WHEEL_RADIUS / WHEEL_INERTIA
        np.testing.assert_allclose(moving["wheel_omega"][0, 0], expected_reaction)
        self.assertLess(moving["speed"][0, 0], 30)

    def test_gas_and_brake_change_speed_without_reversing_brake_force(self):
        actions = np.zeros((3, 4, 3))
        actions[0, :, 1] = 0.4
        actions[2, :, 2] = 0.3
        result = predict(PhysicsState(30), actions)
        self.assertGreater(result["speed"][0, -1], result["speed"][1, -1])
        self.assertGreater(result["speed"][1, -1], result["speed"][2, -1])
        reverse = predict(PhysicsState(-30), [[[0, 0, 1]]])
        self.assertGreater(reverse["forward_speed"][0, 0], -30)
        self.assertLess(reverse["forward_speed"][0, 0], 0)

    def test_straight_motion_stationary_state_and_reverse(self):
        speeds = np.array([0, 20, -10])
        state = PhysicsState(speeds)
        result = predict(state, np.zeros((3, 10, 3)))
        np.testing.assert_allclose(result["relative_y"], speeds[:, None] * (np.arange(1, 11) * 0.08))
        np.testing.assert_allclose(result["relative_x"], 0, atol=1e-14)
        np.testing.assert_allclose(result["yaw_delta"], 0, atol=1e-14)
        np.testing.assert_allclose(result["forward_speed"], np.broadcast_to(speeds[:, None], (3, 10)))

    def test_free_motion_preserves_world_velocity_not_kinematic_yaw(self):
        state = PhysicsState(12, lateral_speed=3, yaw_rate=0.7, wheel_angle=0.2, friction=0)
        result = predict(state, np.zeros((1, 4, 3)))
        times = 0.08 * np.arange(1, 5)
        np.testing.assert_allclose(result["relative_x"][0], 3 * times)
        np.testing.assert_allclose(result["relative_y"][0], 12 * times)
        np.testing.assert_allclose(result["yaw_delta"][0], 0.7 * times)
        np.testing.assert_allclose(result["speed"][0], np.hypot(12, 3))
        np.testing.assert_allclose(result["forward_speed"][0], -3*np.sin(0.7*times) + 12*np.cos(0.7*times))
        np.testing.assert_allclose(result["lateral_speed"][0], 3*np.cos(0.7*times) + 12*np.sin(0.7*times))

    def test_default_wheel_spin_uses_signed_point_velocity_and_joint_angle(self):
        state = PhysicsState(-8, lateral_speed=2, yaw_rate=0.4, wheel_angle=-0.2, friction=0)
        angles = np.array([-0.2, -0.2, 0, 0])
        expected = (
            -(2 - 0.4*WHEEL_Y)*np.sin(angles) + (-8 + 0.4*WHEEL_X)*np.cos(angles)
        ) / WHEEL_RADIUS
        result = predict(state, [[[0, 0, 0]]], decision_dt=RAW_DT)
        np.testing.assert_allclose(result["wheel_omega"][0, 0], expected)

    def test_combined_force_saturation_limits_acceleration(self):
        state = PhysicsState(100, lateral_speed=100, wheel_omega=np.zeros(4))
        result = predict(state, [[[0, 0, 1]]], decision_dt=RAW_DT)
        yaw = result["yaw_delta"][0, 0]
        vf, vl = result["forward_speed"][0, 0], result["lateral_speed"][0, 0]
        vx, vy = vl*np.cos(yaw) - vf*np.sin(yaw), vl*np.sin(yaw) + vf*np.cos(yaw)
        dv = np.hypot(vx - 100, vy - 100)
        self.assertLessEqual(dv, RAW_DT * 4 * FRICTION_LIMIT / BODY_MASS + 1e-12)
        # Separate longitudinal/lateral clipping would violate this vector bound.
        self.assertAlmostEqual(dv, RAW_DT * 4 * FRICTION_LIMIT / BODY_MASS, places=10)

    def test_batch_shapes_finite_reproducible_and_inputs_unchanged(self):
        rng = np.random.default_rng(5)
        actions = rng.uniform(0, 1, (7, 10, 3))
        actions[:, :, 0] = 2 * actions[:, :, 0] - 1
        original = actions.copy()
        speed = np.linspace(-25, 65, 7)
        omega = rng.uniform(-80, 100, (7, 4))
        omega_original = omega.copy()
        state = PhysicsState(speed, lateral_speed=np.linspace(-5, 5, 7), wheel_omega=omega)
        first, second = predict(state, actions), predict(state, actions)
        for name, value in first.items():
            if name == "final_state":
                continue
            value = np.asarray(value)
            self.assertEqual(value.shape, (7, 10, 4) if name == "wheel_omega" else (7, 10))
            self.assertTrue(np.isfinite(value).all(), name)
            np.testing.assert_array_equal(value, second[name])
        np.testing.assert_array_equal(actions, original)
        np.testing.assert_array_equal(omega, omega_original)
        np.testing.assert_array_equal(speed, np.linspace(-25, 65, 7))
        one = predict(PhysicsState(speed[3], lateral_speed=0, wheel_omega=omega[3]), actions[3:4])
        for name in ("speed", "relative_x", "relative_y", "yaw_delta", "wheel_omega"):
            np.testing.assert_allclose(one[name][0], first[name][3], atol=1e-12)

    def test_raw_hold_horizons_and_state_chaining_agree(self):
        actions = np.array([[[0.2, 0.4, 0], [-0.1, 0, 0.15], [0, 0.3, 0], [0.1, 0, 0]]])
        state = PhysicsState(35, lateral_speed=2, yaw_rate=-0.1, wheel_angle=0.04, gas_state=0.2)
        held = predict(state, actions)
        raw = predict(state, np.repeat(actions, 4, axis=1), decision_dt=RAW_DT)
        for name in held:
            if name != "final_state":
                np.testing.assert_allclose(held[name], raw[name][:, 3::4], atol=1e-12)
        first = predict(state, actions[:, :2])
        second = predict(first["final_state"], actions[:, 2:])
        for name in ("speed", "forward_speed", "lateral_speed", "yaw_rate", "wheel_angle", "gas_state", "wheel_omega"):
            np.testing.assert_allclose(second[name], held[name][:, 2:], atol=1e-12)
        yaw = first["yaw_delta"][:, -1, None]
        x = first["relative_x"][:, -1, None] + np.cos(yaw)*second["relative_x"] - np.sin(yaw)*second["relative_y"]
        y = first["relative_y"][:, -1, None] + np.sin(yaw)*second["relative_x"] + np.cos(yaw)*second["relative_y"]
        np.testing.assert_allclose(x, held["relative_x"][:, 2:], atol=1e-12)
        np.testing.assert_allclose(y, held["relative_y"][:, 2:], atol=1e-12)
        np.testing.assert_allclose(yaw + second["yaw_delta"], held["yaw_delta"][:, 2:], atol=1e-12)
        constant = np.array([[[0.2, 0.3, 0]]])
        long_hold = predict(state, constant, decision_dt=0.32)
        four = predict(state, np.repeat(constant, 4, axis=1))
        np.testing.assert_allclose(long_hold["relative_x"][:, 0], four["relative_x"][:, -1])

    def test_mapping_conditions_and_calibration_are_explicit(self):
        actions = np.array([[[0.1, 0.3, 0]], [[0.1, 0.3, 0]]])
        result = predict({"forward_speed": 20, "friction": [[1], [0.6]], "gas_state": 0.2}, actions)
        np.testing.assert_array_equal(result["final_state"].friction, [[1]*4, [0.6]*4])
        slow = predict(PhysicsState(20), actions, parameters=PhysicsParameters(mass_scale=2))
        normal = predict(PhysicsState(20), actions)
        self.assertLess(slow["speed"][0, 0], normal["speed"][0, 0])

    def test_rejects_bad_shapes_nonfinite_actions_and_fractional_raw_ticks(self):
        state, valid = PhysicsState(10), np.zeros((1, 1, 3))
        for actions in (np.zeros((1, 3)), np.zeros((0, 1, 3)), np.zeros((1, 0, 3)), [[[np.nan, 0, 0]]], [[[1.1, 0, 0]]], [[[0, -0.1, 0]]], [[[0, 0, 2]]]):
            with self.subTest(actions=np.shape(actions)), self.assertRaises(ValueError):
                predict(state, actions)
        for dt in (0, -0.02, 0.01, 0.03, np.inf, np.nan):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                predict(state, valid, decision_dt=dt)
        for field, value in (("forward_speed", np.nan), ("wheel_angle", np.inf), ("gas_state", -0.1), ("friction", -1), ("engine_multiplier", np.inf), ("wheel_omega", [1, 2])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                predict(replace(state, **{field: value}), valid)
        for scales in (PhysicsParameters(mass_scale=0), PhysicsParameters(yaw_inertia_scale=-1), PhysicsParameters(tire_stiffness_scale=np.nan)):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                predict(state, valid, parameters=scales)


if __name__ == "__main__":
    unittest.main()
