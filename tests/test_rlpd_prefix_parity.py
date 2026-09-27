"""Synthetic contracts only: no Box2D environment, driving, or road allocation."""

import dataclasses
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from haic.algorithms.rlpd.prefix_parity import (
    INITIAL_HASH, ParityError, _compare_replay_prefix, capture_prefix,
    snapshot, verify_replay_prefix,
)


class RoadTile:
    def __init__(self, idx):
        self.idx = idx
        self.road_visited = False
        self.road_friction = 1.0
        self.position = (0.0, 0.0)
        self.angle = 0.0
        self.fixtures = [SimpleNamespace(
            sensor=True,
            filterData=SimpleNamespace(categoryBits=1, maskBits=65535, groupIndex=0),
            shape=SimpleNamespace(vertices=[
                (float(idx), float(idx)), (float(idx) + 1.0, float(idx)),
                (float(idx) + 1.0, float(idx) + 1.0), (float(idx), float(idx) + 1.0),
            ]))]


class CarRacing:
    def __init__(self, track_id=1, seed=123):
        self.track_id = track_id
        self.track_seed = seed
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.track = [(float(i), float(i), float(i + 1), float(i * 2)) for i in range(4)]
        self.road = [RoadTile(i) for i in range(4)]
        self.obstacles = [SimpleNamespace(position=(3.0, 4.0), angle=0.0,
                                          userData=SimpleNamespace(hit=False),
                                          fixtures=[SimpleNamespace(
                                              shape=SimpleNamespace(radius=1.25), sensor=False)])]
        self.tile_visited_count = 0
        self.t = 0.0
        self.reward = 0.0
        self.prev_reward = 0.0
        self.new_lap = False
        self._collision_this_step = False
        self.finish_qualified_time_s = None
        self.finish_time_s = None
        self.finish_line_tracker = SimpleNamespace(
            center=(1.0, 2.0), forward=(0.0, 1.0), half_width=1.0,
            half_depth=0.5, qualification_ratio=0.95,
            departed_start_area=False, crossing_from_back=False,
            previous_longitudinal=0.0, previous_lateral=0.0,
            qualified_time_s=None, candidate_crossing_time_s=None, finish_time_s=None,
        )
        self.car = SimpleNamespace(
            hull=SimpleNamespace(position=(1.0, 2.0), linearVelocity=(0.0, 0.0),
                                 angle=0.0, angularVelocity=0.0),
            wheels=[SimpleNamespace(
                position=(float(i), 2.0), linearVelocity=(0.0, 0.0),
                angle=0.0, angularVelocity=0.0, gas=0.0, brake=0.0,
                steer=0.0, phase=0.0, omega=0.0, wheel_rad=0.5,
                tiles=set(), skid_start=None, skid_particle=None,
                joint=SimpleNamespace(angle=0.0, motorSpeed=0.0),
            ) for i in range(4)],
            particles=[], fuel_spent=0.0, grass_friction_multiplier=0.6,
            grip_multiplier=1.0, engine_multiplier=1.0,
            steering_multiplier=1.0,
        )
        self.active = False
        self.raw_actions = []
        self.state = np.zeros((96, 96, 3), dtype=np.uint8)
        self.step(None)  # Original CarRacing reset advances one raw frame.
        return self.state.copy(), {}

    def step(self, action):
        self.t += 0.02
        self.state = np.full((96, 96, 3), round(self.t * 10) % 256, dtype=np.uint8)
        if action is not None and self.active:
            self.raw_actions.append(np.asarray(action).copy())
            if not self.road[0].road_visited:
                self.road[0].road_visited = True
                self.tile_visited_count += 1
            self.car.hull.position = (self.car.hull.position[0] + 0.01, 2.0)
            self.car.wheels[0].omega += 0.1
            self.car.wheels[0].tiles.add(self.road[0])
            self.finish_line_tracker.departed_start_area = True
            self.finish_line_tracker.previous_longitudinal += 0.01
            self.reward += 0.25
            self.prev_reward = self.reward
        return self.state.copy(), 0.25 if action is not None and self.active else 0.0, False, False, {}


class TimeLimit:
    def __init__(self, env, max_episode_steps, hook=None):
        self.env = env
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.hook = hook
        self.reward_override = None
        self.flip_terminated = False
        self.flip_truncated = False

    def reset(self):
        self._elapsed_steps = 0
        result = self.env.reset()
        if self.hook is not None:
            self.hook("reset", self)
        return result

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        timed_out = self._elapsed_steps >= self._max_episode_steps
        if self.hook is not None:
            self.hook("step", self)
        if self.reward_override is not None:
            reward = self.reward_override
        return (observation, reward, terminated or self.flip_terminated,
                truncated or self.flip_truncated or timed_out, info)


class DamageManager:
    def __init__(self):
        self.damage = 0.0
        self.override_effects = None

    @property
    def effects(self):
        return self.override_effects or SimpleNamespace(
            grip_multiplier=1.0 - 0.5 * self.damage,
            engine_multiplier=1.0 - 0.25 * self.damage,
            steering_multiplier=1.0 - 0.25 * self.damage,
        )


class CarEnvironment:
    def __init__(self, env):
        self.env = env
        self._skip_frames = 4
        self._stack_frames = 4
        self._no_operation = 50
        self.max_off_track_steps = 100
        self.off_track_counter = 0
        self.damage = DamageManager()
        self.stack_state = None
        self.last_action_bytes = []

    def reset(self):
        self.env.reset()
        for _ in range(self._no_operation):
            image, _, _, _, _ = self.env.step(np.zeros(3, dtype=np.float32))
        self.env.env.active = True
        gray = np.full((84, 84), float(image[0, 0, 0]) / 255.0, dtype=np.float32)
        self.stack_state = np.tile(gray, (4, 1, 1))
        return self.stack_state.copy(), {}

    def step(self, action):
        self.last_action_bytes.append(action.tobytes())
        raw_action = np.asarray(action, dtype=np.float64)
        reward = 0.0
        for _ in range(self._skip_frames):
            image, value, _, _, _ = self.env.step(raw_action)
            reward += value
        gray = np.full((84, 84), float(image[0, 0, 0]) / 255.0, dtype=np.float32)
        self.stack_state = np.concatenate((self.stack_state[1:], gray[None]))
        raw = self.env.env
        return self.stack_state.copy(), reward, False, False, {
            "collision": False, "damage": self.damage.damage, "damage_effects": self.damage.effects,
            "progress": raw.tile_visited_count / len(raw.road), "finish_qualified": False,
            "finish_qualified_time_s": None, "finish_time_s": None, "finished": False,
            "retire_reason": None,
        }


class HaicTrack:
    def __init__(self, env, track_id=1, seed=123):
        self.env = env
        self.track_id = track_id
        self.seed = seed
        self.obstacles = True

    def reset(self):
        observation, info = self.env.reset()
        return observation, {**info, "track_id": self.track_id, "seed": self.seed}

    def step(self, action):
        return self.env.step(action)


class StatelessAgent:
    def __init__(self, action):
        self.action = action
        self.observations = []

    def reset(self, observation):
        self.observations.append(observation.copy())

    def act(self, observation):
        self.observations.append(observation.copy())
        return self.action.copy()


class PrefixParityTests(unittest.TestCase):
    ACTIONS = (np.array([0.25, 0.5, 0.0], dtype=np.float32),
               np.array([-0.2, 0.0, 0.4], dtype=np.float32))

    def environment(self, hook=None):
        raw = CarRacing()
        inner = TimeLimit(raw, 8 * 4 + 200)
        wrapper = CarEnvironment(inner)
        return TimeLimit(HaicTrack(wrapper), 8, hook=hook)

    def setUp(self):
        self.expected = capture_prefix(self.environment(), track_id=1, seed=123,
                                       official_actions=self.ACTIONS)

    def require_blocked(self, env, *, expected=None, actor=None, message=None):
        dispatch_calls = []

        def dispatch(_observation):
            dispatch_calls.append(True)

        with self.assertRaises(ParityError) as error:
            anchor = _compare_replay_prefix(env, expected or self.expected, actor=actor)
            dispatch(anchor)
        if message is not None:
            self.assertIn(message, str(error.exception))
        self.assertEqual(dispatch_calls, [], "branch dispatch was reached on parity failure")

    def test_valid_synthetic_prefix_is_observation_only(self):
        env = self.environment()
        anchor = _compare_replay_prefix(env, self.expected)
        self.assertEqual(anchor.shape, (4, 84, 84))
        self.assertEqual(anchor.dtype, np.float32)
        self.assertEqual(env.env.env.last_action_bytes,
                          [value.tobytes() for value in self.ACTIONS])
        raw_actions = env.env.env.env.env.raw_actions
        self.assertEqual(len(raw_actions), 8)
        for index, raw_action in enumerate(raw_actions):
            self.assertEqual(raw_action.dtype, np.float64)
            np.testing.assert_array_equal(raw_action, self.ACTIONS[index // 4].astype(np.float64))
        self.assertEqual([step.raw_frame_count for step in self.expected.steps], [4, 4])
        self.assertEqual(self.expected.initial.prefix_sha256, INITIAL_HASH)
        self.assertEqual(self.expected.road_sha256, dict(self.expected.initial.state)["road_sha256"])
        self.assertEqual(dict(self.expected.steps[-1].state)["outer.remaining_steps"], 6)

    def test_independent_initial_and_each_action_byte_hash_fail_before_reset(self):
        for changed in (
            dataclasses.replace(self.expected, initial=dataclasses.replace(self.expected.initial,
                                                                            action_sha256="0" * 64)),
            dataclasses.replace(self.expected, initial=dataclasses.replace(self.expected.initial,
                                                                            prefix_sha256="0" * 64)),
            dataclasses.replace(self.expected, actions=(np.array([0.0, 0.5, 0.0], dtype=np.float32).tobytes(),
                                                        self.expected.actions[1])),
            dataclasses.replace(self.expected, steps=(self.expected.steps[0], dataclasses.replace(
                self.expected.steps[1], prefix_sha256="0" * 64))),
        ):
            with self.subTest(changed=changed):
                env = self.environment()
                self.require_blocked(env, expected=changed)
                self.assertIsNone(env._elapsed_steps)

    def test_mutate_exactly_one_source_state_field_per_replay(self):
        changes = {
            "hull pose": lambda e: setattr(e.env.env.env.env.car.hull, "position", (2.0, 3.0)),
            "hull angle": lambda e: setattr(e.env.env.env.env.car.hull, "angle", 0.1),
            "hull linear velocity": lambda e: setattr(e.env.env.env.env.car.hull, "linearVelocity", (1.0, 0.0)),
            "hull angular velocity": lambda e: setattr(e.env.env.env.env.car.hull, "angularVelocity", 0.2),
            "wheel pose": lambda e: setattr(e.env.env.env.env.car.wheels[0], "position", (99.0, 0.0)),
            "wheel omega": lambda e: setattr(e.env.env.env.env.car.wheels[0], "omega", 10.0),
            "wheel linear velocity": lambda e: setattr(e.env.env.env.env.car.wheels[0], "linearVelocity", (0.5, 0.0)),
            "wheel steer": lambda e: setattr(e.env.env.env.env.car.wheels[0], "steer", 0.1),
            "wheel brake": lambda e: setattr(e.env.env.env.env.car.wheels[0], "brake", 0.1),
            "wheel phase": lambda e: setattr(e.env.env.env.env.car.wheels[0], "phase", 0.1),
            "wheel gas": lambda e: setattr(e.env.env.env.env.car.wheels[0], "gas", 0.1),
            "wheel tiles": lambda e: e.env.env.env.env.car.wheels[0].tiles.add(e.env.env.env.env.road[1]),
            "wheel joint": lambda e: setattr(e.env.env.env.env.car.wheels[0].joint, "motorSpeed", 0.2),
            "wheel skid start": lambda e: setattr(e.env.env.env.env.car.wheels[0], "skid_start", (1.0, 2.0)),
            "wheel skid particle": lambda e: setattr(e.env.env.env.env.car.wheels[0], "skid_particle",
                                                     SimpleNamespace(color=(0, 0, 0), ttl=1, poly=[], grass=False)),
            "car particles": lambda e: e.env.env.env.env.car.particles.append(
                SimpleNamespace(color=(0, 0, 0), ttl=1, poly=[(1.0, 2.0)], grass=False)),
            "car fuel spent": lambda e: setattr(e.env.env.env.env.car, "fuel_spent", 0.1),
            "obstacle hit": lambda e: setattr(e.env.env.env.env.obstacles[0].userData, "hit", True),
            "obstacle radius": lambda e: setattr(e.env.env.env.env.obstacles[0].fixtures[0].shape,
                                                 "radius", 2.0),
            "car grass friction": lambda e: setattr(e.env.env.env.env.car,
                                                    "grass_friction_multiplier", 0.9),
            "tile count": lambda e: setattr(e.env.env.env.env, "tile_visited_count", 2),
            "road friction": lambda e: setattr(e.env.env.env.env.road[0], "road_friction", 0.4),
            "road fixture vertex": lambda e: e.env.env.env.env.road[2].fixtures[0].shape.vertices.__setitem__(
                0, (99.0, 99.0)),
            "road fixture sensor": lambda e: setattr(e.env.env.env.env.road[2].fixtures[0],
                                                     "sensor", False),
            "road body position": lambda e: setattr(e.env.env.env.env.road[2],
                                                    "position", (100.0, 0.0)),
            "road body angle": lambda e: setattr(e.env.env.env.env.road[2], "angle", 1.0),
            "road fixture mask": lambda e: setattr(
                e.env.env.env.env.road[2].fixtures[0].filterData, "maskBits", 0),
            "raw time": lambda e: setattr(e.env.env.env.env, "t", 9.0),
            "raw reward": lambda e: setattr(e.env.env.env.env, "reward", 99.0),
            "raw prev_reward": lambda e: setattr(e.env.env.env.env, "prev_reward", 99.0),
            "raw qualified time": lambda e: setattr(e.env.env.env.env, "finish_qualified_time_s", 0.1),
            "raw finish time": lambda e: setattr(e.env.env.env.env, "finish_time_s", 0.1),
            "raw new lap": lambda e: setattr(e.env.env.env.env, "new_lap", True),
            "raw collision flag": lambda e: setattr(e.env.env.env.env, "_collision_this_step", True),
            "frame stack": lambda e: e.env.env.stack_state.__setitem__((0, 0, 0), 0.42),
            "off-track counter": lambda e: setattr(e.env.env, "off_track_counter", 1),
            "damage value": lambda e: setattr(e.env.env.damage, "damage", 0.2),
            "damage effect": lambda e: setattr(e.env.env.damage, "override_effects", SimpleNamespace(
                grip_multiplier=0.7, engine_multiplier=1.0, steering_multiplier=1.0)),
            "car effect": lambda e: setattr(e.env.env.env.env.car, "grip_multiplier", 0.7),
            "engine effect": lambda e: setattr(e.env.env.env.env.car, "engine_multiplier", 0.7),
            "steering effect": lambda e: setattr(e.env.env.env.env.car, "steering_multiplier", 0.7),
            "outer elapsed": lambda e: setattr(e, "_elapsed_steps", 2),
            "inner elapsed": lambda e: setattr(e.env.env.env, "_elapsed_steps", 58),
            "observation image": lambda e: e.env.env.env.env.state.__setitem__((0, 0, 0), 42),
            "road geometry": lambda e: setattr(e.env.env.env.env, "track", [
                (0.0, 0.0, 99.0, 0.0), *e.env.env.env.env.track[1:]]),
        }
        for label, mutate in changes.items():
            with self.subTest(field=label):
                def after_step(phase, outer):
                    if phase == "step" and outer._elapsed_steps == 1:
                        mutate(outer)
                self.require_blocked(self.environment(hook=after_step))

    def test_bitmap_swap_with_unchanged_count_blocks_dispatch(self):
        def swap(phase, env):
            if phase == "step" and env._elapsed_steps == 1:
                raw = env.env.env.env.env
                raw.road[0].road_visited = False
                raw.road[1].road_visited = True
                self.assertEqual(sum(t.road_visited for t in raw.road), raw.tile_visited_count)
        self.require_blocked(self.environment(hook=swap), message="raw.road_visited")

    def test_all_tracker_mutable_fields_have_independent_failures(self):
        for field in sorted(vars(self.environment().env.env.env.env.finish_line_tracker)):
            with self.subTest(field=field):
                def change(phase, env):
                    if phase == "step" and env._elapsed_steps == 1:
                        tracker = env.env.env.env.env.finish_line_tracker
                        value = getattr(tracker, field)
                        if isinstance(value, bool):
                            replacement = not value
                        elif value is None:
                            replacement = 0.1
                        elif isinstance(value, tuple):
                            replacement = (value[0] + 0.1, value[1])
                        else:
                            replacement = value + 0.1
                        setattr(tracker, field, replacement)
                self.require_blocked(self.environment(hook=change), message="finish_tracker." + field)

    def test_missing_or_unrecognized_fields_never_fall_back_to_tile_count(self):
        cases = (
            lambda e: delattr(e.env.env.env.env.road[0], "road_visited"),
            lambda e: delattr(e.env.env.env.env.finish_line_tracker, "candidate_crossing_time_s"),
            lambda e: setattr(e.env.env.env.env.finish_line_tracker, "new_crossing_phase", True),
            lambda e: delattr(e.env.env, "off_track_counter"),
            lambda e: delattr(e, "_elapsed_steps"),
            lambda e: setattr(e.env.env.damage, "override_effects", SimpleNamespace(
                grip_multiplier=1.0, engine_multiplier=1.0)),
        )
        for mutate in cases:
            with self.subTest(mutate=mutate):
                def change(phase, env):
                    if phase == "step" and env._elapsed_steps == 1:
                        mutate(env)
                self.require_blocked(self.environment(hook=change))

    def test_horizon_and_cell_identity_at_reset_block_dispatch(self):
        for mutate in (
            lambda e: setattr(e, "_max_episode_steps", 9),
            lambda e: setattr(e.env.env.env, "_max_episode_steps", 233),
            lambda e: setattr(e.env, "seed", 456),
            lambda e: setattr(e.env.env.env.env, "track_seed", 456),
            lambda e: setattr(e.env.env.env.env, "track_id", 2),
            lambda e: setattr(e.env.env.env.env, "track", [
                (10.0, 0.0, 1.0, 0.0), *e.env.env.env.env.track[1:]]),
        ):
            with self.subTest(mutate=mutate):
                def after_reset(phase, env):
                    if phase == "reset":
                        mutate(env)
                self.require_blocked(self.environment(hook=after_reset))

    def test_step_reward_and_terminal_flags_are_exact_and_terminal_cannot_anchor(self):
        for attr, value in (("reward_override", 42.0), ("flip_terminated", True),
                            ("flip_truncated", True)):
            with self.subTest(field=attr):
                env = self.environment()
                setattr(env, attr, value)
                self.require_blocked(env)
        terminal = dataclasses.replace(self.expected, steps=(self.expected.steps[0], dataclasses.replace(
            self.expected.steps[1], terminated=True)))
        def terminal_step(phase, env):
            if phase == "step" and env._elapsed_steps == 2:
                env.flip_terminated = True
        # Even matching recorded/observed terminal flags do not yield an anchor.
        self.require_blocked(self.environment(hook=terminal_step), expected=terminal,
                             message="terminal prefix")

    def test_actor_action_parity_is_separate_and_does_not_reencode_replay(self):
        action = np.array([0.25, 0.5, 0.0], dtype=np.float32)
        policy_prefix = capture_prefix(self.environment(), track_id=1, seed=123,
                                       official_actions=(action, action), actor=StatelessAgent(action))
        matching_actor = StatelessAgent(action)
        self.assertEqual(_compare_replay_prefix(self.environment(), policy_prefix,
                                              actor=matching_actor).shape, (4, 84, 84))
        self.require_blocked(self.environment(), expected=policy_prefix, message="actor required")
        self.require_blocked(self.environment(), expected=policy_prefix,
                             actor=StatelessAgent(np.array([0.1, 0.5, 0.0], dtype=np.float32)),
                             message="policy-action parity")
        self.require_blocked(self.environment(),
                             actor=StatelessAgent(np.array([0.1, 0.5, 0.0], dtype=np.float32)),
                             message="policy-action parity")

    def test_public_replay_rejects_self_recorded_prefix_before_reset(self):
        env = self.environment()
        with self.assertRaisesRegex(ParityError, "independently frozen original G0"):
            verify_replay_prefix(env, self.expected)
        self.assertIsNone(env._elapsed_steps)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ParityError, "missing frozen source"):
                verify_replay_prefix(env, self.expected, source_root=Path(directory), actor_id="alpha")
        self.assertIsNone(env._elapsed_steps)

    def test_public_replay_binds_and_checks_original_before_reset(self):
        env = self.environment()
        source = object()
        calls = []

        def bound(*args, **kwargs):
            self.assertIsNone(env._elapsed_steps)
            self.assertEqual(kwargs["anchor_decisions"], 2)
            self.assertEqual(kwargs["geometry_seed"], 123)
            calls.append("bound")
            return source

        def checked(recorded, captured):
            self.assertIs(recorded, source)
            self.assertIs(captured, self.expected)
            self.assertIsNone(env._elapsed_steps)
            calls.append("checked")

        with patch("haic.algorithms.rlpd.recorded_prefix.bind_original_g0_prefix", side_effect=bound), \
             patch("haic.algorithms.rlpd.recorded_prefix.verify_captured_prefix", side_effect=checked):
            anchor = verify_replay_prefix(env, self.expected,
                                          source_root=Path("/tmp/kilo"), actor_id="alpha")
        self.assertEqual(calls, ["bound", "checked"])
        self.assertEqual(anchor.shape, (4, 84, 84))

    def test_raw_boundary_rejects_float64_substitution_without_reaching_anchor(self):
        env = self.environment()
        original_step = env.env.env.step

        def substituted(action):
            widened = np.asarray(action, dtype=np.float64)
            widened[1] = 0.9
            return original_step(widened.astype(np.float32))

        env.env.env.step = substituted
        self.require_blocked(env, message="raw CarRacing.step action differs")

    def test_reject_wrong_observation_action_and_wrapper_contract(self):
        env = self.environment()
        env.reset()
        with self.assertRaises(ParityError):
            snapshot(env, np.zeros((84, 84, 4), dtype=np.float32), track_id=1, seed=123,
                     info={"track_id": 1, "seed": 123})
        for action in (np.zeros(3, dtype=np.float64), np.array([0.0, 1.1, 0.0], dtype=np.float32),
                       np.array([np.nan, 0.0, 0.0], dtype=np.float32)):
            with self.subTest(action=action):
                with self.assertRaises(ParityError):
                    capture_prefix(self.environment(), track_id=1, seed=123, official_actions=(action,))
        env = self.environment()
        env.env.env._stack_frames = 5
        self.require_blocked(env)


if __name__ == "__main__":
    unittest.main()
