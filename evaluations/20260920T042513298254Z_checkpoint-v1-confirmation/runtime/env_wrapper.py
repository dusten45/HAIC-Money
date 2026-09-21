import gymnasium as gym
import cv2
import numpy as np

from damage import CollisionDamage

def image_preprocessing(img):
    img = cv2.resize(img, dsize=(84, 84))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (img / 255.0).astype(np.float32)

class CarEnvironment(gym.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, no_operation=50, max_off_track_steps=100, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames
        self.stack_state = None
        self.damage = CollisionDamage()
        self.max_off_track_steps = max_off_track_steps
        self.off_track_counter = 0
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._stack_frames, 84, 84),
            dtype=np.float32,
        )

    @property
    def warmup_steps(self):
        return self._no_operation

    def _apply_damage_effects(self):
        effects = self.damage.effects
        self.unwrapped.car.set_damage_effects(
            effects.grip_multiplier,
            effects.engine_multiplier,
            effects.steering_multiplier,
        )

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)

        for _ in range(self._no_operation):
            observation, _, terminated, truncated, _ = self.env.step(np.array([0.0, 0.0, 0.0]))
            if terminated or truncated:
                observation, info = self.env.reset(seed=seed, options=options)

        self.damage.reset()
        self._apply_damage_effects()
        self.off_track_counter = 0
        observation = image_preprocessing(observation)
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        total_reward = 0
        collision = False
        for _ in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            collision = collision or bool(info.get("collision", False))
            if terminated or truncated:
                break

        crashed = self.damage.update(collision)
        self._apply_damage_effects()

        if total_reward < 0:
            self.off_track_counter += 1
        else:
            self.off_track_counter = 0
        off_track = self.off_track_counter > self.max_off_track_steps

        observation = image_preprocessing(observation)
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)

        progress = self._calculate_progress()

        retire_reason = None
        if crashed:
            retire_reason = "crash"
        elif off_track:
            retire_reason = "off_track"

        info = dict(info)
        info.update(
            collision=collision,
            damage=self.damage.damage,
            damage_effects=self.damage.effects,
            progress=progress,
            finish_qualified=getattr(self.unwrapped, "finish_qualified_time_s", None) is not None,
            finish_qualified_time_s=getattr(self.unwrapped, "finish_qualified_time_s", None),
            finish_time_s=getattr(self.unwrapped, "finish_time_s", None),
            finished=getattr(self.unwrapped, "finish_time_s", None) is not None,
            retire_reason=retire_reason,
        )
        terminated = terminated or crashed or off_track
        return self.stack_state, total_reward, terminated, truncated, info

    def _calculate_progress(self) -> float:
        try:
            visited_tiles = self.unwrapped.tile_visited_count
            total_tiles = len(self.unwrapped.track)
            if total_tiles <= 0:
                return 0.0
            return min(1.0, max(0.0, float(visited_tiles / total_tiles)))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 0.0
