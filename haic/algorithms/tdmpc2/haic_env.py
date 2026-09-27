"""Minimal HAIC observation/action/episode adapter for the TD-MPC2 pixel model."""

from __future__ import annotations

import cv2
import numpy as np


OBS_SHAPE = (4, 64, 64)
ACTION_DIM = 3
FRAME_SKIP = 4


def model_observation(observation: np.ndarray) -> np.ndarray:
    """Retain HAIC's four gray history frames, resize to the official 64px encoder."""
    array = np.asarray(observation)
    if array.shape != (4, 84, 84) or array.dtype != np.float32:
        raise ValueError("HAIC observation must be float32 [4,84,84]")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("HAIC observation must be finite and in [0,1]")
    pixels = np.rint(array * 255.0).astype(np.uint8)
    resized = cv2.resize(pixels.transpose(1, 2, 0), (64, 64), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized.transpose(2, 0, 1))


def environment_action(action: np.ndarray) -> np.ndarray:
    """Map symmetric TD-MPC2 coordinates to HAIC [steer, gas, brake]."""
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("TD-MPC2 action must be a finite 3-vector")
    if (array < -1).any() or (array > 1).any():
        raise ValueError("TD-MPC2 action must be in [-1,1]")
    return np.array([array[0], (array[1] + 1) / 2, (array[2] + 1) / 2], dtype=np.float32)


def model_action(action: np.ndarray) -> np.ndarray:
    """Invert the linear native action mapping without exclusive gas/brake logic."""
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("HAIC action must be a finite 3-vector")
    if array[0] < -1 or array[0] > 1 or (array[1:] < 0).any() or (array[1:] > 1).any():
        raise ValueError("HAIC action must be in [-1,1] x [0,1] x [0,1]")
    return np.array([array[0], array[1] * 2 - 1, array[2] * 2 - 1], dtype=np.float32)


def episode_boundary(terminated: bool, truncated: bool, info: dict) -> tuple[bool, bool]:
    """Return (reset, true terminal); a finish is terminal despite HAIC truncation."""
    reset = bool(terminated or truncated)
    terminal = bool(terminated or info.get("finished", False))
    if terminal and not reset:
        raise ValueError("terminal outcome without environment boundary")
    return reset, terminal


def make_training_env(max_steps: int):
    """Use the unmodified official wrapper, raw reward, frame skip, and time limits."""
    import gymnasium as gym

    from core.vendor.car_racing import CarRacing
    from env_wrapper import CarEnvironment

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    raw = gym.wrappers.TimeLimit(
        CarRacing(continuous=True, render_mode=None),
        max_episode_steps=max_steps * FRAME_SKIP + 200,
    )
    wrapped = CarEnvironment(raw, skip_frames=FRAME_SKIP)
    return gym.wrappers.TimeLimit(wrapped, max_episode_steps=max_steps)
