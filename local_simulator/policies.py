from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Policy interface used by the local simulation runner."""

    def reset(self, observation: np.ndarray) -> None:
        ...

    def act(self, observation: np.ndarray) -> np.ndarray:
        ...


class BaselinePolicy:
    """A small deterministic policy for validating the simulator pipeline."""

    name = "baseline"

    def reset(self, observation: np.ndarray) -> None:
        del observation

    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
