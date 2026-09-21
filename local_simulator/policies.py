from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
from uuid import uuid4
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


class ManualPolicy:
    """Policy whose action is supplied by a local browser session."""

    name = "manual"

    def __init__(self) -> None:
        self._action = np.zeros(3, dtype=np.float32)

    def set_action(self, action: np.ndarray) -> None:
        self._action = np.asarray(action, dtype=np.float32).reshape(-1).copy()

    def reset(self, observation: np.ndarray) -> None:
        del observation
        self._action = np.zeros(3, dtype=np.float32)

    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return self._action.copy()


class AgentPolicy:
    """Adapter that loads the participant Agent from a local workspace."""

    name = "agent"

    def __init__(self, agent_path: Path, project_root: Path) -> None:
        self.agent_path = Path(agent_path).resolve()
        self.project_root = Path(project_root).resolve()
        if not self.agent_path.is_file():
            raise FileNotFoundError(f"agent file does not exist: {self.agent_path}")
        module_name = f"haic_submission_agent_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, self.agent_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load agent module: {self.agent_path}")
        module = importlib.util.module_from_spec(spec)
        project_root_entry = str(self.project_root)
        sys.path.insert(0, project_root_entry)
        try:
            spec.loader.exec_module(module)
            agent_class = getattr(module, "Agent", None)
            if agent_class is None:
                raise ValueError("agent.py must define an Agent class")
            try:
                accepts_project_root = "project_root" in inspect.signature(agent_class).parameters
            except (TypeError, ValueError):
                accepts_project_root = False
            self.agent = (
                agent_class(project_root=str(self.project_root))
                if accepts_project_root
                else agent_class()
            )
        finally:
            for index, entry in enumerate(sys.path):
                if entry is project_root_entry:
                    del sys.path[index]
                    break

    def reset(self, observation: np.ndarray) -> None:
        reset = getattr(self.agent, "reset", None)
        if reset is not None:
            reset(observation)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(self.agent.act(observation), dtype=np.float32)


__all__ = ["AgentPolicy", "BaselinePolicy", "ManualPolicy", "Policy"]
