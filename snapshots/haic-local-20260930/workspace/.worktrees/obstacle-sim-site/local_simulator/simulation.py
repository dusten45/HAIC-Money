from __future__ import annotations

from .policies import Policy
from .schema import MapDocument
from .session import SimulationSession
from .simulation_types import RunLog


def run_episode(
    document: MapDocument,
    policy: Policy,
    record_frames: bool = False,
) -> RunLog:
    session = SimulationSession.start(
        document,
        policy,
        record_frames=record_frames,
    )
    try:
        for _ in range(document.max_steps):
            if session.done:
                break
            session.step()
        return session.finish()
    except Exception:
        session.close()
        raise


__all__ = ["run_episode"]
