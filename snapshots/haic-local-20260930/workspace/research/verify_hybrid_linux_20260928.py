"""Smoke the frozen submission archive in the Linux CPU container."""

import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import zipfile

import numpy as np


ROOT = Path("/workspace")
ARCHIVE = ROOT / "artifacts/haic-research-v2/score-confirm-hybrid-20260928/submission-obstacle-hybrid.zip"


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(destination)
        os.chdir(destination)
        sys.path.insert(0, str(destination))
        import agent

        started = time.monotonic()
        driver = agent.Agent()
        init_s = time.monotonic() - started
        observation = np.zeros((4, 84, 84), dtype=np.float32)
        started = time.monotonic()
        driver.reset(observation)
        reset_s = time.monotonic() - started
        started = time.monotonic()
        action = driver.act(observation)
        act_s = time.monotonic() - started
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        assert Path(agent.__file__).resolve() == destination / "agent.py"
        assert action.shape == (3,) and np.isfinite(action).all()
        assert init_s < 10 and reset_s < 5 and act_s < 5
        assert rss_bytes < 1_024 * 1024 * 1024
        sys.path.append(str(ROOT))
        from training.evaluate_closed_loop import run_episode

        result = run_episode(mode="linux_package_smoke", track_id=2, seed=132,
                             agent=driver, max_decisions=2000, plan_budget_seconds=4.5)
        assert result["invalid_actions"] == 0 and result["act_max_ms"] < 5000
        print(json.dumps({
            "python": sys.version.split()[0],
            "init_s": init_s, "reset_s": reset_s, "act_s": act_s,
            "rss_bytes": rss_bytes,
            "episode": {key: result[key] for key in ("completed", "lapTimeMs", "progress", "retire_reason", "collisions", "damage", "invalid_actions", "act_max_ms")},
        }, indent=2))


if __name__ == "__main__":
    main()
