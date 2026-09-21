import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from local_simulator.schema import MapSpec


class TestRunLog(unittest.TestCase):
    def _run_api(self):
        try:
            from local_simulator.logging import (
                load_run_log,
                run_log_from_dict,
                save_run_log,
            )
            from local_simulator.policies import BaselinePolicy, ManualPolicy
            from local_simulator.session import SimulationSession
            from local_simulator.simulation import run_episode
        except ImportError as error:
            self.fail(f"run logging API is missing: {error}")
        return (
            load_run_log,
            run_log_from_dict,
            save_run_log,
            BaselinePolicy,
            ManualPolicy,
            SimulationSession,
            run_episode,
        )

    def test_baseline_run_has_schema_and_steps(self):
        _, _, _, BaselinePolicy, _, _, run_episode = self._run_api()
        spec = MapSpec(1, 42, "custom_only", (), 2, 4)

        result = run_episode(spec, BaselinePolicy())

        self.assertEqual(result.schema_version, 2)
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].action, (0.0, 1.0, 0.0))
        self.assertGreater(len(result.track.points), 0)
        self.assertIn("lap_time_ms", result.summary)
        self.assertIn("map_ref", result.run)
        self.assertEqual(result.run["environment"], "official-v1")

    def test_log_round_trip(self):
        load_run_log, _, save_run_log, BaselinePolicy, _, _, run_episode = self._run_api()
        spec = MapSpec(1, 42, "custom_only", (), 1, 4)
        result = run_episode(spec, BaselinePolicy())

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "run.json"
            save_run_log(result, path)
            loaded = load_run_log(path)

        self.assertEqual(loaded.summary, result.summary)
        self.assertEqual(loaded.steps, result.steps)
        self.assertEqual(loaded.track, result.track)

    def test_legacy_schema_one_log_still_loads(self):
        _, run_log_from_dict, _, _, _, _, _ = self._run_api()
        legacy_payload = {
            "schema_version": 1,
            "run": {"map": {"track_id": 1, "seed": 42}},
            "track": {"points": [[0.0, 0.0, 0.0, 0.0]], "width": 6.666},
            "steps": [
                {
                    "step": 0,
                    "sim_time_s": 0.08,
                    "action": [0.0, 1.0, 0.0],
                    "position": [0.0, 0.0],
                    "angle": 0.0,
                    "velocity": [0.0, 0.0],
                    "progress": 0.0,
                    "damage": 0.0,
                    "collision": False,
                    "terminated": False,
                    "truncated": False,
                }
            ],
            "summary": {},
        }

        loaded = run_log_from_dict(legacy_payload)

        self.assertEqual(loaded.schema_version, 2)
        self.assertEqual(len(loaded.steps), 1)
        self.assertEqual(loaded.run["map"]["track_id"], 1)

    def test_manual_session_records_explicit_action(self):
        from local_simulator.track_generator import generate_custom_map

        _, _, _, _, ManualPolicy, SimulationSession, _ = self._run_api()
        document = generate_custom_map("custom-track-session", 99, "oval", max_steps=2)
        session = SimulationSession.start(document, ManualPolicy(), record_frames=False)
        try:
            step = session.step((0.25, 0.5, 0.0))
            result = session.finish()
        finally:
            if not session.closed:
                session.close()

        self.assertEqual(step.action, (0.25, 0.5, 0.0))
        self.assertEqual(result.run["map_ref"]["map_id"], "custom-track-session")

    def test_agent_policy_loads_agent_act(self):
        from local_simulator.policies import AgentPolicy

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            agent_path = root_path / "agent.py"
            agent_path.write_text(
                "import numpy as np\n"
                "class Agent:\n"
                "    def reset(self, observation):\n"
                "        self.was_reset = observation.shape\n"
                "    def act(self, observation):\n"
                "        return np.array([0.25, 0.5, 0.0], dtype=np.float32)\n",
                encoding="utf-8",
            )
            policy = AgentPolicy(agent_path, root_path)
            observation = __import__("numpy").zeros((4, 84, 84), dtype="float32")

            policy.reset(observation)

            self.assertEqual(tuple(policy.act(observation)), (0.25, 0.5, 0.0))

    def test_agent_policy_uses_project_root_checkpoint_from_a_different_cwd(self):
        import torch

        from agent import Baseline1Actor

        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as other_cwd:
            project_root = Path(project)
            foreign_cwd = Path(other_cwd)
            for dependency in ("action_smoothing.py", "action_representation.py"):
                shutil.copy2(repository_root / dependency, project_root / dependency)

            for path, steering in (
                (project_root / "model.pt", 0.75),
                (foreign_cwd / "model.pt", -0.75),
            ):
                actor = Baseline1Actor()
                with torch.no_grad():
                    actor.action_net.weight.zero_()
                    actor.action_net.bias.copy_(torch.tensor([steering, 0.25, 0.0]))
                torch.save(actor.state_dict(), path)

            script = r'''import json
import sys
from pathlib import Path

import numpy as np
from local_simulator.policies import AgentPolicy

agent_path = Path(sys.argv[1])
project_root = Path(sys.argv[2])
policy = AgentPolicy(agent_path, project_root)
action = policy.act(np.zeros((4, 84, 84), dtype=np.float32))
print(json.dumps(action.tolist()))'''
            python_path = os.environ.get("PYTHONPATH", "")
            child_environment = os.environ.copy()
            child_environment["PYTHONPATH"] = os.pathsep.join(
                value for value in (str(repository_root), python_path) if value
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(repository_root / "agent.py"),
                    str(project_root),
                ],
                cwd=foreign_cwd,
                env=child_environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            action = json.loads(result.stdout)
            self.assertGreater(action[0], 0.5)


if __name__ == "__main__":
    unittest.main()
