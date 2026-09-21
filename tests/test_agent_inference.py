import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


class _Policy:
    def eval(self):
        return self

    def encode_observation(self, observations):
        return torch.zeros(observations.shape[0], 128)

    def action_parameters(self, latent):
        return torch.tensor([[0.2, 0.25]]), torch.full((1, 2), -3.0)


class _InvalidPolicy(_Policy):
    def action_parameters(self, latent):
        return torch.full((1, 2), float("nan")), torch.zeros(1, 2)


class _Dynamics:
    def eval(self):
        return self

    def predict(self, latent, action):
        return SimpleNamespace(
            next_latent=latent,
            progress_delta=torch.zeros(action.shape[0]),
            reward=torch.zeros(action.shape[0]),
            collision_probability=torch.zeros(action.shape[0]),
            off_track_probability=torch.zeros(action.shape[0]),
            uncertainty=torch.zeros(action.shape[0]),
        )


class _NoPlan:
    def reset(self):
        self.reset_calls = getattr(self, "reset_calls", 0) + 1

    def plan(self, *args, **kwargs):
        return None


class _AdvancingClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class _SlowForwardDynamics(_Dynamics):
    def __init__(self, clock):
        self.clock = clock

    def predict(self, latent, action):
        self.clock.advance(2.0)
        return SimpleNamespace(
            next_latent=latent,
            progress_delta=torch.zeros(action.shape[0]),
            reward=action[:, 0],
            collision_probability=torch.zeros(action.shape[0]),
            off_track_probability=torch.zeros(action.shape[0]),
            uncertainty=torch.zeros(action.shape[0]),
        )


class _NegativeSteerPolicy(_Policy):
    def action_parameters(self, latent):
        return torch.tensor([[-1.0, 0.25]]), torch.full((1, 2), -10.0)


class TestAgentInference(unittest.TestCase):
    def test_default_checkpoint_names_load_the_visual_policy_and_dynamics_ensemble(self):
        # Break caught: default construction must load the trained PPO/CEM
        # pair packaged for evaluation instead of silently disabling planning.
        import agent as agent_module
        from agent import Agent, DYNAMICS_MODEL_FILENAME, POLICY_MODEL_FILENAME
        from haic_agent.dynamics import LatentDynamicsEnsemble
        from haic_agent.networks import VisualActorCritic

        self.assertEqual(POLICY_MODEL_FILENAME, "policy.pt")
        self.assertEqual(DYNAMICS_MODEL_FILENAME, "dynamics.pt")
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            policy_path = directory_path / POLICY_MODEL_FILENAME
            dynamics_path = directory_path / DYNAMICS_MODEL_FILENAME
            policy = VisualActorCritic()
            dynamics = LatentDynamicsEnsemble()
            torch.save({"model_state": policy.state_dict()}, policy_path)
            torch.save({"model": dynamics.state_dict()}, dynamics_path)
            with patch.object(
                agent_module, "MODEL_FILENAME", str(directory_path / "missing-baseline.pt")
            ), patch.object(agent_module, "POLICY_MODEL_FILENAME", str(policy_path)), patch.object(
                agent_module, "DYNAMICS_MODEL_FILENAME", str(dynamics_path)
            ):
                loaded = Agent()

        self.assertIsNotNone(loaded.dynamics)
        torch.testing.assert_close(next(loaded.policy.parameters()), next(policy.parameters()))

    def test_local_runner_exposes_same_default_planning_budget(self):
        # Break caught: a runner-only hardcoded timeout makes closed-loop smoke
        # results use a different planner budget than the submitted Agent.
        from local_runner import build_argument_parser

        args = build_argument_parser().parse_args([])
        self.assertEqual(args.plan_budget, 4.5)

    def test_policy_fallback_and_invalid_policy_output_are_always_bounded(self):
        # Break caught: an unavailable plan or NaN actor parameters can escape
        # directly through act() and cause local invalid-action retirement.
        from agent import Agent

        pixels = np.zeros((4, 84, 84), dtype=np.float32)
        fallback = Agent(policy=_Policy(), dynamics=None, planner=_NoPlan())
        invalid = Agent(policy=_InvalidPolicy(), dynamics=None, planner=_NoPlan())

        action = fallback.act(pixels)
        invalid_action = invalid.act(pixels)
        np.testing.assert_allclose(action, np.array([np.tanh(0.2), 0.02 * np.tanh(0.25), 0.0], dtype=np.float32))
        for value in (action, invalid_action):
            self.assertEqual(value.shape, (3,))
            self.assertTrue(np.all(np.isfinite(value)))
            self.assertGreaterEqual(value[0], -1.0)
            self.assertLessEqual(value[0], 1.0)
            self.assertTrue(np.all((value[1:] >= 0.0) & (value[1:] <= 1.0)))

    def test_reset_clears_planner_state_between_episodes(self):
        # Break caught: planner cache from a prior episode must never decide
        # the first action of a new track/seed episode.
        from agent import Agent

        planner = _NoPlan()
        agent = Agent(policy=_Policy(), dynamics=None, planner=planner)
        agent.reset(np.zeros((4, 84, 84), dtype=np.float32))
        agent.reset(np.zeros((4, 84, 84), dtype=np.float32))
        self.assertEqual(planner.reset_calls, 2)

    def test_input_validation_and_exhausted_plan_budget_return_safe_action(self):
        # Break caught: malformed frames and an immediately elapsed deadline
        # must return a valid fallback before the evaluator's five-second cap.
        from agent import Agent
        from haic_agent.planner import CEMPlanner

        agent = Agent(
            policy=_Policy(),
            dynamics=_Dynamics(),
            planner=CEMPlanner(horizon=6, population=32, iterations=4),
            plan_budget=0.0,
        )
        exhausted = agent.act(np.zeros((4, 84, 84), dtype=np.float32))
        malformed = agent.act(np.zeros((3, 84, 84), dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(exhausted)))
        self.assertTrue(np.all(np.isfinite(malformed)))

    def test_explicit_planner_disable_never_loads_or_calls_dynamics(self):
        # Break caught: PPO-only evaluation must stay PPO-only even when a
        # packaged dynamics.pt is present beside the agent source.
        from agent import Agent

        agent = Agent(policy=_Policy(), dynamics=_Dynamics(), planner=_NoPlan(), planner_enabled=False)
        action = agent.act(np.zeros((4, 84, 84), dtype=np.float32))

        self.assertIsNone(agent.dynamics)
        self.assertFalse(agent.planner_enabled)
        self.assertTrue(np.all(np.isfinite(action)))

    def test_prediction_that_crosses_deadline_returns_immediate_policy_fallback(self):
        # Break caught: a dynamics call can itself consume the remaining
        # budget; its late CEM score must never replace the timely PPO action.
        from agent import Agent
        from haic_agent.planner import CEMPlanner

        clock = _AdvancingClock()
        planner = CEMPlanner(horizon=1, population=2, iterations=1, candidate_batch_size=2, clock=clock)
        planner._cached_unconstrained = torch.tensor([[2.0, 0.25]])
        agent = Agent(
            policy=_NegativeSteerPolicy(),
            dynamics=_SlowForwardDynamics(clock),
            planner=planner,
            plan_budget=1.0,
            clock=clock,
        )

        action = agent.act(np.zeros((4, 84, 84), dtype=np.float32))

        expected = np.array([np.tanh(-1.0), 0.02 * np.tanh(0.25), 0.0], dtype=np.float32)
        np.testing.assert_allclose(action, expected)
        self.assertGreater(clock(), 1.0)

    def test_small_cpu_models_finish_an_act_under_default_budget(self):
        # Break caught: deadline must include encoding and actor work, not only
        # CEM internals, so an ordinary call leaves the five-second margin.
        from agent import Agent
        from haic_agent.planner import CEMPlanner

        agent = Agent(
            policy=_Policy(),
            dynamics=_Dynamics(),
            planner=CEMPlanner(horizon=3, population=16, iterations=2, candidate_batch_size=4),
        )
        started = time.monotonic()
        action = agent.act(np.zeros((4, 84, 84), dtype=np.float32))
        elapsed = time.monotonic() - started
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertLess(elapsed, 4.5)


if __name__ == "__main__":
    unittest.main()
