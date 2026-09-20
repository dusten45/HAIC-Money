import unittest
from types import SimpleNamespace

import torch


class _ForwardDynamics:
    """Small learned-model stand-in with an action-dependent forward reward."""

    def predict(self, latent, action):
        next_latent = latent.clone()
        next_latent[:, 0] = next_latent[:, 0] + action[:, 0]
        return SimpleNamespace(
            next_latent=next_latent,
            progress_delta=action[:, 0],
            reward=action[:, 0],
            collision_probability=torch.zeros(action.shape[0]),
            off_track_probability=torch.zeros(action.shape[0]),
            uncertainty=torch.zeros(action.shape[0]),
        )


class _RiskyDynamics(_ForwardDynamics):
    def predict(self, latent, action):
        prediction = super().predict(latent, action)
        values = dict(prediction.__dict__)
        values["collision_probability"] = (action[:, 0] > 0.0).float()
        values["uncertainty"] = action[:, 0].abs() * 2.0
        return SimpleNamespace(**values)


class _FinalizationClock:
    """Expires only after all candidate scoring/update deadline checks."""

    def __init__(self, expire_after_calls):
        self.calls = 0
        self.expire_after_calls = expire_after_calls

    def __call__(self):
        self.calls += 1
        return 0.0 if self.calls <= self.expire_after_calls else 2.0


class TestCEMPlanner(unittest.TestCase):
    def test_plan_returns_bounded_first_action_and_horizon_sequence(self):
        # Break caught: flattening a CEM population can return a scalar action
        # or an unbounded Normal sample to the CarRacing interface.
        from haic_agent.planner import CEMPlanner

        planner = CEMPlanner(horizon=4, population=24, iterations=3, candidate_batch_size=8)
        result = planner.plan(
            torch.zeros(1, 128),
            torch.full((1, 3), 0.25),
            torch.full((1, 3), -2.0),
            _ForwardDynamics(),
            deadline=float("inf"),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.action.shape, (3,))
        self.assertEqual(result.unconstrained_sequence.shape, (4, 3))
        self.assertTrue(torch.isfinite(result.action).all())
        self.assertGreaterEqual(float(result.action[0]), -1.0)
        self.assertLessEqual(float(result.action[0]), 1.0)
        self.assertTrue(torch.all((result.action[1:] >= 0.0) & (result.action[1:] <= 1.0)))

    def test_warm_start_shifts_previous_unconstrained_sequence_and_reset_isolates_episode(self):
        # Break caught: reusing a plan without a one-decision shift makes the
        # agent execute a stale first action after every fresh observation.
        from haic_agent.planner import CEMPlanner

        planner = CEMPlanner(horizon=3, population=12, iterations=1, candidate_batch_size=6)
        cached = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        planner._cached_unconstrained = cached.clone()

        shifted = planner.warm_start()
        torch.testing.assert_close(shifted, torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.7, 0.8, 0.9]]))
        planner.reset()
        self.assertIsNone(planner.warm_start())

    def test_risk_and_ensemble_uncertainty_change_candidate_score(self):
        # Break caught: scoring only predicted reward lets CEM exploit a model
        # that predicts progress for candidates with collision uncertainty.
        from haic_agent.planner import CEMPlanner

        planner = CEMPlanner(
            horizon=1,
            population=4,
            iterations=1,
            collision_cost=4.0,
            uncertainty_cost=3.0,
        )
        latent = torch.zeros(2, 128)
        actions = torch.tensor([[[-0.25, 0.5, 0.0]], [[0.25, 0.5, 0.0]]])
        scores = planner.score_sequences(latent, actions, _RiskyDynamics(), deadline=float("inf"))

        self.assertIsNotNone(scores)
        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_expired_deadline_or_invalid_model_has_no_plan(self):
        # Break caught: planner exceptions or expired budgets must leave the
        # actor fallback available instead of leaking NaN actions.
        from haic_agent.planner import CEMPlanner

        planner = CEMPlanner(horizon=3, population=8, iterations=1)
        expired = planner.plan(
            torch.zeros(1, 128), torch.zeros(1, 3), torch.zeros(1, 3), _ForwardDynamics(), deadline=-1.0
        )
        invalid = planner.plan(
            torch.zeros(1, 128), torch.full((1, 3), float("nan")), torch.zeros(1, 3), _ForwardDynamics(), deadline=float("inf")
        )

        self.assertIsNone(expired)
        self.assertIsNone(invalid)

    def test_expiry_during_result_finalization_does_not_replace_prior_cache(self):
        # Break caught: result tensor conversion/cache assignment happens after
        # scoring, so a deadline there must reject the plan and preserve the
        # next call's warm start from the prior valid episode state.
        from haic_agent.planner import CEMPlanner

        clock = _FinalizationClock(expire_after_calls=16)
        planner = CEMPlanner(horizon=1, population=2, iterations=1, candidate_batch_size=2, clock=clock)
        prior_cache = torch.tensor([[-3.0, 0.0, 0.0]])
        planner._cached_unconstrained = prior_cache.clone()

        result = planner.plan(
            torch.zeros(1, 128),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.full((1, 3), -10.0),
            _ForwardDynamics(),
            deadline=1.0,
        )

        self.assertIsNone(result)
        torch.testing.assert_close(planner._cached_unconstrained, prior_cache)
        self.assertGreater(clock.calls, 16)


if __name__ == "__main__":
    unittest.main()
