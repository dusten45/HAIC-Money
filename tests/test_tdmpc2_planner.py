import unittest

import torch

from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner


class FakeModel:
    def __init__(self, prior=0.8, noise=0.0, q=0.0):
        self.prior = prior
        self.noise = noise
        self.q = q
        self.policy_states = []
        self.q_calls = 0

    def encode(self, obs, task=None):
        assert task is None
        return obs.float()

    def pi(self, z, task=None):
        assert task is None
        self.policy_states.append(z.clone())
        action = torch.full((z.shape[0], 1), self.prior, device=z.device, dtype=z.dtype)
        action = action + self.noise * torch.randn_like(action)
        return action, {"mean": action}

    def next(self, z, action, task=None):
        assert task is None
        return z + action[:, :1]

    def reward(self, z, action, task=None):
        assert task is None
        return action[:, :1]

    def Q(self, z, action, task=None, return_type="min"):
        assert task is None and return_type == "avg"
        self.q_calls += 1
        return torch.full((z.shape[0], 1), self.q, device=z.device, dtype=z.dtype)

    def termination(self, z, task=None):
        assert task is None
        return (z >= 0.5).float()


def config(**kwargs):
    defaults = dict(
        action_dim=1,
        discount=0.9,
        num_bins=0,
        num_samples=8,
        num_elites=2,
        num_pi_trajs=1,
        iterations=2,
    )
    defaults.update(kwargs)
    return PlannerConfig(**defaults)


class PlannerTests(unittest.TestCase):
    def test_official_planning_defaults_and_config_validation(self):
        cfg = PlannerConfig(action_dim=1, discount=0.95)
        self.assertEqual((cfg.horizon, cfg.iterations, cfg.num_samples, cfg.num_elites, cfg.num_pi_trajs),
                         (3, 6, 512, 64, 24))
        self.assertEqual((cfg.min_std, cfg.max_std, cfg.temperature), (0.05, 2.0, 0.5))
        with self.assertRaises(ValueError):
            config(num_elites=9)
        with self.assertRaises(ValueError):
            config(num_pi_trajs=9)
        with self.assertRaises(ValueError):
            config(discount=float("nan"))

    def test_action_bounds_and_prior_rolls_forward_one_step_at_a_time(self):
        model = FakeModel(prior=5.0)
        planner = TDMPC2Planner(model, config(num_samples=1, num_elites=1, iterations=1, max_std=4.0))
        action = planner.plan(torch.zeros(1, 1), t0=True, eval_mode=False)
        self.assertEqual(action.shape, (1,))
        self.assertTrue((action >= -1).all() and (action <= 1).all())
        self.assertEqual(len(model.policy_states), 4)  # H prior steps, then terminal Q policy
        torch.testing.assert_close(model.policy_states[0], torch.zeros(1, 1))
        torch.testing.assert_close(model.policy_states[1], torch.ones(1, 1))
        torch.testing.assert_close(model.policy_states[2], torch.full((1, 1), 2.0))
        torch.testing.assert_close(planner.prev_mean, torch.ones(3, 1))

    def test_prior_trajectories_compete_against_gaussian_samples(self):
        torch.manual_seed(9)
        prior = TDMPC2Planner(FakeModel(prior=0.9), config(
            horizon=2, iterations=1, num_samples=6, num_elites=1,
            max_std=0.001, min_std=0.001,
        ))
        with_prior = prior.plan(torch.zeros(1, 1), t0=True, eval_mode=True)
        torch.manual_seed(9)
        gaussian = TDMPC2Planner(FakeModel(prior=0.9), config(
            horizon=2, iterations=1, num_samples=6, num_elites=1,
            num_pi_trajs=0, max_std=0.001, min_std=0.001,
        ))
        without_prior = gaussian.plan(torch.zeros(1, 1), t0=True, eval_mode=True)
        self.assertAlmostEqual(with_prior.item(), 0.9, places=6)
        self.assertLess(without_prior.item(), 0.01)

    def test_previous_mean_shift_and_reset(self):
        planner = TDMPC2Planner(FakeModel(), config(
            num_samples=1, num_elites=1, num_pi_trajs=0, iterations=1,
            min_std=1e-8, max_std=1e-8,
        ))
        planner.prev_mean = torch.tensor([[0.1], [0.8], [-0.3]])
        torch.manual_seed(2)
        shifted = planner.plan(torch.zeros(1, 1), t0=False, eval_mode=True)
        self.assertAlmostEqual(shifted.item(), 0.8, places=5)
        torch.testing.assert_close(planner.prev_mean, torch.tensor([[0.8], [-0.3], [0.0]]), atol=1e-6, rtol=0)
        torch.manual_seed(2)
        initial = planner.plan(torch.zeros(1, 1), t0=True, eval_mode=True)
        self.assertAlmostEqual(initial.item(), 0.0, places=5)
        planner.reset()
        torch.testing.assert_close(planner.prev_mean, torch.zeros(3, 1))

    def test_zero_discount_keeps_immediate_reward_and_skips_bootstrap(self):
        model = FakeModel(q=1_000.0)
        planner = TDMPC2Planner(model, config(discount=0, num_samples=2, num_elites=1, num_pi_trajs=0))
        actions = torch.tensor([[[0.3], [-0.4]], [[0.9], [0.9]], [[0.9], [0.9]]])
        values = planner._estimate_value(torch.zeros(2, 1), actions)
        torch.testing.assert_close(values, torch.tensor([[0.3], [-0.4]]))
        self.assertEqual(model.q_calls, 0)

    def test_termination_keeps_terminal_reward_and_masks_future_q(self):
        model = FakeModel(q=100)
        planner = TDMPC2Planner(model, config(
            discount=0.5, episodic=True, horizon=2, num_samples=2,
            num_elites=1, num_pi_trajs=0,
        ))
        actions = torch.tensor([[[0.7], [0.3]], [[0.9], [0.1]]])
        values = planner._estimate_value(torch.zeros(2, 1), actions)
        torch.testing.assert_close(values, torch.tensor([[0.7], [0.3 + 0.5 * 0.1 + 0.25 * 100]]))

    def test_distributional_reward_decodes_official_symlog_bins(self):
        planner = TDMPC2Planner(FakeModel(), config(num_bins=101))
        logits = torch.full((2, 101), -1000.0)
        logits[:, 55] = 1000.0  # center + 5 bins = symlog(1) for [-10, 10]
        decoded = planner._reward(logits, 2)
        torch.testing.assert_close(decoded, torch.expm1(torch.ones(2, 1)), atol=1e-5, rtol=0)

    def test_nonfinite_active_predictions_fail_closed_without_warm_start_update(self):
        class InvalidModel(FakeModel):
            def reward(self, z, action, task=None):
                return torch.full((z.shape[0], 1), float("nan"))

        planner = TDMPC2Planner(InvalidModel(), config())
        planner.prev_mean.fill_(0.6)
        with self.assertRaisesRegex(FloatingPointError, "finite trajectories"):
            planner.plan(torch.zeros(1, 1), eval_mode=True)
        torch.testing.assert_close(planner.prev_mean, torch.full((3, 1), 0.6))

    def test_nonfinite_active_q_is_rejected_but_terminal_q_is_unused(self):
        class InvalidQModel(FakeModel):
            def Q(self, z, action, task=None, return_type="min"):
                return torch.where(z >= 0.5, torch.full_like(z, float("nan")), torch.ones_like(z))

        planner = TDMPC2Planner(InvalidQModel(), config(
            discount=0.5, episodic=True, horizon=1, num_samples=2,
            num_elites=1, num_pi_trajs=0,
        ))
        actions = torch.tensor([[[0.7], [0.2]]])
        values = planner._estimate_value(torch.zeros(2, 1), actions)
        torch.testing.assert_close(values, torch.tensor([[0.7], [0.2 + 0.5]]))

        class InvalidActiveQ(FakeModel):
            def Q(self, z, action, task=None, return_type="min"):
                return torch.full((z.shape[0], 1), float("nan"))

        invalid = TDMPC2Planner(InvalidActiveQ(), config(
            horizon=1, num_samples=2, num_elites=1, num_pi_trajs=0,
        ))
        with self.assertRaisesRegex(FloatingPointError, "finite trajectories"):
            invalid.plan(torch.zeros(1, 1), eval_mode=True)

    def test_seeded_stochastic_plans_are_reproducible(self):
        cfg = config(num_samples=10, num_elites=3)
        first = TDMPC2Planner(FakeModel(prior=0.4, noise=0.2), cfg)
        second = TDMPC2Planner(FakeModel(prior=0.4, noise=0.2), cfg)
        torch.manual_seed(842)
        action1 = first.plan(torch.zeros(1, 1), t0=True, eval_mode=False)
        torch.manual_seed(842)
        action2 = second.plan(torch.zeros(1, 1), t0=True, eval_mode=False)
        torch.testing.assert_close(action1, action2, rtol=0, atol=0)
        torch.testing.assert_close(first.prev_mean, second.prev_mean, rtol=0, atol=0)

    def test_real_pixel_model_replays_full_eval_trace_after_reset_and_seed(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = WorldModel(TDMPC2ModelConfig(
                obs_shape={"rgb": (4, 64, 64)}, episodic=True,
                num_channels=2, latent_dim=32, mlp_dim=32, num_q=3, num_bins=11,
            )).eval()
            cfg = PlannerConfig(
                action_dim=3, discount=0.995, episodic=True, num_bins=11,
                horizon=2, iterations=2, num_samples=16, num_elites=4, num_pi_trajs=4,
            )
            planner = TDMPC2Planner(model, cfg)
            observations = torch.randint(0, 256, (2, 1, 4, 64, 64), dtype=torch.uint8)
            torch.manual_seed(113)
            first = [planner.plan(observations[i], t0=i == 0, eval_mode=True).clone() for i in range(2)]
            planner.reset()
            torch.manual_seed(113)
            second = [planner.plan(observations[i], t0=i == 0, eval_mode=True).clone() for i in range(2)]
            for action, repeat in zip(first, second):
                torch.testing.assert_close(action, repeat, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(action).all())
        finally:
            torch.set_num_threads(old_threads)

    def test_external_inference_mode_mean_can_reset_at_next_episode(self):
        model = WorldModel(TDMPC2ModelConfig(
            obs_shape={"rgb": (4, 64, 64)}, episodic=True,
            num_channels=2, latent_dim=32, mlp_dim=32, num_q=3, num_bins=11,
        )).eval()
        planner = TDMPC2Planner(model, PlannerConfig(
            action_dim=3, discount=0.995, episodic=True, num_bins=11,
            horizon=2, iterations=2, num_samples=16, num_elites=4, num_pi_trajs=4,
        ))
        observation = torch.zeros((1, 4, 64, 64), dtype=torch.uint8)
        with torch.inference_mode():
            planner.plan(observation, t0=True, eval_mode=True)
        self.assertTrue(planner.prev_mean.is_inference())
        planner.reset()
        self.assertFalse(planner.prev_mean.is_inference())
        torch.testing.assert_close(planner.prev_mean, torch.zeros_like(planner.prev_mean))
        with torch.inference_mode():
            action = planner.plan(observation, t0=True, eval_mode=True)
        self.assertTrue(torch.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
