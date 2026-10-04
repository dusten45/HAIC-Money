"""Synthetic-only contracts for the isolated Q-weighted TD-MPC2 planner."""

from dataclasses import replace
from typing import Iterator, cast
from unittest import mock

import pytest
import torch

from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
from haic.algorithms.tdmpc2.q_bootstrap import QWeightedPlanner


class SyntheticModel:
    def __init__(self, q=8.0):
        self.q = q
        self.pi_calls = 0
        self.q_calls = 0
        self.q_actions = []

    def encode(self, obs, task=None):
        return obs.float()

    def pi(self, z, task=None):
        self.pi_calls += 1
        torch.randn_like(z[:, :1])  # The bootstrap policy consumes RNG even at weight zero.
        return torch.full_like(z[:, :1], 2.0), {}

    def next(self, z, action, task=None):
        return z + action[:, :1]

    def reward(self, z, action, task=None):
        return action[:, :1] * 2

    def termination(self, z, task=None):
        return torch.where(z > 0.5, 0.6, 0.5)

    def Q(self, z, action, task=None, return_type="min"):
        assert return_type == "avg"
        self.q_calls += 1
        self.q_actions.append(action.clone())
        torch.randperm(5)  # WorldModel's sampled two-head schedule.
        return torch.full_like(z[:, :1], self.q)


def config(**kwargs) -> PlannerConfig:
    return replace(PlannerConfig(action_dim=1, discount=0.5, horizon=5, num_bins=0,
                                 num_samples=2, num_elites=1, num_pi_trajs=0, iterations=2), **kwargs)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_t0_t4_terminal_and_horizon_timeout_keep_rewards_and_gate_q(weight):
    model = SyntheticModel()
    planner = QWeightedPlanner(model, config(episodic=True), q_weight=weight)
    actions = torch.tensor([
        [[0.7], [0.1]], [[0.9], [0.1]], [[0.9], [0.1]],
        [[0.9], [0.1]], [[0.9], [0.1]],
    ])
    scores = planner._estimate_value(torch.zeros(2, 1), actions)
    # First trajectory ends on step 0; the second reaches the H=5 time limit
    # without a terminal label (z=0.5, whose probability is exactly 0.5).
    expected_timeout_reward = 0.2 * sum(0.5 ** t for t in range(5))
    torch.testing.assert_close(scores, torch.tensor([
        [1.4], [expected_timeout_reward + weight * 0.5 ** 5 * 8],
    ]), rtol=0, atol=1e-6)
    assert model.pi_calls == model.q_calls == 1
    torch.testing.assert_close(model.q_actions[0], torch.ones(2, 1))

    model = SyntheticModel()
    planner = QWeightedPlanner(model, config(episodic=True), q_weight=weight)
    actions = torch.tensor([[[0.1], [0.1]]] * 4 + [[[0.2], [0.1]]])
    scores = planner._estimate_value(torch.tensor([[0.0], [0.45]]), actions)
    # The second trajectory ends on step 0, the first only at step 4.
    expected_t4 = 0.2 * sum(0.5 ** t for t in range(4)) + 0.4 * 0.5 ** 4
    torch.testing.assert_close(scores, torch.tensor([[expected_t4], [0.2]]), rtol=0, atol=1e-6)
    assert model.pi_calls == model.q_calls == 0


@pytest.mark.parametrize("bad", [-1e-15, 1 + 1e-15, float("nan"), float("inf"),
                                      -float("inf"), True, "0", None])
def test_invalid_weights_fail_before_planning(bad):
    with pytest.raises(ValueError, match="q_weight"):
        QWeightedPlanner(SyntheticModel(), config(), q_weight=bad)


def test_weight_zero_consumes_identical_policy_and_head_draws_as_weight_one():
    actions = torch.full((5, 2, 1), 0.1)
    rng_states = []
    scores = []
    for weight in (0.0, 1.0):
        model = SyntheticModel()
        torch.manual_seed(71)
        values = QWeightedPlanner(model, config(), q_weight=weight)._estimate_value(torch.zeros(2, 1), actions)
        assert model.pi_calls == model.q_calls == 1
        rng_states.append(torch.get_rng_state().clone())
        scores.append(values)
    torch.testing.assert_close(scores[1] - scores[0], torch.full((2, 1), 0.5 ** 5 * 8))
    torch.testing.assert_close(rng_states[0], rng_states[1], rtol=0, atol=0)


def test_zero_weight_still_rejects_nonfinite_active_q_and_keeps_base_zero_discount():
    model = SyntheticModel(q=float("nan"))
    planner = QWeightedPlanner(model, config(), q_weight=0)
    actions = torch.zeros(5, 2, 1)
    assert torch.isneginf(planner._estimate_value(torch.zeros(2, 1), actions)).all()
    assert model.q_calls == 1
    zero_discount = QWeightedPlanner(model, config(discount=0), q_weight=0)
    torch.testing.assert_close(zero_discount._estimate_value(torch.zeros(2, 1), actions), torch.zeros(2, 1))
    assert model.q_calls == 1


@pytest.fixture
def world_model() -> Iterator[WorldModel]:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(17)
        cfg = TDMPC2ModelConfig(obs_shape={"rgb": (4, 64, 64)}, episodic=True,
                                num_channels=2, latent_dim=32, mlp_dim=32,
                                num_q=5, num_bins=11)
        model = WorldModel(cfg).eval()
        with torch.no_grad():
            assert model._termination is not None
            termination = cast(torch.nn.Linear, model._termination[-1])
            assert termination.bias is not None
            termination.bias.fill_(-4)
            for i, critic in enumerate(model._Qs):
                last = cast(torch.nn.Linear, cast(torch.nn.Sequential, critic)[-1])
                assert last.bias is not None
                last.bias[i + 4] = 1 + i
        yield model
    finally:
        torch.set_num_threads(old_threads)


def test_real_world_model_weight_one_matches_base_scores_and_rng(world_model):
    cfg = PlannerConfig(action_dim=3, discount=0.97, horizon=3, episodic=True,
                        num_bins=11, num_samples=8, num_elites=2, num_pi_trajs=2,
                        iterations=2)
    obs = torch.full((1, 4, 64, 64), 80, dtype=torch.uint8)
    actions = torch.linspace(-0.9, 0.9, 3 * 8 * 3).reshape(3, 8, 3)
    results = []
    for planner in (TDMPC2Planner(world_model, cfg), QWeightedPlanner(world_model, cfg, q_weight=1)):
        torch.manual_seed(100)
        with torch.no_grad(), mock.patch("haic.algorithms.tdmpc2.model.torch.randperm", wraps=torch.randperm) as draws:
            scores = planner._estimate_value(world_model.encode(obs).repeat(cfg.num_samples, 1), actions)
            results.append((scores, torch.get_rng_state().clone(), draws.call_count))
    assert results[0][2] == results[1][2] == 1
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)


def test_real_world_model_zero_weight_removes_decoded_q_without_skipping_pair_draw(world_model):
    cfg = PlannerConfig(action_dim=3, discount=0.97, horizon=2, episodic=True,
                        num_bins=11, num_samples=8, num_elites=2, num_pi_trajs=2,
                        iterations=2)
    obs = torch.full((1, 4, 64, 64), 80, dtype=torch.uint8)
    actions = torch.full((cfg.horizon, cfg.num_samples, cfg.action_dim), 0.1)
    results = []
    original_q = world_model.Q
    for weight in (0.0, 1.0):
        qs = []

        def record_q(*args, **kwargs):
            value = original_q(*args, **kwargs)
            qs.append(value.clone())
            return value

        torch.manual_seed(100)
        with torch.no_grad(), mock.patch.object(world_model, "Q", side_effect=record_q):
            values = QWeightedPlanner(world_model, cfg, q_weight=weight)._estimate_value(
                world_model.encode(obs).repeat(cfg.num_samples, 1), actions,
            )
            results.append((values, qs, torch.get_rng_state().clone()))
    assert len(results[0][1]) == len(results[1][1]) == 1
    torch.testing.assert_close(results[0][1][0], results[1][1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[1][0] - results[0][0],
                               cfg.discount ** cfg.horizon * results[1][1][0], rtol=0, atol=1e-6)
    torch.testing.assert_close(results[0][2], results[1][2], rtol=0, atol=0)


@pytest.mark.parametrize("eval_mode", [True, False])
def test_real_world_model_weight_one_matches_full_base_plan(world_model, eval_mode):
    cfg = PlannerConfig(action_dim=3, discount=0.97, horizon=3, episodic=True,
                        num_bins=11, num_samples=12, num_elites=3, num_pi_trajs=2,
                        iterations=2)
    obs = torch.full((1, 4, 64, 64), 80, dtype=torch.uint8)
    results = []
    for planner in (TDMPC2Planner(world_model, cfg), QWeightedPlanner(world_model, cfg, q_weight=1)):
        torch.manual_seed(29)
        with torch.no_grad(), mock.patch("haic.algorithms.tdmpc2.model.torch.randperm", wraps=torch.randperm) as draws:
            first = planner.plan(obs, t0=True, eval_mode=eval_mode)
            second = planner.plan(obs, t0=False, eval_mode=eval_mode)
            results.append((first, second, planner.prev_mean.clone(), torch.get_rng_state().clone(), draws.call_count))
    assert results[0][4] == results[1][4] == 2 * cfg.iterations
    for base, weighted in zip(results[0][:4], results[1][:4]):
        torch.testing.assert_close(base, weighted, rtol=0, atol=0)


def test_real_world_model_zero_weight_retains_head_draw_schedule(world_model):
    cfg = PlannerConfig(action_dim=3, discount=0.97, horizon=2, episodic=True,
                        num_bins=11, num_samples=8, num_elites=2, num_pi_trajs=2,
                        iterations=2)
    obs = torch.full((1, 4, 64, 64), 80, dtype=torch.uint8)
    results = []
    for weight in (0.0, 1.0):
        torch.manual_seed(29)
        planner = QWeightedPlanner(world_model, cfg, q_weight=weight)
        with torch.no_grad(), mock.patch("haic.algorithms.tdmpc2.model.torch.randperm", wraps=torch.randperm) as draws:
            planner.plan(obs, t0=True, eval_mode=True)
            results.append((torch.get_rng_state().clone(), draws.call_count))
    assert results[0][1] == results[1][1] == cfg.iterations
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
