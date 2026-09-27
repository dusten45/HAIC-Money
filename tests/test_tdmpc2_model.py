"""Focused contracts for the single-task official TD-MPC2 pixel networks."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from haic.algorithms.tdmpc2.model import (
    PixelPreprocess,
    ShiftAug,
    TDMPC2ModelConfig,
    WorldModel,
    soft_ce,
    symexp,
    symlog,
    two_hot,
    two_hot_inv,
)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old_threads)


def small_cfg(**kwargs):
    return replace(
        TDMPC2ModelConfig(num_channels=2, latent_dim=32, mlp_dim=32), **kwargs
    )


def test_official_5m_pixel_architecture_and_single_task_inputs():
    cfg = TDMPC2ModelConfig()
    model = WorldModel(cfg)
    assert model.total_params == 4_865_540
    assert cfg.obs_shape == {"rgb": (9, 64, 64)}
    assert [layer.kernel_size for layer in model._encoder["rgb"] if isinstance(layer, nn.Conv2d)] == [
        (7, 7), (5, 5), (3, 3), (3, 3)
    ]
    assert model._encoder["rgb"][2].in_channels == 9
    assert model._reward[-1].out_features == 101
    assert len(model._Qs) == 5
    assert model._termination is None
    with pytest.raises(RuntimeError, match="episodic"):
        model.termination(torch.zeros(1, cfg.latent_dim))
    with pytest.raises(ValueError, match="64x64"):
        WorldModel(small_cfg(obs_shape={"rgb": (4, 84, 84)}))
    with pytest.raises(ValueError, match="single-task"):
        WorldModel(small_cfg(multitask=True))


def test_namespace_config_and_haic_channel_adapter_still_use_64_pixel_convolutions():
    cfg = small_cfg(obs_shape={"rgb": (4, 64, 64)})
    namespace = SimpleNamespace(**vars(cfg), bin_size=cfg.bin_size)
    model = WorldModel(namespace)
    with torch.no_grad():
        z = model.encode(torch.randint(0, 256, (1, 4, 64, 64), dtype=torch.uint8))
    assert z.shape == (1, 32)
    assert model._encoder["rgb"][2].in_channels == 4
    with pytest.raises(ValueError, match="task IDs"):
        model.encode(torch.zeros(1, 4, 64, 64), task=0)


def test_preprocessing_and_shift_are_per_observation_and_still_run_in_eval():
    preprocess = PixelPreprocess()
    torch.testing.assert_close(preprocess(torch.tensor([0.0, 255.0])), torch.tensor([-0.5, 0.5]))

    cfg = small_cfg()
    model = WorldModel(cfg).eval()
    shift = next(layer for layer in model._encoder["rgb"] if isinstance(layer, ShiftAug))
    pixels = (torch.arange(64 * 64, dtype=torch.float32) % 256).reshape(1, 1, 64, 64).repeat(2, 9, 1, 1)
    pixels[1] = pixels[1] + 17
    offsets = [torch.tensor([[[[0, 0]]], [[[6, 6]]]], dtype=torch.float32),
               torch.tensor([[[[3, 4]]], [[[2, 1]]]], dtype=torch.float32)]
    with mock.patch("haic.algorithms.tdmpc2.model.torch.randint", side_effect=offsets * 2) as draws:
        shifted_first = shift(pixels)
        shifted_second = shift(pixels)
        with torch.no_grad():
            result = model.encode(pixels.unsqueeze(0).expand(2, -1, -1, -1, -1))
    assert draws.call_count == 4
    assert result.shape == (2, 2, cfg.latent_dim)
    assert shift.pad == 3
    assert not torch.equal(shifted_first, shifted_second)
    torch.testing.assert_close(shifted_first[:, 0], shifted_first[:, 1])


def test_simnorm_rollout_reward_q_and_policy_shapes_gradients():
    torch.manual_seed(17)
    cfg = small_cfg(episodic=True)
    model = WorldModel(cfg)
    obs = torch.randint(0, 256, (2, 2, 9, 64, 64), dtype=torch.uint8)
    z = model.encode(obs)
    assert z.shape == (2, 2, cfg.latent_dim)
    torch.testing.assert_close(z.reshape(2, 2, -1, cfg.simnorm_dim).sum(-1), torch.ones(2, 2, 4))
    a = torch.randn(2, 2, cfg.action_dim, requires_grad=True)
    next_z = model.next(z, a)
    torch.testing.assert_close(next_z.reshape(2, 2, -1, cfg.simnorm_dim).sum(-1), torch.ones(2, 2, 4))
    rewards = model.reward(z, a)
    qs = model.Q(z, a, return_type="all")
    termination = model.termination(next_z)
    assert rewards.shape == (2, 2, cfg.num_bins)
    assert qs.shape == (cfg.num_q, 2, 2, cfg.num_bins)
    assert termination.shape == (2, 2, 1)
    assert (termination > 0).all() and (termination < 1).all()
    assert model.termination(next_z, unnormalized=True).shape == termination.shape
    policy_action, info = model.pi(z)
    assert policy_action.shape == info["mean"].shape == (2, 2, cfg.action_dim)
    assert info["entropy"].shape == info["scaled_entropy"].shape == (2, 2, 1)
    assert info["log_std"].min() >= cfg.log_std_min
    assert info["log_std"].max() <= cfg.log_std_max
    assert torch.all(policy_action.abs() <= 1) and torch.all(info["mean"].abs() <= 1)
    with mock.patch("haic.algorithms.tdmpc2.model.torch.randn_like", side_effect=torch.zeros_like):
        zero_noise_action, zero_info = model.pi(z)
    torch.testing.assert_close(zero_noise_action, zero_info["mean"])
    with mock.patch("haic.algorithms.tdmpc2.model.torch.randn_like", side_effect=torch.ones_like):
        sampled_action, _ = model.pi(z)
    assert not torch.equal(sampled_action, zero_noise_action)
    loss = (next_z.square().sum() + soft_ce(rewards, torch.full((2, 2, 1), -2.0), cfg).sum()
            + soft_ce(qs, torch.zeros(2, 2, 1), cfg).sum() + termination.sum() + policy_action.sum())
    loss.backward()
    assert a.grad is not None and a.grad.abs().sum() > 0
    for module in (model._encoder, model._dynamics, model._reward, model._pi, model._Qs, model._termination):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model._target_Qs.parameters())


def test_negative_symlog_two_hot_saturation_and_soft_ce():
    cfg = TDMPC2ModelConfig()
    raw = torch.tensor([[-torch.expm1(torch.tensor(1.9)).item()], [0.0], [1e20], [-1e20]])
    encoded = two_hot(raw, cfg)
    assert encoded.shape == (4, 101)
    torch.testing.assert_close(encoded.sum(-1), torch.ones(4))
    torch.testing.assert_close(encoded[0, 40:42], torch.tensor([0.5, 0.5]), atol=2e-6, rtol=0)
    assert encoded[2, -1] == encoded[3, 0] == 1.0
    assert symlog(raw[0]).item() < 0
    torch.testing.assert_close(symexp(symlog(raw[:2])), raw[:2])
    logits = torch.zeros((4, 101), requires_grad=True)
    with torch.no_grad():
        logits[0, 40:42] = 7
    assert two_hot_inv(logits, cfg)[0, 0] < 0
    loss = soft_ce(logits, raw, cfg)
    assert loss.shape == (4, 1)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_critic_pair_subsampling_detached_input_grad_and_ema_target():
    cfg = small_cfg(tau=0.2)
    model = WorldModel(cfg).eval()
    model.train()
    assert not model._target_Qs.training
    model.eval()
    assert all(torch.equal(a, b) for a, b in zip(model._Qs.parameters(), model._target_Qs.parameters()))
    assert all(not p.requires_grad for p in model._target_Qs.parameters())
    assert torch.count_nonzero(model._reward[-1].weight) == 0
    assert all(torch.count_nonzero(q[-1].weight) == 0 for q in model._Qs)
    with torch.no_grad():
        model._Qs[0][-1].bias[3] = 5
        model._Qs[1][-1].bias[-4] = 5
        model._Qs[0][-1].weight[0].fill_(0.05)
    z = torch.rand(2, cfg.latent_dim, requires_grad=True)
    a = torch.rand(2, cfg.action_dim, requires_grad=True)
    with mock.patch("haic.algorithms.tdmpc2.model.torch.randperm", return_value=torch.arange(cfg.num_q)) as draw:
        min_q = model.Q(z, a, return_type="min")
        avg_q = model.Q(z, a, return_type="avg", detach=True)
    assert draw.call_count == 2
    pair = two_hot_inv(model.Q(z, a, return_type="all")[:2], cfg)
    torch.testing.assert_close(min_q, pair.min(0).values)
    torch.testing.assert_close(avg_q, pair.mean(0))
    model.zero_grad(set_to_none=True)
    avg_q.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert a.grad is not None and a.grad.abs().sum() > 0
    # Detached critics still transmit gradients to the actor input, not Q parameters.
    assert all(p.grad is None for q in model._Qs for p in q.parameters())
    target_before = model._target_Qs[0][-1].bias.detach().clone()
    target_logits = model.Q(z, a, target=True, return_type="all")
    assert not torch.equal(target_logits[0], model.Q(z, a, return_type="all")[0])
    model.soft_update_target_Q()
    torch.testing.assert_close(model._target_Qs[0][-1].bias, target_before.lerp(model._Qs[0][-1].bias, cfg.tau))
    torch.testing.assert_close(
        model._target_Qs[0][-1].weight[0],
        torch.full_like(model._target_Qs[0][-1].weight[0], 0.05 * cfg.tau),
    )
    assert all(p.grad is None for p in model._target_Qs.parameters())


def test_reward_before_transition_and_termination_after_transition():
    model = WorldModel(small_cfg(episodic=True))
    calls = []

    class InputProbe(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x):
            calls.append((self.name, x.detach().clone()))
            if self.name == "dynamics":
                return x[..., :32] + 1
            return x[..., :1]

    model._dynamics = InputProbe("dynamics")
    model._reward = InputProbe("reward")
    model._termination = InputProbe("termination")
    z = torch.zeros(2, 32)
    a = torch.ones(2, 3)
    reward = model.reward(z, a)
    next_z = model.next(z, a)
    termination_logits = model.termination(next_z, unnormalized=True)
    assert [name for name, _ in calls] == ["reward", "dynamics", "termination"]
    torch.testing.assert_close(calls[0][1], torch.cat([z, a], dim=-1))
    torch.testing.assert_close(calls[2][1], next_z)
    torch.testing.assert_close(reward, torch.zeros(2, 1))
    torch.testing.assert_close(termination_logits, torch.ones(2, 1))
