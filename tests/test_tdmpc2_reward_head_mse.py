"""Generated CPU tensors only, never checkpoints or HAIC environments."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel, soft_ce, two_hot_inv
from haic.algorithms.tdmpc2.reward_head_mse import RawRewardMSEAdapter
from scripts.adapt_tdmpc2_reward_head_mse import _state_parity


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model():
    m = WorldModel(TDMPC2ModelConfig(obs_shape={"rgb": (4, 64, 64)},
        num_channels=2, latent_dim=32, mlp_dim=32, num_q=2, episodic=True))
    with torch.no_grad():
        m._reward[-1].weight.normal_(0, .02)
    return m


def batch():
    return {"obs": torch.randint(0, 256, (256, 4, 64, 64), dtype=torch.uint8),
            "action": torch.rand(256, 3) * 2 - 1,
            "reward": torch.linspace(-2, 4, 256)[:, None]}


def test_raw_mse_gradient_is_standard_decoder_not_raw_categorical_ce():
    cfg = model().cfg
    logits = torch.linspace(-1, 2, 101, dtype=torch.float64)[None].repeat(3, 1).requires_grad_()
    rewards = torch.tensor([[-1.5], [0.0], [4.0]], dtype=torch.float64)
    decoded = two_hot_inv(logits, cfg)
    loss = (decoded - rewards).square().mean()
    actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    bins = torch.linspace(cfg.vmin, cfg.vmax, 101, dtype=logits.dtype)
    probs = logits.softmax(-1)
    mean = (probs * bins).sum(-1, keepdim=True)
    expected = 2 / 3 * (decoded - rewards) * mean.abs().exp() * probs * (bins - mean)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    ce = soft_ce(logits, rewards, cfg).mean()
    ce_gradient = torch.autograd.grad(ce, logits)[0]
    assert not torch.allclose(actual, ce_gradient)
    # Preserve the standard decoder, including its exact-zero derivative caveat.
    zero_cfg = TDMPC2ModelConfig(num_bins=1)
    zero = torch.zeros(1, 1, requires_grad=True)
    assert torch.autograd.grad((two_hot_inv(zero, zero_cfg) - 1).square().mean(), zero)[0].item() == 0


def test_step_exact_manual_mse_adam_parity_freeze_and_rng():
    torch.manual_seed(12)
    source, data = model(), batch()
    adapter = RawRewardMSEAdapter(source)
    twin = RawRewardMSEAdapter(deepcopy(source))
    before, outer = adapter.snapshot(), torch.get_rng_state().clone()
    for index in range(2):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(20260929 + index)
            with torch.no_grad():
                latent = twin.model.encode(data["obs"])
        twin.optim.zero_grad(set_to_none=True)
        expected_loss = (two_hot_inv(twin.model.reward(latent, data["action"]), twin.model.cfg) - data["reward"]).square().mean()
        expected_loss.backward()
        expected_norm = clip_grad_norm_(twin._head_params, 10, error_if_nonfinite=True)
        twin.optim.step()
        twin.optim.zero_grad(set_to_none=True)
        twin.updates += 1
        with patch.object(source, "next", side_effect=AssertionError("imagined latent")), patch.object(
            source, "Q", side_effect=AssertionError("critic")), patch.object(
            source, "pi", side_effect=AssertionError("policy")), patch.object(
            source, "soft_update_target_Q", side_effect=AssertionError("target")):
            report = adapter.step(data)
        assert report == {"loss": expected_loss.item(), "grad_norm": expected_norm.item()}
        assert adapter.snapshot() == twin.snapshot()
        assert torch.equal(outer, torch.get_rng_state())
        _state_parity(adapter, index + 1)
    assert any(adapter.snapshot()[k] != v for k, v in before.items() if k.startswith("_reward."))
    assert all(adapter.snapshot()[k] == v for k, v in before.items() if not k.startswith("_reward."))
    assert all(p.grad is None for p in source.parameters())
    assert all(not module.training for module in source.modules())
    a, b = adapter.optim.state_dict(), twin.optim.state_dict()
    assert a["param_groups"] == b["param_groups"]
    for key in a["state"]:
        for field in a["state"][key]:
            assert torch.equal(a["state"][key][field], b["state"][key][field])


@pytest.mark.parametrize("corrupt", ["step", "lr", "nonhead", "state_nan", "shape"])
def test_tensor_adam_parity_fails_closed(corrupt):
    adapter = RawRewardMSEAdapter(model())
    adapter.step(batch())
    param = next(iter(adapter.optim.state))
    if corrupt == "step":
        adapter.optim.state[param]["step"].fill_(0)
    elif corrupt == "lr":
        adapter.optim.param_groups[0]["lr"] = 1e-3
    elif corrupt == "nonhead":
        with torch.no_grad():
            adapter.model.log_std_min.add_(1)
    elif corrupt == "state_nan":
        adapter.optim.state[param]["exp_avg"].flatten()[0] = float("nan")
    else:
        adapter.optim.state[param]["exp_avg"] = torch.ones(1)
    with pytest.raises((ValueError, RuntimeError)):
        _state_parity(adapter, 1)


@pytest.mark.parametrize("corrupt", ["reward_nan", "bounds", "shape", "dtype", "extra", "logits", "loss", "grad"])
def test_invalid_or_nonfinite_update_not_counted(corrupt):
    adapter, data = RawRewardMSEAdapter(model()), batch()
    if corrupt == "reward_nan":
        data["reward"][0] = float("nan")
    elif corrupt == "bounds":
        data["action"][0] = 1.01
    elif corrupt == "shape":
        data["reward"] = data["reward"].flatten()
    elif corrupt == "dtype":
        data["obs"] = data["obs"].float()
    elif corrupt == "extra":
        data["terminal"] = torch.zeros(256)
    with patch.object(adapter.model, "reward", return_value=torch.full((256, 101), float("nan"))) if corrupt == "logits" else patch(
        "haic.algorithms.tdmpc2.reward_head_mse.two_hot_inv", side_effect=lambda logits, cfg: logits[:, :1] * float("nan")
    ) if corrupt == "loss" else adapter.model._reward[-1].weight.register_hook(
        lambda g: torch.full_like(g, float("nan")) if corrupt == "grad" else g):
        with pytest.raises((ValueError, FloatingPointError)):
            adapter.step(data)
    assert adapter.updates == 0
