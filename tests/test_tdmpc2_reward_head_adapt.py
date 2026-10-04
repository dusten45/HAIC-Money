"""Synthetic-only tests for observed-frame, head-only TD-MPC2 calibration."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from haic.algorithms.tdmpc2.model import (
    TDMPC2ModelConfig, WorldModel, soft_ce, two_hot_inv,
)
from haic.algorithms.tdmpc2.reward_head_adapt import AUGMENTATION_SEED, RewardHeadAdapter


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model():
    return WorldModel(TDMPC2ModelConfig(
        obs_shape={"rgb": (4, 64, 64)}, num_channels=2, latent_dim=32,
        mlp_dim=32, num_q=2, num_bins=101, episodic=True,
    ))


def batch():
    return {
        "obs": torch.randint(0, 256, (256, 4, 64, 64), dtype=torch.uint8),
        "action": torch.rand(256, 3, dtype=torch.float32) * 2 - 1,
        "reward": torch.linspace(-2, 4, 256, dtype=torch.float32)[:, None],
    }


def test_exactly_head_changes_all_nonhead_buffers_target_q_and_rng_unchanged():
    torch.manual_seed(48)
    source = model()
    data = batch()
    original = {key: value.clone() for key, value in source.state_dict().items()}
    source.train()
    for parameter in source.parameters():
        parameter.grad = torch.ones_like(parameter)
    adapter = RewardHeadAdapter(source)
    assert not adapter.verify_only_reward_changed()
    assert not source.training and all(not module.training for module in source.modules())
    assert all(p.grad is None for p in source.parameters())
    assert all(not p.requires_grad for name, p in source.named_parameters()
               if not name.startswith("_reward."))
    rng_before = torch.random.get_rng_state().clone()
    with patch.object(source, "next", side_effect=AssertionError("imagined state")), patch.object(
        source, "Q", side_effect=AssertionError("Q update")
    ), patch.object(source, "pi", side_effect=AssertionError("actor update")), patch.object(
        source, "soft_update_target_Q", side_effect=AssertionError("target Q update")
    ), patch.object(adapter.optim, "step", wraps=adapter.optim.step) as optim_step, patch(
        "haic.algorithms.tdmpc2.reward_head_adapt.clip_grad_norm_",
        wraps=clip_grad_norm_,
    ) as clip:
        result = adapter.step(data)
    assert optim_step.call_count == 1
    assert clip.call_count == 1 and clip.call_args.args[1] == 10.0
    assert adapter.updates == 1
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert result["loss"] > 0 and result["grad_norm"] > 0
    assert adapter.verify_only_reward_changed()
    assert any(not torch.equal(original[key], value) for key, value in source.state_dict().items()
               if key.startswith("_reward."))
    for key, value in source.state_dict().items():
        if not key.startswith("_reward."):
            assert torch.equal(original[key], value), key
    assert all(p.grad is None for p in source.parameters())
    assert set(adapter.optim.param_groups[0]["params"]) == set(source._reward.parameters())
    assert adapter.optim.param_groups[0]["lr"] == 3e-5
    assert adapter.optim.param_groups[0]["weight_decay"] == 0


def test_repeatable_with_restored_rng_and_no_input_gradients():
    source = model()
    data = batch()
    data["action"].requires_grad_()
    data["reward"].requires_grad_()
    a = RewardHeadAdapter(deepcopy(source))
    b = RewardHeadAdapter(deepcopy(source))
    rng = torch.random.get_rng_state().clone()
    result_a = a.step(data)
    torch.random.set_rng_state(rng)
    result_b = b.step(data)
    assert result_a == result_b
    assert a.snapshot() == b.snapshot()
    assert data["action"].grad is None and data["reward"].grad is None


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_augmentation_rng_changes_each_update_without_changing_outer_rng(device):
    source = model().to(device)
    adapter = RewardHeadAdapter(source)
    data = batch()
    observed = []
    encode = source.encode

    def record(obs):
        observed.append(torch.random.get_rng_state().clone())
        if device == "cuda":
            observed.append(torch.cuda.get_rng_state().clone())
        return encode(obs)

    torch.manual_seed(AUGMENTATION_SEED + 23)
    outer_cpu = torch.random.get_rng_state().clone()
    outer_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    with patch.object(source, "encode", side_effect=record):
        adapter.step(data)
        assert torch.equal(outer_cpu, torch.random.get_rng_state())
        adapter.step(data)
        assert torch.equal(outer_cpu, torch.random.get_rng_state())
    assert not torch.equal(observed[0], observed[2 if device == "cuda" else 1])
    for expected, actual in zip(outer_cuda, torch.cuda.get_rng_state_all()):
        assert torch.equal(expected, actual)
    if device == "cuda":
        assert not torch.equal(observed[1], observed[3])


@pytest.mark.parametrize("bad", [
    "extra", "missing", "obs_dtype", "obs_shape", "action_shape", "action_dtype",
    "reward_shape", "reward_dtype", "nonfinite_action", "nonfinite_reward", "action_bounds",
])
def test_invalid_batch_rejected_without_any_update(bad):
    adapter = RewardHeadAdapter(model())
    data = batch()
    if bad == "extra":
        data["additional_reward"] = data["reward"]
    elif bad == "missing":
        del data["reward"]
    elif bad == "obs_dtype":
        data["obs"] = data["obs"].float()
    elif bad == "obs_shape":
        data["obs"] = data["obs"][:255]
    elif bad == "action_shape":
        data["action"] = data["action"][:, :2]
    elif bad == "action_dtype":
        data["action"] = data["action"].double()
    elif bad == "reward_shape":
        data["reward"] = data["reward"].flatten()
    elif bad == "reward_dtype":
        data["reward"] = data["reward"].double()
    elif bad == "nonfinite_action":
        data["action"][0, 0] = float("nan")
    elif bad == "nonfinite_reward":
        data["reward"][0, 0] = float("inf")
    else:
        data["action"][0, 0] = 1.01
    before = adapter.snapshot()
    with pytest.raises(ValueError):
        adapter.step(data)
    assert adapter.updates == 0 and adapter.snapshot() == before


def test_nonhead_mutation_fails_closed_even_if_target_critic_or_buffer():
    for key in ("_encoder.rgb.2.weight", "_target_Qs.0.0.weight", "log_std_min"):
        adapter = RewardHeadAdapter(model())
        with torch.no_grad():
            adapter.model.state_dict()[key].flatten()[0].add_(1)
        with pytest.raises(RuntimeError, match="non-head model state changed"):
            adapter.step(batch())
        assert adapter.updates == 0


def test_nonfinite_logits_and_head_gradients_fail_before_step():
    adapter = RewardHeadAdapter(model())
    with patch.object(adapter.model, "reward", return_value=torch.full((256, 101), float("nan"))):
        with pytest.raises(FloatingPointError, match="reward logits"):
            adapter.step(batch())
    assert adapter.updates == 0
    with patch("haic.algorithms.tdmpc2.reward_head_adapt.soft_ce", side_effect=lambda logits, *_: logits[:, :1] * float("nan")):
        with pytest.raises(FloatingPointError, match="reward loss"):
            adapter.step(batch())
    assert adapter.updates == 0
    with adapter.model._reward[-1].weight.register_hook(
        lambda grad: torch.full_like(grad, float("nan"))
    ):
        with pytest.raises(FloatingPointError, match="gradient"):
            adapter.step(batch())
    assert adapter.updates == 0 and not adapter.verify_only_reward_changed()


@pytest.mark.parametrize("corrupt", ["head", "optimizer"])
def test_post_optimizer_nonfinite_state_is_not_counted_as_success(corrupt):
    adapter = RewardHeadAdapter(model())
    original_step = adapter.optim.step

    def poisoned_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            if corrupt == "head":
                adapter.model._reward[-1].weight.flatten()[0] = float("nan")
            else:
                parameter = next(iter(adapter.optim.state))
                adapter.optim.state[parameter]["exp_avg"].flatten()[0] = float("nan")
        return result

    with patch.object(adapter.optim, "step", side_effect=poisoned_step):
        with pytest.raises(FloatingPointError, match="after (optimizer )?step"):
            adapter.step(batch())
    assert adapter.updates == 0


def test_categorical_reward_target_and_decoding_stay_in_raw_reward_units():
    cfg = model().cfg
    for raw in (-3.0, 0.0, 4.0):
        target = torch.tensor([[raw]])
        center = torch.sign(target) * torch.log1p(target.abs())
        bins = torch.linspace(cfg.vmin, cfg.vmax, 101)
        logits = -1000 * (bins - center).abs().unsqueeze(0)
        position = ((center.item() - cfg.vmin) / cfg.bin_size)
        low = min(int(position), 99)
        upper_weight = position - low
        log_probs = torch.nn.functional.log_softmax(logits, -1)
        expected = -(1 - upper_weight) * log_probs[..., low:low + 1] - upper_weight * log_probs[..., low + 1:low + 2]
        torch.testing.assert_close(soft_ce(logits, target, cfg), expected, rtol=1e-4, atol=1e-3)
        assert abs(two_hot_inv(logits, cfg).item() - raw) < 0.13


def test_requires_categorical_three_action_original_reward_mlp():
    for change in ("action", "bins", "head"):
        source = model()
        if change == "action":
            source.cfg.action_dim = 2
        elif change == "bins":
            source.cfg.num_bins = 11
        else:
            source._reward = torch.nn.Sequential(torch.nn.Linear(35, 101))
        with pytest.raises(ValueError):
            RewardHeadAdapter(source)
