"""Synthetic frozen-probe loss and provenance checks; no HAIC environment."""

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import diagnose_tdmpc2_checkpoint_losses as diagnosis


def _line(row):
    return (json.dumps(row, sort_keys=True) + "\n").encode()


def _ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnosis, "RUN", "runs/synthetic")
    monkeypatch.setattr(diagnosis, "TARGETS", (6,))
    run = tmp_path / "runs/synthetic"
    run.mkdir(parents=True)
    p = {"training": {"max_steps": 3, "seed_steps": 3, "pretrain_updates": 3},
         "source_sha256": {"source.py": "a" * 64},
         "cells": [{"track_id": 1, "geometry_seed": seed} for seed in diagnosis.ROADS]}
    start = {"event": "start", "protocol_sha256": diagnosis.PROTOCOL_SHA256,
             "source_sha256": p["source_sha256"], "resume_supported": False}
    rows = [start]
    steps = []
    action = np.zeros(3, dtype=np.float32).tobytes().hex()
    for eid in range(2):
        cell = p["cells"][eid]
        before = eid * 3
        rows.extend([{"event": event, "episode": eid, "decisions": before, **cell}
                     for event in ("reset_intent", "reset")])
        for offset in range(3):
            steps.append({"episode": eid, "decision": before + offset + 1, **cell,
                          "action_f32_hex": action, "native_action_f32_hex": action,
                          "reward": .25, "terminated": False, "truncated": offset == 2,
                          "terminal": eid == 0 and offset == 2})
        rows.append({"event": "episode", "episode": eid, **cell, "decisions": before + 3,
                     "updates": before + 3, "length": 3, "return": .75,
                     "terminated": False, "truncated": True, "terminal": eid == 0,
                     "action_trace_sha256": hashlib.sha256(bytes.fromhex(action) * 3).hexdigest(),
                     "native_action_trace_sha256": hashlib.sha256(bytes.fromhex(action) * 3).hexdigest()})
    step_bytes = b"".join(map(_line, steps))
    (run / "steps.jsonl").write_bytes(step_bytes)
    name = "checkpoint-at-least-000006-step-000006.pt"
    (run / name).write_bytes(b"synthetic checkpoint, never deserialized")
    rows.append({"event": "checkpoint", "target": 6, "decisions": 6, "updates": 6,
                 "episodes": 2, "path": name, "sha256": diagnosis._digest(run / name),
                 "step_ledger_sha256": hashlib.sha256(step_bytes).hexdigest(),
                 "rolling_update_count": 6, "train_episodes_since_previous": [{}, {}]})
    (run / "training.jsonl").write_bytes(b"".join(map(_line, rows)))
    return p, run, rows, steps


def test_frozen_protocol_and_full_source_map_are_checked():
    p = diagnosis._protocol(diagnosis.ROOT)
    assert p["source_sha256"]["haic/algorithms/tdmpc2/learner.py"] == (
        "e0f1c814a5162b6cf34383a675ec10e768f3a046ac82c6500400cdfed23d2c9a")
    assert p["source_sha256"]["scripts/train_tdmpc2_long.py"] == (
        "9121fee37b5beeeaf7f22f253ab0500896e514fa7e5a140a8257483116df9849")


def test_complete_cursor_checkpoint_hash_and_step_prefix(tmp_path, monkeypatch):
    p, run, rows, steps = _ledger(tmp_path, monkeypatch)
    pin = diagnosis._cursor(tmp_path, p, 6)
    assert (pin["line"], pin["row"]["episodes"], pin["checkpoint_sha256"]) == (
        len(rows), 2, diagnosis._digest(run / rows[-1]["path"]))
    # Later appended (possibly in-flight) TRAIN data does not change a sealed prefix.
    with (run / "training.jsonl").open("ab") as stream:
        stream.write(b'{"event":"reset_intent"')
    with (run / "steps.jsonl").open("ab") as stream:
        stream.write(b'{"decision":7')
    assert diagnosis._cursor(tmp_path, p, 6)["checkpoint_sha256"] == pin["checkpoint_sha256"]
    (run / rows[-1]["path"]).write_bytes(b"bad SHA")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        diagnosis._cursor(tmp_path, p, 6)
    (run / rows[-1]["path"]).write_bytes(b"synthetic checkpoint, never deserialized")
    steps[0]["reward"] = .5
    (run / "steps.jsonl").write_bytes(b"".join(map(_line, steps)))
    with pytest.raises(ValueError, match="episode row differs|prefix SHA"):
        diagnosis._cursor(tmp_path, p, 6)


@pytest.mark.parametrize("broken", ["incomplete", "wrong_source", "wrong_cell", "missing_boundary", "wrong_name"])
def test_cursor_fails_closed_before_torch_load(tmp_path, monkeypatch, broken):
    p, run, rows, _ = _ledger(tmp_path, monkeypatch)
    if broken == "incomplete":
        (run / "training.jsonl").write_bytes(b"".join(map(_line, rows[:-1])) + _line(rows[-1])[:-1])
    else:
        if broken == "wrong_source":
            rows[0]["source_sha256"] = {"source.py": "b" * 64}
        elif broken == "wrong_cell":
            rows[1]["geometry_seed"] = 99
        elif broken == "missing_boundary":
            rows.pop(-1)
        elif broken == "wrong_name":
            rows[-1]["path"] = "../unverified.pt"
        (run / "training.jsonl").write_bytes(b"".join(map(_line, rows)))
    monkeypatch.setattr(diagnosis, "_protocol", lambda root: p)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("deserialized before SHA/cursor"))
    with pytest.raises(ValueError):
        diagnosis.score(root=tmp_path, targets=(6,))


def test_symlink_and_protocol_digest_rejected_without_deserialization(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/link").symlink_to(tmp_path / "runs", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        diagnosis._file(tmp_path, "runs/link/not-a-checkpoint.pt")
    frozen = tmp_path / diagnosis.PROTOCOL
    frozen.parent.mkdir()
    frozen.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="protocol SHA"):
        diagnosis._protocol(tmp_path)


def _probe_state():
    images = np.stack([np.full((4, 64, 64), x, dtype=np.uint8) for x in range(4)])
    actions = np.zeros((3, 3), dtype=np.float32)
    rewards = np.array([.25, .5, -.5], dtype=np.float32)
    terminated = np.zeros(3, dtype=np.bool_)
    truncated = np.array([False, False, True])
    terminal = truncated.copy()
    episode = {"episode_id": 0, "start_step": 0, "observations": images,
               "actions": actions, "rewards": rewards, "terminated": terminated,
               "truncated": truncated, "terminal": terminal}
    def repeat(data, n):
        return torch.from_numpy(np.repeat(data[:, None], n, axis=1).copy())
    probe = {"obs": repeat(images, 256), "action": repeat(actions, 256),
             "reward": repeat(rewards[:, None], 256),
             "terminated": repeat(terminated.astype(np.float32)[:, None], 256),
             "truncated": repeat(truncated.astype(np.float32)[:, None], 256),
             "terminal": repeat(terminal.astype(np.float32)[:, None], 256),
             "episode_id": torch.zeros(256, dtype=torch.int64),
             "start_step": torch.zeros(256, dtype=torch.int64)}
    probe["bootstrap_mask"] = 1 - probe["terminal"]
    state = {"replay": {"format": "haic-tdmpc2-episode-replay-v1", "capacity": 120000,
                        "horizon": 3, "action_dim": 3, "observation_shape": (4, 64, 64),
                        "active": None, "size": 3, "next_episode_id": 1, "episodes": [episode]},
             "probe": probe}
    return state


def test_probe_is_bound_to_stored_replay_and_semantic_mask():
    state = _probe_state()
    p, row = {"training": {"replay_capacity": 120000}}, {"decisions": 3, "episodes": 1}
    _, first = diagnosis._probe(state, p, row)
    assert diagnosis._probe(state, p, row)[1] == first
    state["probe"]["reward"][0, 0] += 1
    with pytest.raises(ValueError, match="reward differs"):
        diagnosis._probe(state, p, row)
    state["probe"]["reward"][0, 0] -= 1
    state["probe"]["bootstrap_mask"][2, 0] = 1
    with pytest.raises(ValueError, match="semantic terminal/bootstrap"):
        diagnosis._probe(state, p, row)


class AnalyticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(num_q=5, num_bins=3, episodic=True, vmin=-1., vmax=1., bin_size=1.)
        self.modes = []

    def encode(self, obs, task):
        self.modes.append(self.training)
        return obs.float()

    def next(self, z, action, task):
        return z

    def pi(self, z, task):
        return torch.zeros((*z.shape[:-1], 3)), {"entropy": torch.full((*z.shape[:-1], 1), .7)}

    def Q(self, z, action, task, *, return_type="all", target=False):
        if target:
            return torch.zeros((*z.shape[:-1], 1))
        return torch.zeros((5, *z.shape[:-1], 3))

    def reward(self, z, action, task):
        return torch.zeros((*z.shape[:-1], 3))

    def termination(self, z, task, *, unnormalized=True):
        return torch.full((*z.shape[:-1], 1), 2.)


def test_analytic_h3_soft_ce_all_five_q_termination_and_total_without_rng_change():
    from haic.algorithms.tdmpc2.learner import TDMPC2LearnerConfig

    model = AnalyticModel()
    cfg = TDMPC2LearnerConfig(episode_length=2000, horizon=3, batch_size=2)
    learner = SimpleNamespace(model=model, cfg=cfg, discount=.995, q_scale=torch.tensor([2.25]))
    learner._td_target = lambda z, reward, terminal, truncated: reward
    obs = torch.arange(4, dtype=torch.float32)[:, None, None].expand(4, 2, 1)
    reward = torch.tensor([[[0.], [1.]], [[0.], [0.]], [[0.], [-1.]]])
    terminal = torch.tensor([[[0.], [0.]], [[0.], [0.]], [[0.], [1.]]])
    probe = {"obs": obs, "action": torch.zeros(3, 2, 3), "reward": reward,
             "terminal": terminal, "truncated": terminal}
    before = torch.get_rng_state().clone()
    result = diagnosis._losses(learner, probe, 123)
    assert torch.equal(torch.get_rng_state(), before)
    assert model.modes == [False, True]
    assert result == diagnosis._losses(learner, probe, 123)
    assert result["semantic_terminal_positive_transitions"] == 1
    assert result["semantic_terminal_transitions"] == 6
    assert result["reward_target_min"] == -1 and result["reward_target_max"] == 1
    assert result["h3_discounted_reward_target_min"] == 0
    assert result["h3_discounted_reward_target_max"] == pytest.approx(1 - .995**2, abs=1e-7)
    weighted_ce = (1 + .5 + .25) * math.log(3) / 3
    assert result["consistency_loss"] == pytest.approx((1 + .5 * 4 + .25 * 9) / 3)
    assert result["reward_loss"] == pytest.approx(weighted_ce)
    assert result["value_loss"] == pytest.approx(weighted_ce)
    semantic_bce = (5 * math.log1p(math.exp(2)) + math.log1p(math.exp(-2))) / 6
    assert result["termination_loss"] == pytest.approx(semantic_bce)
    assert result["weighted_total_loss"] == pytest.approx(
        20 * result["consistency_loss"] + .1 * weighted_ce + .1 * weighted_ce + semantic_bce)
    assert result["q_scale"] == 2.25
    assert result["sampled_policy_entropy_pre_update"] == pytest.approx(.7)


def test_real_tiny_model_repeats_seeded_shifts_without_optimizer_or_rng_mutation(monkeypatch):
    from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)},
                                             num_channels=1, latent_dim=16, mlp_dim=16,
                                             num_q=5, num_bins=11, episodic=True, dropout=.1))
        learner = TDMPC2Learner(model, TDMPC2LearnerConfig(episode_length=2000, horizon=3, batch_size=2))
        monkeypatch.setattr(learner.optim, "step", lambda: pytest.fail("optimizer stepped"))
        monkeypatch.setattr(learner.pi_optim, "step", lambda: pytest.fail("policy optimizer stepped"))
        monkeypatch.setattr(model, "soft_update_target_Q", lambda: pytest.fail("target Q mutated"))
        probe = {"obs": torch.randint(0, 256, (4, 2, 4, 64, 64), dtype=torch.uint8),
                 "action": torch.zeros(3, 2, 3), "reward": torch.ones(3, 2, 1) / 4,
                 "terminal": torch.tensor([[[0.], [0.]], [[0.], [0.]], [[1.], [0.]]]),
                 "truncated": torch.zeros(3, 2, 1)}
        before = {key: value.clone() for key, value in learner.state_dict().items()}
        rng = torch.get_rng_state().clone()
        first = diagnosis._losses(learner, probe, 834)
        second = diagnosis._losses(learner, probe, 834)
        assert first == second and math.isfinite(first["weighted_total_loss"])
        assert torch.equal(torch.get_rng_state(), rng)
        for key, value in learner.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)


def test_cli_writes_exclusive_source_scoped_loss_receipt(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    output = runs / "tdmpc2-long-v2-losses-20000.json"
    monkeypatch.setattr(diagnosis, "ROOT", tmp_path)
    monkeypatch.setattr(diagnosis, "score", lambda **kwargs: {"checkpoints": [{"target": 20000}]})
    monkeypatch.setattr("sys.argv", ["audit", "--targets", "20000", "--output", str(output)])
    diagnosis.main()
    assert json.loads(output.read_text())["checkpoints"][0]["target"] == 20000
    assert json.loads(capsys.readouterr().out)["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        diagnosis.main()
    monkeypatch.setattr("sys.argv", ["audit", "--targets", "20000", "--output", str(runs / "other.json")])
    with pytest.raises(SystemExit):
        diagnosis.main()
