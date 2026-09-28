"""Only synthetic treatment models and environments; no real HAIC reset."""

from copy import deepcopy
import hashlib
import json
import shutil

import numpy as np
import pytest
import torch

from haic.algorithms.tdmpc2.replay import EpisodeReplay
from scripts import evaluate_tdmpc2_damage_train as operator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line(row):
    return (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path
    (root / "experiments").mkdir()
    (root / "runs").mkdir()
    monkeypatch.setattr(operator, "TARGETS", (5,))
    monkeypatch.setattr(operator.torch.version, "cuda", None)
    monkeypatch.setattr(operator.torch, "__version__", "2.1.0+cpu")
    for name in (*operator.TRAIN_SOURCE_PATHS, "scripts/evaluate_tdmpc2_damage_train.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(operator.ROOT / name, path)
    baseline = json.loads((operator.ROOT / operator.BASELINE["training_protocol"]["path"]).read_text())
    baseline["training"].update(seed_steps=5, pretrain_updates=5, batch_size=2)
    baseline["source_sha256"] = {k: sha(root / k) for k in operator.raw.SOURCE_PATHS}
    monkeypatch.setattr(operator, "_baseline", lambda _root, refs: baseline if refs == operator.BASELINE else
                        (_ for _ in ()).throw(ValueError("wrong baseline pin")))
    source = {"format": operator.TRAIN_FORMAT, "purpose": "consumed-TRAIN-development",
              "upstream_revision": baseline["upstream_revision"],
              "source_sha256": {k: sha(root / k) for k in operator.TRAIN_SOURCE_PATHS},
              "seed_schedule": baseline["seed_schedule"],
              "selection": {"arm": "independent_3d", "action_dim": 3},
              "cells": deepcopy(operator.CELLS), "episode_schedule": [0, 1, 2, 3],
              "environment": deepcopy(operator.ENVIRONMENT), "checkpoint_targets": [5],
              "training": deepcopy(baseline["training"]), "resources": deepcopy(baseline["resources"]),
              "reward_target": deepcopy(operator.REWARD_TARGET),
              "run_dir": "runs/tdmpc2-damage-synthetic",
              "baseline_training_protocol": operator.BASELINE["training_protocol"],
              "baseline_training_result": operator.BASELINE["training_result"],
              "baseline_full_eval_protocol": operator.BASELINE["full_eval_protocol"],
              "baseline_full_eval_result": operator.BASELINE["full_eval_result"]}
    source_path = root / "experiments/tdmpc2-damage-synthetic-train.json"
    source_path.write_bytes(line(source))
    run = root / source["run_dir"]
    run.mkdir()
    action = np.zeros(3, np.float32)
    native = np.array([0, .5, .5], np.float32)
    replay = EpisodeReplay(source["training"]["replay_capacity"], 3, action_dim=3,
                           seed=733, augmentation_pad=3)
    step_rows = []
    episode_rows = []
    decisions = 0
    probe = None
    for index, length in enumerate((3, 2)):
        cell = operator.CELLS[index]
        replay.start_episode(np.full((4, 64, 64), index * 10, np.uint8))
        for offset in range(length):
            damage = .2 if (index == 0 or offset == 1) else 0.
            delta = .2 if (index == 0 and offset == 0 or index == 1 and offset == 1) else 0.
            finished = index == 0 and offset == length - 1
            truncated = offset == length - 1
            training_reward = 1. - 5. * delta
            following = np.full((4, 64, 64), index * 10 + offset + 1, np.uint8)
            replay.add_step(following, action, training_reward, truncated=truncated, terminal=finished)
            step_rows.append({"decision": decisions + offset + 1, "episode": index, **cell,
                              "action_f32_hex": action.tobytes().hex(),
                              "native_action_f32_hex": native.tobytes().hex(),
                              "raw_reward_repr": "1.0", "damage_repr": repr(damage),
                              "reward": 1., "training_reward": training_reward,
                              "damage_delta": delta, "damage": damage, "progress": .1 * (offset + 1),
                              "finished": finished, "terminated": False,
                              "truncated": truncated, "terminal": finished})
        decisions += length
        episode_rows.append({"event": "episode", "episode": index, **cell, "decisions": decisions,
                             "updates": 0 if decisions < 5 else 5, "length": length,
                             "return": float(length), "training_return": float(length - 1),
                             "progress": .1 * length, "damage": .2,
                             "finished": index == 0, "terminated": False, "truncated": True,
                             "terminal": index == 0, "finish_time_s": 1. if index == 0 else None,
                             "action_trace_sha256": hashlib.sha256(action.tobytes() * length).hexdigest(),
                             "native_action_trace_sha256": hashlib.sha256(native.tobytes() * length).hexdigest()})
        if index == 0:
            sampler_state = deepcopy(replay.rng.bit_generator.state)
            probe = {k: v.clone() for k, v in replay.sample(2).items()}
            replay.rng.bit_generator.state = sampler_state
    assert probe is not None
    steps = run / "steps.jsonl"
    steps.write_bytes(b"".join(map(line, step_rows)))
    checkpoint = run / "checkpoint-at-least-000005-step-000005.pt"
    state = {"format": operator.TRAIN_FORMAT, "protocol_sha256": sha(source_path),
             "source_sha256": source["source_sha256"], "reward_target": deepcopy(operator.REWARD_TARGET),
             "target": 5, "decisions": 5, "updates": 5, "episodes": 2, "action_dim": 3,
             "learner": {"q_scale": torch.ones(1), "model.weight": torch.tensor([.2])},
             "optim": {}, "pi_optim": {}, "replay": replay.state_dict(), "probe": probe, "rng": {},
             "resume_supported": False}
    torch.save(state, checkpoint)
    runtime = {"device": "cuda", "torch": "2.1.0+cu121"}
    sample = probe["reward"]
    report = {"target": 5, "decisions": 5, "updates": 5, "episodes": 2,
              "reward_prediction_target": "training_reward",
              "frozen_train_probe": {"training_reward_mae": .1, "sample_count": 2,
                                     "episode_ids": probe["episode_id"].tolist(),
                                     "start_steps": probe["start_step"].tolist(),
                                     "distinct_episode_count": 1,
                                     "terminal_positive_transitions": int(probe["terminal"].sum().item()),
                                     "training_reward_min": sample.min().item(),
                                     "training_reward_max": sample.max().item()},
              "train_episodes_since_previous": [{k: ep[k] for k in (
                  "episode", "track_id", "geometry_seed", "decisions", "length", "return",
                  "training_return", "progress", "damage", "finished")}
                                                for ep in episode_rows],
              "path": checkpoint.name, "sha256": sha(checkpoint), "step_ledger_sha256": sha(steps)}
    ledger_rows = [{"event": "start", "protocol_sha256": sha(source_path),
                    "source_sha256": source["source_sha256"], "reward_target": operator.REWARD_TARGET,
                    "seed_schedule": source["seed_schedule"], "runtime": runtime, "resume_supported": False},
                   {"event": "reset_intent", "episode": 0, "decisions": 0, **operator.CELLS[0]},
                   {"event": "reset", "episode": 0, "decisions": 0, **operator.CELLS[0]},
                   episode_rows[0],
                   {"event": "reset_intent", "episode": 1, "decisions": 3, **operator.CELLS[1]},
                   {"event": "reset", "episode": 1, "decisions": 3, **operator.CELLS[1]},
                   episode_rows[1], {"event": "checkpoint", **report}]
    ledger = run / "training.jsonl"
    ledger.write_bytes(b"".join(map(line, ledger_rows)))
    result = {"status": "completed_boundary_at_least_100k", "protocol_sha256": sha(source_path),
              "source_sha256": source["source_sha256"], "reward_target": operator.REWARD_TARGET,
              "seed_schedule": source["seed_schedule"], "pretrain_updates": 5,
              "action_dim": 3, "decisions": 5, "updates": 5, "episodes": 2,
              "reused_train_only": True, "resume_supported": False, "evaluation": None,
              "training_ledger_sha256": sha(ledger), "step_ledger_sha256": sha(steps),
              "runtime": runtime, "checkpoints": [report]}
    result_path = run / "result.json"
    result_path.write_bytes(line(result))
    p = {"format": operator.FORMAT, "purpose": "consumed-TRAIN-development",
         "evaluation_source_sha256": sha(root / "scripts/evaluate_tdmpc2_damage_train.py"),
         "baseline": deepcopy(operator.BASELINE),
         "source": {"protocol": {"path": source_path.relative_to(root).as_posix(), "sha256": sha(source_path)},
                    "result": {"path": result_path.relative_to(root).as_posix(), "sha256": sha(result_path)},
                    "checkpoints": [{"target": 5, "path": checkpoint.relative_to(root).as_posix(),
                                     "sha256": sha(checkpoint), "training_cursor_sha256": sha(ledger),
                                     "step_cursor_sha256": sha(steps)}]},
         "cells": deepcopy(operator.CELLS), "environment": dict(operator.ENVIRONMENT),
         "modes": list(operator.MODES), "repeats": 2, "seed": 20260928,
         "max_steps": 2000, "output_dir": "runs/tdmpc2-damage-full-train-synthetic"}
    path = root / "experiments/tdmpc2-damage-synthetic-eval.json"

    def freeze():
        path.write_bytes(line(p))
        return sha(path)

    def rebind():
        # Freeze a self-consistent synthetic receipt after deliberate data mutation.
        checkpoint_row = next(row for row in ledger_rows if row["event"] == "checkpoint")
        checkpoint_row["step_ledger_sha256"] = sha(steps)
        ledger.write_bytes(b"".join(map(line, ledger_rows)))
        result["checkpoints"] = [{k: v for k, v in checkpoint_row.items() if k != "event"}]
        result["training_ledger_sha256"] = sha(ledger)
        result["step_ledger_sha256"] = sha(steps)
        result_path.write_bytes(line(result))
        p["source"]["result"]["sha256"] = sha(result_path)
        p["source"]["checkpoints"][0].update(
            sha256=checkpoint_row["sha256"], training_cursor_sha256=sha(ledger), step_cursor_sha256=sha(steps))

    return locals()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.]))

    def encode(self, obs, task):
        return self.weight.expand(obs.shape[0], 1)

    def pi(self, latent, task):
        action = torch.tensor([[.2, .1, -.1]]).expand(latent.shape[0], 3)
        return action, {"mean": action}


class FakePlanner:
    def __init__(self, model, cfg):
        assert (cfg.action_dim, cfg.horizon, cfg.num_samples, cfg.iterations,
                cfg.num_elites, cfg.num_pi_trajs, cfg.discount, cfg.episodic) == (3, 3, 512, 6, 64, 24, .995, True)
        self.resets = 0
        self.calls = []

    def reset(self):
        self.resets += 1

    def plan(self, obs, *, t0, eval_mode):
        assert eval_mode
        self.calls.append(t0)
        return torch.tensor([.3, .2, -.2])


class FakeEnv:
    def __init__(self, max_steps):
        assert max_steps == 2000
        self.unwrapped = self
        self.resets = []
        self.steps = 0
        self.closed = False

    def reset(self, *, seed, options):
        assert options == {"track_id": 1} and seed in operator.raw.ROADS
        self.track_id, self.track_seed = 1, seed
        self.resets.append(seed)
        self.steps = 0
        return np.zeros((4, 84, 84), np.float32), {}

    def step(self, action):
        assert action.shape == (3,) and action.dtype == np.float32
        self.steps += 1
        index = len(self.resets) - 1
        boundary = 2000 if index in (0, 4) else 2
        finished = (index == 0 or index in (1, 3, 5, 7)) and self.steps == boundary
        censored = index == 4 and self.steps == 2000
        terminated = self.steps == 2 and index not in (0, 4, 1, 3, 5, 7)
        return np.zeros((4, 84, 84), np.float32), 1., terminated, finished or censored, {
            "progress": 1. if finished else .25, "damage": .2,
            "finished": finished, "finish_time_s": 8. if finished else None}

    def close(self):
        self.closed = True


def test_real_baseline_has_signed_0_of_8_evidence():
    baseline = operator._baseline(operator.ROOT, operator.BASELINE)
    assert baseline["training"]["horizon"] == 3
    assert sha(operator.ROOT / "scripts/evaluate_tdmpc2_full_train.py") == operator.RAW_EVALUATOR_SHA
    with pytest.raises(ValueError, match="baseline"):
        operator._baseline(operator.ROOT, {**operator.BASELINE, "full_eval_result": {
            **operator.BASELINE["full_eval_result"], "sha256": "0" * 64}})


@pytest.mark.parametrize("name", ["checkpoint-at-least-100000-step-100354.pt", "training.jsonl", "steps.jsonl"])
def test_baseline_checks_full_source_artifact_hashes(monkeypatch, name):
    digest = operator.raw._digest

    def drift(path):
        return "0" * 64 if path.name == name and "tdmpc2-long-20260928-v2" in path.as_posix() else digest(path)

    monkeypatch.setattr(operator.raw, "_digest", drift)
    with pytest.raises(ValueError, match="baseline source artifact"):
        operator._baseline(operator.ROOT, operator.BASELINE)


def test_read_only_preflight_requires_complete_result_without_deserialization(frozen, monkeypatch):
    f = frozen
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded in preflight"))
    checked = operator.preflight(f["path"], f["freeze"](), root=f["root"])
    assert checked["planned_episodes"] == 16 and checked["environment_resets"] == 0
    assert checked["source_reward_target"] == operator.REWARD_TARGET
    assert checked["evaluation_reward"] == "raw_environment_only"
    assert not (f["root"] / f["p"]["output_dir"]).exists()
    f["result_path"].unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        operator.preflight(f["path"], sha(f["path"]), root=f["root"])


@pytest.mark.parametrize("producer", ["int_raw_reward", "int_damage"])
def test_canonical_int_producer_repr_survives_complete_source_and_replay_checks(frozen, producer):
    f = frozen
    state = torch.load(f["checkpoint"], map_location="cpu", weights_only=False)
    episode = f["ledger_rows"][6]
    if producer == "int_raw_reward":
        f["step_rows"][3].update(raw_reward_repr="-100", damage_repr="0",
                                 reward=-100., training_reward=-100.)
        state["replay"]["episodes"][1]["rewards"][0] = -100.
        episode.update({"return": -99., "training_return": -100.})
    else:
        f["step_rows"][4].update(damage_repr="1", damage=1., damage_delta=1., training_reward=-4.)
        state["replay"]["episodes"][1]["rewards"][1] = -4.
        episode.update({"damage": 1., "training_return": -3.})
    f["ledger_rows"][7]["train_episodes_since_previous"][1].update(
        {key: episode[key] for key in ("return", "training_return", "damage")})
    f["steps"].write_bytes(b"".join(map(line, f["step_rows"])))
    torch.save(state, f["checkpoint"])
    f["ledger_rows"][7]["sha256"] = sha(f["checkpoint"])
    f["rebind"]()
    checked = operator.preflight(f["path"], f["freeze"](), root=f["root"])
    assert operator._model({**checked["checkpoints"][0], "path": str(f["checkpoint"])}, checked,
                           f["source"], f["result"], f["root"], model_factory=FakeModel).weight.item() == pytest.approx(.2)
    assert not (f["root"] / f["p"]["output_dir"]).exists()


@pytest.mark.parametrize("field,text", [
    ("raw_reward_repr", "1e0"), ("raw_reward_repr", "+1"), ("raw_reward_repr", "01"),
    ("damage_repr", "0e0"), ("damage_repr", "+0"), ("damage_repr", "00")])
def test_noncanonical_numeric_producer_repr_rejected_before_checkpoint_load(frozen, monkeypatch, field, text):
    f = frozen
    f["step_rows"][3][field] = text
    f["steps"].write_bytes(b"".join(map(line, f["step_rows"])))
    f["rebind"]()
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded despite invalid producer repr"))
    with pytest.raises(ValueError, match="raw/shaped step"):
        operator.preflight(f["path"], f["freeze"](), root=f["root"])


@pytest.mark.parametrize("tamper", ["protocol_sha", "executable", "source", "reward_target", "raw_source",
                                     "result_missing", "incomplete", "cursor", "checkpoint", "steps_sha",
                                     "train_ledger", "post_checkpoint_reset", "duplicate_reset", "raw_shaped", "delta",
                                     "producer_repr", "self_consistent_raw_relabel", "episode_return", "capped", "road",
                                     "mode", "seed", "output_exists", "gpu_runtime"])
def test_tamper_refuses_before_model_load_or_env_construction(frozen, monkeypatch, tamper):
    f = frozen
    p = f["p"]
    wrong_sha = "0" * 64
    if tamper == "executable":
        p["evaluation_source_sha256"] = "0" * 64
    elif tamper == "source":
        (f["root"] / "scripts/train_tdmpc2_damage.py").write_bytes(b"changed source")
    elif tamper == "reward_target":
        f["source"]["reward_target"]["coefficient"] = 4.
        f["source_path"].write_bytes(line(f["source"]))
        p["source"]["protocol"]["sha256"] = sha(f["source_path"])
    elif tamper == "raw_source":
        f["source"]["source_sha256"]["haic/algorithms/tdmpc2/model.py"] = "0" * 64
        f["source_path"].write_bytes(line(f["source"]))
        p["source"]["protocol"]["sha256"] = sha(f["source_path"])
    elif tamper == "result_missing":
        f["result_path"].unlink()
    elif tamper == "incomplete":
        f["result"]["status"] = "partial"
        f["result_path"].write_bytes(line(f["result"]))
        p["source"]["result"]["sha256"] = sha(f["result_path"])
    elif tamper == "cursor":
        p["source"]["checkpoints"][0]["step_cursor_sha256"] = "0" * 64
    elif tamper == "checkpoint":
        f["checkpoint"].write_bytes(b"changed checkpoint")
    elif tamper == "steps_sha":
        f["steps"].write_bytes(f["steps"].read_bytes() + b"{}\n")
    elif tamper == "train_ledger":
        f["ledger_rows"].append({"event": "partial", "decisions": 6})
        f["rebind"]()
    elif tamper == "post_checkpoint_reset":
        f["ledger_rows"].append({"event": "reset_intent", "episode": 2,
                                 "decisions": 5, **operator.CELLS[2]})
        f["rebind"]()
    elif tamper == "duplicate_reset":
        f["ledger_rows"].insert(3, deepcopy(f["ledger_rows"][2]))
        f["rebind"]()
    elif tamper in ("raw_shaped", "delta"):
        f["step_rows"][0]["training_reward" if tamper == "raw_shaped" else "damage_delta"] = -1.
        f["steps"].write_bytes(b"".join(map(line, f["step_rows"])))
        f["rebind"]()
    elif tamper == "producer_repr":
        f["step_rows"][0]["damage_repr"] = "0.3"
        f["steps"].write_bytes(b"".join(map(line, f["step_rows"])))
        f["rebind"]()
    elif tamper == "self_consistent_raw_relabel":
        f["step_rows"][0]["reward"] = 2.
        f["step_rows"][0]["training_reward"] = 1.
        f["steps"].write_bytes(b"".join(map(line, f["step_rows"])))
        f["ledger_rows"][3]["return"] = 4.
        f["ledger_rows"][3]["training_return"] = 3.
        f["ledger_rows"][7]["train_episodes_since_previous"][0].update({"return": 4., "training_return": 3.})
        f["rebind"]()
    elif tamper == "episode_return":
        f["ledger_rows"][3]["training_return"] = 3.
        f["ledger_rows"][7]["train_episodes_since_previous"][0]["training_return"] = 3.
        f["rebind"]()
    elif tamper == "capped":
        p["max_steps"] = 500
    elif tamper == "road":
        p["cells"][0]["geometry_seed"] = 42
    elif tamper == "mode":
        p["modes"] = ["mppi", "prior"]
    elif tamper == "seed":
        p["seed"] += 1
    elif tamper == "output_exists":
        (f["root"] / p["output_dir"]).mkdir()
    elif tamper == "gpu_runtime":
        monkeypatch.setattr(operator.torch.version, "cuda", "12.1")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded before full validation"))
    with pytest.raises(ValueError):
        operator.execute(f["path"], wrong_sha if tamper == "protocol_sha" else f["freeze"](), root=f["root"],
                         env_factory=lambda n: pytest.fail("constructed before full validation"))
    assert not (f["root"] / p["output_dir"] / "episodes.jsonl").exists()


def test_checkpoint_format_and_reward_target_must_match_before_first_reset(frozen):
    f = frozen
    state = torch.load(f["checkpoint"], map_location="cpu", weights_only=False)
    state["reward_target"]["coefficient"] = 0.
    torch.save(state, f["checkpoint"])
    f["ledger_rows"][-1]["sha256"] = sha(f["checkpoint"])
    f["rebind"]()
    with pytest.raises(ValueError, match="not the bound damage-target"):
        operator.execute(f["path"], f["freeze"](), root=f["root"], model_factory=FakeModel,
                         env_factory=lambda n: pytest.fail("checkpoint constructed env"))
    assert not (f["root"] / f["p"]["output_dir"] / "result.json").exists()


@pytest.mark.parametrize("alter", ["empty_replay", "raw_replay_reward", "replay_action", "replay_timeout",
                                   "trimmed", "empty_probe", "probe_reward", "probe_action", "probe_terminal",
                                   "probe_bootstrap", "probe_observation", "probe_episode_id"])
def test_resealed_checkpoint_replay_and_probe_must_match_shaped_steps(frozen, alter):
    f = frozen
    state = torch.load(f["checkpoint"], map_location="cpu", weights_only=False)
    replay = state["replay"]
    probe = state["probe"]
    if alter == "empty_replay":
        state["replay"] = {}
    elif alter == "raw_replay_reward":
        replay["episodes"][0]["rewards"][0] = 1.  # Raw instead of the logged shaped zero.
    elif alter == "replay_action":
        replay["episodes"][0]["actions"][0, 0] = .5
    elif alter == "replay_timeout":
        replay["episodes"][1]["terminal"][-1] = True  # Plain truncation must bootstrap.
    elif alter == "trimmed":
        replay["episodes"][0]["start_step"] = 1
    elif alter == "empty_probe":
        state["probe"] = {}
    elif alter == "probe_reward":
        probe["reward"][0, 0, 0] += 1.
    elif alter == "probe_action":
        probe["action"][0, 0, 0] = .5
    elif alter == "probe_terminal":
        probe["terminal"][-1, 0, 0] = 0.
    elif alter == "probe_bootstrap":
        probe["bootstrap_mask"][-1, 0, 0] = 1.
    elif alter == "probe_observation":
        probe["obs"][0, 0, 0, 0, 0] += 1
    elif alter == "probe_episode_id":
        probe["episode_id"][0] = 1
    torch.save(state, f["checkpoint"])
    f["ledger_rows"][-1]["sha256"] = sha(f["checkpoint"])
    f["rebind"]()
    with pytest.raises(ValueError, match="replay|probe"):
        operator.execute(f["path"], f["freeze"](), root=f["root"], model_factory=FakeModel,
                         env_factory=lambda _: pytest.fail("constructed before replay/probe binding"))
    assert not (f["root"] / f["p"]["output_dir"] / "result.json").exists()


@pytest.mark.parametrize("phantom_delta", [False, True])
def test_all_zero_damage_cannot_prove_shape_bound_target_even_when_resealed(frozen, phantom_delta):
    f = frozen
    state = torch.load(f["checkpoint"], map_location="cpu", weights_only=False)
    for step in f["step_rows"]:
        step.update(damage=0., damage_repr="0.0", damage_delta=0., training_reward=step["reward"])
    if phantom_delta:
        f["step_rows"][0]["damage_delta"] = 1e-7  # Within numerical tolerance, not a real reward penalty.
    f["steps"].write_bytes(b"".join(map(line, f["step_rows"])))
    for episode in (f["ledger_rows"][3], f["ledger_rows"][6]):
        episode["damage"] = 0.
        episode["training_return"] = episode["return"]
    report = f["ledger_rows"][7]
    for episode in report["train_episodes_since_previous"]:
        episode["damage"] = 0.
        episode["training_return"] = episode["return"]
    for episode in state["replay"]["episodes"]:
        episode["rewards"][:] = 1.
    state["probe"]["reward"][:] = 1.
    report["frozen_train_probe"].update(training_reward_min=1., training_reward_max=1.)
    torch.save(state, f["checkpoint"])
    report["sha256"] = sha(f["checkpoint"])
    f["rebind"]()
    with pytest.raises(ValueError, match="distinguishable shaped TRAIN steps"):
        operator.execute(f["path"], f["freeze"](), root=f["root"], model_factory=FakeModel,
                         env_factory=lambda _: pytest.fail("reset without distinguishable shaped target"))


def test_raw_outcomes_full_finish_timeout_and_exact_mode_denominators(frozen):
    f = frozen
    envs = []

    def make_env(n):
        envs.append(FakeEnv(n))
        return envs[-1]

    report = operator.execute(f["path"], f["freeze"](), root=f["root"],
                              env_factory=make_env, model_factory=FakeModel, planner_factory=FakePlanner)
    assert len(envs) == 1 and envs[0].closed
    assert envs[0].resets == [seed for _ in range(2) for seed in operator.raw.ROADS for _ in operator.MODES]
    assert report["evaluation_reward"] == "raw_environment_only"
    assert report["training_reward_is_evaluation_metric"] is False
    assert report["source_reward_target"] == operator.REWARD_TARGET
    assert report["checkpoint_replay_probe_verified"] and not report["optimizer_history_proven"]
    assert report["denominators"]["completed_episodes"] == 16
    assert report["denominators"]["distinct_roads"] == 4
    ledger = f["root"] / f["p"]["output_dir"] / "episodes.jsonl"
    rows = [json.loads(row) for row in ledger.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["reset_intent", "episode"] * 16
    episodes = rows[1::2]
    assert all("training_return" not in row and "training_reward" not in row for row in episodes)
    assert episodes[0]["decisions"] == 2000 and episodes[0]["raw_return"] == 2000
    assert episodes[0]["finished"] and episodes[0]["terminal"] and not episodes[0]["censored"]
    assert episodes[0]["truncated"]  # HAIC finish at decision 2000 is not censoring.
    assert episodes[4]["decisions"] == 2000 and episodes[4]["censored"] and not episodes[4]["terminal"]
    by_mode = report["per_checkpoint"]["5"]["per_mode"]
    assert by_mode["mppi"]["finishes"] == 4 and by_mode["mppi"]["episodes"] == 8
    assert by_mode["prior"]["finishes"] == 1 and by_mode["prior"]["episodes"] == 8
    assert by_mode["prior"]["censored"] == 1 and by_mode["prior"]["uncensored"] == 7
    assert by_mode["mppi"]["roads"][0]["episodes"] == 2
    assert report["primary_descriptive_gate"]["met_on_reused_train"] is True
    assert report["primary_descriptive_gate"]["threshold_finishes"] == 4
    assert not report["generalization_claim"] and not report["official_score"]
    with pytest.raises(ValueError, match="unique"):
        operator.execute(f["path"], sha(f["path"]), root=f["root"],
                         env_factory=lambda n: pytest.fail("resumed output"))


def test_denominator_rejects_duplicate_or_missing_road_mode_repeat(frozen):
    f = frozen
    pin = operator.preflight(f["path"], f["freeze"](), root=f["root"])["checkpoints"][0]
    rows = [{"target": 5, "event": "episode", "mode": mode, "repeat": repeat, **cell}
            for repeat in range(2) for cell in operator.CELLS for mode in operator.MODES]
    with pytest.raises(ValueError, match="denominator"):
        operator._summary(rows[:-1], pin)
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="denominator"):
        operator._summary(rows, pin)


def test_source_drift_after_one_synthetic_episode_stops_before_next_reset(frozen):
    f = frozen
    envs = []

    class DriftEnv(FakeEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            if self.steps == 2 and len(self.resets) == 1:
                (f["root"] / "haic/algorithms/tdmpc2/model.py").write_bytes(b"drift")
            return obs, reward, self.steps == 2, False, {**info, "finished": False}

    def make_env(n):
        envs.append(DriftEnv(n))
        return envs[-1]

    with pytest.raises(ValueError, match="source hash mismatch"):
        operator.execute(f["path"], f["freeze"](), root=f["root"], env_factory=make_env,
                         model_factory=FakeModel, planner_factory=FakePlanner)
    assert len(envs) == 1 and envs[0].resets == [operator.raw.ROADS[0]] and envs[0].closed
    events = [json.loads(row)["event"] for row in (
        f["root"] / f["p"]["output_dir"] / "episodes.jsonl").read_text().splitlines()]
    assert events == ["reset_intent", "episode", "partial"]


@pytest.mark.parametrize("after_reset", [False, True])
def test_partial_journal_failure_writes_durable_conservative_receipt(frozen, monkeypatch, after_reset):
    f = frozen
    journal = operator.raw._journal

    def broken_partial(path, row):
        if row["event"] == "partial":
            raise OSError("partial append unavailable")
        journal(path, row)

    monkeypatch.setattr(operator.raw, "_journal", broken_partial)

    class FailingEnv(FakeEnv):
        def step(self, action):
            raise RuntimeError("synthetic applied-step failure")

    if after_reset:
        envs = []

        def make_env(n):
            envs.append(FailingEnv(n))
            return envs[-1]

        with pytest.raises(RuntimeError, match="synthetic applied-step failure"):
            operator.execute(f["path"], f["freeze"](), root=f["root"],
                             model_factory=FakeModel, planner_factory=FakePlanner, env_factory=make_env)
        assert envs[0].closed and envs[0].resets == [operator.raw.ROADS[0]]
    else:
        state = torch.load(f["checkpoint"], map_location="cpu", weights_only=False)
        state["replay"] = {}
        torch.save(state, f["checkpoint"])
        f["ledger_rows"][-1]["sha256"] = sha(f["checkpoint"])
        f["rebind"]()
        with pytest.raises(ValueError, match="replay"):
            operator.execute(f["path"], f["freeze"](), root=f["root"], model_factory=FakeModel,
                             env_factory=lambda _: pytest.fail("reset before replay validation"))
    output = f["root"] / f["p"]["output_dir"]
    receipt = json.loads((output / "failure.json").read_text())
    assert receipt["event"] == "failure" and receipt["complete_episodes"] == 0
    assert receipt["reset_intents"] == int(after_reset)
    assert receipt["environment_resets"] is (None if after_reset else 0)
    assert receipt["resume_supported"] is False
    assert receipt["protocol_sha256"] == sha(f["path"])
    assert not (output / "result.json").exists()
