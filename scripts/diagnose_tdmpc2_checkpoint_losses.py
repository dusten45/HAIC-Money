"""Read-only H3 training-loss replay on the long run's frozen TRAIN probe.

Run with ``python -m scripts.diagnose_tdmpc2_checkpoint_losses`` from the repo
root. No environment is imported, constructed or reset, and no file is written.
This is in-sample, fixed-probe cross-checkpoint fit, NOT interval training means,
road generalization, or an official score. Only trusted local checkpoints should
be supplied: torch.load deserializes pickle after protocol, source, ledger and
checkpoint SHA-256 validation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/tdmpc2-long-reused-train-v2.json"
PROTOCOL_SHA256 = "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec"
RUN = "runs/tdmpc2-long-20260928-v2"
TARGETS = (20000, 40000, 70000, 100000)
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
SOURCE_PATHS = frozenset({
    "scripts/train_tdmpc2_long.py", "haic/algorithms/tdmpc2/__init__.py",
    "haic/algorithms/tdmpc2/action_2d.py", "haic/algorithms/tdmpc2/haic_env.py",
    "haic/algorithms/tdmpc2/model.py", "haic/algorithms/tdmpc2/replay.py",
    "haic/algorithms/tdmpc2/learner.py", "haic/algorithms/tdmpc2/planner.py",
    "env_wrapper.py", "damage.py", "core/__init__.py", "core/vendor/__init__.py",
    "core/vendor/car_racing.py", "core/vendor/car_dynamics.py",
    "core/track_variables.py", "core/obstacle_contacts.py", "core/finish_line.py",
})


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid lowercase SHA-256")
    return value


def _file(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or not relative or any(p in (".", "..") for p in Path(relative).parts):
        raise ValueError(f"noncanonical artifact path: {relative}")
    path = root
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in artifact path: {relative}")
    if not path.is_file():
        raise ValueError(f"missing artifact: {relative}")
    return path


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict:
    if not data or len(data) > 1024 * 1024:
        raise ValueError("invalid JSON record size")
    value = json.loads(data, object_pairs_hook=_unique,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError("JSON record must be an object")
    return value


def _protocol(root: Path) -> dict:
    path = _file(root, PROTOCOL)
    if _digest(path) != PROTOCOL_SHA256:
        raise ValueError("fixed long-run protocol SHA mismatch")
    p = _json(path.read_bytes())
    cfg = p.get("training", {})
    if (p.get("format") != "haic-tdmpc2-long-train-v1"
            or p.get("purpose") != "consumed-TRAIN-development" or p.get("run_dir") != RUN
            or p.get("selection") != {"arm": "independent_3d", "action_dim": 3}
            or p.get("cells") != [{"track_id": 1, "geometry_seed": x} for x in ROADS]
            or p.get("episode_schedule") != [0, 1, 2, 3]
            or p.get("environment") != {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"}
            or p.get("checkpoint_targets") != list(TARGETS)
            or not isinstance(cfg, dict)
            or any(cfg.get(k) != v for k, v in {
                "seed": 733, "horizon": 3, "rho": .5, "discount": .995,
                "action_dim": 3, "observation_shape": [4, 64, 64], "episodic": True,
                "model_size": 5, "num_bins": 101, "augmentation_pad": 3,
                "batch_size": 256, "max_steps": 2000, "replay_capacity": 120000,
                "seed_steps": 10000, "pretrain_updates": 10000,
            }.items())):
        raise ValueError("not the frozen 3D, H3, raw-reward long-run protocol")
    sources = p.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != SOURCE_PATHS:
        raise ValueError("incomplete frozen source map")
    for name, expected in sources.items():
        if _digest(_file(root, name)) != _sha(expected):
            raise ValueError(f"source hash mismatch: {name}")
    return p


def _cursor(root: Path, p: dict, target: int) -> dict:
    """Bind a complete checkpoint row to an episode boundary and step prefix."""
    run = RUN
    ledger = _file(root, f"{run}/training.jsonl")
    rows_sha = hashlib.sha256()
    episodes = []
    decisions = updates = checkpoints = 0
    previous_checkpoint_episodes = previous_checkpoint_updates = 0
    phase = "intent"
    cell: dict = {}
    with ledger.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError("incomplete training-ledger cursor")
            rows_sha.update(line)
            row = _json(line)
            event = row.get("event")
            if line_number == 1:
                if (event != "start" or row.get("protocol_sha256") != PROTOCOL_SHA256
                        or row.get("source_sha256") != p["source_sha256"]
                        or row.get("resume_supported") is not False):
                    raise ValueError("training-ledger start does not bind frozen source/protocol")
                continue
            if phase == "checkpoint":
                if (event != "checkpoint" or row.get("target") != TARGETS[checkpoints]
                        or any(row.get(k) != v for k, v in {
                            "decisions": decisions, "updates": updates, "episodes": len(episodes),
                        }.items()) or row.get("rolling_update_count") != updates - previous_checkpoint_updates
                        or not isinstance(row.get("train_episodes_since_previous"), list)
                        or len(row["train_episodes_since_previous"]) != len(episodes) - previous_checkpoint_episodes):
                    raise ValueError("missing or inconsistent checkpoint boundary row")
                name = f"checkpoint-at-least-{row['target']:06d}-step-{decisions:06d}.pt"
                if row.get("path") != name:
                    raise ValueError("checkpoint filename differs from frozen boundary")
                checksum = _sha(row.get("sha256"))
                steps_sha = _sha(row.get("step_ledger_sha256"))
                if row["target"] == target:
                    checkpoint = _file(root, f"{run}/{name}")
                    if _digest(checkpoint) != checksum:
                        raise ValueError("checkpoint SHA differs from complete ledger row")
                    _step_prefix(root, episodes, decisions, steps_sha)
                    return {"row": row, "line": line_number, "ledger_prefix_sha256": rows_sha.hexdigest(),
                            "checkpoint_sha256": checksum, "checkpoint": checkpoint, "step_prefix_sha256": steps_sha}
                checkpoints += 1
                previous_checkpoint_episodes, previous_checkpoint_updates = len(episodes), updates
                phase = "intent"
                continue
            if event in ("reset_intent", "reset"):
                expected = "reset_intent" if phase == "intent" else "reset"
                cell = p["cells"][len(episodes) % 4]
                if event != expected or any(row.get(k) != v for k, v in {
                    "episode": len(episodes), "decisions": decisions, **cell,
                }.items()):
                    raise ValueError("reset cursor or consumed TRAIN cell differs")
                phase = "reset" if phase == "intent" else "episode"
            elif event == "episode" and phase == "episode":
                length = row.get("length")
                if (type(length) is not int or not 1 <= length <= p["training"]["max_steps"]
                        or type(row.get("updates")) is not int
                        or any(row.get(k) != v for k, v in {
                            "episode": len(episodes), "decisions": decisions + length, **cell,
                        }.items()) or row["updates"] != (
                            0 if decisions + length < p["training"]["seed_steps"]
                            else p["training"]["pretrain_updates"] + decisions + length - p["training"]["seed_steps"])
                        or type(row.get("terminal")) is not bool
                        or type(row.get("terminated")) is not bool
                        or type(row.get("truncated")) is not bool
                        or not (row["terminated"] or row["truncated"])):
                    raise ValueError("episode/decision/update boundary differs")
                decisions += length
                updates = row["updates"]
                episodes.append(row)
                phase = "intent"
                if checkpoints < len(TARGETS) and decisions >= TARGETS[checkpoints]:
                    phase = "checkpoint"
            else:
                raise ValueError(f"unexpected training-ledger event before target: {event}")
    raise ValueError(f"complete checkpoint ledger cursor for target {target} not available")


def _step_prefix(root: Path, episodes: list[dict], decisions: int, expected_sha: str) -> None:
    """Verify the checkpoint-time prefix, even when live training appends later."""
    step_path = _file(root, f"{RUN}/steps.jsonl")
    digest = hashlib.sha256()
    count = 0
    row = None
    with step_path.open("rb") as stream:
        for ep in episodes:
            actions, native = hashlib.sha256(), hashlib.sha256()
            rewards = 0.0
            for offset in range(ep["length"]):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete checkpoint-time step-ledger prefix")
                digest.update(line)
                row = _json(line)
                count += 1
                if (row.get("decision") != count or row.get("episode") != ep["episode"]
                        or row.get("track_id") != ep["track_id"]
                        or row.get("geometry_seed") != ep["geometry_seed"]
                        or type(row.get("reward")) not in (int, float)
                        or not math.isfinite(row["reward"])
                        or any(type(row.get(flag)) is not bool for flag in ("terminated", "truncated", "terminal"))
                        or (offset < ep["length"] - 1 and (row["terminated"] or row["truncated"] or row["terminal"]))
                        or row["terminated"] and not row["terminal"]
                        or row["terminal"] and not (row["terminated"] or row["truncated"])):
                    raise ValueError("step ledger disagrees with episode cursor")
                try:
                    action = bytes.fromhex(row["action_f32_hex"])
                    applied = bytes.fromhex(row["native_action_f32_hex"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("invalid step-ledger action") from exc
                if len(action) != 12 or len(applied) != 12:
                    raise ValueError("invalid 3D step-ledger action")
                actions.update(action)
                native.update(applied)
                rewards += row["reward"]
            if (row is None or any(row[k] != ep[k] for k in ("terminated", "truncated", "terminal"))
                    or actions.hexdigest() != ep.get("action_trace_sha256")
                    or native.hexdigest() != ep.get("native_action_trace_sha256")
                    or not math.isclose(rewards, ep.get("return", math.inf), rel_tol=0, abs_tol=1e-4)):
                raise ValueError("episode row differs from step-ledger actions or return")
    if count != decisions or digest.hexdigest() != expected_sha:
        raise ValueError("checkpoint step-ledger prefix SHA/cursor mismatch")


def _probe(state: dict, p: dict, row: dict) -> tuple[dict, str]:
    """Check the stored probe against its own immutable, completed replay rows."""
    replay = state.get("replay")
    cfg = p["training"]
    if (not isinstance(replay, dict) or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or replay.get("capacity") != cfg["replay_capacity"] or replay.get("horizon") != 3
            or replay.get("action_dim") != 3 or replay.get("observation_shape") != (4, 64, 64)
            or replay.get("active") is not None or replay.get("size") != row["decisions"]
            or replay.get("next_episode_id") != row["episodes"]
            or not isinstance(replay.get("episodes"), list)
            or len(replay["episodes"]) != row["episodes"]):
        raise ValueError("checkpoint replay boundary differs from ledger")
    probe = state.get("probe")
    shapes = {"obs": (4, 256, 4, 64, 64), "action": (3, 256, 3),
              **{key: (3, 256, 1) for key in ("reward", "terminated", "truncated", "terminal", "bootstrap_mask")},
              "episode_id": (256,), "start_step": (256,)}
    if (not isinstance(probe, dict) or set(probe) != set(shapes)
            or any(not isinstance(probe[k], torch.Tensor) or probe[k].device.type != "cpu" or probe[k].requires_grad
                   or tuple(probe[k].shape) != shape for k, shape in shapes.items())
            or probe["obs"].dtype != torch.uint8
            or any(probe[k].dtype != torch.float32 for k in shapes if k not in ("obs", "episode_id", "start_step"))
            or any(probe[k].dtype != torch.int64 for k in ("episode_id", "start_step"))):
        raise ValueError("stored frozen probe has invalid shape/dtype")
    if (not torch.isfinite(probe["reward"]).all() or not torch.isfinite(probe["action"]).all()
            or torch.any(probe["action"].abs() > 1)):
        raise ValueError("nonfinite or out-of-range frozen TRAIN probe")
    for flag in ("terminated", "truncated", "terminal"):
        if not torch.all((probe[flag] == 0) | (probe[flag] == 1)):
            raise ValueError("nonbinary frozen semantic termination label")
    if (torch.any(probe["terminated"] > probe["terminal"])
            or torch.any((probe["terminal"] == 1) & (probe["terminated"] == 0) & (probe["truncated"] == 0))
            or not torch.equal(probe["bootstrap_mask"], 1 - probe["terminal"])):
        raise ValueError("probe semantic terminal/bootstrap disagreement")
    for column, (eid, start) in enumerate(zip(probe["episode_id"].tolist(), probe["start_step"].tolist())):
        if type(eid) is not int or not 0 <= eid < len(replay["episodes"]):
            raise ValueError("probe episode ID missing from checkpoint replay")
        ep = replay["episodes"][eid]
        offset = start - ep["start_step"]
        if ep["episode_id"] != eid or not 0 <= offset <= len(ep["actions"]) - 3:
            raise ValueError("probe crosses replay episode/trim boundary")
        for name, key in (("obs", "observations"), ("action", "actions"), ("reward", "rewards"),
                          ("terminated", "terminated"), ("truncated", "truncated"), ("terminal", "terminal")):
            expected = ep[key][offset:offset + (4 if name == "obs" else 3)]
            actual = probe[name][:, column].numpy()
            if name in ("reward", "terminated", "truncated", "terminal"):
                expected = np.asarray(expected, dtype=np.float32)[:, None]
            if not np.array_equal(actual, expected):
                raise ValueError(f"frozen probe {name} differs from checkpoint replay")
    digest = hashlib.sha256()
    for name in sorted(probe):
        tensor = probe[name].contiguous()
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return probe, digest.hexdigest()


def _losses(learner, probe: dict, seed: int) -> dict:
    """Reproduce learner.update's pre-optimizer joint loss, including stochastic targets."""
    from haic.algorithms.tdmpc2.model import soft_ce

    model, cfg = learner.model, learner.cfg
    if cfg.horizon != 3 or model.cfg.num_q != 5 or not model.cfg.episodic:
        raise ValueError("not the frozen H3 five-Q episodic learner")
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        model.eval()
        obs, action, reward, terminal = (probe[k] for k in ("obs", "action", "reward", "terminal"))
        # Same RNG order as update: online encoder/actor + randomly paired target Q.
        next_z = model.encode(obs[1:], None)
        targets = learner._td_target(next_z, reward, terminal, probe["truncated"])
        model.train()
        z = model.encode(obs[0], None)
        zs = [z]
        consistency = torch.zeros(())
        for t in range(3):
            z = model.next(z, action[t], None)
            consistency = consistency + cfg.rho ** t * F.mse_loss(z, next_z[t])
            zs.append(z)
        zs = torch.stack(zs)
        qs = model.Q(zs[:-1], action, None, return_type="all")
        rewards = model.reward(zs[:-1], action, None)
        if qs.shape != (5, 3, action.shape[1], model.cfg.num_bins) or rewards.shape != (3, action.shape[1], model.cfg.num_bins):
            raise ValueError("model Q/reward logits shape differs from trainer")
        reward_loss = torch.zeros(())
        value_loss = torch.zeros(())
        for t in range(3):
            reward_loss = reward_loss + cfg.rho ** t * soft_ce(rewards[t], reward[t], model.cfg).mean()
            for q in qs[:, t]:
                value_loss = value_loss + cfg.rho ** t * soft_ce(q, targets[t], model.cfg).mean()
        consistency = consistency / 3
        reward_loss = reward_loss / 3
        value_loss = value_loss / 15
        termination_loss = F.binary_cross_entropy_with_logits(
            model.termination(zs[1:], None, unnormalized=True), terminal)
        total = (cfg.consistency_coef * consistency + cfg.reward_coef * reward_loss
                 + cfg.value_coef * value_loss + cfg.termination_coef * termination_loss)
        # The trainer measures this only AFTER an optimizer step; this is explicitly
        # pre-update, frozen-weights sampled policy entropy, not its logged mean.
        _, pi_info = model.pi(zs, None)
        rewards_h3 = reward.squeeze(-1)
        h3 = torch.zeros_like(rewards_h3[0])
        for t in range(3):
            h3 = h3 + learner.discount ** t * rewards_h3[t]
        result = {"sample_count": int(action.shape[1]), "horizon": 3,
                  "reward_target_min": float(reward.min()), "reward_target_max": float(reward.max()),
                  "h3_discounted_reward_target_min": float(h3.min()),
                  "h3_discounted_reward_target_max": float(h3.max()),
                  "semantic_terminal_positive_transitions": int(terminal.sum()),
                  "semantic_terminal_transitions": int(terminal.numel()),
                  "consistency_loss": float(consistency), "reward_loss": float(reward_loss),
                  "value_loss": float(value_loss), "termination_loss": float(termination_loss),
                  "weighted_total_loss": float(total), "q_scale": float(learner.q_scale.item()),
                  "sampled_policy_entropy_pre_update": float(pi_info["entropy"].mean())}
        if not all(math.isfinite(v) for v in result.values() if isinstance(v, float)):
            raise FloatingPointError("nonfinite frozen-probe loss")
        return result


def score(*, root: Path = ROOT, targets: tuple[int, ...] = TARGETS, seed: int = 834) -> dict:
    if not targets or len(set(targets)) != len(targets) or any(x not in TARGETS for x in targets):
        raise ValueError("choose distinct frozen checkpoint targets")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 augmentation/policy seed")
    started = time.perf_counter()
    root = root.resolve(strict=True)
    p = _protocol(root)
    # Complete each ledger/step cursor AND checkpoint SHA before the first torch.load.
    pins = [_cursor(root, p, target) for target in targets]
    from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel

    reports = []
    probe_sha = None
    with torch.random.fork_rng(devices=[]):
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
        learner = TDMPC2Learner(model, TDMPC2LearnerConfig(episode_length=2000, horizon=3, batch_size=256))
        if not math.isclose(learner.discount, p["training"]["discount"], abs_tol=1e-12):
            raise ValueError("learner discount differs from frozen protocol")
        for pin in pins:
            begun = time.perf_counter()
            if _digest(pin["checkpoint"]) != pin["checkpoint_sha256"]:
                raise ValueError("checkpoint changed after pre-deserialization verification")
            state = torch.load(pin["checkpoint"], map_location="cpu", weights_only=False)
            row = pin["row"]
            if (not isinstance(state, dict) or set(state) != {
                    "format", "protocol_sha256", "source_sha256", "target", "decisions", "updates", "episodes",
                    "action_dim", "learner", "optim", "pi_optim", "replay", "probe", "rng", "resume_supported"}
                    or state["format"] != p["format"] or state["protocol_sha256"] != PROTOCOL_SHA256
                    or state["source_sha256"] != p["source_sha256"] or state["resume_supported"] is not False
                    or state["action_dim"] != 3
                    or any(state[k] != row[k] for k in ("target", "decisions", "updates", "episodes"))):
                raise ValueError("deserialized checkpoint differs from bound source/ledger cursor")
            probe, current_sha = _probe(state, p, row)
            if probe_sha is not None and current_sha != probe_sha:
                raise ValueError("checkpoints do not carry the SAME frozen TRAIN probe")
            probe_sha = current_sha
            learner.load_state_dict(state["learner"], strict=True)
            diagnostic = row.get("frozen_train_probe", {})
            if (not isinstance(diagnostic, dict)
                    or diagnostic.get("episode_ids") != probe["episode_id"].tolist()
                    or diagnostic.get("start_steps") != probe["start_step"].tolist()
                    or diagnostic.get("sample_count") != 256
                    or diagnostic.get("terminal_positive_transitions") != int(probe["terminal"].sum())
                    or not isinstance(row.get("q_scale"), (float, int))
                    or not math.isclose(row["q_scale"], learner.q_scale.item(), rel_tol=0, abs_tol=1e-7)):
                raise ValueError("checkpoint probe/Q scale differs from ledger diagnostics")
            metrics = _losses(learner, probe, seed)
            reports.append({"target": row["target"], "decisions": row["decisions"],
                            "updates": row["updates"], "episodes": row["episodes"],
                            "checkpoint_path": pin["checkpoint"].relative_to(root).as_posix(),
                            "checkpoint_sha256": pin["checkpoint_sha256"],
                            "training_ledger_line": pin["line"],
                            "training_ledger_prefix_sha256": pin["ledger_prefix_sha256"],
                            "step_ledger_prefix_sha256": pin["step_prefix_sha256"],
                            "interval_training_update_count": row["rolling_update_count"],
                            "fixed_probe_losses": metrics, "elapsed_seconds": time.perf_counter() - begun})
            del state, probe
            gc.collect()
    return {"scope": "read_only_in_sample_frozen_probe_cross_checkpoint_losses",
            "reused_train_only": True, "generalization_claim": False, "environment_resets": 0,
            "optimizer_steps": 0, "training_interval_means_are_distinct": True,
            "method": "pre-update learner.update H3 joint loss, no optimizer/EMA, same stored probe and seeded CPU shifts/target sampling",
            "augmentation_and_policy_seed": seed, "probe_sha256": probe_sha,
            "protocol_path": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256,
            "source_sha256": p["source_sha256"], "torch_version": str(torch.__version__),
            "checkpoints": reports, "elapsed_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", type=int, default=list(TARGETS), choices=TARGETS,
                        help="complete checkpoint targets to compare; default: all four")
    parser.add_argument("--seed", type=int, default=834, help="shared CPU augmentation/policy seed")
    args = parser.parse_args()
    print(json.dumps(score(targets=tuple(args.targets), seed=args.seed), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
