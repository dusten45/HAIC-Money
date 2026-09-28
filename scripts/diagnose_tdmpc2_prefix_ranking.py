"""Read-only H=3 reward ranking on *logged* TD-MPC2 v2 TRAIN sequences.

The RLPD exact-prefix verifier is bound to original G0 traces and its five-layer
wrapper; TD-MPC2 v2 used a different wrapper and did not record a road signature
or accessible-state prefix. Its replay actions and episode ledger can be bound to
one another, but CANNOT authorize a reset, branch, or same-state counterfactual.
This operator never imports or constructs an environment. It scores the first
three executed actions of each complete episode from its recorded first image;
different episodes are different observed states, even on the same road seed.

Run from the repository root: python -m scripts.diagnose_tdmpc2_prefix_ranking
Only the frozen v2 local artifacts below are accepted. torch.load is used ONLY
after independently pinned SHA-256 checks; supply trusted local artifacts only.
No protocol, output file, or environment interaction is created.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel, two_hot_inv


ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/tdmpc2-reused-train-20260927-v2"
PROTOCOL = "experiments/tdmpc2-reused-train-pilot-v2.json"
PINS = {
    PROTOCOL: "209f8a752a6ef61a3dcec6c770fe10fd71158779b83afbf72f9b5d72410c8bfb",
    f"{RUN}/training.jsonl": "ac2b6728fcd8919b648a9622c7e838e8d4c5982cfb2e46b4c9ed0f54081a1765",
    f"{RUN}/boundary.pt": "08988417fec38b2a5940f3ad3037fc3e67b625735bf13ea90825470073bceee1",
    f"{RUN}/cpu-model.pt": "c13ce3e475e556bf5cbd4a7237341c19ae135db6ce8ad4607661455fc5c4bf89",
    f"{RUN}/cpu-model-export.json": "f33e6ab7773f449d5f868bf30fe4773fd4d5723c27d6513be25388765477d066",
}
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
HORIZON = 3
DISCOUNT = 0.995
PIXELS = (4, 64, 64)
SEED = 73301  # Fixed augmentation RNG, not a new evaluation reset seed.


@dataclass(frozen=True)
class LoggedWindow:
    episode_id: int
    track_id: int
    geometry_seed: int
    observation: np.ndarray  # Stored uint8 model pixels; NOT an official raw frame.
    actions: np.ndarray  # Exactly the first three executed symmetric float32 actions.
    rewards: tuple[float, float, float]  # Ledger's raw (pre-replay-float32) rewards.
    terminal_on_last_step: bool


class RewardModel(Protocol):
    cfg: Any

    def eval(self) -> Any: ...

    def encode(self, obs: torch.Tensor, task: None, /) -> torch.Tensor: ...

    def reward(self, z: torch.Tensor, action: torch.Tensor, task: None, /) -> torch.Tensor: ...

    def next(self, z: torch.Tensor, action: torch.Tensor, task: None, /) -> torch.Tensor: ...


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _pinned(root: Path, relative: str, sha: str) -> Path:
    path = root
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in frozen artifact: {relative}")
    if not path.is_file() or _digest(path) != sha:
        raise ValueError(f"missing or changed frozen artifact: {relative}")
    return path


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text: str) -> dict:
    result = json.loads(text, object_pairs_hook=_unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def _check_protocol(protocol: dict) -> None:
    train = protocol.get("training", {})
    if (protocol.get("format") != "haic-tdmpc2-reused-train-pilot-v1"
            or protocol.get("purpose") != "reused-TRAIN-engineering-pilot"
            or protocol.get("run_dir") != RUN
            or protocol.get("cells") != [{"track_id": 1, "geometry_seed": seed} for seed in ROADS]
            or protocol.get("episode_schedule") != [0, 1, 2, 3]
            or protocol.get("environment") != {
                "track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"
            }
            or train.get("horizon") != HORIZON or train.get("discount") != DISCOUNT
            or train.get("observation_shape") != list(PIXELS)
            or train.get("episodic") is not True or train.get("replay_capacity") != 14000
            or train.get("max_steps") != 2000):
        raise ValueError("not the frozen four-cell raw-reward TRAIN v2 protocol")


def _native_action_trace(actions: np.ndarray) -> str:
    trace = hashlib.sha256()
    for action in actions:
        native = np.array([action[0], (action[1] + 1) / 2, (action[2] + 1) / 2], dtype=np.float32)
        trace.update(native.tobytes())
    return trace.hexdigest()


def bind_logged_windows(protocol: dict, checkpoint: dict, rows: list[dict]) -> tuple[list[LoggedWindow], int]:
    """Pure seam: bind *all* replay steps to the ledger before selecting windows.

    This seam is for synthetic tests. Only score_frozen() pins the source bytes;
    a caller-supplied replay/ledger is not independent proof of an actual road.
    """
    _check_protocol(protocol)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    replay = checkpoint.get("replay")
    if (checkpoint.get("format") != protocol["format"]
            or checkpoint.get("protocol_sha256") != PINS[PROTOCOL]
            or checkpoint.get("source_sha256") != protocol.get("source_sha256")
            or not isinstance(replay, dict)
            or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or replay.get("horizon") != HORIZON or replay.get("action_dim") != 3
            or replay.get("observation_shape") != PIXELS
            or replay.get("capacity") != protocol["training"]["replay_capacity"]
            or replay.get("active") is not None):
        raise ValueError("checkpoint/replay source, format or boundary differs")
    episodes = replay.get("episodes")
    if (not isinstance(episodes, list) or not episodes
            or type(checkpoint.get("episodes")) is not int
            or len(episodes) != checkpoint["episodes"]
            or replay.get("next_episode_id") != len(episodes)
            or replay.get("size") != checkpoint.get("decisions")
            or not isinstance(rows, list) or len(rows) < 2
            or not all(isinstance(row, dict) for row in rows)
            or rows[0].get("event") != "start"
            or rows[0].get("protocol_sha256") != PINS[PROTOCOL]
            or rows[0].get("source_sha256") != protocol["source_sha256"]
            or rows[-1].get("event") != "checkpoint"
            or rows[-1].get("sha256") != PINS[f"{RUN}/boundary.pt"]):
        raise ValueError("replay cursor or ledger source/last boundary differs")
    allowed = {"start", "reset_intent", "reset", "step", "episode", "checkpoint",
               "metric", "diagnostic", "updates"}
    if any(not isinstance(row, dict) or row.get("event") not in allowed for row in rows):
        raise ValueError("partial, unknown, or malformed ledger event")
    events = [row for row in rows[1:] if row["event"] in {
        "reset_intent", "reset", "step", "episode", "checkpoint"
    }]
    cursor = decisions = skipped = 0
    windows = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError("malformed replay episode")
        actions, observations = episode.get("actions"), episode.get("observations")
        rewards = episode.get("rewards")
        flags = tuple(episode.get(key) for key in ("terminated", "truncated", "terminal"))
        if not isinstance(actions, np.ndarray) or actions.ndim != 2:
            raise ValueError("missing episode action vectors")
        length = len(actions)
        if (episode.get("episode_id") != index or episode.get("start_step") != 0
                or not 1 <= length <= protocol["training"]["max_steps"]
                or actions.shape != (length, 3) or actions.dtype != np.float32
                or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)
                or not isinstance(observations, np.ndarray)
                or observations.shape != (length + 1, *PIXELS) or observations.dtype != np.uint8
                or not isinstance(rewards, np.ndarray) or rewards.shape != (length,)
                or rewards.dtype != np.float32 or not np.isfinite(rewards).all()
                or any(not isinstance(flag, np.ndarray) or flag.shape != (length,)
                       or flag.dtype != np.bool_ for flag in flags)):
            raise ValueError("replay episode has invalid shape, dtype, value or trimmed prefix")
        actions = cast(np.ndarray, actions)
        observations = cast(np.ndarray, observations)
        rewards = cast(np.ndarray, rewards)
        term, trunc, terminal = (cast(np.ndarray, flag) for flag in flags)
        if (np.any(term[:-1] | trunc[:-1] | terminal[:-1])
                or not (term[-1] or trunc[-1]) or np.any(term & ~terminal)
                or np.any(terminal & ~(term | trunc))):
            raise ValueError("replay episode crosses a terminal/reset boundary")
        cell = protocol["cells"][protocol["episode_schedule"][index % 4]]
        for event in ("reset_intent", "reset"):
            if cursor >= len(events) or any(events[cursor].get(k) != v for k, v in {
                "event": event, "episode": index, "decisions": decisions, **cell
            }.items()):
                raise ValueError("ledger reset or consumed TRAIN cell differs from schedule")
            cursor += 1
        logged_rewards = []
        for step in range(length):
            decisions += 1
            if cursor >= len(events):
                raise ValueError("ledger omits a replay action/step")
            row = events[cursor]
            if (row.get("event") != "step" or row.get("episode") != index
                    or row.get("decisions") != decisions
                    or any(type(row.get(key)) is not bool or row[key] != bool(flag[step])
                           for key, flag in zip(("terminated", "truncated", "terminal"),
                                                (term, trunc, terminal)))
                    or type(row.get("reward")) not in (float, int)
                    or not math.isfinite(row["reward"])
                    or np.float32(row["reward"]) != rewards[step]):
                raise ValueError("ledger step/reward/terminal differs from replay")
            logged_rewards.append(float(row["reward"]))
            cursor += 1
        if cursor + 1 >= len(events):
            raise ValueError("missing episode/checkpoint receipts")
        end, boundary = events[cursor:cursor + 2]
        if (any(end.get(k) != v for k, v in {
                "event": "episode", "episode": index, "decisions": decisions,
                "length": length, "terminated": bool(term[-1]),
                "truncated": bool(trunc[-1]), "terminal": bool(terminal[-1]), **cell
            }.items())
                or end.get("action_trace_sha256") != hashlib.sha256(actions.tobytes()).hexdigest()
                or end.get("native_action_trace_sha256") != _native_action_trace(actions)
                or type(end.get("reward")) not in (float, int)
                or not math.isclose(end["reward"], sum(logged_rewards), rel_tol=0, abs_tol=1e-6)
                or any(boundary.get(k) != v for k, v in {
                    "event": "checkpoint", "episodes": index + 1, "decisions": decisions,
                    "updates": end.get("updates"), "pretrain_updates": end.get("pretrain_updates")
                }.items())):
            raise ValueError("episode action hash/return or checkpoint cursor differs")
        cursor += 2
        if length < HORIZON:
            skipped += 1
            continue
        windows.append(LoggedWindow(index, cell["track_id"], cell["geometry_seed"],
                                    observations[0].copy(), actions[:HORIZON].copy(),
                                    tuple(logged_rewards[:HORIZON]), bool(term[HORIZON - 1] or trunc[HORIZON - 1])))
    if (cursor != len(events) or decisions != checkpoint["decisions"]
            or replay["size"] != decisions
            or any(rows[-1].get(k) != checkpoint.get(k)
                   for k in ("decisions", "episodes", "updates", "pretrain_updates"))
            or not windows):
        raise ValueError("ledger/replay final boundary differs or no full H=3 windows")
    return windows, skipped


def rank_logged_windows(model: RewardModel, windows: list[LoggedWindow]) -> dict:
    """Score reward + open-loop dynamics only; no terminal mask or Q bootstrap."""
    if not windows:
        raise ValueError("no bound logged windows")
    model.eval()
    scored = []
    # fork_rng prevents the encoder's stochastic ShiftAug from changing caller RNG.
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        for window in windows:
            if (window.observation.shape != PIXELS or window.observation.dtype != np.uint8
                    or window.actions.shape != (HORIZON, 3) or window.actions.dtype != np.float32
                    or not np.isfinite(window.actions).all() or np.any(np.abs(window.actions) > 1)
                    or len(window.rewards) != HORIZON or not all(math.isfinite(x) for x in window.rewards)):
                raise ValueError("invalid logged H=3 window")
            torch.manual_seed(SEED + window.episode_id)
            z = model.encode(torch.from_numpy(window.observation.copy())[None], None)
            if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[0] != 1 or not torch.isfinite(z).all():
                raise ValueError("invalid encoded first observation")
            predicted = []
            for action in window.actions:
                tensor = torch.from_numpy(action.copy())[None]
                reward = two_hot_inv(model.reward(z, tensor, None), model.cfg)
                if reward.shape != (1, 1) or not torch.isfinite(reward).all():
                    raise ValueError("invalid decoded model reward")
                predicted.append(float(reward.item()))
                next_z = model.next(z, tensor, None)
                if not isinstance(next_z, torch.Tensor) or next_z.shape != z.shape or not torch.isfinite(next_z).all():
                    raise ValueError("invalid open-loop predicted latent")
                z = next_z
            observed_return = math.fsum(DISCOUNT**t * r for t, r in enumerate(window.rewards))
            predicted_return = math.fsum(DISCOUNT**t * r for t, r in enumerate(predicted))
            if not math.isfinite(predicted_return):
                raise ValueError("nonfinite model reward return")
            scored.append({"episode_id": window.episode_id, "track_id": window.track_id,
                           "geometry_seed": window.geometry_seed, "start_step": 0,
                           "terminal_on_last_step": window.terminal_on_last_step,
                           "predicted_reward_return": predicted_return,
                           "logged_raw_reward_return": observed_return})
    pairs = {"concordant": 0, "discordant": 0, "predicted_tie": 0, "logged_tie": 0}
    tie_tolerance = 1e-6  # Raw ledger/replay rounding must not manufacture return order.
    for index, left in enumerate(scored):
        for right in scored[index + 1:]:
            dp = left["predicted_reward_return"] - right["predicted_reward_return"]
            dr = left["logged_raw_reward_return"] - right["logged_raw_reward_return"]
            if abs(dr) <= tie_tolerance:
                pairs["logged_tie"] += 1
            elif abs(dp) <= tie_tolerance:
                pairs["predicted_tie"] += 1
            else:
                pairs["concordant" if (dp > 0) == (dr > 0) else "discordant"] += 1
    comparable = pairs["concordant"] + pairs["discordant"]
    return {"scope": "observational_logged_replay_only", "reused_train_only": True,
            "in_sample_training_replay": True,
            "same_state_counterfactual_ranking": False, "environment_resets": 0,
            "source_road_signature_available": False, "exact_prefix_parity_verified": False,
            "limitation": "Episode starts are different observed states; ledger cell labels have no independent road SHA or exact-prefix state binder. Rankings do not compare counterfactual actions at one anchor.",
            "method": "first H=3 executed actions per complete episode; open-loop model reward only; no Q/termination weighting",
            "horizon": HORIZON, "discount": DISCOUNT, "augmentation_seed": SEED,
            "tie_tolerance_raw_return": tie_tolerance,
            "logged_return_range": max(row["logged_raw_reward_return"] for row in scored)
                                   - min(row["logged_raw_reward_return"] for row in scored),
            "ranking_identifiable": comparable > 0,
            "concordance_on_comparable_pairs": pairs["concordant"] / comparable if comparable else None,
            "episodes_scored": len(scored), "pairs_across_distinct_episode_starts": pairs,
            "logged_windows": scored}


def score_frozen(root: Path = ROOT) -> dict:
    """Only public artifact entry: hash-pin and check v2 before model inference."""
    root = Path(root).resolve(strict=True)
    paths = {name: _pinned(root, name, sha) for name, sha in PINS.items()}
    protocol = _json(paths[PROTOCOL].read_text(encoding="utf-8"))
    _check_protocol(protocol)
    source = protocol.get("source_sha256", {})
    model_source = "haic/algorithms/tdmpc2/model.py"
    if not isinstance(source, dict) or model_source not in source:
        raise ValueError("model source is not frozen")
    _pinned(root, model_source, source[model_source])
    receipt = _json(paths[f"{RUN}/cpu-model-export.json"].read_text(encoding="utf-8"))
    if (receipt.get("format") != "haic-tdmpc2-cpu-export-v1"
            or receipt.get("protocol_sha256") != PINS[PROTOCOL]
            or receipt.get("source_sha256") != source
            or receipt.get("checkpoint_sha256") != PINS[f"{RUN}/boundary.pt"]
            or receipt.get("sha256") != PINS[f"{RUN}/cpu-model.pt"]
            or receipt.get("path") != "cpu-model.pt"):
        raise ValueError("CPU export receipt does not bind frozen v2 artifacts")
    rows = [_json(line) for line in paths[f"{RUN}/training.jsonl"].read_text(encoding="utf-8").splitlines()]
    checkpoint = torch.load(paths[f"{RUN}/boundary.pt"], map_location="cpu", weights_only=False)
    windows, skipped = bind_logged_windows(protocol, checkpoint, rows)
    export = torch.load(paths[f"{RUN}/cpu-model.pt"], map_location="cpu", weights_only=False)
    if (not isinstance(export, dict) or export.get("format") != "haic-tdmpc2-cpu-model-v1"
            or export.get("protocol_sha256") != PINS[PROTOCOL]
            or export.get("source_sha256") != source
            or export.get("checkpoint_sha256") != PINS[f"{RUN}/boundary.pt"]
            or any(export.get(key) != checkpoint.get(key) or receipt.get(key) != checkpoint.get(key)
                   for key in ("decisions", "episodes", "updates", "pretrain_updates"))
            or not isinstance(export.get("model_state"), dict)):
        raise ValueError("CPU model export and replay checkpoint differ")
    model = WorldModel(TDMPC2ModelConfig(obs_shape={"rgb": PIXELS}, episodic=True))
    model.load_state_dict(export["model_state"], strict=True)
    result = rank_logged_windows(model, windows)
    result.update({"episodes_shorter_than_h3": skipped, "protocol_sha256": PINS[PROTOCOL],
                   "checkpoint_sha256": PINS[f"{RUN}/boundary.pt"],
                   "ledger_sha256": PINS[f"{RUN}/training.jsonl"],
                   "cpu_model_sha256": PINS[f"{RUN}/cpu-model.pt"]})
    return result


def main() -> None:
    print(json.dumps(score_frozen(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
