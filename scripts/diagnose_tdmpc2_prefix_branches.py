"""Optional, protocol-gated v2 TRAIN replay -> same-state H=3 branch diagnosis.

No environment is constructed until the v2 checkpoint, complete ledger, CPU export,
new protocol, environment/physics sources and runtime have passed independent pins.
The default CLI is a no-reset preflight; --execute additionally requires an unused
output path. This compares reconstructed accessible states, NOT historical hidden
Box2D contacts (v2 did not record a historical road/state signature).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import torch

from haic.algorithms.rlpd import prefix_parity as parity
from haic.algorithms.tdmpc2.haic_env import environment_action, episode_boundary, model_observation
from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel, two_hot_inv


ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/tdmpc2-reused-train-20260927-v2"
V2_PROTOCOL = "experiments/tdmpc2-reused-train-pilot-v2.json"
V2_PINS = {
    V2_PROTOCOL: "209f8a752a6ef61a3dcec6c770fe10fd71158779b83afbf72f9b5d72410c8bfb",
    f"{RUN}/training.jsonl": "ac2b6728fcd8919b648a9622c7e838e8d4c5982cfb2e46b4c9ed0f54081a1765",
    f"{RUN}/boundary.pt": "08988417fec38b2a5940f3ad3037fc3e67b625735bf13ea90825470073bceee1",
    f"{RUN}/cpu-model.pt": "c13ce3e475e556bf5cbd4a7237341c19ae135db6ce8ad4607661455fc5c4bf89",
    f"{RUN}/cpu-model-export.json": "f33e6ab7773f449d5f868bf30fe4773fd4d5723c27d6513be25388765477d066",
}
V2_SOURCES = frozenset({
    "haic/algorithms/tdmpc2/__init__.py", "scripts/train_tdmpc2.py",
    "haic/algorithms/tdmpc2/haic_env.py", "haic/algorithms/tdmpc2/model.py",
    "haic/algorithms/tdmpc2/replay.py", "haic/algorithms/tdmpc2/learner.py",
    "haic/algorithms/tdmpc2/planner.py", "env_wrapper.py", "damage.py",
    "core/__init__.py", "core/vendor/__init__.py", "core/vendor/car_racing.py",
    "core/vendor/car_dynamics.py", "core/track_variables.py",
    "core/obstacle_contacts.py", "core/finish_line.py",
})
NEW_SOURCES = frozenset({
    "scripts/diagnose_tdmpc2_prefix_branches.py", "haic/algorithms/rlpd/prefix_parity.py",
    "train.py", "action_representation.py", "action_smoothing.py", "tracking.py",
})
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
ANCHORS = tuple((episode, step) for episode in range(4) for step in (16, 50, 100))
HORIZON = 3
DISCOUNT = 0.995
PIXELS = (4, 64, 64)
TIE_TOLERANCE = 1e-6
EXPECTED_COUNTS = {"decisions": 12058, "episodes": 37, "updates": 12057, "pretrain_updates": 10000}
FIXED_SUFFIXES = {
    "coast": [[0.0, -1.0, -1.0]] * HORIZON,
    "gas": [[0.0, 1.0, -1.0]] * HORIZON,
    "brake": [[0.0, -1.0, 1.0]] * HORIZON,
    "left_gas": [[-0.5, 1.0, -1.0]] * HORIZON,
}


@dataclass(frozen=True)
class BoundAnchor:
    episode_id: int
    track_id: int
    seed: int
    step: int
    observations: np.ndarray  # Every v2 model pixel from reset through anchor, inclusive.
    actions: np.ndarray  # Every executed float32 v2 model action before anchor.
    rewards: tuple[float, ...]  # Ledger raw precision (replay itself is float32).
    logged_actions: np.ndarray  # The actually executed H3 baseline suffix.
    logged_observations: np.ndarray
    logged_rewards: tuple[float, ...]
    logged_flags: tuple[tuple[bool, bool, bool], ...]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def pinned(root: Path, relative: str, sha: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))
            or not isinstance(sha, str) or len(sha) != 64
            or any(letter not in "0123456789abcdef" for letter in sha)):
        raise ValueError("invalid frozen path or SHA-256")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in frozen source: {relative}")
    if not path.is_file() or digest(path) != sha:
        raise ValueError(f"missing or changed frozen source: {relative}")
    return path


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def strict_json(text: str) -> dict:
    obj = json.loads(text, object_pairs_hook=_unique,
                     parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


def runtime_identity() -> dict[str, str]:
    """Include the native physics binary and TimeLimit implementation in the freeze."""
    import Box2D
    import Box2D._Box2D as native_box2d
    import cv2
    import gymnasium
    from gymnasium.wrappers.time_limit import TimeLimit

    if native_box2d.__file__ is None or Box2D.__file__ is None:
        raise ValueError("Box2D native/source file is unavailable")

    return {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "opencv": str(getattr(cv2, "__version__")),
        "gymnasium": gymnasium.__version__, "box2d": Box2D.__version__,
        "box2d_native_sha256": digest(Path(native_box2d.__file__)),
        "box2d_python_sha256": digest(Path(Box2D.__file__)),
        "timelimit_source_sha256": digest(Path(inspect.getfile(TimeLimit))),
    }


def check_protocol(spec: dict, v2: dict) -> None:
    """Pure structural check, with *hard-coded* v2 identity and fixed cells/branches."""
    expected = {
        "format", "purpose", "v2_sha256", "source_sha256", "runtime",
        "anchors", "candidates", "environment", "horizon", "discount",
        "tie_tolerance", "augmentation_seed", "max_resets",
    }
    if (not isinstance(spec, dict) or set(spec) != expected
            or spec["format"] != "haic-tdmpc2-v2-prefix-branches-v1"
            or spec["purpose"] != "consumed-TRAIN-same-state-diagnostic"
            or spec["v2_sha256"] != V2_PINS
            or spec["anchors"] != [{"episode_id": ep, "start_step": step} for ep, step in ANCHORS]
            or spec["candidates"] != {"logged": None, **FIXED_SUFFIXES}
            or spec["environment"] != {
                "track_id": 1, "geometry_seeds": list(ROADS), "max_steps": 2000,
                "frame_skip": 4, "obstacles": True, "reward_shaping": False,
            }
            or spec["horizon"] != HORIZON or spec["discount"] != DISCOUNT
            or spec["tie_tolerance"] != TIE_TOLERANCE
            or type(spec["augmentation_seed"]) is not int
            or not 0 <= spec["augmentation_seed"] < 2**32
            or spec["max_resets"] != len(ANCHORS) * (1 + 1 + len(FIXED_SUFFIXES))
            or not isinstance(spec["runtime"], dict)
            or set(spec["runtime"]) != set(runtime_identity())
            or any(not isinstance(value, str) for value in spec["runtime"].values())
            or not isinstance(v2, dict)
            or v2.get("format") != "haic-tdmpc2-reused-train-pilot-v1"
            or v2.get("purpose") != "reused-TRAIN-engineering-pilot"
            or v2.get("run_dir") != RUN
            or v2.get("cells") != [{"track_id": 1, "geometry_seed": seed} for seed in ROADS]
            or v2.get("episode_schedule") != [0, 1, 2, 3]
            or v2.get("environment") != {
                "track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"
            } or not isinstance(v2.get("training"), dict)
            or any(v2["training"].get(key) != value for key, value in {
                "max_steps": 2000, "horizon": HORIZON, "discount": DISCOUNT,
                "observation_shape": list(PIXELS), "replay_capacity": 14000,
                "episodic": True,
            }.items())
            or not isinstance(v2.get("source_sha256"), dict)
            or set(v2["source_sha256"]) != V2_SOURCES
            or not isinstance(spec["source_sha256"], dict)
            or set(spec["source_sha256"]) != V2_SOURCES | NEW_SOURCES
            or any(spec["source_sha256"].get(name) != sha
                   for name, sha in v2["source_sha256"].items())):
        raise ValueError("not the fixed source-bound v2 consumed-TRAIN branch contract")


def _native_trace(actions: np.ndarray) -> str:
    trace = hashlib.sha256()
    for action in actions:
        trace.update(environment_action(action).tobytes())
    return trace.hexdigest()


def bind_anchors(v2: dict, checkpoint: dict, rows: list[dict]) -> tuple[BoundAnchor, ...]:
    """Bind ALL recorded episodes/steps/hashes before selecting fixed anchors.

    Pure seam for synthetic tests. Only preflight() admits its inputs from pinned
    bytes; passing caller-built dictionaries is never authorization for a reset.
    """
    replay = checkpoint.get("replay") if isinstance(checkpoint, dict) else None
    if (not isinstance(replay, dict) or checkpoint.get("format") != v2["format"]
            or checkpoint.get("protocol_sha256") != V2_PINS[V2_PROTOCOL]
            or checkpoint.get("source_sha256") != v2["source_sha256"]
            or any(checkpoint.get(key) != expected for key, expected in EXPECTED_COUNTS.items())
            or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or replay.get("capacity") != 14000 or replay.get("horizon") != HORIZON
            or replay.get("action_dim") != 3 or replay.get("observation_shape") != PIXELS
            or replay.get("size") != checkpoint["decisions"]
            or replay.get("next_episode_id") != checkpoint["episodes"]
            or replay.get("active") is not None
            or not isinstance(replay.get("episodes"), list)
            or len(replay["episodes"]) != checkpoint["episodes"]
            or not isinstance(rows, list) or len(rows) < 2
            or any(not isinstance(row, dict) or row.get("event") not in {
                "start", "reset_intent", "reset", "step", "episode", "checkpoint",
                "metric", "diagnostic", "updates",
            } for row in rows)
            or rows[0].get("event") != "start"
            or rows[0].get("protocol_sha256") != V2_PINS[V2_PROTOCOL]
            or rows[0].get("source_sha256") != v2["source_sha256"]
            or rows[-1].get("event") != "checkpoint"
            or rows[-1].get("sha256") != V2_PINS[f"{RUN}/boundary.pt"]
            or any(rows[-1].get(key) != checkpoint.get(key)
                   for key in ("decisions", "episodes", "updates", "pretrain_updates"))):
        raise ValueError("v2 checkpoint/replay/ledger identity or final boundary differs")
    assert isinstance(replay, dict)
    events = [row for row in rows[1:] if row["event"] in {
        "reset_intent", "reset", "step", "episode", "checkpoint"
    }]
    selected = {}
    cursor = decisions = 0
    for index, episode in enumerate(replay["episodes"]):
        if not isinstance(episode, dict):
            raise ValueError("invalid v2 replay episode")
        actions, obs, rewards = (episode.get(key) for key in ("actions", "observations", "rewards"))
        flags = tuple(episode.get(key) for key in ("terminated", "truncated", "terminal"))
        if not isinstance(actions, np.ndarray) or actions.ndim != 2:
            raise ValueError("missing v2 replay actions")
        length = len(actions)
        if (type(episode.get("episode_id")) is not int or episode["episode_id"] != index
                or episode.get("start_step") != 0 or not 1 <= length <= 2000
                or actions.shape != (length, 3) or actions.dtype != np.float32
                or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)
                or not isinstance(obs, np.ndarray) or obs.dtype != np.uint8
                or obs.shape != (length + 1, *PIXELS)
                or not isinstance(rewards, np.ndarray) or rewards.dtype != np.float32
                or rewards.shape != (length,) or not np.isfinite(rewards).all()
                or any(not isinstance(flag, np.ndarray) or flag.dtype != np.bool_
                       or flag.shape != (length,) for flag in flags)):
            raise ValueError("v2 replay has invalid shape, dtype, boundary or values")
        term, trunc, terminal = (cast(np.ndarray, flag) for flag in flags)
        if (np.any(term[:-1] | trunc[:-1] | terminal[:-1])
                or not (term[-1] or trunc[-1]) or np.any(term & ~terminal)
                or np.any(terminal & ~(term | trunc))):
            raise ValueError("v2 replay crosses terminal boundary")
        cell = v2["cells"][v2["episode_schedule"][index % len(ROADS)]]
        for event in ("reset_intent", "reset"):
            if cursor >= len(events) or any(events[cursor].get(key) != value for key, value in {
                "event": event, "episode": index, "decisions": decisions, **cell,
            }.items()):
                raise ValueError("v2 episode cell/reset schedule differs")
            cursor += 1
        raw_rewards = []
        for step in range(length):
            decisions += 1
            if cursor >= len(events):
                raise ValueError("missing v2 step")
            row = events[cursor]
            if (row.get("event") != "step" or row.get("episode") != index
                    or row.get("decisions") != decisions
                    or any(type(row.get(key)) is not bool or row[key] != bool(flag[step])
                            for key, flag in zip(("terminated", "truncated", "terminal"),
                                                 (term, trunc, terminal)))
                    or type(row.get("reward")) not in (int, float)
                    or not math.isfinite(row["reward"])
                    or np.float32(row["reward"]) != rewards[step]):
                raise ValueError("v2 replay reward/flags differ from step ledger")
            raw_rewards.append(float(row["reward"]))
            cursor += 1
        if cursor + 1 >= len(events):
            raise ValueError("missing v2 episode/checkpoint")
        end, boundary = events[cursor:cursor + 2]
        if (any(end.get(key) != value for key, value in {
                "event": "episode", "episode": index, "decisions": decisions,
                "length": length, "terminated": bool(term[-1]),
                "truncated": bool(trunc[-1]), "terminal": bool(terminal[-1]), **cell,
            }.items())
                or end.get("action_trace_sha256") != hashlib.sha256(actions.tobytes()).hexdigest()
                or end.get("native_action_trace_sha256") != _native_trace(actions)
                or type(end.get("reward")) not in (float, int)
                or not math.isclose(end["reward"], math.fsum(raw_rewards), rel_tol=0, abs_tol=1e-6)
                or any(boundary.get(key) != value for key, value in {
                    "event": "checkpoint", "episodes": index + 1, "decisions": decisions,
                    "updates": end.get("updates"), "pretrain_updates": end.get("pretrain_updates"),
                }.items())):
            raise ValueError("v2 action/native hash, return or episode boundary differs")
        cursor += 2
        for ep, start in ANCHORS:
            if ep != index:
                continue
            if start + HORIZON > length or np.any(term[:start] | trunc[:start]):
                raise ValueError(f"predeclared episode {ep} step {start} is not a full nonterminal anchor")
            selected[ep, start] = BoundAnchor(
                ep, cell["track_id"], cell["geometry_seed"], start,
                obs[:start + 1].copy(), actions[:start].copy(), tuple(raw_rewards[:start]),
                actions[start:start + HORIZON].copy(), obs[start:start + HORIZON + 1].copy(),
                tuple(raw_rewards[start:start + HORIZON]),
                tuple((bool(term[t]), bool(trunc[t]), bool(terminal[t]))
                      for t in range(start, start + HORIZON)),
            )
    if (cursor != len(events) or decisions != checkpoint["decisions"]
            or set(selected) != set(ANCHORS)):
        raise ValueError("v2 ledger does not cover all replay steps and fixed anchors")
    return tuple(selected[anchor] for anchor in ANCHORS)


def preflight(root: Path, protocol_path: Path, protocol_sha256: str) -> tuple[dict, tuple[BoundAnchor, ...], WorldModel]:
    """Only actual source binder; caller cannot inject replacement replay or physics."""
    root = Path(root).resolve(strict=True)
    protocol_path = Path(protocol_path).absolute()
    if (protocol_path.is_symlink() or protocol_path.parent != root / "experiments"
            or protocol_path.suffix != ".json" or not protocol_path.is_file()
            or protocol_path.stat().st_size > 1024 * 1024):
        raise ValueError("protocol must be an existing nonsymlink experiments/*.json")
    spec = strict_json(pinned(root, f"experiments/{protocol_path.name}", protocol_sha256).read_text())
    files = {name: pinned(root, name, sha) for name, sha in V2_PINS.items()}
    v2 = strict_json(files[V2_PROTOCOL].read_text())
    check_protocol(spec, v2)
    for name, sha in spec["source_sha256"].items():
        pinned(root, name, sha)
    if spec["runtime"] != runtime_identity():
        raise ValueError("frozen Python/Box2D/TimeLimit/Torch pixel runtime changed")
    receipt = strict_json(files[f"{RUN}/cpu-model-export.json"].read_text())
    if (receipt.get("format") != "haic-tdmpc2-cpu-export-v1"
            or receipt.get("protocol_sha256") != V2_PINS[V2_PROTOCOL]
            or receipt.get("source_sha256") != v2["source_sha256"]
            or receipt.get("checkpoint_sha256") != V2_PINS[f"{RUN}/boundary.pt"]
            or receipt.get("sha256") != V2_PINS[f"{RUN}/cpu-model.pt"]
            or receipt.get("path") != "cpu-model.pt"):
        raise ValueError("v2 CPU export receipt identity differs")
    rows = [strict_json(line) for line in files[f"{RUN}/training.jsonl"].read_text().splitlines()]
    # torch.load is permitted only AFTER source, artifact SHA and runtime checks.
    checkpoint = torch.load(files[f"{RUN}/boundary.pt"], map_location="cpu", weights_only=False)
    anchors = bind_anchors(v2, checkpoint, rows)
    export = torch.load(files[f"{RUN}/cpu-model.pt"], map_location="cpu", weights_only=False)
    if (not isinstance(export, dict) or export.get("format") != "haic-tdmpc2-cpu-model-v1"
            or export.get("protocol_sha256") != V2_PINS[V2_PROTOCOL]
            or export.get("source_sha256") != v2["source_sha256"]
            or export.get("checkpoint_sha256") != V2_PINS[f"{RUN}/boundary.pt"]
            or any(export.get(key) != checkpoint.get(key) or receipt.get(key) != checkpoint.get(key)
                   for key in ("decisions", "episodes", "updates", "pretrain_updates"))
            or not isinstance(export.get("model_state"), dict)):
        raise ValueError("v2 CPU model export does not match bound replay checkpoint")
    model = WorldModel(TDMPC2ModelConfig(obs_shape={"rgb": PIXELS}, episodic=True))
    model.load_state_dict(export["model_state"], strict=True)
    model.eval()
    return spec, anchors, model


def _pixel_parity(observation: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.array_equal(model_observation(observation), expected):
        raise parity.ParityError(f"v2 model_observation differs at {label}")


def _step_parity(reward: Any, terminated: Any, truncated: Any, info: dict,
                 expected_reward: float, expected_flags: tuple[bool, bool, bool], label: str) -> None:
    done, terminal = episode_boundary(terminated, truncated, info)
    if (type(reward) not in (int, float, np.float32, np.float64)
            or not math.isfinite(float(reward)) or float(reward) != expected_reward
            or type(terminated) not in (bool, np.bool_)
            or type(truncated) not in (bool, np.bool_)
            or (bool(terminated), bool(truncated), terminal) != expected_flags
            or done != (bool(terminated) or bool(truncated))):
        raise parity.ParityError(f"v2 raw reward/terminal flags differ at {label}")


def capture_bound_prefix(env: Any, anchor: BoundAnchor) -> tuple[parity.Prefix, np.ndarray]:
    """First actual reset; stop on the first v2 replay mismatch, not after capture."""
    parity._chain(env)
    observation, info = env.reset()
    _pixel_parity(observation, anchor.observations[0], "reset")
    initial = parity.snapshot(env, observation, track_id=anchor.track_id, seed=anchor.seed, info=info)
    road_sha = dict(initial.state)["road_sha256"]
    steps = []
    history = parity.INITIAL_HASH
    for index, (action, raw_reward) in enumerate(zip(anchor.actions, anchor.rewards)):
        native = environment_action(action)
        history = hashlib.sha256(bytes.fromhex(history) + native.tobytes()).hexdigest()
        observation, reward, term, trunc, info, commands = parity._step_with_raw_tap(env, native)
        signature = parity.snapshot(env, observation, track_id=anchor.track_id, seed=anchor.seed,
                                    action=native, prefix_sha256=history, reward=reward,
                                    terminated=term, truncated=trunc, info=info, raw_commands=commands)
        _pixel_parity(observation, anchor.observations[index + 1], f"prefix[{index}]")
        _step_parity(reward, term, trunc, info, raw_reward, (False, False, False), f"prefix[{index}]")
        if (dict(signature.state)["road_sha256"] != road_sha
                or dict(signature.state)["outer.elapsed_steps"] != index + 1):
            raise parity.ParityError("v2 prefix road or decision clock differs")
        steps.append(signature)
    return parity.Prefix(anchor.track_id, anchor.seed, road_sha, initial,
                         tuple(environment_action(action).tobytes() for action in anchor.actions),
                         tuple(steps)), observation.copy()


def candidate_actions(anchor: BoundAnchor) -> dict[str, np.ndarray]:
    return {"logged": anchor.logged_actions.copy(), **{
        name: np.asarray(actions, dtype=np.float32) for name, actions in FIXED_SUFFIXES.items()
    }}


def predicted_returns(model: Any, pixels: np.ndarray, candidates: dict[str, np.ndarray],
                      seed: int) -> dict[str, float]:
    model.eval()
    results = {}
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        z0 = model.encode(torch.from_numpy(pixels.copy())[None], None)
        if not isinstance(z0, torch.Tensor) or z0.ndim != 2 or z0.shape[0] != 1 or not torch.isfinite(z0).all():
            raise ValueError("invalid model encoding of parity-gated anchor")
        for name, actions in candidates.items():
            if (actions.shape != (HORIZON, 3) or actions.dtype != np.float32
                    or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
                raise ValueError("invalid fixed H3 candidate")
            z = z0
            values = []
            for action in actions:
                tensor = torch.from_numpy(action.copy())[None]
                reward = two_hot_inv(model.reward(z, tensor, None), model.cfg)
                if reward.shape != (1, 1) or not torch.isfinite(reward).all():
                    raise ValueError("nonfinite model reward head")
                values.append(float(reward.item()))
                z = model.next(z, tensor, None)
                if not isinstance(z, torch.Tensor) or z.shape != z0.shape or not torch.isfinite(z).all():
                    raise ValueError("nonfinite model next latent")
            results[name] = math.fsum(DISCOUNT**t * value for t, value in enumerate(values))
            if not math.isfinite(results[name]):
                raise ValueError("nonfinite predicted H3 return")
    return results


def execute_suffix(env: Any, anchor: BoundAnchor, name: str, actions: np.ndarray,
                   road_sha256: str) -> dict:
    """Called only AFTER source-bound baseline and fresh full-prefix state equality."""
    rewards = []
    last_term = last_trunc = False
    for index, action in enumerate(actions):
        native = environment_action(action)
        obs, reward, term, trunc, info, commands = parity._step_with_raw_tap(env, native)
        sig = parity.snapshot(env, obs, track_id=anchor.track_id, seed=anchor.seed,
                              action=native, reward=reward, terminated=term, truncated=trunc,
                              info=info, raw_commands=commands)
        if (dict(sig.state)["outer.elapsed_steps"] != anchor.step + index + 1
                or dict(sig.state)["road_sha256"] != road_sha256):
            raise parity.ParityError("suffix road or decision clock differs")
        if name == "logged":
            _pixel_parity(obs, anchor.logged_observations[index + 1], f"logged_suffix[{index}]")
            _step_parity(reward, term, trunc, info, anchor.logged_rewards[index],
                         anchor.logged_flags[index], f"logged_suffix[{index}]")
        rewards.append(float(reward))
        last_term, last_trunc = bool(term), bool(trunc)
        if term or trunc:
            break
    return {
        "candidate": name, "actions_model": actions.tolist(), "steps": len(rewards),
        "raw_rewards": rewards, "real_discounted_raw_return": math.fsum(
            DISCOUNT**t * value for t, value in enumerate(rewards)),
        "terminated": last_term, "truncated": last_trunc,
        "full_h3": len(rewards) == HORIZON,
        "terminal_excluded": last_term or last_trunc,
        "ranking_eligible": len(rewards) == HORIZON and not (last_term or last_trunc),
    }


def rank(rows: list[dict]) -> dict:
    pairs = {"concordant": 0, "discordant": 0, "real_tie": 0, "predicted_tie": 0,
             "terminal_or_short_excluded": 0}
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if not left["ranking_eligible"] or not right["ranking_eligible"]:
                pairs["terminal_or_short_excluded"] += 1
                continue
            actual = left["real_discounted_raw_return"] - right["real_discounted_raw_return"]
            pred = left["predicted_reward_return"] - right["predicted_reward_return"]
            if abs(actual) <= TIE_TOLERANCE:
                pairs["real_tie"] += 1
            elif abs(pred) <= TIE_TOLERANCE:
                pairs["predicted_tie"] += 1
            else:
                pairs["concordant" if (actual > 0) == (pred > 0) else "discordant"] += 1
    comparable = pairs["concordant"] + pairs["discordant"]
    return {"pairs": pairs, "comparable_pairs": comparable,
            "concordance": pairs["concordant"] / comparable if comparable else None,
            "ranking_identifiable": comparable > 0,
            "full_h3_candidates": sum(row["full_h3"] for row in rows),
            "terminal_candidates": sum(row["terminal_excluded"] for row in rows),
            "ranking_eligible_candidates": sum(row["ranking_eligible"] for row in rows)}


def _save(path: Path, report: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _recheck_runtime_sources(root: Path, spec: dict) -> None:
    for name, sha in spec["source_sha256"].items():
        pinned(root, name, sha)
    if runtime_identity() != spec["runtime"]:
        raise ValueError("frozen environment/physics runtime changed before reset")


def run(root: Path, protocol_path: Path, sha: str, *, execute: bool = False,
        output: Path | None = None) -> dict:
    spec, anchors, model = preflight(root, protocol_path, sha)
    report = {
        "format": "haic-tdmpc2-v2-prefix-branches-result-v1", "status": "preflight_only",
        "protocol_sha256": sha, "v2_sha256": V2_PINS, "source_sha256": spec["source_sha256"],
        "runtime": spec["runtime"], "environment_resets_attempted": 0,
        "scope": "reused_consumed_TRAIN_same_reconstructed_accessible_state_only",
        "historical_road_or_hidden_box2d_state_proven": False,
        "same_state_counterfactual_ranking": False, "official_or_protected_result": False,
        "anchor_count": len(anchors), "candidate_count_per_anchor": 1 + len(FIXED_SUFFIXES),
        "horizon": HORIZON, "discount": DISCOUNT, "tie_tolerance": TIE_TOLERANCE,
        "anchors": [],
    }
    if not execute:
        return report
    root = Path(root).resolve(strict=True)
    if (output is None or not output.is_absolute() or output.is_symlink()
            or output.parent != root / "runs" or output.parent.is_symlink()
            or not output.parent.is_dir() or output.suffix != ".json" or output.exists()):
        raise ValueError("execution requires an unused absolute runs/*.json output")
    # Delayed import: this exact unmodified factory is the sole production env path.
    from train import build_env

    report["status"] = "running"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())  # Exclusive receipt before the FIRST actual reset.
    try:
        for anchor in anchors:
            entry = {"episode_id": anchor.episode_id, "track_id": anchor.track_id,
                     "geometry_seed": anchor.seed, "start_step": anchor.step,
                     "anchor_model_observation_sha256": hashlib.sha256(anchor.observations[-1].tobytes()).hexdigest(),
                     "anchor_model_observation_hex": anchor.observations[-1].tobytes().hex(),
                     "candidates": []}
            report["anchors"].append(entry)
            _recheck_runtime_sources(root, spec)
            env = build_env(anchor.track_id, anchor.seed, 2000, 4,
                            reward_shaping=False, obstacles=True)
            try:
                report["environment_resets_attempted"] += 1
                _save(output, report)
                captured, pixels = capture_bound_prefix(env, anchor)
            finally:
                env.close()
            entry["road_sha256"] = captured.road_sha256
            entry["anchor_observation_sha256"] = captured.steps[-1].observation_sha256
            entry["anchor_accessible_state_sha256"] = hashlib.sha256(repr(captured.steps[-1].state).encode()).hexdigest()
            candidate_map = candidate_actions(anchor)
            predicted = predicted_returns(model, anchor.observations[-1], candidate_map,
                                          spec["augmentation_seed"] + anchor.episode_id * 1000 + anchor.step)
            for name, actions in candidate_map.items():
                _recheck_runtime_sources(root, spec)
                env = build_env(anchor.track_id, anchor.seed, 2000, 4,
                                reward_shaping=False, obstacles=True)
                try:
                    report["environment_resets_attempted"] += 1
                    if report["environment_resets_attempted"] > spec["max_resets"]:
                        raise ValueError("frozen reset budget exceeded")
                    _save(output, report)
                    replayed = parity._compare_replay_prefix(env, captured)
                    _pixel_parity(replayed, anchor.observations[-1], "branch_anchor")
                    if not np.array_equal(model_observation(replayed), model_observation(pixels)):
                        raise parity.ParityError("branch accessible anchor pixels differ")
                    row = execute_suffix(env, anchor, name, actions, captured.road_sha256)
                finally:
                    env.close()
                row["predicted_reward_return"] = predicted[name]
                row["model_action_bytes_hex"] = actions.tobytes().hex()
                entry["candidates"].append(row)
                _save(output, report)
            entry["ranking"] = rank(entry["candidates"])
            _save(output, report)
        report["aggregate_pair_counts"] = {key: sum(a["ranking"]["pairs"][key] for a in report["anchors"])
                                           for key in report["anchors"][0]["ranking"]["pairs"]}
        comparable = report["aggregate_pair_counts"]["concordant"] + report["aggregate_pair_counts"]["discordant"]
        report["concordance_on_comparable_pairs"] = (report["aggregate_pair_counts"]["concordant"] / comparable
                                                      if comparable else None)
        report["same_state_counterfactual_ranking"] = True
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "stopped_parity_or_execution_failure"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _save(output, report)
        raise
    _save(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--execute", action="store_true", help="explicit, authorized reset/branch opt-in")
    parser.add_argument("--output", type=Path, help="unused absolute runs/*.json failure/result receipt")
    parser.add_argument("--print-runtime", action="store_true", help="print runtime identity; no artifact/env load")
    args = parser.parse_args()
    if args.print_runtime:
        print(json.dumps(runtime_identity(), sort_keys=True))
        return
    if args.protocol is None or args.protocol_sha256 is None:
        parser.error("--protocol and --protocol-sha256 are required for preflight or execution")
    result = run(ROOT, args.protocol, args.protocol_sha256, execute=args.execute, output=args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
