"""Conditional RAW100k H5 real-suffix ranking on reconstructed consumed TRAIN states.

Default --preflight performs only SHA/ledger/quality checks, with no torch.load or
environment construction. --execute additionally requires a NEW, externally
SHA-frozen experiments/tdmpc2-h5-branches-v1.json protocol and an unused runs/
receipt. No such branch protocol or passing logged-H5 result is supplied here.
This uses the original H3-trained model at five inference steps, NOT an H5-trained
model. Reconstructed accessible state is not historical hidden Box2D solver state.
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
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner


ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/tdmpc2-long-20260928-v2"
PROTOCOL = "experiments/tdmpc2-h5-branches-v1.json"
LOGGED_PROTOCOL = "experiments/tdmpc2-h5-logged-v1.json"
LOGGED_PROTOCOL_SHA = "202ba8a484bb1c787fcf4524c8d0373376362a0efb1de390a976a37d8b987223"
LOGGED_RESULT = "runs/tdmpc2-long-v2-h5-logged-20260929-v1.json"
LOGGED_RESULT_SHA = "47baa5b9196fa55ca4eff492cab0bf559fb7a819a6c6e0d88b8c9481db7df042"
LOGGED_MANIFEST_SHA = "27514704f38a090535040b74adb78b715b98e370556dbe7c0c7d2f326541f4fc"
HELPER = "scripts/diagnose_tdmpc2_checkpoint_losses.py"
HELPER_SHA = "ef6552479192e2e113951edb3bfbf7a4c87b3a508dd884c45ce6a270a0aebe6f"
RAW_PROTOCOL_SHA = "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec"
RAW_RESULT_SHA = "287470932dbba9a7eeed778a5be7762b736e6679d99fd29078c5103890a75a4b"
CHECKPOINT = f"{RUN}/checkpoint-at-least-100000-step-100354.pt"
CHECKPOINT_SHA = "aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295"
TRAIN_SHA = "84f06ee4dcc8568bd5d384d1811252321ad5dc6a98ce19d18badb448d09123d7"
STEPS_SHA = "a94ce156afcb6e4f706474d6cf2d2ccb4754a868eb75a3eb4b53df5d2be37945"
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
ANCHORS = tuple((ep, step) for ep in range(4) for step in (16, 50, 100))
HORIZON, DISCOUNT, TIE_TOLERANCE = 5, 0.995, 1e-6
PIXELS = (4, 64, 64)
FINAL = {"target": 100000, "decisions": 100354, "updates": 100354, "episodes": 307,
         "action_dim": 3, "resume_supported": False}
ENVIRONMENT = {"track_id": 1, "geometry_seeds": list(ROADS), "max_steps": 2000,
               "frame_skip": 4, "obstacles": True, "reward_shaping": False}
GATE = {"min_informative_real_pairs": 40, "min_informative_roads": 3,
        "min_concordance": 0.60}
EXTRA_SOURCES = frozenset({
    "scripts/diagnose_tdmpc2_h5_branches.py", HELPER, "scripts/diagnose_tdmpc2_h5_logged.py",
    "haic/algorithms/rlpd/prefix_parity.py", "train.py", "action_representation.py",
    "action_smoothing.py", "tracking.py",
})
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
    observations: np.ndarray
    actions: np.ndarray
    rewards: tuple[float, ...]
    logged_actions: np.ndarray
    logged_observations: np.ndarray
    logged_rewards: tuple[float, ...]
    logged_flags: tuple[tuple[bool, bool, bool], ...]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return digest_stream(stream)


def digest_stream(stream: Any) -> str:
    sha = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        sha.update(block)
    return sha.hexdigest()


def _sha(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("expected lowercase SHA-256")
    return value


def pinned(root: Path, name: str, expected: str) -> Path:
    _sha(expected)
    if (not isinstance(name, str) or not name or "\\" in name or name.startswith("/")
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("noncanonical pinned path")
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in pinned path: {name}")
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"missing or changed pinned source: {name}")
    return path


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict:
    if not 0 < len(data) <= 1024 * 1024:
        raise ValueError("invalid JSON record length")
    value = json.loads(data, object_pairs_hook=_unique,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def runtime_identity() -> dict[str, str]:
    import Box2D
    import Box2D._Box2D as native
    import cv2
    import gymnasium
    from gymnasium.wrappers.time_limit import TimeLimit

    if Box2D.__file__ is None or native.__file__ is None:
        raise ValueError("native Box2D runtime unavailable")
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "opencv": str(getattr(cv2, "__version__")), "gymnasium": gymnasium.__version__, "box2d": Box2D.__version__,
            "box2d_native_sha256": digest(Path(native.__file__)),
            "box2d_python_sha256": digest(Path(Box2D.__file__)),
            "timelimit_source_sha256": digest(Path(inspect.getfile(TimeLimit)))}


def _ref(root: Path, value: Any, name: str | None = None) -> tuple[dict, Path]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("incomplete frozen artifact reference")
    path = value["path"]
    if name is not None and path != name:
        raise ValueError("frozen artifact path differs")
    return _json(pinned(root, path, value["sha256"]).read_bytes()), root / path


def check_protocol(spec: dict, raw: dict, logged: dict) -> None:
    """Pure structural contract; each source SHA is checked against bytes in preflight."""
    keys = {"format", "purpose", "source", "logged_quality", "source_sha256", "runtime",
            "anchors", "candidates", "environment", "horizon", "discount", "tie_tolerance",
            "augmentation_seed", "max_resets", "quality_gate"}
    raw_refs = {"protocol": {"path": "experiments/tdmpc2-long-reused-train-v2.json", "sha256": RAW_PROTOCOL_SHA},
                "result": {"path": f"{RUN}/result.json", "sha256": RAW_RESULT_SHA},
                "checkpoint": {"path": CHECKPOINT, "sha256": CHECKPOINT_SHA},
                "training_ledger": {"path": f"{RUN}/training.jsonl", "sha256": TRAIN_SHA},
                "step_ledger": {"path": f"{RUN}/steps.jsonl", "sha256": STEPS_SHA}}
    logged_refs = {"protocol": {"path": LOGGED_PROTOCOL, "sha256": LOGGED_PROTOCOL_SHA},
                   "result": {"path": LOGGED_RESULT, "sha256": LOGGED_RESULT_SHA}}
    quality = spec.get("logged_quality", {}) if isinstance(spec, dict) else {}
    if (not isinstance(spec, dict) or set(spec) != keys
            or spec["format"] != "haic-tdmpc2-raw100k-h5-branches-v1"
            or spec["purpose"] != "consumed-TRAIN-same-reconstructed-accessible-state-H5-raw-reward"
            or spec["source"] != raw_refs or quality != logged_refs
            or spec["anchors"] != [{"episode_id": ep, "start_step": step} for ep, step in ANCHORS]
            or spec["candidates"] != {"logged": None, **FIXED_SUFFIXES}
            or spec["environment"] != ENVIRONMENT or spec["horizon"] != HORIZON
            or spec["discount"] != DISCOUNT or spec["tie_tolerance"] != TIE_TOLERANCE
            or spec["quality_gate"] != GATE or spec["max_resets"] != len(ANCHORS) * 6
            or type(spec["augmentation_seed"]) is not int
            or not 0 <= spec["augmentation_seed"] < 2**32
            or not isinstance(spec["runtime"], dict)
            or set(spec["runtime"]) != set(runtime_identity())
            or not all(isinstance(v, str) for v in spec["runtime"].values())
            or raw.get("cells") != [{"track_id": 1, "geometry_seed": seed} for seed in ROADS]
            or raw.get("episode_schedule") != [0, 1, 2, 3]
            or raw.get("selection") != {"arm": "independent_3d", "action_dim": 3}
            or raw.get("training", {}).get("horizon") != 3
            or raw.get("training", {}).get("discount") != DISCOUNT
            or logged.get("format") != "haic-tdmpc2-h5-logged-v1"
            or logged.get("source_protocol") != raw_refs["protocol"]
            or logged.get("source_result") != raw_refs["result"]
            or logged.get("checkpoint") != raw_refs["checkpoint"]
            or logged.get("training_ledger") != raw_refs["training_ledger"]
            or logged.get("step_ledger") != raw_refs["step_ledger"]
            or logged.get("environment_resets") != 0
            or logged.get("selection", {}).get("manifest_sha256") != LOGGED_MANIFEST_SHA
            or not isinstance(logged.get("source_sha256"), dict)
            or set(logged["source_sha256"]) != {"scripts/diagnose_tdmpc2_h5_logged.py", HELPER}
            or logged["source_sha256"].get(HELPER) != HELPER_SHA
            or not isinstance(spec["source_sha256"], dict)
            or set(spec["source_sha256"]) != set(raw.get("source_sha256", {})) | EXTRA_SOURCES
            or any(spec["source_sha256"].get(name) != sha for name, sha in raw["source_sha256"].items())
            or any(spec["source_sha256"].get(name) != sha for name, sha in logged["source_sha256"].items())):
        raise ValueError("not the fixed RAW100k H5 source-bound branch contract")


def check_logged_quality(receipt: dict, logged: dict, spec: dict) -> dict:
    """Recompute the exact primary no-reset H5 screen; its claim alone is insufficient."""
    selection = logged.get("selection", {})
    rows = receipt.get("windows")
    if (receipt.get("format") != "haic-tdmpc2-h5-logged-result-v1"
            or receipt.get("quality_gate_passed") is not True
            or spec.get("logged_quality") != {
                "protocol": {"path": LOGGED_PROTOCOL, "sha256": LOGGED_PROTOCOL_SHA},
                "result": {"path": LOGGED_RESULT, "sha256": LOGGED_RESULT_SHA}}
            or receipt.get("scope") != "in_replay_observational_fit_only"
            or receipt.get("same_state_counterfactual_ranking") is not False
            or receipt.get("environment_resets") != 0 or receipt.get("optimizer_steps") != 0
            or receipt.get("protocol_path") != LOGGED_PROTOCOL
            or receipt.get("protocol_sha256") != spec["logged_quality"]["protocol"]["sha256"]
            or receipt.get("source_protocol_sha256") != RAW_PROTOCOL_SHA
            or receipt.get("source_result_sha256") != RAW_RESULT_SHA
            or receipt.get("source_checkpoint_sha256") != CHECKPOINT_SHA
            or receipt.get("training_ledger_sha256") != TRAIN_SHA
            or receipt.get("step_ledger_sha256") != STEPS_SHA
            or receipt.get("source_sha256") != logged["source_sha256"]
            or selection.get("manifest_sha256") != LOGGED_MANIFEST_SHA
            or receipt.get("window_manifest_sha256") != LOGGED_MANIFEST_SHA
            or selection.get("window_count") != 256 or selection.get("horizons") != [3, 5]
            or selection.get("discount") != DISCOUNT or selection.get("constant_reward_per_step") != 0
            or receipt.get("window_count") != 256 or not isinstance(rows, list) or len(rows) != 256
            or not isinstance(receipt.get("horizons"), dict) or "h5" not in receipt["horizons"]):
        raise ValueError("separately frozen H5 logged-replay receipt is missing or unbound")
    manifest, targets, estimates, positives, guesses = [], [], [], [], []
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("episode_id")) is not int
                or row["episode_id"] < 0 or type(row.get("start_step")) is not int
                or row["start_step"] < 0 or row.get("track_id") != 1
                or row.get("geometry_seed") not in ROADS):
            raise ValueError("invalid logged H5 window identity")
        action_sha = _sha(row.get("actions_sha256"))
        manifest.append({"episode_id": row["episode_id"], "start_step": row["start_step"],
                         "actions_sha256": action_sha})
        fields = [row.get(key) for key in ("actual_rewards", "predicted_rewards",
                                           "terminal", "predicted_terminal_probability")]
        if any(not isinstance(values, list) or len(values) != 5 for values in fields):
            raise ValueError("invalid logged H5 rewards or semantic-terminal probabilities")
        actual_rewards, predicted_rewards, labels, probabilities = (cast(list, field) for field in fields)
        if (any(type(v) not in (int, float) or not math.isfinite(v)
                for values in (actual_rewards, predicted_rewards, probabilities) for v in values)
                or any(type(v) is not bool for v in labels)
                or any(not 0 <= v <= 1 for v in probabilities)):
            raise ValueError("invalid logged H5 rewards or semantic-terminal probabilities")
        term, trunc = row.get("terminated"), row.get("truncated")
        if (not isinstance(term, list) or not isinstance(trunc, list)
                or len(term) != 5 or len(trunc) != 5
                or any(type(a) is not bool or type(b) is not bool or a and not label
                       or label and not (a or b) for a, b, label in zip(term, trunc, labels))
                or any(any(values[:4]) for values in (term, trunc, labels))):
            raise ValueError("logged H5 terminal/timeout labels differ")
        targets.append(math.fsum(DISCOUNT**t * value for t, value in enumerate(actual_rewards)))
        estimates.append(math.fsum(DISCOUNT**t * value for t, value in enumerate(predicted_rewards)))
        positives.extend(labels)
        guesses.extend(value > 0.5 for value in probabilities)
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (len({(r["episode_id"], r["start_step"]) for r in rows}) != 256
            or manifest_sha != selection["manifest_sha256"]
            or receipt.get("road_window_counts") != {
                str(seed): sum(row["geometry_seed"] == seed for row in rows) for seed in ROADS}):
        raise ValueError("logged H5 window/road manifest differs from frozen selection")
    mae = math.fsum(abs(a - b) for a, b in zip(targets, estimates)) / 256
    baseline = math.fsum(map(abs, targets)) / 256
    positive = sum(positives)
    negative = len(positives) - positive
    tp = sum(a and b for a, b in zip(positives, guesses))
    fp = sum(not a and b for a, b in zip(positives, guesses))
    summary = receipt["horizons"]["h5"]
    if (not isinstance(summary, dict) or summary.get("windows") != 256
            or summary.get("reward_transitions") != 1280
            or summary.get("semantic_terminal_positive_transitions") != positive
            or summary.get("semantic_terminal_negative_transitions") != negative
            or summary.get("terminal_true_positive") != tp or summary.get("terminal_false_positive") != fp
            or type(summary.get("mae_per_discounted_step")) not in (int, float)
            or type(summary.get("constant_mae_per_discounted_step")) not in (int, float)
            or not math.isclose(summary["mae_per_discounted_step"], mae / sum(DISCOUNT**t for t in range(5)),
                                rel_tol=0, abs_tol=1e-7)
            or not math.isclose(summary["constant_mae_per_discounted_step"],
                                baseline / sum(DISCOUNT**t for t in range(5)), rel_tol=0, abs_tol=1e-7)
            or not (math.isfinite(mae) and math.isfinite(baseline) and 0 < positive < 1280)
            or not (mae < baseline and tp * negative > fp * positive)):
        raise ValueError("separately frozen H5 logged-replay quality gate did not pass")
    return {"windows": 256, "positive_transitions": positive, "negative_transitions": negative,
            "true_positive": tp, "false_positive": fp, "discounted_return_mae": mae,
            "zero_constant_discounted_return_mae": baseline}


def preflight(root: Path, protocol_sha256: str) -> dict:
    """No checkpoint deserialization and no environment construction/reset."""
    root = Path(root).resolve(strict=True)
    spec = _json(pinned(root, PROTOCOL, _sha(protocol_sha256)).read_bytes())
    quality_refs = spec.get("logged_quality")
    if (not isinstance(quality_refs, dict) or set(quality_refs) != {"protocol", "result"}
            or any(not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                   for ref in quality_refs.values())):
        raise ValueError("missing separately frozen H5 logged quality references")
    pinned(root, HELPER, HELPER_SHA)
    from scripts import diagnose_tdmpc2_checkpoint_losses as source

    raw = source._protocol(root)  # Hard-coded RAW identity and every original source byte.
    result = _json(pinned(root, f"{RUN}/result.json", RAW_RESULT_SHA).read_bytes())
    for name, sha in ((CHECKPOINT, CHECKPOINT_SHA), (f"{RUN}/training.jsonl", TRAIN_SHA),
                      (f"{RUN}/steps.jsonl", STEPS_SHA)):
        pinned(root, name, sha)
    if (result.get("status") != "completed_boundary_at_least_100k"
            or result.get("protocol_sha256") != RAW_PROTOCOL_SHA
            or result.get("source_sha256") != raw["source_sha256"]
            or result.get("training_ledger_sha256") != TRAIN_SHA
            or result.get("step_ledger_sha256") != STEPS_SHA
            or any(result.get(key) != val for key, val in FINAL.items() if key != "target")
            or result.get("reused_train_only") is not True or result.get("evaluation") is not None
            or not isinstance(result.get("checkpoints"), list) or len(result["checkpoints"]) != 4):
        raise ValueError("RAW100k result does not bind complete original source")
    pin = source._cursor(root, raw, 100000)
    if (pin["row"] != {"event": "checkpoint", **result["checkpoints"][-1]}
            or pin["checkpoint_sha256"] != CHECKPOINT_SHA
            or pin["ledger_prefix_sha256"] != TRAIN_SHA or pin["step_prefix_sha256"] != STEPS_SHA):
        raise ValueError("RAW100k complete checkpoint cursor differs from ledgers")
    logged, _ = _ref(root, quality_refs["protocol"], LOGGED_PROTOCOL)
    check_protocol(spec, raw, logged)
    receipt, _ = _ref(root, quality_refs["result"])
    quality = check_logged_quality(receipt, logged, spec)
    for name, sha in spec["source_sha256"].items():
        pinned(root, name, sha)
    if runtime_identity() != spec["runtime"]:
        raise ValueError("frozen Python/Box2D/TimeLimit/Torch runtime changed")
    episodes = []
    with (root / f"{RUN}/training.jsonl").open("rb") as stream:
        for line in stream:
            row = _json(line)
            if row.get("event") == "episode":
                episodes.append(row)
    if (len(episodes) != FINAL["episodes"]
            or any(row.get("episode") != i or row.get("track_id") != 1
                   or row.get("geometry_seed") != ROADS[i % 4] for i, row in enumerate(episodes))):
        raise ValueError("RAW100k episode/road schedule differs")
    return {"spec": spec, "raw": raw, "quality": quality, "episodes": episodes,
            "checkpoint": pin["checkpoint"], "protocol_sha256": protocol_sha256}


def bind_replay(state: dict, raw: dict, episodes: list[dict], steps_path: Path) -> tuple[BoundAnchor, ...]:
    """Bind EVERY recorded action/native command/reward/flag/pixel array before reset."""
    replay = state.get("replay") if isinstance(state, dict) else None
    if (not isinstance(state, dict) or state.get("format") != raw["format"]
            or state.get("protocol_sha256") != RAW_PROTOCOL_SHA
            or state.get("source_sha256") != raw["source_sha256"]
            or any(state.get(k) != v for k, v in FINAL.items())
            or not isinstance(state.get("learner"), dict) or "q_scale" not in state["learner"]
            or not isinstance(replay, dict) or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or any(replay.get(k) != v for k, v in {
                "capacity": 120000, "horizon": 3, "action_dim": 3, "augmentation_pad": 3,
                "include_partial": False, "bootstrap_on_truncation": True,
                "observation_shape": PIXELS, "next_episode_id": len(episodes),
                "size": FINAL["decisions"], "active": None}.items())
            or not isinstance(replay.get("episodes"), list) or len(replay["episodes"]) != len(episodes)):
        raise ValueError("not the complete RAW100k H3-trained checkpoint/replay")
    selected = {}
    decisions = 0
    with steps_path.open("rb") as stream:
        for eid, (episode, ledger) in enumerate(zip(replay["episodes"], episodes)):
            if not isinstance(episode, dict) or type(ledger.get("length")) is not int:
                raise ValueError("invalid complete RAW replay episode")
            length = ledger["length"]
            actions, obs, rewards = (episode.get(k) for k in ("actions", "observations", "rewards"))
            flags = [episode.get(k) for k in ("terminated", "truncated", "terminal")]
            if (episode.get("episode_id") != eid or episode.get("start_step") != 0
                    or not 1 <= length <= 2000 or not isinstance(actions, np.ndarray)
                    or actions.shape != (length, 3) or actions.dtype != np.float32
                    or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)
                    or not isinstance(obs, np.ndarray) or obs.shape != (length + 1, *PIXELS)
                    or obs.dtype != np.uint8 or not isinstance(rewards, np.ndarray)
                    or rewards.shape != (length,) or rewards.dtype != np.float32
                    or not np.isfinite(rewards).all()
                    or any(not isinstance(f, np.ndarray) or f.shape != (length,) or f.dtype != np.bool_
                           for f in flags)):
                raise ValueError("RAW replay pixel/action/reward/flag shape or dtype differs")
            actions, obs, rewards = cast(np.ndarray, actions), cast(np.ndarray, obs), cast(np.ndarray, rewards)
            term, trunc, terminal = (cast(np.ndarray, flag) for flag in flags)
            if (np.any(term[:-1] | trunc[:-1] | terminal[:-1])
                    or not (term[-1] or trunc[-1]) or np.any(term & ~terminal)
                    or np.any(terminal & ~(term | trunc))
                    or any(bool(f[-1]) is not ledger[k] for f, k in zip((term, trunc, terminal),
                                                                         ("terminated", "truncated", "terminal")))):
                raise ValueError("RAW replay terminal/timeout boundary differs")
            action_sha, native_sha, total = hashlib.sha256(), hashlib.sha256(), 0.0
            raw_rewards = []
            for offset, action in enumerate(actions):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("RAW step ledger ends before complete replay")
                row = _json(line)
                native = environment_action(action)
                if (row.get("episode") != eid or row.get("decision") != decisions + 1
                        or row.get("track_id") != 1 or row.get("geometry_seed") != ROADS[eid % 4]
                        or row.get("action_f32_hex") != action.tobytes().hex()
                        or row.get("native_action_f32_hex") != native.tobytes().hex()
                        or type(row.get("reward")) not in (int, float)
                        or not math.isfinite(row["reward"]) or np.float32(row["reward"]) != rewards[offset]
                        or any(type(row.get(k)) is not bool or row[k] != bool(f[offset])
                               for f, k in zip((term, trunc, terminal), ("terminated", "truncated", "terminal")))):
                    raise ValueError(f"RAW replay/road/native action/reward/flag lineage differs at {eid}:{offset}")
                action_sha.update(action.tobytes())
                native_sha.update(native.tobytes())
                total += row["reward"]
                raw_rewards.append(row["reward"])
                decisions += 1
            if (action_sha.hexdigest() != ledger.get("action_trace_sha256")
                    or native_sha.hexdigest() != ledger.get("native_action_trace_sha256")
                    or not math.isclose(total, ledger.get("return", math.inf), rel_tol=0, abs_tol=1e-4)
                    or ledger.get("decisions") != decisions):
                raise ValueError("RAW replay full-episode action/native/return lineage differs")
            for ep, start in ANCHORS:
                if ep != eid:
                    continue
                if start + HORIZON > length or np.any(term[:start] | trunc[:start]):
                    raise ValueError("fixed RAW anchor has no five logged decisions")
                selected[ep, start] = BoundAnchor(
                    ep, 1, ROADS[ep], start, obs[:start + 1].copy(), actions[:start].copy(),
                    tuple(raw_rewards[:start]), actions[start:start + HORIZON].copy(),
                    obs[start:start + HORIZON + 1].copy(), tuple(raw_rewards[start:start + HORIZON]),
                    tuple((bool(term[t]), bool(trunc[t]), bool(terminal[t]))
                          for t in range(start, start + HORIZON)))
        if stream.read(1) or decisions != FINAL["decisions"] or set(selected) != set(ANCHORS):
            raise ValueError("RAW complete step/replay cursor or fixed anchors differ")
    return tuple(selected[key] for key in ANCHORS)


def load_bound_model(bundle: dict, root: Path) -> tuple[tuple[BoundAnchor, ...], WorldModel]:
    """Only the SHA-gated production path deserializes the trusted RAW checkpoint."""
    _recheck(root, bundle, artifacts=True)
    with torch.random.fork_rng(devices=[]):
        with bundle["checkpoint"].open("rb") as stream:
            if digest_stream(stream) != CHECKPOINT_SHA:
                raise ValueError("RAW checkpoint changed before torch.load")
            stream.seek(0)
            state = torch.load(stream, map_location="cpu", weights_only=False)
        anchors = bind_replay(state, bundle["raw"], bundle["episodes"], root / f"{RUN}/steps.jsonl")
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": PIXELS}, episodic=True))
        weights = {key.removeprefix("model."): value for key, value in state["learner"].items()
                   if key.startswith("model.")}
        model.load_state_dict(weights, strict=True)
        model.eval()
    return anchors, model


def _pixel(observation: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.array_equal(model_observation(observation), expected):
        raise parity.ParityError(f"RAW model_observation differs at {label}")


def _step(reward: Any, term: Any, trunc: Any, info: dict, expected: float,
          flags: tuple[bool, bool, bool], label: str) -> None:
    done, terminal = episode_boundary(term, trunc, info)
    if (type(reward) not in (int, float, np.float32, np.float64)
            or not math.isfinite(float(reward)) or float(reward) != expected
            or type(term) not in (bool, np.bool_) or type(trunc) not in (bool, np.bool_)
            or (bool(term), bool(trunc), terminal) != flags
            or done != (bool(term) or bool(trunc))):
        raise parity.ParityError(f"RAW raw reward/terminal/timeout flags differ at {label}")


def capture_bound_prefix(env: Any, anchor: BoundAnchor) -> parity.Prefix:
    parity._chain(env)
    observation, info = env.reset()
    _pixel(observation, anchor.observations[0], "reset")
    initial = parity.snapshot(env, observation, track_id=anchor.track_id, seed=anchor.seed, info=info)
    road_sha = dict(initial.state)["road_sha256"]
    signatures = []
    history = parity.INITIAL_HASH
    for index, (action, reward) in enumerate(zip(anchor.actions, anchor.rewards)):
        native = environment_action(action)
        history = hashlib.sha256(bytes.fromhex(history) + native.tobytes()).hexdigest()
        observation, observed, term, trunc, info, commands = parity._step_with_raw_tap(env, native)
        signature = parity.snapshot(env, observation, track_id=anchor.track_id, seed=anchor.seed,
                                    action=native, prefix_sha256=history, reward=observed,
                                    terminated=term, truncated=trunc, info=info, raw_commands=commands)
        _pixel(observation, anchor.observations[index + 1], f"prefix[{index}]")
        _step(observed, term, trunc, info, reward, (False, False, False), f"prefix[{index}]")
        if (dict(signature.state)["road_sha256"] != road_sha
                or dict(signature.state)["outer.elapsed_steps"] != index + 1):
            raise parity.ParityError("RAW prefix road/clock differs")
        signatures.append(signature)
    return parity.Prefix(anchor.track_id, anchor.seed, road_sha, initial,
                         tuple(environment_action(action).tobytes() for action in anchor.actions),
                         tuple(signatures))


def candidate_actions(anchor: BoundAnchor) -> dict[str, np.ndarray]:
    return {"logged": anchor.logged_actions.copy(), **{
        name: np.asarray(value, dtype=np.float32) for name, value in FIXED_SUFFIXES.items()}}


def predicted_returns(model: Any, pixels: np.ndarray, candidates: dict[str, np.ndarray],
                       seed: int) -> dict[str, float]:
    model.eval()
    results = {}
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        z0 = model.encode(torch.from_numpy(pixels.copy())[None], None)
        if not isinstance(z0, torch.Tensor) or z0.ndim != 2 or z0.shape[0] != 1 or not torch.isfinite(z0).all():
            raise ValueError("invalid model encoding of bound RAW prebranch observation")
        for name, actions in candidates.items():
            if (actions.shape != (HORIZON, 3) or actions.dtype != np.float32
                    or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
                raise ValueError("invalid fixed H5 symmetric action bytes")
            z = z0
            rewards = []
            for index, action in enumerate(actions):
                tensor = torch.from_numpy(action.copy())[None]
                reward = two_hot_inv(model.reward(z, tensor, None), model.cfg)
                if reward.shape != (1, 1) or not torch.isfinite(reward).all():
                    raise ValueError("nonfinite world-model raw reward prediction")
                rewards.append(float(reward.item()))
                if index + 1 < HORIZON:
                    z = model.next(z, tensor, None)
                    if not isinstance(z, torch.Tensor) or z.shape != z0.shape or not torch.isfinite(z).all():
                        raise ValueError("nonfinite world-model next latent")
            results[name] = math.fsum(DISCOUNT**t * value for t, value in enumerate(rewards))
            if not math.isfinite(results[name]):
                raise ValueError("nonfinite predicted H5 reward-only return")
    return results


def matched_model_scores(model: Any, pixels: np.ndarray, candidates: dict[str, np.ndarray],
                         seed: int) -> dict[str, dict[str, float]]:
    """Score identical candidate bytes at both horizons, with and without MPPI bootstrap."""
    model.eval()
    names = list(candidates)
    if len(names) != 5 or any(a.shape != (HORIZON, 3) or a.dtype != np.float32
                              or not np.isfinite(a).all() or np.any(np.abs(a) > 1)
                              for a in candidates.values()):
        raise ValueError("matched scores require five finite fixed H5 candidates")
    results = {name: {} for name in names}
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        z0 = model.encode(torch.from_numpy(pixels.copy())[None], None)
        if not isinstance(z0, torch.Tensor) or z0.ndim != 2 or z0.shape[0] != 1 or not torch.isfinite(z0).all():
            raise ValueError("invalid matched-score anchor encoding")
        for name in names:
            z = z0
            rewards = []
            for index, action in enumerate(candidates[name]):
                tensor = torch.from_numpy(action.copy())[None]
                reward = two_hot_inv(model.reward(z, tensor, None), model.cfg)
                if reward.shape != (1, 1) or not torch.isfinite(reward).all():
                    raise ValueError("nonfinite matched-score predicted reward")
                rewards.append(float(reward.item()))
                if index + 1 < HORIZON:
                    z = model.next(z, tensor, None)
                    if not isinstance(z, torch.Tensor) or z.shape != z0.shape or not torch.isfinite(z).all():
                        raise ValueError("nonfinite matched-score predicted latent")
            for horizon in (3, 5):
                results[name][f"h{horizon}_reward"] = math.fsum(
                    DISCOUNT**t * reward for t, reward in enumerate(rewards[:horizon]))

        # _estimate_value is the actual MPPI score: alive-gated reward plus
        # a sampled prior-action Q at the last predicted latent. Five fixed
        # trajectories are scored, not a full MPPI search population.
        for horizon in (3, 5):
            config = PlannerConfig(action_dim=3, discount=DISCOUNT, horizon=horizon,
                                   episodic=True, num_samples=5, num_elites=1,
                                   num_pi_trajs=0, num_bins=model.cfg.num_bins)
            planner = TDMPC2Planner(model, config)
            actions = torch.from_numpy(np.stack([candidates[name][:horizon] for name in names], axis=1).copy())
            torch.manual_seed(seed + horizon)
            values = planner._estimate_value(z0.repeat(5, 1), actions)
            if values.shape != (5, 1) or not torch.isfinite(values).all():
                raise ValueError("nonfinite termination/Q-gated candidate planner score")
            for name, value in zip(names, values[:, 0].tolist()):
                results[name][f"h{horizon}_planner"] = float(value)
    if any(not math.isfinite(value) for scores in results.values() for value in scores.values()):
        raise ValueError("nonfinite matched candidate score")
    return results


def execute_suffix(env: Any, anchor: BoundAnchor, name: str, actions: np.ndarray,
                   road_sha: str) -> dict:
    if (name not in candidate_actions(anchor) or actions.shape != (HORIZON, 3)
            or actions.dtype != np.float32
            or actions.tobytes() != candidate_actions(anchor)[name].tobytes()):
        raise ValueError("suffix action bytes differ from predeclared candidate")
    rewards, ending = [], None
    for index, action in enumerate(actions):
        native = environment_action(action)
        obs, reward, term, trunc, info, commands = parity._step_with_raw_tap(env, native)
        signature = parity.snapshot(env, obs, track_id=anchor.track_id, seed=anchor.seed,
                                    action=native, reward=reward, terminated=term, truncated=trunc,
                                    info=info, raw_commands=commands)
        if (dict(signature.state)["outer.elapsed_steps"] != anchor.step + index + 1
                or dict(signature.state)["road_sha256"] != road_sha):
            raise parity.ParityError("H5 suffix road/clock differs")
        if name == "logged":
            _pixel(obs, anchor.logged_observations[index + 1], f"logged_suffix[{index}]")
            _step(reward, term, trunc, info, anchor.logged_rewards[index],
                  anchor.logged_flags[index], f"logged_suffix[{index}]")
        if type(reward) not in (int, float, np.float32, np.float64) or not math.isfinite(float(reward)):
            raise parity.ParityError("nonfinite real raw H5 reward")
        rewards.append(float(reward))
        if term or trunc:
            ending = "terminated" if term else ("finished" if info.get("finished") else "timeout")
            break
    reason = None if len(rewards) == HORIZON else (
        "terminated_before_fifth" if ending == "terminated" else
        "finished_before_fifth" if ending == "finished" else
        "timeout_before_fifth" if ending == "timeout" else "short_without_boundary")
    return {"candidate": name, "model_action_bytes_hex": actions.tobytes().hex(),
            "native_action_bytes_hex": b"".join(environment_action(a).tobytes() for a in actions).hex(),
            "steps": len(rewards), "raw_rewards": rewards,
             "real_discounted_raw_return": math.fsum(DISCOUNT**t * r for t, r in enumerate(rewards)),
             "real_h3_prefix_raw_return": (math.fsum(DISCOUNT**t * r for t, r in enumerate(rewards[:3]))
                                           if len(rewards) >= 3 else None),
            "ending": ending, "full_h5": len(rewards) == HORIZON,
            "ranking_eligible": len(rewards) == HORIZON, "exclusion_reason": reason}


def rank(rows: list[dict]) -> dict:
    pairs = {"concordant": 0, "discordant": 0, "real_tie": 0,
             "predicted_tie": 0, "terminal_or_short_excluded": 0}
    if len(rows) != 5:
        raise ValueError("exactly five independently branched H5 candidates required")
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if not left["ranking_eligible"] or not right["ranking_eligible"]:
                pairs["terminal_or_short_excluded"] += 1
                continue
            actual = left["real_discounted_raw_return"] - right["real_discounted_raw_return"]
            predicted = left["predicted_reward_return"] - right["predicted_reward_return"]
            if abs(actual) <= TIE_TOLERANCE:
                pairs["real_tie"] += 1
            elif abs(predicted) <= TIE_TOLERANCE:
                pairs["predicted_tie"] += 1
            else:
                pairs["concordant" if (actual > 0) == (predicted > 0) else "discordant"] += 1
    informative = pairs["concordant"] + pairs["discordant"] + pairs["predicted_tie"]
    return {"pairs": pairs, "informative_real_pairs": informative,
            "concordance_on_informative_real_pairs": pairs["concordant"] / informative if informative else None,
            "full_h5_candidates": sum(r["full_h5"] for r in rows),
            "excluded_candidates_by_reason": {reason: sum(r["exclusion_reason"] == reason for r in rows)
                                              for reason in ("terminated_before_fifth", "finished_before_fifth",
                                                              "timeout_before_fifth", "short_without_boundary")}}


def matched_action_quality(rows: list[dict]) -> dict:
    """Compare H3/H5 choices against SAME five-action real H5 outcomes."""
    eligible = [row for row in rows if row["ranking_eligible"]]
    if len(rows) != 5 or any(row.get("real_h3_prefix_raw_return") is None for row in eligible):
        raise ValueError("matched action quality requires five candidates and full H3 prefixes")
    if not eligible:
        return {"eligible_candidates": 0, "rankings": {}, "selected": {}}
    scores = {"h3_reward": "h3_predicted_reward_return",
              "h5_reward": "predicted_reward_return",
              "h3_planner": "h3_predicted_planner_score",
              "h5_planner": "h5_predicted_planner_score"}
    actual_h5 = "real_discounted_raw_return"
    rankings = {}
    for label, predicted in scores.items():
        outcomes = {"real_h5": actual_h5}
        if label == "h3_reward":
            outcomes["real_h3_prefix"] = "real_h3_prefix_raw_return"
        for target_label, actual in outcomes.items():
            counts = {"concordant": 0, "discordant": 0, "real_tie": 0, "predicted_tie": 0}
            for index, left in enumerate(eligible):
                for right in eligible[index + 1:]:
                    difference = left[actual] - right[actual]
                    guess = left[predicted] - right[predicted]
                    if not math.isfinite(difference) or not math.isfinite(guess):
                        raise ValueError("nonfinite matched action quality")
                    if abs(difference) <= TIE_TOLERANCE:
                        counts["real_tie"] += 1
                    elif abs(guess) <= TIE_TOLERANCE:
                        counts["predicted_tie"] += 1
                    else:
                        counts["concordant" if (difference > 0) == (guess > 0) else "discordant"] += 1
            informative = counts["concordant"] + counts["discordant"] + counts["predicted_tie"]
            rankings[f"{label}_vs_{target_label}"] = {
                "pairs": counts, "informative_real_pairs": informative,
                "concordance": counts["concordant"] / informative if informative else None}
    best = max(row[actual_h5] for row in eligible)
    selected = {}
    for label, predicted in scores.items():
        # Stable input order resolves near-ties without looking at actual outcomes.
        choice = max(eligible, key=lambda row: row[predicted])
        selected[label] = {"candidate": choice["candidate"],
                           "real_h5_return": choice[actual_h5],
                           "real_h5_regret": best - choice[actual_h5],
                           "best_real_h5_tie": best - choice[actual_h5] <= TIE_TOLERANCE}
    return {"eligible_candidates": len(eligible), "rankings": rankings, "selected": selected,
            "best_real_h5_return": best}


def summarize(entries: list[dict]) -> dict:
    if len(entries) != len(ANCHORS):
        raise ValueError("cannot rank incomplete fixed H5 anchor set")
    by_road = {}
    for seed in ROADS:
        rankings = [row["ranking"] for row in entries if row["geometry_seed"] == seed]
        counts = {key: sum(r["pairs"][key] for r in rankings) for key in rankings[0]["pairs"]}
        by_road[str(seed)] = {"anchors": len(rankings), "pairs": counts,
                              "informative_real_pairs": sum(r["informative_real_pairs"] for r in rankings)}
    counts = {key: sum(road["pairs"][key] for road in by_road.values())
              for key in next(iter(by_road.values()))["pairs"]}
    informative = counts["concordant"] + counts["discordant"] + counts["predicted_tie"]
    roads = sum(row["informative_real_pairs"] > 0 for row in by_road.values())
    reasons = {reason: sum(entry["ranking"]["excluded_candidates_by_reason"][reason]
                           for entry in entries)
               for reason in ("terminated_before_fifth", "finished_before_fifth",
                              "timeout_before_fifth", "short_without_boundary")}
    result = {"by_road": by_road, "pairs": counts, "informative_real_pairs": informative,
            "full_h5_candidates": sum(entry["ranking"]["full_h5_candidates"] for entry in entries),
            "excluded_candidates_by_reason": reasons,
            "informative_roads": roads, "concordance_on_informative_real_pairs": (
                counts["concordant"] / informative if informative else None),
            "quality_gate_passed": (informative >= GATE["min_informative_real_pairs"]
                                    and roads >= GATE["min_informative_roads"]
                                    and counts["concordant"] / informative > GATE["min_concordance"])
            if informative else False}
    if all("matched_action_quality" in entry for entry in entries):
        qualities = [entry["matched_action_quality"] for entry in entries]
        result["matched_action_quality"] = {
            "eligible_anchors": sum(q["eligible_candidates"] > 0 for q in qualities),
            "rankings": {name: {
                "pairs": {key: sum(q["rankings"].get(name, {}).get("pairs", {}).get(key, 0) for q in qualities)
                          for key in ("concordant", "discordant", "real_tie", "predicted_tie")}}
                for name in ("h3_reward_vs_real_h5", "h5_reward_vs_real_h5",
                             "h3_reward_vs_real_h3_prefix", "h3_planner_vs_real_h5", "h5_planner_vs_real_h5")},
            "selected": {name: {
                "best_real_h5_ties": sum(q["selected"].get(name, {}).get("best_real_h5_tie", False) for q in qualities),
                "mean_real_h5_regret": (sum(q["selected"][name]["real_h5_regret"]
                                             for q in qualities if name in q["selected"])
                                        / sum(name in q["selected"] for q in qualities)
                                        if any(name in q["selected"] for q in qualities) else None)}
                for name in ("h3_reward", "h5_reward", "h3_planner", "h5_planner")}}
        for value in result["matched_action_quality"]["rankings"].values():
            pairs = value["pairs"]
            count = sum(pairs[k] for k in ("concordant", "discordant", "predicted_tie"))
            value["informative_real_pairs"] = count
            value["concordance"] = pairs["concordant"] / count if count else None
    return result


def _save(path: Path, report: dict) -> None:
    body = {key: value for key, value in report.items() if key != "body_sha256"}
    report["body_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _recheck(root: Path, bundle: dict, *, artifacts: bool = False) -> None:
    spec = bundle["spec"]
    pinned(root, PROTOCOL, bundle["protocol_sha256"])
    for value in spec["logged_quality"].values():
        pinned(root, value["path"], value["sha256"])
    for name, sha in spec["source_sha256"].items():
        pinned(root, name, sha)
    if artifacts:
        for value in spec["source"].values():
            pinned(root, value["path"], value["sha256"])
    if runtime_identity() != spec["runtime"]:
        raise ValueError("runtime changed before H5 branch reset")


def run(root: Path, protocol_sha256: str, *, execute: bool = False,
        output: Path | None = None) -> dict:
    bundle = preflight(root, protocol_sha256)
    root = Path(root).resolve(strict=True)
    spec = bundle["spec"]
    report = {"format": "haic-tdmpc2-raw100k-h5-branches-result-v1", "status": "preflight_only",
              "protocol_sha256": protocol_sha256, "source": spec["source"],
              "logged_quality": spec["logged_quality"], "logged_quality_metrics": bundle["quality"],
              "source_sha256": spec["source_sha256"], "runtime": spec["runtime"],
              "historical_hidden_box2d_state_proven": False, "fresh_or_official_score": False,
              "scope": "four_reused_consumed_TRAIN_roads_same_reconstructed_accessible_state_only",
              "same_state_counterfactual_ranking": False, "exact_resume_supported": False,
              "environment_resets_attempted": 0, "maximum_resets": len(ANCHORS) * 6,
              "horizon": HORIZON, "discount": DISCOUNT, "tie_tolerance": TIE_TOLERANCE,
              "anchors": []}
    if not execute:
        if output is not None:
            raise ValueError("preflight must not create a result receipt")
        return report
    if (output is None or not output.is_absolute() or output.is_symlink()
            or output.parent != root / "runs" or output.parent.is_symlink()
            or not output.parent.is_dir() or output.suffix != ".json" or output.exists()
            or not output.name.startswith("tdmpc2-raw100k-h5-branches-")):
        raise ValueError("execution requires an unused absolute runs/tdmpc2-raw100k-h5-branches-*.json")
    report["status"] = "binding"
    with output.open("x", encoding="utf-8") as stream:
        report["body_sha256"] = hashlib.sha256(json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        anchors, model = load_bound_model(bundle, root)  # All 100354 replay steps before FIRST reset.
        report["replay_bound_before_first_reset"] = True
        for anchor in anchors:
            candidates = candidate_actions(anchor)
            report["anchors"].append({
                "episode_id": anchor.episode_id, "track_id": 1, "geometry_seed": anchor.seed,
                "start_step": anchor.step,
                "anchor_model_observation_hex": anchor.observations[-1].tobytes().hex(),
                "anchor_model_observation_sha256": hashlib.sha256(anchor.observations[-1].tobytes()).hexdigest(),
                "candidate_action_bytes_hex": {name: value.tobytes().hex() for name, value in candidates.items()},
                "candidates": [],
            })
        _save(output, report)  # All 12 pixel/action manifests precede the FIRST capture reset.
        # The step ledger cannot change after the complete replay/ledger byte comparison.
        pinned(root, f"{RUN}/steps.jsonl", STEPS_SHA)
        pinned(root, f"{RUN}/training.jsonl", TRAIN_SHA)
        from train import build_env

        report["status"] = "running"
        for index, anchor in enumerate(anchors):
            candidates = candidate_actions(anchor)
            entry = report["anchors"][index]
            _recheck(root, bundle)
            if report["environment_resets_attempted"] >= spec["max_resets"]:
                raise ValueError("frozen H5 reset cap exceeded")
            env = build_env(1, anchor.seed, 2000, 4, reward_shaping=False, obstacles=True)
            try:
                report["environment_resets_attempted"] += 1
                _save(output, report)
                captured = capture_bound_prefix(env, anchor)
            finally:
                env.close()
            entry["road_sha256"] = captured.road_sha256
            entry["anchor_accessible_state_sha256"] = hashlib.sha256(
                repr(captured.steps[-1].state).encode()).hexdigest()
            entry["anchor_observation_sha256"] = captured.steps[-1].observation_sha256
            _save(output, report)
            predicted = predicted_returns(model, anchor.observations[-1], candidates,
                                           spec["augmentation_seed"] + anchor.episode_id * 1000 + anchor.step)
            matched = matched_model_scores(model, anchor.observations[-1], candidates,
                                           spec["augmentation_seed"] + anchor.episode_id * 1000 + anchor.step)
            if any(not math.isclose(predicted[name], matched[name]["h5_reward"], rel_tol=0, abs_tol=1e-5)
                   for name in candidates):
                raise ValueError("matched H5 reward prediction differs from original fixed gate")
            for name, actions in candidates.items():
                _recheck(root, bundle)
                if report["environment_resets_attempted"] >= spec["max_resets"]:
                    raise ValueError("frozen H5 reset cap exceeded")
                env = build_env(1, anchor.seed, 2000, 4, reward_shaping=False, obstacles=True)
                try:
                    report["environment_resets_attempted"] += 1
                    _save(output, report)
                    replayed = parity._compare_replay_prefix(env, captured)
                    _pixel(replayed, anchor.observations[-1], "branch_anchor")
                    row = execute_suffix(env, anchor, name, actions, captured.road_sha256)
                finally:
                    env.close()
                row["predicted_reward_return"] = predicted[name]
                row["h3_predicted_reward_return"] = matched[name]["h3_reward"]
                row["h3_predicted_planner_score"] = matched[name]["h3_planner"]
                row["h5_predicted_planner_score"] = matched[name]["h5_planner"]
                entry["candidates"].append(row)
                _save(output, report)
            entry["ranking"] = rank(entry["candidates"])
            entry["matched_action_quality"] = matched_action_quality(entry["candidates"])
            _save(output, report)
        _recheck(root, bundle, artifacts=True)
        report["summary"] = summarize(report["anchors"])
        report["same_state_counterfactual_ranking"] = True
        report["status"] = "complete" if report["summary"]["quality_gate_passed"] else "quality_gate_failed"
    except BaseException as exc:
        report["status"] = "stopped_binder_parity_or_execution_failure"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _save(output, report)
        raise
    _save(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="default: source/ledger/quality checks only")
    mode.add_argument("--execute", action="store_true", help="explicit reset opt-in after frozen quality gate")
    mode.add_argument("--print-runtime", action="store_true", help="runtime pins without files, load, or resets")
    parser.add_argument("--output", type=Path, help="new exclusive direct runs/ failure/result receipt")
    args = parser.parse_args()
    if args.print_runtime:
        if args.output is not None or args.protocol_sha256 is not None:
            parser.error("--print-runtime takes no protocol or output")
        print(json.dumps(runtime_identity(), sort_keys=True))
        return
    if args.protocol_sha256 is None:
        parser.error("--protocol-sha256 is required for preflight or execution")
    print(json.dumps(run(ROOT, args.protocol_sha256, execute=args.execute, output=args.output),
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
