"""Zero-reset reward scoring on the original 12 consumed-TRAIN H5 branches.

The score protocol must be frozen separately AFTER the overshoot trainer has
completed its first whole-episode >=100k target. Neither mode accepts a partial
checkpoint, constructs an environment, or changes model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from haic.algorithms.tdmpc2.haic_env import environment_action
from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel, two_hot_inv
from scripts import diagnose_tdmpc2_h5_branches as branch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/tdmpc2-overshoot-old-branch-score-v1.json"
OUTPUT = "runs/tdmpc2-overshoot-old-branch-score-20260929-v1.json"
TRAIN_PROTOCOL = "experiments/tdmpc2-reward-overshoot-train-v1.json"
TRAIN_PROTOCOL_SHA = "ecceda92de76c2decdde13bc6661c706adfda22b8e8067e84ebd8cd2697a692b"
RUN = "runs/tdmpc2-overshoot-20260929-v1"
OLD_PRIMARY = "runs/tdmpc2-raw100k-h5-branches-20260929-v1.json"
OLD_PRIMARY_SHA = "23f7e6898b362cd789b92d6a1f2f184e1de551596458ae08117058570f1bcefd"
OLD_PROTOCOL = "experiments/tdmpc2-h5-branches-v1.json"
OLD_PROTOCOL_SHA = "a29cbcd2e59687201a7b0fba68e119f596e88633e06a51300639bf78461b33c9"
OLD_OPERATOR = "scripts/diagnose_tdmpc2_h5_branches.py"
OLD_OPERATOR_SHA = "85742b0623c25e6ddb8c2291b274b012c21a5002082dc0c60e424dbcea2f3e88"
OLD_SUMMARY = "experiments/tdmpc2-h5-branches-v1-result.json"
OLD_SUMMARY_SHA = "0efcceb8d09ea4dac72f350d400ac2f2ad4d193e300f483eb712f65cd05c2913"
OLD_CHECKPOINT = branch.CHECKPOINT
OLD_CHECKPOINT_SHA = branch.CHECKPOINT_SHA
FORMAT = "haic-tdmpc2-overshoot-old-branch-score-v1"
RESULT_FORMAT = "haic-tdmpc2-overshoot-old-branch-score-result-v1"
OLD_ROAD_COUNTS = dict(zip(branch.ROADS, ((13, 23), (16, 22), (12, 23), (15, 23))))
ERROR_LIMIT = 0.39667698614253405
SELF = "scripts/score_tdmpc2_overshoot_archived_branches.py"
TEST = "tests/test_score_tdmpc2_overshoot_archived_branches.py"


def _ref(path: str, sha: str) -> dict:
    return {"path": path, "sha256": sha}


def _file(root: Path, ref: dict, expected: str | None = None) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("incomplete score artifact reference")
    if expected is not None and ref["path"] != expected:
        raise ValueError("score artifact path differs from frozen contract")
    return branch.pinned(root, ref["path"], ref["sha256"])


def _body_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _check_body(value: dict) -> None:
    if value.get("body_sha256") != _body_sha({k: v for k, v in value.items() if k != "body_sha256"}):
        raise ValueError("archived branch body SHA differs")


def _finished_result(root: Path, spec: dict, training: dict) -> tuple[dict, dict]:
    """Do not even open the active ledgers until the final result AND model exist."""
    refs = spec["training_source"]
    result = branch._json(_file(root, refs["result"], f"{RUN}/result.json").read_bytes())
    if (result.get("format") != training["format"] or result.get("status") != "completed_boundary_at_least_100k"
            or result.get("protocol_sha256") != TRAIN_PROTOCOL_SHA
            or result.get("source_sha256") != training["source_sha256"]
            or result.get("reward_overshoot") != training["reward_overshoot"]
            or result.get("runtime") != training["runtime"]
            or result.get("seed_schedule") != training["seed_schedule"]
            or any(result.get(key) != training[key] for key in (
                "baseline_training_result", "replay_audit", "throughput_benchmark"))
            or result.get("pretrain_updates") != 10000 or result.get("action_dim") != 3
            or result.get("reused_train_only") is not True or result.get("resume_supported") is not False
            or result.get("evaluation") is not None or type(result.get("decisions")) is not int
            or not 100000 <= result["decisions"] <= 102000
            or type(result.get("updates")) is not int or result["updates"] != result["decisions"]
            or type(result.get("episodes")) is not int
            or result["episodes"] < 4 or result.get("training_ledger_sha256") != refs["training_ledger"]["sha256"]
            or result.get("step_ledger_sha256") != refs["step_ledger"]["sha256"]):
        raise ValueError("first >=100k overshoot TRAIN result is not complete and bound")
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 4:
        raise ValueError("four completed first-crossing checkpoints required")
    last = checkpoints[-1]
    if (not isinstance(last, dict) or last.get("target") != 100000
            or last.get("decisions") != result["decisions"] or last.get("updates") != result["updates"]
            or last.get("episodes") != result["episodes"] or last.get("sha256") != refs["checkpoint"]["sha256"]
            or type(last.get("decisions")) is not int
            or last.get("path") != f"checkpoint-at-least-100000-step-{last['decisions']:06d}.pt"
            or refs["checkpoint"]["path"] != f"{RUN}/{last['path']}"):
        raise ValueError("selected checkpoint is not first completed >=100k boundary")
    _file(root, refs["checkpoint"], f"{RUN}/{last['path']}")
    return result, last


def _ledgers(root: Path, spec: dict, training: dict, result: dict) -> list[dict]:
    """Audit the COMPLETE frozen ledger, each first-crossing row and every step."""
    refs = spec["training_source"]
    ledger = _file(root, refs["training_ledger"], f"{RUN}/training.jsonl")
    steps = _file(root, refs["step_ledger"], f"{RUN}/steps.jsonl")
    episodes: list[dict] = []
    digest = hashlib.sha256()
    step_digest = hashlib.sha256()
    decisions = updates = checkpoint_index = previous_episodes = previous_updates = 0
    targets = training["checkpoint_targets"]
    phase = "start"
    with ledger.open("rb") as rows, steps.open("rb") as step_rows:
        for line in rows:
            if not line.endswith(b"\n"):
                raise ValueError("truncated training ledger")
            row = branch._json(line)
            before = digest.hexdigest()
            digest.update(line)
            event = row.get("event")
            if phase == "start":
                if (event != "start" or row.get("protocol_sha256") != TRAIN_PROTOCOL_SHA
                        or row.get("source_sha256") != training["source_sha256"]
                        or row.get("reward_overshoot") != training["reward_overshoot"]
                        or row.get("runtime") != training["runtime"]
                        or row.get("resume_supported") is not False):
                    raise ValueError("training ledger start is not frozen")
                phase = "intent"
            elif phase == "checkpoint":
                if (event != "checkpoint" or checkpoint_index >= len(targets)
                        or row != {"event": "checkpoint", **result["checkpoints"][checkpoint_index]}
                        or row.get("target") != targets[checkpoint_index]
                        or row.get("path") != f"checkpoint-at-least-{targets[checkpoint_index]:06d}-step-{decisions:06d}.pt"
                        or row.get("decisions") != decisions or row.get("updates") != updates
                        or row.get("episodes") != len(episodes)
                        or row.get("rolling_update_count") != updates - previous_updates
                        or row.get("training_ledger_sha256_before_checkpoint") != before
                        or row.get("step_ledger_sha256") != step_digest.hexdigest()
                        or row.get("train_episodes_since_previous") != [
                            {key: ep[key] for key in ("episode", "track_id", "geometry_seed", "decisions",
                                                 "length", "return", "progress", "damage", "finished")}
                            for ep in episodes[previous_episodes:]]):
                    raise ValueError("checkpoint first-crossing ledger/prefix/probe row differs")
                _file(root, _ref(f"{RUN}/{row['path']}", row["sha256"]))
                previous_episodes, previous_updates = len(episodes), updates
                checkpoint_index += 1
                phase = "intent"
            elif phase in ("intent", "reset"):
                if event != ("reset_intent" if phase == "intent" else "reset") or row != {
                    "event": event, "episode": len(episodes), "decisions": decisions,
                    **training["cells"][len(episodes) % 4],
                }:
                    raise ValueError("TRAIN reset intent/cell sequence differs")
                phase = "reset" if phase == "intent" else "episode"
            elif phase == "episode":
                length = row.get("length")
                cell = training["cells"][len(episodes) % 4]
                if (event != "episode" or type(length) is not int
                        or not 1 <= length <= training["training"]["max_steps"]
                        or type(row.get("updates")) is not int
                        or any(row.get(k) != v for k, v in {
                            "episode": len(episodes), "decisions": decisions + length, **cell,
                        }.items())
                        or row["updates"] != max(0, decisions + length - 10000) + (
                            10000 if decisions + length >= 10000 else 0)
                        or any(type(row.get(key)) is not bool for key in (
                            "terminated", "truncated", "terminal", "finished"))
                        or not (row["terminated"] or row["truncated"])
                        or row["terminal"] != (row["terminated"] or row["finished"])
                        or row["finished"] and not row["truncated"]
                        or row.get("finish_time_s") is not None and not row["finished"]
                        or row["finished"] and (type(row.get("finish_time_s")) not in (int, float)
                                                or not math.isfinite(row["finish_time_s"]))
                        or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                               for key in ("progress", "damage"))):
                    raise ValueError("TRAIN episode boundary/updates differ")
                actions, native, total = hashlib.sha256(), hashlib.sha256(), 0.0
                item: dict | None = None
                for offset in range(length):
                    step = step_rows.readline()
                    if not step.endswith(b"\n"):
                        raise ValueError("truncated step ledger")
                    step_digest.update(step)
                    item = branch._json(step)
                    try:
                        action = bytes.fromhex(item["action_f32_hex"])
                        applied = bytes.fromhex(item["native_action_f32_hex"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError("invalid step action bytes") from exc
                    if (len(action) != 12 or len(applied) != 12
                            or not np.isfinite(np.frombuffer(action, dtype=np.float32)).all()
                            or np.any(np.abs(np.frombuffer(action, dtype=np.float32)) > 1)
                            or environment_action(np.frombuffer(action, dtype=np.float32)).tobytes() != applied
                            or any(item.get(k) != v for k, v in {
                                "episode": len(episodes), "decision": decisions + offset + 1, **cell,
                            }.items()) or type(item.get("reward")) not in (int, float)
                            or not math.isfinite(item["reward"])
                            or any(type(item.get(k)) is not bool for k in ("terminated", "truncated", "terminal"))
                            or offset < length - 1 and any(item[k] for k in ("terminated", "truncated", "terminal"))
                            or item["terminated"] and not item["terminal"]
                            or item["terminal"] and not (item["terminated"] or item["truncated"])):
                        raise ValueError("TRAIN step action/reward/terminal lineage differs")
                    actions.update(action)
                    native.update(applied)
                    total += item["reward"]
                if (item is None or any(item[k] != row[k] for k in ("terminated", "truncated", "terminal"))
                        or actions.hexdigest() != row.get("action_trace_sha256")
                        or native.hexdigest() != row.get("native_action_trace_sha256")
                        or type(row.get("return")) not in (float, int) or not math.isfinite(row["return"])
                        or not math.isclose(total, row["return"], rel_tol=0, abs_tol=1e-4)):
                    raise ValueError("TRAIN episode/step ledger lineage differs")
                decisions += length
                updates = row["updates"]
                episodes.append(row)
                phase = "checkpoint" if checkpoint_index < len(targets) and decisions >= targets[checkpoint_index] else "intent"
            else:
                raise ValueError("unexpected TRAIN ledger event")
        if (phase != "intent" or step_rows.read(1) or digest.hexdigest() != refs["training_ledger"]["sha256"]
                or step_digest.hexdigest() != refs["step_ledger"]["sha256"]
                or (decisions, updates, len(episodes), checkpoint_index) != (
                    result["decisions"], result["updates"], result["episodes"], 4)):
            raise ValueError("truncated/extra TRAIN ledger or missing final checkpoint")
    return episodes


def _archived(root: Path, spec: dict) -> tuple[dict, dict]:
    refs = spec["original_branch"]
    old_spec = branch._json(_file(root, refs["protocol"], OLD_PROTOCOL).read_bytes())
    summary = branch._json(_file(root, refs["summary"], OLD_SUMMARY).read_bytes())
    primary = branch._json(_file(root, refs["primary"], OLD_PRIMARY).read_bytes())
    _check_body(primary)
    if (old_spec.get("source", {}).get("checkpoint") != refs["model"]
            or old_spec.get("source_sha256", {}).get(OLD_OPERATOR) != OLD_OPERATOR_SHA
            or summary.get("primary_receipt_sha256") != OLD_PRIMARY_SHA
            or summary.get("source_model_sha256") != OLD_CHECKPOINT_SHA
            or summary.get("operator_sha256") != OLD_OPERATOR_SHA
            or summary.get("protocol_sha256") != OLD_PROTOCOL_SHA
            or primary.get("format") != "haic-tdmpc2-raw100k-h5-branches-result-v1"
            or primary.get("status") != "complete" or primary.get("protocol_sha256") != OLD_PROTOCOL_SHA
            or primary.get("source") != old_spec["source"]
            or primary.get("source_sha256") != old_spec["source_sha256"]
            or primary.get("runtime") != old_spec["runtime"]
            or primary.get("environment_resets_attempted") != 72
            or primary.get("replay_bound_before_first_reset") is not True
            or primary.get("same_state_counterfactual_ranking") is not True
            or primary.get("fresh_or_official_score") is not False
            or primary.get("discount") != branch.DISCOUNT or primary.get("horizon") != 5
            or primary.get("tie_tolerance") != branch.TIE_TOLERANCE
            or not isinstance(primary.get("anchors"), list) or len(primary["anchors"]) != 12):
        raise ValueError("original five-action branch provenance differs")
    _file(root, refs["operator"], OLD_OPERATOR)
    _file(root, refs["model"], OLD_CHECKPOINT)
    return primary, summary


def preflight(root: Path, protocol_sha256: str) -> dict:
    """No torch.load, environment construction, optimizer operation or output."""
    root = Path(root).resolve(strict=True)
    spec = branch._json(branch.pinned(root, PROTOCOL, branch._sha(protocol_sha256)).read_bytes())
    old = {"primary": _ref(OLD_PRIMARY, OLD_PRIMARY_SHA), "protocol": _ref(OLD_PROTOCOL, OLD_PROTOCOL_SHA),
           "operator": _ref(OLD_OPERATOR, OLD_OPERATOR_SHA), "model": _ref(OLD_CHECKPOINT, OLD_CHECKPOINT_SHA),
           "summary": _ref(OLD_SUMMARY, OLD_SUMMARY_SHA)}
    refs = spec.get("training_source")
    if (not isinstance(spec, dict) or set(spec) != {"format", "purpose", "training_source",
            "original_branch", "source_sha256", "runtime", "output", "gate"}
            or spec["format"] != FORMAT or spec["purpose"] != "consumed-TRAIN-archived-branch-reward-only"
            or spec["output"] != OUTPUT or spec["original_branch"] != old
            or not isinstance(refs, dict) or set(refs) != {
                "protocol", "result", "checkpoint", "training_ledger", "step_ledger"}
            or refs["protocol"] != _ref(TRAIN_PROTOCOL, TRAIN_PROTOCOL_SHA)
            or spec["gate"] != {"min_h5_concordant": 66, "h5_informative_pairs": 91,
                                    "min_nonregressing_roads": 3, "min_h3_concordant": 31,
                                    "h3_informative_pairs": 40, "h5_error_exclusive": ERROR_LIMIT}):
        raise ValueError("score protocol must be separately frozen to exact sources and four gates")
    training = branch._json(_file(root, refs["protocol"], TRAIN_PROTOCOL).read_bytes())
    old_spec = branch._json(_file(root, old["protocol"], OLD_PROTOCOL).read_bytes())
    expected_sources = {**training["source_sha256"], **old_spec["source_sha256"]}
    if (training.get("format") != "haic-tdmpc2-reward-overshoot-train-v1"
            or training.get("run_dir") != RUN or training.get("checkpoint_targets") != [20000, 40000, 70000, 100000]
            or training.get("cells") != [{"track_id": 1, "geometry_seed": seed} for seed in branch.ROADS]
            or training.get("episode_schedule") != [0, 1, 2, 3]
            or training.get("training", {}).get("horizon") != 3
            or training["training"].get("discount") != branch.DISCOUNT
            or old_spec.get("augmentation_seed") != 834
            or old_spec.get("anchors") != [{"episode_id": ep, "start_step": step} for ep, step in branch.ANCHORS]
            or old_spec.get("candidates") != {"logged": None, **branch.FIXED_SUFFIXES}
            or not isinstance(spec["source_sha256"], dict)
            or set(spec["source_sha256"]) != set(expected_sources) | {SELF, TEST}
            or any(spec["source_sha256"].get(name) != sha for name, sha in expected_sources.items())):
        raise ValueError("score protocol source/anchor/trainer contract differs")
    # Source and runtime checks precede checkpoint deserialization, including the evaluator itself.
    for name, sha in spec["source_sha256"].items():
        branch.pinned(root, name, sha)
    if branch.runtime_identity() != spec["runtime"] or spec["runtime"] != old_spec["runtime"]:
        raise ValueError("source-bound scorer runtime differs from original branch runtime")
    primary, old_summary = _archived(root, spec)
    # This gate is intentionally BEFORE *any* open/read of the in-progress TRAIN ledgers.
    result, last = _finished_result(root, spec, training)
    episodes = _ledgers(root, spec, training, result)
    return {"spec": spec, "training": training, "result": result, "last": last,
            "episodes": episodes, "primary": primary, "old_summary": old_summary,
            "protocol_sha256": protocol_sha256}


def _optimizer(state: object, updates: int) -> None:
    if (not isinstance(state, dict) or set(state) != {"state", "param_groups"}
            or not isinstance(state["state"], dict) or not state["state"]
            or not isinstance(state["param_groups"], list) or not state["param_groups"]):
        raise ValueError("checkpoint optimizer state missing")
    ids = [param for group in state["param_groups"] for param in group["params"]]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(state["state"]):
        raise ValueError("checkpoint optimizer parameter coverage differs")
    for record in state["state"].values():
        step = record.get("step") if isinstance(record, dict) else None
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or not math.isfinite(float(step.item())) or float(step.item()) != updates
                or not all(k in record for k in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("checkpoint optimizer update count/state differs")


def _overshoot_probe(state: dict, training: dict, last: dict) -> None:
    """Bind H3 plus the variant's masked step-4/5 tensors to saved replay."""
    from scripts import diagnose_tdmpc2_checkpoint_losses as probe_source

    probe = state.get("probe")
    extra = {"overshoot_action": (2, 256, 3), "overshoot_reward": (2, 256, 1),
             "overshoot_mask": (2, 256, 1)}
    base_keys = {"obs", "action", "reward", "terminated", "truncated", "terminal",
                 "bootstrap_mask", "episode_id", "start_step"}
    if not isinstance(probe, dict) or set(probe) != base_keys | set(extra):
        raise ValueError("overshoot checkpoint frozen probe keys differ")
    probe_source._probe({**state, "probe": {key: probe[key] for key in base_keys}}, training, last)
    for key, shape in extra.items():
        tensor = probe[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.requires_grad or tensor.dtype != torch.float32
                or tuple(tensor.shape) != shape or not torch.isfinite(tensor).all()):
            raise ValueError("overshoot checkpoint suffix probe tensor differs")
    action, reward, mask = (probe[name].numpy() for name in extra)
    if (np.any(np.abs(action) > 1) or np.any((mask != 0) & (mask != 1))):
        raise ValueError("overshoot checkpoint suffix action/mask invalid")
    replay = state["replay"]["episodes"]
    for column, (eid, start) in enumerate(zip(probe["episode_id"].tolist(), probe["start_step"].tolist())):
        episode = replay[eid]
        offset = start - episode["start_step"]
        for t in range(2):
            index = offset + 3 + t
            expected_mask = float(index < len(episode["actions"]))
            if expected_mask and (np.any(episode["terminated"][offset:index])
                                  or np.any(episode["truncated"][offset:index])):
                raise ValueError("overshoot checkpoint probe crosses episode boundary")
            expected_action = episode["actions"][index] if expected_mask else np.zeros(3, np.float32)
            expected_reward = episode["rewards"][index] if expected_mask else np.float32(0)
            if (mask[t, column, 0] != expected_mask
                    or not np.array_equal(action[t, column], expected_action)
                    or reward[t, column, 0] != expected_reward):
                raise ValueError("overshoot checkpoint suffix probe/replay lineage differs")


def _bind_checkpoint_replay(root: Path, state: dict, bundle: dict) -> None:
    """Check every saved episode against its already-hashed, complete step ledger."""
    stored = state["replay"]["episodes"]
    if not isinstance(stored, list) or len(stored) != len(bundle["episodes"]):
        raise ValueError("checkpoint replay complete episode count differs")
    with _file(root, bundle["spec"]["training_source"]["step_ledger"]).open("rb") as stream:
        for eid, ledger in enumerate(bundle["episodes"]):
            episode = stored[eid]
            length = ledger["length"]
            if not isinstance(episode, dict):
                raise ValueError("checkpoint replay episode metadata invalid")
            actions, rewards, observations = (episode.get(key) for key in ("actions", "rewards", "observations"))
            flags = [episode.get(key) for key in ("terminated", "truncated", "terminal")]
            if (episode.get("episode_id") != eid or episode.get("start_step") != 0
                    or not isinstance(actions, np.ndarray) or actions.dtype != np.float32
                    or actions.shape != (length, 3) or not np.isfinite(actions).all()
                    or np.any(np.abs(actions) > 1) or not isinstance(rewards, np.ndarray)
                    or rewards.dtype != np.float32 or rewards.shape != (length,) or not np.isfinite(rewards).all()
                    or not isinstance(observations, np.ndarray) or observations.dtype != np.uint8
                    or observations.shape != (length + 1, *branch.PIXELS)
                    or any(not isinstance(flag, np.ndarray) or flag.dtype != np.bool_
                           or flag.shape != (length,) for flag in flags)):
                raise ValueError("checkpoint replay episode boundary or pixel/action shape differs")
            actions = cast(np.ndarray, actions)
            rewards = cast(np.ndarray, rewards)
            flags = cast(list[np.ndarray], flags)
            if (any(bool(flag[-1]) != ledger[key] for flag, key in zip(
                    flags, ("terminated", "truncated", "terminal")))
                    or any(np.any(flag[:-1]) for flag in flags)):
                raise ValueError("checkpoint replay episode terminal boundary differs")
            action_hash, native_hash = hashlib.sha256(), hashlib.sha256()
            for offset, action in enumerate(actions):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("checkpoint replay step ledger truncated")
                row = branch._json(line)
                native = environment_action(action)
                if (row.get("decision") != ledger["decisions"] - length + offset + 1
                        or row.get("episode") != eid
                        or row.get("action_f32_hex") != action.tobytes().hex()
                        or row.get("native_action_f32_hex") != native.tobytes().hex()
                        or np.float32(row["reward"]) != rewards[offset]
                        or any(row.get(key) != bool(flag[offset]) for flag, key in zip(
                            flags, ("terminated", "truncated", "terminal")))):
                    raise ValueError("checkpoint replay action/reward/terminal differs from steps")
                action_hash.update(action.tobytes())
                native_hash.update(native.tobytes())
            if (action_hash.hexdigest() != ledger["action_trace_sha256"]
                    or native_hash.hexdigest() != ledger["native_action_trace_sha256"]):
                raise ValueError("checkpoint replay complete episode action trace differs")
        if stream.read(1):
            raise ValueError("checkpoint replay has fewer transitions than complete step ledger")


def _models(root: Path, bundle: dict) -> tuple[WorldModel, WorldModel]:
    # The protocol and ALL producer/dependency/artifact hashes are rechecked before pickle.
    preflight(root, bundle["protocol_sha256"])
    spec, last = bundle["spec"], bundle["last"]
    checkpoint = _file(root, spec["training_source"]["checkpoint"])
    with torch.random.fork_rng(devices=[]), checkpoint.open("rb") as stream:
        if branch.digest_stream(stream) != spec["training_source"]["checkpoint"]["sha256"]:
            raise ValueError("checkpoint changed before torch.load")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=False)
    training = bundle["training"]
    if (not isinstance(state, dict) or state.get("format") != training["format"]
            or state.get("protocol_sha256") != TRAIN_PROTOCOL_SHA
            or state.get("source_sha256") != training["source_sha256"]
            or state.get("reward_overshoot") != training["reward_overshoot"]
            or any(state.get(key) != training[key] for key in (
                "baseline_training_protocol", "baseline_training_result", "replay_audit", "throughput_benchmark"))
            or any(state.get(key) != last[key] for key in ("target", "decisions", "updates", "episodes"))
            or state.get("action_dim") != 3 or state.get("resume_supported") is not False
            or state.get("step_ledger_sha256") != last["step_ledger_sha256"]
            or state.get("training_ledger_sha256_before_checkpoint") != last[
                "training_ledger_sha256_before_checkpoint"]
            or not isinstance(state.get("rng"), dict) or not isinstance(state.get("learner"), dict)):
        raise ValueError("overshoot checkpoint episode/producer boundary differs")
    replay = state.get("replay")
    if (not isinstance(replay, dict) or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or replay.get("active") is not None or replay.get("size") != last["decisions"]
            or replay.get("next_episode_id") != last["episodes"]
            or not isinstance(replay.get("episodes"), list) or len(replay["episodes"]) != last["episodes"]
            or replay.get("horizon") != 3 or replay.get("action_dim") != 3
            or replay.get("capacity") != 120000 or replay.get("observation_shape") != (4, 64, 64)):
        raise ValueError("overshoot checkpoint replay episode boundary differs")
    _bind_checkpoint_replay(root, state, bundle)
    _overshoot_probe(state, training, last)
    _optimizer(state.get("optim"), last["updates"])
    _optimizer(state.get("pi_optim"), last["updates"])
    model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": branch.PIXELS}, episodic=True))
    if not isinstance(state["learner"].get("q_scale"), torch.Tensor):
        raise ValueError("checkpoint learner scale missing")
    weights = {key.removeprefix("model."): value for key, value in state["learner"].items()
               if key.startswith("model.")}
    model.load_state_dict(weights, strict=True)
    model.eval()
    # The old branch binder independently validates original checkpoint replay and every original step.
    original_bundle = branch.preflight(root, OLD_PROTOCOL_SHA)
    bound, original = branch.load_bound_model(original_bundle, root)
    if len(bound) != 12:
        raise ValueError("original branch checkpoint has incomplete anchors")
    for anchor, archived in zip(bound, bundle["primary"]["anchors"]):
        if (anchor.episode_id != archived.get("episode_id") or anchor.step != archived.get("start_step")
                or anchor.seed != archived.get("geometry_seed")
                or anchor.observations[-1].tobytes().hex() != archived.get("anchor_model_observation_hex")
                or branch.candidate_actions(anchor)["logged"].tobytes().hex() != archived.get(
                    "candidate_action_bytes_hex", {}).get("logged")):
            raise ValueError("original checkpoint pixel/action anchor differs from archived branch")
    return original, model


def _predict_both(model: Any, pixels: np.ndarray, candidates: dict[str, np.ndarray], seed: int) -> dict:
    """Original per-anchor crop/RNG, identical action order, decoded reward before latent next."""
    model.eval()
    predicted = {}
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        latent = model.encode(torch.from_numpy(pixels.copy())[None], None)
        if not isinstance(latent, torch.Tensor) or latent.ndim != 2 or latent.shape[0] != 1 or not torch.isfinite(latent).all():
            raise ValueError("nonfinite anchor encoding")
        for name, actions in candidates.items():
            z, rewards = latent, []
            for index, action in enumerate(actions):
                a = torch.from_numpy(action.copy())[None]
                reward = two_hot_inv(model.reward(z, a, None), model.cfg)
                if reward.shape != (1, 1) or not torch.isfinite(reward).all():
                    raise ValueError("nonfinite imagined reward")
                rewards.append(float(reward.item()))
                if index < 4:
                    z = model.next(z, a, None)
                    if z.shape != latent.shape or not torch.isfinite(z).all():
                        raise ValueError("nonfinite imagined next latent")
            predicted[name] = {f"h{h}": math.fsum(branch.DISCOUNT ** t * v
                                                   for t, v in enumerate(rewards[:h])) for h in (3, 5)}
    return predicted


def _pairs(rows: list[dict], predicted: str, actual: str) -> dict:
    counts = {"concordant": 0, "discordant": 0, "real_tie": 0, "predicted_tie": 0}
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            real = left[actual] - right[actual]
            estimate = left[predicted] - right[predicted]
            if abs(real) <= branch.TIE_TOLERANCE:
                counts["real_tie"] += 1
            elif abs(estimate) <= branch.TIE_TOLERANCE:
                counts["predicted_tie"] += 1
            else:
                counts["concordant" if (real > 0) == (estimate > 0) else "discordant"] += 1
    return counts


def _aggregate(rows: list[dict], predicted: str, actual: str) -> dict:
    counts = {key: sum(_pairs(row["candidates"], predicted, actual)[key] for row in rows)
              for key in ("concordant", "discordant", "real_tie", "predicted_tie")}
    return {"pairs": counts, "informative_real_pairs": sum(counts[k] for k in (
        "concordant", "discordant", "predicted_tie"))}


def score_rows(primary: dict, old_summary: dict, original: Any, new: Any) -> tuple[list[dict], dict, dict]:
    """Require original-model parity on all 60 actions before assessing the new model."""
    if not isinstance(primary.get("anchors"), list) or len(primary["anchors"]) != len(branch.ANCHORS):
        raise ValueError("archived anchor set incomplete")
    entries = []
    prepared = []
    old_error = new_error = 0.0
    for (ep, step), archived in zip(branch.ANCHORS, primary["anchors"]):
        if (archived.get("episode_id"), archived.get("start_step"), archived.get("track_id"),
                archived.get("geometry_seed")) != (ep, step, 1, branch.ROADS[ep]):
            raise ValueError("archived anchor order/road differs")
        try:
            pixels = np.frombuffer(bytes.fromhex(archived["anchor_model_observation_hex"]), dtype=np.uint8)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid archived anchor pixels") from exc
        if (pixels.size != math.prod(branch.PIXELS)
                or hashlib.sha256(pixels.tobytes()).hexdigest() != archived.get("anchor_model_observation_sha256")):
            raise ValueError("archived pixel shape/SHA differs")
        pixels = pixels.reshape(branch.PIXELS)
        originals = archived.get("candidates")
        if (not isinstance(originals, list) or [r.get("candidate") for r in originals] != [
                "logged", "coast", "gas", "brake", "left_gas"]):
            raise ValueError("archived five-action candidate order differs")
        candidates = {}
        for row in originals:
            name = row["candidate"]
            value = row.get("model_action_bytes_hex")
            if (not isinstance(value, str) or len(value) != 120 or value != archived.get(
                    "candidate_action_bytes_hex", {}).get(name)):
                raise ValueError("archived candidate action byte manifest differs")
            try:
                action = np.frombuffer(bytes.fromhex(value), dtype=np.float32).reshape(5, 3).copy()
            except ValueError as exc:
                raise ValueError("invalid archived action bytes") from exc
            if (not np.isfinite(action).all() or np.any(np.abs(action) > 1)
                    or name != "logged" and action.tobytes() != np.asarray(
                        branch.FIXED_SUFFIXES[name], np.float32).tobytes()):
                raise ValueError("archived fixed action bytes differ")
            candidates[name] = action
        seed = 834 + ep * 1000 + step
        old_prediction = branch.predicted_returns(original, pixels, candidates, seed)
        candidate_rows = []
        for row in originals:
            name = row["candidate"]
            rewards = row.get("raw_rewards")
            if (row.get("steps") != 5 or row.get("full_h5") is not True
                    or row.get("ranking_eligible") is not True or row.get("exclusion_reason") is not None
                    or row.get("ending") not in (None, "terminated", "finished", "timeout")
                    or not isinstance(rewards, list) or len(rewards) != 5
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in rewards)
                    or not math.isclose(old_prediction[name], row.get("predicted_reward_return", math.inf),
                                        abs_tol=1e-5, rel_tol=0)):
                raise ValueError("archived original H5 prediction/terminal parity failed")
            real3 = math.fsum(branch.DISCOUNT**t * v for t, v in enumerate(rewards[:3]))
            real5 = math.fsum(branch.DISCOUNT**t * v for t, v in enumerate(rewards))
            if (not math.isclose(real3, row.get("real_h3_prefix_raw_return", math.inf), abs_tol=1e-9, rel_tol=0)
                    or not math.isclose(real5, row.get("real_discounted_raw_return", math.inf), abs_tol=1e-9, rel_tol=0)):
                raise ValueError("archived real H3/H5 returns invalid")
            old_error += abs(row["predicted_reward_return"] - real5)
            candidate_rows.append({"candidate": name, "model_action_bytes_hex": row["model_action_bytes_hex"],
                                   "real_h5_raw_return": real5, "real_h3_prefix_raw_return": real3,
                                   "old_h5_reward_return": row["predicted_reward_return"],
                                   "ending": row["ending"], "steps": 5})
        entries.append({"episode_id": ep, "start_step": step, "geometry_seed": branch.ROADS[ep],
                        "anchor_model_observation_sha256": archived["anchor_model_observation_sha256"],
                        "candidates": candidate_rows})
        prepared.append((seed, pixels, candidates))
    denom = 60 * sum(branch.DISCOUNT**t for t in range(5))
    if (not math.isclose(old_error / denom, ERROR_LIMIT, rel_tol=0, abs_tol=1e-9)
            or not math.isclose(old_summary["same_candidate_rollout_error_diagnostic"][
                "h5_absolute_error_per_discounted_step"], old_error / denom, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("original normalized H5 error does not match archived summary")
    old_h5 = _aggregate(entries, "old_h5_reward_return", "real_h5_raw_return")
    if (old_h5["pairs"] != {"concordant": 56, "discordant": 35, "real_tie": 29, "predicted_tie": 0}
            or old_h5["informative_real_pairs"] != 91):
        raise ValueError("original archived H5 pair parity failed")
    for entry, (seed, pixels, candidates) in zip(entries, prepared):
        new_prediction = _predict_both(new, pixels, candidates, seed)
        if set(new_prediction) != set(candidates):
            raise ValueError("new model returned incomplete candidate predictions")
        for row in entry["candidates"]:
            name = row["candidate"]
            if (not isinstance(new_prediction[name], dict)
                    or set(new_prediction[name]) != {"h3", "h5"}
                    or any(type(new_prediction[name][key]) not in (int, float)
                           or not math.isfinite(new_prediction[name][key]) for key in ("h3", "h5"))):
                raise ValueError("new model returned invalid H3/H5 reward")
            row["new_h3_reward_return"] = new_prediction[name]["h3"]
            row["new_h5_reward_return"] = new_prediction[name]["h5"]
            new_error += abs(row["new_h5_reward_return"] - row["real_h5_raw_return"])
    h5 = _aggregate(entries, "new_h5_reward_return", "real_h5_raw_return")
    h3 = _aggregate(entries, "new_h3_reward_return", "real_h3_prefix_raw_return")
    if (h5["informative_real_pairs"] != 91 or h5["pairs"]["real_tie"] != 29
            or h3["informative_real_pairs"] != 40 or h3["pairs"]["real_tie"] != 80):
        raise ValueError("archived real H5/H3 pair denominators differ")
    by_road = {}
    road_gate = {}
    for seed, (old_concordant, denominator) in OLD_ROAD_COUNTS.items():
        road = _aggregate([entry for entry in entries if entry["geometry_seed"] == seed],
                          "new_h5_reward_return", "real_h5_raw_return")
        old_road = _aggregate([entry for entry in entries if entry["geometry_seed"] == seed],
                              "old_h5_reward_return", "real_h5_raw_return")
        if (old_road["pairs"]["concordant"] != old_concordant
                or old_road["informative_real_pairs"] != denominator
                or road["informative_real_pairs"] != denominator):
            raise ValueError("archived old per-road comparison differs")
        by_road[str(seed)] = {"anchors": 3, **road}
        road_gate[str(seed)] = {"new_concordant": road["pairs"]["concordant"],
                                "old_concordant": old_concordant, "informative_real_pairs": denominator,
                                "passed": road["pairs"]["concordant"] >= old_concordant}
    summary = {"h5": {**h5, "by_road": by_road}, "h3": h3,
               "h5_absolute_error_per_discounted_step": new_error / denom}
    gate: dict = {
        "h5_reward_concordance": {"concordant": h5["pairs"]["concordant"],
                                  "informative_real_pairs": 91, "minimum_concordant": 66,
                                  "passed": h5["pairs"]["concordant"] >= 66},
        "road_nonregression": {"nonregressing_roads": sum(v["passed"] for v in road_gate.values()),
                               "minimum_roads": 3, "by_road": road_gate,
                               "passed": sum(v["passed"] for v in road_gate.values()) >= 3},
        "h3_prefix_concordance": {"concordant": h3["pairs"]["concordant"],
                                  "informative_real_pairs": 40, "minimum_concordant": 31,
                                  "passed": h3["pairs"]["concordant"] >= 31},
        "h5_normalized_error": {"value": new_error / denom, "threshold_exclusive": ERROR_LIMIT,
                                "passed": new_error / denom < ERROR_LIMIT},
    }
    gate["passed"] = all(v["passed"] for v in gate.values())
    return entries, summary, gate


def _write_exclusive(root: Path, value: dict) -> None:
    path = root / OUTPUT
    if path.parent.is_symlink() or not path.parent.is_dir() or path.exists() or path.is_symlink():
        raise ValueError("score output must be absent in real runs directory")
    value["body_sha256"] = _body_sha(value)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o644), "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(root: Path, protocol_sha256: str, *, score: bool = False) -> dict:
    root = Path(root).resolve(strict=True)
    bundle = preflight(root, protocol_sha256)
    if not score:
        return {"status": "preflight_only", "protocol_sha256": protocol_sha256,
                "checkpoint": bundle["spec"]["training_source"]["checkpoint"],
                "environment_resets": 0, "optimizer_updates": 0}
    if (root / OUTPUT).exists() or (root / OUTPUT).is_symlink():
        raise ValueError("score output already exists; no overwrite or checkpoint-specific retry")
    original, new = _models(root, bundle)
    entries, summary, gate = score_rows(bundle["primary"], bundle["old_summary"], original, new)
    preflight(root, protocol_sha256)  # Final rehash, including both complete ledgers and checkpoint.
    report = {"format": RESULT_FORMAT, "status": "PASS" if gate["passed"] else "FAIL",
              "protocol": _ref(PROTOCOL, protocol_sha256),
              "training_source": bundle["spec"]["training_source"],
              "original_branch": bundle["spec"]["original_branch"],
              "source_sha256": bundle["spec"]["source_sha256"],
              "runtime": bundle["spec"]["runtime"], "anchors": entries,
              "summary": summary, "gate": gate, "environment_resets": 0,
              "optimizer_updates": 0, "scope": "reused_TRAIN_archived_branches_only"}
    _write_exclusive(root, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="read-only source/complete-ledger gate")
    mode.add_argument("--score", action="store_true", help="exclusive zero-reset archived score")
    args = parser.parse_args()
    if args.protocol.as_posix() != PROTOCOL:
        parser.error("only the separately frozen dedicated score protocol is accepted")
    print(json.dumps(run(ROOT, args.protocol_sha256, score=args.score), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
