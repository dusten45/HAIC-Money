"""Isolated, masked recovery replay and fixed-source SAC batches."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common_adapter import ActionAdapter
from haic.algorithms.rlpd.replay import FrameStackReplay, PixelTransition


DATASET_FORMAT = "haic-rlpd-recovery-dataset-v1"
FINISH_POLICY = "paired-finish-recovery-plus-handoff-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_stack(frames: np.ndarray, step: int) -> np.ndarray:
    return frames[np.maximum(np.arange(step - 3, step + 1), 0)].copy()


def load_recovery_replay(dataset_dir: Path, *, manifest_sha256: str,
                         allowed_cells: list[dict], seed: int,
                         minimum_transitions: int = 128,
                         minimum_geometries: int = 3,
                         required_policy: str | None = None) -> tuple[FrameStackReplay, dict]:
    """Verify full episodes, then restrict sampling without discarding stack context."""
    if (type(minimum_transitions) is not int or minimum_transitions <= 0
            or type(minimum_geometries) is not int or minimum_geometries <= 0):
        raise ValueError("recovery sufficiency thresholds must be positive integers")
    root = Path(dataset_dir).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("recovery manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError("unsupported recovery dataset format")
    policy = manifest.get("eligibility_policy")
    if policy not in (None, FINISH_POLICY) or required_policy is not None and policy != required_policy:
        raise ValueError("recovery eligibility policy mismatch")
    if policy == FINISH_POLICY:
        digest = manifest.get("source_manifest_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or not isinstance(manifest.get("source_cost_counts"), dict)):
            raise ValueError("finish-qualified recovery source lineage is missing")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("recovery manifest has no episodes")
    allowed = {(c["track_id"], c["geometry_seed"]) for c in allowed_cells
               if c.get("partition") == "TRAIN" and c.get("obstacles") is True}
    capacity = 0
    stored_transitions = 0
    ids = set()
    paths = set()
    for row in episodes:
        if (type(row.get("episode_id")) is not int or row["episode_id"] < 0
                or row["episode_id"] in ids or type(row.get("steps")) is not int
                or not 0 < row["steps"] <= 2000
                or type(row.get("censored")) is not bool
                or row.get("stratum") not in ("failure", "finish-control")
                or (row.get("track_id"), row.get("geometry_seed")) not in allowed):
            raise ValueError("invalid recovery episode identity, length or TRAIN cell")
        ids.add(row["episode_id"])
        relative = Path(row["path"])
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or any((root.joinpath(*relative.parts[:i])).is_symlink()
                       for i in range(1, len(relative.parts) + 1))
                or path.resolve() in paths or file_sha256(path) != row.get("sha256")):
            raise ValueError("unsafe, duplicate or changed recovery episode path")
        paths.add(path.resolve())
        stored_transitions += row["steps"]
        if not row["censored"]:
            capacity += row["steps"]
    replay = FrameStackReplay(max(1, capacity), seed=seed, source="recovery")
    accepted_rows = []
    geometries = set()
    unique = {"failure": set(), "finish-control": set()}
    geometry_by_stratum = {"failure": set(), "finish-control": set()}
    adapter = ActionAdapter()
    for row in episodes:
        with np.load(root / row["path"], allow_pickle=False) as data:
            n = row["steps"]
            frames = data["frames"]
            if frames.shape != (n + 1, 84, 84) or frames.dtype != np.uint8:
                raise ValueError("recovery frames must be N+1 uint8 frames")
            for name, step in (("initial_stack", 0), ("final_stack", n)):
                if (data[name].dtype != np.uint8
                        or not np.array_equal(data[name], frame_stack(frames, step))):
                    raise ValueError("recovery initial/final stack reconstruction mismatch")
            fields = {name: data[name] for name in (
                "proposed_action", "executed_action", "applied_action", "reward",
                "terminated", "truncated", "terminal", "episode_end", "step", "role",
                "recovery_mask",
            )}
            for name, value in fields.items():
                expected = (n, 3) if name.endswith("action") else (n,)
                if value.shape != expected:
                    raise ValueError(f"invalid recovery field shape: {name}")
            for name in ("terminated", "truncated", "terminal", "episode_end", "recovery_mask"):
                if fields[name].dtype != np.bool_:
                    raise ValueError(f"recovery flags must be boolean: {name}")
            if (not np.issubdtype(fields["step"].dtype, np.integer)
                    or not np.array_equal(fields["step"], np.arange(n))):
                raise ValueError("recovery steps are not contiguous from reset")
            ended = fields["terminated"] | fields["truncated"]
            if (not np.array_equal(fields["episode_end"], ended) or ended[:-1].any()
                    or bool(ended[-1]) == row["censored"]
                    or (fields["terminated"] & ~fields["terminal"]).any()
                    or (fields["terminal"] & ~ended).any()):
                raise ValueError("recovery episode lacks a real, terminal-safe ending")
            mask = fields["recovery_mask"]
            if (int(mask.sum()) != row.get("accepted_transitions")
                    or mask.any() and (row.get("local_recovery_qualified") is not True
                                       or row.get("mode") == "actor" or row["censored"])):
                raise ValueError("recovery mask includes unqualified or non-Oracle rows")
            if policy == FINISH_POLICY:
                if type(row.get("paired_finish_qualified")) is not bool:
                    raise ValueError("finish-qualified recovery lacks its paired eligibility flag")
                expected_mask = np.zeros(n, dtype=np.bool_)
                if row["paired_finish_qualified"]:
                    start, end = row.get("training_window_start"), row.get("training_window_end")
                    anchor, horizon = row.get("anchor_step"), row.get("horizon")
                    if (type(start) is not int or type(end) is not int or type(anchor) is not int
                            or type(horizon) is not int or horizon not in (12, 25)
                            or not 0 <= start == anchor < anchor + horizon < end <= n
                            or end != anchor + horizon + 63
                            or row.get("mode") != f"oracle-{horizon}"
                            or row.get("finished") is not True or row["censored"]
                            or not fields["terminal"][-1]
                            or row.get("local_recovery_qualified") is not True
                            or not (fields["role"][start:anchor + horizon] == "oracle").all()
                            or not (fields["role"][anchor + horizon:end] == "actor").all()):
                        raise ValueError("invalid paired-finish recovery/handoff window")
                    expected_mask[start:end] = True
                if not np.array_equal(mask, expected_mask):
                    raise ValueError("recovery mask differs from the paired-finish bounded window")
            elif (mask & (fields["role"] != "oracle")).any():
                raise ValueError("legacy recovery masks may include only Oracle rows")
            offset = len(replay)
            for step in range(n):
                executed = fields["executed_action"][step]
                applied = fields["applied_action"][step]
                if not np.array_equal(adapter.to_native(applied), executed):
                    raise ValueError("executed native action does not match applied official action")
                transition = PixelTransition(
                    observation=frame_stack(frames, step),
                    next_observation=frame_stack(frames, step + 1),
                    proposed_action=fields["proposed_action"][step], executed_action=executed,
                    applied_action=applied, reward=float(fields["reward"][step]),
                    terminated=bool(fields["terminated"][step]),
                    truncated=bool(fields["truncated"][step]), terminal=bool(fields["terminal"][step]),
                    episode_id=row["episode_id"], step=step, track_id=row["track_id"],
                    geometry_seed=row["geometry_seed"],
                )
                if not row["censored"]:
                    replay.add(transition)
                if mask[step]:
                    identity = hashlib.sha256(transition.observation.tobytes()
                                              + transition.executed_action.tobytes()).digest()
                    unique[row["stratum"]].add(identity)
            accepted_rows.extend((offset + np.flatnonzero(mask)).tolist())
            if mask.any():
                geometry = (row["track_id"], row["geometry_seed"])
                geometries.add(geometry)
                geometry_by_stratum[row["stratum"]].add(geometry)
    if replay.valid_count != len(replay):
        raise ValueError("recovery full-episode reconstruction is incomplete")
    if (len(unique["failure"]) < minimum_transitions
            or len(geometry_by_stratum["failure"]) < minimum_geometries):
        raise ValueError("recovery dataset fails the frozen count/geometry sufficiency gate")
    # Retain every frame for stack reconstruction; only valid-row sampling is masked.
    accepted = set(accepted_rows)
    for index in range(len(replay)):
        if index not in accepted:
            replay._remove_valid(index)
    replay.finalize()
    receipt = {"manifest_sha256": manifest_sha256, "stored_transitions": stored_transitions,
                    "replay_transitions": len(replay),
                    "accepted_transitions": replay.valid_count,
                    "unique_accepted_transitions": len(unique["failure"] | unique["finish-control"]),
                    "unique_failure_transitions": len(unique["failure"]),
                    "failure_geometries": len(geometry_by_stratum["failure"]),
                    "accepted_geometries": len(geometries), "episode_count": len(episodes),
                    "censored_episodes_excluded": sum(row["censored"] for row in episodes)}
    if policy is not None:
        receipt.update({"eligibility_policy": policy,
                        "source_manifest_sha256": manifest["source_manifest_sha256"],
                        "source_cost_counts": manifest["source_cost_counts"],
                        "dataset_source_sha256": manifest.get("dataset_source_sha256", {})})
    return replay, receipt


def sample_recovery_batch(online, prior, recovery=None, *, arm: str) -> dict:
    """Exactly 32 online + 32 prior, or 32 online + 16 prior + 16 recovery."""
    if arm not in ("control", "treatment"):
        raise ValueError("arm must be control or treatment")
    sources = [("online", online, 32), ("prior", prior, 32 if arm == "control" else 16)]
    if arm == "treatment":
        sources.append(("recovery", recovery, 16))
    if any(replay is None or replay.valid_count == 0 for _, replay, _ in sources):
        raise ValueError("all requested replay sources must have valid rows; no fallback")
    for name, replay, _ in sources:
        if replay.source != ("offline" if name == "prior" else name):
            raise ValueError("replay source identity does not match its fixed sampling role")
        if name != "online" and not replay.immutable:
            raise ValueError("prior and recovery sources must be frozen")
    batches = [replay.sample(count) for _, replay, count in sources]
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches):
        raise ValueError("replay source fields differ")
    result: dict = {key: np.concatenate([batch[key] for batch in batches]) for key in keys}
    result["source_counts"] = {name: count for name, _, count in sources}
    return result


def import_learning_state(agent, state: dict, *, seed: int) -> dict:
    """Import model/optimizers only. No historical replay, RNG, or env continuation."""
    if (state.get("format") != "haic-rlpd-training-checkpoint-v1"
            or state.get("config") != asdict(agent.config)):
        raise ValueError("source learning checkpoint format/config mismatch")
    for name in ("actor", "critic", "target_critic"):
        getattr(agent, name).load_state_dict(state[name], strict=True)
    agent.log_alpha.data.copy_(state["log_alpha"].to(agent.device))
    for name in ("actor_optimizer", "critic_optimizer", "temperature_optimizer"):
        optimizer = getattr(agent, name)
        optimizer.load_state_dict(state[name])
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(agent.device)
    agent.rng.manual_seed(seed + 0x524C5044)
    agent.environment_steps = 0
    agent.gradient_steps = 0
    return {"source_environment_steps": int(state["environment_steps"]),
            "source_gradient_steps": int(state["gradient_steps"]),
            "model_and_optimizers_imported": True, "fresh_online_replay": True,
            "rng_reset_seed": seed, "exact_continuation": False}
