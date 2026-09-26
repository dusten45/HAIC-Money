"""Immutable, compact teacher trajectories and fixed-quota DrQ replay sampling.

Episodes retain one uint8 grayscale latest frame per decision plus the actual
final frame. Four-frame float32 observations are reconstructed only when sampled.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from common_adapter import ActionAdapter, ObservationSpec, Transition


_FORMAT_VERSION = 1
_ARRAY_FIELDS = (
    "frames",
    "actions",
    "applied_actions",
    "rewards",
    "progress",
    "damage",
    "terminated",
    "truncated",
    "finished",
    "terminal",
)
_BATCH_FIELDS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminated",
    "truncated",
    "finished",
    "terminal",
    "discount",
    "horizon",
    "episode_index",
    "episode_id",
    "step",
    "source_id",
    "source_actor_sha256",
    "geometry_id",
    "track_id",
    "applied_action",
    "progress",
    "damage",
    "retire_reason",
)
_CORE_BATCH_FIELDS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminated",
    "truncated",
    "terminal",
    "discount",
    "horizon",
)
_OBSERVATIONS = ObservationSpec()
_ACTIONS = ActionAdapter()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else _unsupported_json(item),
    ).encode("utf-8")


def _unsupported_json(value: Any) -> None:
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def _frame_stack(frames: np.ndarray, index: int) -> np.ndarray:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("frame index must be an integer")
    index = int(index)
    if not 0 <= index < len(frames):
        raise IndexError("frame index is outside the episode")
    return np.stack([frames[max(0, index - 3 + channel)] for channel in range(4)], axis=0)


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """Back an array with immutable bytes so views cannot be made writable."""
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _readonly(values: Any, *, dtype: np.dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} has an invalid representation") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return _immutable_array(array)


def _float_rows(values: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        original = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if original.shape != shape or not np.isfinite(original).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if np.any(np.abs(original) > np.finfo(np.float32).max):
        raise ValueError(f"{name} cannot be represented as float32")
    result = original.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} cannot be represented as float32")
    return result


def _bool_rows(values: Any, *, length: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if raw.dtype.kind not in "bui" or np.any((raw != 0) & (raw != 1)):
        raise ValueError(f"{name} must contain only boolean values")
    return raw.astype(np.bool_, copy=True)


def _metadata_copy(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("episode metadata must be a JSON-compatible mapping")
    try:
        return json.loads(_canonical_json(dict(metadata)).decode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("episode metadata must be finite JSON-compatible values") from exc


def _latest_frame_stack(observation: Any, *, name: str) -> np.ndarray:
    array = np.asarray(observation)
    if array.dtype == np.uint8 and array.shape == _OBSERVATIONS.shape:
        return array.copy()
    try:
        return _OBSERVATIONS.to_uint8(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid (4,84,84) observation in [0,1]") from exc


class TeacherEpisode:
    """A complete episode encoded as ``T+1`` latest grayscale uint8 frames.

    ``frames[t]`` is the newest pixel in observation ``t``; the last frame is
    the recorded final observation, including on terminal and timeout rows.
    Arrays are copied and read-only. Actions are the actually executed native
    values, and each stored official action is checked against their mapping.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("TeacherEpisode is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        frames: Any,
        actions: Any,
        applied_actions: Any,
        rewards: Any,
        progress: Any | None = None,
        damage: Any | None = None,
        retire_reasons: Sequence[str | None] | None = None,
        terminated: Any,
        truncated: Any,
        finished: Any,
        terminal: Any | None = None,
        episode_id: str | int,
        source_id: str,
        source_actor_sha256: str,
        geometry_id: str,
        track_id: int,
        complete: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ):
        raw_frames = np.asarray(frames)
        if raw_frames.dtype != np.uint8 or raw_frames.ndim != 3 or raw_frames.shape[1:] != (84, 84):
            raise ValueError("frames must be uint8 latest-frame rows with shape (T+1,84,84)")
        if len(raw_frames) < 2:
            raise ValueError("an episode must contain at least one transition and its next frame")
        steps = len(raw_frames) - 1
        frozen_frames = _readonly(raw_frames, dtype=np.dtype(np.uint8),
                                  shape=(steps + 1, 84, 84), name="frames")

        native_actions = _float_rows(actions, shape=(steps, 3), name="actions")
        if np.any(native_actions < -1.0) or np.any(native_actions > 1.0):
            raise ValueError("executed native actions must lie in [-1,1]")
        official_actions = _float_rows(applied_actions, shape=(steps, 3), name="applied_actions")
        try:
            for row in official_actions:
                _ACTIONS.to_native(row, clip=False)
        except ValueError as exc:
            raise ValueError("applied official actions must lie within official bounds") from exc
        for native, official in zip(native_actions, official_actions, strict=True):
            expected = _ACTIONS.to_official(native, clip=False)
            if not np.allclose(expected, official, rtol=0.0, atol=1e-6):
                raise ValueError("applied official action does not audit against executed native action")

        reward_rows = _float_rows(rewards, shape=(steps,), name="rewards")
        progress_rows = _float_rows(
            np.zeros(steps, dtype=np.float32) if progress is None else progress,
            shape=(steps,), name="progress",
        )
        damage_rows = _float_rows(
            np.zeros(steps, dtype=np.float32) if damage is None else damage,
            shape=(steps,), name="damage",
        )
        if retire_reasons is None:
            retire_reason_rows = (None,) * steps
        else:
            retire_reason_rows = tuple(retire_reasons)
            if len(retire_reason_rows) != steps or any(
                reason is not None and not isinstance(reason, str)
                for reason in retire_reason_rows
            ):
                raise ValueError("retire_reasons must have one string-or-None value per step")
        terminated_rows = _bool_rows(terminated, length=steps, name="terminated")
        truncated_rows = _bool_rows(truncated, length=steps, name="truncated")
        finished_rows = _bool_rows(finished, length=steps, name="finished")
        episode_metadata = _metadata_copy(metadata)
        if terminal is None:
            retire_reason = episode_metadata.get("retire_reason")
            is_retired = retire_reason in {"crash", "off_track", "out_of_bounds"}
            terminal_rows = terminated_rows | finished_rows
            if is_retired:
                terminal_rows[-1] = True
        else:
            terminal_rows = _bool_rows(terminal, length=steps, name="terminal")

        boundaries = terminated_rows | truncated_rows | finished_rows | terminal_rows
        if np.any(boundaries[:-1]):
            raise ValueError("episode boundary flags may only occur on its final transition")
        if not isinstance(complete, (bool, np.bool_)):
            raise ValueError("complete must be a boolean")
        complete = bool(complete)
        if complete and not boundaries[-1]:
            raise ValueError("complete episodes require a terminal, finished, or truncated final row")

        actor_hash = str(source_actor_sha256)
        if re.fullmatch(r"[0-9a-f]{64}", actor_hash) is None:
            raise ValueError("source_actor_sha256 must be a lowercase SHA-256 digest")
        for name, value in (("episode_id", episode_id), ("source_id", source_id),
                            ("geometry_id", geometry_id)):
            if value is None or isinstance(value, bool) or not str(value).strip():
                raise ValueError(f"{name} must be a non-empty identifier")
        if isinstance(track_id, bool) or not isinstance(track_id, (int, np.integer)):
            raise ValueError("track_id must be an integer")
        if int(track_id) < 0:
            raise ValueError("track_id must be non-negative")

        for name, array in (
            ("frames", frozen_frames),
            ("actions", native_actions),
            ("applied_actions", official_actions),
            ("rewards", reward_rows),
            ("progress", progress_rows),
            ("damage", damage_rows),
            ("terminated", terminated_rows),
            ("truncated", truncated_rows),
            ("finished", finished_rows),
            ("terminal", terminal_rows),
        ):
            setattr(self, f"_{name}", _immutable_array(array))
        self.episode_id = str(episode_id)
        self.source_id = str(source_id)
        self.source_actor_sha256 = actor_hash
        self.geometry_id = str(geometry_id)
        self.track_id = int(track_id)
        self.complete = bool(complete)
        self._metadata = episode_metadata
        self._retire_reasons = tuple(retire_reason_rows)
        object.__setattr__(self, "_frozen", True)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(_metadata_copy(self._metadata))

    @classmethod
    def from_transitions(
        cls,
        transitions: Sequence[Transition],
        *,
        source_id: str,
        source_actor_sha256: str,
        geometry_id: str,
        track_id: int,
        complete: bool = True,
        metadata: Mapping[str, Any] | None = None,
        progress: Any | None = None,
        damage: Any | None = None,
        retire_reasons: Sequence[str | None] | None = None,
    ) -> "TeacherEpisode":
        """Freeze collector transitions after checking sequence and action parity."""
        rows = tuple(transitions)
        if not rows:
            raise ValueError("cannot create a teacher episode without transitions")
        episode_key = rows[0].episode_id
        if any(row.episode_id != episode_key for row in rows):
            raise ValueError("transitions from different episodes cannot be combined")
        if any(row.step != step for step, row in enumerate(rows)):
            raise ValueError("episode transition steps must be contiguous from reset step zero")

        observations: list[np.ndarray] = []
        next_observations: list[np.ndarray] = []
        native_actions: list[np.ndarray] = []
        official_actions: list[np.ndarray] = []
        for index, row in enumerate(rows):
            try:
                observations.append(_latest_frame_stack(row.observation, name=f"observation[{index}]"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"transition {index} has an invalid current observation") from exc
            try:
                next_observations.append(
                    _latest_frame_stack(row.next_observation, name=f"next_observation[{index}]")
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"transition {index} is missing a valid next/boundary frame") from exc
            if row.applied_action is None:
                raise ValueError(f"transition {index} has no applied official action audit")
            native_actions.append(row.action)
            official_actions.append(row.applied_action)

        frames = np.stack([obs[-1] for obs in observations] + [next_observations[-1][-1]])
        for index, observation in enumerate(observations):
            if not np.array_equal(observation, _frame_stack(frames, index)):
                raise ValueError(f"observation stack {index} does not match latest-frame sequence")
            if not np.array_equal(next_observations[index], _frame_stack(frames, index + 1)):
                raise ValueError(f"transition {index} next observation is not frame-aligned")

        terminated = np.asarray([row.terminated for row in rows], dtype=np.bool_)
        truncated = np.asarray([row.truncated for row in rows], dtype=np.bool_)
        finished = np.asarray([bool(row.info.get("finished", False)) for row in rows], dtype=np.bool_)
        terminal = np.asarray([row.is_terminal for row in rows], dtype=np.bool_)
        progress = [row.info.get("progress", 0.0) for row in rows]
        damage = [row.info.get("damage", 0.0) for row in rows]
        retire_reasons = [row.info.get("retire_reason") for row in rows]
        final_info = dict(rows[-1].info)
        episode_metadata = dict(metadata or {})
        for key in ("retire_reason", "progress", "damage"):
            if key in final_info:
                episode_metadata.setdefault(key, final_info[key])
        return cls(
            frames=frames,
            actions=native_actions,
            applied_actions=official_actions,
            rewards=[row.reward for row in rows],
            progress=progress,
            damage=damage,
            retire_reasons=retire_reasons,
            terminated=terminated,
            truncated=truncated,
            finished=finished,
            terminal=terminal,
            episode_id=episode_key,
            source_id=source_id,
            source_actor_sha256=source_actor_sha256,
            geometry_id=geometry_id,
            track_id=track_id,
            complete=complete,
            metadata=episode_metadata,
        )

    @property
    def steps(self) -> int:
        return len(self._actions)

    def _record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_id": self.source_id,
            "source_actor_sha256": self.source_actor_sha256,
            "geometry_id": self.geometry_id,
            "track_id": self.track_id,
            "complete": self.complete,
            "metadata": self._metadata,
            "retire_reasons": list(self._retire_reasons),
        }

    def _array(self, name: str) -> np.ndarray:
        result = getattr(self, f"_{name}").view()
        result.flags.writeable = False
        return result

    @property
    def frames(self) -> np.ndarray:
        return self._array("frames")

    @property
    def actions(self) -> np.ndarray:
        return self._array("actions")

    @property
    def applied_actions(self) -> np.ndarray:
        return self._array("applied_actions")

    @property
    def rewards(self) -> np.ndarray:
        return self._array("rewards")

    @property
    def terminated(self) -> np.ndarray:
        return self._array("terminated")

    @property
    def truncated(self) -> np.ndarray:
        return self._array("truncated")

    @property
    def finished(self) -> np.ndarray:
        return self._array("finished")

    @property
    def terminal(self) -> np.ndarray:
        return self._array("terminal")

    @property
    def progress(self) -> np.ndarray:
        return self._array("progress")

    @property
    def damage(self) -> np.ndarray:
        return self._array("damage")

    @property
    def retire_reasons(self) -> tuple[str | None, ...]:
        return self._retire_reasons

    def observation(self, step: int) -> np.ndarray:
        """Reconstruct one float32 four-frame input with reset stack padding."""
        return _OBSERVATIONS.from_uint8(_frame_stack(self._frames, step))


@dataclass(frozen=True)
class NstepTarget:
    """One aligned decision-level n-step target from a sealed teacher dataset."""

    observation: np.ndarray
    action: np.ndarray
    applied_action: np.ndarray
    progress: np.float32
    damage: np.float32
    retire_reason: str | None
    reward: np.float32
    next_observation: np.ndarray
    terminated: np.bool_
    truncated: np.bool_
    finished: np.bool_
    terminal: np.bool_
    discount: np.float32
    horizon: np.int64
    episode_index: np.int64
    episode_id: str
    step: np.int64
    source_id: str
    source_actor_sha256: str
    geometry_id: str
    track_id: np.int64

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _BATCH_FIELDS}


class TeacherDataset:
    """Seal complete episodes and expose verifiable, latest-frame storage.

    The digest covers pixels, actions, rewards, progress/damage labels, end flags,
    retirement reasons, identities, and episode metadata. Once sealed, episodes
    cannot be added and their arrays are read-only. Serialization uses numeric
    NPZ arrays only; object pickling is not used.
    """

    def __init__(self, episodes: Sequence[TeacherEpisode] = ()):
        self._episodes: list[TeacherEpisode] | tuple[TeacherEpisode, ...] = []
        self._sealed = False
        self._digest: str | None = None
        for episode in episodes:
            self.add_episode(episode)

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def episodes(self) -> tuple[TeacherEpisode, ...]:
        return tuple(self._episodes)

    @property
    def digest(self) -> str:
        if not self._sealed or self._digest is None:
            raise RuntimeError("teacher dataset must be sealed before reading its digest")
        return self._digest

    @property
    def transition_count(self) -> int:
        return sum(episode.steps for episode in self._episodes)

    @property
    def memory_bytes(self) -> int:
        return int(sum(getattr(episode, f"_{name}").nbytes
                       for episode in self._episodes for name in _ARRAY_FIELDS))

    def add_episode(self, episode: TeacherEpisode) -> None:
        if self._sealed:
            raise RuntimeError("a sealed teacher dataset is immutable")
        if not isinstance(episode, TeacherEpisode):
            raise TypeError("episode must be a TeacherEpisode")
        assert isinstance(self._episodes, list)
        self._episodes.append(episode)

    @staticmethod
    def _digest_episodes(episodes: Sequence[TeacherEpisode]) -> str:
        digest = hashlib.sha256()
        digest.update(b"HAIC-DRQV2-TEACHER-DATASET\x00v1\x00")
        for episode in episodes:
            metadata = _canonical_json(episode._record())
            digest.update(len(metadata).to_bytes(8, "little"))
            digest.update(metadata)
            for name in _ARRAY_FIELDS:
                array = getattr(episode, f"_{name}")
                descriptor = _canonical_json({
                    "name": name,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                })
                raw = memoryview(np.ascontiguousarray(array)).cast("B")
                digest.update(len(descriptor).to_bytes(8, "little"))
                digest.update(descriptor)
                digest.update(len(raw).to_bytes(8, "little"))
                digest.update(raw)
        return digest.hexdigest()

    def seal(self) -> str:
        """Reject partial/duplicate episodes, freeze membership, and return SHA-256."""
        if self._sealed:
            return self.digest
        if not self._episodes:
            raise ValueError("cannot seal an empty teacher dataset")
        identities = [(episode.source_id, episode.episode_id) for episode in self._episodes]
        if len(set(identities)) != len(identities):
            raise ValueError("teacher episode identities must be unique within each source")
        for episode in self._episodes:
            if not episode.complete:
                raise ValueError("cannot seal an incomplete teacher episode")
            if len(episode._frames) != episode.steps + 1:
                raise ValueError("complete episode is missing its final observation frame")
        self._episodes = tuple(self._episodes)
        self._digest = self._digest_episodes(self._episodes)
        self._sealed = True
        return self._digest

    def _require_sealed(self) -> tuple[TeacherEpisode, ...]:
        if not self._sealed:
            raise RuntimeError("teacher dataset must be sealed before sampling")
        return self._episodes  # type: ignore[return-value]

    def _offsets(self) -> list[int]:
        offsets = [0]
        for episode in self._require_sealed():
            offsets.append(offsets[-1] + episode.steps)
        return offsets

    def _locate(self, index: int) -> tuple[int, int]:
        offsets = self._offsets()
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("teacher row index must be an integer")
        index = int(index)
        if index < 0 or index >= offsets[-1]:
            raise IndexError("teacher row index is outside the dataset")
        episode_index = bisect_right(offsets, index) - 1
        return episode_index, index - offsets[episode_index]

    @staticmethod
    def _validate_return(gamma: float, n_step: int) -> tuple[float, int]:
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be finite and in (0,1]")
        if type(n_step) is not int or not 1 <= n_step <= 3:
            raise ValueError("n_step must be an integer in [1,3]")
        return float(gamma), n_step

    def build_n_step(self, index: int, *, gamma: float = 0.99, n_step: int = 3) -> NstepTarget:
        """Build up to three rewards without crossing an episode boundary."""
        gamma, n_step = self._validate_return(gamma, n_step)
        episode_index, start = self._locate(index)
        episode = self._episodes[episode_index]
        total_reward = 0.0
        horizon = 0
        endpoint = start
        for offset in range(n_step):
            endpoint = start + offset
            total_reward += gamma**offset * float(episode._rewards[endpoint])
            horizon = offset + 1
            if (
                episode._terminated[endpoint]
                or episode._truncated[endpoint]
                or episode._finished[endpoint]
                or episode._terminal[endpoint]
            ):
                break
        if not math.isfinite(total_reward) or abs(total_reward) > np.finfo(np.float32).max:
            raise ValueError("n-step reward is not representable as float32")
        terminal = bool(episode._terminal[endpoint] or episode._terminated[endpoint]
                        or episode._finished[endpoint])
        return NstepTarget(
            observation=episode.observation(start),
            action=episode._actions[start].copy(),
            applied_action=episode._applied_actions[start].copy(),
            progress=np.float32(episode._progress[start]),
            damage=np.float32(episode._damage[start]),
            retire_reason=episode._retire_reasons[start],
            reward=np.float32(total_reward),
            next_observation=episode.observation(endpoint + 1),
            terminated=np.bool_(episode._terminated[endpoint]),
            truncated=np.bool_(episode._truncated[endpoint]),
            finished=np.bool_(episode._finished[endpoint]),
            terminal=np.bool_(terminal),
            discount=np.float32(0.0 if terminal else gamma**horizon),
            horizon=np.int64(horizon),
            episode_index=np.int64(episode_index),
            episode_id=episode.episode_id,
            step=np.int64(start),
            source_id=episode.source_id,
            source_actor_sha256=episode.source_actor_sha256,
            geometry_id=episode.geometry_id,
            track_id=np.int64(episode.track_id),
        )

    def valid_indices(self, *, gamma: float = 0.99, n_step: int = 3) -> list[int]:
        """List rows with complete within-episode n-step targets."""
        self._validate_return(gamma, n_step)
        self._require_sealed()
        return list(range(self.transition_count))

    def sample_at_indices(
        self, indices: Sequence[int], *, gamma: float = 0.99, n_step: int = 3
    ) -> dict[str, np.ndarray]:
        if len(indices) == 0:
            raise ValueError("at least one teacher index is required")
        rows = [self.build_n_step(index, gamma=gamma, n_step=n_step) for index in indices]
        return {name: np.stack([row.as_dict()[name] for row in rows]) for name in _BATCH_FIELDS}

    def sample(
        self,
        batch_size: int,
        *,
        gamma: float = 0.99,
        n_step: int = 3,
        indices: Sequence[int] | None = None,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        valid = self.valid_indices(gamma=gamma, n_step=n_step)
        if indices is None:
            if len(valid) < batch_size:
                raise ValueError("teacher pool has fewer valid rows than requested")
            chosen = np.random.default_rng(seed).choice(valid, size=batch_size, replace=False).tolist()
        else:
            chosen = list(indices)
            if len(chosen) != batch_size:
                raise ValueError("indices length must equal batch_size")
        return self.sample_at_indices(chosen, gamma=gamma, n_step=n_step)

    def to_bytes(self) -> bytes:
        """Serialize a sealed dataset with a content digest and no object arrays."""
        episodes = self._require_sealed()
        manifest = _canonical_json({"version": _FORMAT_VERSION,
                                    "episodes": [episode._record() for episode in episodes]})
        transition_offsets = np.zeros(len(episodes) + 1, dtype=np.int64)
        frame_offsets = np.zeros(len(episodes) + 1, dtype=np.int64)
        for index, episode in enumerate(episodes):
            transition_offsets[index + 1] = transition_offsets[index] + episode.steps
            frame_offsets[index + 1] = frame_offsets[index] + episode.steps + 1
        arrays: dict[str, np.ndarray] = {
            "manifest": np.frombuffer(manifest, dtype=np.uint8),
            "digest": np.frombuffer(self.digest.encode("ascii"), dtype=np.uint8),
            "transition_offsets": transition_offsets,
            "frame_offsets": frame_offsets,
        }
        for name in _ARRAY_FIELDS:
            arrays[name] = np.concatenate([getattr(ep, f"_{name}") for ep in episodes], axis=0)
        stream = BytesIO()
        np.savez_compressed(stream, **arrays)
        return stream.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_digest: str | None = None) -> "TeacherDataset":
        """Load, validate, and seal bytes; reject corrupt or unexpected digests."""
        try:
            with np.load(BytesIO(data), allow_pickle=False) as archive:
                expected_keys = set(_ARRAY_FIELDS) | {
                    "manifest", "digest", "transition_offsets", "frame_offsets"
                }
                if set(archive.files) != expected_keys:
                    raise ValueError("serialized teacher dataset has unsupported fields")
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        except Exception as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("serialized teacher dataset"):
                raise
            raise ValueError("invalid serialized teacher dataset") from exc
        try:
            manifest_array = arrays.pop("manifest")
            digest_array = arrays.pop("digest")
            if (
                manifest_array.dtype != np.uint8
                or manifest_array.ndim != 1
                or digest_array.dtype != np.uint8
                or digest_array.ndim != 1
            ):
                raise ValueError("invalid teacher dataset manifest or digest arrays")
            manifest_bytes = manifest_array.tobytes()
            stored_digest = digest_array.tobytes().decode("ascii")
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, ValueError) as exc:
            raise ValueError("invalid teacher dataset manifest or digest") from exc
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"version", "episodes"}
            or manifest.get("version") != _FORMAT_VERSION
            or manifest_bytes != _canonical_json(manifest)
        ):
            raise ValueError("unsupported teacher dataset serialization version")
        records = manifest.get("episodes")
        if not isinstance(records, list) or not records:
            raise ValueError("serialized dataset must contain complete episodes")
        for name, dtype in (("frames", np.uint8), ("actions", np.float32),
                            ("applied_actions", np.float32), ("rewards", np.float32),
                            ("progress", np.float32), ("damage", np.float32),
                            ("terminated", np.bool_), ("truncated", np.bool_),
                            ("finished", np.bool_), ("terminal", np.bool_),
                            ("transition_offsets", np.int64), ("frame_offsets", np.int64)):
            if arrays[name].dtype != np.dtype(dtype):
                raise ValueError(f"serialized teacher dataset has invalid {name} dtype")
        trans_offsets = arrays["transition_offsets"]
        pix_offsets = arrays["frame_offsets"]
        if (
            trans_offsets.shape != (len(records) + 1,)
            or pix_offsets.shape != (len(records) + 1,)
            or trans_offsets[0] != 0
            or pix_offsets[0] != 0
            or np.any(np.diff(trans_offsets) <= 0)
            or np.any(np.diff(pix_offsets) != np.diff(trans_offsets) + 1)
            or trans_offsets[-1] != len(arrays["actions"])
            or pix_offsets[-1] != len(arrays["frames"])
        ):
            raise ValueError("serialized teacher dataset offsets are invalid")
        transition_count = int(trans_offsets[-1])
        frame_count = int(pix_offsets[-1])
        expected_shapes = {
            "frames": (frame_count, 84, 84),
            "actions": (transition_count, 3),
            "applied_actions": (transition_count, 3),
            "rewards": (transition_count,),
            "progress": (transition_count,),
            "damage": (transition_count,),
            "terminated": (transition_count,),
            "truncated": (transition_count,),
            "finished": (transition_count,),
            "terminal": (transition_count,),
        }
        if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
            raise ValueError("serialized teacher dataset arrays do not match episode offsets")
        episodes = []
        episode_record_fields = {
            "episode_id", "source_id", "source_actor_sha256", "geometry_id",
            "track_id", "complete", "metadata", "retire_reasons",
        }
        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != episode_record_fields:
                raise ValueError("serialized episode metadata has unsupported fields")
            start, stop = map(int, trans_offsets[index : index + 2])
            frame_start, frame_stop = map(int, pix_offsets[index : index + 2])
            if not isinstance(record["retire_reasons"], list) or len(record["retire_reasons"]) != stop - start:
                raise ValueError("serialized retirement labels do not align with episode transitions")
            episodes.append(TeacherEpisode(
                frames=arrays["frames"][frame_start:frame_stop],
                actions=arrays["actions"][start:stop],
                applied_actions=arrays["applied_actions"][start:stop],
                rewards=arrays["rewards"][start:stop],
                progress=arrays["progress"][start:stop],
                damage=arrays["damage"][start:stop],
                retire_reasons=record["retire_reasons"],
                terminated=arrays["terminated"][start:stop],
                truncated=arrays["truncated"][start:stop],
                finished=arrays["finished"][start:stop],
                terminal=arrays["terminal"][start:stop],
                episode_id=record["episode_id"],
                source_id=record["source_id"],
                source_actor_sha256=record["source_actor_sha256"],
                geometry_id=record["geometry_id"],
                track_id=record["track_id"],
                complete=record["complete"],
                metadata=record.get("metadata", {}),
            ))
        dataset = cls(episodes)
        actual_digest = dataset.seal()
        if not re.fullmatch(r"[0-9a-f]{64}", stored_digest) or actual_digest != stored_digest:
            raise ValueError("teacher dataset digest mismatch")
        if expected_digest is not None and actual_digest != expected_digest:
            raise ValueError("teacher dataset digest does not match expected digest")
        return dataset


class TwoSourceReplay:
    """Sample fixed online-only ``64:0`` or teacher ``48:16`` minibatches.

    Adapter contract: the online pool is the existing ``Uint8Replay`` API
    (``valid_indices()`` and ``sample(batch_size, indices=...)``); the sealed
    teacher dataset supplies matching n-step fields. Keep returned provenance
    fields beside the numeric batch, and pass executed native ``action`` values
    unchanged to the learner. To insert teacher rows into ``Uint8Replay``, adapt
    them as ``common_adapter.Transition`` records with unique episode IDs and
    real reconstructed next observations; let that replay own latest-frame
    compression rather than storing four uint8 channels per row here.
    """

    _QUOTAS = {"online-only": (64, 0), "teacher-replay": (48, 16)}

    def __init__(
        self,
        online_replay: Any,
        teacher_dataset: TeacherDataset | None,
        *,
        mode: str = "teacher-replay",
        seed: int = 0,
        online_source_id: str = "online",
    ):
        if mode not in self._QUOTAS:
            raise ValueError("mode must be 'online-only' or 'teacher-replay'")
        if not callable(getattr(online_replay, "valid_indices", None)) or not callable(
            getattr(online_replay, "sample", None)
        ):
            raise TypeError("online_replay must provide Uint8Replay sampling methods")
        if mode == "teacher-replay" and (
            not isinstance(teacher_dataset, TeacherDataset) or not teacher_dataset.sealed
        ):
            raise ValueError("teacher-replay mode requires a sealed TeacherDataset")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if not isinstance(online_source_id, str) or not online_source_id:
            raise ValueError("online_source_id must be a non-empty string")
        self.online_replay = online_replay
        self.teacher_dataset = teacher_dataset
        self.mode = mode
        self.online_source_id = online_source_id
        self.rng = np.random.default_rng(seed)
        self.n_step = int(online_replay.n_step)
        self.gamma = float(online_replay.gamma)
        TeacherDataset._validate_return(self.gamma, self.n_step)
        self._online_valid_indices: list[int] = []
        self._online_seen_next = 0
        self._online_seen_oldest = int(getattr(online_replay, "oldest_sequence", 0))
        self._teacher_valid_indices = (
            teacher_dataset.valid_indices(gamma=self.gamma, n_step=self.n_step)
            if mode == "teacher-replay" and teacher_dataset is not None else []
        )

    def _refresh_online_valid_indices(self) -> list[int]:
        """Incrementally validate only newly completed n-step windows between wraps."""
        next_sequence = int(getattr(self.online_replay, "_next_sequence"))
        oldest_sequence = int(getattr(self.online_replay, "oldest_sequence"))
        if next_sequence == self._online_seen_next and oldest_sequence == self._online_seen_oldest:
            return self._online_valid_indices
        if oldest_sequence != self._online_seen_oldest or next_sequence < self._online_seen_next:
            self._online_valid_indices = self.online_replay.valid_indices()
        else:
            first_candidate = max(oldest_sequence, self._online_seen_next - self.n_step)
            known = set(self._online_valid_indices)
            for sequence in range(first_candidate, next_sequence):
                if sequence not in known and self.online_replay._build_n_step(sequence) is not None:
                    self._online_valid_indices.append(sequence)
                    known.add(sequence)
            self._online_valid_indices.sort()
        self._online_seen_next = next_sequence
        self._online_seen_oldest = oldest_sequence
        return self._online_valid_indices

    @staticmethod
    def _choose(pool: Sequence[int], count: int, name: str, rng: np.random.Generator) -> list[int]:
        if count == 0:
            return []
        if len(pool) < count:
            raise ValueError(f"{name} pool has {len(pool)} valid rows; {count} are required")
        return [int(index) for index in rng.choice(np.asarray(pool), size=count, replace=False)]

    def sample(self, batch_size: int = 64) -> dict[str, Any]:
        """Return a shuffled 64-row batch with local indices and source tags.

        ``source`` is numeric (0 online, 1 teacher); ``sources`` and
        ``source_tags`` retain human-readable provenance. Sampling is without
        replacement within each call and follows the constructor seed.
        """
        if type(batch_size) is not int or batch_size != 64:
            raise ValueError("the study sampler uses fixed 64-row minibatches")
        online_count, teacher_count = self._QUOTAS[self.mode]
        online_valid = self._refresh_online_valid_indices()
        online_indices = self._choose(online_valid, online_count, "online", self.rng)
        teacher_indices: list[int] = []
        if teacher_count:
            assert self.teacher_dataset is not None
            teacher_indices = self._choose(
                self._teacher_valid_indices, teacher_count, "teacher", self.rng
            )

        online_rows = None
        if online_count:
            online_rows = self.online_replay.sample(online_count, indices=online_indices)
        teacher_rows = None
        if teacher_count:
            assert self.teacher_dataset is not None
            teacher_rows = self.teacher_dataset.sample_at_indices(
                teacher_indices, gamma=self.gamma, n_step=self.n_step
            )

        core: dict[str, np.ndarray] = {}
        for name in _CORE_BATCH_FIELDS:
            parts = []
            if online_rows is not None:
                value = online_rows[name]
                if name in {"observation", "next_observation"}:
                    value = value.astype(np.float32) / np.float32(255.0)
                parts.append(value)
            if teacher_rows is not None:
                parts.append(teacher_rows[name])
            core[name] = np.concatenate(parts, axis=0)

        source_names = np.asarray(["online"] * online_count + ["teacher"] * teacher_count,
                                  dtype="U7")
        source_codes = np.asarray([0] * online_count + [1] * teacher_count, dtype=np.uint8)
        source_indices = np.asarray(online_indices + teacher_indices, dtype=np.int64)
        tags: list[dict[str, Any]] = []
        if online_rows is not None:
            for offset, index in enumerate(online_indices):
                tags.append({
                    "source": "online",
                    "source_id": self.online_source_id,
                    "source_index": index,
                    "episode_id": int(online_rows["episode_id"][offset]),
                })
        if teacher_rows is not None:
            for offset, index in enumerate(teacher_indices):
                tags.append({
                    "source": "teacher",
                    "source_id": str(teacher_rows["source_id"][offset]),
                    "source_actor_sha256": str(teacher_rows["source_actor_sha256"][offset]),
                    "geometry_id": str(teacher_rows["geometry_id"][offset]),
                    "track_id": int(teacher_rows["track_id"][offset]),
                    "episode_index": int(teacher_rows["episode_index"][offset]),
                    "episode_id": str(teacher_rows["episode_id"][offset]),
                    "step": int(teacher_rows["step"][offset]),
                    "source_index": index,
                })
        order = self.rng.permutation(batch_size)
        result: dict[str, Any] = {name: value[order] for name, value in core.items()}
        result["source"] = source_codes[order]
        result["sources"] = source_names[order]
        result["source_indices"] = source_indices[order]
        result["source_tags"] = tuple(tags[int(index)] for index in order)
        return result

    def state_dict(self) -> dict[str, Any]:
        """Persist sampler RNG and immutable-source identity for exact continuation."""
        return {
            "format": "haic-drq-two-source-replay-v1",
            "mode": self.mode,
            "online_source_id": self.online_source_id,
            "teacher_dataset_digest": (
                self.teacher_dataset.digest if self.teacher_dataset is not None else None
            ),
            "gamma": self.gamma,
            "n_step": self.n_step,
            "rng_state": self.rng.bit_generator.state,
            "online_valid_indices": list(self._online_valid_indices),
            "online_seen_next": self._online_seen_next,
            "online_seen_oldest": self._online_seen_oldest,
            "teacher_valid_count": len(self._teacher_valid_indices),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "format", "mode", "online_source_id", "teacher_dataset_digest",
            "gamma", "n_step", "rng_state", "online_valid_indices",
            "online_seen_next", "online_seen_oldest", "teacher_valid_count",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("unsupported two-source replay checkpoint fields")
        if state["format"] != "haic-drq-two-source-replay-v1":
            raise ValueError("unsupported two-source replay checkpoint format")
        digest = self.teacher_dataset.digest if self.teacher_dataset is not None else None
        if (
            state["mode"] != self.mode
            or state["online_source_id"] != self.online_source_id
            or state["teacher_dataset_digest"] != digest
            or float(state["gamma"]) != self.gamma
            or int(state["n_step"]) != self.n_step
            or int(state["teacher_valid_count"]) != len(self._teacher_valid_indices)
        ):
            raise ValueError("two-source replay checkpoint does not match frozen sources or return")
        online_valid = self.online_replay.valid_indices()
        stored_online_valid = [int(index) for index in state["online_valid_indices"]]
        if online_valid != stored_online_valid:
            raise ValueError("two-source replay checkpoint valid online rows do not match replay contents")
        next_sequence = int(getattr(self.online_replay, "_next_sequence"))
        oldest_sequence = int(getattr(self.online_replay, "oldest_sequence"))
        if int(state["online_seen_next"]) != next_sequence or int(state["online_seen_oldest"]) != oldest_sequence:
            raise ValueError("two-source replay checkpoint online cursor does not match replay contents")
        self._online_valid_indices = online_valid
        self._online_seen_next = next_sequence
        self._online_seen_oldest = oldest_sequence
        self.rng.bit_generator.state = state["rng_state"]


# Keep the study-only fork and learner update next to the immutable replay API
# without changing the public legacy DrQv2Agent checkpoint/update behavior.
from haic.algorithms.drq_v2.teacher_study import (  # noqa: E402
    GeometryPoolRNG,
    RNGStreams,
    audit_source_actor_pair,
    file_sha256,
    find_sampled_track_env,
    fork_from_source,
    learner_rng_state,
    restore_learner_rng_state,
    update_from_mixture,
)
