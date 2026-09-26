"""Validate a sealed TRAIN-only episode corpus before Dreamer replay ingestion."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from io import BytesIO
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np

from common_adapter import Transition
from dreamer_v3 import Uint8SequenceReplay
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def replay_from_dataset_bytes(
    data: bytes,
    *,
    archive_sha256: str,
    dataset_digest: str,
    source_id: str,
    source_actor_sha256: str,
    allowed_cells: Sequence[tuple[int, int]],
    excluded_seeds: Sequence[int],
    capacity: int,
) -> tuple[Uint8SequenceReplay, dict[str, Any]]:
    """Import complete, source-pinned TRAIN episodes without collecting new roads.

    The caller must verify the archive's repository-relative TRAIN path and its
    frozen protocol/seed-audit receipt before reading bytes. No data is truncated
    to fit replay: a smaller capacity or undeclared cell fails closed.
    """
    if not isinstance(data, bytes) or not data:
        raise ValueError("a nonempty immutable dataset archive is required")
    for name, digest in (
        ("archive_sha256", archive_sha256),
        ("dataset_digest", dataset_digest),
        ("source_actor_sha256", source_actor_sha256),
    ):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if hashlib.sha256(data).hexdigest() != archive_sha256:
        raise ValueError("dataset archive hash does not match the frozen protocol")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must identify exactly one collection arm")
    if type(capacity) is not int or capacity < 1:
        raise ValueError("replay capacity must be a positive integer")
    if not allowed_cells or any(
        not isinstance(cell, tuple) or len(cell) != 2
        or any(type(value) is not int for value in cell)
        or cell[0] < 1 or not 0 <= cell[1] < 2**32
        for cell in allowed_cells
    ):
        raise ValueError("allowed_cells must list positive tracks and uint32 geometry seeds")
    if len(set(allowed_cells)) != len(allowed_cells) or len({seed for _, seed in allowed_cells}) != len(allowed_cells):
        raise ValueError("TRAIN cells must use distinct geometry seeds across tracks")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in excluded_seeds):
        raise ValueError("excluded_seeds must be uint32 IDs")
    if {seed for _, seed in allowed_cells} & set(excluded_seeds):
        raise ValueError("TRAIN cells overlap reserved or held-out geometry seed IDs")

    # Bound compressed input before TeacherDataset eagerly materializes all arrays.
    try:
        with ZipFile(BytesIO(data)) as archive:
            if sum(member.file_size for member in archive.infolist()) > (
                capacity * (2 * 84 * 84 + 128) + 65536
            ):
                raise ValueError("dataset archive exceeds replay capacity memory bound")
            with archive.open("actions.npy") as action_rows:
                if np.lib.format.read_magic(action_rows) != (1, 0):
                    raise ValueError("unsupported dataset actions header")
                shape, _, dtype = np.lib.format.read_array_header_1_0(action_rows)
                if len(shape) != 2 or shape[1] != 3 or dtype != np.dtype(np.float32):
                    raise ValueError("dataset actions have an invalid header")
                if not 1 <= shape[0] <= capacity:
                    raise ValueError("replay capacity would silently overwrite frozen TRAIN episodes")
    except (BadZipFile, KeyError, OSError, EOFError) as exc:
        raise ValueError("invalid serialized teacher dataset archive") from exc

    dataset = TeacherDataset.from_bytes(data, expected_digest=dataset_digest)
    if dataset.transition_count > capacity:
        raise ValueError("replay capacity would silently overwrite frozen TRAIN episodes")
    allowed = set(allowed_cells)
    finished_cells: set[tuple[int, int]] = set()
    seen_cells: set[tuple[int, int]] = set()
    for episode in dataset.episodes:
        if not episode.complete or episode.source_id != source_id or episode.source_actor_sha256 != source_actor_sha256:
            raise ValueError("dataset mixes incomplete episodes or source actor identities")
        if not episode.geometry_id.isdecimal() or str(int(episode.geometry_id)) != episode.geometry_id:
            raise ValueError("episode geometry_id must be a canonical uint32 seed")
        cell = (episode.track_id, int(episode.geometry_id))
        if cell not in allowed:
            raise ValueError("dataset episode belongs to an undeclared TRAIN cell")
        if not (episode.terminated[-1] or episode.truncated[-1]):
            raise ValueError("complete episode must end in a real terminated/truncated boundary")
        for step in range(episode.steps):
            expected_terminal = (
                bool(episode.terminated[step]) or bool(episode.finished[step])
                or episode.retire_reasons[step] in {"crash", "off_track", "out_of_bounds"}
            )
            if bool(episode.terminal[step]) != expected_terminal:
                raise ValueError("terminal label contradicts executed episode outcome")
        seen_cells.add(cell)
        if episode.finished[-1]:
            finished_cells.add(cell)

    replay = Uint8SequenceReplay(capacity=capacity)
    for episode_index, episode in enumerate(dataset.episodes):
        for step in range(episode.steps):
            replay.add(Transition(
                observation=episode.observation(step),
                action=episode.actions[step],
                applied_action=episode.applied_actions[step],
                reward=float(episode.rewards[step]),
                next_observation=episode.observation(step + 1),
                terminated=bool(episode.terminated[step]),
                truncated=bool(episode.truncated[step]),
                terminal=bool(episode.terminal[step]),
                info={
                    "finished": bool(episode.finished[step]),
                    "progress": float(episode.progress[step]),
                    "damage": float(episode.damage[step]),
                    "retire_reason": episode.retire_reasons[step],
                },
                episode_id=episode_index,
                step=step,
            ))
    return replay, {
        "archive_sha256": archive_sha256,
        "dataset_digest": dataset_digest,
        "source_id": source_id,
        "source_actor_sha256": source_actor_sha256,
        "decisions": dataset.transition_count,
        "episodes": len(dataset.episodes),
        "covered_cells": sorted(seen_cells),
        "distinct_finished_cells": len(finished_cells),
    }
