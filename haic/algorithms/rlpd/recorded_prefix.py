"""Read-only original-G0 action/observation binder, before any environment reset.

The production entry point pins the *original* protocol, manifest and ledger
bytes, not caller-produced replacement hashes. The selected trace SHA comes
only from that ledger. No actor is invoked and diagnostic telemetry is never
used as an actor input. A captured Prefix must be checked here before replay.

G0 recorded pixels and commands, not full accessible-state signatures. Even a
successful observational comparison cannot prove hidden Box2D state equality.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from haic.algorithms.rlpd.prefix_parity import INITIAL_HASH, ParityError, Prefix


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ACTOR_ID = re.compile(r"[a-z0-9-]{1,48}\Z")
_TRACE_KEYS = frozenset({
    "initial_stack", "observation_frame", "next_frame", "proposed_native_action",
    "executed_native_action", "commanded_official_action", "raw_official_action",
    "summed_reward", "new_tiles", "directed_delta", "centerline_distance_m",
    "centerline_fraction", "speed_m_s", "heading_error_rad", "x_m", "y_m",
    "progress", "damage", "contact", "off_track_counter", "finish_qualified",
    "finished", "terminated", "truncated", "finish_phase", "raw_frames",
    "raw_finish_phase_bits",
})


@dataclass(frozen=True)
class _FrozenSource:
    protocol_path: str
    protocol_sha256: str
    manifest_path: str
    manifest_sha256: str
    ledger_path: str
    ledger_sha256: str


_ORIGINAL_G0 = _FrozenSource(
    "experiments/rlpd-g0-completion-v1.json",
    "6d4c50aaa03a68bad27bafc8eccf99f2df4377908a21fd31ec7729736f29946b",
    "runs/20260926-rlpd-g0-completion-v1/manifest.json",
    "52abe52381009f392b849c475afea7fa15aec93aaf8001fbc5eb14611ecbf358",
    "runs/20260926-rlpd-g0-completion-v1/cells.jsonl",
    "924d3ea84b6a393ab962e3b2821d49b11e2c5b23c21f898caf4b795fd58d9559",
)


@dataclass(frozen=True)
class RecordedStep:
    official_action: bytes  # Exactly 12 float32 bytes passed to the official wrapper.
    raw_action: bytes  # Exactly 24 float64 bytes sent to each raw CarRacing step.
    raw_frame_count: int
    observation_sha256: str
    next_observation_sha256: str
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class RecordedPrefix:
    track_id: int
    geometry_seed: int
    actor_id: str
    anchor_decisions: int
    road_centerline_sha256: str
    actor_sha256: str
    actor_source_sha256: str
    actor_export_protocol_sha256: str
    protocol_sha256: str
    manifest_sha256: str
    ledger_sha256: str
    trace_sha256: str
    initial_observation_sha256: str
    steps: tuple[RecordedStep, ...]

    @property
    def actions(self) -> tuple[bytes, ...]:
        return tuple(step.official_action for step in self.steps)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ParityError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ParityError(f"{label} is outside its integer range")
    return value


def _pinned_bytes(root: Path, relative: str, digest: str, prefix: str) -> bytes:
    _sha(digest, relative)
    if not isinstance(relative, str) or "\\" in relative or not relative:
        raise ParityError("unsafe source path")
    parts = relative.split("/")
    if len(parts) < 2 or parts[0] != prefix or any(part in ("", ".", "..") for part in parts):
        raise ParityError(f"source path must stay under {prefix}/")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ParityError(f"symlink in source path: {relative}")
    if not current.is_file():
        raise ParityError(f"missing frozen source: {relative}")
    payload = current.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ParityError(f"frozen SHA mismatch: {relative}")
    return payload


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ParityError(f"invalid JSON constant: {value}")


def _json(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_keys,
                       parse_constant=_bad_constant)
    if not isinstance(value, dict):
        raise ParityError("frozen JSON must be an object")
    return value


def _observation_sha(pixels: np.ndarray) -> str:
    observation = np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255.0))
    return hashlib.sha256(observation.tobytes()).hexdigest()


def _trace_steps(payload: bytes, count: int, anchor: int) -> tuple[str, tuple[RecordedStep, ...], float, int]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != _TRACE_KEYS or len(archive.files) != len(_TRACE_KEYS):
            raise ParityError("G0 trace has missing or unexpected columns")
        arrays = {key: archive[key] for key in _TRACE_KEYS}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ParityError("object arrays are not permitted in G0 traces")
    initial = arrays["initial_stack"]
    if initial.shape != (4, 84, 84) or initial.dtype != np.uint8:
        raise ParityError("G0 initial_stack must be uint8 (4,84,84)")
    for key, array in arrays.items():
        if key != "initial_stack" and (array.ndim == 0 or array.shape[0] != count):
            raise ParityError(f"G0 {key} has the wrong step count")
    for key in ("observation_frame", "next_frame"):
        if arrays[key].shape != (count, 84, 84) or arrays[key].dtype != np.uint8:
            raise ParityError(f"G0 {key} must be uint8 (N,84,84)")
    official = arrays["commanded_official_action"]
    raw = arrays["raw_official_action"]
    if official.shape != (count, 3) or official.dtype != np.float32:
        raise ParityError("G0 commanded_official_action must be float32 (N,3)")
    if raw.shape != (count, 3) or raw.dtype != np.float64:
        raise ParityError("G0 raw_official_action must be float64 (N,3)")
    if (not np.isfinite(official).all() or not np.isfinite(raw).all()
            or np.any(official < np.array([-1, 0, 0], dtype=np.float32))
            or np.any(official > 1)
            or np.ascontiguousarray(raw).tobytes()
            != np.ascontiguousarray(official.astype(np.float64)).tobytes()):
        raise ParityError("G0 raw command is not the exact widened official action")
    rewards = arrays["summed_reward"]
    if rewards.shape != (count,) or rewards.dtype != np.float64 or not np.isfinite(rewards).all():
        raise ParityError("G0 summed_reward must be finite float64 (N,)")
    for key in ("terminated", "truncated"):
        if arrays[key].shape != (count,) or arrays[key].dtype != np.bool_:
            raise ParityError(f"G0 {key} must be bool (N,)")
    frames = arrays["raw_frames"]
    if (frames.shape != (count,) or frames.dtype.kind not in "iu"
            or np.any(frames < 1) or np.any(frames > 4)):
        raise ParityError("G0 raw_frames must be integers in [1,4]")
    if (np.any(arrays["terminated"][:-1] | arrays["truncated"][:-1])
            or not (arrays["terminated"][-1] or arrays["truncated"][-1])):
        raise ParityError("G0 complete trace continues after terminal or lacks final terminal")
    if anchor >= count or np.any(arrays["terminated"][:anchor] | arrays["truncated"][:anchor]):
        raise ParityError("anchor_decisions must stop within a nonterminal G0 prefix")

    stack = initial.copy()
    initial_sha = _observation_sha(stack)
    steps: list[RecordedStep] = []
    for index in range(count):
        if not np.array_equal(stack[-1], arrays["observation_frame"][index]):
            raise ParityError(f"G0 observation_frame is off by one at step {index}")
        obs_sha = _observation_sha(stack)
        stack[:-1] = stack[1:]
        stack[-1] = arrays["next_frame"][index]
        next_sha = _observation_sha(stack)
        if index < anchor:
            steps.append(RecordedStep(
                np.ascontiguousarray(official[index]).tobytes(),
                np.ascontiguousarray(raw[index]).tobytes(), int(frames[index]),
                obs_sha, next_sha, float(rewards[index]),
                bool(arrays["terminated"][index]), bool(arrays["truncated"][index]),
            ))
    return initial_sha, tuple(steps), float(math.fsum(float(x) for x in rewards)), int(frames.sum())


def _bind_frozen_source(
    root: Path, source: _FrozenSource, *, track_id: int, geometry_seed: int,
    actor_id: str, anchor_decisions: int,
) -> RecordedPrefix:
    """Internal seam for synthetic pinned-file tests; not a source override on the public API."""
    _integer(track_id, "track_id", minimum=1, maximum=4)
    _integer(geometry_seed, "geometry_seed", minimum=0, maximum=2**32 - 1)
    _integer(anchor_decisions, "anchor_decisions", minimum=1, maximum=2000)
    if not isinstance(actor_id, str) or _ACTOR_ID.fullmatch(actor_id) is None:
        raise ParityError("actor_id is malformed")
    root = Path(root).resolve()
    protocol = _json(_pinned_bytes(root, source.protocol_path, source.protocol_sha256, "experiments"))
    manifest = _json(_pinned_bytes(root, source.manifest_path, source.manifest_sha256, "runs"))
    ledger = _pinned_bytes(root, source.ledger_path, source.ledger_sha256, "runs")
    if (protocol.get("format") != "haic-rlpd-g0-diagnostic-v1"
            or protocol.get("status") != "frozen" or protocol.get("partition") != "TRAIN"
            or protocol.get("frame_skip") != 4 or protocol.get("max_steps") != 2000
            or protocol.get("reward_shaping") is not False
            or protocol.get("collision_penalty") != 0.0
            or protocol.get("interventions") is not False or protocol.get("learner_updates") != 0):
        raise ParityError("not the original frozen TRAIN G0 control protocol")
    if (manifest.get("format") != "haic-rlpd-g0-diagnostic-result-v1"
            or manifest.get("role") != "TRAIN-only-failure-diagnostic"
            or manifest.get("ranked") is not False
            or manifest.get("protocol_path") != source.protocol_path
            or manifest.get("protocol_sha256") != source.protocol_sha256
            or manifest.get("cells_sha256") != source.ledger_sha256
            or manifest.get("geometry_audit_sha256") != protocol.get("geometry_audit_sha256")):
        raise ParityError("G0 manifest does not bind protocol and ledger")
    cells, actors = protocol.get("cells"), protocol.get("actors")
    if (not isinstance(cells, list) or not cells or not isinstance(actors, list) or len(actors) != 2
            or any(not isinstance(c, dict) or c.get("partition") != "TRAIN"
                   or c.get("obstacles") is not True
                   or type(c.get("track_id")) is not int or type(c.get("geometry_seed")) is not int
                   for c in cells)
            or len({(c["track_id"], c["geometry_seed"]) for c in cells}) != len(cells)
            or any(not isinstance(a, dict) or not isinstance(a.get("id"), str)
                   or _ACTOR_ID.fullmatch(a["id"]) is None
                   or a.get("action_mode") != "exported_tanh_mean"
                   or _SHA.fullmatch(str(a.get("sha256"))) is None for a in actors)
            or actors[0]["id"] == actors[1]["id"]):
        raise ParityError("G0 protocol has malformed paired TRAIN cells/actors")
    rows = [_json(line.encode("utf-8")) for line in ledger.decode("utf-8").splitlines()]
    if (len(rows) != len(cells) * len(actors)
            or manifest.get("cell_count") != len(rows)
            or manifest.get("geometry_count") != len(cells)):
        raise ParityError("G0 ledger does not cover the frozen paired cell schedule")
    selected = selected_actor = None
    for index, row in enumerate(rows):
        cell, actor = cells[index // len(actors)], actors[index % len(actors)]
        expected_path = (f"traces/seed-{cell['geometry_seed']}-track-"
                         f"{cell['track_id']}-{actor['id']}.npz")
        if (row.get("partition") != "TRAIN" or row.get("track_id") != cell["track_id"]
                or row.get("geometry_seed") != cell["geometry_seed"]
                or row.get("actor_id") != actor["id"] or row.get("actor_sha256") != actor["sha256"]
                or row.get("trace_path") != expected_path
                or _SHA.fullmatch(str(row.get("trace_sha256"))) is None
                or _SHA.fullmatch(str(row.get("road_centerline_sha256"))) is None
                or type(row.get("steps")) is not int or not 1 <= row["steps"] <= 2000):
            raise ParityError("G0 ledger row differs from protocol cell/actor schedule")
        if index % 2 and rows[index - 1]["road_centerline_sha256"] != row["road_centerline_sha256"]:
            raise ParityError("paired G0 actors have different road centerline SHAs")
        if (cell["track_id"], cell["geometry_seed"], actor["id"]) == (track_id, geometry_seed, actor_id):
            selected, selected_actor = row, actor
    if selected is None or selected_actor is None:
        raise ParityError("requested track/geometry/actor is not a frozen G0 ledger row")
    if anchor_decisions >= selected["steps"]:
        raise ParityError("cannot anchor after the G0 terminal decision")
    actor_sha = _sha(selected_actor["sha256"], "actor SHA")
    _pinned_bytes(root, selected_actor["path"], actor_sha, "runs")
    trace_path = (Path(source.manifest_path).parent / selected["trace_path"]).as_posix()
    trace_sha = _sha(selected["trace_sha256"], "ledger trace SHA")
    trace = _pinned_bytes(root, trace_path, trace_sha, "runs")
    initial_sha, steps, reward_sum, raw_frames = _trace_steps(trace, selected["steps"], anchor_decisions)
    if (type(selected.get("driven_raw_frames")) is not int
            or raw_frames != selected["driven_raw_frames"]
            or type(selected.get("raw_reward_sum")) not in (int, float)
            or not math.isfinite(selected["raw_reward_sum"])
            or not math.isclose(reward_sum, selected["raw_reward_sum"], rel_tol=0, abs_tol=1e-6)):
        raise ParityError("G0 trace reward/raw-frame totals differ from ledger")
    return RecordedPrefix(
        track_id, geometry_seed, actor_id, anchor_decisions,
        _sha(selected["road_centerline_sha256"], "G0 road SHA"), actor_sha,
        _sha(selected_actor["source_sha256"], "actor source SHA"),
        _sha(selected_actor["export_protocol_sha256"], "actor export protocol SHA"),
        source.protocol_sha256, source.manifest_sha256, source.ledger_sha256,
        trace_sha, initial_sha, steps,
    )


def bind_original_g0_prefix(
    root: Path, *, expected_protocol_sha256: str, expected_manifest_sha256: str,
    expected_ledger_sha256: str, track_id: int, geometry_seed: int,
    actor_id: str, anchor_decisions: int,
) -> RecordedPrefix:
    """Bind independently predeclared original-G0 files/cell/anchor before reset.

    A caller cannot swap all three expected hashes for a self-made dataset.
    The original trace hash is taken exclusively from the verified ledger.
    """
    if (expected_protocol_sha256 != _ORIGINAL_G0.protocol_sha256
            or expected_manifest_sha256 != _ORIGINAL_G0.manifest_sha256
            or expected_ledger_sha256 != _ORIGINAL_G0.ledger_sha256):
        raise ParityError("expected G0 hashes are not the original frozen source")
    return _bind_frozen_source(root, _ORIGINAL_G0, track_id=track_id,
                               geometry_seed=geometry_seed, actor_id=actor_id,
                               anchor_decisions=anchor_decisions)


def verify_captured_prefix(recorded: RecordedPrefix, captured: Prefix) -> None:
    """PURE pre-replay check of source official actions and observational signatures.

    The full (N,4) track SHA remains an internal replay invariant; the XY-only
    centerline SHA uses G0's exact canonicalization to bind that track to its
    separately recorded road. Hidden Box2D state still needs empirical replay.
    """
    if (not isinstance(recorded, RecordedPrefix) or not isinstance(captured, Prefix)
            or recorded.track_id != captured.track_id or recorded.geometry_seed != captured.seed
            or not recorded.steps or len(recorded.steps) != recorded.anchor_decisions
            or len(captured.actions) != len(recorded.steps)
            or len(captured.steps) != len(recorded.steps)
            or captured.policy_action_sha256 is not None
            and len(captured.policy_action_sha256) != len(recorded.steps)):
        raise ParityError("captured prefix has wrong source cell or decision count")
    _sha(recorded.road_centerline_sha256, "G0 road SHA")
    _sha(captured.road_sha256, "captured full-track SHA")
    states = (dict(captured.initial.state), *(dict(step.state) for step in captured.steps))
    if any(state.get("road_sha256") != captured.road_sha256 for state in states):
        raise ParityError("captured full-track SHA drifted within the prefix")
    if any(state.get("road_centerline_sha256") != recorded.road_centerline_sha256
           for state in states):
        raise ParityError("captured road centerline SHA differs from original G0")
    if any(state.get("outer.max_episode_steps") != 2000
           or state.get("inner.max_episode_steps") != 8200 for state in states):
        raise ParityError("captured original G0 decision/raw deadline differs")
    if (captured.initial.observation_sha256 != recorded.initial_observation_sha256
            or captured.initial.action_sha256 != hashlib.sha256(b"").hexdigest()
            or captured.initial.prefix_sha256 != INITIAL_HASH):
        raise ParityError("captured initial observation/action history differs from G0")
    prior_sha = recorded.initial_observation_sha256
    history = INITIAL_HASH
    for index, (source, action, signature) in enumerate(
        zip(recorded.steps, captured.actions, captured.steps)
    ):
        if not isinstance(action, bytes) or action != source.official_action:
            raise ParityError(f"G0 official action bytes differ at step {index}")
        if (len(action) != 12 or not isinstance(source.raw_action, bytes) or len(source.raw_action) != 24
                or np.frombuffer(source.raw_action, dtype=np.float64).tobytes()
                != np.frombuffer(action, dtype=np.float32).astype(np.float64).tobytes()):
            raise ParityError(f"G0 raw command widening differs at step {index}")
        history = hashlib.sha256(bytes.fromhex(history) + action).hexdigest()
        if (source.observation_sha256 != prior_sha
                or signature.observation_sha256 != source.next_observation_sha256
                or signature.action_sha256 != hashlib.sha256(action).hexdigest()
                or signature.prefix_sha256 != history
                or signature.raw_frame_count != source.raw_frame_count
                or signature.raw_action_sha256
                != hashlib.sha256(source.raw_action * source.raw_frame_count).hexdigest()
                or type(signature.reward) is not float or signature.reward != source.reward
                or type(signature.terminated) is not bool or signature.terminated != source.terminated
                or type(signature.truncated) is not bool or signature.truncated != source.truncated
                or signature.terminated or signature.truncated
                or captured.policy_action_sha256 is not None
                and captured.policy_action_sha256[index] != signature.action_sha256):
            raise ParityError(f"captured G0 observation/reward/flags/actions differ at step {index}")
        prior_sha = source.next_observation_sha256
