"""Strict, injectable RLPD reset-and-official-action prefix parity gate.

The caller supplies the original unmodified ``train.build_env`` environment with
reward shaping, collision penalty, and action adapters disabled. This module does
not construct an environment, select anchors or branches, or dispatch interventions.
Accessible-state equality cannot prove equality of hidden Box2D contacts/solver
state. A separate frozen TRAIN interaction protocol, coverage gate and harmed-finish
controls are required before using a verified prefix for any driving experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np


class ParityError(ValueError):
    """Missing source state, malformed evidence, or a divergent prefix."""


INITIAL_HASH = hashlib.sha256(b"rlpd-official-prefix-v1:reset").hexdigest()
TRACKER_FIELDS = frozenset({
    "center", "forward", "half_width", "half_depth", "qualification_ratio",
    "departed_start_area", "crossing_from_back", "previous_longitudinal",
    "previous_lateral", "qualified_time_s", "candidate_crossing_time_s",
    "finish_time_s",
})
EFFECT_FIELDS = frozenset({"grip_multiplier", "engine_multiplier", "steering_multiplier"})
STEP_INFO_FIELDS = frozenset({
    "collision", "damage", "damage_effects", "progress", "finish_qualified",
    "finish_qualified_time_s", "finish_time_s", "finished", "retire_reason",
})


@dataclass(frozen=True)
class Signature:
    action_sha256: str
    prefix_sha256: str
    raw_action_sha256: str
    raw_frame_count: int
    observation_sha256: str
    raw_image_sha256: str
    reward: float | None
    terminated: bool | None
    truncated: bool | None
    info: tuple[tuple[str, Any], ...]
    state: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class Prefix:
    track_id: int
    seed: int
    road_sha256: str
    initial: Signature
    actions: tuple[bytes, ...]
    steps: tuple[Signature, ...]
    policy_action_sha256: tuple[str, ...] | None = None


def _field(obj: Any, name: str, path: str) -> Any:
    try:
        return getattr(obj, name)
    except AttributeError as exc:
        raise ParityError(f"missing {path}.{name}") from exc


def _number(value: Any, path: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ParityError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ParityError(f"{path} must be finite")
    return result


def _integer(value: Any, path: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ParityError(f"{path} must be an integer")
    return int(value)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ParityError(f"{path} must be a boolean")
    return bool(value)


def _optional_number(value: Any, path: str) -> float | None:
    return None if value is None else _number(value, path)


def _vector(value: Any, path: str) -> tuple[float, float]:
    try:
        if len(value) != 2:
            raise ParityError(f"{path} must have two coordinates")
        return (_number(value[0], path + "[0]"), _number(value[1], path + "[1]"))
    except (TypeError, IndexError) as exc:
        raise ParityError(f"{path} must have two coordinates") from exc


def _image(value: Any, shape: tuple[int, ...], dtype: Any, path: str) -> tuple[str, np.ndarray]:
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
        raise ParityError(f"{path} must be {np.dtype(dtype)} {shape}")
    if not np.isfinite(value).all():
        raise ParityError(f"{path} must be finite")
    if dtype == np.float32 and (np.any(value < 0.0) or np.any(value > 1.0)):
        raise ParityError(f"{path} must be in [0, 1]")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(), value


def _action_bytes(action: Any) -> bytes:
    if not isinstance(action, np.ndarray) or action.shape != (3,) or action.dtype != np.float32:
        raise ParityError("official action must be float32 (3,)")
    if not np.isfinite(action).all() or not (
        -1.0 <= action[0] <= 1.0 and 0.0 <= action[1] <= 1.0 and 0.0 <= action[2] <= 1.0
    ):
        raise ParityError("official action is outside finite official bounds")
    return np.ascontiguousarray(action).tobytes()


def _step_with_raw_tap(env: Any, action: np.ndarray) -> tuple[Any, ...]:
    """Check what the original float64 wrapper sends to each raw physics step."""
    _, _, _, _, raw = _chain(env)
    expected = np.ascontiguousarray(action.astype(np.float64)).tobytes()
    commands: list[bytes] = []
    original = raw.step

    def observed(raw_action: Any):
        if (not isinstance(raw_action, np.ndarray) or raw_action.shape != (3,)
                or raw_action.dtype != np.float64 or raw_action.tobytes() != expected):
            raise ParityError("raw CarRacing.step action differs from widened official float32")
        result = original(raw_action)
        commands.append(np.ascontiguousarray(raw_action).tobytes())
        return result

    raw.step = observed
    try:
        result = env.step(action)
    finally:
        raw.step = original
    if not 1 <= len(commands) <= 4:
        raise ParityError("original frame skip used an unexpected raw action count")
    return (*result, tuple(commands))


def _chain(env: Any) -> tuple[Any, Any, Any, Any, Any]:
    names = ("TimeLimit", "HaicTrack", "CarEnvironment", "TimeLimit", "CarRacing")
    chain = [env]
    for name in names[:-1]:
        current = chain[-1]
        if type(current).__name__ != name:
            raise ParityError(f"expected {name}, found {type(current).__name__}")
        chain.append(_field(current, "env", name))
    if type(chain[-1]).__name__ != names[-1]:
        raise ParityError("expected raw CarRacing without extra wrappers")
    outer, haic, wrapper, inner, raw = chain
    if (_integer(_field(wrapper, "_skip_frames", "CarEnvironment"), "frame_skip") != 4
            or _integer(_field(wrapper, "_stack_frames", "CarEnvironment"), "stack_frames") != 4
            or _integer(_field(wrapper, "_no_operation", "CarEnvironment"), "warmup") != 50
            or _integer(_field(wrapper, "max_off_track_steps", "CarEnvironment"), "off_track_limit") != 100
            or _boolean(_field(haic, "obstacles", "HaicTrack"), "obstacles") is not True):
        raise ParityError("not the original four-frame RLPD wrapper configuration")
    outer_max = _integer(_field(outer, "_max_episode_steps", "outer TimeLimit"), "outer max")
    inner_max = _integer(_field(inner, "_max_episode_steps", "inner TimeLimit"), "inner max")
    if outer_max <= 0 or inner_max != outer_max * 4 + 200:
        raise ParityError("original decision/raw-frame horizons differ")
    return outer, haic, wrapper, inner, raw


def _effects(effects: Any, path: str) -> tuple[float, float, float]:
    if not hasattr(effects, "__dict__") or set(vars(effects)) != EFFECT_FIELDS:
        raise ParityError(f"{path} has missing or unrecognized damage-effect fields")
    return tuple(_number(_field(effects, key, path), path + "." + key) for key in (
        "grip_multiplier", "engine_multiplier", "steering_multiplier"
    ))


def _particle(particle: Any, path: str) -> tuple[Any, ...]:
    if not hasattr(particle, "__dict__") or set(vars(particle)) != {"color", "ttl", "poly", "grass"}:
        raise ParityError(f"{path} has missing or unrecognized skid-particle fields")
    color = _field(particle, "color", path)
    poly = _field(particle, "poly", path)
    if (not isinstance(color, (tuple, list)) or len(color) != 3
            or not isinstance(poly, (tuple, list))):
        raise ParityError(f"{path} has malformed color or polyline")
    return (tuple(_integer(channel, path + ".color") for channel in color),
            _number(_field(particle, "ttl", path), path + ".ttl"),
            tuple(_vector(point, path + ".poly") for point in poly),
            _boolean(_field(particle, "grass", path), path + ".grass"))


def _state(env: Any, observation: np.ndarray, track_id: int, seed: int) -> tuple[dict[str, Any], str]:
    outer, haic, wrapper, inner, raw = _chain(env)
    state: dict[str, Any] = {}
    for path, value in (("haic.track_id", _field(haic, "track_id", "HaicTrack")),
                        ("haic.seed", _field(haic, "seed", "HaicTrack")),
                        ("raw.track_id", _field(raw, "track_id", "CarRacing")),
                        ("raw.track_seed", _field(raw, "track_seed", "CarRacing"))):
        state[path] = _integer(value, path)
    if any(state[key] != expected for key, expected in (
        ("haic.track_id", track_id), ("raw.track_id", track_id),
        ("haic.seed", seed), ("raw.track_seed", seed)
    )):
        raise ParityError("track ID or geometry seed differs from requested same-road cell")
    points = np.asarray(_field(raw, "track", "CarRacing"), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 4 or not len(points) or not np.isfinite(points).all():
        raise ParityError("raw.track must be a finite (N, 4) original centerline")
    road_sha = hashlib.sha256(np.ascontiguousarray(points, dtype="<f8").tobytes()).hexdigest()
    state["road_sha256"] = road_sha
    state["road_centerline_sha256"] = hashlib.sha256(
        np.ascontiguousarray(points[:, 2:4], dtype="<f8").tobytes()
    ).hexdigest()
    road = _field(raw, "road", "CarRacing")
    if not isinstance(road, (list, tuple)) or len(road) != len(points):
        raise ParityError("raw.road must have one tile body per centerline entry")
    bitmap = []
    friction = []
    fixtures = []
    body_transforms = []
    collision_filters = []
    for idx, tile in enumerate(road):
        if _integer(_field(tile, "idx", "road tile"), "road.idx") != idx:
            raise ParityError("road tile order/identity changed")
        body_transforms.append((
            _vector(_field(tile, "position", "road tile"), "road tile position"),
            _number(_field(tile, "angle", "road tile"), "road tile angle"),
        ))
        bitmap.append(_boolean(_field(tile, "road_visited", "road tile"), "road_visited"))
        friction.append(_number(_field(tile, "road_friction", "road tile"), "road_friction"))
        tile_fixtures = _field(tile, "fixtures", "road tile")
        if len(tile_fixtures) != 1:
            raise ParityError("road tile must have exactly one polygon fixture")
        fixture = tile_fixtures[0]
        vertices = _field(_field(fixture, "shape", "road tile"), "vertices", "road fixture")
        if len(vertices) != 4:
            raise ParityError("road tile polygon must have four vertices")
        fixtures.append((
            _boolean(_field(fixture, "sensor", "road fixture"), "road fixture sensor"),
            tuple(_vector(vertex, "road fixture vertex") for vertex in vertices),
        ))
        filter_data = _field(fixture, "filterData", "road fixture")
        collision_filters.append(tuple(
            _integer(_field(filter_data, name, "road fixture filter"),
                     "road fixture filter." + name)
            for name in ("categoryBits", "maskBits", "groupIndex")
        ))
    state["raw.road_visited"] = tuple(bitmap)
    state["raw.road_friction"] = tuple(friction)
    state["raw.road_fixtures"] = tuple(fixtures)
    state["raw.road_body_transforms"] = tuple(body_transforms)
    state["raw.road_collision_filters"] = tuple(collision_filters)
    state["raw.tile_visited_count"] = _integer(_field(raw, "tile_visited_count", "CarRacing"), "tile count")
    if state["raw.tile_visited_count"] != sum(bitmap):
        raise ParityError("tile count does not agree with complete road-visited bitmap")
    obstacles = _field(raw, "obstacles", "CarRacing")
    if not isinstance(obstacles, (tuple, list)):
        raise ParityError("raw.obstacles must be an ordered collection")
    obstacle_state = []
    for idx, body in enumerate(obstacles):
        path = f"obstacle[{idx}]"
        data = _field(body, "userData", path)
        if not hasattr(data, "__dict__") or set(vars(data)) != {"hit"}:
            raise ParityError(path + " has missing or unrecognized contact fields")
        fixtures = _field(body, "fixtures", path)
        if len(fixtures) != 1:
            raise ParityError(path + " must have exactly one frozen circle fixture")
        fixture = fixtures[0]
        radius = _number(_field(_field(fixture, "shape", path), "radius", path + ".shape"),
                         path + ".shape.radius")
        if radius <= 0 or _boolean(_field(fixture, "sensor", path), path + ".sensor"):
            raise ParityError(path + " must have a positive non-sensor obstacle radius")
        obstacle_state.append((_vector(_field(body, "position", path), path + ".position"),
                                _number(_field(body, "angle", path), path + ".angle"),
                                _boolean(_field(data, "hit", path + ".userData"), path + ".hit"),
                                radius))
    state["raw.obstacles"] = tuple(obstacle_state)
    for key in ("t", "reward", "prev_reward"):
        state["raw." + key] = _number(_field(raw, key, "CarRacing"), "raw." + key)
    for key in ("finish_qualified_time_s", "finish_time_s"):
        state["raw." + key] = _optional_number(_field(raw, key, "CarRacing"), "raw." + key)
    state["raw.new_lap"] = _boolean(_field(raw, "new_lap", "CarRacing"), "raw.new_lap")
    state["raw._collision_this_step"] = _boolean(
        _field(raw, "_collision_this_step", "CarRacing"), "raw._collision_this_step"
    )
    tracker = _field(raw, "finish_line_tracker", "CarRacing")
    if tracker is None or not hasattr(tracker, "__dict__") or set(vars(tracker)) != TRACKER_FIELDS:
        raise ParityError("finish tracker has missing or unrecognized mutable fields")
    for key in sorted(TRACKER_FIELDS):
        value = _field(tracker, key, "finish tracker")
        path = "finish_tracker." + key
        if key in ("center", "forward"):
            state[path] = _vector(value, path)
        elif key in ("departed_start_area", "crossing_from_back"):
            state[path] = _boolean(value, path)
        elif key.endswith("_time_s"):
            state[path] = _optional_number(value, path)
        else:
            state[path] = _number(value, path)
    car = _field(raw, "car", "CarRacing")
    hull = _field(car, "hull", "car")
    wheels = _field(car, "wheels", "car")
    if not isinstance(wheels, (list, tuple)) or len(wheels) != 4:
        raise ParityError("car must have four accessible wheels")
    for name, body in (("hull", hull), *((f"wheel[{i}]", w) for i, w in enumerate(wheels))):
        for key in ("position", "linearVelocity"):
            state[f"{name}.{key}"] = _vector(_field(body, key, name), f"{name}.{key}")
        for key in ("angle", "angularVelocity"):
            state[f"{name}.{key}"] = _number(_field(body, key, name), f"{name}.{key}")
    state["car.fuel_spent"] = _number(_field(car, "fuel_spent", "car"), "car.fuel_spent")
    state["car.grass_friction_multiplier"] = _number(
        _field(car, "grass_friction_multiplier", "car"), "car.grass_friction_multiplier"
    )
    for key in EFFECT_FIELDS:
        state["car." + key] = _number(_field(car, key, "car"), "car." + key)
    particles = _field(car, "particles", "car")
    if not isinstance(particles, (list, tuple)):
        raise ParityError("car.particles must be an ordered collection")
    state["car.particles"] = tuple(_particle(p, f"car.particles[{i}]") for i, p in enumerate(particles))
    for idx, wheel in enumerate(wheels):
        name = f"wheel[{idx}]"
        for key in ("gas", "brake", "steer", "phase", "omega", "wheel_rad"):
            state[f"{name}.{key}"] = _number(_field(wheel, key, name), f"{name}.{key}")
        joint = _field(wheel, "joint", name)
        for key in ("angle", "motorSpeed"):
            state[f"{name}.joint.{key}"] = _number(_field(joint, key, name + ".joint"), f"{name}.joint.{key}")
        tiles = _field(wheel, "tiles", name)
        if not isinstance(tiles, set):
            raise ParityError(f"{name}.tiles must be a set of road bodies")
        indices = tuple(sorted(_integer(_field(tile, "idx", name + ".tiles"), name + ".tiles.idx") for tile in tiles))
        if any(index < 0 or index >= len(road) or road[index] not in tiles for index in indices):
            raise ParityError(f"{name}.tiles contains an unknown road body")
        state[f"{name}.tiles"] = indices
        skid_start = _field(wheel, "skid_start", name)
        skid_particle = _field(wheel, "skid_particle", name)
        state[f"{name}.skid_start"] = None if skid_start is None else _vector(skid_start, name + ".skid_start")
        state[f"{name}.skid_particle"] = (None if skid_particle is None else
                                          _particle(skid_particle, name + ".skid_particle"))
    damage = _field(wrapper, "damage", "CarEnvironment")
    state["damage.value"] = _number(_field(damage, "damage", "DamageManager"), "damage.value")
    state["damage.effects"] = _effects(_field(damage, "effects", "DamageManager"), "damage.effects")
    state["wrapper.off_track_counter"] = _integer(
        _field(wrapper, "off_track_counter", "CarEnvironment"), "off_track_counter"
    )
    stack_sha, stack = _image(_field(wrapper, "stack_state", "CarEnvironment"),
                              (4, 84, 84), np.float32, "wrapper.stack_state")
    if not np.array_equal(stack, observation):
        raise ParityError("observation differs from internal frame-stack history")
    state["wrapper.stack_sha256"] = stack_sha
    for label, limit in (("outer", outer), ("inner", inner)):
        maximum = _integer(_field(limit, "_max_episode_steps", label), label + ".max")
        elapsed = _integer(_field(limit, "_elapsed_steps", label), label + ".elapsed")
        if not 0 <= elapsed <= maximum:
            raise ParityError(label + " TimeLimit has invalid elapsed steps")
        state[label + ".max_episode_steps"] = maximum
        state[label + ".elapsed_steps"] = elapsed
        state[label + ".remaining_steps"] = maximum - elapsed
    return state, road_sha


def snapshot(
    env: Any, observation: np.ndarray, *, track_id: int, seed: int,
    action: np.ndarray | None = None, prefix_sha256: str = INITIAL_HASH,
    reward: Any = None, terminated: Any = None, truncated: Any = None,
    info: Any = None, raw_commands: tuple[bytes, ...] | None = None,
) -> Signature:
    """Capture only the explicitly recognized accessible state, with no actor inputs."""
    if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id <= 0:
        raise ParityError("track_id must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ParityError("seed must be an unsigned 32-bit geometry seed")
    obs_sha, observation = _image(observation, (4, 84, 84), np.float32, "observation")
    outer, _, _, _, raw = _chain(env)
    state, _ = _state(env, observation, track_id, seed)
    raw_sha, _ = _image(_field(raw, "state", "CarRacing"), (96, 96, 3), np.uint8, "raw.state")
    if not isinstance(info, dict):
        raise ParityError("reset/step info must be a dictionary")
    if action is None:
        if (reward is not None or terminated is not None or truncated is not None
                or raw_commands is not None
                or set(info) != {"track_id", "seed"}
                or _integer(info["track_id"], "reset track_id") != track_id
                or _integer(info["seed"], "reset seed") != seed
                or state["outer.elapsed_steps"] != 0 or prefix_sha256 != INITIAL_HASH):
            raise ParityError("initial reset identity/time/action is not original")
        action_sha, fields = hashlib.sha256(b"").hexdigest(), tuple(sorted(info.items()))
        raw_action_sha, raw_frame_count = hashlib.sha256(b"").hexdigest(), 0
        step_reward = term = trunc = None
    else:
        action_sha = hashlib.sha256(_action_bytes(action)).hexdigest()
        if (not isinstance(raw_commands, tuple) or not 1 <= len(raw_commands) <= 4
                or any(not isinstance(command, bytes) or len(command) != 24
                       or command != np.ascontiguousarray(action.astype(np.float64)).tobytes()
                       for command in raw_commands)):
            raise ParityError("observed raw float64 commands do not match the official action")
        raw_action_sha = hashlib.sha256(b"".join(raw_commands)).hexdigest()
        raw_frame_count = len(raw_commands)
        if set(info) != STEP_INFO_FIELDS:
            raise ParityError("step info has missing or unrecognized original fields")
        if info["retire_reason"] not in (None, "crash", "off_track"):
            raise ParityError("unrecognized retirement reason")
        fields = tuple(sorted({
            "collision": _boolean(info["collision"], "info.collision"),
            "damage": _number(info["damage"], "info.damage"),
            "damage_effects": _effects(info["damage_effects"], "info.damage_effects"),
            "progress": _number(info["progress"], "info.progress"),
            "finish_qualified": _boolean(info["finish_qualified"], "info.finish_qualified"),
            "finish_qualified_time_s": _optional_number(info["finish_qualified_time_s"], "info.qualified_time"),
            "finish_time_s": _optional_number(info["finish_time_s"], "info.finish_time"),
            "finished": _boolean(info["finished"], "info.finished"),
            "retire_reason": info["retire_reason"],
        }.items()))
        step_reward = _number(reward, "reward")
        term = _boolean(terminated, "terminated")
        trunc = _boolean(truncated, "truncated")
        if state["outer.elapsed_steps"] < 1:
            raise ParityError("step did not advance original outer TimeLimit")
    return Signature(action_sha, prefix_sha256, raw_action_sha, raw_frame_count,
                      obs_sha, raw_sha, step_reward,
                      term, trunc, fields, tuple(sorted(state.items())))


def _assert_same(expected: Signature, actual: Signature, step: str) -> None:
    for name in ("action_sha256", "prefix_sha256", "raw_action_sha256", "raw_frame_count",
                 "observation_sha256", "raw_image_sha256",
                 "reward", "terminated", "truncated"):
        if getattr(expected, name) != getattr(actual, name):
            raise ParityError(f"{step}.{name} differs")
    for name in ("info", "state"):
        left, right = dict(getattr(expected, name)), dict(getattr(actual, name))
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right or left[key] != right[key]:
                raise ParityError(f"{step}.{name}.{key} differs")


def _policy_hash(actor: Any, observation: np.ndarray, action_bytes: bytes) -> str:
    if not callable(getattr(actor, "act", None)):
        raise ParityError("stateless policy must expose Agent.act(observation)")
    output = _action_bytes(actor.act(observation.copy()))
    if output != action_bytes:
        raise ParityError("frozen Agent.act official policy-action parity differs")
    return hashlib.sha256(output).hexdigest()


def _reset_actor(actor: Any, observation: np.ndarray) -> None:
    if not callable(getattr(actor, "reset", None)):
        raise ParityError("frozen stateless Agent must expose reset(observation)")
    actor.reset(observation.copy())


def capture_prefix(
    env: Any, *, track_id: int, seed: int, official_actions: Sequence[np.ndarray],
    actor: Any = None,
) -> Prefix:
    """Capture a caller-authorized original reset and *executed* official prefix."""
    _chain(env)
    action_data = tuple(_action_bytes(action) for action in official_actions)
    if not action_data:
        raise ParityError("a prefix must contain at least one executed decision")
    observation, info = env.reset()
    initial = snapshot(env, observation, track_id=track_id, seed=seed, info=info)
    road_sha = dict(initial.state)["road_sha256"]
    if actor is not None:
        _reset_actor(actor, observation)
    actions: list[bytes] = []
    steps: list[Signature] = []
    policy_hashes: list[str] = []
    prefix_sha = INITIAL_HASH
    for index, action_bytes in enumerate(action_data):
        if actor is not None:
            policy_hashes.append(_policy_hash(actor, observation, action_bytes))
        action = np.frombuffer(action_bytes, dtype=np.float32).copy()
        prefix_sha = hashlib.sha256(bytes.fromhex(prefix_sha) + action_bytes).hexdigest()
        observation, reward, terminated, truncated, info, raw_commands = _step_with_raw_tap(env, action)
        signature = snapshot(env, observation, track_id=track_id, seed=seed, action=action,
                             prefix_sha256=prefix_sha, reward=reward, terminated=terminated,
                             truncated=truncated, info=info, raw_commands=raw_commands)
        if dict(signature.state)["outer.elapsed_steps"] != index + 1 or dict(signature.state)["road_sha256"] != road_sha:
            raise ParityError("capture changed the original decision clock or road")
        actions.append(action_bytes)
        steps.append(signature)
        if (terminated or truncated) and index + 1 != len(action_data):
            raise ParityError("recorded actions continue beyond a terminal transition")
    return Prefix(track_id, seed, road_sha, initial, tuple(actions), tuple(steps),
                  tuple(policy_hashes) if actor is not None else None)


def _compare_replay_prefix(env: Any, expected: Prefix, *, actor: Any = None) -> np.ndarray:
    """Internal accessible-state comparator; NOT source-bound branch authorization.

    Replay the stored float32 official bytes DIRECTLY through env.step, without
    translating to/from native actions. No branch is dispatched by this module.
    """
    if not isinstance(expected, Prefix) or not expected.actions or len(expected.actions) != len(expected.steps):
        raise ParityError("expected a nonempty, complete prefix")
    if expected.policy_action_sha256 is not None and (
        actor is None or len(expected.policy_action_sha256) != len(expected.actions)
    ):
        raise ParityError("frozen actor required for recorded policy-action parity")
    prefix_sha = INITIAL_HASH
    if expected.initial.action_sha256 != hashlib.sha256(b"").hexdigest() or expected.initial.prefix_sha256 != prefix_sha:
        raise ParityError("initial action/history hash differs")
    for index, (data, signature) in enumerate(zip(expected.actions, expected.steps)):
        if not isinstance(data, bytes) or len(data) != 12:
            raise ParityError("recorded official action must be exactly 12 bytes")
        _action_bytes(np.frombuffer(data, dtype=np.float32))
        prefix_sha = hashlib.sha256(bytes.fromhex(prefix_sha) + data).hexdigest()
        if signature.action_sha256 != hashlib.sha256(data).hexdigest() or signature.prefix_sha256 != prefix_sha:
            raise ParityError(f"action/history byte hash differs at step {index}")
    _chain(env)
    observation, info = env.reset()
    initial = snapshot(env, observation, track_id=expected.track_id, seed=expected.seed, info=info)
    _assert_same(expected.initial, initial, "initial")
    if dict(initial.state)["road_sha256"] != expected.road_sha256:
        raise ParityError("reset road hash differs")
    if actor is not None:
        _reset_actor(actor, observation)
    prefix_sha = INITIAL_HASH
    for index, (data, signature) in enumerate(zip(expected.actions, expected.steps)):
        if actor is not None:
            policy_sha = _policy_hash(actor, observation, data)
            if expected.policy_action_sha256 is not None and policy_sha != expected.policy_action_sha256[index]:
                raise ParityError(f"policy-action hash differs at step {index}")
        action = np.frombuffer(data, dtype=np.float32).copy()
        prefix_sha = hashlib.sha256(bytes.fromhex(prefix_sha) + data).hexdigest()
        observation, reward, terminated, truncated, info, raw_commands = _step_with_raw_tap(env, action)
        actual = snapshot(env, observation, track_id=expected.track_id, seed=expected.seed,
                          action=action, prefix_sha256=prefix_sha, reward=reward,
                          terminated=terminated, truncated=truncated, info=info,
                          raw_commands=raw_commands)
        _assert_same(signature, actual, f"step[{index}]")
        if dict(actual.state)["outer.elapsed_steps"] != index + 1:
            raise ParityError("original outer decision clock drifted")
        if terminated or truncated:
            raise ParityError("terminal prefix is not an intervention anchor")
    return observation.copy()


def verify_replay_prefix(
    env: Any, expected: Prefix, *, source_root: Any = None,
    actor_id: str | None = None, actor: Any = None,
) -> np.ndarray:
    """Bind exact original G0 bytes before replay or releasing an anchor."""
    if source_root is None or actor_id is None:
        raise ParityError("independently frozen original G0 source binding is required")
    if not isinstance(expected, Prefix) or not expected.actions:
        raise ParityError("expected a nonempty recorded prefix")
    from haic.algorithms.rlpd.recorded_prefix import (
        _ORIGINAL_G0, bind_original_g0_prefix, verify_captured_prefix,
    )

    recorded = bind_original_g0_prefix(
        source_root, expected_protocol_sha256=_ORIGINAL_G0.protocol_sha256,
        expected_manifest_sha256=_ORIGINAL_G0.manifest_sha256,
        expected_ledger_sha256=_ORIGINAL_G0.ledger_sha256,
        track_id=expected.track_id, geometry_seed=expected.seed,
        actor_id=actor_id, anchor_decisions=len(expected.actions),
    )
    verify_captured_prefix(recorded, expected)
    return _compare_replay_prefix(env, expected, actor=actor)
