"""Dependency-light stateful smoothing for bounded vector actions.

``ActionSmoother`` accepts ordinary Python sequences, so callers can keep their
existing NumPy or Torch boundary outside this module.  The requested action is
clipped to the configured bounds before either exponential smoothing or a
per-component maximum delta is applied.  The result is clipped once more and
stored as the state used by the next call.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

__all__ = [
    "ACTION_BOUNDS_HIGH",
    "ACTION_BOUNDS_LOW",
    "DEFAULT_ACTION_SMOOTHING",
    "ActionSmoother",
    "StatefulActionSmoother",
    "action_smoothing_fingerprint",
    "build_action_smoother",
    "canonical_action_smoothing",
    "normalize_action_smoothing",
]


_SCHEMA_VERSION = 1
ACTION_BOUNDS_LOW = (-1.0, 0.0, 0.0)
ACTION_BOUNDS_HIGH = (1.0, 1.0, 1.0)


DEFAULT_ACTION_SMOOTHING = {
    "method": "none",
    "alpha": 0.35,
    "max_delta": None,
    "initial_action": [0.0, 0.0, 0.0],
}


def _is_iterable(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must contain real numbers")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite real numbers") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def _parameter_spec(value: Any, name: str) -> float | tuple[float, ...]:
    if _is_iterable(value):
        values = tuple(_finite_float(item, name) for item in value)
        if not values:
            raise ValueError(f"{name} must not be empty")
        return values
    return _finite_float(value, name)


def _spec_length(value: float | tuple[float, ...]) -> int | None:
    return len(value) if isinstance(value, tuple) else None


def _json_spec(value: float | tuple[float, ...] | None):
    if isinstance(value, tuple):
        return list(value)
    return value


def _canonical_action_vector(value: Any, name: str) -> list[float]:
    if not _is_iterable(value):
        raise TypeError(f"{name} must be a vector")
    values = [_finite_float(item, name) for item in value]
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def canonical_action_smoothing(
    method: str = "none",
    alpha: Real | Any | None = 0.35,
    initial_action: Any = (0.0, 0.0, 0.0),
    max_delta: Real | Any | None = None,
) -> dict[str, Any]:
    """Build a validated, JSON-compatible action smoothing configuration."""

    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.strip().lower()
    method_aliases = {"ema": "alpha", "exponential": "alpha"}
    method = method_aliases.get(method, method)
    if method not in {"none", "alpha", "max_delta"}:
        raise ValueError("method must be one of: none, alpha, max_delta")

    if alpha is None:
        canonical_alpha = None
    else:
        canonical_alpha = _parameter_spec(alpha, "alpha")
        alpha_values = (
            canonical_alpha
            if isinstance(canonical_alpha, tuple)
            else (canonical_alpha,)
        )
        if any(value < 0.0 or value > 1.0 for value in alpha_values):
            raise ValueError("alpha must be between 0 and 1")

    if max_delta is None:
        canonical_delta = None
    else:
        canonical_delta = _parameter_spec(max_delta, "max_delta")
        delta_values = (
            canonical_delta
            if isinstance(canonical_delta, tuple)
            else (canonical_delta,)
        )
        if any(value < 0.0 for value in delta_values):
            raise ValueError("max_delta must be non-negative")

    if method == "none":
        # Keep the no-op path explicit and independent of the intervention's
        # requested alpha.  This makes old configs normalize to identical
        # semantics instead of accidentally smoothing a control arm.
        canonical_alpha = 1.0
    elif method == "alpha":
        if canonical_delta is not None:
            raise ValueError("max_delta is not valid when method is alpha")
    elif method == "max_delta":
        # The alpha default exists for the public helper's ergonomic signature,
        # but it is not part of max-delta semantics or its fingerprint.
        canonical_alpha = None
    if method == "alpha" and canonical_alpha is None:
        raise ValueError("alpha is required when method is alpha")
    if method == "max_delta" and canonical_delta is None:
        raise ValueError("max_delta is required when method is max_delta")
    if method == "none" and canonical_delta is not None:
        raise ValueError("max_delta is not valid when method is none")

    return {
        "method": method,
        "alpha": _json_spec(canonical_alpha),
        "max_delta": _json_spec(canonical_delta),
        "initial_action": _canonical_action_vector(initial_action, "initial_action"),
    }


def normalize_action_smoothing(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize an optional smoothing config to a deterministic dictionary."""

    if config is None:
        config = DEFAULT_ACTION_SMOOTHING
    if not isinstance(config, Mapping):
        raise TypeError("action smoothing config must be a mapping")
    values = dict(DEFAULT_ACTION_SMOOTHING)
    values.update(config)
    return canonical_action_smoothing(
        method=values["method"],
        alpha=values.get("alpha"),
        initial_action=values["initial_action"],
        max_delta=values.get("max_delta"),
    )


def action_smoothing_fingerprint(config: Mapping[str, Any] | None = None) -> str:
    """Return the SHA-256 fingerprint of a normalized smoothing config."""

    encoded = json.dumps(
        normalize_action_smoothing(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_action_smoother(
    config: Mapping[str, Any] | None = None,
) -> "StatefulActionSmoother":
    """Build an initialized smoother with the official action bounds."""

    normalized = normalize_action_smoothing(config)
    method = normalized["method"]
    initial_action = normalized["initial_action"]
    if len(initial_action) != len(ACTION_BOUNDS_LOW):
        raise ValueError(
            f"initial_action must have length {len(ACTION_BOUNDS_LOW)}"
        )
    kwargs = {
        "low": ACTION_BOUNDS_LOW,
        "high": ACTION_BOUNDS_HIGH,
        "method": method,
        "initial_action": initial_action,
    }
    if method == "max_delta":
        kwargs["max_delta"] = normalized["max_delta"]
    else:
        kwargs["alpha"] = normalized["alpha"]
    return StatefulActionSmoother(**kwargs)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


class StatefulActionSmoother:
    """Smooth bounded vector actions with independent batch state.

    Exactly one of ``alpha`` and ``max_delta`` must be supplied.  With
    ``alpha``, each output is ``previous + alpha * (clipped_request -
    previous)``.  With ``max_delta``, each output moves toward the clipped
    request by at most that component's configured delta.  The first request
    for each state slot is clipped and returned unchanged.

    ``low``, ``high``, ``alpha`` and ``max_delta`` may each be scalars or
    one-dimensional iterables.  Scalars broadcast over the action dimension;
    the dimension can be inferred from the first action when all specifications
    are scalar.  ``smooth`` returns a Python list (or list of lists for a
    batch), which keeps this module free of numerical framework dependencies.
    """

    schema_version = _SCHEMA_VERSION

    def __init__(
        self,
        low: Real | Any = -1.0,
        high: Real | Any = 1.0,
        *,
        alpha: Real | Any | None = None,
        max_delta: Real | Any | None = None,
        method: str | None = None,
        initial_action: Any | None = None,
    ):
        if method is not None:
            if not isinstance(method, str):
                raise TypeError("method must be a string")
            method = {"ema": "alpha", "exponential": "alpha"}.get(
                method.strip().lower(), method.strip().lower()
            )
            if method not in {"none", "alpha", "max_delta"}:
                raise ValueError("method must be one of: none, alpha, max_delta")
            if method == "none":
                if max_delta is not None or (alpha is not None and alpha != 1.0):
                    raise ValueError("method none cannot define alpha or max_delta")
                alpha = 1.0
            elif method == "alpha" and max_delta is not None:
                raise ValueError("method alpha cannot define max_delta")
            elif method == "max_delta" and alpha is not None:
                raise ValueError("method max_delta cannot define alpha")
            elif method == "max_delta" and max_delta is None:
                raise ValueError("max_delta is required when method is max_delta")
            elif method == "alpha" and alpha is None:
                raise ValueError("alpha is required when method is alpha")

        if (alpha is None) == (max_delta is None):
            raise ValueError("exactly one of alpha or max_delta must be provided")

        self._low_spec = _parameter_spec(low, "low")
        self._high_spec = _parameter_spec(high, "high")
        self._alpha_spec = None if alpha is None else _parameter_spec(alpha, "alpha")
        self._max_delta_spec = (
            None if max_delta is None else _parameter_spec(max_delta, "max_delta")
        )
        self._mode = method or ("alpha" if alpha is not None else "max_delta")

        lengths = {
            length
            for length in (
                _spec_length(self._low_spec),
                _spec_length(self._high_spec),
                _spec_length(self._alpha_spec) if self._alpha_spec is not None else None,
                _spec_length(self._max_delta_spec)
                if self._max_delta_spec is not None
                else None,
            )
            if length is not None
        }
        if len(lengths) > 1:
            raise ValueError("low, high, and smoothing parameters must have one shared length")

        self._action_dim = next(iter(lengths), None)
        self._low: tuple[float, ...] | None = None
        self._high: tuple[float, ...] | None = None
        self._alpha: tuple[float, ...] | None = None
        self._max_delta: tuple[float, ...] | None = None
        self._resolve_specs()

        self._batch_size: int | None = None
        self._state: list[tuple[float, ...] | None] | None = None
        if initial_action is not None:
            self.reset(initial_action=initial_action)

    @property
    def mode(self) -> str:
        """Return ``"alpha"`` or ``"max_delta"``."""

        return self._mode

    @property
    def action_dim(self) -> int | None:
        """Return the configured or inferred action dimension."""

        return self._action_dim

    @property
    def batch_size(self) -> int | None:
        """Return the current batch size, if one has been established."""

        return self._batch_size

    @property
    def state(self):
        """Return an immutable snapshot of per-slot previous actions."""

        if self._state is None:
            return None
        return tuple(None if row is None else tuple(row) for row in self._state)

    @property
    def config(self) -> dict[str, Any]:
        """Return the JSON-compatible smoothing configuration."""

        return {
            "type": type(self).__name__,
            "schema_version": self.schema_version,
            "method": self._mode,
            "low": _json_spec(self._low_spec),
            "high": _json_spec(self._high_spec),
            "alpha": _json_spec(self._alpha_spec),
            "max_delta": _json_spec(self._max_delta_spec),
        }

    @property
    def config_fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the canonical configuration."""

        encoded = json.dumps(
            self.config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fingerprint(self) -> str:
        """Return :attr:`config_fingerprint` as a method for caller convenience."""

        return self.config_fingerprint

    def _resolve_specs(self) -> None:
        if self._action_dim is None:
            if isinstance(self._low_spec, float) and isinstance(self._high_spec, float):
                if self._low_spec > self._high_spec:
                    raise ValueError("low must not exceed high")
            self._validate_smoothing_specs()
            return

        dimension = self._action_dim
        self._low = self._expand(self._low_spec, dimension, "low")
        self._high = self._expand(self._high_spec, dimension, "high")
        if any(low > high for low, high in zip(self._low, self._high)):
            raise ValueError("low must not exceed high")

        self._alpha = (
            self._expand(self._alpha_spec, dimension, "alpha")
            if self._alpha_spec is not None
            else None
        )
        self._max_delta = (
            self._expand(self._max_delta_spec, dimension, "max_delta")
            if self._max_delta_spec is not None
            else None
        )
        self._validate_smoothing_specs()

    @staticmethod
    def _expand(
        value: float | tuple[float, ...] | None,
        dimension: int,
        name: str,
    ) -> tuple[float, ...]:
        if value is None:
            raise ValueError(f"{name} is required")
        if isinstance(value, tuple):
            if len(value) != dimension:
                raise ValueError(f"{name} must have length {dimension}")
            return value
        return (value,) * dimension

    def _validate_smoothing_specs(self) -> None:
        values = self._alpha if self._action_dim is not None else self._alpha_spec
        if values is not None:
            values_to_check = values if isinstance(values, tuple) else (values,)
            if any(value < 0.0 or value > 1.0 for value in values_to_check):
                raise ValueError("alpha must be between 0 and 1")

        values = self._max_delta if self._action_dim is not None else self._max_delta_spec
        if values is not None:
            values_to_check = values if isinstance(values, tuple) else (values,)
            if any(value < 0.0 for value in values_to_check):
                raise ValueError("max_delta must be non-negative")

    def _ensure_dimension(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("action dimension must be positive")
        if self._action_dim is not None and self._action_dim != dimension:
            raise ValueError(f"action dimension must be {self._action_dim}")
        if self._action_dim is None:
            self._action_dim = dimension
            self._resolve_specs()

    @staticmethod
    def _vector(value: Any, name: str) -> tuple[float, ...]:
        if not _is_iterable(value):
            raise TypeError(f"{name} must be a vector or batch of vectors")
        values = tuple(_finite_float(item, name) for item in value)
        if not values:
            raise ValueError(f"{name} must not be empty")
        return values

    def _coerce_actions(
        self,
        actions: Any,
        expected_batch: int | None = None,
    ) -> tuple[list[tuple[float, ...]], bool]:
        if not _is_iterable(actions):
            raise TypeError("actions must be a vector or batch of vectors")
        values = list(actions)
        if not values:
            raise ValueError("actions must not be empty")

        batched = _is_iterable(values[0])
        if batched:
            rows = [self._vector(row, "actions") for row in values]
        else:
            rows = [self._vector(values, "actions")]

        if expected_batch is not None and len(rows) != expected_batch:
            raise ValueError(f"actions must contain {expected_batch} vector(s)")
        dimension = len(rows[0])
        if any(len(row) != dimension for row in rows):
            raise ValueError("all action vectors must have the same length")
        self._ensure_dimension(dimension)
        return rows, batched

    def _clip_row(self, row: tuple[float, ...]) -> tuple[float, ...]:
        assert self._low is not None and self._high is not None
        return tuple(
            low if value < low else high if value > high else value
            for value, low, high in zip(row, self._low, self._high)
        )

    def smooth(self, actions: Any):
        """Smooth one action vector or a batch of vectors and update state."""

        rows, batched = self._coerce_actions(actions, self._batch_size)
        batch_size = len(rows)
        if self._batch_size is None:
            self._batch_size = batch_size
        if self._state is None:
            self._state = [None] * batch_size

        assert self._alpha is not None or self._max_delta is not None
        output_rows: list[tuple[float, ...]] = []
        for row, previous in zip(rows, self._state):
            target = self._clip_row(row)
            if previous is None:
                result = target
            elif self._alpha is not None:
                result = tuple(
                    self._clip_value(
                        old + alpha * (new - old), low, high
                    )
                    for old, new, alpha, low, high in zip(
                        previous, target, self._alpha, self._low, self._high
                    )
                )
            else:
                assert self._max_delta is not None
                result = tuple(
                    self._clip_value(
                        old + self._bounded_delta(new - old, limit), low, high
                    )
                    for old, new, limit, low, high in zip(
                        previous, target, self._max_delta, self._low, self._high
                    )
                )
            output_rows.append(result)

        self._state = output_rows
        if batched:
            return [list(row) for row in output_rows]
        return list(output_rows[0])

    __call__ = smooth
    apply = smooth

    @property
    def last_action(self):
        """Return a mutable-list snapshot of the previous action state."""

        if self._state is None:
            return None
        if self._batch_size == 1:
            row = self._state[0]
            return None if row is None else list(row)
        return [None if row is None else list(row) for row in self._state]

    @staticmethod
    def _bounded_delta(delta: float, limit: float) -> float:
        if delta < -limit:
            return -limit
        if delta > limit:
            return limit
        return delta

    @staticmethod
    def _clip_value(value: float, low: float, high: float) -> float:
        if value < low:
            return low
        if value > high:
            return high
        return value

    def reset(
        self,
        initial_action: Any | None = None,
        *,
        batch_size: int | None = None,
        mask: Any | None = None,
        indices: Any | None = None,
    ) -> None:
        """Clear state, optionally initialize it, or reset selected batch slots.

        ``mask`` is a boolean batch mask and ``indices`` is an iterable of
        batch positions.  Either selectively clears existing slots to make
        their next action a first action, which is useful when vectorized
        environments finish independently.
        """

        if mask is not None and indices is not None:
            raise ValueError("mask and indices are mutually exclusive")
        if initial_action is not None and (mask is not None or indices is not None):
            raise ValueError("initial_action cannot be combined with selective reset")

        if mask is not None or indices is not None:
            if self._batch_size is None:
                if batch_size is None:
                    raise ValueError("batch_size is required for selective reset")
                self._batch_size = _positive_int(batch_size, "batch_size")
            elif batch_size is not None and _positive_int(batch_size, "batch_size") != self._batch_size:
                raise ValueError(f"batch_size must be {self._batch_size}")
            if self._state is None:
                self._state = [None] * self._batch_size

            if mask is not None:
                selected = self._mask_indices(mask, self._batch_size)
            else:
                selected = self._indices(indices, self._batch_size)
            for index in selected:
                self._state[index] = None
            return

        requested_batch = (
            None if batch_size is None else _positive_int(batch_size, "batch_size")
        )
        if initial_action is not None:
            rows, _ = self._coerce_actions(initial_action, requested_batch)
            self._batch_size = len(rows)
            self._state = [self._clip_row(row) for row in rows]
            return

        self._batch_size = requested_batch
        self._state = None if requested_batch is None else [None] * requested_batch

    @staticmethod
    def _mask_indices(mask: Any, batch_size: int) -> tuple[int, ...]:
        if not _is_iterable(mask):
            raise TypeError("mask must be an iterable of booleans")
        values = list(mask)
        if len(values) != batch_size:
            raise ValueError(f"mask must have length {batch_size}")
        selected = []
        for index, value in enumerate(values):
            if isinstance(value, bool):
                enabled = value
            elif isinstance(value, Integral) and int(value) in (0, 1):
                enabled = bool(value)
            else:
                raise TypeError("mask must contain only booleans")
            if enabled:
                selected.append(index)
        return tuple(selected)

    @staticmethod
    def _indices(indices: Any, batch_size: int) -> tuple[int, ...]:
        if not _is_iterable(indices):
            raise TypeError("indices must be an iterable of integers")
        selected = []
        for value in indices:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("indices must contain integers")
            index = int(value)
            if index < 0 or index >= batch_size:
                raise IndexError(f"batch index out of range: {index}")
            if index in selected:
                raise ValueError("indices must not contain duplicates")
            selected.append(index)
        return tuple(selected)

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration and state snapshot."""

        previous = None
        if self._state is not None:
            previous = [None if row is None else list(row) for row in self._state]
        return {
            "schema_version": self.schema_version,
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "state": {
                "action_dim": self._action_dim,
                "batch_size": self._batch_size,
                "previous": previous,
            },
        }

    to_dict = state_dict

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore state after verifying schema and configuration identity."""

        if not isinstance(payload, Mapping):
            raise TypeError("state payload must be a mapping")
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("unsupported action smoother state schema")
        config = payload.get("config")
        if config != self.config:
            raise ValueError("state configuration does not match this smoother")
        if payload.get("config_fingerprint") != self.config_fingerprint:
            raise ValueError("state configuration fingerprint does not match")
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("state payload is missing state data")

        raw_dimension = state.get("action_dim")
        if raw_dimension is not None:
            dimension = _positive_int(raw_dimension, "action_dim")
            self._ensure_dimension(dimension)

        raw_batch_size = state.get("batch_size")
        restored_batch_size = (
            None
            if raw_batch_size is None
            else _positive_int(raw_batch_size, "batch_size")
        )
        if restored_batch_size is not None and self._action_dim is None:
            # A reset batch may be serialized before the first action.  Its
            # dimension remains intentionally unknown until a vector arrives.
            pass

        previous = state.get("previous")
        if previous is None:
            self._batch_size = restored_batch_size
            self._state = None
            return
        if not _is_iterable(previous):
            raise ValueError("state previous value must be a batch of vectors")
        rows = list(previous)
        if not rows:
            raise ValueError("state previous value must not be empty")
        if restored_batch_size is None:
            restored_batch_size = len(rows)
        if len(rows) != restored_batch_size:
            raise ValueError("state batch size does not match previous state")

        decoded: list[tuple[float, ...] | None] = []
        inferred_dimension = self._action_dim
        for row in rows:
            if row is None:
                decoded.append(None)
                continue
            decoded_row = self._vector(row, "state previous")
            if inferred_dimension is None:
                inferred_dimension = len(decoded_row)
                self._ensure_dimension(inferred_dimension)
            if len(decoded_row) != inferred_dimension:
                raise ValueError("state vectors must have a consistent length")
            clipped = self._clip_row(decoded_row)
            if clipped != decoded_row:
                raise ValueError("state previous actions must be within bounds")
            decoded.append(decoded_row)

        if inferred_dimension is not None:
            self._ensure_dimension(inferred_dimension)
        self._batch_size = restored_batch_size
        self._state = decoded

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "ActionSmoother":
        if not isinstance(payload, Mapping):
            raise TypeError("state payload must be a mapping")
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("state payload is missing configuration")
        if config.get("type") != cls.__name__:
            raise ValueError("state configuration has an incompatible type")
        smoother = cls(
            low=config.get("low"),
            high=config.get("high"),
            alpha=config.get("alpha"),
            max_delta=config.get("max_delta"),
            method=config.get("method"),
        )
        smoother.load_state_dict(payload)
        return smoother

    def to_json(self) -> str:
        """Return a deterministic JSON serialization of configuration and state."""

        return json.dumps(
            self.state_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    serialize = to_json

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionSmoother":
        return cls.from_state_dict(payload)

    @classmethod
    def from_json(cls, encoded: str) -> "ActionSmoother":
        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid action smoother JSON") from error
        return cls.from_state_dict(payload)

    deserialize = from_json


ActionSmoother = StatefulActionSmoother
