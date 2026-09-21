import hashlib
import json
from collections.abc import Mapping

import numpy as np


CONTINUOUS_BOX = {"schema_version": 1, "method": "continuous_box", "action_shape": [3]}
MULTIDISCRETE_STEER_LONGITUDINAL = {
    "schema_version": 1,
    "method": "multidiscrete_steer_longitudinal_v1",
    "nvec": [5, 3],
    "steering": [0.0, -0.5, 0.5, -1.0, 1.0],
    "longitudinal": [[1.0, 0.0], [0.0, 0.0], [0.0, 0.8]],
}
MULTIDISCRETE_STEER_LONGITUDINAL_V2 = {
    "schema_version": 1,
    "method": "multidiscrete_steer_longitudinal_v2",
    "nvec": [7, 3],
    "steering": [0.0, -0.5, 0.5, -1.0, 1.0, -0.25, 0.25],
    "longitudinal": [[1.0, 0.0], [0.0, 0.0], [0.0, 0.8]],
}


def canonical_action_representation(method="continuous-box"):
    aliases = {
        "continuous-box": "continuous_box",
        "multidiscrete-steer-longitudinal-v1": "multidiscrete_steer_longitudinal_v1",
        "multidiscrete-steer-longitudinal-v2": "multidiscrete_steer_longitudinal_v2",
    }
    method = aliases.get(method, method)
    if method == "continuous_box":
        return dict(CONTINUOUS_BOX)
    if method == "multidiscrete_steer_longitudinal_v1":
        return json.loads(json.dumps(MULTIDISCRETE_STEER_LONGITUDINAL))
    if method == "multidiscrete_steer_longitudinal_v2":
        return json.loads(json.dumps(MULTIDISCRETE_STEER_LONGITUDINAL_V2))
    raise ValueError("unsupported action representation")


def normalize_action_representation(config=None):
    if config is None:
        return canonical_action_representation()
    if not isinstance(config, Mapping):
        raise TypeError("action representation must be a mapping")
    canonical = canonical_action_representation(config.get("method", "continuous_box"))
    normalized = dict(config)
    normalized["method"] = canonical["method"]
    for key, value in canonical.items():
        if key in normalized and normalized[key] != value:
            raise ValueError(f"action representation {key} does not match schema")
    return canonical


def action_representation_fingerprint(config=None):
    return hashlib.sha256(
        json.dumps(normalize_action_representation(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def policy_action_shape(config=None):
    representation = normalize_action_representation(config)
    if representation["method"] == "continuous_box":
        return (3,)
    return tuple(representation["nvec"])


def map_policy_action(action, config=None):
    representation = normalize_action_representation(config)
    values = np.asarray(action)
    if representation["method"] == "continuous_box":
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("continuous action must be a finite shape-(3,) vector")
        return np.clip(values.astype(np.float32), [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    if values.shape != (2,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("MultiDiscrete action must be an integer shape-(2,) vector")
    steer_index, longitudinal_index = (int(values[0]), int(values[1]))
    if not 0 <= steer_index < representation["nvec"][0] or not 0 <= longitudinal_index < representation["nvec"][1]:
        raise ValueError("MultiDiscrete action index is out of range")
    steering = representation["steering"][steer_index]
    gas, brake = representation["longitudinal"][longitudinal_index]
    return np.asarray([steering, gas, brake], dtype=np.float32)
