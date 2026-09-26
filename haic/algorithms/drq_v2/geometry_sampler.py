"""Training-only sampling of frozen, seeded official CarRacing roads.

The catalog's measured families are labels, not generator controls. Road geometry
is selected solely by its uint32 seed; track IDs select obstacle layouts only.
This module never reads diagnostic actor outcomes or generates candidate roads.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit


CATALOG_PATH = "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json"
CATALOG_SHA256 = "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b"
PROTOCOL_PATH = "experiments/drqv2-geometry-augmentation-v1.json"
PROTOCOL_SHA256 = "be6d1d1c3b1b16f6b56fd4720097ca8fea8b64c5c962d07183961cdfa0d87294"
_GENERATION_SOURCES = frozenset({
    "core/vendor/car_racing.py", "core/finish_line.py",
    "haic/algorithms/drq_v2/geometry_features.py", "scripts/generate_drq_training_geometry.py",
})
_RUNTIME_SOURCES = frozenset({"core/track_variables.py", "train.py", "env_wrapper.py"})
_TIERS = frozenset({"easy", "boundary", "difficult"})
_UINT32_LIMIT = 2**32


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _seeds(values: Any, label: str) -> list[int]:
    if (not isinstance(values, list)
            or any(type(value) is not int or not 0 <= value < _UINT32_LIMIT for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} must contain distinct uint32 seeds")
    return values


def _inside(root: Path, path: str | Path, directory: str) -> Path:
    if not isinstance(path, (str, Path)) or not str(path) or Path(path).is_absolute():
        raise ValueError(f"{directory} path must be relative to the repository")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to((root / directory).resolve()):
        raise ValueError(f"path must stay under {directory}/")
    return resolved


def _load_catalog(root: Path, catalog_path: str | Path, protocol_path: str | Path,
                  catalog_sha256: str, protocol_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if not _digest(catalog_sha256) or not _digest(protocol_sha256):
        raise ValueError("catalog and protocol require pinned SHA-256 digests")
    catalog_raw = _inside(root, catalog_path, "runs").read_bytes()
    protocol_raw = _inside(root, protocol_path, "experiments").read_bytes()
    if _sha256(catalog_raw) != catalog_sha256 or _sha256(protocol_raw) != protocol_sha256:
        raise ValueError("catalog or generation protocol SHA-256 mismatch")
    catalog, protocol = json.loads(catalog_raw), json.loads(protocol_raw)
    if (not isinstance(catalog, dict) or not isinstance(protocol, dict)
            or catalog.get("format") != "haic-drq-training-geometry-catalog-v1"
            or protocol.get("format") != "haic-drq-training-geometry-protocol-v1"
            or catalog.get("protocol_sha256") != protocol_sha256
            or protocol.get("train_target") != 120 or protocol.get("diagnostic_target") != 16):
        raise ValueError("catalog generation protocol or quota mismatch")

    protocol_sources = protocol.get("source_sha256")
    catalog_sources = catalog.get("source_sha256")
    required = _GENERATION_SOURCES | _RUNTIME_SOURCES
    if (not isinstance(protocol_sources, dict) or not isinstance(catalog_sources, dict)
            or not required <= protocol_sources.keys()
            or set(catalog_sources) != _GENERATION_SOURCES
            or any(catalog_sources[name] != protocol_sources[name] for name in _GENERATION_SOURCES)):
        raise ValueError("missing or inconsistent frozen generator/runtime source hashes")
    source_hashes = {name: protocol_sources[name] for name in sorted(required)}
    for name, expected in source_hashes.items():
        if not _digest(expected) or _sha256((root / name).read_bytes()) != expected:
            raise ValueError(f"frozen geometry/runtime source SHA-256 mismatch: {name}")
    source_hashes["haic/algorithms/drq_v2/geometry_sampler.py"] = _sha256(Path(__file__).read_bytes())

    rules = protocol.get("family_rules")
    if (not isinstance(rules, list) or len(rules) != 6
            or any(not isinstance(rule, dict) or not isinstance(rule.get("name"), str)
                   or not rule["name"] for rule in rules)):
        raise ValueError("generation protocol must define six structural families")
    names = [rule["name"] for rule in rules]
    if len(set(names)) != len(names):
        raise ValueError("generation families must be distinct")
    pool = set(_seeds(protocol.get("candidate_seeds"), "candidate_seeds"))
    exclusions = protocol.get("exclusion_seed_ids")
    if not isinstance(exclusions, dict) or set(exclusions) != {"reserved", "heldout", "blind"}:
        raise ValueError("generation protocol needs reserved/heldout/blind exclusion lists")
    forbidden = set().union(*(_seeds(exclusions[key], key) for key in exclusions))
    if pool & forbidden:
        raise ValueError("candidate pool overlaps excluded geometry across track IDs")
    audit = catalog.get("seed_audit")
    if (not isinstance(audit, dict) or audit.get("passed") is not True
            or audit.get("matched_collisions") != [] or audit.get("proposed_seeds") != protocol["candidate_seeds"]):
        raise ValueError("catalog lacks its passed, exact-pool seed audit")

    train, diagnostic = catalog.get("train"), catalog.get("train_diagnostic")
    if not isinstance(train, list) or not isinstance(diagnostic, list) or (len(train), len(diagnostic)) != (120, 16):
        raise ValueError("catalog TRAIN and TRAIN-DIAGNOSTIC quotas differ")
    all_rows = train + diagnostic
    seen_seeds: set[int] = set()
    seen_roads: set[str] = set()
    for row in all_rows:
        if (not isinstance(row, dict) or type(row.get("geometry_seed")) is not int
                or not 0 <= row["geometry_seed"] < _UINT32_LIMIT or row["geometry_seed"] not in pool
                or row["geometry_seed"] in forbidden or row["geometry_seed"] in seen_seeds
                or row.get("track_id") != 1 or type(row.get("track_id")) is not int
                or row.get("family") not in names or not _digest(row.get("road_coordinate_sha256"))
                or row["road_coordinate_sha256"] in seen_roads):
            raise ValueError("catalog contains duplicate, unapproved or excluded geometry")
        seen_seeds.add(row["geometry_seed"])
        seen_roads.add(row["road_coordinate_sha256"])
    if (Counter((row["family"], row.get("stage")) for row in train)
            != Counter({(name, stage): 10 for name in names for stage in ("representative", "variant")})
            or Counter(row["family"] for row in diagnostic)
            != Counter(dict(zip(names, (3, 3, 3, 3, 2, 2))))
            or any(row.get("stage") != "diagnostic" for row in diagnostic)):
        raise ValueError("catalog family/stage quotas differ from the frozen generation protocol")
    return catalog, protocol, source_hashes


class CatalogGeometrySampler(gym.Wrapper):
    """Select only catalog TRAIN seeds at episode boundaries.

    ``env`` must be the base ``train.HaicTrack``, not ``SampledHaicTrack``;
    use :func:`build_catalog_env` to construct the official wrapper stack. Pass
    explicit, predeclared family-to-tier assignments and relative tier weights.
    Checkpoint state restores the *next reset*, not mid-episode physics state.
    """

    def __init__(self, env: gym.Env, *, track_ids: Sequence[int],
                 track_seed: int, geometry_seed: int,
                 family_tiers: Mapping[str, str], tier_weights: Mapping[str, float],
                 families: Sequence[str] | None = None,
                 repo_root: Path | None = None, catalog_path: str | Path = CATALOG_PATH,
                 protocol_path: str | Path = PROTOCOL_PATH,
                 expected_catalog_sha256: str = CATALOG_SHA256,
                 expected_protocol_sha256: str = PROTOCOL_SHA256):
        root = (Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]).resolve()
        catalog, protocol, source_hashes = _load_catalog(
            root, catalog_path, protocol_path, expected_catalog_sha256, expected_protocol_sha256,
        )
        ids = list(track_ids)
        if not ids or any(type(value) is not int or value <= 0 for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("track_ids must be distinct positive official track IDs")
        for label, value in (("track_seed", track_seed), ("geometry_seed", geometry_seed)):
            if type(value) is not int or not 0 <= value < _UINT32_LIMIT:
                raise ValueError(f"{label} must be a uint32 sampler seed")
        names = [rule["name"] for rule in protocol["family_rules"]]
        if (not isinstance(family_tiers, Mapping) or set(family_tiers) != set(names)
                or any(tier not in _TIERS for tier in family_tiers.values())):
            raise ValueError("family_tiers must predeclare easy/boundary/difficult for every frozen family")
        if not isinstance(tier_weights, Mapping) or set(tier_weights) != _TIERS:
            raise ValueError("tier_weights must provide easy, boundary and difficult weights")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in tier_weights.values()):
            raise ValueError("tier weights must be finite nonnegative numbers")
        selected = names if families is None else list(families)
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(names):
            raise ValueError("families must be a nonempty, distinct TRAIN-family subset")
        active = [name for name in names if name in selected]
        tier_counts = Counter(family_tiers[name] for name in active)
        if any(tier_weights[tier] and not tier_counts[tier] for tier in _TIERS):
            raise ValueError("a positive tier weight has no selected family")
        probabilities = [tier_weights[family_tiers[name]] / tier_counts[family_tiers[name]]
                         for name in active]
        total = sum(probabilities)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("selected families require a positive total weight")
        from train import HaicTrack

        if type(env) is not HaicTrack or not hasattr(env, "track_id") or not hasattr(env, "seed"):
            raise ValueError("geometry sampler requires the base HaicTrack, not a resampling wrapper")

        super().__init__(env)
        self.catalog_sha256 = expected_catalog_sha256
        self.protocol_sha256 = expected_protocol_sha256
        self.source_sha256 = source_hashes
        self.track_ids = ids
        self.track_seed = track_seed
        self.geometry_seed = geometry_seed
        self.family_tiers = {name: family_tiers[name] for name in names}
        self.tier_weights = {tier: float(tier_weights[tier]) for tier in sorted(_TIERS)}
        self.families = active
        self._probabilities = np.asarray(probabilities, dtype=np.float64) / total
        self._roads = {name: tuple(row["geometry_seed"] for row in catalog["train"] if row["family"] == name)
                       for name in active}
        self._track_rng = np.random.default_rng(track_seed)
        self._geometry_rng = np.random.default_rng(geometry_seed)
        self._orders: dict[str, list[int]] = {name: [] for name in active}
        self._cursors = {name: 0 for name in active}
        self.episode_count = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None or options is not None:
            raise ValueError("external reset seed/options cannot override frozen training geometry")
        track_id = int(self._track_rng.choice(self.track_ids))
        family = self.families[int(self._geometry_rng.choice(len(self.families), p=self._probabilities))]
        if self._cursors[family] == len(self._orders[family]):
            self._orders[family] = [int(value) for value in self._geometry_rng.permutation(self._roads[family])]
            self._cursors[family] = 0
        road_seed = self._orders[family][self._cursors[family]]
        self._cursors[family] += 1
        self.env.track_id = track_id
        self.env.seed = road_seed
        observation, info = self.env.reset()
        info = dict(info or {})
        if ("seed" in info and info["seed"] != road_seed
                or "track_id" in info and info["track_id"] != track_id):
            raise ValueError("wrapped environment did not use the selected geometry/track ID")
        info.update(seed=road_seed, track_id=track_id, geometry_seed=road_seed,
                    geometry_family=family, catalog_sha256=self.catalog_sha256)
        self.episode_count += 1
        return observation, info

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-compatible schedule state for an episode-boundary restart."""
        return deepcopy({
            "format": "haic-drq-geometry-sampler-v1", "catalog_sha256": self.catalog_sha256,
            "protocol_sha256": self.protocol_sha256, "source_sha256": self.source_sha256,
            "track_ids": self.track_ids, "track_seed": self.track_seed,
            "geometry_seed": self.geometry_seed, "family_tiers": self.family_tiers,
            "tier_weights": self.tier_weights, "families": self.families,
            "track_rng": self._track_rng.bit_generator.state,
            "geometry_rng": self._geometry_rng.bit_generator.state,
            "orders": self._orders, "cursors": self._cursors, "episode_count": self.episode_count,
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Reject incompatible or poisoned schedules without changing current state."""
        current = self.state_dict()
        dynamic = {"track_rng", "geometry_rng", "orders", "cursors", "episode_count"}
        if (not isinstance(state, Mapping) or set(state) != set(current)
                or any(state[key] != value for key, value in current.items() if key not in dynamic)):
            raise ValueError("geometry sampler restart lineage or configuration mismatch")
        orders, cursors = state["orders"], state["cursors"]
        if (not isinstance(orders, dict) or not isinstance(cursors, dict)
                or set(orders) != set(self.families) or set(cursors) != set(self.families)
                or type(state["episode_count"]) is not int or state["episode_count"] < 0):
            raise ValueError("invalid geometry sampler restart state")
        for family in self.families:
            order, cursor = orders[family], cursors[family]
            if (not isinstance(order, list) or (order and (len(order) != len(self._roads[family])
                    or any(type(seed) is not int for seed in order)
                    or set(order) != set(self._roads[family])))
                    or type(cursor) is not int or not 0 <= cursor <= len(order)):
                raise ValueError("restart permutation contains an unapproved geometry")
        track_rng = np.random.default_rng()
        geometry_rng = np.random.default_rng()
        try:
            for rng, key in ((track_rng, "track_rng"), (geometry_rng, "geometry_rng")):
                rng.bit_generator.state = deepcopy(state[key])
                if rng.bit_generator.state["bit_generator"] != "PCG64":
                    raise ValueError("unexpected RNG type")
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("invalid geometry sampler RNG state") from error
        self._track_rng = track_rng
        self._geometry_rng = geometry_rng
        self._orders = deepcopy(orders)
        self._cursors = deepcopy(cursors)
        self.episode_count = state["episode_count"]


def build_catalog_env(*, track_ids: Sequence[int], track_seed: int, geometry_seed: int,
                      family_tiers: Mapping[str, str], tier_weights: Mapping[str, float],
                      max_steps: int, families: Sequence[str] | None = None,
                      repo_root: Path | None = None, catalog_path: str | Path = CATALOG_PATH,
                      protocol_path: str | Path = PROTOCOL_PATH,
                      expected_catalog_sha256: str = CATALOG_SHA256,
                      expected_protocol_sha256: str = PROTOCOL_SHA256) -> TimeLimit:
    """Construct the existing raw TimeLimit -> CarEnvironment(skip=4) -> HaicTrack stack.

    An additional decision-level TimeLimit matches ``train.build_sampled_env``.
    No reward shaping, collision penalty, or alternative track generator is used.
    """
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive number of frame-skip-4 decisions")
    from train import HaicTrack

    inner = HaicTrack(track_id=1, seed=0, max_steps=max_steps, frame_skip=4, obstacles=True)
    try:
        sampler = CatalogGeometrySampler(
            inner, track_ids=track_ids, track_seed=track_seed, geometry_seed=geometry_seed,
            family_tiers=family_tiers, tier_weights=tier_weights, families=families,
            repo_root=repo_root, catalog_path=catalog_path, protocol_path=protocol_path,
            expected_catalog_sha256=expected_catalog_sha256,
            expected_protocol_sha256=expected_protocol_sha256,
        )
    except Exception:
        inner.close()
        raise
    return TimeLimit(sampler, max_episode_steps=max_steps)
