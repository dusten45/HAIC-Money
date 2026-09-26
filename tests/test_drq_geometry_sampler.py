"""Synthetic-only tests: no catalog road or blind geometry is reset here."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from haic.algorithms.drq_v2.geometry_sampler import CatalogGeometrySampler, build_catalog_env


SOURCE_FILES = (
    "core/vendor/car_racing.py", "core/finish_line.py", "core/track_variables.py",
    "haic/algorithms/drq_v2/geometry_features.py", "scripts/generate_drq_training_geometry.py",
    "train.py", "env_wrapper.py",
)
GENERATOR_FILES = (
    "core/vendor/car_racing.py", "core/finish_line.py",
    "haic/algorithms/drq_v2/geometry_features.py", "scripts/generate_drq_training_geometry.py",
)
FAMILIES = tuple(f"synthetic-family-{index}" for index in range(6))
FAMILY_TIERS = dict(zip(FAMILIES, ("easy", "boundary", "boundary", "difficult", "difficult", "difficult")))
WEIGHTS = {"easy": 1.0, "boundary": 2.0, "difficult": 3.0}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class FakeHaicTrack(gym.Env):
    def __init__(self):
        self.track_id = 1
        self.seed = 0
        self.calls = []
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4, 84, 84), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        assert seed is None and options is None
        road_fingerprint = digest(f"seeded road {self.seed}".encode())
        self.calls.append((self.seed, self.track_id, road_fingerprint))
        return np.zeros((4, 84, 84), dtype=np.float32), {
            "seed": self.seed, "track_id": self.track_id, "road_fingerprint": road_fingerprint,
        }

    def step(self, action):
        return np.zeros((4, 84, 84), dtype=np.float32), -2.5, False, False, {}


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    import train

    real_haic_track = train.HaicTrack
    monkeypatch.setattr(train, "HaicTrack", FakeHaicTrack)
    sources = {}
    for name in SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic frozen source: {name}\n".encode())
        sources[name] = digest(path.read_bytes())
    protocol = {
        "format": "haic-drq-training-geometry-protocol-v1",
        "source_sha256": sources, "train_target": 120, "diagnostic_target": 16,
        "family_rules": [{"name": family} for family in FAMILIES],
        "exclusion_seed_ids": {"reserved": [42], "heldout": [31001], "blind": [31201]},
    }
    train = [{"geometry_seed": 10_000 + family_index * 20 + index, "track_id": 1,
              "family": family, "stage": "representative" if index < 10 else "variant",
              "road_coordinate_sha256": digest(f"synthetic road {family_index}:{index}".encode())}
             for family_index, family in enumerate(FAMILIES) for index in range(20)]
    diagnostic = [{"geometry_seed": 20_000 + family_index * 3 + index, "track_id": 1,
                   "family": family, "stage": "diagnostic",
                   "road_coordinate_sha256": digest(f"synthetic diagnostic {family_index}:{index}".encode())}
                  for family_index, family in enumerate(FAMILIES)
                  for index in range((3, 3, 3, 3, 2, 2)[family_index])]
    protocol["candidate_seeds"] = [row["geometry_seed"] for row in train + diagnostic]
    protocol_path = tmp_path / "experiments/frozen.json"
    protocol_path.parent.mkdir()
    protocol_path.write_text(json.dumps(protocol))
    protocol_sha256 = digest(protocol_path.read_bytes())
    catalog = {
        "format": "haic-drq-training-geometry-catalog-v1", "protocol_sha256": protocol_sha256,
        "source_sha256": {name: sources[name] for name in GENERATOR_FILES},
        "seed_audit": {"passed": True, "matched_collisions": [],
                       "proposed_seeds": protocol["candidate_seeds"]},
        "train": train, "train_diagnostic": diagnostic,
    }
    catalog_path = tmp_path / "runs/catalog.json"
    catalog_path.parent.mkdir()
    catalog_path.write_text(json.dumps(catalog))
    args = {
        "repo_root": tmp_path, "protocol_path": "experiments/frozen.json",
        "catalog_path": "runs/catalog.json", "expected_protocol_sha256": protocol_sha256,
        "expected_catalog_sha256": digest(catalog_path.read_bytes()),
    }
    return SimpleNamespace(root=tmp_path, protocol=protocol, catalog=catalog,
                           real_haic_track=real_haic_track,
                           catalog_path=catalog_path, protocol_path=protocol_path, args=args)


def sampler(frozen, *, env=None, track_ids=(1, 2, 3), track_seed=17, geometry_seed=23,
            family_tiers=None, tier_weights=None, families=None):
    env = env if env is not None else FakeHaicTrack()
    return CatalogGeometrySampler(
        env, track_ids=track_ids, track_seed=track_seed, geometry_seed=geometry_seed,
        family_tiers=FAMILY_TIERS if family_tiers is None else family_tiers,
        tier_weights=WEIGHTS if tier_weights is None else tier_weights,
        families=families, **frozen.args,
    )


def repin_catalog(frozen):
    frozen.catalog_path.write_text(json.dumps(frozen.catalog))
    frozen.args["expected_catalog_sha256"] = digest(frozen.catalog_path.read_bytes())


def test_only_train_seeds_across_all_track_ids_and_same_road_is_not_new_geometry(frozen):
    selected = set(frozen.protocol["candidate_seeds"][:120])
    blocked = {row["geometry_seed"] for row in frozen.catalog["train_diagnostic"]}
    blocked.update((42, 31001, 31201))
    wrapper = sampler(frozen, track_ids=(1, 2, 3, 4))
    observations = [wrapper.reset()[1] for _ in range(400)]
    assert {item["seed"] for item in observations} <= selected
    assert not {item["seed"] for item in observations} & blocked
    assert {item["track_id"] for item in observations} == {1, 2, 3, 4}
    assert wrapper.env.calls == [(item["seed"], item["track_id"], item["road_fingerprint"])
                                 for item in observations]
    roads = {}
    for item in observations:
        roads.setdefault(item["geometry_seed"], set()).add((item["track_id"], item["road_fingerprint"]))
    assert any(len({track_id for track_id, _road in layouts}) > 1 for layouts in roads.values())
    assert all(len({road for _track_id, road in layouts}) == 1 for layouts in roads.values())
    assert len({item["road_fingerprint"] for item in observations}) == len(roads)


def test_subset_selection_weights_and_per_family_permutation_cycles(frozen):
    wrapper = sampler(frozen, families=[FAMILIES[0]],
                      tier_weights={"easy": 1, "boundary": 0, "difficult": 0})
    chosen = [wrapper.reset()[1] for _ in range(40)]
    seeds = {row["geometry_seed"] for row in frozen.catalog["train"] if row["family"] == FAMILIES[0]}
    assert {item["geometry_family"] for item in chosen} == {FAMILIES[0]}
    assert set(item["seed"] for item in chosen[:20]) == seeds
    assert set(item["seed"] for item in chosen[20:]) == seeds
    assert len({item["seed"] for item in chosen[:20]}) == 20
    interleaved = sampler(frozen, families=[FAMILIES[1], FAMILIES[2]],
                          tier_weights={"easy": 0, "boundary": 1, "difficult": 0})
    per_family = {name: [] for name in (FAMILIES[1], FAMILIES[2])}
    for _ in range(160):
        info = interleaved.reset()[1]
        per_family[info["geometry_family"]].append(info["seed"])
    for name, drawn in per_family.items():
        expected = {row["geometry_seed"] for row in frozen.catalog["train"] if row["family"] == name}
        assert len(drawn) >= 40
        assert set(drawn[:20]) == set(drawn[20:40]) == expected
    with pytest.raises(ValueError, match="positive tier weight has no selected family"):
        sampler(frozen, families=[FAMILIES[0]])
    with pytest.raises(ValueError, match="TRAIN-family subset"):
        sampler(frozen, families=[FAMILIES[0], "diagnostic-only"])
    with pytest.raises(ValueError, match="predeclare"):
        sampler(frozen, family_tiers={FAMILIES[0]: "easy"})
    with pytest.raises(ValueError, match="tier weights"):
        sampler(frozen, tier_weights={"easy": 1, "boundary": float("nan"), "difficult": 1})


def test_independent_rngs_and_deterministic_sequences(frozen):
    first = sampler(frozen)
    identical = sampler(frozen)
    other_ids = sampler(frozen, track_ids=(77,), track_seed=928)
    other_geometry = sampler(frozen, geometry_seed=429,
                             tier_weights={"easy": 0, "boundary": 0, "difficult": 1})
    sequence = [[wrapper.reset()[1] for _ in range(80)] for wrapper in
                (first, identical, other_ids, other_geometry)]
    identity = lambda row: (row["track_id"], row["geometry_seed"], row["geometry_family"])
    assert list(map(identity, sequence[0])) == list(map(identity, sequence[1]))
    assert [(row["geometry_seed"], row["geometry_family"]) for row in sequence[0]] == [
        (row["geometry_seed"], row["geometry_family"]) for row in sequence[2]]
    assert [row["track_id"] for row in sequence[0]] == [row["track_id"] for row in sequence[3]]
    assert {row["geometry_family"] for row in sequence[3]} <= set(FAMILIES[3:])


def test_caller_weights_apply_to_tiers_not_family_counts(frozen):
    wrapper = sampler(frozen, tier_weights={"easy": 6, "boundary": 3, "difficult": 1})
    counts = Counter(FAMILY_TIERS[wrapper.reset()[1]["geometry_family"]] for _ in range(1_000))
    assert 550 < counts["easy"] < 650
    assert 250 < counts["boundary"] < 350
    assert 70 < counts["difficult"] < 130


def test_untrusted_reset_seed_and_options_are_rejected_before_rng_consumption(frozen):
    wrapper = sampler(frozen)
    outer = gym.wrappers.TimeLimit(wrapper, max_episode_steps=2)
    before = wrapper.state_dict()
    for kwargs in ({"seed": 31001}, {"seed": 31201}, {"seed": 0},
                   {"options": {"track_id": 999}}, {"options": {}},
                   {"seed": 3, "options": {"track_id": 4}}):
        with pytest.raises(ValueError, match="external reset seed/options"):
            outer.reset(**kwargs)
        assert wrapper.state_dict() == before
        assert wrapper.env.calls == []
    outer.reset()
    assert wrapper.episode_count == 1
    outer.close()


def test_resampling_subclass_is_rejected_before_it_can_reset_any_road(frozen):
    class UnsafeSampledTrack(FakeHaicTrack):
        def reset(self, *, seed=None, options=None):
            raise AssertionError("unexpected fresh or excluded road reset")

    fake = UnsafeSampledTrack()
    with pytest.raises(ValueError, match="base HaicTrack"):
        sampler(frozen, env=fake)
    assert fake.calls == []


@pytest.mark.parametrize("partition", ("train_diagnostic", "heldout", "blind", "reserved"))
def test_even_re_pinned_catalog_rejects_all_excluded_road_ids(frozen, partition):
    if partition == "train_diagnostic":
        frozen.catalog["train"][0]["geometry_seed"] = frozen.catalog["train_diagnostic"][0]["geometry_seed"]
    else:
        frozen.catalog["train"][0]["geometry_seed"] = {
            "heldout": 31001, "blind": 31201, "reserved": 42,
        }[partition]
    repin_catalog(frozen)
    fake = FakeHaicTrack()
    with pytest.raises(ValueError, match="unapproved or excluded geometry"):
        sampler(frozen, env=fake, track_ids=(1, 2, 3))
    assert fake.calls == []


def test_hash_source_and_generation_protocol_mismatches_stop_before_reset(frozen):
    fake = FakeHaicTrack()
    frozen.catalog_path.write_bytes(frozen.catalog_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        sampler(frozen, env=fake)
    repin_catalog(frozen)
    frozen.protocol_path.write_bytes(frozen.protocol_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        sampler(frozen, env=fake)
    frozen.protocol_path.write_text(json.dumps(frozen.protocol))
    (frozen.root / "core/vendor/car_racing.py").write_bytes(b"changed generator")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        sampler(frozen, env=fake)
    assert fake.calls == []


def test_exact_restart_and_invalid_state_cannot_inject_geometry(frozen):
    first = sampler(frozen, tier_weights={"easy": 1, "boundary": 0, "difficult": 0})
    for _ in range(23):
        first.reset()
    snapshot = json.loads(json.dumps(first.state_dict()))
    assert snapshot["source_sha256"]["core/vendor/car_racing.py"]
    assert snapshot["source_sha256"]["haic/algorithms/drq_v2/geometry_sampler.py"]
    second = sampler(frozen, tier_weights={"easy": 1, "boundary": 0, "difficult": 0})
    second.load_state_dict(snapshot)
    snapshot["orders"][FAMILIES[0]][0] = 31201
    assert second.state_dict()["orders"][FAMILIES[0]][0] != 31201
    assert [first.reset()[1] for _ in range(60)] == [second.reset()[1] for _ in range(60)]

    unchanged = second.state_dict()
    invalid = deepcopy(unchanged)
    invalid["orders"][FAMILIES[0]][0] = frozen.catalog["train_diagnostic"][0]["geometry_seed"]
    with pytest.raises(ValueError, match="unapproved geometry"):
        second.load_state_dict(invalid)
    invalid = deepcopy(unchanged)
    invalid["track_rng"] = {"bit_generator": "MT19937"}
    with pytest.raises(ValueError, match="invalid geometry sampler RNG"):
        second.load_state_dict(invalid)
    invalid = deepcopy(unchanged)
    invalid["catalog_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage"):
        second.load_state_dict(invalid)
    assert second.state_dict() == unchanged
    incompatible = sampler(frozen, track_seed=4, tier_weights={"easy": 1, "boundary": 0, "difficult": 0})
    with pytest.raises(ValueError, match="lineage"):
        incompatible.load_state_dict(unchanged)


def test_factory_preserves_raw_reward_skip_four_and_two_time_limits(frozen, monkeypatch):
    import train

    class FakeRawCarRacing(gym.Env):
        instances = []

        def __init__(self, *, continuous, render_mode):
            assert continuous and render_mode is None
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(0, 255, shape=(96, 96, 3), dtype=np.uint8)
            self.car = SimpleNamespace(set_damage_effects=lambda *_args: None)
            self.track = [(0, 0, 0, 0)] * 100
            self.tile_visited_count = 0
            self.finish_time_s = None
            self.steps = 0
            self.resets = []
            self.instances.append(self)

        def reset(self, *, seed=None, options=None):
            self.resets.append((seed, options))
            self.steps = 0
            return np.zeros((96, 96, 3), dtype=np.uint8), {}

        def step(self, action):
            self.steps += 1
            return np.zeros((96, 96, 3), dtype=np.uint8), -0.25, False, False, {}

    monkeypatch.setattr(train, "CarRacing", FakeRawCarRacing)
    monkeypatch.setattr(train, "HaicTrack", frozen.real_haic_track)
    env = build_catalog_env(
        track_ids=(1, 2), track_seed=17, geometry_seed=23, family_tiers=FAMILY_TIERS,
        tier_weights=WEIGHTS, max_steps=1, **frozen.args,
    )
    try:
        observation, info = env.reset()
        raw = FakeRawCarRacing.instances[-1]
        assert observation.shape == (4, 84, 84)
        assert raw.resets == [(info["seed"], {"track_id": info["track_id"]})]
        assert raw.steps == 50  # Existing CarEnvironment warmup.
        _, reward, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
        assert reward == -1.0  # Four unchanged raw rewards, no shaping.
        assert not terminated and truncated
        assert raw.steps == 54
        assert env.env.env.env.env._max_episode_steps == 204  # Raw-frame TimeLimit budget.
        with pytest.raises(ValueError, match="external reset seed/options"):
            env.reset(seed=31201)
        assert raw.resets == [(info["seed"], {"track_id": info["track_id"]})]
    finally:
        env.close()
