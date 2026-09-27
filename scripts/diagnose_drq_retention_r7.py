"""Separate, unranked r7 diagnostic on the 16 already-consumed r6 TRAIN-DIAGNOSTIC roads.

No training, threshold tuning, source re-rollout, held-out evaluation or actor
selection occurs here. Run only after all 12 r7 runs have completed and the r7
protocol has been frozen; this command intentionally has no default protocol hash.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from drq_v2 import load_exported_actor
from scripts import diagnose_drq_geometry_mix as r6


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("experiments/drqv2-retention-r7.json")
RUN_ROOT = Path("runs/20260926-drqv2-retention-r7")
R6_PROTOCOL_SHA = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
R6_MANIFEST_SHA = "fb14fe9eab14f61cdecc45453b69e44d34fdb967368c44c255e827292804b380"
R6_EPISODES_SHA = "aee2ae9b85172feb0e9a09697bdd92a34b29631cfc094586364f7c0b9f283441"
R6_PAIRED_SHA = "29b6688413738f5a0ae7faaa950d202c53c73c1c14735f9e205d3ec01ed2cdd7"
R6_DIAGNOSTIC = Path("runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic")
R6_PAIRED = Path("runs/20260925-drqv2-geometry-mix-v1-r6/paired-regression-v1.json")
FORMAT = "haic-drq-retention-study-v1"
RESULT_FORMAT = "haic-drq-retention-run-result-v1"
CATALOG_FORMAT = "haic-drq-retention-checkpoint-catalog-v1"
EPISODE_FORMAT = "haic-drq-retention-training-diagnostic-episode-v1"
DIAGNOSTIC_FORMAT = "haic-drq-retention-training-diagnostic-v1"
VARIANTS = r6.VARIANTS
CONDITIONS = ("r7a", "r7b")
ROLES = {(seed, variant, condition) for seed in (0, 1)
         for variant in VARIANTS for condition in CONDITIONS}
FINAL_STEP = 32768
FINAL_UPDATES = 22768
RUNTIME_SOURCES = (
    "scripts/diagnose_drq_geometry_mix.py", "train.py", "common_adapter.py", "drq_v2.py",
    "agent.py", "env_wrapper.py", "action_representation.py", "action_smoothing.py",
    "tracking.py", "damage.py", "core/finish_line.py", "core/obstacle_contacts.py",
    "core/track_variables.py", "core/vendor/car_dynamics.py", "core/vendor/car_racing.py",
    "haic/algorithms/drq_v2/geometry_features.py",
    "haic/algorithms/drq_v2/geometry_sampler.py",
)


class DiagnosticError(r6.DiagnosticError):
    """Malformed or incomplete r7 input; never silently skip a diagnostic cell."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _hash(value: Any, label: str) -> str:
    _check(isinstance(value, str) and r6.SHA256_RE.fullmatch(value) is not None,
           f"{label} must be a lowercase SHA-256")
    return value


def _file(root: Path, value: Any, label: str) -> Path:
    path = r6._repo_file(root, value, "runs", label)
    _check(path.is_file() and not path.is_symlink(), f"{label} is missing or not a regular file: {path}")
    return path


def _pinned(root: Path, value: Any, digest: str, label: str) -> dict[str, Any]:
    directory = "experiments" if isinstance(value, str) and value.startswith("experiments/") else "runs"
    try:
        path = r6._repo_file(root, value, directory, label)
        _check(path.is_file() and not path.is_symlink(), f"{label} is missing or not a regular file: {path}")
        return r6._pinned_json(path, _hash(digest, label), label)
    except r6.DiagnosticError as error:
        raise DiagnosticError(str(error)) from error


def _verified(root: Path, value: Any, digest: Any, label: str) -> Path:
    path = _file(root, value, label)
    _check(r6.sha256_file(path) == _hash(digest, label), f"{label} SHA-256 mismatch: {path}")
    return path


def _r6_reference(root: Path, r6_protocol: dict[str, Any], rows: list[dict[str, Any]]
                  ) -> tuple[dict[tuple[int, int], dict[str, Any]],
                             dict[tuple[int, int, str], dict[str, Any]], dict[str, str]]:
    diagnostic = root / R6_DIAGNOSTIC
    manifest = _pinned(root, (R6_DIAGNOSTIC / "manifest.json").as_posix(),
                       R6_MANIFEST_SHA, "original r6 diagnostic manifest")
    _check(manifest.get("protocol_sha256") == R6_PROTOCOL_SHA
           and manifest.get("episode_count") == 256 and manifest.get("role") == "TRAIN-DIAGNOSTIC"
           and manifest.get("ranked") is False and manifest.get("score_selection") is False
           and manifest.get("catalog", {}).get("catalog_sha256") == r6.CATALOG_SHA256,
           "original r6 diagnostic identity changed")
    sidecar = _file(root, (R6_DIAGNOSTIC / "manifest.sha256").as_posix(), "r6 manifest sidecar")
    _check(sidecar.read_text(encoding="ascii") == f"{R6_MANIFEST_SHA}  manifest.json\n",
           "r6 manifest sidecar mismatch")
    files = manifest.get("files_sha256")
    _check(isinstance(files, dict) and files.get("episodes.jsonl") == R6_EPISODES_SHA,
           "r6 episode file is not bound to the original manifest")
    episode_path = _verified(root, (R6_DIAGNOSTIC / "episodes.jsonl").as_posix(),
                             R6_EPISODES_SHA, "original r6 episodes")
    paired = _pinned(root, R6_PAIRED.as_posix(), R6_PAIRED_SHA, "original r6 paired regression")
    _check(paired.get("protocol_sha256") == R6_PROTOCOL_SHA
           and paired.get("diagnostic_manifest_sha256") == R6_MANIFEST_SHA
           and paired.get("episodes_sha256") == R6_EPISODES_SHA
           and paired.get("catalog_sha256") == r6.CATALOG_SHA256,
           "r6 paired regression lineage mismatch")

    by_road = {row["geometry_seed"]: row for row in rows}
    expected = {(seed, road, role, repeat) for seed in (0, 1) for road in by_road
                for role in ("unchanged-source", *VARIANTS) for repeat in range(2)}
    observed: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    for line in episode_path.read_bytes().splitlines():
        row = r6._parse_json(line, episode_path)
        _check(isinstance(row, dict), "r6 episode must be an object")
        seed, road, role, repeat = (row.get("source_learner_seed"), row.get("geometry_seed"),
                                    row.get("arm"), row.get("repeat"))
        key = (seed, road, role, repeat)
        _check(key in expected and key not in observed and type(seed) is int
               and type(road) is int and type(repeat) is int,
               "r6 diagnostic has an unexpected or duplicate cell")
        role_key = f"{role}-seed{seed}"
        trace = f"traces/seed-{road}/{role_key}/repeat-{repeat}.npz"
        _check(row.get("trace_path") == trace and row.get("trace_sha256") == files.get(trace)
               and row.get("family") == by_road[road]["family"] and row.get("track_id") == 1
               and row.get("protocol_sha256") == R6_PROTOCOL_SHA
               and row.get("catalog_sha256") == r6.CATALOG_SHA256
               and row.get("canonical_repeat") is (repeat == 0)
               and row.get("repeat_trace_agrees_with_repeat0") is True
               and type(row.get("finished")) is bool,
               "original r6 diagnostic cell lineage/outcome mismatch")
        observed[key] = row
    _check(set(observed) == expected and len(files) == 258,
           "r6 diagnostic is not the complete 256-episode inventory")
    for seed, road, role, _ in expected:
        first, second = observed[(seed, road, role, 0)], observed[(seed, road, role, 1)]
        _check(first["trace_deterministic_sha256"] == second["trace_deterministic_sha256"]
               and first["finished"] == second["finished"], "r6 repeat pair diverged")

    sources = {(seed, road): observed[(seed, road, "unchanged-source", 0)]
               for seed in (0, 1) for road in by_road}
    prior = {(seed, road, variant): observed[(seed, road, variant, 0)]
             for seed in (0, 1) for road in by_road for variant in VARIANTS}
    _check(sum(row["finished"] for row in sources.values()) == 11,
           "original source outcome is not 11/32")
    _check({variant: sum(row["finished"] for key, row in prior.items() if key[2] == variant)
            for variant in VARIANTS} == {"uniform": 1, "failure_weighted": 4, "easy_retention": 3},
           "original r6 outcomes differ from the frozen 1/4/3 of 32")
    original_pairs = paired.get("pairs")
    _check(isinstance(original_pairs, list) and len(original_pairs) == 96,
           "r6 original paired regression lacks its full grid")
    found = set()
    transitions: dict[str, Counter] = defaultdict(Counter)
    families: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for pair in original_pairs:
        _check(isinstance(pair, dict), "r6 original paired row is malformed")
        seed, road, variant = pair.get("source_seed"), pair.get("geometry_seed"), pair.get("variant")
        key = (seed, road, variant)
        _check(key in prior and key not in found and type(seed) is int and type(road) is int,
               "r6 original paired grid is duplicate or incomplete")
        source, candidate = sources[(seed, road)], prior[key]
        transition = r6_transition(source["finished"], candidate["finished"])
        _check(pair.get("source_finished") is source["finished"]
               and pair.get("candidate_finished") is candidate["finished"]
               and pair.get("transition") == transition
               and pair.get("family") == source["family"] and pair.get("track_id") == 1
               and pair.get("source_trace_sha256") == source["trace_sha256"]
               and pair.get("candidate_trace_sha256") == candidate["trace_sha256"],
               "r6 paired outcome/trace disagrees with original episodes")
        transitions[variant][transition] += 1
        families[(variant, source["family"])][transition] += 1
        found.add(key)
    _check(found == set(prior) and all(
        paired.get("variant_transitions", {}).get(variant)
        == {name: transitions[variant][name] for name in ("lost", "gained", "both_finished", "both_failed")}
        for variant in VARIANTS), "r6 original paired transition counts mismatch")
    _check({(item["variant"], item["family"]): item for item in paired.get("family_transitions", [])}
           == {(variant, family): {"variant": variant, "family": family,
                                   "n": sum(count.values()),
                                   **{name: count[name] for name in ("lost", "gained", "both_finished", "both_failed")}}
               for (variant, family), count in families.items()},
           "r6 original family paired counts mismatch")
    _check(r6_protocol["diagnostics"]["geometry_seeds"] == list(r6.DIAGNOSTIC_SEEDS),
           "r6 diagnostic seed order differs from frozen allocation")
    return sources, prior, {"r6_protocol_sha256": R6_PROTOCOL_SHA,
                            "r6_manifest_sha256": R6_MANIFEST_SHA,
                            "r6_episodes_sha256": R6_EPISODES_SHA,
                            "r6_paired_regression_sha256": R6_PAIRED_SHA}


def r6_transition(source: bool, candidate: bool) -> str:
    if source:
        return "both_finished" if candidate else "lost"
    return "gained" if candidate else "both_failed"


def _runtime_sources(root: Path, protocol: dict[str, Any]) -> dict[str, str]:
    frozen = protocol.get("source_sha256")
    _check(isinstance(frozen, dict) and all(name in frozen for name in RUNTIME_SOURCES),
           "r6 original CPU diagnostic/runtime source inventory is incomplete")
    checked = {}
    for relative in RUNTIME_SOURCES:
        path = root
        for part in Path(relative).parts:
            path /= part
            _check(not path.is_symlink(), f"r6 runtime source path is a symlink: {path}")
        _check(path.is_file() and r6.sha256_file(path) == _hash(frozen[relative], relative),
               f"r6 CPU diagnostic/runtime source hash changed: {relative}")
        checked[relative] = frozen[relative]
    return checked


def _grid(protocol: dict[str, Any]) -> dict[tuple[int, str, str], Path]:
    _check(protocol.get("format") == FORMAT and protocol.get("run_root") == RUN_ROOT.as_posix()
           and protocol.get("r6_protocol_path") == r6.PROTOCOL_PATH.as_posix()
           and protocol.get("r6_protocol_sha256") == R6_PROTOCOL_SHA,
           "r7 protocol format/run root/r6 lineage mismatch")
    declared = protocol.get("runs")
    _check(isinstance(declared, list) and len(declared) == 12, "r7 must declare exactly 12 runs")
    roles: dict[tuple[int, str, str], Path] = {}
    for run in declared:
        _check(isinstance(run, dict), "invalid r7 run")
        seed, variant, condition = run.get("source_seed"), run.get("variant"), run.get("condition")
        key = (seed, variant, condition)
        _check(type(seed) is int and key in ROLES and key not in roles,
               "duplicate or unexpected r7 seed/variant/condition")
        relative = RUN_ROOT / f"learner-{seed}-{variant}-{condition}"
        _check(run.get("run_dir") == relative.as_posix(), "r7 candidate run directory differs from frozen layout")
        roles[key] = relative
    _check(set(roles) == ROLES, "r7 run grid is incomplete")
    return roles


def _verify_step_ledger(path: Path, source_rows: set[int], *, max_step: int = FINAL_STEP,
                        warmup: int = r6.WARMUP_STEPS) -> None:
    count = 0
    with path.open("rb") as stream:
        for count, line in enumerate(stream, 1):
            _check(count <= max_step, "r7 step ledger exceeds exact online budget")
            entry = r6._parse_json(line, path)
            updates = max(0, count - warmup)
            _check(isinstance(entry, dict) and type(entry.get("additional_online_step")) is int
                   and entry["additional_online_step"] == count
                   and type(entry.get("gradient_steps")) is int and entry["gradient_steps"] == updates
                   and type(entry.get("geometry_seed")) is int and entry["geometry_seed"] in source_rows
                   and type(entry.get("track_id")) is int and entry["track_id"] in (1, 2, 3, 4)
                   and type(entry.get("source_samples")) is int
                   and type(entry.get("online_samples")) is int
                   and entry["source_samples"] == updates * 32
                   and entry["online_samples"] == updates * 32,
                   f"r7 step ledger row {count} violates TRAIN-only exact-budget 32/32 contract")
    _check(count == max_step, "r7 step ledger is truncated before exact online budget")


def _candidate(root: Path, run_dir: Path, key: tuple[int, str, str],
               protocol_sha: str, source: dict[str, Any], protocol: dict[str, Any],
               training_roads: set[int]) -> dict[str, Any]:
    seed, variant, condition = key
    _verified(root, (run_dir / "study_protocol.json").as_posix(), protocol_sha,
              "r7 local frozen protocol copy")
    config_path = _file(root, (run_dir / "run-config.json").as_posix(), "r7 run config")
    config = r6._read_json(config_path)
    expected_lambda = protocol["lambda_preserve"] if condition == "r7b" else 0.0
    _check(isinstance(config, dict) and config.get("format") == "haic-drq-retention-run-config-v1"
           and config.get("protocol_sha256") == protocol_sha and config.get("source_seed") == seed
           and config.get("variant") == variant and config.get("condition") == condition
           and config.get("source_actor_sha256") == source["actor_sha256"]
           and config.get("source_replay_sha256") == source["checkpoint_sha256"]
           and config.get("source_rows_per_batch") == 32
           and config.get("online_rows_per_batch") == 32
           and config.get("lambda_preserve") == expected_lambda
           and config.get("diagnostic_cache_sha256") == protocol["diagnostic_cache"]["sha256"]
           and config.get("rng_seeds") == next(item["rng_seeds"] for item in protocol["runs"]
                                               if (item["source_seed"], item["variant"], item["condition"]) == key),
           f"r7 run config treatment/source/replay identity mismatch for {key}")
    result_path = _file(root, (run_dir / "result.json").as_posix(), "r7 completed run receipt")
    receipt = r6._read_json(result_path)
    _check(isinstance(receipt, dict) and receipt.get("format") == RESULT_FORMAT
           and receipt.get("study_id") == "drqv2-retention-r7"
           and receipt.get("completed") is True and receipt.get("source_seed") == seed
           and receipt.get("variant") == variant and receipt.get("condition") == condition
           and receipt.get("study_protocol_sha256") == protocol_sha
           and receipt.get("additional_online_steps") == FINAL_STEP
           and receipt.get("study_gradient_steps") == FINAL_UPDATES
           and receipt.get("source_samples") == FINAL_UPDATES * 32
           and receipt.get("online_samples") == FINAL_UPDATES * 32
           and receipt.get("source_actor_sha256") == source["actor_sha256"]
           and receipt.get("source_checkpoint_sha256") == source["checkpoint_sha256"]
           and receipt.get("source_replay_sha256") == source["checkpoint_sha256"],
           f"r7 result not a completed exact-budget receipt for {key}")
    catalog_path = _file(root, (run_dir / "checkpoint-catalog.json").as_posix(), "r7 checkpoint catalog")
    catalog = r6._read_json(catalog_path)
    _check(isinstance(catalog, dict) and catalog.get("format") == CATALOG_FORMAT
           and catalog.get("study_protocol_sha256") == protocol_sha
           and catalog.get("source_seed") == seed and catalog.get("variant") == variant
           and catalog.get("condition") == condition,
           f"r7 checkpoint catalog identity mismatch for {key}")
    candidates, catalog_candidates = receipt.get("candidates"), catalog.get("candidates")
    _check(isinstance(candidates, list) and isinstance(catalog_candidates, list)
           and len(candidates) == len(catalog_candidates) == 2
           and {c.get("checkpoint_online_step") for c in candidates if isinstance(c, dict)} == {16384, FINAL_STEP}
           and candidates == catalog_candidates,
           f"r7 result/catalog must agree on both checkpoint budgets for {key}")
    final = next(c for c in candidates if c["checkpoint_online_step"] == FINAL_STEP)
    for field, expected in (("study_gradient_steps", FINAL_UPDATES),):
        _check(final.get(field) == expected, f"r7 final {field} mismatch for {key}")
    step = run_dir / "checkpoints" / "step-000032768"
    paths = {"actor": step / "actor.pt", "checkpoint": step / "checkpoint.pt",
             "checkpoint_manifest": step / "checkpoint.manifest.json"}
    for name in ("actor", "checkpoint"):
        relative = paths[name]
        _check(final.get(f"{name}_path") == relative.as_posix(),
               f"r7 final {name} is not the frozen step-32768 artifact for {key}")
        _verified(root, relative.as_posix(), final.get(f"{name}_sha256"), f"r7 final {name}")
    manifest_path = _file(root, paths["checkpoint_manifest"].as_posix(), "r7 final checkpoint manifest")
    manifest_sha = r6.sha256_file(manifest_path)
    manifest = r6._read_json(manifest_path)
    extra = manifest.get("extra")
    _check(manifest.get("algorithm") == "drq-v2" and isinstance(extra, dict)
           and extra.get("environment_steps") == r6.SOURCE_STEPS + FINAL_STEP
           and extra.get("study_id") == "drqv2-retention-r7"
           and extra.get("study_protocol_sha256") == protocol_sha
           and extra.get("source_checkpoint_sha256") == source["checkpoint_sha256"]
           and extra.get("source_replay_sha256") == source["checkpoint_sha256"]
           and extra.get("catalog_sha256") == r6.CATALOG_SHA256,
           f"r7 final checkpoint manifest lineage/budget mismatch for {key}")
    trace_rel = run_dir / "replay-sample-trace-step-000032768.npz"
    _check(final.get("sample_trace_path") == trace_rel.as_posix(),
           f"r7 final sampled-update trace receipt is incomplete for {key}")
    _verified(root, trace_rel.as_posix(), final.get("sample_trace_sha256"), "r7 sampled-update trace")
    step_path = _verified(root, (run_dir / "step-metrics.jsonl").as_posix(),
                          receipt.get("online_replay_manifest_sha256"), "r7 exact-budget step ledger")
    _verify_step_ledger(step_path, training_roads)
    _verified(root, (run_dir / "drift.jsonl").as_posix(),
              receipt.get("diagnostic_trace_sha256"), "r7 offline drift ledger")
    return {
        "role": f"{variant}-{condition}", "arm": f"{variant}-{condition}",
        "variant": variant, "condition": condition, "source_learner_seed": seed,
        "source_actor_sha256": source["actor_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "actor_path": root / paths["actor"], "actor_sha256": final["actor_sha256"],
        "actor_weights_sha256": final.get("actor_weights_sha256"),
        "checkpoint_path": root / paths["checkpoint"], "checkpoint_sha256": final["checkpoint_sha256"],
        "checkpoint_manifest_path": root / paths["checkpoint_manifest"],
        "checkpoint_manifest_sha256": manifest_sha,
        "checkpoint_online_step": FINAL_STEP,
        "catalog_path": catalog_path, "catalog_sha256": r6.sha256_file(catalog_path),
        "result_path": result_path, "result_sha256": r6.sha256_file(result_path),
        "sample_trace_path": root / trace_rel, "sample_trace_sha256": final["sample_trace_sha256"],
        "step_ledger_sha256": receipt["online_replay_manifest_sha256"],
        "run_config_sha256": r6.sha256_file(config_path),
    }


def _prepare(root: Path, protocol_sha256: str, loader: Callable[..., Any]
             ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]],
                        dict[tuple[int, int], dict[str, Any]],
                        dict[tuple[int, int, str], dict[str, Any]], dict[str, Any]]:
    protocol = _pinned(root, PROTOCOL.as_posix(), protocol_sha256, "r7 study protocol")
    grid = _grid(protocol)
    r6_protocol = _pinned(root, r6.PROTOCOL_PATH.as_posix(), R6_PROTOCOL_SHA, "frozen r6 protocol")
    _check(r6_protocol.get("format") == r6.PROTOCOL_FORMAT, "r6 protocol format mismatch")
    r6._validate_study_contract(r6_protocol)
    _check(protocol.get("study_id") == "drqv2-retention-r7"
           and protocol.get("r6_contract") == "inherit-learner-budget-geometry-environment-verbatim"
           and protocol.get("catalog_sha256") == r6.CATALOG_SHA256
           and protocol.get("diagnostic_manifest_sha256") == R6_MANIFEST_SHA
           and type(protocol.get("lambda_preserve")) in (float, int)
           and protocol["lambda_preserve"] == 0.5
           and protocol.get("interpretation", {}).get("no_post_result_lambda_or_threshold_change") is True,
           "r7 protocol changed r6 inheritance, frozen lambda, or original diagnostic contract")
    runtime = r6._validate_cpu21(r6_protocol)
    runtime_sources = _runtime_sources(root, r6_protocol)
    rows, lineage = r6._catalog_rows(root, r6_protocol)
    source_protocol = _pinned(root, r6.CATALOG_PROTOCOL_PATH.as_posix(),
                              r6.CATALOG_PROTOCOL_SHA256, "catalog generator protocol")
    sources = r6._source_records(root, r6_protocol, source_protocol)
    source_replay = protocol.get("source_replay")
    cache = protocol.get("diagnostic_cache")
    _check(isinstance(source_replay, dict) and set(source_replay) == {"0", "1"}
           and isinstance(cache, dict) and isinstance(cache.get("path"), str)
           and isinstance(cache.get("sha256"), str),
           "r7 source replay or read-only diagnostic cache lineage missing")
    for seed in (0, 1):
        _check(isinstance(source_replay[str(seed)], dict)
               and source_replay[str(seed)].get("checkpoint_path")
               == sources[seed]["checkpoint_path"].relative_to(root).as_posix()
               and source_replay[str(seed)].get("checkpoint_sha256") == sources[seed]["checkpoint_sha256"],
               "r7 source replay is not the original source checkpoint")
        _verified(root, source_replay[str(seed)].get("episode_ledger_path"),
                  source_replay[str(seed)].get("episode_ledger_sha256"),
                  "r7 source-training episode provenance")
    probe = protocol.get("gradient_probe")
    _check(isinstance(probe, dict), "r7 independent gradient probe lineage missing")
    _verified(root, probe.get("path"), probe.get("sha256"), "r7 fixed gradient-scale probe")
    _verified(root, cache["path"], cache["sha256"], "r7 TRAIN-DIAGNOSTIC drift-only cache")
    _verified(root, cache.get("hybrid_result_path"), cache.get("hybrid_result_sha256"),
              "r7 offline hybrid cache receipt")
    previous_source, previous_r6, reference = _r6_reference(root, r6_protocol, rows)
    for seed in (0, 1):
        _check(all(previous_source[(seed, road["geometry_seed"])]["actor_sha256"]
                   == sources[seed]["actor_sha256"] for road in rows),
               "r6 source outcome is not the frozen source actor")

    # All 12 complete receipts and their final artifacts are checked before loading
    # actors or constructing even one simulator instance.
    training_roads = set(r6_protocol["environment"]["geometry_seeds"])
    candidates = {key: _candidate(root, path, key, protocol_sha256, sources[key[0]],
                                  protocol, training_roads)
                  for key, path in sorted(grid.items())}
    _check(len({value["actor_sha256"] for value in candidates.values()}) == 12,
           "r7 candidate actor exports are reused between treatment arms")
    roles: dict[str, dict[str, Any]] = {}
    for key, candidate in candidates.items():
        actor, adapter, observation_spec, weights_sha = r6._validate_actor(
            candidate["actor_path"], candidate["actor_sha256"],
            candidate["actor_weights_sha256"], loader=loader,
        )
        candidate["actor_weights_sha256"] = weights_sha
        role_key = f"{key[1]}-{key[2]}-seed{key[0]}"
        roles[role_key] = {**candidate, "actor": actor, "action_adapter": adapter,
                           "observation_spec": observation_spec}
    return rows, roles, previous_source, previous_r6, {
        **reference, **lineage, "runtime": runtime, "protocol_sha256": protocol_sha256,
        "protocol_path": PROTOCOL.as_posix(), "r6_runtime_sources_sha256": runtime_sources,
    }


def _paired_summary(episodes: list[dict[str, Any]], rows: list[dict[str, Any]],
                    sources: dict[tuple[int, int], dict[str, Any]],
                    previous: dict[tuple[int, int, str], dict[str, Any]],
                    lineage: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    roads = {row["geometry_seed"]: row for row in rows}
    expected = {(seed, road, variant, condition, repeat)
                for seed in (0, 1) for road in roads for variant in VARIANTS
                for condition in CONDITIONS for repeat in (0, 1)}
    indexed = {}
    for episode in episodes:
        key = (episode["source_learner_seed"], episode["geometry_seed"],
               episode["variant"], episode["condition"], episode["repeat"])
        _check(key in expected and key not in indexed
               and episode["track_id"] == 1
               and episode["family"] == roads[key[1]]["family"]
               and type(episode["finished"]) is bool
               and episode.get("repeat_trace_agrees_with_repeat0") is True,
               "r7 episode grid is duplicate, divergent or changed geometry")
        indexed[key] = episode
    _check(set(indexed) == expected, "r7 canonical/repeat episode grid is incomplete")
    pairs = []
    for seed, road, variant, condition in sorted({key[:4] for key in expected}):
        current = indexed[(seed, road, variant, condition, 0)]
        source, old = sources[(seed, road)], previous[(seed, road, variant)]
        _check(current["source_actor_sha256"] == source["actor_sha256"]
               and current["source_checkpoint_sha256"] == source["checkpoint_sha256"],
               "r7 result is not paired to its original source actor")
        pairs.append({
            "source_seed": seed, "geometry_seed": road, "track_id": 1,
            "family": current["family"], "variant": variant, "condition": condition,
            "source_finished": source["finished"], "r6_finished": old["finished"],
            "r7_finished": current["finished"],
            "source_to_r7": r6_transition(source["finished"], current["finished"]),
            "r6_to_r7": r6_transition(old["finished"], current["finished"]),
            "source_trace_sha256": source["trace_sha256"],
            "r6_trace_sha256": old["trace_sha256"],
            "r7_trace_sha256": current["trace_sha256"],
            "r7_max_progress": current["max_progress"],
            "r7_raw_reward_sum": current["raw_reward_sum"],
        })
    groups = []
    for condition in CONDITIONS:
        for variant in VARIANTS:
            selected = [pair for pair in pairs if pair["condition"] == condition and pair["variant"] == variant]
            _check(len(selected) == 32 and sum(pair["source_finished"] for pair in selected) == 11,
                   "r7 arm does not contain all 32 source-actor/road pairs")
            for seed, family in ([(None, None)]
                                 + [(seed, None) for seed in (0, 1)]
                                 + [(None, family) for family in r6.EXPECTED_FAMILIES]
                                 + [(seed, family) for seed in (0, 1)
                                    for family in r6.EXPECTED_FAMILIES]):
                group = [pair for pair in selected
                         if (seed is None or pair["source_seed"] == seed)
                         and (family is None or pair["family"] == family)]
                counts = Counter(pair["source_to_r7"] for pair in group)
                baseline_counts = Counter(pair["r6_to_r7"] for pair in group)
                groups.append({
                    "condition": condition, "variant": variant, "source_seed": seed, "family": family,
                    "paired_cells": len(group), "source_finishes": sum(pair["source_finished"] for pair in group),
                    "r6_finishes": sum(pair["r6_finished"] for pair in group),
                    "r7_finishes": sum(pair["r7_finished"] for pair in group),
                    "source_to_r7": {kind: counts[kind] for kind in
                                     ("lost", "gained", "both_finished", "both_failed")},
                    "r6_to_r7": {kind: baseline_counts[kind] for kind in
                                 ("lost", "gained", "both_finished", "both_failed")},
                    "source_success_roads": sorted(pair["geometry_seed"] for pair in group
                                                   if pair["source_finished"]),
                    "source_success_cells": [{"source_seed": pair["source_seed"],
                                              "geometry_seed": pair["geometry_seed"]}
                                             for pair in group if pair["source_finished"]],
                    "kept_source_success_roads": sorted(pair["geometry_seed"] for pair in group
                                                        if pair["source_to_r7"] == "both_finished"),
                    "lost_source_success_roads": sorted(pair["geometry_seed"] for pair in group
                                                        if pair["source_to_r7"] == "lost"),
                    "gained_roads": sorted(pair["geometry_seed"] for pair in group
                                           if pair["source_to_r7"] == "gained"),
                    "gained_cells": [{"source_seed": pair["source_seed"],
                                     "geometry_seed": pair["geometry_seed"]}
                                    for pair in group if pair["source_to_r7"] == "gained"],
                    "original_r6_gained_roads": sorted(pair["geometry_seed"] for pair in group
                                                       if not pair["source_finished"] and pair["r6_finished"]),
                })
    summary = {
        "format": DIAGNOSTIC_FORMAT, "role": "TRAIN-DIAGNOSTIC", "ranked": False,
        "score_selection": False, "canonical_repeat": 0, "repeats": 2,
        "geometry_count": 16, "source_seed_count": 2, "candidate_role_count": 12,
        "expected_episode_count": 384, "observed_episode_count": len(episodes),
        "comparison": "32 paired source-actor/road cells per condition/variant on reused TRAIN-DIAGNOSTIC; repeats are determinism checks, not additional independent trials",
        "performance_scope": "internal raw-reward development proxy only; not fresh generalization, screen, confirmation, blind or official HAIC performance",
        "paired_source_and_r6": groups, "paired_json": "paired.json",
        **lineage,
    }
    return summary, {"format": DIAGNOSTIC_FORMAT, "role": "TRAIN-DIAGNOSTIC",
                     "ranked": False, "score_selection": False,
                     "canonical_repeat": 0, "paired_cells": len(pairs), "pairs": pairs,
                     **lineage}


def run_diagnostic(*, root: Path, protocol_sha256: str, output_root: Path,
                   environment_factory: Callable[[int, int], Any] | None = None,
                   actor_loader: Callable[..., Any] | None = None) -> dict[str, Any]:
    root = root.resolve()
    output = r6._user_repo_file(root, output_root, "runs", "r7 diagnostic output")
    _check(output == root / RUN_ROOT / "train-diagnostic",
           "r7 output must be the separate new run root's train-diagnostic/ directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"r7 TRAIN-DIAGNOSTIC output already exists: {output}")
    _check(output.parent.is_dir() and not output.parent.is_symlink(),
           "r7 output parent must exist and must not be a symlink")
    rows, roles, sources, old, lineage = _prepare(root, protocol_sha256,
                                                  actor_loader or load_exported_actor)
    factory = environment_factory or r6.build_environment
    output.mkdir(exist_ok=False)
    (output / "traces").mkdir()
    episodes: list[dict[str, Any]] = []
    inventory: dict[str, str] = {}
    for road in rows:
        for role_name, role in roles.items():
            env = factory(road["geometry_seed"], r6.MAX_STEPS)
            try:
                canonical = None
                for repeat in range(r6.REPEATS):
                    episode, arrays = r6._collect_episode(
                        env, role, road, repeat, episode_id=len(episodes) // 2,
                        max_steps=r6.MAX_STEPS, root=root,
                    )
                    episode["format"] = EPISODE_FORMAT
                    episode["condition"] = role["condition"]
                    episode["variant"] = role["variant"]
                    episode["result_path"] = role["result_path"].relative_to(root).as_posix()
                    episode["result_sha256"] = role["result_sha256"]
                    relative = Path("traces") / f"seed-{road['geometry_seed']}" / role_name / f"repeat-{repeat}.npz"
                    trace = output / relative
                    trace.parent.mkdir(parents=True, exist_ok=True)
                    with trace.open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    episode["trace_path"] = relative.as_posix()
                    episode["trace_sha256"] = r6.sha256_file(trace)
                    episode["trace_deterministic_sha256"] = r6._trace_digest(arrays)
                    episode["cell_id"] = f"seed-{road['geometry_seed']}:track-1:{role_name}:repeat-{repeat}"
                    episode["catalog_sha256"] = r6.CATALOG_SHA256
                    episode["protocol_sha256"] = protocol_sha256
                    if repeat == 0:
                        canonical = episode
                    else:
                        _check(canonical is not None
                               and episode["trace_deterministic_sha256"] == canonical["trace_deterministic_sha256"],
                               f"r7 repeated trace disagreed with repeat 0: {episode['cell_id']}")
                        canonical["repeat_trace_agrees_with_repeat0"] = True
                        episode["repeat_trace_agrees_with_repeat0"] = True
                    inventory[relative.as_posix()] = episode["trace_sha256"]
                    episodes.append(episode)
            finally:
                env.close()
    summary, paired = _paired_summary(episodes, rows, sources, old, lineage)
    episodes_path = output / "episodes.jsonl"
    with episodes_path.open("x", encoding="utf-8") as stream:
        for row in episodes:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    r6._write_json(output / "summary.json", summary)
    r6._write_json(output / "paired.json", paired)
    inventory.update({name: r6.sha256_file(output / name)
                      for name in ("episodes.jsonl", "summary.json", "paired.json")})
    manifest = {
        "format": DIAGNOSTIC_FORMAT, "role": "TRAIN-DIAGNOSTIC", "ranked": False,
        "score_selection": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell_inventory": "16 reused TRAIN-DIAGNOSTIC roads x 12 full-budget r7 actors x 2 repeats",
        "episode_count": len(episodes), "repeat_replay_determinism": "all 192 repeat pairs match",
        "evaluator_source_sha256": r6.sha256_file(Path(__file__)),
        "candidate_receipts": [{key: value for key, value in role.items()
                                if key in ("variant", "condition", "source_learner_seed", "actor_sha256",
                                           "actor_weights_sha256", "checkpoint_sha256", "checkpoint_manifest_sha256",
                                           "catalog_sha256", "result_sha256", "sample_trace_sha256")}
                               for role in roles.values()],
        **lineage, "files_sha256": dict(sorted(inventory.items())),
    }
    r6._write_json(output / "manifest.json", manifest)
    with (output / "manifest.sha256").open("x", encoding="ascii") as stream:
        stream.write(f"{r6.sha256_file(output / 'manifest.json')}  manifest.json\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha256", required=True,
                        help="frozen SHA-256 of experiments/drqv2-retention-r7.json")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT / "train-diagnostic")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        result = run_diagnostic(root=args.repo_root, protocol_sha256=args.protocol_sha256,
                                output_root=args.output_root)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        parser.exit(2, f"r7 TRAIN-DIAGNOSTIC stopped before completion: {error}\n")
    print(json.dumps({"output_root": str(args.output_root), "episodes": result["observed_episode_count"],
                      "ranked": False, "protocol_sha256": result["protocol_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
