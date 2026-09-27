"""CPU21 diagnostic for the six frozen final-source-replay DrQ treatments.

Only the 16 already-reused TRAIN-DIAGNOSTIC roads are opened. Source and r7b
controls are read from their immutable r6/r7 traces, never rolled out again.
Preflight is read-only and must succeed for the entire grid before any reset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from drq_v2 import load_exported_actor
from scripts import diagnose_drq_geometry_mix as r6
from scripts import diagnose_drq_retention_r7 as r7


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-drq-final-source-replay-study-v1"
STUDY_ID = "drqv2-final-source-replay-v1"
RUN_ROOT = Path("runs/20260927-drqv2-final-source-replay-v1")
CONDITION = "final_source"
DIAGNOSTIC_FORMAT = "haic-drq-final-source-replay-diagnostic-v1"
EPISODE_FORMAT = "haic-drq-final-source-replay-diagnostic-episode-v1"
ROLES = {(seed, variant) for seed in (0, 1) for variant in r6.VARIANTS}
R7_PROTOCOL_SHA = "774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9"
R7_MANIFEST_SHA = "0e4662102bd1d2f10536a04e9c7dfa8e48ef3952ba4d0782e6e4c89791835678"
R7_EPISODES_SHA = "0ce47ff1f3c00535d200e085c7d8c72405bfce0773311edb836cf0dfde1c8924"
R7_PAIRED_SHA = "8f10b5dae1701c5832079bfec2298fd8f9f916cffc5a839ea0a4669bea9d5f37"
R7_DIAGNOSTIC = Path("runs/20260926-drqv2-retention-r7/train-diagnostic")
SAMPLE_AUDIT = RUN_ROOT / "pre-evaluation-sample-audit.json"
TRANSITIONS = ("both_finished", "lost", "gained", "both_failed")


class DiagnosticError(r7.DiagnosticError):
    """An input, provenance, parity, or gate contract is incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _verified(root: Path, relative: Any, digest: Any, label: str) -> Path:
    try:
        return r7._verified(root, relative, digest, label)
    except r7.DiagnosticError as error:
        raise DiagnosticError(str(error)) from error


def _read_pinned(root: Path, relative: Any, digest: Any, label: str) -> dict[str, Any]:
    try:
        return r7._pinned(root, relative, digest, label)
    except r7.DiagnosticError as error:
        raise DiagnosticError(str(error)) from error


def _grid(protocol: dict[str, Any]) -> dict[tuple[int, str], Path]:
    _require(protocol.get("format") == FORMAT and protocol.get("study_id") == STUDY_ID
             and protocol.get("run_root") == RUN_ROOT.as_posix()
             and protocol.get("r6_protocol_path") == r6.PROTOCOL_PATH.as_posix()
             and protocol.get("r6_protocol_sha256") == r7.R6_PROTOCOL_SHA
             and protocol.get("r7_protocol_path") == r7.PROTOCOL.as_posix()
             and protocol.get("r7_protocol_sha256") == R7_PROTOCOL_SHA
             and protocol.get("r6_manifest_sha256") == r7.R6_MANIFEST_SHA
             and protocol.get("r7_manifest_sha256") == R7_MANIFEST_SHA,
             "new protocol format, study, root or original control pin changed")
    _require(protocol.get("lambda_preserve") == .5
             and type(protocol.get("lambda_preserve")) in (float, int),
             "new protocol changes fixed r7b preservation weight")
    replay = protocol.get("replay_contract")
    _require(isinstance(replay, dict) and replay.get("batch_size") == 64
             and replay.get("source_rows") == 32 and replay.get("online_rows") == 32
             and replay.get("source_kind") == "final_source_policy",
             "new protocol changes exact final-source 32:32 replay treatment")
    runs = protocol.get("runs")
    _require(isinstance(runs, list) and len(runs) == 6, "protocol must declare exactly six treatments")
    grid: dict[tuple[int, str], Path] = {}
    for run in runs:
        _require(isinstance(run, dict), "malformed treatment arm")
        seed, variant = run.get("source_seed"), run.get("variant")
        key = (seed, variant)
        _require(type(seed) is int and key in ROLES and key not in grid
                 and run.get("condition") == CONDITION,
                 "duplicate or unexpected source-seed/mixture/final-source arm")
        expected = RUN_ROOT / f"learner-{seed}-{variant}-{CONDITION}"
        _require(run.get("run_dir") == expected.as_posix(), "treatment run path is not the new root")
        grid[key] = expected
    _require(set(grid) == ROLES, "six-treatment grid is incomplete")
    return grid


def _protocol(root: Path, protocol_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    path = r6._user_repo_file(root, protocol_path, "experiments", "final-source study protocol")
    _require(path.is_file() and not path.is_symlink(), "new study protocol is missing")
    digest = r6.sha256_file(path)
    protocol = _read_pinned(root, path.relative_to(root).as_posix(), digest, "new study protocol")
    _grid(protocol)
    old = _read_pinned(root, r7.PROTOCOL.as_posix(), R7_PROTOCOL_SHA, "original r7 protocol")
    _require(old.get("format") == r7.FORMAT and old.get("study_id") == "drqv2-retention-r7"
             and old.get("r6_protocol_sha256") == r7.R6_PROTOCOL_SHA
             and old.get("diagnostic_manifest_sha256") == r7.R6_MANIFEST_SHA
             and old.get("lambda_preserve") == .5,
             "original r7b comparator contract changed")
    matched_rng = {(row["source_seed"], row["variant"]): row["rng_seeds"]
                   for row in old["runs"] if row["condition"] == "r7b"}
    _require(set(matched_rng) == ROLES and protocol.get("catalog_sha256") == r6.CATALOG_SHA256
             and protocol.get("diagnostic_cache") == old.get("diagnostic_cache")
             and all(row.get("rng_seeds") == matched_rng[(row["source_seed"], row["variant"])]
                     for row in protocol["runs"]),
             "final-source treatment changes r7b source RNG, catalog or diagnostic-only cache")
    frozen = _read_pinned(root, r6.PROTOCOL_PATH.as_posix(), r7.R6_PROTOCOL_SHA,
                          "original r6 protocol")
    _require(frozen.get("format") == r6.PROTOCOL_FORMAT, "original r6 protocol format changed")
    r6._validate_study_contract(frozen)
    _require(frozen["diagnostics"]["geometry_seeds"] == list(r6.DIAGNOSTIC_SEEDS),
             "diagnostic road order differs from original r6")
    code = protocol.get("code_sha256")
    imported = ("scripts/diagnose_drq_final_replay.py", "scripts/diagnose_drq_geometry_mix.py",
                "scripts/diagnose_drq_retention_r7.py", "scripts/audit_drq_final_replay_samples.py")
    _require(isinstance(code, dict) and all(name in code for name in imported),
             "new protocol must pin this evaluator and both imported evaluator helpers")
    for relative, expected in code.items():
        _require(isinstance(relative, str) and bool(Path(relative).parts),
                 "invalid executable source path in frozen protocol")
        path = r6._repo_file(root, relative, Path(relative).parts[0], "pinned executable source")
        _require(path.is_file() and not path.is_symlink()
                 and r6.sha256_file(path) == r7._hash(expected, relative),
                 f"pinned executable source changed: {relative}")
    return protocol, digest, frozen


def _trace(root: Path, directory: Path, row: dict[str, Any], files: dict[str, str]) -> None:
    relative = row.get("trace_path")
    _require(isinstance(relative, str) and relative.startswith("traces/")
             and row.get("trace_sha256") == files.get(relative),
             "archived control trace is not bound to its immutable manifest")
    path = _verified(root, (directory / relative).as_posix(), files[relative], "archived control trace")
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == set(r6.TRACE_ARRAYS), "archived control trace arrays changed")
        digest = r6._trace_digest({name: archive[name] for name in r6.TRACE_ARRAYS})
    _require(digest == row.get("trace_deterministic_sha256"),
             "archived control trace contents disagree with episode digest")


def _controls(root: Path, rows: list[dict[str, Any]], frozen: dict[str, Any]
              ) -> tuple[dict[tuple[int, int], dict[str, Any]],
                         dict[tuple[int, int, str], dict[str, Any]], dict[str, str]]:
    sources, old_r6, lineage = r7._r6_reference(root, frozen, rows)
    r6_manifest = _read_pinned(root, (r7.R6_DIAGNOSTIC / "manifest.json").as_posix(),
                               r7.R6_MANIFEST_SHA, "old r6 manifest")
    source_files = r6_manifest["files_sha256"]
    source_episodes = r7._file(root, (r7.R6_DIAGNOSTIC / "episodes.jsonl").as_posix(),
                               "old source episode inventory")
    for line in source_episodes.read_bytes().splitlines():
        entry = r6._parse_json(line, source_episodes)
        if entry.get("arm") == "unchanged-source":
            if entry.get("repeat") == 0:
                _require(entry.get("trace_sha256") == sources[(entry["source_learner_seed"],
                                                                 entry["geometry_seed"])]["trace_sha256"],
                         "old source canonical row changed")
            _trace(root, r7.R6_DIAGNOSTIC, entry, source_files)

    manifest = _read_pinned(root, (R7_DIAGNOSTIC / "manifest.json").as_posix(),
                            R7_MANIFEST_SHA, "original r7 diagnostic manifest")
    sidecar = r7._file(root, (R7_DIAGNOSTIC / "manifest.sha256").as_posix(), "r7 manifest sidecar")
    _require(sidecar.read_text(encoding="ascii") == f"{R7_MANIFEST_SHA}  manifest.json\n"
             and manifest.get("format") == r7.DIAGNOSTIC_FORMAT
             and manifest.get("role") == "TRAIN-DIAGNOSTIC"
             and manifest.get("ranked") is False and manifest.get("score_selection") is False
             and manifest.get("protocol_sha256") == R7_PROTOCOL_SHA
             and manifest.get("r6_manifest_sha256") == r7.R6_MANIFEST_SHA
             and manifest.get("catalog_sha256") == r6.CATALOG_SHA256
             and manifest.get("episode_count") == 384
             and manifest.get("evaluator_source_sha256") == r6.sha256_file(root / "scripts/diagnose_drq_retention_r7.py"),
             "original r7 manifest lineage changed")
    files = manifest.get("files_sha256")
    _require(isinstance(files, dict) and len(files) == 387
             and files.get("episodes.jsonl") == R7_EPISODES_SHA
             and files.get("paired.json") == R7_PAIRED_SHA,
             "r7 immutable trace/episode/paired inventory is incomplete")
    path = _verified(root, (R7_DIAGNOSTIC / "episodes.jsonl").as_posix(), R7_EPISODES_SHA,
                     "original r7 episodes")
    _verified(root, (R7_DIAGNOSTIC / "summary.json").as_posix(), files.get("summary.json"),
              "original r7 summary")
    paired = _read_pinned(root, (R7_DIAGNOSTIC / "paired.json").as_posix(), R7_PAIRED_SHA,
                          "original r7 paired cells")
    _require(paired.get("format") == r7.DIAGNOSTIC_FORMAT and paired.get("paired_cells") == 192
             and paired.get("catalog_sha256") == r6.CATALOG_SHA256
             and paired.get("protocol_sha256") == R7_PROTOCOL_SHA,
             "r7 paired control lineage changed")
    roads = {row["geometry_seed"]: row for row in rows}
    expected_all = {(seed, road, variant, condition, repeat)
                    for seed in (0, 1) for road in roads for variant in r6.VARIANTS
                    for condition in r7.CONDITIONS for repeat in (0, 1)}
    entries: dict[tuple[int, int, str, str, int], dict[str, Any]] = {}
    for line in path.read_bytes().splitlines():
        entry = r6._parse_json(line, path)
        _require(isinstance(entry, dict), "old r7 episode is not an object")
        key = (entry.get("source_learner_seed"), entry.get("geometry_seed"),
               entry.get("variant"), entry.get("condition"), entry.get("repeat"))
        _require(key in expected_all and key not in entries
                 and entry.get("family") == roads[key[1]]["family"]
                 and entry.get("track_id") == 1 and entry.get("protocol_sha256") == R7_PROTOCOL_SHA
                 and entry.get("catalog_sha256") == r6.CATALOG_SHA256
                 and entry.get("canonical_repeat") is (key[4] == 0)
                 and entry.get("repeat_trace_agrees_with_repeat0") is True
                 and type(entry.get("finished")) is bool,
                 "old r7 episode grid or outcome has changed")
        role = f"{key[2]}-{key[3]}-seed{key[0]}"
        _require(entry.get("trace_path") == f"traces/seed-{key[1]}/{role}/repeat-{key[4]}.npz"
                 and entry.get("trace_sha256") == files.get(entry["trace_path"]),
                 "old r7 episode trace path/hash mismatch")
        entries[key] = entry
    _require(set(entries) == expected_all, "old r7 episode inventory incomplete")
    controls: dict[tuple[int, int, str], dict[str, Any]] = {}
    for seed, road, variant in sorted(old_r6):
        first = entries[(seed, road, variant, "r7b", 0)]
        second = entries[(seed, road, variant, "r7b", 1)]
        _require(first["trace_deterministic_sha256"] == second["trace_deterministic_sha256"]
                 and first["finished"] is second["finished"]
                 and first["source_actor_sha256"] == sources[(seed, road)]["actor_sha256"]
                 and first["source_checkpoint_sha256"] == sources[(seed, road)]["checkpoint_sha256"],
                 "old r7b control repeat/source actor changed")
        _trace(root, R7_DIAGNOSTIC, first, files)
        _trace(root, R7_DIAGNOSTIC, second, files)
        controls[(seed, road, variant)] = first
    original_pairs = paired.get("pairs")
    _require(isinstance(original_pairs, list) and len(original_pairs) == 192,
             "original r7 paired control inventory incomplete")
    seen: set[tuple[int, int, str, str]] = set()
    for pair in original_pairs:
        _require(isinstance(pair, dict), "old paired control is not an object")
        key = (pair.get("source_seed"), pair.get("geometry_seed"), pair.get("variant"),
               pair.get("condition"))
        _require(key not in seen and key[:3] in old_r6 and key[3] in r7.CONDITIONS,
                 "old r7 paired control duplicated or wrong role")
        seen.add(key)
        if key[3] == "r7b":
            source, old, control = sources[key[:2]], old_r6[key[:3]], controls[key[:3]]
            _require(pair.get("source_finished") is source["finished"]
                     and pair.get("r6_finished") is old["finished"]
                     and pair.get("r7_finished") is control["finished"]
                     and pair.get("family") == control["family"] and pair.get("track_id") == 1
                     and pair.get("source_trace_sha256") == source["trace_sha256"]
                     and pair.get("r6_trace_sha256") == old["trace_sha256"]
                     and pair.get("r7_trace_sha256") == control["trace_sha256"]
                     and pair.get("source_to_r7") == r7.r6_transition(source["finished"], control["finished"]),
                     "old r7b paired control disagrees with archived source/r6/r7b episodes")
    _require(len(seen) == 192 and {variant: (
        sum(sources[(seed, road)]["finished"] and control["finished"]
            for (seed, road, name), control in controls.items() if name == variant),
        sum(not sources[(seed, road)]["finished"] and control["finished"]
            for (seed, road, name), control in controls.items() if name == variant))
        for variant in r6.VARIANTS} == {"uniform": (4, 5), "failure_weighted": (1, 7),
                                       "easy_retention": (5, 5)},
             "old r7b comparator is not its frozen K/G control")
    return sources, controls, {**lineage, "r7_protocol_sha256": R7_PROTOCOL_SHA,
                               "r7_manifest_sha256": R7_MANIFEST_SHA,
                               "r7_episodes_sha256": R7_EPISODES_SHA,
                               "r7_paired_sha256": R7_PAIRED_SHA}


def _pools(root: Path, protocol: dict[str, Any], sources: dict[int, dict[str, Any]],
           diagnostic_roads: set[int]) -> dict[int, dict[str, str]]:
    declared = protocol.get("source_replay")
    _require(isinstance(declared, dict) and set(declared) == {"0", "1"},
             "exactly two separately pinned final-source pools required")
    pools = {}
    for seed in (0, 1):
        item = declared[str(seed)]
        _require(isinstance(item, dict), "invalid final-source pool declaration")
        pool = _verified(root, item.get("pool_path"), item.get("pool_sha256"), "final-source replay pool")
        receipt = _read_pinned(root, item.get("receipt_path"), item.get("receipt_sha256"),
                               "final-source collection receipt")
        source = sources[seed]
        _require(pool != source["checkpoint_path"]
                 and receipt.get("format") == "haic-drq-final-source-pool-v1"
                 and receipt.get("completed") is True
                 and type(receipt.get("source_seed")) is int and receipt["source_seed"] == seed
                 and receipt.get("source_actor_sha256") == source["actor_sha256"]
                 and receipt.get("source_checkpoint_sha256") == source["checkpoint_sha256"]
                 and receipt.get("pool_path") == item["pool_path"]
                 and receipt.get("pool_sha256") == item["pool_sha256"]
                 and receipt.get("partition") == "TRAIN"
                 and receipt.get("excluded_diagnostic_roads") is True
                 and receipt.get("decisions") == 100000 and receipt.get("capacity") == 100000
                 and type(receipt.get("valid_n_step_starts")) is int
                 and receipt["valid_n_step_starts"] >= 32
                 and receipt.get("catalog_sha256") == r6.CATALOG_SHA256,
                 "final-source replay pool has incomplete source/TRAIN provenance")
        old_ledger = _read_pinned(root, r7.PROTOCOL.as_posix(), R7_PROTOCOL_SHA,
                                  "original r7 source protocol")["source_replay"][str(seed)]
        _require(receipt.get("original_ledger_sha256") == old_ledger["episode_ledger_sha256"]
                 and r6.SHA256_RE.fullmatch(str(receipt.get("schedule_sha256"))) is not None,
                 "pool schedule/original source ledger pin differs from frozen control")
        copy = (pool.parent / "collection_protocol.json").relative_to(root).as_posix()
        _verified(root, copy, receipt.get("collection_protocol_sha256"), "pool collection protocol copy")
        collection = _read_pinned(root, copy, receipt["collection_protocol_sha256"],
                                  "original final-source collection protocol")
        collection_sources = collection.get("sources")
        declared_seed = collection_sources.get(str(seed)) if isinstance(collection_sources, dict) else None
        _require(collection.get("format") == "haic-drq-final-source-collection-v1"
                 and collection.get("study_id") == STUDY_ID
                 and collection.get("run_root") == RUN_ROOT.as_posix()
                 and collection.get("r6_protocol_sha256") == r7.R6_PROTOCOL_SHA
                 and collection.get("r7_protocol_sha256") == R7_PROTOCOL_SHA
                 and collection.get("catalog_sha256") == r6.CATALOG_SHA256
                 and collection.get("collection_noise_std") == receipt.get("collection_noise_std") == .05
                 and isinstance(declared_seed, dict)
                 and declared_seed.get("schedule_sha256") == receipt["schedule_sha256"]
                 and declared_seed.get("original_ledger_sha256") == receipt["original_ledger_sha256"]
                 and declared_seed.get("source_actor_sha256") == source["actor_sha256"]
                 and declared_seed.get("source_checkpoint_sha256") == source["checkpoint_sha256"]
                 and declared_seed.get("collection_rng_seeds") == receipt.get("collection_rng_seeds"),
                 "final-source collection receipt differs from frozen schedule/actor/noise protocol")
        _verify_pool_ledgers(root, receipt, old_ledger, seed, diagnostic_roads)
        pools[seed] = {"pool_path": pool.relative_to(root).as_posix(),
                       "pool_sha256": item["pool_sha256"],
                       "receipt_path": item["receipt_path"],
                       "receipt_sha256": item["receipt_sha256"],
                       "schedule_sha256": receipt["schedule_sha256"],
                       "original_ledger_sha256": receipt["original_ledger_sha256"]}
    _require(pools[0]["pool_sha256"] != pools[1]["pool_sha256"],
             "seed 0 and seed 1 must not share one final-source replay pool")
    return pools


def _verify_pool_ledgers(root: Path, receipt: dict[str, Any], original: dict[str, Any],
                         seed: int, diagnostic_roads: set[int], *, decisions: int = 100000) -> None:
    original_path = _verified(root, original.get("episode_ledger_path"),
                              original.get("episode_ledger_sha256"), "original source TRAIN ledger")
    known = {}
    for raw in original_path.read_bytes().splitlines():
        row = r6._parse_json(raw, original_path)
        if row.get("event") == "reset":
            key = row.get("episode_id")
            _require(type(key) is int and key not in known and type(row.get("seed")) is int
                     and type(row.get("track_id")) is int and row["track_id"] in (1, 2, 3, 4),
                     "original source TRAIN reset ledger has an invalid road")
            known[key] = (row["track_id"], row["seed"])
    _require(bool(known), "original source TRAIN ledger has no reset pairs")
    episodes = _verified(root, receipt.get("episode_ledger_path"),
                         receipt.get("episode_ledger_sha256"), "new pool episode ledger")
    steps = _verified(root, receipt.get("step_ledger_path"),
                      receipt.get("step_ledger_sha256"), "new pool step ledger")
    resets = {}
    ended = set()
    for raw in episodes.read_bytes().splitlines():
        row = r6._parse_json(raw, episodes)
        event, index, original_id = row.get("event"), row.get("episode_id"), row.get("original_episode_id")
        _require(type(index) is int and type(original_id) is int and original_id in known
                 and row.get("source_seed") == seed and row.get("schedule_index") == index
                 and (row.get("track_id"), row.get("seed")) == known[original_id]
                 and row.get("seed") not in diagnostic_roads,
                 "new pool reset is not one original source TRAIN pair")
        if event == "reset":
            _require(index == len(resets) and row.get("partition") == "TRAIN",
                     "new pool reset schedule is discontinuous or not TRAIN")
            resets[index] = known[original_id]
        else:
            _require(event in ("end", "capped_partial") and index in resets and index not in ended,
                     "new pool terminal event does not close its reset")
            ended.add(index)
    _require(bool(resets) and receipt.get("geometry_seeds") == sorted({road for _, road in resets.values()})
             and len(resets) == receipt.get("scheduled_episodes_consumed")
             and len(resets) == len(ended), "new pool has a missing or incomplete episode receipt")
    count = 0
    with steps.open("rb") as stream:
        for count, raw in enumerate(stream, 1):
            _require(count <= decisions, "new pool step ledger exceeds fixed collection budget")
            row = r6._parse_json(raw, steps)
            index = row.get("episode_id")
            _require(row.get("decision") == count and row.get("sequence_id") == count - 1
                     and type(index) is int and index in resets
                     and row.get("schedule_index") == index
                     and row.get("original_episode_id") in known
                     and (row.get("track_id"), row.get("geometry_seed")) == resets[index]
                     == known[row["original_episode_id"]]
                     and row.get("source_seed") == seed and row.get("partition") == "TRAIN",
                     f"new pool step {count} is not a recorded original TRAIN source road")
    _require(count == decisions, "new pool step ledger is truncated")


def _candidate(root: Path, run_dir: Path, key: tuple[int, str], protocol: dict[str, Any],
               protocol_sha: str, source: dict[str, Any], pool: dict[str, str],
               training_roads: set[int]) -> dict[str, Any]:
    seed, variant = key
    prefix = run_dir.as_posix()
    _verified(root, f"{prefix}/study_protocol.json", protocol_sha, "treatment protocol copy")
    config_path = r7._file(root, f"{prefix}/run-config.json", "final-source run config")
    config = r6._read_json(config_path)
    run = next(entry for entry in protocol["runs"] if (entry["source_seed"], entry["variant"]) == key)
    _require(isinstance(config, dict) and config.get("format") == "haic-drq-retention-run-config-v1"
             and config.get("protocol_sha256") == protocol_sha
             and (config.get("source_seed"), config.get("variant"), config.get("condition"))
             == (seed, variant, CONDITION)
             and config.get("source_actor_sha256") == source["actor_sha256"]
             and config.get("source_checkpoint_sha256") == source["checkpoint_sha256"]
             and config.get("source_replay_sha256") == pool["pool_sha256"]
             and config.get("source_pool_receipt_sha256") == pool["receipt_sha256"]
             and config.get("source_rows_per_batch") == 32
             and config.get("online_rows_per_batch") == 32
             and config.get("lambda_preserve") == .5
             and config.get("rng_seeds") == run.get("rng_seeds")
             and config.get("encoder_update") is True
             and config.get("diagnostic_cache_sha256") == protocol["diagnostic_cache"]["sha256"]
             and config.get("optimizer") == "Adam"
             and config.get("actor_lr") == config.get("critic_lr") == 1e-4,
             f"treatment config provenance/ratio differs from r7b for {key}")
    result_path = r7._file(root, f"{prefix}/result.json", "completed treatment result")
    result = r6._read_json(result_path)
    _require(isinstance(result, dict) and result.get("format") == r7.RESULT_FORMAT
             and result.get("study_id") == STUDY_ID and result.get("completed") is True
             and (result.get("source_seed"), result.get("variant"), result.get("condition"))
             == (seed, variant, CONDITION)
             and result.get("study_protocol_sha256") == protocol_sha
             and result.get("source_actor_sha256") == source["actor_sha256"]
             and result.get("source_checkpoint_sha256") == source["checkpoint_sha256"]
             and result.get("source_replay_sha256") == pool["pool_sha256"]
             and result.get("additional_online_steps") == r7.FINAL_STEP
             and result.get("study_gradient_steps") == r7.FINAL_UPDATES
             and result.get("source_samples") == r7.FINAL_UPDATES * 32
             and result.get("online_samples") == r7.FINAL_UPDATES * 32,
             f"treatment {key} lacks full-budget completed result")
    catalog_path = r7._file(root, f"{prefix}/checkpoint-catalog.json", "treatment checkpoint catalog")
    catalog = r6._read_json(catalog_path)
    candidates = result.get("candidates")
    _require(isinstance(catalog, dict) and catalog.get("format") == r7.CATALOG_FORMAT
             and catalog.get("study_protocol_sha256") == protocol_sha
             and (catalog.get("source_seed"), catalog.get("variant"), catalog.get("condition"))
             == (seed, variant, CONDITION)
             and catalog.get("candidates") == candidates and isinstance(candidates, list)
             and len(candidates) == 2 and all(isinstance(item, dict) for item in candidates)
             and [item.get("checkpoint_online_step") for item in candidates] == [16384, r7.FINAL_STEP],
             f"treatment {key} checkpoint catalog differs from full-budget receipt")
    final = candidates[1]
    step = run_dir / "checkpoints" / "step-000032768"
    paths = {"actor": step / "actor.pt", "checkpoint": step / "checkpoint.pt"}
    _require(final.get("study_gradient_steps") == r7.FINAL_UPDATES,
             f"treatment {key} final checkpoint update count differs")
    for name, relative in paths.items():
        _require(final.get(f"{name}_path") == relative.as_posix(),
                 f"treatment {key} final {name} is not step 32768")
        _verified(root, relative.as_posix(), final.get(f"{name}_sha256"), f"final {name}")
    manifest_path = step / "checkpoint.manifest.json"
    manifest = r6._read_json(r7._file(root, manifest_path.as_posix(), "final checkpoint manifest"))
    _require(isinstance(manifest, dict) and manifest.get("algorithm") == "drq-v2"
             and isinstance(manifest.get("extra"), dict)
             and manifest["extra"].get("environment_steps") == r6.SOURCE_STEPS + r7.FINAL_STEP
             and manifest["extra"].get("study_id") == STUDY_ID
             and manifest["extra"].get("study_protocol_sha256") == protocol_sha
             and manifest["extra"].get("source_checkpoint_sha256") == source["checkpoint_sha256"]
             and manifest["extra"].get("source_replay_sha256") == pool["pool_sha256"]
             and manifest["extra"].get("catalog_sha256") == r6.CATALOG_SHA256,
             f"treatment {key} checkpoint manifest/source-pool lineage differs")
    sample_path = run_dir / "replay-sample-trace-step-000032768.npz"
    _require(final.get("sample_trace_path") == sample_path.as_posix(),
             f"treatment {key} lacks full sampled-update trace")
    _verified(root, sample_path.as_posix(), final.get("sample_trace_sha256"), "final sampled-update trace")
    with np.load(root / sample_path, allow_pickle=False) as archive:
        _require(set(archive.files) == {"source", "source_indices", "episode_id", "protocol_sha256"},
                 f"treatment {key} sampled-update provenance arrays missing")
        tags, indices, episode_ids, pinned = (archive[name] for name in
                                             ("source", "source_indices", "episode_id", "protocol_sha256"))
        _require(tags.shape == indices.shape == episode_ids.shape == (r7.FINAL_UPDATES, 64)
                 and tags.dtype == np.uint8 and indices.dtype == np.int64
                 and episode_ids.dtype == np.int64
                 and np.array_equal(pinned, np.frombuffer(protocol_sha.encode("ascii"), dtype=np.uint8))
                 and np.all((tags == 0) | (tags == 1)) and np.all(tags.sum(axis=1) == 32),
                 f"treatment {key} sampled-update trace violates 32:32/protocol identity")
        for is_source in (True, False):
            chosen = np.where(tags == int(is_source), indices, -1)
            _require(np.all(np.diff(np.sort(chosen, axis=1)[:, 32:], axis=1) != 0),
                     f"treatment {key} samples duplicate rows in one update")
    ledger = _verified(root, f"{prefix}/step-metrics.jsonl",
                       result.get("online_replay_manifest_sha256"), "exact-budget online ledger")
    r7._verify_step_ledger(ledger, training_roads)
    _verified(root, f"{prefix}/drift.jsonl", result.get("diagnostic_trace_sha256"),
              "frozen diagnostic drift ledger")
    return {
        "role": f"{variant}-{CONDITION}", "arm": f"{variant}-{CONDITION}",
        "variant": variant, "condition": CONDITION, "source_learner_seed": seed,
        "source_actor_sha256": source["actor_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_replay_sha256": pool["pool_sha256"], "pool_receipt_sha256": pool["receipt_sha256"],
        "actor_path": root / paths["actor"], "actor_sha256": final["actor_sha256"],
        "actor_weights_sha256": final.get("actor_weights_sha256"),
        "checkpoint_path": root / paths["checkpoint"], "checkpoint_sha256": final["checkpoint_sha256"],
        "checkpoint_manifest_path": root / manifest_path,
        "checkpoint_manifest_sha256": r6.sha256_file(root / manifest_path),
        "checkpoint_online_step": r7.FINAL_STEP,
        "catalog_path": catalog_path, "catalog_sha256": r6.sha256_file(catalog_path),
        "result_path": result_path, "result_sha256": r6.sha256_file(result_path),
        "run_config_sha256": r6.sha256_file(config_path),
        "sample_trace_sha256": final["sample_trace_sha256"],
        "step_ledger_sha256": result["online_replay_manifest_sha256"],
    }


def _sample_audit(root: Path, protocol_sha: str, pools: dict[int, dict[str, str]],
                  records: dict[tuple[int, str], dict[str, Any]]) -> str:
    try:
        path = r7._file(root, SAMPLE_AUDIT.as_posix(), "independent semantic sample audit receipt")
    except r7.DiagnosticError as error:
        raise DiagnosticError(str(error)) from error
    sha = r6.sha256_file(path)
    receipt = _read_pinned(root, SAMPLE_AUDIT.as_posix(), sha, "independent semantic sample audit")
    _require(receipt.get("format") == "haic-drq-final-source-sample-audit-v1"
             and receipt.get("passed") is True and receipt.get("no_environment_resets") is True
             and receipt.get("protocol_sha256") == protocol_sha,
             "independent semantic source/online sample audit did not pass this protocol")
    sources = receipt.get("source_pool")
    _require(isinstance(sources, dict) and set(sources) == {"0", "1"},
             "independent sample audit lacks both final-source pools")
    for seed, pool in pools.items():
        row = sources[str(seed)]
        _require(isinstance(row, dict) and row.get("pool_sha256") == pool["pool_sha256"]
                 and row.get("receipt_sha256") == pool["receipt_sha256"]
                 and row.get("verified_source_decisions") == 100000
                 and type(row.get("valid_n_step_starts")) is int and row["valid_n_step_starts"] >= 32,
                 "independent semantic sample audit did not cover the sealed source pool")
    runs = receipt.get("runs")
    _require(isinstance(runs, list) and len(runs) == 6,
             "independent semantic sample audit lacks six completed treatments")
    seen = set()
    for row in runs:
        _require(isinstance(row, dict), "malformed independent sample-audit run")
        key = (row.get("source_seed"), row.get("variant"))
        _require(key in records and key not in seen
                 and row.get("run_dir") == (RUN_ROOT / f"learner-{key[0]}-{key[1]}-{CONDITION}").as_posix()
                 and row.get("result_sha256") == records[key]["result_sha256"]
                 and row.get("final_checkpoint_sha256") == records[key]["checkpoint_sha256"]
                 and row.get("final_sample_trace_sha256") == records[key]["sample_trace_sha256"]
                 and row.get("updates") == r7.FINAL_UPDATES and row.get("batch_size") == 64
                 and row.get("source_rows") == r7.FINAL_UPDATES * 32
                 and row.get("online_rows") == r7.FINAL_UPDATES * 32,
                 "semantic sample audit differs from treatment result/checkpoint/pool/trace")
        seen.add(key)
    _require(seen == ROLES, "independent sample audit omitted a seed/mixture")
    return sha


def _prepare(root: Path, protocol_path: Path, loader: Callable[..., Any]
             ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]],
                        dict[tuple[int, int], dict[str, Any]],
                        dict[tuple[int, int, str], dict[str, Any]], dict[str, Any]]:
    protocol, digest, frozen = _protocol(root, protocol_path)
    runtime = r6._validate_cpu21(frozen)
    runtime_sources = r7._runtime_sources(root, frozen)
    rows, lineage = r6._catalog_rows(root, frozen)
    catalog_protocol = _read_pinned(root, r6.CATALOG_PROTOCOL_PATH.as_posix(),
                                    r6.CATALOG_PROTOCOL_SHA256, "catalog protocol")
    seeds = r6._source_records(root, frozen, catalog_protocol)
    sources, controls, old = _controls(root, rows, frozen)
    for seed, road in sources:
        _require(sources[(seed, road)]["actor_sha256"] == seeds[seed]["actor_sha256"],
                 "archived source actor does not match frozen control")
    pools = _pools(root, protocol, seeds, {row["geometry_seed"] for row in rows})
    training = set(frozen["environment"]["geometry_seeds"])
    roles: dict[str, dict[str, Any]] = {}
    records = {}
    for key, directory in sorted(_grid(protocol).items()):
        records[key] = _candidate(root, directory, key, protocol, digest, seeds[key[0]],
                                  pools[key[0]], training)
    _require(len({item["actor_sha256"] for item in records.values()}) == 6,
             "treatment actor export reused between seed/mixture arms")
    audit_sha = _sample_audit(root, digest, pools, records)
    for (seed, variant), record in sorted(records.items()):
        actor, adapter, observation_spec, weights = r6._validate_actor(
            record["actor_path"], record["actor_sha256"], record["actor_weights_sha256"], loader=loader)
        record["actor_weights_sha256"] = weights
        roles[f"{variant}-{CONDITION}-seed{seed}"] = {
            **record, "actor": actor, "action_adapter": adapter, "observation_spec": observation_spec,
        }
    return rows, roles, sources, controls, {
        **old, **lineage, "runtime": runtime, "r6_runtime_sources_sha256": runtime_sources,
        "protocol_path": r6._user_repo_file(root, protocol_path, "experiments", "study").relative_to(root).as_posix(),
        "protocol_sha256": digest,
        "sample_audit_path": SAMPLE_AUDIT.as_posix(), "sample_audit_sha256": audit_sha,
        "source_pool_provenance": {str(seed): value for seed, value in pools.items()},
    }


def _paired_summary(episodes: list[dict[str, Any]], rows: list[dict[str, Any]],
                    sources: dict[tuple[int, int], dict[str, Any]],
                    controls: dict[tuple[int, int, str], dict[str, Any]],
                    lineage: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    roads = {row["geometry_seed"]: row for row in rows}
    expected = {(seed, road, variant, repeat) for seed in (0, 1) for road in roads
                for variant in r6.VARIANTS for repeat in (0, 1)}
    indexed: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    for episode in episodes:
        key = (episode.get("source_learner_seed"), episode.get("geometry_seed"),
               episode.get("variant"), episode.get("repeat"))
        _require(key in expected and key not in indexed
                 and episode.get("condition") == CONDITION
                 and episode.get("track_id") == 1 and episode.get("family") == roads[key[1]]["family"]
                 and type(episode.get("finished")) is bool
                 and episode.get("repeat_trace_agrees_with_repeat0") is True,
                 "new diagnostic episode grid/geometry/repeat is invalid")
        indexed[key] = episode
    _require(set(indexed) == expected, "new diagnostic is not exactly 192 episodes")
    pairs = []
    for seed, road, variant in sorted(controls):
        current, repeat = indexed[(seed, road, variant, 0)], indexed[(seed, road, variant, 1)]
        source, control = sources[(seed, road)], controls[(seed, road, variant)]
        _require(current["trace_deterministic_sha256"] == repeat["trace_deterministic_sha256"]
                 and current["finished"] is repeat["finished"]
                 and current["source_actor_sha256"] == source["actor_sha256"]
                 and current["source_checkpoint_sha256"] == source["checkpoint_sha256"],
                 "new diagnostic repeat/source pairing changed")
        pairs.append({
            "source_seed": seed, "geometry_seed": road, "track_id": 1,
            "family": current["family"], "variant": variant,
            "source_finished": source["finished"], "r7b_finished": control["finished"],
            "treatment_finished": current["finished"],
            "source_to_treatment": r7.r6_transition(source["finished"], current["finished"]),
            "r7b_to_treatment": r7.r6_transition(control["finished"], current["finished"]),
            "source_trace_sha256": source["trace_sha256"],
            "r7b_trace_sha256": control["trace_sha256"],
            "treatment_trace_sha256": current["trace_sha256"],
            "treatment_max_progress": current["max_progress"],
            "treatment_raw_reward_sum": current["raw_reward_sum"],
        })
    _require(len(pairs) == 96 and sum(pair["source_finished"] for pair in pairs) == 33,
             "matched source comparison is not 11/32 per mixture")
    source_never_finished = {road for road in roads if not any(
        sources[(seed, road)]["finished"] for seed in (0, 1))}
    groups = []
    for variant in r6.VARIANTS:
        selected = [pair for pair in pairs if pair["variant"] == variant]
        _require(len(selected) == 32, "mixture is not 32 paired actor/road cells")
        for seed, family in ([(None, None)] + [(s, None) for s in (0, 1)]
                             + [(None, f) for f in r6.EXPECTED_FAMILIES]
                             + [(s, f) for s in (0, 1) for f in r6.EXPECTED_FAMILIES]):
            matched = [pair for pair in selected if (seed is None or pair["source_seed"] == seed)
                       and (family is None or pair["family"] == family)]
            counts = Counter(pair["source_to_treatment"] for pair in matched)
            old_counts = Counter(pair["r7b_to_treatment"] for pair in matched)
            gained = [pair for pair in matched if pair["source_to_treatment"] == "gained"]
            source_success = [pair for pair in matched if pair["source_finished"]]
            groups.append({
                "variant": variant, "source_seed": seed, "family": family,
                "paired_actor_road_cells": len(matched), "distinct_reused_road_count": len({p["geometry_seed"] for p in matched}),
                "source_success_cells": len(source_success),
                "source_success_distinct_roads": len({p["geometry_seed"] for p in source_success}),
                "source_failure_cells": len(matched) - len(source_success),
                "source_finishes": len(source_success),
                "r7b_finishes": sum(p["r7b_finished"] for p in matched),
                "treatment_finishes": sum(p["treatment_finished"] for p in matched),
                "retention_cells": {"K": counts["both_finished"], "L": counts["lost"],
                                    "G": counts["gained"], "neither": counts["both_failed"]},
                "source_to_treatment": {kind: counts[kind] for kind in TRANSITIONS},
                "r7b_to_treatment": {kind: old_counts[kind] for kind in TRANSITIONS},
                "kept_cells": [{"source_seed": p["source_seed"], "geometry_seed": p["geometry_seed"]}
                               for p in matched if p["source_to_treatment"] == "both_finished"],
                "lost_cells": [{"source_seed": p["source_seed"], "geometry_seed": p["geometry_seed"]}
                               for p in matched if p["source_to_treatment"] == "lost"],
                "gained_cells": [{"source_seed": p["source_seed"], "geometry_seed": p["geometry_seed"]}
                                 for p in gained],
                "neither_cells": [{"source_seed": p["source_seed"], "geometry_seed": p["geometry_seed"]}
                                  for p in matched if p["source_to_treatment"] == "both_failed"],
                "distinct_gained_roads": len({p["geometry_seed"] for p in gained}),
                "distinct_gained_roads_unfinished_by_both_sources": len({p["geometry_seed"] for p in gained}
                                                                         & source_never_finished),
                **({"fixed_gate_pass": counts["both_finished"] >= 9 and counts["gained"] >= 2}
                   if seed is None and family is None else {}),
            })
    top = [group for group in groups if group["source_seed"] is None and group["family"] is None]
    _require(all(group["paired_actor_road_cells"] == 32 and group["source_success_cells"] == 11
                 and group["source_failure_cells"] == 21 and group["distinct_reused_road_count"] == 16
                 for group in top), "fixed retention gate denominator differs from 11/21/32")
    summary = {
        "format": DIAGNOSTIC_FORMAT, "role": "reused TRAIN-DIAGNOSTIC", "ranked": False,
        "score_selection": False, "canonical_repeat": 0, "repeats": 2,
        "geometry_count": 16, "candidate_role_count": 6,
        "expected_episode_count": 192, "observed_episode_count": len(episodes),
        "source_success_cells_per_mixture": 11, "source_failure_cells_per_mixture": 21,
        "fixed_gate": "kept >=9/11 AND gained >=2/21 on paired source-actor/road cells",
        "any_treatment_passes_fixed_gate": any(group["fixed_gate_pass"] for group in top),
        "comparison": "same 16 previously reused TRAIN-DIAGNOSTIC roads; old source/r7b traces read only; repeats test deterministic replay, not 192 independent roads",
        "performance_scope": "internal TRAIN-DIAGNOSTIC development proxy; no promotion, confirmation, blind or official HAIC score",
        "paired_source_and_r7b": groups, "paired_json": "paired.json", **lineage,
    }
    return summary, {"format": DIAGNOSTIC_FORMAT, "role": "reused TRAIN-DIAGNOSTIC",
                     "ranked": False, "score_selection": False, "canonical_repeat": 0,
                     "paired_cells": len(pairs), "pairs": pairs, **lineage}


def run_diagnostic(*, root: Path, protocol_path: Path, output_root: Path,
                   preflight_only: bool = False,
                   environment_factory: Callable[[int, int], Any] | None = None,
                   actor_loader: Callable[..., Any] | None = None) -> dict[str, Any]:
    root = root.resolve()
    output = r6._user_repo_file(root, output_root, "runs", "new diagnostic output")
    _require(output == root / RUN_ROOT / "train-diagnostic",
             "output must be the new final-source run root's train-diagnostic directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"TRAIN-DIAGNOSTIC output already exists: {output}")
    _require(output.parent.is_dir() and not output.parent.is_symlink(),
             "new treatment run root must exist and must not be a symlink")
    rows, roles, sources, controls, lineage = _prepare(root, protocol_path,
                                                       actor_loader or load_exported_actor)
    if preflight_only:
        return {"protocol_sha256": lineage["protocol_sha256"], "candidate_role_count": len(roles),
                "verified_old_source_cells": len(sources), "verified_old_r7b_cells": len(controls),
                "expected_episode_count": 192, "environment_resets": 0, "output_created": False}
    factory = environment_factory or r6.build_environment
    output.mkdir(exist_ok=False)
    (output / "traces").mkdir()
    episodes: list[dict[str, Any]] = []
    inventory: dict[str, str] = {}
    for road in rows:
        for name, role in roles.items():
            env = factory(road["geometry_seed"], r6.MAX_STEPS)
            try:
                first = None
                for repeat in (0, 1):
                    episode, arrays = r6._collect_episode(
                        env, role, road, repeat, episode_id=len(episodes) // 2,
                        max_steps=r6.MAX_STEPS, root=root)
                    episode.update({
                        "format": EPISODE_FORMAT, "variant": role["variant"], "condition": CONDITION,
                        "result_path": role["result_path"].relative_to(root).as_posix(),
                        "result_sha256": role["result_sha256"],
                        "source_replay_sha256": role["source_replay_sha256"],
                        "pool_receipt_sha256": role["pool_receipt_sha256"],
                        "protocol_sha256": lineage["protocol_sha256"],
                        "catalog_sha256": r6.CATALOG_SHA256,
                    })
                    relative = Path("traces") / f"seed-{road['geometry_seed']}" / name / f"repeat-{repeat}.npz"
                    trace = output / relative
                    trace.parent.mkdir(parents=True, exist_ok=True)
                    with trace.open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    episode["trace_path"] = relative.as_posix()
                    episode["trace_sha256"] = r6.sha256_file(trace)
                    episode["trace_deterministic_sha256"] = r6._trace_digest(arrays)
                    episode["cell_id"] = f"seed-{road['geometry_seed']}:track-1:{name}:repeat-{repeat}"
                    if repeat == 0:
                        first = episode
                    else:
                        _require(first is not None
                                 and episode["trace_deterministic_sha256"] == first["trace_deterministic_sha256"]
                                 and episode["finished"] is first["finished"],
                                 f"new repeated trace diverged: {episode['cell_id']}")
                        first["repeat_trace_agrees_with_repeat0"] = True
                        episode["repeat_trace_agrees_with_repeat0"] = True
                    inventory[relative.as_posix()] = episode["trace_sha256"]
                    episodes.append(episode)
            finally:
                env.close()
    summary, paired = _paired_summary(episodes, rows, sources, controls, lineage)
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for row in episodes:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    r6._write_json(output / "summary.json", summary)
    r6._write_json(output / "paired.json", paired)
    inventory.update({name: r6.sha256_file(output / name)
                      for name in ("episodes.jsonl", "summary.json", "paired.json")})
    manifest = {
        "format": DIAGNOSTIC_FORMAT, "role": "reused TRAIN-DIAGNOSTIC",
        "ranked": False, "score_selection": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell_inventory": "16 reused roads x six final-source treatment actors x two deterministic repeats",
        "episode_count": len(episodes), "repeat_replay_determinism": "all 96 repeat pairs match",
        "evaluator_source_sha256": r6.sha256_file(Path(__file__)),
        "candidate_receipts": [{field: role[field] for field in (
            "variant", "condition", "source_learner_seed", "source_replay_sha256",
            "pool_receipt_sha256", "actor_sha256", "actor_weights_sha256",
            "checkpoint_sha256", "checkpoint_manifest_sha256", "catalog_sha256",
            "result_sha256", "sample_trace_sha256")}
            for role in roles.values()],
        **lineage, "files_sha256": dict(sorted(inventory.items())),
    }
    r6._write_json(output / "manifest.json", manifest)
    with (output / "manifest.sha256").open("x", encoding="ascii") as stream:
        stream.write(f"{r6.sha256_file(output / 'manifest.json')}  manifest.json\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="frozen new protocol under experiments/")
    parser.add_argument("--preflight-only", action="store_true", help="validate all inputs with zero resets/writes")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        result = run_diagnostic(root=args.repo_root, protocol_path=args.protocol,
                                output_root=args.output_root, preflight_only=args.preflight_only)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        parser.exit(2, f"final-source TRAIN-DIAGNOSTIC stopped: {error}\n")
    print(json.dumps({"output_root": str(args.output_root), **{
        key: result[key] for key in ("protocol_sha256", "expected_episode_count")},
        "preflight_only": args.preflight_only, "ranked": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
