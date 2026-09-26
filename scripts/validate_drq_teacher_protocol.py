"""Fail-closed static and source-ledger checks for the DrQ teacher study."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


class ProtocolError(ValueError):
    pass


_TOP_LEVEL_KEYS = {
    "name",
    "purpose",
    "frame_skip",
    "max_steps",
    "partitions",
    "reserved_training_seeds",
    "schema_version",
    "study_id",
    "run_root",
    "hypothesis",
    "source_revision",
    "working_tree_dirty",
    "working_tree_status_sha256",
    "spec_fingerprints",
    "source_snapshots",
    "source_compatibility",
    "source_actors",
    "training_pools",
    "known_excluded_geometry_seeds",
    "geometry_audit",
    "environment",
    "learner",
    "budgets",
    "runtime",
    "selection",
}
_PARTITION_NAMES = {
    "screen",
    "confirmation",
    "blind",
}
_TRAINING_POOL_NAMES = {"teacher_training", "online_training"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GEOMETRY_SEED_KEYS = ("geometry_seed", "sampled_seed", "reset_seed", "seed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_transition_aliases(source: str, *, label: str) -> tuple[ast.Module, list[str]]:
    module = ast.parse(source, filename=label)
    transition = next(
        (node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Transition"),
        None,
    )
    _require(transition is not None, f"{label} has no Transition class")
    aliases = {"is_first", "is_last", "is_terminal"}
    methods = [node for node in transition.body if isinstance(node, ast.FunctionDef) and node.name in aliases]
    if label.endswith("common_adapter.py source"):
        _require(not methods, "source-era common_adapter unexpectedly defines compatibility aliases")
        return module, []

    found = {method.name: method for method in methods}
    _require(set(found) == aliases, "study common_adapter compatibility aliases changed")
    expected_returns = {
        "is_first": ast.parse("return self.step == 0").body[0],
        "is_last": ast.parse("return self.terminated or self.truncated").body[0],
        "is_terminal": ast.parse("return bool(self.terminal)").body[0],
    }
    for name, method in found.items():
        _require(len(method.decorator_list) == 1
                 and isinstance(method.decorator_list[0], ast.Name)
                 and method.decorator_list[0].id == "property"
                 and len(method.body) == 1
                 and ast.dump(method.body[0], include_attributes=False)
                 == ast.dump(expected_returns[name], include_attributes=False),
                 f"study common_adapter compatibility alias changed behavior: {name}")
    transition.body = [node for node in transition.body if node not in methods]
    return module, sorted(found)


def _adapter_compatibility_record(source_path: Path, study_path: Path) -> dict[str, Any]:
    source_module, source_aliases = _strip_transition_aliases(
        source_path.read_text(encoding="utf-8"), label="common_adapter.py source"
    )
    study_module, study_aliases = _strip_transition_aliases(
        study_path.read_text(encoding="utf-8"), label="common_adapter.py study"
    )
    _require(not source_aliases and study_aliases == ["is_first", "is_last", "is_terminal"],
             "common_adapter source compatibility alias set changed")
    _require(ast.dump(source_module, include_attributes=False)
             == ast.dump(study_module, include_attributes=False),
             "common_adapter contains changes beyond the audited convenience aliases")
    return {
        "path": "common_adapter.py",
        "source_sha256": sha256_file(source_path),
        "study_sha256": sha256_file(study_path),
        "ignored_methods": study_aliases,
        "equivalence": "AST-identical after removing the three exact Transition convenience properties",
    }


def _agent_drq_compatibility_record(source_path: Path, study_path: Path) -> dict[str, Any]:
    source_module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    study_module = ast.parse(study_path.read_text(encoding="utf-8"), filename=str(study_path))

    def class_node(module: ast.Module, name: str) -> ast.ClassDef:
        node = next((item for item in module.body if isinstance(item, ast.ClassDef) and item.name == name), None)
        _require(node is not None, f"agent source is missing DrQ inference class {name}")
        return node

    def method_node(node: ast.ClassDef, name: str) -> ast.FunctionDef:
        method = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == name), None)
        _require(method is not None, f"agent source is missing DrQ inference method {name}")
        return method

    def mode_branch(method: ast.FunctionDef, mode: str) -> ast.If:
        expected = ast.dump(ast.parse(f'self._runtime_mode == "{mode}"').body[0].value,
                             include_attributes=False)
        branch = next((item for item in method.body if isinstance(item, ast.If)
                       and ast.dump(item.test, include_attributes=False) == expected), None)
        _require(branch is not None, f"agent source is missing the {mode} mode branch in {method.name}")
        return branch

    source_agent = class_node(source_module, "Agent")
    study_agent = class_node(study_module, "Agent")
    for class_name in ("DrQFeatures", "DrQActor"):
        _require(ast.dump(class_node(source_module, class_name), include_attributes=False)
                 == ast.dump(class_node(study_module, class_name), include_attributes=False),
                 f"DrQ inference network changed in agent.py: {class_name}")
    _require(ast.dump(method_node(source_agent, "_init_drq"), include_attributes=False)
             == ast.dump(method_node(study_agent, "_init_drq"), include_attributes=False),
             "DrQ export loading changed in agent.py")
    for name in ("reset", "act"):
        _require(ast.dump(mode_branch(method_node(source_agent, name), "drq"), include_attributes=False)
                 == ast.dump(mode_branch(method_node(study_agent, name), "drq"), include_attributes=False),
                 f"DrQ inference {name} behavior changed in agent.py")

    def drq_dispatch(method: ast.FunctionDef) -> tuple[str, str]:
        expected = ast.dump(ast.parse("model_format == DRQ_ACTOR_FORMAT").body[0].value,
                             include_attributes=False)
        branch = next((node for node in ast.walk(method) if isinstance(node, ast.If)
                       and ast.dump(node.test, include_attributes=False) == expected), None)
        _require(branch is not None, "Agent.__init__ no longer dispatches the tagged DrQ export")
        return (
            ast.dump(branch.test, include_attributes=False),
            ast.dump(ast.Module(body=branch.body, type_ignores=[]), include_attributes=False),
        )

    _require(drq_dispatch(method_node(source_agent, "__init__"))
             == drq_dispatch(method_node(study_agent, "__init__")),
             "tagged DrQ Agent dispatch changed in agent.py")
    return {
        "path": "agent.py",
        "source_sha256": sha256_file(source_path),
        "study_sha256": sha256_file(study_path),
        "equivalence": "exact DrQ network, load, dispatch, reset and action AST",
        "checked_units": [
            "DrQFeatures", "DrQActor", "Agent.__init__.DRQ_ACTOR_FORMAT",
            "Agent._init_drq", "Agent.reset.drq", "Agent.act.drq",
        ],
    }


def _evaluation_harness_record(source_path: Path, study_path: Path) -> dict[str, Any]:
    return {
        "path": "evaluate_policy.py",
        "source_sha256": sha256_file(source_path),
        "study_sha256": sha256_file(study_path),
        "scope": "evaluation-only; learner transition generation and optimizer updates do not depend on this module",
        "comparison": "all study source actors and candidates are re-evaluated with this frozen study copy",
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _repo_file(root: Path, relative_path: Any, label: str) -> Path:
    _require(isinstance(relative_path, str) and bool(relative_path), f"{label} path is absent")
    candidate = Path(relative_path)
    _require(not candidate.is_absolute(), f"{label} path must be repository-relative")
    resolved = (root / candidate).resolve()
    _require(resolved.is_relative_to(root.resolve()), f"{label} path escapes the repository")
    _require(resolved.is_file(), f"{label} file does not exist: {relative_path}")
    return resolved


def _geometry_seeds_from_ledger(path: Path) -> set[int]:
    seeds: set[int] = set()
    with path.open("r", encoding="utf-8") as ledger:
        for line_number, line in enumerate(ledger, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolError(f"invalid source episode ledger at {path}:{line_number}") from error
            seed = _read_geometry_seed(record)
            _require(seed is not None, f"no geometry seed in {path}:{line_number}")
            seeds.add(seed)
    _require(bool(seeds), f"source episode ledger has no geometry seeds: {path}")
    return seeds


def _read_geometry_seed(record: Any) -> int | None:
    if not isinstance(record, dict):
        return None
    for key in _GEOMETRY_SEED_KEYS:
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for key in ("sampling", "reset", "environment", "episode"):
        nested = record.get(key)
        if isinstance(nested, dict):
            seed = _read_geometry_seed(nested)
            if seed is not None:
                return seed
    return None


def _cell_set(partition: Any, name: str) -> tuple[set[tuple[int, int]], set[int]]:
    _require(isinstance(partition, dict), f"partition {name} must be an object")
    _require(set(partition) == {"track_ids", "seeds", "repeats"},
             f"partition {name} has unsupported fields")
    track_ids, seeds = partition["track_ids"], partition["seeds"]
    _require(isinstance(track_ids, list) and track_ids
             and all(isinstance(value, int) and not isinstance(value, bool) and value >= 1
                     for value in track_ids), f"partition {name} has invalid track IDs")
    _require(isinstance(seeds, list) and seeds
             and all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**32
                     for value in seeds), f"partition {name} has invalid geometry seeds")
    _require(len(track_ids) == len(set(track_ids)) and len(seeds) == len(set(seeds)),
             f"partition {name} has duplicate track IDs or geometry seeds")
    _require(type(partition["repeats"]) is int and partition["repeats"] >= 2,
             f"partition {name} requires at least two CPU reload repeats")
    exact_cells: set[tuple[int, int]] = set()
    for track_id in track_ids:
        for seed in seeds:
            exact_cells.add((track_id, seed))
    return exact_cells, set(seeds)


def validate_protocol(
    protocol: Any,
    repo_root: Path,
    *,
    audit_live_geometry: bool = False,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Validate immutable source lineage and geometry-level partition exclusions."""
    _require(isinstance(protocol, dict), "protocol must be a JSON object")
    _require(set(protocol) == _TOP_LEVEL_KEYS, "protocol has missing or unsupported top-level fields")
    _require(protocol["schema_version"] == 1, "unsupported protocol schema_version")
    _require(protocol["study_id"] in {
        "drqv2-teacher-replay-v1",
        "drqv2-teacher-replay-v1-r2",
        "drqv2-teacher-replay-v1-r3",
    }, "unexpected study_id")
    _require(isinstance(protocol["run_root"], str)
             and protocol["run_root"].startswith("runs/")
             and not Path(protocol["run_root"]).is_absolute()
             and ".." not in Path(protocol["run_root"]).parts,
             "study run_root must be a safe path under runs/")
    _require(protocol["name"] == protocol["study_id"], "evaluator protocol name must match study_id")
    _require(protocol["purpose"] == "research", "teacher-replay requires a research evaluator protocol")
    _require(protocol["frame_skip"] == 4 and protocol["max_steps"] == 2000,
             "evaluator protocol horizon/frame skip differs from the source contract")
    _require(isinstance(protocol["hypothesis"], str) and protocol["hypothesis"].strip(),
             "hypothesis is absent")
    _require(isinstance(protocol["source_revision"], str) and protocol["source_revision"],
             "source_revision is absent")
    _require(type(protocol["working_tree_dirty"]) is bool
             and isinstance(protocol["working_tree_status_sha256"], str)
             and _SHA256_RE.fullmatch(protocol["working_tree_status_sha256"]) is not None,
             "working tree state must be frozen")
    _require(isinstance(protocol["spec_fingerprints"], dict)
             and set(protocol["spec_fingerprints"]) == {"action", "observation"}
             and all(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
                     for value in protocol["spec_fingerprints"].values()),
             "action and observation fingerprints must be frozen")

    snapshots = protocol["source_snapshots"]
    _require(isinstance(snapshots, dict) and snapshots, "source_snapshots is absent")
    verified_snapshots: dict[str, str] = {}
    for relative_path, expected_hash in snapshots.items():
        _require(isinstance(expected_hash, str) and _SHA256_RE.fullmatch(expected_hash) is not None,
                 f"invalid snapshot SHA-256 for {relative_path}")
        source_path = _repo_file(repo_root, relative_path, "source snapshot")
        actual_hash = sha256_file(source_path)
        _require(actual_hash == expected_hash, f"source snapshot hash mismatch: {relative_path}")
        verified_snapshots[relative_path] = actual_hash

    compatibilities = protocol["source_compatibility"]
    _require(isinstance(compatibilities, list) and len(compatibilities) == 3
             and {record.get("path") for record in compatibilities if isinstance(record, dict)}
             == {"agent.py", "common_adapter.py", "evaluate_policy.py"},
             "source compatibility records must define the audited source/runtime differences")
    compatibility_by_path = {record["path"]: record for record in compatibilities}

    actors = protocol["source_actors"]
    _require(isinstance(actors, list) and len(actors) == 2, "exactly two source actors are required")
    source_training_seeds: set[int] = set()
    actor_seeds: set[int] = set()
    for actor in actors:
        required_actor_keys = {
            "learner_seed",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_manifest_path",
            "checkpoint_manifest_sha256",
            "actor_path",
            "actor_sha256",
            "run_config_path",
            "run_config_sha256",
            "episodes_path",
            "episodes_sha256",
            "source_training_snapshot_dir",
            "source_training_hashes",
            "checkpoint_source_hashes",
            "source_revision",
            "source_dirty",
            "training_steps",
            "training_track_ids",
            "training_geometry_seeds",
            "source_pair_audit",
        }
        _require(isinstance(actor, dict) and set(actor) == required_actor_keys,
                 "source actor has missing or unsupported fields")
        learner_seed = actor["learner_seed"]
        _require(learner_seed in (0, 1) and learner_seed not in actor_seeds,
                 "source learner seeds must be exactly 0 and 1")
        actor_seeds.add(learner_seed)
        for field in (
            "checkpoint_sha256", "checkpoint_manifest_sha256", "actor_sha256",
            "run_config_sha256", "episodes_sha256",
        ):
            _require(isinstance(actor[field], str) and _SHA256_RE.fullmatch(actor[field]) is not None,
                     f"invalid {field} for source learner {learner_seed}")
        for path_field, hash_field in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("checkpoint_manifest_path", "checkpoint_manifest_sha256"),
            ("actor_path", "actor_sha256"),
            ("run_config_path", "run_config_sha256"),
            ("episodes_path", "episodes_sha256"),
        ):
            source_path = _repo_file(repo_root, actor[path_field], f"source {path_field}")
            _require(sha256_file(source_path) == actor[hash_field],
                     f"source {path_field} hash mismatch for learner {learner_seed}")
        ledger_path = _repo_file(repo_root, actor["episodes_path"], "source episodes")
        run_config_path = _repo_file(repo_root, actor["run_config_path"], "source run config")
        checkpoint_manifest_path = _repo_file(
            repo_root, actor["checkpoint_manifest_path"], "source checkpoint manifest"
        )
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        frozen_config = run_config.get("config", {})
        frozen_drq = frozen_config.get("drq_config", {})
        source_dir_rel = Path(actor["source_training_snapshot_dir"])
        _require(not source_dir_rel.is_absolute()
                 and (repo_root / source_dir_rel).resolve().is_relative_to(repo_root.resolve())
                 and (repo_root / source_dir_rel).is_dir(),
                 f"source training snapshot directory is invalid for learner {learner_seed}")
        training_hashes = actor["source_training_hashes"]
        checkpoint_hashes = actor["checkpoint_source_hashes"]
        _require(isinstance(training_hashes, dict) and training_hashes
                 and isinstance(checkpoint_hashes, dict) and checkpoint_hashes,
                 f"source training code hashes are missing for learner {learner_seed}")
        for hash_map_name, hashes in (
            ("training_source_hashes", training_hashes),
            ("checkpoint_source_hashes", checkpoint_hashes),
        ):
            for relative_path, expected_hash in hashes.items():
                _require(isinstance(relative_path, str) and not Path(relative_path).is_absolute()
                         and ".." not in Path(relative_path).parts
                         and isinstance(expected_hash, str) and _SHA256_RE.fullmatch(expected_hash),
                         f"invalid {hash_map_name} entry for learner {learner_seed}")
                source_copy = _repo_file(
                    repo_root, (source_dir_rel / relative_path).as_posix(), "source code snapshot"
                )
                _require(sha256_file(source_copy) == expected_hash,
                         f"source code snapshot hash mismatch for learner {learner_seed}: {relative_path}")
                study_hash = protocol["source_snapshots"].get(relative_path)
                if study_hash != expected_hash:
                    compatibility = compatibility_by_path.get(relative_path)
                    _require(compatibility is not None
                             and compatibility.get("source_sha256") == expected_hash
                             and compatibility.get("study_sha256") == study_hash,
                             f"study code differs from source actor code for learner {learner_seed}: {relative_path}")
        _require(frozen_config.get("training_source_sha256") == training_hashes
                 and manifest.get("extra", {}).get("training_source_sha256") == training_hashes,
                 f"training source hashes disagree across source artifacts for learner {learner_seed}")
        source_dir = (repo_root / source_dir_rel).resolve()
        manifest_source_hashes = {
            (Path(path).resolve().relative_to(source_dir)).as_posix(): value
            for path, value in manifest.get("source_hashes", {}).items()
        }
        _require(manifest_source_hashes == checkpoint_hashes,
                 f"checkpoint manifest source hashes disagree for learner {learner_seed}")
        _require(run_config.get("git", {}).get("commit") == actor["source_revision"],
                 f"source revision disagrees with run config for learner {learner_seed}")
        _require(run_config.get("git", {}).get("dirty") is actor["source_dirty"],
                 f"source dirty status disagrees with run config for learner {learner_seed}")
        _require(type(actor["source_dirty"]) is bool
                 and frozen_config.get("seed") == learner_seed,
                 f"source seed or dirty status is invalid for learner {learner_seed}")
        _require(actor["training_steps"] == 131072
                 and frozen_config.get("total_steps") == 131072
                 and frozen_config.get("algorithm") == "drq-v2",
                 f"source learner {learner_seed} did not train the declared 131072 DrQ steps")
        _require(actor["training_track_ids"] == [1, 2, 3, 4]
                 and frozen_config.get("track_ids") == [1, 2, 3, 4]
                 and frozen_drq.get("augmentation_pad") == 4,
                 f"source learner {learner_seed} does not match the pad-4 track contract")
        _require(manifest.get("algorithm") == "drq-v2"
                 and manifest.get("source_hashes")
                 and manifest.get("extra", {}).get("environment_steps") == 131072,
                 f"source checkpoint manifest is incomplete for learner {learner_seed}")
        source_pair = actor["source_pair_audit"]
        _require(isinstance(source_pair, dict)
                 and source_pair.get("learner_seed") == learner_seed
                 and source_pair.get("checkpoint_sha256") == actor["checkpoint_sha256"]
                 and source_pair.get("actor_sha256") == actor["actor_sha256"]
                 and source_pair.get("actor_weights_sha256")
                 and source_pair.get("action_fingerprint") == protocol["spec_fingerprints"]["action"]
                 and source_pair.get("observation_fingerprint") == protocol["spec_fingerprints"]["observation"]
                 and source_pair.get("cpu_smoke_observations") == 3,
                 f"source full checkpoint and exported actor parity was not audited for learner {learner_seed}")
        agent_parity = source_pair.get("agent_source_parity", {})
        _require(agent_parity == {
            "source_sha256": compatibility_by_path["agent.py"]["source_sha256"],
            "study_sha256": compatibility_by_path["agent.py"]["study_sha256"],
            "observations": 3,
            "actions_exactly_equal": True,
        }, f"source-era and study DrQ CPU Agent action parity was not audited for learner {learner_seed}")
        observed_seeds = _geometry_seeds_from_ledger(ledger_path)
        declared_seeds = actor["training_geometry_seeds"]
        _require(isinstance(declared_seeds, list)
                 and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in declared_seeds),
                 f"invalid declared training geometry seeds for learner {learner_seed}")
        _require(set(declared_seeds) == observed_seeds,
                 f"declared training geometry seeds disagree with source ledger for learner {learner_seed}")
        source_training_seeds |= observed_seeds

    _require(actor_seeds == {0, 1}, "source learner seeds must be 0 and 1")
    source_dir = Path(actors[0]["source_training_snapshot_dir"])
    compatibility_audits = [
        _agent_drq_compatibility_record(
            _repo_file(repo_root, (source_dir / "agent.py").as_posix(), "source Agent"),
            _repo_file(repo_root, "agent.py", "study Agent"),
        ),
        _adapter_compatibility_record(
            _repo_file(repo_root, (source_dir / "common_adapter.py").as_posix(), "source common adapter"),
            _repo_file(repo_root, "common_adapter.py", "study common adapter"),
        ),
        _evaluation_harness_record(
            _repo_file(repo_root, (source_dir / "evaluate_policy.py").as_posix(), "source evaluator"),
            _repo_file(repo_root, "evaluate_policy.py", "study evaluator"),
        ),
    ]
    expected_compatibilities = sorted(compatibility_audits, key=lambda row: row["path"])
    _require(compatibilities == expected_compatibilities
             and all(actor["source_training_hashes"].get("common_adapter.py")
                     == compatibility_by_path["common_adapter.py"]["source_sha256"]
                     and actor["checkpoint_source_hashes"].get("agent.py")
                     == compatibility_by_path["agent.py"]["source_sha256"]
                     and actor["checkpoint_source_hashes"].get("evaluate_policy.py")
                     == compatibility_by_path["evaluate_policy.py"]["source_sha256"]
                     for actor in actors),
             "source-era Agent/common_adapter differences exceed the audited DrQ API compatibility")
    partitions = protocol["partitions"]
    _require(isinstance(partitions, dict) and set(partitions) == _PARTITION_NAMES,
             "partitions must define exactly screen, confirmation and blind")
    parsed_partitions = {name: _cell_set(partitions[name], name) for name in _PARTITION_NAMES}

    pools = protocol["training_pools"]
    _require(isinstance(pools, dict) and set(pools) == _TRAINING_POOL_NAMES,
             "training_pools must define teacher_training and online_training")
    parsed_pools: dict[str, set[int]] = {}
    for name, pool in pools.items():
        _require(isinstance(pool, dict) and set(pool) == {"track_ids", "seeds", "sampler_seed"},
                 f"training pool {name} has missing or unsupported fields")
        track_ids, seeds, sampler_seed = pool["track_ids"], pool["seeds"], pool["sampler_seed"]
        _require(track_ids == [1, 2, 3, 4], f"training pool {name} must use track IDs 1-4")
        _require(isinstance(seeds, list) and seeds
                 and all(isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**32
                         for seed in seeds), f"training pool {name} has invalid geometry seeds")
        _require(len(seeds) == len(set(seeds)), f"training pool {name} has duplicate geometry seeds")
        _require(isinstance(sampler_seed, int) and not isinstance(sampler_seed, bool),
                 f"training pool {name} requires a fixed sampler_seed")
        parsed_pools[name] = set(seeds)

    prior = protocol["known_excluded_geometry_seeds"]
    _require(isinstance(prior, list) and prior
             and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in prior),
             "known_excluded_geometry_seeds must be a non-empty integer list")
    prior_seeds = set(prior)
    _require(len(prior_seeds) == len(prior), "known_excluded_geometry_seeds contains duplicates")

    eval_names = ("screen", "confirmation", "blind")
    eval_geometry: set[int] = set()
    eval_cells: set[tuple[int, int]] = set()
    for name in eval_names:
        cells, geometry = parsed_partitions[name]
        _require(not (geometry & (prior_seeds | source_training_seeds)),
                 f"{name} geometry overlaps prior or source-training geometry")
        _require(not (cells & eval_cells), f"{name} contains a cell reused by another evaluation partition")
        eval_cells |= cells
        eval_geometry |= geometry

    training_geometry: set[int] = set()
    for name in ("teacher_training", "online_training"):
        geometry = parsed_pools[name]
        _require(not (geometry & (eval_geometry | prior_seeds)),
                 f"{name} geometry overlaps evaluation or prior excluded geometry")
        _require(not (geometry & training_geometry),
                 "teacher and online training geometry pools must be disjoint")
        training_geometry |= geometry
    reserved = protocol["reserved_training_seeds"]
    required_reserved = prior_seeds | source_training_seeds | eval_geometry
    _require(isinstance(reserved, list)
             and all(isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**32
                     for seed in reserved), "reserved_training_seeds must be uint32 values")
    _require(len(reserved) == len(set(reserved)) and set(reserved) == required_reserved,
             "reserved_training_seeds must exactly union prior, source-training and evaluation geometry")

    geometry_audit = protocol["geometry_audit"]
    _require(isinstance(geometry_audit, dict)
             and set(geometry_audit) == {
                 "report_path", "report_sha256", "candidate_seeds", "source_snapshot_count",
                 "known_excluded_geometry_seed_count", "global_freshness_claim",
             }, "geometry_audit has missing or unsupported fields")
    audit_path = _repo_file(repo_root, geometry_audit["report_path"], "geometry audit report")
    _require(isinstance(geometry_audit["report_sha256"], str)
             and _SHA256_RE.fullmatch(geometry_audit["report_sha256"]) is not None
             and sha256_file(audit_path) == geometry_audit["report_sha256"],
             "geometry audit report SHA-256 mismatch")
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_candidates = sorted(training_geometry | eval_geometry)
    _require(report.get("passed") is True and not report.get("parse_errors")
             and all(not paths for paths in report.get("candidate_hits", {}).values()),
             "geometry audit did not pass exact recorded-use scan")
    _require(report.get("candidate_seeds") == expected_candidates
             and geometry_audit["candidate_seeds"] == expected_candidates,
             "geometry audit candidate set differs from frozen study partitions")
    _require(report.get("known_excluded_geometry_seeds") == sorted(prior_seeds)
             and geometry_audit["known_excluded_geometry_seed_count"] == len(prior_seeds),
             "known-excluded geometry set differs from the frozen audit report")
    _require(report.get("source_snapshot_count") == geometry_audit["source_snapshot_count"]
             and report["global_freshness_claim"] == "no known recorded overlap; historical pilot schedules are incomplete"
             and geometry_audit["global_freshness_claim"] == report["global_freshness_claim"],
             "geometry audit scope or freshness caveat changed")
    if audit_live_geometry:
        from scripts.audit_drq_teacher_geometry import audit_geometry

        excluded = {
            geometry_audit["report_path"],
            "experiments/drqv2-teacher-replay-v1-geometry-audit.json",
        }
        if protocol_path is not None:
            excluded.add(protocol_path.resolve().relative_to(repo_root.resolve()).as_posix())
        live = audit_geometry(repo_root, expected_candidates, exclude_paths=excluded)
        _require(live["passed"], "live exact-token geometry audit found a prior candidate use or parse gap")
        _require(live["known_excluded_geometry_seeds"] == sorted(prior_seeds),
                 "live recorded geometry union changed since study audit")
    else:
        live = None

    _validate_fixed_contract(protocol)
    return {
        "study_id": protocol["study_id"],
        "source_actor_count": len(actors),
        "source_training_geometry_seed_count": len(source_training_seeds),
        "prior_excluded_geometry_seed_count": len(prior_seeds),
        "partition_cell_counts": {name: len(parsed_partitions[name][0]) for name in sorted(parsed_partitions)},
        "training_pool_geometry_seed_counts": {name: len(parsed_pools[name]) for name in sorted(parsed_pools)},
        "verified_source_snapshot_count": len(verified_snapshots),
        "live_geometry_audit": live,
    }


def _validate_fixed_contract(protocol: dict[str, Any]) -> None:
    _require(protocol["partitions"]["screen"]["track_ids"] == [101, 102, 103]
             and len(protocol["partitions"]["screen"]["seeds"]) == 8,
             "screen shape must be 3 track IDs by 8 geometries")
    _require(protocol["partitions"]["confirmation"]["track_ids"] == [111, 112, 113, 114]
             and len(protocol["partitions"]["confirmation"]["seeds"]) == 8,
             "confirmation shape must be 4 track IDs by 8 geometries")
    _require(protocol["partitions"]["blind"]["track_ids"] == [121, 122, 123]
             and len(protocol["partitions"]["blind"]["seeds"]) == 8,
             "blind shape must be 3 track IDs by 8 geometries")
    _require(protocol["partitions"]["screen"]["repeats"] == 2
             and protocol["partitions"]["confirmation"]["repeats"] == 2
             and protocol["partitions"]["blind"]["repeats"] == 2,
             "all partitions require exactly two CPU reload repeats")
    environment = protocol["environment"]
    _require(environment == {
        "observation_shape": [4, 84, 84],
        "observation_dtype": "float32",
        "observation_range": [0.0, 1.0],
        "frame_skip": 4,
        "action_axes": ["steer", "gas", "brake"],
        "native_action_range": [-1.0, 1.0],
        "raw_reward": True,
    }, "environment contract differs from the frozen teacher-replay plan")
    learner = protocol["learner"]
    _require(learner == {
        "algorithm": "DrQ-v2",
        "padding": 4,
        "feature_dim": 256,
        "hidden_dim": 256,
        "n_step": 3,
        "gamma": 0.99,
        "actor_lr": 0.0001,
        "critic_lr": 0.0001,
        "tau": 0.01,
        "target_update_frequency": 2,
        "actor_update_frequency": 2,
        "target_noise_std": 0.2,
        "target_noise_clip": 0.5,
        "steering_logit_l2": 0.0,
        "reward_shaping": False,
        "reward_normalization": False,
    }, "learner configuration differs from the frozen teacher-replay plan")
    budgets = protocol["budgets"]
    _require(budgets == {
        "teacher_decisions_per_source_cap": 16384,
        "additional_online_decisions_per_arm_source": 32768,
        "online_startup_decisions_without_updates": 10000,
        "critic_updates_per_learning_decision": 1,
        "batch_size": 64,
        "teacher_rows_per_treatment_batch": 16,
        "online_rows_per_treatment_batch": 48,
        "checkpoint_online_steps": [16384, 32768],
        "online_replay_capacity": 100000,
        "teacher_replay_capacity_per_source": 16384,
    }, "budgets differ from the frozen teacher-replay plan")
    runtime = protocol["runtime"]
    runtime_keys = {
        "training_device", "training_executable", "training_python", "dependencies",
        "hardware", "evaluation_executable", "evaluation_runtime", "worker_limits",
    }
    _require(isinstance(runtime, dict) and set(runtime) == runtime_keys
             and runtime.get("training_device") and runtime.get("training_executable")
             and runtime.get("evaluation_executable")
             and isinstance(runtime.get("dependencies"), dict)
             and isinstance(runtime.get("evaluation_runtime"), dict)
             and isinstance(runtime.get("hardware"), dict),
             "runtime dependency versions and train/evaluation runtimes must be pinned")
    _require(all(isinstance(package, str) and isinstance(version, str) and version
                 for package, version in runtime["dependencies"].items()),
             "runtime dependencies must contain explicit versions")
    _require(runtime["evaluation_runtime"].get("torch") == "2.1.0+cpu"
             and runtime["evaluation_runtime"].get("cuda_available") is False,
             "screen/confirmation evaluator must use the pinned CPU Torch 2.1 runtime")
    _require(runtime["worker_limits"] == {
        "actor_load_seconds_max": 10,
        "reset_act_seconds_max": 5,
        "peak_rss_bytes_max": 1073741824,
        "workers": 1,
        "frame_skip": 4,
        "max_steps": 2000,
    }, "CPU evaluator operational limits differ from the teacher-replay plan")
    selection = protocol["selection"]
    _require(selection == {
        "screen_cells": 24,
        "confirmation_cells": 32,
        "blind_cells": 24,
        "repeats_per_cell": 2,
        "canonical_repeat": 0,
        "checkpoint_tie_break": "earlier",
        "checkpoint_candidate_steps": [16384, 32768],
        "screen_sort": "completions-desc,canonical-mean-progress-desc,completed-lap-time-asc",
        "screen_requires_nonzero_teacher_finishes_per_source": True,
        "screen_teacher_finishes_at_least_online_control_per_source": True,
        "confirmation_minimum_gain_vs_online_only_per_source": 2,
        "confirmation_at_least_unchanged_source_per_source": True,
        "confirmation_winning_geometry_minimum": 2,
        "promotion_requires_both_source_actors": True,
        "blind_finalist_learner_zero_breaks_exact_tie": True,
        "blind_opens_only_after_confirmation_pass": True,
        "blind_minimum_canonical_finishes": 1,
    }, "selection/acceptance rules differ from the frozen teacher-replay plan")


def validate_protocol_file(
    protocol_path: Path,
    repo_root: Path,
    *,
    audit_live_geometry: bool = False,
) -> dict[str, Any]:
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read protocol JSON: {protocol_path}") from error
    return validate_protocol(
        protocol,
        repo_root,
        audit_live_geometry=audit_live_geometry,
        protocol_path=protocol_path,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--audit-live-geometry", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = validate_protocol_file(
            args.protocol_file, args.repo_root, audit_live_geometry=args.audit_live_geometry
        )
    except ProtocolError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
