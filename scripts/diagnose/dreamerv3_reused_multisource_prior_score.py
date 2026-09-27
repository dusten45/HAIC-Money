"""Read-only paired H8 vs pure-256 score on already consumed r6 TRAIN development.

Freeze a separate protocol at PROTOCOL_PATH and pass its SHA-256. Exact top-level
schema: format, purpose, study_id, prior_protocol, pure_256_score_protocol,
pure_256_score_result, training, development, source_sha256, scoring, resources,
local_gate. The two training rows contain seed, result_path/result_sha256,
checkpoint_path/checkpoint_sha256, lineage_path/lineage_sha256. Development,
scoring and resources equal the original score protocol; source_sha256 is the
union of its executable map, the H8 training protocol map and this module.
No ZIP is decoded until the code, tests and separate protocol are frozen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamer_v3 import DreamerV3Agent, DreamerV3Config
from scripts.diagnose import dreamerv3_reused_multisource_open_loop as mixed
from scripts.diagnose import dreamerv3_reused_train_open_loop as original


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "haic-dreamerv3-reused-train-multisource-prior-score-v1"
PURPOSE = "reused-TRAIN-consumed-development-prior-interaction-diagnostic"
STUDY_ID = "dreamerv3-reused-train-multisource-prior-score-v1"
PROTOCOL_PATH = "experiments/dreamerv3-reused-train-multisource-prior-score-v1.json"
PRIOR_PATH = "experiments/dreamerv3-reused-train-multisource-prior-v1.json"
PRIOR_SHA256 = "4dad8f8be6ec6f56cc766ade6b3a6f4f8ddb44424c6a6b33e4f2f22293718c80"
PURE_PROTOCOL_PATH = mixed.PROTOCOL_PATH
PURE_PROTOCOL_SHA256 = "dc40240247673ba26e6a722c68ffd7b93c501a2881d8bc01a1a631ed4d3615f1"
PURE_RESULT_PATH = mixed.OUTPUT + "/score-result.json"
PURE_RESULT_SHA256 = "93174115d081a04344b91a31fe42ad4ac2020b69afcc655e921c589013fd2e93"
RUN_ROOT = mixed.TRAIN_ROOT + "/prior-interaction-v1"
OUTPUT = mixed.TRAIN_ROOT + "/prior-interaction-score-v1"
SOURCE = "scripts/diagnose/dreamerv3_reused_multisource_prior_score.py"
RESULT_SHAS = ("f9b07cde32894b55993fd454202e05506c6ea3eb1f1310373502884147816aff",
               "65e5527f3b562ab5d0fe64093641a846fc0b30df56157c12577fac9b99c4b815")
CHECKPOINT_SHAS = ("a67c35b4289be70731b0f16eeb4a9cbeef344cfd1e81d074f1a328b0c4b0c464",
                   "f74bc3d01607a9db10af918fc949d1e790a90f124bed9f38b33f9504026417d7")
LINEAGE_SHA256 = mixed.LINEAGE_SHA256
GATE = {"metric": "image_mse", "learner_seeds": [0, 1],
        "development_action_strata": ["source0", "random"],
        "strictly_below_own_pure_256_on_both_strata": True,
        "strictly_below_frozen_anchor_repeat_on_both_strata": True,
        "min_independent_complete_roads_per_stratum": 4}
PREDICTIONS = frozenset({"image_mse", "reward_mse", "terminal_bce"})
BASELINES = frozenset({"shifted_repeat_mse", "training_constant_reward_mse",
                       "last_logged_reward_mse", "training_prevalence_bce"})
METRICS = PREDICTIONS | BASELINES
STRATA = ("source0", "random")


def _reference_cap(kind: str) -> int:
    return 4 * 1024**2 if kind == "catalog" else 1024**2


def _exact(actual: Any, expected: dict[str, Any], label: str) -> None:
    if (not isinstance(actual, dict) or set(actual) != set(expected)
            or any(type(actual[key]) is not type(value) or actual[key] != value
                   for key, value in expected.items())):
        raise ValueError(f"{label}: frozen identity, metadata or count mismatch")


def _metrics(value: Any, label: str) -> None:
    if (not isinstance(value, dict) or set(value) != METRICS
            or any(type(item) not in (float, int) or not math.isfinite(item)
                   for item in value.values())):
        raise ValueError(f"{label}: missing or nonfinite metrics")


def _pair_metrics(now: Any, before: Any, label: str) -> dict[str, float]:
    _metrics(now, label)
    _metrics(before, label)
    if any(now[key] != before[key] for key in BASELINES):
        raise ValueError(f"{label}: frozen window/reward/terminal baseline differs")
    return {key: original._finite(now[key] - before[key]) for key in sorted(PREDICTIONS)}


def _window(episode: dict[str, Any]) -> None:
    length = episode.get("decisions")
    if type(length) is not int or length < 41 or len(episode.get("windows", [])) != 2:
        raise ValueError("score requires two nonoverlapping eligible 8+32 windows")
    for row, (kind, anchor) in zip(episode["windows"],
                                   (("reset", 8), ("terminal", length - 32)), strict=True):
        _exact({key: row.get(key) for key in ("kind", "context_anchor_decision",
                "reset_origin_prefix_decisions", "first_target_decision", "last_target_decision",
                "label_uses")},
               {"kind": kind, "context_anchor_decision": anchor,
                "reset_origin_prefix_decisions": anchor, "first_target_decision": anchor,
                "last_target_decision": anchor + 31, "label_uses": 32}, "window identity")
        if type(row.get("terminal_positive_label_uses")) is not int or not 0 <= row["terminal_positive_label_uses"] <= 32:
            raise ValueError("window terminal labels invalid")
        _metrics(row.get("metrics"), "window")


def _reference_score(frozen: dict[str, Any], pure: dict[str, Any],
                     train: dict[str, Any], receipts: dict[str, Any]) -> dict[str, float]:
    for key, value in {"format": mixed.FORMAT, "purpose": mixed.PURPOSE,
                       "study_id": mixed.STUDY_ID, "status": "engineering_proxy_only",
                       "score_protocol_sha256": PURE_PROTOCOL_SHA256,
                       "training_protocol_sha256": mixed.TRAIN_SHA256,
                       "development_collection_protocol_sha256": mixed.DEV_SHA256,
                       "source_sha256": pure["source_sha256"], "sampling": pure["scoring"],
                       "fresh_claim": False, "p1b_claim": False,
                       "promotion_eligible": False, "student_actor_trained": False,
                       "distinct_training_roads": 12, "training_episode_attempts": 24}.items():
        if type(frozen.get(key)) is not type(value) or frozen[key] != value:
            raise ValueError(f"pure-256 score {key} differs from frozen reference")
    baseline = frozen.get("training_baselines")
    if (not isinstance(baseline, dict) or set(baseline) != {
            "training_decisions", "training_terminal_events", "constant_reward", "terminal_prevalence"}
            or baseline["training_decisions"] != mixed.TRAIN_DECISIONS
            or type(baseline["training_terminal_events"]) is not int
            or not 0 < baseline["training_terminal_events"] <= mixed.TRAIN_DECISIONS
            or baseline["terminal_prevalence"] != baseline["training_terminal_events"] / mixed.TRAIN_DECISIONS
            or not math.isfinite(baseline["constant_reward"])
            or set(frozen.get("training_source_denominators", {})) != {"source0", "source1"}
            or sum(frozen["training_source_denominators"][s]["training_decisions"] for s in ("source0", "source1"))
            != baseline["training_decisions"]
            or sum(frozen["training_source_denominators"][s]["training_terminal_events"]
                   for s in ("source0", "source1")) != baseline["training_terminal_events"]
            or any(frozen["training_source_denominators"][s]["training_decisions"]
                   != train["datasets"][s]["stored_decisions"] for s in ("source0", "source1"))):
        raise ValueError("pure-256 training baselines or denominator drift")
    if set(frozen.get("strata", {})) != set(STRATA):
        raise ValueError("pure-256 score lacks both development action strata")
    if frozen.get("gate") != {"name": "both-seeds-both-action-strata-strict-image-mse-below-shifted-repeat",
                              "passed": False, "meaning": "non-promoting internal engineering proxy only"}:
        raise ValueError("pure-256 frozen image gate differs")
    for label in STRATA:
        stratum = frozen["strata"][label]
        row = pure["development"][label]
        receipt = receipts[label]
        if (stratum.get("source_id") != row["source_id"]
                or stratum.get("development_archive_sha256") != row["archive_sha256"]
                or any(stratum.get(key) != value for key, value in (
                    ("collected_complete_episodes", 4), ("collected_independent_roads", 4),
                    ("collected_finished_episodes", len(receipt["distinct_finished_geometries"])),
                    ("collected_terminal_episodes", sum(ep["terminal"] for ep in receipt["episode_rows"]))))
                or not isinstance(stratum.get("models"), list) or len(stratum["models"]) != 2):
            raise ValueError(f"{label}: pure-256 source or independent road counts differ")
        anchors = None
        for seed, model in enumerate(stratum["models"]):
            if (type(model.get("learner_seed")) is not int or model["learner_seed"] != seed
                    or model.get("checkpoint_sha256") != pure["training"][seed]["checkpoint_sha256"]
                    or not isinstance(model.get("episodes"), list) or len(model["episodes"]) != 4
                    or not isinstance(model.get("roads"), list) or len(model["roads"]) != 4):
                raise ValueError(f"{label}/{seed}: pure-256 model/episode count mismatch")
            roads = defaultdict(list)
            for index, episode in enumerate(model["episodes"]):
                receipt_ep = receipt["episode_rows"][index]
                for key, value in {"episode_id": str(receipt_ep["episode_id"]),
                                   "track_id": receipt_ep["track_id"],
                                   "geometry_seed": receipt_ep["geometry_seed"],
                                   "decisions": receipt_ep["decisions"],
                                   "terminal_event": receipt_ep["terminal"],
                                   "finished": receipt_ep["finished"]}.items():
                    if type(episode.get(key)) is not type(value) or episode[key] != value:
                        raise ValueError("pure-256 episode identity differs from development receipt")
                _window(episode)
                _metrics(episode.get("metrics"), "pure-256 episode")
                if (episode["metrics"] != {key: original._mean([w["metrics"][key] for w in episode["windows"]])
                                           for key in METRICS}
                        or episode.get("unique_scored_decisions") != 64
                        or episode.get("duplicate_label_uses") != 0
                        or episode.get("unique_terminal_positive_labels") != 1
                        or [w["terminal_positive_label_uses"] for w in episode["windows"]] != [0, 1]):
                    raise ValueError("pure-256 episode/window baseline or label count drift")
                roads[(episode["track_id"], episode["geometry_seed"])].append(episode)
            expected_roads = [{"track_id": track, "geometry_seed": road, **original._aggregate(items)}
                              for (track, road), items in sorted(roads.items())]
            if (len(roads) != 4 or model["roads"] != expected_roads
                    or model.get("aggregate") != original._aggregate(model["episodes"])):
                raise ValueError("pure-256 road, episode or aggregate denominator mismatch")
            metrics = model["aggregate"]["metrics_episode_mean"]
            _metrics(metrics, "pure-256 aggregate")
            reference_baselines = {key: metrics[key] for key in BASELINES}
            if anchors is None:
                anchors = reference_baselines
            elif reference_baselines != anchors:
                raise ValueError("pure-256 baseline changed across model seeds")
    return baseline


def _step_stream(root: Path, prefix: str, result: dict[str, Any], kind: str,
                 count: int, pins: dict[str, tuple[str, int]]) -> None:
    name = f"{kind}-update-metrics.jsonl"
    if result.get(f"{kind}_metrics_path") != name:
        raise ValueError(f"{kind}: unexpected optimizer metrics path")
    path_name = f"{prefix}/{name}"
    digest = result.get(f"{kind}_metrics_sha256")
    path = mixed._pin(root, path_name, digest, max_bytes=16 * 1024**2)
    pins[path_name] = (digest, 16 * 1024**2)
    with path.open(encoding="utf-8") as stream:
        rows = 0
        for rows, line in enumerate(stream, 1):
            if rows > count or len(line) > 4096:
                raise ValueError(f"{kind}: optimizer metrics stream exceeds count/row cap")
            item = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            required = {f"{kind}_update", "metrics"} | ({"after_base_update"} if kind == "aux" else set())
            if (not isinstance(item, dict) or set(item) != required
                    or type(item.get(f"{kind}_update")) is not int or item[f"{kind}_update"] != rows
                    or (kind == "aux" and item["after_base_update"] != rows * 4)
                    or not isinstance(item["metrics"], dict) or not item["metrics"]
                    or ("loss_wm" not in item["metrics"] if kind == "base" else
                        not {"aux_loss", "frame_mse", "anchor_count", "target_count", "effective_horizon"}
                        <= item["metrics"].keys())
                    or any(not isinstance(key, str) or type(value) not in (float, int)
                           or not math.isfinite(value) for key, value in item["metrics"].items())
                    or (kind == "aux" and (item["metrics"]["effective_horizon"] != 8
                           or item["metrics"]["target_count"] != 64))):
                raise ValueError(f"{kind}: invalid optimizer metrics")
        if rows != count:
            raise ValueError(f"{kind}: wrong optimizer metrics count")


def _checkpoint(path: Path, reference: Path, *, metadata: dict[str, Any],
                baseline_metadata: dict[str, Any], config: dict[str, Any],
                lineage: list[dict[str, Any]], seed: int) -> tuple[Any, ...]:
    # Only fixed, SHA-checked, size-capped files reach the pickle loader.
    baseline = torch.load(reference, map_location="cpu", weights_only=False)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(payload, dict) or payload.get("format") != "haic-dreamerv3-checkpoint-v3"
            or payload.get("run_metadata") != metadata or payload.get("config") != config
            or type(payload.get("gradient_steps")) is not int or payload["gradient_steps"] != 256
            or type(payload.get("environment_steps")) is not int or payload["environment_steps"] != 0
            or payload.get("trainer_state") is not None
            or baseline.get("format") != "haic-dreamerv3-checkpoint-v3"
            or baseline.get("config") != config or baseline.get("gradient_steps") != 256
            or baseline.get("environment_steps") != 0):
        raise ValueError(f"learner-{seed}: checkpoint metadata/config/model-only counters mismatch")
    if baseline.get("run_metadata") != baseline_metadata:
        raise ValueError(f"learner-{seed}: pure-256 checkpoint provenance mismatch")
    count = sum(row["sequence_id_end"] - row["sequence_id_start"] + 1 for row in lineage)
    for candidate in (baseline, payload):
        replay = candidate.get("replay")
        if (not isinstance(replay, dict) or any(replay.get(key) != value for key, value in (
                ("capacity", config["replay_capacity"]), ("size", count), ("cursor", count),
                ("total_steps", count)))
                or not np.array_equal(replay.get("sequence_ids"), np.arange(count))
                or not np.array_equal(np.flatnonzero(replay.get("is_first")),
                                      [row["sequence_id_start"] for row in lineage])
                or not np.array_equal(np.flatnonzero(replay.get("is_last")),
                                      [row["sequence_id_end"] for row in lineage])
                or set(replay.get("boundary_observations", {})) != {
                    row["sequence_id_end"] for row in lineage}):
            raise ValueError(f"learner-{seed}: checkpoint replay lineage lost or overwritten")
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            reference_agent = DreamerV3Agent(DreamerV3Config(**config), seed=seed)
            for name in ("actor", "critic", "critic_target", "actor_optimizer", "critic_optimizer"):
                expected = getattr(reference_agent, name).state_dict()
                if (not mixed._equal(baseline.get(name), expected)
                        or not mixed._equal(payload.get(name), expected)):
                    raise ValueError(f"learner-{seed}: seeded actor/critic/target or optimizer changed: {name}")
    finally:
        np.random.set_state(numpy_state)
    del baseline, reference_agent
    modules = (original.ConvEncoder(4, config["embed_dim"]),
               original.CategoricalRSSM(action_dim=3, embed_dim=config["embed_dim"],
                   hidden_dim=config["hidden_dim"], num_categoricals=config["num_categoricals"],
                   num_classes=config["num_classes"], unimix=config["unimix"]))
    encoder, rssm = modules
    modules += (original.ConvDecoder(rssm.state_dim, 1),
                original.RewardHead(rssm.state_dim, config["twohot_bins"]),
                original.ContinueHead(rssm.state_dim))
    for name, module in zip(("encoder", "rssm", "decoder", "reward_head", "continue_head"), modules, strict=True):
        module.load_state_dict(payload[name], strict=True)
        module.eval()
    return modules


def _pair(current: dict[str, Any], frozen: dict[str, Any]) -> None:
    if (current["learner_seed"] != frozen["learner_seed"]
            or current["pure_256_checkpoint_sha256"] != frozen["checkpoint_sha256"]):
        raise ValueError("paired learner seed/checkpoint identity differs")
    for section, key in (("aggregate", "metrics_episode_mean"), ("roads", "metrics_episode_mean"),
                         ("episodes", "metrics")):
        now_rows = [current[section]] if section == "aggregate" else current[section]
        old_rows = [frozen[section]] if section == "aggregate" else frozen[section]
        if len(now_rows) != len(old_rows):
            raise ValueError(f"{section}: paired row count differs")
        for now, old in zip(now_rows, old_rows, strict=True):
            if ({field: value for field, value in now.items() if field not in (key, "windows")}
                    != {field: value for field, value in old.items() if field not in (key, "windows")}):
                raise ValueError(f"{section}: paired road/episode/label denominator differs")
            now["paired_delta_vs_pure_256"] = _pair_metrics(now[key], old[key], section)
            now["pure_256_metrics"] = old[key]
            if section == "episodes":
                _window(now)
                for window, reference in zip(now["windows"], old["windows"], strict=True):
                    if ({field: value for field, value in window.items() if field != "metrics"}
                            != {field: value for field, value in reference.items() if field != "metrics"}):
                        raise ValueError("paired reset/terminal window or label identity differs")
                    window["paired_delta_vs_pure_256"] = _pair_metrics(
                        window["metrics"], reference["metrics"], "window")
                    window["pure_256_metrics"] = reference["metrics"]


def _gate(strata: dict[str, Any]) -> dict[str, Any]:
    seeds = []
    for seed in (0, 1):
        comparisons = {}
        for label in STRATA:
            model = strata[label]["models"][seed]
            aggregate = model["aggregate"]
            if (aggregate["episode_count"] != 4 or aggregate["window_count"] != 8
                    or aggregate["window_label_uses"] != 256
                    or strata[label]["collected_independent_roads"] != 4):
                raise ValueError("gate requires four independently scored complete roads in both strata")
            now = original._finite(aggregate["metrics_episode_mean"]["image_mse"])
            before = original._finite(aggregate["pure_256_metrics"]["image_mse"])
            anchor = original._finite(aggregate["pure_256_metrics"]["shifted_repeat_mse"])
            comparisons[label] = {"prior_image_mse": now, "pure_256_image_mse": before,
                                  "frozen_anchor_repeat_mse": anchor,
                                  "paired_delta_vs_pure_256": original._finite(now - before),
                                  "strict_improvement": now < before,
                                  "strictly_below_anchor": now < anchor}
        seeds.append({"learner_seed": seed, "development_action_strata": comparisons,
                      "passed": all(row["strict_improvement"] and row["strictly_below_anchor"]
                                    for row in comparisons.values())})
    return {"rule": GATE, "seeds": seeds, "passed": all(row["passed"] for row in seeds),
            "interpretation": "Descriptive stop rule on consumed r6 TRAIN development, not fresh P1b, "
                              "actor benefit, official performance or a loss-shape cause (320 vs 256 steps)."}


def preflight(protocol_path: Path, protocol_sha256: str, output_dir: Path,
              *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Pin all identities and audit both checkpoints without decoding a ZIP."""
    root = Path(repo_root).resolve()
    if original._relative(root, protocol_path) != PROTOCOL_PATH:
        raise ValueError("separate, fixed-name H8 scoring protocol required")
    pins: dict[str, tuple[str, int]] = {}

    def pin(name: str, digest: str, cap: int = 1024**2) -> Path:
        path = mixed._pin(root, name, digest, max_bytes=cap)
        pins[name] = (digest, cap)
        return path

    protocol = original._json(pin(PROTOCOL_PATH, protocol_sha256))
    if (set(protocol) != {"format", "purpose", "study_id", "prior_protocol", "pure_256_score_protocol",
                          "pure_256_score_result", "training", "development", "source_sha256",
                          "scoring", "resources", "local_gate"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or protocol["study_id"] != STUDY_ID):
        raise ValueError("invalid isolated H8 scoring protocol schema")
    _exact(protocol["local_gate"], GATE, "predeclared descriptive-only gate")
    if original._relative(root, output_dir) != OUTPUT:
        raise ValueError("isolated score output directory is fixed")
    output = root / OUTPUT
    if (not output.parent.is_dir() or any(path.is_symlink() for path in
            (root / "runs", root / mixed.TRAIN_ROOT, output.parent))):
        raise ValueError("score output parent must exist without symlinks")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"score output collision: {output}")

    def reference(field: str, name: str, digest: str) -> dict[str, Any]:
        _exact(protocol[field], {"path": name, "sha256": digest}, field)
        return original._json(pin(name, digest))

    prior = reference("prior_protocol", PRIOR_PATH, PRIOR_SHA256)
    pure = reference("pure_256_score_protocol", PURE_PROTOCOL_PATH, PURE_PROTOCOL_SHA256)
    frozen = reference("pure_256_score_result", PURE_RESULT_PATH, PURE_RESULT_SHA256)
    train = original._json(pin(mixed.TRAIN_PATH, mixed.TRAIN_SHA256))
    dev = original._json(pin(mixed.DEV_PATH, mixed.DEV_SHA256))
    if (prior.get("format") != "haic-dreamerv3-reused-train-multisource-prior-v1"
            or prior.get("purpose") != "reused-TRAIN-prior-image-interaction-diagnostic"
            or prior.get("study_id") != "dreamerv3-reused-train-multisource-prior-v1"
            or prior.get("mixed_protocol") != pure["training_protocol"]
            or prior["mixed_protocol"] != {"path": mixed.TRAIN_PATH, "sha256": mixed.TRAIN_SHA256}
            or pure["development_collection_protocol"] != {"path": mixed.DEV_PATH, "sha256": mixed.DEV_SHA256}
            or prior.get("cells") != train.get("cells")
            or prior.get("datasets") != train.get("datasets")
            or prior.get("baseline_results") != [{key: row[key] for key in (
                "seed", "result_path", "result_sha256", "lineage_path", "lineage_sha256")}
                for row in pure["training"]]
            or prior.get("expected_lineage_sha256") != LINEAGE_SHA256
            or prior.get("output_root") != RUN_ROOT
            or prior.get("learner") != {"seeds": [0, 1], "config": train["learner"]["config"],
                    "base_updates": 256, "aux_updates": 64, "aux_every_base_updates": 4,
                    "prior_horizon": 8, "prior_weight": 0.25,
                    "anchor_mode": "deterministic-first-last-full-horizon",
                    "aux_rng_mode": "fork-torch-numpy-python", "aux_rng_seed_base": 260926000,
                    "aux_rng_seed_stride": 1000}
            or prior.get("resources") != {**train["resources"], "min_cgroup_available_bytes": 12 * 1024**3}
            or train.get("learner", {}).get("seeds") != [0, 1] or train["learner"].get("updates") != 256
            or train.get("cells") != [{"track_id": 1, "geometry_seed": road} for road in mixed.ROADS]
            or dev.get("cells") != [{"track_id": 1, "geometry_seed": road} for road in mixed.DEV_ROADS]
            or dev.get("r6_protocol") != train.get("r6_protocol")
            or set(mixed.ROADS) & set(mixed.DEV_ROADS)
            or sum(row["stored_decisions"] for row in train["datasets"].values()) != mixed.TRAIN_DECISIONS):
        raise ValueError("H8 source/actor/archive/lineage/budget or original training identity differs")
    config = train["learner"]["config"]
    if (not isinstance(config, dict) or set(config) != {field.name for field in fields(DreamerV3Config)}
            or config.get("device") != "cpu" or config.get("replay_capacity") != mixed.REPLAY_CAPACITY):
        raise ValueError("original CPU model configuration differs")
    for field, value in (("development", pure["development"]), ("scoring", pure["scoring"]),
                         ("resources", pure["resources"])):
        _exact(protocol[field], value, field)
    if (pure["scoring"] != {"context_decisions": 8, "window_decisions": 32,
            "latent_seed": 120927, "context_mode": "reset-origin-full-prefix-terminal",
            "baselines": original.BASELINES, "min_complete_roads_per_stratum": 4}):
        raise ValueError("original 8+32 score/window contract differs")
    limits = {"max_archive_bytes": 64 * 1024**2, "max_uncompressed_bytes": 512 * 1024**2,
              "max_decisions_per_archive": 32768, "max_checkpoint_bytes": 1024**3,
              "max_cgroup_memory_bytes": 128 * 1024**3,
              "min_cgroup_available_bytes": 128 * 1024**3, "min_disk_available_bytes": 1024**4}
    if set(pure["resources"]) != set(limits):
        raise ValueError("original ZIP/checkpoint/cgroup/disk cap schema differs")
    for key, ceiling in limits.items():
        original._positive(pure["resources"][key], key, ceiling)
    resources = pure["resources"]
    if (resources["min_cgroup_available_bytes"] < 8 * 1024**3
            or resources["min_disk_available_bytes"] < 2 * 1024**3):
        raise ValueError("frozen score cgroup/disk floor is below safe minimum")
    needed = max(resources["min_cgroup_available_bytes"], 512 * 1024**2
                 + 2 * resources["max_checkpoint_bytes"] + 2 * resources["max_archive_bytes"]
                 + 2 * resources["max_uncompressed_bytes"])
    cgroup = original._cgroup()
    if cgroup["limit_bytes"] > resources["max_cgroup_memory_bytes"] or cgroup["available_bytes"] < needed:
        raise ValueError("cgroup lacks frozen checkpoint/ZIP scoring headroom")
    if os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize < resources["min_disk_available_bytes"]:
        raise ValueError("disk lacks frozen score headroom")
    expected_sources = {**pure["source_sha256"], **prior["source_sha256"]}
    if any(pure["source_sha256"][key] != prior["source_sha256"][key]
           for key in pure["source_sha256"].keys() & prior["source_sha256"].keys()):
        raise ValueError("old and H8 source SHA maps disagree")
    if (not isinstance(protocol["source_sha256"], dict)
            or set(protocol["source_sha256"]) != set(expected_sources) | {SOURCE}
            or any(protocol["source_sha256"].get(name) != digest for name, digest in expected_sources.items())):
        raise ValueError("complete old/new executable source SHA map required")
    for name, digest in protocol["source_sha256"].items():
        pin(name, digest, 4 * 1024**2)
    for name, path in {**mixed.RUNTIME_SOURCES, SOURCE: Path(__file__)}.items():
        if original._sha256(path) != protocol["source_sha256"][name]:
            raise ValueError(f"executing scorer source SHA mismatch: {name}")
    for name, digest in train["source_sha256"].items():
        pin(name, digest, 4 * 1024**2)
    for name in ("common_adapter.py", "haic/algorithms/drq_v2/teacher_replay.py"):
        if dev["source_sha256"].get(name) != protocol["source_sha256"][name]:
            raise ValueError(f"development decoder source SHA differs: {name}")
    pin("scripts/diagnose/collect_dreamerv3_reused_train.py",
        dev["source_sha256"]["scripts/diagnose/collect_dreamerv3_reused_train.py"], 4 * 1024**2)
    for name, ref in ((mixed.TRAIN_PATH, pure["training_protocol"]),
                      (mixed.DEV_PATH, pure["development_collection_protocol"])):
        _exact(ref, {"path": name, "sha256": mixed.TRAIN_SHA256 if name == mixed.TRAIN_PATH else mixed.DEV_SHA256},
               "original protocol reference")
    for name, ref in train.items():
        if name in ("r6_protocol", "source0_protocol", "source1_protocol", "base_model_protocol",
                     "catalog", "historical_train_summary", "source1_root_receipt"):
            if prior.get(name) != ref:
                raise ValueError(f"H8 reference differs from original: {name}")
            pin(ref["path"], ref["sha256"], _reference_cap(name))

    train_receipts = {}
    for label in ("source0", "source1"):
        row = train["datasets"][label]
        receipt = original._json(pin(row["receipt_path"], row["receipt_sha256"]))
        if (receipt.get("status") != "completed" or receipt.get("arm") != "teacher"
                or receipt.get("protocol_sha256") != train[f"{label}_protocol"]["sha256"]
                or receipt.get("r6_protocol_sha256") != train["r6_protocol"]["sha256"]
                or receipt.get("source_checkpoint_sha256") != row["source_checkpoint_sha256"]
                or receipt.get("schedule_exhausted") is not True
                or receipt.get("partial_decisions") != 0
                or receipt.get("unresolved_decision_calls") != 0
                or receipt.get("source_id") != row["source_id"]
                or receipt.get("source_actor_sha256") != row["source_actor_sha256"]
                or receipt.get("archive_sha256") != row["archive_sha256"]
                or receipt.get("dataset_digest") != row["dataset_digest"]
                or receipt.get("stored_decisions") != row["stored_decisions"]
                or receipt.get("complete_episode_count") != 12
                or receipt.get("allowed_cells") != train["cells"]
                or not isinstance(receipt.get("episode_rows"), list) or len(receipt["episode_rows"]) != 12
                or sum(item["decisions"] for item in receipt["episode_rows"]) != row["stored_decisions"]):
            raise ValueError(f"{label}: training source/actor/archive receipt differs")
        pin(row["archive_path"], row["archive_sha256"], resources["max_archive_bytes"])
        train_receipts[label] = receipt
    lineage = mixed._lineage(root, train, pure["training"], train_receipts)
    for row in pure["training"]:
        pin(row["lineage_path"], LINEAGE_SHA256)
        pin(row["result_path"], row["result_sha256"])
        pin(row["checkpoint_path"], row["checkpoint_sha256"], resources["max_checkpoint_bytes"])
    checked_dev = {}
    for label, arm in (("source0", "teacher"), ("random", "random")):
        row = pure["development"][label]
        source = dev["source_actor"] if label == "source0" else dev["random"]
        actor = source["actor_sha256"] if label == "source0" else source["source_actor_sha256"]
        if (row["source_id"] != source["source_id"] or row["source_actor_sha256"] != actor
                or row["receipt_path"] != f"{mixed.DEV_ROOT}/{arm}/collection-result.json"
                or row["archive_path"] != f"{mixed.DEV_ROOT}/{arm}/support-dataset.npz"):
            raise ValueError(f"{label}: development path or source actor differs")
        receipt = original._json(pin(row["receipt_path"], row["receipt_sha256"]))
        if (receipt.get("status") != "completed" or receipt.get("protocol_sha256") != mixed.DEV_SHA256
                or receipt.get("allowed_cells") != dev["cells"] or receipt.get("arm") != arm
                or receipt.get("study_id") != dev["study_id"]
                or receipt.get("source_id") != row["source_id"]
                or receipt.get("source_actor_sha256") != actor
                or receipt.get("source_checkpoint_sha256") != (source["checkpoint_sha256"] if label == "source0" else None)
                or receipt.get("random_rng_seed") != (7395 if label == "random" else None)
                or receipt.get("archive_sha256") != row["archive_sha256"]
                or receipt.get("dataset_digest") != row["dataset_digest"]
                or receipt.get("complete_episode_count") != 4 or receipt.get("schedule_attempts") != 4
                or receipt.get("schedule_exhausted") is not True or receipt.get("partial_decisions") != 0
                or receipt.get("unresolved_decision_calls") != 0 or not isinstance(receipt.get("episode_rows"), list)
                or len(receipt["episode_rows"]) != 4 or any(
                    ep.get("episode_id") != index or ep.get("attempt") != index
                    or ep.get("status") != "complete" or ep.get("complete") is not True
                    or type(ep.get("terminal")) is not bool or ep["terminal"] is not True
                    or type(ep.get("finished")) is not bool
                    or ep.get("decisions", 0) < 41
                    or (ep.get("track_id"), ep.get("geometry_seed")) != (1, mixed.DEV_ROADS[index])
                    for index, ep in enumerate(receipt["episode_rows"]))
                or receipt.get("stored_decisions") != sum(ep["decisions"] for ep in receipt["episode_rows"])
                or receipt.get("decisions_spent") != receipt["stored_decisions"]
                or receipt.get("distinct_finished_geometries") != sorted(
                    ep["geometry_seed"] for ep in receipt["episode_rows"] if ep["finished"])):
            raise ValueError(f"{label}: development receipt/road/label provenance differs")
        archive = pin(row["archive_path"], row["archive_sha256"], resources["max_archive_bytes"])
        checked_dev[label] = {"row": row, "receipt": receipt, "archive": archive, "arm": arm}
    baseline = _reference_score(frozen, pure, train,
                                {label: data["receipt"] for label, data in checked_dev.items()})
    training = []
    if not isinstance(protocol["training"], list) or len(protocol["training"]) != 2:
        raise ValueError("exactly two H8 training seed pairs required")
    for seed, row in enumerate(protocol["training"]):
        prefix = f"{RUN_ROOT}/learner-{seed}"
        _exact(row, {"seed": seed, "result_path": prefix + "/training-result.json",
                     "result_sha256": RESULT_SHAS[seed], "checkpoint_path": prefix + "/world-model-checkpoint.pt",
                     "checkpoint_sha256": CHECKPOINT_SHAS[seed], "lineage_path": prefix + "/lineage.json",
                     "lineage_sha256": LINEAGE_SHA256}, f"learner-{seed} fixed SHA pair")
        result = original._json(pin(row["result_path"], row["result_sha256"]))
        pin(row["lineage_path"], LINEAGE_SHA256)
        if original._json(root / row["lineage_path"])["episodes"] != lineage:
            raise ValueError(f"learner-{seed}: original lineage ranges/bytes differ")
        checkpoint = pin(row["checkpoint_path"], row["checkpoint_sha256"], resources["max_checkpoint_bytes"])
        expected = {"format": "haic-dreamerv3-reused-train-multisource-prior-v1-result",
                    "status": "complete", "purpose": prior["purpose"], "study_id": prior["study_id"],
                    "seed": seed, "protocol_sha256": PRIOR_SHA256, "mixed_protocol": prior["mixed_protocol"],
                    "pure_256_baselines": prior["baseline_results"], "source_sha256": prior["source_sha256"],
                    "sources": train["datasets"], "checkpoint_path": "world-model-checkpoint.pt",
                    "checkpoint_sha256": row["checkpoint_sha256"], "lineage_path": row["lineage_path"],
                    "lineage_sha256": LINEAGE_SHA256, "base_model_only_updates": 256,
                    "aux_world_model_optimizer_steps": 64, "world_model_optimizer_steps": 320,
                    "aux_after_every_base_updates": 4, "prior_horizon": 8, "prior_weight": 0.25,
                    "anchor_mode": prior["learner"]["anchor_mode"],
                    "aux_rng_mode": prior["learner"]["aux_rng_mode"],
                    "aux_rng_seed_base": prior["learner"]["aux_rng_seed_base"],
                    "aux_rng_seed_stride": prior["learner"]["aux_rng_seed_stride"],
                    "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                    "promotion_eligible": False, "matched_pure_256_step_model": False,
                    "actor_critic_target_optimizers_unchanged": True,
                    "environment_steps": 0, "distinct_training_roads": 12, "episodes": 24,
                    "decisions": mixed.TRAIN_DECISIONS}
        for key, value in expected.items():
            if type(result.get(key)) is not type(value) or result[key] != value:
                raise ValueError(f"learner-{seed}: H8 training result {key} differs")
        pure_result = original._json(root / pure["training"][seed]["result_path"])
        if (not isinstance(result.get("source_audits"), dict)
                or set(result["source_audits"]) != {"source0", "source1"}
                or any(result["source_audits"][label].get("source_actor_sha256")
                       != train["datasets"][label]["source_actor_sha256"]
                       or result["source_audits"][label].get("archive_sha256")
                       != train["datasets"][label]["archive_sha256"]
                       or result["source_audits"][label].get("dataset_digest")
                       != train["datasets"][label]["dataset_digest"] for label in ("source0", "source1"))
                or result["source_audits"] != pure_result.get("source_audits")
                or not isinstance(result.get("union_finished_roads"), list)
                or result["union_finished_roads"] != pure_result.get("union_finished_roads")
                or pure_result.get("sources") != train["datasets"]
                or pure_result.get("model_only_updates") != 256
                or pure_result.get("actor_critic_target_optimizers_unchanged") is not True
                or pure_result.get("actor_trained") is not False):
            raise ValueError(f"learner-{seed}: source audit or pure-256 receipt differs")
        for kind, count in (("base", 256), ("aux", 64)):
            _step_stream(root, prefix, result, kind, count, pins)
        metadata = {key: result[key] for key in ("purpose", "study_id", "seed", "protocol_sha256",
                   "mixed_protocol", "pure_256_baselines", "source_sha256", "sources", "lineage_path",
                   "lineage_sha256", "base_metrics_sha256", "aux_metrics_sha256",
                   "base_model_only_updates", "aux_world_model_optimizer_steps", "world_model_optimizer_steps",
                   "aux_after_every_base_updates", "prior_horizon", "prior_weight", "anchor_mode",
                   "aux_rng_mode", "aux_rng_seed_base", "aux_rng_seed_stride", "actor_trained",
                   "fresh_claim", "p1b_claim", "promotion_eligible", "matched_pure_256_step_model")}
        baseline_metadata = {"purpose": train["purpose"], "study_id": train["study_id"],
                             "seed": seed, "protocol_sha256": mixed.TRAIN_SHA256,
                             "lineage_path": pure["training"][seed]["lineage_path"],
                             "lineage_sha256": LINEAGE_SHA256, "sources": train["datasets"],
                             "actor_trained": False, "fresh_claim": False,
                             "p1b_claim": False, "promotion_eligible": False}
        modules = _checkpoint(checkpoint, root / pure["training"][seed]["checkpoint_path"],
                              metadata=metadata, baseline_metadata=baseline_metadata,
                              config=config, lineage=lineage, seed=seed)
        training.append({"row": row, "result": result, "modules": modules})
    return {"root": root, "output": output, "protocol": protocol, "frozen": frozen,
            "prior": prior, "pure": pure, "train": train, "dev": dev, "baseline": baseline,
            "training": training, "development": checked_dev, "pins": pins,
            "required_available_bytes": needed}


def score(protocol_path: Path, protocol_sha256: str, output_dir: Path,
          *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, output_dir, repo_root=repo_root)
    resources = checked["protocol"]["resources"]
    if original._cgroup()["available_bytes"] < checked["required_available_bytes"]:
        raise ValueError("cgroup lacks development ZIP load headroom")
    torch.set_num_threads(1)
    datasets = {}
    for label in STRATA:
        entry = checked["development"][label]
        datasets[label] = mixed._dataset(entry["archive"], entry["row"], resources,
                                         mixed.DEV_ROADS, checked["dev"]["study_id"],
                                         entry["arm"], entry["receipt"])
        for episode, reference in zip(datasets[label].episodes,
                                      checked["frozen"]["strata"][label]["models"][0]["episodes"], strict=True):
            if (episode.episode_id != reference["episode_id"]
                    or episode.track_id != reference["track_id"]
                    or int(episode.geometry_id) != reference["geometry_seed"]
                    or episode.steps != reference["decisions"]
                    or bool(episode.terminal[-1]) != reference["terminal_event"]
                    or bool(episode.finished[-1]) != reference["finished"]):
                raise ValueError("development ZIP episode/road differs from original pure-256 score")
    report = {"format": FORMAT, "purpose": PURPOSE, "study_id": STUDY_ID,
              "status": "iterative_tuning_descriptive_only", "score_protocol_sha256": protocol_sha256,
              "prior_protocol_sha256": PRIOR_SHA256, "pure_256_score_protocol_sha256": PURE_PROTOCOL_SHA256,
              "pure_256_score_result_sha256": PURE_RESULT_SHA256,
              "development_collection_protocol_sha256": mixed.DEV_SHA256,
              "source_sha256": checked["protocol"]["source_sha256"],
              "sampling": checked["protocol"]["scoring"], "training_baselines": checked["baseline"],
              "fresh_claim": False, "p1b_claim": False, "promotion_eligible": False,
              "student_actor_trained": False,
              "interpretation": "Paired prediction proxies on the SAME already consumed r6 TRAIN roads, "
                                "source0/random action episodes and seeded reset/terminal 8+32 windows. "
                                "256 base plus 64 extra H8 optimizer steps versus pure-256 are not "
                                "compute-matched; no causal loss-shape, driving, fresh P1b or official claim.",
              "strata": {}}
    with torch.no_grad():
        for stratum_index, label in enumerate(STRATA):
            frozen = checked["frozen"]["strata"][label]
            stratum = {key: value for key, value in frozen.items() if key != "models"}
            stratum["models"] = []
            for seed, entry in enumerate(checked["training"]):
                episodes = [original._score_episode(ep, entry["modules"],
                             checked["baseline"]["constant_reward"],
                             checked["baseline"]["terminal_prevalence"],
                             checked["protocol"]["scoring"]["latent_seed"]
                             + stratum_index * 1_000_003 + index * 1009 + seed)
                            for index, ep in enumerate(datasets[label].episodes)]
                roads: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
                for episode in episodes:
                    roads[(episode["track_id"], episode["geometry_seed"])].append(episode)
                model = {"learner_seed": seed, "checkpoint_sha256": entry["row"]["checkpoint_sha256"],
                         "pure_256_checkpoint_sha256": checked["pure"]["training"][seed]["checkpoint_sha256"],
                         "aggregate": original._aggregate(episodes),
                         "roads": [{"track_id": track, "geometry_seed": road, **original._aggregate(items)}
                                   for (track, road), items in sorted(roads.items())], "episodes": episodes}
                _pair(model, frozen["models"][seed])
                stratum["models"].append(model)
            report["strata"][label] = stratum
    report["local_engineering_gate"] = _gate(report["strata"])
    for name, (digest, cap) in checked["pins"].items():
        mixed._pin(checked["root"], name, digest, max_bytes=cap)
    cgroup = original._cgroup()
    if cgroup["limit_bytes"] > resources["max_cgroup_memory_bytes"] or cgroup["available_bytes"] < checked["required_available_bytes"]:
        raise ValueError("cgroup lacks write-time headroom")
    if os.statvfs(checked["output"].parent).f_bavail * os.statvfs(checked["output"].parent).f_frsize < resources["min_disk_available_bytes"]:
        raise ValueError("disk lacks write-time headroom")
    if checked["output"].exists() or checked["output"].is_symlink():
        raise FileExistsError("score output collision")
    checked["output"].mkdir(exist_ok=False)
    with (checked["output"] / "score-result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = score(args.protocol, args.protocol_sha256, args.output, repo_root=args.repo_root)
    print(json.dumps({"status": result["status"], "local_engineering_gate_passed":
                      result["local_engineering_gate"]["passed"],
                      "output": str(args.output / "score-result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
