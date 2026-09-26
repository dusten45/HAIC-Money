"""Read-only, paired 64-vs-256-update Dreamer world-model budget probe.

The development episodes were already scored in v1 and are consumed iterative
tuning data, not a fresh holdout or a policy/official evaluation. Run from the
repository root with ``python -m scripts.diagnose.dreamerv3_reused_train_budget_probe``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamer_v3 import DreamerV3Config
from scripts.diagnose import dreamerv3_reused_train_open_loop as original


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "haic-dreamerv3-reused-train-budget-probe-v1"
PURPOSE = "reused-TRAIN-consumed-development-update-budget-probe"
PROBE_SOURCE = "scripts/diagnose/dreamerv3_reused_train_budget_probe.py"
V2_OFFLINE_PATH = "experiments/dreamerv3-reused-train-offline-v2.json"
V2_OFFLINE_SHA256 = "ba837ebc3f64282bbd77b4444c6f337c23a3ebce12602d328e9626c1c406b290"
ORIGINAL_SCORE_PROTOCOL_PATH = "experiments/dreamerv3-reused-train-score-v1.json"
ORIGINAL_SCORE_PROTOCOL_SHA256 = "8fc90d1024e6bc55d1e42d3ba442bd3c493bfece195571fa34063a15203c4a5d"
ORIGINAL_SCORE_RESULT_PATH = (
    "runs/20260926-dreamerv3-reused-train-diagnostic-v1/scoring-v1/score-result.json"
)
ORIGINAL_SCORE_RESULT_SHA256 = "748de44441dedf08b98d2fc28464578cf28b7d9d0483eb762552cb142ad6b4b5"
TRAIN_ROOT = "runs/20260926-dreamerv3-reused-train-diagnostic-v1/offline-v2"
RUN_ROOT = "runs/20260926-dreamerv3-reused-train-diagnostic-v1"
METRICS = frozenset({"image_mse", "shifted_repeat_mse", "reward_mse", "training_constant_reward_mse",
                     "last_logged_reward_mse", "terminal_bce", "training_prevalence_bce"})
BASELINE_METRICS = METRICS - {"image_mse", "reward_mse", "terminal_bce"}


def _fixed_file(root: Path, name: str, expected: str, digest: str, *, cap: int) -> Path:
    if name != expected:
        raise ValueError(f"expected pinned path {expected}")
    path = root
    for part in expected.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink input forbidden: {expected}")
    if not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError(f"missing or oversized input: {expected}")
    if original._sha256(path) != original._hash(digest):
        raise ValueError(f"SHA-256 mismatch: {expected}")
    return path


def _ref(root: Path, ref: Any, name: str, sha256: str) -> dict[str, Any]:
    if ref != {"path": name, "sha256": sha256}:
        raise ValueError(f"exact frozen reference required: {name}")
    return original._json(_fixed_file(root, name, name, sha256, cap=1024 * 1024))


def _model_only_checkpoint(path: Path, *, arm: str, seed: int, offline_sha: str,
                           dataset_digest: str, config: dict[str, Any],
                           reference_path: Path | None = None) -> dict[str, Any]:
    # Call only after fixed location, SHA, size and cgroup preflight. Checkpoints
    # are trusted, frozen torch artifacts, not arbitrary user-supplied pickles.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"purpose": "reused-TRAIN-engineering-diagnostic",
                "study_id": "dreamerv3-reused-train-diagnostic-v1", "arm": arm, "seed": seed,
                "offline_protocol_sha256": offline_sha, "dataset_digest": dataset_digest,
                "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                "promotion_eligible": False}
    metadata = payload.get("run_metadata") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict) or payload.get("format") != "haic-dreamerv3-checkpoint-v3"
            or not isinstance(metadata, dict) or set(metadata) != set(expected)
            or any(type(metadata[key]) is not type(value) or metadata[key] != value
                   for key, value in expected.items())
            or type(payload.get("environment_steps")) is not int or payload["environment_steps"] != 0
            or type(payload.get("gradient_steps")) is not int or payload["gradient_steps"] != 256
            or payload.get("config") != config or set(config) != {field.name for field in fields(DreamerV3Config)}
            or config.get("device") != "cpu"):
        raise ValueError(f"{arm}/{seed}: 256-update model-only checkpoint metadata/config mismatch")
    for name in ("actor_optimizer", "critic_optimizer"):
        optimizer = payload.get(name)
        if not isinstance(optimizer, dict) or optimizer.get("state") != {}:
            raise ValueError(f"{arm}/{seed}: actor/critic optimizer has update state")
    if reference_path is not None:
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        for name in ("actor", "critic", "critic_target"):
            before, after = reference.get(name), payload.get(name)
            if (not isinstance(before, dict) or not isinstance(after, dict) or not before
                    or before.keys() != after.keys() or any(
                        not isinstance(before[key], torch.Tensor) or not isinstance(after[key], torch.Tensor)
                        or not torch.equal(before[key], after[key]) for key in before
                    )):
                raise ValueError(f"{arm}/{seed}: {name} differs from pinned 64-update model")
        del reference
    return payload


def _modules(path: Path, *, arm: str, seed: int, offline_sha: str,
             dataset_digest: str, config: dict[str, Any]) -> tuple[Any, ...]:
    payload = _model_only_checkpoint(path, arm=arm, seed=seed, offline_sha=offline_sha,
                                     dataset_digest=dataset_digest, config=config)
    encoder = original.ConvEncoder(in_channels=4, embed_dim=config["embed_dim"])
    rssm = original.CategoricalRSSM(action_dim=3, embed_dim=config["embed_dim"],
                                    hidden_dim=config["hidden_dim"], num_categoricals=config["num_categoricals"],
                                    num_classes=config["num_classes"], unimix=config["unimix"])
    decoder = original.ConvDecoder(in_features=rssm.state_dim, out_channels=1)
    reward = original.RewardHead(in_features=rssm.state_dim, bins=config["twohot_bins"])
    continuation = original.ContinueHead(in_features=rssm.state_dim)
    for name, module in (("encoder", encoder), ("rssm", rssm), ("decoder", decoder),
                         ("reward_head", reward), ("continue_head", continuation)):
        module.load_state_dict(payload[name], strict=True)
        module.eval()
    del payload
    return encoder, rssm, decoder, reward, continuation


def _deltas(current: Any, reference: Any, *, group: str) -> dict[str, float]:
    if (not isinstance(current, dict) or not isinstance(reference, dict)
            or set(current) != METRICS or set(reference) != METRICS):
        raise ValueError(f"{group}: metric names differ from original score")
    delta = {}
    for key in sorted(METRICS):
        now, before = current[key], reference[key]
        if (type(now) not in (int, float) or type(before) not in (int, float)
                or not math.isfinite(now) or not math.isfinite(before)):
            raise ValueError(f"{group}: nonfinite or missing paired metric")
        if key in BASELINE_METRICS and now != before:
            raise ValueError(f"{group}: unchanged-data baseline differs from original")
        delta[key] = original._finite(now - before)
    return delta


def _pair(current: dict[str, Any], reference: dict[str, Any]) -> None:
    if (current["model_arm"] != reference["model_arm"] or current["learner_seed"] != reference["learner_seed"]
            or current["training_baselines"] != reference["training_baselines"]):
        raise ValueError("model identity or TRAIN-only baselines differ from 64-update score")
    for section, metric_name, keys in (("aggregate", "metrics_episode_mean", ()),
                                        ("roads", "metrics_episode_mean", ("track_id", "geometry_seed")),
                                        ("episodes", "metrics", ("track_id", "geometry_seed", "episode_id"))):
        left = [current[section]] if section == "aggregate" else current[section]
        right = [reference[section]] if section == "aggregate" else reference[section]
        if len(left) != len(right):
            raise ValueError(f"{section}: episode/road denominators differ")
        if section != "aggregate" and len({tuple(row[k] for k in keys) for row in right}) != len(right):
            raise ValueError(f"{section}: duplicate original paired identity")
        for now, before in zip(left, right, strict=True):
            expected = set(before) - {metric_name, "windows", "paired_delta_vs_64"}
            if ({key: now.get(key) for key in expected} != {key: before[key] for key in expected}
                    or set(now) - {metric_name, "windows", "paired_delta_vs_64"} != expected):
                raise ValueError(f"{section}: identity or label/window denominators differ")
            now["paired_delta_vs_64"] = _deltas(now[metric_name], before[metric_name], group=section)
            if section == "episodes":
                if len(now["windows"]) != 2 or len(before["windows"]) != 2 or now["decisions"] < 41:
                    raise ValueError("episode: exact two original scoring windows required")
                for index, (window, old) in enumerate(zip(now["windows"], before["windows"], strict=True)):
                    anchor = 8 if index == 0 else now["decisions"] - 32
                    if (window["kind"] != ("reset" if index == 0 else "terminal")
                            or window["context_anchor_decision"] != anchor
                            or window["first_target_decision"] != anchor
                            or window["last_target_decision"] != anchor + 31
                            or window["reset_origin_prefix_decisions"] != anchor
                            or {key: value for key, value in window.items() if key not in ("metrics", "paired_delta_vs_64")}
                            != {key: value for key, value in old.items() if key != "metrics"}):
                        raise ValueError("episode: original reset/terminal score window differs")
                    window["paired_delta_vs_64"] = _deltas(window["metrics"], old["metrics"], group="window")


def preflight(protocol_path: Path, protocol_sha256: str, output_dir: Path,
              *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Check both pinned experiments before reading any development ZIP bytes."""
    root = Path(repo_root).resolve()
    name = original._relative(root, protocol_path)
    if not name.startswith("experiments/dreamerv3-reused-train-budget-probe-"):
        raise ValueError("a separately frozen budget-probe protocol is required")
    protocol = original._json(original._pinned(root, name, protocol_sha256, "protocol"))
    if (set(protocol) != {"format", "purpose", "study_id", "offline_protocol", "training",
                          "development_collection_protocol", "development", "source_sha256",
                          "scoring", "resources", "original_score_protocol", "original_score_result"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or protocol["study_id"] != original.DEVELOPMENT_ID):
        raise ValueError("invalid consumed-development budget-probe schema")
    v1_protocol = _ref(root, protocol["original_score_protocol"], ORIGINAL_SCORE_PROTOCOL_PATH,
                       ORIGINAL_SCORE_PROTOCOL_SHA256)
    v1_result = _ref(root, protocol["original_score_result"], ORIGINAL_SCORE_RESULT_PATH,
                     ORIGINAL_SCORE_RESULT_SHA256)
    if (v1_result.get("format") != original.FORMAT or v1_result.get("purpose") != original.PURPOSE
            or v1_result.get("score_protocol_sha256") != ORIGINAL_SCORE_PROTOCOL_SHA256
            or v1_result.get("offline_protocol_sha256") != original.OFFLINE_SHA256
            or v1_result.get("status") != "descriptive_only"
            or any(v1_result.get(key) is not False for key in ("fresh_claim", "p1b_claim", "promotion_eligible", "student_actor_trained"))
            or v1_result.get("sampling") != v1_protocol["scoring"]
            or v1_result.get("source_sha256") != v1_protocol["source_sha256"]):
        raise ValueError("original pinned score result/protocol identity mismatch")
    output_name = original._relative(root, output_dir)
    output = original._path(root, output_name, "output", exists=False)
    if output.parent != root / RUN_ROOT or not output.parent.is_dir():
        raise ValueError("probe output must be a new sibling runs directory")
    checked_v1 = original.preflight(root / ORIGINAL_SCORE_PROTOCOL_PATH, ORIGINAL_SCORE_PROTOCOL_SHA256,
                                    output, repo_root=root)
    v1_offline = checked_v1["offline"]
    if (protocol["offline_protocol"] != {"path": V2_OFFLINE_PATH, "sha256": V2_OFFLINE_SHA256}
            or v1_protocol["offline_protocol"] != {"path": original.OFFLINE_PATH, "sha256": original.OFFLINE_SHA256}):
        raise ValueError("exact first- and second-loop offline protocol pins required")
    v2_offline = _ref(root, protocol["offline_protocol"], V2_OFFLINE_PATH, V2_OFFLINE_SHA256)
    unchanged = dict(v2_offline)
    learner = unchanged.get("learner")
    if not isinstance(learner, dict):
        raise ValueError("second-loop learner missing")
    unchanged["learner"] = dict(learner)
    if (type(learner.get("updates")) is not int or learner["updates"] != 256
            or v1_offline["learner"].get("updates") != 64):
        raise ValueError("256-vs-64 model-only update budget required")
    unchanged["learner"]["updates"] = 64
    if unchanged != v1_offline:
        raise ValueError("offline v2 differs from v1 beyond learner.updates")
    if (protocol["development_collection_protocol"] != v1_protocol["development_collection_protocol"]
            or protocol["development"] != v1_protocol["development"]
            or protocol["scoring"] != v1_protocol["scoring"]
            or protocol["resources"] != v1_protocol["resources"]
            or protocol["scoring"]["latent_seed"] != 120926):
        raise ValueError("development archives, fixed score windows or resource caps differ from first loop")
    sources = protocol["source_sha256"]
    if (not isinstance(sources, dict) or set(sources) != original.SOURCE_PATHS | {PROBE_SOURCE}
            or {key: sources[key] for key in original.SOURCE_PATHS} != v1_protocol["source_sha256"]):
        raise ValueError("original and new scorer/model executable source pins required")
    for source, digest in sources.items():
        if source == PROBE_SOURCE:
            _fixed_file(root, source, PROBE_SOURCE, digest, cap=1024 * 1024)
        else:
            original._pinned(root, source, digest, "source")
    if sources[PROBE_SOURCE] != original._sha256(Path(__file__)):
        raise ValueError("executing probe differs from pinned source")
    if (v1_result.get("development_collection_protocol_sha256")
            != protocol["development_collection_protocol"]["sha256"]
            or set(v1_result.get("strata", {})) != {"random", "teacher"}
            or set(protocol.get("training", {})) != {"random", "teacher"}):
        raise ValueError("original score strata or second-loop training arms missing")
    resources = protocol["resources"]
    training = {}
    for arm in ("random", "teacher"):
        rows = protocol["training"][arm]
        original_models = v1_result["strata"][arm].get("models")
        if (not isinstance(rows, list) or len(rows) != 2 or not isinstance(original_models, list)
                or len(original_models) != 4
                or v1_result["strata"][arm].get("development_archive_sha256")
                != protocol["development"][arm]["archive_sha256"]
                or v1_result["strata"][arm].get("coverage") != "descriptive_sufficient_minimum"
                or [(item.get("model_arm"), item.get("learner_seed")) for item in original_models]
                != [(model_arm, seed) for model_arm in ("random", "teacher") for seed in (0, 1)]):
            raise ValueError("original paired model/source stratum identity mismatch")
        training[arm] = []
        for seed, row in enumerate(rows):
            if (not isinstance(row, dict) or set(row) != {"seed", "result_path", "result_sha256",
                                                        "checkpoint_path", "checkpoint_sha256"}
                    or type(row["seed"]) is not int or row["seed"] != seed):
                raise ValueError("exact two fixed seeds and SHA-pinned result/checkpoint pairs required")
            prefix = f"{TRAIN_ROOT}/{arm}-seed-{seed}"
            result_file = _fixed_file(root, row["result_path"], f"{prefix}/training-result.json",
                                      row["result_sha256"], cap=1024 * 1024)
            checkpoint = _fixed_file(root, row["checkpoint_path"], f"{prefix}/world-model-checkpoint.pt",
                                     row["checkpoint_sha256"], cap=resources["max_checkpoint_bytes"])
            result = original._json(result_file)
            for key, expected in (("format", "haic-dreamerv3-reused-train-offline-result-v1"),
                                  ("status", "complete"), ("purpose", v2_offline["purpose"]),
                                  ("study_id", v2_offline["study_id"]), ("arm", arm), ("seed", seed),
                                  ("offline_protocol_sha256", V2_OFFLINE_SHA256),
                                  ("collection_protocol_sha256", v2_offline["collection_protocol"]["sha256"]),
                                  ("collection_receipt_path", v2_offline["datasets"][arm]["receipt_path"]),
                                  ("collection_receipt_sha256", v2_offline["datasets"][arm]["receipt_sha256"]),
                                  ("checkpoint_path", "world-model-checkpoint.pt"),
                                  ("checkpoint_sha256", row["checkpoint_sha256"]),
                                  ("model_only_updates", 256), ("environment_steps", 0),
                                  ("actor_trained", False), ("actor_critic_target_unchanged", True),
                                  ("fresh_claim", False), ("p1b_claim", False), ("promotion_eligible", False)):
                if type(result.get(key)) is not type(expected) or result[key] != expected:
                    raise ValueError(f"{arm}/{seed}: training result {key} disagrees with freeze")
            train_row = v2_offline["datasets"][arm]
            for key in ("archive_path", "archive_sha256", "dataset_digest", "source_id", "source_actor_sha256"):
                if result.get(key) != train_row[key]:
                    raise ValueError(f"{arm}/{seed}: training result {key} differs from sealed source")
            memory = original._cgroup()
            if (memory["limit_bytes"] > resources["max_cgroup_memory_bytes"]
                    or memory["available_bytes"] < checked_v1["required_available_bytes"]):
                raise ValueError("cgroup lacks frozen checkpoint-comparison headroom")
            v1_row = v1_protocol["training"][arm][seed]
            reference = _fixed_file(root, v1_row["checkpoint_path"], v1_row["checkpoint_path"],
                                    v1_row["checkpoint_sha256"], cap=resources["max_checkpoint_bytes"])
            _model_only_checkpoint(checkpoint, arm=arm, seed=seed, offline_sha=V2_OFFLINE_SHA256,
                                   dataset_digest=train_row["dataset_digest"],
                                   config=learner["config"], reference_path=reference)
            training[arm].append({"checkpoint": checkpoint, "sha256": row["checkpoint_sha256"]})
    return {"root": root, "protocol": protocol, "protocol_path": name, "output": output,
            "offline": v2_offline, "original": v1_result, "checked_v1": checked_v1,
            "training": training, "required_available_bytes": checked_v1["required_available_bytes"]}


def score(protocol_path: Path, protocol_sha256: str, output_dir: Path,
          *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, output_dir, repo_root=repo_root)
    root, protocol, offline = checked["root"], checked["protocol"], checked["offline"]
    resources = protocol["resources"]
    cgroup = original._cgroup()
    if (cgroup["limit_bytes"] > resources["max_cgroup_memory_bytes"]
            or cgroup["available_bytes"] < checked["required_available_bytes"]):
        raise ValueError("cgroup lacks frozen archive-load headroom")
    torch.set_num_threads(1)
    baselines = {}
    datasets = {}
    for arm in ("random", "teacher"):
        train_row = offline["datasets"][arm]
        train = original._load_dataset(checked["checked_v1"]["training"][arm][0]["archive"],
                                       train_row, resources, checked["checked_v1"]["training_cells"],
                                       offline["study_id"], arm)
        count = train.transition_count
        if (count != train_row["stored_decisions"]
                or {(ep.track_id, int(ep.geometry_id)) for ep in train.episodes}
                != set(checked["checked_v1"]["training_cells"])):
            raise ValueError("TRAIN archive cells/decisions differ from frozen original")
        baselines[arm] = {"training_decisions": count,
                          "training_terminal_events": sum(int(ep.terminal.sum()) for ep in train.episodes),
                          "constant_reward": original._finite(sum(float(ep.rewards.astype(np.float64).sum())
                                                                 for ep in train.episodes) / count)}
        baselines[arm]["terminal_prevalence"] = original._finite(baselines[arm]["training_terminal_events"] / count)
        del train
        dev = checked["checked_v1"]["development"][arm]
        dataset = original._load_dataset(dev["archive"], dev["row"], resources,
                                         checked["checked_v1"]["development_cells"], original.DEVELOPMENT_ID, arm)
        receipt = dev["receipt"]
        rows = [row for row in receipt["episode_rows"] if row.get("status") == "complete"]
        if (dataset.transition_count != receipt["stored_decisions"]
                or len(dataset.episodes) != receipt["complete_episode_count"] or len(rows) != len(dataset.episodes)):
            raise ValueError("development archive counts differ from pinned receipt")
        for episode, row in zip(dataset.episodes, rows, strict=True):
            if (str(row.get("episode_id")) != episode.episode_id or type(row.get("episode_id")) is not int
                    or episode.metadata.get("attempt") != row.get("attempt")
                    or (episode.track_id, int(episode.geometry_id)) != (row.get("track_id"), row.get("geometry_seed"))
                    or episode.steps != row.get("decisions") or bool(episode.terminal[-1]) != row.get("terminal")
                    or bool(episode.finished[-1]) != row.get("finished")):
                raise ValueError("development episode differs from its sealed receipt")
        datasets[arm] = dataset
    report = {"format": FORMAT, "purpose": PURPOSE, "status": "iterative_tuning_descriptive_only",
              "score_protocol_sha256": protocol_sha256, "offline_protocol_sha256": V2_OFFLINE_SHA256,
              "original_score_protocol_sha256": ORIGINAL_SCORE_PROTOCOL_SHA256,
              "original_score_result_sha256": ORIGINAL_SCORE_RESULT_SHA256,
              "development_collection_protocol_sha256": protocol["development_collection_protocol"]["sha256"],
              "source_sha256": protocol["source_sha256"], "sampling": protocol["scoring"],
              "fresh_claim": False, "p1b_claim": False, "promotion_eligible": False,
              "student_actor_trained": False,
              "interpretation": "Previously scored, training-excluded reused TRAIN development data are consumed "
                                "for iterative tuning. Paired model-only update-budget sensitivity, not a fresh "
                                "holdout, pass, policy result, promotion, blind, confirmation or official metric; "
                                "windows on one road are not independent episodes.",
              "strata": {}}
    with torch.no_grad():
        for source_arm in ("random", "teacher"):
            original_stratum = checked["original"]["strata"][source_arm]
            dataset = datasets[source_arm]
            usable = [ep for ep in dataset.episodes if ep.steps >= 41]
            geometry_count = len({(ep.track_id, ep.geometry_id) for ep in usable})
            scored_terminal = sum(bool(ep.terminal[-1]) for ep in usable)
            if (len(usable) != original_stratum["scored_episode_count"]
                    or len(dataset.episodes) != original_stratum["collected_complete_episodes"]
                    or sum(bool(ep.terminal[-1]) for ep in dataset.episodes)
                    != original_stratum["collected_terminal_episodes"]
                    or scored_terminal != original_stratum["scored_independent_terminal_episodes"]
                    or geometry_count != original_stratum["scored_independent_geometries"]
                    or scored_terminal < protocol["scoring"]["min_terminal_episodes_per_stratum"]
                    or geometry_count < protocol["scoring"]["min_geometries_per_stratum"]):
                raise ValueError("paired development stratum coverage differs from original score")
            stratum = {key: value for key, value in original_stratum.items() if key != "models"}
            stratum["models"] = []
            for model_arm in ("random", "teacher"):
                baseline = baselines[model_arm]
                for seed, row in enumerate(checked["training"][model_arm]):
                    _fixed_file(root, f"{TRAIN_ROOT}/{model_arm}-seed-{seed}/world-model-checkpoint.pt",
                                f"{TRAIN_ROOT}/{model_arm}-seed-{seed}/world-model-checkpoint.pt",
                                row["sha256"], cap=resources["max_checkpoint_bytes"])
                    modules = _modules(row["checkpoint"], arm=model_arm, seed=seed,
                                       offline_sha=V2_OFFLINE_SHA256,
                                       dataset_digest=offline["datasets"][model_arm]["dataset_digest"],
                                       config=offline["learner"]["config"])
                    episodes = [original._score_episode(ep, modules, baseline["constant_reward"],
                                                        baseline["terminal_prevalence"],
                                                        protocol["scoring"]["latent_seed"]
                                                        + (source_arm == "teacher") * 1_000_003
                                                        + index * 1009 + seed)
                                for index, ep in enumerate(dataset.episodes)]
                    del modules
                    roads = defaultdict(list)
                    for episode in episodes:
                        roads[(episode["track_id"], episode["geometry_seed"])].append(episode)
                    model = {"model_arm": model_arm, "learner_seed": seed,
                             "checkpoint_sha256": row["sha256"], "training_baselines": baseline,
                             "aggregate": original._aggregate(episodes),
                             "roads": [{"track_id": track, "geometry_seed": road, **original._aggregate(items)}
                                       for (track, road), items in sorted(roads.items())],
                             "episodes": episodes}
                    reference = original_stratum["models"][(0 if model_arm == "random" else 2) + seed]
                    if reference["checkpoint_sha256"] != checked["checked_v1"]["training"][model_arm][seed]["checkpoint_sha256"]:
                        raise ValueError("original score checkpoint identity differs from protocol")
                    _pair(model, reference)
                    stratum["models"].append(model)
            report["strata"][source_arm] = stratum
    # Recheck all pins before creating a NEW output directory; no replay/training/environment access.
    _fixed_file(root, checked["protocol_path"], checked["protocol_path"], protocol_sha256, cap=1024 * 1024)
    _ref(root, protocol["offline_protocol"], V2_OFFLINE_PATH, V2_OFFLINE_SHA256)
    _ref(root, protocol["original_score_protocol"], ORIGINAL_SCORE_PROTOCOL_PATH, ORIGINAL_SCORE_PROTOCOL_SHA256)
    _ref(root, protocol["original_score_result"], ORIGINAL_SCORE_RESULT_PATH, ORIGINAL_SCORE_RESULT_SHA256)
    for ref in (checked["checked_v1"]["protocol"]["offline_protocol"],
                checked["checked_v1"]["offline"]["collection_protocol"],
                protocol["development_collection_protocol"],
                checked["checked_v1"]["training_collection"]["r6_protocol"]):
        original._pinned(root, ref["path"], ref["sha256"],
                         "r6" if ref["path"] == "experiments/drqv2-geometry-mix-v1-r6.json" else "protocol")
    for name, sha in protocol["source_sha256"].items():
        _fixed_file(root, name, name, sha, cap=1024 * 1024)
    for arm in ("random", "teacher"):
        for row in checked["checked_v1"]["protocol"]["training"][arm]:
            original._pinned(root, row["result_path"], row["result_sha256"], "training_result")
            _fixed_file(root, row["checkpoint_path"], row["checkpoint_path"], row["checkpoint_sha256"],
                        cap=resources["max_checkpoint_bytes"])
        for row in protocol["training"][arm]:
            _fixed_file(root, row["result_path"], row["result_path"], row["result_sha256"], cap=1024 * 1024)
            _fixed_file(root, row["checkpoint_path"], row["checkpoint_path"], row["checkpoint_sha256"],
                        cap=resources["max_checkpoint_bytes"])
        for item in (offline["datasets"][arm], protocol["development"][arm]):
            for kind in ("receipt", "archive"):
                original._pinned(root, item[f"{kind}_path"], item[f"{kind}_sha256"], kind)
    original._path(root, original._relative(root, checked["output"]), "output", exists=False)
    checked["output"].mkdir(exist_ok=False)
    with (checked["output"] / "score-result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = score(args.protocol, args.protocol_sha256, args.output, repo_root=args.repo_root)
    print(json.dumps({"status": result["status"], "output": str(args.output / "score-result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
