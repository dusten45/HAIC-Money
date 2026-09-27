"""Read-only, consumed-development prior-image vs original 256-update score.

Requires a separately frozen protocol and a new output directory. No simulator,
replay updates, actor scoring, or fresh P1/P1b evaluation is performed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import json
import math
from pathlib import Path
from typing import Any

import torch

from dreamer_v3 import DreamerV3Config
from scripts.diagnose import dreamerv3_reused_train_budget_probe as budget
from scripts.diagnose import dreamerv3_reused_train_open_loop as original


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "haic-dreamerv3-reused-prior-image-score-v1"
PURPOSE = "reused-TRAIN-consumed-development-prior-image-diagnostic"
SCORE_PROTOCOL_PATH = "experiments/dreamerv3-reused-train-prior-image-score-v1.json"
PRIOR_PROTOCOL_PATH = "experiments/dreamerv3-reused-train-prior-image-v1.json"
PRIOR_PROTOCOL_SHA256 = "bddec29a834f68ac9cd25f151fdf3da30c798b9f0be7adc93c2e7c74080eaa71"
BUDGET_PROTOCOL_PATH = "experiments/dreamerv3-reused-train-budget-probe-v1.json"
BUDGET_PROTOCOL_SHA256 = "ddcb33d19b36f6c0c17942263da835a45bf87f76271f550027e6113824ebf994"
BUDGET_RESULT_PATH = f"{budget.RUN_ROOT}/scoring-v2/score-result.json"
BUDGET_RESULT_SHA256 = "94bbd7457283d5ec8f0ad4a4ff2e654b97839d09a7eca764615de6fae625ff5d"
TRAIN_ROOT = f"{budget.RUN_ROOT}/offline-prior-v1"
SOURCE = "scripts/diagnose/dreamerv3_reused_prior_image_score.py"
ANCHOR_MSE = 0.01488432277896834
GATE = {
    "metric": "image_mse", "window_decisions": 32, "model_arm": "teacher",
    "learner_seeds": [0, 1], "source_strata": ["random", "teacher"],
    "strictly_below_own_256_on_both_strata": True,
    "strictly_below_shifted_repeat_on_teacher_stratum": True,
    "teacher_shifted_repeat_mse": ANCHOR_MSE,
}
_ARMS = ("random", "teacher")


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _exact(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    if (not isinstance(actual, dict) or set(actual) != set(expected)
            or any(type(actual[key]) is not type(value) or actual[key] != value
                   for key, value in expected.items())):
        raise ValueError(f"{label}: metadata or budget differs from frozen contract")


def _metrics_stream(root: Path, prefix: str, result: dict[str, Any], kind: str, count: int) -> Path:
    name = f"{kind}-update-metrics.jsonl"
    if result.get(f"{kind}_metrics_path") != name:
        raise ValueError(f"{kind}: unexpected metrics stream path")
    path = budget._fixed_file(root, f"{prefix}/{name}", f"{prefix}/{name}",
                              result.get(f"{kind}_metrics_sha256"), cap=16 * 1024**2)
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            rows = index
            if index > count or len(line) > 4096:
                raise ValueError(f"{kind}: metrics stream exceeds frozen count/row cap")
            item = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if (not isinstance(item, dict) or set(item) != ({"base_update", "metrics"} if kind == "base" else
                                                            {"aux_update", "after_base_update", "metrics"})
                    or type(item.get(f"{kind}_update")) is not int or item[f"{kind}_update"] != index
                    or (kind == "aux" and (type(item["after_base_update"]) is not int
                                           or item["after_base_update"] != index * 4))
                    or not isinstance(item["metrics"], dict) or not item["metrics"]
                    or ("loss_wm" not in item["metrics"] if kind == "base" else
                        not {"aux_loss", "frame_mse", "anchor_count", "target_count", "effective_horizon"}
                        <= item["metrics"].keys())
                    or any(not isinstance(key, str) or type(value) not in (float, int)
                           or not math.isfinite(value) for key, value in item["metrics"].items())):
                raise ValueError(f"{kind}: invalid optimizer-step metrics")
        if rows != count:
            raise ValueError(f"{kind}: wrong optimizer-step metrics count")
    return path


def _aux_payload(path: Path, *, arm: str, seed: int, result: dict[str, Any],
                 prior: dict[str, Any], prior_sha: str, offline: dict[str, Any],
                 reference: Path) -> dict[str, Any]:
    # Torch deserialization occurs only after fixed path, SHA, size, source and cgroup checks.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    row = offline["datasets"][arm]
    learner = prior["learner"]
    metadata = {
        "purpose": prior["purpose"], "study_id": prior["study_id"], "arm": arm, "seed": seed,
        "prior_protocol_sha256": prior_sha, "base_offline_protocol_sha256": budget.V2_OFFLINE_SHA256,
        "source_sha256": prior["source_sha256"], "collection_receipt_sha256": row["receipt_sha256"],
        "archive_sha256": row["archive_sha256"], "dataset_digest": row["dataset_digest"],
        "base_metrics_sha256": result["base_metrics_sha256"],
        "aux_metrics_sha256": result["aux_metrics_sha256"],
        "base_model_only_updates": 256, "aux_world_model_optimizer_steps": 64,
        "world_model_optimizer_steps": 320, "prior_horizon": 8, "prior_weight": 0.25,
        "anchor_mode": learner["anchor_mode"], "aux_rng_mode": learner["aux_rng_mode"],
        "aux_rng_seed_base": learner["aux_rng_seed_base"],
        "aux_rng_seed_stride": learner["aux_rng_seed_stride"],
        "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
        "promotion_eligible": False,
    }
    config = offline["learner"]["config"]
    if (not isinstance(payload, dict) or payload.get("format") != "haic-dreamerv3-checkpoint-v3"
            or type(payload.get("environment_steps")) is not int or payload["environment_steps"] != 0
            or type(payload.get("gradient_steps")) is not int or payload["gradient_steps"] != 256
            or payload.get("config") != config or set(config) != {field.name for field in fields(DreamerV3Config)}
            or config.get("device") != "cpu"):
        raise ValueError(f"{arm}/{seed}: auxiliary checkpoint metadata/config/gradient steps mismatch")
    _exact(payload.get("run_metadata"), metadata, f"{arm}/{seed}: auxiliary checkpoint run_metadata")
    baseline = torch.load(reference, map_location="cpu", weights_only=False)
    for name in ("actor", "critic", "critic_target", "actor_optimizer", "critic_optimizer"):
        before, after = baseline.get(name), payload.get(name)
        if (not isinstance(before, dict) or not before or not isinstance(after, dict)
                or not _same(before, after)
                or (name.endswith("optimizer") and after.get("state") != {})):
            raise ValueError(f"{arm}/{seed}: {name} differs from seeded original 256-only checkpoint")
    return payload


def _modules(payload: dict[str, Any], config: dict[str, Any]) -> tuple[Any, ...]:
    encoder = original.ConvEncoder(in_channels=4, embed_dim=config["embed_dim"])
    rssm = original.CategoricalRSSM(action_dim=3, embed_dim=config["embed_dim"],
                                    hidden_dim=config["hidden_dim"], num_categoricals=config["num_categoricals"],
                                    num_classes=config["num_classes"], unimix=config["unimix"])
    modules = (encoder, rssm, original.ConvDecoder(in_features=rssm.state_dim, out_channels=1),
               original.RewardHead(in_features=rssm.state_dim, bins=config["twohot_bins"]),
               original.ContinueHead(in_features=rssm.state_dim))
    for name, module in zip(("encoder", "rssm", "decoder", "reward_head", "continue_head"), modules, strict=True):
        module.load_state_dict(payload[name], strict=True)
        module.eval()
    return modules


def _window_identity(episode: dict[str, Any]) -> None:
    length = episode.get("decisions")
    if type(length) is not int or length < 41 or len(episode.get("windows", [])) != 2:
        raise ValueError("original score lacks two eligible windows")
    for window, (kind, anchor) in zip(episode["windows"], (("reset", 8), ("terminal", length - 32)), strict=True):
        for key, value in {"kind": kind, "context_anchor_decision": anchor,
                           "reset_origin_prefix_decisions": anchor, "first_target_decision": anchor,
                           "last_target_decision": anchor + 31, "label_uses": 32}.items():
            if window.get(key) != value or type(window.get(key)) is not type(value):
                raise ValueError("original score window identity differs from frozen 8+32 rule")
        budget._deltas(window.get("metrics"), window.get("metrics"), group="frozen window")


def _reference(checked: dict[str, Any]) -> None:
    """Reject broken reference identities and gates before opening a development ZIP."""
    frozen, protocol = checked["original"], checked["protocol"]
    expected = {"format": budget.FORMAT, "purpose": budget.PURPOSE,
                "score_protocol_sha256": BUDGET_PROTOCOL_SHA256,
                "offline_protocol_sha256": budget.V2_OFFLINE_SHA256,
                "original_score_protocol_sha256": budget.ORIGINAL_SCORE_PROTOCOL_SHA256,
                "original_score_result_sha256": budget.ORIGINAL_SCORE_RESULT_SHA256,
                "status": "iterative_tuning_descriptive_only", "fresh_claim": False,
                "p1b_claim": False, "promotion_eligible": False, "student_actor_trained": False,
                "sampling": protocol["scoring"], "source_sha256": protocol["source_sha256"],
                "development_collection_protocol_sha256": protocol["development_collection_protocol"]["sha256"]}
    for key, value in expected.items():
        if type(frozen.get(key)) is not type(value) or frozen[key] != value:
            raise ValueError(f"original 256-only score {key} differs from frozen contract")
    if set(frozen.get("strata", {})) != set(_ARMS):
        raise ValueError("original score missing a development stratum")
    for source in _ARMS:
        stratum = frozen["strata"][source]
        if (stratum.get("development_archive_sha256") != protocol["development"][source]["archive_sha256"]
                or stratum.get("source_id") != protocol["development"][source]["source_id"]
                or stratum.get("coverage") != "descriptive_sufficient_minimum"
                or stratum.get("coverage_reasons") != []
                or any(stratum.get(key) != 4 for key in ("collected_complete_episodes",
                      "collected_terminal_episodes", "scored_episode_count",
                      "scored_independent_terminal_episodes", "scored_independent_geometries"))
                or not isinstance(stratum.get("models"), list) or len(stratum["models"]) != 4):
            raise ValueError("original score development coverage or source mismatch")
        anchor = None
        for index, (arm, seed) in enumerate((arm, seed) for arm in _ARMS for seed in (0, 1)):
            model = stratum["models"][index]
            train = checked["offline"]["datasets"][arm]
            baseline = model.get("training_baselines")
            if (model.get("model_arm") != arm or type(model.get("learner_seed")) is not int
                    or model["learner_seed"] != seed
                    or model.get("checkpoint_sha256") != protocol["training"][arm][seed]["checkpoint_sha256"]
                    or not isinstance(baseline, dict) or set(baseline) != {
                        "training_decisions", "training_terminal_events", "constant_reward", "terminal_prevalence"}
                    or type(baseline["training_decisions"]) is not int
                    or baseline["training_decisions"] != train["stored_decisions"]
                    or type(baseline["training_terminal_events"]) is not int
                    or not 0 <= baseline["training_terminal_events"] <= baseline["training_decisions"]
                    or type(baseline["constant_reward"]) not in (int, float)
                    or not math.isfinite(baseline["constant_reward"])
                    or baseline["terminal_prevalence"] != baseline["training_terminal_events"] / baseline["training_decisions"]
                    or not isinstance(model.get("episodes"), list) or len(model["episodes"]) != 4
                    or not isinstance(model.get("roads"), list) or len(model["roads"]) != 4):
                raise ValueError("original score model identity or training baselines mismatch")
            if baseline != frozen["strata"]["random"]["models"][(0 if arm == "random" else 2) + seed]["training_baselines"]:
                raise ValueError("original score training baseline changed across source strata")
            aggregate = model["aggregate"]
            if (aggregate.get("episode_count") != 4 or aggregate.get("window_count") != 8
                    or aggregate.get("window_label_uses") != 256
                    or aggregate.get("unique_terminal_positive_labels") != 4):
                raise ValueError("original score aggregate denominators differ")
            budget._deltas(aggregate.get("metrics_episode_mean"), aggregate.get("metrics_episode_mean"), group="frozen aggregate")
            repeat = aggregate["metrics_episode_mean"]["shifted_repeat_mse"]
            if anchor is None:
                anchor = repeat
            elif repeat != anchor:
                raise ValueError("original model anchor baseline changed")
            roads = set()
            road_rows = {(row.get("track_id"), row.get("geometry_seed")): row for row in model["roads"]}
            if len(road_rows) != 4:
                raise ValueError("original score has duplicate roads")
            for episode in model["episodes"]:
                _window_identity(episode)
                budget._deltas(episode.get("metrics"), episode.get("metrics"), group="frozen episode")
                cell = (episode.get("track_id"), episode.get("geometry_seed"))
                road = road_rows.get(cell, {})
                if (cell not in checked["checked_v1"]["development_cells"] or cell in roads
                        or road.get("track_id") != cell[0] or road.get("geometry_seed") != cell[1]
                        or road.get("episode_count") != 1 or road.get("window_count") != 2
                        or road.get("metrics_episode_mean") != episode["metrics"]):
                    raise ValueError("original score road or episode identity mismatch")
                roads.add(cell)
            if roads != set(checked["checked_v1"]["development_cells"]):
                raise ValueError("original score misses a frozen development road")
        if source == "teacher" and anchor != ANCHOR_MSE:
            raise ValueError("teacher anchor differs from predeclared frozen-repeat baseline")


def _pair(current: dict[str, Any], frozen: dict[str, Any]) -> None:
    if (current["model_arm"] != frozen["model_arm"] or current["learner_seed"] != frozen["learner_seed"]
            or current["training_baselines"] != frozen["training_baselines"]):
        raise ValueError("paired model identity or TRAIN baselines differ")
    for section, metric_key, keys in (("aggregate", "metrics_episode_mean", ()),
                                      ("roads", "metrics_episode_mean", ("track_id", "geometry_seed")),
                                      ("episodes", "metrics", ("track_id", "geometry_seed", "episode_id"))):
        now_rows = [current[section]] if section == "aggregate" else current[section]
        old_rows = [frozen[section]] if section == "aggregate" else frozen[section]
        if len(now_rows) != len(old_rows) or (keys and len({tuple(row[k] for k in keys) for row in old_rows}) != len(old_rows)):
            raise ValueError(f"{section}: paired rows missing or duplicated")
        for now, old in zip(now_rows, old_rows, strict=True):
            ignored = {metric_key, "windows", "paired_delta_vs_64", "paired_delta_vs_256", "original_256_metrics"}
            old_identity = {key: value for key, value in old.items() if key not in ignored}
            now_identity = {key: value for key, value in now.items() if key not in ignored}
            if old_identity != now_identity:
                raise ValueError(f"{section}: paired label, geometry or denominator mismatch")
            now["paired_delta_vs_256"] = budget._deltas(now[metric_key], old[metric_key], group=section)
            now["original_256_metrics"] = old[metric_key]
            if section == "episodes":
                _window_identity(now)
                _window_identity(old)
                for window, reference in zip(now["windows"], old["windows"], strict=True):
                    ignored_window = {"metrics", "paired_delta_vs_64", "paired_delta_vs_256", "original_256_metrics"}
                    if ({key: value for key, value in window.items() if key not in ignored_window}
                            != {key: value for key, value in reference.items() if key not in ignored_window}):
                        raise ValueError("episode: paired reset/terminal window mismatch")
                    window["paired_delta_vs_256"] = budget._deltas(window["metrics"], reference["metrics"], group="window")
                    window["original_256_metrics"] = reference["metrics"]


def preflight(protocol_path: Path, protocol_sha256: str, output_dir: Path,
              *, repo_root: Path = ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    name = original._relative(root, protocol_path)
    if name != SCORE_PROTOCOL_PATH:
        raise ValueError("a separate, fixed-name prior-image scoring protocol is required")
    protocol = original._json(budget._fixed_file(root, name, SCORE_PROTOCOL_PATH,
                                                 protocol_sha256, cap=1024 * 1024))
    if (not isinstance(protocol, dict) or set(protocol) != {"format", "purpose", "study_id",
            "prior_protocol", "budget_score_protocol", "budget_score_result", "training",
            "source_sha256", "local_gate"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or protocol["study_id"] != original.DEVELOPMENT_ID):
        raise ValueError("invalid isolated prior-image scoring schema or predeclared gate")
    _exact(protocol["local_gate"], GATE, "predeclared local engineering gate")
    prior = budget._ref(root, protocol["prior_protocol"], PRIOR_PROTOCOL_PATH, PRIOR_PROTOCOL_SHA256)
    frozen = budget._ref(root, protocol["budget_score_result"], BUDGET_RESULT_PATH, BUDGET_RESULT_SHA256)
    budget_protocol = budget._ref(root, protocol["budget_score_protocol"], BUDGET_PROTOCOL_PATH,
                                  BUDGET_PROTOCOL_SHA256)
    output = original._path(root, original._relative(root, output_dir), "output", exists=False)
    if output.parent != root / budget.RUN_ROOT or not output.parent.is_dir():
        raise ValueError("output must be a new sibling diagnostic runs directory")
    checked = budget.preflight(root / BUDGET_PROTOCOL_PATH, BUDGET_PROTOCOL_SHA256,
                               output, repo_root=root)
    if (budget_protocol != checked["protocol"] or frozen != original._json(root / BUDGET_RESULT_PATH)
            or prior.get("format") != "haic-dreamerv3-reused-prior-image-offline-v1"
            or prior.get("purpose") != "reused-TRAIN-prior-image-auxiliary-diagnostic"
            or prior.get("study_id") != checked["offline"]["study_id"]
            or prior.get("base_offline_protocol") != checked["protocol"]["offline_protocol"]
            or prior.get("output_root") != TRAIN_ROOT
            or prior.get("learner") != {
                "base_updates": 256, "aux_updates": 64, "aux_every_base_updates": 4,
                "prior_horizon": 8, "prior_weight": 0.25,
                "anchor_mode": "deterministic-first-last-full-horizon",
                "aux_rng_mode": "fork-torch-numpy-python", "aux_rng_seed_base": 260926000,
                "aux_rng_seed_stride": 1000}
            or prior.get("resources") != {key: checked["protocol"]["resources"][key]
                                                for key in ("max_archive_bytes", "max_cgroup_memory_bytes",
                                                            "min_cgroup_available_bytes")}
            or ("resources" in checked["offline"] and prior["resources"] != checked["offline"]["resources"])
            or set(protocol.get("training", {})) != set(_ARMS)):
        raise ValueError("prior treatment and frozen 256-only score protocols disagree")
    sources = {**checked["protocol"]["source_sha256"], **prior["source_sha256"]}
    if (any(prior["source_sha256"].get(key) != checked["protocol"]["source_sha256"][key]
            for key in original.SOURCE_PATHS - {"scripts/diagnose/dreamerv3_reused_train_open_loop.py"})
            or set(protocol["source_sha256"]) != set(sources) | {SOURCE}
            or {key: protocol["source_sha256"][key] for key in sources} != sources):
        raise ValueError("auxiliary and 256-only executable source SHA maps disagree")
    for source, digest in protocol["source_sha256"].items():
        budget._fixed_file(root, source, source, digest, cap=1024 * 1024)
    if protocol["source_sha256"][SOURCE] != original._sha256(Path(__file__)):
        raise ValueError("executing prior-image scorer differs from frozen source")
    checked.update(prior=prior, score_protocol=protocol, original=frozen,
                   protocol_path=name, prior_training={})
    _reference(checked)
    for arm in _ARMS:
        rows = protocol["training"][arm]
        if (not isinstance(rows, list) or len(rows) != 2
                or [row.get("seed") for row in rows if isinstance(row, dict)] != [0, 1]):
            raise ValueError(f"{arm}: exactly two prior-image learner seeds required")
        checked["prior_training"][arm] = []
        train = checked["offline"]["datasets"][arm]
        for seed, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"seed", "result_path", "result_sha256",
                                                      "checkpoint_path", "checkpoint_sha256"} or type(row["seed"]) is not int:
                raise ValueError("exact four prior-image result/checkpoint SHA pairs required")
            prefix = f"{TRAIN_ROOT}/{arm}-seed-{seed}"
            result_path = budget._fixed_file(root, row["result_path"], f"{prefix}/training-result.json",
                                             row["result_sha256"], cap=1024 * 1024)
            checkpoint = budget._fixed_file(root, row["checkpoint_path"],
                                             f"{prefix}/world-model-checkpoint.pt", row["checkpoint_sha256"],
                                             cap=checked["protocol"]["resources"]["max_checkpoint_bytes"])
            result = original._json(result_path)
            expected = {
                "format": "haic-dreamerv3-reused-prior-image-result-v1", "status": "complete",
                "purpose": prior["purpose"], "study_id": prior["study_id"], "arm": arm, "seed": seed,
                "prior_protocol_sha256": PRIOR_PROTOCOL_SHA256,
                "base_offline_protocol_path": budget.V2_OFFLINE_PATH,
                "base_offline_protocol_sha256": budget.V2_OFFLINE_SHA256,
                "collection_protocol_sha256": checked["offline"]["collection_protocol"]["sha256"],
                "source_sha256": prior["source_sha256"], "collection_receipt_path": train["receipt_path"],
                "collection_receipt_sha256": train["receipt_sha256"],
                "checkpoint_path": "world-model-checkpoint.pt", "checkpoint_sha256": row["checkpoint_sha256"],
                "environment_steps": 0, "base_model_only_updates": 256,
                "aux_world_model_optimizer_steps": 64, "world_model_optimizer_steps": 320,
                "aux_after_every_base_updates": 4, "prior_horizon": 8, "prior_weight": 0.25,
                "anchor_mode": prior["learner"]["anchor_mode"],
                "aux_rng_mode": prior["learner"]["aux_rng_mode"],
                "aux_rng_seed_base": prior["learner"]["aux_rng_seed_base"],
                "aux_rng_seed_stride": prior["learner"]["aux_rng_seed_stride"],
                "actor_critic_target_unchanged": True, "actor_critic_optimizers_unchanged": True,
                "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                "promotion_eligible": False, "matched_pure_256_step_model": False,
            }
            for key in ("archive_path", "archive_sha256", "dataset_digest", "source_id", "source_actor_sha256"):
                expected[key] = train[key]
            for key, value in expected.items():
                if type(result.get(key)) is not type(value) or result[key] != value:
                    raise ValueError(f"{arm}/{seed}: auxiliary training result {key} mismatch")
            if (type(result.get("collection_decisions_spent")) is not int
                    or not train["stored_decisions"] <= result["collection_decisions_spent"] <= 8000):
                raise ValueError("auxiliary collection decision count mismatch")
            for kind, count in (("base", 256), ("aux", 64)):
                _metrics_stream(root, prefix, result, kind, count)
            memory = original._cgroup()
            if (memory["limit_bytes"] > checked["protocol"]["resources"]["max_cgroup_memory_bytes"]
                    or memory["available_bytes"] < checked["required_available_bytes"]):
                raise ValueError("cgroup lacks pinned checkpoint comparison headroom")
            payload = _aux_payload(checkpoint, arm=arm, seed=seed, result=result,
                                   prior=prior, prior_sha=PRIOR_PROTOCOL_SHA256,
                                   offline=checked["offline"], reference=checked["training"][arm][seed]["checkpoint"])
            del payload
            checked["prior_training"][arm].append({"checkpoint": checkpoint, "row": row, "result": result})
    # Hash both ZIPs without decoding them; all other discrepancies abort before ZIP open.
    for arm in _ARMS:
        for data in (checked["offline"]["datasets"][arm], checked["protocol"]["development"][arm]):
            budget._fixed_file(root, data["archive_path"], data["archive_path"], data["archive_sha256"],
                               cap=checked["protocol"]["resources"]["max_archive_bytes"])
    return checked


def _development(checked: dict[str, Any], arm: str) -> Any:
    dev = checked["checked_v1"]["development"][arm]
    dataset = original._load_dataset(dev["archive"], dev["row"], checked["protocol"]["resources"],
                                     checked["checked_v1"]["development_cells"], original.DEVELOPMENT_ID, arm)
    receipt = dev["receipt"]
    rows = [row for row in receipt["episode_rows"] if row.get("status") == "complete"]
    frozen = checked["original"]["strata"][arm]
    if (dataset.transition_count != receipt["stored_decisions"] or len(dataset.episodes) != 4
            or len(rows) != 4 or len(dataset.episodes) != receipt["complete_episode_count"]):
        raise ValueError("development dataset episode/decision count differs from receipt")
    for index, (episode, row) in enumerate(zip(dataset.episodes, rows, strict=True)):
        reference = frozen["models"][0]["episodes"][index]
        if (type(row.get("episode_id")) is not int or episode.episode_id != str(row["episode_id"])
                or episode.metadata.get("attempt") != row.get("attempt")
                or (episode.track_id, int(episode.geometry_id)) != (row.get("track_id"), row.get("geometry_seed"))
                or episode.steps != row.get("decisions") or bool(episode.terminal[-1]) != row.get("terminal")
                or bool(episode.finished[-1]) != row.get("finished")
                or (episode.episode_id, episode.track_id, int(episode.geometry_id), episode.steps,
                    bool(episode.terminal[-1]), bool(episode.finished[-1])) !=
                   (reference["episode_id"], reference["track_id"], reference["geometry_seed"],
                    reference["decisions"], reference["terminal_event"], reference["finished"])):
            raise ValueError("development episode/road/window differs from frozen score and receipt")
    return dataset


def _gate(strata: dict[str, Any]) -> dict[str, Any]:
    seeds = []
    for seed in (0, 1):
        comparisons = {}
        for source in _ARMS:
            model = strata[source]["models"][2 + seed]
            now = original._finite(model["aggregate"]["metrics_episode_mean"]["image_mse"])
            before = original._finite(model["aggregate"]["original_256_metrics"]["image_mse"])
            comparisons[source] = {"aux_image_mse": now, "original_256_image_mse": before,
                                   "paired_delta_image_mse": original._finite(now - before),
                                   "strict_improvement": now < before}
        below_anchor = comparisons["teacher"]["aux_image_mse"] < ANCHOR_MSE
        seeds.append({"learner_seed": seed, "source_strata": comparisons,
                      "below_teacher_shifted_repeat": below_anchor,
                      "passed": below_anchor and all(value["strict_improvement"] for value in comparisons.values())})
    return {"rule": GATE, "teacher_seeds": seeds, "passed": all(row["passed"] for row in seeds),
            "interpretation": "Local engineering stop rule on consumed reused TRAIN; not fresh P1b, "
                              "official performance, actor improvement or causal loss-shape proof (320 vs 256 steps)."}


def score(protocol_path: Path, protocol_sha256: str, output_dir: Path,
          *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, output_dir, repo_root=repo_root)
    resources, root = checked["protocol"]["resources"], checked["root"]
    memory = original._cgroup()
    if (memory["limit_bytes"] > resources["max_cgroup_memory_bytes"]
            or memory["available_bytes"] < checked["required_available_bytes"]):
        raise ValueError("cgroup lacks archive-load headroom")
    torch.set_num_threads(1)
    # Validate BOTH sealed source strata against receipts and the frozen score before scoring either.
    datasets = {arm: _development(checked, arm) for arm in _ARMS}
    report = {
        "format": FORMAT, "purpose": PURPOSE, "status": "iterative_tuning_descriptive_only",
        "score_protocol_sha256": protocol_sha256, "prior_protocol_sha256": PRIOR_PROTOCOL_SHA256,
        "budget_score_protocol_sha256": BUDGET_PROTOCOL_SHA256,
        "budget_score_result_sha256": BUDGET_RESULT_SHA256,
        "source_sha256": checked["score_protocol"]["source_sha256"],
        "sampling": checked["protocol"]["scoring"], "fresh_claim": False,
        "p1b_claim": False, "promotion_eligible": False, "student_actor_trained": False,
        "interpretation": "Same previously consumed reused TRAIN development roads and seeded 8+32 windows. "
                          "256 base plus 64 extra optimizer steps versus original 256-only models; descriptive "
                          "proxy differences do not establish a loss-shape cause, fresh P1b, or policy performance.",
        "strata": {},
    }
    with torch.no_grad():
        for source in _ARMS:
            frozen = checked["original"]["strata"][source]
            stratum = {key: value for key, value in frozen.items() if key != "models"}
            stratum["models"] = []
            for arm in _ARMS:
                for seed, entry in enumerate(checked["prior_training"][arm]):
                    row = entry["row"]
                    budget._fixed_file(root, row["checkpoint_path"], row["checkpoint_path"],
                                       row["checkpoint_sha256"], cap=resources["max_checkpoint_bytes"])
                    reference_row = checked["protocol"]["training"][arm][seed]
                    budget._fixed_file(root, reference_row["checkpoint_path"], reference_row["checkpoint_path"],
                                       reference_row["checkpoint_sha256"], cap=resources["max_checkpoint_bytes"])
                    payload = _aux_payload(entry["checkpoint"], arm=arm, seed=seed, result=entry["result"],
                                           prior=checked["prior"], prior_sha=PRIOR_PROTOCOL_SHA256,
                                           offline=checked["offline"],
                                           reference=checked["training"][arm][seed]["checkpoint"])
                    modules = _modules(payload, checked["offline"]["learner"]["config"])
                    del payload
                    reference = frozen["models"][(0 if arm == "random" else 2) + seed]
                    baseline = reference["training_baselines"]
                    episodes = [original._score_episode(ep, modules, baseline["constant_reward"],
                                                        baseline["terminal_prevalence"],
                                                        checked["protocol"]["scoring"]["latent_seed"]
                                                        + (source == "teacher") * 1_000_003 + index * 1009 + seed)
                                for index, ep in enumerate(datasets[source].episodes)]
                    del modules
                    roads: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
                    for episode in episodes:
                        roads[(episode["track_id"], episode["geometry_seed"])].append(episode)
                    model = {"model_arm": arm, "learner_seed": seed,
                             "checkpoint_sha256": row["checkpoint_sha256"],
                             "original_256_checkpoint_sha256": reference["checkpoint_sha256"],
                             "training_baselines": baseline, "aggregate": original._aggregate(episodes),
                             "roads": [{"track_id": track, "geometry_seed": road, **original._aggregate(items)}
                                       for (track, road), items in sorted(roads.items())],
                             "episodes": episodes}
                    _pair(model, reference)
                    stratum["models"].append(model)
            report["strata"][source] = stratum
    report["local_engineering_gate"] = _gate(report["strata"])
    # Recheck all primary bytes before exclusive output creation; no success receipt on drift.
    budget._fixed_file(root, checked["protocol_path"], SCORE_PROTOCOL_PATH, protocol_sha256, cap=1024 * 1024)
    for name, digest in checked["score_protocol"]["source_sha256"].items():
        budget._fixed_file(root, name, name, digest, cap=1024 * 1024)
    for ref in (checked["score_protocol"]["prior_protocol"],
                checked["score_protocol"]["budget_score_protocol"],
                checked["score_protocol"]["budget_score_result"],
                checked["protocol"]["offline_protocol"],
                checked["protocol"]["original_score_protocol"],
                checked["protocol"]["original_score_result"],
                checked["protocol"]["development_collection_protocol"],
                checked["offline"]["collection_protocol"],
                checked["checked_v1"]["training_collection"]["r6_protocol"]):
        budget._fixed_file(root, ref["path"], ref["path"], ref["sha256"], cap=1024 * 1024)
    for arm in _ARMS:
        for original_rows in (checked["protocol"]["training"][arm],
                              checked["checked_v1"]["protocol"]["training"][arm]):
            for row in original_rows:
                for key, cap in (("result", 1024 * 1024), ("checkpoint", resources["max_checkpoint_bytes"])):
                    budget._fixed_file(root, row[f"{key}_path"], row[f"{key}_path"], row[f"{key}_sha256"], cap=cap)
        for entry in checked["prior_training"][arm]:
            row, result = entry["row"], entry["result"]
            for key, cap in (("result", 1024 * 1024), ("checkpoint", resources["max_checkpoint_bytes"])):
                budget._fixed_file(root, row[f"{key}_path"], row[f"{key}_path"], row[f"{key}_sha256"], cap=cap)
            prefix = f"{TRAIN_ROOT}/{arm}-seed-{row['seed']}"
            for kind, count in (("base", 256), ("aux", 64)):
                _metrics_stream(root, prefix, result, kind, count)
        for data in (checked["offline"]["datasets"][arm], checked["protocol"]["development"][arm]):
            budget._fixed_file(root, data["archive_path"], data["archive_path"], data["archive_sha256"],
                               cap=resources["max_archive_bytes"])
        for data in (checked["offline"]["datasets"][arm], checked["protocol"]["development"][arm]):
            budget._fixed_file(root, data["receipt_path"], data["receipt_path"], data["receipt_sha256"],
                               cap=1024 * 1024)
    memory = original._cgroup()
    if (memory["limit_bytes"] > resources["max_cgroup_memory_bytes"]
            or memory["available_bytes"] < checked["required_available_bytes"]):
        raise ValueError("cgroup lacks write-time headroom")
    output = original._path(root, original._relative(root, checked["output"]), "output", exists=False)
    output.mkdir(exist_ok=False)
    with (output / "score-result.json").open("x", encoding="utf-8") as stream:
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
    report = score(args.protocol, args.protocol_sha256, args.output, repo_root=args.repo_root)
    print(json.dumps({"status": report["status"], "local_engineering_gate_passed":
                      report["local_engineering_gate"]["passed"],
                      "output": str(args.output / "score-result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
