"""Read-only action-input sensitivity on consumed reused-TRAIN development windows.

Run from the repository root with ``python -m scripts.diagnose.dreamerv3_reused_action_probe``.
The changed actions are inputs to a frozen world model, NOT environment actions or
counterfactual outcomes. No policy, learner, environment, or protected cell is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.diagnose import dreamerv3_reused_train_budget_probe as budget
from scripts.diagnose import dreamerv3_reused_train_open_loop as original


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "haic-dreamerv3-reused-train-action-input-probe-v1"
SOURCE = "scripts/diagnose/dreamerv3_reused_action_probe.py"
V2_SCORE_PROTOCOL = "experiments/dreamerv3-reused-train-budget-probe-v1.json"
V2_SCORE_PROTOCOL_SHA256 = "ddcb33d19b36f6c0c17942263da835a45bf87f76271f550027e6113824ebf994"
V2_SCORE_RESULT = f"{budget.RUN_ROOT}/scoring-v2/score-result.json"
V2_SCORE_RESULT_SHA256 = "94bbd7457283d5ec8f0ad4a4ff2e654b97839d09a7eca764615de6fae625ff5d"
OUTPUT = f"{budget.RUN_ROOT}/action-sensitivity-v1"
SHIFT = 11  # np.roll: action at future offset t comes from (t - 11) mod 32.
ERRORS = ("image_mse", "reward_mse", "terminal_bce")
DIFFERENCES = ("image_prediction_mse", "reward_prediction_abs", "terminal_probability_abs")
CONDITIONS = ("logged", "shifted", "zero")
# Fixed engineering display heuristic, not a pass gate or a threshold fit on development data.
SMALL_RESPONSE = {"image_prediction_mse": 1e-6, "reward_prediction_abs": 0.01,
                  "terminal_probability_abs": 0.01}


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty prediction group")
    return original._finite(sum(values) / len(values))


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = {}
    for condition in CONDITIONS:
        items = [row["conditions"][condition] for row in rows]
        conditions[condition] = {
            "errors": {key: _mean([item["errors"][key] for item in items]) for key in ERRORS},
            "delta_error_vs_logged": {
                key: _mean([item["delta_error_vs_logged"][key] for item in items]) for key in ERRORS
            },
            "prediction_diff_vs_logged": {
                key: _mean([item["prediction_diff_vs_logged"][key] for item in items]) for key in DIFFERENCES
            },
            "changed_native_action_uses": sum(item["changed_native_action_uses"] for item in items),
            "native_action_uses": sum(item["native_action_uses"] for item in items),
        }
    informative = all(conditions[name]["changed_native_action_uses"] > 0 for name in ("shifted", "zero"))
    return {"conditions": conditions, "action_insensitive_small_response": (
        all(conditions[name]["prediction_diff_vs_logged"][metric] <= ceiling
            for name in ("shifted", "zero") for metric, ceiling in SMALL_RESPONSE.items())
        if informative else None
    )}


def _episode(episode: Any, modules: tuple[Any, ...], latent_seed: int,
             reference: dict[str, Any]) -> dict[str, Any]:
    """Use one observed reset-origin prefix, then replay three future inputs per anchor."""
    length = episode.steps
    if (length < 41 or episode.actions.shape != (length, 3)
            or not np.isfinite(episode.actions).all() or np.any(np.abs(episode.actions) > 1.0)):
        raise ValueError("scored episode requires bounded native actions and two 8+32 windows")
    anchors = (("reset", 8), ("terminal", length - 32))
    if (reference.get("episode_id") != episode.episode_id or reference.get("decisions") != length
            or (reference.get("track_id"), reference.get("geometry_seed"))
            != (episode.track_id, int(episode.geometry_id))
            or len(reference.get("windows", [])) != 2):
        raise ValueError("episode differs from pinned score identity")
    encoder, rssm, decoder, reward_head, continue_head = modules
    result = {"episode_id": episode.episode_id, "track_id": episode.track_id,
              "geometry_seed": int(episode.geometry_id), "decisions": length,
              "terminal_event": bool(episode.terminal[-1]), "finished": bool(episode.finished[-1]),
              "windows": []}
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(latent_seed)
        max_anchor = anchors[-1][1]
        embeds = []
        for start in range(0, max_anchor + 1, 32):
            stop = min(start + 32, max_anchor + 1)
            stacks = np.stack([episode.observation(i) for i in range(start, stop)])
            embeds.append(encoder(torch.from_numpy(stacks)).unsqueeze(0))
        embeddings = torch.cat(embeds, dim=1)
        context_actions = torch.from_numpy(episode.actions[:max_anchor].copy()).unsqueeze(0)
        first = torch.zeros((1, max_anchor), dtype=torch.bool)
        first[:, 0] = True
        states, _, _ = rssm.observe_sequence(embeddings, context_actions, first)
        for window_index, (kind, anchor) in enumerate(anchors):
            old = reference["windows"][window_index]
            if (old.get("kind") != kind or old.get("context_anchor_decision") != anchor
                    or old.get("reset_origin_prefix_decisions") != anchor
                    or old.get("first_target_decision") != anchor
                    or old.get("last_target_decision") != anchor + 31
                    or old.get("label_uses") != 32):
                raise ValueError("window differs from pinned 8+32 score")
            native = episode.actions[anchor:anchor + 32].copy()
            variants = {"logged": native, "shifted": np.roll(native, SHIFT, axis=0).copy(),
                        "zero": np.zeros_like(native)}
            rng_start = torch.random.get_rng_state()
            raw: dict[str, dict[str, Any]] = {}
            for name, actions in variants.items():
                torch.random.set_rng_state(rng_start)
                state = states[:, anchor].clone()
                stack = torch.from_numpy(episode.observation(anchor).copy()).unsqueeze(0)
                errors: dict[str, list[float]] = {key: [] for key in ERRORS}
                predictions: list[tuple[torch.Tensor, float, float]] = []
                for offset, action in enumerate(actions):
                    h, z = torch.split(state, [rssm.hidden_dim, rssm.stoch_dim], dim=-1)
                    next_h, next_z, _, _ = rssm.step_prior(h, z, torch.from_numpy(action.copy()).unsqueeze(0))
                    state = torch.cat((next_h, next_z), dim=-1)
                    latest = (stack[:, -1:] + decoder(state)).clamp(0.0, 1.0)
                    stack = torch.cat((stack[:, 1:], latest), dim=1)
                    target = torch.from_numpy(episode.observation(anchor + offset + 1)[-1].copy()).unsqueeze(0)
                    predicted_reward = original._finite(reward_head.pred(state).item())
                    terminal_probability = min(max(original._finite(
                        1.0 - torch.sigmoid(continue_head(state)).item()), 1e-7), 1 - 1e-7)
                    terminal = bool(episode.terminal[anchor + offset])
                    errors["image_mse"].append(original._finite(torch.mean((latest[:, -1] - target) ** 2).item()))
                    errors["reward_mse"].append(original._finite(
                        (predicted_reward - original._finite(episode.rewards[anchor + offset])) ** 2))
                    errors["terminal_bce"].append(original._finite(-math.log(
                        terminal_probability if terminal else 1 - terminal_probability)))
                    predictions.append((latest[:, -1].clone(), predicted_reward, terminal_probability))
                if name == "logged":
                    rng_after_logged = torch.random.get_rng_state()
                raw[name] = {"errors": {key: _mean(values) for key, values in errors.items()},
                             "predictions": predictions}
            torch.random.set_rng_state(rng_after_logged)
            for key in ERRORS:
                before = old["metrics"].get(key)
                if (type(before) not in (int, float) or not math.isfinite(before)
                        or not math.isclose(raw["logged"]["errors"][key], before,
                                            rel_tol=1e-6, abs_tol=1e-7)):
                    raise ValueError(f"logged {key} differs from pinned score window")
            window = {"kind": kind, "context_anchor_decision": anchor,
                      "reset_origin_prefix_decisions": anchor, "first_target_decision": anchor,
                      "last_target_decision": anchor + 31, "label_uses": 32,
                      "terminal_positive_label_uses": sum(bool(episode.terminal[i])
                                                          for i in range(anchor, anchor + 32)),
                      "conditions": {}}
            if window["terminal_positive_label_uses"] != old.get("terminal_positive_label_uses"):
                raise ValueError("terminal labels differ from pinned score")
            for name in CONDITIONS:
                comparisons = list(zip(raw[name]["predictions"], raw["logged"]["predictions"], strict=True))
                difference = {
                    "image_prediction_mse": _mean([original._finite(torch.mean((now[0] - old[0]) ** 2).item())
                                                   for now, old in comparisons]),
                    "reward_prediction_abs": _mean([original._finite(abs(now[1] - old[1]))
                                                    for now, old in comparisons]),
                    "terminal_probability_abs": _mean([original._finite(abs(now[2] - old[2]))
                                                        for now, old in comparisons]),
                }
                window["conditions"][name] = {
                    "errors": raw[name]["errors"],
                    "delta_error_vs_logged": {key: original._finite(raw[name]["errors"][key]
                                                               - raw["logged"]["errors"][key]) for key in ERRORS},
                    "prediction_diff_vs_logged": difference,
                    "changed_native_action_uses": int(np.count_nonzero(np.any(variants[name] != native, axis=1))),
                    "native_action_uses": 32,
                }
            window.update(_condition_summary([window]))
            result["windows"].append(window)
    scored = set(range(8, 40)) | set(range(length - 32, length))
    result.update(unique_scored_decisions=len(scored), duplicate_label_uses=64 - len(scored),
                  unique_terminal_positive_labels=sum(bool(episode.terminal[i]) for i in scored))
    if (result["terminal_event"] != reference.get("terminal_event")
            or result["finished"] != reference.get("finished")
            or result["unique_scored_decisions"] != reference.get("unique_scored_decisions")
            or result["duplicate_label_uses"] != reference.get("duplicate_label_uses")
            or result["unique_terminal_positive_labels"] != reference.get("unique_terminal_positive_labels")):
        raise ValueError("episode label denominators differ from pinned score")
    result.update(_condition_summary(result["windows"]))
    return result


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("no independent scored episodes")
    result = {"independent_episode_count": len(episodes),
              "independent_road_count": len({(row["track_id"], row["geometry_seed"]) for row in episodes}),
              "independent_terminal_episode_count": sum(row["terminal_event"] for row in episodes),
              "window_count": sum(len(row["windows"]) for row in episodes),
              "window_label_uses": sum(32 * len(row["windows"]) for row in episodes),
              "unique_scored_decisions": sum(row["unique_scored_decisions"] for row in episodes),
              "duplicate_label_uses": sum(row["duplicate_label_uses"] for row in episodes),
              "terminal_positive_label_uses": sum(sum(w["terminal_positive_label_uses"] for w in row["windows"])
                                                  for row in episodes),
              "unique_terminal_positive_labels": sum(row["unique_terminal_positive_labels"] for row in episodes)}
    result.update(_condition_summary(episodes))
    return result


def _model_mean(models: list[dict[str, Any]]) -> dict[str, Any]:
    # An average over models does not multiply the number of independent episodes.
    denominators = {key: value for key, value in models[0].items()
                    if key not in ("conditions", "action_insensitive_small_response")}
    if any({key: item[key] for key in denominators} != denominators for item in models[1:]):
        raise ValueError("model groups use different episode/label denominators")
    result = {"model_count": len(models), **denominators}
    summary = _condition_summary(models)
    # Per-model native action uses repeat the same episode labels; keep one copy.
    for name in CONDITIONS:
        summary["conditions"][name]["native_action_uses"] = models[0]["conditions"][name]["native_action_uses"]
        summary["conditions"][name]["changed_native_action_uses"] = models[0]["conditions"][name]["changed_native_action_uses"]
    result.update(summary)
    return result


def preflight(output_dir: Path, source_sha256: str, *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Validate fixed inputs, both score results and cgroup before opening a dataset ZIP."""
    root = Path(repo_root).resolve()
    if original._relative(root, output_dir) != OUTPUT:
        raise ValueError("output must be the new fixed action-sensitivity sibling directory")
    budget._fixed_file(root, SOURCE, SOURCE, source_sha256, cap=1024 * 1024)
    if original._sha256(Path(__file__)) != source_sha256:
        raise ValueError("executing action probe differs from pinned source")
    checked = budget.preflight(root / V2_SCORE_PROTOCOL, V2_SCORE_PROTOCOL_SHA256,
                               root / OUTPUT, repo_root=root)
    protocol = checked["protocol"]
    v2_result = budget._ref(root, {"path": V2_SCORE_RESULT, "sha256": V2_SCORE_RESULT_SHA256},
                            V2_SCORE_RESULT, V2_SCORE_RESULT_SHA256)
    if (v2_result.get("format") != budget.FORMAT or v2_result.get("purpose") != budget.PURPOSE
            or v2_result.get("status") != "iterative_tuning_descriptive_only"
            or v2_result.get("score_protocol_sha256") != V2_SCORE_PROTOCOL_SHA256
            or v2_result.get("original_score_protocol_sha256") != budget.ORIGINAL_SCORE_PROTOCOL_SHA256
            or v2_result.get("original_score_result_sha256") != budget.ORIGINAL_SCORE_RESULT_SHA256
            or v2_result.get("offline_protocol_sha256") != budget.V2_OFFLINE_SHA256
            or v2_result.get("development_collection_protocol_sha256")
            != protocol["development_collection_protocol"]["sha256"]
            or v2_result.get("source_sha256") != protocol["source_sha256"]
            or v2_result.get("sampling") != protocol["scoring"]
            or any(v2_result.get(key) is not False for key in (
                "fresh_claim", "p1b_claim", "promotion_eligible", "student_actor_trained"))
            or set(v2_result.get("strata", {})) != {"random", "teacher"}):
        raise ValueError("v2 pinned score result identity/status mismatch")
    for source_arm in ("random", "teacher"):
        v1_stratum = checked["original"]["strata"][source_arm]
        v2_stratum = v2_result["strata"][source_arm]
        if ({key: value for key, value in v2_stratum.items() if key != "models"}
                != {key: value for key, value in v1_stratum.items() if key != "models"}
                or len(v2_stratum.get("models", [])) != 4):
            raise ValueError("v2 development stratum or denominators differ from v1")
        for index, model in enumerate(v2_stratum["models"]):
            arm, seed = ("random", "teacher")[index // 2], index % 2
            if (model.get("model_arm") != arm or type(model.get("learner_seed")) is not int
                    or model["learner_seed"] != seed
                    or model.get("checkpoint_sha256") != checked["training"][arm][seed]["sha256"]
                    or v1_stratum["models"][index].get("checkpoint_sha256")
                    != checked["checked_v1"]["training"][arm][seed]["checkpoint_sha256"]):
                raise ValueError("v2 model/checkpoint identity differs from pinned budget protocol")
            paired = copy.deepcopy(model)
            budget._pair(paired, v1_stratum["models"][index])
            for section in ("aggregate", "roads", "episodes"):
                left = [model[section]] if section == "aggregate" else model[section]
                right = [paired[section]] if section == "aggregate" else paired[section]
                for a, b in zip(left, right, strict=True):
                    if a.get("paired_delta_vs_64") != b["paired_delta_vs_64"]:
                        raise ValueError("v2 paired score deltas differ from pinned v1 result")
                    if section == "episodes":
                        for window, expected in zip(a["windows"], b["windows"], strict=True):
                            if window.get("paired_delta_vs_64") != expected["paired_delta_vs_64"]:
                                raise ValueError("v2 paired window deltas differ from pinned v1 result")
    return {**checked, "v2_result": v2_result, "source_sha256": source_sha256}


def score(output_dir: Path, source_sha256: str, *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(output_dir, source_sha256, repo_root=repo_root)
    root, protocol = checked["root"], checked["protocol"]
    resources = protocol["resources"]
    memory = original._cgroup()
    if (memory["limit_bytes"] > resources["max_cgroup_memory_bytes"]
            or memory["available_bytes"] < checked["required_available_bytes"]):
        raise ValueError("cgroup lacks frozen archive-load headroom")
    torch.set_num_threads(1)
    datasets = {}
    for source_arm in ("random", "teacher"):
        dev = checked["checked_v1"]["development"][source_arm]
        dataset = original._load_dataset(dev["archive"], dev["row"], resources,
                                         checked["checked_v1"]["development_cells"],
                                         original.DEVELOPMENT_ID, source_arm)
        receipt = dev["receipt"]
        rows = [row for row in receipt["episode_rows"] if row.get("status") == "complete"]
        v2_stratum = checked["v2_result"]["strata"][source_arm]
        if (dataset.transition_count != receipt["stored_decisions"]
                or len(dataset.episodes) != receipt["complete_episode_count"]
                or len(dataset.episodes) != len(rows)
                or len(dataset.episodes) != v2_stratum["collected_complete_episodes"]):
            raise ValueError("sealed development archive counts differ from score/receipt")
        for episode, row in zip(dataset.episodes, rows, strict=True):
            if (type(row.get("episode_id")) is not int or episode.episode_id != str(row["episode_id"])
                    or episode.metadata.get("attempt") != row.get("attempt")
                    or (episode.track_id, int(episode.geometry_id)) != (row.get("track_id"), row.get("geometry_seed"))
                    or episode.steps != row.get("decisions") or bool(episode.terminal[-1]) != row.get("terminal")
                    or bool(episode.finished[-1]) != row.get("finished") or episode.steps < 41):
                raise ValueError("development episode differs from its pinned receipt/windows")
        if (sum(bool(ep.terminal[-1]) for ep in dataset.episodes)
                != v2_stratum["scored_independent_terminal_episodes"]
                or len({(ep.track_id, ep.geometry_id) for ep in dataset.episodes})
                != v2_stratum["scored_independent_geometries"]):
            raise ValueError("development episode/road coverage differs from pinned scores")
        datasets[source_arm] = dataset

    report: dict[str, Any] = {
        "format": FORMAT, "status": "iterative_tuning_descriptive_only",
        "source_sha256": {**protocol["source_sha256"], SOURCE: source_sha256},
        "pinned_inputs_sha256": {
            "v1_score_protocol": budget.ORIGINAL_SCORE_PROTOCOL_SHA256,
            "v1_score_result": budget.ORIGINAL_SCORE_RESULT_SHA256,
            "v2_score_protocol": V2_SCORE_PROTOCOL_SHA256,
            "v2_score_result": V2_SCORE_RESULT_SHA256,
            "v1_offline_protocol": original.OFFLINE_SHA256, "v2_offline_protocol": budget.V2_OFFLINE_SHA256,
            "development_collection_protocol": protocol["development_collection_protocol"]["sha256"],
        },
        "sampling": {"context_decisions": 8, "future_decisions": 32, "shift_roll": SHIFT,
                     "context_mode": "reset-origin-full-prefix-terminal",
                     "latent_seed": protocol["scoring"]["latent_seed"],
                     "conditions": list(CONDITIONS), "small_response_heuristic": SMALL_RESPONSE},
        "fresh_claim": False, "p1b_claim": False, "promotion_eligible": False,
        "student_actor_trained": False,
        "interpretation": "Previously scored, consumed iterative-tuning reused TRAIN episodes. Only future "
                          "native action inputs differ within the same observed context and latent RNG. Fixed "
                          "logged next-frame/reward/terminal labels under mismatch are NOT counterfactual outcomes; "
                          "higher mismatched-label error is model-input sensitivity, not safe action ranking. "
                          "Small-response flags are fixed descriptive heuristics, not a fitted threshold or "
                          "promotion gate. Windows and models on one road do not add independent episodes. "
                          "No simulator, updates, fresh holdout, blind, confirmation or official inference.",
        "strata": {},
    }
    for source_arm in ("random", "teacher"):
        dataset = datasets[source_arm]
        stratum = {"development_archive_sha256": checked["checked_v1"]["development"][source_arm]["row"]["archive_sha256"],
                   "source_id": checked["checked_v1"]["development"][source_arm]["row"]["source_id"],
                   "models": []}
        for updates in (64, 256):
            for arm in ("random", "teacher"):
                for seed in (0, 1):
                    old = (checked["original"] if updates == 64 else checked["v2_result"])["strata"][source_arm]["models"][2 * (arm == "teacher") + seed]
                    row = (checked["checked_v1"]["training"][arm][seed]
                           if updates == 64 else checked["training"][arm][seed])
                    checkpoint = row["checkpoint"]
                    digest = row["checkpoint_sha256"] if updates == 64 else row["sha256"]
                    budget._fixed_file(root, original._relative(root, checkpoint),
                                       original._relative(root, checkpoint), digest,
                                       cap=resources["max_checkpoint_bytes"])
                    if updates == 64:
                        modules = original._checkpoint_modules(
                            checkpoint, arm=arm, seed=seed,
                            dataset_digest=checked["checked_v1"]["offline"]["datasets"][arm]["dataset_digest"],
                            config=checked["checked_v1"]["offline"]["learner"]["config"])
                    else:
                        modules = budget._modules(checkpoint, arm=arm, seed=seed,
                                                  offline_sha=budget.V2_OFFLINE_SHA256,
                                                  dataset_digest=checked["offline"]["datasets"][arm]["dataset_digest"],
                                                  config=checked["offline"]["learner"]["config"])
                    episodes = [_episode(ep, modules,
                                         protocol["scoring"]["latent_seed"] + (source_arm == "teacher") * 1_000_003
                                         + index * 1009 + seed, old["episodes"][index])
                                for index, ep in enumerate(dataset.episodes)]
                    del modules
                    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
                    for ep in episodes:
                        grouped[(ep["track_id"], ep["geometry_seed"])].append(ep)
                    aggregate = _aggregate(episodes)
                    if (old["aggregate"]["episode_count"] != aggregate["independent_episode_count"]
                            or old["aggregate"]["window_label_uses"] != aggregate["window_label_uses"]
                            or old["aggregate"]["unique_scored_decisions"] != aggregate["unique_scored_decisions"]
                            or old["aggregate"]["terminal_positive_label_uses"] != aggregate["terminal_positive_label_uses"]):
                        raise ValueError("aggregate label/episode denominators differ from pinned score")
                    stratum["models"].append({"model_arm": arm, "learner_seed": seed, "updates": updates,
                                              "checkpoint_sha256": digest, "aggregate": aggregate,
                                              "roads": [{"track_id": track, "geometry_seed": road, **_aggregate(rows)}
                                                        for (track, road), rows in sorted(grouped.items())],
                                              "episodes": episodes})
        stratum["descriptive_model_mean"] = _model_mean([model["aggregate"] for model in stratum["models"]])
        by_road: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for model in stratum["models"]:
            for road in model["roads"]:
                by_road[(road["track_id"], road["geometry_seed"])].append(
                    {key: value for key, value in road.items() if key not in ("track_id", "geometry_seed")})
        stratum["roads_descriptive_model_mean"] = [
            {"track_id": track, "geometry_seed": road, **_model_mean(rows)}
            for (track, road), rows in sorted(by_road.items())]
        report["strata"][source_arm] = stratum

    # Recheck all input pins after computation, before atomically claiming the new directory.
    budget._fixed_file(root, SOURCE, SOURCE, source_sha256, cap=1024 * 1024)
    if original._sha256(Path(__file__)) != source_sha256:
        raise ValueError("executing action probe changed while scoring")
    budget._fixed_file(root, V2_SCORE_PROTOCOL, V2_SCORE_PROTOCOL, V2_SCORE_PROTOCOL_SHA256, cap=1024 * 1024)
    for name, digest in ((budget.ORIGINAL_SCORE_PROTOCOL_PATH, budget.ORIGINAL_SCORE_PROTOCOL_SHA256),
                         (budget.ORIGINAL_SCORE_RESULT_PATH, budget.ORIGINAL_SCORE_RESULT_SHA256),
                         (V2_SCORE_RESULT, V2_SCORE_RESULT_SHA256),
                         (original.OFFLINE_PATH, original.OFFLINE_SHA256),
                         (budget.V2_OFFLINE_PATH, budget.V2_OFFLINE_SHA256)):
        budget._fixed_file(root, name, name, digest, cap=1024 * 1024)
    for ref in (checked["checked_v1"]["offline"]["collection_protocol"],
                protocol["development_collection_protocol"],
                checked["checked_v1"]["training_collection"]["r6_protocol"]):
        budget._fixed_file(root, ref["path"], ref["path"], ref["sha256"], cap=1024 * 1024)
    for name, digest in protocol["source_sha256"].items():
        budget._fixed_file(root, name, name, digest, cap=1024 * 1024)
    for arm in ("random", "teacher"):
        for rows in (checked["checked_v1"]["protocol"]["training"][arm], protocol["training"][arm]):
            for row in rows:
                for kind, cap in (("result", 1024 * 1024), ("checkpoint", resources["max_checkpoint_bytes"])):
                    budget._fixed_file(root, row[f"{kind}_path"], row[f"{kind}_path"],
                                       row[f"{kind}_sha256"], cap=cap)
        for item in (checked["checked_v1"]["offline"]["datasets"][arm], protocol["development"][arm]):
            for kind, cap in (("receipt", 1024 * 1024), ("archive", resources["max_archive_bytes"])):
                budget._fixed_file(root, item[f"{kind}_path"], item[f"{kind}_path"],
                                   item[f"{kind}_sha256"], cap=cap)
    original._path(root, OUTPUT, "output", exists=False)
    (root / OUTPUT).mkdir(exist_ok=False)
    with (root / OUTPUT / "action-result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True,
                        help="SHA-256 of this frozen executable, externally recorded before any real archive read")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = score(args.output, args.source_sha256, repo_root=args.repo_root)
    print(json.dumps({"status": result["status"], "output": str(args.output / "action-result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
