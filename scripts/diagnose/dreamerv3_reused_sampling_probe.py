"""Read-only K=8 prior-sampling diagnosis on consumed reused-TRAIN development episodes.

Run from the repository root with ``python -m scripts.diagnose.dreamerv3_reused_sampling_probe``.
This is iterative tuning on already scored roads, not a fresh holdout or policy gate.
Freeze this file's SHA-256 externally before passing --source-sha256 for a real score.
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
FORMAT = "haic-dreamerv3-reused-train-sampling-probe-v1"
SOURCE = "scripts/diagnose/dreamerv3_reused_sampling_probe.py"
SCORE_PROTOCOL = "experiments/dreamerv3-reused-train-budget-probe-v1.json"
SCORE_PROTOCOL_SHA256 = "ddcb33d19b36f6c0c17942263da835a45bf87f76271f550027e6113824ebf994"
SCORE_RESULT = f"{budget.RUN_ROOT}/scoring-v2/score-result.json"
SCORE_RESULT_SHA256 = "94bbd7457283d5ec8f0ad4a4ff2e654b97839d09a7eca764615de6fae625ff5d"
OUTPUT = f"{budget.RUN_ROOT}/sampling-v1"
K = 8
HORIZONS = {"t1": (0, 1), "t1_8": (0, 8), "t9_32": (8, 32)}


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty scored window group")
    return original._finite(sum(values) / len(values))


def _distribution(path_errors: list[float]) -> dict[str, Any]:
    if len(path_errors) != K:
        raise ValueError("exactly eight prior paths required")
    mean = _mean(path_errors)
    return {"sampled_path_mse": path_errors, "sampled_mse_mean": mean,
            "sampled_mse_std": original._finite(math.sqrt(
                sum((item - mean) ** 2 for item in path_errors) / K))}


def _close(actual: float, expected: Any, description: str) -> None:
    if (type(expected) not in (float, int) or not math.isfinite(expected)
            or not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7)):
        raise ValueError(f"{description} differs from pinned single-prior score")


def _window(episode: Any, rssm: Any, decoder: Any, state: torch.Tensor,
            anchor: int, kind: str, reference: dict[str, Any]) -> dict[str, Any]:
    if (reference.get("kind") != kind or reference.get("context_anchor_decision") != anchor
            or reference.get("reset_origin_prefix_decisions") != anchor
            or reference.get("first_target_decision") != anchor
            or reference.get("last_target_decision") != anchor + 31
            or reference.get("label_uses") != 32):
        raise ValueError("window differs from pinned reset/terminal anchor")
    stack_at_anchor = torch.from_numpy(episode.observation(anchor).copy()).unsqueeze(0)
    targets = [torch.from_numpy(episode.observation(anchor + t + 1)[-1].copy()).unsqueeze(0)
               for t in range(32)]
    actions = torch.from_numpy(episode.actions[anchor:anchor + 32].copy())
    repeat = stack_at_anchor.clone()
    repeat_errors = []
    for target in targets:
        repeat = torch.cat((repeat[:, 1:], repeat[:, -1:]), dim=1)
        repeat_errors.append(original._finite(torch.mean((repeat[:, -1] - target) ** 2).item()))

    # The first path consumes exactly the original scorer's prior draws. Restore
    # its post-path RNG for the next anchor; seven extra paths must not shift it.
    errors = np.empty((K, 32), dtype=np.float64)
    summed_images = [torch.zeros_like(target) for target in targets]
    for sample in range(K):
        prior_state = state.clone()
        stack = stack_at_anchor.clone()
        for offset, (action, target) in enumerate(zip(actions, targets, strict=True)):
            h, z = torch.split(prior_state, [rssm.hidden_dim, rssm.stoch_dim], dim=-1)
            next_h, next_z, _, _ = rssm.step_prior(h, z, action.unsqueeze(0))
            prior_state = torch.cat((next_h, next_z), dim=-1)
            latest = (stack[:, -1:] + decoder(prior_state)).clamp(0.0, 1.0)
            stack = torch.cat((stack[:, 1:], latest), dim=1)
            predicted = latest[:, -1]
            errors[sample, offset] = original._finite(torch.mean((predicted - target) ** 2).item())
            summed_images[offset].add_(predicted)
        if sample == 0:
            after_original_path = torch.random.get_rng_state()
    torch.random.set_rng_state(after_original_path)

    predictive_errors = [original._finite(torch.mean((image / K - target) ** 2).item())
                         for image, target in zip(summed_images, targets, strict=True)]
    _close(_mean(errors[0].tolist()), reference["metrics"].get("image_mse"), "one sampled prior")
    _close(_mean(repeat_errors), reference["metrics"].get("shifted_repeat_mse"), "shifted repeat")
    positive = sum(bool(episode.terminal[anchor + index]) for index in range(32))
    if positive != reference.get("terminal_positive_label_uses"):
        raise ValueError("terminal labels differ from pinned window")
    horizons = {}
    for name, (start, stop) in HORIZONS.items():
        paths = [original._finite(float(errors[i, start:stop].mean())) for i in range(K)]
        horizons[name] = {"label_uses": stop - start, "unique_scored_decisions": stop - start,
                          "duplicate_label_uses": 0,
                          "metrics": {
                              "single_prior_mse": _mean(errors[0, start:stop].tolist()),
                              "predictive_mean_mse": _mean(predictive_errors[start:stop]),
                              **_distribution(paths),
                              "shifted_repeat_mse": _mean(repeat_errors[start:stop]),
                          }}
    return {"kind": kind, "context_anchor_decision": anchor,
            "reset_origin_prefix_decisions": anchor, "first_target_decision": anchor,
            "last_target_decision": anchor + 31, "label_uses": 32,
            "terminal_positive_label_uses": positive, "horizons": horizons,
            "per_step": [{"target_decision": anchor + offset,
                          "target_observation": anchor + offset + 1,
                          "single_prior_mse": original._finite(float(errors[0, offset])),
                          "predictive_mean_mse": predictive_errors[offset],
                          "shifted_repeat_mse": repeat_errors[offset]}
                         for offset in range(32)]}


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("no scored episodes")
    windows = [window for episode in episodes for window in episode["windows"]]
    unique = sum(len(set().union(*(set(range(window["context_anchor_decision"],
                                         window["context_anchor_decision"] + 32))
                                   for window in episode["windows"]))) for episode in episodes)
    result = {"independent_episode_count": len(episodes),
              "independent_road_count": len({(ep["track_id"], ep["geometry_seed"]) for ep in episodes}),
              "window_count": len(windows), "window_label_uses": 32 * len(windows),
              "unique_scored_decisions": unique, "duplicate_label_uses": 32 * len(windows) - unique,
              "unique_terminal_positive_labels": sum(ep["unique_terminal_positive_labels"] for ep in episodes),
              "terminal_positive_label_uses": sum(w["terminal_positive_label_uses"] for w in windows),
              "horizons": {}}
    for name, (start, stop) in HORIZONS.items():
        label_uses = (stop - start) * len(windows)
        unique_bucket = sum(len(set().union(*(set(range(window["context_anchor_decision"] + start,
                                                        window["context_anchor_decision"] + stop))
                                          for window in episode["windows"]))) for episode in episodes)
        groups = [window["horizons"][name]["metrics"] for window in windows]
        path_errors = [_mean([group["sampled_path_mse"][index] for group in groups])
                       for index in range(K)]
        result["horizons"][name] = {
            "label_uses": label_uses, "unique_scored_decisions": unique_bucket,
            "duplicate_label_uses": label_uses - unique_bucket,
            "metrics": {"single_prior_mse": _mean([group["single_prior_mse"] for group in groups]),
                        "predictive_mean_mse": _mean([group["predictive_mean_mse"] for group in groups]),
                        **_distribution(path_errors),
                        "shifted_repeat_mse": _mean([group["shifted_repeat_mse"] for group in groups])}}
    return result


def _episode(episode: Any, modules: tuple[Any, ...], latent_seed: int,
             reference: dict[str, Any]) -> dict[str, Any]:
    length = episode.steps
    if (length < 41 or episode.actions.shape != (length, 3)
            or not np.isfinite(episode.actions).all() or np.any(np.abs(episode.actions) > 1.0)
            or reference.get("episode_id") != episode.episode_id or reference.get("decisions") != length
            or (reference.get("track_id"), reference.get("geometry_seed"))
            != (episode.track_id, int(episode.geometry_id)) or len(reference.get("windows", [])) != 2):
        raise ValueError("episode differs from pinned 8+32 native-action score")
    encoder, rssm, decoder, _, _ = modules
    anchors = (("reset", 8), ("terminal", length - 32))
    row = {"episode_id": episode.episode_id, "track_id": episode.track_id,
           "geometry_seed": int(episode.geometry_id), "decisions": length,
           "terminal_event": bool(episode.terminal[-1]), "finished": bool(episode.finished[-1]),
           "windows": []}
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(latent_seed)
        max_anchor = anchors[-1][1]
        embeds = []
        for start in range(0, max_anchor + 1, 32):
            stop = min(start + 32, max_anchor + 1)
            stacks = np.stack([episode.observation(index) for index in range(start, stop)])
            embeds.append(encoder(torch.from_numpy(stacks)).unsqueeze(0))
        embeddings = torch.cat(embeds, dim=1)
        context_actions = torch.from_numpy(episode.actions[:max_anchor].copy()).unsqueeze(0)
        first = torch.zeros((1, max_anchor), dtype=torch.bool)
        first[:, 0] = True
        states, _, _ = rssm.observe_sequence(embeddings, context_actions, first)
        for index, (name, anchor) in enumerate(anchors):
            row["windows"].append(_window(episode, rssm, decoder, states[:, anchor].clone(), anchor,
                                           name, reference["windows"][index]))
    scored = set(range(8, 40)) | set(range(length - 32, length))
    row["unique_terminal_positive_labels"] = sum(bool(episode.terminal[index]) for index in scored)
    summary = _summarize([row])
    row.update({key: value for key, value in summary.items() if key not in (
        "independent_episode_count", "independent_road_count", "horizons")})
    row["horizons"] = summary["horizons"]
    if (row["terminal_event"] != reference.get("terminal_event")
            or row["finished"] != reference.get("finished")
            or any(row[key] != reference.get(key) for key in (
                "unique_scored_decisions", "duplicate_label_uses", "unique_terminal_positive_labels"))):
        raise ValueError("episode labels or actual overlap differ from pinned score")
    _close(_mean([w["horizons"]["t1_8"]["metrics"]["single_prior_mse"] * 8 / 32
                  + w["horizons"]["t9_32"]["metrics"]["single_prior_mse"] * 24 / 32
                  for w in row["windows"]]), reference["metrics"].get("image_mse"), "episode prior")
    return row


def preflight(output_dir: Path, source_sha256: str, *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Check every fixed input, output and memory limit before opening any dataset ZIP."""
    root = Path(repo_root).resolve()
    if original._relative(root, output_dir) != OUTPUT:
        raise ValueError("output must be the fixed new sampling-v1 sibling directory")
    budget._fixed_file(root, SOURCE, SOURCE, source_sha256, cap=1024 * 1024)
    if original._sha256(Path(__file__)) != source_sha256:
        raise ValueError("executing sampling probe differs from frozen source")
    checked = budget.preflight(root / SCORE_PROTOCOL, SCORE_PROTOCOL_SHA256,
                               root / OUTPUT, repo_root=root)
    protocol = checked["protocol"]
    scored = budget._ref(root, {"path": SCORE_RESULT, "sha256": SCORE_RESULT_SHA256},
                         SCORE_RESULT, SCORE_RESULT_SHA256)
    if (scored.get("format") != budget.FORMAT or scored.get("purpose") != budget.PURPOSE
            or scored.get("status") != "iterative_tuning_descriptive_only"
            or scored.get("score_protocol_sha256") != SCORE_PROTOCOL_SHA256
            or scored.get("original_score_protocol_sha256") != budget.ORIGINAL_SCORE_PROTOCOL_SHA256
            or scored.get("original_score_result_sha256") != budget.ORIGINAL_SCORE_RESULT_SHA256
            or scored.get("offline_protocol_sha256") != budget.V2_OFFLINE_SHA256
            or scored.get("development_collection_protocol_sha256")
            != protocol["development_collection_protocol"]["sha256"]
            or scored.get("source_sha256") != protocol["source_sha256"]
            or scored.get("sampling") != protocol["scoring"]
            or any(scored.get(key) is not False for key in (
                "fresh_claim", "p1b_claim", "promotion_eligible", "student_actor_trained"))
            or set(scored.get("strata", {})) != {"random", "teacher"}):
        raise ValueError("pinned 256-update score identity/status mismatch")
    for source_arm in ("random", "teacher"):
        before = checked["original"]["strata"][source_arm]
        after = scored["strata"][source_arm]
        if ({key: value for key, value in after.items() if key != "models"}
                != {key: value for key, value in before.items() if key != "models"}
                or len(after.get("models", [])) != 4):
            raise ValueError("scored development strata differ from pinned v1")
        for index, model in enumerate(after["models"]):
            arm, seed = ("random", "teacher")[index // 2], index % 2
            if (model.get("model_arm") != arm or type(model.get("learner_seed")) is not int
                    or model["learner_seed"] != seed
                    or model.get("checkpoint_sha256") != checked["training"][arm][seed]["sha256"]):
                raise ValueError("v2 model/checkpoint identity differs from frozen protocol")
            paired = copy.deepcopy(model)
            budget._pair(paired, before["models"][index])
            for section in ("aggregate", "roads", "episodes"):
                rows = [model[section]] if section == "aggregate" else model[section]
                expected_rows = [paired[section]] if section == "aggregate" else paired[section]
                for row, expected in zip(rows, expected_rows, strict=True):
                    if row.get("paired_delta_vs_64") != expected["paired_delta_vs_64"]:
                        raise ValueError("pinned paired score deltas differ from v1")
                    if section == "episodes":
                        for window, old in zip(row["windows"], expected["windows"], strict=True):
                            if window.get("paired_delta_vs_64") != old["paired_delta_vs_64"]:
                                raise ValueError("pinned paired window deltas differ from v1")
    return {**checked, "v2_result": scored, "source_sha256": source_sha256}


def _recheck(checked: dict[str, Any]) -> None:
    root, protocol = checked["root"], checked["protocol"]
    resources = protocol["resources"]
    for name, digest in ((SOURCE, checked["source_sha256"]),
                         (SCORE_PROTOCOL, SCORE_PROTOCOL_SHA256),
                         (SCORE_RESULT, SCORE_RESULT_SHA256),
                         (budget.ORIGINAL_SCORE_PROTOCOL_PATH, budget.ORIGINAL_SCORE_PROTOCOL_SHA256),
                         (budget.ORIGINAL_SCORE_RESULT_PATH, budget.ORIGINAL_SCORE_RESULT_SHA256),
                         (original.OFFLINE_PATH, original.OFFLINE_SHA256),
                         (budget.V2_OFFLINE_PATH, budget.V2_OFFLINE_SHA256)):
        budget._fixed_file(root, name, name, digest, cap=1024 * 1024)
    if original._sha256(Path(__file__)) != checked["source_sha256"]:
        raise ValueError("executing sampling probe changed during scoring")
    for ref in (checked["checked_v1"]["offline"]["collection_protocol"],
                protocol["development_collection_protocol"],
                checked["checked_v1"]["training_collection"]["r6_protocol"]):
        budget._fixed_file(root, ref["path"], ref["path"], ref["sha256"], cap=1024 * 1024)
    for name, sha in protocol["source_sha256"].items():
        budget._fixed_file(root, name, name, sha, cap=1024 * 1024)
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


def score(output_dir: Path, source_sha256: str, *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(output_dir, source_sha256, repo_root=repo_root)
    root, protocol = checked["root"], checked["protocol"]
    resources = protocol["resources"]
    current = original._cgroup()
    if (current["limit_bytes"] > resources["max_cgroup_memory_bytes"]
            or current["available_bytes"] < checked["required_available_bytes"]):
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
        stratum = checked["v2_result"]["strata"][source_arm]
        if (dataset.transition_count != receipt["stored_decisions"] or len(dataset.episodes) != 4
                or len(dataset.episodes) != len(rows)
                or len(dataset.episodes) != receipt["complete_episode_count"]
                or len(dataset.episodes) != stratum["scored_episode_count"]):
            raise ValueError("development archive counts differ from pinned receipt/score")
        for episode, row in zip(dataset.episodes, rows, strict=True):
            if (type(row.get("episode_id")) is not int or episode.episode_id != str(row["episode_id"])
                    or episode.metadata.get("attempt") != row.get("attempt")
                    or (episode.track_id, int(episode.geometry_id)) != (row.get("track_id"), row.get("geometry_seed"))
                    or episode.steps != row.get("decisions") or bool(episode.terminal[-1]) != row.get("terminal")
                    or bool(episode.finished[-1]) != row.get("finished") or episode.steps < 41):
                raise ValueError("development episode differs from sealed receipt/windows")
        if (sum(bool(ep.terminal[-1]) for ep in dataset.episodes)
                != stratum["scored_independent_terminal_episodes"]
                or len({(ep.track_id, ep.geometry_id) for ep in dataset.episodes})
                != stratum["scored_independent_geometries"]):
            raise ValueError("development terminal/road coverage differs from pinned score")
        datasets[source_arm] = dataset

    report: dict[str, Any] = {
        "format": FORMAT, "status": "iterative_tuning_descriptive_only",
        "source_sha256": {**protocol["source_sha256"], SOURCE: source_sha256},
        "pinned_inputs_sha256": {
            "v1_score_protocol": budget.ORIGINAL_SCORE_PROTOCOL_SHA256,
            "v1_score_result": budget.ORIGINAL_SCORE_RESULT_SHA256,
            "v2_score_protocol": SCORE_PROTOCOL_SHA256, "v2_score_result": SCORE_RESULT_SHA256,
            "v1_offline_protocol": original.OFFLINE_SHA256,
            "v2_offline_protocol": budget.V2_OFFLINE_SHA256,
            "development_collection_protocol": protocol["development_collection_protocol"]["sha256"],
        },
        "sampling": {"context_decisions": 8, "future_decisions": 32, "prior_paths": K,
                     "horizon_offsets_0_based_end_exclusive": {name: list(span) for name, span in HORIZONS.items()},
                     "context_mode": "reset-origin-full-prefix-terminal",
                     "latent_seed": protocol["scoring"]["latent_seed"],
                     "sampled_mse_std_ddof": 0, "t1_subset_of_t1_8": True},
        "fresh_claim": False, "p1b_claim": False, "promotion_eligible": False,
        "student_actor_trained": False,
        "interpretation": "Eight independently drawn conditional prior paths per original posterior anchor, "
                          "not eight independent episodes. Path 0 reproduces pinned single-sample image MSE. "
                          "Predictive-mean MSE is MSE of the eight averaged predicted images, NOT mean MSE. "
                          "Sample distribution mean/std summarizes eight path-wise bucket MSEs (population std); "
                          "the zero-residual shifted-repeat baseline freezes the anchor image. t1 is contained "
                          "in t1_8; reset/terminal windows can overlap and only roads/episodes are independent. "
                          "Both development strata are consumed reused r6 TRAIN roads; neither a fresh holdout "
                          "nor a policy, driving, confirmation, blind, official or promotion result.",
        "strata": {},
    }
    with torch.no_grad():
        for source_arm in ("random", "teacher"):
            dataset = datasets[source_arm]
            stratum = {"development_archive_sha256": checked["checked_v1"]["development"][source_arm]["row"]["archive_sha256"],
                       "source_id": checked["checked_v1"]["development"][source_arm]["row"]["source_id"],
                       "models": []}
            for updates in (64, 256):
                for arm in ("random", "teacher"):
                    for seed in (0, 1):
                        previous = (checked["original"] if updates == 64 else checked["v2_result"])["strata"][source_arm]["models"][2 * (arm == "teacher") + seed]
                        model_row = (checked["checked_v1"]["training"][arm][seed]
                                     if updates == 64 else checked["training"][arm][seed])
                        checkpoint = model_row["checkpoint"]
                        digest = model_row["checkpoint_sha256"] if updates == 64 else model_row["sha256"]
                        budget._fixed_file(root, original._relative(root, checkpoint),
                                           original._relative(root, checkpoint), digest,
                                           cap=resources["max_checkpoint_bytes"])
                        if updates == 64:
                            modules = original._checkpoint_modules(
                                checkpoint, arm=arm, seed=seed,
                                dataset_digest=checked["checked_v1"]["offline"]["datasets"][arm]["dataset_digest"],
                                config=checked["checked_v1"]["offline"]["learner"]["config"])
                        else:
                            modules = budget._modules(
                                checkpoint, arm=arm, seed=seed, offline_sha=budget.V2_OFFLINE_SHA256,
                                dataset_digest=checked["offline"]["datasets"][arm]["dataset_digest"],
                                config=checked["offline"]["learner"]["config"])
                        episodes = [_episode(ep, modules,
                                             protocol["scoring"]["latent_seed"]
                                             + (source_arm == "teacher") * 1_000_003 + index * 1009 + seed,
                                             previous["episodes"][index])
                                    for index, ep in enumerate(dataset.episodes)]
                        del modules
                        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
                        for ep in episodes:
                            grouped[(ep["track_id"], ep["geometry_seed"])].append(ep)
                        aggregate = _summarize(episodes)
                        old_aggregate = previous["aggregate"]
                        if any(aggregate[new] != old_aggregate[old] for new, old in (
                                ("independent_episode_count", "episode_count"),
                                ("window_count", "window_count"), ("window_label_uses", "window_label_uses"),
                                ("unique_scored_decisions", "unique_scored_decisions"),
                                ("duplicate_label_uses", "duplicate_label_uses"),
                                ("terminal_positive_label_uses", "terminal_positive_label_uses"),
                                ("unique_terminal_positive_labels", "unique_terminal_positive_labels"))):
                            raise ValueError("aggregate denominators differ from pinned score")
                        mean_single = _mean([_mean([w["horizons"]["t1_8"]["metrics"]["single_prior_mse"] / 4
                                                    + w["horizons"]["t9_32"]["metrics"]["single_prior_mse"] * 3 / 4
                                                    for w in ep["windows"]]) for ep in episodes])
                        _close(mean_single, old_aggregate["metrics_episode_mean"]["image_mse"], "aggregate prior")
                        stratum["models"].append({"model_arm": arm, "learner_seed": seed, "updates": updates,
                                                  "checkpoint_sha256": digest, "aggregate": aggregate,
                                                  "roads": [{"track_id": track, "geometry_seed": road,
                                                             **_summarize(items)}
                                                            for (track, road), items in sorted(grouped.items())],
                                                  "episodes": episodes})
            report["strata"][source_arm] = stratum

    _recheck(checked)
    output = checked["output"]
    output.mkdir(exist_ok=False)
    with (output / "sampling-result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True,
                        help="externally frozen SHA-256 of this executable before opening real archives")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--preflight", action="store_true", help="validate without opening dataset ZIPs or scoring")
    args = parser.parse_args(argv)
    if args.preflight:
        checked = preflight(args.output, args.source_sha256, repo_root=args.repo_root)
        print(json.dumps({"status": "preflight_only", "source_sha256": checked["source_sha256"],
                          "output": str(checked["output"])}, sort_keys=True))
        return 0
    result = score(args.output, args.source_sha256, repo_root=args.repo_root)
    print(json.dumps({"status": result["status"],
                      "output": str(args.output / "sampling-result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
