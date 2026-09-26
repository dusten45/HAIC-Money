"""Score Dreamer prior rollouts on protocol-reserved development episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dreamer_v3 import DreamerV3Agent, DreamerV3Config
from scripts.diagnose.collect_dreamerv3_development import (
    load_development_cells,
    verify_protocol_sources,
)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    x, y = _rankdata(np.asarray(left)), _rankdata(np.asarray(right))
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / (np.arange(len(labels)) + 1)
    return float(precision[sorted_labels].sum() / positives)


def _episode_bootstrap_ci(values: dict[int, list[float]], seed: int, draws: int = 1000):
    per_episode = np.asarray([np.mean(items) for items in values.values()], dtype=np.float64)
    if len(per_episode) < 2:
        return None
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(per_episode), size=(draws, len(per_episode)))
    means = per_episode[sampled].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _validate_dataset(dataset: dict, protocol: dict, protocol_bytes: bytes) -> dict:
    metadata = json.loads(str(dataset["metadata_json"].item()))
    expected_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if metadata.get("protocol_sha256") != expected_sha:
        raise ValueError("development dataset protocol hash does not match the frozen protocol")
    if metadata.get("format") != "haic-dreamerv3-development-v1":
        raise ValueError("unsupported development dataset format")
    episodes = metadata.get("episodes", [])
    obs_offsets = dataset["observation_offsets"]
    transition_offsets = dataset["transition_offsets"]
    if len(obs_offsets) != len(episodes) + 1 or len(transition_offsets) != len(episodes) + 1:
        raise ValueError("development dataset offsets do not match episode manifest")
    if obs_offsets[0] != 0 or transition_offsets[0] != 0:
        raise ValueError("development dataset offsets must start at zero")
    if obs_offsets[-1] != len(dataset["observations"]):
        raise ValueError("development observation offsets do not cover the stored frames")
    if transition_offsets[-1] != len(dataset["actions"]):
        raise ValueError("development transition offsets do not cover the stored actions")
    for index, episode in enumerate(episodes):
        length = int(transition_offsets[index + 1] - transition_offsets[index])
        if obs_offsets[index + 1] - obs_offsets[index] != length + 1:
            raise ValueError("each development episode must contain one terminal successor frame")
        if episode.get("decisions") != length:
            raise ValueError("episode manifest length disagrees with transition offsets")
    return metadata


def evaluate_world_model(
    checkpoint_path: Path,
    dataset_path: Path,
    protocol_path: Path,
    output_path: Path | None = None,
) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    verify_protocol_sources(protocol)
    gate = protocol.get("world_model_gate")
    if not isinstance(gate, dict):
        raise ValueError("protocol must freeze world_model_gate thresholds")
    contexts = gate.get("context_lengths", [])
    horizons = gate.get("horizons", [])
    draws = gate.get("latent_draws")
    stride = gate.get("anchor_stride")
    threshold = gate.get("terminal_probability_threshold")
    if (
        not contexts
        or not horizons
        or any(type(value) is not int or value < 1 for value in contexts + horizons)
        or type(draws) is not int
        or draws < 1
        or type(stride) is not int
        or stride < 1
        or not 0.0 < threshold < 1.0
    ):
        raise ValueError("invalid frozen world_model_gate sampling configuration")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "haic-dreamerv3-checkpoint-v3":
        raise ValueError("open-loop evaluation requires a residual-decoder checkpoint v3")
    expected_protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    run_metadata = payload.get("run_metadata") or {}
    if run_metadata.get("protocol_sha256") != expected_protocol_sha:
        raise ValueError("learner checkpoint was not produced under this frozen protocol")
    run_config = run_metadata.get("run_config") or {}
    budget = protocol.get("training_budget") or {}
    learner_seed = run_config.get("seed")
    if "learner_seeds" not in protocol or learner_seed not in protocol["learner_seeds"]:
        raise ValueError("learner checkpoint seed is not declared by the frozen protocol")
    budget_mapping = {
        "total_steps": "total_environment_decisions",
        "warmup_steps": "random_prefill_decisions",
        "replay_pretrain_updates": "replay_pretrain_updates",
        "policy_start_step": "policy_start_step",
        "updates_per_step": "online_updates_per_decision",
        "seq_len": "sequence_length",
        "burnin_steps": "burnin_steps",
        "terminal_window_fraction": "terminal_window_fraction",
        "reset_start_fraction": "reset_start_fraction",
        "short_episode_fraction": "short_episode_fraction",
        "observation_loss_scale": "observation_loss_scale",
        "kl_free_nats": "kl_free_nats",
        "overshoot_horizon": "overshoot_horizon",
        "overshoot_kl_weight": "overshoot_kl_weight",
        "overshoot_free_nats": "overshoot_free_nats",
        "continue_positive_weight": "continue_positive_weight",
        "batch_size": "batch_size",
        "replay_capacity": "replay_capacity",
        "track_sampler_seed": "track_sampler_seed",
        "device": "device",
    }
    if not budget or any(
        run_config.get(run_key) != budget.get(protocol_key)
        for run_key, protocol_key in budget_mapping.items()
    ):
        raise ValueError("learner checkpoint training budget does not match the frozen protocol")
    if run_config.get("track_ids") != protocol.get("training_track_ids"):
        raise ValueError("learner track IDs do not match the frozen protocol")
    excluded_seeds = set(run_config.get("excluded_training_seeds", []))
    required_exclusions = set(protocol.get("reserved_training_seeds", []))
    required_exclusions.update(
        seed
        for partition in protocol.get("partitions", {}).values()
        for seed in partition.get("seeds", [])
    )
    if not required_exclusions.issubset(excluded_seeds):
        raise ValueError("learner training did not exclude every development and evaluation seed")
    config = DreamerV3Config(**payload["config"])
    config.device = device
    agent = DreamerV3Agent(config)
    agent.load_checkpoint(checkpoint_path)
    for module in (
        agent.encoder,
        agent.rssm,
        agent.decoder,
        agent.reward_head,
        agent.continue_head,
        agent.actor,
        agent.critic,
    ):
        module.eval()

    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = _validate_dataset(data, protocol, protocol_bytes)
    declared_cells = load_development_cells(protocol)
    recorded_cells = [
        (episode["track_id"], episode["geometry_seed"])
        for episode in metadata["episodes"]
    ]
    if recorded_cells != declared_cells:
        raise ValueError("development dataset episode cells do not match the frozen protocol")
    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    terminal_labels = data["terminal"].astype(bool)
    obs_offsets = data["observation_offsets"]
    transition_offsets = data["transition_offsets"]
    episodes = metadata["episodes"]

    train_size = agent.replay.size
    if train_size < 1:
        raise ValueError("checkpoint contains no training replay for trivial baselines")
    training_rewards = agent.replay.rewards[:train_size]
    constant_reward = float(np.mean(training_rewards))
    training_prevalence = float(np.mean(agent.replay.is_terminal[:train_size]))
    training_prevalence = float(np.clip(training_prevalence, 1e-6, 1.0 - 1e-6))
    training_mean_observation = (
        agent.replay.observations[:train_size].astype(np.float32).mean(axis=0) / 255.0
    )
    training_mean_baseline = torch.from_numpy(
        training_mean_observation.copy()
    ).to(device).unsqueeze(0)

    buckets: dict[tuple[int, int, int, int], list[dict]] = {}
    cumulative: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
    max_horizon = max(horizons)
    episode_lengths = []

    with torch.no_grad():
        for episode_id, episode in enumerate(episodes):
            obs_start, obs_end = map(int, obs_offsets[episode_id:episode_id + 2])
            trans_start, trans_end = map(int, transition_offsets[episode_id:episode_id + 2])
            length = trans_end - trans_start
            episode_lengths.append(length)
            for context_len in contexts:
                if length <= context_len:
                    continue
                anchors = set(range(context_len, length, stride))
                for horizon in horizons:
                    if length - horizon >= context_len:
                        anchors.add(length - horizon)
                for anchor in sorted(anchors):
                    available = min(max_horizon, length - anchor)
                    if available <= 0:
                        continue
                    context_start = anchor - context_len
                    context_frames = torch.from_numpy(
                        observations[
                            obs_start + context_start:obs_start + anchor + 1
                        ].copy()
                    ).to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0
                    context_actions = torch.from_numpy(
                        actions[
                            trans_start + context_start:trans_start + anchor
                        ].copy()
                    ).to(device=device, dtype=torch.float32).unsqueeze(0)
                    context_first = torch.zeros((1, context_len), dtype=torch.bool, device=device)
                    context_first[:, 0] = True
                    context_embed = agent.encoder(
                        context_frames.reshape(context_len + 1, 4, 84, 84)
                    ).unsqueeze(0)

                    for draw in range(draws):
                        torch.manual_seed(
                            int(gate.get("rng_seed", 0))
                            + episode_id * 1_000_003
                            + anchor * 1009
                            + draw
                        )
                        states, _, _ = agent.rssm.observe_sequence(
                            context_embed, context_actions, context_first
                        )
                        state = states[:, -1]
                        model_stack = context_frames[:, -1].clone()
                        repeat_stack = context_frames[:, -1].clone()
                        pred_sum = 0.0
                        actual_sum = 0.0
                        for step in range(available):
                            transition_index = trans_start + anchor + step
                            action = torch.from_numpy(actions[transition_index].copy()).to(
                                device=device, dtype=torch.float32
                            ).unsqueeze(0).clamp(-1.0, 1.0)
                            h, z = torch.split(
                                state, [agent.rssm.hidden_dim, agent.rssm.stoch_dim], dim=-1
                            )
                            next_h, next_z, _, _ = agent.rssm.step_prior(h, z, action)
                            next_state = torch.cat([next_h, next_z], dim=-1)
                            predicted_delta = agent.decoder(next_state)
                            model_next_frame = (
                                model_stack[:, -1:] + predicted_delta
                            ).clamp(0.0, 1.0)
                            model_stack = torch.cat(
                                [model_stack[:, 1:], model_next_frame], dim=1
                            )
                            target = torch.from_numpy(
                                observations[obs_start + anchor + step + 1].copy()
                            ).to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0
                            repeat_stack = torch.cat(
                                [repeat_stack[:, 1:], repeat_stack[:, -1:]], dim=1
                            )
                            reward_prediction = float(agent.reward_head.pred(next_state).item())
                            continue_probability = float(
                                torch.sigmoid(agent.continue_head(next_state)).item()
                            )
                            reward_actual = float(rewards[transition_index])
                            previous_reward = (
                                float(rewards[transition_index - 1])
                                if transition_index > trans_start
                                else constant_reward
                            )
                            record = {
                                "reward_prediction": reward_prediction,
                                "reward_actual": reward_actual,
                                "reward_constant_baseline_abs_error": abs(constant_reward - reward_actual),
                                "reward_last_baseline_abs_error": abs(previous_reward - reward_actual),
                                "continue_probability": continue_probability,
                                "terminal": bool(terminal_labels[transition_index]),
                                "image_mse_full": float(F.mse_loss(model_stack, target).item()),
                                "image_mse_latest": float(
                                    F.mse_loss(model_next_frame[:, -1], target[:, -1]).item()
                                ),
                                "repeat_mse_latest": float(
                                    F.mse_loss(repeat_stack[:, -1], target[:, -1]).item()
                                ),
                                "repeat_mse_full": float(F.mse_loss(repeat_stack, target).item()),
                                "mean_mse_latest": float(
                                    F.mse_loss(training_mean_baseline[:, -1], target[:, -1]).item()
                                ),
                            }
                            key = (context_len, step + 1, episode_id, anchor + step)
                            buckets.setdefault(key, []).append(record)
                            pred_sum += reward_prediction
                            actual_sum += record["reward_actual"]
                            if step + 1 in horizons:
                                cumulative.setdefault(
                                    (context_len, step + 1, episode_id), []
                                ).append((pred_sum, actual_sum))
                            state = next_state
                            if record["terminal"]:
                                break

    result_rows = []
    failed = []
    inconclusive = []
    seed = int(gate.get("rng_seed", 0))
    min_positive = int(gate["min_positive_episodes"])
    min_negative = int(gate["min_negative_episodes"])
    threshold = float(threshold)

    for context_len in contexts:
        for horizon in horizons:
            episode_records: dict[int, list[dict]] = {}
            for (ctx, h, episode_id, _), predictions in buckets.items():
                if ctx != context_len or h != horizon:
                    continue
                averaged = {
                    name: float(np.mean([row[name] for row in predictions]))
                    for name in predictions[0]
                    if name != "terminal"
                }
                averaged["terminal"] = predictions[0]["terminal"]
                episode_records.setdefault(episode_id, []).append(averaged)

            flat = [row for rows in episode_records.values() for row in rows]
            labels = np.asarray([row["terminal"] for row in flat], dtype=bool)
            probabilities = np.asarray(
                [1.0 - row["continue_probability"] for row in flat], dtype=np.float64
            )
            positives = int(labels.sum())
            negatives = int((~labels).sum())
            positive_episodes = len({
                episode_id for episode_id, rows in episode_records.items()
                if any(row["terminal"] for row in rows)
            })
            negative_episodes = len({
                episode_id for episode_id, rows in episode_records.items()
                if any(not row["terminal"] for row in rows)
            })

            if flat:
                clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
                bce = float(-np.mean(labels * np.log(clipped) + (~labels) * np.log(1.0 - clipped)))
                baseline_bce = float(-np.mean(
                    labels * np.log(training_prevalence)
                    + (~labels) * np.log(1.0 - training_prevalence)
                ))
                prevalence = float(labels.mean())
                scored_prevalence = float(np.clip(prevalence, 1e-6, 1.0 - 1e-6))
                matched_constant_bce = float(-np.mean(
                    labels * np.log(scored_prevalence)
                    + (~labels) * np.log(1.0 - scored_prevalence)
                ))
                recall = float(((probabilities >= threshold) & labels).sum() / max(positives, 1))
                specificity = float(
                    ((probabilities < threshold) & (~labels)).sum() / max(negatives, 1)
                )
                balanced_accuracy = 0.5 * (recall + specificity)
                average_precision = _average_precision(labels, probabilities)
                image_differences, image_by_episode = {}, {}
                full_image_by_episode = {}
                mean_image_by_episode = {}
                for episode_id, rows in episode_records.items():
                    image_by_episode[episode_id] = [
                        row["image_mse_latest"] - row["repeat_mse_latest"]
                        for row in rows
                    ]
                    image_differences[episode_id] = float(np.mean(image_by_episode[episode_id]))
                    full_image_by_episode[episode_id] = [
                        row["image_mse_full"] - row["repeat_mse_full"]
                        for row in rows
                    ]
                    mean_image_by_episode[episode_id] = [
                        row["image_mse_latest"] - row["mean_mse_latest"]
                        for row in rows
                    ]
                image_ci = _episode_bootstrap_ci(image_by_episode, seed + context_len * 100 + horizon)
                full_image_ci = _episode_bootstrap_ci(
                    full_image_by_episode, seed + context_len * 200 + horizon
                )
                mean_image_ci = _episode_bootstrap_ci(
                    mean_image_by_episode, seed + context_len * 300 + horizon
                )
                reward_constant_mae = float(np.mean([
                    np.mean([row["reward_constant_baseline_abs_error"] for row in rows])
                    for rows in episode_records.values()
                ]))
                reward_last_mae = float(np.mean([
                    np.mean([row["reward_last_baseline_abs_error"] for row in rows])
                    for rows in episode_records.values()
                ]))
                episode_reward_mae = float(np.mean([
                    np.mean([abs(row["reward_prediction"] - row["reward_actual"]) for row in rows])
                    for rows in episode_records.values()
                ]))
            else:
                bce = baseline_bce = matched_constant_bce = recall = specificity = balanced_accuracy = None
                average_precision = prevalence = episode_reward_mae = None
                image_ci = full_image_ci = mean_image_ci = None
                reward_constant_mae = reward_last_mae = None
                image_differences = {}

            returns_by_episode = {}
            for (ctx, h, episode_id), values in cumulative.items():
                if ctx == context_len and h == horizon:
                    returns_by_episode[episode_id] = np.mean(np.asarray(values), axis=0).tolist()
            reward_rank = _spearman(
                [value[0] for value in returns_by_episode.values()],
                [value[1] for value in returns_by_episode.values()],
            )

            enough_events = positive_episodes >= min_positive and negative_episodes >= min_negative
            if not enough_events:
                inconclusive.append(f"context={context_len},horizon={horizon}:insufficient_terminal_episodes")
            if enough_events and flat:
                gates = {
                    "repeat_frame_baseline": image_ci is not None and image_ci[1] < 0.0,
                    "full_stack_repeat_baseline": full_image_ci is not None and full_image_ci[1] < 0.0,
                    "reward_rank": horizon != 5 or (
                        reward_rank is not None and reward_rank >= float(gate["reward_spearman_min"])
                    ),
                    "terminal_recall": recall is not None and recall >= float(gate["terminal_recall_min"]),
                    "terminal_balanced_accuracy": balanced_accuracy is not None
                    and balanced_accuracy >= float(gate["terminal_balanced_accuracy_min"]),
                    "terminal_pr_auc": average_precision is not None
                    and prevalence is not None
                    and average_precision >= float(gate["pr_auc_multiple"]) * prevalence,
                    "terminal_bce": bce is not None
                    and matched_constant_bce is not None
                    and bce <= matched_constant_bce * (1.0 - float(gate["bce_relative_improvement_min"])),
                }
                if not all(gates.values()):
                    failed.extend(
                        f"context={context_len},horizon={horizon}:{name}"
                        for name, passed in gates.items()
                        if not passed
                    )
            else:
                gates = None

            result_rows.append({
                "context_length": context_len,
                "horizon": horizon,
                "unique_transition_predictions": len(flat),
                "positive_transitions": positives,
                "negative_transitions": negatives,
                "positive_episodes": positive_episodes,
                "negative_episodes": negative_episodes,
                "terminal_prevalence": prevalence,
                "terminal_pr_auc": average_precision,
                "terminal_recall": recall,
                "terminal_balanced_accuracy": balanced_accuracy,
                "terminal_bce": bce,
                "constant_prevalence_bce": baseline_bce,
                "scored_window_fitted_constant_bce": matched_constant_bce,
                "latest_frame_mse_delta_vs_repeat": float(np.mean(list(image_differences.values())))
                if image_differences else None,
                "latest_frame_mse_delta_episode_bootstrap_ci95": image_ci,
                "full_stack_mse_delta_vs_repeat_episode_bootstrap_ci95": full_image_ci,
                "latest_frame_mse_delta_vs_training_mean_episode_bootstrap_ci95": mean_image_ci,
                "reward_mae": episode_reward_mae,
                "reward_constant_baseline_mae": reward_constant_mae,
                "reward_last_baseline_mae": reward_last_mae,
                "cumulative_reward_spearman_across_episodes": reward_rank,
                "gates": gates,
            })

    gate_status = "inconclusive" if inconclusive else "fail" if failed else "pass"
    report = {
        "format": "haic-dreamerv3-open-loop-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "learner_environment_steps": agent.environment_steps,
        "learner_gradient_steps": agent.gradient_steps,
        "training_terminal_prevalence": training_prevalence,
        "development_episode_count": len(episodes),
        "development_episode_length_mean": float(np.mean(episode_lengths)),
        "world_model_gate": gate_status,
        "failed_conditions": failed,
        "inconclusive_conditions": inconclusive,
        "metrics": result_rows,
        "interpretation": (
            "Training-excluded development diagnostic, not official score evidence. "
            "The scored-window-fitted BCE reference uses evaluation labels retrospectively: "
            "it is not a deployable or independently fitted predictor. Logged-action "
            "reward Spearman ranks episodes, not alternative actions at one state."
        ),
    }
    if output_path is not None:
        output_path = output_path.resolve()
        if output_path.exists():
            raise FileExistsError(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_world_model(
        args.checkpoint.resolve(),
        args.dataset.resolve(),
        args.protocol.resolve(),
        args.output,
    )
    print(json.dumps({
        "world_model_gate": report["world_model_gate"],
        "failed_conditions": report["failed_conditions"],
        "inconclusive_conditions": report["inconclusive_conditions"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
