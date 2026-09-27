"""Matched short/full-context audit of a frozen Dreamer B1 checkpoint.

This is deterministic latent-mode inference on existing development episodes,
not a rerun of the stochastic B1 gate or a new policy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dreamer_v3 import DreamerV3Agent, DreamerV3Config


ROOT = Path(__file__).resolve().parents[2]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode_prior(agent: DreamerV3Agent, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
    next_h, _, logits, _ = agent.rssm.step_prior(h, z, action)
    index = logits.argmax(dim=-1)
    next_z = F.one_hot(index, num_classes=agent.rssm.c).float().reshape(1, -1)
    return next_h, next_z


def _average_precision(labels: list[bool], scores: list[float]) -> float | None:
    if not any(labels):
        return None
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    positives = np.asarray(labels, dtype=bool)[order]
    return float((np.cumsum(positives)[positives] / (np.flatnonzero(positives) + 1)).mean())


def audit(checkpoint: Path, dataset: Path, protocol: Path) -> dict:
    torch.set_num_threads(2)
    protocol_bytes = protocol.read_bytes()
    frozen = json.loads(protocol_bytes)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if frozen["source_sha256"]["dreamer_v3.py"] != _hash(ROOT / "dreamer_v3.py"):
        raise ValueError("current Dreamer inference source differs from frozen checkpoint source")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "haic-dreamerv3-checkpoint-v3":
        raise ValueError("a frozen residual-decoder checkpoint v3 is required")
    if payload.get("run_metadata", {}).get("protocol_sha256") != protocol_sha:
        raise ValueError("checkpoint protocol mismatch")
    config = DreamerV3Config(**payload["config"])
    config.device = "cpu"
    agent = DreamerV3Agent(config)
    agent.load_checkpoint(checkpoint)
    for module in (agent.encoder, agent.rssm, agent.decoder, agent.reward_head, agent.continue_head):
        module.eval()

    with np.load(dataset, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data["metadata_json"].item()))
    if metadata.get("protocol_sha256") != protocol_sha:
        raise ValueError("development dataset protocol mismatch")
    recorded_cells = [
        (row["track_id"], row["geometry_seed"]) for row in metadata["episodes"]
    ]
    expected_cells = [
        (row["track_id"], row["seed"]) for row in frozen["training_development"]["cells"]
    ]
    if recorded_cells != expected_cells:
        raise ValueError("development cells differ from frozen protocol")

    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    terminals = data["terminal"]
    obs_offsets = data["observation_offsets"]
    trans_offsets = data["transition_offsets"]
    rows = []
    with torch.inference_mode():
        for episode_id in range(len(recorded_cells)):
            os0, os1 = map(int, obs_offsets[episode_id:episode_id + 2])
            ts0, ts1 = map(int, trans_offsets[episode_id:episode_id + 2])
            length = ts1 - ts0
            if length < 16 or os1 - os0 != length + 1:
                continue
            frames = torch.from_numpy(observations[os0:os1].copy()).float() / 255.0
            embeds = torch.cat([
                agent.encoder(frames[start:start + 32])
                for start in range(0, len(frames), 32)
            ])
            anchors = sorted({10, length // 2, length - 5, length - 1})
            for t in anchors:
                for context in (5, 8, 10, "full"):
                    start = 0 if context == "full" else t - context
                    if start < 0:
                        continue
                    h = torch.zeros(1, agent.rssm.hidden_dim)
                    z = torch.zeros(1, agent.rssm.stoch_dim)
                    previous_action = torch.zeros(1, 3)
                    for index in range(start, t + 1):
                        h, z = agent.rssm.step_post_deterministic(
                            h, z, previous_action, embeds[index:index + 1]
                        )
                        if index < t:
                            previous_action = torch.from_numpy(actions[ts0 + index].copy()).view(1, 3)

                    predicted_stack = frames[t:t + 1].clone()
                    repeated_stack = frames[t:t + 1].clone()
                    predicted_rewards = []
                    actual_rewards = []
                    survival = 1.0
                    for step in range(min(5, length - t)):
                        native_action = torch.from_numpy(actions[ts0 + t + step].copy()).view(1, 3)
                        h, z = _mode_prior(agent, h, z, native_action)
                        state = torch.cat([h, z], dim=-1)
                        next_frame = (predicted_stack[:, -1:] + agent.decoder(state)).clamp(0.0, 1.0)
                        predicted_stack = torch.cat([predicted_stack[:, 1:], next_frame], dim=1)
                        repeated_stack = torch.cat([repeated_stack[:, 1:], repeated_stack[:, -1:]], dim=1)
                        survival *= float(torch.sigmoid(agent.continue_head(state)).item())
                        predicted_rewards.append(float(agent.reward_head.pred(state).item()))
                        actual_rewards.append(float(rewards[ts0 + t + step]))
                        if step + 1 in (1, 5):
                            target = frames[t + step + 1:t + step + 2]
                            rows.append({
                                "episode_id": episode_id,
                                "track_id": recorded_cells[episode_id][0],
                                "anchor": t,
                                "context": context,
                                "horizon": step + 1,
                                "actual_terminal": bool(np.any(terminals[ts0 + t:ts0 + t + step + 1])),
                                "predicted_terminal_risk": 1.0 - survival,
                                "model_latest_mse": float(F.mse_loss(predicted_stack[:, -1], target[:, -1]).item()),
                                "repeat_latest_mse": float(F.mse_loss(repeated_stack[:, -1], target[:, -1]).item()),
                                "reward_sum_prediction": float(sum(predicted_rewards)),
                                "reward_sum_actual": float(sum(actual_rewards)),
                            })
                        if terminals[ts0 + t + step]:
                            break

    summaries = []
    for context in (5, 8, 10, "full"):
        for horizon in (1, 5):
            selected = [row for row in rows if row["context"] == context and row["horizon"] == horizon]
            if not selected:
                continue
            label = [row["actual_terminal"] for row in selected]
            risks = [row["predicted_terminal_risk"] for row in selected]
            summaries.append({
                "context": context,
                "horizon": horizon,
                "n": len(selected),
                "terminal_n": sum(label),
                "terminal_prevalence": float(np.mean(label)),
                "terminal_pr_auc": _average_precision(label, risks),
                "terminal_recall_at_half": float(np.mean([risk >= 0.5 for risk, positive in zip(risks, label) if positive])) if any(label) else None,
                "risk_mean_terminal": float(np.mean([risk for risk, positive in zip(risks, label) if positive])) if any(label) else None,
                "risk_mean_nonterminal": float(np.mean([risk for risk, positive in zip(risks, label) if not positive])) if not all(label) else None,
                "latest_mse_delta_model_minus_repeat": float(np.mean([
                    row["model_latest_mse"] - row["repeat_latest_mse"] for row in selected
                ])),
                "reward_sum_mae": float(np.mean([
                    abs(row["reward_sum_prediction"] - row["reward_sum_actual"]) for row in selected
                ])),
            })
    return {
        "format": "haic-dreamerv3-b1-matched-context-audit-v1",
        "checkpoint_sha256": _hash(checkpoint),
        "dataset_sha256": _hash(dataset),
        "protocol_sha256": protocol_sha,
        "source_sha256": _hash(ROOT / "dreamer_v3.py"),
        "method": "fixed identical episode/anchor/native action; deterministic posterior/prior categorical argmax; actual future observations used only as targets; episode-level cells are distinct",
        "limitations": "Not the stochastic frozen B1 gate; handpicked 4 anchors/episode enrich termination; no alternative-action ground truth; no actor learning",
        "episode_count": len(recorded_cells),
        "rows": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(args.checkpoint.resolve(), args.dataset.resolve(), args.protocol.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "episode_count": report["episode_count"], "rows": len(report["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
