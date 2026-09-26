"""Audit saved residual-option Q heads and replay without environment access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from haic.algorithms.drq_v2 import (
    OPTION_COUNT,
    ResidualOption,
    ResidualOptionQ,
    apply_native_residual,
    select_greedy_option,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = {
    "v1": ROOT / "runs/20260924-drqv2-residual-options-pilot/iteration-1-retry",
    "v2": ROOT / "runs/20260924-drqv2-residual-options-pilot/iteration-2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option_counts(indices: np.ndarray) -> dict[str, int]:
    return {
        option.name: int(np.count_nonzero(indices == int(option)))
        for option in ResidualOption
    }


def q_gap_summary(q_values: np.ndarray) -> dict:
    q_values = np.asarray(q_values, dtype=np.float64)
    if q_values.ndim != 2 or q_values.shape[1] != OPTION_COUNT or not np.isfinite(q_values).all():
        raise ValueError("q_values must be a finite (N, option_count) array")
    gaps = q_values[:, 1:].max(axis=1) - q_values[:, int(ResidualOption.KEEP)]
    return {
        "states": int(len(gaps)),
        "positive_non_keep_advantage_fraction": float(np.mean(gaps > 0.0)) if len(gaps) else None,
        "gap_quantiles": {
            str(q): float(np.quantile(gaps, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        } if len(gaps) else {},
    }


def policy_choice_summary(
    q_values: np.ndarray,
    base_actions: np.ndarray,
    observed_options: np.ndarray,
    *,
    margin: float,
) -> dict:
    q_values = np.asarray(q_values, dtype=np.float64)
    base_actions = np.asarray(base_actions, dtype=np.float32)
    observed_options = np.asarray(observed_options, dtype=np.int64)
    if q_values.ndim != 2 or q_values.shape[1] != OPTION_COUNT:
        raise ValueError("q_values has an invalid shape")
    if base_actions.shape != (len(q_values), 3) or observed_options.shape != (len(q_values),):
        raise ValueError("base actions and observed options must match Q rows")
    if np.any(observed_options < 0) or np.any(observed_options >= OPTION_COUNT):
        raise ValueError("observed option outside the known option set")

    unrestricted = q_values.argmax(axis=1)
    gated = np.asarray([
        int(select_greedy_option(row, intervention_margin=margin)) for row in q_values
    ], dtype=np.int64)

    effective_changes = np.zeros(len(q_values), dtype=np.bool_)
    for index, option_index in enumerate(gated):
        applied = apply_native_residual(base_actions[index], ResidualOption(int(option_index)))
        effective_changes[index] = not np.array_equal(applied, base_actions[index])

    return {
        "margin": float(margin),
        "states": int(len(q_values)),
        "unrestricted_argmax_counts": _option_counts(unrestricted),
        "margin_gated_choice_counts": _option_counts(gated),
        "recorded_behavior_option_counts": _option_counts(observed_options),
        "unrestricted_vs_gated_choice_mismatch_fraction": (
            float(np.mean(unrestricted != gated)) if len(q_values) else None
        ),
        "current_head_greedy_vs_recorded_behavior_agreement_fraction": (
            float(np.mean(unrestricted == observed_options)) if len(q_values) else None
        ),
        "margin_gated_option_selection_fraction": (
            float(np.mean(gated != int(ResidualOption.KEEP))) if len(q_values) else None
        ),
        "margin_gated_effective_action_change_fraction": (
            float(np.mean(effective_changes)) if len(q_values) else None
        ),
    }


def target_choice_summary(
    online_next_q: np.ndarray,
    terminal: np.ndarray,
    *,
    margin: float,
) -> dict:
    online_next_q = np.asarray(online_next_q, dtype=np.float64)
    terminal = np.asarray(terminal, dtype=np.bool_)
    if online_next_q.ndim != 2 or online_next_q.shape[1] != OPTION_COUNT:
        raise ValueError("online_next_q has an invalid shape")
    if terminal.shape != (len(online_next_q),):
        raise ValueError("terminal mask must match next-Q rows")
    bootstrap = ~terminal
    if not bootstrap.any():
        return {
            "bootstrap_rows": 0,
            "unrestricted_next_action_counts": _option_counts(np.asarray([], dtype=np.int64)),
            "margin_gated_next_action_counts": _option_counts(np.asarray([], dtype=np.int64)),
            "selector_mismatch_count": 0,
            "selector_mismatch_fraction": None,
            "interpretation": "No nonterminal rows; no target-selector comparison available.",
        }
    q_values = online_next_q[bootstrap]
    unrestricted = q_values.argmax(axis=1)
    gated = np.asarray([
        int(select_greedy_option(row, intervention_margin=margin)) for row in q_values
    ], dtype=np.int64)
    mismatch = unrestricted != gated
    return {
        "bootstrap_rows": int(bootstrap.sum()),
        "unrestricted_next_action_counts": _option_counts(unrestricted),
        "margin_gated_next_action_counts": _option_counts(gated),
        "selector_mismatch_count": int(mismatch.sum()),
        "selector_mismatch_fraction": float(mismatch.mean()),
        "interpretation": (
            "Current training code chooses the unrestricted online argmax for its Double-Q bootstrap. "
            "The saved target-head weights are absent, so this compares action indices only; "
            "it does not reconstruct target values or prove the off-policy target is invalid."
        ),
    }


def _load_training_artifacts(label: str, run_dir: Path) -> dict:
    checkpoint_path = run_dir / "option_q.pt"
    replay_path = run_dir / "option_replay.pt"
    result_path = run_dir / "result.json"
    manifest_path = run_dir / "manifest.completed.json"
    for path in (checkpoint_path, replay_path, result_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: required training artifact is missing: {path}")

    checkpoint_hash = sha256_file(checkpoint_path)
    replay_hash = sha256_file(replay_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # These are locally produced NumPy replay snapshots; no external input is accepted.
    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
    if checkpoint_hash != result.get("checkpoint_sha256"):
        raise ValueError(f"{label}: Q checkpoint hash disagrees with result receipt")
    if replay_hash != result.get("replay_sha256"):
        raise ValueError(f"{label}: replay hash disagrees with result receipt")
    if checkpoint.get("base_actor_sha256") != manifest.get("base_actor_sha256"):
        raise ValueError(f"{label}: Q head and run manifest reference different base actors")
    if replay.get("frozen_actor_sha256") != manifest.get("base_actor_sha256"):
        raise ValueError(f"{label}: replay and run manifest reference different base actors")
    if checkpoint.get("format") != "haic-drq-v2-residual-option-q-v1":
        raise ValueError(f"{label}: unsupported option Q checkpoint format")
    size = replay.get("size")
    feature_dim = checkpoint.get("feature_dim")
    if type(size) is not int or size <= 0:
        raise ValueError(f"{label}: replay is empty or malformed")
    if replay["features"].shape != (replay["capacity"], feature_dim):
        raise ValueError(f"{label}: replay feature dimensions do not match Q head")

    head = ResidualOptionQ(feature_dim, checkpoint["hidden_dim"])
    head.load_state_dict(checkpoint["option_q"])
    head.eval()
    stored_margin = float(checkpoint.get("option_spec", {}).get("intervention_margin_q_minus_keep", 0.0))
    return {
        "label": label,
        "run_dir": str(run_dir),
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "replay_path": replay_path,
        "replay_hash": replay_hash,
        "result": result,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "replay": replay,
        "head": head,
        "stored_margin": stored_margin,
    }


@torch.inference_mode()
def _q_for_rows(head: ResidualOptionQ, features: np.ndarray, actions: np.ndarray) -> np.ndarray:
    q_values = head(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.float32),
    )
    return q_values.detach().cpu().numpy().astype(np.float64, copy=False)


def _replay_support(replay: dict) -> dict:
    size = int(replay["size"])
    options = np.asarray(replay["options"][:size], dtype=np.int64)
    rewards = np.asarray(replay["rewards"][:size], dtype=np.float64)
    durations = np.asarray(replay["durations"][:size], dtype=np.int64)
    terminal = np.asarray(replay["terminal"][:size], dtype=np.bool_)
    truncated = np.asarray(replay["truncated"][:size], dtype=np.bool_)
    terminated = np.asarray(replay["terminated"][:size], dtype=np.bool_)
    options_by_name = {}
    for option in ResidualOption:
        mask = options == int(option)
        count = int(mask.sum())
        options_by_name[option.name] = {
            "transition_count": count,
            "mean_recorded_option_return": float(rewards[mask].mean()) if count else None,
            "median_recorded_option_return": float(np.median(rewards[mask])) if count else None,
            "mean_duration_decisions": float(durations[mask].mean()) if count else None,
            "terminal_count": int(terminal[mask].sum()),
            "terminated_count": int(terminated[mask].sum()),
            "truncated_count": int(truncated[mask].sum()),
            "both_terminated_and_truncated_count": int((terminated[mask] & truncated[mask]).sum()),
        }
    return {
        "stored_rows": size,
        "capacity": int(replay["capacity"]),
        "cursor": int(replay["cursor"]),
        "feature_dim": int(replay["feature_dim"]),
        "total_recorded_option_return": "discounted raw reward accumulated only over this observed option; not episodic return or counterfactual advantage",
        "episode_ids_saved": False,
        "geometry_ids_saved": False,
        "executed_applied_action_saved_per_step": False,
        "option_support": options_by_name,
    }


def build_audit(run_dirs: dict[str, Path]) -> dict:
    torch.set_num_threads(1)
    artifacts = {
        label: _load_training_artifacts(label, run_dir.resolve())
        for label, run_dir in run_dirs.items()
    }
    actor_hashes = {row["manifest"]["base_actor_sha256"] for row in artifacts.values()}
    if len(actor_hashes) != 1:
        raise ValueError("v1 and v2 runs do not share a single frozen DrQ actor")

    replay_q = {}
    own_margins = {}
    base_data = {}
    for data_label, artifact in artifacts.items():
        replay = artifact["replay"]
        size = int(replay["size"])
        features = np.asarray(replay["features"][:size], dtype=np.float32)
        base_actions = np.asarray(replay["base_actions"][:size], dtype=np.float32)
        observed_options = np.asarray(replay["options"][:size], dtype=np.int64)
        terminal = np.asarray(replay["terminal"][:size], dtype=np.bool_)
        next_features = np.asarray(replay["next_features"][:size], dtype=np.float32)
        next_actions = np.asarray(replay["next_base_actions"][:size], dtype=np.float32)
        if features.shape[1] != artifact["head"].feature_dim or base_actions.shape != (size, 3):
            raise ValueError(f"{data_label}: replay observation/action shape is invalid")
        base_data[data_label] = {
            "features": features,
            "base_actions": base_actions,
            "observed_options": observed_options,
            "terminal": terminal,
            "next_features": next_features,
            "next_base_actions": next_actions,
        }
        own_q = _q_for_rows(artifact["head"], features, base_actions)
        own_margins[data_label] = float(
            np.quantile(own_q[:, 1:].max(axis=1) - own_q[:, int(ResidualOption.KEEP)], 0.75)
        )

    for head_label, artifact in artifacts.items():
        replay_q[head_label] = {}
        for data_label, data in base_data.items():
            q_values = _q_for_rows(artifact["head"], data["features"], data["base_actions"])
            replay_q[head_label][data_label] = q_values

    v1_reused_margin = own_margins["v1"]
    comparisons = {}
    for head_label, artifact in artifacts.items():
        comparisons[head_label] = {}
        for data_label, data in base_data.items():
            q_values = replay_q[head_label][data_label]
            margins = {
                "zero": 0.0,
                "v1_training_replay_q75": v1_reused_margin,
                f"{head_label}_head_on_{head_label}_replay_q75": own_margins[head_label],
            }
            comparisons[head_label][data_label] = {
                "q_gap_distribution": q_gap_summary(q_values),
                "gate_scenarios": {
                    name: policy_choice_summary(
                        q_values,
                        data["base_actions"],
                        data["observed_options"],
                        margin=margin,
                    )
                    for name, margin in margins.items()
                },
                "training_option_support": _replay_support(artifacts[data_label]["replay"]),
                "target_selector_vs_gated_behavior": target_choice_summary(
                    _q_for_rows(artifact["head"], data["next_features"], data["next_base_actions"]),
                    data["terminal"],
                    margin=artifact["stored_margin"],
                ),
            }

    run_rows = {}
    for label, artifact in artifacts.items():
        checkpoint = artifact["checkpoint"]
        run_rows[label] = {
            "run_dir": artifact["run_dir"],
            "protocol": artifact["manifest"].get("protocol"),
            "protocol_sha256": artifact["manifest"].get("protocol_sha256"),
            "training_source_sha256": artifact["manifest"].get("source_sha256"),
            "base_actor_sha256": artifact["manifest"].get("base_actor_sha256"),
            "option_checkpoint_sha256": artifact["checkpoint_hash"],
            "replay_sha256": artifact["replay_hash"],
            "stored_intervention_margin": artifact["stored_margin"],
            "head_feature_dim": int(checkpoint["feature_dim"]),
            "target_head_saved": "target_option_q" in checkpoint,
            "optimizer_saved": "optimizer" in checkpoint,
            "gradient_steps": checkpoint.get("gradient_steps"),
            "training_result": {
                "environment_decisions": artifact["result"].get("environment_decisions"),
                "completed_episodes": artifact["result"].get("completed_episodes"),
                "completed_episode_finishes": artifact["result"].get("completed_episode_finishes"),
            },
        }

    return {
        "schema_version": 1,
        "name": "drqv2-residual-options-offline-training-audit",
        "status": "read-only-training-artifact-diagnostic",
        "environment_interaction": False,
        "development_evaluation_cells_accessed": False,
        "confirmation_or_blind_cells_accessed": False,
        "base_actor_sha256": next(iter(actor_hashes)),
        "runs": run_rows,
        "head_specific_q75_training_gaps": own_margins,
        "cross_head_replay_policy_comparisons": comparisons,
        "target_value_reconstruction": {
            "possible": False,
            "reason": "The saved pilot checkpoint includes the online option head, but not the target option head or optimizer. This audit compares online action indices only and cannot reproduce historical Double-Q target values.",
        },
        "limits": [
            "Replay rows are transitions from the option behavior policy, not independent states; counts are not episode-level coverage.",
            "The stored per-option reward is a discounted raw return over the actually executed option only. No unchosen-option counterfactual outcome exists.",
            "Episode IDs, geometry IDs, and per-decision applied actions were not saved in option replay, so option reward summaries cannot identify a causal failure mechanism.",
            "The q-gap quantiles are training-replay diagnostics, not calibration on held-out states and not confidence bounds on actual finish value.",
            "The v2 reused margin was derived from the v1 head/replay; recomputed head-specific quantiles diagnose scale transfer but do not validate a new margin.",
        ],
        "next_gate": "If the target-versus-deployed selector differs materially, decide before any new study whether Q estimates the unconstrained optimal option or the margin-gated deployed policy; then freeze a policy-consistent target and a head-specific training-only calibration rule. No new rollout follows from this audit alone.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-run", type=Path, default=DEFAULT_RUNS["v1"])
    parser.add_argument("--v2-run", type=Path, default=DEFAULT_RUNS["v2"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluations/drqv2-residual-options-training-offline-audit.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {output}")
    result = build_audit({"v1": args.v1_run, "v2": args.v2_run})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
