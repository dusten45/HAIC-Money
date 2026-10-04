"""Validate the paired Lagrangian PPO screen without training or simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from haic_agent.networks import (
    CPU_INFERENCE_THREADS,
    DEFAULT_PEDAL_EXPANSION,
    VisualActorCritic,
)
from training.lagrangian_ppo import LagrangianActorCritic, LagrangianPPOUpdater
from training.ppo import PPOConfig, PPOUpdater
from training.train_policy import (
    actor_state_sha256,
    load_train_only_site_map_split,
    set_reproducible_seed,
)


REGISTERED_TRAIN_CELLS = (
    ("custom-track-haic-obstacles-20260920", 20260920),
    ("custom-track-haic-obstacles-20260920", 20260924),
    ("custom-track-haic-train-20260921", 20260921),
    ("custom-track-haic-train-20260921", 20260925),
)
MODEL_SEEDS = (8104, 8105)
LAGRANGIAN_MODES = ("off", "adaptive")
SCREEN_ACTOR_SETTINGS = {
    "use_hud": False,
    "use_visual_features": True,
    "use_temporal_features": False,
}


def _new_actor(
    seed: int,
    *,
    use_hud: bool = False,
    use_visual_features: bool = True,
    throttle_expansion: float = 3.5,
) -> VisualActorCritic:
    torch.set_num_threads(CPU_INFERENCE_THREADS)
    set_reproducible_seed(seed)
    actor = VisualActorCritic(
        use_hud=use_hud,
        use_visual_features=use_visual_features,
        use_temporal_features=False,
        throttle_expansion=throttle_expansion,
        brake_expansion=DEFAULT_PEDAL_EXPANSION,
    )
    return actor


def build_preflight_report(
    *,
    site_map_split: Path | str,
    preflight_report_path: Path | str | None = None,
    run_directory: Path | str | None = None,
    model_seeds: tuple[int, ...] = MODEL_SEEDS,
    total_steps: int = 8192,
    updates: int = 8,
    throttle_expansion: float = 3.5,
    teacher_warmup_epochs: int = 0,
    teacher_max_decisions: int = 800,
    teacher_warmup_batch_size: int = 32,
    teacher_warmup_learning_rate: float = 1e-3,
    initialize_from: Path | str | None = None,
    resume: Path | str | None = None,
) -> dict[str, Any]:
    """Prove TRAIN-only loading and paired actor initialization for the frozen screen."""
    if tuple(model_seeds) not in (MODEL_SEEDS, (8104,)):
        raise ValueError("the paired screen supports model seed 8104 and optionally 8105")
    if initialize_from is not None:
        raise ValueError("the clean-start preflight must not initialize from checkpoint weights")
    initial_checkpoint_sha256 = None
    actor_settings = dict(SCREEN_ACTOR_SETTINGS)
    if total_steps != 8192 or updates != 8:
        raise ValueError("the paired screen is frozen at 8,192 decisions and 8 updates")
    if float(throttle_expansion) != 3.5:
        raise ValueError("the paired screen requires throttle expansion 3.5")
    if teacher_warmup_epochs < 0:
        raise ValueError("teacher warm-up epochs must be non-negative")
    if teacher_max_decisions < 1:
        raise ValueError("teacher max decisions must be positive")
    if teacher_warmup_batch_size < 1:
        raise ValueError("teacher warm-up batch size must be positive")
    if not math.isfinite(teacher_warmup_learning_rate) or teacher_warmup_learning_rate <= 0:
        raise ValueError("teacher warm-up learning rate must be finite and positive")
    if resume is not None:
        raise ValueError("the paired screen does not support optimizer resume")

    split = load_train_only_site_map_split(site_map_split)
    actual_cells = tuple((episode.map_id, int(episode.seed)) for episode in split.train)
    if actual_cells != REGISTERED_TRAIN_CELLS:
        raise ValueError(
            "TRAIN-only manifest must contain exactly the four registered TRAIN cells"
        )
    if split.tune or split.held_out:
        raise ValueError("non-TRAIN episode objects must not enter the paired screen")
    if any(
        episode.site_map.map_kind != "custom"
        or episode.site_map.obstacle_mode != "custom_only"
        for episode in split.train
    ):
        raise ValueError("paired screen TRAIN cells must be custom_only maps")

    config = PPOConfig(
        learning_rate=2e-6,
        policy_mean_learning_rate=2e-6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        auxiliary_coefficient=0.1,
        max_grad_norm=0.5,
        epochs=4,
        minibatch_size=32,
    )
    initial_actor_hashes: dict[str, dict[str, str]] = {}
    optimizer_state_entries = {mode: 0 for mode in LAGRANGIAN_MODES}
    for seed in model_seeds:
        initial_actor_hashes[str(seed)] = {}
        for mode in LAGRANGIAN_MODES:
            actor = _new_actor(
                seed,
                use_hud=actor_settings["use_hud"],
                use_visual_features=actor_settings["use_visual_features"],
                throttle_expansion=throttle_expansion,
            )
            initial_actor_hashes[str(seed)][mode] = actor_state_sha256(actor)
            model = actor if mode == "off" else LagrangianActorCritic(actor)
            updater = (
                PPOUpdater(model, config)
                if mode == "off"
                else LagrangianPPOUpdater(model, config)
            )
            state_count = len(updater.optimizer.state)
            if state_count != 0:
                raise ValueError(f"{mode} optimizer is not fresh for seed {seed}")
            optimizer_state_entries[mode] = max(
                optimizer_state_entries[mode], state_count
            )

    for seed, hashes in initial_actor_hashes.items():
        if hashes["off"] != hashes["adaptive"]:
            raise ValueError(f"initial actor hash mismatch for seed {seed}")

    manifest_path = Path(site_map_split).expanduser().resolve()
    train_map_files = sorted({episode.source_path.name for episode in split.train})
    train_map_hashes = {
        episode.source_path.name: hashlib.sha256(
            episode.source_path.read_bytes()
        ).hexdigest()
        for episode in split.train
    }
    default_preflight_path = Path(
        "artifacts/haic/lagrangian-fresh-runtime-architecture-20260925/preflight.json"
    )
    resolved_preflight_path = (
        Path(preflight_report_path).expanduser().resolve()
        if preflight_report_path is not None
        else default_preflight_path.resolve()
    )
    resolved_run_directory = (
        Path(run_directory).expanduser()
        if run_directory is not None
        else Path("artifacts/haic/lagrangian-fresh-runtime-architecture-20260925")
    )
    command_run_directory = str(resolved_run_directory).replace("/", "\\")
    shared_arguments = [
        "--total-steps 8192",
        "--updates 8",
        "--max-decisions 2000",
        "--learning-rate 0.000002",
        "--gamma 0.99",
        "--gae-lambda 0.95",
        "--pedal-expansion 1.0",
        "--seed {seed}",
        "--paired-rollout-seed {seed}",
        f"--teacher-warmup-epochs {teacher_warmup_epochs}",
        f"--teacher-max-decisions {teacher_max_decisions}",
        f"--teacher-warmup-batch-size {teacher_warmup_batch_size}",
        f"--teacher-warmup-learning-rate {teacher_warmup_learning_rate}",
        "--throttle-expansion 3.5",
        "--brake-expansion 1.0",
        "--speed-target 70.0",
        "--speed-shortfall-penalty 0.6",
        "--obstacle-brake-reward 6.0",
        "--curve-brake-reward 0.0",
        "--recovery-clearance-reward 0.0",
        "--disable-recovery-action-rewards",
        "--disable-hud-branch",
        "--enable-visual-features",
        "--train-only-site-map-split \"" + str(manifest_path) + "\"",
        "--preflight-report \"" + str(resolved_preflight_path) + "\"",
        "--defer-tune",
        "--lagrangian-initial-lambda 30",
        "--lagrangian-cost-budget 0.05",
        "--lagrangian-dual-step-size 5",
        "--lagrangian-window-size 8",
        "--lagrangian-min-lambda 0",
        "--lagrangian-max-lambda 60",
        "--lagrangian-cost-gamma 1",
    ]
    runner_commands = [
        {
            "seed": seed,
            "mode": mode,
            "command": (
                ".\\.venv\\Scripts\\python.exe -m training.train_policy "
                f"--output {command_run_directory}\\seed{seed}-{mode} "
                + " ".join(shared_arguments).format(seed=seed)
                + f" --lagrangian-mode {mode}"
            ),
        }
        for seed in model_seeds
        for mode in LAGRANGIAN_MODES
    ]
    return {
        "schema_version": 1,
        "preflight_kind": "paired_lagrangian_ppo_fresh_actor_runtime_architecture",
        "passed": True,
        "execution_started": False,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "run_directory": str(resolved_run_directory),
        "loaded_groups": ["train"],
        "loaded_map_files": train_map_files,
        "train_map_sha256": train_map_hashes,
        "non_train_map_files_opened": [],
        "official_map_files_opened": [],
        "train_cells": [
            {"map_id": map_id, "seed": seed} for map_id, seed in actual_cells
        ],
        "train_episode_count": len(actual_cells),
        "actor_input": "pixel_observation_only",
        "privileged_labels_in_actor_input": False,
        "safety_cost_label_source": "TRAIN simulator labels only",
        "teacher_warmup_epochs": teacher_warmup_epochs,
        "teacher_warmup": {
            "enabled": teacher_warmup_epochs > 0,
            "max_decisions_per_episode": teacher_max_decisions,
            "batch_size": teacher_warmup_batch_size,
            "learning_rate": teacher_warmup_learning_rate,
            "demonstration_source": "TRAIN-only simulator teacher",
            "teacher_loss_used_during_ppo": False,
            "privileged_state_in_actor_input": False,
        },
        "demonstrations_used": 0,
        "initialization_mode": "fresh_seeded_actor",
        "actor_settings": actor_settings,
        "initialize_from": None,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "strict_actor_load": False,
        "source_checkpoint_optimizer_state_used": False,
        "resume": None,
        "optimizer_state_entries": optimizer_state_entries,
        "model_seeds": list(model_seeds),
        "initial_actor_hashes": initial_actor_hashes,
        "paired_actor_hashes_match": True,
        "shared_config": {
            "total_steps": total_steps,
            "updates": updates,
            "learning_rate": config.learning_rate,
            "policy_mean_learning_rate": config.policy_mean_learning_rate,
            "throttle_expansion": throttle_expansion,
            "brake_expansion": DEFAULT_PEDAL_EXPANSION,
            "max_decisions": 2000,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_ratio": config.clip_ratio,
            "entropy_coefficient": config.entropy_coefficient,
            "value_coefficient": config.value_coefficient,
            "auxiliary_coefficient": config.auxiliary_coefficient,
            "max_grad_norm": config.max_grad_norm,
            "epochs": config.epochs,
            "minibatch_size": config.minibatch_size,
            "teacher_warmup_epochs": teacher_warmup_epochs,
            "teacher_max_decisions": teacher_max_decisions,
            "teacher_warmup_batch_size": teacher_warmup_batch_size,
            "teacher_warmup_learning_rate": teacher_warmup_learning_rate,
            "recovery_action_rewards_enabled": False,
            "speed_target": 70.0,
            "speed_shortfall_penalty": 0.6,
            "obstacle_brake_reward": 6.0,
            "curve_brake_reward": 0.0,
            "recovery_clearance_reward": 0.0,
            "use_hud": actor_settings["use_hud"],
            "use_visual_features": actor_settings["use_visual_features"],
            "use_temporal_features": actor_settings["use_temporal_features"],
            "tune_selection": False,
            "resume": None,
            "initialize_from": None,
            "lagrangian_initial_lambda": 30.0,
            "lagrangian_cost_budget": 0.05,
            "lagrangian_dual_step_size": 5.0,
            "lagrangian_window_size": 8,
            "lagrangian_min_lambda": 0.0,
            "lagrangian_max_lambda": 60.0,
            "lagrangian_cost_gamma": 1.0,
        },
        "arms": list(LAGRANGIAN_MODES),
        "only_arm_argument_changed": "--lagrangian-mode",
        "runner_commands": runner_commands,
        "adaptive_config": {
            "initial_lambda": 30.0,
            "cost_budget": 0.05,
            "dual_step_size": 5.0,
            "window_size": 8,
            "min_lambda": 0.0,
            "max_lambda": 60.0,
            "cost_gamma": 1.0,
            "lambda_updates_require_completed_train_episodes": True,
        },
        "deferred_tune": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-map-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=MODEL_SEEDS)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=0)
    parser.add_argument("--teacher-max-decisions", type=int, default=800)
    parser.add_argument("--teacher-warmup-batch-size", type=int, default=32)
    parser.add_argument("--teacher-warmup-learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = build_preflight_report(
        site_map_split=args.site_map_split,
        preflight_report_path=args.output,
        run_directory=args.run_directory,
        model_seeds=tuple(args.model_seeds),
        teacher_warmup_epochs=args.teacher_warmup_epochs,
        teacher_max_decisions=args.teacher_max_decisions,
        teacher_warmup_batch_size=args.teacher_warmup_batch_size,
        teacher_warmup_learning_rate=args.teacher_warmup_learning_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
