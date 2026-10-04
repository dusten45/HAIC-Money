"""Freeze and validate the paired pixel hazard-potential PPO screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from haic_agent.networks import CPU_INFERENCE_THREADS, DEFAULT_PEDAL_EXPANSION, VisualActorCritic
from training.ppo import PPOConfig, PPOUpdater


ROOT = Path(__file__).resolve().parents[1]
REGISTERED_TRAIN_CELLS = (
    ("custom-track-haic-obstacles-20260920", 20260920),
    ("custom-track-haic-obstacles-20260920", 20260924),
    ("custom-track-haic-train-20260921", 20260921),
    ("custom-track-haic-train-20260921", 20260925),
)
MODEL_SEEDS = (8104, 8105)
SCALES = (0.0, 0.10)
WARMUP = {
    "teacher_warmup_epochs": 30,
    "teacher_max_decisions": 800,
    "teacher_warmup_batch_size": 32,
    "teacher_warmup_learning_rate": 1e-3,
}
SOURCE_FILES = (
    "training/train_policy.py",
    "training/rollout.py",
    "training/hazard_potential.py",
    "training/evaluate_hazard_potential_tune.py",
    "training/preflight_hazard_potential_screen.py",
    "haic_agent/networks.py",
    "haic_agent/pixel_features.py",
    "training/site_maps.py",
)


def _scale_label(scale: float) -> str:
    return f"beta-{float(scale):.2f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def shared_config(
    *,
    total_steps: int,
    updates: int,
    max_decisions: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    pedal_expansion: float,
    throttle_expansion: float,
    brake_expansion: float,
    teacher_warmup_epochs: int,
    teacher_max_decisions: int,
    teacher_warmup_batch_size: int,
    teacher_warmup_learning_rate: float,
    speed_target: float,
    speed_shortfall_penalty: float,
    obstacle_brake_reward: float,
    curve_brake_reward: float,
    recovery_clearance_reward: float,
    recovery_action_rewards_enabled: bool,
    use_hud: bool,
    use_visual_features: bool,
    use_temporal_features: bool,
    lagrangian_mode: str,
    lagrangian_initial_lambda: float,
    lagrangian_cost_budget: float,
    lagrangian_dual_step_size: float,
    lagrangian_window_size: int,
    lagrangian_min_lambda: float,
    lagrangian_max_lambda: float,
    lagrangian_cost_gamma: float,
    resume: str | None,
    initialize_from: str | None,
    tune_selection: bool,
) -> dict[str, Any]:
    config = PPOConfig(
        learning_rate=learning_rate,
        policy_mean_learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return {
        "total_steps": int(total_steps),
        "updates": int(updates),
        "max_decisions": int(max_decisions),
        "learning_rate": float(learning_rate),
        "policy_mean_learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_ratio": float(config.clip_ratio),
        "entropy_coefficient": float(config.entropy_coefficient),
        "value_coefficient": float(config.value_coefficient),
        "auxiliary_coefficient": float(config.auxiliary_coefficient),
        "max_grad_norm": float(config.max_grad_norm),
        "epochs": int(config.epochs),
        "minibatch_size": min(32, math.ceil(total_steps / updates)),
        "pedal_expansion": float(pedal_expansion),
        "throttle_expansion": float(throttle_expansion),
        "brake_expansion": float(brake_expansion),
        "teacher_warmup_epochs": int(teacher_warmup_epochs),
        "teacher_max_decisions": int(teacher_max_decisions),
        "teacher_warmup_batch_size": int(teacher_warmup_batch_size),
        "teacher_warmup_learning_rate": float(teacher_warmup_learning_rate),
        "speed_target": float(speed_target),
        "speed_shortfall_penalty": float(speed_shortfall_penalty),
        "obstacle_brake_reward": float(obstacle_brake_reward),
        "curve_brake_reward": float(curve_brake_reward),
        "recovery_clearance_reward": float(recovery_clearance_reward),
        "recovery_action_rewards_enabled": bool(recovery_action_rewards_enabled),
        "use_hud": bool(use_hud),
        "use_visual_features": bool(use_visual_features),
        "use_temporal_features": bool(use_temporal_features),
        "lagrangian_mode": str(lagrangian_mode),
        "lagrangian_initial_lambda": float(lagrangian_initial_lambda),
        "lagrangian_cost_budget": float(lagrangian_cost_budget),
        "lagrangian_dual_step_size": float(lagrangian_dual_step_size),
        "lagrangian_window_size": int(lagrangian_window_size),
        "lagrangian_min_lambda": float(lagrangian_min_lambda),
        "lagrangian_max_lambda": float(lagrangian_max_lambda),
        "lagrangian_cost_gamma": float(lagrangian_cost_gamma),
        "resume": resume,
        "initialize_from": initialize_from,
        "tune_selection": bool(tune_selection),
    }


def frozen_shared_config() -> dict[str, Any]:
    return shared_config(
        total_steps=8192,
        updates=8,
        max_decisions=2000,
        learning_rate=2e-6,
        gamma=0.99,
        gae_lambda=0.95,
        pedal_expansion=DEFAULT_PEDAL_EXPANSION,
        throttle_expansion=3.5,
        brake_expansion=1.0,
        teacher_warmup_epochs=WARMUP["teacher_warmup_epochs"],
        teacher_max_decisions=WARMUP["teacher_max_decisions"],
        teacher_warmup_batch_size=WARMUP["teacher_warmup_batch_size"],
        teacher_warmup_learning_rate=WARMUP["teacher_warmup_learning_rate"],
        speed_target=70.0,
        speed_shortfall_penalty=0.6,
        obstacle_brake_reward=6.0,
        curve_brake_reward=0.0,
        recovery_clearance_reward=0.0,
        recovery_action_rewards_enabled=False,
        use_hud=False,
        use_visual_features=True,
        use_temporal_features=False,
        lagrangian_mode="off",
        lagrangian_initial_lambda=30.0,
        lagrangian_cost_budget=0.05,
        lagrangian_dual_step_size=5.0,
        lagrangian_window_size=8,
        lagrangian_min_lambda=0.0,
        lagrangian_max_lambda=60.0,
        lagrangian_cost_gamma=1.0,
        resume=None,
        initialize_from=None,
        tune_selection=False,
    )


def _source_hashes() -> dict[str, str]:
    return {
        name: _sha256(ROOT / name)
        for name in SOURCE_FILES
    }


def build_preflight_report(
    *,
    site_map_split: Path | str,
    preflight_report_path: Path | str | None = None,
    run_directory: Path | str = "artifacts/haic/ppo-hazard-potential-2seed-u8-20260925/runs",
    model_seeds: tuple[int, ...] = MODEL_SEEDS,
) -> dict[str, Any]:
    """Prove paired fresh initialization and TRAIN-only inputs without running PPO."""
    if tuple(model_seeds) not in (MODEL_SEEDS, (8104,)):
        raise ValueError("the paired screen requires seeds 8104 and 8105; 8104-only is test mode")
    manifest_path = Path(site_map_split).expanduser().resolve()
    from training.train_policy import (
        actor_state_sha256,
        load_train_only_site_map_split,
        set_reproducible_seed,
    )

    split = load_train_only_site_map_split(manifest_path)
    actual_cells = tuple((episode.map_id, int(episode.seed)) for episode in split.train)
    if actual_cells != REGISTERED_TRAIN_CELLS:
        raise ValueError("TRAIN-only manifest differs from the four registered cells")
    if split.tune or split.held_out:
        raise ValueError("preflight must not materialize TUNE or held-out episodes")

    config = frozen_shared_config()
    hashes: dict[str, dict[str, str]] = {}
    optimizer_entries: dict[str, int] = {_scale_label(scale): 0 for scale in SCALES}
    for seed in model_seeds:
        hashes[str(seed)] = {}
        for scale in SCALES:
            torch.set_num_threads(CPU_INFERENCE_THREADS)
            set_reproducible_seed(seed)
            actor = VisualActorCritic(
                use_hud=False,
                use_visual_features=True,
                use_temporal_features=False,
                throttle_expansion=3.5,
                brake_expansion=1.0,
            )
            hashes[str(seed)][_scale_label(scale)] = actor_state_sha256(actor)
            updater = PPOUpdater(
                actor,
                PPOConfig(
                    learning_rate=2e-6,
                    policy_mean_learning_rate=2e-6,
                    gamma=0.99,
                    gae_lambda=0.95,
                    minibatch_size=32,
                ),
            )
            state_count = len(updater.optimizer.state)
            if state_count != 0:
                raise ValueError(f"{_scale_label(scale)} optimizer is not fresh for seed {seed}")
            optimizer_entries[_scale_label(scale)] = max(
                optimizer_entries[_scale_label(scale)], state_count
            )
        if hashes[str(seed)][_scale_label(SCALES[0])] != hashes[str(seed)][_scale_label(SCALES[1])]:
            raise ValueError(f"paired beta arms have different initial actors for seed {seed}")

    train_map_hashes = {
        episode.source_path.name: _sha256(episode.source_path)
        for episode in split.train
    }
    output_path = (
        Path(preflight_report_path).expanduser().resolve()
        if preflight_report_path is not None
        else ROOT / "artifacts/haic/ppo-hazard-potential-2seed-u8-20260925/preflight.json"
    )
    run_path = Path(run_directory).expanduser()
    command_run_directory = _root_relative(run_path)
    command_preflight_path = _root_relative(output_path)
    command_manifest = _root_relative(manifest_path)
    source_hashes = _source_hashes()
    runner_commands = []
    for seed in model_seeds:
        for scale in SCALES:
            label = _scale_label(scale)
            output = f"{command_run_directory}\\seed{seed}-{label}"
            command = (
                ".\\.venv\\Scripts\\python.exe -m training.train_policy "
                f"--output {output} --total-steps 8192 --updates 8 "
                "--max-decisions 2000 --learning-rate 0.000002 --gamma 0.99 "
                "--gae-lambda 0.95 --seed {seed} --paired-rollout-seed {seed} "
                "--pedal-expansion 1.0 "
                "--teacher-warmup-epochs 30 --teacher-max-decisions 800 "
                "--teacher-warmup-batch-size 32 --teacher-warmup-learning-rate 0.001 "
                "--throttle-expansion 3.5 --brake-expansion 1.0 "
                "--speed-target 70.0 --speed-shortfall-penalty 0.6 "
                "--obstacle-brake-reward 6.0 --curve-brake-reward 0.0 "
                "--recovery-clearance-reward 0.0 --disable-recovery-action-rewards "
                "--disable-hud-branch --enable-visual-features "
                f"--hazard-potential-scale {scale:.2f} "
                f"--train-only-site-map-split {command_manifest} "
                f"--hazard-potential-preflight {command_preflight_path} "
                "--defer-tune --lagrangian-mode off"
            ).format(seed=seed)
            runner_commands.append(
                {"seed": seed, "arm": label, "scale": scale, "command": command}
            )

    return {
        "schema_version": 1,
        "preflight_kind": "paired_pixel_hazard_potential_ppo_fresh_actor",
        "passed": True,
        "execution_started": False,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "run_directory": str(run_path),
        "model_seeds": list(model_seeds),
        "arms": [_scale_label(scale) for scale in SCALES],
        "arm_coefficients": {_scale_label(scale): scale for scale in SCALES},
        "initial_actor_hashes": hashes,
        "optimizer_state_entries": optimizer_entries,
        "paired_actor_hashes_match": True,
        "actor_settings": {
            "use_hud": False,
            "use_visual_features": True,
            "use_temporal_features": False,
        },
        "initialization_mode": "fresh_seeded_actor_then_shared_train_only_teacher_warmup",
        "teacher_warmup": {
            "enabled": True,
            **WARMUP,
            "teacher_loss_used_during_ppo": False,
            "privileged_state_in_actor_input": False,
        },
        "shared_config": config,
        "loaded_groups": ["train"],
        "train_cells": [
            {"map_id": map_id, "seed": seed}
            for map_id, seed in actual_cells
        ],
        "loaded_map_files": sorted(train_map_hashes),
        "train_map_sha256": train_map_hashes,
        "non_train_map_files_opened": [],
        "official_map_files_opened": [],
        "source_sha256": source_hashes,
        "runner_commands": runner_commands,
        "tune_or_held_out_opened": False,
    }


def validate_preflight_report(
    *,
    preflight_path: Path | str,
    manifest_path: Path | str,
    split: Any,
    seed: int,
    scale: float,
    shared_config: dict[str, Any],
) -> str:
    """Bind one training arm to the frozen preflight, source, map, and actor hashes."""
    if scale not in SCALES:
        raise ValueError("hazard-potential scale is not a registered paired arm")
    report_path = Path(preflight_path).expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read hazard-potential preflight report: {error}") from error
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("hazard-potential preflight report did not pass")
    if report.get("preflight_kind") != "paired_pixel_hazard_potential_ppo_fresh_actor":
        raise ValueError("preflight report is not for the hazard-potential PPO screen")
    if report.get("execution_started") is not False:
        raise ValueError("preflight report must be non-executing")
    if report.get("loaded_groups") != ["train"]:
        raise ValueError("preflight must load only the TRAIN group")
    if report.get("non_train_map_files_opened") != [] or report.get("official_map_files_opened") != []:
        raise ValueError("preflight opened a non-TRAIN or official map")
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    if str(resolved_manifest) != report.get("manifest"):
        raise ValueError("training map manifest does not match the preflight")
    if _sha256(resolved_manifest) != report.get("manifest_sha256"):
        raise ValueError("training map manifest changed after preflight")
    actual_cells = [
        {"map_id": episode.map_id, "seed": int(episode.seed)}
        for episode in split.train
    ]
    if actual_cells != report.get("train_cells"):
        raise ValueError("loaded TRAIN cells do not match the preflight")
    actual_map_hashes = {
        episode.source_path.name: _sha256(episode.source_path)
        for episode in split.train
    }
    if actual_map_hashes != report.get("train_map_sha256"):
        raise ValueError("a registered TRAIN map changed after preflight")
    if split.tune or split.held_out:
        raise ValueError("paired hazard-potential runs must not load non-TRAIN episodes")
    if report.get("shared_config") != shared_config:
        raise ValueError("run settings do not match the preflight shared run configuration")
    if seed not in report.get("model_seeds", []):
        raise ValueError(f"preflight does not include model seed {seed}")
    label = _scale_label(scale)
    hashes_by_seed = report.get("initial_actor_hashes", {}).get(str(seed), {})
    expected_hash = hashes_by_seed.get(label) if isinstance(hashes_by_seed, dict) else None
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"preflight is missing the actor hash for seed {seed}/{label}")
    if hashes_by_seed.get(_scale_label(SCALES[0])) != hashes_by_seed.get(_scale_label(SCALES[1])):
        raise ValueError(f"paired beta arms do not share an initial actor for seed {seed}")
    if report.get("source_sha256") != _source_hashes():
        raise ValueError("trainer or actor source changed after preflight")
    return expected_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-map-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("artifacts/haic/ppo-hazard-potential-2seed-u8-20260925/runs"),
    )
    parser.add_argument("--model-seeds", type=int, nargs="+", default=list(MODEL_SEEDS))
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight report: {output}")
    report = build_preflight_report(
        site_map_split=args.site_map_split,
        preflight_report_path=output,
        run_directory=args.run_directory,
        model_seeds=tuple(args.model_seeds),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "execution_started": False, "output": _root_relative(output), "arms": report["arms"], "model_seeds": report["model_seeds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
