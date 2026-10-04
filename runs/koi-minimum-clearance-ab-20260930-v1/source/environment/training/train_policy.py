"""Reproducible CPU training entry point for the pixel-only PPO policy."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from haic_agent.networks import (
    CPU_INFERENCE_THREADS,
    DEFAULT_PEDAL_EXPANSION,
    MAX_BRAKE,
    MAX_BRAKE_EXPANSION,
    MAX_GAS,
    MAX_THROTTLE_EXPANSION,
    PedalScale,
    PolicyOutput,
    VisualActorCritic,
)
from haic_agent.pixel_features import FEATURE_NAMES
from training.env_factory import (
    CollectedTransition,
    TrainingEpisode,
    TrainingSplit,
    create_episode_environment,
    create_training_environment,
    describe_episode,
    describe_split,
    split_track_seeds,
)
from training.hazard_potential import hazard_potential_shaping_reward
from training.imitation import behavioral_cloning_warmup, collect_teacher_demonstrations
from training.lagrangian import (
    CostComponents,
    DEFAULT_COST_BUDGET,
    DEFAULT_DUAL_STEP_SIZE,
    DEFAULT_MAX_MULTIPLIER,
    DEFAULT_MIN_MULTIPLIER,
    DEFAULT_WINDOW_SIZE,
    REFERENCE_LAMBDA,
    EpisodeBoundary,
    EpisodeCostAccumulator,
    LagrangeMultiplier,
    compute_transition_cost,
)
from training.lagrangian_ppo import (
    LagrangianActorCritic,
    LagrangianPPOUpdater,
    LagrangianPolicyOutput,
    LagrangianRolloutStorage,
)
from training.ppo import PPOConfig, PPOUpdater
from training.rollout import RolloutStorage
from training.site_maps import (
    SiteMapSplit,
    load_site_map_split,
    load_train_site_map_episodes,
)


DEFAULT_SPLIT = split_track_seeds(
    train=((1, 101), (2, 102)), tune=((3, 201),), held_out=((4, 301),)
)
REWARD_SCALE = 0.1
AUXILIARY_TARGET_SCALES = (40.0, 100.0, 100.0, 100.0, 100.0, 0.4, 3.0, 1.0, 1.0, 1.0)
ROAD_TRACKING_LATERAL_WEIGHT = 0.5
ROAD_TRACKING_HEADING_WEIGHT = 0.5
MAX_NORMALIZED_LATERAL_ERROR = 2.0
TIME_PENALTY = 0.05
# A per-decision bonus proportional to speed sums to a constant over a lap
# (speed x elapsed time is distance, and the lap distance is fixed), so the
# earlier ``SAFE_SPEED_REWARD_WEIGHT`` term expressed no preference at all
# between a fast lap and a slow one. Charging for the shortfall below a target
# cruise speed keeps the same dense per-decision shape while making a slower
# lap strictly more expensive than a faster one.
# Target set from the tracks, not from the actor's current habits. Official
# track lengths measured from the simulator's own tile centres are 962-1197,
# so a 15s lap needs a 64-80 mean speed; the car tops out at 100. The earlier
# 50.0 target sat below the requirement, so a lap driven at 50 paid nothing
# even though it takes ~21s. Deterministic traces on official track 2 seed 101
# show the actor commanding gas 0.071 mean / 0.121 max even when the transform
# allows 1.00, holding a 41 mean speed: the throttle ceiling is not what keeps
# the lap at 25s, the policy's own speed preference is.
SPEED_TARGET = 70.0
SPEED_SHORTFALL_PENALTY = 0.6
# Slowing for a visible hazard should not be charged as if the road were clear,
# but waiving the charge entirely would make idling in front of an obstacle the
# cheapest place on the track. Lowering the target instead keeps braking free
# while still charging a car that stops and stays stopped.
HAZARD_SPEED_RELIEF = 0.5
# Trace evidence (official-track1-seed42): the obstacle that ends every S1
# lineage run sits inside a sharp bend (road_curve_magnitude rising from 0.19
# to 0.58 over the six decisions before impact) where the actor holds a flat
# 41 m/s cruise target and nearly runs off the outside of the turn before the
# obstacle is even visible. A curvature-aware speed target both improves
# generic cornering and, as a side effect, buys reaction time for hazards
# hidden behind blind bends -- a different lever than braking/steering after
# the hazard is already visible.
CURVE_SPEED_RELIEF = 0.4
COLLISION_PENALTY = 60.0
TERMINAL_FAILURE_PENALTY = 60.0
OBSTACLE_THROTTLE_PENALTY = 8.0
OBSTACLE_LATERAL_CLEARANCE_PIXELS = 12.0
# Kahn 2017/2020: penalizing gas near a hazard is not the same as rewarding the
# braking action that actually removes the hazard. Trace evidence from the S1
# obstacle-risk checkpoint (site-map-official-domain-obstacle-risk-ppo8192-...)
# shows brake commands staying under 0.09 even when obstacle_urgency == 1.0, so
# add a direct positive term for braking while hazard_risk is high.
OBSTACLE_BRAKE_REWARD = 6.0
# Evans 2023 + collision trace evidence: after a collision the car often gets
# pinned near the obstacle (speed collapses toward 0 for tens of decisions)
# and progress stays frozen even though damage saturates. Training rarely
# samples this state, so the actor never learns to re-accelerate out of it.
# Reward gas usage specifically while damaged, not currently colliding, and
# nearly stopped so PPO sees a clear signal to escape rather than idle.
RECOVERY_MIN_DAMAGE = 0.15
RECOVERY_MAX_SPEED = 8.0
RECOVERY_MAX_HAZARD_RISK = 0.5
RECOVERY_GAS_REWARD = 5.0
# Trace evidence (official-track1-seed42, brake-reward checkpoint): once pinned
# nose-first against an obstacle, the actor settles into near-zero steer while
# holding gas, which cannot produce a lateral escape force against a head-on
# static body. Reward large steering magnitude in addition to gas while pinned
# so PPO can discover the slip-and-turn escape instead of idling at the wall.
RECOVERY_STEER_REWARD = 4.0


def close_training_environment(environment: Any) -> None:
    """Close either a direct Gym environment or Task 1's collecting wrapper."""
    close = getattr(environment, "close", None)
    if callable(close):
        close()
        return
    environment.environment.close()


def set_reproducible_seed(seed: int) -> None:
    """Set every random source used by the CPU smoke and full training paths."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def auxiliary_targets(transition: CollectedTransition) -> torch.Tensor:
    """Normalize labels for the exact pixels used to choose the action."""
    labels = transition.observation_labels or transition.labels
    half_width = max(float(labels.road_half_width), 1e-6)
    normalized_lateral = float(
        np.clip(float(labels.lateral_error) / half_width, -2.0, 2.0)
    )
    values = (
        labels.speed,
        *labels.wheel_omega,
        labels.steering_angle,
        labels.yaw_rate,
        normalized_lateral,
        math.sin(float(labels.heading_error)),
        math.cos(float(labels.heading_error)),
    )
    return torch.tensor(values, dtype=torch.float32) / torch.tensor(
        AUXILIARY_TARGET_SCALES, dtype=torch.float32
    )


def visible_hazard_risk(feature_values: np.ndarray) -> float:
    """Estimate how much a visible obstacle blocks the car's own lane."""
    obstacle_present = float(np.clip(feature_values[4], 0.0, 1.0))
    obstacle_lateral = float(np.clip(feature_values[5], -1.0, 1.0))
    obstacle_urgency = float(np.clip(feature_values[6], 0.0, 1.0))
    road_center_offset = float(np.clip(feature_values[1], -1.0, 1.0))
    # Pixel features use different normalizations: obstacle offset spans
    # 12 px, while road-center offset spans 42 px. Convert both to pixels
    # before estimating whether the visible obstacle intersects the car's
    # current lane.
    obstacle_offset_pixels = obstacle_lateral * OBSTACLE_LATERAL_CLEARANCE_PIXELS
    car_offset_from_road_center = -road_center_offset * 42.0
    relative_obstacle_offset_pixels = (
        obstacle_offset_pixels - car_offset_from_road_center
    )
    lateral_distance = abs(relative_obstacle_offset_pixels)
    lane_overlap = max(0.0, 1.0 - lateral_distance / OBSTACLE_LATERAL_CLEARANCE_PIXELS)
    return obstacle_present * obstacle_urgency**2 * lane_overlap


def summarize_hazard_potential_transitions(
    transitions: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    """Summarize visible-risk changes on transitions seen by the PPO rollout."""
    exposed = [item for item in transitions if float(item["current_risk"]) > 0.0]
    urgent = [item for item in transitions if float(item["current_risk"]) >= 0.8]

    def decreased_fraction(items: list[dict[str, Any]]) -> float | None:
        if not items:
            return None
        return sum(
            float(item["next_risk"]) < float(item["current_risk"])
            for item in items
        ) / len(items)

    return {
        "transition_count": len(transitions),
        "risk_exposed_transition_count": len(exposed),
        "risk_decreased_fraction": decreased_fraction(exposed),
        "urgent_transition_count": len(urgent),
        "urgent_risk_decreased_fraction": decreased_fraction(urgent),
        "mean_urgent_risk_delta": (
            float(
                np.mean(
                    [
                        float(item["next_risk"]) - float(item["current_risk"])
                        for item in urgent
                    ]
                )
            )
            if urgent
            else None
        ),
        "urgent_collision_count": sum(bool(item["collision"]) for item in urgent),
        "mean_shaping_reward": (
            float(np.mean([float(item["shaping_reward"]) for item in transitions]))
            if transitions
            else None
        ),
    }


def _visual_feature_values(
    visual_features: np.ndarray | torch.Tensor | None,
) -> np.ndarray | None:
    if visual_features is None:
        return None
    if isinstance(visual_features, torch.Tensor):
        feature_values = visual_features.detach().cpu().numpy()
    else:
        feature_values = np.asarray(visual_features, dtype=np.float32)
    if feature_values.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"visual_features must have shape ({len(FEATURE_NAMES)},)")
    return feature_values


def shape_transition_reward(
    transition: CollectedTransition,
    previous_progress: float,
    *,
    previous_speed: float = 0.0,
    gamma: float = PPOConfig.gamma,
    hazard_potential_scale: float = 0.0,
    visual_features: np.ndarray | torch.Tensor | None = None,
    next_visual_features: np.ndarray | torch.Tensor | None = None,
    throttle_limit: float = MAX_GAS,
    brake_limit: float = MAX_BRAKE,
    obstacle_brake_reward: float = OBSTACLE_BRAKE_REWARD,
    curve_brake_reward: float = 0.0,
    direct_gas_reward: float = 0.0,
    recovery_clearance_reward: float = 0.0,
    enable_recovery_action_rewards: bool = True,
    safety_costs_separate: bool = False,
) -> float:
    """Shape PPO feedback for progress, speed, safety, and completion."""
    if not np.isfinite(obstacle_brake_reward) or obstacle_brake_reward < 0.0:
        raise ValueError("obstacle_brake_reward must be finite and non-negative")
    if not np.isfinite(curve_brake_reward) or curve_brake_reward < 0.0:
        raise ValueError("curve_brake_reward must be finite and non-negative")
    if not np.isfinite(direct_gas_reward) or direct_gas_reward < 0.0:
        raise ValueError("direct_gas_reward must be finite and non-negative")
    if not np.isfinite(recovery_clearance_reward) or recovery_clearance_reward < 0.0:
        raise ValueError("recovery_clearance_reward must be finite and non-negative")
    if not np.isfinite(hazard_potential_scale) or hazard_potential_scale < 0.0:
        raise ValueError("hazard_potential_scale must be finite and non-negative")
    labels = transition.labels
    simulator_reward = float(transition.reward)
    if (
        safety_costs_separate
        and bool(getattr(transition, "simulator_out_of_bounds_failure", False))
        and not labels.finished
    ):
        # CarRacing replaces its final out-of-bounds reward with -100. Undo
        # that simulator failure penalty so the Lagrangian failure cost is the
        # only direct failure signal in this mode.
        simulator_reward += 100.0
    new_road = max(0.0, labels.tile_progress - previous_progress)
    feature_values = _visual_feature_values(visual_features)
    next_feature_values = _visual_feature_values(next_visual_features)
    hazard_risk = (
        visible_hazard_risk(feature_values) if feature_values is not None else 0.0
    )
    curve_magnitude = (
        float(np.clip(feature_values[3], 0.0, 1.0))
        if feature_values is not None
        else 0.0
    )
    speed_shortfall_cost = 0.0
    if not labels.collision and not labels.off_track:
        target_speed = SPEED_TARGET * (
            1.0 - HAZARD_SPEED_RELIEF * hazard_risk
        ) * (1.0 - CURVE_SPEED_RELIEF * curve_magnitude)
        shortfall = max(0.0, target_speed - max(float(labels.speed), 0.0)) / SPEED_TARGET
        speed_shortfall_cost = SPEED_SHORTFALL_PENALTY * shortfall
    shaped = (
        simulator_reward
        + 10.0 * new_road
        - TIME_PENALTY
        + route_tracking_reward(labels)
        - speed_shortfall_cost
    )
    if feature_values is not None:
        gas_fraction = float(
            np.clip(float(transition.action[1]) / throttle_limit, 0.0, 1.0)
        )
        brake_fraction = float(
            np.clip(float(transition.action[2]) / brake_limit, 0.0, 1.0)
        )
        shaped -= OBSTACLE_THROTTLE_PENALTY * hazard_risk * gas_fraction
        shaped += obstacle_brake_reward * hazard_risk * brake_fraction
        shaped += curve_brake_reward * curve_magnitude * brake_fraction
        if direct_gas_reward > 0.0 and not labels.collision and not labels.off_track:
            half_width = max(float(labels.road_half_width), 1e-6)
            road_alignment = max(0.0, 1.0 - abs(float(labels.lateral_error)) / half_width)
            heading_alignment = max(0.0, 1.0 - abs(float(labels.heading_error)) / 0.35)
            clear_road = (
                road_alignment
                * heading_alignment
                * (1.0 - curve_magnitude) ** 2
                * (1.0 - hazard_risk)
            )
            shaped += direct_gas_reward * clear_road * gas_fraction * shortfall
    if (
        not safety_costs_separate
        and
        recovery_clearance_reward > 0.0
        and feature_values is not None
        and next_feature_values is not None
        and labels.damage >= RECOVERY_MIN_DAMAGE
        and not labels.collision
        and not labels.off_track
        and previous_speed <= RECOVERY_MAX_SPEED
        and hazard_risk >= RECOVERY_MAX_HAZARD_RISK
    ):
        next_hazard_risk = visible_hazard_risk(next_feature_values)
        clearance_gain = max(0.0, hazard_risk - next_hazard_risk)
        shaped += recovery_clearance_reward * clearance_gain
    if labels.finished:
        shaped += 100.0
    if (
        not safety_costs_separate
        and labels.collision
    ):
        shaped -= COLLISION_PENALTY
    if not safety_costs_separate and labels.off_track:
        shaped -= 10.0
    if not safety_costs_separate:
        shaped -= 0.1 * labels.damage
    if (
        not safety_costs_separate
        and enable_recovery_action_rewards
        and
        labels.damage >= RECOVERY_MIN_DAMAGE
        and not labels.collision
        and previous_speed <= RECOVERY_MAX_SPEED
        and hazard_risk < RECOVERY_MAX_HAZARD_RISK
    ):
        gas_fraction = float(
            np.clip(float(transition.action[1]) / throttle_limit, 0.0, 1.0)
        )
        shaped += RECOVERY_GAS_REWARD * gas_fraction
        steer_magnitude = float(np.clip(abs(float(transition.action[0])), 0.0, 1.0))
        shaped += RECOVERY_STEER_REWARD * steer_magnitude * gas_fraction
    if (
        not safety_costs_separate
        and bool(getattr(transition, "terminated", False))
        and not labels.finished
    ):
        shaped -= TERMINAL_FAILURE_PENALTY
    reward = shaped * REWARD_SCALE
    if hazard_potential_scale > 0.0:
        if feature_values is None or next_feature_values is None:
            raise ValueError(
                "hazard-potential shaping requires current and next pixel features"
            )
        reward += hazard_potential_shaping_reward(
            current_risk=hazard_risk,
            next_risk=visible_hazard_risk(next_feature_values),
            gamma=gamma,
            scale=hazard_potential_scale,
            terminated=bool(getattr(transition, "terminated", False)),
            truncated=bool(getattr(transition, "truncated", False)),
        )
    return reward


def route_tracking_reward(labels: Any) -> float:
    """Give PPO dense training feedback for route alignment and road centering."""
    half_width = max(float(labels.road_half_width), 1e-6)
    normalized_lateral_error = float(
        np.clip(
            float(labels.lateral_error) / half_width,
            -MAX_NORMALIZED_LATERAL_ERROR,
            MAX_NORMALIZED_LATERAL_ERROR,
        )
    )
    heading_error = float(labels.heading_error)
    return -ROAD_TRACKING_LATERAL_WEIGHT * normalized_lateral_error**2 - (
        ROAD_TRACKING_HEADING_WEIGHT * (1.0 - math.cos(heading_error))
    )


def _create_environment_for_episode(episode: TrainingEpisode, *, max_decisions: int):
    if isinstance(episode, tuple):
        track_id, seed = episode
        return create_training_environment(
            track_id=int(track_id), seed=int(seed), max_decisions=max_decisions
        )
    return create_episode_environment(episode, max_decisions=max_decisions)


def interleave_training_episodes(
    episodes: Iterable[TrainingEpisode],
) -> tuple[TrainingEpisode, ...]:
    """Shuffle within domains, then round-robin maps and official tracks."""
    groups: dict[tuple[str, int | None], list[TrainingEpisode]] = {}
    for episode in episodes:
        if isinstance(episode, tuple):
            group_key = ("official", int(episode[0]))
        elif episode.site_map.map_kind == "official":
            group_key = ("official", int(episode.site_map.track_id))
        else:
            group_key = (episode.site_map.map_kind, None)
        groups.setdefault(group_key, []).append(episode)

    for group in groups.values():
        random.shuffle(group)

    ordered: list[TrainingEpisode] = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.pop(0))
    return tuple(ordered)


def load_train_only_site_map_split(path: Path | str) -> SiteMapSplit:
    """Load TRAIN map files without opening TUNE or held-out map paths."""
    train_episodes = load_train_site_map_episodes(path)
    if not train_episodes:
        raise ValueError("TRAIN-only map split must contain at least one episode")
    return SiteMapSplit(train=train_episodes, tune=(), held_out=())


def actor_state_sha256(actor: VisualActorCritic) -> str:
    """Hash an actor's named tensor state for paired initialization checks."""
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_lagrangian_screen_preflight(
    *,
    preflight_path: Path | str,
    manifest_path: Path | str,
    split: SiteMapSplit,
    seed: int,
    mode: str,
    shared_config: dict[str, Any],
) -> str:
    """Bind one arm to the passing TRAIN-only preflight and its actor hash."""
    report_path = Path(preflight_path).expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read Lagrangian preflight report: {error}") from error
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("Lagrangian preflight report did not pass")
    if report.get("execution_started") is not False:
        raise ValueError("Lagrangian preflight report must be non-executing")
    if report.get("loaded_groups") != ["train"]:
        raise ValueError("preflight must load only the TRAIN group")
    if report.get("non_train_map_files_opened") != [] or report.get(
        "official_map_files_opened"
    ) != []:
        raise ValueError("preflight opened a non-TRAIN or official map file")
    if mode not in report.get("arms", ()):
        raise ValueError(f"preflight does not include the requested {mode!r} arm")

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    if str(resolved_manifest) != report.get("manifest"):
        raise ValueError("training map manifest does not match the preflight")
    manifest_hash = hashlib.sha256(resolved_manifest.read_bytes()).hexdigest()
    if manifest_hash != report.get("manifest_sha256"):
        raise ValueError("training map manifest changed after the preflight")

    actual_cells = [
        {"map_id": episode.map_id, "seed": int(episode.seed)}
        for episode in split.train
    ]
    if actual_cells != report.get("train_cells"):
        raise ValueError("loaded TRAIN cells do not match the preflight")
    actual_map_hashes = {
        episode.source_path.name: hashlib.sha256(
            episode.source_path.read_bytes()
        ).hexdigest()
        for episode in split.train
    }
    if actual_map_hashes != report.get("train_map_sha256"):
        raise ValueError("a registered TRAIN map changed after the preflight")
    if split.tune or split.held_out:
        raise ValueError("paired Lagrangian runs must not load TUNE or held-out episodes")

    configured = report.get("shared_config")
    if not isinstance(configured, dict):
        raise ValueError("preflight is missing its shared run configuration")
    for name, actual in shared_config.items():
        if configured.get(name) != actual:
            raise ValueError(
                f"run setting {name!r} does not match the passing preflight"
            )
    if seed not in report.get("model_seeds", ()):
        raise ValueError(f"preflight does not include model seed {seed}")
    hashes = report.get("initial_actor_hashes", {}).get(str(seed), {})
    expected_hash = hashes.get(mode) if isinstance(hashes, dict) else None
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"preflight is missing the initial actor hash for seed {seed}/{mode}")
    return expected_hash


class _EpisodeOutcomeAccumulator:
    """Keep comparable TRAIN-only driving outcomes for both PPO arms."""

    def __init__(self, episode: TrainingEpisode) -> None:
        self.episode = describe_episode(episode)
        self.decisions = 0
        self.speeds: list[float] = []
        self.actions: list[tuple[float, float, float]] = []
        self.collision_decisions = 0
        self.collision_onsets = 0
        self.off_track_events = 0
        self.previous_collision = False
        self.progress_start: float | None = None
        self.progress_end = 0.0
        self.progress_max = 0.0
        self.damage_start: float | None = None
        self.damage_final = 0.0
        self.damage_max = 0.0
        self.finish_time_s: float | None = None
        self.cost_components = CostComponents()

    def add(self, transition: CollectedTransition, cost: CostComponents) -> None:
        labels = transition.labels
        previous = transition.observation_labels
        if self.decisions == 0:
            self.progress_start = float(
                previous.tile_progress if previous is not None else labels.tile_progress
            )
            self.damage_start = float(
                previous.damage if previous is not None else labels.damage
            )
        self.decisions += 1
        self.speeds.append(float(labels.speed))
        action = np.asarray(transition.action, dtype=np.float64)
        self.actions.append(tuple(float(value) for value in action[:3]))
        collision = bool(labels.collision)
        self.collision_decisions += int(collision)
        self.collision_onsets += int(collision and not self.previous_collision)
        self.previous_collision = collision
        self.off_track_events += int(bool(labels.off_track))
        self.progress_end = float(labels.tile_progress)
        self.progress_max = max(self.progress_max, self.progress_end)
        self.damage_final = float(labels.damage)
        self.damage_max = max(self.damage_max, self.damage_final)
        self.finish_time_s = transition.finish_time_s
        self.cost_components = self.cost_components + cost

    def record(
        self,
        transition: CollectedTransition,
        boundary: EpisodeBoundary,
        *,
        interrupted: bool = False,
    ) -> dict[str, Any]:
        labels = transition.labels
        if labels.finished:
            boundary_name = "finished"
        elif boundary.terminated:
            if transition.simulator_out_of_bounds_failure:
                boundary_name = "out_of_bounds"
            elif labels.off_track:
                boundary_name = "off_track"
            elif labels.damage >= 1.0:
                boundary_name = "damage_cap"
            else:
                boundary_name = "environment_terminated"
        elif boundary.environment_truncated:
            boundary_name = "environment_truncated"
        elif boundary.episode_time_limit:
            boundary_name = "episode_time_limit"
        elif boundary.collector_only:
            boundary_name = "collector_cutoff"
        elif interrupted:
            boundary_name = "collector_interrupted"
        else:
            boundary_name = "unknown"

        components = {
            "contact_cost": self.cost_components.contact_cost,
            "damage_stock_cost": self.cost_components.damage_stock_cost,
            "off_track_cost": self.cost_components.off_track_cost,
            "failure_cost": self.cost_components.failure_cost,
        }
        actions = np.asarray(self.actions, dtype=np.float64).reshape((-1, 3))
        speeds = np.asarray(self.speeds, dtype=np.float64)
        return {
            "episode": self.episode,
            "episode_end_observed": boundary.episode_done,
            "finished": bool(labels.finished),
            "boundary": boundary_name,
            "finish_time_s": self.finish_time_s,
            "decisions": self.decisions,
            "progress_start": self.progress_start,
            "progress_end": self.progress_end,
            "progress_delta": self.progress_end - (self.progress_start or 0.0),
            "progress_max": self.progress_max,
            "collision_decisions": self.collision_decisions,
            "collision_onsets": self.collision_onsets,
            "off_track_events": self.off_track_events,
            "damage_start": self.damage_start,
            "damage_final": self.damage_final,
            "damage_max": self.damage_max,
            "mean_speed": float(speeds.mean()) if speeds.size else None,
            "max_speed": float(speeds.max()) if speeds.size else None,
            "steer_mean": float(actions[:, 0].mean()) if actions.size else None,
            "gas_mean": float(actions[:, 1].mean()) if actions.size else None,
            "brake_mean": float(actions[:, 2].mean()) if actions.size else None,
            "cost_components": components,
            "total_cost": self.cost_components.total_cost,
        }


def collect_rollout(
    model: VisualActorCritic | LagrangianActorCritic,
    episodes: Iterable[TrainingEpisode],
    *,
    total_steps: int,
    max_decisions: int,
    gamma: float = PPOConfig.gamma,
    cost_gamma: float = 1.0,
    gae_lambda: float = PPOConfig.gae_lambda,
    lagrangian_mode: str = "off",
    obstacle_brake_reward: float = OBSTACLE_BRAKE_REWARD,
    curve_brake_reward: float = 0.0,
    direct_gas_reward: float = 0.0,
    recovery_clearance_reward: float = 0.0,
    hazard_potential_scale: float = 0.0,
    enable_recovery_action_rewards: bool = True,
    reuse_next_policy_output: bool = True,
    hazard_exposure_recorder: Any | None = None,
    update_number: int = 0,
) -> RolloutStorage:
    """Collect a fixed number of four-raw-tick training transitions on CPU."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    if lagrangian_mode not in ("off", "fixed", "adaptive"):
        raise ValueError("lagrangian_mode must be 'off', 'fixed', or 'adaptive'")
    use_lagrangian = lagrangian_mode != "off"
    if use_lagrangian and not isinstance(model, LagrangianActorCritic):
        raise TypeError("Lagrangian rollout collection requires LagrangianActorCritic")
    if not use_lagrangian and isinstance(model, LagrangianActorCritic):
        raise TypeError("standard PPO rollout collection requires VisualActorCritic")
    if not np.isfinite(obstacle_brake_reward) or obstacle_brake_reward < 0.0:
        raise ValueError("obstacle_brake_reward must be finite and non-negative")
    if not np.isfinite(direct_gas_reward) or direct_gas_reward < 0.0:
        raise ValueError("direct_gas_reward must be finite and non-negative")
    if not np.isfinite(recovery_clearance_reward) or recovery_clearance_reward < 0.0:
        raise ValueError("recovery_clearance_reward must be finite and non-negative")
    if not np.isfinite(hazard_potential_scale) or hazard_potential_scale < 0.0:
        raise ValueError("hazard_potential_scale must be finite and non-negative")
    throttle_limit = float(getattr(model, "throttle_limit", MAX_GAS))
    brake_limit = float(getattr(model, "brake_limit", MAX_BRAKE))
    episode_list = interleave_training_episodes(episodes)
    if not episode_list:
        raise ValueError("at least one train episode is required")
    storage = LagrangianRolloutStorage() if use_lagrangian else RolloutStorage()
    storage.hazard_potential_transitions = []
    storage_speeds: list[float] = []
    episode_index = 0
    while len(storage) < total_steps:
        episode = episode_list[episode_index % len(episode_list)]
        episode_index += 1
        environment = _create_environment_for_episode(episode, max_decisions=max_decisions)
        observation, _ = environment.reset()
        previous_progress = 0.0
        previous_damage = 0.0
        previous_speed = 0.0
        recorder_active = False
        episode_cost_accumulator = EpisodeCostAccumulator()
        episode_outcome = _EpisodeOutcomeAccumulator(episode)
        outcome_recorded = False
        try:
            if hazard_exposure_recorder is not None:
                hazard_exposure_recorder.begin_episode(
                    update_number=update_number,
                    episode_number=episode_index - 1,
                    episode=describe_episode(episode),
                )
                recorder_active = True
            cached_output: PolicyOutput | LagrangianPolicyOutput | None = None
            cached_visual_features: torch.Tensor | None = None
            for decision_index in range(max_decisions):
                observation_tensor = torch.from_numpy(observation).unsqueeze(0)
                with torch.no_grad():
                    if reuse_next_policy_output and cached_output is not None:
                        output = cached_output
                        visual_features = cached_visual_features
                        cached_output = None
                        cached_visual_features = None
                    else:
                        output = model(observation_tensor)
                        visual_features = getattr(model, "last_visual_features", None)
                    action, log_probability, pretransform_action = model.sample_actions_with_pretransform(output)
                if visual_features is not None:
                    visual_features = visual_features.squeeze(0).detach()
                transition = environment.step_transition(action.squeeze(0).cpu().numpy())
                if hazard_exposure_recorder is not None:
                    feature_values = _visual_feature_values(visual_features)
                    observation_labels = transition.observation_labels
                    hazard_exposure_recorder.record_decision(
                        progress=(
                            float(observation_labels.tile_progress)
                            if observation_labels is not None
                            else previous_progress
                        ),
                        damage_delta=float(transition.labels.damage)
                        - float(
                            observation_labels.damage
                            if observation_labels is not None
                            else previous_damage
                        ),
                        features=feature_values,
                        risk=(
                            visible_hazard_risk(feature_values)
                            if feature_values is not None
                            else None
                        ),
                        action=transition.action,
                        next_collision=bool(transition.labels.collision),
                        next_progress=float(transition.labels.tile_progress),
                        pre_speed=float(
                            observation_labels.speed
                            if observation_labels is not None
                            else previous_speed
                        ),
                        next_speed=float(transition.labels.speed),
                        next_damage=float(transition.labels.damage),
                        throttle_limit=throttle_limit,
                        brake_limit=brake_limit,
                    )
                with torch.no_grad():
                    next_output = model(
                        torch.from_numpy(transition.next_observation).unsqueeze(0)
                    )
                    next_value = next_output.value.item()
                    next_cost_value = (
                        next_output.cost_value.item() if use_lagrangian else 0.0
                    )
                next_visual_features = getattr(model, "last_visual_features", None)
                if next_visual_features is not None:
                    next_visual_features = next_visual_features.squeeze(0).detach()
                episode_decision_cap = decision_index + 1 >= max_decisions
                rollout_step_cap = len(storage) + 1 >= total_steps
                ended_by_environment = transition.terminated or transition.truncated
                episode_time_limit = not ended_by_environment and episode_decision_cap
                collector_truncated = (
                    not ended_by_environment
                    and not episode_time_limit
                    and rollout_step_cap
                )
                boundary = EpisodeBoundary(
                    terminated=bool(transition.terminated),
                    environment_truncated=bool(transition.truncated),
                    episode_time_limit=episode_time_limit,
                    collector_truncated=collector_truncated,
                )
                rollout_truncated = (
                    transition.truncated
                    or boundary.episode_time_limit
                    or boundary.collector_only
                )
                if (
                    reuse_next_policy_output
                    and not transition.terminated
                    and not rollout_truncated
                ):
                    cached_output = next_output
                    cached_visual_features = next_visual_features
                else:
                    cached_output = None
                    cached_visual_features = None
                shaped_reward = shape_transition_reward(
                    transition,
                    previous_progress,
                    previous_speed=float(
                        (transition.observation_labels or transition.labels).speed
                    ),
                    gamma=gamma,
                    hazard_potential_scale=hazard_potential_scale,
                    visual_features=visual_features,
                    next_visual_features=next_visual_features,
                    throttle_limit=throttle_limit,
                    brake_limit=brake_limit,
                    obstacle_brake_reward=obstacle_brake_reward,
                    curve_brake_reward=curve_brake_reward,
                    direct_gas_reward=direct_gas_reward,
                    recovery_clearance_reward=recovery_clearance_reward,
                    enable_recovery_action_rewards=enable_recovery_action_rewards,
                    safety_costs_separate=use_lagrangian,
                )
                current_features = _visual_feature_values(visual_features)
                next_features = _visual_feature_values(next_visual_features)
                if current_features is not None and next_features is not None:
                    current_risk = visible_hazard_risk(current_features)
                    next_risk = visible_hazard_risk(next_features)
                    potential_reward = hazard_potential_shaping_reward(
                        current_risk=current_risk,
                        next_risk=next_risk,
                        gamma=gamma,
                        scale=hazard_potential_scale,
                        terminated=bool(transition.terminated),
                        truncated=bool(transition.truncated),
                    )
                    storage.hazard_potential_transitions.append(
                        {
                            "current_risk": current_risk,
                            "next_risk": next_risk,
                            "shaping_reward": potential_reward,
                            "collision": bool(transition.labels.collision),
                        }
                    )
                targets = auxiliary_targets(transition)
                observation_labels = transition.observation_labels
                cost_components = compute_transition_cost(
                    damage_before=float(
                        observation_labels.damage
                        if observation_labels is not None
                        else previous_damage
                    ),
                    damage_after=float(transition.labels.damage),
                    collision=bool(transition.labels.collision),
                    off_track=bool(transition.labels.off_track),
                    finished=bool(transition.labels.finished),
                    boundary=boundary,
                )
                completed_cost = episode_cost_accumulator.add_transition(
                    cost_components, boundary, split="train"
                )
                episode_outcome.add(transition, cost_components)
                if use_lagrangian:
                    if completed_cost is not None:
                        storage.completed_episode_costs.append(completed_cost)
                    storage.add(
                        observation=observation_tensor.squeeze(0),
                        action=action.squeeze(0),
                        pretransform_action=pretransform_action.squeeze(0),
                        old_log_probability=log_probability.item(),
                        reward=shaped_reward,
                        reward_value=output.value.item(),
                        next_reward_value=next_value,
                        cost=cost_components.total_cost,
                        cost_value=output.cost_value.item(),
                        next_cost_value=next_cost_value,
                        boundary=boundary,
                        auxiliary_targets=targets,
                    )
                else:
                    storage.add(
                        observation=observation_tensor.squeeze(0),
                        action=action.squeeze(0),
                        pretransform_action=pretransform_action.squeeze(0),
                        log_probability=log_probability.item(),
                        value=output.value.item(),
                        next_value=next_value,
                        reward=shaped_reward,
                        terminated=transition.terminated,
                        truncated=rollout_truncated,
                        auxiliary_targets=targets,
                    )
                previous_progress = transition.labels.tile_progress
                previous_damage = transition.labels.damage
                previous_speed = transition.labels.speed
                storage_speeds.append(float(transition.labels.speed))
                observation = transition.next_observation
                if transition.terminated or rollout_truncated:
                    outcome = episode_outcome.record(transition, boundary)
                    if boundary.episode_done:
                        storage.train_episode_outcomes.append(outcome)
                    else:
                        storage.incomplete_train_episode_outcomes.append(outcome)
                    outcome_recorded = True
                    if hazard_exposure_recorder is not None:
                        if transition.terminated:
                            recorder_boundary = "terminated"
                        elif transition.truncated:
                            recorder_boundary = "environment_truncated"
                        elif decision_index + 1 >= max_decisions:
                            recorder_boundary = "max_decisions"
                        else:
                            recorder_boundary = "rollout_cutoff"
                        recorder_active = False
                        hazard_exposure_recorder.end_episode(
                            boundary=recorder_boundary,
                            end_progress=float(transition.labels.tile_progress),
                            terminated=bool(transition.terminated),
                            truncated=bool(rollout_truncated),
                            finished=bool(transition.labels.finished),
                            off_track=bool(transition.labels.off_track),
                        )
                    break
        finally:
            if recorder_active:
                hazard_exposure_recorder.end_episode(
                    boundary="collector_interrupted",
                    end_progress=float(previous_progress),
                    terminated=False,
                    truncated=True,
                    finished=False,
                    off_track=False,
                )
            incomplete_cost = episode_cost_accumulator.discard_partial()
            if incomplete_cost is not None:
                if use_lagrangian:
                    storage.incomplete_episode_costs.append(incomplete_cost)
                if not outcome_recorded and episode_outcome.decisions:
                    interrupted_boundary = EpisodeBoundary(collector_truncated=True)
                    storage.incomplete_train_episode_outcomes.append(
                        episode_outcome.record(
                            transition,
                            interrupted_boundary,
                            interrupted=True,
                        )
                    )
            close_training_environment(environment)
    if use_lagrangian:
        storage.compute_returns_and_advantages(
            reward_gamma=gamma, cost_gamma=cost_gamma, gae_lambda=gae_lambda
        )
    else:
        storage.compute_returns_and_advantages(gamma=gamma, gae_lambda=gae_lambda)
    storage.rollout_speeds = tuple(storage_speeds)
    return storage


def save_checkpoint(
    path: Path,
    model: VisualActorCritic | LagrangianActorCritic,
    updater: PPOUpdater | LagrangianPPOUpdater,
    *,
    step: int,
    metadata: dict[str, Any],
    training_state: dict[str, Any] | None = None,
) -> None:
    """Save a resumable CPU checkpoint under the configured artifacts path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": updater.optimizer.state_dict(),
        "step": int(step),
        "metadata": metadata,
    }
    if training_state is not None:
        checkpoint["training_state"] = training_state
    torch.save(checkpoint, path)


def save_actor_snapshot(
    path: Path,
    model: VisualActorCritic | LagrangianActorCritic,
    *,
    step: int,
    metadata: dict[str, Any],
) -> None:
    """Save an inference-only actor snapshot without optimizer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    actor_state_dict = getattr(model, "actor_state_dict", None)
    inference_state = (
        actor_state_dict()
        if callable(actor_state_dict)
        else {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    )
    torch.save(
        {
            "model_state": inference_state,
            "step": int(step),
            "metadata": {**metadata, "actor_snapshot_only": True},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: VisualActorCritic | LagrangianActorCritic,
    updater: PPOUpdater | LagrangianPPOUpdater,
    *,
    required_training_state: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint without changing the fixed inference model contract."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if required_training_state is not None and required_training_state not in checkpoint.get(
        "training_state", {}
    ):
        raise ValueError(
            f"resume checkpoint lacks {required_training_state}; use --initialize-from "
            "for actor-only weights"
        )
    model.load_state_dict(checkpoint["model_state"])
    updater.optimizer.load_state_dict(checkpoint["optimizer_state"])
    return {
        "step": int(checkpoint["step"]),
        "metadata": dict(checkpoint["metadata"]),
        "training_state": dict(checkpoint.get("training_state", {})),
    }


def load_actor_weights(path: Path, model: VisualActorCritic) -> dict[str, Any]:
    """Strictly initialize model weights without restoring optimizer or step state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state"), dict
    ):
        raise ValueError("actor checkpoint must contain a model_state mapping")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("actor checkpoint metadata must be a mapping")
    for name in ("use_hud", "use_visual_features", "use_temporal_features"):
        if name in metadata and bool(metadata[name]) is not bool(getattr(model, name)):
            raise ValueError(f"actor checkpoint {name} setting does not match the requested actor")
    symmetric_expansion = metadata.get("pedal_expansion", DEFAULT_PEDAL_EXPANSION)
    requested_scale = getattr(model, "pedal_scale", PedalScale())
    for field, stored_expansion in (
        ("throttle_expansion", metadata.get("throttle_expansion", symmetric_expansion)),
        ("brake_expansion", metadata.get("brake_expansion", symmetric_expansion)),
    ):
        if float(stored_expansion) > getattr(requested_scale, field) + 1e-9:
            raise ValueError(
                f"actor checkpoint {field} is larger than the requested actor; "
                "narrowing the pedal range would silently change its actions"
            )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return {
        "checkpoint_step_metadata": int(checkpoint.get("step", 0)),
        "optimizer_state_used": False,
        "strict_model_state_load": True,
        "source_metadata": metadata,
    }


def save_best_checkpoint(
    path: Path,
    model: VisualActorCritic | LagrangianActorCritic,
    updater: PPOUpdater | LagrangianPPOUpdater,
    *,
    step: int,
    tune_metrics: dict[str, float | None],
    metadata: dict[str, Any],
    training_state: dict[str, Any] | None = None,
) -> bool:
    """Persist a candidate only when its tune rank beats the stored checkpoint."""
    if path.is_file():
        stored = torch.load(path, map_location="cpu")
        stored_metrics = stored.get("metadata", {}).get("tune_metrics")
        if stored_metrics is not None and selection_score(tune_metrics) <= selection_score(stored_metrics):
            return False
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata["tune_metrics"] = tune_metrics
    save_kwargs = (
        {} if training_state is None else {"training_state": training_state}
    )
    save_checkpoint(
        path,
        model,
        updater,
        step=step,
        metadata=checkpoint_metadata,
        **save_kwargs,
    )
    return True


def evaluate_policy(
    model: VisualActorCritic,
    episodes: Iterable[TrainingEpisode],
    *,
    max_decisions: int,
) -> dict[str, float | None]:
    """Measure tune/held-out driving success; completion rate is the first metric."""
    torch.set_num_threads(CPU_INFERENCE_THREADS)
    results: list[dict[str, float | bool | None]] = []
    for episode in episodes:
        environment = _create_environment_for_episode(episode, max_decisions=max_decisions)
        observation, _ = environment.reset()
        start_simulation_time = float(getattr(environment.unwrapped, "t", 0.0))
        total_reward = 0.0
        speeds: list[float] = []
        auxiliary_squared_errors: list[float] = []
        final_transition: CollectedTransition | None = None
        finish_time: float | None = None
        try:
            for _ in range(max_decisions):
                with torch.no_grad():
                    policy_output = model(torch.from_numpy(observation).unsqueeze(0))
                    action = model.deterministic_actions(policy_output)
                final_transition = environment.step_transition(action.squeeze(0).cpu().numpy())
                total_reward += final_transition.reward
                with torch.no_grad():
                    auxiliary_squared_errors.append(
                        torch.nn.functional.mse_loss(
                            policy_output.auxiliary_predictions.squeeze(0),
                            auxiliary_targets(final_transition),
                        ).item()
                    )
                speeds.append(float(final_transition.labels.speed))
                observation = final_transition.next_observation
                if final_transition.terminated or final_transition.truncated:
                    break
            raw_finish_time = getattr(environment.unwrapped, "finish_time_s", None)
            finish_time = (
                float(raw_finish_time) - start_simulation_time
                if raw_finish_time is not None
                else None
            )
        finally:
            close_training_environment(environment)
        labels = final_transition.labels if final_transition is not None else None
        results.append(
            {
                "finished": bool(labels.finished) if labels else False,
                "progress": float(labels.tile_progress) if labels else 0.0,
                "reward": total_reward,
                "lap_time_s": finish_time,
                "auxiliary_mse": float(np.mean(auxiliary_squared_errors))
                if auxiliary_squared_errors
                else 0.0,
                "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                "max_speed": float(np.max(speeds)) if speeds else 0.0,
            }
        )
    finished = [result for result in results if result["finished"]]
    finish_times = [result["lap_time_s"] for result in finished if result["lap_time_s"] is not None]
    return {
        "finish_rate": float(np.mean([result["finished"] for result in results])) if results else 0.0,
        "median_finished_lap_time_s": float(np.median(finish_times)) if finish_times else None,
        "p90_finished_lap_time_s": float(np.percentile(finish_times, 90)) if finish_times else None,
        "mean_progress": float(np.mean([result["progress"] for result in results])) if results else 0.0,
        "mean_reward": float(np.mean([result["reward"] for result in results])) if results else 0.0,
        "auxiliary_mse": float(np.mean([result["auxiliary_mse"] for result in results])) if results else 0.0,
        "mean_speed": float(np.mean([result["mean_speed"] for result in results])) if results else 0.0,
        "max_speed": float(np.max([result["max_speed"] for result in results])) if results else 0.0,
        "episodes": float(len(results)),
        "completed_episodes": float(len(finished)),
    }


def selection_score(metrics: dict[str, float | None]) -> tuple[float, float, float, float]:
    """Order checkpoints by completion, median/p90 finished lap time, then progress."""
    lap_time = metrics["median_finished_lap_time_s"]
    p90_lap_time = metrics.get("p90_finished_lap_time_s", lap_time)
    lap_component = -float(lap_time) if lap_time is not None else float("-inf")
    p90_component = -float(p90_lap_time) if p90_lap_time is not None else float("-inf")
    return float(metrics["finish_rate"]), lap_component, p90_component, float(metrics["mean_progress"])


def hud_ablation(
    model: VisualActorCritic, tune_episodes: Iterable[TrainingEpisode], *, max_decisions: int
) -> dict[str, dict[str, float | None]]:
    """Measure the HUD branch contribution on the tune split using the same weights.

    An actor trained without the HUD branch has nothing to ablate, and running
    the sweep anyway cost 14-22% of a continuation's wall clock while both arms
    returned byte-identical metrics. Report the skip instead of paying for it.
    """
    if not model.use_hud:
        return {"skipped": True, "reason": "actor was trained without the HUD branch"}
    enabled = evaluate_policy(model, tune_episodes, max_decisions=max_decisions)
    previous = model.use_hud
    model.use_hud = False
    try:
        disabled = evaluate_policy(model, tune_episodes, max_decisions=max_decisions)
    finally:
        model.use_hud = previous
    return {"hud_enabled": enabled, "hud_disabled": disabled}


def distribute_training_steps(total_steps: int, updates: int) -> tuple[int, ...]:
    """Split one run's environment budget into positive on-policy updates."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if updates < 1 or updates > total_steps:
        raise ValueError("updates must be between 1 and total_steps")
    steps_per_update, remainder = divmod(total_steps, updates)
    return tuple(
        steps_per_update + (1 if update_index < remainder else 0)
        for update_index in range(updates)
    )


def rollout_action_metrics(
    rollout: RolloutStorage, *, throttle_limit: float = MAX_GAS
) -> dict[str, float]:
    """Summarize actual sampled simulator actions for exploration diagnosis.

    ``gas_saturation_fraction`` answers the question that lap-time debugging
    kept running into: whether the actor is slow because it chooses to be or
    because the action transform will not let it ask for more throttle.
    """
    if not len(rollout):
        raise ValueError("cannot summarize an empty rollout")
    actions = torch.stack([step.action for step in rollout.steps]).float()
    speeds = tuple(getattr(rollout, "rollout_speeds", ()))
    speed_metrics = (
        {
            "mean_speed": float(np.mean(speeds)),
            "max_speed": float(np.max(speeds)),
        }
        if speeds
        else {}
    )
    return {
        "throttle_limit": float(throttle_limit),
        "gas_saturation_fraction": float(
            (actions[:, 1] >= 0.98 * throttle_limit).float().mean()
        ),
        **speed_metrics,
        "steer_mean": float(actions[:, 0].mean()),
        "steer_std": float(actions[:, 0].std(unbiased=False)),
        "mean_abs_steer": float(actions[:, 0].abs().mean()),
        "gas_mean": float(actions[:, 1].mean()),
        "gas_max": float(actions[:, 1].max()),
        "brake_mean": float(actions[:, 2].mean()),
        "brake_fraction": float((actions[:, 2] > 0.001).float().mean()),
        "pedal_overlap_fraction": float(
            ((actions[:, 1] > 0.0) & (actions[:, 2] > 0.0)).float().mean()
        ),
    }


def train(
    *,
    output_directory: Path,
    total_steps: int,
    max_decisions: int,
    seed: int,
    updates: int = 4,
    evaluation_max_decisions: int | None = None,
    learning_rate: float = 3e-4,
    policy_mean_learning_rate: float | None = None,
    resume: Path | None = None,
    initialize_from: Path | None = None,
    split: TrainingSplit = DEFAULT_SPLIT,
    teacher_warmup_epochs: int = 3,
    teacher_max_decisions: int = 800,
    teacher_warmup_batch_size: int = 32,
    teacher_warmup_learning_rate: float = 1e-3,
    use_hud: bool = True,
    use_visual_features: bool = False,
    use_temporal_features: bool = False,
    pedal_expansion: float = DEFAULT_PEDAL_EXPANSION,
    throttle_expansion: float | None = None,
    brake_expansion: float | None = None,
    gamma: float = PPOConfig.gamma,
    gae_lambda: float = PPOConfig.gae_lambda,
    obstacle_brake_reward: float = OBSTACLE_BRAKE_REWARD,
    curve_brake_reward: float = 0.0,
    direct_gas_reward: float = 0.0,
    recovery_clearance_reward: float = 0.0,
    hazard_potential_scale: float = 0.0,
    enable_recovery_action_rewards: bool = True,
    lagrangian_mode: str = "off",
    lagrangian_initial_lambda: float = REFERENCE_LAMBDA,
    lagrangian_cost_budget: float = DEFAULT_COST_BUDGET,
    lagrangian_dual_step_size: float = DEFAULT_DUAL_STEP_SIZE,
    lagrangian_window_size: int = DEFAULT_WINDOW_SIZE,
    lagrangian_min_lambda: float = DEFAULT_MIN_MULTIPLIER,
    lagrangian_max_lambda: float = DEFAULT_MAX_MULTIPLIER,
    lagrangian_cost_gamma: float = 1.0,
    paired_rollout_seed: int | None = None,
    expected_initial_actor_sha256: str | None = None,
    preflight_report_sha256: str | None = None,
    tune_selection: bool = True,
    actor_snapshot_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Warm-start the pixel actor from train-only demonstrations, then run PPO."""
    torch.set_num_threads(CPU_INFERENCE_THREADS)
    started = time.perf_counter()
    selection_max_decisions = (
        max_decisions if evaluation_max_decisions is None else evaluation_max_decisions
    )
    if max_decisions < 1 or selection_max_decisions < 1:
        raise ValueError("max_decisions and evaluation_max_decisions must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if policy_mean_learning_rate is not None and (
        not np.isfinite(policy_mean_learning_rate) or policy_mean_learning_rate <= 0.0
    ):
        raise ValueError("policy_mean_learning_rate must be finite and positive")
    effective_policy_mean_learning_rate = (
        learning_rate
        if policy_mean_learning_rate is None
        else float(policy_mean_learning_rate)
    )
    if not np.isfinite(obstacle_brake_reward) or obstacle_brake_reward < 0.0:
        raise ValueError("obstacle_brake_reward must be finite and non-negative")
    if not np.isfinite(curve_brake_reward) or curve_brake_reward < 0.0:
        raise ValueError("curve_brake_reward must be finite and non-negative")
    if not np.isfinite(direct_gas_reward) or direct_gas_reward < 0.0:
        raise ValueError("direct_gas_reward must be finite and non-negative")
    if not np.isfinite(recovery_clearance_reward) or recovery_clearance_reward < 0.0:
        raise ValueError("recovery_clearance_reward must be finite and non-negative")
    if not np.isfinite(hazard_potential_scale) or hazard_potential_scale < 0.0:
        raise ValueError("hazard_potential_scale must be finite and non-negative")
    if hazard_potential_scale > 0.0 and not use_visual_features:
        raise ValueError("hazard-potential shaping requires visual pixel features")
    if hazard_potential_scale > 0.0 and lagrangian_mode != "off":
        raise ValueError("hazard-potential screen requires ordinary PPO without Lagrangian mode")
    if lagrangian_mode not in ("off", "fixed", "adaptive"):
        raise ValueError("lagrangian_mode must be 'off', 'fixed', or 'adaptive'")
    if not np.isfinite(lagrangian_cost_gamma) or not 0.0 <= lagrangian_cost_gamma <= 1.0:
        raise ValueError("lagrangian_cost_gamma must be finite and between 0 and 1")
    if not isinstance(tune_selection, bool):
        raise ValueError("tune_selection must be a boolean")
    if not isinstance(enable_recovery_action_rewards, bool):
        raise ValueError("enable_recovery_action_rewards must be a boolean")
    if paired_rollout_seed is not None and (
        isinstance(paired_rollout_seed, bool)
        or not isinstance(paired_rollout_seed, int)
        or paired_rollout_seed < 0
    ):
        raise ValueError("paired_rollout_seed must be a non-negative integer")
    for name, value in (
        ("expected_initial_actor_sha256", expected_initial_actor_sha256),
        ("preflight_report_sha256", preflight_report_sha256),
    ):
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    if resume is not None and initialize_from is not None:
        raise ValueError("resume and actor-only initialize_from are mutually exclusive")
    if teacher_warmup_epochs < 0:
        raise ValueError("teacher_warmup_epochs must be non-negative")
    if teacher_max_decisions < 1:
        raise ValueError("teacher_max_decisions must be positive")
    if teacher_warmup_batch_size < 1:
        raise ValueError("teacher_warmup_batch_size must be positive")
    if not np.isfinite(teacher_warmup_learning_rate) or teacher_warmup_learning_rate <= 0:
        raise ValueError("teacher_warmup_learning_rate must be finite and positive")
    if not isinstance(use_visual_features, bool):
        raise ValueError("use_visual_features must be a boolean")
    if not isinstance(use_temporal_features, bool):
        raise ValueError("use_temporal_features must be a boolean")
    if use_temporal_features and not use_visual_features:
        raise ValueError("use_temporal_features requires use_visual_features")
    set_reproducible_seed(seed)
    update_step_counts = distribute_training_steps(total_steps, updates)
    actor = VisualActorCritic(
        use_hud=use_hud,
        use_visual_features=use_visual_features,
        use_temporal_features=use_temporal_features,
        throttle_expansion=(
            pedal_expansion if throttle_expansion is None else throttle_expansion
        ),
        brake_expansion=(
            pedal_expansion if brake_expansion is None else brake_expansion
        ),
    )
    lagrange_multiplier = (
        None
        if lagrangian_mode == "off"
        else LagrangeMultiplier(
            mode=lagrangian_mode,
            initial_value=lagrangian_initial_lambda,
            cost_budget=lagrangian_cost_budget,
            dual_step_size=lagrangian_dual_step_size,
            window_size=lagrangian_window_size,
            minimum=lagrangian_min_lambda,
            maximum=lagrangian_max_lambda,
        )
    )
    model: VisualActorCritic | LagrangianActorCritic = (
        actor if lagrange_multiplier is None else LagrangianActorCritic(actor)
    )
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        policy_mean_learning_rate=effective_policy_mean_learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        minibatch_size=min(32, min(update_step_counts)),
    )
    updater: PPOUpdater | LagrangianPPOUpdater = (
        PPOUpdater(model, ppo_config)
        if lagrange_multiplier is None
        else LagrangianPPOUpdater(model, ppo_config)
    )
    optimizer = getattr(updater, "optimizer", None)
    optimizer_state = getattr(optimizer, "state", {})
    if optimizer_state:
        raise RuntimeError("a newly created PPO optimizer must start empty")
    optimizer_state_entries_at_start = len(optimizer_state)
    start_step = 0
    resume_metadata: dict[str, Any] = {}
    actor_initialization: dict[str, Any] | None = None
    if resume is not None:
        loaded = load_checkpoint(
            resume,
            model,
            updater,
            required_training_state=(
                "lagrange_multiplier" if lagrange_multiplier is not None else None
            ),
        )
        start_step = loaded["step"]
        resume_metadata = loaded["metadata"]
        if lagrange_multiplier is not None:
            saved_lagrange_state = loaded["training_state"].get("lagrange_multiplier")
            if not isinstance(saved_lagrange_state, dict):
                raise ValueError("Lagrangian resume checkpoint has no multiplier state")
            lagrange_multiplier.load_state_dict(saved_lagrange_state)
        if resume_metadata.get("use_hud", True) is not use_hud:
            raise ValueError("resume checkpoint use_hud setting does not match the requested actor")
        if resume_metadata.get("use_visual_features", False) is not use_visual_features:
            raise ValueError(
                "resume checkpoint visual feature setting does not match the requested actor"
            )
        if resume_metadata.get("use_temporal_features", False) and not use_temporal_features:
            raise ValueError(
                "resume checkpoint temporal feature setting does not match the requested actor"
            )
        resumed_symmetric = resume_metadata.get(
            "pedal_expansion", DEFAULT_PEDAL_EXPANSION
        )
        requested = getattr(model, "pedal_scale", PedalScale())
        for field, resumed in (
            ("throttle_expansion", resume_metadata.get("throttle_expansion", resumed_symmetric)),
            ("brake_expansion", resume_metadata.get("brake_expansion", resumed_symmetric)),
        ):
            if float(resumed) > getattr(requested, field) + 1e-9:
                raise ValueError(
                    f"resume checkpoint {field} is larger than the requested actor; "
                    "narrowing the pedal range would silently change every stored action"
                )
        updater.set_learning_rates(
            learning_rate, effective_policy_mean_learning_rate
        )
    elif initialize_from is not None:
        if optimizer is None:
            raise RuntimeError("actor-only initialization requires a PPO optimizer")
        actor_initialization = load_actor_weights(
            initialize_from,
            model.policy if isinstance(model, LagrangianActorCritic) else model,
        )
        actor_initialization["optimizer_state_entries_at_actor_load"] = len(
            optimizer.state
        )
        if optimizer.state:
            raise RuntimeError("actor-only initialization must leave Adam state empty")

    initial_actor_sha256 = actor_state_sha256(actor)
    if (
        expected_initial_actor_sha256 is not None
        and initial_actor_sha256 != expected_initial_actor_sha256
    ):
        raise ValueError("initialized actor does not match the paired preflight hash")

    if any(
        isinstance(step, bool) or not isinstance(step, int) or step <= 0
        for step in actor_snapshot_steps
    ):
        raise ValueError("actor_snapshot_steps must contain positive integer steps")
    if len(set(actor_snapshot_steps)) != len(actor_snapshot_steps):
        raise ValueError("actor_snapshot_steps must not contain duplicates")
    update_boundaries: set[int] = set()
    cumulative_steps = 0
    for update_steps in update_step_counts:
        cumulative_steps += update_steps
        update_boundaries.add(cumulative_steps)
    if any(step not in update_boundaries for step in actor_snapshot_steps):
        raise ValueError("actor snapshots must align with completed PPO update boundaries")

    checkpoint = output_directory / "policy.pt"
    training_checkpoint = (
        checkpoint
        if lagrange_multiplier is None
        else output_directory / "lagrangian-training.pt"
    )
    update_results: list[dict[str, Any]] = []
    best_tune_metrics: dict[str, float | None] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_lagrange_state: dict[str, Any] | None = None
    best_checkpoint_metadata: dict[str, Any] | None = None
    best_checkpoint_step: int | None = None
    checkpoint_updated = False
    completed_steps = 0
    timings_s = {
        "teacher_data_collection_s": 0.0,
        "teacher_warmup_s": 0.0,
        "rollout_s": 0.0,
        "ppo_update_s": 0.0,
        "tune_selection_eval_s": 0.0,
        "full_tune_eval_s": 0.0,
        "hud_ablation_s": 0.0,
    }
    previous_warmup = resume_metadata.get("teacher_warmup")
    teacher_decision_limit = min(teacher_max_decisions, max_decisions)
    if isinstance(previous_warmup, dict) and previous_warmup.get("enabled"):
        teacher_warmup_metrics = dict(previous_warmup)
        teacher_warmup_metrics["reused_from_checkpoint"] = True
    elif teacher_warmup_epochs == 0:
        teacher_warmup_metrics = {
            "enabled": False,
            "teacher": "vision_corridor_training_only",
            "epochs": 0,
            "demonstration_episodes": 0,
            "demonstration_steps": 0,
            "teacher_loss_used_during_ppo": False,
        }
    else:
        collection_started = time.perf_counter()
        demonstrations = collect_teacher_demonstrations(
            split,
            max_decisions=teacher_decision_limit,
            pedal_expansion=getattr(model, "pedal_scale", PedalScale()),
        )
        timings_s["teacher_data_collection_s"] = (
            time.perf_counter() - collection_started
        )
        warmup_started = time.perf_counter()
        teacher_warmup_metrics = behavioral_cloning_warmup(
            actor,
            demonstrations,
            epochs=teacher_warmup_epochs,
            batch_size=teacher_warmup_batch_size,
            learning_rate=teacher_warmup_learning_rate,
        )
        timings_s["teacher_warmup_s"] = time.perf_counter() - warmup_started
        teacher_warmup_metrics["data_collection_s"] = timings_s[
            "teacher_data_collection_s"
        ]
        teacher_warmup_metrics["warmup_s"] = timings_s["teacher_warmup_s"]
        teacher_warmup_metrics["max_decisions_per_episode"] = teacher_decision_limit

    # A paired ablation can reset stochastic sampling after mode-specific model
    # construction. This keeps initial action noise, episode shuffles, and PPO
    # minibatch permutations aligned while leaving ordinary PPO's RNG sequence
    # unchanged when the option is omitted.
    if paired_rollout_seed is not None:
        set_reproducible_seed(paired_rollout_seed)

    last_update_metadata: dict[str, Any] | None = None
    for update_number, update_steps in enumerate(update_step_counts, start=1):
        rollout_started = time.perf_counter()
        rollout = collect_rollout(
            model,
            split.train,
            total_steps=update_steps,
            max_decisions=max_decisions,
            gamma=ppo_config.gamma,
            cost_gamma=lagrangian_cost_gamma,
            gae_lambda=ppo_config.gae_lambda,
            lagrangian_mode=lagrangian_mode,
            obstacle_brake_reward=obstacle_brake_reward,
            curve_brake_reward=curve_brake_reward,
            direct_gas_reward=direct_gas_reward,
            recovery_clearance_reward=recovery_clearance_reward,
            hazard_potential_scale=hazard_potential_scale,
            enable_recovery_action_rewards=enable_recovery_action_rewards,
        )
        action_metrics = rollout_action_metrics(
            rollout, throttle_limit=float(getattr(model, "throttle_limit", MAX_GAS))
        )
        rollout_time = time.perf_counter() - rollout_started
        timings_s["rollout_s"] += rollout_time
        update_started = time.perf_counter()
        lagrangian_update: dict[str, Any] | None = None
        if lagrange_multiplier is None:
            losses = dict(updater.update(rollout))
        else:
            lambda_for_update = lagrange_multiplier.value
            losses = dict(
                updater.update(
                    rollout, lagrange_multiplier=lambda_for_update
                )
            )
            windows = []
            for episode_cost in rollout.completed_episode_costs:
                window = lagrange_multiplier.observe_completed_episode(episode_cost)
                if window is not None:
                    windows.append(
                        {
                            "mode": window.mode,
                            "episode_count": window.episode_count,
                            "mean_cost": window.mean_cost,
                            "multiplier_before": window.multiplier_before,
                            "multiplier_after": window.multiplier_after,
                        }
                    )
            completed_costs = [
                episode.total_cost for episode in rollout.completed_episode_costs
            ]
            lagrangian_update = {
                "mode": lagrangian_mode,
                "lambda_for_update": lambda_for_update,
                "lambda_after_completed_episodes": lagrange_multiplier.value,
                "completed_train_episodes": len(completed_costs),
                "mean_completed_episode_cost": (
                    float(np.mean(completed_costs)) if completed_costs else None
                ),
                "multiplier_windows": windows,
                "discarded_collector_fragments": len(
                    rollout.incomplete_episode_costs
                ),
            }
        ppo_update_time = time.perf_counter() - update_started
        timings_s["ppo_update_s"] += ppo_update_time
        completed_steps += len(rollout)
        if tune_selection:
            tune_started = time.perf_counter()
            tune_metrics = evaluate_policy(
                model, split.tune, max_decisions=selection_max_decisions
            )
            tune_time = time.perf_counter() - tune_started
        else:
            tune_metrics = None
            tune_time = 0.0
        timings_s["tune_selection_eval_s"] += tune_time
        metadata = {
            "seed": seed,
            "initial_actor_sha256": initial_actor_sha256,
            "preflight_report_sha256": preflight_report_sha256,
            "split": describe_split(split),
            "training_steps_this_run": completed_steps,
            "training_steps_requested": total_steps,
            "updates_completed": update_number,
            "updates_requested": updates,
            "learning_rate": learning_rate,
            "policy_mean_learning_rate": effective_policy_mean_learning_rate,
            "use_hud": bool(use_hud),
            "use_visual_features": use_visual_features,
            "use_temporal_features": use_temporal_features,
            "pedal_expansion": float(
                getattr(model, "pedal_expansion", DEFAULT_PEDAL_EXPANSION)
            ),
            "throttle_expansion": float(
                getattr(model, "pedal_scale", PedalScale()).throttle_expansion
            ),
            "brake_expansion": float(
                getattr(model, "pedal_scale", PedalScale()).brake_expansion
            ),
            "throttle_limit": float(getattr(model, "throttle_limit", MAX_GAS)),
            "brake_limit": float(getattr(model, "brake_limit", MAX_BRAKE)),
            "gamma": float(ppo_config.gamma),
            "gae_lambda": float(ppo_config.gae_lambda),
            "teacher_warmup": teacher_warmup_metrics,
            "reward_scale": REWARD_SCALE,
            "obstacle_brake_reward": float(obstacle_brake_reward),
            "obstacle_brake_formula": (
                "coefficient * visible_hazard_risk * normalized_brake_fraction * reward_scale"
            ),
            "obstacle_brake_training_only": True,
            "curve_brake_reward": float(curve_brake_reward),
            "direct_gas_reward": float(direct_gas_reward),
            "direct_gas_formula": "coefficient * clear_road * normalized_gas * speed_shortfall * reward_scale",
            "direct_gas_training_only": True,
            "curve_brake_formula": (
                "coefficient * road_curve_magnitude * normalized_brake_fraction * reward_scale"
            ),
            "curve_brake_training_only": True,
            "recovery_clearance_reward": float(recovery_clearance_reward),
            "hazard_potential_scale": float(hazard_potential_scale),
            "hazard_potential_formula": (
                "beta * (gamma * Phi(next) - Phi(current)), Phi(risk)=-risk; "
                "added after base reward scaling"
            ),
            "hazard_potential_training_only": True,
            "hazard_potential_diagnostics": summarize_hazard_potential_transitions(
                getattr(rollout, "hazard_potential_transitions", [])
            ),
            "recovery_clearance_formula": (
                "coefficient * max(0, current_visible_hazard_risk - next_visible_hazard_risk) * reward_scale "
                "when damaged, low-speed, high-risk, non-collision, and on-track"
            ),
            "recovery_clearance_training_only": True,
            "tune_selection_enabled": tune_selection,
            "actor_initialization": actor_initialization,
            "optimizer_state_entries_at_start": optimizer_state_entries_at_start,
            "reward_config": {
                "time_penalty": TIME_PENALTY,
                "speed_shortfall_penalty": SPEED_SHORTFALL_PENALTY,
                "speed_target": SPEED_TARGET,
                "hazard_speed_relief": HAZARD_SPEED_RELIEF,
                "curve_speed_relief": CURVE_SPEED_RELIEF,
                "obstacle_throttle_penalty": OBSTACLE_THROTTLE_PENALTY,
                "obstacle_brake_reward": float(obstacle_brake_reward),
                "curve_brake_reward": float(curve_brake_reward),
                "direct_gas_reward": float(direct_gas_reward),
                "recovery_clearance_reward": float(recovery_clearance_reward),
                "hazard_potential_scale": float(hazard_potential_scale),
                "recovery_action_rewards_enabled": enable_recovery_action_rewards,
                "obstacle_lateral_clearance_pixels": OBSTACLE_LATERAL_CLEARANCE_PIXELS,
                "obstacle_throttle_penalty_training_only": True,
                "collision_penalty": COLLISION_PENALTY,
                "terminal_failure_penalty": TERMINAL_FAILURE_PENALTY,
                "recovery_min_damage": RECOVERY_MIN_DAMAGE,
                "recovery_max_speed": RECOVERY_MAX_SPEED,
                "recovery_max_hazard_risk": RECOVERY_MAX_HAZARD_RISK,
                "recovery_gas_reward": RECOVERY_GAS_REWARD,
                "recovery_steer_reward": RECOVERY_STEER_REWARD,
            },
            "auxiliary_target_scales": AUXILIARY_TARGET_SCALES,
            "selection_max_decisions": selection_max_decisions,
            "full_evaluation_max_decisions": max_decisions,
            "training_action_metrics": action_metrics,
            "tune_metrics": tune_metrics,
            "selection_metric": (
                "finish_rate, negative_median_finished_lap_time_s, "
                "negative_p90_finished_lap_time_s, mean_progress"
                if tune_selection
                else None
            ),
        }
        training_state = None
        if lagrangian_update is not None and lagrange_multiplier is not None:
            metadata["lagrangian_mode"] = lagrangian_mode
            metadata["lagrangian"] = {
                **lagrangian_update,
                "initial_lambda": float(lagrangian_initial_lambda),
                "cost_budget": float(lagrangian_cost_budget),
                "dual_step_size": float(lagrangian_dual_step_size),
                "window_size": int(lagrangian_window_size),
                "minimum_lambda": float(lagrangian_min_lambda),
                "maximum_lambda": float(lagrangian_max_lambda),
                "cost_gamma": float(lagrangian_cost_gamma),
                "safety_costs_separate_from_reward": True,
                "reward_excluded_safety_terms": [
                    "collision_penalty",
                    "damage_stock_penalty",
                    "off_track_penalty",
                    "terminal_failure_penalty",
                    "simulator_out_of_bounds_penalty",
                ],
            }
            training_state = {
                "lagrange_multiplier": lagrange_multiplier.state_dict()
            }
        last_update_metadata = metadata
        save_kwargs = (
            {} if training_state is None else {"training_state": training_state}
        )
        updated = save_best_checkpoint(
            training_checkpoint,
            model,
            updater,
            step=start_step + completed_steps,
            tune_metrics=tune_metrics,
            metadata=metadata,
            **save_kwargs,
        ) if tune_selection else False
        checkpoint_updated = checkpoint_updated or updated
        if updated:
            best_tune_metrics = tune_metrics
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_lagrange_state = training_state
            best_checkpoint_metadata = dict(metadata)
            best_checkpoint_step = start_step + completed_steps
        if completed_steps in actor_snapshot_steps:
            save_actor_snapshot(
                output_directory / f"actor-step-{completed_steps}.pt",
                model,
                step=start_step + completed_steps,
                metadata=metadata,
            )
        update_result = {
            "update": update_number,
            "steps": len(rollout),
            "step": start_step + completed_steps,
            "losses": losses,
            "action_metrics": action_metrics,
            "train_episode_outcomes": list(
                getattr(rollout, "train_episode_outcomes", [])
            ),
            "incomplete_train_episode_outcomes": list(
                getattr(rollout, "incomplete_train_episode_outcomes", [])
            ),
            "hazard_potential_diagnostics": summarize_hazard_potential_transitions(
                getattr(rollout, "hazard_potential_transitions", [])
            ),
            "tune_metrics": tune_metrics,
            "checkpoint_updated": updated,
            "timings_s": {
                "rollout_s": rollout_time,
                "ppo_update_s": ppo_update_time,
                "tune_selection_eval_s": tune_time,
            },
        }
        if lagrangian_update is not None:
            update_result["lagrangian"] = lagrangian_update
        update_results.append(update_result)

    if tune_selection:
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            if lagrange_multiplier is not None and best_lagrange_state is not None:
                lagrange_multiplier.load_state_dict(
                    best_lagrange_state["lagrange_multiplier"]
                )
        elif training_checkpoint.is_file():
            stored_checkpoint = torch.load(
                training_checkpoint, map_location="cpu", weights_only=False
            )
            model.load_state_dict(stored_checkpoint["model_state"])
            best_tune_metrics = stored_checkpoint.get("metadata", {}).get("tune_metrics")
            best_checkpoint_metadata = dict(stored_checkpoint.get("metadata", {}))
            best_checkpoint_step = int(stored_checkpoint.get("step", start_step))
            if lagrange_multiplier is not None:
                saved_state = stored_checkpoint.get("training_state", {}).get(
                    "lagrange_multiplier"
                )
                if not isinstance(saved_state, dict):
                    raise ValueError("selected Lagrangian checkpoint has no multiplier state")
                lagrange_multiplier.load_state_dict(saved_state)
        if best_tune_metrics is None:
            best_tune_metrics = update_results[-1]["tune_metrics"]
        full_tune_started = time.perf_counter()
        full_tune_metrics = evaluate_policy(model, split.tune, max_decisions=max_decisions)
        timings_s["full_tune_eval_s"] = time.perf_counter() - full_tune_started
        ablation_started = time.perf_counter()
        ablation = hud_ablation(
            model, split.tune, max_decisions=selection_max_decisions
        )
        timings_s["hud_ablation_s"] = time.perf_counter() - ablation_started
    else:
        if last_update_metadata is None:
            raise RuntimeError("deferred-TUNE training finished without an update")
        final_metadata = dict(last_update_metadata)
        final_metadata["checkpoint_selection"] = "final_update_without_tune"
        final_metadata["tune_metrics"] = None
        training_state = (
            None
            if lagrange_multiplier is None
            else {"lagrange_multiplier": lagrange_multiplier.state_dict()}
        )
        save_kwargs = (
            {} if training_state is None else {"training_state": training_state}
        )
        save_checkpoint(
            training_checkpoint,
            model,
            updater,
            step=start_step + completed_steps,
            metadata=final_metadata,
            **save_kwargs,
        )
        checkpoint_updated = True
        full_tune_metrics = None
        ablation = {"skipped": True, "reason": "TUNE deferred until all experiment arms finish"}
        best_checkpoint_metadata = final_metadata
        best_checkpoint_step = start_step + completed_steps
    if lagrange_multiplier is not None:
        if best_checkpoint_metadata is None:
            raise RuntimeError("Lagrangian training completed without checkpoint metadata")
        save_actor_snapshot(
            checkpoint,
            model,
            step=(
                start_step + completed_steps
                if best_checkpoint_step is None
                else best_checkpoint_step
            ),
            metadata=best_checkpoint_metadata,
        )
        checkpoint_updated = True
    timings_s["wall_s"] = time.perf_counter() - started
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_updated": checkpoint_updated,
        "learning_rate": learning_rate,
        "policy_mean_learning_rate": effective_policy_mean_learning_rate,
        "use_hud": bool(use_hud),
        "use_visual_features": use_visual_features,
        "pedal_expansion": float(
            getattr(model, "pedal_expansion", DEFAULT_PEDAL_EXPANSION)
        ),
        "throttle_expansion": float(
            getattr(model, "pedal_scale", PedalScale()).throttle_expansion
        ),
        "brake_expansion": float(
            getattr(model, "pedal_scale", PedalScale()).brake_expansion
        ),
        "throttle_limit": float(getattr(model, "throttle_limit", MAX_GAS)),
        "obstacle_brake_reward": float(obstacle_brake_reward),
        "curve_brake_reward": float(curve_brake_reward),
        "direct_gas_reward": float(direct_gas_reward),
        "recovery_clearance_reward": float(recovery_clearance_reward),
        "recovery_action_rewards_enabled": enable_recovery_action_rewards,
        "initial_actor_sha256": initial_actor_sha256,
        "preflight_report_sha256": preflight_report_sha256,
        "paired_rollout_seed": paired_rollout_seed,
        "tune_selection_enabled": tune_selection,
        "actor_initialization": actor_initialization,
        "optimizer_state_entries_at_start": optimizer_state_entries_at_start,
        "checkpoint_selection": (
            "best_tune" if tune_selection else "final_update_without_tune"
        ),
        "losses": update_results[-1]["losses"],
        "teacher_warmup": teacher_warmup_metrics,
        "tune_metrics": best_tune_metrics,
        "full_tune_metrics": full_tune_metrics,
        "ablation": ablation,
        "updates": update_results,
        "updates_completed": len(update_results),
        "training_steps": completed_steps,
        "step": start_step + completed_steps,
        "timings_s": timings_s,
        "wall_time_s": timings_s["wall_s"],
    }
    if lagrange_multiplier is not None:
        result["training_checkpoint"] = str(training_checkpoint)
        result["lagrangian"] = {
            "mode": lagrangian_mode,
            "lambda_after": lagrange_multiplier.value,
            "completed_train_episodes": lagrange_multiplier.completed_episode_count,
            "cost_budget": lagrangian_cost_budget,
            "dual_step_size": lagrangian_dual_step_size,
            "window_size": lagrangian_window_size,
            "cost_gamma": lagrangian_cost_gamma,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/haic/visual-ppo"))
    parser.add_argument("--total-steps", type=int, default=2_048)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--evaluation-max-decisions", type=int, default=800)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-mean-learning-rate", type=float)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=3)
    parser.add_argument("--teacher-max-decisions", type=int, default=800)
    parser.add_argument("--teacher-warmup-batch-size", type=int, default=32)
    parser.add_argument("--teacher-warmup-learning-rate", type=float, default=1e-3)
    parser.add_argument("--disable-hud-branch", action="store_true")
    parser.add_argument("--enable-visual-features", action="store_true")
    parser.add_argument("--enable-temporal-features", action="store_true")
    parser.add_argument(
        "--pedal-expansion",
        type=float,
        default=DEFAULT_PEDAL_EXPANSION,
        help=(
            "Widen the pedal ceilings by this factor while holding the cruise "
            "throttle fixed. 1.0 keeps the historical 0.12/0.28 limits."
        ),
    )
    parser.add_argument(
        "--throttle-expansion",
        type=float,
        default=None,
        help=(
            "Open only the throttle branch, up to "
            f"{MAX_THROTTLE_EXPANSION:.3f} (gas 1.0). Overrides --pedal-expansion."
        ),
    )
    parser.add_argument(
        "--brake-expansion",
        type=float,
        default=None,
        help=(
            "Open only the brake branch, up to "
            f"{MAX_BRAKE_EXPANSION:.3f} (brake 1.0). Overrides --pedal-expansion."
        ),
    )
    parser.add_argument(
        "--speed-target",
        type=float,
        default=SPEED_TARGET,
        help="Cruise speed below which every decision is charged.",
    )
    parser.add_argument(
        "--speed-shortfall-penalty",
        type=float,
        default=SPEED_SHORTFALL_PENALTY,
        help="Reward charged per decision at a standstill on clear road.",
    )
    parser.add_argument(
        "--obstacle-brake-reward", type=float, default=OBSTACLE_BRAKE_REWARD
    )
    parser.add_argument("--curve-brake-reward", type=float, default=0.0)
    parser.add_argument("--direct-gas-reward", type=float, default=0.0)
    parser.add_argument("--recovery-clearance-reward", type=float, default=0.0)
    parser.add_argument(
        "--hazard-potential-scale",
        type=float,
        default=0.0,
        help="Add pixel-only potential-difference reward to ordinary PPO transitions.",
    )
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Initialize actor weights without restoring PPO optimizer state.",
    )
    parser.add_argument(
        "--lagrangian-mode",
        choices=("off", "fixed", "adaptive"),
        default="off",
        help="Enable the separate TRAIN-only safety-cost critic; default keeps standard PPO.",
    )
    parser.add_argument(
        "--lagrangian-initial-lambda", type=float, default=REFERENCE_LAMBDA
    )
    parser.add_argument(
        "--lagrangian-cost-budget", type=float, default=DEFAULT_COST_BUDGET
    )
    parser.add_argument(
        "--lagrangian-dual-step-size", type=float, default=DEFAULT_DUAL_STEP_SIZE
    )
    parser.add_argument(
        "--lagrangian-window-size", type=int, default=DEFAULT_WINDOW_SIZE
    )
    parser.add_argument(
        "--lagrangian-min-lambda", type=float, default=DEFAULT_MIN_MULTIPLIER
    )
    parser.add_argument(
        "--lagrangian-max-lambda", type=float, default=DEFAULT_MAX_MULTIPLIER
    )
    parser.add_argument("--lagrangian-cost-gamma", type=float, default=1.0)
    parser.add_argument("--site-map-split", type=Path)
    parser.add_argument(
        "--train-only-site-map-split",
        type=Path,
        help="Load only TRAIN map files; TUNE and held-out map files are not opened.",
    )
    parser.add_argument(
        "--defer-tune",
        action="store_true",
        help="Skip all TUNE evaluation and checkpoint selection during this run.",
    )
    parser.add_argument(
        "--disable-recovery-action-rewards",
        action="store_true",
        help="Disable damaged-state gas/steer action bonuses for matched experiments.",
    )
    parser.add_argument(
        "--paired-rollout-seed",
        type=int,
        help="Reset RNG streams after model construction to align paired rollout sampling.",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        help="Require the passing paired-screen preflight and its actor/map hashes.",
    )
    parser.add_argument(
        "--hazard-potential-preflight",
        type=Path,
        help="Require the frozen paired pixel hazard-potential TRAIN-only preflight.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    global SPEED_TARGET, SPEED_SHORTFALL_PENALTY
    args = parse_args()
    # The shaping constants are module-level so every caller sees one reward;
    # let a sweep override them here and record the values in checkpoint
    # metadata rather than editing the file per experiment.
    if not math.isfinite(args.speed_target) or args.speed_target <= 0.0:
        raise ValueError("--speed-target must be finite and positive")
    if not math.isfinite(args.speed_shortfall_penalty) or args.speed_shortfall_penalty < 0.0:
        raise ValueError("--speed-shortfall-penalty must be finite and non-negative")
    if not math.isfinite(args.hazard_potential_scale) or args.hazard_potential_scale < 0.0:
        raise ValueError("--hazard-potential-scale must be finite and non-negative")
    if not math.isfinite(args.direct_gas_reward) or args.direct_gas_reward < 0.0:
        raise ValueError("--direct-gas-reward must be finite and non-negative")
    if args.hazard_potential_scale > 0.0 and not args.enable_visual_features:
        raise ValueError("--hazard-potential-scale requires --enable-visual-features")
    if args.hazard_potential_scale > 0.0 and args.lagrangian_mode != "off":
        raise ValueError("--hazard-potential-scale screen requires --lagrangian-mode off")
    SPEED_TARGET = float(args.speed_target)
    SPEED_SHORTFALL_PENALTY = float(args.speed_shortfall_penalty)
    total_steps = 8 if args.smoke else args.total_steps
    max_decisions = 4 if args.smoke else args.max_decisions
    evaluation_max_decisions = 4 if args.smoke else args.evaluation_max_decisions
    updates = 1 if args.smoke else args.updates
    teacher_warmup_epochs = min(args.teacher_warmup_epochs, 1) if args.smoke else args.teacher_warmup_epochs
    if args.site_map_split is not None and args.train_only_site_map_split is not None:
        raise ValueError(
            "--site-map-split and --train-only-site-map-split are mutually exclusive"
        )
    if args.train_only_site_map_split is not None:
        if not args.defer_tune:
            raise ValueError("TRAIN-only map loading requires --defer-tune")
        split = load_train_only_site_map_split(args.train_only_site_map_split)
    elif args.site_map_split is not None:
        if args.defer_tune:
            raise ValueError(
                "--defer-tune paired screens must use --train-only-site-map-split "
                "so non-TRAIN map files are never opened"
            )
        split = load_site_map_split(args.site_map_split)
    else:
        split = DEFAULT_SPLIT
    if args.lagrangian_mode != "off" and args.train_only_site_map_split is None:
        raise ValueError(
            "Lagrangian CLI runs require --train-only-site-map-split to prevent split leakage"
        )
    expected_actor_sha256 = None
    preflight_report_sha256 = None
    if args.paired_rollout_seed is not None:
        if (args.preflight_report is None) == (args.hazard_potential_preflight is None):
            raise ValueError(
                "paired rollouts require exactly one of --preflight-report or "
                "--hazard-potential-preflight"
            )
        if args.train_only_site_map_split is None or not args.defer_tune:
            raise ValueError(
                "paired screen runs require TRAIN-only map loading and deferred TUNE"
            )
        if args.seed != args.paired_rollout_seed:
            raise ValueError("paired_rollout_seed must equal the model seed")
        if args.smoke:
            raise ValueError("smoke mode cannot be used with a frozen paired preflight")
        effective_throttle = (
            args.pedal_expansion
            if args.throttle_expansion is None
            else args.throttle_expansion
        )
        effective_brake = (
            args.pedal_expansion
            if args.brake_expansion is None
            else args.brake_expansion
        )
        if args.hazard_potential_preflight is not None:
            if args.lagrangian_mode != "off":
                raise ValueError(
                    "hazard-potential paired runs require ordinary PPO with "
                    "--lagrangian-mode off"
                )
            from training.preflight_hazard_potential_screen import (
                shared_config as hazard_shared_config,
                validate_preflight_report as validate_hazard_preflight,
            )

            preflight_shared_config = hazard_shared_config(
                total_steps=total_steps,
                updates=updates,
                max_decisions=max_decisions,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                pedal_expansion=args.pedal_expansion,
                throttle_expansion=effective_throttle,
                brake_expansion=effective_brake,
                teacher_warmup_epochs=teacher_warmup_epochs,
                teacher_max_decisions=args.teacher_max_decisions,
                teacher_warmup_batch_size=args.teacher_warmup_batch_size,
                teacher_warmup_learning_rate=args.teacher_warmup_learning_rate,
                speed_target=args.speed_target,
                speed_shortfall_penalty=args.speed_shortfall_penalty,
                obstacle_brake_reward=args.obstacle_brake_reward,
                curve_brake_reward=args.curve_brake_reward,
                recovery_clearance_reward=args.recovery_clearance_reward,
                recovery_action_rewards_enabled=not args.disable_recovery_action_rewards,
                use_hud=not args.disable_hud_branch,
                use_visual_features=args.enable_visual_features,
                use_temporal_features=args.enable_temporal_features,
                lagrangian_mode=args.lagrangian_mode,
                lagrangian_initial_lambda=args.lagrangian_initial_lambda,
                lagrangian_cost_budget=args.lagrangian_cost_budget,
                lagrangian_dual_step_size=args.lagrangian_dual_step_size,
                lagrangian_window_size=args.lagrangian_window_size,
                lagrangian_min_lambda=args.lagrangian_min_lambda,
                lagrangian_max_lambda=args.lagrangian_max_lambda,
                lagrangian_cost_gamma=args.lagrangian_cost_gamma,
                resume=str(args.resume) if args.resume is not None else None,
                initialize_from=(
                    str(args.initialize_from)
                    if args.initialize_from is not None
                    else None
                ),
                tune_selection=not args.defer_tune,
            )
            expected_actor_sha256 = validate_hazard_preflight(
                preflight_path=args.hazard_potential_preflight,
                manifest_path=args.train_only_site_map_split,
                split=split,
                seed=args.seed,
                scale=args.hazard_potential_scale,
                shared_config=preflight_shared_config,
            )
            report_path = args.hazard_potential_preflight
        else:
            if args.hazard_potential_scale > 0.0:
                raise ValueError(
                    "positive hazard-potential scale requires "
                    "--hazard-potential-preflight"
                )
            screen_ppo = PPOConfig(learning_rate=args.learning_rate)
            preflight_shared_config = {
                "total_steps": total_steps,
                "updates": updates,
                "learning_rate": args.learning_rate,
                "policy_mean_learning_rate": args.learning_rate,
                "throttle_expansion": effective_throttle,
                "brake_expansion": effective_brake,
                "max_decisions": max_decisions,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_ratio": screen_ppo.clip_ratio,
                "entropy_coefficient": screen_ppo.entropy_coefficient,
                "value_coefficient": screen_ppo.value_coefficient,
                "auxiliary_coefficient": screen_ppo.auxiliary_coefficient,
                "max_grad_norm": screen_ppo.max_grad_norm,
                "epochs": screen_ppo.epochs,
                "minibatch_size": min(32, math.ceil(total_steps / updates)),
                "teacher_warmup_epochs": teacher_warmup_epochs,
                "teacher_max_decisions": args.teacher_max_decisions,
                "teacher_warmup_batch_size": args.teacher_warmup_batch_size,
                "teacher_warmup_learning_rate": args.teacher_warmup_learning_rate,
                "recovery_action_rewards_enabled": not args.disable_recovery_action_rewards,
                "speed_target": args.speed_target,
                "speed_shortfall_penalty": args.speed_shortfall_penalty,
                "obstacle_brake_reward": args.obstacle_brake_reward,
                "curve_brake_reward": args.curve_brake_reward,
                "recovery_clearance_reward": args.recovery_clearance_reward,
                "use_hud": not args.disable_hud_branch,
                "use_visual_features": args.enable_visual_features,
                "use_temporal_features": args.enable_temporal_features,
                "tune_selection": not args.defer_tune,
                "resume": str(args.resume) if args.resume is not None else None,
                "initialize_from": (
                    str(args.initialize_from)
                    if args.initialize_from is not None
                    else None
                ),
                "lagrangian_initial_lambda": args.lagrangian_initial_lambda,
                "lagrangian_cost_budget": args.lagrangian_cost_budget,
                "lagrangian_dual_step_size": args.lagrangian_dual_step_size,
                "lagrangian_window_size": args.lagrangian_window_size,
                "lagrangian_min_lambda": args.lagrangian_min_lambda,
                "lagrangian_max_lambda": args.lagrangian_max_lambda,
                "lagrangian_cost_gamma": args.lagrangian_cost_gamma,
            }
            expected_actor_sha256 = validate_lagrangian_screen_preflight(
                preflight_path=args.preflight_report,
                manifest_path=args.train_only_site_map_split,
                split=split,
                seed=args.seed,
                mode=args.lagrangian_mode,
                shared_config=preflight_shared_config,
            )
            report_path = args.preflight_report
        preflight_report_sha256 = hashlib.sha256(
            report_path.expanduser().resolve().read_bytes()
        ).hexdigest()
    elif args.preflight_report is not None or args.hazard_potential_preflight is not None:
        raise ValueError(
            "paired preflight reports require --paired-rollout-seed"
        )
    if args.hazard_potential_scale > 0.0 and args.hazard_potential_preflight is None:
        raise ValueError(
            "positive hazard-potential scale requires the paired "
            "--hazard-potential-preflight screen"
        )
    result = train(
        output_directory=args.output,
        total_steps=total_steps,
        max_decisions=max_decisions,
        seed=args.seed,
        updates=updates,
        evaluation_max_decisions=evaluation_max_decisions,
        learning_rate=args.learning_rate,
        policy_mean_learning_rate=args.policy_mean_learning_rate,
        resume=args.resume,
        initialize_from=args.initialize_from,
        split=split,
        teacher_warmup_epochs=teacher_warmup_epochs,
        teacher_max_decisions=args.teacher_max_decisions,
        teacher_warmup_batch_size=args.teacher_warmup_batch_size,
        teacher_warmup_learning_rate=args.teacher_warmup_learning_rate,
        use_hud=not args.disable_hud_branch,
        use_visual_features=args.enable_visual_features,
        use_temporal_features=args.enable_temporal_features,
        pedal_expansion=args.pedal_expansion,
        throttle_expansion=args.throttle_expansion,
        brake_expansion=args.brake_expansion,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lagrangian_mode=args.lagrangian_mode,
        lagrangian_initial_lambda=args.lagrangian_initial_lambda,
        lagrangian_cost_budget=args.lagrangian_cost_budget,
        lagrangian_dual_step_size=args.lagrangian_dual_step_size,
        lagrangian_window_size=args.lagrangian_window_size,
        lagrangian_min_lambda=args.lagrangian_min_lambda,
        lagrangian_max_lambda=args.lagrangian_max_lambda,
        lagrangian_cost_gamma=args.lagrangian_cost_gamma,
        obstacle_brake_reward=args.obstacle_brake_reward,
        curve_brake_reward=args.curve_brake_reward,
        direct_gas_reward=args.direct_gas_reward,
        recovery_clearance_reward=args.recovery_clearance_reward,
        hazard_potential_scale=args.hazard_potential_scale,
        enable_recovery_action_rewards=not args.disable_recovery_action_rewards,
        paired_rollout_seed=args.paired_rollout_seed,
        expected_initial_actor_sha256=expected_actor_sha256,
        preflight_report_sha256=preflight_report_sha256,
        tune_selection=not args.defer_tune,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
