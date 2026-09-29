"""Benchmark the pixel corridor controller on official and Track Lab maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from agent import Agent
from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent
from haic_agent.apex_line_runtime import ApexLineAgent
from haic_agent.corridor_agent import VisionCorridorAgent
from haic_agent.fast_pedal_stable_steer_runtime import FastPedalStableSteerAgent
from haic_agent.high_speed_pivot_runtime import HighSpeedPivotAgent
from haic_agent.networks import VisualActorCritic
from haic_agent.obstacle_full_road_guard_runtime import ObstacleFullRoadGuardAgent
from haic_agent.stable_risk_envelope_runtime import StableRiskEnvelopeAgent
from haic_agent.speed_coupled_preview_runtime import SpeedCoupledPreviewAgent
from training.env_factory import TrainingEpisode
from training.evaluate_closed_loop import aggregate_episode_results, run_episode
from training.site_maps import SiteMapEpisode, SiteMapSplit, load_site_map, load_site_map_split


DEFAULT_TRACKS = ((1, 42), (2, 101))
DEFAULT_SITE_MAP_SPLIT = Path("training/maps/site/site_map_split.json")
CONTROLLER_PROFILES = {
    "safe": {"cruise_speed": 62.0, "curve_speed_penalty": 2.0, "max_gas": 0.12},
    "fast": {"cruise_speed": 70.0, "curve_speed_penalty": 2.0, "max_gas": 0.18},
    "fast_plus": {"cruise_speed": 72.0, "curve_speed_penalty": 2.0, "max_gas": 0.20},
    "sprint_guarded": {"cruise_speed": 80.0, "curve_speed_penalty": 3.0, "max_gas": 0.24},
    "race": {"cruise_speed": 76.0, "curve_speed_penalty": 1.5, "max_gas": 0.24},
    "slow_obstacle": {
        "cruise_speed": 62.0,
        "curve_speed_penalty": 2.0,
        "max_gas": 0.12,
        "obstacle_far_speed": 36.0,
        "obstacle_near_speed": 26.0,
    },
    "strong_avoid": {
        "cruise_speed": 62.0,
        "curve_speed_penalty": 2.0,
        "max_gas": 0.12,
        "obstacle_steer_scale": 1.7,
    },
}
RUNTIME_PROFILES = ("selected_full_road_guard", "fast_pedal_stable", "apex_line", "anticipatory_bend",
                    "stable_risk_envelope", "speed_coupled_reference", "pivot_road_memory",
                    "pivot_path_exit", "pivot_obstacle_side", "pivot_speed_budget")
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CHECKPOINTS = {
    "stable": (
        ROOT / "artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt",
        "3ceb5ebbe2a4bd96944649a693b8ef924c0252fa620b0fd547c7fb494afff199",
    ),
    "fast": (
        ROOT / "artifacts/haic/ppo-speed-shortfall-0p6-vs-0p0-kl0p5-u8-20260924T133743Z/run/seed8104/penalty_0p6/policy.pt",
        "647fb7c831c0b61dc78bcc416867f9f7d6070b4fa6aad51c299f7c0ae48d145d",
    ),
}


def _guarded_checkpoint_agent(name: str) -> ObstacleFullRoadGuardAgent:
    path, expected_hash = RUNTIME_CHECKPOINTS[name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"{name} checkpoint hash differs from the registered diagnostic")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved.get("metadata", {})
    config = metadata.get("model_config", metadata)
    pedal = config.get("pedal_scale", {})
    model = VisualActorCritic(
        use_hud=bool(config.get("use_hud", True)),
        use_visual_features=bool(config.get("use_visual_features", False)),
        use_temporal_features=bool(config.get("use_temporal_features", False)),
        throttle_expansion=float(pedal.get("throttle_expansion", config.get("throttle_expansion", 1.0))),
        brake_expansion=float(pedal.get("brake_expansion", config.get("brake_expansion", 1.0))),
    )
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    return ObstacleFullRoadGuardAgent(
        Agent(policy=model, planner_enabled=False, strict_checkpoint_loading=True)
    )


def _runtime_agent(profile: str):
    stable = _guarded_checkpoint_agent("stable")
    if profile == "selected_full_road_guard":
        return stable
    if profile == "stable_risk_envelope":
        return StableRiskEnvelopeAgent(stable)
    if profile == "anticipatory_bend":
        return AnticipatoryBendAgent(stable)
    if profile in {"speed_coupled_reference", "pivot_road_memory", "pivot_path_exit",
                   "pivot_obstacle_side", "pivot_speed_budget"}:
        speed = SpeedCoupledPreviewAgent(AnticipatoryBendAgent(stable))
        if profile == "speed_coupled_reference":
            return speed
        return HighSpeedPivotAgent(speed, profile.removeprefix("pivot_"))
    fast = _guarded_checkpoint_agent("fast")
    if profile == "fast_pedal_stable":
        return FastPedalStableSteerAgent(fast, stable)
    if profile == "apex_line":
        return ApexLineAgent(fast, stable)
    raise ValueError(f"unknown runtime profile: {profile}")


def _parse_track(value: str) -> tuple[int, int]:
    try:
        track, seed = value.split(":", maxsplit=1)
        track_id = int(track)
        track_seed = int(seed)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("track must be written as TRACK_ID:SEED") from error
    if track_id <= 0 or not 0 <= track_seed <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("track ID must be positive and seed must fit uint32")
    return track_id, track_seed


def select_site_episodes(
    split: SiteMapSplit, *, group: str = "held_out", limit: int | None = None
) -> tuple[SiteMapEpisode, ...]:
    """Choose one split group while preserving its map and seed identities."""
    if group not in {"train", "tune", "held_out"}:
        raise ValueError("group must be train, tune, or held_out")
    episodes = tuple(getattr(split, group))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        episodes = episodes[:limit]
    return episodes


def load_direct_site_episodes(paths: Iterable[Path | str]) -> tuple[SiteMapEpisode, ...]:
    """Load explicit Track Lab map files in addition to a split group."""
    episodes = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        site_map = load_site_map(resolved)
        episodes.append(
            SiteMapEpisode(site_map=site_map, seed=site_map.seed, source_path=resolved)
        )
    return tuple(episodes)


def run_corridor_benchmark(
    *,
    tracks: Iterable[tuple[int, int]] = DEFAULT_TRACKS,
    site_episodes: Iterable[SiteMapEpisode] = (),
    profiles: Iterable[str] = ("safe",),
    max_decisions: int = 800,
) -> dict[str, object]:
    """Run repeated pixel-only driving checks and retain failures and motion data."""
    episodes: tuple[TrainingEpisode, ...] = tuple(tracks) + tuple(site_episodes)
    selected_profiles = tuple(dict.fromkeys(profiles))
    if not episodes:
        raise ValueError("at least one official track or site-map episode is required")
    if not selected_profiles:
        raise ValueError("at least one controller profile is required")
    invalid_profiles = set(selected_profiles) - set(CONTROLLER_PROFILES) - set(RUNTIME_PROFILES)
    if invalid_profiles:
        raise ValueError(f"unknown controller profile(s): {sorted(invalid_profiles)}")
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")

    records = []
    for profile in selected_profiles:
        for episode in episodes:
            agent = (_runtime_agent(profile) if profile in RUNTIME_PROFILES
                     else VisionCorridorAgent(**CONTROLLER_PROFILES[profile]))
            if isinstance(episode, SiteMapEpisode):
                record = run_episode(
                    mode="corridor",
                    episode=episode,
                    agent=agent,
                    max_decisions=max_decisions,
                    plan_budget_seconds=0.0,
                    capture_trace=True,
                )
            else:
                track_id, seed = episode
                record = run_episode(
                    mode="corridor",
                    track_id=int(track_id),
                    seed=int(seed),
                    agent=agent,
                    max_decisions=max_decisions,
                    plan_budget_seconds=0.0,
                    capture_trace=True,
                )
            record["controller_mode"] = "pixel_runtime" if profile in RUNTIME_PROFILES else "corridor"
            record["controller_profile"] = profile
            record["controller_diagnostics"] = (
                agent.diagnostics() if profile in CONTROLLER_PROFILES
                else {"apex_decisions": getattr(agent, "apex_decisions", 0),
                      "risk_clear_decisions": getattr(agent, "risk_clear_decisions", 0),
                      "preview_count": getattr(agent, "preview_count", 0),
                      "activation_count": getattr(agent, "activation_count", 0),
                      "fast_pedal_decisions": getattr(agent, "fast_pedal_decisions", 0),
                      "stable_decisions": getattr(agent, "stable_decisions", 0)}
            )
            records.append(record)

    summary = aggregate_episode_results(records)
    summary["by_profile"] = {
        profile: aggregate_episode_results(
            [record for record in records if record["controller_profile"] == profile]
        )["by_mode"]["corridor"]
        for profile in selected_profiles
    }

    return {
        "schema_version": 1,
        "controller": "pixel-only corridor or registered checkpoint runtime diagnostic",
        "controller_inputs": "84x84 grayscale image stack only",
        "motion_metrics_source": "local simulator speed magnitude sampled at the decision interval",
        "acceleration_note": (
            "legacy peak acceleration/deceleration include nonterminal speed deltas, including collision frames; "
            "non_collision_* metrics exclude collision frames, collision_peak_deceleration reports impact speed drops, "
            "and terminal-step speed change is separate"
        ),
        "trace_note": "per-decision action, progress, speed, collision, and damage are retained",
        "max_decisions": int(max_decisions),
        "profiles": {
            profile: (CONTROLLER_PROFILES[profile] if profile in CONTROLLER_PROFILES
                      else {"runtime": profile, "checkpoints": {
                          name: {"path": str(path.relative_to(ROOT)), "sha256": digest}
                          for name, (path, digest) in RUNTIME_CHECKPOINTS.items()
                          if name == "stable" or profile != "selected_full_road_guard"
                      }})
            for profile in selected_profiles
        },
        "episodes": records,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        dest="tracks",
        action="append",
        type=_parse_track,
        help="official track and seed, written as TRACK_ID:SEED; repeat to add checks",
    )
    parser.add_argument("--site-map-split", type=Path)
    parser.add_argument("--site-group", choices=("train", "tune", "held_out"), default="held_out")
    parser.add_argument("--site-limit", type=int)
    parser.add_argument(
        "--site-map",
        dest="site_maps",
        action="append",
        type=Path,
        help="validated Track Lab map JSON to add directly; repeat to add more checks",
    )
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        choices=tuple(CONTROLLER_PROFILES) + RUNTIME_PROFILES,
        help="controller profile to compare; repeat to compare several (default: safe)",
    )
    parser.add_argument("--max-decisions", type=int, default=800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/haic/corridor-benchmark/report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks = tuple(args.tracks) if args.tracks else DEFAULT_TRACKS
    site_episodes: tuple[SiteMapEpisode, ...] = ()
    if args.site_map_split is not None:
        split = load_site_map_split(args.site_map_split)
        site_episodes = select_site_episodes(
            split,
            group=args.site_group,
            limit=args.site_limit,
        )
    site_episodes = site_episodes + load_direct_site_episodes(args.site_maps or ())
    report = run_corridor_benchmark(
        tracks=tracks,
        site_episodes=site_episodes,
        profiles=tuple(args.profiles) if args.profiles else ("safe",),
        max_decisions=args.max_decisions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
