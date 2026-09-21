"""Benchmark the pixel corridor controller on official and Track Lab maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from haic_agent.corridor_agent import VisionCorridorAgent
from training.env_factory import TrainingEpisode
from training.evaluate_closed_loop import aggregate_episode_results, run_episode
from training.site_maps import SiteMapEpisode, SiteMapSplit, load_site_map_split


DEFAULT_TRACKS = ((1, 42), (2, 101))
DEFAULT_SITE_MAP_SPLIT = Path("training/maps/site/site_map_split.json")


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


def run_corridor_benchmark(
    *,
    tracks: Iterable[tuple[int, int]] = DEFAULT_TRACKS,
    site_episodes: Iterable[SiteMapEpisode] = (),
    max_decisions: int = 800,
) -> dict[str, object]:
    """Run repeated pixel-only driving checks and retain failures and motion data."""
    episodes: tuple[TrainingEpisode, ...] = tuple(tracks) + tuple(site_episodes)
    if not episodes:
        raise ValueError("at least one official track or site-map episode is required")
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")

    records = []
    for episode in episodes:
        agent = VisionCorridorAgent()
        if isinstance(episode, SiteMapEpisode):
            record = run_episode(
                mode="corridor",
                episode=episode,
                agent=agent,
                max_decisions=max_decisions,
                plan_budget_seconds=0.0,
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
            )
        record["controller_mode"] = "corridor"
        records.append(record)

    return {
        "schema_version": 1,
        "controller": "pixel corridor, HUD speed governor, and obstacle avoidance",
        "controller_inputs": "84x84 grayscale image stack only",
        "motion_metrics_source": "local simulator telemetry sampled at the decision interval",
        "max_decisions": int(max_decisions),
        "episodes": records,
        "summary": aggregate_episode_results(records),
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
    report = run_corridor_benchmark(
        tracks=tracks,
        site_episodes=site_episodes,
        max_decisions=args.max_decisions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
