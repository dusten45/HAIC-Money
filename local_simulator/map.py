from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Sequence

from .environment import build_map_bundle
from .schema import CustomObstacle, MapSpec, map_to_dict


DEFAULT_ARTIFACT_ROOT = Path("D:/HAIC")


def default_artifact_root() -> Path:
    """Prefer the requested D: workspace, with a local fallback."""
    if DEFAULT_ARTIFACT_ROOT.drive and DEFAULT_ARTIFACT_ROOT.exists():
        return DEFAULT_ARTIFACT_ROOT
    return Path.cwd() / ".haic-artifacts"


def _obstacle_value(value: str) -> CustomObstacle:
    try:
        progress, lateral, radius = (float(part.strip()) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "obstacle must use progress,lateral,radius"
        ) from error
    try:
        return CustomObstacle(progress, lateral, radius)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _auto_obstacles(seed: int, count: int) -> tuple[CustomObstacle, ...]:
    if count < 0:
        raise ValueError("auto obstacle count must be non-negative")
    rng = random.Random(seed ^ 0xC0DE)
    if count == 0:
        return ()
    obstacles: list[CustomObstacle] = []
    for index in range(count):
        ratio = (index + 1) / (count + 1)
        progress = 0.12 + ratio * 0.76
        lateral = rng.uniform(-0.55, 0.55)
        obstacles.append(CustomObstacle(progress, lateral, 1.2))
    return tuple(obstacles)


def _build_parser() -> argparse.ArgumentParser:
    root = default_artifact_root()
    parser = argparse.ArgumentParser(
        description="Create a reproducible HAIC track and obstacle map."
    )
    parser.add_argument("--track-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--obstacle-mode",
        choices=("official", "custom_only", "official_plus_custom"),
        default="official",
    )
    parser.add_argument(
        "--obstacle",
        action="append",
        type=_obstacle_value,
        default=[],
        metavar="PROGRESS,LATERAL,RADIUS",
        help="custom obstacle; may be repeated",
    )
    parser.add_argument(
        "--auto-obstacles",
        type=int,
        default=0,
        metavar="COUNT",
        help="add deterministic custom obstacles from the map seed",
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "maps" / "map.json",
    )
    return parser


def _preview(bundle: Any) -> dict[str, Any]:
    return {
        "track": {
            "points": [list(point) for point in bundle.track.points],
            "width": bundle.track.width,
        },
        "official_obstacles": [list(position) for position in bundle.official_obstacles],
        "custom_obstacles": [
            {
                "position": list(obstacle.position),
                "radius": obstacle.radius,
            }
            for obstacle in bundle.custom_obstacles
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.obstacle_mode == "official" and (args.obstacle or args.auto_obstacles):
        parser.error("custom obstacles require custom_only or official_plus_custom")
    try:
        obstacles = tuple(args.obstacle) + _auto_obstacles(args.seed, args.auto_obstacles)
        spec = MapSpec(
            track_id=args.track_id,
            seed=args.seed,
            obstacle_mode=args.obstacle_mode,
            obstacles=obstacles,
            max_steps=args.max_steps,
            frame_skip=args.frame_skip,
            metadata=(("generator", "local_simulator.map"),),
        )
        bundle = build_map_bundle(spec)
    except (TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    payload = map_to_dict(spec)
    payload["preview"] = _preview(bundle)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"map: {output_path}")
    print(f"track points: {len(bundle.track.points)}")
    print(f"official obstacles: {len(bundle.official_obstacles)}")
    print(f"custom obstacles: {len(bundle.custom_obstacles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
