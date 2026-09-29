from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Sequence

from .preview import build_map_preview
from .schema import (
    CustomMapSpec,
    CustomObstacle,
    CustomTrackGeometry,
    MapSpec,
    map_to_dict,
)
from .track_generator import TEMPLATES, generate_custom_map, validate_custom_geometry


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


def _control_point_value(value: str) -> tuple[float, float]:
    try:
        x, y = (float(part.strip()) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "control-point must use x,y"
        ) from error
    return x, y


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
    parser.add_argument("--kind", choices=("official", "custom"), default="official")
    parser.add_argument("--track-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-id")
    parser.add_argument("--design-seed", type=int, default=42)
    parser.add_argument("--template", choices=tuple(sorted(TEMPLATES)), default="oval")
    parser.add_argument(
        "--width",
        type=float,
        default=8.0,
        help="generated custom-track half-width (0.5 to 9 world units)",
    )
    parser.add_argument(
        "--control-point",
        action="append",
        type=_control_point_value,
        default=[],
        metavar="X,Y",
        help="custom track centerline point; may be repeated",
    )
    parser.add_argument(
        "--obstacle-mode",
        choices=("official", "custom_only", "official_plus_custom"),
        default=None,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    obstacle_mode = args.obstacle_mode
    if args.kind == "official" and obstacle_mode is None:
        obstacle_mode = "official"
    if args.kind == "custom" and obstacle_mode is None:
        obstacle_mode = "custom_only"
    if obstacle_mode == "official" and (args.obstacle or args.auto_obstacles):
        parser.error("custom obstacles require custom_only or official_plus_custom")
    try:
        obstacle_seed = args.design_seed if args.kind == "custom" else args.seed
        obstacles = tuple(args.obstacle) + _auto_obstacles(
            obstacle_seed,
            args.auto_obstacles,
        )
        if args.kind == "custom":
            if obstacle_mode == "official":
                parser.error("custom maps cannot use official obstacles")
            map_id = args.map_id or f"custom-track-{args.design_seed:04d}"
            if args.control_point:
                geometry = CustomTrackGeometry(
                    centerline=tuple(args.control_point),
                    width=args.width,
                )
                validate_custom_geometry(geometry)
                spec = CustomMapSpec(
                    map_id=map_id,
                    geometry=geometry,
                    obstacles=obstacles,
                    max_steps=args.max_steps,
                    frame_skip=args.frame_skip,
                    generator=(("design_seed", args.design_seed), ("source", "manual")),
                    metadata=(("generator", "local_simulator.map"),),
                )
            else:
                generated = generate_custom_map(
                    map_id=map_id,
                    design_seed=args.design_seed,
                    template=args.template,
                    width=args.width,
                    max_steps=args.max_steps,
                    frame_skip=args.frame_skip,
                )
                spec = replace(generated, obstacles=obstacles)
        else:
            spec = MapSpec(
                track_id=args.track_id,
                seed=args.seed,
                obstacle_mode=obstacle_mode,
                obstacles=obstacles,
                max_steps=args.max_steps,
                frame_skip=args.frame_skip,
                metadata=(("generator", "local_simulator.map"),),
            )
        preview = build_map_preview(spec)
    except (TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    payload = map_to_dict(spec)
    payload["preview"] = preview
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"map: {output_path}")
    print(f"track points: {len(preview['track']['points'])}")
    print(f"official obstacles: {len(preview['official_obstacles'])}")
    print(f"custom obstacles: {len(preview['custom_obstacles'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
