from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from .logging import save_run_log
from .map import default_artifact_root
from .policies import BaselinePolicy
from .schema import map_from_dict
from .simulation import run_episode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official local simulator and save a replay log."
    )
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_artifact_root() / "runs" / "run.json",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--frame-skip", type=int)
    parser.add_argument(
        "--record-frames",
        action="store_true",
        help="embed JPEG camera frames in the JSON log",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.map_path.read_text(encoding="utf-8"))
        spec = map_from_dict(payload)
        overrides = {}
        if args.max_steps is not None:
            overrides["max_steps"] = args.max_steps
        if args.frame_skip is not None:
            overrides["frame_skip"] = args.frame_skip
        if overrides:
            spec = replace(spec, **overrides)
        result = run_episode(spec, BaselinePolicy(), record_frames=args.record_frames)
        save_run_log(result, args.output)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(f"run: {args.output}")
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
