"""Generate official Track Lab map descriptors and an official-heavy split.

Scoring uses official tracks only (COMPETITION_INFO.md), yet the existing
`full_site_map_split.json` spends 14 of its 22 train episodes on custom maps
the actor already completes 7/7, and gives the official distribution only 8.
Official descriptors carry no geometry -- the simulator builds the track and
its six built-in obstacles from ``track_id`` and ``seed`` -- so widening the
official coverage costs nothing but rollout time.

The generated split keeps every existing tune and held-out episode so earlier
results stay comparable, and never reuses a train seed in tune or held-out.

    python -m tools.make_official_split
"""

import argparse
import json
from pathlib import Path

MAPS_DIRECTORY = Path("training/maps/site/official")
BASE_SPLIT = Path("training/maps/site/full_site_map_split.json")
OUTPUT_SPLIT = Path("training/maps/site/official_heavy_split.json")

# Existing assignments that must not move: track 1 seed 42 and track 2 seed 101
# are held out, 47/48 and 106/107 are tune, 43-46 and 102-105 are train.
NEW_TRAIN_SEEDS = {1: range(49, 69), 2: range(108, 128)}
NEW_TUNE_SEEDS = {1: range(69, 73), 2: range(128, 132)}


def descriptor_path(track_id: int, seed: int) -> Path:
    return MAPS_DIRECTORY / f"official-track{track_id}-seed{seed}.json"


def write_descriptor(track_id: int, seed: int) -> Path:
    path = descriptor_path(track_id, seed)
    payload = {
        "schema_version": 1,
        "map_kind": "official",
        "map_id": f"official-track{track_id}-seed{seed}",
        "track_id": int(track_id),
        "seed": int(seed),
        "obstacle_mode": "official",
        "obstacles": [],
        "max_steps": 2000,
        "frame_skip": 4,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def entries(seeds_by_track: dict[int, range]) -> list[dict]:
    return [
        {
            "map": f"official/official-track{track_id}-seed{seed}.json",
            "seeds": [int(seed)],
        }
        for track_id, seeds in seeds_by_track.items()
        for seed in seeds
    ]


def collect_seeds(rows: list[dict]) -> set[tuple[str, int]]:
    return {(row["map"], int(seed)) for row in rows for seed in row["seeds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_SPLIT)
    parser.add_argument("--output", type=Path, default=OUTPUT_SPLIT)
    arguments = parser.parse_args()

    base = json.loads(arguments.base.read_text(encoding="utf-8"))
    written = [
        write_descriptor(track_id, seed)
        for group in (NEW_TRAIN_SEEDS, NEW_TUNE_SEEDS)
        for track_id, seeds in group.items()
        for seed in seeds
    ]

    split = {
        "schema_version": base["schema_version"],
        "train": list(base["train"]) + entries(NEW_TRAIN_SEEDS),
        "tune": list(base["tune"]) + entries(NEW_TUNE_SEEDS),
        "held_out": list(base["held_out"]),
    }

    train, tune, held_out = (collect_seeds(split[name]) for name in ("train", "tune", "held_out"))
    for left_name, left, right_name, right in (
        ("train", train, "tune", tune),
        ("train", train, "held_out", held_out),
        ("tune", tune, "held_out", held_out),
    ):
        overlap = left & right
        if overlap:
            raise SystemExit(f"{left_name} and {right_name} share episodes: {sorted(overlap)}")

    arguments.output.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} official map descriptors")
    for name, rows in (("train", split["train"]), ("tune", split["tune"]), ("held_out", split["held_out"])):
        official = sum(len(row["seeds"]) for row in rows if "official" in row["map"])
        total = sum(len(row["seeds"]) for row in rows)
        print(f"  {name:9s} {total:3d} episodes ({official} official, {total - official} custom)")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
