"""Render one hash-verified G0 window contact sheet for exploratory inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.analyze_rlpd_g0_windows import ROOT, _pinned, _json, sha256_file


def render_window(
    root: Path, run_dir: str, *, geometry_seed: int, actor_id: str,
    anchor: str, manifest_sha256: str, output_path: Path,
    allowed_output_dir: Path = Path("/tmp/kilo"),
) -> dict:
    root = root.resolve()
    relative = Path(run_dir)
    if (
        not isinstance(run_dir, str) or relative.is_absolute()
        or len(relative.parts) != 2 or relative.parts[0] != "runs"
        or ".." in relative.parts or relative.as_posix() != run_dir
    ):
        raise ValueError("window source must be a repository-relative runs/<name> directory")
    if (
        not isinstance(output_path, Path) or not output_path.is_absolute()
        or output_path.suffix.lower() != ".png" or output_path.exists()
        or output_path.parent.resolve() != allowed_output_dir.resolve()
        or output_path.is_symlink()
    ):
        raise ValueError("render output must be a new PNG in the approved temporary directory")
    manifest_path = (relative / "manifest.json").as_posix()
    manifest = _json(_pinned(root, manifest_path, manifest_sha256, "runs"))
    if (manifest.get("format") != "haic-rlpd-g0-fixed-windows-result-v1"
            or manifest.get("labels_assigned") is not False or manifest.get("causal_claim") is not None
            or manifest.get("row_count") != 24 or manifest.get("window_count") != 144):
        raise ValueError("window manifest is not a complete, unlabelled TRAIN diagnostic")
    freeze = _json(_pinned(
        root, manifest["freeze_path"], manifest["freeze_sha256"],
        Path(manifest["freeze_path"]).parts[0],
    ))
    if freeze.get("format") != "haic-rlpd-g0-fixed-windows-freeze-v1" or freeze.get("status") != "frozen":
        raise ValueError("window freeze is not the declared fixed rule")
    if sha256_file(root / "scripts/analyze_rlpd_g0_windows.py") != freeze.get("analysis_source_sha256"):
        raise ValueError("window extractor source changed after its freeze")
    for key in ("manifest", "protocol", "cells"):
        _pinned(root, manifest["source"][f"{key}_path"], manifest["source"][f"{key}_sha256"],
                "experiments" if key == "protocol" else "runs")
    ledger_relative = (relative / manifest["windows_path"]).as_posix()
    ledger = _pinned(root, ledger_relative, manifest["windows_sha256"], "runs")
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 144:
        raise ValueError("window ledger is incomplete")
    matches = [row for row in rows if row.get("geometry_seed") == geometry_seed
               and row.get("actor_id") == actor_id and row.get("anchor") == anchor]
    if len(matches) != 1 or not matches[0].get("tiles_path"):
        raise ValueError("requested window is missing or ambiguous")
    item = matches[0]
    tile_relative = (relative / item["tiles_path"]).as_posix()
    tile_path = _pinned(root, tile_relative, item["tiles_sha256"], "runs")
    with np.load(tile_path, allow_pickle=False) as archive:
        if "contact_sheet_tiles" not in archive.files or "contact_sheet_steps" not in archive.files:
            raise ValueError("window tile archive has no contact-sheet frames")
        tiles = archive["contact_sheet_tiles"]
        steps = archive["contact_sheet_steps"]
    if (tiles.ndim != 4 or tiles.shape[1:] != (2, 84, 84) or tiles.dtype != np.uint8
            or steps.ndim != 1 or len(steps) != len(tiles)
            or steps.tolist() != item["contact_sheet_steps"]
            or not np.all(np.diff(steps) > 0)
            or len(tiles) > 12):
        raise ValueError("window contact-sheet indexing/pixels are malformed")

    height, width = len(tiles) * 112, 2 * 84 + 32
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for index, (step, pair) in enumerate(zip(steps, tiles)):
        y = index * 112
        canvas[y + 23:y + 107, 8:92] = cv2.cvtColor(pair[0], cv2.COLOR_GRAY2BGR)
        canvas[y + 23:y + 107, 100:184] = cv2.cvtColor(pair[1], cv2.COLOR_GRAY2BGR)
        color = (0, 230, 255) if step == item["anchor_step"] else (255, 255, 255)
        cv2.putText(canvas, f"step {int(step)}{' ANCHOR' if step == item['anchor_step'] else ''}",
                    (8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not write temporary contact sheet: {output_path}")
    return {
        "source_window_path": tile_relative, "source_window_sha256": item["tiles_sha256"],
        "geometry_seed": geometry_seed, "actor_id": actor_id, "anchor": anchor,
        "window_start": item["window_start"], "window_end_inclusive": item["window_end_inclusive"],
        "image_path": str(output_path), "image_sha256": sha256_file(output_path),
        "labels_assigned": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--geometry-seed", type=int, required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render_window(
        ROOT, args.run_dir, geometry_seed=args.geometry_seed,
        actor_id=args.actor_id, anchor=args.anchor,
        manifest_sha256=args.manifest_sha256, output_path=args.output,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
