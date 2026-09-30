"""Render recorded policy observations with detected road rows for debugging."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", type=Path)
    parser.add_argument("steps", type=int, nargs="+")
    args = parser.parse_args()
    obs = np.load(args.cell / "observations.npz")["observations"]
    rows = [json.loads(line) for line in (args.cell / "telemetry.jsonl").read_text().splitlines()]
    panels = []
    for index in args.steps:
        gray = (obs[index, -1] * 255).astype(np.uint8)
        panel = cv2.cvtColor(cv2.resize(gray, (336, 336), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        for hook in rows[index]["hooks"]:
            if hook["method"] == "_road_centers":
                for y, x in hook["result"].items():
                    cv2.circle(panel, (round(x * 4), int(y) * 4), 3, (0, 255, 0), -1)
                break
        panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(panel, f"step {index} steer {rows[index]['action'][0]:.3f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        panels.append(panel)
    cv2.imwrite(str(args.cell / "camera-sheet.png"), np.concatenate(panels, axis=1))
    print(args.cell / "camera-sheet.png")


if __name__ == "__main__":
    main()
