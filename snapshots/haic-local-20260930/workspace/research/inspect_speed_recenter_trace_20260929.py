"""Summarize the longest wheel departure in the consumed track-2 trace."""

import json
from pathlib import Path
import sys


path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/haic-research-v2/speed-preview-road-recenter-track2-trace-20260929/events.jsonl")
for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
    samples = row["road_samples"]
    decisions = row["decision_trace"]
    runs = []
    index = 0
    while index < len(samples):
        if samples[index]["off_wheels"] != 4:
            index += 1
            continue
        end = index + 1
        while end < len(samples) and samples[end]["off_wheels"] == 4:
            end += 1
        runs.append((index, end))
        index = end
    start, end = max(runs, key=lambda item: item[1] - item[0])
    print(row["arm"], "start", start, "length", end - start)
    for index in range(max(0, start - 8), min(len(samples), start + 14)):
        decision, sample = decisions[index], samples[index]
        print(index, round(decision["progress"], 3), round(decision["speed"], 1),
              round(decision["steer"], 2), round(decision["gas"], 2),
              sample["centers"].get("54"), sample["centers"].get("30"), sample["off_wheels"])
