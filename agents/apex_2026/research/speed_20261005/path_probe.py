"""Development-only replay inspection; privileged fields never enter inference."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--track", type=int, default=1)
    parser.add_argument("--seed", type=int, default=516237)
    parser.add_argument("--steps", type=int, default=65)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("path_probe_submission", args.source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    agent = module.Agent()
    map_spec = MapSpec(args.track, args.seed, "official", (), 700, 4)
    environment, raw = create_environment(map_spec, render_mode=None)
    original = json.loads(args.trace.read_text())
    rows, frames = [], []
    try:
        observation, _ = reset_environment(environment, map_spec)
        agent.reset(observation)
        for step, saved in enumerate(original[:args.steps]):
            frame = agent._frame(observation)
            road = agent._road(frame)
            circles = agent._circles(frame, road) if road else []
            action = agent.act(observation)
            np.testing.assert_array_equal(action, np.asarray(saved["action"], np.float32))
            rows.append({"step": step, "circles": circles, "speed": agent.last_speed,
                         "yaw": agent.last_yaw, "target": agent.last_target,
                         "steer": agent.last_steer, "pass_side": agent.pass_side,
                         "pass_y": agent.pass_y, "center": agent.last_center,
                         "road_near": float(np.interp(0., road[0], road[1])) if road else None,
                         "road_preview": float(np.interp(20., road[0], road[1])) if road else None})
            frames.append(frame)
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                break
    finally:
        environment.close()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"inspection.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(args.output/"frames.npz", frames=frames)
    print(json.dumps(rows[-1]))


if __name__ == "__main__":
    main()
