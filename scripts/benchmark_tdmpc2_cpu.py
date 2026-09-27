"""Measure TD-MPC2's full CPU planner contract without environment interaction."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time


def resident_mib() -> float:
    with open("/proc/self/statm", encoding="ascii") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1048576


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trials < 2 or args.threads < 1:
        parser.error("trials must be at least 2 and threads positive")

    start = time.perf_counter()
    import torch

    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
    from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = WorldModel(TDMPC2ModelConfig(obs_shape={"rgb": (4, 64, 64)}, episodic=True)).eval()
    planner = TDMPC2Planner(model, PlannerConfig(action_dim=3, discount=0.995, episodic=True))
    constructed_at = time.perf_counter()
    initial_rss = resident_mib()
    obs = torch.zeros((1, 4, 64, 64), dtype=torch.uint8)
    times = []
    rss = [initial_rss]
    for index in range(args.trials):
        tick = time.perf_counter()
        action = planner.plan(obs, t0=index == 0, eval_mode=True)
        elapsed = time.perf_counter() - tick
        if action.shape != (3,) or not torch.isfinite(action).all() or torch.any(action.abs() > 1):
            raise RuntimeError("invalid full-planner action")
        times.append(elapsed)
        rss.append(resident_mib())
    warm = sorted(times[1:])
    print(json.dumps({
        "scope": "untrained-model-shape-latency-proxy-not-official-acceptance",
        "torch": torch.__version__,
        "threads": args.threads,
        "trials": args.trials,
        "import_and_construct_s": constructed_at - start,
        "first_act_s": times[0],
        "warm_act_median_s": statistics.median(warm),
        "warm_act_p95_s": warm[int((len(warm) - 1) * 0.95)],
        "max_act_s": max(times),
        "initial_resident_mib": initial_rss,
        "max_sampled_resident_mib": max(rss),
        "official_limits": {"construct_s": 10, "reset_s": 5, "act_s": 5, "resident_mib": 1024},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
