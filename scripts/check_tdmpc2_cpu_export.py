"""Read-only CPU Torch 2.1 smoke for TD-MPC2 model-only exported weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def _rss_mib() -> float:
    with open("/proc/self/statm", encoding="ascii") as stream:
        pages = int(stream.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1048576


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-only-state", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--trace-length", type=int, default=4)
    args = parser.parse_args()
    if args.trace_length < 2 or args.threads < 1:
        parser.error("trace-length >= 2 and threads >= 1 required")
    if not args.model_only_state.is_file():
        parser.error("model-only export is missing")

    start = time.perf_counter()
    import torch

    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
    from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner

    if not torch.__version__.startswith("2.1.0") or torch.cuda.is_available():
        raise RuntimeError("probe must run with official CPU-only Torch 2.1.0")
    torch.set_num_threads(args.threads)
    model = WorldModel(TDMPC2ModelConfig(obs_shape={"rgb": (4, 64, 64)}, episodic=True))
    state = torch.load(args.model_only_state, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("expected a model-only state dictionary")
    if set(state) == {"model"} and isinstance(state["model"], dict):
        weights, export_kind = state["model"], "synthetic_plain"
        lineage = {}
    elif (state.get("format") == "haic-tdmpc2-cpu-model-v1"
          and isinstance(state.get("model_state"), dict)
          and all(isinstance(state.get(key), str) and len(state[key]) == 64
                  for key in ("protocol_sha256", "checkpoint_sha256"))
          and isinstance(state.get("source_sha256"), dict)):
        weights, export_kind = state["model_state"], "source_bound_training_export"
        lineage = {"protocol_sha256": state["protocol_sha256"],
                   "checkpoint_sha256": state["checkpoint_sha256"],
                   "decisions": state.get("decisions"), "updates": state.get("updates")}
    else:
        raise ValueError("expected a plain synthetic model or source-bound CPU model export")
    model.load_state_dict(weights, strict=True)
    model.eval()
    planner = TDMPC2Planner(model, PlannerConfig(action_dim=3, discount=.995, episodic=True))
    construction_s = time.perf_counter() - start
    pixels = torch.zeros((1, 4, 64, 64), dtype=torch.uint8)
    traces = []
    latencies = []
    max_rss = _rss_mib()
    for _ in range(2):
        torch.manual_seed(args.seed)
        planner.reset()
        trace = hashlib.sha256()
        for index in range(args.trace_length):
            tick = time.perf_counter()
            action = planner.plan(pixels, t0=index == 0, eval_mode=True)
            elapsed = time.perf_counter() - tick
            if action.shape != (3,) or not torch.isfinite(action).all() or torch.any(action.abs() > 1):
                raise ValueError("model-only export produced invalid planner action")
            trace.update(action.cpu().numpy().astype("<f4").tobytes())
            latencies.append(elapsed)
            max_rss = max(max_rss, _rss_mib())
        traces.append(trace.hexdigest())
    if traces[0] != traces[1]:
        raise AssertionError("CPU model-only export does not reproduce an action trace")
    print(json.dumps({
        "scope": "standalone CPU-only synthetic observation checkpoint-load proxy; not official Agent",
        "torch": torch.__version__,
        "export_kind": export_kind,
        "lineage": lineage,
        "model_only_sha256": hashlib.sha256(args.model_only_state.read_bytes()).hexdigest(),
        "seed": args.seed,
        "trace_length": args.trace_length,
        "identical_traces": True,
        "action_trace_sha256": traces[0],
        "construct_and_load_s": construction_s,
        "first_act_s": latencies[0],
        "max_act_s": max(latencies),
        "max_sampled_process_rss_mib": max_rss,
        "official_limits": {"construction_s": 10, "each_act_s": 5, "process_mib": 1024},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
