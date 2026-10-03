"""Exact, source-bound telemetry for the consumed v4 track-2 contact cell."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import compare_feasible_corridor_dev as v1
from tools import compare_feasible_corridor_dev_v4 as v4
from tools import evaluate_bare_generalization as fresh
from tools import trace_feasible_corridor_dev as previous_trace
from tools.trace_feasible_corridor_dev import act_with_diagnostics as BASE_ACT_WITH_DIAGNOSTICS


TARGET = (2, 4089604952)
ARM = "candidate"
PROTOCOL_PATH = v4.PROTOCOL_PATH
RUN_ROOT = v4.OUTPUT_ROOT
OUTPUT = ROOT / ".haic-artifacts/feasible-corridor-dev-telemetry-v4/track-2-seed-4089604952/candidate.json"


def preflight() -> tuple[dict, dict]:
    protocol = fresh._read_json(PROTOCOL_PATH)
    previous_trace.require_consumed_cell(protocol, *TARGET)
    margin = fresh._read_json(v1.MARGIN_SCREEN_PROTOCOL)
    ego = fresh._read_json(v1.EGO_SCREEN_PROTOCOL)
    v4.validate_protocol(
        protocol, margin, ego, fresh.digest(v1.MARGIN_SCREEN_PROTOCOL),
        fresh.digest(v1.EGO_SCREEN_PROTOCOL),
        v4.prior.sealed_holdout_seeds(ROOT / "experiments"),
    )
    paths = previous_trace._paths(protocol, PROTOCOL_PATH)
    freeze_path = RUN_ROOT / "freeze.json"
    summary_path = RUN_ROOT / "mechanism-summary.json"
    if not freeze_path.is_file() or not summary_path.is_file():
        raise ValueError("complete frozen v4 mechanism triage is required")
    identity = v4.build_identity(PROTOCOL_PATH, protocol, paths, margin, ego)
    if fresh._read_json(freeze_path) != identity:
        raise ValueError("v4 freeze differs from frozen source/environment")
    v4.check_frozen_inputs(identity, paths, {"margin": margin, "ego": ego})
    ego_cells = v1.ego_screen_cells(ego)
    for track, seed in v1.MECHANISM_CELLS:
        previous_trace.require_frozen_receipts(
            RUN_ROOT, identity, track, seed, ego_cells,
        )
    summary = v4.report(
        RUN_ROOT, identity, v1.MECHANISM_CELLS,
        mechanism_only=True, ego_cells=ego_cells,
    )
    if fresh._read_json(summary_path) != summary:
        raise ValueError("v4 mechanism summary differs from its 20 receipts")
    rows, hashes = previous_trace.require_frozen_receipts(
        RUN_ROOT, identity, *TARGET, ego_cells,
    )
    diagnostic_identity = {
        **identity,
        "development_freeze_sha256": fresh.digest(freeze_path),
        "mechanism_summary_sha256": fresh.digest(summary_path),
        "target_receipt_sha256": hashes,
        "previous_trace_sha256": fresh.digest(Path(previous_trace.__file__)),
        "telemetry_tool_sha256": fresh.digest(Path(__file__)),
    }
    return diagnostic_identity, {
        "source": paths[ARM], "model": paths["model"],
        "class": protocol["controller_classes"][ARM],
        "reference": rows[ARM],
    }


def act_with_temporal_switches(agent, observation):
    controller = agent._forward_controller
    original = controller._allow_obstacle_side_switch
    decisions = []

    def observed_switch(*, obstacle_y, obstacle_x, candidate_side):
        before_side = float(controller._obstacle_side)
        previous = controller._temporal_obstacle
        current = controller._corridor_detection
        bbox = controller._corridor_bbox
        nearest = sorted(
            (row for row in controller._corridor_edges
             if abs(row - obstacle_y) <= 13.0),
            key=lambda row: (abs(row - obstacle_y), row),
        )[:3]
        result = original(
            obstacle_y=obstacle_y, obstacle_x=obstacle_x,
            candidate_side=candidate_side,
        )
        decisions.append({
            "obstacle_y": float(obstacle_y),
            "obstacle_x": float(obstacle_x),
            "candidate_side": float(candidate_side),
            "current_side": before_side,
            "allowed": bool(result),
            "previous_obstacle": None if previous is None else list(previous),
            "current_obstacle": None if current is None else list(current),
            "bbox": None if bbox is None else list(bbox),
            "nearest_edge_rows": [
                {"row": row,
                 "left": float(controller._corridor_edges[row][0]),
                 "right": float(controller._corridor_edges[row][1])}
                for row in nearest
            ],
        })
        return result

    with patch.object(controller, "_allow_obstacle_side_switch", observed_switch):
        action, detail = BASE_ACT_WITH_DIAGNOSTICS(agent, observation)
    detail["temporal_side_switch_decisions"] = decisions
    return action, detail


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    identity, inputs = preflight()
    if args.preflight_only:
        print(json.dumps({
            "preflight": "PASS", "arm": ARM, "cell": list(TARGET),
            "candidate_sha256": identity["source_sha256"][ARM],
            "frozen_mechanism_rows": 20,
        }, sort_keys=True))
        return 0
    with fresh._run_lock(RUN_ROOT):
        if OUTPUT.exists():
            envelope = fresh._read_json(OUTPUT)
            payload = {"identity": identity, "episode": envelope.get("episode")}
            if (envelope.get("identity") != identity
                    or envelope.get("digest") != hashlib.sha256(fresh._canonical(payload)).hexdigest()):
                raise ValueError("existing v4 telemetry differs from frozen inputs")
            print(json.dumps({"artifact": str(OUTPUT), "status": "EXISTING"}))
            return 0
        with patch.object(previous_trace, "act_with_diagnostics", act_with_temporal_switches):
            episode = previous_trace._episode(ARM, *TARGET, inputs)
        after, _ = preflight()
        if after != identity:
            raise RuntimeError("frozen v4 inputs changed during diagnostic replay")
        payload = {"identity": identity, "episode": episode}
        envelope = {**payload, "digest": hashlib.sha256(fresh._canonical(payload)).hexdigest()}
        fresh._atomic_json(OUTPUT, envelope)
    print(json.dumps({
        "artifact": str(OUTPUT), "status": "RECORDED",
        "action_trace_sha256": episode["action_trace_sha256"],
        "summary": episode["summary"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
