"""Source-bound observation-only controller tracing on consumed diagnostic cells."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CELLS = ((1, 11), (1, 21), (1, 42), (2, 644062))
DOCUMENTS = ("RESTRICTIONs.md", "COMPETITION_INFO.md", "RESULTS.md", "SOTA.md", "report.pdf")
HOOKS = (
    "_road_centers", "_road_sweep", "_nearest_obstacle", "_estimate_speed",
    "_adjust_road_steering", "_adjust_obstacle_steering", "_adjust_obstacle_urgency",
    "_adjust_target_speed", "_adjust_target_speed_for_steering", "_pedals",
    "_adjust_pedals", "_curve_brake_envelope", "_lost_road_action",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(protocol_path: Path, expected: dict[str, str]) -> dict:
    for name in DOCUMENTS:
        if not (ROOT / name).is_file() or not (ROOT / name).stat().st_size:
            raise ValueError(f"Required document missing/empty: {name}")
    paths = {"source": ROOT / "agent.py", "model": ROOT / "model.pt", "protocol": protocol_path}
    for name, path in paths.items():
        if digest(path) != expected[name].lower():
            raise ValueError(f"Frozen {name} hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cells = tuple(
        (cell["track_id"], cell["seed"]) if isinstance(cell, dict) else tuple(cell)
        for cell in protocol["diagnostic_cells"]
    )
    if cells != CELLS:
        raise ValueError("Only the four ordered, consumed diagnostic cells are permitted")
    if type(protocol["max_steps"]) is not int or not 1 <= protocol["max_steps"] <= 400:
        raise ValueError("Diagnostic max_steps must be an integer in [1, 400]")
    if protocol.get("frame_skip") != 4:
        raise ValueError("Official frame_skip must be 4")
    if any(protocol.get(key) is not False for key in ("training", "submission", "sota_promotion")):
        raise ValueError("Protocol must explicitly disable training/submission/SOTA promotion")
    for key, kind in (("source_sha256", "source"), ("model_sha256", "model")):
        if key in protocol and protocol[key].lower() != expected[kind].lower():
            raise ValueError(f"Protocol {key} disagrees with frozen CLI hash")
    protected: set[int] = set()
    for path in sorted((ROOT / "experiments").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, partition in record.get("partitions", {}).items():
            if name not in {"development", "screen"}:
                protected.update(partition.get("seeds", []))
        for key, value in record.items():
            if "reserved" in key and "seed" in key and key != "reserved_training_seeds":
                if isinstance(value, list) and all(type(seed) is int for seed in value):
                    protected.update(value)
    overlap = {seed for _, seed in CELLS} & protected
    if overlap:
        raise ValueError(f"Reserved holdout geometry requested: {sorted(overlap)}")
    return {"protocol": protocol, "hashes": expected,
            "holdout_overlap": [],
            "documents": {name: digest(ROOT / name) for name in DOCUMENTS}}


def diagnostic_json(value):
    import numpy as np
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


def install_hooks(controller, events: list[dict]) -> None:
    """Record only actual method invocations; never invoke extra perception calls."""
    import numpy as np

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.size <= 8:
                return value.tolist()
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported hook value: {type(value).__name__}")

    def make_hook(name, original):
        def traced(*args, **kwargs):
            event = {"method": name, "args": encode(args), "kwargs": encode(kwargs)}
            events.append(event)
            result = original(*args, **kwargs)
            event["result"] = encode(result)
            return result
        return traced

    for name in HOOKS:
        setattr(controller, name, make_hook(name, getattr(controller, name)))


def scalar_state(controller) -> dict:
    import numpy as np
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in vars(controller).items()
        if value is None or isinstance(value, (str, int, float, bool, np.generic))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-sha256", "--agent-sha256", dest="source_sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    expected = {kind: getattr(args, f"{kind}_sha256").lower() for kind in ("source", "model", "protocol")}
    receipt = preflight(args.protocol, expected)
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", **receipt}, indent=2))
        return 0

    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from agent import Agent
    from local_simulator.logging import run_log_to_dict
    from local_simulator.schema import MapSpec
    from local_simulator.session import SimulationSession

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = ROOT / ".haic-artifacts" / "corridor-failure-trace-v1" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    receipt["harness_sha256"] = digest(Path(__file__))
    receipt["environment_hashes"] = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((ROOT / "core").rglob("*.py")) + [ROOT / "env_wrapper.py", ROOT / "damage.py"]
    }
    (output / "freeze.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(str(output), flush=True)
    max_steps = receipt["protocol"]["max_steps"]
    for track, seed in CELLS:
        preflight(args.protocol, expected)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        cell_output = output / f"track-{track}-seed-{seed}"
        cell_output.mkdir()
        observations = []
        session = None
        failure = None
        try:
            policy = Agent(model_path=str(ROOT / "model.pt"))
            controller = policy._forward_controller
            if controller is None:
                raise ValueError("This diagnostic requires the active bare controller")
            expected_class = receipt["protocol"].get("controller_class", receipt["protocol"].get("controller"))
            if expected_class and type(controller).__name__ != expected_class:
                raise ValueError("Active controller does not match the protocol")
            events: list[dict] = []
            install_hooks(controller, events)
            session = SimulationSession.start(MapSpec(track, seed, "official", (), max_steps, 4), policy)
            with (cell_output / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
                while not session.done and len(session.steps) < max_steps:
                    observation = np.asarray(session.observation)
                    if observation.shape != (4, 84, 84) or observation.dtype != np.float32:
                        raise ValueError("Unexpected official observation contract")
                    observations.append(observation.copy())
                    events.clear()
                    row = {"step": len(session.steps), "controller": type(controller).__name__,
                           "state_before": scalar_state(controller)}
                    try:
                        # The policy receives pixels only. Environment telemetry is collected afterward.
                        action = np.asarray(policy.act(observation))
                        row["action"] = action.tolist()
                        if action.shape != (3,) or not np.isfinite(action).all():
                            raise ValueError("Policy emitted invalid action")
                        if np.any(action < [-1, 0, 0]) or np.any(action > [1, 1, 1]):
                            raise ValueError("Policy action out of official bounds")
                        row["outcome"] = asdict(session.step(action))
                        row["environment_info"] = dict(session.last_info)
                    except BaseException:
                        row["error"] = traceback.format_exc()
                        raise
                    finally:
                        row["hooks"] = list(events)
                        row["state_after"] = scalar_state(controller)
                        stream.write(json.dumps(row, allow_nan=False, default=diagnostic_json) + "\n")
                        stream.flush()
        except BaseException:
            failure = traceback.format_exc()
            raise
        finally:
            np.savez_compressed(cell_output / "observations.npz", observations=(
                np.stack(observations) if observations else np.empty((0, 4, 84, 84), dtype=np.float32)
            ))
            if session is not None:
                try:
                    raw = run_log_to_dict(session.finish(reason="diagnostic_error" if failure else None))
                    (cell_output / "run.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
                finally:
                    session.close()
            if failure:
                (cell_output / "error.txt").write_text(failure, encoding="utf-8")
        print(json.dumps({"cell": [track, seed], "steps": len(observations), "artifact": str(cell_output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
