"""Conditional CPU21 H3 MPPI evaluation on four repeatedly consumed TRAIN roads.

Default --preflight is read-only: no torch.load, environment construction or reset.
--execute is conditional on a separately frozen protocol and an independently
SHA-bound PASS from the archived five-action branch scorer. No protocol or score
receipt is generated here. Only trusted checkpoints may be used: torch.load is
pickle and is reached only after complete source/ledger/score verification.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import torch

from scripts import evaluate_tdmpc2_full_train as raw
from scripts import evaluate_tdmpc2_h3_noise_train as noise


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-reward-overshoot-full-consumed-train-v1"
TRAIN_FORMAT = "haic-tdmpc2-reward-overshoot-train-v1"
TRAIN_PROTOCOL = {"path": "experiments/tdmpc2-reward-overshoot-train-v1.json",
                  "sha256": "ecceda92de76c2decdde13bc6661c706adfda22b8e8067e84ebd8cd2697a692b"}
RAW_EVALUATOR_SHA = "d2f3df9522257f2dfbffe5167571c5a491b7929acbbbcb3b84ebb0c2f8b91089"
NOISE_EVALUATOR_SHA = "64518efb0a33efdd33ccdcec44c7d140e6859e60f2f0bb57d3e178c22597e17c"
RESOURCE_PROTOCOL = {"path": "experiments/tdmpc2-h3-noise-full-consumed-train-v1.json",
                     "sha256": "c82b9ac2b5f20d4b6115f07010d2c38d81c895eea7e35f17f2c288a7dca8d40d"}
BRANCH_PROTOCOL = {"path": "experiments/tdmpc2-h5-branches-v1.json",
                   "sha256": "a29cbcd2e59687201a7b0fba68e119f596e88633e06a51300639bf78461b33c9"}
BRANCH_PRIMARY = {"path": "runs/tdmpc2-raw100k-h5-branches-20260929-v1.json",
                  "sha256": "23f7e6898b362cd789b92d6a1f2f184e1de551596458ae08117058570f1bcefd"}
PLANNER = {"action_dim": 3, "horizon": 3, "num_samples": 512, "num_pi_trajs": 24,
           "iterations": 6, "num_elites": 64, "discount": .995, "episodic": True,
           "eval_mode": True}
RESOURCE_KEYS = frozenset({"measured_peak_rss_bytes", "additional_memory_bytes",
                           "memory_reserve_bytes", "remaining_disk_bytes", "disk_reserve_bytes",
                           "max_wall_seconds"})


def _ref(root: Path, ref: object, prefix: str) -> tuple[dict, Path]:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str) or not ref["path"].startswith(prefix)):
        raise ValueError(f"invalid reference: {prefix}")
    return raw._reference(root, ref, prefix)


def _runtime() -> dict:
    """Read native Box2D/TimeLimit identity without constructing the HAIC env."""
    import Box2D
    import Box2D._Box2D as native
    import cv2
    import gymnasium
    from gymnasium.wrappers.time_limit import TimeLimit

    if Box2D.__file__ is None or native.__file__ is None:
        raise ValueError("native Box2D is unavailable")
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda, "numpy": np.__version__,
            "opencv": str(getattr(cv2, "__version__")), "gymnasium": gymnasium.__version__,
            "box2d": Box2D.__version__, "box2d_native_sha256": raw._digest(Path(native.__file__)),
            "box2d_python_sha256": raw._digest(Path(Box2D.__file__)),
            "timelimit_source_sha256": raw._digest(Path(inspect.getfile(TimeLimit)))}


def _source(root: Path, source: object) -> tuple[dict, dict, dict]:
    """Validate the separate trainer protocol, complete result and fixed final pin."""
    if (not isinstance(source, dict) or set(source) != {"protocol", "result", "checkpoint"}
            or source["protocol"] != TRAIN_PROTOCOL):
        raise ValueError("evaluation must bind the separate frozen overshoot trainer")
    protocol, _ = _ref(root, source["protocol"], "experiments/")
    result, result_path = _ref(root, source["result"], "runs/tdmpc2-overshoot-")
    run = protocol.get("run_dir")
    if (protocol.get("format") != TRAIN_FORMAT or protocol.get("purpose") != "consumed-TRAIN-development"
            or run != "runs/tdmpc2-overshoot-20260929-v1"
            or result_path != raw._path(root, f"{run}/result.json")
            or protocol.get("cells") != raw.CELLS or protocol.get("environment") != raw.ENVIRONMENT
            or protocol.get("episode_schedule") != [0, 1, 2, 3]
            or protocol.get("selection") != {"arm": "independent_3d", "action_dim": 3}
            or protocol.get("checkpoint_targets") != [20000, 40000, 70000, 100000]
            or protocol.get("reward_overshoot") != {
                "kind": "masked_imagined_reward_only", "base_horizon": 3,
                "reward_steps": [4, 5], "same_episode_suffix": True, "reward_target": "raw",
                "rho": .5, "reward_coef": .1, "normalization_horizon": 3, "aux_enabled": True}
            or protocol.get("training", {}).get("device") != "cuda"
            or any(protocol.get("training", {}).get(k) != v for k, v in raw.TRAIN_SETTINGS.items())):
        raise ValueError("source differs from separate 3D/raw/H3 overshoot TRAIN protocol")
    baseline, _ = _ref(root, protocol["baseline_training_protocol"], "experiments/")
    if (protocol["baseline_training_protocol"] != noise.SOURCE["protocol"]
            or protocol["baseline_training_result"] != noise.BASELINE["training_result"]
            or protocol.get("seed_schedule") != baseline.get("seed_schedule")
            or protocol.get("training") != baseline.get("training")
            or protocol.get("resources") != baseline.get("resources")
            or protocol.get("source_sha256", {}).keys() != (set(baseline.get("source_sha256", {})) | {
                "scripts/train_tdmpc2_reward_overshoot.py", "haic/algorithms/tdmpc2/reward_overshoot.py"})
            or any(protocol["source_sha256"].get(k) != v for k, v in baseline["source_sha256"].items())):
        raise ValueError("overshoot source changed frozen RAW training settings or source bytes")
    for ref in ("baseline_training_result", "replay_audit", "throughput_benchmark"):
        _ref(root, protocol[ref], "experiments/" if ref != "throughput_benchmark" else "runs/")
    for name, digest in protocol["source_sha256"].items():
        if raw._digest(raw._path(root, name)) != raw._sha(digest):
            raise ValueError(f"overshoot source hash mismatch: {name}")
    checkpoints = result.get("checkpoints")
    cfg = protocol["training"]
    if (result.get("status") != "completed_boundary_at_least_100k"
            or result.get("format") != TRAIN_FORMAT
            or result.get("protocol_sha256") != TRAIN_PROTOCOL["sha256"]
            or result.get("source_sha256") != protocol["source_sha256"]
            or result.get("reward_overshoot") != protocol["reward_overshoot"]
            or any(result.get(key) != protocol[key] for key in (
                "baseline_training_result", "replay_audit", "throughput_benchmark", "seed_schedule"))
            or result.get("runtime") != protocol.get("runtime")
            or result.get("reused_train_only") is not True or result.get("evaluation") is not None
            or result.get("resume_supported") is not False or result.get("action_dim") != 3
            or result.get("pretrain_updates") != cfg["pretrain_updates"]
            or type(result.get("decisions")) is not int
            or not 100000 <= result["decisions"] <= cfg["decision_cap"]
            or result.get("updates") != result["decisions"]
            or type(result.get("episodes")) is not int or result["episodes"] < 1
            or not isinstance(checkpoints, list) or len(checkpoints) != 4
            or [row.get("target") for row in checkpoints if isinstance(row, dict)] != [20000, 40000, 70000, 100000]):
        raise ValueError("overshoot 100k completed result is missing or partial")
    pin = source["checkpoint"]
    if (not isinstance(pin, dict) or set(pin) != {
            "target", "path", "sha256", "training_cursor_sha256", "step_cursor_sha256"}
            or pin.get("target") != 100000
            or pin.get("path") != f"{run}/{checkpoints[-1]['path']}"
            or pin.get("sha256") != checkpoints[-1].get("sha256")):
        raise ValueError("select only the first completed whole-episode >=100k checkpoint")
    return protocol, result, _ledger(root, protocol, result, pin)


def _ledger(root: Path, protocol: dict, result: dict, selected: dict) -> dict:
    """Verify every new TRAIN step, semantic boundary and checkpoint SHA cursor."""
    run = protocol["run_dir"]
    training = raw._path(root, f"{run}/training.jsonl")
    steps = raw._path(root, f"{run}/steps.jsonl")
    if (raw._digest(training) != raw._sha(result.get("training_ledger_sha256"))
            or raw._digest(steps) != raw._sha(result.get("step_ledger_sha256"))):
        raise ValueError("overshoot complete training/step ledger SHA mismatch")
    td, sd = hashlib.sha256(), hashlib.sha256()
    decisions = updates = checkpoint_index = 0
    episodes = []
    phase = "start"
    pin = None
    with training.open("rb") as journal, steps.open("rb") as step_stream:
        for line_number, line in enumerate(journal, 1):
            if not line.endswith(b"\n"):
                raise ValueError("torn overshoot training ledger")
            before = td.hexdigest()
            td.update(line)
            row = raw._json(line)
            event = row.get("event")
            if phase == "start":
                if (event != "start" or row.get("protocol_sha256") != TRAIN_PROTOCOL["sha256"]
                        or row.get("source_sha256") != protocol["source_sha256"]
                        or row.get("reward_overshoot") != protocol["reward_overshoot"]
                        or row.get("runtime") != protocol["runtime"]
                        or row.get("seed_schedule") != protocol["seed_schedule"]
                        or row.get("resume_supported") is not False):
                    raise ValueError("overshoot TRAIN start does not bind variant source")
                phase = "intent"
            elif phase == "checkpoint":
                expected = result["checkpoints"][checkpoint_index]
                target = protocol["checkpoint_targets"][checkpoint_index]
                if (row != {"event": "checkpoint", **expected} or row.get("target") != target
                        or any(row.get(k) != v for k, v in (
                            ("decisions", decisions), ("updates", updates), ("episodes", len(episodes))))
                        or row.get("step_ledger_sha256") != sd.hexdigest()
                        or row.get("training_ledger_sha256_before_checkpoint") != before
                        or row.get("train_episodes_since_previous") != [
                            {k: ep[k] for k in ("episode", "track_id", "geometry_seed", "decisions",
                                                  "length", "return", "progress", "damage", "finished")}
                            for ep in episodes[0 if checkpoint_index == 0 else
                                               result["checkpoints"][checkpoint_index - 1]["episodes"]:]]
                        or row.get("path") != f"checkpoint-at-least-{target:06d}-step-{decisions:06d}.pt"
                        or raw._digest(raw._path(root, f"{run}/{row['path']}")) != raw._sha(row["sha256"])):
                    raise ValueError("overshoot checkpoint/whole-episode SHA cursor differs")
                if target == 100000:
                    expected_pin = {"target": target, "path": f"{run}/{row['path']}",
                                    "sha256": row["sha256"], "training_cursor_sha256": td.hexdigest(),
                                    "step_cursor_sha256": sd.hexdigest()}
                    if selected != expected_pin:
                        raise ValueError("selected final checkpoint SHA cursor differs")
                    pin = {**expected_pin, "line": line_number, "decisions": decisions,
                           "updates": updates, "episodes": len(episodes),
                           "training_ledger_sha256_before_checkpoint": before}
                checkpoint_index += 1
                phase = "intent"
            elif event in ("reset_intent", "reset") and checkpoint_index < 4:
                expected = "reset_intent" if phase == "intent" else "reset"
                cell = raw.CELLS[len(episodes) % 4]
                if phase not in ("intent", "reset") or row != {
                        "event": expected, "episode": len(episodes), "decisions": decisions, **cell}:
                    raise ValueError("overshoot reset intent/cell differs from TRAIN schedule")
                phase = "reset" if phase == "intent" else "episode"
            elif event == "episode" and phase == "episode":
                length = row.get("length")
                cell = raw.CELLS[len(episodes) % 4]
                if (type(length) is not int or not 1 <= length <= 2000
                        or row.get("episode") != len(episodes)
                        or any(row.get(k) != v for k, v in cell.items())
                        or row.get("decisions") != decisions + length
                        or row.get("updates") != (0 if decisions + length < 10000 else decisions + length)
                        or any(type(row.get(k)) is not bool for k in (
                            "finished", "terminated", "truncated", "terminal"))
                        or not (row["terminated"] or row["truncated"])
                        or row["terminal"] != (row["terminated"] or row["finished"])
                        or row["finished"] and not row["truncated"]
                        or row.get("finish_time_s") is not None and not (row["finished"] and
                            _finite(row["finish_time_s"]))
                        or row["finished"] and row.get("finish_time_s") is None):
                    raise ValueError("overshoot episode lacks complete TRAIN boundary")
                action_hash, native_hash = hashlib.sha256(), hashlib.sha256()
                total = 0.0
                step = None
                for offset in range(length):
                    step_line = step_stream.readline()
                    if not step_line.endswith(b"\n"):
                        raise ValueError("missing overshoot TRAIN step")
                    sd.update(step_line)
                    step = raw._json(step_line)
                    if (step.get("episode") != len(episodes) or step.get("decision") != decisions + offset + 1
                            or any(step.get(k) != v for k, v in cell.items())
                            or any(type(step.get(k)) is not bool for k in (
                                "finished", "terminated", "truncated", "terminal"))
                            or (offset < length - 1 and any(step[k] for k in (
                                "finished", "terminated", "truncated", "terminal")))
                            or step["terminal"] != (step["terminated"] or step["finished"])
                            or not _finite(step.get("reward"))
                            or not _producer_repr(step.get("raw_reward_repr"), step["reward"])):
                        raise ValueError("overshoot raw step/flags/producer reward differ")
                    try:
                        action = bytes.fromhex(step["action_f32_hex"])
                        native = bytes.fromhex(step["native_action_f32_hex"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError("invalid overshoot 3D action bytes") from exc
                    if len(action) != 12 or len(native) != 12:
                        raise ValueError("overshoot TRAIN action is not 3D float32")
                    action_hash.update(action)
                    native_hash.update(native)
                    total += step["reward"]
                if (step is None or any(step[k] != row[k] for k in (
                        "finished", "terminated", "truncated", "terminal"))
                        or not all(_finite(row.get(k)) for k in ("return", "progress", "damage"))
                        or not math.isclose(total, row["return"], abs_tol=1e-4, rel_tol=0)
                        or action_hash.hexdigest() != row.get("action_trace_sha256")
                        or native_hash.hexdigest() != row.get("native_action_trace_sha256")):
                    raise ValueError("overshoot episode differs from full step ledger")
                decisions += length
                updates = row["updates"]
                episodes.append(row)
                phase = "checkpoint" if checkpoint_index < 4 and decisions >= protocol[
                    "checkpoint_targets"][checkpoint_index] else "intent"
            else:
                raise ValueError(f"partial or unexpected overshoot TRAIN event: {event}")
        if (phase != "intent" or checkpoint_index != 4 or pin is None
                or step_stream.read(1) or decisions != result["decisions"]
                or updates != result["updates"] or len(episodes) != result["episodes"]
                or pin["decisions"] != decisions or pin["episodes"] != len(episodes)
                or td.hexdigest() != result["training_ledger_sha256"]
                or sd.hexdigest() != result["step_ledger_sha256"]):
            raise ValueError("overshoot 100k result lacks final complete ledger cursor")
    return pin


def _finite(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _producer_repr(text: object, value: object) -> bool:
    if not isinstance(text, str) or len(text) > 128 or not _finite(value):
        return False
    try:
        assert isinstance(value, (float, int))
        number = float(value)
        return math.isfinite(float(text)) and float(text) == number and (
            text == repr(number) or number.is_integer() and text == repr(int(number)))
    except ValueError:
        return False


def _baseline(root: Path) -> dict:
    """Bind original RAW 0/8 and original code/environment before any new model load."""
    if (raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_full_train.py")) != RAW_EVALUATOR_SHA
            or raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_h3_noise_train.py")) != NOISE_EVALUATOR_SHA):
        raise ValueError("original RAW/noise evaluator source SHA mismatch")
    _, pin = noise._source(root)
    noise._baseline(root, pin)
    spec, _ = _ref(root, RESOURCE_PROTOCOL, "experiments/")
    measured = spec.get("resources", {}).get("measured_peak_rss_bytes")
    if (spec.get("raw_evaluator_sha256") != RAW_EVALUATOR_SHA
            or spec.get("cells") != raw.CELLS or spec.get("environment") != raw.ENVIRONMENT
            or spec.get("planner", {}).get("eval_mode") is not False
            or measured != 6462996480):
        raise ValueError("measured original CPU21 peak is not the pinned prior evaluation")
    return {"baseline_checkpoint_sha256": pin["sha256"], "mppi_episodes": 8,
            "mppi_finishes": 0, "mppi_censored": 0, "measured_peak_rss_bytes": measured}


def _score(root: Path, refs: object, source: dict, pin: dict) -> dict:
    """Reject a declared PASS unless archived raw outcomes reproduce all four gates."""
    if not isinstance(refs, dict) or set(refs) != {"protocol", "receipt"}:
        raise ValueError("missing separate archived-branch score protocol and receipt")
    score_protocol, _ = _ref(root, refs["protocol"], "experiments/tdmpc2-overshoot-")
    receipt, receipt_path = _ref(root, refs["receipt"], "runs/tdmpc2-overshoot-old-branch-score-")
    if (set(score_protocol) != {"format", "purpose", "training_source", "original_branch",
                                "source_sha256", "runtime", "output", "gate"}
            or score_protocol.get("format") != "haic-tdmpc2-overshoot-old-branch-score-v1"
            or score_protocol.get("purpose") != "consumed-TRAIN-archived-branch-reward-only"
            or receipt_path != raw._path(root, score_protocol.get("output", "missing"))
            or score_protocol.get("output") != "runs/tdmpc2-overshoot-old-branch-score-20260929-v1.json"
            or score_protocol.get("source_sha256") != receipt.get("source_sha256")
            or score_protocol.get("runtime") != receipt.get("runtime")):
        raise ValueError("old-branch score protocol/source/runtime differs from PASS receipt")
    expected_training = {"protocol": TRAIN_PROTOCOL, "result": source["result"],
                         "checkpoint": {"path": pin["path"], "sha256": pin["sha256"]},
                         "training_ledger": {"path": "runs/tdmpc2-overshoot-20260929-v1/training.jsonl",
                                             "sha256": source["training_ledger_sha256"]},
                         "step_ledger": {"path": "runs/tdmpc2-overshoot-20260929-v1/steps.jsonl",
                                         "sha256": source["step_ledger_sha256"]}}
    original = {"primary": BRANCH_PRIMARY, "protocol": BRANCH_PROTOCOL,
                "operator": {"path": "scripts/diagnose_tdmpc2_h5_branches.py",
                             "sha256": "85742b0623c25e6ddb8c2291b274b012c21a5002082dc0c60e424dbcea2f3e88"},
                "model": {"path": noise.SOURCE["checkpoints"][0]["path"],
                          "sha256": noise.SOURCE["checkpoints"][0]["sha256"]},
                "summary": {"path": "experiments/tdmpc2-h5-branches-v1-result.json",
                            "sha256": "0efcceb8d09ea4dac72f350d400ac2f2ad4d193e300f483eb712f65cd05c2913"}}
    if (receipt.get("format") != "haic-tdmpc2-overshoot-old-branch-score-result-v1"
            or receipt.get("status") != "PASS" or receipt.get("environment_resets") != 0
            or receipt.get("optimizer_updates") != 0
            or receipt.get("scope") != "reused_TRAIN_archived_branches_only"
            or receipt.get("protocol") != refs["protocol"]
            or receipt.get("training_source") != expected_training
            or receipt.get("original_branch") != original
            or score_protocol.get("training_source") != expected_training
            or score_protocol.get("original_branch") != original
            or score_protocol.get("gate") != {
                "min_h5_concordant": 66, "h5_informative_pairs": 91,
                "min_nonregressing_roads": 3, "min_h3_concordant": 31,
                "h3_informative_pairs": 40, "h5_error_exclusive": .39667698614253405}
            or not isinstance(receipt.get("source_sha256"), dict)
            or "scripts/score_tdmpc2_overshoot_archived_branches.py" not in receipt["source_sha256"]):
        raise ValueError("archived-branch PASS is absent, incomplete, or bound to a different model")
    for ref in (*expected_training.values(), *original.values()):
        if raw._digest(raw._path(root, ref["path"])) != raw._sha(ref["sha256"]):
            raise ValueError(f"score lineage SHA mismatch: {ref['path']}")
    body = {key: value for key, value in receipt.items() if key != "body_sha256"}
    body_sha = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    if receipt.get("body_sha256") != body_sha:
        raise ValueError("archived-branch receipt body SHA differs")
    archived, _ = _ref(root, BRANCH_PRIMARY, "runs/")
    old_protocol, _ = _ref(root, BRANCH_PROTOCOL, "experiments/")
    new_protocol, _ = _ref(root, TRAIN_PROTOCOL, "experiments/")
    source_map = receipt["source_sha256"]
    original_sources = old_protocol.get("source_sha256")
    training_sources = new_protocol.get("source_sha256")
    if (not isinstance(original_sources, dict) or not isinstance(training_sources, dict)
            or set(source_map) != set(original_sources) | set(training_sources) | {
                "scripts/score_tdmpc2_overshoot_archived_branches.py",
                "tests/test_score_tdmpc2_overshoot_archived_branches.py"}
            or any(source_map.get(name) != expected for mapping in (
                original_sources, training_sources) for name, expected in mapping.items())
            or score_protocol.get("runtime") != old_protocol.get("runtime")):
        raise ValueError("score did not pin complete trainer/archived branch source and runtime")
    for name, expected in source_map.items():
        if raw._digest(raw._path(root, name)) != raw._sha(expected):
            raise ValueError(f"archived scorer source SHA mismatch: {name}")
    if (archived.get("status") != "complete" or archived.get("protocol_sha256") != BRANCH_PROTOCOL["sha256"]
            or old_protocol.get("source", {}).get("checkpoint") != original["model"]
            or archived.get("source", {}).get("checkpoint") != original["model"]
            or archived.get("environment_resets_attempted") != 72):
        raise ValueError("original 12-anchor branch evidence is not the pinned receipt")
    candidates = ("logged", "coast", "gas", "brake", "left_gas")
    rows = receipt.get("anchors")
    old_rows = archived.get("anchors")
    if not isinstance(rows, list) or len(rows) != 12 or not isinstance(old_rows, list) or len(old_rows) != 12:
        raise ValueError("archived score requires all twelve original anchors")

    def pairs(cells: list[dict], actual: str, predicted: str) -> dict:
        counts = {"concordant": 0, "discordant": 0, "real_tie": 0, "predicted_tie": 0}
        for index, left in enumerate(cells):
            for right in cells[index + 1:]:
                a = left[actual] - right[actual]
                b = left[predicted] - right[predicted]
                key = ("real_tie" if abs(a) <= 1e-6 else "predicted_tie" if abs(b) <= 1e-6
                       else "concordant" if (a > 0) == (b > 0) else "discordant")
                counts[key] += 1
        return counts

    h5 = {key: 0 for key in ("concordant", "discordant", "real_tie", "predicted_tie")}
    h3 = h5.copy()
    by_road = {}
    road_pairs = {}
    absolute = []
    for index, (row, old) in enumerate(zip(rows, old_rows)):
        road = raw.ROADS[index // 3]
        if (not isinstance(row, dict) or row.get("episode_id") != old.get("episode_id")
                or row.get("start_step") != old.get("start_step")
                or row.get("geometry_seed") != road or old.get("geometry_seed") != road
                or row.get("anchor_model_observation_sha256") != old.get("anchor_model_observation_sha256")):
            raise ValueError("old branch anchor, road or observation hash differs")
        new_candidates = row.get("candidates")
        old_candidates = old.get("candidates")
        if (not isinstance(new_candidates, list) or not isinstance(old_candidates, list)
                or len(new_candidates) != len(old_candidates) or len(new_candidates) != 5):
            raise ValueError("five archived candidate action sequences required")
        for name, new, prior in zip(candidates, new_candidates, old_candidates):
            if (not isinstance(new, dict) or new.get("candidate") != name
                    or prior.get("candidate") != name
                    or new.get("model_action_bytes_hex") != prior.get("model_action_bytes_hex")
                    or new.get("model_action_bytes_hex") != old.get("candidate_action_bytes_hex", {}).get(name)
                    or new.get("real_h5_raw_return") != prior.get("real_discounted_raw_return")
                    or new.get("real_h3_prefix_raw_return") != prior.get("real_h3_prefix_raw_return")
                    or new.get("old_h5_reward_return") != prior.get("predicted_reward_return")
                    or new.get("ending") != prior.get("ending") or new.get("steps") != 5
                    or prior.get("steps") != 5
                    or any(not _finite(new.get(k)) for k in (
                        "new_h3_reward_return", "new_h5_reward_return"))):
                raise ValueError("archived branch candidate/real outcome bytes differ")
            absolute.append(abs(new["new_h5_reward_return"] - new["real_h5_raw_return"]))
        for fields, actual, predicted in ((h5, "real_h5_raw_return", "new_h5_reward_return"),
                                          (h3, "real_h3_prefix_raw_return", "new_h3_reward_return")):
            for key, count in pairs(new_candidates, actual, predicted).items():
                fields[key] += count
        road_data = by_road.setdefault(str(road), {"anchors": 0, "new_concordant": 0,
                                                   "old_concordant": 0, "informative_real_pairs": 0})
        road_data["anchors"] += 1
        new_pairs = pairs(new_candidates, "real_h5_raw_return", "new_h5_reward_return")
        old_pairs = pairs(new_candidates, "real_h5_raw_return", "old_h5_reward_return")
        if new_pairs["real_tie"] != old_pairs["real_tie"]:
            raise ValueError("old and new branch score disagree on actual H5 ties")
        road_data["new_concordant"] += new_pairs["concordant"]
        road_data["old_concordant"] += old_pairs["concordant"]
        road_data["informative_real_pairs"] += 10 - new_pairs["real_tie"]
        counts = road_pairs.setdefault(str(road), {key: 0 for key in h5})
        for key, count in new_pairs.items():
            counts[key] += count
    if (h5["real_tie"] != 29 or sum(h5[k] for k in h5 if k != "real_tie") != 91
            or h3["real_tie"] != 80 or sum(h3[k] for k in h3 if k != "real_tie") != 40
            or len(absolute) != 60 or len(by_road) != 4
            or [by_road[str(seed)]["old_concordant"] for seed in raw.ROADS] != [13, 16, 12, 15]
            or [by_road[str(seed)]["informative_real_pairs"] for seed in raw.ROADS] != [23, 22, 23, 23]):
        raise ValueError("score did not use original 91 H5 and 40 H3 informative pairs")
    error = math.fsum(absolute) / 60 / math.fsum(.995 ** t for t in range(5))
    nonregressing = sum(value["new_concordant"] >= value["old_concordant"] for value in by_road.values())
    summary = receipt.get("summary")
    if (not isinstance(summary, dict) or not isinstance(summary.get("h5"), dict)
            or summary["h5"].get("pairs") != h5
            or summary["h5"].get("informative_real_pairs") != 91
            or summary["h5"].get("by_road") != {
                seed: {"anchors": d["anchors"], "informative_real_pairs": d["informative_real_pairs"],
                       "pairs": road_pairs[seed]} for seed, d in by_road.items()}
            or not isinstance(summary.get("h3"), dict) or summary["h3"].get("pairs") != h3
            or summary["h3"].get("informative_real_pairs") != 40
            or not _finite(summary.get("h5_absolute_error_per_discounted_step"))
            or not math.isclose(summary["h5_absolute_error_per_discounted_step"], error,
                                rel_tol=0, abs_tol=1e-9)):
        raise ValueError("archived score summary disagrees with per-anchor predictions")
    gate = receipt.get("gate")
    if (not isinstance(gate, dict) or set(gate) != {
            "h5_reward_concordance", "road_nonregression", "h3_prefix_concordance",
            "h5_normalized_error", "passed"}
            or gate["h5_reward_concordance"] != {
                "concordant": h5["concordant"], "informative_real_pairs": 91,
                "minimum_concordant": 66, "passed": h5["concordant"] >= 66}
            or gate["road_nonregression"] != {
                "nonregressing_roads": nonregressing, "minimum_roads": 3,
                "by_road": {seed: {"new_concordant": d["new_concordant"],
                                   "old_concordant": d["old_concordant"],
                                   "informative_real_pairs": d["informative_real_pairs"],
                                   "passed": d["new_concordant"] >= d["old_concordant"]}
                            for seed, d in by_road.items()}, "passed": nonregressing >= 3}
            or gate["h3_prefix_concordance"] != {
                "concordant": h3["concordant"], "informative_real_pairs": 40,
                "minimum_concordant": 31, "passed": h3["concordant"] >= 31}
            or not _finite(gate["h5_normalized_error"].get("value"))
            or not math.isclose(gate["h5_normalized_error"]["value"], error, rel_tol=0, abs_tol=1e-9)
            or gate["h5_normalized_error"].get("threshold_exclusive") != .39667698614253405
            or gate["h5_normalized_error"].get("passed") != (error < .39667698614253405)
            or gate["passed"] is not True or not (h5["concordant"] >= 66 and nonregressing >= 3
                                                   and h3["concordant"] >= 31 and error < .39667698614253405)):
        raise ValueError("four predeclared archived-branch gates did not independently PASS")
    return {"receipt_sha256": refs["receipt"]["sha256"], "body_sha256": body_sha,
            "h5_concordant": h5["concordant"], "h3_concordant": h3["concordant"],
            "nonregressing_roads": nonregressing, "normalized_h5_error": error}


def _resources(output: Path, forecast: dict, *, allocated_bytes: int = 0) -> dict:
    memory = noise._headroom()
    disk = shutil.disk_usage(output).free
    needed_memory = max(0, forecast["additional_memory_bytes"] - allocated_bytes) + forecast["memory_reserve_bytes"]
    needed_disk = forecast["remaining_disk_bytes"] + forecast["disk_reserve_bytes"]
    if memory < needed_memory or disk < needed_disk:
        raise ValueError("measured host/cgroup or disk headroom below forecast plus 24GiB reserve")
    return {"host_and_cgroup_raw_headroom_bytes": memory, "output_disk_free_bytes": disk,
            "required_memory_bytes": needed_memory, "required_disk_bytes": needed_disk}


def _check(protocol_path: Path, sha: str, *, root: Path, reserved: bool = False,
           allocated_bytes: int = 0) -> tuple[dict, dict]:
    path = raw._evaluation_path(root, protocol_path)
    if path.parent != root / "experiments" or raw._digest(path) != raw._sha(sha):
        raise ValueError("separate overshoot evaluation protocol SHA mismatch")
    p = raw._json(path.read_bytes())
    if (set(p) != {"format", "purpose", "evaluation_source_sha256", "raw_evaluator_sha256",
                   "noise_evaluator_sha256", "source", "branch_gate", "baseline", "resource_reference",
                   "cells", "environment", "modes", "repeats", "seed", "max_steps", "planner",
                   "runtime", "resources", "output_dir"}
            or p["format"] != FORMAT or p["purpose"] != "consumed-TRAIN-development"
            or p["raw_evaluator_sha256"] != RAW_EVALUATOR_SHA
            or p["noise_evaluator_sha256"] != NOISE_EVALUATOR_SHA
            or p["resource_reference"] != RESOURCE_PROTOCOL or p["baseline"] != noise.BASELINE
            or p["cells"] != raw.CELLS or p["environment"] != raw.ENVIRONMENT
            or p["modes"] != ["mppi"] or type(p["repeats"]) is not int or p["repeats"] != 2
            or type(p["seed"]) is not int or p["seed"] != 20260928
            or type(p["max_steps"]) is not int or p["max_steps"] != 2000
            or p["planner"] != PLANNER or any(type(p["planner"].get(k)) is not type(v)
                                                  for k, v in PLANNER.items())):
        raise ValueError("only eight raw H3 default-MPPI consumed-TRAIN episodes are allowed")
    forecast = p["resources"]
    if (not isinstance(forecast, dict) or set(forecast) != RESOURCE_KEYS
            or any(type(v) is not int for v in forecast.values())
            or forecast["measured_peak_rss_bytes"] != 6462996480
            or forecast["additional_memory_bytes"] < 8589934592
            or forecast["memory_reserve_bytes"] < 24 * 1024**3
            or forecast["remaining_disk_bytes"] < 67108864
            or forecast["disk_reserve_bytes"] < 24 * 1024**3
            or not 0 < forecast["max_wall_seconds"] <= 21600):
        raise ValueError("measured original CPU peak/incremental budgets or >=24GiB reserves missing")
    if (raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_reward_overshoot_train.py"))
            != raw._sha(p["evaluation_source_sha256"])):
        raise ValueError("overshoot evaluation executable SHA mismatch")
    output = raw._path(root, p["output_dir"], existing=False)
    if (output.parent != root / "runs" or not output.name.startswith("tdmpc2-overshoot-full-train-")
            or not output.parent.is_dir() or (not output.is_dir() if reserved else output.exists())
            or reserved and any((output / name).exists() for name in ("result.json", "failure.json"))):
        raise ValueError("evaluation output must be a new exclusive runs/tdmpc2-overshoot-full-train-* directory")
    if torch.version.cuda is not None or str(torch.__version__) != "2.1.0+cpu":
        raise ValueError("evaluation requires isolated CPU-only Torch 2.1.0+cpu")
    if p["runtime"] != _runtime():
        raise ValueError("CPU21/native Box2D/TimeLimit evaluation runtime changed")
    resources = _resources(output if reserved else output.parent, forecast, allocated_bytes=allocated_bytes)
    baseline = _baseline(root)
    protocol, result, pin = _source(root, p["source"])
    score = _score(root, p["branch_gate"], {
        "result": p["source"]["result"],
        "training_ledger_sha256": result["training_ledger_sha256"],
        "step_ledger_sha256": result["step_ledger_sha256"]}, pin)
    return p, {"status": "preflight_only", "environment_resets": 0, "torch_load_calls": 0,
               "reused_train_only": True, "generalization_claim": False, "official_score": False,
               "protocol_sha256": sha, "source_protocol_sha256": TRAIN_PROTOCOL["sha256"],
               "source_result_sha256": p["source"]["result"]["sha256"],
               "source_sha256": protocol["source_sha256"], "checkpoint": pin,
               "branch_gate": score, "baseline": baseline, "planner": PLANNER,
               "planned_episodes": 8, "max_steps": 2000, "output_dir": p["output_dir"],
               "resources": resources}


def preflight(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    """Strict file-only source/score/runtime/resource gate with zero environment resets."""
    return _check(protocol_path, protocol_sha256, root=root.resolve(strict=True))[1]


def _model(pin: dict, p: dict, root: Path, *, model_factory=None):
    """Only the production caller can pass the fully checked trusted .pt file."""
    checkpoint = raw._path(root, pin["path"])
    with checkpoint.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != pin["sha256"]:
            raise ValueError("overshoot checkpoint changed before torch.load")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=False)
    if (not isinstance(state, dict) or set(state) != {
            "format", "protocol_sha256", "source_sha256", "baseline_training_protocol",
            "baseline_training_result", "replay_audit", "throughput_benchmark", "reward_overshoot",
            "target", "decisions", "updates", "episodes", "action_dim", "step_ledger_sha256",
            "training_ledger_sha256_before_checkpoint", "learner", "optim", "pi_optim", "replay",
            "probe", "rng", "resume_supported"}
            or state["format"] != TRAIN_FORMAT or state["protocol_sha256"] != TRAIN_PROTOCOL["sha256"]
            or state["source_sha256"] != p["source_sha256"]
            or any(state[k] != p[k] for k in (
                "baseline_training_protocol", "baseline_training_result", "replay_audit",
                "throughput_benchmark", "reward_overshoot"))
            or any(state.get(k) != pin[k] for k in ("target", "decisions", "updates", "episodes"))
            or state["step_ledger_sha256"] != pin["step_cursor_sha256"]
            or state["training_ledger_sha256_before_checkpoint"] != pin[
                "training_ledger_sha256_before_checkpoint"]
            or state["action_dim"] != 3 or state["resume_supported"] is not False
            or any(not isinstance(state[k], dict) for k in (
                "learner", "optim", "pi_optim", "replay", "probe", "rng"))
            or "q_scale" not in state["learner"]
            or not all(isinstance(k, str) and (k == "q_scale" or k.startswith("model."))
                       for k in state["learner"])
            or state["replay"].get("format") != "haic-tdmpc2-episode-replay-v1"
            or state["replay"].get("active") is not None
            or state["replay"].get("next_episode_id") != pin["episodes"]
            or state["replay"].get("size") != pin["decisions"]):
        raise ValueError("checkpoint is not the bound complete reward-overshoot learner")
    if model_factory is None:
        from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
    else:
        model = model_factory()
    model.load_state_dict({k[6:]: v for k, v in state["learner"].items() if k.startswith("model.")}, strict=True)
    return model.to("cpu").eval()


def _episode(env, model, planner, cell: dict, *, repeat: int, seed: int, budget,
             clock=time.perf_counter) -> dict:
    from haic.algorithms.tdmpc2.haic_env import environment_action, episode_boundary, model_observation

    raw._seed(seed)
    planner.reset()
    obs, _ = env.reset(seed=cell["geometry_seed"], options={"track_id": cell["track_id"]})
    if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
            or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
        raise ValueError("environment reset did not honor consumed TRAIN cell")
    pixels = model_observation(obs)
    total = latency_sum = latency_max = first_latency = 0.0
    actions, native_actions = hashlib.sha256(), hashlib.sha256()
    for decision in range(1, 2001):
        budget()
        started = clock()
        with torch.inference_mode():
            tensor = torch.as_tensor(pixels[None], device="cpu")
            action = planner.plan(tensor, t0=(decision == 1), eval_mode=True)
            action = action.detach().cpu().numpy().astype(np.float32)
        latency = clock() - started
        if not math.isfinite(latency) or not 0 <= latency <= 5.0:
            raise TimeoutError("CPU MPPI action latency exceeded five seconds or is invalid")
        native = environment_action(action)
        latency_sum += latency
        latency_max = max(latency_max, latency)
        actions.update(action.tobytes())
        native_actions.update(native.tobytes())
        obs, reward, terminated, truncated, info = env.step(native)
        if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
            raise ValueError("environment road changed during episode")
        if (type(terminated) is not bool or type(truncated) is not bool or not isinstance(info, dict)
                or type(info.get("finished")) is not bool or not _finite(reward)
                or any(not _finite(info.get(k)) for k in ("progress", "damage"))):
            raise ValueError("invalid raw environment reward, flags or metrics")
        total += reward
        if not math.isfinite(total):
            raise ValueError("nonfinite raw return")
        done, terminal = episode_boundary(terminated, truncated, info)
        if decision == 1:
            first_latency = latency
        if done:
            finished = info["finished"]
            if (finished and not truncated or truncated and not finished and not terminated and decision != 2000
                    or finished != (info.get("finish_time_s") is not None)
                    or finished and not _finite(info["finish_time_s"])):
                raise ValueError("finish/truncation conflicts with full-episode semantics")
            return {"event": "episode", "target": 100000, "mode": "mppi", "repeat": repeat,
                    "episode_seed": seed, **cell, "decisions": decision, "max_steps": 2000,
                    "raw_return": total, "progress": float(info["progress"]), "damage": float(info["damage"]),
                    "finished": finished, "terminated": terminated, "truncated": truncated,
                    "terminal": terminal, "censored": bool(truncated and not finished and not terminated),
                    "finish_time_s": info.get("finish_time_s"),
                    "action_trace_sha256": actions.hexdigest(), "native_action_trace_sha256": native_actions.hexdigest(),
                    "action_latency_first_s": first_latency, "action_latency_mean_s": latency_sum / decision,
                    "action_latency_max_s": latency_max, "action_latency_total_s": latency_sum}
        pixels = model_observation(obs)
    raise ValueError("environment did not end its full 2000-decision episode")


def _summary(rows: list[dict], pin: dict) -> dict:
    schedule = [(repeat, cell) for repeat in range(2) for cell in raw.CELLS]
    if (len(rows) != 8 or any(row.get("event") != "episode" or row.get("target") != 100000
            or row.get("mode") != "mppi" or row.get("repeat") != repeat
            or row.get("episode_seed") != 20260928 + index
            or any(row.get(k) != v for k, v in cell.items()) or row.get("max_steps") != 2000
            or type(row.get("decisions")) is not int or not 1 <= row["decisions"] <= 2000
            or "training_return" in row or "training_reward" in row
            for index, (row, (repeat, cell)) in enumerate(zip(rows, schedule)))):
        raise ValueError("not the complete eight-episode RAW MPPI schedule")
    roads = []
    for cell in raw.CELLS:
        cohort = [r for r in rows if r["geometry_seed"] == cell["geometry_seed"]]
        roads.append({**cell, "episodes": 2, "finishes": sum(r["finished"] for r in cohort),
                      "censored": sum(r["censored"] for r in cohort),
                      "uncensored": sum(not r["censored"] for r in cohort),
                      "mean_progress": sum(r["progress"] for r in cohort) / 2,
                      "mean_raw_return": sum(r["raw_return"] for r in cohort) / 2,
                      "mean_damage": sum(r["damage"] for r in cohort) / 2})
    steps = sum(row["decisions"] for row in rows)
    return {"checkpoint_sha256": pin["sha256"], "mode": "mppi", "episodes": 8,
            "finishes": sum(row["finished"] for row in rows),
            "censored": sum(row["censored"] for row in rows),
            "uncensored": sum(not row["censored"] for row in rows), "decisions": steps,
            "mean_progress": sum(row["progress"] for row in rows) / 8,
            "mean_raw_return": sum(row["raw_return"] for row in rows) / 8,
            "mean_damage": sum(row["damage"] for row in rows) / 8,
            "mean_action_latency_s": sum(row["action_latency_total_s"] for row in rows) / steps,
            "max_action_latency_s": max(row["action_latency_max_s"] for row in rows), "roads": roads}


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _run_core(root: Path, p: dict, protocol_path: Path, sha: str, checked: dict, model, planner,
              *, env_factory, clock=time.perf_counter, initial_peak_rss_bytes: int | None = None) -> dict:
    """Private synthetic-only seam; public execute cannot inject model or environment."""
    from haic.algorithms.tdmpc2.planner import PlannerConfig

    config = PlannerConfig(action_dim=3, discount=.995, episodic=True, horizon=3,
                           num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64)
    if (getattr(planner, "config", None) != config or p.get("planner") != PLANNER
            or checked.get("branch_gate", {}).get("receipt_sha256") != p["branch_gate"]["receipt"]["sha256"]):
        raise ValueError("private core requires source-gated default H3 MPPI")
    output = raw._path(root, p["output_dir"], existing=False)
    if output.parent != root / "runs" or not output.name.startswith("tdmpc2-overshoot-full-train-"):
        raise ValueError("invalid exclusive output")
    output.mkdir(mode=0o700, exist_ok=False)
    ledger = output / "episodes.jsonl"
    rows = []
    intents = 0
    env = None
    started = clock()
    peak_at_start = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                     if initial_peak_rss_bytes is None else initial_peak_rss_bytes)

    def allocated() -> int:
        return max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - peak_at_start)

    def budget() -> None:
        if clock() - started > p["resources"]["max_wall_seconds"]:
            raise TimeoutError("overshoot full-episode wall cap exceeded")
        _resources(output, p["resources"], allocated_bytes=allocated())

    try:
        raw._journal(ledger, {"event": "start", "protocol_sha256": sha,
                              "source_result_sha256": checked["source_result_sha256"],
                              "source_model_sha256": checked["checkpoint"]["sha256"],
                              "branch_score_sha256": checked["branch_gate"]["receipt_sha256"],
                              "resume_supported": False})
        for repeat in range(2):
            for index, cell in enumerate(raw.CELLS):
                budget()
                latest_p, latest = _check(protocol_path, sha, root=root, reserved=True,
                                          allocated_bytes=allocated())
                if latest_p != p or any(latest[k] != checked[k] for k in (
                        "checkpoint", "branch_gate", "baseline", "source_result_sha256")):
                    raise ValueError("source, archived score or baseline changed before reset intent")
                seed = 20260928 + repeat * 4 + index
                intents += 1
                raw._journal(ledger, {"event": "reset_intent", "target": 100000, "mode": "mppi",
                                      "repeat": repeat, "episode_seed": seed, **cell})
                if env is None:
                    env = env_factory(2000)
                budget()
                _, latest = _check(protocol_path, sha, root=root, reserved=True,
                                   allocated_bytes=allocated())
                if latest["checkpoint"] != checked["checkpoint"] or latest["branch_gate"] != checked["branch_gate"]:
                    raise ValueError("score/source changed immediately before environment reset")
                row = _episode(env, model, planner, cell, repeat=repeat, seed=seed,
                               budget=budget, clock=clock)
                raw._journal(ledger, row)
                rows.append(row)
        budget()
        latest_p, latest = _check(protocol_path, sha, root=root, reserved=True,
                                  allocated_bytes=allocated())
        if latest_p != p or latest["checkpoint"] != checked["checkpoint"] or latest[
                "branch_gate"] != checked["branch_gate"]:
            raise ValueError("source or score changed before complete result")
        summary = _summary(rows, checked["checkpoint"])
        distinct = sum(road["finishes"] > 0 for road in summary["roads"])
        report = {"status": "complete", "scope": "reused_consumed_TRAIN_development_only",
                  "reused_train_only": True, "generalization_claim": False, "official_score": False,
                  "evaluation_reward": "raw_environment_only", "training_reward_is_evaluation_metric": False,
                  "protocol_sha256": sha, "source_protocol_sha256": TRAIN_PROTOCOL["sha256"],
                  "source_result_sha256": checked["source_result_sha256"],
                  "source_model_sha256": checked["checkpoint"]["sha256"],
                  "branch_gate": checked["branch_gate"],
                  "baseline_full_eval_result_sha256": noise.BASELINE["full_eval_result"]["sha256"],
                  "baseline_h3_eval_mode_true": {"episodes": 8, "finishes": 0, "censored": 0},
                  "source_model_trained_horizon": 3, "planner": PLANNER, "cpu_only_torch": str(torch.__version__),
                  "full_episode_finish_comparison_valid": summary["censored"] == 0,
                  "pairing": "same reused TRAIN road/reset seed; independent actions and trajectories",
                  "denominators": {"distinct_training_roads": 4, "repeats_per_road": 2,
                                   "planned_episodes": 8, "completed_episodes": len(rows), "max_steps": 2000},
                  "episodes_sha256": raw._digest(ledger), "primary_mppi": summary,
                  "primary_local_gate": {"min_finishes": 4, "min_distinct_finished_roads": 2,
                                         "observed_finishes": summary["finishes"],
                                         "observed_distinct_finished_roads": distinct,
                                         "uncensored_episodes": summary["uncensored"],
                                         "met_on_reused_train": summary["finishes"] >= 4 and distinct >= 2
                                         and summary["censored"] == 0,
                                         "fresh_generalization_claim": False},
                  "resources": p["resources"],
                  "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "elapsed_seconds": clock() - started}
        if env is not None:
            env.close()
            env = None
        _write_json(output / "result.json", report)
        return report
    except BaseException as exc:
        try:
            raw._journal(ledger, {"event": "partial", "reason": type(exc).__name__,
                                  "complete_episodes": len(rows), "reset_intents": intents,
                                  "environment_resets": None if intents else 0,
                                  "resume_supported": False})
        except BaseException:
            try:
                _write_json(output / "failure.json", {
                    "event": "failure", "reason": type(exc).__name__, "protocol_sha256": sha,
                    "source_result_sha256": checked["source_result_sha256"],
                    "branch_score_sha256": checked["branch_gate"]["receipt_sha256"],
                    "episode_ledger_sha256": raw._digest(ledger) if ledger.is_file() else None,
                    "complete_episodes": len(rows), "reset_intents": intents,
                    "environment_resets": None if intents else 0, "resume_supported": False})
            except BaseException as preserve_error:
                raise RuntimeError("cannot durably preserve partial evaluation; exposure unknown") from preserve_error
        raise
    finally:
        if env is not None:
            env.close()


def execute(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    """Production-only; no model/env injection or resumption of any partial output."""
    root = root.resolve(strict=True)
    p, checked = _check(protocol_path, protocol_sha256, root=root)
    if _check(protocol_path, protocol_sha256, root=root)[0] != p:
        raise ValueError("frozen evaluation source changed before model load")
    from haic.algorithms.tdmpc2.haic_env import make_training_env
    from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner

    initial_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    source, _ = _ref(root, TRAIN_PROTOCOL, "experiments/")
    model = _model(checked["checkpoint"], source, root)
    planner = TDMPC2Planner(model, PlannerConfig(action_dim=3, discount=.995, episodic=True,
                                                 horizon=3, num_samples=512, num_pi_trajs=24,
                                                 iterations=6, num_elites=64))
    allocated = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial_peak)
    _, latest = _check(protocol_path, protocol_sha256, root=root, allocated_bytes=allocated)
    if latest["checkpoint"] != checked["checkpoint"] or latest["branch_gate"] != checked["branch_gate"]:
        raise ValueError("source/score drift after trusted model load")
    return _run_core(root, p, protocol_path, protocol_sha256, checked, model, planner,
                     env_factory=make_training_env, initial_peak_rss_bytes=initial_peak)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-runtime", action="store_true", help="native CPU runtime only; no protocol or env")
    parser.add_argument("--protocol", type=Path, help="independently frozen experiments/*.json")
    parser.add_argument("--protocol-sha256", help="external lowercase protocol SHA-256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="default, read-only and zero resets")
    mode.add_argument("--execute", action="store_true", help="conditional eight consumed TRAIN episodes")
    args = parser.parse_args()
    if args.print_runtime:
        if args.protocol or args.protocol_sha256 or args.preflight or args.execute:
            parser.error("--print-runtime takes no protocol or execution options")
        print(json.dumps(_runtime(), sort_keys=True, allow_nan=False))
        return
    if not args.protocol or not args.protocol_sha256:
        parser.error("--protocol and --protocol-sha256 are required")
    result = execute(args.protocol, args.protocol_sha256) if args.execute else preflight(
        args.protocol, args.protocol_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
