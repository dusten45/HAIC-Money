"""Separate Oracle throttle/brake effects from its steering correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from scripts import branch_oracle_policy_action as full_action_branch
from scripts import diagnose_oracle_policy_failures as diagnosis


ROOT = diagnosis.ROOT
PROTOCOL_PATH = ROOT / "experiments/oracle-policy-longitudinal-branch-v1.json"
SOURCE_RESULT = ROOT / "experiments/oracle-policy-diagnosis-v1-result.json"
FULL_ACTION_RESULT = ROOT / "experiments/oracle-policy-action-branch-v1-result.json"
OUTPUT = ROOT / "runs/oracle-policy-longitudinal-branch-v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any, *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _trace(policy_id: str, seed: int):
    result = json.loads(SOURCE_RESULT.read_text())
    row = next(value for value in result["comparisons"] if
               value["study"] == "rlpd-g0" and value["policy_id"] == policy_id
               and value["geometry_seed"] == seed)
    relative = f"policies/{policy_id}-track1-seed{seed}.npz"
    path = ROOT / "runs/oracle-policy-diagnosis-v1" / relative
    if _sha(path) != result["rollout_files_sha256"][relative]:
        raise diagnosis.DiagnosisError(f"source policy trace changed: {relative}")
    with np.load(path, allow_pickle=False) as archive:
        return row, {name: archive[name] for name in archive.files}


def _rollout(row: dict[str, Any], trace: dict[str, np.ndarray], event_step: int, mode: str):
    env, base, observation = diagnosis._make_env(row["track_id"], row["geometry_seed"], 2000)
    try:
        from haic.oracle_v1 import OracleController

        controller = OracleController(base, target_speed=12.0, avoid_obstacles=True)
        if diagnosis._obs_hash(observation) != trace["observation_sha256"][0]:
            raise diagnosis.DiagnosisError("longitudinal branch reset differs from policy input")
        for step in range(event_step):
            observation, _, terminated, truncated, _ = env.step(
                np.asarray(trace["policy_or_oracle_action"][step], dtype=np.float32),
            )
            if terminated or truncated:
                raise diagnosis.DiagnosisError("saved policy prefix terminated before branch")
        before = diagnosis._state(base, controller)
        if diagnosis._obs_hash(observation) != trace["observation_sha256"][event_step]:
            raise diagnosis.DiagnosisError("prefix observation differs at the frozen branch state")
        checks = {
            "position_m": float(np.linalg.norm(before["position"] - trace["pre_position"][event_step])),
            "speed_m_s": abs(before["speed"] - float(trace["pre_speed"][event_step])),
            "oracle_action_linf": float(np.max(np.abs(
                before["oracle_action"] - trace["oracle_action_at_state"][event_step]
            ))),
        }
        if any(value > 1e-5 for value in checks.values()):
            raise diagnosis.DiagnosisError(f"longitudinal branch state mismatch: {checks}")
        policy_action = np.asarray(trace["policy_or_oracle_action"][event_step], dtype=np.float32)
        first_action = policy_action.copy()
        if mode == "oracle-longitudinal":
            first_action[1:] = before["oracle_action"][1:]
        states = []
        actions = []
        for offset in range(4):
            index = event_step + offset
            action = first_action if offset == 0 else np.asarray(
                trace["policy_or_oracle_action"][index], dtype=np.float32,
            )
            observation, _, terminated, truncated, _ = env.step(action)
            state = diagnosis._state(base, controller)
            states.append({
                "position": state["position"],
                "speed": state["speed"],
                "heading": state["heading"],
                "center_error": state["center_error"],
                "heading_error": state["heading_error"],
                "curvature": state["curvature"],
                "progress": float(env._calculate_progress()),
                "damage": float(env.damage.damage),
            })
            actions.append(action.copy())
            if (terminated or truncated) and offset < 3:
                raise diagnosis.DiagnosisError("longitudinal branch ended before four decisions")
        arrays = {
            "actions": np.stack(actions).astype(np.float32),
            "position": np.stack([state["position"] for state in states]).astype(np.float64),
            **{key: np.asarray([state[key] for state in states], dtype=np.float64)
               for key in ("speed", "heading", "center_error", "heading_error", "curvature", "progress", "damage")},
        }
        return {"state_match": checks, "event_policy_action": policy_action.astype(float).tolist(),
                "event_oracle_action": before["oracle_action"].astype(float).tolist()}, arrays
    finally:
        env.close()


def _verify() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = diagnosis._read_json(PROTOCOL_PATH)
    if protocol["runner_sha256"] != _sha(Path(__file__).resolve()):
        raise diagnosis.DiagnosisError("longitudinal runner does not match its frozen hash")
    for relative, expected in protocol["source_sha256"].items():
        path = SOURCE_RESULT if relative == "diagnosis_result" else ROOT / relative
        if _sha(path) != expected:
            raise diagnosis.DiagnosisError(f"longitudinal source hash mismatch: {relative}")
    source = json.loads(SOURCE_RESULT.read_text())
    full = json.loads(FULL_ACTION_RESULT.read_text())
    if full["speed_gate_passes"] != 10 or full["lane_guard_passes"] != 7:
        raise diagnosis.DiagnosisError("the frozen full-action branch result no longer qualifies for this isolation test")
    return protocol, source


def run(*, preflight_only: bool = False) -> dict[str, Any]:
    protocol, source = _verify()
    samples = [tuple(value) for value in protocol["samples"]]
    if samples != full_action_branch.SAMPLES:
        raise diagnosis.DiagnosisError("sample list differs from the full-action intervention")
    if preflight_only:
        return {"status": "preflight_passed_no_environment_resets", "paired_interventions": len(samples)}
    if OUTPUT.exists():
        raise diagnosis.DiagnosisError("longitudinal output exists; refusing overwrite")
    OUTPUT.mkdir()
    _write(OUTPUT / "metadata.json", {
        "format": "haic-oracle-longitudinal-branch-run-v1",
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "runner_sha256": _sha(Path(__file__).resolve()),
        "source_diagnosis_sha256": _sha(SOURCE_RESULT),
        "full_action_branch_sha256": _sha(FULL_ACTION_RESULT),
        "intervention": "Oracle gas/brake only; preserve policy steer; then three recorded policy actions",
    }, exclusive=True)
    results = []
    for index, (policy_id, seed) in enumerate(samples, 1):
        row, trace = _trace(policy_id, seed)
        step = int(row["first_failure_precursor"]["step"])
        control_summary, control = _rollout(row, trace, step, "policy")
        treatment_summary, treatment = _rollout(row, trace, step, "oracle-longitudinal")
        if not np.allclose(control_summary["state_match"]["position_m"],
                           treatment_summary["state_match"]["position_m"], atol=1e-9, rtol=0):
            raise diagnosis.DiagnosisError("policy and longitudinal branches used different prefixes")
        speed_delta = float(treatment["speed"][-1] - control["speed"][-1])
        lane_delta = float(abs(treatment["center_error"][-1]) - abs(control["center_error"][-1]))
        damage_delta = float(treatment["damage"][-1] - control["damage"][-1])
        item = {
            "policy_id": policy_id,
            "track_id": row["track_id"],
            "geometry_seed": seed,
            "event_step": step,
            "source_trace_sha256": row["source_trace_sha256"],
            "speed_delta_m_s_treatment_minus_control": speed_delta,
            "abs_center_error_delta_m_treatment_minus_control": lane_delta,
            "damage_delta_treatment_minus_control": damage_delta,
            "speed_pass": speed_delta <= -0.5,
            "lane_pass": lane_delta <= 0.5,
            "damage_pass": damage_delta <= 0.0,
        }
        for mode, arrays in (("policy", control), ("oracle-longitudinal", treatment)):
            path = OUTPUT / f"{policy_id}-seed{seed}-step{step}-{mode}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            item[f"{mode}_trace_sha256"] = _sha(path)
        results.append(item)
        with (OUTPUT / "branches.jsonl").open("a") as stream:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"longitudinal-branch {index:02d}/{len(samples)} {policy_id} seed={seed}: "
              f"speed={speed_delta:+.3f} lane={lane_delta:+.3f} damage={damage_delta:+.3f}", flush=True)
    speed_pass = sum(row["speed_pass"] for row in results)
    lane_pass = sum(row["lane_pass"] for row in results)
    damage_pass = sum(row["damage_pass"] for row in results)
    supported = speed_pass >= 8 and lane_pass == len(results) and damage_pass == len(results)
    result = {
        "format": "haic-oracle-longitudinal-branch-result-v1",
        "status": "complete_longitudinal_action_isolation",
        "hypothesis": "The speed improvement from the Oracle branch can be retained by replacing only throttle/brake while preserving policy steering.",
        "samples": len(results),
        "speed_passes": speed_pass,
        "lane_guard_passes": lane_pass,
        "damage_guard_passes": damage_pass,
        "median_speed_delta_m_s": float(np.median([row["speed_delta_m_s_treatment_minus_control"] for row in results])),
        "hypothesis_supported": supported,
        "next_step": ("pilot masked RLPD oracle supervision on throttle/brake only, TRAIN cells only"
                      if supported else "reject longitudinal-only oracle imitation; prioritize steering/heading-state diagnosis instead of training"),
        "resource_use": "20 short branches, no full episodes, no model update",
        "parent_full_action_branch_result_sha256": _sha(FULL_ACTION_RESULT),
        "source_diagnosis_result_sha256": _sha(SOURCE_RESULT),
        "pairs": results,
    }
    _write(OUTPUT / "result.json", result, exclusive=True)
    _write(ROOT / "experiments/oracle-policy-longitudinal-branch-v1-result.json", result, exclusive=True)
    return {key: result[key] for key in ("status", "samples", "speed_passes", "lane_guard_passes",
                                         "damage_guard_passes", "median_speed_delta_m_s",
                                         "hypothesis_supported", "next_step")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(preflight_only=args.preflight_only), sort_keys=True), flush=True)
    except Exception as error:
        print(f"longitudinal branch failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
