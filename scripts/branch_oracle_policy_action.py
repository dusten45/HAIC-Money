"""Test one Oracle action substitution at frozen RLPD failure precursors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from scripts import diagnose_oracle_policy_failures as diagnosis


ROOT = diagnosis.ROOT
PROTOCOL_PATH = ROOT / "experiments/oracle-policy-action-branch-v1.json"
RESULT_PATH = ROOT / "experiments/oracle-policy-diagnosis-v1-result.json"
OUTPUT = ROOT / "runs/oracle-policy-action-branch-v1"
SAMPLES = [
    ("rlpd-seed11", 4272000001),
    ("rlpd-seed11", 4272000003),
    ("rlpd-seed11", 4272000005),
    ("rlpd-seed11", 4272000008),
    ("rlpd-seed11", 4272000009),
    ("rlpd-seed11", 4272000010),
    ("rlpd-seed50", 4272000003),
    ("rlpd-seed50", 4272000005),
    ("rlpd-seed50", 4272000010),
    ("rlpd-seed50", 4272000011),
]
BRANCH_DECISIONS = 4
BRANCH_CONTRACT = {
    "samples": [[policy, seed] for policy, seed in SAMPLES],
    "event_rule": "current replay failed and first measured precursor is curve-entry-overspeed",
    "intervention": "replace one action with Oracle action from identical policy state; replay the next three logged policy actions",
    "prefix_state_tolerance": 1e-5,
    "primary_speed_delta_m_s": -0.5,
    "primary_pass_count": 8,
    "lane_abs_center_error_delta_m_max": 0.5,
    "damage_delta_max": 0.0,
}


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


def _append(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = diagnosis._read_json(PROTOCOL_PATH)
    if protocol.get("branch_contract") != BRANCH_CONTRACT:
        raise diagnosis.DiagnosisError("branch sample or outcome gate differs from frozen contract")
    if protocol.get("runner_sha256") != _sha(Path(__file__).resolve()):
        raise diagnosis.DiagnosisError("branch runner does not match frozen code hash")
    if protocol.get("source_sha256", {}).get("diagnosis_result") != _sha(RESULT_PATH):
        raise diagnosis.DiagnosisError("source diagnosis result changed after branch freeze")
    for relative, expected in protocol.get("source_sha256", {}).items():
        if relative == "diagnosis_result":
            continue
        if _sha(ROOT / relative) != expected:
            raise diagnosis.DiagnosisError(f"branch input hash changed: {relative}")
    result = json.loads(RESULT_PATH.read_text())
    selected = [row for row in result["comparisons"] if
                (row["policy_id"], row["geometry_seed"]) in SAMPLES and
                row["study"] == "rlpd-g0" and not row["replay_finished"] and
                row["first_failure_precursor"]["mode"] == "curve-entry-overspeed"]
    observed = [(row["policy_id"], row["geometry_seed"]) for row in selected]
    if observed != SAMPLES:
        raise diagnosis.DiagnosisError(f"branch sample no longer matches its freeze: {observed}")
    return protocol, result


def _trace_for(policy_id: str, seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result = json.loads(RESULT_PATH.read_text())
    row = next(row for row in result["comparisons"] if
               row["study"] == "rlpd-g0" and row["policy_id"] == policy_id
               and row["geometry_seed"] == seed)
    path = OUTPUT.parent / "oracle-policy-diagnosis-v1" / "policies" / (
        f"{policy_id}-track1-seed{seed}.npz"
    )
    expected = result["rollout_files_sha256"][f"policies/{policy_id}-track1-seed{seed}.npz"]
    if _sha(path) != expected:
        raise diagnosis.DiagnosisError(f"policy trace changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    return row, trace


def _capture(base: Any, controller: Any, env: Any) -> dict[str, Any]:
    state = diagnosis._state(base, controller)
    return {
        "position": state["position"].copy(),
        "speed": state["speed"],
        "heading": state["heading"],
        "center_error": state["center_error"],
        "path_error": state["path_error"],
        "heading_error": state["heading_error"],
        "curvature": state["curvature"],
        "arc_length": state["arc_length"],
        "oracle_action": state["oracle_action"].copy(),
        "progress": float(env._calculate_progress()),
        "damage": float(env.damage.damage),
        "off_track_counter": int(env.off_track_counter),
    }


def _one_branch(row: dict[str, Any], trace: dict[str, np.ndarray], event_step: int,
                mode: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env, base, observation = diagnosis._make_env(
        int(row["track_id"]), int(row["geometry_seed"]), 2000,
    )
    try:
        from haic.oracle_v1 import OracleController

        controller = OracleController(base, target_speed=12.0, avoid_obstacles=True)
        if diagnosis._obs_hash(observation) != trace["observation_sha256"][0]:
            raise diagnosis.DiagnosisError("branch reset observation differs from policy trace")
        for step in range(event_step):
            action = np.asarray(trace["policy_or_oracle_action"][step], dtype=np.float32)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise diagnosis.DiagnosisError(f"policy prefix terminated before branch at {step}")
        before = _capture(base, controller, env)
        state_step = event_step
        if diagnosis._obs_hash(observation) != trace["observation_sha256"][state_step]:
            raise diagnosis.DiagnosisError("branch observation does not match frozen policy state")
        checks = {
            "position_m": float(np.linalg.norm(before["position"] - trace["pre_position"][state_step])),
            "speed_m_s": abs(before["speed"] - float(trace["pre_speed"][state_step])),
            "heading_rad": abs(float(np.arctan2(
                np.sin(before["heading"] - trace["pre_heading"][state_step]),
                np.cos(before["heading"] - trace["pre_heading"][state_step]),
            ))),
            "oracle_action_linf": float(np.max(np.abs(
                before["oracle_action"] - trace["oracle_action_at_state"][state_step]
            ))),
        }
        if any(value > 1e-5 for value in checks.values()):
            raise diagnosis.DiagnosisError(f"branch prefix did not reach the frozen state: {checks}")
        first_action = (np.asarray(trace["policy_or_oracle_action"][state_step], dtype=np.float32)
                        if mode == "policy" else before["oracle_action"])
        snapshots = [before]
        applied = []
        for offset in range(BRANCH_DECISIONS):
            index = state_step + offset
            action = first_action if offset == 0 else np.asarray(
                trace["policy_or_oracle_action"][index], dtype=np.float32,
            )
            if not env.action_space.contains(action):
                raise diagnosis.DiagnosisError(f"invalid branch action: {action}")
            observation, _, terminated, truncated, _ = env.step(action)
            applied.append(action.copy())
            snapshots.append(_capture(base, controller, env))
            if (terminated or truncated) and offset + 1 < BRANCH_DECISIONS:
                raise diagnosis.DiagnosisError("branch ended before its four-decision horizon")
        summary = {
            "mode": mode,
            "track_id": row["track_id"],
            "geometry_seed": row["geometry_seed"],
            "policy_id": row["policy_id"],
            "event_step": event_step,
            "source_trace_sha256": row["source_trace_sha256"],
            "branch_state_match": checks,
            "substituted_action": first_action.astype(float).tolist(),
            "state_at_event": {key: (value.astype(float).tolist() if isinstance(value, np.ndarray) else value)
                               for key, value in before.items()},
            "state_after_four_decisions": {key: (value.astype(float).tolist() if isinstance(value, np.ndarray) else value)
                                           for key, value in snapshots[-1].items()},
        }
        arrays = {
            "actions": np.stack(applied).astype(np.float32),
            "position": np.stack([state["position"] for state in snapshots]).astype(np.float64),
            "speed": np.asarray([state["speed"] for state in snapshots], dtype=np.float64),
            "heading": np.asarray([state["heading"] for state in snapshots], dtype=np.float64),
            "center_error": np.asarray([state["center_error"] for state in snapshots], dtype=np.float64),
            "path_error": np.asarray([state["path_error"] for state in snapshots], dtype=np.float64),
            "heading_error": np.asarray([state["heading_error"] for state in snapshots], dtype=np.float64),
            "curvature": np.asarray([state["curvature"] for state in snapshots], dtype=np.float64),
            "progress": np.asarray([state["progress"] for state in snapshots], dtype=np.float64),
            "damage": np.asarray([state["damage"] for state in snapshots], dtype=np.float64),
        }
        return summary, arrays
    finally:
        env.close()


def run(*, preflight_only: bool = False) -> dict[str, Any]:
    protocol, source_result = verify_protocol()
    if preflight_only:
        return {"status": "preflight_passed_no_environment_resets", "paired_interventions": len(SAMPLES)}
    if OUTPUT.exists():
        raise diagnosis.DiagnosisError("branch output exists; refusing overwrite")
    OUTPUT.mkdir()
    _write(OUTPUT / "metadata.json", {
        "format": "haic-oracle-action-branch-run-v1",
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "runner_sha256": _sha(Path(__file__).resolve()),
        "parent_result_sha256": _sha(RESULT_PATH),
        "paired_interventions": len(SAMPLES),
        "learner_updates": 0,
        "action_conditioning": "recorded current policy prefix; exact state parity required; one action substituted then three recorded policy actions",
    }, exclusive=True)
    pairs = []
    for index, (policy_id, seed) in enumerate(SAMPLES, 1):
        row, trace = _trace_for(policy_id, seed)
        event_step = int(row["first_failure_precursor"]["step"])
        control, control_trace = _one_branch(row, trace, event_step, "policy")
        treatment, treatment_trace = _one_branch(row, trace, event_step, "oracle")
        before = control["state_at_event"]
        treated_before = treatment["state_at_event"]
        if (control["branch_state_match"] != treatment["branch_state_match"]
                or not np.allclose(before["position"], treated_before["position"], atol=1e-5, rtol=0)):
            raise diagnosis.DiagnosisError("policy and Oracle branches did not start from one state")
        speed_delta = treatment_trace["speed"][-1] - control_trace["speed"][-1]
        center_delta = abs(treatment_trace["center_error"][-1]) - abs(control_trace["center_error"][-1])
        damage_delta = treatment_trace["damage"][-1] - control_trace["damage"][-1]
        pair = {
            "policy_id": policy_id,
            "track_id": row["track_id"],
            "geometry_seed": seed,
            "event_step": event_step,
            "source_trace_sha256": row["source_trace_sha256"],
            "policy_branch": control,
            "oracle_action_branch": treatment,
            "four_decision_delta": {
                "speed_m_s_treatment_minus_control": float(speed_delta),
                "abs_center_error_m_treatment_minus_control": float(center_delta),
                "damage_treatment_minus_control": float(damage_delta),
            },
            "speed_gate_pass": bool(speed_delta <= -0.5),
            "lane_guard_pass": bool(center_delta <= 0.5),
            "damage_guard_pass": bool(damage_delta <= 0.0),
        }
        stem = f"{policy_id}-seed{seed}-step{event_step}"
        for mode, arrays in (("policy", control_trace), ("oracle", treatment_trace)):
            path = OUTPUT / f"{stem}-{mode}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            pair[f"{mode}_trace_path"] = path.relative_to(OUTPUT).as_posix()
            pair[f"{mode}_trace_sha256"] = _sha(path)
        pairs.append(pair)
        _append(OUTPUT / "branches.jsonl", pair)
        print(f"branch {index:02d}/{len(SAMPLES)} {policy_id} seed={seed} step={event_step}: "
              f"speed_delta={speed_delta:+.3f} lane_delta={center_delta:+.3f} damage_delta={damage_delta:+.3f}",
              flush=True)

    speed_passes = sum(row["speed_gate_pass"] for row in pairs)
    result = {
        "format": "haic-oracle-action-branch-result-v1",
        "status": "complete_local_one_action_intervention",
        "hypothesis": "The leading RLPD curve-entry precursor is an action-side speed/braking mismatch; replacing only the first qualifying policy action with the fixed Oracle action improves short-horizon speed control without worsening lane error or damage.",
        "primary_metric": "four-decision speed after one substituted action and three recorded policy actions",
        "predeclared_gate": {
            "speed_improved_at_least_0_5_m_s_in": "8 of 10",
            "abs_center_error_delta_m_max": 0.5,
            "damage_delta_max": 0.0,
        },
        "paired_interventions": len(pairs),
        "speed_gate_passes": speed_passes,
        "median_speed_delta_m_s": float(np.median([
            row["four_decision_delta"]["speed_m_s_treatment_minus_control"] for row in pairs
        ])),
        "lane_guard_passes": sum(row["lane_guard_pass"] for row in pairs),
        "damage_guard_passes": sum(row["damage_guard_pass"] for row in pairs),
        "hypothesis_supported": bool(speed_passes >= 8 and all(
            row["lane_guard_pass"] and row["damage_guard_pass"] for row in pairs
        )),
        "next_step": ("pilot a small RLPD Oracle-labelled recovery/action replay update on TRAIN only; hold completion claims until a separate predeclared cell cohort"
                      if speed_passes >= 8 and all(row["lane_guard_pass"] and row["damage_guard_pass"] for row in pairs)
                      else "reject the simple speed-action repair hypothesis; do not add Oracle action imitation based on this result"),
        "resource_use": "20 short deterministic prefixes/branches; no actor training or full-episode reruns",
        "fresh_access": "none",
        "oracle_mutated": False,
        "parent_result_sha256": _sha(RESULT_PATH),
        "branch_protocol_sha256": _sha(PROTOCOL_PATH),
        "source_comparisons": len(source_result["comparisons"]),
        "pairs": pairs,
    }
    _write(OUTPUT / "result.json", result, exclusive=True)
    _write(ROOT / "experiments/oracle-policy-action-branch-v1-result.json", result, exclusive=True)
    return {key: result[key] for key in ("status", "paired_interventions", "speed_gate_passes",
                                         "median_speed_delta_m_s", "lane_guard_passes",
                                         "damage_guard_passes", "hypothesis_supported", "next_step")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(preflight_only=args.preflight_only), sort_keys=True), flush=True)
    except Exception as error:
        print(f"branch experiment failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
