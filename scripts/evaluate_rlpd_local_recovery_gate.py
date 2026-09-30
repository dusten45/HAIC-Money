"""Source-pinned, pixel-only local recovery comparison on consumed G0 roads.

This operator never tunes the gate and never changes the shared environment.
Preflight is zero-reset; invoking without --preflight-only requires separate
run authorization. Outputs are exclusive, nonresumable evidence directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

from scripts import evaluate_rlpd_recovery as original

ROOT = original.ROOT
ARMS = ("original", "local-gate")
GATE_MODULE = "haic/algorithms/rlpd/local_recovery_gate.py"


def action(value: Any) -> np.ndarray:
    result = np.asarray(value)
    if (result.shape != (3,) or result.dtype != np.float32
            or not np.isfinite(result).all() or np.any(result < [-1, 0, 0])
            or np.any(result > [1, 1, 1])):
        raise ValueError("invalid official float32 action")
    return result.copy()


class AuditActor:
    """Only observation reaches act(); privileged collector fields stay outside."""

    def __init__(self, actor: Any, gated: bool):
        self.actor, self.gated = actor, gated
        self.audit: list[dict[str, Any]] = []
        self.triggers = 0

    def reset(self, observation: np.ndarray) -> None:
        self.audit.clear()
        self.triggers = 0
        if self.gated:
            self.actor.reset()
        else:
            self.actor.reset(observation)

    def act(self, observation: np.ndarray) -> np.ndarray:
        applied = action(self.actor.act(observation))
        if self.gated:
            data = self.actor.diagnostics
            triggers = int(data["triggers"])
            if triggers < self.triggers or triggers > self.triggers + 1:
                raise ValueError("invalid cumulative gate trigger count")
            distance = float(data["distance"])
            if not np.isfinite(distance):
                raise ValueError("nonfinite gate distance")
            row = {"gate_active": bool(data["active"]),
                   "gate_trigger": triggers > self.triggers,
                   "gate_distance": distance, "gate_proto_idx": -1 if data["prototype_id"] is None else int(data["prototype_id"]),
                   "gate_holdcounter": int(data["remaining"]),
                   "gate_source_action": action(data["source_action"]),
                   "gate_corrected_action": action(data["corrected_action"])}
            if not 0 <= row["gate_holdcounter"] <= 11:
                raise ValueError("gate hold exceeds predeclared twelve decisions")
            previous = self.audit[-1]["gate_holdcounter"] if self.audit else 0
            if (row["gate_trigger"] and previous > 0
                    or row["gate_active"] != (row["gate_trigger"] or previous > 0)
                    or row["gate_holdcounter"] != (11 if row["gate_trigger"] else max(0, previous - 1))):
                raise ValueError("gate hold extended or miscounted")
            expected = row["gate_corrected_action"] if row["gate_active"] else row["gate_source_action"]
            if not np.array_equal(applied, expected):
                raise ValueError("gate action and audit disagree")
            self.triggers = triggers
        else:
            row = {"gate_active": False, "gate_trigger": False, "gate_distance": 0.0,
                   "gate_proto_idx": -1, "gate_holdcounter": 0,
                   "gate_source_action": applied, "gate_corrected_action": applied}
        self.audit.append(row)
        return applied

    def merge(self, trace: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.audit or len(self.audit) != len(trace["step"]):
            raise ValueError("gate audit/trace decision counts differ")
        arrays = {key: np.asarray([row[key] for row in self.audit]) for key in self.audit[0]}
        applied = np.where(arrays["gate_active"][:, None], arrays["gate_corrected_action"],
                           arrays["gate_source_action"])
        if not np.array_equal(applied, trace["policy_or_oracle_action"]):
            raise ValueError("executed action differs from pixel audit")
        return {**trace, **arrays}


def factory(kind: str, path: Path, digest: str) -> AuditActor:
    if kind == "original":
        from agent import Agent
        return AuditActor(Agent(model_path=str(path)), False)
    from haic.algorithms.rlpd.local_recovery_gate import LocalRecoveryGate
    return AuditActor(LocalRecoveryGate.load(path, expected_sha256=digest, device="cpu"), True)


def inspect_gate(path: Path, digest: str, root: Path) -> dict[str, Any]:
    """Validate a standalone gate without applying PixelActor export validation."""
    import torch
    if original.sha(path) != digest:
        raise ValueError("gate artifact SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "haic-rlpd-local-recovery-gate-v1":
        raise ValueError("not a local recovery gate artifact")
    provenance = payload["provenance"]
    source = provenance["source_actor"]
    corrected = provenance["corrected_actor"]
    if source["sha256"] != original.V5_SHA:
        raise ValueError("source is not immutable V5")
    for item in (source, corrected):
        original.checked(root, item["path"], item["sha256"])
    policy = payload["policy"]
    if policy["hold_decisions"] != 12 or policy["radius_fraction"] != .5:
        raise ValueError("gate menu changed")
    calibration = provenance["calibration"]
    if (calibration["protected_triggers"] != 0 or calibration["protected_points"] <= 0
            or calibration["protected_images_exact_source_action_checked"] != calibration["protected_points"]
            or calibration["guide_rows"] != 87):
        raise ValueError("protected pool calibration failed")
    sources = provenance["source_hashes"]
    if GATE_MODULE not in sources:
        raise ValueError("gate implementation source is not pinned")
    for name, expected in sources.items():
        original.checked(root, name, expected)
    protocol = path.with_name("protocol.json")
    protocol_relative = protocol.relative_to(root).as_posix()
    original.checked(root, protocol_relative, provenance["protocol_sha256"])
    frozen_builder = json.loads(protocol.read_text())
    if (frozen_builder["source_actor"] != source or frozen_builder["corrected_actor"] != corrected
            or frozen_builder["source_hashes"] != sources or frozen_builder["policy"] != policy
            or frozen_builder["inputs"] != provenance["inputs"]):
        raise ValueError("gate/builder provenance differs")
    evidence = {protocol_relative: provenance["protocol_sha256"]}
    for key in ("primary_result", "repair_receipt"):
        item = frozen_builder[key]
        original.checked(root, item["path"], item["sha256"])
        evidence[item["path"]] = item["sha256"]
    from haic.algorithms.rlpd.local_recovery_gate import tensor_state_sha
    for name, item in (("source", source), ("corrected", corrected)):
        exported = torch.load(root / item["path"], map_location="cpu", weights_only=True)
        if (tensor_state_sha(payload["actor_state_dicts"][name]) != item["state_sha256"]
                or tensor_state_sha(exported["actor_state_dict"]) != item["state_sha256"]):
            raise ValueError("embedded actor differs from pinned export")
    return {"source_actor": source, "corrected_actor": corrected,
            "sources_sha256": sources, "policy": policy,
            "calibration": calibration, "evidence_sha256": evidence}


def preflight(actors: dict[str, dict[str, str]], make=factory) -> dict[str, Any]:
    fixtures = [np.zeros((4, 84, 84), np.float32),
                np.linspace(0, 1, 4 * 84 * 84, dtype=np.float32).reshape(4, 84, 84)]
    loaded = {kind: [make(kind, Path(item["path"]), item["sha256"]) for _ in range(2)]
              for kind, item in actors.items()}
    records = []
    for obs in fixtures:
        values = {}
        for kind, pair in loaded.items():
            outputs = []
            for actor in pair:
                actor.reset(obs)
                outputs.append(actor.act(obs))
                # Check a second decision as well as the forced initial fallback.
                outputs.append(actor.act(obs))
            if not np.array_equal(outputs[0], outputs[2]) or not np.array_equal(outputs[1], outputs[3]):
                raise ValueError("double reload action mismatch")
            values[kind] = [v.tolist() for v in outputs[:2]]
        for gate, source in zip(loaded["local-gate"], loaded["original"]):
            if gate.audit[0]["gate_active"] or gate.audit[0]["gate_trigger"]:
                raise ValueError("initial synthetic action triggered gate")
            for gate_row, source_row in zip(gate.audit, source.audit):
                if not np.array_equal(gate_row["gate_source_action"], source_row["gate_source_action"]):
                    raise ValueError("gate source differs from root Agent")
                if gate_row["gate_active"] or gate_row["gate_trigger"]:
                    raise ValueError("outside synthetic fixture triggered gate")
        records.append(values)
    return {"environment_resets": 0, "fixtures": 2, "double_reload_bit_exact": True,
            "initial_and_outside_source_exact": True, "actions": records}


def telemetry(trace: dict[str, np.ndarray]) -> dict[str, Any]:
    n = len(trace["step"])
    fields = ("heading_error", "center_error", "pre_speed", "damage", "progress",
              "curvature", "off_track_counter", "collision", "reward")
    for key in fields:
        if trace[key].shape != (n,) or not np.isfinite(trace[key]).all():
            raise ValueError(f"misaligned/nonfinite primary telemetry: {key}")
    windows = []
    for index in np.flatnonzero(trace["gate_trigger"]):
        index = int(index)
        end = min(n - 1, index + 63)
        handoff = index + 12
        later = min(n - 1, handoff + 63)
        windows.append({"trigger_index": index, "end_index": end,
                        "complete_5s_window": end - index == 63,
                        "elapsed_seconds": (end - index) * .08,
                        "source_action": trace["gate_source_action"][index].tolist(),
                        "switched_action": trace["policy_or_oracle_action"][index].tolist(),
                        "before": {key: float(trace[key][index]) for key in fields},
                        "later": {key: float(trace[key][end]) for key in fields},
                        "window_max_abs_heading": float(np.max(np.abs(trace["heading_error"][index:end+1]))),
                        "window_max_abs_lateral": float(np.max(np.abs(trace["center_error"][index:end+1]))),
                        "window_collision_decisions": int(np.sum(trace["collision"][index:end+1])),
                        "handoff_index": handoff if handoff < n else None,
                        "complete_post_hold_5s_window": later - handoff == 63,
                        "post_hold_later_index": later if handoff < n else None,
                        "post_hold_later": {key: float(trace[key][later]) for key in fields} if handoff < n else None})
    return {"decisions": n, "active_decisions": int(np.sum(trace["gate_active"])),
            "guard_active_fraction": float(np.mean(trace["gate_active"])),
            "trigger_count": int(np.sum(trace["gate_trigger"])),
            "mean_abs_heading": float(np.mean(np.abs(trace["heading_error"]))),
            "mean_abs_lateral": float(np.mean(np.abs(trace["center_error"]))),
            "mean_speed": float(np.mean(trace["pre_speed"])),
            "mean_damage": float(np.mean(trace["damage"])),
            "mean_progress": float(np.mean(trace["progress"])),
            "raw_reward_sum": float(np.sum(trace["reward"])), "trigger_followups": windows,
            "timing": "heading/lateral/speed/curvature pre-decision; reward/progress/damage/contact post-decision",
            "post_hold_limit": "Gate may re-trigger during followup; this is not guaranteed source-only continuation.",
            "independent_active_episodes": False, "causal": False}


def evaluate(gate_artifact: Path, gate_sha256: str, output: str, *, preflight_only=False,
             root: Path = ROOT, make=factory, collect=None) -> dict[str, Any]:
    if (original.diagnosis.FRAME_SKIP != 4 or original.diagnosis.WARMUP != 50
            or original.diagnosis.TARGET_SPEED != 12.0):
        raise ValueError("shared collector contract changed")
    if re.fullmatch(r"runs/[a-z0-9][a-z0-9-]*", output) is None:
        raise ValueError("output must be runs/<newslug>")
    target = root / output
    if target.exists() or target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("output must be new with existing nonsymlink runs parent")
    gate = (root / gate_artifact).resolve()
    if not gate.is_relative_to(root.resolve()) or re.fullmatch(r"[0-9a-f]{64}", gate_sha256) is None:
        raise ValueError("gate requires repository-local path and exact SHA256")
    metadata = inspect_gate(gate, gate_sha256, root)
    archive, evidence = original.archived_screen(root)
    ledger = original.checked(root, original.G0, original.G0_SHA)
    historical = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    roads = {}
    for row in historical:
        seed, digest = row["geometry_seed"], row["road_centerline_sha256"]
        if row["track_id"] != 1 or (seed in roads and roads[seed] != digest):
            raise ValueError("inconsistent consumed G0 evidence")
        roads[seed] = digest
    if set(roads) != set(original.SEEDS) or any(r["road_centerline_sha256"] != roads[r["geometry_seed"]] for r in archive):
        raise ValueError("consumed road census/hash mismatch")
    actors = {"original": {"path": str(root / metadata["source_actor"]["path"]), "sha256": original.V5_SHA},
              "local-gate": {"path": str(gate), "sha256": gate_sha256}}
    sources = original.source_hashes(root)
    for name, digest in metadata["sources_sha256"].items():
        if name in sources and sources[name] != digest:
            raise ValueError("gate/evaluator source closure disagrees")
        sources[name] = digest
    own = "scripts/evaluate_rlpd_local_recovery_gate.py"
    sources[own] = original.sha(root / own)
    frozen = {"format": "haic-rlpd-local-recovery-gate-evaluation-v1",
              "contract": original.CONTRACT, "thresholds": original.THRESHOLDS,
              "actors": actors, "gate": metadata, "sources_sha256": sources,
              "runtime": original.runtime(), "roads": roads,
              "evidence_sha256": {**evidence, **metadata["evidence_sha256"], original.G0: original.G0_SHA},
              "oracle_contract": {"target_speed_m_s": 12.0, "avoid_obstacles": True}}
    frozen["preflight"] = preflight(actors, make)
    frozen_hash = original.canonical_sha(frozen)
    target.mkdir()
    original.write_json(target / "protocol.json", frozen)
    original.write_json(target / "preflight.json", {"status": "passed", "environment_resets": 0,
                                                  "protocol_content_sha256": frozen_hash})

    def recheck():
        if original.canonical_sha(json.loads((target / "protocol.json").read_text())) != frozen_hash:
            raise ValueError("protocol changed")
        for name, expected in {**sources, **frozen["evidence_sha256"]}.items():
            original.checked(root, name, expected)
        for item in (*actors.values(), metadata["corrected_actor"]):
            if original.sha(root / item["path"]) != item["sha256"]:
                raise ValueError("actor/artifact changed")
        if original.runtime() != frozen["runtime"]:
            raise ValueError("runtime changed")

    rows = []
    intents = 0

    def manifest(status):
        original.write_json(target / "manifest.json", {"status": status,
            "files_sha256": {p.name: original.sha(p) for p in target.iterdir() if p.is_file()},
            "protocol_content_sha256": frozen_hash, "episodes_complete": len(rows),
            "reset_intents": intents,
            "environment_resets": 0 if preflight_only else len(rows) if status == "complete" else None,
            "failed_attempt_reset_count_known": status != "failed"})

    try:
        recheck()
        if preflight_only:
            manifest("preflight_only")
            return {"status": "preflight_only", "environment_resets": 0, "output": output}
        collector = collect or original.diagnosis._collect_episode
        for kind in ARMS:
            for seed in original.SEEDS:
                recheck()
                actor = make(kind, Path(actors[kind]["path"]), actors[kind]["sha256"])
                recheck()
                name = f"{kind}-seed-{seed}"
                case = {"track_id": 1, "geometry_seed": seed, "max_steps": 2000,
                        "source_road_centerline_sha256": roads[seed]}
                original.write_json(target / f"{name}-reset-intent.json", {
                    "case": case, "actor": actors[kind], "phase": "full-episode",
                    "protocol_content_sha256": frozen_hash, "no_retry": True})
                intents += 1
                summary, trace = collector(case, "policy", actor)
                recheck()
                n = summary["steps"]
                if (not 0 < n <= 2000 or len(trace["step"]) != n
                        or (not summary["terminated"] and not summary["truncated"] and n != 2000)
                        or summary["track_id"] != 1 or summary["geometry_seed"] != seed
                        or summary["road_centerline_sha256"] != roads[seed]):
                    raise ValueError("incomplete or mismatched episode")
                trace = actor.merge(trace)
                if trace["gate_active"][0] or trace["gate_trigger"][0]:
                    raise ValueError("protected initial road image activated gate")
                metrics = telemetry(trace)
                if not np.isclose(metrics["raw_reward_sum"], summary["total_reward"], rtol=1e-10, atol=1e-8):
                    raise ValueError("primary raw reward disagrees with summary")
                trace_path = target / f"{name}.npz"
                trace_hash = original.diagnosis._write_trace(trace_path, trace)
                row = {**summary, "phase": "full-episode", "actor_id": kind,
                       "actor_sha256": actors[kind]["sha256"], "events": original.events(trace),
                       "terminal_curve_association": original.terminal_curve_association(trace, summary),
                       "censored": not summary["terminated"] and not summary["truncated"] and not summary["finished"],
                       "telemetry": metrics, "trace_path": trace_path.name, "trace_sha256": trace_hash,
                       "initial_source_action": trace["gate_source_action"][0].tolist(),
                       "initial_executed_action": trace["policy_or_oracle_action"][0].tolist(),
                       "reset_accessible_state_sha256": original.canonical_sha({
                           key: np.asarray(trace[key][0]).tolist() for key in
                           ("pre_position", "pre_velocity", "pre_heading", "pre_speed",
                            "center_error", "heading_error", "curvature")})}
                original.write_json(target / f"{name}-receipt.json", row)
                rows.append(row)
        left, right = ([r for r in rows if r["actor_id"] == kind] for kind in ARMS)
        if any(a["initial_observation_sha256"] != b["initial_observation_sha256"] for a, b in zip(left, right)):
            raise ValueError("paired reset image mismatch")
        if any(a["reset_accessible_state_sha256"] != b["reset_accessible_state_sha256"] for a, b in zip(left, right)):
            raise ValueError("paired accessible reset state mismatch")
        if any(a["initial_executed_action"] != b["initial_source_action"]
               or a["initial_executed_action"] != b["initial_executed_action"] for a, b in zip(left, right)):
            raise ValueError("protected initial source action mismatch")
        per_actor = {}
        for kind in ARMS:
            selected = [r for r in rows if r["actor_id"] == kind]
            decisions = sum(r["steps"] for r in selected)
            per_actor[kind] = {"denominator": 12, "finish_count": sum(r["finished"] for r in selected),
                "censored": sum(r["censored"] for r in selected),
                "curve_entry_associated_terminal_failures": sum(r["terminal_curve_association"]["associated_terminal_failure"] for r in selected),
                "precursors": original.census(selected), "decisions": decisions,
                "trigger_count": sum(r["telemetry"]["trigger_count"] for r in selected),
                "guard_active_fraction": sum(r["telemetry"]["active_decisions"] for r in selected) / decisions,
                "mean_final_damage": float(np.mean([r["damage"] for r in selected])),
                "mean_final_progress": float(np.mean([r["progress"] for r in selected])),
                "mean_raw_return": float(np.mean([r["total_reward"] for r in selected])),
                **{key: sum(r["telemetry"][key] * r["steps"] for r in selected) / decisions
                   for key in ("mean_abs_heading", "mean_abs_lateral", "mean_speed", "mean_damage", "mean_progress")}}
        baseline = [r for r in archive if r["actor_sha256"] == original.V5_SHA]
        result = {"format": "haic-rlpd-local-recovery-gate-evaluation-result-v1", "status": "complete",
                  "episodes": rows, "per_actor": per_actor, "official_score": False,
                  "fresh_generalization": False, "environment_resets": intents,
                  "paired_reset_checks": {"roads": 12, "geometry_exact": True,
                      "image_exact": True, "accessible_state_exact": True, "initial_source_action_exact": True,
                      "hidden_solver_state_identity_proven": False},
                  "contemporaneous_pairs": {"original->local-gate": original.transition_table(left, right)},
                  "versus_archived_immutable_v5": {kind: original.transition_table(baseline, [r for r in rows if r["actor_id"] == kind]) for kind in ARMS},
                  "limitations": ["Consumed TRAIN; one actor seed; no official or fresh-generalization claim.",
                      "Active states/episodes are selected, dependent visits, not independent experiments.",
                      "Five-second followups and terrain telemetry are descriptive closed-loop proxies, not causal effects.",
                      "Collector caps without real done are censored; no environment modification or retry."]}
        original.write_json(target / "result.json", result)
        manifest("complete")
        return result
    except BaseException as error:
        original.write_json(target / "failure.json", {"completed_episodes": len(rows), "reset_intents": intents,
            "no_retry": True, "error": f"{type(error).__name__}: {error}",
            "limitation": "Collector exceptions expose no partial primary trace; preserve reset intent and unknown cost."})
        manifest("failed")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-artifact", type=Path, required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.gate_artifact, args.gate_sha256, args.output, preflight_only=args.preflight_only)
        print(json.dumps({key: result[key] for key in ("status", "environment_resets")}))
        return 0
    except Exception as error:
        print(f"local recovery evaluation failed closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
