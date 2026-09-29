"""Collect privileged coupled recovery branches on consumed RLPD G0 cells only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from scripts import branch_oracle_policy_action as branch
from scripts import diagnose_oracle_policy_failures as diagnosis

ROOT = diagnosis.ROOT
PROTOCOL = ROOT / "experiments/rlpd-coupled-recovery-v1.json"
OUTPUT = ROOT / "runs/rlpd-coupled-recovery-v1"
CONTRACT = {
    "failure_samples": [list(x) for x in branch.SAMPLES],
    "finish_controls": "first four parent replay_finished RLPD rows sorted policy_id/seed",
    "control_anchor": "first curve-entry-overspeed by fixed parent rule, otherwise step40",
    "horizons": [12, 25], "followup_decisions": 63, "max_steps": 2000,
    "oracle": {"target_speed": 12.0, "avoid_obstacles": True, "action": "full steering gas brake"},
    "qualification": {"final_abs_lateral_max_m": 6.0, "minimum_progress_delta": 0.02,
                      "added_damage_max": 0.0, "no_failure_before_handoff_plus_63": True},
    "parity": "current-runtime actor executed prefix; every observation and accessible state; no historical parity",
    "tolerance": 1e-5, "threads": 1, "fresh_cells": 0,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def native(action):
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (3,) or not np.isfinite(action).all() or np.any(action < [-1, 0, 0]) or np.any(action > 1):
        raise diagnosis.DiagnosisError("invalid official action")
    action[1:] = action[1:] * 2 - 1
    return action


def terminal_flag(terminated, info):
    return bool(terminated or info.get("finished", False) or info.get("retire_reason") in {"crash", "off_track", "out_of_bounds"})


def select_cases():
    result = diagnosis._read_json(branch.RESULT_PATH)
    rows = [x for x in result["comparisons"] if x["study"] == "rlpd-g0"]
    failures = []
    for policy, seed in branch.SAMPLES:
        row = next(x for x in rows if (x["policy_id"], x["geometry_seed"]) == (policy, seed))
        if row["replay_finished"] or row["first_failure_precursor"]["mode"] != "curve-entry-overspeed":
            raise diagnosis.DiagnosisError("parent failure sample changed")
        failures.append(dict(row, anchor_step=int(row["first_failure_precursor"]["step"]), stratum="failure"))
    controls = sorted([x for x in rows if x["replay_finished"]], key=lambda x: (x["policy_id"], x["geometry_seed"]))[:4]
    if len(controls) != 4:
        raise diagnosis.DiagnosisError("missing finished parent controls")
    for row in controls:
        path = ROOT / "runs/oracle-policy-diagnosis-v1/policies" / f"{row['policy_id']}-track1-seed{row['geometry_seed']}.npz"
        with np.load(path, allow_pickle=False) as t:
            a, o = t["policy_or_oracle_action"], t["oracle_action_at_state"]
            qualifying = np.flatnonzero((np.abs(t["curvature"]) >= .025) & (t["pre_speed"] >= 14)
                                       & (a[:, 1] >= .35) & (a[:, 2] < .05) & (o[:, 2] >= .10))
        failures.append(dict(row, anchor_step=int(qualifying[0]) if len(qualifying) else 40, stratum="finish-control"))
    if any(x["track_id"] != 1 or not 4272000001 <= x["geometry_seed"] <= 4272000012 for x in failures):
        raise diagnosis.DiagnosisError("selection escaped consumed G0")
    return failures, result


def freeze():
    cases, result = select_cases()
    paths = {"scripts/collect_rlpd_coupled_recovery.py", "scripts/branch_oracle_policy_action.py",
             "scripts/diagnose_oracle_policy_failures.py", "experiments/oracle-policy-diagnosis-v1-result.json",
             "runs/20260926-rlpd-g0-completion-v1/cells.jsonl", "runs/20260926-rlpd-g0-completion-v1/manifest.json",
             "agent.py", "common_adapter.py", "env_wrapper.py", "damage.py"}
    for directory in ("haic/oracle_v1", "core/vendor"):
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / directory).rglob("*.py"))
    paths.add("haic/oracle_v1/PROVENANCE.json")
    for row in cases:
        spec = diagnosis.RLPD_ACTORS[row["policy_id"]]
        diagnosis._actor_path(row["policy_id"])
        paths.add(spec["actor_path"])
        if branch._sha(ROOT / row["source_trace_path"]) != row["source_trace_sha256"]:
            raise diagnosis.DiagnosisError("original G0 trace hash mismatch")
        paths.add(row["source_trace_path"])
        relative = f"policies/{row['policy_id']}-track1-seed{row['geometry_seed']}.npz"
        path = ROOT / "runs/oracle-policy-diagnosis-v1" / relative
        if branch._sha(path) != result["rollout_files_sha256"][relative]:
            raise diagnosis.DiagnosisError("parent trace hash mismatch")
        paths.add(path.relative_to(ROOT).as_posix())
    import torch
    protocol = {"format": "haic-rlpd-coupled-recovery-protocol-v1", "status": "frozen", "created_at": utc(),
                "contract": CONTRACT, "cases": cases, "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
                "source_sha256": {p: branch._sha(ROOT / p) for p in sorted(paths)},
                "limitation": "privileged upper-bound successful branches, not deployable policy or generalization"}
    branch._write(PROTOCOL, protocol, exclusive=True)
    return {"status": "frozen", "anchors": len(cases), "protocol_sha256": branch._sha(PROTOCOL)}


def preflight():
    p = diagnosis._read_json(PROTOCOL)
    if p["contract"] != CONTRACT or p["status"] != "frozen":
        raise diagnosis.DiagnosisError("contract mismatch")
    for name, expected in p["source_sha256"].items():
        if branch._sha(ROOT / name) != expected:
            raise diagnosis.DiagnosisError(f"source drift: {name}")
    cases, _ = select_cases()
    if cases != p["cases"]:
        raise diagnosis.DiagnosisError("anchor drift")
    import torch
    if p["runtime"] != {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__}:
        raise diagnosis.DiagnosisError("runtime drift")
    return p


def parity(observation, state, reference, index):
    if diagnosis._obs_hash(observation) != reference["observation_sha256"][index]:
        raise diagnosis.DiagnosisError(f"current prefix pixel mismatch at {index}")
    for key in ("position", "speed", "heading", "center_error", "path_error", "heading_error", "curvature", "arc_length", "progress", "damage", "off_track_counter", "oracle_action"):
        if not np.allclose(state[key], reference[key][index], atol=1e-5, rtol=0):
            raise diagnosis.DiagnosisError(f"current prefix state mismatch {key} at {index}")


def qualify(arrays, anchor, horizon):
    end = anchor + horizon + 63
    n = len(arrays["reward"])
    available = end <= n
    if not available:
        return {"qualified": False, "followup_complete": False, "assessment_step": end}
    damage = arrays["damage"][anchor:end+1]
    early_failure = bool(np.any(arrays["episode_end"][anchor:end] & ~arrays["finished"][anchor:end]))
    progress_delta = float(arrays["progress"][end] - arrays["progress"][anchor])
    lateral = float(abs(arrays["center_error"][end]))
    added_damage = float(np.max(damage) - damage[0])
    return {"qualified": bool(not early_failure and lateral <= 6 and progress_delta >= .02 and added_damage <= 0),
            "followup_complete": True, "assessment_step": end, "progress_delta": progress_delta,
            "final_abs_lateral_m": lateral, "added_damage": added_damage, "early_failure": early_failure,
            "followup_seconds": 63 * 4 / 50}


def collect(case, actor, horizon, reference=None, partial_path=None):
    from haic.oracle_v1 import OracleController
    env, base, obs = diagnosis._make_env(1, case["geometry_seed"], 2000)
    values = defaultdict(list)
    anchor = case["anchor_step"]
    try:
        controller = OracleController(base, target_speed=12.0, avoid_obstacles=True)
        actor.reset(obs)
        initial = np.rint(obs * 255).astype(np.uint8)
        pixels = initial
        info = {}
        capped = False
        values["frames"].append(initial[-1])
        if not np.array_equal(initial, np.tile(initial[-1], (4, 1, 1))):
            raise diagnosis.DiagnosisError("reset frames not identical")
        for step in range(2000):
            state = branch._capture(base, controller, env)
            values["observation_sha256"].append(diagnosis._obs_hash(obs))
            for key, value in state.items():
                values[key].append(value)
            if reference is not None and step <= anchor:
                parity(obs, state, reference, step)
            proposed = np.asarray(actor.act(obs), dtype=np.float32)
            role = "prefix" if step < anchor else "oracle" if step < anchor + horizon else "actor"
            official = (reference["applied_action"][step].copy() if reference is not None and step < anchor
                        else state["oracle_action"].copy() if role == "oracle" else proposed)
            if not env.action_space.contains(official):
                raise diagnosis.DiagnosisError("invalid applied action")
            next_obs, reward, terminated, truncated, info = env.step(official)
            capped = step == 1999 and not (terminated or truncated)
            for key, value in {"proposed_action": native(proposed), "executed_action": native(official),
                               "applied_action": official.copy(), "reward": reward, "step": step, "role": role,
                               "terminated": bool(terminated), "truncated": bool(truncated),
                               "terminal": terminal_flag(terminated, info), "episode_end": bool(terminated or truncated),
                               "finished": bool(info.get("finished", False)), "timestamp_utc": utc(),
                               "simulation_time_s": float(getattr(base, "t", 0.0)),
                               "next_observation_sha256": diagnosis._obs_hash(next_obs)}.items():
                values[key].append(value)
            pixels = np.rint(next_obs * 255).astype(np.uint8)
            values["frames"].append(pixels[-1])
            if not np.array_equal(pixels[:-1], np.rint(obs[1:] * 255).astype(np.uint8)):
                raise diagnosis.DiagnosisError("frame stack shift mismatch")
            obs = next_obs
            if terminated or truncated or capped:
                break
        final_state = branch._capture(base, controller, env)
        for key, value in final_state.items():
            values[key].append(value)
        arrays = {key: np.asarray(value) for key, value in values.items()}
        arrays["initial_stack"] = initial
        arrays["final_stack"] = pixels
        if len(arrays["reward"]) <= anchor:
            assessment = {"qualified": False, "followup_complete": False, "anchor_reached": False}
        else:
            assessment = qualify(arrays, anchor, horizon)
        if capped:
            assessment["qualified"] = False
            assessment["censored_episode_excluded"] = True
        arrays["recovery_mask"] = (arrays["role"] == "oracle") & bool(assessment["qualified"])
        summary = {"steps": len(arrays["reward"]), "finished": bool(info.get("finished", False)),
                   "censored": bool(capped), "reason": info.get("retire_reason"), "anchor_step": anchor,
                   "horizon": horizon, "local_recovery_qualified": bool(assessment["qualified"]),
                   "assessment": assessment, "accepted_transitions": int(arrays["recovery_mask"].sum()),
                   "prefix_decisions": min(anchor, len(arrays["reward"])), "total_reward": float(arrays["reward"].sum()),
                   "road_centerline_sha256": diagnosis._track_hash(base)}
        summary["current_anchor_curve_entry_overspeed"] = bool(len(arrays["reward"]) > anchor
            and abs(arrays["curvature"][anchor]) >= .025 and arrays["speed"][anchor] >= 14
            and arrays["applied_action"][anchor, 1] >= .35 and arrays["applied_action"][anchor, 2] < .05
            and arrays["oracle_action"][anchor, 2] >= .10) if horizon == 0 else None
        return summary, arrays
    except Exception:
        if partial_path is not None:
            diagnosis._write_trace(partial_path, {key: np.asarray(value) for key, value in values.items()})
        raise
    finally:
        env.close()


def run():
    p = preflight()
    if OUTPUT.exists():
        raise diagnosis.DiagnosisError("output exists; no overwrite/resume")
    import torch
    from agent import Agent
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUTPUT.mkdir()
    started = time.monotonic()
    branch._write(OUTPUT / "metadata.json", {"protocol_sha256": branch._sha(PROTOCOL), "started_at": utc(),
                                            "runtime": p["runtime"], "threads": 1, "learner_updates": 0}, exclusive=True)
    actors = {name: Agent(model_path=str(diagnosis._actor_path(name)[0])) for name in diagnosis.RLPD_ACTORS}
    episodes, pairs, unique = [], [], set()
    unique_by_stratum = {"failure": set(), "finish-control": set()}
    try:
        for index, case in enumerate(p["cases"]):
            reference = None
            outcomes = []
            for horizon in (0, 12, 25):
                preflight()
                stem = f"{case['policy_id']}-seed{case['geometry_seed']}-h{horizon}"
                branch._append(OUTPUT / "reset-intents.jsonl", {"stem": stem, "timestamp": utc(), "geometry_seed": case["geometry_seed"], "track_id": 1})
                summary, arrays = collect(case, actors[case["policy_id"]], horizon, reference, OUTPUT / f"{stem}-partial.npz")
                if horizon == 0:
                    reference = arrays
                path = OUTPUT / f"{stem}.npz"
                diagnosis._write_trace(path, arrays)
                for t in np.flatnonzero(arrays["recovery_mask"]):
                    identity = (str(arrays["observation_sha256"][t]), arrays["executed_action"][t].tobytes())
                    unique.add(identity)
                    unique_by_stratum[case["stratum"]].add(identity)
                row = dict(summary, episode_id=len(episodes), path=path.name, sha256=branch._sha(path), mode="actor" if not horizon else f"oracle-{horizon}",
                           policy_id=case["policy_id"], geometry_seed=case["geometry_seed"], track_id=1, stratum=case["stratum"],
                           parent_trace_sha256=p["source_sha256"][f"runs/oracle-policy-diagnosis-v1/policies/{case['policy_id']}-track1-seed{case['geometry_seed']}.npz"],
                           actor_sha256=case["actor_sha256"])
                episodes.append(row)
                outcomes.append(row)
                branch._append(OUTPUT / "episodes.jsonl", row)
                print(f"{index+1}/14 {stem} steps={row['steps']} qualified={row['local_recovery_qualified']} finish={row['finished']}", flush=True)
            for treatment in outcomes[1:]:
                known = not outcomes[0]["censored"] and not treatment["censored"]
                pairs.append({"policy_id": case["policy_id"], "geometry_seed": case["geometry_seed"], "stratum": case["stratum"],
                              "horizon": treatment["horizon"], "control_finished": outcomes[0]["finished"], "treatment_finished": treatment["finished"],
                              "finish_pair_known": known,
                              "rescued": known and not outcomes[0]["finished"] and treatment["finished"],
                              "harmed": known and outcomes[0]["finished"] and not treatment["finished"],
                              "local_recovery_qualified": treatment["local_recovery_qualified"]})
        preflight()
        accepted = sum(x["accepted_transitions"] for x in episodes)
        geometries = sorted({x["geometry_seed"] for x in episodes if x["accepted_transitions"]})
        manifest = {"format": "haic-rlpd-recovery-dataset-v1", "status": "complete", "protocol_sha256": branch._sha(PROTOCOL),
                    "episodes": episodes, "pairs": pairs, "counts": {"anchors": 14, "episodes": len(episodes), "reset_intents": len(episodes),
                    "decisions": sum(x["steps"] for x in episodes), "prefix_decisions": sum(x["prefix_decisions"] for x in episodes),
                    "accepted_transitions": accepted, "unique_accepted_transitions": len(unique), "accepted_geometries": len(geometries),
                    "support_by_stratum": {s: {"accepted_transitions": sum(x["accepted_transitions"] for x in episodes if x["stratum"] == s),
                                              "unique_accepted_transitions": len(unique_by_stratum[s]),
                                              "accepted_geometries": len({x["geometry_seed"] for x in episodes if x["stratum"] == s and x["accepted_transitions"]})}
                                            for s in unique_by_stratum},
                    "unknown_finish_pairs": sum(not x["finish_pair_known"] for x in pairs),
                    "rejected_oracle_transitions": sum(min(x["horizon"], max(0, x["steps"]-x["anchor_step"])) for x in episodes if x["mode"] != "actor") - accepted,
                    "qualified_branches": sum(x["local_recovery_qualified"] for x in episodes if x["mode"] != "actor"),
                    "rescued": sum(x["rescued"] for x in pairs), "harmed": sum(x["harmed"] for x in pairs)},
                    "accepted_geometry_seeds": geometries, "elapsed_wall_seconds": time.monotonic()-started,
                    "finished_at": utc(), "deployable_policy": False, "fresh_generalization": False,
                    "limitation": "privileged branch-selection upper bound on repeatedly consumed TRAIN; local qualification is separate from finish"}
        branch._write(OUTPUT / "manifest.json", manifest, exclusive=True)
        branch._write(OUTPUT / "result.json", manifest, exclusive=True)
        return manifest["counts"]
    except Exception as error:
        branch._write(OUTPUT / "failure.json", {"timestamp": utc(), "error": repr(error), "completed_episodes": len(episodes),
                                              "decisions_in_completed_episodes": sum(x["steps"] for x in episodes)}, exclusive=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = freeze() if args.freeze else ({"status": "preflight_passed_no_resets", "anchors": len(preflight()["cases"])} if args.preflight_only else run())
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(f"collector failed closed: {error!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
