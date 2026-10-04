"""No-reset, fixed-candidate reward/termination/Q decomposition for TD-MPC2.

This is not a policy gate. A separate SHA-frozen protocol is required before any
real checkpoint can be deserialized; no environment or optimizer is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from haic.algorithms.tdmpc2.model import two_hot_inv
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
from scripts import diagnose_tdmpc2_h5_branches as branch
from scripts import diagnose_tdmpc2_h5_fixed_q as fixed
from scripts import score_tdmpc2_overshoot_archived_branches as score


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/tdmpc2-overshoot-planner-terms-v1.json"
OUTPUT = "runs/tdmpc2-overshoot-planner-terms-20260929-v1.json"
SCORE_SHA = "ff9511b83952e92b14bd8f1aa9f7e93f34ea5670c13528a9179c5b159ae2c994"
SCORE_PROTOCOL_SHA = "44edd04cc7a579c3867ed35f5763645d0c92b2971fd979d5d141ea87f5802aa8"
FIXED_SHA = "6b6bf566973c8a3b57b732343f4f6a74dbf8ec449f66d19c7e70a4af9f2e6219"
SELF = "scripts/diagnose_tdmpc2_overshoot_planner_terms.py"
TEST = "tests/test_diagnose_tdmpc2_overshoot_planner_terms.py"
NAMES = fixed.NAMES
PAIRS = tuple(combinations(range(5), 2))
BOOTSTRAP_OFFSET = 100_000


def _ref(path: str, sha: str) -> dict:
    return {"path": path, "sha256": sha}


def _body(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _archived_inputs(original: dict, scored: dict, old_spec: dict) -> list[dict]:
    """Validate both archived receipts' identities, bytes and real outcomes."""
    fixed._verify_receipt(original, old_spec)
    if (scored.get("body_sha256") != _body({k: v for k, v in scored.items() if k != "body_sha256"})
            or scored.get("format") != score.RESULT_FORMAT or scored.get("status") != "FAIL"
            or scored.get("protocol") != _ref(score.PROTOCOL, scored["protocol"]["sha256"])
            or scored.get("original_branch", {}).get("primary") != _ref(score.OLD_PRIMARY, score.OLD_PRIMARY_SHA)
            or scored.get("environment_resets") != 0 or scored.get("optimizer_updates") != 0
            or scored.get("scope") != "reused_TRAIN_archived_branches_only"
            or scored.get("gate", {}).get("passed") is not False
            or scored.get("gate", {}).get("h5_normalized_error", {}).get("passed") is not False):
        raise ValueError("overshoot score body/failed gate/branch lineage differs")
    old_entries, new_entries = original["anchors"], scored.get("anchors")
    if not isinstance(new_entries, list) or len(new_entries) != len(old_entries):
        raise ValueError("overshoot score anchors differ")
    inputs = []
    for old, new, (episode, start) in zip(old_entries, new_entries, branch.ANCHORS):
        if (new.get("episode_id"), new.get("start_step"), new.get("geometry_seed"),
                new.get("anchor_model_observation_sha256")) != (
                    episode, start, branch.ROADS[episode], old["anchor_model_observation_sha256"]):
            raise ValueError("overshoot score anchor bytes/identity differ")
        if (not isinstance(new.get("candidates"), list) or
                len(new["candidates"]) != len(NAMES)):
            raise ValueError("overshoot score candidate count differs")
        for before, after, name in zip(old["candidates"], new["candidates"], NAMES):
            if (before["candidate"] != name or after.get("candidate") != name
                    or after.get("model_action_bytes_hex") != before["model_action_bytes_hex"]
                    or after.get("steps") != 5 or after.get("ending") is not None
                    or not math.isclose(after.get("real_h5_raw_return", math.inf),
                                        before["real_discounted_raw_return"], abs_tol=1e-9, rel_tol=0)
                    or not math.isclose(after.get("real_h3_prefix_raw_return", math.inf),
                                        before["real_h3_prefix_raw_return"], abs_tol=1e-9, rel_tol=0)
                    or not math.isclose(after.get("old_h5_reward_return", math.inf),
                                        before["predicted_reward_return"], abs_tol=1e-9, rel_tol=0)
                    or not all(type(after.get(key)) in (float, int) and math.isfinite(after[key])
                               for key in ("new_h3_reward_return", "new_h5_reward_return"))):
                raise ValueError("overshoot score candidate action/return/prediction differs")
        pixels = np.frombuffer(bytes.fromhex(old["anchor_model_observation_hex"]), np.uint8).copy()
        actions = {name: np.frombuffer(bytes.fromhex(old["candidate_action_bytes_hex"][name]),
                                           dtype="<f4").copy().reshape(5, 3) for name in NAMES}
        inputs.append({"episode_id": episode, "start_step": start, "geometry_seed": branch.ROADS[episode],
                       "pixels": pixels.reshape(branch.PIXELS), "actions": actions,
                       "original": old, "overshoot": new})
    return inputs


def preflight(root: Path, protocol_sha256: str) -> dict:
    """Pin all source and primary receipts BEFORE any checkpoint deserialization."""
    root = Path(root).resolve(strict=True)
    spec = branch._json(branch.pinned(root, PROTOCOL, branch._sha(protocol_sha256)).read_bytes())
    if (set(spec) != {"format", "purpose", "score_source", "fixed_q_summary", "source_sha256",
                      "runtime", "output", "augmentation_seed", "bootstrap_offset", "discount",
                      "tie_tolerance", "anchors", "candidates", "policy_release"}
            or spec["format"] != "haic-tdmpc2-overshoot-planner-terms-v1"
            or spec["purpose"] != "read-only-consumed-TRAIN-fixed-candidate-mechanism"
            or spec["score_source"] != {"protocol": _ref(score.PROTOCOL, SCORE_PROTOCOL_SHA),
                                        "primary": _ref(score.OUTPUT, SCORE_SHA)}
            or spec["fixed_q_summary"] != _ref("experiments/tdmpc2-h5-fixed-q-v1-result.json", FIXED_SHA)
            or spec["output"] != OUTPUT or spec["augmentation_seed"] != 834
            or spec["bootstrap_offset"] != BOOTSTRAP_OFFSET
            or spec["discount"] != branch.DISCOUNT or spec["tie_tolerance"] != branch.TIE_TOLERANCE
            or spec["anchors"] != [{"episode_id": ep, "start_step": start} for ep, start in branch.ANCHORS]
            or spec["candidates"] != list(NAMES) or spec["policy_release"] is not False):
        raise ValueError("not a separately frozen diagnostic-only protocol")
    score_sha = spec["score_source"]["protocol"]["sha256"]
    bundle = score.preflight(root, score_sha)  # Both complete run lineages and checkpoint SHA pins.
    if bundle["protocol_sha256"] != score_sha:
        raise ValueError("score protocol lineage differs")
    expected = {**bundle["spec"]["source_sha256"],
                "scripts/diagnose_tdmpc2_h5_fixed_q.py": branch.digest(root / "scripts/diagnose_tdmpc2_h5_fixed_q.py")}
    # The fixed-Q operator is separately pinned by its frozen summary below.
    fixed_summary = branch._json(branch.pinned(root, spec["fixed_q_summary"]["path"], FIXED_SHA).read_bytes())
    if (fixed_summary.get("operator") != "scripts/diagnose_tdmpc2_h5_fixed_q.py"
            or fixed_summary.get("operator_sha256") != expected["scripts/diagnose_tdmpc2_h5_fixed_q.py"]
            or fixed_summary.get("source_h5_primary_receipt_sha256") != score.OLD_PRIMARY_SHA
            or fixed_summary.get("source_h5_branch_protocol_sha256") != score.OLD_PROTOCOL_SHA
            or fixed_summary.get("environment_resets") != 0 or fixed_summary.get("optimizer_steps") != 0
            or fixed_summary.get("informative_real_h5_pairs") != 91):
        raise ValueError("fixed-Q source/primary summary differs")
    expected.update({SELF: branch.digest(root / SELF), TEST: branch.digest(root / TEST)})
    if spec["source_sha256"] != expected or spec["runtime"] != bundle["spec"]["runtime"]:
        raise ValueError("diagnostic source/runtime pins differ")
    for name, sha in expected.items():
        branch.pinned(root, name, sha)
    if branch.runtime_identity() != spec["runtime"]:
        raise ValueError("diagnostic runtime differs")
    scored = branch._json(branch.pinned(root, score.OUTPUT, SCORE_SHA).read_bytes())
    score._check_body(scored)
    if (scored.get("protocol") != spec["score_source"]["protocol"]
            or scored.get("training_source") != bundle["spec"]["training_source"]
            or scored.get("original_branch") != bundle["spec"]["original_branch"]
            or scored.get("source_sha256") != bundle["spec"]["source_sha256"]
            or scored.get("runtime") != spec["runtime"]):
        raise ValueError("overshoot score model/source receipt differs")
    inputs = _archived_inputs(bundle["primary"], scored,
                              branch._json(branch.pinned(root, score.OLD_PROTOCOL, score.OLD_PROTOCOL_SHA).read_bytes()))
    return {"spec": spec, "score_bundle": bundle, "fixed_summary": fixed_summary, "inputs": inputs,
            "protocol_sha256": protocol_sha256, "score_primary": scored}


def _terms(model: Any, pixels: np.ndarray, actions: dict[str, np.ndarray], seed: int,
           horizon: int) -> dict:
    """One common sampled-policy draw across ten fixed Q pairs at this anchor/H."""
    if horizon not in (3, 5) or tuple(actions) != NAMES or pixels.shape != branch.PIXELS or pixels.dtype != np.uint8:
        raise ValueError("invalid fixed-candidate decomposition input")
    if any(a.shape != (5, 3) or a.dtype != np.float32 or not np.isfinite(a).all() or
           np.any(np.abs(a) > 1) for a in actions.values()):
        raise ValueError("invalid five-step candidate action bytes")
    bootstrap_seed = seed + BOOTSTRAP_OFFSET
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        z0 = model.encode(torch.from_numpy(pixels.copy())[None], None)
        if z0.ndim != 2 or z0.shape[0] != 1 or not torch.isfinite(z0).all():
            raise ValueError("invalid encoded anchor")
        z = z0.repeat(5, 1)
        sequence = torch.from_numpy(np.stack([actions[name] for name in NAMES], axis=1).copy())
        alive = torch.ones((5, 1), dtype=torch.bool)
        ungated = torch.zeros((5, 1), dtype=z.dtype)
        gated = torch.zeros_like(ungated)
        rewards, probabilities, alive_before, alive_after = [], [], [], []
        for t in range(horizon):
            r = two_hot_inv(model.reward(z, sequence[t], None), model.cfg)
            next_z = model.next(z, sequence[t], None)
            if (r.shape != (5, 1) or next_z.shape != z.shape or
                    not torch.isfinite(r).all() or not torch.isfinite(next_z).all()):
                raise ValueError("nonfinite or malformed model reward/next latent")
            alive_before.append(alive[:, 0].tolist())
            ungated += branch.DISCOUNT ** t * r
            gated += branch.DISCOUNT ** t * torch.where(alive, r, 0)
            rewards.append(r[:, 0].tolist())
            z = next_z
            p = model.termination(z, None)
            if p.shape != (5, 1) or not torch.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
                raise ValueError("invalid predicted termination probability")
            probabilities.append(p[:, 0].tolist())
            alive = alive & ~(p > .5)
            alive_after.append(alive[:, 0].tolist())
        torch.manual_seed(bootstrap_seed)
        bootstrap_action, _ = model.pi(z, None)
        if (bootstrap_action.shape != (5, 3) or not torch.isfinite(bootstrap_action).all()):
            raise ValueError("invalid sampled bootstrap action")
        bootstrap_action = bootstrap_action.clamp(-1, 1)
        logits = model.Q(z, bootstrap_action, None, return_type="all")
        if (logits.shape != (5, 5, model.cfg.num_bins) or not torch.isfinite(logits).all()):
            raise ValueError("invalid five-head categorical online Q")
        decoded = two_hot_inv(logits, model.cfg)
        if decoded.shape != (5, 5, 1) or not torch.isfinite(decoded).all():
            raise ValueError("invalid decoded online Q")
        output = {}
        for index, name in enumerate(NAMES):
            per_pair = {}
            for left, right in PAIRS:
                pair_name = f"{left},{right}"
                q_term = (branch.DISCOUNT ** horizon *
                          (decoded[left, index, 0] + decoded[right, index, 0]) / 2
                          if alive[index, 0] else torch.zeros((), dtype=z.dtype))
                per_pair[pair_name] = {"bootstrap_q_term": float(q_term),
                                       "full_score": float(gated[index, 0] + q_term)}
            output[name] = {"raw_reward_sequence": [row[index] for row in rewards],
                            "termination_probability": [row[index] for row in probabilities],
                            "alive_before_reward": [row[index] for row in alive_before],
                            "alive_after_transition": [row[index] for row in alive_after],
                            "ungated_reward": float(ungated[index, 0]),
                            "gated_reward": float(gated[index, 0]),
                            "bootstrap_action": bootstrap_action[index].tolist(),
                            "decoded_online_q_heads": decoded[:, index, 0].tolist(),
                            "bootstrap_alive": bool(alive[index, 0]), "fixed_pairs": per_pair}
        # Independently compare with the actual planner scorer, including t0/t4 endings.
        for pair in PAIRS:
            planner = TDMPC2Planner(fixed._FixedPair(model, pair), PlannerConfig(
                action_dim=3, discount=branch.DISCOUNT, horizon=horizon, episodic=True,
                num_samples=5, num_elites=1, num_pi_trajs=0, num_bins=model.cfg.num_bins))
            torch.manual_seed(bootstrap_seed)
            expected = planner._estimate_value(z0.repeat(5, 1), sequence[:horizon])
            for index, name in enumerate(NAMES):
                observed = output[name]["fixed_pairs"][f"{pair[0]},{pair[1]}"]["full_score"]
                if not math.isclose(observed, float(expected[index, 0]), abs_tol=2e-4, rel_tol=1e-6):
                    raise ValueError("term decomposition differs from production planner score")
    return {"bootstrap_seed": bootstrap_seed, "candidates": output}


def _quality(entries: list[dict], model: str, horizon: str, component: str) -> dict:
    grouped = {}
    for seed in branch.ROADS:
        results = []
        for entry in entries:
            if entry["geometry_seed"] != seed:
                continue
            result = entry["models"][model][horizon]["candidates"]
            values = {name: (result[name]["fixed_pairs"][component]["full_score"]
                             if component in result[name]["fixed_pairs"] else result[name][component])
                      for name in NAMES}
            results.append(fixed._quality(values, entry["real_h5_return"]))
        if len(results) != 3:
            raise ValueError("road anchor count differs")
        # _aggregate expects twelve; aggregate three anchor outcomes locally instead.
        counts = {key: sum(q["pairs"][key] for q in results)
                  for key in ("concordant", "discordant", "predicted_tie", "real_tie")}
        grouped[str(seed)] = {"pairs": counts, "informative_real_pairs": sum(counts[k] for k in
                              ("concordant", "discordant", "predicted_tie")),
                              "tied_best": sum(q["best_real_h5_tie"] for q in results),
                              "mean_regret": math.fsum(q["real_h5_regret"] for q in results) / 3}
    all_rows = []
    for entry in entries:
        result = entry["models"][model][horizon]["candidates"]
        values = {name: (result[name]["fixed_pairs"][component]["full_score"]
                         if component in result[name]["fixed_pairs"] else result[name][component])
                  for name in NAMES}
        all_rows.append(fixed._quality(values, entry["real_h5_return"]))
    total = fixed._aggregate(all_rows)
    if total["informative_real_pairs"] != 91 or total["pairs"]["real_tie"] != 29:
        raise ValueError("real H5 pair denominator changed")
    return {"total": total, "by_road": grouped}


def _write_exclusive(root: Path, report: dict) -> None:
    path = root / OUTPUT
    if (path.parent != root / "runs" or not path.parent.is_dir() or path.parent.is_symlink()
            or path.exists() or path.is_symlink()):
        raise ValueError("diagnostic receipt must be absent under real runs directory")
    report["body_sha256"] = _body(report)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o644), "w", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(root: Path, protocol_sha256: str, *, execute: bool = False) -> dict:
    bound = preflight(root, protocol_sha256)
    if not execute:
        return {"status": "preflight_only", "environment_resets": 0, "optimizer_steps": 0,
                "policy_release": False, "protocol_sha256": protocol_sha256}
    root = Path(root).resolve(strict=True)
    if (root / OUTPUT).exists() or (root / OUTPUT).is_symlink():
        raise ValueError("diagnostic receipt already exists")
    original, new = score._models(root, bound["score_bundle"])
    entries = []
    # Verify ALL sixty original/new reward predictions before scoring any Q/terminal terms.
    for item in bound["inputs"]:
        seed = 834 + item["episode_id"] * 1000 + item["start_step"]
        old_prediction = branch.predicted_returns(original, item["pixels"], item["actions"], seed)
        new_prediction = score._predict_both(new, item["pixels"], item["actions"], seed)
        for old_row, new_row in zip(item["original"]["candidates"], item["overshoot"]["candidates"]):
            name = old_row["candidate"]
            if (not math.isclose(old_prediction[name], old_row["predicted_reward_return"], abs_tol=1e-4, rel_tol=0)
                    or not math.isclose(new_prediction[name]["h3"], new_row["new_h3_reward_return"], abs_tol=1e-4, rel_tol=0)
                    or not math.isclose(new_prediction[name]["h5"], new_row["new_h5_reward_return"], abs_tol=1e-4, rel_tol=0)):
                raise ValueError("archived reward prediction reproduction failed")
    for item in bound["inputs"]:
        seed = 834 + item["episode_id"] * 1000 + item["start_step"]
        terms = {label: {f"h{h}": _terms(model, item["pixels"], item["actions"], seed, h)
                         for h in (3, 5)} for label, model in (("raw100k", original),
                                                               ("overshoot100k", new))}
        for before, after in zip(item["original"]["candidates"], item["overshoot"]["candidates"]):
            name = before["candidate"]
            for label, horizon, archived in (("raw100k", "h3", before["h3_predicted_reward_return"]),
                                             ("raw100k", "h5", before["predicted_reward_return"]),
                                             ("overshoot100k", "h3", after["new_h3_reward_return"]),
                                             ("overshoot100k", "h5", after["new_h5_reward_return"])):
                if not math.isclose(terms[label][horizon]["candidates"][name]["ungated_reward"],
                                    archived, abs_tol=1e-4, rel_tol=0):
                    raise ValueError("decomposed ungated reward differs from archived prediction")
        entries.append({"episode_id": item["episode_id"], "start_step": item["start_step"],
                        "geometry_seed": item["geometry_seed"],
                        "anchor_model_observation_sha256": item["original"]["anchor_model_observation_sha256"],
                        "candidate_action_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                                                    for name, value in item["actions"].items()},
                        "real_h5_return": {row["candidate"]: row["real_discounted_raw_return"]
                                           for row in item["original"]["candidates"]},
                        "models": terms})
    summary = {label: {h: {component: _quality(entries, label, h, component) for component in
                          ("ungated_reward", "gated_reward", *[f"{a},{b}" for a, b in PAIRS])}
                       for h in ("h3", "h5")} for label in ("raw100k", "overshoot100k")}
    if (summary["raw100k"]["h5"]["ungated_reward"]["total"]["pairs"]["concordant"] != 56
            or summary["overshoot100k"]["h5"]["ungated_reward"]["total"]["pairs"]["concordant"] != 74):
        raise ValueError("original/new reward-only ranking reproduction differs")
    for record in bound["fixed_summary"]["fixed_critic_pair_concordant_over_91"]:
        pair = f"{record['heads'][0]},{record['heads'][1]}"
        for horizon in ("h3", "h5"):
            observed = summary["raw100k"][horizon][pair]["total"]
            if (observed["pairs"]["concordant"] != record[horizon]
                    or observed["selected_best_real_h5_ties"] != record[f"{horizon}_best_of_12"]):
                raise ValueError("original fixed-Q control could not be reproduced")
    preflight(root, protocol_sha256)  # Rehash protocol, complete runs and receipts before output.
    report = {"format": "haic-tdmpc2-overshoot-planner-terms-result-v1",
              "status": "complete_diagnostic_only", "policy_release": False,
              "prior_four_way_gate_passed": False, "environment_resets": 0, "optimizer_steps": 0,
              "scope": "same_12_consumed_TRAIN_anchors_five_fixed_candidates_not_full_MPPI",
              "protocol": _ref(PROTOCOL, protocol_sha256), "score_source": bound["spec"]["score_source"],
              "fixed_q_summary": bound["spec"]["fixed_q_summary"],
              "source_sha256": bound["spec"]["source_sha256"], "runtime": bound["spec"]["runtime"],
              "checkpoint_sources": {"raw100k": bound["score_bundle"]["spec"]["original_branch"],
                                     "overshoot100k": bound["score_bundle"]["spec"]["training_source"]},
              "summary": summary, "anchors": entries}
    _write_exclusive(root, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.protocol.as_posix() != PROTOCOL:
        parser.error("only the separately frozen diagnostic protocol is accepted")
    print(json.dumps(run(ROOT, args.protocol_sha256, execute=args.execute), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
