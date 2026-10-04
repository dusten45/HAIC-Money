"""Read-only fixed-Q-pair audit of the frozen RAW100k H3/H5 five-action branches.

No environment is constructed or reset. This scores archived actions, not an MPPI
search or a driving policy. Only SHA-verified local checkpoint pickle is loaded.
The original stochastic planner receipt remains unchanged and is not a matched
fixed-pair result.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from haic.algorithms.tdmpc2.haic_env import environment_action
from haic.algorithms.tdmpc2.model import two_hot_inv
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
from scripts import diagnose_tdmpc2_h5_branches as branch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "a29cbcd2e59687201a7b0fba68e119f596e88633e06a51300639bf78461b33c9"
RESULT = "experiments/tdmpc2-h5-branches-v1-result.json"
RESULT_SHA = "0efcceb8d09ea4dac72f350d400ac2f2ad4d193e300f483eb712f65cd05c2913"
RECEIPT = "runs/tdmpc2-raw100k-h5-branches-20260929-v1.json"
RECEIPT_SHA = "23f7e6898b362cd789b92d6a1f2f184e1de551596458ae08117058570f1bcefd"
NAMES = ("logged", "coast", "gas", "brake", "left_gas")
PAIRS = tuple(combinations(range(5), 2))


def _finite(value: object) -> float:
    if not isinstance(value, (int, float)) or type(value) is bool or not math.isfinite(value):
        raise ValueError("nonfinite or nonnumeric archived branch outcome")
    return float(value)


def _quality(scores: dict[str, float], real: dict[str, float]) -> dict:
    if set(scores) != set(real) or tuple(scores) != NAMES:
        raise ValueError("fixed five-candidate input set/order differs")
    counts = {"concordant": 0, "discordant": 0, "predicted_tie": 0, "real_tie": 0}
    for index, left in enumerate(NAMES):
        for right in NAMES[index + 1:]:
            actual = _finite(real[left]) - _finite(real[right])
            predicted = _finite(scores[left]) - _finite(scores[right])
            if abs(actual) <= branch.TIE_TOLERANCE:
                counts["real_tie"] += 1
            elif abs(predicted) <= branch.TIE_TOLERANCE:
                counts["predicted_tie"] += 1
            else:
                counts["concordant" if (actual > 0) == (predicted > 0) else "discordant"] += 1
    selected = max(NAMES, key=scores.__getitem__)
    regret = max(real.values()) - real[selected]
    return {"pairs": counts, "selected_candidate": selected,
            "selected_real_h5_return": real[selected], "real_h5_regret": regret,
            "best_real_h5_tie": regret <= branch.TIE_TOLERANCE}


def _aggregate(qualities: list[dict]) -> dict:
    if len(qualities) != len(branch.ANCHORS):
        raise ValueError("fixed-Q summary requires exactly twelve anchors")
    counts = {key: sum(q["pairs"][key] for q in qualities)
              for key in ("concordant", "discordant", "predicted_tie", "real_tie")}
    informative = counts["concordant"] + counts["discordant"] + counts["predicted_tie"]
    if informative != 91 or counts["real_tie"] != 29 or sum(counts.values()) != 120:
        raise ValueError("archived real H5 informative/tied pair denominator changed")
    return {"pairs": counts, "informative_real_pairs": informative,
            "concordance": counts["concordant"] / informative,
            "eligible_anchors": len(qualities),
            "selected_best_real_h5_ties": sum(q["best_real_h5_tie"] for q in qualities),
            "mean_real_h5_regret": math.fsum(q["real_h5_regret"] for q in qualities) / len(qualities)}


def _verify_receipt(receipt: dict, spec: dict) -> list[dict]:
    """Check primary body, candidate inputs and outcome totals before torch.load."""
    body = {key: value for key, value in receipt.items() if key != "body_sha256"}
    body_sha = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if (receipt.get("body_sha256") != body_sha
            or receipt.get("format") != "haic-tdmpc2-raw100k-h5-branches-result-v1"
            or receipt.get("status") != "complete" or receipt.get("protocol_sha256") != PROTOCOL_SHA
            or receipt.get("source") != spec["source"]
            or receipt.get("logged_quality") != spec["logged_quality"]
            or receipt.get("source_sha256") != spec["source_sha256"]
            or receipt.get("runtime") != spec["runtime"]
            or receipt.get("scope") != "four_reused_consumed_TRAIN_roads_same_reconstructed_accessible_state_only"
            or receipt.get("environment_resets_attempted") != 72
            or receipt.get("maximum_resets") != 72
            or receipt.get("replay_bound_before_first_reset") is not True
            or receipt.get("same_state_counterfactual_ranking") is not True
            or receipt.get("historical_hidden_box2d_state_proven") is not False
            or receipt.get("fresh_or_official_score") is not False
            or receipt.get("horizon") != 5 or receipt.get("discount") != branch.DISCOUNT
            or receipt.get("tie_tolerance") != branch.TIE_TOLERANCE):
        raise ValueError("frozen H5 receipt/body/RAW lineage differs")
    entries = receipt.get("anchors")
    if not isinstance(entries, list) or len(entries) != len(branch.ANCHORS):
        raise ValueError("expected exactly twelve archived H5 anchors")
    original = {label: [] for label in ("h3_reward", "h5_reward", "h3_planner", "h5_planner")}
    for entry, (episode, start) in zip(entries, branch.ANCHORS):
        if (not isinstance(entry, dict) or entry.get("episode_id") != episode
                or entry.get("start_step") != start or entry.get("track_id") != 1
                or entry.get("geometry_seed") != branch.ROADS[episode]):
            raise ValueError("archived H5 anchor identity/order differs")
        pixel_hex = entry.get("anchor_model_observation_hex")
        if not isinstance(pixel_hex, str):
            raise ValueError("missing archived anchor pixel bytes")
        pixels = bytes.fromhex(pixel_hex)
        if (len(pixels) != math.prod(branch.PIXELS)
                or hashlib.sha256(pixels).hexdigest() != entry.get("anchor_model_observation_sha256")):
            raise ValueError("archived anchor pixel SHA/shape differs")
        actions = entry.get("candidate_action_bytes_hex")
        rows = entry.get("candidates")
        if (not isinstance(actions, dict) or set(actions) != set(NAMES)
                or not isinstance(rows, list) or len(rows) != len(NAMES)):
            raise ValueError("expected exactly five archived action suffixes")
        real = {}
        for name, row in zip(NAMES, rows):
            if not isinstance(row, dict) or row.get("candidate") != name:
                raise ValueError("archived candidate identity/order differs")
            action_hex = actions[name]
            if not isinstance(action_hex, str) or row.get("model_action_bytes_hex") != action_hex:
                raise ValueError("archived candidate action bytes differ")
            raw_action = bytes.fromhex(action_hex)
            if len(raw_action) != 5 * 3 * 4:
                raise ValueError("archived action length is not five steps")
            array = np.frombuffer(raw_action, dtype="<f4").reshape(5, 3)
            if not np.isfinite(array).all() or np.any(np.abs(array) > 1):
                raise ValueError("archived action outside model bounds")
            if name != "logged" and raw_action != np.asarray(spec["candidates"][name], dtype="<f4").tobytes():
                raise ValueError("archived fixed suffix differs from protocol")
            native = b"".join(environment_action(a).tobytes() for a in array).hex()
            if row.get("native_action_bytes_hex") != native:
                raise ValueError("archived native action bytes differ")
            rewards = row.get("raw_rewards")
            if (not isinstance(rewards, list) or len(rewards) != 5 or row.get("steps") != 5
                    or row.get("full_h5") is not True or row.get("ranking_eligible") is not True
                    or row.get("exclusion_reason") is not None):
                raise ValueError("archived H5 suffix incomplete")
            rewards = [_finite(reward) for reward in rewards]
            actual = math.fsum(branch.DISCOUNT**t * reward for t, reward in enumerate(rewards))
            prefix = math.fsum(branch.DISCOUNT**t * reward for t, reward in enumerate(rewards[:3]))
            if (not math.isclose(actual, _finite(row.get("real_discounted_raw_return")), rel_tol=0, abs_tol=1e-8)
                    or not math.isclose(prefix, _finite(row.get("real_h3_prefix_raw_return")), rel_tol=0, abs_tol=1e-8)):
                raise ValueError("archived real H5/H3 raw return differs from rewards")
            real[name] = actual
        for label, field in (("h3_reward", "h3_predicted_reward_return"),
                             ("h5_reward", "predicted_reward_return"),
                             ("h3_planner", "h3_predicted_planner_score"),
                             ("h5_planner", "h5_predicted_planner_score")):
            quality = _quality({row["candidate"]: _finite(row.get(field)) for row in rows}, real)
            archived = entry.get("matched_action_quality", {})
            selected = archived.get("selected", {}).get(label, {})
            ranked = archived.get("rankings", {}).get(f"{label}_vs_real_h5", {})
            if (ranked.get("pairs") != quality["pairs"]
                    or selected.get("candidate") != quality["selected_candidate"]
                    or selected.get("best_real_h5_tie") is not quality["best_real_h5_tie"]
                    or not math.isclose(_finite(selected.get("real_h5_regret")),
                                        quality["real_h5_regret"], rel_tol=0, abs_tol=1e-8)):
                raise ValueError("original branch per-anchor ranking/selection differs")
            original[label].append(quality)
    summary = receipt.get("summary", {})
    matched = summary.get("matched_action_quality", {})
    if (summary.get("full_h5_candidates") != 60 or summary.get("informative_real_pairs") != 91
            or summary.get("quality_gate_passed") is not True
            or summary.get("pairs") != {**_aggregate(original["h5_reward"])["pairs"],
                                        "terminal_or_short_excluded": 0}):
        raise ValueError("original branch summary/denominator differs")
    for label, qualities in original.items():
        computed = _aggregate(qualities)
        ranking = matched.get("rankings", {}).get(f"{label}_vs_real_h5", {})
        selected = matched.get("selected", {}).get(label, {})
        if (ranking.get("pairs") != computed["pairs"]
                or ranking.get("informative_real_pairs") != 91
                or selected.get("best_real_h5_ties") != computed["selected_best_real_h5_ties"]
                or not math.isclose(_finite(selected.get("mean_real_h5_regret")),
                                    computed["mean_real_h5_regret"], rel_tol=0, abs_tol=1e-8)):
            raise ValueError("original branch matched H3/H5 summary differs")
    return entries


class _FixedPair:
    """Only Q's random two-of-five draw is replaced; MPPI scoring is unmodified."""

    def __init__(self, model: Any, pair: tuple[int, int]):
        self.model, self.pair = model, pair

    def reward(self, z: torch.Tensor, action: torch.Tensor, task: None) -> torch.Tensor:
        return self.model.reward(z, action, task)

    def next(self, z: torch.Tensor, action: torch.Tensor, task: None) -> torch.Tensor:
        return self.model.next(z, action, task)

    def termination(self, z: torch.Tensor, task: None) -> torch.Tensor:
        return self.model.termination(z, task)

    def pi(self, z: torch.Tensor, task: None) -> tuple:
        return self.model.pi(z, task)

    def Q(self, z: torch.Tensor, action: torch.Tensor, task: None, *, return_type: str) -> torch.Tensor:
        if return_type != "avg":
            raise ValueError("fixed-pair bootstrap requires MPPI's average Q")
        all_logits = self.model.Q(z, action, task, return_type="all")
        if (not isinstance(all_logits, torch.Tensor) or all_logits.shape != (5, z.shape[0], 101)
                or not torch.isfinite(all_logits).all()):
            raise ValueError("fixed-Q requires five finite categorical critic heads")
        return two_hot_inv(all_logits[list(self.pair)], self.model.cfg).mean(dim=0)


def _score(model: Any, pixels: np.ndarray, actions: dict[str, np.ndarray], seed: int) -> dict:
    if (tuple(actions) != NAMES or pixels.shape != branch.PIXELS or pixels.dtype != np.uint8
            or any(a.shape != (5, 3) or a.dtype != np.float32 or not np.isfinite(a).all()
                   or np.any(np.abs(a) > 1) for a in actions.values())
            or model.cfg.num_q != 5 or model.cfg.num_bins != 101):
        raise ValueError("fixed-Q requires one pixel anchor and exactly five H5 actions/five critics")
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        z0 = model.encode(torch.from_numpy(pixels.copy())[None], None)
        if not isinstance(z0, torch.Tensor) or z0.ndim != 2 or z0.shape[0] != 1 or not torch.isfinite(z0).all():
            raise ValueError("nonfinite fixed-Q anchor encoding")
        # Same encoded anchor and sampled-policy random stream for every pair/H.
        trajectories = torch.from_numpy(np.stack([actions[name] for name in NAMES], axis=1).copy())
        reward_only = {}
        for horizon in (3, 5):
            z = z0.repeat(5, 1)
            totals = torch.zeros((5, 1), dtype=z.dtype, device=z.device)
            for step in range(horizon):
                reward = two_hot_inv(model.reward(z, trajectories[step], None), model.cfg)
                if reward.shape != (5, 1) or not torch.isfinite(reward).all():
                    raise ValueError("nonfinite fixed-Q reward-only prediction")
                totals += branch.DISCOUNT**step * reward
                if step < horizon - 1:
                    z = model.next(z, trajectories[step], None)
                    if z.shape != z0.repeat(5, 1).shape or not torch.isfinite(z).all():
                        raise ValueError("nonfinite fixed-Q latent")
            reward_only[f"h{horizon}"] = dict(zip(NAMES, map(float, totals[:, 0].tolist())))
        fixed = []
        bootstrap_seed = seed + 100_000
        for pair in PAIRS:
            scores = {}
            for horizon in (3, 5):
                config = PlannerConfig(action_dim=3, discount=branch.DISCOUNT, horizon=horizon,
                                       episodic=True, num_samples=5, num_elites=1,
                                       num_pi_trajs=0, num_bins=model.cfg.num_bins)
                planner = TDMPC2Planner(_FixedPair(model, pair), config)
                torch.manual_seed(bootstrap_seed)
                values = planner._estimate_value(z0.repeat(5, 1), trajectories[:horizon])
                if values.shape != (5, 1) or not torch.isfinite(values).all():
                    raise ValueError("nonfinite fixed-Q terminal/discount/bootstrap planner score")
                scores[f"h{horizon}"] = dict(zip(NAMES, map(float, values[:, 0].tolist())))
            fixed.append({"pair": list(pair), "scores": scores})
    return {"reward_only": reward_only, "fixed_pairs": fixed, "bootstrap_seed": bootstrap_seed}


def run(root: Path = ROOT) -> dict:
    root = Path(root).resolve(strict=True)
    bundle = branch.preflight(root, PROTOCOL_SHA)  # Source, runtime, RAW ledgers/checkpoint and protocol pins.
    summary = branch._json(branch.pinned(root, RESULT, RESULT_SHA).read_bytes())
    if (summary.get("format") != "haic-tdmpc2-raw100k-h5-branches-summary-v1"
            or summary.get("status") != "complete_weak_reward_gate_passed_planner_gate_held"
            or summary.get("protocol") != branch.PROTOCOL
            or summary.get("operator") != "scripts/diagnose_tdmpc2_h5_branches.py"
            or summary.get("primary_receipt") != RECEIPT
            or summary.get("primary_receipt_sha256") != RECEIPT_SHA
            or summary.get("protocol_sha256") != PROTOCOL_SHA
            or summary.get("operator_sha256") != bundle["spec"]["source_sha256"]["scripts/diagnose_tdmpc2_h5_branches.py"]
            or summary.get("source_model_sha256") != bundle["spec"]["source"]["checkpoint"]["sha256"]
            or summary.get("source_model_trained_horizon") != 3
            or summary.get("preflight_environment_resets") != 0
            or summary.get("execution_environment_resets_attempted") != 72
            or summary.get("anchors") != 12 or summary.get("full_h5_candidate_suffixes") != 60
            or summary.get("informative_h5_pairs") != 91 or summary.get("real_h5_ties_at_1e-6") != 29):
        raise ValueError("frozen branch summary does not bind receipt/protocol/operator/checkpoint")
    receipt = branch._json(branch.pinned(root, RECEIPT, RECEIPT_SHA).read_bytes())
    entries = _verify_receipt(receipt, bundle["spec"])
    original = receipt["summary"]["matched_action_quality"]
    for label in ("h3_reward", "h5_reward", "h3_planner", "h5_planner"):
        record = summary.get("matched_same_five_candidates_vs_real_h5", {}).get(
            f"{label}_only" if label.endswith("reward") else f"{label.replace('planner', 'terminal_q_planner_score')}", {})
        if (record.get("concordant") != original["rankings"][f"{label}_vs_real_h5"]["pairs"]["concordant"]
                or record.get("discordant") != original["rankings"][f"{label}_vs_real_h5"]["pairs"]["discordant"]
                or record.get("informative_pairs") != 91
                or record.get("best_action_tied_at_1e-6") != original["selected"][label]["best_real_h5_ties"]
                or not math.isclose(_finite(record.get("mean_real_h5_regret")),
                                    original["selected"][label]["mean_real_h5_regret"],
                                    rel_tol=0, abs_tol=1e-8)):
            raise ValueError("frozen branch result/primary body comparisons differ")
    # The trusted checkpoint pickle is opened only after all source and receipt checks above.
    anchors, model = branch.load_bound_model(bundle, root)
    if len(anchors) != len(entries):
        raise ValueError("RAW replay/archived anchor count differs")
    output = []
    for entry, anchor in zip(entries, anchors):
        if (anchor.episode_id != entry["episode_id"] or anchor.step != entry["start_step"]
                or anchor.seed != entry["geometry_seed"]
                or anchor.observations[-1].tobytes().hex() != entry["anchor_model_observation_hex"]):
            raise ValueError("archived anchor pixels do not match complete RAW checkpoint replay")
        archived = entry["candidate_action_bytes_hex"]
        if any(branch.candidate_actions(anchor)[name].tobytes().hex() != archived[name] for name in NAMES):
            raise ValueError("archived suffix bytes do not match RAW replay/protocol")
        actions = {name: np.frombuffer(bytes.fromhex(archived[name]), dtype="<f4").copy().reshape(5, 3)
                   for name in NAMES}
        seed = bundle["spec"]["augmentation_seed"] + anchor.episode_id * 1000 + anchor.step
        result = _score(model, anchor.observations[-1], actions, seed)
        real = {row["candidate"]: row["real_discounted_raw_return"] for row in entry["candidates"]}
        for name, row in zip(NAMES, entry["candidates"]):
            if (not math.isclose(result["reward_only"]["h3"][name], row["h3_predicted_reward_return"],
                                 rel_tol=0, abs_tol=1e-4)
                    or not math.isclose(result["reward_only"]["h5"][name], row["predicted_reward_return"],
                                        rel_tol=0, abs_tol=1e-4)):
                raise ValueError("independently recomputed reward-only branch score differs")
        for horizon in ("h3", "h5"):
            result["reward_only"][horizon] = {"scores": result["reward_only"][horizon],
                                                "quality": _quality(result["reward_only"][horizon], real)}
        for pair in result["fixed_pairs"]:
            for horizon in ("h3", "h5"):
                pair["scores"][horizon] = {"scores": pair["scores"][horizon],
                                            "quality": _quality(pair["scores"][horizon], real)}
        output.append({"episode_id": anchor.episode_id, "start_step": anchor.step,
                       "geometry_seed": anchor.seed, "anchor_model_observation_sha256":
                       entry["anchor_model_observation_sha256"],
                       "candidate_action_sha256": {name: hashlib.sha256(actions[name].tobytes()).hexdigest()
                                                   for name in NAMES},
                       "real_h5_return": real, **result})
    branch._recheck(root, bundle, artifacts=True)
    branch.pinned(root, RESULT, RESULT_SHA)
    branch.pinned(root, RECEIPT, RECEIPT_SHA)
    reward = {h: _aggregate([row["reward_only"][h]["quality"] for row in output]) for h in ("h3", "h5")}
    if (reward["h3"]["pairs"]["concordant"] != 63
            or reward["h5"]["pairs"]["concordant"] != 56):
        raise ValueError("reward-only five-action H3/H5 baseline changed")
    pairs = [{"pair": list(pair), **{h: _aggregate([row["fixed_pairs"][index]["scores"][h]["quality"]
                                                  for row in output]) for h in ("h3", "h5")}}
             for index, pair in enumerate(PAIRS)]
    report = {"format": "haic-tdmpc2-h5-fixed-q-read-only-v1",
              "scope": "fixed_pair_observational_reused_TRAIN_not_original_stochastic_MPPI",
              "environment_resets": 0, "optimizer_steps": 0, "fresh_cells": False,
              "policy_benefit_established": False, "original_stochastic_MPPI_reproduced": False,
              "source": bundle["spec"]["source"], "source_sha256": bundle["spec"]["source_sha256"],
              "branch_protocol_sha256": PROTOCOL_SHA, "branch_result_path": RESULT,
              "branch_result_sha256": RESULT_SHA, "primary_receipt_path": RECEIPT,
              "primary_receipt_sha256": RECEIPT_SHA, "primary_body_sha256": receipt["body_sha256"],
              "operator_sha256": branch.digest(Path(__file__)),
              "bootstrap_rng": "CPU torch.fork_rng; common manual_seed(834 + episode_id*1000 + start_step + 100000) before each pair/horizon; same five policy draws per anchor",
              "summary": {"anchors": 12, "candidate_suffixes": 60, "actual_h5_pairs": 120,
                          "real_h5_ties": 29, "informative_real_h5_pairs": 91,
                          "reward_only": reward, "fixed_pairs": pairs,
                          "original_stochastic_planner_concordant_reference": {
                              h: original["rankings"][f"{h}_planner_vs_real_h5"]["pairs"]["concordant"]
                              for h in ("h3", "h5")}},
              "anchors": output}
    report["body_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"),
                                                     allow_nan=False).encode()).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha256", required=True, help="frozen H5 branch protocol SHA-256")
    args = parser.parse_args()
    if args.protocol_sha256 != PROTOCOL_SHA:
        parser.error("not the pinned original H5 branch protocol SHA-256")
    print(json.dumps(run(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
