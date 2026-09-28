"""Read-only H3 reward-ranking comparison on the frozen v2 TRAIN branches.

No environment import, construction, reset or optimizer step; an optional
exclusive runs/ output receipt can preserve the read-only score. A
checkpoint must have a complete episode-boundary training row, matching step
prefix and SHA before torch.load (which requires trusted local pickle files).
The 40 informative pairs share anchors/candidates on four reused TRAIN roads;
this is a dependent development proxy, not survival or generalization proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BRANCH_PROTOCOL = "experiments/tdmpc2-v2-prefix-branches-v1.json"
BRANCH_PROTOCOL_SHA256 = "d6b32770b250ff764e944839d910cc3e29eb77ad7626ab7b37af4bed7d96bf93"
BRANCH_RECEIPT = "runs/tdmpc2-prefix-branches-20260928-v1.json"
BRANCH_RECEIPT_SHA256 = "b6dfc5901e9f2425d1af27ec9c95906d2d5c736ebf1475a01bc4e32688339414"
BRANCH_SUMMARY = "experiments/tdmpc2-v2-prefix-branches-v1-result.json"
BRANCH_SUMMARY_SHA256 = "55fc15b7d6b8951cfc77b051b7719520289a4338d3431783a552c84aa64c343a"
LONG_PROTOCOL = "experiments/tdmpc2-long-reused-train-v2.json"
LONG_PROTOCOL_SHA256 = "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec"
CURSOR_SOURCE = "scripts/diagnose_tdmpc2_checkpoint_losses.py"
CURSOR_SOURCE_SHA256 = "ef6552479192e2e113951edb3bfbf7a4c87b3a508dd884c45ce6a270a0aebe6f"
RUN = "runs/tdmpc2-long-20260928-v2"
TARGETS = (20000, 40000, 70000, 100000)
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
ANCHORS = tuple((episode, step) for episode in range(4) for step in (16, 50, 100))
NAMES = ("logged", "coast", "gas", "brake", "left_gas")
TOLERANCE = 1e-6
DISCOUNT = .995
PIXELS = (4, 64, 64)
FIXED_ACTIONS = {"coast": (0, -1, -1), "gas": (0, 1, -1),
                 "brake": (0, -1, 1), "left_gas": (-.5, 1, -1)}
PAIR_KEYS = ("concordant", "discordant", "real_tie", "predicted_tie", "terminal_or_short_excluded")
BASELINE_PAIRS = {"concordant": 25, "discordant": 15, "real_tie": 80,
                  "predicted_tie": 0, "terminal_or_short_excluded": 0}


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
        raise ValueError("invalid lowercase SHA-256")
    return value


def _file(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or name.startswith("/")
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("noncanonical artifact/source path")
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in artifact/source path: {name}")
    if not path.is_file():
        raise ValueError(f"missing artifact/source: {name}")
    return path


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _pinned_json(root: Path, name: str, expected: str) -> dict:
    path = _file(root, name)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"frozen artifact SHA mismatch: {name}")
    if not data or len(data) > 2 * 1024 * 1024:
        raise ValueError(f"invalid frozen JSON size: {name}")
    value = json.loads(data, object_pairs_hook=_unique,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError("frozen JSON must be an object")
    return value


def _finite(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("nonfinite or nonnumeric branch value")
    return float(value)


def _bytes(value: object, size: int) -> bytes:
    if (not isinstance(value, str) or len(value) != 2 * size
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("missing/noncanonical fixed observation or action bytes")
    return bytes.fromhex(value)


def _pairs(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(PAIR_KEYS, 0)
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            actual = left["real_discounted_raw_return"] - right["real_discounted_raw_return"]
            predicted = left["predicted_reward_return"] - right["predicted_reward_return"]
            if abs(actual) <= TOLERANCE:
                counts["real_tie"] += 1
            elif abs(predicted) <= TOLERANCE:
                counts["predicted_tie"] += 1
            else:
                counts["concordant" if (actual > 0) == (predicted > 0) else "discordant"] += 1
    return counts


def _validate_branch(protocol: dict, receipt: dict, summary: dict) -> list[dict]:
    if (protocol.get("format") != "haic-tdmpc2-v2-prefix-branches-v1"
            or protocol.get("purpose") != "consumed-TRAIN-same-state-diagnostic"
            or protocol.get("anchors") != [{"episode_id": ep, "start_step": step} for ep, step in ANCHORS]
            or protocol.get("candidates") != {"logged": None, **{
                name: [list(action)] * 3 for name, action in FIXED_ACTIONS.items()}}
            or protocol.get("environment") != {"track_id": 1, "geometry_seeds": list(ROADS),
                                                    "max_steps": 2000, "frame_skip": 4,
                                                    "obstacles": True, "reward_shaping": False}
            or protocol.get("horizon") != 3 or protocol.get("discount") != DISCOUNT
            or protocol.get("tie_tolerance") != TOLERANCE
            or type(protocol.get("augmentation_seed")) is not int
            or not 0 <= protocol["augmentation_seed"] < 2**32 - 3100
            or protocol.get("max_resets") != 72
            or not isinstance(protocol.get("source_sha256"), dict)
            or not isinstance(protocol.get("v2_sha256"), dict)
            or not isinstance(protocol.get("runtime"), dict)):
        raise ValueError("not the frozen H3 branch protocol")
    if (set(receipt) != {"aggregate_pair_counts", "anchor_count", "anchors", "candidate_count_per_anchor",
                         "concordance_on_comparable_pairs", "discount", "environment_resets_attempted",
                         "format", "historical_road_or_hidden_box2d_state_proven", "horizon",
                         "official_or_protected_result", "protocol_sha256", "runtime",
                         "same_state_counterfactual_ranking", "scope", "source_sha256", "status",
                         "tie_tolerance", "v2_sha256"}
            or receipt["format"] != "haic-tdmpc2-v2-prefix-branches-result-v1"
            or receipt["status"] != "complete" or receipt["protocol_sha256"] != BRANCH_PROTOCOL_SHA256
            or receipt["source_sha256"] != protocol["source_sha256"]
            or receipt["v2_sha256"] != protocol["v2_sha256"]
            or receipt["runtime"] != protocol["runtime"]
            or receipt["scope"] != "reused_consumed_TRAIN_same_reconstructed_accessible_state_only"
            or receipt["same_state_counterfactual_ranking"] is not True
            or receipt["historical_road_or_hidden_box2d_state_proven"] is not False
            or receipt["official_or_protected_result"] is not False
            or receipt["horizon"] != 3 or receipt["discount"] != DISCOUNT
            or receipt["tie_tolerance"] != TOLERANCE
            or type(receipt["anchor_count"]) is not int or receipt["anchor_count"] != 12
            or type(receipt["candidate_count_per_anchor"]) is not int
            or receipt["candidate_count_per_anchor"] != 5
            or type(receipt["environment_resets_attempted"]) is not int
            or receipt["environment_resets_attempted"] != 72
            or not isinstance(receipt["anchors"], list) or len(receipt["anchors"]) != 12):
        raise ValueError("incomplete or altered 72-reset branch receipt")
    if (summary.get("status") != "complete" or summary.get("protocol_sha256") != BRANCH_PROTOCOL_SHA256
            or summary.get("primary_receipt") != BRANCH_RECEIPT
            or summary.get("primary_receipt_sha256") != BRANCH_RECEIPT_SHA256
            or any(summary.get(k) != v for k, v in {
                "predeclared_anchors": 12, "completed_anchors": 12, "candidate_sequences": 60,
                "full_h3_candidates": 60, "execution_environment_resets_attempted": 72,
                "real_return_ties_at_1e-6": 80, "concordant_pairs": 25,
                "discordant_pairs": 15, "informative_pairs": 40, "informative_anchors": 7,
                "terminal_or_short_excluded_candidates": 0,
            }.items())):
        raise ValueError("frozen summary differs from complete primary branch receipt")

    baseline = dict.fromkeys(PAIR_KEYS, 0)
    informative_anchors = 0
    roads = {}
    for (episode, step), anchor in zip(ANCHORS, receipt["anchors"]):
        if (not isinstance(anchor, dict) or set(anchor) != {
                "anchor_accessible_state_sha256", "anchor_model_observation_hex",
                "anchor_model_observation_sha256", "anchor_observation_sha256", "candidates",
                "episode_id", "geometry_seed", "ranking", "road_sha256", "start_step", "track_id"}
                or type(anchor["episode_id"]) is not int or anchor["episode_id"] != episode
                or type(anchor["start_step"]) is not int or anchor["start_step"] != step
                or type(anchor["track_id"]) is not int or anchor["track_id"] != 1
                or type(anchor["geometry_seed"]) is not int or anchor["geometry_seed"] != ROADS[episode]
                or not isinstance(anchor["candidates"], list) or len(anchor["candidates"]) != 5):
            raise ValueError("branch anchor identity/count differs from frozen protocol")
        pixels = _bytes(anchor["anchor_model_observation_hex"], math.prod(PIXELS))
        if hashlib.sha256(pixels).hexdigest() != _sha(anchor["anchor_model_observation_sha256"]):
            raise ValueError("anchor model observation bytes/SHA mismatch")
        _sha(anchor["anchor_observation_sha256"])
        _sha(anchor["anchor_accessible_state_sha256"])
        road = _sha(anchor["road_sha256"])
        if episode in roads and roads[episode] != road:
            raise ValueError("road signature changed between anchors on the same episode")
        roads[episode] = road
        rows = []
        for name, candidate in zip(NAMES, anchor["candidates"]):
            if (not isinstance(candidate, dict) or set(candidate) != {
                    "actions_model", "candidate", "full_h3", "model_action_bytes_hex",
                    "predicted_reward_return", "ranking_eligible", "raw_rewards",
                    "real_discounted_raw_return", "steps", "terminal_excluded",
                    "terminated", "truncated"}
                    or candidate["candidate"] != name or type(candidate["steps"]) is not int
                    or candidate["steps"] != 3 or candidate["full_h3"] is not True
                    or candidate["ranking_eligible"] is not True
                    or any(candidate[k] is not False for k in ("terminal_excluded", "terminated", "truncated"))
                    or not isinstance(candidate["raw_rewards"], list)
                    or len(candidate["raw_rewards"]) != 3):
                raise ValueError("branch candidate is missing, terminal, short, or reordered")
            actions = np.frombuffer(_bytes(candidate["model_action_bytes_hex"], 3 * 3 * 4), dtype="<f4")
            if (not isinstance(candidate["actions_model"], list)
                    or len(candidate["actions_model"]) != 3
                    or any(not isinstance(row, list) or len(row) != 3
                           or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
                           for row in candidate["actions_model"])
                    or not np.array_equal(np.asarray(candidate["actions_model"], dtype=np.float32).ravel(), actions)
                    or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)
                    or name != "logged" and not np.array_equal(
                        actions.reshape(3, 3), np.asarray([FIXED_ACTIONS[name]] * 3, dtype=np.float32))):
                raise ValueError("candidate actions differ from exact 3D float32 bytes")
            rewards = [_finite(value) for value in candidate["raw_rewards"]]
            real = _finite(candidate["real_discounted_raw_return"])
            predicted = _finite(candidate["predicted_reward_return"])
            if not math.isclose(real, math.fsum(DISCOUNT**t * r for t, r in enumerate(rewards)),
                                rel_tol=0, abs_tol=1e-10):
                raise ValueError("branch real H3 return differs from discounted raw rewards")
            rows.append({"candidate": name, "actions": actions.reshape(3, 3).copy(),
                         "real_discounted_raw_return": real, "predicted_reward_return": predicted})
        pairs = _pairs(rows)
        ranking = anchor["ranking"]
        comparable = pairs["concordant"] + pairs["discordant"]
        if (not isinstance(ranking, dict) or ranking.get("pairs") != pairs
                or ranking.get("comparable_pairs") != comparable
                or ranking.get("ranking_identifiable") is not (comparable > 0)
                or any(ranking.get(k) != 5 for k in ("full_h3_candidates", "ranking_eligible_candidates"))
                or ranking.get("terminal_candidates") != 0
                or (ranking.get("concordance") is not None if comparable == 0 else
                    not math.isclose(_finite(ranking.get("concordance")), pairs["concordant"] / comparable,
                                     rel_tol=0, abs_tol=1e-12))):
            raise ValueError("stored per-anchor baseline pair counts differ from raw branches")
        informative_anchors += comparable > 0
        for key in PAIR_KEYS:
            baseline[key] += pairs[key]
        anchor["_decoded"] = (np.frombuffer(pixels, dtype=np.uint8).reshape(PIXELS).copy(), rows)
    if (baseline != BASELINE_PAIRS or informative_anchors != 7
            or receipt["aggregate_pair_counts"] != baseline
            or not math.isclose(_finite(receipt["concordance_on_comparable_pairs"]), .625,
                                rel_tol=0, abs_tol=1e-12)):
        raise ValueError("v2 25/15/80 baseline pair counts cannot be reproduced")
    return receipt["anchors"]


def _bind_long(root: Path, targets: tuple[int, ...]) -> tuple[dict, list[dict]]:
    # Imported helper is *itself* pinned before use, not part of the running trainer.
    if _digest(_file(root, CURSOR_SOURCE)) != CURSOR_SOURCE_SHA256:
        raise ValueError("checkpoint cursor executable SHA mismatch")
    from scripts import diagnose_tdmpc2_checkpoint_losses as cursor

    if (cursor.PROTOCOL != LONG_PROTOCOL or cursor.PROTOCOL_SHA256 != LONG_PROTOCOL_SHA256
            or cursor.RUN != RUN or cursor.TARGETS != TARGETS):
        raise ValueError("checkpoint cursor is not pinned to long retry v2")
    protocol = cursor._protocol(root)
    if protocol.get("source_sha256", {}).get("scripts/train_tdmpc2_long.py") != (
            "9121fee37b5beeeaf7f22f253ab0500896e514fa7e5a140a8257483116df9849"):
        raise ValueError("long retry runner source differs")
    pins = [cursor._cursor(root, protocol, target) for target in targets]
    return protocol, pins


def _predictions(model, pixels: np.ndarray, rows: list[dict], seed: int) -> list[dict]:
    from haic.algorithms.tdmpc2.model import two_hot_inv

    model.eval()
    scored = []
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        z0 = model.encode(torch.from_numpy(pixels.copy())[None], None)
        if not isinstance(z0, torch.Tensor) or z0.ndim != 2 or z0.shape[0] != 1 or not torch.isfinite(z0).all():
            raise ValueError("nonfinite/malformed model anchor encoding")
        for row in rows:
            z = z0
            predicted_rewards = []
            for action in row["actions"]:
                tensor = torch.from_numpy(action.copy())[None]
                reward = two_hot_inv(model.reward(z, tensor, None), model.cfg)
                if reward.shape != (1, 1) or not torch.isfinite(reward).all():
                    raise ValueError("nonfinite/malformed reward head")
                predicted_rewards.append(float(reward.item()))
                z = model.next(z, tensor, None)
                if not isinstance(z, torch.Tensor) or z.shape != z0.shape or not torch.isfinite(z).all():
                    raise ValueError("nonfinite/malformed model dynamics")
            value = math.fsum(DISCOUNT**t * r for t, r in enumerate(predicted_rewards))
            if not math.isfinite(value):
                raise ValueError("nonfinite predicted H3 reward-only return")
            scored.append({"candidate": row["candidate"], "real_discounted_raw_return": row["real_discounted_raw_return"],
                           "predicted_reward_return": value})
    return scored


def _checkpoint_model(pin: dict, protocol: dict):
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel

    # Hash and deserialize the SAME open file, not a path reopened after hashing.
    with pin["checkpoint"].open("rb") as stream:
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
        if sha.hexdigest() != pin["checkpoint_sha256"]:
            raise ValueError("sealed checkpoint changed before torch.load")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=False)
    row = pin["row"]
    replay = state.get("replay") if isinstance(state, dict) else None
    learner = state.get("learner") if isinstance(state, dict) else None
    if (not isinstance(state, dict) or set(state) != {
            "format", "protocol_sha256", "source_sha256", "target", "decisions", "updates", "episodes",
            "action_dim", "learner", "optim", "pi_optim", "replay", "probe", "rng", "resume_supported"}
            or state["format"] != protocol.get("format")
            or state["protocol_sha256"] != LONG_PROTOCOL_SHA256
            or state["source_sha256"] != protocol["source_sha256"]
            or state["resume_supported"] is not False or state["action_dim"] != 3
            or any(state[k] != row[k] for k in ("target", "decisions", "updates", "episodes"))
            or not isinstance(replay, dict) or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or replay.get("active") is not None
            or replay.get("size") != row["decisions"] or replay.get("next_episode_id") != row["episodes"]
            or not isinstance(replay.get("episodes"), list) or len(replay["episodes"]) != row["episodes"]
            or replay.get("capacity") != 120000 or replay.get("horizon") != 3
            or replay.get("action_dim") != 3 or replay.get("observation_shape") != PIXELS
            or not isinstance(learner, dict) or "q_scale" not in learner
            or any(not isinstance(k, str) or k != "q_scale" and not k.startswith("model.") for k in learner)):
        raise ValueError("checkpoint contents do not match sealed 3D whole-episode ledger")
    model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": PIXELS}, episodic=True))
    model.load_state_dict({key[6:]: value for key, value in learner.items() if key.startswith("model.")}, strict=True)
    return model.to("cpu").eval()


def score(*, root: Path = ROOT, targets: tuple[int, ...] = TARGETS) -> dict:
    if (not targets or len(set(targets)) != len(targets) or tuple(sorted(targets)) != targets
            or any(type(t) is not int or t not in TARGETS for t in targets)):
        raise ValueError("select a nonempty ascending subset of 20k/40k/70k/100k")
    root = root.resolve(strict=True)
    branch = _pinned_json(root, BRANCH_PROTOCOL, BRANCH_PROTOCOL_SHA256)
    receipt = _pinned_json(root, BRANCH_RECEIPT, BRANCH_RECEIPT_SHA256)
    summary = _pinned_json(root, BRANCH_SUMMARY, BRANCH_SUMMARY_SHA256)
    anchors = _validate_branch(branch, receipt, summary)
    for name, sha in branch["source_sha256"].items():
        if _digest(_file(root, name)) != _sha(sha):
            raise ValueError(f"branch source SHA mismatch: {name}")
    # Bind ALL selected sources, ledger prefixes and checkpoint bytes before ANY torch.load.
    protocol, pins = _bind_long(root, targets)
    reports = []
    for pin in pins:
        model = _checkpoint_model(pin, protocol)
        per_anchor = []
        counts = dict.fromkeys(PAIR_KEYS, 0)
        for anchor in anchors:
            pixels, rows = anchor["_decoded"]
            seed = branch["augmentation_seed"] + anchor["episode_id"] * 1000 + anchor["start_step"]
            scored = _predictions(model, pixels, rows, seed)
            pairs = _pairs(scored)
            if pairs["real_tie"] != anchor["ranking"]["pairs"]["real_tie"]:
                raise ValueError("checkpoint did not use the same real-return pairs")
            for key in PAIR_KEYS:
                counts[key] += pairs[key]
            per_anchor.append({"episode_id": anchor["episode_id"], "track_id": anchor["track_id"],
                               "geometry_seed": anchor["geometry_seed"], "start_step": anchor["start_step"],
                               "anchor_model_observation_sha256": anchor["anchor_model_observation_sha256"],
                               "augmentation_seed": seed, "pairs": pairs, "candidates": scored})
        if (counts["real_tie"] != 80 or counts["terminal_or_short_excluded"] != 0
                or sum(counts.values()) != 120):
            raise ValueError("checkpoint comparison changed the fixed 120 real pairs")
        comparable = counts["concordant"] + counts["discordant"]
        reports.append({"target": pin["row"]["target"], "decisions": pin["row"]["decisions"],
                        "updates": pin["row"]["updates"], "episodes": pin["row"]["episodes"],
                        "checkpoint_path": pin["checkpoint"].relative_to(root).as_posix(),
                        "checkpoint_sha256": pin["checkpoint_sha256"],
                        "training_ledger_line": pin["line"],
                        "training_ledger_prefix_sha256": pin["ledger_prefix_sha256"],
                        "step_ledger_prefix_sha256": pin["step_prefix_sha256"],
                        "pairs": counts, "concordant_on_fixed_informative_pairs": {
                            "numerator": counts["concordant"], "denominator": 40},
                        "concordant_on_nonpredicted_ties": {"numerator": counts["concordant"],
                                                           "denominator": comparable},
                        "delta_concordant_on_same_40_vs_v2": counts["concordant"] - 25,
                        "anchors": per_anchor})
        del model
    return {"scope": "read_only_same_frozen_anchor_reward_ranking_reused_TRAIN_proxy",
            "environment_resets": 0, "optimizer_steps": 0, "generalization_or_survival_proven": False,
            "dependent_pairs_shared_across_four_consumed_roads": True,
            "historical_hidden_box2d_state_proven": False,
            "model_scoring": "same exact anchor uint8 pixels and 3D float32 action bytes; H3 reward only, no Q bootstrap; per-anchor shared seeded CPU augmentation",
            "branch_protocol": BRANCH_PROTOCOL, "branch_protocol_sha256": BRANCH_PROTOCOL_SHA256,
            "branch_receipt": BRANCH_RECEIPT, "branch_receipt_sha256": BRANCH_RECEIPT_SHA256,
            "long_protocol": LONG_PROTOCOL, "long_protocol_sha256": LONG_PROTOCOL_SHA256,
            "source_sha256": protocol["source_sha256"], "cursor_source_sha256": CURSOR_SOURCE_SHA256,
            "baseline_v2": {"pairs": BASELINE_PAIRS, "concordant_on_fixed_informative_pairs": {
                "numerator": 25, "denominator": 40}, "informative_anchors": 7},
            "checkpoints": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", type=int, choices=TARGETS, default=list(TARGETS),
                        help="ascending sealed checkpoint subset; default all four")
    parser.add_argument("--output", type=Path, help="exclusive runs/ receipt for this target subset")
    args = parser.parse_args()
    output = args.output.absolute() if args.output is not None else None
    if output is not None:
        name = f"tdmpc2-long-v2-ranking-{'-'.join(map(str, args.targets))}.json"
        if output != ROOT / "runs" / name or output.parent.is_symlink():
            parser.error("output must be the exact source-bound direct runs/ receipt")
    payload = json.dumps(score(targets=tuple(args.targets)), sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(payload, end="")
        return
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                      "targets": args.targets}, sort_keys=True))


if __name__ == "__main__":
    main()
