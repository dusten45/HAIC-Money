"""Exclusive, source-bound five-action real-branch Q0 screen on consumed TRAIN.

Default preflight is protocol/source/ledger only: no checkpoint load or reset.
Execution requires a separately frozen protocol SHA, an unused runs/ receipt,
and an explicit --execute. Neither mode trains, evaluates a policy or opens a
fresh, protected or official cell. Reconstructed accessible state does not
establish historical hidden Box2D solver-state identity.
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

from haic.algorithms.rlpd import prefix_parity as parity
from haic.algorithms.tdmpc2.haic_env import environment_action
from haic.algorithms.tdmpc2.planner import PlannerConfig
from haic.algorithms.tdmpc2.q_bootstrap import QWeightedPlanner
from scripts import diagnose_tdmpc2_h5_branches as branch
from scripts import diagnose_tdmpc2_h5_fixed_q as fixed
from scripts import score_tdmpc2_overshoot_archived_branches as score


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/tdmpc2-overshoot-q0-branches-v1.json"
SELF = "scripts/diagnose_tdmpc2_overshoot_q0_branches.py"
TEST = "tests/test_diagnose_tdmpc2_overshoot_q0_branches.py"
Q_MODULE = "haic/algorithms/tdmpc2/q_bootstrap.py"
SCORE_PROTOCOL_SHA = "44edd04cc7a579c3867ed35f5763645d0c92b2971fd979d5d141ea87f5802aa8"
ANCHORS = tuple((ep, step) for ep in range(4, 8) for step in (16, 50, 100))
LENGTHS = (319, 314, 293, 303)
NAMES = fixed.NAMES
PAIRS = tuple(combinations(range(5), 2))
BOOTSTRAP_OFFSET = 100_000
GATE = {"full_h5_suffixes": 60, "min_informative_pairs": 40,
        "min_informative_roads": 3, "min_q0_concordance": 0.70,
        "min_q0_tied_best": 8, "min_q0_tied_best_roads": 2}


def ref(path: str, sha: str) -> dict:
    return {"path": path, "sha256": sha}


def body_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def preflight(root: Path, protocol_sha256: str) -> dict:
    """Fail closed before checkpoint deserialization or environment construction."""
    root = Path(root).resolve(strict=True)
    spec = branch._json(branch.pinned(root, PROTOCOL, branch._sha(protocol_sha256)).read_bytes())
    if (set(spec) != {"format", "purpose", "score_source", "old_primary", "source_sha256",
                      "runtime", "anchors", "episode_lengths", "candidates", "environment",
                      "horizons", "discount", "tie_tolerance", "augmentation_seed",
                      "bootstrap_offset", "q_weights", "fixed_q_pairs", "max_resets",
                      "quality_gate", "output", "policy_release"}
            or spec["format"] != "haic-tdmpc2-overshoot-q0-real-branches-v1"
            or spec["purpose"] != "consumed-TRAIN-disjoint-RAW-replay-real-five-action-Q0-screen"
            or spec["score_source"] != ref(score.PROTOCOL, SCORE_PROTOCOL_SHA)
            or spec["old_primary"] != ref(score.OLD_PRIMARY, score.OLD_PRIMARY_SHA)
            or spec["anchors"] != [{"episode_id": ep, "start_step": step} for ep, step in ANCHORS]
            or spec["episode_lengths"] != list(LENGTHS)
            or spec["candidates"] != {"logged": None, **branch.FIXED_SUFFIXES}
            or spec["environment"] != branch.ENVIRONMENT
            or spec["horizons"] != [3, 5] or spec["discount"] != branch.DISCOUNT
            or spec["tie_tolerance"] != branch.TIE_TOLERANCE
            or type(spec["augmentation_seed"]) is not int or not 0 <= spec["augmentation_seed"] < 2**32
            or spec["bootstrap_offset"] != BOOTSTRAP_OFFSET
            or spec["q_weights"] != {"q0": 0, "q1": 1}
            or spec["fixed_q_pairs"] != [list(pair) for pair in PAIRS]
            or spec["max_resets"] != 72 or spec["quality_gate"] != GATE
            or spec["policy_release"] is not False
            or not isinstance(spec["output"], str)
            or not spec["output"].startswith("runs/tdmpc2-overshoot-q0-branches-")
            or not spec["output"].endswith(".json")
            or "/" in spec["output"][5:]):
        raise ValueError("not the fixed disjoint Q0 real-branch protocol")
    bound = score.preflight(root, SCORE_PROTOCOL_SHA)
    raw = branch.preflight(root, score.OLD_PROTOCOL_SHA)
    expected = {**bound["spec"]["source_sha256"],
                Q_MODULE: branch.digest(root / Q_MODULE), SELF: branch.digest(root / SELF),
                TEST: branch.digest(root / TEST)}
    if (spec["source_sha256"] != expected or spec["runtime"] != bound["spec"]["runtime"]
            or branch.runtime_identity() != spec["runtime"]
            or [raw["episodes"][ep]["length"] for ep in range(4, 8)] != list(LENGTHS)
            or set(ANCHORS) & set(branch.ANCHORS)):
        raise ValueError("Q0 source/runtime/episode lengths or old-anchor overlap differs")
    for name, sha in expected.items():
        branch.pinned(root, name, sha)
    return {"spec": spec, "score": bound, "raw": raw, "protocol_sha256": protocol_sha256}


def recheck(root: Path, bound: dict) -> None:
    spec = bound["spec"]
    branch.pinned(root, PROTOCOL, bound["protocol_sha256"])
    for name, sha in spec["source_sha256"].items():
        branch.pinned(root, name, sha)
    for refs in (bound["score"]["spec"]["training_source"],
                 bound["score"]["spec"]["original_branch"]):
        for item in refs.values():
            branch.pinned(root, item["path"], item["sha256"])
    branch._recheck(root, bound["raw"], artifacts=True)
    if branch.runtime_identity() != spec["runtime"]:
        raise ValueError("Q0 runtime changed")


def bind_anchors(state: dict, bound: dict, root: Path) -> tuple[branch.BoundAnchor, ...]:
    """Bind the ORIGINAL RAW100k replay and every RAW ledger step before reset."""
    raw = bound["raw"]
    replay = state.get("replay") if isinstance(state, dict) else None
    if (not isinstance(state, dict) or state.get("format") != raw["raw"]["format"]
            or state.get("protocol_sha256") != branch.RAW_PROTOCOL_SHA
            or state.get("source_sha256") != raw["raw"]["source_sha256"]
            or any(state.get(k) != v for k, v in branch.FINAL.items())
            or not isinstance(state.get("learner"), dict) or "q_scale" not in state["learner"]
            or not isinstance(replay, dict) or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or any(replay.get(k) != v for k, v in {
                "capacity": 120000, "horizon": 3, "action_dim": 3, "augmentation_pad": 3,
                "include_partial": False, "bootstrap_on_truncation": True,
                "observation_shape": branch.PIXELS, "next_episode_id": len(raw["episodes"]),
                "size": branch.FINAL["decisions"], "active": None}.items())
            or not isinstance(replay.get("episodes"), list)
            or len(replay["episodes"]) != len(raw["episodes"])):
        raise ValueError("not the complete ORIGINAL RAW100k checkpoint/replay")
    selected = {}
    decisions = 0
    with (root / f"{branch.RUN}/steps.jsonl").open("rb") as stream:
        for ep, (record, ledger) in enumerate(zip(replay["episodes"], raw["episodes"])):
            length = ledger["length"]
            actions, obs, rewards = (record.get(k) for k in ("actions", "observations", "rewards"))
            flags = [record.get(k) for k in ("terminated", "truncated", "terminal")]
            if (record.get("episode_id") != ep or record.get("start_step") != 0
                    or not 1 <= length <= 2000 or not isinstance(actions, np.ndarray)
                    or actions.shape != (length, 3) or actions.dtype != np.float32
                    or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)
                    or not isinstance(obs, np.ndarray) or obs.shape != (length + 1, *branch.PIXELS)
                    or obs.dtype != np.uint8 or not isinstance(rewards, np.ndarray)
                    or rewards.shape != (length,) or rewards.dtype != np.float32
                    or not np.isfinite(rewards).all()
                    or any(not isinstance(flag, np.ndarray) or flag.shape != (length,) or flag.dtype != np.bool_
                           for flag in flags)):
                raise ValueError("ORIGINAL RAW replay pixel/action/reward/flag shape differs")
            term, trunc, terminal = flags
            if (np.any(term[:-1] | trunc[:-1] | terminal[:-1])
                    or not (term[-1] or trunc[-1]) or np.any(term & ~terminal)
                    or np.any(terminal & ~(term | trunc))
                    or any(bool(flag[-1]) is not ledger[key] for flag, key in zip(
                        flags, ("terminated", "truncated", "terminal")))):
                raise ValueError("ORIGINAL RAW replay episode boundary differs")
            action_sha, native_sha, total = hashlib.sha256(), hashlib.sha256(), 0.0
            raw_rewards = []
            for offset, action in enumerate(actions):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("ORIGINAL RAW step ledger ends before complete replay")
                row = branch._json(line)
                native = environment_action(action)
                if (row.get("episode") != ep or row.get("decision") != decisions + 1
                        or row.get("track_id") != 1 or row.get("geometry_seed") != branch.ROADS[ep % 4]
                        or row.get("action_f32_hex") != action.tobytes().hex()
                        or row.get("native_action_f32_hex") != native.tobytes().hex()
                        or type(row.get("reward")) not in (int, float)
                        or not math.isfinite(row["reward"]) or np.float32(row["reward"]) != rewards[offset]
                        or any(type(row.get(key)) is not bool or row[key] != bool(flag[offset])
                               for flag, key in zip(flags, ("terminated", "truncated", "terminal")))):
                    raise ValueError(f"ORIGINAL RAW replay/road/action/reward/flag lineage differs at {ep}:{offset}")
                action_sha.update(action.tobytes())
                native_sha.update(native.tobytes())
                total += row["reward"]
                raw_rewards.append(row["reward"])
                decisions += 1
            if (action_sha.hexdigest() != ledger.get("action_trace_sha256")
                    or native_sha.hexdigest() != ledger.get("native_action_trace_sha256")
                    or not math.isclose(total, ledger.get("return", math.inf), rel_tol=0, abs_tol=1e-4)
                    or ledger.get("decisions") != decisions):
                raise ValueError("ORIGINAL RAW full-episode action/native/return lineage differs")
            if ep not in range(4, 8):
                continue
            if length != LENGTHS[ep - 4] or ledger["geometry_seed"] != branch.ROADS[ep % 4]:
                raise ValueError("disjoint ORIGINAL RAW anchor length/road differs")
            for anchor_ep, start in ANCHORS:
                if anchor_ep != ep:
                    continue
                if start + 5 > length or np.any(term[:start + 5] | trunc[:start + 5]):
                    raise ValueError("disjoint ORIGINAL RAW anchor lacks five logged decisions")
                selected[ep, start] = branch.BoundAnchor(
                    ep, 1, branch.ROADS[ep % 4], start, obs[:start + 1].copy(),
                    actions[:start].copy(), tuple(raw_rewards[:start]),
                    actions[start:start + 5].copy(), obs[start:start + 6].copy(),
                    tuple(raw_rewards[start:start + 5]),
                    tuple((bool(term[t]), bool(trunc[t]), bool(terminal[t])) for t in range(start, start + 5)))
        if stream.read(1) or decisions != branch.FINAL["decisions"] or set(selected) != set(ANCHORS):
            raise ValueError("ORIGINAL RAW complete replay/step cursor or disjoint anchors differ")
    return tuple(selected[key] for key in ANCHORS)


def model_scores(model: Any, pixels: np.ndarray, actions: dict[str, np.ndarray], seed: int) -> dict:
    """Use the actual weighted scorer, identical five action bytes and paired RNG."""
    if (tuple(actions) != NAMES or pixels.shape != branch.PIXELS or pixels.dtype != np.uint8
            or any(a.shape != (5, 3) or a.dtype != np.float32 or not np.isfinite(a).all()
                   or np.any(np.abs(a) > 1) for a in actions.values())):
        raise ValueError("invalid fixed action/pixel input")
    output = {label: {} for label in ("h3_q0", "h5_q0", *[f"h5_q1_{a}_{b}" for a, b in PAIRS])}
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        z = model.encode(torch.from_numpy(pixels.copy())[None], None)
        if z.ndim != 2 or z.shape[0] != 1 or not torch.isfinite(z).all():
            raise ValueError("invalid Q0 anchor encoding")
        for h in (3, 5):
            sequence = torch.from_numpy(np.stack([actions[name][:h] for name in NAMES], axis=1).copy())
            config = PlannerConfig(action_dim=3, discount=branch.DISCOUNT, horizon=h,
                                   episodic=True, num_samples=5, num_elites=1,
                                   num_pi_trajs=0, num_bins=model.cfg.num_bins)
            for pair in PAIRS:
                proxy = fixed._FixedPair(model, pair)
                bootstrap_seed = seed + BOOTSTRAP_OFFSET + h
                for label, weight in ((f"h{h}_q0", 0), (f"h5_q1_{pair[0]}_{pair[1]}", 1)):
                    if h == 3 and weight == 1:
                        continue
                    torch.manual_seed(bootstrap_seed)
                    values = QWeightedPlanner(proxy, config, q_weight=weight)._estimate_value(
                        z.repeat(5, 1), sequence)
                    if values.shape != (5, 1) or not torch.isfinite(values).all():
                        raise ValueError("nonfinite weighted planner candidate score")
                    item = dict(zip(NAMES, (float(v) for v in values[:, 0])))
                    if weight == 0:
                        if output[label] and any(not math.isclose(item[name], output[label][name],
                                                                  rel_tol=0, abs_tol=1e-6) for name in NAMES):
                            raise ValueError("Q0 changed across fixed head pairs")
                    output[label] = item
    return output


def aggregate(entries: list[dict]) -> dict:
    """One shared real-H5 denominator for H3/H5 Q0 and all ten H5 Q1 pairs."""
    labels = ("h3_q0", "h5_q0", *[f"h5_q1_{a}_{b}" for a, b in PAIRS])
    if (len(entries) != 12 or [(e.get("episode_id"), e.get("start_step")) for e in entries] != list(ANCHORS)):
        raise ValueError("incomplete/disordered Q0 anchor set")
    rows = {label: [] for label in labels}
    informative_roads = set()
    for entry in entries:
        candidates = entry["candidates"]
        if len(candidates) != 5 or [r.get("candidate") for r in candidates] != list(NAMES):
            raise ValueError("incomplete five-action real branches")
        real = {}
        for row in candidates:
            rewards = row.get("raw_rewards")
            if (row.get("steps") != 5 or row.get("full_h5") is not True
                    or row.get("ranking_eligible") is not True
                    or not isinstance(rewards, list) or len(rewards) != 5
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in rewards)):
                raise ValueError("all sixty real five-step suffixes required")
            value = math.fsum(branch.DISCOUNT**t * v for t, v in enumerate(rewards))
            if not math.isclose(value, row.get("real_discounted_raw_return", math.inf), abs_tol=1e-9, rel_tol=0):
                raise ValueError("real discounted return differs from independent suffix")
            real[row["candidate"]] = value
        scores = entry["scores"]
        if set(scores) != set(labels):
            raise ValueError("missing Q0/Q1 matched candidate scores")
        for label in labels:
            q = fixed._quality(scores[label], real)
            rows[label].append(q)
            if q["pairs"]["concordant"] + q["pairs"]["discordant"] + q["pairs"]["predicted_tie"]:
                informative_roads.add(entry["geometry_seed"])
    totals = {}
    for label, qualities in rows.items():
        pairs = {key: sum(q["pairs"][key] for q in qualities)
                 for key in ("concordant", "discordant", "predicted_tie", "real_tie")}
        informative = sum(pairs[key] for key in ("concordant", "discordant", "predicted_tie"))
        totals[label] = {"pairs": pairs, "informative_real_pairs": informative,
                         "concordance": pairs["concordant"] / informative if informative else None,
                         "tied_best": sum(q["best_real_h5_tie"] for q in qualities),
                         "tied_best_roads": len({entry["geometry_seed"] for entry, q in zip(entries, qualities)
                                                if q["best_real_h5_tie"]}),
                         "mean_real_h5_regret": math.fsum(q["real_h5_regret"] for q in qualities) / 12}
    q0, h3 = totals["h5_q0"], totals["h3_q0"]
    comparisons = [totals[label] for label in labels[2:]]
    informative = q0["informative_real_pairs"]
    if any(total["informative_real_pairs"] != informative or total["pairs"]["real_tie"] != q0["pairs"]["real_tie"]
           for total in totals.values()) or informative + q0["pairs"]["real_tie"] != 120:
        raise ValueError("matched real H5 pair denominators differ")
    passed = (informative >= GATE["min_informative_pairs"]
              and len(informative_roads) >= GATE["min_informative_roads"]
              and q0["concordance"] >= GATE["min_q0_concordance"]
              and all(q0["concordance"] > q["concordance"] for q in comparisons)
              and q0["concordance"] >= h3["concordance"]
              and q0["tied_best"] >= GATE["min_q0_tied_best"]
              and q0["tied_best_roads"] >= GATE["min_q0_tied_best_roads"]
              and all(q0["mean_real_h5_regret"] < q["mean_real_h5_regret"] for q in comparisons)
              and q0["mean_real_h5_regret"] <= h3["mean_real_h5_regret"])
    return {"full_h5_suffixes": 60, "informative_real_pairs": informative,
            "informative_roads": len(informative_roads), "scores": totals, "quality_gate_passed": passed}


def save(path: Path, report: dict, *, exclusive: bool = False) -> None:
    report["body_sha256"] = body_sha({k: v for k, v in report.items() if k != "body_sha256"})
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags, 0o644), "w", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(root: Path, protocol_sha256: str, *, execute: bool = False) -> dict:
    bound = preflight(root, protocol_sha256)
    spec = bound["spec"]
    if not execute:
        return {"status": "preflight_only", "environment_resets_attempted": 0,
                "optimizer_updates": 0, "protocol_sha256": protocol_sha256, "policy_release": False}
    root = Path(root).resolve(strict=True)
    output = root / spec["output"]
    if (output.parent != root / "runs" or output.parent.is_symlink() or not output.parent.is_dir()
            or output.exists() or output.is_symlink() or output.with_name(output.name + ".tmp").exists()):
        raise ValueError("requires unused exclusive direct runs/ receipt")
    report = {"format": "haic-tdmpc2-overshoot-q0-real-branches-result-v1", "status": "binding",
              "protocol": ref(PROTOCOL, protocol_sha256), "score_source": spec["score_source"],
              "old_primary": spec["old_primary"], "source_sha256": spec["source_sha256"],
              "runtime": spec["runtime"], "environment_resets_attempted": 0,
              "maximum_resets": 72, "optimizer_updates": 0, "policy_release": False,
              "fresh_or_official_score": False, "historical_hidden_box2d_state_proven": False,
              "exact_resume_supported": False, "anchors": [], "reset_intents": []}
    save(output, report, exclusive=True)
    try:
        # score._models verifies overshoot and RAW producer, ledger, optimizer,
        # model and replay lineage; RAW replay alone supplies the branch anchors.
        original_model, model = score._models(root, bound["score"])
        del original_model
        recheck(root, bound)
        checkpoint = bound["score"]["spec"]["original_branch"]["model"]
        with (root / checkpoint["path"]).open("rb") as stream:
            if branch.digest_stream(stream) != checkpoint["sha256"]:
                raise ValueError("ORIGINAL RAW checkpoint changed before replay deserialization")
            stream.seek(0)
            state = torch.load(stream, map_location="cpu", weights_only=False)
        anchors = bind_anchors(state, bound, root)
        del state
        report["replay_bound_before_first_reset"] = True
        for anchor in anchors:
            candidates = branch.candidate_actions(anchor)
            report["anchors"].append({"episode_id": anchor.episode_id, "track_id": 1,
                "geometry_seed": anchor.seed, "start_step": anchor.step,
                "anchor_model_observation_hex": anchor.observations[-1].tobytes().hex(),
                "anchor_model_observation_sha256": hashlib.sha256(anchor.observations[-1].tobytes()).hexdigest(),
                "candidate_action_bytes_hex": {name: a.tobytes().hex() for name, a in candidates.items()},
                "candidates": []})
        save(output, report)
        from train import build_env

        report["status"] = "running"
        for index, anchor in enumerate(anchors):
            entry = report["anchors"][index]
            candidates = branch.candidate_actions(anchor)
            captured = None
            for candidate in (None, *NAMES):
                recheck(root, bound)
                if report["environment_resets_attempted"] >= spec["max_resets"]:
                    raise ValueError("Q0 reset cap exceeded")
                # Durable intent is recorded BEFORE constructing/resetting each environment.
                report["reset_intents"].append({"episode_id": anchor.episode_id,
                    "start_step": anchor.step, "candidate": candidate or "prefix_capture",
                    "ordinal": report["environment_resets_attempted"] + 1})
                report["environment_resets_attempted"] += 1
                save(output, report)
                env = build_env(1, anchor.seed, 2000, 4, reward_shaping=False, obstacles=True)
                try:
                    if candidate is None:
                        captured = branch.capture_bound_prefix(env, anchor)
                        entry["road_sha256"] = captured.road_sha256
                        entry["anchor_accessible_state_sha256"] = hashlib.sha256(
                            repr(captured.steps[-1].state).encode()).hexdigest()
                        entry["anchor_observation_sha256"] = captured.steps[-1].observation_sha256
                    else:
                        if captured is None:
                            raise ValueError("missing captured accessible prefix")
                        observation = parity._compare_replay_prefix(env, captured)
                        branch._pixel(observation, anchor.observations[-1], "branch_anchor")
                        row = branch.execute_suffix(env, anchor, candidate, candidates[candidate], captured.road_sha256)
                        entry["candidates"].append(row)
                finally:
                    env.close()
                save(output, report)
            entry["scores"] = model_scores(model, anchor.observations[-1], candidates,
                                             spec["augmentation_seed"] + anchor.episode_id * 1000 + anchor.step)
            save(output, report)
        recheck(root, bound)
        report["summary"] = aggregate(report["anchors"])
        report["status"] = "PASS" if report["summary"]["quality_gate_passed"] else "FAIL"
    except BaseException as exc:
        report["status"] = "stopped_partial_no_resume"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save(output, report)
        raise
    save(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--print-runtime", action="store_true")
    args = parser.parse_args()
    if args.print_runtime:
        if args.protocol_sha256 is not None:
            parser.error("--print-runtime takes no protocol")
        print(json.dumps(branch.runtime_identity(), sort_keys=True))
        return
    if args.protocol_sha256 is None:
        parser.error("--protocol-sha256 required")
    print(json.dumps(run(ROOT, args.protocol_sha256, execute=args.execute), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
