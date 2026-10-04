"""DORMANT consumed-TRAIN CPU21 H5/Q0 full-episode evaluation.

No protocol is produced here. Freeze an independent SHA-bound protocol ONLY
after the disjoint real-branch producer has passed every declared gate.
Default preflight is read-only, without torch.load or environment construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import torch

from scripts import evaluate_tdmpc2_full_train as raw
from scripts import evaluate_tdmpc2_reward_overshoot_train as prior


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-overshoot-h5-q0-full-consumed-train-v1"
BRANCH_FORMAT = "haic-tdmpc2-overshoot-q0-real-branches-v1"
PLANNER = {**prior.PLANNER, "horizon": 5, "q_weight": 0.0}
SOURCE_SUMMARY = {"path": "experiments/tdmpc2-reward-overshoot-train-v1-result.json",
                  "sha256": "15b5e795d6e6c3e7127a7d804e41c428e9ad2904fdcec527ce27c7983da3a63e"}
PRIOR_EVALUATOR_SHA = "e28f50bf625c8093e04cb4dceb42633b1afc956849480cf4a990216c596fd0bd"
OVERSHOOT_CHECKPOINT_SHA = "51b1da8c4e2c2584524a3d51ccbd82daa1121479c123237d2e809cbde1bcdaa0"
H3_REFERENCES = {
    "fixed": prior.noise.BASELINE["full_eval_result"],
    "noise": {"path": "experiments/tdmpc2-h3-noise-full-consumed-train-v1-result.json",
              "sha256": "182a0426a9717fc55f5a191519163b59dc4a8674fac0af2c50997be0f82f15a5"},
}


def _ref(root: Path, value: object, prefix: str) -> tuple[dict, Path]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value["path"], str) or not value["path"].startswith(prefix)):
        raise ValueError("missing or invalid SHA-bound reference")
    return raw._reference(root, value, prefix)


def _branch(root: Path, refs: object, source: dict, pin: dict, runtime: dict) -> dict:
    """Verify a complete *new* disjoint-branch PASS, not a claimed PASS flag."""
    if not isinstance(refs, dict) or set(refs) != {"protocol", "receipt"}:
        raise ValueError("new disjoint branch protocol and receipt required")
    protocol, _ = _ref(root, refs["protocol"], "experiments/tdmpc2-overshoot-q0-branches-")
    receipt, _ = _ref(root, refs["receipt"], "runs/tdmpc2-overshoot-q0-branches-")
    if (protocol.get("format") != BRANCH_FORMAT
            or protocol.get("purpose") != "consumed-TRAIN-disjoint-RAW-replay-real-five-action-Q0-screen"
            or receipt.get("format") != "haic-tdmpc2-overshoot-q0-real-branches-result-v1"
            or receipt.get("status") != "PASS" or receipt.get("protocol") != refs["protocol"]
            or receipt.get("environment_resets_attempted") != 72
            or len(receipt.get("reset_intents", [])) != 72
            or receipt.get("replay_bound_before_first_reset") is not True
            or receipt.get("optimizer_updates") != 0
            or protocol.get("output") != refs["receipt"]["path"]
            or receipt.get("runtime") != protocol.get("runtime")
            or receipt.get("runtime") != {k: v for k, v in runtime.items() if k != "torch_cuda"}
            or receipt.get("score_source") != protocol.get("score_source")
            or receipt.get("old_primary") != protocol.get("old_primary")
            or protocol.get("max_resets") != 72 or protocol.get("policy_release") is not False
            or receipt.get("policy_release") is not False
            or protocol.get("anchors") != [{"episode_id": ep, "start_step": step}
                                          for ep in range(4, 8) for step in (16, 50, 100)]
            or protocol.get("episode_lengths") != [319, 314, 293, 303]
            or protocol.get("q_weights") != {"q0": 0, "q1": 1}
            or protocol.get("horizons") != [3, 5]
            or protocol.get("discount") != .995 or protocol.get("tie_tolerance") != 1e-6
            or protocol.get("bootstrap_offset") != 100000
            or protocol.get("environment") != {
                "track_id": 1, "geometry_seeds": list(raw.ROADS), "max_steps": 2000,
                "frame_skip": 4, "obstacles": True, "reward_shaping": False}
            or protocol.get("quality_gate") != {
                "full_h5_suffixes": 60, "min_informative_pairs": 40,
                "min_informative_roads": 3, "min_q0_concordance": .70,
                "min_q0_tied_best": 8, "min_q0_tied_best_roads": 2}):
        raise ValueError("absent, FAIL or partial disjoint branch receipt/runtime")
    body = {key: value for key, value in receipt.items() if key != "body_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    if receipt.get("body_sha256") != digest:
        raise ValueError("branch receipt body SHA mismatch")
    sources = protocol.get("source_sha256")
    if (not isinstance(sources, dict) or receipt.get("source_sha256") != sources
            or "scripts/diagnose_tdmpc2_overshoot_q0_branches.py" not in sources
            or "haic/algorithms/tdmpc2/q_bootstrap.py" not in sources
            or "tests/test_diagnose_tdmpc2_overshoot_q0_branches.py" not in sources):
        raise ValueError("new branch operator/planner source bytes not pinned")
    for name, expected in sources.items():
        if raw._digest(raw._path(root, name)) != raw._sha(expected):
            raise ValueError(f"branch source drift: {name}")
    old_score, _ = _ref(root, protocol.get("score_source"),
                        "experiments/tdmpc2-overshoot-old-branch-score-")
    old_primary, _ = _ref(root, protocol.get("old_primary"), "runs/tdmpc2-raw100k-h5-branches-")
    if (old_score.get("training_source", {}).get("checkpoint") != {
            "path": pin["path"], "sha256": pin["sha256"]}
            or old_score.get("training_source", {}).get("protocol") != prior.TRAIN_PROTOCOL
            or old_score.get("training_source", {}).get("result") != source["result"]
            or old_score.get("original_branch", {}).get("primary") != protocol["old_primary"]
            or old_primary.get("status") != "complete"
            or protocol["score_source"]["sha256"] != "44edd04cc7a579c3867ed35f5763645d0c92b2971fd979d5d141ea87f5802aa8"
            or protocol["old_primary"] != prior.BRANCH_PRIMARY):
        raise ValueError("branch is not bound to complete overshoot 100k source")
    for ref in (*old_score["training_source"].values(), *old_score["original_branch"].values()):
        if raw._digest(raw._path(root, ref["path"])) != raw._sha(ref["sha256"]):
            raise ValueError("branch source artifact drift")
    if any(sources.get(name) != value for name, value in old_score["source_sha256"].items()):
        raise ValueError("branch lost old-score source lineage")
    if set(sources) != set(old_score["source_sha256"]) | {
            "haic/algorithms/tdmpc2/q_bootstrap.py",
            "scripts/diagnose_tdmpc2_overshoot_q0_branches.py",
            "tests/test_diagnose_tdmpc2_overshoot_q0_branches.py"}:
        raise ValueError("branch includes unreviewed or missing executable sources")
    if protocol.get("fixed_q_pairs") != [[i, j] for i in range(5) for j in range(i + 1, 5)]:
        raise ValueError("branch missing ten fixed Q1 pairs")
    expected_intents = [{"episode_id": ep, "start_step": step, "candidate": candidate,
                         "ordinal": ordinal}
                        for ordinal, (ep, step, candidate) in enumerate((
                            (ep, step, candidate) for ep in range(4, 8) for step in (16, 50, 100)
                            for candidate in ("prefix_capture", "logged", "coast", "gas", "brake", "left_gas")), 1)]
    if receipt["reset_intents"] != expected_intents:
        raise ValueError("branch reset intents incomplete or duplicated")
    anchors = receipt.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 12:
        raise ValueError("branch requires twelve disjoint anchors")
    names = ("logged", "coast", "gas", "brake", "left_gas")
    labels = ("h3_q0", "h5_q0", *(f"h5_q1_{i}_{j}" for i in range(5) for j in range(i + 1, 5)))
    pairs = 0
    informative_roads = set()
    counts = {key: {"concordant": 0, "discordant": 0, "predicted_tie": 0, "real_tie": 0}
              for key in labels}
    regrets = {key: [] for key in labels}
    ties = {key: 0 for key in labels}
    tie_roads = {key: set() for key in labels}
    suffixes = 0
    for index, anchor in enumerate(anchors):
        road = raw.ROADS[index // 3]
        if (anchor.get("geometry_seed") != road or anchor.get("episode_id") != index // 3 + 4
                or anchor.get("start_step") != (16, 50, 100)[index % 3]
                or anchor.get("track_id") != 1
                or any(not isinstance(anchor.get(key), str) or not anchor[key] for key in (
                    "road_sha256", "anchor_accessible_state_sha256", "anchor_observation_sha256",
                    "anchor_model_observation_sha256"))):
            raise ValueError("branch anchor is not disjoint or prefix-parity checked")
        candidates = anchor.get("candidates")
        if (not isinstance(candidates, list) or len(candidates) != 5
                or [c.get("candidate") for c in candidates] != list(names)
                or not isinstance(anchor.get("candidate_action_bytes_hex"), dict)):
            raise ValueError("branch lacks five actual candidates")
        if any(c.get("steps") != 5 or c.get("full_h5") is not True
               or c.get("ranking_eligible") is not True or not isinstance(c.get("raw_rewards"), list)
               or len(c["raw_rewards"]) != 5
               or c.get("model_action_bytes_hex") != anchor["candidate_action_bytes_hex"].get(c["candidate"])
               or any(type(value) not in (float, int) or not math.isfinite(value)
                      for value in c["raw_rewards"])
               or not math.isclose(math.fsum(.995 ** t * v for t, v in enumerate(c["raw_rewards"])),
                                   c.get("real_discounted_raw_return", math.inf), abs_tol=1e-9, rel_tol=0)
               for c in candidates):
            raise ValueError("branch has partial suffix or nonfinite score")
        suffixes += len(candidates)
        scores = anchor.get("scores")
        if (not isinstance(scores, dict) or set(scores) != set(labels)
                or any(not isinstance(scores[label], dict) or tuple(scores[label]) != names
                       or any(type(value) not in (int, float) or not math.isfinite(value)
                              for value in scores[label].values()) for label in labels)):
            raise ValueError("branch missing matched Q0/Q1 candidate score")
        actual = [c["real_discounted_raw_return"] for c in candidates]
        top = max(actual)
        for key in labels:
            selected = max(range(5), key=lambda i: scores[key][names[i]])
            regrets[key].append(top - actual[selected])
            if top - actual[selected] <= 1e-6:
                ties[key] += 1
                tie_roads[key].add(road)
        for i in range(5):
            for j in range(i + 1, 5):
                delta = actual[i] - actual[j]
                if abs(delta) <= 1e-6:
                    for key in labels:
                        counts[key]["real_tie"] += 1
                    continue
                pairs += 1
                informative_roads.add(road)
                for key in labels:
                    predicted = scores[key][names[i]] - scores[key][names[j]]
                    verdict = ("predicted_tie" if abs(predicted) <= 1e-6 else
                               "concordant" if (predicted > 0) == (delta > 0) else "discordant")
                    counts[key][verdict] += 1
    mean = {key: math.fsum(values) / 12 for key, values in regrets.items()}
    q0 = counts["h5_q0"]["concordant"]
    passed = (suffixes == 60 and pairs >= 40 and len(informative_roads) >= 3
              and q0 / pairs >= .70 and all(q0 > counts[key]["concordant"] for key in labels[2:])
              and q0 >= counts["h3_q0"]["concordant"]
              and ties["h5_q0"] >= 8 and len(tie_roads["h5_q0"]) >= 2
              and all(mean["h5_q0"] < mean[key] for key in labels[2:])
              and mean["h5_q0"] <= mean["h3_q0"])
    totals = {key: {"pairs": counts[key], "informative_real_pairs": pairs,
                    "concordance": counts[key]["concordant"] / pairs if pairs else None,
                    "tied_best": ties[key], "tied_best_roads": len(tie_roads[key]),
                    "mean_real_h5_regret": mean[key]} for key in labels}
    metrics = {"full_h5_suffixes": suffixes, "informative_real_pairs": pairs,
               "informative_roads": len(informative_roads), "scores": totals,
               "quality_gate_passed": passed}
    if not passed or receipt.get("summary") != metrics:
        raise ValueError("new disjoint branch fails one or more predeclared Q0 gates")
    return {"receipt_sha256": refs["receipt"]["sha256"], "body_sha256": digest, **metrics}


def _check(path: Path, sha: str, *, root: Path, reserved: bool = False,
           allocated_bytes: int = 0) -> tuple[dict, dict]:
    path = raw._evaluation_path(root, path)
    if path.parent != root / "experiments" or raw._digest(path) != raw._sha(sha):
        raise ValueError("independent evaluation protocol SHA mismatch")
    p = raw._json(path.read_bytes())
    required = {"format", "purpose", "evaluation_source_sha256", "planner_source_sha256",
                "source", "source_summary", "branch_gate", "descriptive_references",
                "cells", "environment", "modes", "repeats", "seed", "max_steps",
                "planner", "runtime", "resources", "output_dir"}
    if (set(p) != required or p["format"] != FORMAT or p["purpose"] != "consumed-TRAIN-development"
            or p["cells"] != raw.CELLS or p["environment"] != raw.ENVIRONMENT
            or p["modes"] != ["mppi"] or p["repeats"] != 2 or p["seed"] != 20260928
            or p["max_steps"] != 2000 or p["planner"] != PLANNER
            or p["source_summary"] != SOURCE_SUMMARY
            or p["descriptive_references"] != H3_REFERENCES):
        raise ValueError("evaluation settings or descriptive references differ")
    if (raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_overshoot_h5_q0.py"))
            != raw._sha(p["evaluation_source_sha256"])
            or raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_reward_overshoot_train.py"))
            != PRIOR_EVALUATOR_SHA
            or raw._digest(raw._path(root, "haic/algorithms/tdmpc2/q_bootstrap.py"))
            != raw._sha(p["planner_source_sha256"])):
        raise ValueError("evaluation or Q0 planner executable source drift")
    output = raw._path(root, p["output_dir"], existing=False)
    if (output.parent != root / "runs" or not output.name.startswith("tdmpc2-overshoot-h5-q0-full-train-")
            or not output.parent.is_dir() or (not output.is_dir() if reserved else output.exists())
            or reserved and (output / "result.json").exists()):
        raise ValueError("output requires new exclusive no-resume runs directory")
    if torch.version.cuda is not None or str(torch.__version__) != "2.1.0+cpu":
        raise ValueError("isolated Torch 2.1.0+cpu required")
    runtime = prior._runtime()
    if p["runtime"] != runtime:
        raise ValueError("CPU21/native environment runtime drift")
    forecast = p["resources"]
    if (not isinstance(forecast, dict) or set(forecast) != prior.RESOURCE_KEYS
            or any(type(value) is not int or value <= 0 for value in forecast.values())
            or forecast["measured_peak_rss_bytes"] < 6462996480
            or forecast["additional_memory_bytes"] < 8589934592
            or forecast["remaining_disk_bytes"] < 67108864
            or forecast["max_wall_seconds"] > 21600):
        raise ValueError("measured CPU RSS and incremental resource forecast required")
    resources = prior._resources(output if reserved else output.parent, forecast,
                                 allocated_bytes=allocated_bytes)
    summary, _ = _ref(root, p["source_summary"], "experiments/")
    source_protocol, result, pin = prior._source(root, p["source"])
    if (summary.get("status") != "complete_100k_reused_train_model_internal_only"
            or summary.get("primary_result_sha256") != p["source"]["result"]["sha256"]
            or summary.get("selected_checkpoint_sha256") != pin["sha256"]
            or summary.get("training_ledger_sha256") != result["training_ledger_sha256"]
            or summary.get("step_ledger_sha256") != result["step_ledger_sha256"]
            or pin["sha256"] != OVERSHOOT_CHECKPOINT_SHA):
        raise ValueError("overshoot final checkpoint/result/complete ledgers disagree")
    baseline = prior._baseline(root)
    fixed, _ = _ref(root, p["descriptive_references"]["fixed"], "experiments/")
    noise, _ = _ref(root, p["descriptive_references"]["noise"], "experiments/")
    if (fixed.get("mppi", {}).get("finishes") != 0 or fixed.get("mppi", {}).get("episodes") != 8
            or noise.get("performance_evidence", {}).get("finishes") != 2
            or noise.get("performance_evidence", {}).get("completed_episodes") != 8
            or noise.get("performance_evidence", {}).get("distinct_finished_roads") != 2
            or noise.get("performance_evidence", {}).get("censored") != 0):
        raise ValueError("original H3 0/8 or H3 noise 2/8 references changed")
    gate = _branch(root, p["branch_gate"], p["source"], pin, runtime)
    return p, {"status": "preflight_only", "environment_resets": 0, "torch_load_calls": 0,
               "protocol_sha256": sha, "source_result_sha256": p["source"]["result"]["sha256"],
               "source_sha256": source_protocol["source_sha256"], "checkpoint": pin,
               "branch_gate": gate, "baseline": baseline,
               "descriptive_references": p["descriptive_references"],
               "resources": resources, "planner": PLANNER, "planned_episodes": 8,
               "generalization_claim": False, "official_score": False}


def preflight(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    return _check(protocol_path, protocol_sha256, root=root.resolve(strict=True))[1]


def _run_core(root: Path, p: dict, protocol_path: Path, sha: str, checked: dict,
              model, planner, *, env_factory, clock=time.perf_counter,
              initial_peak_rss_bytes: int | None = None) -> dict:
    """Synthetic-test seam; public execute has no injectable model/environment."""
    from haic.algorithms.tdmpc2.planner import PlannerConfig

    config = PlannerConfig(action_dim=3, discount=.995, episodic=True, horizon=5,
                           num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64)
    if (getattr(planner, "config", None) != config or getattr(planner, "q_weight", None) != 0
            or p.get("planner") != PLANNER or checked.get("branch_gate", {}).get("receipt_sha256")
            != p["branch_gate"]["receipt"]["sha256"]):
        raise ValueError("private core requires source-gated H5 QWeightedPlanner(q_weight=0)")
    output = raw._path(root, p["output_dir"], existing=False)
    output.mkdir(mode=0o700, exist_ok=False)
    ledger = output / "episodes.jsonl"
    rows = []
    intents = 0
    env = None
    started = clock()
    initial_peak = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    if initial_peak_rss_bytes is None else initial_peak_rss_bytes)

    def allocated():
        return max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial_peak)

    def budget():
        if clock() - started > p["resources"]["max_wall_seconds"]:
            raise TimeoutError("local H5 Q0 wall budget exceeded")
        prior._resources(output, p["resources"], allocated_bytes=allocated())

    try:
        raw._journal(ledger, {"event": "start", "protocol_sha256": sha,
                              "source_model_sha256": checked["checkpoint"]["sha256"],
                              "branch_receipt_sha256": checked["branch_gate"]["receipt_sha256"],
                              "resume_supported": False})
        for repeat in range(2):
            for index, cell in enumerate(raw.CELLS):
                budget()
                latest_p, latest = _check(protocol_path, sha, root=root, reserved=True,
                                          allocated_bytes=allocated())
                if latest_p != p or any(latest[key] != checked[key] for key in (
                        "checkpoint", "branch_gate", "source_result_sha256")):
                    raise ValueError("protocol/source/branch drift before reset")
                seed = 20260928 + repeat * 4 + index
                intents += 1
                raw._journal(ledger, {"event": "reset_intent", "target": 100000,
                                      "mode": "mppi", "repeat": repeat, "episode_seed": seed, **cell})
                if env is None:
                    env = env_factory(2000)
                budget()
                latest = _check(protocol_path, sha, root=root, reserved=True,
                                allocated_bytes=allocated())[1]
                if any(latest[key] != checked[key] for key in (
                        "checkpoint", "branch_gate", "source_result_sha256")):
                    raise ValueError("source/branch drift immediately before reset")
                row = prior._episode(env, model, planner, cell, repeat=repeat, seed=seed,
                                     budget=budget, clock=clock)
                raw._journal(ledger, row)
                rows.append(row)
        budget()
        latest = _check(protocol_path, sha, root=root, reserved=True,
                        allocated_bytes=allocated())[1]
        if latest["checkpoint"] != checked["checkpoint"] or latest["branch_gate"] != checked["branch_gate"]:
            raise ValueError("source/branch drift before final result")
        summary = prior._summary(rows, checked["checkpoint"])
        distinct = sum(road["finishes"] > 0 for road in summary["roads"])
        report = {"status": "complete", "scope": "reused_consumed_TRAIN_development_only",
                  "protocol_sha256": sha, "source_model_sha256": checked["checkpoint"]["sha256"],
                  "source_result_sha256": checked["source_result_sha256"],
                  "branch_gate": checked["branch_gate"], "planner": PLANNER,
                  "evaluation_reward": "raw_environment_only", "generalization_claim": False,
                  "official_score": False, "agent_act_compliance_claim": False,
                  "descriptive_references": checked["descriptive_references"],
                  "pairing": "same reused TRAIN roads/reset IDs, not action trajectories",
                  "episodes_sha256": raw._digest(ledger), "primary_mppi": summary,
                  "full_episode_finish_comparison_valid": summary["censored"] == 0,
                  "primary_local_gate": {"min_finishes": 4, "min_distinct_finished_roads": 2,
                                         "observed_finishes": summary["finishes"],
                                         "observed_distinct_finished_roads": distinct,
                                         "met_on_reused_train": summary["finishes"] >= 4 and distinct >= 2
                                         and summary["censored"] == 0},
                  "resources": p["resources"], "peak_process_rss_bytes": resource.getrusage(
                      resource.RUSAGE_SELF).ru_maxrss * 1024, "elapsed_seconds": clock() - started}
        if env is not None:
            env.close()
            env = None
        prior._write_json(output / "result.json", report)
        return report
    except BaseException as exc:
        try:
            raw._journal(ledger, {"event": "partial", "reason": type(exc).__name__,
                                  "complete_episodes": len(rows), "reset_intents": intents,
                                  "environment_resets": None if intents else 0,
                                  "resume_supported": False})
        except BaseException:
            try:
                prior._write_json(output / "failure.json", {
                    "event": "failure", "reason": type(exc).__name__, "protocol_sha256": sha,
                    "episode_ledger_sha256": raw._digest(ledger) if ledger.is_file() else None,
                    "complete_episodes": len(rows), "reset_intents": intents,
                    "environment_resets": None if intents else 0, "resume_supported": False})
            except BaseException as preserve_error:
                raise RuntimeError("cannot preserve partial Q0 exposure; status unknown") from preserve_error
        raise
    finally:
        if env is not None:
            env.close()


def execute(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    root = root.resolve(strict=True)
    p, checked = _check(protocol_path, protocol_sha256, root=root)
    if _check(protocol_path, protocol_sha256, root=root)[0] != p:
        raise ValueError("evaluation source drift before trusted checkpoint load")
    # Imports and pickle deserialization are strictly after the complete branch/source gate.
    from haic.algorithms.tdmpc2.haic_env import make_training_env
    from haic.algorithms.tdmpc2.planner import PlannerConfig
    from haic.algorithms.tdmpc2.q_bootstrap import QWeightedPlanner

    initial_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    source, _ = _ref(root, prior.TRAIN_PROTOCOL, "experiments/")
    model = prior._model(checked["checkpoint"], source, root)
    planner = QWeightedPlanner(model, PlannerConfig(action_dim=3, discount=.995, episodic=True,
                                                   horizon=5, num_samples=512, num_pi_trajs=24,
                                                   iterations=6, num_elites=64), q_weight=0)
    allocated = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - initial_peak)
    latest = _check(protocol_path, protocol_sha256, root=root, allocated_bytes=allocated)[1]
    if latest["checkpoint"] != checked["checkpoint"] or latest["branch_gate"] != checked["branch_gate"]:
        raise ValueError("source/branch drift after model load")
    return _run_core(root, p, protocol_path, protocol_sha256, checked, model, planner,
                     env_factory=make_training_env, initial_peak_rss_bytes=initial_peak)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-runtime", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.print_runtime:
        if args.protocol or args.protocol_sha256 or args.preflight or args.execute:
            parser.error("--print-runtime takes no protocol/execution flags")
        print(json.dumps(prior._runtime(), sort_keys=True, allow_nan=False))
        return
    if not args.protocol or not args.protocol_sha256:
        parser.error("--protocol and --protocol-sha256 are required")
    report = (execute(args.protocol, args.protocol_sha256) if args.execute else
              preflight(args.protocol, args.protocol_sha256))
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
