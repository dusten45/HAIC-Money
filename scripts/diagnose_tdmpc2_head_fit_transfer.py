"""Frozen reward-head fit/transfer audit; no environment or optimizer operations.

Preflight prints a deterministic protocol proposal without torch.load. Scoring
requires a separately frozen protocol SHA and an exclusive output file.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts import adapt_tdmpc2_reward_head as head


ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/diagnose_tdmpc2_head_fit_transfer.py"
TEST = "tests/test_diagnose_tdmpc2_head_fit_transfer.py"
PROTOCOL = "experiments/tdmpc2-head-fit-transfer-v1.json"
OUTPUT = "runs/tdmpc2-head-fit-transfer-20260929-v1.json"
PILOT = ("runs/tdmpc2-head-only-20260929-v2/result.json",
         "72b4f5019ca21f7c86dd5c8527a3af0739fc5349f8d88aa2637b4a368b59da4f")
PILOT_PROTOCOL_SHA = "d2e7c833671c1e16e529172ed7b38ffc2be9a107a2c57242334277fac67a206b"
ADAPTED = ("runs/tdmpc2-head-only-20260929-v2/adapted-model.pt",
           "c8ba10ee4a1287bc3d26b76e645e66b9b020c74c749d060cc3fcd567fb611510")
PHASES = ("fit_planned", "excluded_random", "excluded_early_planned")
SIGNS = ("positive", "nonpositive")
SEED = 20260929
BATCH = 256
GATE = {"scope": "natural_fit_RAW_episode44_through306",
        "positive_mae_improvement_at_least": 0.10,
        "positive_absolute_signed_bias_improvement_at_least": 0.10,
        "minimum_qualifying_roads": 3,
        "nonpositive_mae_worsening_at_most": 0.05,
        "failure_interpretation": "reject_transfer_only_explanation_choose_objective_head_discrimination_next",
        "policy_gate": False}
EXPECTED_COUNTS = {
    "fit_planned": ((7492, 13704), (6443, 15187), (7130, 13360), (7453, 15812)),
    "excluded_random": ((97, 1478), (138, 1943), (75, 1133), (78, 1214)),
    "excluded_early_planned": ((47, 659), (61, 883), (62, 940), (73, 1048)),
}


def phase(episode: int, decision: int) -> str | None:
    if type(episode) is not int or not 0 <= episode <= 306 or type(decision) is not int or decision < 1:
        raise ValueError("invalid original RAW episode/decision")
    if episode >= 44:
        if decision <= 10000:
            raise ValueError("fitting episode cannot contain a seed action")
        return "fit_planned"
    if episode >= 12:
        return "excluded_random" if decision <= 10000 else "excluded_early_planned"
    return None


def groups_from_ledger(episodes: list[dict], path: Path) -> dict:
    """Order every target by original episode/step, including ending transitions."""
    groups = {p: {str(r): {s: [] for s in SIGNS} for r in head.raw.ROADS} for p in PHASES}
    decision = 0
    if len(episodes) != 307:
        raise ValueError("only complete original RAW307 episodes are allowed")
    with path.open("rb") as stream:
        for eid, ep in enumerate(episodes):
            if (ep.get("episode") != eid or ep.get("geometry_seed") != head.raw.ROADS[eid % 4]
                    or ep.get("track_id") != 1 or type(ep.get("length")) is not int
                    or not 1 <= ep["length"] <= 2000 or ep.get("decisions") != decision + ep["length"]):
                raise ValueError("original RAW episode schedule differs")
            for step in range(ep["length"]):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete RAW step ledger")
                row = head.raw._json(line)
                decision += 1
                if (row.get("episode") != eid or row.get("decision") != decision
                        or row.get("geometry_seed") != ep["geometry_seed"] or row.get("track_id") != 1
                        or type(row.get("reward")) not in (int, float)
                        or not np.isfinite(row["reward"])):
                    raise ValueError("original RAW transition identity/reward differs")
                p = phase(eid, decision)
                if p is not None:
                    s = "positive" if np.float32(row["reward"]) > 0 else "nonpositive"
                    groups[p][str(ep["geometry_seed"])][s].append((eid, step, decision, row["reward"]))
        if stream.read(1) or decision != 100354:
            raise ValueError("extra/missing RAW transitions")
    return groups


def _refs(value):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for child in value.values():
                yield from _refs(child)


def _sources(root: Path) -> tuple[dict, dict, list[dict], dict]:
    pilot = head.raw._json(head._pinned(root, PILOT).read_bytes())
    spec = head.raw._json(head._pinned(root, (head.PROTOCOL, PILOT_PROTOCOL_SHA)).read_bytes())
    if (pilot.get("format") != "haic-tdmpc2-head-only-result-v2" or pilot.get("status") != "complete"
            or pilot.get("source") != spec or pilot.get("protocol") != head.ref(head.PROTOCOL, PILOT_PROTOCOL_SHA)
            or pilot.get("checkpoint") != head.ref(*ADAPTED) or pilot.get("optimizer_updates") != 512
            or pilot.get("environment_resets") != 0 or pilot.get("gate", {}).get("status") != "FAIL"
            or spec.get("split") != head.SPLIT or spec.get("training") != head.TRAINING
            or spec.get("holdout_gate") != head.GATE
            or spec.get("unique_step_counts") != head.EXPECTED_SPLIT_COUNTS
            or pilot.get("nonhead_bitwise_parity", {}).get("passed") is not True
            or pilot["nonhead_bitwise_parity"].get("sha256_before") !=
            pilot["nonhead_bitwise_parity"].get("sha256_after")):
        raise ValueError("complete negative head pilot binding differs")
    for reference in _refs(spec):
        head._pinned(root, (reference["path"], reference["sha256"]))
    head._pinned(root, ADAPTED)
    journal = head._pinned(root, (f"{head.OUTPUT}/journal.jsonl", pilot["journal_sha256"]))
    records = [head.raw._json(line) for line in journal.read_bytes().splitlines()]
    intents = [r for r in records if r.get("event") == "update_intent"]
    if ([r.get("update") for r in intents] != list(range(1, 513))
            or records[-1].get("event") != "complete" or records[-1].get("updates") != 512
            or records[-1].get("checkpoint_sha256") != ADAPTED[1]
            or any(r.get("event") == "partial" or r.get("protocol_sha256") != PILOT_PROTOCOL_SHA for r in records)):
        raise ValueError("pilot journal incomplete or different")
    for name, sha in spec["source_sha256"].items():
        head._pinned(root, (name, sha))
    _, episodes, context = head._source(root)  # No load or optimizer construction.
    pools, holdout = head._indices(episodes, root / spec["raw_source"]["step_ledger"]["path"])
    counts = {str(r): {"adaptation": {s: len(pools[r][s]) for s in SIGNS},
                        "holdout": {s: len(holdout[r][s]) for s in SIGNS}} for r in head.raw.ROADS}
    if counts != spec["unique_step_counts"]:
        raise ValueError("pilot RAW label counts differ")
    return pilot, spec, episodes, context


def preflight(root: Path = ROOT, protocol_sha256: str | None = None, *, device: str = "cpu",
              allow_output: bool = False) -> dict:
    root = Path(root).resolve(strict=True)
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("explicit available cpu/cuda device required")
    output = root / OUTPUT
    if (output.is_symlink() or (root / "runs").is_symlink()
            or output.exists() and not allow_output):
        raise ValueError("exclusive diagnostic output already exists or is symlink")
    pilot, spec, episodes, context = _sources(root)
    groups = groups_from_ledger(episodes, root / spec["raw_source"]["step_ledger"]["path"])
    counts = {p: {str(r): {s: len(groups[p][str(r)][s]) for s in SIGNS} for r in head.raw.ROADS}
              for p in PHASES}
    expected = {p: {str(r): dict(zip(SIGNS, EXPECTED_COUNTS[p][i]))
                    for i, r in enumerate(head.raw.ROADS)} for p in PHASES}
    if counts != expected:
        raise ValueError("exact phase/road/sign unique RAW counts differ")
    sources = {**spec["source_sha256"], **{name: head.raw._digest(head.raw._file(root, name))
                                           for name in (SELF, TEST)}}
    schema = {"format": "haic-tdmpc2-head-fit-transfer-protocol-v1",
              "scope": "all_unique_original_RAW_logged_consumed_TRAIN_no_policy_gate",
              "pilot_result": head.ref(*PILOT), "pilot_protocol": head.ref(head.PROTOCOL, PILOT_PROTOCOL_SHA),
              "adapted_checkpoint": head.ref(*ADAPTED), "raw_source": spec["raw_source"],
              "overshoot_source": spec["overshoot_source"],
              "pilot_journal": head.ref(f"{head.OUTPUT}/journal.jsonl", pilot["journal_sha256"]),
              "source_sha256": sources, "runtime": head._runtime(), "device": device,
              "selection": {"fit_episode_range_inclusive": [44, 306],
                            "excluded_episode_range_inclusive": [12, 43], "seed_last_decision": 10000,
                            "phases": list(PHASES), "roads": list(head.raw.ROADS), "signs": list(SIGNS),
                            "method": "all_unique_sorted_original_episode_then_step_in_each_stratum",
                            "counts": counts, "manifest_sha256": head._sha_body(groups),
                            "batch_size": BATCH, "augmentation_seed": SEED,
                            "seed_rule": "20260929+road_index*1000000+phase_index*100000+sign_index*10000+batch_start",
                            "augmentation": "same_seed_reset_before_each_model_encode_fork_selected_device_rng",
                            "reward_units": "RAW_sum_of_up_to_four_native_frames_per_decision"},
              "metrics": {"per_sign": ["count", "ce", "mae", "signed_bias", "prediction_mean", "target_mean"],
                          "natural_ce": "count_weighted_all_unique_transitions",
                          "balanced_ce": "0.25_positive_CE+0.75_nonpositive_CE_per_road_then_uniform_four_roads",
                          "target": "original_soft_ce_raw_float32_symlog_two_hot_101_bins"},
              "fit_gate": GATE, "output": OUTPUT, "environment_resets": 0,
              "optimizer_updates": 0, "policy_or_official_claim": False}
    if protocol_sha256 is not None:
        frozen = head.raw._json(head._pinned(root, (PROTOCOL, head.raw._sha(protocol_sha256))).read_bytes())
        if frozen != schema:
            raise ValueError("frozen diagnostic protocol differs from exact source/runtime/schema")
    return {"status": "preflight_only", "schema": schema, "environment_resets": 0,
            "optimizer_updates": 0, "torch_load_calls": 0,
            "_groups": groups, "_episodes": episodes, "_context": context, "_pilot": pilot}


def _tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str(value.dtype).encode("ascii") + str(tuple(value.shape)).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_adapted(state: dict, parent: dict, pilot: dict) -> dict:
    spec = pilot["source"]
    if (not isinstance(state, dict) or state.get("format") != "haic-tdmpc2-head-only-model-v2"
            or state.get("protocol_sha256") != PILOT_PROTOCOL_SHA or state.get("source") != spec["overshoot_source"]
            or state.get("source_sha256") != spec["source_sha256"] or state.get("updates") != 512
            or state.get("seed") != SEED or state.get("nonhead_bitwise_parity") is not True
            or state.get("throughput_benchmark") != spec["throughput_benchmark"]
            or not isinstance(state.get("model"), dict) or state["model"].keys() != parent.keys()):
        raise ValueError("adapted model metadata/key binding differs")
    changed = []
    for key, old in parent.items():
        new = state["model"][key]
        if (not isinstance(old, torch.Tensor) or not isinstance(new, torch.Tensor)
                or old.device.type != "cpu" or new.device.type != "cpu"
                or new.dtype != old.dtype or new.shape != old.shape
                or not torch.isfinite(old).all() or not torch.isfinite(new).all()):
            raise ValueError("invalid/nonfinite adapted or parent tensor")
        old_sha, new_sha = _tensor_sha(old), _tensor_sha(new)
        if key.startswith("_reward."):
            if old_sha != new_sha:
                changed.append(key)
        elif (old_sha != new_sha or old_sha != pilot["nonhead_bitwise_parity"]["sha256_before"].get(key)):
            raise ValueError(f"adapted nonhead byte parity failed: {key}")
    if not changed or sorted(changed) != sorted(pilot["nonhead_bitwise_parity"]["changed_reward_keys"]):
        raise ValueError("adapted changed-head keys differ from complete result")
    return state["model"]


def _load(root: Path, pre: dict, protocol_sha256: str) -> tuple[dict, dict]:
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
    from scripts import diagnose_tdmpc2_h5_branches as binder

    spec = pre["schema"]
    loaded = []
    for ref in (spec["raw_source"]["checkpoint"], spec["overshoot_source"]["checkpoint"], spec["adapted_checkpoint"]):
        checked = preflight(root, protocol_sha256, device=spec["device"], allow_output=True)
        if checked["schema"] != spec:
            raise ValueError("pins changed before checkpoint load")
        with head._pinned(root, (ref["path"], ref["sha256"])).open("rb") as stream:
            if head.raw._digest_stream(stream) != ref["sha256"]:
                raise ValueError("SHA mismatch before torch.load")
            stream.seek(0)
            loaded.append(torch.load(stream, map_location="cpu", weights_only=False))
    original, parent, adapted = loaded
    raw_protocol = head.raw._json(head._pinned(root, (spec["raw_source"]["protocol"]["path"],
                                                    spec["raw_source"]["protocol"]["sha256"])).read_bytes())
    binder.bind_replay(original, raw_protocol, pre["_episodes"], root / spec["raw_source"]["step_ledger"]["path"])
    head.overshoot._optimizer(original.get("optim"), 100354)
    head.overshoot._optimizer(original.get("pi_optim"), 100354)
    context = pre["_context"]
    training, last = context["training"], context["last"]
    if (not isinstance(parent, dict) or parent.get("format") != training["format"]
            or parent.get("protocol_sha256") != head.overshoot.TRAIN_PROTOCOL_SHA
            or parent.get("source_sha256") != training["source_sha256"]
            or parent.get("reward_overshoot") != training["reward_overshoot"]
            or any(parent.get(k) != training[k] for k in ("baseline_training_protocol", "baseline_training_result",
                                                         "replay_audit", "throughput_benchmark"))
            or any(parent.get(k) != last[k] for k in ("target", "decisions", "updates", "episodes",
                                                     "step_ledger_sha256", "training_ledger_sha256_before_checkpoint"))
            or parent.get("action_dim") != 3 or parent.get("resume_supported") is not False
            or not isinstance(parent.get("learner"), dict)):
        raise ValueError("complete overshoot parent metadata differs")
    replay = parent.get("replay")
    if (not isinstance(replay, dict) or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or any(replay.get(k) != v for k, v in {"active": None, "size": 100159, "next_episode_id": 309,
                   "horizon": 3, "action_dim": 3, "capacity": 120000, "observation_shape": (4, 64, 64)}.items())):
        raise ValueError("complete overshoot replay metadata differs")
    bundle = {"spec": {"training_source": context["refs"]},
              "episodes": head.overshoot._ledgers(root, {"training_source": context["refs"]},
                                                  training, context["result"])}
    head.overshoot._bind_checkpoint_replay(root, parent, bundle)
    head.overshoot._overshoot_probe(parent, training, last)
    head.overshoot._optimizer(parent.get("optim"), 100159)
    head.overshoot._optimizer(parent.get("pi_optim"), 100159)
    if not isinstance(parent["learner"].get("q_scale"), torch.Tensor):
        raise ValueError("parent learner scale missing")
    weights = {k.removeprefix("model."): v for k, v in parent["learner"].items() if k.startswith("model.")}
    adapted_weights = validate_adapted(adapted, weights, pre["_pilot"])
    models = {}
    with torch.random.fork_rng(devices=[]):
        for name, state in (("parent", weights), ("adapted", adapted_weights)):
            model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
            model.load_state_dict(state, strict=True)
            model.requires_grad_(False).eval().to(spec["device"])
            models[name] = model
    return models, original["replay"]  # Never score the parent's different 309-episode replay.


def summarize(target, prediction, ce) -> dict:
    arrays = [np.asarray(x, np.float64) for x in (target, prediction, ce)]
    if (arrays[0].ndim != 1 or not len(arrays[0]) or any(a.shape != arrays[0].shape for a in arrays)
            or any(not np.isfinite(a).all() for a in arrays) or np.any(arrays[2] < 0)):
        raise ValueError("empty/malformed/nonfinite reward metrics")
    y, pred, losses = arrays
    error = pred - y
    return {"count": len(y), "ce": float(losses.mean()), "mae": float(np.abs(error).mean()),
            "signed_bias": float(error.mean()), "prediction_mean": float(pred.mean()), "target_mean": float(y.mean())}


def ce_summary(roads: dict) -> dict:
    total = sum(roads[r][s]["count"] for r in roads for s in SIGNS)
    if total <= 0 or any(roads[r][s]["count"] <= 0 or not np.isfinite(roads[r][s]["ce"])
                         for r in roads for s in SIGNS):
        raise ValueError("incomplete/nonfinite CE strata")
    by_road = {r: {"natural_ce": sum(roads[r][s]["count"] * roads[r][s]["ce"] for s in SIGNS) /
                               sum(roads[r][s]["count"] for s in SIGNS),
                   "balanced_ce_25_75": .25 * roads[r]["positive"]["ce"] + .75 * roads[r]["nonpositive"]["ce"]}
               for r in roads}
    return {"count": total, "by_road": by_road,
            "natural_ce": sum(roads[r][s]["count"] * roads[r][s]["ce"] for r in roads for s in SIGNS) / total,
            "balanced_ce_25_75_uniform_roads": float(np.mean([x["balanced_ce_25_75"] for x in by_road.values()]))}


def score_models(models: dict, replay: dict, groups: dict, *, device: str = "cpu") -> dict:
    from haic.algorithms.tdmpc2.model import soft_ce, two_hot_inv

    if set(models) != {"parent", "adapted"}:
        raise ValueError("exactly parent and adapted frozen models required")
    metrics = {name: {} for name in models}
    rng_devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices), torch.inference_mode():
        for pi, p in enumerate(PHASES):
            for name in models:
                metrics[name][p] = {}
            for ri, road in enumerate(head.raw.ROADS):
                road = str(road)
                for name in models:
                    metrics[name][p][road] = {}
                for si, sign in enumerate(SIGNS):
                    entries = groups[p][road][sign]
                    values = {name: {"prediction": [], "ce": []} for name in models}
                    targets = []
                    for first in range(0, len(entries), BATCH):
                        batch = entries[first:first + BATCH]
                        episodes = replay["episodes"]
                        obs = torch.from_numpy(np.stack([episodes[e]["observations"][s] for e, s, _, _ in batch])).to(device)
                        action = torch.from_numpy(np.stack([episodes[e]["actions"][s] for e, s, _, _ in batch])).to(device)
                        reward = np.asarray([episodes[e]["rewards"][s] for e, s, _, _ in batch], np.float32)
                        labels = np.asarray([r for _, _, _, r in batch], np.float64)
                        if (obs.dtype != torch.uint8 or tuple(obs.shape) != (len(batch), 4, 64, 64)
                                or action.dtype != torch.float32 or tuple(action.shape) != (len(batch), 3)
                                or not torch.isfinite(action).all() or (action.abs() > 1).any()
                                or not np.isfinite(reward).all() or not np.array_equal(reward, labels.astype(np.float32))
                                or np.any((reward > 0) != (sign == "positive"))):
                            raise ValueError("scored ORIGINAL RAW pixel/action/reward/sign binding differs")
                        targets.extend(labels.tolist())
                        target = torch.from_numpy(reward[:, None]).to(device)
                        seed = SEED + ri * 1000000 + pi * 100000 + si * 10000 + first
                        for name, model in models.items():
                            model.eval()
                            torch.default_generator.manual_seed(seed)
                            if rng_devices:
                                torch.cuda.default_generators[rng_devices[0]].manual_seed(seed)
                            z = model.encode(obs, None)
                            if z.shape != (len(batch), model.cfg.latent_dim) or not torch.isfinite(z).all():
                                raise ValueError("nonfinite/malformed true encoded latent")
                            logits = model.reward(z, action, None)
                            if logits.shape != (len(batch), 101) or not torch.isfinite(logits).all():
                                raise ValueError("nonfinite/malformed reward logits")
                            pred, loss = two_hot_inv(logits, model.cfg), soft_ce(logits, target, model.cfg)
                            if pred.shape != (len(batch), 1) or loss.shape != (len(batch), 1):
                                raise ValueError("malformed categorical reward metrics")
                            values[name]["prediction"].extend(pred[:, 0].cpu().numpy().astype(np.float64).tolist())
                            values[name]["ce"].extend(loss[:, 0].cpu().numpy().astype(np.float64).tolist())
                    for name in models:
                        metrics[name][p][road][sign] = summarize(targets, values[name]["prediction"], values[name]["ce"])
    return {name: {"strata": phases, "ce_summaries": {p: ce_summary(phases[p]) for p in PHASES}}
            for name, phases in metrics.items()}


def fit_gate(parent: dict, adapted: dict) -> dict:
    checks = {}
    for road in map(str, head.raw.ROADS):
        old, new = parent[road], adapted[road]
        for sign in SIGNS:
            if (type(old[sign]["count"]) is not int or old[sign]["count"] <= 0
                    or old[sign]["count"] != new[sign]["count"]
                    or any(not np.isfinite(row[k]) for row in (old[sign], new[sign]) for k in ("mae", "signed_bias"))):
                raise ValueError("invalid fit gate counts/metrics")
        d = lambda x: Decimal(str(x))
        mae = d(old["positive"]["mae"]) - d(new["positive"]["mae"])
        bias = abs(d(old["positive"]["signed_bias"])) - abs(d(new["positive"]["signed_bias"]))
        neg = d(new["nonpositive"]["mae"]) - d(old["nonpositive"]["mae"])
        checks[road] = {"positive_mae_improvement": float(mae), "positive_absolute_signed_bias_improvement": float(bias),
                        "nonpositive_mae_worsening": float(neg),
                        "qualifies": mae >= d(.10) and bias >= d(.10), "nonpositive_pass": neg <= d(.05)}
    qualifies = sum(c["qualifies"] for c in checks.values())
    passed = qualifies >= 3 and all(c["nonpositive_pass"] for c in checks.values())
    return {"status": "PASS" if passed else "FAIL", "by_road": checks, "qualifying_roads": qualifies,
            "transfer_only_explanation_rejected": not passed,
            "next_mechanism": "assess_excluded_phase_transfer" if passed else "objective_head_discrimination",
            "policy_gate": False, "policy_release": False}


def execute(root: Path, protocol_sha256: str, *, device: str = "cpu") -> dict:
    root = Path(root).resolve(strict=True)
    pre = preflight(root, protocol_sha256, device=device)
    # Reserve the exclusive artifact before loading; failures remain non-resumable.
    with (root / OUTPUT).open("x", encoding="utf-8") as stream:
        try:
            models, replay = _load(root, pre, protocol_sha256)
            before = {name: {k: _tensor_sha(v) for k, v in m.state_dict().items()} for name, m in models.items()}
            report = score_models(models, replay, pre["_groups"], device=device)
            if any(before[name] != {k: _tensor_sha(v) for k, v in m.state_dict().items()} for name, m in models.items()):
                raise ValueError("frozen model state changed during diagnostic")
            checked = preflight(root, protocol_sha256, device=device, allow_output=True)
            if checked["schema"] != pre["schema"]:
                raise ValueError("source drift during scoring")
            result = {"format": "haic-tdmpc2-head-fit-transfer-result-v1", "status": "complete",
                      "protocol": head.ref(PROTOCOL, protocol_sha256), "source": pre["schema"], "metrics": report,
                      "fit_gate": fit_gate(report["parent"]["strata"]["fit_planned"], report["adapted"]["strata"]["fit_planned"]),
                      "frozen_model_bitwise_parity": True, "torch_load_calls": 3,
                      "environment_resets": 0, "optimizer_updates": 0, "policy_or_official_claim": False}
        except BaseException as exc:
            json.dump({"status": "failed_no_resume", "reason": type(exc).__name__,
                       "protocol_sha256": protocol_sha256, "environment_resets": 0,
                       "optimizer_updates": 0, "policy_release": False}, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            raise
        result["body_sha256"] = head._sha_body(result)
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-runtime", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--score", action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.score and not args.protocol_sha256:
        parser.error("--score requires separately frozen --protocol-sha256")
    if args.print_runtime:
        print(json.dumps({"runtime": head._runtime(), "environment_resets": 0, "optimizer_updates": 0,
                          "torch_load_calls": 0}, sort_keys=True))
    elif args.preflight:
        pre = preflight(ROOT, args.protocol_sha256, device=args.device)
        print(json.dumps({k: pre[k] for k in ("status", "schema", "environment_resets", "optimizer_updates", "torch_load_calls")},
                         sort_keys=True, allow_nan=False))
    else:
        print(json.dumps(execute(ROOT, args.protocol_sha256, device=args.device)["fit_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
