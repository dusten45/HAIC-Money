"""Read-only balanced H3 semantic-terminal probe on sealed 20k TRAIN replay.

Run from the repository root with ``python -m scripts.diagnose_tdmpc2_terminal_probe``.
Default targets require all four sealed checkpoints; use --targets for a subset.
This is in-sample training-replay event detection, not driving/generalization or
an official ranking. Only trusted local checkpoints may be supplied: their full
source, ledger cursor and SHA are checked before pickle deserialization.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts import diagnose_tdmpc2_checkpoint_losses as cursor


ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/diagnose_tdmpc2_checkpoint_losses.py"
HELPER_SHA256 = "ef6552479192e2e113951edb3bfbf7a4c87b3a508dd884c45ce6a270a0aebe6f"
ANCHOR_SHA256 = "0915be9a448be9628d2668880f23258c9c7e5e282aa1e9cfe22983c02f0145bd"
SEED_PROBE_SHA256 = "ff0731ac2589c21662bcd941c12e857972af02b4756476ac41c0309e9234b6f5"
TARGETS = (20000, 40000, 70000, 100000)
CLASSES = ("raw_termination", "finish", "time_limit_truncation", "ordinary_nonterminal")
HORIZON = 3


def _bind(root: Path, targets: tuple[int, ...]) -> tuple[dict, list[dict]]:
    if (not targets or tuple(sorted(set(targets))) != targets
            or any(type(t) is not int or t not in TARGETS for t in targets)):
        raise ValueError("select distinct ascending checkpoint targets")
    if cursor._digest(cursor._file(root, HELPER)) != HELPER_SHA256:
        raise ValueError("checkpoint cursor helper SHA mismatch")
    if (cursor.PROTOCOL_SHA256 != "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec"
            or cursor.RUN != "runs/tdmpc2-long-20260928-v2" or cursor.TARGETS != TARGETS):
        raise ValueError("checkpoint cursor is not bound to the long v2 run")
    protocol = cursor._protocol(root)
    # The anchor is mandatory even when scoring only a later checkpoint.
    pins = [cursor._cursor(root, protocol, t) for t in (20000, *[t for t in targets if t != 20000])]
    if pins[0]["checkpoint_sha256"] != ANCHOR_SHA256:
        raise ValueError("20k sealed checkpoint SHA mismatch")
    return protocol, pins


def _episodes(root: Path, pin: dict, max_steps: int) -> list[dict]:
    rows = []
    digest = hashlib.sha256()
    with cursor._file(root, f"{cursor.RUN}/training.jsonl").open("rb") as stream:
        for line in range(1, pin["line"] + 1):
            data = stream.readline()
            if not data.endswith(b"\n"):
                raise ValueError("incomplete sealed training cursor")
            digest.update(data)
            entry = cursor._json(data)
            if entry.get("event") != "episode":
                continue
            if entry.get("episode") != len(rows) or type(entry.get("finished")) is not bool:
                raise ValueError("ambiguous episode identity/finish semantics")
            if entry.get("terminal") is not (entry.get("terminated") or entry["finished"]):
                raise ValueError("semantic terminal differs from termination or finish")
            if entry.get("terminated") and entry.get("truncated"):
                raise ValueError("simultaneous termination/truncation has ambiguous event class")
            if entry["finished"]:
                if entry.get("truncated") is not True or entry.get("finish_time_s") is None:
                    raise ValueError("finish is not a confirmed truncation")
            elif entry.get("finish_time_s") is not None:
                raise ValueError("unlabeled finish time")
            if entry.get("truncated") and not entry.get("terminated") and not entry["finished"]:
                if entry.get("length") != max_steps:
                    raise ValueError("nonfinish truncation is not a time limit")
            if entry.get("length") == max_steps and not entry.get("terminated") and not entry["finished"]:
                if entry.get("truncated") is not True:
                    raise ValueError("time-limit boundary is missing truncation")
            rows.append(entry)
    if len(rows) != pin["row"]["episodes"] or digest.hexdigest() != pin["ledger_prefix_sha256"]:
        raise ValueError("sealed training prefix SHA or episode count differs from checkpoint")
    return rows


def _replay(state: dict, protocol: dict, pin: dict) -> dict:
    row = pin["row"]
    replay = state.get("replay") if isinstance(state, dict) else None
    if (not isinstance(state, dict) or set(state) != {
            "format", "protocol_sha256", "source_sha256", "target", "decisions", "updates", "episodes",
            "action_dim", "learner", "optim", "pi_optim", "replay", "probe", "rng", "resume_supported"}
            or state["format"] != protocol["format"] or state["protocol_sha256"] != cursor.PROTOCOL_SHA256
            or state["source_sha256"] != protocol["source_sha256"] or state["resume_supported"] is not False
            or state["action_dim"] != 3
            or any(state[k] != row[k] for k in ("target", "decisions", "updates", "episodes"))
            or not isinstance(replay, dict) or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or replay.get("active") is not None or replay.get("size") != row["decisions"]
            or replay.get("next_episode_id") != row["episodes"]
            or replay.get("capacity") != protocol["training"]["replay_capacity"]
            or replay.get("horizon") != HORIZON or replay.get("action_dim") != 3
            or replay.get("observation_shape") != (4, 64, 64)
            or not isinstance(replay.get("episodes"), list) or len(replay["episodes"]) != row["episodes"]):
        raise ValueError("checkpoint replay differs from sealed whole-episode cursor")
    _, seed_sha = cursor._probe(state, protocol, row)
    if seed_sha != SEED_PROBE_SHA256:
        raise ValueError("checkpoint seed replay probe SHA differs from 20k anchor")
    return replay


def _validate_anchor(root: Path, replay: dict, episodes: list[dict], pin: dict) -> None:
    """Check every 20k replay transition against the SHA-bound step-ledger prefix."""
    decision = 0
    digest = hashlib.sha256()
    with cursor._file(root, f"{cursor.RUN}/steps.jsonl").open("rb") as stream:
        for eid, (ep, row) in enumerate(zip(replay["episodes"], episodes)):
            n = row["length"]
            if (ep.get("episode_id") != eid or ep.get("start_step") != 0
                    or not isinstance(ep.get("observations"), np.ndarray)
                    or ep["observations"].dtype != np.uint8 or ep["observations"].shape != (n + 1, 4, 64, 64)
                    or not isinstance(ep.get("actions"), np.ndarray)
                    or ep["actions"].dtype != np.float32 or ep["actions"].shape != (n, 3)
                    or any(not isinstance(ep.get(k), np.ndarray) or ep[k].shape != (n,)
                           or ep[k].dtype != (np.float32 if k == "rewards" else np.bool_)
                           for k in ("rewards", "terminated", "truncated", "terminal"))):
                raise ValueError("20k replay episode is incomplete, trimmed or malformed")
            for offset in range(n):
                data = stream.readline()
                if not data.endswith(b"\n"):
                    raise ValueError("missing 20k step prefix")
                digest.update(data)
                step = cursor._json(data)
                decision += 1
                if (step.get("decision") != decision or step.get("episode") != eid
                        or step.get("action_f32_hex") != ep["actions"][offset].tobytes().hex()
                        or np.float32(step.get("reward")) != ep["rewards"][offset]
                        or any(step.get(k) is not bool(ep[k][offset]) for k in ("terminated", "truncated", "terminal"))):
                    raise ValueError("20k replay action/reward/semantic label differs from step ledger")
            if (any(bool(ep[k][n - 1]) is not row[k] for k in ("terminated", "truncated", "terminal"))
                    or np.any(ep["terminated"][:-1] | ep["truncated"][:-1] | ep["terminal"][:-1])):
                raise ValueError("replay episode has ambiguous internal or final boundary")
    if decision != pin["row"]["decisions"] or digest.hexdigest() != pin["step_prefix_sha256"]:
        raise ValueError("20k replay step prefix SHA or decision count differs from sealed cursor")


def _windows(replay: dict, episodes: list[dict], seed: int, checkpoint_sha: str) -> tuple[list[tuple[int, int]], dict]:
    positive, negative = [], []
    for eid, (ep, row) in enumerate(zip(replay["episodes"], episodes)):
        if ep.get("episode_id") != eid or ep.get("start_step") != 0 or len(ep["actions"]) != row["length"]:
            raise ValueError("replay window ID cannot be bound to sealed episode")
        if row["terminal"] and row["length"] < HORIZON:
            raise ValueError("positive terminal has no complete H3 window")
        for start in range(row["length"] - HORIZON + 1):
            flags = ep["terminal"][start:start + HORIZON]
            (positive if bool(np.any(flags)) else negative).append((eid, start))
    if not positive or len(negative) < len(positive):
        raise ValueError("cannot construct positive/negative-balanced H3 TRAIN windows")
    key = hashlib.sha256(f"{checkpoint_sha}:{SEED_PROBE_SHA256}:{seed}".encode("ascii")).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "big"))
    chosen = sorted(negative[int(i)] for i in rng.choice(len(negative), len(positive), replace=False))
    ids = sorted(positive + chosen)
    return ids, {"positive_windows": len(positive), "negative_windows": len(chosen),
                 "eligible_negative_windows": len(negative), "window_count": len(ids),
                 "window_ids": [{"episode_id": eid, "start_step": start} for eid, start in ids],
                 "selection_seed_sha256": key.hex()}


def _batch(replay: dict, episodes: list[dict], ids: list[tuple[int, int]]) -> tuple[dict, np.ndarray, str]:
    obs, actions, labels, classes = [], [], [], []
    for eid, start in ids:
        ep, row = replay["episodes"][eid], episodes[eid]
        if ep.get("episode_id") != eid or ep.get("start_step") != 0 or start < 0 or start + HORIZON > row["length"]:
            raise ValueError("window crosses an episode or replay trim")
        obs.append(ep["observations"][start:start + HORIZON + 1])
        actions.append(ep["actions"][start:start + HORIZON])
        flags = ep["terminal"][start:start + HORIZON]
        labels.append(flags.astype(np.float32))
        classes.append([("raw_termination" if row["terminated"] else "finish") if bool(flag)
                        else ("time_limit_truncation" if start + t == row["length"] - 1 and row["truncated"]
                              and not row["terminal"] else "ordinary_nonterminal")
                        for t, flag in enumerate(flags)])
    probe = {"obs": torch.from_numpy(np.stack(obs, axis=1).copy()),
             "action": torch.from_numpy(np.stack(actions, axis=1).copy()),
             "terminal": torch.from_numpy(np.stack(labels, axis=1)[:, :, None].copy())}
    tags = np.asarray(classes).T
    digest = hashlib.sha256()
    for name in sorted(probe):
        tensor = probe[name]
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    digest.update(json.dumps(classes, separators=(",", ":")).encode("ascii"))
    return probe, tags, digest.hexdigest()


def _summarize(logits: torch.Tensor, labels: torch.Tensor, classes: np.ndarray) -> dict:
    if (logits.shape != labels.shape or tuple(logits.shape[:-1]) != classes.shape
            or logits.shape[-1] != 1 or not torch.isfinite(logits).all()):
        raise ValueError("nonfinite or misaligned termination predictions")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("nonbinary termination labels")
    probabilities = torch.sigmoid(logits).numpy().reshape(-1)
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").numpy().reshape(-1)
    targets = labels.numpy().reshape(-1)
    tags = classes.reshape(-1)
    by_class = {}
    for name in CLASSES:
        selected = tags == name
        count = int(selected.sum())
        expected = 0 if name in ("time_limit_truncation", "ordinary_nonterminal") else 1
        if count and not np.all(targets[selected] == expected):
            raise ValueError("semantic class contradicts terminal target")
        by_class[name] = {"count": count, "label": expected,
                          "bce": float(losses[selected].mean()) if count else None,
                          "mean_probability": float(probabilities[selected].mean()) if count else None,
                          "correct_at_threshold": int(np.sum((probabilities[selected] >= .5) == expected)),
                          "positive_recall": float(np.mean(probabilities[selected] >= .5)) if count and expected else None,
                          "specificity": float(np.mean(probabilities[selected] < .5)) if count and not expected else None}
    positives, negatives = targets == 1, targets == 0
    if not positives.any() or not negatives.any():
        raise ValueError("missing positive or negative transition denominator")
    return {"transitions": int(targets.size), "positive_transitions": int(positives.sum()),
            "negative_transitions": int(negatives.sum()), "bce": float(losses.mean()),
            "true_positive_count": int(np.sum(probabilities[positives] >= .5)),
            "true_negative_count": int(np.sum(probabilities[negatives] < .5)),
            "mean_positive_probability": float(probabilities[positives].mean()),
            "mean_negative_probability": float(probabilities[negatives].mean()),
            "positive_recall": float(np.mean(probabilities[positives] >= .5)),
            "specificity": float(np.mean(probabilities[negatives] < .5)), "per_class": by_class}


def _predict(model, probe: dict, seed: int, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    real, predicted = [], []
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        for start in range(0, probe["action"].shape[1], batch_size):
            stop = start + batch_size
            obs, action = probe["obs"][:, start:stop], probe["action"][:, start:stop]
            true_z = model.encode(obs[1:], None)
            z = model.encode(obs[0], None)
            steps = []
            for t in range(HORIZON):
                z = model.next(z, action[t], None)
                steps.append(z)
            actual = model.termination(true_z, None, unnormalized=True)
            imagined = model.termination(torch.stack(steps), None, unnormalized=True)
            if actual.shape != (HORIZON, action.shape[1], 1) or imagined.shape != actual.shape:
                raise ValueError("termination head returned unexpected H3 shape")
            real.append(actual.cpu())
            predicted.append(imagined.cpu())
    return torch.cat(real, dim=1), torch.cat(predicted, dim=1)


def score(*, root: Path = ROOT, targets: tuple[int, ...] = TARGETS, seed: int = 834) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    root = root.resolve(strict=True)
    protocol, pins = _bind(root, targets)
    # Each selected checkpoint has a complete SHA-bound ledger cursor before ANY torch.load.
    episodes = _episodes(root, pins[0], protocol["training"]["max_steps"])
    reports = []
    ids = selection = anchor_batch = tags = batch_sha = None
    for pin in pins:
        with pin["checkpoint"].open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != pin["checkpoint_sha256"]:
                raise ValueError("checkpoint changed before torch.load")
            stream.seek(0)
            state = torch.load(stream, map_location="cpu", weights_only=False)
        replay = _replay(state, protocol, pin)
        if pin is pins[0]:
            _validate_anchor(root, replay, episodes, pin)
            ids, selection = _windows(replay, episodes, seed, pin["checkpoint_sha256"])
            anchor_batch, tags, batch_sha = _batch(replay, episodes, ids)
        else:
            if ids is None or tags is None or batch_sha is None:
                raise ValueError("missing 20k anchor before later checkpoint")
            other, other_tags, other_sha = _batch(replay, episodes, ids)
            if other_sha != batch_sha or not np.array_equal(other_tags, tags):
                raise ValueError("later checkpoint changed the 20k replay observations/actions/labels")
            del other
        if pin["row"]["target"] not in targets:
            del state, replay
            gc.collect()
            continue
        from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel

        learner = state.get("learner")
        if (not isinstance(learner, dict) or "q_scale" not in learner
                or any(not isinstance(key, str) or key != "q_scale" and not key.startswith("model.")
                       for key in learner)):
            raise ValueError("checkpoint learner weights malformed")
        if anchor_batch is None or tags is None:
            raise ValueError("missing 20k replay window probe")
        with torch.random.fork_rng(devices=[]):
            model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
        model.load_state_dict({key[6:]: value for key, value in learner.items() if key.startswith("model.")}, strict=True)
        del state, replay, learner
        gc.collect()
        true_logits, predicted_logits = _predict(model, anchor_batch, seed)
        report = {"target": pin["row"]["target"], "decisions": pin["row"]["decisions"],
                  "checkpoint_path": pin["checkpoint"].relative_to(root).as_posix(),
                  "checkpoint_sha256": pin["checkpoint_sha256"], "training_ledger_line": pin["line"],
                  "training_ledger_prefix_sha256": pin["ledger_prefix_sha256"],
                  "step_ledger_prefix_sha256": pin["step_prefix_sha256"],
                  "true_next": _summarize(true_logits, anchor_batch["terminal"], tags),
                  "predicted_latent": _summarize(predicted_logits, anchor_batch["terminal"], tags)}
        if not all(math.isfinite(metric["bce"]) for metric in (report["true_next"], report["predicted_latent"])):
            raise FloatingPointError("nonfinite terminal BCE")
        reports.append(report)
        del model
        gc.collect()
    return {"scope": "read_only_in_sample_balanced_h3_terminal_detection", "reused_train_only": True,
            "official_ranking_claim": False, "generalization_claim": False, "environment_resets": 0,
            "optimizer_steps": 0, "threshold": .5, "augmentation_seed": seed,
            "encoder_and_shift_comparability_caveat": "Encoder weights change and seeded per-frame shifts yield different latent representations; raw replay inputs and labels are identical, not fixed latents.",
            "semantic_classes": "raw_termination includes off-track/crash/out-of-bounds; finish is a semantic terminal despite truncation; plain time-limit truncation is NOT terminal. Raw termination cause is not recorded in replay.",
            "protocol_sha256": cursor.PROTOCOL_SHA256, "source_sha256": protocol["source_sha256"],
            "checkpoint_helper_sha256": HELPER_SHA256, "anchor_checkpoint_sha256": ANCHOR_SHA256,
            "seed_replay_probe_sha256": SEED_PROBE_SHA256, "balanced_probe_sha256": batch_sha,
            "selection": selection, "checkpoints": reports, "torch_version": str(torch.__version__)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", type=int, default=list(TARGETS), choices=TARGETS,
                        help="ascending sealed checkpoint targets; default: all four")
    parser.add_argument("--seed", type=int, default=834, help="shared window-selection and CPU shift seed")
    parser.add_argument("--output", type=Path, help="exclusive source-bound direct runs/ receipt")
    args = parser.parse_args()
    output = args.output.absolute() if args.output is not None else None
    if output is not None:
        name = f"tdmpc2-long-v2-terminal-{'-'.join(map(str, args.targets))}.json"
        if output != ROOT / "runs" / name or output.parent.is_symlink():
            parser.error("output must be the exact direct runs/ terminal receipt")
    report = score(targets=tuple(args.targets), seed=args.seed)
    payload = json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(payload, end="")
        return
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
