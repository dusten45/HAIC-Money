"""Offline shared-replay actor diagnostics; the CLI requires the pinned CPU21 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from agent import Agent, DRQ_ACTOR_FORMAT
from common_adapter import ActionSpec, ObservationSpec, file_sha256
from drq_v2 import DrQv2Config, Uint8Replay


BATCH_SIZE = 64
SAMPLE_SEED = 20260921


def array_metadata(values):
    values = np.ascontiguousarray(values)
    return {"dtype": values.dtype.str, "shape": list(values.shape),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("diagnostic values must be nonempty and finite")
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "min": float(values.min()),
            "max": float(values.max())}


def load_sample(checkpoint, sample_size=2048, sample_seed=SAMPLE_SEED):
    """Restore one replay, sample without advancing its RNG, then release it."""
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if type(sample_seed) is not int or sample_seed < 0:
        raise ValueError("sample_seed must be a nonnegative integer")
    checkpoint = Path(checkpoint).resolve()
    checkpoint_hash = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "haic-drq-v2-checkpoint-v1":
        raise ValueError("source must be a full DrQ replay checkpoint")
    config = DrQv2Config(**payload["config"])
    observation = ObservationSpec(**payload["observation_spec"])
    action = ActionSpec(**payload["action_spec"])
    if observation.fingerprint != ObservationSpec().fingerprint or action.fingerprint != ActionSpec().fingerprint:
        raise ValueError("source must use the frozen CHW observation and action contracts")
    stored = payload["replay"]
    if any(stored[key] != expected for key, expected in (
        ("capacity", config.replay_capacity), ("action_dim", config.action_dim),
        ("n_step", config.n_step), ("gamma", config.gamma),
    )):
        raise ValueError("replay metadata disagrees with checkpoint configuration")
    if (type(stored["size"]) is not int or type(stored["next_sequence"]) is not int
            or not 0 < stored["size"] <= config.replay_capacity
            or stored["next_sequence"] < stored["size"]):
        raise ValueError("invalid replay sequence bounds")
    replay = Uint8Replay(config.replay_capacity, n_step=config.n_step, gamma=config.gamma)
    for name in ("frames", "actions", "rewards", "terminated", "truncated", "terminal",
                 "episode_ids", "episode_steps", "sequence_ids"):
        expected, actual = getattr(replay, name), stored[name]
        if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"invalid replay {name} dtype/shape")
    for value in stored["boundary_observations"].values():
        if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != observation.shape:
            raise ValueError("invalid replay boundary observation dtype/shape")
    replay.load_state_dict(stored)
    source = {"path": str(checkpoint), "sha256": checkpoint_hash, "config": payload["config"],
              "saved_device": config.device, "environment_steps": payload["environment_steps"],
              "gradient_steps": payload["gradient_steps"], "replay_size": replay.size,
              "observation_fingerprint": observation.fingerprint, "action_fingerprint": action.fingerprint}
    del payload, stored
    retained = np.arange(replay.oldest_sequence, replay.newest_sequence + 1, dtype=np.int64)
    slots = retained % replay.capacity
    if (not np.array_equal(replay.sequence_ids[slots], retained)
            or np.any(replay.episode_ids[slots] < 0) or np.any(replay.episode_steps[slots] < 0)
            or not np.isfinite(replay.rewards[slots]).all()
            or not np.isfinite(replay.actions[slots]).all() or np.any(np.abs(replay.actions[slots]) > 1)):
        raise ValueError("invalid retained replay records")
    rows = []
    for index in np.random.default_rng(sample_seed).permutation(retained):
        row = replay._build_n_step(int(index))
        if row is not None:
            rows.append(row)
            if len(rows) == sample_size:
                break
    if len(rows) != sample_size:
        raise ValueError(f"requested {sample_size} states, but only {len(rows)} valid replay starts exist")
    observations = np.stack([row["observation"] for row in rows])
    indices = np.asarray([row["sequence"] for row in rows], dtype="<i8")
    slots = indices % replay.capacity
    pixel_changes = []
    for offset in range(0, sample_size, BATCH_SIZE):
        pixels = observations[offset:offset + BATCH_SIZE].astype(np.int16)
        pixel_changes.extend(np.abs(np.diff(pixels, axis=1)).mean(axis=(1, 2, 3)) / 255)
    sample = {
        "size": sample_size, "seed": sample_seed,
        "selection": "first valid n-step starts in a seeded permutation of retained sequence IDs; no replacement",
        "indices": array_metadata(indices), "observations": array_metadata(observations),
        "layout": "BCHW, oldest to newest; uint8 / 255 once before minimal Agent inference",
        "distribution": {
            "episode_count": int(len(np.unique(replay.episode_ids[slots]))),
            "episode_step": statistics(replay.episode_steps[slots]),
            "raw_reward": statistics(replay.rewards[slots]),
            "negative_raw_reward_fraction": float(np.mean(replay.rewards[slots] < 0)),
            "terminal_nstep_endpoint_fraction": float(np.mean([row["terminal"] for row in rows])),
            "all_frames_identical_fraction": float(np.mean(np.all(observations == observations[:, -1:], axis=(1, 2, 3)))),
            "adjacent_frame_pixel_mae": statistics(pixel_changes),
            "recorded_native_action_mean": replay.actions[slots].mean(axis=0).tolist(),
        },
    }
    return observations, source, sample


def shift_uint8(observations, offsets, pad):
    """The learner's replicate-pad integer crop, with one shift shared by all frames."""
    if observations.dtype != torch.uint8 or observations.ndim != 4 or tuple(observations.shape[1:]) != (4, 84, 84):
        raise ValueError("shifts require uint8 BCHW (N, 4, 84, 84)")
    offsets = np.asarray(offsets)
    if (type(pad) is not int or pad < 0 or offsets.shape != (len(observations), 2)
            or offsets.dtype.kind not in "iu" or np.any(offsets < -pad) or np.any(offsets > pad)):
        raise ValueError("invalid integer shift offsets/padding")
    if pad == 0:
        return observations.clone()
    padded = F.pad(observations, (pad, pad, pad, pad), mode="replicate")
    return torch.stack([padded[i, :, pad + int(dy):pad + int(dy) + 84,
                               pad + int(dx):pad + int(dx) + 84]
                        for i, (dy, dx) in enumerate(offsets)])


def perturbations(size, seed):
    variants = {"reverse_past_keep_newest": None, "duplicate_newest": None}
    for amount in (1, 4):
        for dy, dx in ((0, amount), (0, -amount), (amount, 0), (-amount, 0)):
            variants[f"shift_dy{dy}_dx{dx}"] = (4, np.tile([dy, dx], (size, 1)).astype("<i8"))
    rng = np.random.default_rng(seed)
    for pad in (1, 4):
        for repeat in (0, 1):
            variants[f"random_pad{pad}_{repeat}"] = (pad, rng.integers(-pad, pad + 1, (size, 2), dtype=np.int64))
    return variants


@torch.inference_mode()
def predict(model, observations, variant="baseline", shift=None):
    if observations.dtype != np.uint8 or observations.ndim != 4 or observations.shape[1:] != (4, 84, 84):
        raise ValueError("inference requires uint8 BCHW (N, 4, 84, 84)")
    predictions = []
    for offset in range(0, len(observations), BATCH_SIZE):
        pixels = torch.from_numpy(observations[offset:offset + BATCH_SIZE])
        if variant == "reverse_past_keep_newest":
            pixels = pixels[:, [2, 1, 0, 3]]
        elif variant == "duplicate_newest":
            pixels = pixels[:, -1:].expand(-1, 4, -1, -1)
        elif shift is not None:
            pad, offsets = shift
            pixels = shift_uint8(pixels, offsets[offset:offset + BATCH_SIZE], pad)
        elif variant != "baseline":
            raise ValueError("unknown perturbation")
        logits = model.policy(model.trunk(model.encoder(pixels.float() / 255)))
        predictions.append(logits.numpy())
    logits = np.concatenate(predictions)
    if logits.shape != (len(observations), 3) or not np.isfinite(logits).all():
        raise ValueError("invalid actor predictions")
    native = torch.from_numpy(logits).tanh().numpy()
    official = native.copy()
    official[:, 1:] = (official[:, 1:] + 1) * .5
    return logits, native, official


def action_difference(reference, changed):
    logits, native, official = changed
    before_logits, before_native, before_official = reference
    delta = np.abs(official - before_official)
    flips = native[:, 0] * before_native[:, 0] < 0
    return {
        "official_action_mae": delta.astype(np.float64).mean(axis=0).tolist(),
        "steering_abs_delta_p95": float(np.quantile(delta[:, 0], .95)),
        "steering_sign_flip_fraction": float(flips.mean()),
        "opposite_saturated_steering_fraction": float(np.mean(flips & (np.abs(native[:, 0]) >= .99)
                                                            & (np.abs(before_native[:, 0]) >= .99))),
        "steering_logit_abs_delta": statistics(np.abs(logits[:, 0] - before_logits[:, 0])),
        "official_predictions": array_metadata(official),
    }


def build_report(checkpoint, actors, *, sample_size=2048, sample_seed=SAMPLE_SEED):
    if not actors:
        raise ValueError("at least one explicit actor is required")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    observations, source, sample = load_sample(checkpoint, sample_size, sample_seed)
    variants = perturbations(sample_size, sample_seed)
    report = {
        "format": "haic-drq-v2-diagnostics-v1", "diagnostic_only": True,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                    "device": "cpu", "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()},
        "source_sha256": {name: file_sha256(Path(__file__).parent / name) for name in (
            "diagnose_drqv2.py", "agent.py", "drq_v2.py", "common_adapter.py", "action_smoothing.py", "action_representation.py")},
        "checkpoint": source, "sample": sample, "batch_size": BATCH_SIZE,
        "perturbations": {name: ({"pad": shift[0], "offset_order": "dy,dx", "offsets": array_metadata(shift[1])}
                                 if shift is not None else {"kind": name}) for name, shift in variants.items()},
        "limitations": ["All actors use the same source-policy replay states, not their own visitation distributions.",
                        "Frame/shift interventions are not behavioral evaluations or a study decision.",
                        "CPU mapping is diagnostic only; no training continuation or RNG restoration is attempted.",
                        "The 0.46 motor plateau assumes nominal wheel limits; actual wheel states are not retained."],
        "actors": [],
    }
    for path in actors:
        path = Path(path).resolve()
        actor_hash = file_sha256(path)
        runtime = Agent(model_path=str(path))
        if runtime.format != DRQ_ACTOR_FORMAT:
            raise ValueError("diagnostics require an exported DrQ actor")
        baseline = predict(runtime.model, observations)
        logits, native, official = baseline
        derivative = 1 - native[:, 0] ** 2
        row = {
            "path": str(path), "sha256": actor_hash, "config": runtime.export_metadata["config"],
            "sample_indices_sha256": sample["indices"]["sha256"],
            "observations_sha256": sample["observations"]["sha256"],
            "baseline": {
                "steering_abs_ge_099_fraction": float(np.mean(np.abs(native[:, 0]) >= .99)),
                "steering_abs_ge_046_fraction": float(np.mean(np.abs(native[:, 0]) >= .46)),
                "steering_logit": statistics(logits[:, 0]), "absolute_steering_logit": statistics(np.abs(logits[:, 0])),
                "steering_tanh_derivative": statistics(derivative),
                "steering_zero_derivative_fraction": float(np.mean(derivative == 0)),
                "steering_derivative_le_1e_minus4_fraction": float(np.mean(derivative <= 1e-4)),
                "official_action_mean": official.astype(np.float64).mean(axis=0).tolist(),
                "gas": statistics(official[:, 1]), "brake": statistics(official[:, 2]),
                "gas_brake_both_gt_01_fraction": float(np.mean((official[:, 1] > .1) & (official[:, 2] > .1))),
                "logits": array_metadata(logits), "official_predictions": array_metadata(official),
            },
            "perturbations": {},
        }
        random_views = {}
        for name, shift in variants.items():
            changed = predict(runtime.model, observations, name, shift)
            row["perturbations"][name] = action_difference(baseline, changed)
            if name.startswith("random_"):
                random_views[name] = changed
        row["independent_random_view_pairs"] = {
            f"pad{pad}": action_difference(random_views[f"random_pad{pad}_0"], random_views[f"random_pad{pad}_1"])
            for pad in (1, 4)
        }
        if file_sha256(path) != actor_hash:
            raise RuntimeError("actor archive changed during diagnosis")
        report["actors"].append(row)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--actor", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"diagnostics will not overwrite {args.output}")
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {args.output.parent}")
    if (sys.version_info[:2] != (3, 11) or torch.__version__.split("+")[0] != "2.1.0"
            or torch.version.cuda is not None or np.__version__ != "1.26.0"):
        raise RuntimeError("run diagnostics with Python 3.11 / Torch 2.1.0+cpu / NumPy 1.26.0")
    report = build_report(args.checkpoint, args.actor, sample_size=args.sample_size, sample_seed=args.sample_seed)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as handle:
        handle.write(serialized)
    print(json.dumps({"output": str(args.output.resolve()), "sample_size": args.sample_size, "actors": len(args.actor)}))


if __name__ == "__main__":
    main()
