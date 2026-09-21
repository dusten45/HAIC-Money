import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from action_smoothing import (
    action_control_fingerprint,
    action_smoothing_fingerprint,
    build_action_smoother,
    normalize_action_smoothing,
    normalize_action_control,
    read_embedded_action_smoothing,
)
from action_representation import (
    action_representation_fingerprint,
    normalize_action_representation,
)
from agent import Baseline1Actor


ACTOR_STATE_KEYS = (
    "features_extractor.cnn.0.weight",
    "features_extractor.cnn.0.bias",
    "features_extractor.cnn.2.weight",
    "features_extractor.cnn.2.bias",
    "features_extractor.cnn.4.weight",
    "features_extractor.cnn.4.bias",
    "features_extractor.linear.0.weight",
    "features_extractor.linear.0.bias",
    "action_net.weight",
    "action_net.bias",
)


def load_sb3_policy_state(archive_path: Path):
    with zipfile.ZipFile(archive_path) as archive:
        try:
            payload = archive.read("policy.pth")
        except KeyError as error:
            raise ValueError(f"policy.pth is missing from {archive_path}") from error
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def extract_actor_state(policy_state):
    missing = [key for key in ACTOR_STATE_KEYS if key not in policy_state]
    if missing:
        raise ValueError(f"SB3 policy is not the expected CnnPolicy: missing {missing}")
    return {key: policy_state[key].detach().cpu() for key in ACTOR_STATE_KEYS}


def build_actor(archive_path: Path):
    state = extract_actor_state(load_sb3_policy_state(archive_path))
    input_channels = int(state["features_extractor.cnn.0.weight"].shape[1])
    action_dimensions = int(state["action_net.bias"].shape[0])
    actor = Baseline1Actor(
        input_channels=input_channels,
        action_dimensions=action_dimensions,
    )
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def source_action_smoothing(archive_path: Path, encoded_config: str | None = None):
    embedded = read_embedded_action_smoothing(archive_path)
    external = None
    if encoded_config:
        external = normalize_action_smoothing(json.loads(encoded_config))
    else:
        run_dir = (
            archive_path.parent.parent
            if archive_path.parent.name == "checkpoints"
            else archive_path.parent
        )
        config_path = run_dir / "config.json"
        if config_path.is_file():
            recorded = json.loads(config_path.read_text())
            external = normalize_action_smoothing(
                recorded.get("config", {}).get("action_smoothing")
            )
        elif archive_path.parent.name == "candidates":
            candidates_path = archive_path.parent.parent / "candidates.json"
            if not candidates_path.is_file():
                raise ValueError(
                    "immutable evaluation snapshot requires --action-smoothing-config"
                )
            candidates = json.loads(candidates_path.read_text())
            for candidate in candidates:
                if Path(candidate.get("evaluation_archive_path", "")).name == archive_path.name:
                    if candidate.get("action_smoothing_present", True):
                        external = normalize_action_smoothing(candidate.get("action_smoothing"))
                    break
            else:
                raise ValueError(
                    "immutable evaluation snapshot is missing action smoothing provenance"
                )
    if embedded is not None and external is not None:
        if action_smoothing_fingerprint(embedded) != action_smoothing_fingerprint(external):
            raise ValueError("checkpoint and external action smoothing configs do not match")
    return embedded or external or normalize_action_smoothing()


def source_action_control(archive_path: Path):
    run_dir = (
        archive_path.parent.parent
        if archive_path.parent.name == "checkpoints"
        else archive_path.parent
    )
    config_path = run_dir / "config.json"
    if config_path.is_file():
        recorded = json.loads(config_path.read_text())
        return normalize_action_control(recorded.get("config", {}).get("action_control"))
    if archive_path.parent.name == "candidates":
        candidates_path = archive_path.parent.parent / "candidates.json"
        if not candidates_path.is_file():
            raise ValueError("immutable evaluation snapshot requires action control provenance")
        for candidate in json.loads(candidates_path.read_text()):
            if Path(candidate.get("evaluation_archive_path", "")).name == archive_path.name:
                return normalize_action_control(candidate.get("action_control"))
    return normalize_action_control()


def source_action_representation(archive_path: Path):
    run_dir = archive_path.parent.parent if archive_path.parent.name == "checkpoints" else archive_path.parent
    config_path = run_dir / "config.json"
    if config_path.is_file():
        recorded = json.loads(config_path.read_text())
        return normalize_action_representation(
            recorded.get("config", {}).get("action_representation")
        )
    return normalize_action_representation()


def export_payload(actor: Baseline1Actor, action_smoothing=None, action_control=None, action_representation=None):
    action_smoothing = normalize_action_smoothing(action_smoothing)
    action_control = normalize_action_control(action_control)
    action_representation = normalize_action_representation(action_representation)
    if actor.input_channels != action_control["input_channels"]:
        raise ValueError("actor input channels do not match action control config")
    expected_dimensions = (
        sum(action_representation["nvec"])
        if action_representation["method"] != "continuous_box"
        else 3
    )
    if actor.action_dimensions != expected_dimensions:
        raise ValueError("actor action head does not match action representation")
    return {
        "state_dict": actor.state_dict(),
        "action_smoothing": action_smoothing,
        "action_smoothing_fingerprint": action_smoothing_fingerprint(action_smoothing),
        "action_control": action_control,
        "action_control_fingerprint": action_control_fingerprint(action_control),
        "input_channels": actor.input_channels,
        "action_representation": action_representation,
        "action_representation_fingerprint": action_representation_fingerprint(action_representation),
    }


def verify_smoothing(actor: Baseline1Actor, action_smoothing, samples: int):
    observations = verification_observations(samples, actor.input_channels)
    smoother = build_action_smoother(action_smoothing)
    with torch.inference_mode():
        raw_actions = actor.predict_action(
            torch.as_tensor(observations, dtype=torch.float32)
        ).cpu().numpy()
    smoothed = np.asarray(
        [smoother.smooth(action) for action in raw_actions], dtype=np.float32
    )
    if smoothed.shape != raw_actions.shape or not np.isfinite(smoothed).all():
        raise ValueError("action smoothing produced an invalid export trace")
    return float(np.max(np.abs(smoothed - raw_actions)))


def verification_observations(samples: int, input_channels=4):
    if samples < 4:
        raise ValueError("--verify-samples must be at least 4")
    ramp = np.linspace(0.0, 1.0, input_channels * 84 * 84, dtype=np.float32).reshape(input_channels, 84, 84)
    random = np.random.default_rng(0).random(
        (samples - 3, input_channels, 84, 84), dtype=np.float32
    )
    return np.concatenate(
        (
            np.zeros((1, input_channels, 84, 84), dtype=np.float32),
            np.ones((1, input_channels, 84, 84), dtype=np.float32),
            ramp[np.newaxis],
            random,
        )
    )


def verify_parity(archive_path: Path, actor: Baseline1Actor, samples: int):
    observations = verification_observations(samples, actor.input_channels)
    source = PPO.load(str(archive_path), device="cpu")
    with torch.inference_mode():
        source_tensor, _ = source.policy.obs_to_tensor(observations)
        source_mean = source.policy.get_distribution(source_tensor).distribution.mean
        actor_mean = actor(source_tensor)
        actor_actions = actor.predict_action(source_tensor).cpu().numpy()
    source_actions, _ = source.predict(observations, deterministic=True)

    np.testing.assert_allclose(actor_mean.cpu().numpy(), source_mean.cpu().numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actor_actions, source_actions, atol=1e-6, rtol=1e-6)
    return float(np.max(np.abs(actor_actions - source_actions)))


def benchmark_actor(actor: Baseline1Actor, calls: int):
    if calls <= 0:
        return None
    observation = np.random.default_rng(1).random((actor.input_channels, 84, 84), dtype=np.float32)

    with torch.inference_mode():
        actor.predict_action(torch.as_tensor(observation).unsqueeze(0))

    timings = []
    for _ in range(calls):
        start = time.perf_counter()
        with torch.inference_mode():
            actor.predict_action(torch.as_tensor(observation).unsqueeze(0))
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "max_ms": float(np.max(timings)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Export a deterministic SB3 CnnPolicy actor")
    parser.add_argument("--input", type=Path, required=True, help="SB3 model .zip path")
    parser.add_argument("--output", type=Path, default=Path("model.pt"))
    parser.add_argument("--verify-samples", type=int, default=256)
    parser.add_argument("--benchmark-calls", type=int, default=500)
    parser.add_argument(
        "--action-smoothing-config",
        help="JSON smoothing config; default reads the source run config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    actor = build_actor(args.input)
    action_smoothing = source_action_smoothing(args.input, args.action_smoothing_config)
    action_control = source_action_control(args.input)
    action_representation = source_action_representation(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        export_payload(actor, action_smoothing, action_control, action_representation),
        args.output,
    )
    print(f"exported actor: {args.output}")

    action_error = verify_parity(args.input, actor, args.verify_samples)
    print(
        f"parity passed: samples={args.verify_samples} "
        f"max_action_abs_error={action_error:.2e}"
    )
    smoothing_error = verify_smoothing(actor, action_smoothing, args.verify_samples)
    print(
        f"smoothing trace passed: fingerprint="
        f"{action_smoothing_fingerprint(action_smoothing)[:16]} "
        f"max_delta={smoothing_error:.2e}"
    )

    benchmark = benchmark_actor(actor, args.benchmark_calls)
    if benchmark is not None:
        print(
            f"cpu action latency: calls={args.benchmark_calls} "
            f"mean_ms={benchmark['mean_ms']:.3f} "
            f"p95_ms={benchmark['p95_ms']:.3f} "
            f"max_ms={benchmark['max_ms']:.3f}"
        )


if __name__ == "__main__":
    main()
