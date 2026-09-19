import argparse
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

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
    actor = Baseline1Actor()
    actor.load_state_dict(extract_actor_state(load_sb3_policy_state(archive_path)), strict=True)
    actor.eval()
    return actor


def verification_observations(samples: int):
    if samples < 4:
        raise ValueError("--verify-samples must be at least 4")
    ramp = np.linspace(0.0, 1.0, 4 * 84 * 84, dtype=np.float32).reshape(4, 84, 84)
    random = np.random.default_rng(0).random(
        (samples - 3, 4, 84, 84), dtype=np.float32
    )
    return np.concatenate(
        (
            np.zeros((1, 4, 84, 84), dtype=np.float32),
            np.ones((1, 4, 84, 84), dtype=np.float32),
            ramp[np.newaxis],
            random,
        )
    )


def verify_parity(archive_path: Path, actor: Baseline1Actor, samples: int):
    observations = verification_observations(samples)
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
    observation = np.random.default_rng(1).random((4, 84, 84), dtype=np.float32)

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
    return parser.parse_args()


def main():
    args = parse_args()
    actor = build_actor(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), args.output)
    print(f"exported actor: {args.output}")

    action_error = verify_parity(args.input, actor, args.verify_samples)
    print(
        f"parity passed: samples={args.verify_samples} "
        f"max_action_abs_error={action_error:.2e}"
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
