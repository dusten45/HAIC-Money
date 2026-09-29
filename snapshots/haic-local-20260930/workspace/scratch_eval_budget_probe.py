"""Ask whether the continuations were limited by training budget.

Two measurements, both on train/tune material only:

1. Replay the source actor on tune episodes, then compare every candidate's
   deterministic action on those same on-distribution pixels, evaluated at the
   source's own pedal expansion. This isolates what PPO changed in the weights
   from what the action transform changed.
2. Score each candidate's weights at expansion 1.0 on the tune split. If the
   score matches the source, the continuation did not move the policy.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from haic_agent.networks import DEFAULT_PEDAL_EXPANSION, VisualActorCritic
from training.site_maps import load_site_map_split
from training.train_policy import _create_environment_for_episode, evaluate_policy

SPLIT = Path("training/maps/site/full_site_map_split.json")
SOURCE = Path(
    "artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt"
)
CANDIDATES = {
    "k2.0_4096steps": Path(
        "artifacts/haic/pedal-expansion2-speed-reward-ppo4096-lr2e6-seed8101/policy.pt"
    ),
    "k1.5_8192steps": Path(
        "artifacts/haic/pedal-expansion1p5-speed-reward-ppo8192-u4-lr2e6-seed8101/policy.pt"
    ),
}


def load(path: Path, expansion: float) -> VisualActorCritic:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    model = VisualActorCritic(
        use_hud=metadata.get("use_hud", True),
        use_visual_features=metadata.get("use_visual_features", False),
        use_temporal_features=metadata.get("use_temporal_features", False),
        pedal_expansion=expansion,
    )
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model


def collect_source_observations(model, episodes, *, decisions: int) -> torch.Tensor:
    frames = []
    for episode in episodes:
        environment = _create_episode(episode, decisions)
        observation, _ = environment.reset()
        try:
            for _ in range(decisions):
                frames.append(torch.from_numpy(np.array(observation, copy=True)))
                with torch.no_grad():
                    action = model.deterministic_actions(
                        model(torch.from_numpy(observation).unsqueeze(0))
                    )
                transition = environment.step_transition(action.squeeze(0).numpy())
                observation = transition.next_observation
                if transition.terminated or transition.truncated:
                    break
        finally:
            environment.close()
    return torch.stack(frames)


def _create_episode(episode, decisions):
    return _create_environment_for_episode(episode, max_decisions=decisions)


def main() -> None:
    split = load_site_map_split(SPLIT)
    tune_episodes = tuple(split.tune)
    source = load(SOURCE, DEFAULT_PEDAL_EXPANSION)

    observations = collect_source_observations(source, tune_episodes[:2], decisions=200)
    print(f"on-distribution frames: {tuple(observations.shape)}", flush=True)
    with torch.no_grad():
        reference = source.deterministic_actions(source(observations))

    report = {
        "frames": int(observations.shape[0]),
        "reference_action_scale": {
            "mean_abs_steer": float(reference[:, 0].abs().mean()),
            "mean_gas": float(reference[:, 1].mean()),
            "mean_brake": float(reference[:, 2].mean()),
        },
        "candidates": {},
    }

    for name, path in CANDIDATES.items():
        matched = load(path, DEFAULT_PEDAL_EXPANSION)
        with torch.no_grad():
            actions = matched.deterministic_actions(matched(observations))
        delta = (actions - reference).abs().mean(0)
        started = time.perf_counter()
        metrics = evaluate_policy(matched, tune_episodes, max_decisions=400)
        metrics["eval_s"] = round(time.perf_counter() - started, 1)
        report["candidates"][name] = {
            "weight_only_action_delta": {
                "mean_abs_steer": float(delta[0]),
                "mean_abs_gas": float(delta[1]),
                "mean_abs_brake": float(delta[2]),
            },
            "tune_at_expansion_1.0": metrics,
        }
        print(json.dumps({name: report["candidates"][name]}, indent=2, sort_keys=True), flush=True)

    Path("tmp").mkdir(exist_ok=True)
    Path("tmp/budget-probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
