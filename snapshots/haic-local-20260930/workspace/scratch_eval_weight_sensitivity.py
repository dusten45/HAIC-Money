"""Control experiment for the 'we just need more training' hypothesis.

The two continuations moved the actor's weights by 0.13% and 0.27% yet swung
tune completion from 4/6 to 2/6 and 1/6. If random perturbations of the same
norm swing it just as much, the tune score is not measuring what PPO learned;
it is measuring where on a jagged surface the run happened to stop.

Train/tune material only.
"""

import json
import time
from pathlib import Path

import torch

from haic_agent.networks import DEFAULT_PEDAL_EXPANSION, VisualActorCritic
from training.site_maps import load_site_map_split
from training.train_policy import evaluate_policy

SPLIT = Path("training/maps/site/full_site_map_split.json")
SOURCE = Path(
    "artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt"
)
# Matched to |dtheta| of the k=1.5 / 8,192-step continuation.
PERTURBATION_NORM = 0.0495
SEEDS = (101, 202, 303)


def build(state: dict[str, torch.Tensor]) -> VisualActorCritic:
    model = VisualActorCritic(
        use_hud=False,
        use_visual_features=True,
        pedal_expansion=DEFAULT_PEDAL_EXPANSION,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def perturbed_state(
    state: dict[str, torch.Tensor], *, seed: int, norm: float
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    noise = {
        name: torch.randn(value.shape, generator=generator, dtype=torch.float32)
        for name, value in state.items()
        if value.is_floating_point()
    }
    total = sum(float(tensor.pow(2).sum()) for tensor in noise.values()) ** 0.5
    scale = norm / total
    return {
        name: (value.float() + noise[name] * scale) if name in noise else value
        for name, value in state.items()
    }


def main() -> None:
    episodes = tuple(load_site_map_split(SPLIT).tune)
    state = torch.load(SOURCE, map_location="cpu", weights_only=False)["model_state"]

    results = {}
    started = time.perf_counter()
    results["source"] = evaluate_policy(build(state), episodes, max_decisions=400)
    results["source"]["eval_s"] = round(time.perf_counter() - started, 1)
    print(json.dumps({"source": results["source"]}, indent=2, sort_keys=True), flush=True)

    for seed in SEEDS:
        model = build(perturbed_state(state, seed=seed, norm=PERTURBATION_NORM))
        started = time.perf_counter()
        metrics = evaluate_policy(model, episodes, max_decisions=400)
        metrics["eval_s"] = round(time.perf_counter() - started, 1)
        results[f"random_perturbation_seed{seed}"] = metrics
        print(
            json.dumps({f"random_perturbation_seed{seed}": metrics}, indent=2, sort_keys=True),
            flush=True,
        )

    Path("tmp").mkdir(exist_ok=True)
    Path("tmp/weight-sensitivity.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(SOURCE),
                "perturbation_norm": PERTURBATION_NORM,
                "tune_episodes": len(episodes),
                "max_decisions": 400,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
