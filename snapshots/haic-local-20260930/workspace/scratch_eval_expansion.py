"""Measure how the pedal expansion changes lap time for one fixed weight set.

Train/tune episodes only: this is an engineering check that the action
transform is what limits speed, not a held-out performance claim.
"""

import json
import sys
import time
from pathlib import Path

import torch

from haic_agent.networks import VisualActorCritic
from training.site_maps import load_site_map_split
from training.train_policy import evaluate_policy

CHECKPOINT = Path(
    "artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt"
)
SPLIT = Path("training/maps/site/full_site_map_split.json")


def main() -> None:
    expansions = [float(value) for value in sys.argv[1:]] or [1.0, 1.5, 2.0]
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    split = load_site_map_split(SPLIT)
    episodes = tuple(split.tune)[:4]
    print(f"tune episodes: {len(episodes)}", flush=True)

    results = {}
    for expansion in expansions:
        model = VisualActorCritic(
            use_hud=metadata.get("use_hud", True),
            use_visual_features=metadata.get("use_visual_features", False),
            use_temporal_features=metadata.get("use_temporal_features", False),
            pedal_expansion=expansion,
        )
        model.load_state_dict(saved["model_state"])
        model.eval()
        started = time.perf_counter()
        metrics = evaluate_policy(model, episodes, max_decisions=600)
        metrics["eval_s"] = round(time.perf_counter() - started, 1)
        metrics["throttle_limit"] = model.throttle_limit
        results[str(expansion)] = metrics
        print(json.dumps({str(expansion): metrics}, indent=2, sort_keys=True), flush=True)

    Path("tmp").mkdir(exist_ok=True)
    Path("tmp/pedal-expansion-tune-sweep.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
