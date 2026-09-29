"""Held-out comparison of the source actor and the pedal-expanded candidate.

Run only after the candidate won on tune, per RULES.md step 7.
"""

import json
import time
from pathlib import Path

import torch

from haic_agent.networks import DEFAULT_PEDAL_EXPANSION, VisualActorCritic
from training.site_maps import load_site_map_split
from training.train_policy import evaluate_policy

SPLIT = Path("training/maps/site/full_site_map_split.json")
CANDIDATES = {
    "source_expansion1": Path(
        "artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt"
    ),
    "pedal_expansion2_speed_reward": Path(
        "artifacts/haic/pedal-expansion2-speed-reward-ppo4096-lr2e6-seed8101/policy.pt"
    ),
}


def main() -> None:
    episodes = tuple(load_site_map_split(SPLIT).held_out)
    results = {}
    for name, checkpoint in CANDIDATES.items():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = saved["metadata"]
        model = VisualActorCritic(
            use_hud=metadata.get("use_hud", True),
            use_visual_features=metadata.get("use_visual_features", False),
            use_temporal_features=metadata.get("use_temporal_features", False),
            pedal_expansion=float(
                metadata.get("pedal_expansion", DEFAULT_PEDAL_EXPANSION)
            ),
        )
        model.load_state_dict(saved["model_state"])
        model.eval()
        started = time.perf_counter()
        metrics = evaluate_policy(model, episodes, max_decisions=600)
        metrics["eval_s"] = round(time.perf_counter() - started, 1)
        metrics["throttle_limit"] = model.throttle_limit
        metrics["checkpoint"] = str(checkpoint)
        results[name] = metrics
        print(json.dumps({name: metrics}, indent=2, sort_keys=True), flush=True)

    Path("tmp").mkdir(exist_ok=True)
    Path("tmp/pedal-expansion-heldout.json").write_text(
        json.dumps(
            {
                "split": str(SPLIT),
                "held_out_episodes": len(episodes),
                "max_decisions": 600,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
