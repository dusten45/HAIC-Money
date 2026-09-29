"""How much lap time does throttle headroom buy on the official tracks?

Zero-shot: one fixed weight set, only the action transform changes. Run on the
two official episodes the current actor actually completes, so lap time is
defined. Diagnostic, not a promotion claim.
"""

import json
import time
from pathlib import Path

import torch

from haic_agent.networks import PedalScale, VisualActorCritic
from training.site_maps import load_site_map_split
from training.train_policy import evaluate_policy

CHECKPOINT = Path(
    "artifacts/haic/site-map-official-domain-steer-recovery-fullsplit-ppo8192-u8-lr2e6-seed8101/policy.pt"
)
SPLIT = Path("training/maps/site/full_site_map_split.json")
# track1 seed48 (tune) and track2 seed101 (held out) are the official episodes
# the current actor completes, so a lap time exists to compare.
TARGET_MAPS = ("official-track1-seed48", "official-track2-seed101")
SCALES = (
    PedalScale(1.0, 1.0),
    PedalScale(2.0, 1.0),
    PedalScale(3.5, 1.0),
    PedalScale(3.5, 3.5),
    PedalScale(6.0, 3.0),
    PedalScale(8.333333333333334, 3.571428571428571),
)


def main() -> None:
    split = load_site_map_split(SPLIT)
    episodes = [
        episode
        for group in (split.train, split.tune, split.held_out)
        for episode in group
        if getattr(getattr(episode, "site_map", None), "map_id", None) in TARGET_MAPS
    ]
    print("episodes:", [e.site_map.map_id for e in episodes], flush=True)

    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    results = {}
    for scale in SCALES:
        model = VisualActorCritic(
            use_hud=metadata.get("use_hud", True),
            use_visual_features=metadata.get("use_visual_features", False),
            use_temporal_features=metadata.get("use_temporal_features", False),
            throttle_expansion=scale.throttle_expansion,
            brake_expansion=scale.brake_expansion,
        )
        model.load_state_dict(saved["model_state"])
        model.eval()
        started = time.perf_counter()
        metrics = evaluate_policy(model, episodes, max_decisions=800)
        metrics["eval_s"] = round(time.perf_counter() - started, 1)
        metrics["throttle_limit"] = round(model.throttle_limit, 4)
        metrics["brake_limit"] = round(model.brake_limit, 4)
        key = f"gas{model.throttle_limit:.2f}_brake{model.brake_limit:.2f}"
        results[key] = metrics
        print(
            f"{key:24s} 완주 {metrics['completed_episodes']:.0f}/{metrics['episodes']:.0f}"
            f"  중앙 랩 {metrics['median_finished_lap_time_s']}"
            f"  평균속도 {metrics['mean_speed']:.2f}"
            f"  최고속도 {metrics['max_speed']:.2f}"
            f"  진행률 {metrics['mean_progress']:.3f}",
            flush=True,
        )

    Path("tmp").mkdir(exist_ok=True)
    Path("tmp/official-throttle-sweep.json").write_text(
        json.dumps(
            {"checkpoint": str(CHECKPOINT), "maps": list(TARGET_MAPS), "results": results},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
