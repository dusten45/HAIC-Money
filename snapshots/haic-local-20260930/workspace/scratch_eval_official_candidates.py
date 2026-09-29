"""Rank the available checkpoints on the official tracks only.

Official tracks are what the competition scores, and completion is the hard
gate, so rank by official completion first and official lap time second.
Runs every official episode across train/tune/held-out splits the actor has a
recorded result for, at the full evaluation cap.
"""

import json
import time
from pathlib import Path

import torch

from haic_agent.networks import DEFAULT_PEDAL_EXPANSION, VisualActorCritic
from training.site_maps import load_site_map_split
from training.train_policy import _create_environment_for_episode

SPLIT = Path("training/maps/site/full_site_map_split.json")
CANDIDATES = {
    "obstacle-risk (현재 SOTA/제출 ZIP)": Path(
        "artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt"
    ),
    "brake-reward-fullsplit": Path(
        "artifacts/haic/site-map-official-domain-brake-reward-fullsplit-ppo8192-u8-lr2e6-seed8101/policy.pt"
    ),
    "steer-recovery-fullsplit": Path(
        "artifacts/haic/site-map-official-domain-steer-recovery-fullsplit-ppo8192-u8-lr2e6-seed8101/policy.pt"
    ),
}


def load(path: Path) -> VisualActorCritic:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    symmetric = metadata.get("pedal_expansion", DEFAULT_PEDAL_EXPANSION)
    model = VisualActorCritic(
        use_hud=metadata.get("use_hud", True),
        use_visual_features=metadata.get("use_visual_features", False),
        use_temporal_features=metadata.get("use_temporal_features", False),
        throttle_expansion=float(metadata.get("throttle_expansion", symmetric)),
        brake_expansion=float(metadata.get("brake_expansion", symmetric)),
    )
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model


def run_episode(model, episode, *, max_decisions: int) -> dict:
    environment = _create_environment_for_episode(episode, max_decisions=max_decisions)
    observation, _ = environment.reset()
    start = float(getattr(environment.unwrapped, "t", 0.0))
    transition = None
    try:
        for _ in range(max_decisions):
            with torch.no_grad():
                action = model.deterministic_actions(
                    model(torch.from_numpy(observation).unsqueeze(0))
                )
            transition = environment.step_transition(action.squeeze(0).numpy())
            observation = transition.next_observation
            if transition.terminated or transition.truncated:
                break
        raw_finish = getattr(environment.unwrapped, "finish_time_s", None)
        lap = float(raw_finish) - start if raw_finish is not None else None
    finally:
        environment.close()
    labels = transition.labels if transition is not None else None
    return {
        "map_id": episode.site_map.map_id,
        "finished": bool(labels.finished) if labels else False,
        "progress": float(labels.tile_progress) if labels else 0.0,
        "damage": float(labels.damage) if labels else 0.0,
        "lap_time_s": lap,
    }


def main() -> None:
    split = load_site_map_split(SPLIT)
    episodes = [
        episode
        for group in (split.train, split.tune, split.held_out)
        for episode in group
        if str(getattr(getattr(episode, "site_map", None), "map_id", "")).startswith("official")
    ]
    # Train seeds were optimized on; keep them out of the ranking.
    train_ids = {
        str(e.site_map.map_id) for e in split.train if hasattr(e, "site_map")
    }
    scored = [e for e in episodes if str(e.site_map.map_id) not in train_ids]
    print(f"official episodes not in train: {[e.site_map.map_id for e in scored]}", flush=True)

    report = {}
    for name, path in CANDIDATES.items():
        model = load(path)
        started = time.perf_counter()
        rows = [run_episode(model, episode, max_decisions=800) for episode in scored]
        finished = [r for r in rows if r["finished"]]
        laps = sorted(r["lap_time_s"] for r in finished if r["lap_time_s"] is not None)
        report[name] = {
            "checkpoint": str(path),
            "official_completed": len(finished),
            "official_episodes": len(rows),
            "median_lap_s": laps[len(laps) // 2] if laps else None,
            "mean_progress": sum(r["progress"] for r in rows) / len(rows),
            "eval_s": round(time.perf_counter() - started, 1),
            "episodes": rows,
        }
        print(
            f"{name:34s} 공식 완주 {len(finished)}/{len(rows)}"
            f"  중앙 랩 {report[name]['median_lap_s']}"
            f"  평균 진행 {report[name]['mean_progress']:.3f}",
            flush=True,
        )
        for row in rows:
            print(
                f"     {row['map_id']:32s} done={str(row['finished']):5s}"
                f" prog={row['progress']:.3f} dmg={row['damage']:.1f} lap={row['lap_time_s']}",
                flush=True,
            )

    Path("tmp").mkdir(exist_ok=True)
    Path("tmp/official-candidate-ranking.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
