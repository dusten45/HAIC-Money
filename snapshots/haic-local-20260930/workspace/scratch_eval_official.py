import json
import sys
from pathlib import Path

from training.evaluate_closed_loop import make_agent, run_episode
from training.site_maps import SiteMapEpisode, SiteMapSpec, load_site_map

checkpoint = Path(sys.argv[1])

official_plain = SiteMapSpec(
    map_id="official-track1-seed42",
    map_kind="official",
    track_id=1,
    seed=42,
    obstacle_mode="official",
    obstacles=(),
    max_steps=2000,
    frame_skip=4,
)
official_plus_obstacle = load_site_map(
    "training/maps/site/official-track1-seed42-plus-obstacle.json"
)

results = {}
for label, site_map in [
    ("official-track1-seed42", official_plain),
    ("official-track1-seed42-plus-obstacle", official_plus_obstacle),
]:
    episode = SiteMapEpisode(site_map=site_map, seed=42, source_path=Path("."))
    agent = make_agent(
        policy_checkpoint=checkpoint,
        dynamics_checkpoint=None,
        plan_budget_seconds=4.5,
        planner_settings={},
    )
    result = run_episode(
        mode="ppo_only",
        episode=episode,
        agent=agent,
        max_decisions=600,
        plan_budget_seconds=4.5,
        capture_trace=True,
    )
    results[label] = {
        "completed": result["completed"],
        "progress": result["progress"],
        "collisions": result["collisions"],
        "damage": result["damage"],
        "retire_reason": result["retire_reason"],
        "lapTimeMs": result["lapTimeMs"],
    }
    trace_path = Path(f"/tmp_trace_{label}.json") if False else Path(f"scratch_trace_{label}.json")
    trace_path.write_text(json.dumps(result["decision_trace"]), encoding="utf-8")

print(json.dumps(results, indent=2))
