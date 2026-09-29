"""Fresh matched tune of fixed first and second recenter ZIPs."""

from research import speed_coupled_preview_tune_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-road-recenter-r2.md"
comparison.PLAN_SHA = "b94930bdc4a7a08d692bf9fca34dc5b6900e7770210064b32b2f3c7a3b7c2ef6"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r2-tune-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-candidate-20260929/submission-speed-preview-road-recenter.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r2-candidate-20260929/submission-speed-preview-road-recenter-r2.zip",
}
comparison.EXPECTED = {
    "preview": "711729b6ea3ea2cc981a479c6f2405d5b968230462884b6361442ece1a0a210f",
    "preview_speed": "673415f4effc808b54f36184b8cd77423a3a153a1ce8684d2537c2490b43dff0",
}
comparison.CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (2200, 2201, 2202, 2203))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_tune_20260929.py",
    "research/speed_preview_road_recenter_r2_tune_20260929.py",
)
comparison.CHILD = comparison.CHILD.replace(
    "controller = driver._controller\n",
    "controller = driver._controller\n"
    "row['recenter_count'] = getattr(controller, 'recenter_count', 0)\n"
    "row['held_direction_count'] = getattr(controller, 'held_direction_count', 0)\n"
    "controller = getattr(controller, 'base', controller)\n",
)


if __name__ == "__main__":
    comparison.main()
