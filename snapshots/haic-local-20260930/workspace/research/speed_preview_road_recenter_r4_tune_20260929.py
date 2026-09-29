"""Fresh matched tune of fixed first and conditional recenter ZIPs."""

from research import speed_coupled_preview_tune_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-road-recenter-r4.md"
comparison.PLAN_SHA = "405c77ceb93d622aab2d7f60f9427a07aa9565b0daf2aad95c6e3fac4f4a97bc"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r4-tune-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-candidate-20260929/submission-speed-preview-road-recenter.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r4-candidate-20260929/submission-speed-preview-road-recenter-r4.zip",
}
comparison.EXPECTED = {
    "preview": "711729b6ea3ea2cc981a479c6f2405d5b968230462884b6361442ece1a0a210f",
    "preview_speed": "168093ce191011b36e37d3bcfd926c5d3d02f9397deb82ad2b160734541b0b96",
}
comparison.CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (2400, 2401, 2402, 2403))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_tune_20260929.py",
    "research/speed_preview_road_recenter_r4_tune_20260929.py",
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
