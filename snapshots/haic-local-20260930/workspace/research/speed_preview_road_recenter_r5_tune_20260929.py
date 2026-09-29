"""Fresh matched tune of fixed recenter and joint obstacle-road ZIPs."""

from research import speed_coupled_preview_tune_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-obstacle-road-arbitration.md"
comparison.PLAN_SHA = "e9ba179aaf286c9a72ff05473568dda9d453174b841f9dd5ee15be88fd853347"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r5-tune-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r4-candidate-20260929/submission-speed-preview-road-recenter-r4.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r5-candidate-20260929/submission-speed-preview-road-recenter-r5.zip",
}
comparison.EXPECTED = {
    "preview": "168093ce191011b36e37d3bcfd926c5d3d02f9397deb82ad2b160734541b0b96",
    "preview_speed": "43d961dbcb12b6a54a84aa479f8f1dadbe763097885994b9da00e22953d80312",
}
comparison.CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (2500, 2501, 2502, 2503))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_tune_20260929.py",
    "research/speed_preview_road_recenter_r5_tune_20260929.py",
)
comparison.CHILD = comparison.CHILD.replace(
    "controller = driver._controller\n",
    "controller = driver._controller\n"
    "row['recenter_count'] = getattr(controller, 'recenter_count', 0)\n"
    "row['held_direction_count'] = getattr(controller, 'held_direction_count', 0)\n"
    "row['obstacle_road_count'] = getattr(controller, 'obstacle_road_count', 0)\n"
    "controller = getattr(controller, 'base', controller)\n",
)


if __name__ == "__main__":
    comparison.main()
