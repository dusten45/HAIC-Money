"""Fresh matched tune of fixed obstacle timing ZIPs."""

from research import speed_coupled_preview_tune_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-obstacle-timing-r2.md"
comparison.PLAN_SHA = "44ce84909fb063c6678002151ced3bc3309231bf11e83105d3883885c1cc4544"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r8-tune-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r7-candidate-20260929/submission-speed-preview-road-recenter-r7.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r8-candidate-20260929/submission-speed-preview-road-recenter-r8.zip",
}
comparison.EXPECTED = {
    "preview": "7f113bbfb764830fd2ef4ffde7111e2548be1ab055245d7b00f5280fedf16f6c",
    "preview_speed": "e3e6be15d1c79ceeccd2ae73804c25e0ecc1c8df3b23cec76f0418abc5dd0b03",
}
comparison.CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (2800, 2801, 2802, 2803))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_tune_20260929.py",
    "research/speed_preview_road_recenter_r8_tune_20260929.py",
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
