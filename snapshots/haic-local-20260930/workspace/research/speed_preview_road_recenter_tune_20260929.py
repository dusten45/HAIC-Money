"""Fresh matched tune of fixed speed-coupled and road-recenter ZIPs."""

from research import speed_coupled_preview_tune_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-road-recenter.md"
comparison.PLAN_SHA = "6c44bcb3a8e828ddf604dffa4d3f33b8bdf475ba0f4c8328ade754ed0cb7fa46"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-tune-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-coupled-preview-candidate-20260929/submission-speed-coupled-preview.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-candidate-20260929/submission-speed-preview-road-recenter.zip",
}
comparison.EXPECTED = {
    "preview": "54c117287390704acc52505d680c31574fdcf53ac062cbb0ce5258a57058c324",
    "preview_speed": "711729b6ea3ea2cc981a479c6f2405d5b968230462884b6361442ece1a0a210f",
}
comparison.CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (2100, 2101, 2102, 2103))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_tune_20260929.py",
    "research/speed_preview_road_recenter_tune_20260929.py",
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
