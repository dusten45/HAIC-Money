"""Trace the consumed rescue lost by the opposing-steer-only gate."""

from research import speed_coupled_preview_road_excursion_trace_20260929 as trace


trace.PACKAGES = {
    "preview": trace.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r5-candidate-20260929/submission-speed-preview-road-recenter-r5.zip",
    "preview_speed": trace.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r6-candidate-20260929/submission-speed-preview-road-recenter-r6.zip",
}
trace.RUN = trace.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r6-divergence-trace-20260929"
trace.CELLS = ((2, 2401),)
trace.TRACE_CHILD = trace.TRACE_CHILD.replace(
    "row['decision_trace'] = raw['decision_trace']",
    "row['decision_trace'] = raw['decision_trace']\n"
    "row['recenter_count'] = getattr(driver._controller, 'recenter_count', 0)\n"
    "row['held_direction_count'] = getattr(driver._controller, 'held_direction_count', 0)",
)


if __name__ == "__main__":
    trace.main()
