"""Trace the consumed track-2 regression after obstacle-road arbitration."""

from research import speed_coupled_preview_road_excursion_trace_20260929 as trace


trace.PACKAGES = {
    "preview": trace.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r4-candidate-20260929/submission-speed-preview-road-recenter-r4.zip",
    "preview_speed": trace.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r5-candidate-20260929/submission-speed-preview-road-recenter-r5.zip",
}
trace.RUN = trace.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r5-regression-trace-20260929"
trace.CELLS = ((2, 2503),)
trace.TRACE_CHILD = trace.TRACE_CHILD.replace(
    "row['decision_trace'] = raw['decision_trace']",
    "row['decision_trace'] = raw['decision_trace']\n"
    "row['recenter_count'] = getattr(driver._controller, 'recenter_count', 0)\n"
    "row['held_direction_count'] = getattr(driver._controller, 'held_direction_count', 0)",
)


if __name__ == "__main__":
    trace.main()
