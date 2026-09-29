"""Trace the consumed track-2 road excursion in both fixed ZIPs."""

from research import speed_coupled_preview_road_excursion_trace_20260929 as trace


trace.PACKAGES = {
    "preview": trace.ROOT / "artifacts/haic-research-v2/speed-coupled-preview-candidate-20260929/submission-speed-coupled-preview.zip",
    "preview_speed": trace.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-candidate-20260929/submission-speed-preview-road-recenter.zip",
}
trace.RUN = trace.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-track2-trace-20260929"
trace.CELLS = ((2, 2101),)
trace.TRACE_CHILD = trace.TRACE_CHILD.replace(
    "row['decision_trace'] = raw['decision_trace']",
    "row['decision_trace'] = raw['decision_trace']\n"
    "row['recenter_count'] = getattr(driver._controller, 'recenter_count', 0)\n"
    "row['held_direction_count'] = getattr(driver._controller, 'held_direction_count', 0)",
)


if __name__ == "__main__":
    trace.main()
