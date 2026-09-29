"""Trace consumed r5 versus r7 pixel-obstacle timing divergence."""

from research.speed_preview_road_recenter_r5_regression_trace_20260929 import trace


trace.PACKAGES = {
    "preview": trace.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r5-candidate-20260929/submission-speed-preview-road-recenter-r5.zip",
    "preview_speed": trace.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r7-candidate-20260929/submission-speed-preview-road-recenter-r7.zip",
}
trace.RUN = trace.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r7-divergence-trace-20260929"
trace.CELLS = ((2, 2401),)
trace.TRACE_CHILD = trace.TRACE_CHILD.replace(
    "self.road_samples.append({'off_wheels': off_count",
    "self.road_samples.append({'obstacle_diag': driver._controller.base.base.base.corridor.last_step_diagnostics(), 'off_wheels': off_count",
)


if __name__ == "__main__":
    trace.main()
