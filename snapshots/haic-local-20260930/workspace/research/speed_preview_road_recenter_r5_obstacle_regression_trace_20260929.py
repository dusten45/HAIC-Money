"""Inspect pixel obstacle placement at the consumed r5 regression."""

from research.speed_preview_road_recenter_r5_regression_trace_20260929 import trace


trace.RUN = trace.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r5-obstacle-regression-trace-20260929"
trace.TRACE_CHILD = trace.TRACE_CHILD.replace(
    "self.road_samples.append({'off_wheels': off_count",
    "self.road_samples.append({'obstacle_diag': driver._controller.base.base.base.corridor.last_step_diagnostics(), 'off_wheels': off_count",
)


if __name__ == "__main__":
    trace.main()
