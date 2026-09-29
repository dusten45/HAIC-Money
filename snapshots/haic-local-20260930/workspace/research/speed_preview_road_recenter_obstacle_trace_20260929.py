"""Inspect pixel obstacle diagnostics during the consumed shared failure."""

from research.speed_preview_road_recenter_r4_failure_trace_20260929 import trace


trace.RUN = trace.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-obstacle-trace-20260929"
trace.TRACE_CHILD = trace.TRACE_CHILD.replace(
    "self.road_samples.append({'off_wheels': off_count",
    "self.road_samples.append({'obstacle_diag': driver._controller.base.base.base.corridor.last_step_diagnostics(), 'off_wheels': off_count",
)


if __name__ == "__main__":
    trace.main()
