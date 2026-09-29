"""Mechanism-only repeat of consumed cells for obstacle/road arbitration."""

import json

from research import speed_coupled_preview_consumed_probe_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-obstacle-road-arbitration.md"
comparison.PLAN_SHA = "e9ba179aaf286c9a72ff05473568dda9d453174b841f9dd5ee15be88fd853347"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r5-consumed-probe-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r4-candidate-20260929/submission-speed-preview-road-recenter-r4.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r5-candidate-20260929/submission-speed-preview-road-recenter-r5.zip",
}
comparison.EXPECTED = {
    "preview": "168093ce191011b36e37d3bcfd926c5d3d02f9397deb82ad2b160734541b0b96",
    "preview_speed": "43d961dbcb12b6a54a84aa479f8f1dadbe763097885994b9da00e22953d80312",
}
comparison.CELLS = ((2, 2401), (2, 2101), (3, 2203))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_consumed_probe_20260929.py",
    "research/speed_preview_road_recenter_r5_consumed_probe_20260929.py",
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
    report = json.loads((comparison.RUN / "integration_report.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (comparison.RUN / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    cell = {(row["arm"], row["track_id"], row["seed"]): row for row in rows if row["type"] == "episode"}
    baseline = cell["preview", 2, 2401]
    candidate = cell["preview_speed", 2, 2401]
    assessment = {
        "kind": "consumed_cell_mechanism_diagnostic",
        "fresh_comparison": False,
        "control_all_wheels_off_max_streak": baseline["all_wheels_off_max_streak"],
        "candidate_all_wheels_off_max_streak": candidate["all_wheels_off_max_streak"],
        "candidate_completed": candidate["completed"],
        "candidate_recenter_count": sum(row.get("recenter_count", 0) for row in rows if row.get("arm") == "preview_speed"),
        "candidate_hold_count": sum(row.get("held_direction_count", 0) for row in rows if row.get("arm") == "preview_speed"),
        "candidate_obstacle_road_count": sum(row.get("obstacle_road_count", 0) for row in rows if row.get("arm") == "preview_speed"),
        "candidate_only_failures": report["candidate_only_failures"],
        "mechanism_probe_pass": bool(candidate.get("obstacle_road_count", 0) > 0 and not report["candidate_only_failures"] and not report["material_road_exposure_cells"]),
    }
    (comparison.RUN / "diagnostic_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2), flush=True)
