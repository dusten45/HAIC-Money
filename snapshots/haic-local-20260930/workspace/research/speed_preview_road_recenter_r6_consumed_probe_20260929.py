"""Mechanism-only repeat of consumed cells for opposing-steer arbitration."""

import json

from research import speed_coupled_preview_consumed_probe_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-opposing-steer-only.md"
comparison.PLAN_SHA = "8617e48c9406d3b1e8c3a0453050c2444b68041d2a89a70569f4c69c426ce1d4"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r6-consumed-probe-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r5-candidate-20260929/submission-speed-preview-road-recenter-r5.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r6-candidate-20260929/submission-speed-preview-road-recenter-r6.zip",
}
comparison.EXPECTED = {
    "preview": "43d961dbcb12b6a54a84aa479f8f1dadbe763097885994b9da00e22953d80312",
    "preview_speed": "c3979fe29627413a12c21f3a395054f67eccc21f5bf8f24d308625ae6a9653a4",
}
comparison.CELLS = ((2, 2401), (2, 2503))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_consumed_probe_20260929.py",
    "research/speed_preview_road_recenter_r6_consumed_probe_20260929.py",
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
