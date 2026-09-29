"""Mechanism-only repeat of consumed cells for road recentering."""

import json

from research import speed_coupled_preview_consumed_probe_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-road-recenter.md"
comparison.PLAN_SHA = "6c44bcb3a8e828ddf604dffa4d3f33b8bdf475ba0f4c8328ade754ed0cb7fa46"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-consumed-probe-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-coupled-preview-candidate-20260929/submission-speed-coupled-preview.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-candidate-20260929/submission-speed-preview-road-recenter.zip",
}
comparison.EXPECTED = {
    "preview": "54c117287390704acc52505d680c31574fdcf53ac062cbb0ce5258a57058c324",
    "preview_speed": "711729b6ea3ea2cc981a479c6f2405d5b968230462884b6361442ece1a0a210f",
}
comparison.CELLS = ((1, 2002), (2, 1700), (1, 1920), (2, 1921))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_consumed_probe_20260929.py",
    "research/speed_preview_road_recenter_consumed_probe_20260929.py",
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
    report = json.loads((comparison.RUN / "integration_report.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (comparison.RUN / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    cell = {(row["arm"], row["track_id"], row["seed"]): row for row in rows if row["type"] == "episode"}
    baseline = cell["preview", 1, 2002]
    candidate = cell["preview_speed", 1, 2002]
    assessment = {
        "kind": "consumed_cell_mechanism_diagnostic",
        "fresh_comparison": False,
        "control_all_wheels_off_max_streak": baseline["all_wheels_off_max_streak"],
        "candidate_all_wheels_off_max_streak": candidate["all_wheels_off_max_streak"],
        "candidate_completed": candidate["completed"],
        "candidate_recenter_count": sum(row.get("recenter_count", 0) for row in rows if row.get("arm") == "preview_speed"),
        "candidate_hold_count": sum(row.get("held_direction_count", 0) for row in rows if row.get("arm") == "preview_speed"),
        "candidate_only_failures": report["candidate_only_failures"],
        "mechanism_probe_pass": bool(candidate["completed"] and candidate["all_wheels_off_max_streak"] <= 10 and not report["candidate_only_failures"] and sum(row.get("recenter_count", 0) for row in rows if row.get("arm") == "preview_speed") > 0),
    }
    (comparison.RUN / "diagnostic_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2), flush=True)
