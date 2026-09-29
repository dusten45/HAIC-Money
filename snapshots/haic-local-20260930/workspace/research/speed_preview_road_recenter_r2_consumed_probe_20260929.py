"""Mechanism-only repeat of consumed cells for earlier recentering."""

import json

from research import speed_coupled_preview_consumed_probe_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-road-recenter-r2.md"
comparison.PLAN_SHA = "b94930bdc4a7a08d692bf9fca34dc5b6900e7770210064b32b2f3c7a3b7c2ef6"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r2-consumed-probe-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-candidate-20260929/submission-speed-preview-road-recenter.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r2-candidate-20260929/submission-speed-preview-road-recenter-r2.zip",
}
comparison.EXPECTED = {
    "preview": "711729b6ea3ea2cc981a479c6f2405d5b968230462884b6361442ece1a0a210f",
    "preview_speed": "673415f4effc808b54f36184b8cd77423a3a153a1ce8684d2537c2490b43dff0",
}
comparison.CELLS = ((2, 2101), (2, 2103), (1, 2002))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_consumed_probe_20260929.py",
    "research/speed_preview_road_recenter_r2_consumed_probe_20260929.py",
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
    baseline = cell["preview", 2, 2101]
    candidate = cell["preview_speed", 2, 2101]
    assessment = {
        "kind": "consumed_cell_mechanism_diagnostic",
        "fresh_comparison": False,
        "control_all_wheels_off_max_streak": baseline["all_wheels_off_max_streak"],
        "candidate_all_wheels_off_max_streak": candidate["all_wheels_off_max_streak"],
        "candidate_completed": candidate["completed"],
        "candidate_recenter_count": sum(row.get("recenter_count", 0) for row in rows if row.get("arm") == "preview_speed"),
        "candidate_hold_count": sum(row.get("held_direction_count", 0) for row in rows if row.get("arm") == "preview_speed"),
        "candidate_only_failures": report["candidate_only_failures"],
        "mechanism_probe_pass": bool(candidate["completed"] and candidate["all_wheels_off_max_streak"] < 20 and not report["candidate_only_failures"] and sum(row.get("recenter_count", 0) for row in rows if row.get("arm") == "preview_speed") > 0),
    }
    (comparison.RUN / "diagnostic_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2), flush=True)
