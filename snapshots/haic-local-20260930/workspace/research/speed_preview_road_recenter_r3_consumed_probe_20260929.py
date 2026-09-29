"""Mechanism-only repeat of consumed cells for momentum-preserving recentering."""

import json

from research import speed_coupled_preview_consumed_probe_20260929 as comparison


comparison.PLAN = comparison.ROOT / "docs/plans/2026-09-29-speed-preview-road-recenter-r3.md"
comparison.PLAN_SHA = "2ab03fc5d77da239df2914c113b2657493b65291bf161af7853ecdb098ecd107"
comparison.RUN = comparison.ROOT / "runs/haic-research-v2/speed-preview-road-recenter-r3-consumed-probe-20260929"
comparison.PACKAGES = {
    "preview": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-candidate-20260929/submission-speed-preview-road-recenter.zip",
    "preview_speed": comparison.ROOT / "artifacts/haic-research-v2/speed-preview-road-recenter-r3-candidate-20260929/submission-speed-preview-road-recenter-r3.zip",
}
comparison.EXPECTED = {
    "preview": "711729b6ea3ea2cc981a479c6f2405d5b968230462884b6361442ece1a0a210f",
    "preview_speed": "51406c1f9cf821a42470235a2311e7bcbfc0a2a5eeba17c99e03acdf141a06de",
}
comparison.CELLS = ((2, 2101), (3, 2203), (1, 2002))
comparison.SOURCES = (
    "training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py",
    "core/vendor/car_racing.py", "research/speed_coupled_preview_consumed_probe_20260929.py",
    "research/speed_preview_road_recenter_r3_consumed_probe_20260929.py",
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
        "mechanism_probe_pass": bool(candidate["completed"] and candidate["all_wheels_off_max_streak"] < 20 and cell["preview_speed", 3, 2203]["all_wheels_off_max_streak"] <= cell["preview", 3, 2203]["all_wheels_off_max_streak"] + 10 and not report["candidate_only_failures"] and sum(row.get("recenter_count", 0) for row in rows if row.get("arm") == "preview_speed") > 0),
    }
    (comparison.RUN / "diagnostic_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2), flush=True)
