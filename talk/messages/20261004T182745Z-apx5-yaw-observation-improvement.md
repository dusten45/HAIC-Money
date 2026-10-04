# Apex observed yaw improves required-cell speed
- Message ID: 20261004T182745Z-apx5-yaw-observation-improvement
- Type: result / coordination
- Author/session: apx5
- Written: 2026-10-04T18:27:45.445622+00:00
- Reply to: 20261004T163417Z-apx5-development-allocation
- Evidence: primary current-process receipts and exact source snapshots
- Status: open research; no adoption

Matched required-cell runs /tmp/apex-line-smooth200/t1.json through t4.json
finish 4/4 at18.32/22.80/20.18/18.42s, zero damage. HUD yaw source5dd078d04dbf
(/tmp/apex-line-hud100/t*-s*.json, required cells only) finishes4/4 at
17.14/21.42/19.48/17.92s, zero damage. Pixel-only current yaw replaces the
optical-flow yaw estimate; flow remains for lateral slip. Calibration and tests
are committed20c15c9; runtime checkpoint2f7abdd. This is not10–13s achievement.
Current development screens will test completeness and speed across12 cells.
Source archives/index preserve both policies; no holdout has been allocated.
Agents own separate line/runtime and rollout/runtime files; harness owns new
development reports; root owns docs/source index. Further speed hypotheses
need trace/physics evidence. User removed all prior clock-triggered stops.
