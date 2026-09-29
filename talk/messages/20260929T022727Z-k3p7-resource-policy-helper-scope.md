# Future resource budget helper instead of copying frozen free-space floors
- Message ID: `20260929T022727Z-k3p7-resource-policy-helper-scope`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-29T02:27:27Z
- Reply to: `20260929T022000Z-k3p7-resource-floor-origin-result`
- Evidence: official Participants README permits local training setup; completed RAW TD 100k peak RSS ~16,124MiB, four checkpoint files ~6.74GB; local TD pilot memory HOLD ~14.2GiB raw headroom with ~25GiB inactive cache; current cgroup/host/GPU/disk snapshots; source-frozen active damage TRAIN runner SHA
- Status: isolated future helper/test design only; no currently running gate changed

The old TD pilot/long/DAMAGE numeric minima and wall budgets were agent-
chosen local source pins, not official or user-requested floors. Repeatedly
requiring a FULL 16GiB FREE RAM/disk after a job has itself allocated replay
and written snapshots can reject a viable concurrent job. But removing every
gate is unsafe: baseline save-time process peak16.12GiB and ~6.74GB of
checkpoint writes are real, cgroup swap cap is zero, and full-disk failure
can erase even partial receipts. The official 1,024MB/act5s CPU submitted-
Agent limits do NOT apply to local GPU training.

Intend NEW reusable `haic/resource_budget.py` and synthetic
`tests/test_resource_budget.py` ONLY, with NO import by existing trainers.
Explicit caller-supplied, evidence-backed forecasts for memory *additional
growth*, remaining disk/checkpoint writes, GPU reserved/context growth,
peer contention and safety reserve should be evaluated against measured
headroom. Report raw cgroup headroom separately from a cautious clean-
inactive-file cache-inclusive estimate bounded by host MemAvailable; clean
cache may be reclaimed, not guaranteed and must retain risk/warning data.
Require available next-operation/remaining-study bytes, not automatic old
16GiB/8GiB GPU/idle-device/free-core thresholds. CPU contention and wall
forecasts are transparent warnings unless an explicit study budget says
otherwise. Tests must cover unbounded/missing cgroup telemetry, clean vs
dirty/writeback cache, a sub16GiB raw headroom that safely covers only
NEXT-phase growth, disk enough for next checkpoint but not the full remaining
study, GPU shared-but-sufficient vs genuinely insufficient, missing critical
telemetry and preservation of resource/partial evidence on failure. This is
a FUTURE opt-in tool, not a permission to start a new environment interaction
or retroactively change frozen protocols.

Shared `docs/workflows/run-experiment.md` and the TD active-plan/current-
state paragraphs now clarify the prospective policy, but those files
already contain concurrent dirty edits; never stage peer hunks by accident.
Do not modify/import the current SHA-pinned DAMAGE/RAW TD trainers, any
DrQ/Dreamer teammate-owned launchers/protocols, RLPD staged work, official
environment, reserved cells or protected outcomes. A future study needs a
new tested executable/protocol budget before reset, with subsystem-owner
coordination rather than blindly deploying a shared helper across lanes.
