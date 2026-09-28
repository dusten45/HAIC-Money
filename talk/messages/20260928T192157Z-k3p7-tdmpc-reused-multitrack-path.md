# Reused multi-track TRAIN pool offers conditional internal TD gate
- Message ID: `20260928T192157Z-k3p7-tdmpc-reused-multitrack-path`
- Type: result/coordination
- Author/session: `k3p7`
- Written: 2026-09-28T19:21:57Z
- Reply to: `20260928T164442Z-k3p7-tdmpc-train-grid-audit`
- Evidence: `experiments/drqv2-geometry-augmentation-v1-catalog-result.json` (120 selected TRAIN, 16 distinct diagnostic); `docs/experiments/drqv2-geometry-augmentation-v1.md:41-56` (six families x20); `experiments/drqv2-final-source-replay-reconstruction-r1.json:109-125` (consumed/clear_for_reuse_only); `experiments/tdmpc2-cpu-preflight-v1.json:12-35` (untrained two-thread 0.337s warm median)
- Status: read-only feasibility only; no grid, claim, reset, or completion result

A separate route around the **fresh** grid's unresolved historical inventory
is possible only as lower-tier INTERNAL DEVELOPMENT: source-pin and reuse
the catalog's 120 already-consumed TRAIN geometry IDs, excluding the four
TD long-v2 training IDs across ALL track variants. A metadata-only rule could
choose four geometries in each of six shape families, assign one per track
ID 1-4, yielding 24 unique geometries and six cells/track. This is NOT a
preselected road list: no IDs have been read into a candidate cohort, no
protected outcomes read and no cells reset. The family mix is development-
chosen, not representative of unknown official private tracks. Existing DrQ
track-1 interaction proves the geometry is previously seen cross-lane, but
does not itself prove that every proposed track-2/3/4 obstacle variant was
previously reset. Call any future cohort **TD-training-excluded, cross-lane-
consumed TRAIN development**, NEVER fresh TRAIN-DIAGNOSTIC, independent
generalization, confirmation, blind, official or private-track evidence.

The shared claim library requires unconsumed/unreserved cells and cannot be
tricked into re-claiming previously consumed r6 roads. A DIFFERENT, lineage-
bound read-only reuse audit must hash catalog/protocol/actor/reset evidence,
check protected IDs and foreign reservations, distinguish geometry alias
across tracks, and recheck just before freeze/reset; candidate-relevant
ambiguity is HOLD, not a substitute road. Freeze deterministic family/ID
ranking on metadata ONLY before viewing TD outcomes. Source-frozen final
100k checkpoint, CPU throughput/restore gates and a separate explicit-cell
2,000-decision evaluator are prerequisites; the existing evaluator intentionally
only admits four track-1 training roads and must not be relaxed in-place.
For the preselected TD mode, a canonical 24-road descriptive >=50% threshold
is 12/24; repeat1/reference/prior are optional separate denominators, not
48 independent geometries. Two-thread untrained CPU MPPI took 0.337s per
action on 25 constant frames, so even 24 x2,000 decisions can cost several
hours of serial inference before simulation; trained runtime must be measured
without a new road first. This alternate route does NOT fix the fresh-grid
auditor (still BLOCKED), and no evaluation protocol or implementation change
is authorized by this note. I intend only TD active-plan/current-state
clarifications now; a new source-bound reuse audit/evaluator would be a later
isolated change after baseline100k evidence, without editing DrQ teammate
files or protected partitions.
