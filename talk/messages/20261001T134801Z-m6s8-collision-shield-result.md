# Short collision shield: safety gain, frozen efficiency gate fails
- Message ID: 20261001T134801Z-m6s8-collision-shield-result
- Type: result
- Author/session: m6s8
- Written: 2026-10-01T13:48:01Z
- Reply to: 20261001T131540Z-m6s8-collision-shield-scope
- Evidence: completed protocol-bound episode/raw/decision streams and primary result
- Status: frozen candidate NOT ADOPTED; crossing remains baseline

Implemented independent collision_shield.py, not recovery v2. Full footprint/all-
object short projection, first .08s candidate then .24s assumed baseline, unchanged
pedals, minimum finite-grid action-deviation + soft road cost. No road veto or
heading/reentry state. Six-action encounter cap; immediate baseline handback.
Unresolved threat dropout inhibits rearm rather than extending control.

Primary experiments/koi-collision-shield-v1-result.json and
runs/koi-collision-shield-v1/protocol.json (afeeb356...) verify all12 episodes,
six matched reused TRAIN cells/five roads, sourcead772bde... over crossinga4b35c56....
Finishes3/6->5/6, kept3/lost0/gained2; damage1.4->0, collision-positive decisions7->0,
hit objects2->0 and no new baseline-clean hit (36 objects each arm).17 changed
decisions in16 bursts; longest2 decisions/.16s, largest encounter5 (cap6).
Physical minimum fixture gap-.012577->.990591m; longest full offroad1.14s unchanged,
total full-offroad occupancy1.20->1.28s. Exact no-op/pedals/encounter and logged
geometry/state/pixel/raw pre-divergence checks pass.

Eight of nine frozen gates pass. Sole failure: retained3/3184000002 lap+180ms
violates the predeclared per-kept-lap+20ms tolerance. Kept3 mean+40ms/path+.714544m.
Ordinary controls remain2/2 clean, but combined lateral integral64.843274->65.955569
m*s and path+0.582198m do not establish reduced overavoidance. One is exact no-op;
the other uses4 one-action interventions, lap-60ms but max lateral+0.209987m.
Remaining2/3184000006 departure lies in the identical115-decision prefix, before
the first shield change116; do not infer that a new heading-recovery patch works.

The safety/completion observation is materially positive, unlike the old recovery,
but neither fresh generalization nor both user objectives is established. Preserve
the frozen NOT_ADOPTED verdict without changing the efficiency limit after results.
No policy tuning, repeat, fresh/protected/Track4 geometry, official action, model
promotion or Git write. All old root/crossing/v2/recovery hashes remain unchanged.
51 tests+31 subtests pass; source review issues resolved before freeze; CPU21
zero-reset import preflight passed. Focused independent postrun audit is pending.
Durable details: architecture section25, current-state and experiment index.
