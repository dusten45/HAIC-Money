# DrQ-v2 Training-Only Geometry Augmentation

This study diagnoses frozen pad-4 DrQ actors and prepares a reproducible training
road catalog. It does **not** train a new actor or measure official competition
performance. Source and receipt identities are frozen in
[`analysis`](../../experiments/drqv2-geometry-augmentation-v1-analysis.json),
[`catalog protocol`](../../experiments/drqv2-geometry-augmentation-v1.json),
[`catalog result`](../../experiments/drqv2-geometry-augmentation-v1-catalog-result.json),
the [`CPU diagnostic protocol`](../../experiments/drqv2-geometry-augmentation-v1-diagnostic-protocol.json),
the [`272-cell result`](../../experiments/drqv2-geometry-augmentation-v1-training-summary.json),
and the [`per-road final set`](../../experiments/drqv2-geometry-augmentation-v1-final-set.json).

## Historical Failure Evidence

The two frozen source actors are pad-4 DrQ learner 0
(`433115ce…4ad37`) and learner 1 (`c0ceab5e…992954`). On the already-consumed
pad confirmation, they finished 4/32 and 7/32 canonical obstacle cells. Learner
0 had 10/32 progress-at-least-0.9 nonfinishes versus learner 1's 5/32. The
analyst grouped results by **road seed**, not by obstacle `track_id`, source
checkpoint reload, or repeated test of the same actor on the reused screen.
Historical evaluator receipts contain terminal progress, damage, steps and action
sequence but not per-step pose, speed or progress. Their terminal progress
location is cumulative unique road tiles visited, **not** physical arclength or
measured vehicle position. The environment's `off_track` retirement means a
100-decision run of negative reward; it does not by itself prove the car was
physically outside a road polygon.

| Observation | Repeated road evidence | Structural observation | Hypothesis, not cause |
|---|---|---|---|
| Early, collision-free stops | Learner 0 terminated at progress 0.0667 and 0.0739, about steps 139 and 135, on all four obstacle variants of consumed road seeds `33101` and `33103`; damage and collision counts were zero. Learner 1 traversed farther or completed variants. | The first major positive-curvature bend begins about 59.5 m after spawn, with net turn roughly 1.20-1.24 rad over 14 m. | Turn-entry speed/control/visual anticipation may be vulnerable. This is only **two** roads, not eight independent failures; `33102` has a sharper opening bend yet learner 0 reaches progress 1.0, ruling out a simple monotone curvature explanation. |
| Dense reversals and larger first bend | In the consumed r3 teacher-training pool, learner 1 finished on three of eight road seeds; its no-finish road seeds (n=5) had mean early major bend 1.73 rad versus 1.18 rad for seeds with a finish (n=3). | Mean short-gap mid-road reversals were 2.6 versus 1.67 respectively. | Test the *combination* of entry demand and recovery between opposite bends, not a universal left/right weakness. This comparison is exploratory and small. |
| Late high-progress nonfinish | On the consumed pad confirmation, learner 0 reached progress 1.0 and `finish_qualified=true` in two examples yet did not complete a directed finish crossing. | Closing bend and approach differ among roads; old receipts have no crossing pose/velocity. | Final approach heading, speed and directional crossing may matter. A late bend is an exposure to test, not proof that geometry alone causes failed finishes. |
| Obstacle dependence | Some variants of the same road have different outcomes, and other source-training episodes duplicate their entire action/progress trace. | `track_id` affects obstacles but not the seeded road centerline. | Compare road-level effects separately from obstacle interactions; repeated deterministic trajectories are not independent confirmations. |

The analysis materializes 36 **already-consumed** road signatures and hashes 43
named allowed source artifacts. No blind road/outcome was opened. Existing
validation and confirmation cells are diagnostic references only and are **never
inserted into training**. A consumed confirmation cohort cannot be described as
a fresh future confirmation holdout.

## Generator And Families

The local official Participants mirror's `CarRacing.reset(seed, options={"track_id": ...})`
controls road shape by a uint32 seed. It fixes 12 checkpoints, a 3.5-world-unit
tile step and half-width `40/6`; `track_id` controls obstacle layout rather than
road geometry. It does **not** expose independent segment length, entry angle,
curvature, width or exact left/right reflection parameters. A separate Track Lab
custom-map generator can vary width/templates, but those roads are a different
local distribution and were not passed off as official-generated DrQ roads.

The six catalog families use *measured features* of roads from the unmodified
generator. Each has ten representative seeds and ten structural variants in
TRAIN; 16 disjoint roads are TRAIN-DIAGNOSTIC. Family assignment is exclusive,
but measured shape neighborhoods may overlap: a road can fill a later family's
predeclared quota only after earlier quotas fill, and **never** as a reaction to
actor scores.

| Family (TRAIN count) | Intended contrast | Actual generated range/side balance |
|---|---|---|
| `opening-short-entry-left-turn` (20) | First left bend near the collision-free early exits; vary the opening turn while keeping entry short. | Major early turn 1.15-1.69 rad; 20 left / 0 right. |
| `opening-delayed-high-turn` (20) | Contrast longer acceleration/straight exposure before a larger first turn. | Major early turn 2.40-2.88 rad; 20 left / 0 right. |
| `easy-curvature-anchor` (20) | Retain lower opening-turn and reversal exposure so training is not only hard exits. | Opening turn 0.93-1.15 rad, at most two mid-road short-gap reversals; 20 left / 0 right. Lower structural demand does **not** imply proven policy success. |
| `mid-road-left-right-reversal` (20) | Short-gap S-like opposite turns and action recovery. | Two to five mid-road reversals; 10 left / 10 right. |
| `mid-road-sustained-or-same-turn` (20) | Repeated same-sign bends separated by limited recovery. | 17 left / 3 right; actual generator underproduced the right form in the bounded scan. |
| `finish-approach-turn` (20) | Approach variation before directional, progress-qualified finish crossing. | Late major turn 1.43-2.20 rad; 19 left / 1 right. Finish crossing has not been caused by this feature. |

In a preliminary check of 16 consumed training roads, **all 16 first major bends
turned left**. The current official generator does not offer a mirror knob. The
catalog records this asymmetry instead of inventing early-right counterparts;
the mid-road S family achieves an actual 10/10 left/right split. A future custom
Track Lab mirror study would require a separately labeled simulation/protocol.
Width-change and narrow-road families are unsupported by this official generator.

## Allocation And Sanity

The frozen candidate reservation contains 512 training-only seeds beginning
`3910800001`. A blind-safe, SHA-pinned ID audit read only six explicitly named
protocol/seed-register/source-training-ledger files; its receipt is at
[`candidate-seed-audit.json`](../../runs/20260925-drqv2-geometry-augmentation-v1/candidate-seed-audit.json).
It checks blind **IDs only for exclusion**, and makes the limited claim "no known
recorded overlap" because the historical DrQ pilot schedule is incomplete.
The protocol's exclusion inventory holds 1,821 reserved IDs, including 72
screen/confirmation IDs and 44 blind IDs; those lists overlap and must not be
added as if they were independent roads.

The catalog inspected 187 of the 512 reserved seeds before all quotas filled:
120 unique TRAIN roads and 16 separate TRAIN-DIAGNOSTIC roads. Of 51 unselected
scans, 49 had no remaining family quota; two candidate roads, `3910800051`
and `3910800055`, were rejected solely for broken closing seams (about 10.70 m
and 10.86 m against a 3.5 m median tile step). Neither had a measured centerline
intersection; these were static generation defects, **not** actor failures.
Every selected road passed finite coordinate/closed-centerline/no-intersection
checks, road-overlap warnings were zero, spawn and finish-tracker geometry were
legal, a neutral raw physics step succeeded, and a second same-seed reset
reproduced the full coordinate hash with a fresh finish tracker. This static
finish plausibility is not a proof that an actor can finish a lap; the frozen
actor rollouts provide the subsequent termination check.

An additional [`finish-logic probe`](../../runs/20260925-drqv2-geometry-augmentation-v1/catalog/finish-logic-probe.json)
replayed a synthetic 95%-qualified, forward centerline crossing against each of
the **136** seeded finish trackers. All accepted the legal crossing while
leaving the original tracker untouched. This proves the trigger is logically
available at the declared start/finish segment, **not** that a Box2D car can
drive the synthetic path or that a frozen actor will finish the road.

The signatures contain 64 arclength-binned signed turns, normalized width,
absolute perimeter and width; comparisons allow cyclic start shifts and an
optional sign reflection. The threshold `0.06` was fixed before candidate
resets. Calibration on 24 distinct consumed nonblind roads found a closest
distinct-pair distance of about `0.139`; shifted copies of one consumed road
scored about `0.024-0.036`. The closest selected road to those 24 already
consumed held-out references scored `0.128609` (`3910800090` versus `33103`).
Eight consumed teacher-training roads were also protected by signatures. Blind
road signatures were **never read**, so blind structural near-duplication cannot
be certified; exact blind seed IDs are excluded across all training track IDs.

## Frozen Actor Difficulty

The 120 TRAIN and 16 TRAIN-DIAGNOSTIC cells were sealed before any frozen actor
rollout. Both original pad-4 actors were evaluated under the separate CPU-only
[`diagnostic protocol`](../../experiments/drqv2-geometry-augmentation-v1-diagnostic-protocol.json)
with one 1,200-decision attempt per actor/road and **no actor-based road deletion**.
Four immutable shards each sealed 34 roads x 2 actor traces. The independent
[`training-only summary`](../../experiments/drqv2-geometry-augmentation-v1-training-summary.json)
validated exact file inventories, trace values, source/checkpoint/catalog hashes
and 272/272 expected actor-road cells before reporting any outcome. Source 0
finished 37/136 roads and source 1 finished 28/136. At least one actor
actually completed 55/136 roads; physical driving finishability remains
unverified for the other 81 despite the synthetic finish-logic check.

Difficulty labels are deliberately **provisional** and per road (two actors,
one rollout each), not statistical certainty:

| Label | Count among 136 | Meaning |
|---|---:|---|
| Too easy | 10 | Both actors finished once. Not replicated ease. |
| Useful / boundary | 74 | At least one finish or >=0.9 maximum visited-tile progress, but not both actors finishing. |
| Difficult but learnable | 50 | No finish, but at least one actor reached >=0.2 maximum progress. |
| Unresolved difficult | 2 | Neither actor reached 0.2; does **not** prove a malformed road. |
| Pathological / malformed | 0 | All selected roads passed static, step and finish-logic checks. |

The 120 TRAIN roads contain 8 too-easy, 64 boundary, 46 difficult-but-learnable
and 2 unresolved cases; the 16 separately held TRAIN-DIAGNOSTIC roads contain
2, 10, 4 and 0 respectively. Their disjoint IDs and sample roles are preserved
in the catalog. The [`per-road final set`](../../experiments/drqv2-geometry-augmentation-v1-final-set.json)
joins **all 136 IDs** to family hypothesis, actual generator seed/fixed width,
selection reason, measured features, closest consumed nonblind reference,
per-source actor result and difficulty; no map was removed for low policy score.

| Family | Total road seeds including TRAIN-DIAGNOSTIC | Source 0 / source 1 single-attempt finishes | Provisional too easy / boundary / difficult / unresolved |
|---|---:|---:|---|
| Easy-curvature anchor | 23 | 7 / 9 | 4 / 11 / 8 / 0 |
| Finish-approach turn | 22 | 4 / 2 | 0 / 13 / 9 / 0 |
| Mid-road left-right reversal | 23 | 10 / 2 | 2 / 11 / 9 / 1 |
| Mid-road sustained/same turn | 22 | 6 / 5 | 2 / 12 / 8 / 0 |
| Opening delayed high turn | 23 | 4 / 5 | 1 / 12 / 9 / 1 |
| Opening short-entry left turn | 23 | 6 / 5 | 1 / 15 / 7 / 0 |

The old early-failure location is independently visible in the new **training**
roads without reopening it: source 0 ended before progress 0.1 on eight new
seeds, generally after 133-140 decisions and without collision or damage. Its
nearest road-centerline fraction at termination was roughly 0.054-0.070, while
the first major bend begins near 0.046-0.064. Source 1 progressed much further
on these same eight roads, with one finish; it had one separate early stop.
Terminal *road-nearest position* now strengthens the opening-turn association
beyond the old receipts' terminal visited-tile progress, but camera visibility,
speed, steering control and hidden obstacle effects remain competing causes.
`off_track` was still the wrapper's 100+ consecutive negative-reward rule (86
source-0 and 102 source-1 cells), not independently measured time outside the
official road boundary. No map was individually deleted after these outcomes.

Neither 37/136 nor 28/136 is an official HAIC score or a matched improvement
against a new learner. These one-attempt training-road cells were deliberately
selected for measured structure; their finish frequencies cannot be compared
as if they were drawn from the independent old confirmation distribution.

## Subsequent Training Boundary

No long DrQ-v2 training is run in this task. A later training experiment should
start with a separately frozen **fixed mixture**, not an automatic curriculum:
only two roads were low-progress for both actors, while 74 are boundary and
50 show meaningful progress. A defensible initial catalog-only episode mixture
is 30% lower-curvature anchors, 50% opening-short/S-reversal/same-turn
boundary families (split equally), and 20% opening-delayed/late-approach
families (split equally). These are proposed weights, not a trained result;
the anchor family itself includes eight difficult roads, so a future experiment
should also compare retention on known, previously consumed *training-only*
easy roads rather than assuming the family label means policy success.

The new [`geometry_sampler.py`](../../haic/algorithms/drq_v2/geometry_sampler.py)
provides a SHA-bound factory for the existing official CarRacing/CarEnvironment
frame-skip-4 stack, takes **only** catalog TRAIN seeds, rejects external reset
seed overrides, cycles within each family and checkpoints track-ID/geometry RNGs
for an episode-boundary restart. A one-step smoke on a selected training-only
road and restart-next-seed parity passed. This does **not** change the existing
`train_drqv2.py` call site or its checkpoint resume format. A later learner
experiment must explicitly integrate this factory and its sampler state,
preregister its comparison, track every selected road ID and preserve source
weight/action/replay parity; do not start long training under this protocol.

After any new DrQ learner training, a teacher-replay retry needs its own fresh
source checkpoint and training episode ledgers, a newly allocated teacher
collection pool, separate online TRAIN pool and fresh screen/confirmation/blind
IDs audited across all track IDs. The r1-r3 teacher-replay datasets and the 512
candidate IDs here are consumed or reserved training research, **not** reusable
fresh confirmation cells. Do not relax the four-distinct-finish/source A3 gate
post hoc; a new gate/budget is a new preregistered hypothesis, not an extension
of the failed r3 pilot. Blind remains unopened until a separately authorized,
successful confirmation gate.

For that later teacher-replay attempt, freeze a *new* actor checkpoint and
training reset ledger for each source seed, verify exact actor/critic/target
lineage and immutable CPU action parity, then allocate disjoint fresh teacher
collection, online TRAIN, screen and confirmation geometry pools in a
blind-safe protocol audit before interaction. The current 136 training roads
may be reused as TRAIN but never recast as new holdout cells. Keep any future
blind IDs reserved only; this task has neither opened nor analyzed their road
shapes or outcomes. Any structural near-duplicate guarantee against unopened
blind roads would require an independent custodian returning only a sealed
one-shot accept/reject result, never interactive road-shape feedback.
