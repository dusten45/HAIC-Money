# Current Research State

Last refreshed: 2026-09-26. This is the current-state source of truth, not an
experiment changelog. Evidence and historical decisions are linked below.

## Current Position

| Area | Current status |
|---|---|
| Validated internal baseline | Native DrQ-v2 control with augmentation pad 4. Two unique control actors, one per training seed, were evaluated on two fresh internal confirmation cohorts; the recorded control outcomes range from 4 to 7 finishes in 32 cells. |
| Active research direction | Diagnose DreamerV3 v1-v9 world-model B1 failure with frozen artifacts before deciding whether a differently designed, newly frozen experiment is warranted. No new Dreamer actor training is authorized by the B1 record. |
| Current blocker | Two learner seeds failed B1 across all nine random-data world-model studies. v8's tiny image gain did not yield reliable terminal/reward signal. v9's `pos_weight=56` upweighted common `continue=1` rather than rare terminal events; the original BCE comparator mixed training and sampled-window prevalences. See the indexed B1 diagnosis. The earlier steering-saturated policy pilot also remains a separate failure. |
| DrQ-v2 narrow tuning | Closed. The predeclared steering-logit L2 and pad=1 follow-ups both regressed; do not launch a third narrow tuning axis. |
| DrQ-v2 teacher-replay plan | The isolated r3 collection completed its fixed 16,384 decisions/source cap, but one source had only 3/4 required finished geometries. A3 is inconclusive; no paired learner training, screen, confirmation, or blind evaluation occurred. Preserve both datasets as consumed and do not extend the cap or substitute a source. See `docs/experiments/INDEX.md`. |
| DrQ-v2 training-only geometry study | Completed: consumed-road failure analysis, blind-safe seed audit, 120 distinct TRAIN roads + 16 separate TRAIN-DIAGNOSTIC roads in six measured families, static/finish-logic sanity, and 272 sealed frozen actor diagnostics. The same 136 training-only roads gave 10 both-actor finishes, 74 provisional boundary, 50 difficult-with-progress, 2 unresolved and zero selected malformed geometries. No new learner training or held-out/blind action. See `docs/experiments/drqv2-geometry-augmentation-v1.md`. |
| DrQ-v2 geometry-mix fine-tuning | All six frozen r6 online-only runs completed 32,768 additional decisions and 22,768 updates each, sampling TRAIN only. On the same 16 previously designated TRAIN-DIAGNOSTIC roads, canonical repeat-0 finishes were uniform 1/32, failure-weighted 4/32, and easy-retention 3/32 across the two learner seeds; the unchanged source actors finished 11/32. This is descriptive, unranked development evidence, not fresh generalization, confirmation, blind, or official HAIC performance; no weights were selected or promoted. r4/r5 partial attempts remain preserved and are not resumed or counted. See the r6 [`active plan`](../plans/active/drqv2-geometry-mix-plan.md), [`protocol`](../../experiments/drqv2-geometry-mix-v1-r6.json), six run `result.json` files, and [`diagnostic manifest`](../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json). |
| User-directed pixel RLPD pilot | V2 is closed stop/hold. The separate long-horizon v1 passed screen, strict confirmation, and blind; RLPD seed 11 at 131,072 steps remains an internal candidate. Entropy V1–V3 aborted before interaction. V4 consumed fresh teacher/student runs but stopped before screen on source-hash drift; its held-outs are retired. V5 used fresh prior data and four 131,072-step target-arm runs; its screen had 29/192 canonical finishes and passed all target/seed gates. Strict confirmations were author-target 17/32 and 7/32 versus +1.5 target 6/32 and 6/32; all were eligible and operationally clean. Author-target passed the paired dominance gate, and its selected seed-50 actor finished 7/24 on the internal blind (mean progress 0.657). This remains two-seed internal evidence, not an official result. See `docs/experiments/INDEX.md`. |
| Pixel RLPD completion-first G0 | Complete TRAIN-only observational diagnosis: two frozen actors on 12 shared geometries, 24 episodes and 10,049 decisions; each finished 3/12 and failed 9/12. Contact and centerline-distance events also occurred on successful controls; the 20-decision low-directed-motion event appeared only on nonfinishes in this cohort. No causal remedy or learner experiment is selected. All cells are consumed; see `experiments/rlpd-g0-completion-v1-result.json`. |
| Best single-model designation | No official or competition-confirmed model is designated. Pixel RLPD seed 11 is an internal, limited-geometry blind-tested candidate only. See `docs/results/MODEL_STATUS.md`. |
| Official external state | This repository has no committed official server submission identifier, public-track result, or model-confirmation receipt. A local package archive is not proof of an official submission. |

## Active Work

The user-authorized DrQ-v2 geometry-mix study is a separate matched experiment;
it neither reopens teacher-replay r1-r3 nor changes the active Dreamer B1 plan.
Its three fixed family mixtures and diagnostic-only protocol are frozen at
[`drqv2-geometry-mix-v1-r6`](../../experiments/drqv2-geometry-mix-v1-r6.json).
Mix r1-r3 were superseded before any interaction. R4 and r5 each had one partial
learner-0 uniform TRAIN run (16,384 decisions, 6,384 updates) and stopped before
full checkpoint: r4 had an undefined catalog-index helper, r5 detected mutable
documentation in the checkpoint source-hash list. Both were mid-episode and
cannot be resumed. R6 restarts from the frozen source checkpoint with an
explicit TRAIN pool, new RNG schedule and executable-only runtime source hashes.
Partial TRAIN exposure and failure receipts are preserved but are not candidates
or evaluations.

R6 completed with all six run results and both step-16,384 and step-32,768
checkpoint/replay-sample trace hashes preserved under
[`runs/20260925-drqv2-geometry-mix-v1-r6`](../../runs/20260925-drqv2-geometry-mix-v1-r6/);
the active plan indexes the final checkpoint and replay-trace hashes. The frozen
CPU21 development diagnostic used only the 16 catalog TRAIN-DIAGNOSTIC roads:
256 episodes across eight actor roles and two repeats, with all 128 repeat pairs
deterministically identical. Its manifest binds every trace hash and the family
summary is linked from the experiment index. This is not a fresh holdout,
confirmation, blind, or official evaluation, and model updates are not themselves
evidence of generalization or a promotion decision.

The subsequent [r6 regression diagnosis](../experiments/drqv2-geometry-mix-r6-regression.md)
paired all 32 source-actor/road cells with each variant on those **reused**
TRAIN-DIAGNOSTIC roads: uniform lost 11 source wins and gained 1 new win;
failure-weighted lost 10/gained 3/kept 1; easy-retention lost 10/gained 2/kept
1. A bounded action-replay of the 11 archived source-success episodes (5,998
decisions) matched the old traces exactly and measured deterministic actor,
twin-critic and encoder outputs on identical source observations without a new
policy rollout or update. Early actor/encoder drift is directly observed; Q1
reordering on some cells and incomplete source-state replay retention remain
contributors to test, not proven causes. This does not authorize training,
promotion or a new evaluation partition.

The active plan is
[`docs/plans/active/dreamerv3-recovery-strategy.md`](../plans/active/dreamerv3-recovery-strategy.md).
It is a correctness-and-feasibility plan, not authorization to start a 131,072-
decision run. The initial fidelity work and nine B1 static-random-data trials
are complete; all failed their frozen gates. The
[`B1 failure diagnosis`](../../experiments/dreamerv3-b1-failure-diagnosis-v1.json)
separates implementation errors, sparse training signal, and gate design defects.
No completion or official score claim follows from its offline diagnostics.

Dreamer P0/D0 has synthetic-tested actor/return contracts, reset-origin and
completed short-episode replay, and a retrospective scored-window BCE reference.
The regression suite covering the P1 teacher collector, sealed-dataset replay
bridge, fail-closed cross-lane ID inspector and adjacent contracts passed 184
synthetic/mock tests, not driving evidence.
Fractional short-episode sampling now skips unavailable windows; the collector
recomputes pinned audit evidence instead of trusting a self-declared pass receipt.
The inspector parses r6's geometry-sampler RNG separately from road IDs but
cannot certify exhaustive historical allocations and never issues a pass.
B1 remains no-go; no repaired Dreamer driving policy or P1/P1b dataset exists.
A complete independently verifiable seed inventory, fresh immutable protocol,
random-arm collector and offline learner runner remain necessary for P1/P1b.

A separate, non-promoting [reused-TRAIN engineering diagnostic](../../experiments/dreamerv3-reused-train-diagnostic-v1.json)
has since completed on four previously allocated r6 TRAIN roads. Its
[random](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/collection/random/collection-result.json)
and [DrQ-source](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/collection/teacher/collection-result.json)
collections used 1,306 and 2,548 decisions respectively, with zero versus one
local finish in four complete episodes each. Under a separate
[offline protocol](../../experiments/dreamerv3-reused-train-offline-v1.json),
two learner seeds per source completed 64 model-only updates each, without new
environment steps or actor/critic training; the four run receipts are linked in
the [experiment index](../experiments/INDEX.md). Differently sized reused data
and in-distribution training losses do not establish a matched prediction gain
or driving improvement. Fresh P1/P1b remains blocked, with no new Dreamer actor,
held-out result, or official result.

The separately frozen [reused-TRAIN development and update-budget result](../../experiments/dreamerv3-reused-train-development-v1-result.json)
scored two world-model seeds per source on the same four *training-excluded* but
previously allocated r6 TRAIN roads in each action-source stratum. Both new
collections finished 0/4 roads. At 64 model-only updates, all models missed the
shifted-repeat image baseline on teacher-action development; increasing only the
model update count to 256 worsened both teacher-trained seeds' image error on
both reused development strata. Each stratum had four independent terminal
events among 256 scored decisions and no finished development geometry;
average terminal BCE is not evidence of finish-event discrimination. The
second score reused development already inspected after 64 updates, so it is
consumed tuning feedback, not a fresh holdout or P1b result. The B1/P1 gates,
no-trained-student-actor status and prohibition on official claims remain.

## Completed Teacher-Replay Gate

The user-directed DrQ-v2 teacher-replay protocol is frozen at
[`r3`](../../experiments/drqv2-teacher-replay-v1-r3.json). The r3 datasets passed
source and integrity checks, but one actor missed the preregistered coverage
hurdle; therefore there is no learner/evaluation result or candidate promotion.
The original r1 wrapper-seam attempt and r2 trainer-preflight attempt are retained
as zero-learning aborts, not model failures. Do not reuse r2/r3 teacher datasets,
open the reserved evaluation pools, or infer performance from collection metrics.
The stop receipt and per-source evidence are indexed in
[`docs/experiments/INDEX.md`](../experiments/INDEX.md).

## Completed Pixel RLPD Pilot Gate

The user-directed pixel-RLPD v2 protocol and result are recorded at
[`protocol`](../../experiments/pixel-rlpd-offpolicy-pilot-v2.json) and
[`result`](../../experiments/pixel-rlpd-offpolicy-pilot-v2-result.json). The
teacher dataset met its fixed 8,192-decision/two-finish coverage gate, and each of
the four SAC/RLPD student runs completed 16,384 decisions. The custom 12-cell per-
actor screen (two repeats per cell) produced one canonical finish in 96 canonical
episodes: the selected
RLPD seed-1 8,192-step actor finished once; selected seed 0 finished zero times.
The pilot therefore fails its preregistered two-seed promotion gate. All eight
screened actors passed CPU reload, determinism, and operational eligibility, but
this does not offset the completion gate. Do not open v2's reserved conditional
full, confirmation, or blind cells. The user requested a new feedback/retraining
iteration; any continuation must be a separate protocol with fresh code/data/screen
and held-out allocations, not an extension of v2.

## Fresh RLPD Follow-up (Completed)

The separate hypothesis protocol is
[`pixel-rlpd-long-horizon-followup-v1`](../../experiments/pixel-rlpd-long-horizon-followup-v1.json)
(SHA-256 `2960b561f24f6dc23a42dd0d49fd1ef7a5a0864e0e3f7543cb6920bc465a4329`). Its
seed and both teacher-ledger audits were completed before interaction, with no
known exact recorded overlap; historical schedules remain incomplete. The fresh
16,384-decision teacher collection had 6 distinct-geometry finishes; the four
matched 131,072-decision student runs completed. Both screen gates passed on 24
canonical cells per actor: seed 10 selected RLPD/SAC finished 7/24 and 3/24; seed
11 finished 12/24 and 3/24. Both strict confirmations passed (3/32 vs 2/32, and
12/32 vs 7/32). The first confirmation command failed before validation/workers and
consumed zero cells; actor-specific, hash-linked projections of the immutable
screen rows then passed `previous_evaluation_metadata()` without replaying screen
cells. The preselected RLPD seed-11, 131,072-step finalist passed the 24-cell blind
with 9 canonical finishes (mean progress 0.679). All CPU reload/determinism/resource
checks passed. This remains internal CarRacing evidence, not an official score.
Full detail is at
[`pixel-rlpd-long-horizon-followup-v1-result`](../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json).

## Next RLPD Question

The frozen RLPD temperature convention remains `target_entropy=-1.5`; it is not
proven optimal. Entropy V1–V3 aborted before environment interaction and their
allocations are retired. V4 passed initial protocol/runtime/geometry checks, collected
fresh prior data and completed four matched runs, but the pre-screen source recheck
detected a changed `generalization-policy.md`; V4 evaluation never opened. V5 screen
gate passed all four target/seed minima with 29 finishes in 192 canonical episodes.
Fresh strict confirmations were 17/32 and 7/32 for the author target versus 6/32
and 6/32 for the +1.5 target; all four were eligible, deterministic, CPU-reload
identical, and had zero operational failures. Author-target passed the paired
dominance gate, and the frozen tie-break selected its seed-50 actor for the one
24-cell internal blind, which finished 7 episodes (mean progress 0.657). This
two-seed CarRacing ablation is internal proxy evidence, not official performance or
model confirmation. See the
[`V5 result`](../../experiments/pixel-rlpd-entropy-target-ablation-v5-result.json).

The separately authorized [RLPD completion-first G0](../../experiments/rlpd-g0-completion-v1-result.json)
closed its fixed TRAIN-only diagnostic budget without learner updates or held-out
evaluation. Its 12 geometry clusters produced 24 paired observational episodes,
not a matched training-treatment comparison. Read-only trace review found no
unique initiating cause: two qualified nonfinishes were road-near and oppositely
headed long before 95% visited-tile progress, while an early contact/centerline
warning can also precede a real finish. The [fixed-window offline extraction](../../experiments/rlpd-g0-fixed-windows-v1-result.json)
sealed 144 slots across those same 24 consumed TRAIN traces, with 101 anchored
windows and 43 explicit missing anchors; all six successful-parent controls were
retained, but no manual or causal labels exist. A separate
[pixel-only motion score](../../experiments/rlpd-g0-pixel-motion-v1-result.json)
covered all 10,049 decisions; 74 retrospectively satisfy the contact-stall
telemetry definition and all six finishes remain in its control distribution.
Those labels occupy four failed episodes on three road geometries; their score
range overlaps finished-control decisions. The [gate decision](../../experiments/rlpd-g0-pixel-motion-v1-decision.json)
selects no threshold or policy. The [frozen-encoder representation probe](../../experiments/rlpd-visual-representation-probe-v2-result.json)
completed 256 CPU head-only updates with the actor unchanged, but all 188
diagnostic true positives came from two correlated episodes on ONE geometry and
none of its diagnostic roads supplied a finished parent. It can at most decode
the existing visual speed HUD on reused TRAIN roads, not justify an intervention.
A new source-hashed geometry-level positive and successful-parent coverage gate,
followed by separate exact-prefix parity/harm evidence, must precede G1. None
of these observations proves recovery data, value shaping, or memory is the
causal fix. The r5 seed-audit
erratum is narrow and does not retroactively attest the malformed receipt.

## Competition Schedule And Access

The first mock in the current site-verified schedule has passed; the next event
listed is the second mock. The old hostname failed DNS resolution on 2026-09-25,
but the [replacement competition site](https://scholarships-hardwood-headers-influenced.trycloudflare.com/)
was reachable on 2026-09-26 and its bundle still published the recorded dates.
The owner reports a higher daily submission quota than the bundle's fixed upload
text; verify the authenticated effective limit before any approved upload. Dates,
timezones, the site-access check, quota provenance, and the unverified
confirmation-cutoff report are recorded only in
[`docs/competition/info.md`](../competition/info.md). Recheck the official
schedule before any external action.

## Read Next

- Current candidate provenance and confirmation status:
  [`docs/results/MODEL_STATUS.md`](../results/MODEL_STATUS.md)
- Evaluation and partition discipline:
  [`docs/evaluation/protocol.md`](../evaluation/protocol.md) and
  [`docs/evaluation/generalization-policy.md`](../evaluation/generalization-policy.md)
- Durable evidence and prior decisions:
  [`docs/experiments/INDEX.md`](../experiments/INDEX.md) and
  [`docs/decisions/INDEX.md`](../decisions/INDEX.md)
- External actions and official rules:
  [`docs/competition/`](../competition/)
- Multi-agent discussion protocol and recent working messages:
  [`talk/README.md`](../../talk/README.md)

## Evidence Boundary

The DrQ-v2 and DreamerV3 statements above summarize frozen experiment records, not
new measurements. Consult the linked JSON protocol/result artifacts before making a
new comparison or changing a candidate status.
