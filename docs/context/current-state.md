# Current Research State

Last refreshed: 2026-09-25. This is the current-state source of truth, not an
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
| User-directed pixel RLPD pilot | V2 is closed stop/hold. Its separate fresh long-horizon v1 follow-up passed screen, strict matched confirmation, and blind; RLPD seed 11 at 131,072 student steps is an internal candidate. Three entropy-ablation drafts aborted at zero-interaction preflight; their allocations are retired. V4's fresh `-1.5` vs `+1.5` matched ablation is now running under a 47-file source/runtime lock on new geometry. No V4 result exists yet. See `docs/experiments/INDEX.md`. |
| Best single-model designation | No official or competition-confirmed model is designated. Pixel RLPD seed 11 is an internal, limited-geometry blind-tested candidate only. See `docs/results/MODEL_STATUS.md`. |
| Official external state | This repository has no committed official server submission identifier, public-track result, or model-confirmation receipt. A local package archive is not proof of an official submission. |

## Active Work

The active plan is
[`docs/plans/active/dreamerv3-recovery-strategy.md`](../plans/active/dreamerv3-recovery-strategy.md).
It is a correctness-and-feasibility plan, not authorization to start a 131,072-
decision run. The initial fidelity work and nine B1 static-random-data trials
are complete; all failed their frozen gates. The
[`B1 failure diagnosis`](../../experiments/dreamerv3-b1-failure-diagnosis-v1.json)
separates implementation errors, sparse training signal, and gate design defects.
No completion or official score claim follows from its offline diagnostics.

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
proven optimal. Three entropy-ablation protocol drafts (metadata budget, filename /
internal-name mismatch, and learner-seed/source-hash mismatch) each aborted before
environment interaction, and each geometry allocation is retired. The V4 one-factor
`-1.5` versus `+1.5` protocol passed source/runtime/schema and seed-token preflight.
It uses fresh data and seeds, with no long-horizon or aborted-ablation cell/data
reuse. Its persistent runner has started teacher collection and matched training on
the fresh geometry; no V4 training or screen result is recorded yet.

## Competition Schedule And Access

The first mock in the last site-verified schedule has passed. The next event in
that historical snapshot is the second mock, but the competition-site hostname
failed DNS resolution on 2026-09-25, so the **current official schedule remains
unverified**. Dates, timezones, the site-access check, and the unverified
confirmation-cutoff report are recorded only in
[`docs/competition/info.md`](../competition/info.md). Recheck a working official
site before any external action; the old schedule is for internal planning only.

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
