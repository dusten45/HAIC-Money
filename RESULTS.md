# Results database index

Reconstructed 2026-09-30. The earlier DB was unavailable. Existing versioned JSON
artifacts remain the detailed records; this index does not recreate missing runs,
change old decisions or retroactively lift their documented blockers. Read each
artifact for source/model hashes, exact validation, protocol and limitations.

## Relevant current and prior evidence

| Record | Recorded outcome and scope |
| --- | --- |
| [Compound clearing carry](experiments/compound-clearing-brake-carry-v1-result.json) | IMPLEMENTED_DIAGNOSTIC_PASS_INCOMPARABLE; fifth-frame brake relief max0.04; 248 static tests; no completed driving evaluation |
| [Aggressive compound pace](experiments/aggressive-compound-pace-v1-result.json) | IMPLEMENTED_DIAGNOSTIC_PASS_INCOMPARABLE; qualified visible compound target ceiling38; near/missing-obstacle target30 |
| [Adaptive compound target](experiments/adaptive-compound-target-v1-result.json) | IMPLEMENTED_DIAGNOSTIC_PASS_INCOMPARABLE; earlier qualified ceiling36 |
| [Compound latch release](experiments/compound-obstacle-latch-release-v1-result.json) | IMPLEMENTED_DIAGNOSTIC_PASS_INCOMPARABLE; supplemental brake removal during missing-obstacle latch |
| [Compound obstacle brake carry](experiments/compound-obstacle-brake-carry-v1-result.json) | IMPLEMENTED_DIAGNOSTIC_PASS_INCOMPARABLE; softer compound brake; source-unbound submission11 video evidence |
| [Submission12 evidence](experiments/video-4-12-latch-brake-evidence-v1.json) | Rendered-video diagnostic, not source-bound policy evaluation |
| [Submission11 evidence](experiments/video-4-11-compound-brake-evidence-v1.json) | Rendered-video diagnostic; motion proxy is not physical speed or clearance |
| [Hairpin failures](experiments/official-seed-compound-hairpin-evidence-v1.json) | Prior official-generator failure evidence; consulted for limitations, not new evaluation |
| [Continuity evidence](experiments/compound-obstacle-continuity-evidence-v1.json) | Historical curve/obstacle failure diagnosis |
| [Post-obstacle curve retention](experiments/post-obstacle-curve-retention-v1-result.json) | IMPLEMENTED_DIAGNOSTIC_PASS_INCOMPARABLE; no claim seeds11/17/21/42 were fixed |
| [Fast corner carry](experiments/fast-corner-carry-v1-result.json) | IMPLEMENTED_DIAGNOSTIC_PASS_INCOMPARABLE; later historical regression evidence must also be considered |
| [Double straight throttle](experiments/double-clear-straight-throttle-v1-result.json) | IMPLEMENTED_DIAGNOSTIC_PASS_INCOMPARABLE; requested throttle is not doubled physical speed |
| [Clear straight sustain](experiments/clear-straight-sustain-pace-v1-result.json) | rejected-static-review; simple constant override rejected |
| [Racing-line hazard recovery](experiments/racing-line-hazard-recovery-v1-result.json) | rejected; preserve failure record |
| [DrQ L2 study](experiments/drqv2-steering-logit-v1-result.json) | rejected; confirmation finish deltas -3 and -7 in CONTEXT summary; no promotion |
| [DrQ promotion decision](experiments/drqv2-promotion-v1-result.json) | Predeclared screen gate failed; checkpoint not promoted despite diagnostic nonzero completion |
| [DrQ padding execution](experiments/drqv2-pad-execution.json) | Recorded as training, no outcome; process liveness not reverified by this restoration |

This is a navigation index, not an exhaustive rewrite: all other
`experiments/*-result.json`, evidence JSONs, protocol JSONs and their referenced raw
artifacts remain part of the DB. No failure or neutral record is deleted. Historical
"current candidate" wording is as-of that record; [CONTEXT.md](CONTEXT.md) identifies
the active source. Incomparable protocols must not be merged into a leaderboard.

## New records

- [Feasible corridor consumed triage](experiments/feasible-corridor-dev-v1-result.json):
  TRIAGE_BLOCK_FULL on ten already used cells. Finishes3->6, mean progress
  0.732->0.876, contacts23->13, but track1/17 lost a control finish and
  track3/3857792434 plus track2/4089604952 added crash DNFs. All20 rows
  and frozen hashes passed audit. Full55 and fresh partitions stayed closed;
  active route unchanged. Six exact source-bound replays show all three hard
  failures occur under no-corridor fallback; the earlier plans' causal role is
  unresolved.
- [Observed ego-side switch fresh screen](experiments/observed-ego-side-switch-generalization-v1-result.json):
  REJECT on eight new geometry seeds crossed with IDs1--4. Finishes11/32->17/32
  and mean progress0.675->0.777, but contacts65->73 and damage13.0->14.6;
  seven paired cells violated the preregistered safety veto. All68 cold runs
  and deterministic repeats passed independent audit. Source-bound replays of
  two regressions found that the ego-side rule blocked beneficial baseline
  obstacle-side switches. Confirmation/blind remain sealed; active route unchanged.
- [Consumed screen obstacle stalls](experiments/consumed-screen-obstacle-stall-v1-result.json):
  exact baseline replays on track1/3 seed3857792434 reproduced original action
  hashes. Both cars saw road and an obstacle, kept all wheels on road, but
  barely moved and visited no new tile during the final101 decisions. Their
  `off_track` retirements were reward starvation on road, not road-loss fallback.
- [Observed ego-side switch development](experiments/observed-ego-side-switch-dev-v1-result.json):
  RETAIN_DIAGNOSTIC_CANDIDATE on23 reused cells/69 cold episodes. Versus active
  baseline, finishes13/23->17/23, mean progress0.819->0.932, contacts32->31;
  versus rejected margin candidate, finishes16/23->17/23, contacts33->31.
  No paired safety veto. All32 repeated prior-screen control/margin rows matched
  original action hashes and outcomes. This reused-cell gain did not pass its
  subsequent fresh-geometry screen; active route stayed unchanged.
- [Observed margin fresh screen](experiments/observed-margin-generalization-v1-result.json):
  REJECT despite finishes9/16->10/16 and mean progress0.851->0.902 on four new
  geometry seeds crossed with IDs1--4. Track1/3892761381 contacts2->3 and
  damage0.4->0.6 violated the preregistered safety veto; confirmation and blind
  remain sealed. No active route or SOTA promotion.
- [Observed margin arbitration](experiments/observed-margin-arbitration-v1-result.json):
  retained as a diagnostic candidate after seven reused pairs; finishes4/7->6/7,
  no collision or damage regression, but +0.24s across four mutually completed
  laps, more partial off-track samples on seed17, and unchanged seed42 crash.
  Requires fresh evaluation before activation or any SOTA claim.
- [Observed centerline arbitration](experiments/observed-centerline-arbitration-v1-result.json):
  REJECT after six reused pairs; finishes3/6->5/6 but track2 contacts2->4 and
  damage0.4->0.8. Track3/1007 was stopped before execution.
- [Camera failure trace](experiments/corridor-failure-trace-v1-result.json):
  four baseline trajectories exactly reproduced with pixel/hook telemetry.
  Missing row42 was replaced by image center, reversing requested turn on
  seed11/21 despite three still-visible leftward road points.
- [Observed road target](experiments/observed-road-target-v1-result.json):
  REJECT after two pairs. Seed11 recovered from DNF18.92% to clean22.20s;
  seed21 recovered from DNF33.73% to27.92s but gained one collision/damage0.2.
  Five remaining cells unrun. This is not an activated or SOTA policy.
- `observed-road-curb-filter-v1` was preregistered but deferred without
  implementation/environment execution. Offline traces identify white-curb
  false positives; whole-component suppression could also hide a real obstacle
  touching a curb. Seed11/21 false selections have action-equivalent real
  obstacles behind them, so filtering alone does not explain their recovery.
- [Near passing commitment](experiments/observed-road-side-commit-v1-result.json):
  REJECT after two pairs. Seed11 remains clean22.20s; seed21 crashes at81.95%
  with5 collision events/damage1.0. At steps273/274 small left recentering
  suppresses right avoidance despite a rightward far bend; side remains stable.
  Remaining5 cells unrun; original
  Agent route preserved. No additional threshold sweep or SOTA promotion.
- [Geometry-supported arbitration](experiments/observed-curve-arbitration-v1-result.json):
  REJECT after6 pairs/12 runs. Finishes3/6->6/6, recovering seed11/21 clean
  in22.20/27.84s and seed42 in24.98s with3contacts (control crash5).
  Track2/644062 increases collision2->3 and damage0.4->0.6, time28.12->28.98s;
  paired completed time sum increases1.64s. Track3/1007 unrun. Active route
  unchanged despite recovered failures; neither generalized speed nor safety
  improvement is established.

- [Compound onset](experiments/compound-brake-onset-v1-result.json): 14 completed
  development runs, both arms 4/7 finishes, zero changed actions and no time gain;
  INCONCLUSIVE, not activated.
- [Visible compound base-brake result](experiments/visible-compound-base-brake-v1-result.json):
  REJECT; seed17 changes from clean25.32s finish to off-track DNF54.82%; seed42/21
  also increase off-road samples. Stopped after9 completed runs; next candidate
  interrupted, remaining cells unrun. Active route unchanged.

Preregister protocols before execution. Record control/candidate hashes, exact
cells and reservation audit, environment version, raw result locations, failures,
metrics, operational checks and comparator state. Short development diagnostics
are not fresh confirmation or SOTA evidence. Source-unbound user videos cannot
certify the current artifact. No new environment performance result was created
by reconstructing this index.
