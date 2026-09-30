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

Preregister protocols before execution. Record control/candidate hashes, exact
cells and reservation audit, environment version, raw result locations, failures,
metrics, operational checks and comparator state. Short development diagnostics
are not fresh confirmation or SOTA evidence. Source-unbound user videos cannot
certify the current artifact. No new environment performance result was created
by reconstructing this index.
