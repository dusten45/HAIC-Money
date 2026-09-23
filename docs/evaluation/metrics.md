# Evaluation Metrics

## Metric Names Matter

Official ranking semantics belong to
[`docs/competition/info.md`](../competition/info.md). A local metric must never be
renamed or reported as an official HAIC score.

| Metric | Meaning | Permitted use |
|---|---|---|
| Completion / finish | A run reached the local mirror's finish condition | Primary internal outcome when the protocol and environment are fixed; compare on the same cells. |
| Completed lap time | Time for a completed local run | Tie-break evidence after completion, subject to the same protocol. |
| Progress | Local mirror progress for an unfinished run | Secondary proxy only; cannot promote a candidate with no required completion. |
| Damage, collision, retire reason | Reliability diagnostics | Explain failure modes and safety tradeoffs. |
| Action trace, latency, RSS, package size | Operational evidence | Required runtime/package gates, not driving quality. |
| Raw CarRacing reward, training reward, loss | Optimization diagnostics | Never an official rank metric and not a promotion substitute. |
| Custom robustness/geometric scores or smoothness | Local analysis | Label explicitly as a proxy and retain its definition with the result artifact. |

## Selection Order

For an internally comparable candidate set, use the frozen protocol's declared
order. Existing DrQ-v2 protocols use completion first, then progress, then completed
lap time. A protocol may add operational eligibility before ranking.

For current matched DrQ studies, aggregates use canonical `repeat == 0` cells only.
`finish_rate` is finished canonical cells divided by canonical cells. Progress,
reward, steps, damage, and steering delta are canonical-cell means; reward and damage
are diagnostics and never selection tie-breakers. Average and best lap time include
finishers only and are `null` without a finish.

Do not pool different algorithms, source revisions, seed grids, environment modes,
or checkpoint-selection rules into one numeric ranking. If those differ, report the
comparison as non-matched evidence.

## Single-Model Rule

`docs/results/MODEL_STATUS.md` distinguishes a validated baseline, a best
single-model candidate, official submissions, and the confirmed model. A per-track
best assembled from different submissions is not a single-model result. A model can
become the best single-model candidate only when one immutable checkpoint/package
has evidence across the declared evaluation scope.

## Reporting Template

Every durable result should identify the candidate/run/checkpoint hash, source
revision, environment/protocol, exact tracks and seeds, repeat semantics, outcome
counts, operational status, and artifact path. State whether a causal explanation
is observed, inferred, or untested.
