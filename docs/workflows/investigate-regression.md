# Investigate Regression Workflow

Do not mask a regression by immediately switching algorithms. First establish
whether the measured change is real and comparable.

## Investigation Order

1. Source and code revision: inspect the diff, source hash, active configuration,
   and checkpoint/model hash.
2. Environment and evaluation: verify environment version, track/seed/cell scope,
   termination semantics, screen/confirmation role, and candidate selection path.
3. Data path: check observation preprocessing, frame order, action mapping,
   recurrent/reset state, reward/terminal flags, and training exclusions.
4. Runtime path: check dependency/runtime version, CPU export, package loading,
   action traces, latency, RSS, and stale artifacts.
5. Experimental validity: check matched-control equivalence, seed leakage,
   consumed-cell reuse, and whether a reported aggregate combines multiple models.

## Resolution

- Add a focused regression test when the cause is a reproducible implementation
  defect.
- Preserve the failed artifact and distinguish an observed regression from an
  unproven mechanism.
- Re-run only the smallest valid comparison needed to confirm a fix.
- Update architecture, evaluation, or current-state documentation only when their
  source-of-truth facts changed.
